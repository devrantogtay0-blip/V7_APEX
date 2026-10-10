# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ATTRIBUTION
PnL atıf · Brinson · Faktör ayrıştırma · Hisse/sektör/rejim dağılımı
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("v7.attribution")


# ═══════════════════════════════════════════════════════════════════
# PnL ATIF (Hisse bazlı)
# ═══════════════════════════════════════════════════════════════════

def pnl_by_stock(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Hisseye göre PnL toplamı."""
    if not trades:
        return pd.DataFrame(columns=["stock", "pnl", "n", "avg_pnl", "win_rate"])
    df = pd.DataFrame(trades)
    g = df.groupby("stock").agg(
        pnl=("pnl", "sum"),
        n=("pnl", "count"),
        avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    return g.sort_values("pnl", ascending=False).reset_index(drop=True)


def pnl_by_reason(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Kapanış nedenine göre PnL."""
    if not trades:
        return pd.DataFrame(columns=["reason", "pnl", "n"])
    df = pd.DataFrame(trades)
    g = df.groupby("reason").agg(
        pnl=("pnl", "sum"),
        n=("pnl", "count"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    return g.sort_values("pnl", ascending=False).reset_index(drop=True)


def pnl_by_sector(trades: List[Dict], sector_map: Dict[str, int]) -> pd.DataFrame:
    """Sektöre göre PnL."""
    if not trades:
        return pd.DataFrame(columns=["sector", "pnl", "n"])
    df = pd.DataFrame(trades)
    df["sector"] = df["stock"].map(sector_map).fillna(-1).astype(int)
    g = df.groupby("sector").agg(
        pnl=("pnl", "sum"),
        n=("pnl", "count"),
    ).reset_index()
    return g.sort_values("pnl", ascending=False).reset_index(drop=True)


def pnl_by_time(trades: List[Dict], freq: str = "W") -> pd.DataFrame:
    """Zaman bazlı PnL."""
    if not trades:
        return pd.DataFrame(columns=["date", "pnl", "cum_pnl"])
    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["t"], unit="D", errors="coerce")
    df = df.set_index("date").sort_index()
    res = df["pnl"].resample(freq).sum().reset_index()
    res["cum_pnl"] = res["pnl"].cumsum()
    return res


# ═══════════════════════════════════════════════════════════════════
# BRINSON ATIF
# ═══════════════════════════════════════════════════════════════════

def brinson_attribution(portfolio_weights: np.ndarray,
                        benchmark_weights: np.ndarray,
                        portfolio_returns: np.ndarray,
                        benchmark_returns: np.ndarray) -> Dict[str, Any]:
    """
    Brinson-Fachler atıf (1985):
    Allocation = (w_p - w_b) × (r_b - r_total_b)
    Selection = w_b × (r_p - r_b)
    Interaction = (w_p - w_b) × (r_p - r_b)
    """
    w_p = np.asarray(portfolio_weights, dtype=np.float64)
    w_b = np.asarray(benchmark_weights, dtype=np.float64)
    r_p = np.asarray(portfolio_returns, dtype=np.float64)
    r_b = np.asarray(benchmark_returns, dtype=np.float64)

    r_b_total = float((w_b * r_b).sum())

    alloc = (w_p - w_b) * (r_b - r_b_total)
    select = w_b * (r_p - r_b)
    interact = (w_p - w_b) * (r_p - r_b)

    return {
        "allocation": float(alloc.sum()),
        "selection": float(select.sum()),
        "interaction": float(interact.sum()),
        "total": float(alloc.sum() + select.sum() + interact.sum()),
        "per_asset": pd.DataFrame({
            "w_p": w_p, "w_b": w_b, "r_p": r_p, "r_b": r_b,
            "allocation": alloc, "selection": select, "interaction": interact,
        }),
    }


# ═══════════════════════════════════════════════════════════════════
# FAKTÖR AYRIŞTIRMA
# ═══════════════════════════════════════════════════════════════════

def factor_attribution(returns: np.ndarray, factor_returns: np.ndarray,
                       factor_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Basit çok faktörlü regresyon: r = α + Σ β_i · F_i + ε
    factor_returns: (T, K) matrisi
    """
    r = np.asarray(returns, dtype=np.float64)
    F = np.asarray(factor_returns, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    T, K = F.shape
    if T != len(r) or T < 20:
        return {"error": "Boyut uyumsuz"}

    # OLS
    X = np.column_stack([np.ones(T), F])
    try:
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        resid = r - X @ beta
        ss_total = float(((r - r.mean()) ** 2).sum())
        ss_resid = float((resid ** 2).sum())
        r2 = 1.0 - ss_resid / max(ss_total, 1e-12)
        alpha = float(beta[0])
        betas = beta[1:]
    except Exception as e:
        log.warning("Faktör atıf hatası: %s", e)
        return {"error": str(e)}

    names = factor_names or [f"F{i+1}" for i in range(K)]
    contrib = betas * F.mean(axis=0)  # ortalama faktör katkısı
    return {
        "alpha": alpha,
        "alpha_ann": alpha * 252,
        "betas": dict(zip(names, betas.tolist())),
        "contributions": dict(zip(names, contrib.tolist())),
        "r2": r2,
        "residual_vol": float(resid.std(ddof=1) * np.sqrt(252)),
    }


# ═══════════════════════════════════════════════════════════════════
# REJİM BAZLI PERFORMANS
# ═══════════════════════════════════════════════════════════════════

def regime_performance(returns: np.ndarray, regimes: np.ndarray,
                       regime_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Rejim bazlı getiri istatistikleri."""
    r = np.asarray(returns, dtype=np.float64)
    g = np.asarray(regimes, dtype=np.int64)
    if len(r) != len(g):
        return pd.DataFrame()
    names = regime_names or ["BULL", "BEAR", "VOL", "RANGE"]
    rows = []
    for i, name in enumerate(names):
        sel = g == i
        if sel.sum() < 2:
            continue
        rr = r[sel]
        rows.append({
            "regime": name,
            "n": int(sel.sum()),
            "avg_ret": float(rr.mean() * 252),
            "vol": float(rr.std(ddof=1) * np.sqrt(252)),
            "sharpe": float(rr.mean() / (rr.std(ddof=1) + 1e-12) * np.sqrt(252)),
            "win_rate": float((rr > 0).mean()),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# DRAWDOWN ATIF
# ═══════════════════════════════════════════════════════════════════

def drawdown_attribution(trades: List[Dict], equity_hist: List) -> Dict[str, Any]:
    """DD'nin hangi işlemlerden geldiği."""
    if not trades or not equity_hist:
        return {}
    eq = np.array([e for _, e in equity_hist])
    if len(eq) < 2:
        return {}
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.maximum(peak, 1e-9)

    # En kötü 5 DD dönemi
    worst_idx = np.argsort(dd)[-5:]
    worst_periods = []
    for i in worst_idx:
        if i < len(equity_hist):
            worst_periods.append({
                "t": equity_hist[i][0],
                "dd_pct": float(dd[i] * 100),
                "equity": float(eq[i]),
            })

    # En çok zarar veren hisseler
    df = pd.DataFrame(trades)
    if "pnl" not in df.columns:
        return {}
    losing = df[df["pnl"] < 0].groupby("stock")["pnl"].sum().sort_values()
    worst_stocks = losing.head(5).to_dict()

    return {
        "max_dd": float(dd.max()),
        "max_dd_pct": float(dd.max() * 100),
        "worst_periods": worst_periods,
        "worst_stocks": worst_stocks,
    }


# ═══════════════════════════════════════════════════════════════════
# TAM RAPOR
# ═══════════════════════════════════════════════════════════════════

def full_attribution_report(trades: List[Dict],
                            sector_map: Optional[Dict[str, int]] = None,
                            equity_hist: Optional[List] = None) -> Dict[str, Any]:
    """Kapsamlı atıf raporu."""
    out: Dict[str, Any] = {}
    out["by_stock"] = pnl_by_stock(trades)
    out["by_reason"] = pnl_by_reason(trades)
    out["by_time"] = pnl_by_time(trades)
    if sector_map:
        out["by_sector"] = pnl_by_sector(trades, sector_map)
    if equity_hist:
        out["drawdown"] = drawdown_attribution(trades, equity_hist)
    return out


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Attribution Test")
    rng = np.random.default_rng(7)

    # Sentetik trades
    trades = []
    stocks = ["GARAN", "AKBNK", "THYAO", "TUPRS", "ASELS"]
    reasons = ["stop", "tp_partial", "max_hold", "model"]
    for i in range(50):
        trades.append({
            "t": i * 2,
            "stock": str(rng.choice(stocks)),
            "qty": int(rng.integers(10, 200)),
            "entry": float(rng.uniform(50, 300)),
            "exit": float(rng.uniform(50, 300)),
            "pnl": float(rng.normal(100, 500)),
            "reason": str(rng.choice(reasons)),
        })

    # 1) Hisse bazlı
    df1 = pnl_by_stock(trades)
    print(f"\n✅ Hisse bazlı PnL:\n{df1.to_string(index=False)}")

    # 2) Neden bazlı
    df2 = pnl_by_reason(trades)
    print(f"\n✅ Neden bazlı PnL:\n{df2.to_string(index=False)}")

    # 3) Sektör bazlı
    sector_map = {s: i % 3 for i, s in enumerate(stocks)}
    df3 = pnl_by_sector(trades, sector_map)
    print(f"\n✅ Sektör bazlı:\n{df3.to_string(index=False)}")

    # 4) Zaman bazlı
    df4 = pnl_by_time(trades, freq="W")
    print(f"\n✅ Zaman bazlı: {len(df4)} hafta")

    # 5) Brinson
    w_p = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    w_b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    r_p = np.array([0.05, 0.03, 0.02, -0.01, 0.04])
    r_b = np.array([0.04, 0.02, 0.03, 0.01, 0.03])
    br = brinson_attribution(w_p, w_b, r_p, r_b)
    print(f"\n✅ Brinson:")
    print(f"   Allocation: {br['allocation']*100:.3f}%")
    print(f"   Selection: {br['selection']*100:.3f}%")
    print(f"   Interaction: {br['interaction']*100:.3f}%")

    # 6) Faktör atıf
    T = 250
    F = rng.normal(0, 0.01, (T, 3))
    r_true = 0.0002 + 0.7 * F[:, 0] + 0.5 * F[:, 1] - 0.3 * F[:, 2] + rng.normal(0, 0.005, T)
    fa = factor_attribution(r_true, F, ["mkt", "size", "value"])
    print(f"\n✅ Faktör atıf: α_yıl={fa['alpha_ann']*100:.2f}%, R²={fa['r2']:.3f}")
    print(f"   Betas: {fa['betas']}")

    # 7) Rejim performans
    regimes = rng.integers(0, 4, T)
    rp_df = regime_performance(r_true, regimes)
    print(f"\n✅ Rejim performans:\n{rp_df.to_string(index=False)}")

    # 8) Drawdown atıf
    eq_hist = [(i, 100000 + i * 100 + float(rng.normal(0, 200))) for i in range(100)]
    dd = drawdown_attribution(trades, eq_hist)
    print(f"\n✅ DD atıf: max_dd={dd['max_dd_pct']:.2f}%")

    # 9) Full report
    full = full_attribution_report(trades, sector_map, eq_hist)
    print(f"\n✅ Full report anahtarları: {list(full.keys())}")

    print("\n✅ Attribution testi geçti")
