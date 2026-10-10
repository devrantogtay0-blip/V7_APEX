# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — STRESS TEST (TAM SÜRÜM)
Tarihsel senaryolar · Monte Carlo · Ters stres · Duyarlılık · Korelasyon · Likidite
Tutarlı anahtar formatı · Boş portföy koruması
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("v7.stress")


# ═══════════════════════════════════════════════════════════════════
# SABİTLER
# ═══════════════════════════════════════════════════════════════════
HISTORICAL_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "2008 Küresel Kriz", "shock": -0.45, "duration_days": 250},
    {"name": "2020 COVID Çöküşü", "shock": -0.30, "duration_days": 30},
    {"name": "2022 Enflasyon Şoku", "shock": -0.20, "duration_days": 120},
    {"name": "2018 TL Krizi", "shock": -0.25, "duration_days": 60},
    {"name": "Sert Düzeltme", "shock": -0.10, "duration_days": 20},
    {"name": "Kara Pazartesi", "shock": -0.05, "duration_days": 1},
    {"name": "Dot-com Çöküşü", "shock": -0.40, "duration_days": 300},
    {"name": "Flash Crash", "shock": -0.08, "duration_days": 1},
]

DEFAULT_SHOCKS = [-0.50, -0.40, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                  0.00, 0.05, 0.10, 0.15, 0.20, 0.30]


# ═══════════════════════════════════════════════════════════════════
# 1. TARİHSEL SENARYO
# ═══════════════════════════════════════════════════════════════════

def historical_stress(equity: float, exposure: float,
                      scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Tarihsel senaryo tablosu.
    equity: mevcut özkaynak (TL)
    exposure: maruziyet (0-1)
    Dönüş: DataFrame
    """
    scen = scenarios or HISTORICAL_SCENARIOS
    rows = []
    for s in scen:
        shock = float(s["shock"])
        loss = exposure * equity * shock
        post_eq = equity + loss
        rows.append({
            "senaryo": s["name"],
            "şok_%": round(shock * 100, 2),
            "etki_TL": round(loss, 2),
            "sonrası_özkaynak": round(post_eq, 2),
            "etki_%": round((loss / equity * 100) if equity > 0 else 0.0, 2),
            "süre_gün": s["duration_days"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 2. MONTE CARLO
# ═══════════════════════════════════════════════════════════════════

def monte_carlo_stress(returns: np.ndarray, equity: float,
                       n_sims: int = 1000, horizon: int = 20,
                       seed: int = 0) -> Dict[str, Any]:
    """
    Monte Carlo: geçmiş getiri dağılımından örnekle.
    Dönüş: kuantiller.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 30 or equity <= 0:
        return {
            "error": "Yetersiz veri",
            "mean": equity, "median": equity, "std": 0.0,
            "p05": equity, "p25": equity, "p75": equity, "p95": equity,
            "var_95": 0.0, "cvar_95": 0.0,
            "worst": equity, "best": equity, "prob_loss": 0.0,
        }

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_sims)
    for i in range(n_sims):
        sample = rng.choice(r, size=horizon, replace=True)
        eq = equity * np.prod(1.0 + sample)
        sims[i] = eq

    p05 = float(np.percentile(sims, 5))
    var_95 = float(equity - p05)
    tail = sims[sims <= p05]
    cvar_95 = float(equity - tail.mean()) if tail.size else var_95

    return {
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "std": float(sims.std()),
        "p05": p05,
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "p95": float(np.percentile(sims, 95)),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "worst": float(sims.min()),
        "best": float(sims.max()),
        "prob_loss": float((sims < equity).mean()),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. TERS STRES (DÜZELTİLDİ)
# ═══════════════════════════════════════════════════════════════════

def reverse_stress(equity: float, exposure: float,
                   target_loss_pct: float = 20.0) -> Dict[str, Any]:
    """
    Ters stres: hedef kayba neden olacak şok ne olmalı?
    target_loss_pct: hedef kayıp yüzdesi (örn. 20)

    DÖNÜŞ: Her zaman AYNI anahtarlar:
        - shock_needed_pct: şok yüzdesi
        - loss_amount: kayıp tutarı (TL)
        - post_equity: sonrası özkaynak
        - note: durum notu
    """
    if exposure <= 0 or equity <= 0:
        return {
            "shock_needed_pct": 0.0,
            "loss_amount": 0.0,
            "post_equity": float(equity),
            "note": "Pozisyon yok veya geçersiz",
        }
    target_loss = equity * target_loss_pct / 100.0
    shock_needed = -target_loss / (exposure * equity)
    return {
        "shock_needed_pct": round(shock_needed * 100, 2),
        "loss_amount": round(target_loss, 2),
        "post_equity": round(equity - target_loss, 2),
        "note": "OK",
    }


def reverse_stress_all(equity: float, exposure: float,
                       targets: Optional[List[float]] = None) -> Dict[str, Dict]:
    """Birden fazla hedef için ters stres."""
    targets = targets or [10.0, 15.0, 20.0, 30.0, 50.0]
    out = {}
    for t in targets:
        key = f"loss_{int(t)}pct"
        out[key] = reverse_stress(equity, exposure, t)
    return out


# ═══════════════════════════════════════════════════════════════════
# 4. DUYARLILIK ANALİZİ
# ═══════════════════════════════════════════════════════════════════

def sensitivity_analysis(equity: float, exposure: float,
                         shocks: Optional[List[float]] = None) -> pd.DataFrame:
    """Farklı şok seviyelerinde portföy etkisi."""
    shocks = shocks or DEFAULT_SHOCKS
    rows = []
    for s in shocks:
        loss = exposure * equity * s
        rows.append({
            "şok_%": round(s * 100, 2),
            "etki_TL": round(loss, 2),
            "sonrası_özkaynak": round(equity + loss, 2),
            "etki_%": round((loss / equity * 100) if equity > 0 else 0.0, 2),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 5. KORELASYON ŞOKU
# ═══════════════════════════════════════════════════════════════════

def correlation_shock_stress(returns_matrix: np.ndarray, weights: np.ndarray,
                             equity: float, target_corr: float = 0.95
                             ) -> Dict[str, Any]:
    """
    Korelasyon şoku: tüm korelasyonları target_corr'a çek.
    Portföy volünü yeniden hesapla.
    """
    R = np.asarray(returns_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != len(w) or equity <= 0:
        return {
            "error": "Geçersiz giriş",
            "vol_normal": 0.0, "vol_shocked": 0.0,
            "var_normal_tl": 0.0, "var_shocked_tl": 0.0,
            "amplification": 1.0,
        }

    vol = R.std(axis=0, ddof=1)
    corr = np.corrcoef(R, rowvar=False)
    cov_normal = np.outer(vol, vol) * corr
    vol_normal = float(np.sqrt(max(w @ cov_normal @ w, 0.0)))

    corr_shocked = np.full_like(corr, target_corr)
    np.fill_diagonal(corr_shocked, 1.0)
    cov_shocked = np.outer(vol, vol) * corr_shocked
    vol_shocked = float(np.sqrt(max(w @ cov_shocked @ w, 0.0)))

    var_normal = 1.645 * vol_normal * equity
    var_shocked = 1.645 * vol_shocked * equity

    return {
        "vol_normal": vol_normal,
        "vol_shocked": vol_shocked,
        "var_normal_tl": var_normal,
        "var_shocked_tl": var_shocked,
        "amplification": vol_shocked / max(vol_normal, 1e-9),
    }


# ═══════════════════════════════════════════════════════════════════
# 6. LİKİDİTE ŞOKU
# ═══════════════════════════════════════════════════════════════════

def liquidity_stress(exposure_value: float, adv_tl: float,
                     participation_rate: float = 0.20,
                     impact_coef: float = 0.1) -> Dict[str, Any]:
    """
    Likidite şoku: pozisyonu ADV'nin belirli yüzdesiyle tasfiye et.
    Almgren-Chriss basit versiyonu.
    """
    if adv_tl <= 0 or exposure_value <= 0:
        return {
            "days_to_liquidate": 0.0,
            "impact_pct": 0.0,
            "cost_tl": 0.0,
            "post_equity": float(exposure_value),
            "note": "Pozisyon yok veya likidite yok",
        }
    days_to_liquidate = exposure_value / max(adv_tl * participation_rate, 1.0)
    q_over_adv = exposure_value / adv_tl
    impact = impact_coef * math.sqrt(max(q_over_adv, 0.0))
    cost_tl = exposure_value * impact
    return {
        "days_to_liquidate": round(days_to_liquidate, 2),
        "impact_pct": round(impact * 100, 4),
        "cost_tl": round(cost_tl, 2),
        "post_equity": round(exposure_value - cost_tl, 2),
        "note": "OK",
    }


# ═══════════════════════════════════════════════════════════════════
# 7. VOLATİLİTE ŞOKU
# ═══════════════════════════════════════════════════════════════════

def volatility_shock(returns: np.ndarray, equity: float,
                     vol_multiplier: float = 2.0,
                     confidence: float = 0.95) -> Dict[str, Any]:
    """Volatilite şoku: vol 2x olsa VaR ne olur?"""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 30 or equity <= 0:
        return {"var_normal": 0.0, "var_shocked": 0.0, "amplification": 1.0}
    from scipy.stats import norm
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    z = float(norm.ppf(confidence))
    var_normal = (-mu + z * sd) * equity
    var_shocked = (-mu + z * sd * vol_multiplier) * equity
    return {
        "var_normal": var_normal,
        "var_shocked": var_shocked,
        "amplification": var_shocked / max(var_normal, 1e-9),
    }


# ═══════════════════════════════════════════════════════════════════
# 8. TAM RAPOR
# ═══════════════════════════════════════════════════════════════════

def full_stress_report(equity: float, exposure: float,
                       returns: Optional[np.ndarray] = None,
                       returns_matrix: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None,
                       adv_tl: float = 1e7) -> Dict[str, Any]:
    """
    Tüm stres testlerini çalıştırır.
    DÖNÜŞ anahtarları:
        - equity
        - exposure
        - historical: DataFrame
        - sensitivity: DataFrame
        - monte_carlo: Dict (varsa)
        - correlation_shock: Dict (varsa)
        - liquidity: Dict
        - reverse: Dict (loss_10pct, loss_20pct, ...)
        - volatility_shock: Dict (varsa)
    """
    out: Dict[str, Any] = {
        "equity": float(equity),
        "exposure": float(exposure),
    }

    # Tarihsel
    out["historical"] = historical_stress(equity, exposure)

    # Duyarlılık
    out["sensitivity"] = sensitivity_analysis(equity, exposure)

    # Monte Carlo
    if returns is not None and len(returns) >= 30:
        out["monte_carlo"] = monte_carlo_stress(returns, equity, n_sims=500)
    else:
        out["monte_carlo"] = monte_carlo_stress(np.array([]), equity)

    # Korelasyon şoku
    if returns_matrix is not None and weights is not None:
        out["correlation_shock"] = correlation_shock_stress(
            returns_matrix, weights, equity, 0.95)

    # Likidite
    out["liquidity"] = liquidity_stress(exposure * equity, adv_tl)

    # Ters stres (her zaman aynı anahtarlar)
    out["reverse"] = reverse_stress_all(equity, exposure)

    # Volatilite şoku
    if returns is not None and len(returns) >= 30:
        out["volatility_shock"] = volatility_shock(returns, equity)

    return out


# ═══════════════════════════════════════════════════════════════════
# 9. RAPOR ÖZETİ (konsol/metin)
# ═══════════════════════════════════════════════════════════════════

def print_report(rep: Dict[str, Any]) -> None:
    """Konsola okunabilir rapor."""
    print(f"\n{'=' * 60}")
    print(f"🐋 STRES TESTİ RAPORU")
    print(f"{'=' * 60}")
    print(f"Özkaynak: ₺{rep.get('equity', 0):,.2f}")
    print(f"Maruziyet: %{rep.get('exposure', 0) * 100:.1f}")

    if "historical" in rep and not rep["historical"].empty:
        print(f"\n📍 Tarihsel Senaryolar:")
        print(rep["historical"].to_string(index=False))

    if "monte_carlo" in rep:
        mc = rep["monte_carlo"]
        if "error" not in mc:
            print(f"\n🎲 Monte Carlo (20g):")
            print(f"   Medyan: ₺{mc['median']:,.0f}")
            print(f"   VaR₉₅: ₺{mc['var_95']:,.0f}")
            print(f"   CVaR₉₅: ₺{mc['cvar_95']:,.0f}")
            print(f"   Kayıp Olasılığı: %{mc['prob_loss'] * 100:.1f}")

    if "reverse" in rep:
        print(f"\n🔄 Ters Stres:")
        for k, v in rep["reverse"].items():
            print(f"   {k}: Şok %{v.get('shock_needed_pct', 0):.1f}")

    if "liquidity" in rep:
        lq = rep["liquidity"]
        print(f"\n💧 Likidite: {lq.get('days_to_liquidate', 0):.1f} gün, maliyet ₺{lq.get('cost_tl', 0):,.0f}")

    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Stress Test")
    rng = np.random.default_rng(7)

    equity = 100_000.0
    exposure = 0.65

    # 1) Tarihsel
    df_h = historical_stress(equity, exposure)
    print(f"\n✅ Tarihsel: {len(df_h)} senaryo")
    print(df_h.head(3).to_string(index=False))

    # 2) Monte Carlo
    rets = rng.normal(0.0003, 0.015, 500)
    mc = monte_carlo_stress(rets, equity, n_sims=200)
    print(f"\n✅ Monte Carlo: medyan ₺{mc['median']:,.0f}, VaR₉₅ ₺{mc['var_95']:,.0f}")

    # 3) Ters stres — POZISYON VAR
    rev = reverse_stress(equity, exposure, 20.0)
    print(f"\n✅ Ters stres (pozisyon var): şok %{rev['shock_needed_pct']:.1f}")
    assert "shock_needed_pct" in rev

    # 4) Ters stres — POZISYON YOK (BUG FIX)
    rev_empty = reverse_stress(equity, 0.0, 20.0)
    print(f"✅ Ters stres (pozisyon yok): şok %{rev_empty['shock_needed_pct']:.1f}")
    assert "shock_needed_pct" in rev_empty
    assert rev_empty["shock_needed_pct"] == 0.0

    # 5) Tüm hedefler
    rev_all = reverse_stress_all(equity, exposure)
    print(f"✅ Reverse all: {list(rev_all.keys())}")
    assert "loss_10pct" in rev_all
    assert "loss_20pct" in rev_all
    assert "loss_30pct" in rev_all

    # 6) Duyarlılık
    df_s = sensitivity_analysis(equity, exposure)
    print(f"✅ Duyarlılık: {len(df_s)} satır")

    # 7) Korelasyon şoku
    R = rng.normal(0.0003, 0.015, (300, 5))
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cs = correlation_shock_stress(R, w, equity, 0.95)
    print(f"✅ Korelasyon şoku: {cs['amplification']:.2f}x")

    # 8) Likidite
    ls = liquidity_stress(exposure * equity, adv_tl=1e7)
    print(f"✅ Likidite: {ls['days_to_liquidate']:.1f} gün")

    # 9) Volatilite şoku
    vs = volatility_shock(rets, equity, 2.0)
    print(f"✅ Volatilite şoku: {vs['amplification']:.2f}x")

    # 10) Full report — POZISYON VAR
    full = full_stress_report(equity, exposure, rets, R, w)
    print(f"\n✅ Full report anahtarları: {list(full.keys())}")
    assert "reverse" in full
    assert "loss_10pct" in full["reverse"]
    assert "shock_needed_pct" in full["reverse"]["loss_10pct"]

    # 11) Full report — POZISYON YOK (KRİTİK)
    full_empty = full_stress_report(equity, 0.0, rets)
    print(f"\n✅ Full report (pozisyon yok):")
    print(f"   reverse.loss_10pct: {full_empty['reverse']['loss_10pct']}")
    assert "shock_needed_pct" in full_empty["reverse"]["loss_10pct"]
    print(f"   ✅ KeyError olmayacak")

    # 12) Print report
    print_report(full)

    print("✅ v7_stress.py TAM testi geçti")
