# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — STRESS TEST
Tarihsel senaryolar · Monte Carlo · Ters stres · Duyarlılık analizi
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("v7.stress")


# ═══════════════════════════════════════════════════════════════════
# SENARYO TANIMLARI
# ═══════════════════════════════════════════════════════════════════

HISTORICAL_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "2008 Küresel Kriz", "shock": -0.45, "duration_days": 250},
    {"name": "2020 COVID Çöküşü", "shock": -0.30, "duration_days": 30},
    {"name": "2022 Enflasyon Şoku", "shock": -0.20, "duration_days": 120},
    {"name": "2018 TL Krizi", "shock": -0.25, "duration_days": 60},
    {"name": "Sert Düzeltme", "shock": -0.10, "duration_days": 20},
    {"name": "Kara Pazartesi", "shock": -0.05, "duration_days": 1},
]


# ═══════════════════════════════════════════════════════════════════
# TARİHSEL SENARYO
# ═══════════════════════════════════════════════════════════════════

def historical_stress(equity: float, exposure: float,
                      scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Tarihsel senaryo tablosu.
    equity: mevcut özkaynak (TL)
    exposure: maruziyet (0-1)
    """
    scen = scenarios or HISTORICAL_SCENARIOS
    rows = []
    for s in scen:
        shock = float(s["shock"])
        loss = exposure * equity * shock
        rows.append({
            "senaryo": s["name"],
            "şok %": shock * 100,
            "etki TL": loss,
            "sonrası özkaynak": equity + loss,
            "etki %": (loss / equity * 100) if equity > 0 else 0.0,
            "süre (gün)": s["duration_days"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# MONTE CARLO
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
        return {"error": "Yetersiz veri"}

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_sims)
    for i in range(n_sims):
        sample = rng.choice(r, size=horizon, replace=True)
        eq = equity * np.prod(1.0 + sample)
        sims[i] = eq

    return {
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "std": float(sims.std()),
        "p05": float(np.percentile(sims, 5)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "p95": float(np.percentile(sims, 95)),
        "var_95": float(equity - np.percentile(sims, 5)),
        "cvar_95": float(equity - sims[sims <= np.percentile(sims, 5)].mean()),
        "worst": float(sims.min()),
        "best": float(sims.max()),
        "prob_loss": float((sims < equity).mean()),
    }


# ═══════════════════════════════════════════════════════════════════
# TERS STRES TESTİ
# ═══════════════════════════════════════════════════════════════════

def reverse_stress(equity: float, exposure: float,
                   target_loss_pct: float = 20.0) -> Dict[str, float]:
    """
    Ters stres: hedef kayba neden olacak şok ne olmalı?
    target_loss_pct: hedef kayıp yüzdesi (örn. 20)
    """
    if exposure <= 0 or equity <= 0:
        return {"shock_needed": 0.0, "loss_amount": 0.0}
    target_loss = equity * target_loss_pct / 100.0
    shock_needed = -target_loss / (exposure * equity)
    return {
        "shock_needed_pct": shock_needed * 100,
        "loss_amount": target_loss,
        "post_equity": equity - target_loss,
    }


# ═══════════════════════════════════════════════════════════════════
# DUYARLILIK ANALİZİ
# ═══════════════════════════════════════════════════════════════════

def sensitivity_analysis(equity: float, exposure: float,
                         shocks: Optional[List[float]] = None) -> pd.DataFrame:
    """Farklı şok seviyelerinde portföy etkisi."""
    shocks = shocks or [-0.50, -0.40, -0.30, -0.20, -0.10, -0.05,
                        0.0, 0.05, 0.10, 0.20, 0.30]
    rows = []
    for s in shocks:
        loss = exposure * equity * s
        rows.append({
            "şok %": s * 100,
            "etki TL": loss,
            "sonrası özkaynak": equity + loss,
            "etki %": (loss / equity * 100) if equity > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# KORELASYON ŞOKU
# ═══════════════════════════════════════════════════════════════════

def correlation_shock_stress(returns_matrix: np.ndarray, weights: np.ndarray,
                             equity: float, target_corr: float = 0.95) -> Dict[str, float]:
    """
    Korelasyon şoku: tüm korelasyonları target_corr'a çek.
    Portföy volünü yeniden hesapla.
    """
    R = np.asarray(returns_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != len(w) or equity <= 0:
        return {}

    # Normal vol + korelasyon
    vol = R.std(axis=0, ddof=1)
    corr = np.corrcoef(R, rowvar=False)
    cov_normal = np.outer(vol, vol) * corr
    vol_normal = float(np.sqrt(w @ cov_normal @ w))

    # Şok korelasyon
    corr_shocked = np.full_like(corr, target_corr)
    np.fill_diagonal(corr_shocked, 1.0)
    cov_shocked = np.outer(vol, vol) * corr_shocked
    vol_shocked = float(np.sqrt(max(w @ cov_shocked @ w, 0.0)))

    # 1 günlük %5 VaR
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
# LİKİDİTE ŞOKU
# ═══════════════════════════════════════════════════════════════════

def liquidity_stress(exposure_value: float, adv_tl: float,
                     participation_rate: float = 0.20,
                     impact_coef: float = 0.1) -> Dict[str, float]:
    """
    Likidite şoku: pozisyonu ADV'nin belirli bir yüzdesiyle tasfiye et.
    Almgren-Chriss basit versiyonu.
    """
    if adv_tl <= 0 or exposure_value <= 0:
        return {}
    days_to_liquidate = exposure_value / max(adv_tl * participation_rate, 1.0)
    # Basit etki: √(Q/ADV) × coef
    q_over_adv = exposure_value / adv_tl
    impact = impact_coef * math.sqrt(max(q_over_adv, 0.0))
    cost_tl = exposure_value * impact
    return {
        "days_to_liquidate": days_to_liquidate,
        "impact_pct": impact * 100,
        "cost_tl": cost_tl,
        "post_equity": exposure_value - cost_tl,
    }


# ═══════════════════════════════════════════════════════════════════
# KAPSAMLI RAPOR
# ═══════════════════════════════════════════════════════════════════

def full_stress_report(equity: float, exposure: float,
                       returns: Optional[np.ndarray] = None,
                       returns_matrix: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None,
                       adv_tl: float = 1e7) -> Dict[str, Any]:
    """Tüm stres testlerini çalıştırır."""
    out: Dict[str, Any] = {"equity": equity, "exposure": exposure}

    # Tarihsel
    out["historical"] = historical_stress(equity, exposure)

    # Duyarlılık
    out["sensitivity"] = sensitivity_analysis(equity, exposure)

    # Monte Carlo
    if returns is not None and len(returns) >= 30:
        out["monte_carlo"] = monte_carlo_stress(returns, equity, n_sims=500)

    # Korelasyon şoku
    if returns_matrix is not None and weights is not None:
        out["correlation_shock"] = correlation_shock_stress(
            returns_matrix, weights, equity, 0.95)

    # Likidite
    out["liquidity"] = liquidity_stress(exposure * equity, adv_tl)

    # Ters stres
    out["reverse"] = {
        "loss_10pct": reverse_stress(equity, exposure, 10.0),
        "loss_20pct": reverse_stress(equity, exposure, 20.0),
        "loss_30pct": reverse_stress(equity, exposure, 30.0),
    }
    return out


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
    print(f"\n✅ Tarihsel senaryolar: {len(df_h)} adet")
    print(df_h.to_string(index=False))

    # 2) Monte Carlo
    rets = rng.normal(0.0003, 0.015, 500)
    mc = monte_carlo_stress(rets, equity, n_sims=200, horizon=20)
    print(f"\n✅ Monte Carlo:")
    print(f"   Medyan: ₺{mc['median']:.0f}")
    print(f"   P05: ₺{mc['p05']:.0f}, P95: ₺{mc['p95']:.0f}")
    print(f"   VaR₉₅: ₺{mc['var_95']:.0f}")
    print(f"   Kayıp olasılığı: %{mc['prob_loss']*100:.1f}")

    # 3) Ters stres
    rev = reverse_stress(equity, exposure, 20.0)
    print(f"\n✅ Ters stres: %20 kayıp için şok = %{rev['shock_needed_pct']:.1f}")

    # 4) Duyarlılık
    df_s = sensitivity_analysis(equity, exposure)
    print(f"\n✅ Duyarlılık: {len(df_s)} senaryo")

    # 5) Korelasyon şoku
    R = rng.normal(0.0003, 0.015, (300, 5))
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cs = correlation_shock_stress(R, w, equity, 0.95)
    print(f"\n✅ Korelasyon şoku:")
    print(f"   Normal vol: {cs['vol_normal']:.4f}")
    print(f"   Şok vol: {cs['vol_shocked']:.4f}")
    print(f"   Amplifikasyon: {cs['amplification']:.2f}x")

    # 6) Likidite
    ls = liquidity_stress(exposure * equity, adv_tl=1e7)
    print(f"\n✅ Likidite: {ls['days_to_liquidate']:.1f} günde tasfiye, maliyet ₺{ls['cost_tl']:.0f}")

    # 7) Full report
    full = full_stress_report(equity, exposure, rets, R, w)
    print(f"\n✅ Full report anahtarları: {list(full.keys())}")

    print("\n✅ Stress testi geçti")
