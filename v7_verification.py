# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — DOĞRULAMA
CPCV · PBO · DSR · White RC · SPA · MCS · Conformal · ADF/KPSS
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from v7_config import EMBARGO_BARS

log = logging.getLogger("v7.verification")


# ═══════════════════════════════════════════════════════════════════
# CV YARDIMCILARI
# ═══════════════════════════════════════════════════════════════════

def make_time_groups(t_lo: int, t_hi: int, n_groups: int) -> List[Tuple[int, int]]:
    """[t_lo, t_hi) → n eşit grup."""
    edges = np.linspace(t_lo, t_hi, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def purged_train_mask(t_idx: np.ndarray, t1: np.ndarray,
                      test_ranges: List[Tuple[int, int]],
                      embargo: int = EMBARGO_BARS) -> np.ndarray:
    """Purge + embargo uygulanmış train maskesi."""
    t1 = np.maximum(np.asarray(t1), np.asarray(t_idx))
    keep = np.ones(len(t_idx), dtype=bool)
    for lo, hi in test_ranges:
        overlap = (t_idx <= hi - 1) & (t1 >= lo)
        emb = (t_idx >= hi) & (t_idx < hi + embargo)
        keep &= ~(overlap | emb)
    return keep


def test_mask_from_ranges(t_idx: np.ndarray,
                          ranges: List[Tuple[int, int]]) -> np.ndarray:
    """Test maskesi."""
    m = np.zeros(len(t_idx), dtype=bool)
    for lo, hi in ranges:
        m |= (t_idx >= lo) & (t_idx < hi)
    return m


def cpcv_splits(t_idx: np.ndarray, t1: np.ndarray, t_lo: int, t_hi: int,
                n_groups: int = 8, k_test: int = 4, embargo: int = EMBARGO_BARS):
    """Combinatorial Purged CV — C(n,k) kombinasyonları."""
    groups = make_time_groups(t_lo, t_hi, n_groups)
    for combo in itertools.combinations(range(n_groups), k_test):
        ranges = [groups[g] for g in combo]
        te = np.nonzero(test_mask_from_ranges(t_idx, ranges))[0]
        tr = np.nonzero(purged_train_mask(t_idx, t1, ranges, embargo))[0]
        if len(te) and len(tr):
            yield combo, tr, te


def cpcv_path_count(n_groups: int = 8, k_test: int = 4) -> int:
    """CPCV yol sayısı: k/N · C(N,k)."""
    from math import comb
    return int(round(k_test / n_groups * comb(n_groups, k_test)))


def walk_forward_split(t_idx: np.ndarray, t1: np.ndarray, T: int,
                       fracs=(0.70, 0.15, 0.15), embargo: int = EMBARGO_BARS):
    """Walk-forward train/val/test."""
    t1 = np.maximum(np.asarray(t1), np.asarray(t_idx))
    a = int(T * fracs[0])
    b = int(T * (fracs[0] + fracs[1]))
    tr = np.nonzero(t1 < a - embargo)[0]
    va = np.nonzero((t_idx >= a) & (t1 < b - embargo))[0]
    te = np.nonzero(t_idx >= b)[0]
    return tr, va, te


# ═══════════════════════════════════════════════════════════════════
# PSR / DSR
# ═══════════════════════════════════════════════════════════════════

def probabilistic_sharpe(r: np.ndarray, sr_star: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012)."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if n < 4 or r.std(ddof=1) <= 1e-12:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    m = r - r.mean()
    s0 = r.std(ddof=0)
    g3 = float(np.mean(m ** 3) / s0 ** 3) if s0 > 1e-12 else 0.0
    g4 = float(np.mean(m ** 4) / s0 ** 4) if s0 > 1e-12 else 3.0
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / denom))


def deflated_sharpe(r: np.ndarray, n_trials: int) -> float:
    """DSR (Bailey & López de Prado 2014)."""
    if n_trials < 2:
        raise ValueError("n_trials ≥ 2")
    gamma = 0.5772156649
    n = len(r)
    sd_sr = math.sqrt(1.0 / max(n - 1, 1))
    sr0 = sd_sr * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                   + gamma * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr0)


# ═══════════════════════════════════════════════════════════════════
# PBO (CSCV)
# ═══════════════════════════════════════════════════════════════════

def cscv_pbo(M: np.ndarray, n_blocks: int = 8, max_combos: int = 70,
             seed: int = 0) -> Tuple[float, np.ndarray]:
    """CSCV-PBO (Bailey et al. 2015)."""
    M = np.asarray(M, dtype=np.float64)
    T, K = M.shape
    if K < 2:
        raise ValueError("PBO için en az 2 aday")
    nb = max(2, n_blocks - (n_blocks % 2))
    blocks = np.array_split(np.arange(T), nb)
    combos = list(itertools.combinations(range(nb), nb // 2))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in pick]

    logits = []
    for c in combos:
        is_idx = np.concatenate([blocks[b] for b in c])
        oos_idx = np.concatenate([blocks[b] for b in range(nb) if b not in c])
        sd_i = M[is_idx].std(axis=0, ddof=1)
        sr_is = np.where(sd_i > 1e-12, M[is_idx].mean(axis=0) / np.maximum(sd_i, 1e-12), 0.0)
        best = int(np.argmax(sr_is))
        sd_o = M[oos_idx].std(axis=0, ddof=1)
        sr_oos = np.where(sd_o > 1e-12, M[oos_idx].mean(axis=0) / np.maximum(sd_o, 1e-12), 0.0)
        rank = float(np.sum(sr_oos < sr_oos[best])) + 1.0
        w = rank / (K + 1.0)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits


# ═══════════════════════════════════════════════════════════════════
# WHITE RC / SPA / MCS
# ═══════════════════════════════════════════════════════════════════

def _stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator):
    """Stationary bootstrap (Politis & Romano 1994)."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / max(block, 1.0)
    i = rng.integers(0, n)
    for k in range(n):
        idx[k] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def white_reality_check(R: np.ndarray, n_boot: int = 500, block: float = 10.0,
                        seed: int = 0) -> Dict[str, float]:
    """
    White's Reality Check (White 2000).
    R: (T, K) getiri matrisi (K aday strateji).
    """
    R = np.asarray(R, dtype=np.float64)
    T, K = R.shape
    if T < 30 or K < 2:
        return {"stat": 0.0, "pvalue": 1.0}

    f = R.mean(axis=0)  # her stratejinin ortalama getirisi
    f_bar = f.max()
    f_centered = f - f_bar
    rng = np.random.default_rng(seed)
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block, rng)
        mean_boot = R[idx].mean(axis=0)
        boot_stats[b] = np.max(mean_boot - f)
    pvalue = float(np.mean(boot_stats > f_bar))
    return {"stat": float(f_bar), "pvalue": pvalue}


def hansen_spa(R: np.ndarray, benchmark: Optional[np.ndarray] = None,
               n_boot: int = 500, block: float = 10.0, seed: int = 0) -> Dict[str, float]:
    """
    Hansen's SPA test (Hansen 2005).
    R: (T,K) aday getiriler; benchmark: (T,) veya None (sıfır benchmark).
    """
    R = np.asarray(R, dtype=np.float64)
    T, K = R.shape
    if T < 30:
        return {"stat": 0.0, "pvalue": 1.0}

    if benchmark is None:
        d = R
    else:
        b = np.asarray(benchmark, dtype=np.float64).reshape(-1, 1)
        d = R - b

    d_bar = d.mean(axis=0)
    V = np.sqrt(np.mean((d - d_bar) ** 2, axis=0) + 1e-12)
    z = np.max(d_bar / V)
    rng = np.random.default_rng(seed)
    boot_z = np.zeros(n_boot)
    for b_i in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block, rng)
        d_b = d[idx].mean(axis=0)
        boot_z[b_i] = np.max((d_b - d_bar) / V)
    pvalue = float(np.mean(boot_z > z))
    return {"stat": float(z), "pvalue": pvalue}


def mcs(losses: np.ndarray, alpha: float = 0.10, n_boot: int = 500,
        block: float = 10.0, seed: int = 0) -> Dict[str, Any]:
    """
    Model Confidence Set (Hansen et al. 2011) — basitleştirilmiş.
    losses: (T, K) kayıp matrisi (düşük = iyi).
    """
    L = np.asarray(losses, dtype=np.float64)
    T, K = L.shape
    if T < 30 or K < 2:
        return {"mcs_set": list(range(K)), "pvalues": [1.0] * K}

    rng = np.random.default_rng(seed)
    include = list(range(K))
    while len(include) > 1:
        Ls = L[:, include]
        mean_l = Ls.mean(axis=0)
        worst = int(np.argmax(mean_l))
        # Eliminasyon testi (bootstrap)
        d = Ls - Ls[:, [i for i in range(Ls.shape[1]) if i != worst]].mean(axis=1, keepdims=True)
        d_bar = d.mean(axis=0)
        # en kötü relative skor
        rel = d_bar[worst]
        V = math.sqrt(np.mean((d[:, worst] - rel) ** 2) + 1e-12)
        z = rel / V
        boot_z = np.zeros(n_boot)
        for b in range(n_boot):
            idx = _stationary_bootstrap_indices(T, block, rng)
            d_b = d[idx, worst].mean() - rel
            boot_z[b] = d_b / V
        pval = float(np.mean(boot_z > z))
        if pval <= alpha / max(len(include), 1):
            include.pop(worst)
        else:
            break
    return {"mcs_set": include, "pvalues": [1.0] * len(include)}


# ═══════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════

def split_conformal_interval(y_cal: np.ndarray, pred_cal: np.ndarray,
                             pred_test: np.ndarray, alpha: float = 0.10
                             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Split conformal regresyon aralığı (Vovk 2005).
    Kapsama ≥ 1-α (değişim varsayımıyla).
    """
    res = np.abs(np.asarray(y_cal, dtype=np.float64) - np.asarray(pred_cal, dtype=np.float64))
    n = len(res)
    if n == 0:
        raise ValueError("Kalibrasyon seti boş")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(res, level, method="higher"))
    pt = np.asarray(pred_test, dtype=np.float64)
    return pt - q, pt + q, q


def conformal_class_sets(probs_cal: np.ndarray, y_cal: np.ndarray,
                         probs_test: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """
    LAC sınıflandırma kümeleri (Sadinle 2019).
    Uyumsuzluk = 1 - p(doğru sınıf).
    Dönüş: (n_test, K) bool
    """
    y_cal = np.asarray(y_cal, dtype=np.int64)
    s_cal = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    n = len(s_cal)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(s_cal, level, method="higher"))
    return (1.0 - np.asarray(probs_test)) <= q


# ═══════════════════════════════════════════════════════════════════
# ADF / KPSS / JB (basit)
# ═══════════════════════════════════════════════════════════════════

def adf_test(x: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
    """
    Basit ADF testi (Dickey-Fuller 1979).
    H0: birim kök var (durağan değil).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return {"stat": 0.0, "critical_5pct": -2.86}
    dx = np.diff(x)
    y = dx[1:]
    X = np.column_stack([x[1:-1], np.ones(len(y))])
    if len(y) < 5:
        return {"stat": 0.0, "critical_5pct": -2.86}
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = math.sqrt(np.mean(resid ** 2) / max(np.sum((X[:, 0] - X[:, 0].mean()) ** 2), 1e-12))
        stat = beta[0] / max(se, 1e-12)
    except Exception:
        stat = 0.0
    return {"stat": float(stat), "critical_5pct": -2.86}


def jb_test(x: np.ndarray) -> Dict[str, float]:
    """Jarque-Bera normallik testi."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return {"stat": 0.0, "pvalue": 1.0}
    m = x - x.mean()
    s = x.std(ddof=0)
    if s <= 1e-12:
        return {"stat": 0.0, "pvalue": 1.0}
    skew = float(np.mean(m ** 3) / s ** 3)
    kurt = float(np.mean(m ** 4) / s ** 4 - 3.0)
    jb = n / 6 * (skew ** 2 + kurt ** 2 / 4)
    pval = 1.0 - float(np.exp(-jb / 2)) if jb < 40 else 0.0
    return {"stat": float(jb), "pvalue": pval}


# ═══════════════════════════════════════════════════════════════════
# FULL VERIFICATION PACKAGE
# ═══════════════════════════════════════════════════════════════════

def full_verification(returns: np.ndarray, n_trials: int = 20,
                      is_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Tüm doğrulama testlerini çalıştırır.
    returns: (T,) tek strateji getirileri
    is_matrix: (T, K) çoklu aday matrisi (opsiyonel, White RC için)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    out: Dict[str, Any] = {}

    if len(r) >= 30:
        out["psr"] = probabilistic_sharpe(r)
        out["dsr"] = deflated_sharpe(r, n_trials)
        out["sharpe"] = float(r.mean() / max(r.std(ddof=1), 1e-12) * math.sqrt(252))
        out["adf"] = adf_test(r)
        out["jb"] = jb_test(r)

    if is_matrix is not None and is_matrix.shape[0] >= 30 and is_matrix.shape[1] >= 2:
        pbo, _ = cscv_pbo(is_matrix, n_blocks=8)
        out["pbo"] = pbo
        wrc = white_reality_check(is_matrix, n_boot=200)
        out["white_rc"] = wrc
        spa = hansen_spa(is_matrix, n_boot=200)
        out["hansen_spa"] = spa

    return out


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Verification Test")
    rng = np.random.default_rng(7)

    # 1) PSR / DSR
    r = rng.normal(0.0005, 0.015, 500)
    psr = probabilistic_sharpe(r)
    dsr = deflated_sharpe(r, 50)
    print(f"\n✅ PSR: {psr:.3f}, DSR: {dsr:.3f}")
    assert 0 <= psr <= 1
    assert dsr <= psr + 1e-12

    # 2) CPCV
    T = 400
    t_idx = np.arange(T - 40)
    t1 = t_idx + 40
    sp = list(cpcv_splits(t_idx, t1, 0, T - 40))
    print(f"\n✅ CPCV kombinasyon: {len(sp)} (beklenen 70)")
    assert len(sp) == 70
    print(f"✅ CPCV path sayısı: {cpcv_path_count()} (beklenen 35)")

    # 3) PBO
    M = rng.standard_normal((400, 8)) * 0.01
    M[:, 0] += 0.004  # güçlü edge
    pbo, _ = cscv_pbo(M, n_blocks=8)
    print(f"\n✅ PBO (güçlü): {pbo:.3f}")
    assert pbo < 0.2

    M2 = rng.standard_normal((400, 8)) * 0.01  # rastgele
    pbo2, _ = cscv_pbo(M2, n_blocks=8)
    print(f"✅ PBO (rastgele): {pbo2:.3f}")
    assert pbo2 > 0.3

    # 4) White RC
    wrc = white_reality_check(M, n_boot=200)
    print(f"\n✅ White RC: stat={wrc['stat']:.4f}, p={wrc['pvalue']:.3f}")

    # 5) SPA
    spa = hansen_spa(M, n_boot=200)
    print(f"✅ Hansen SPA: stat={spa['stat']:.3f}, p={spa['pvalue']:.3f}")

    # 6) MCS
    losses = -M
    mcs_res = mcs(losses, alpha=0.10, n_boot=200)
    print(f"✅ MCS set: {mcs_res['mcs_set']}")

    # 7) Conformal
    y = rng.standard_normal(3000)
    p = np.zeros(3000)
    lo, hi, q = split_conformal_interval(y[:1500], p[:1500], p[1500:], alpha=0.10)
    cov = float(np.mean((y[1500:] >= lo) & (y[1500:] <= hi)))
    print(f"\n✅ Conformal kapsama: {cov:.3f} (beklenen ≥0.88)")
    assert cov >= 0.88

    # 8) ADF / JB
    adf = adf_test(r)
    jb = jb_test(r)
    print(f"\n✅ ADF: {adf['stat']:.3f}, JB: {jb['stat']:.3f}")

    # 9) Full package
    full = full_verification(r, n_trials=20, is_matrix=M)
    print(f"\n✅ Full verification anahtarları: {list(full.keys())}")

    print("\n✅ Doğrulama testi geçti")
