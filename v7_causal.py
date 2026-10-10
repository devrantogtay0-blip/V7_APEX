# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — CAUSAL INFERENCE
PC skeleton · Backdoor etki · Causal features · Basit SCM
"""
from __future__ import annotations

import itertools
import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from v7_config import CAUSAL_NAMES, MACRO_KEYS

log = logging.getLogger("v7.causal")


# ═══════════════════════════════════════════════════════════════════
# KISMİ KORELASYON
# ═══════════════════════════════════════════════════════════════════

def partial_corr(rij: float, rik: float, rjk: float) -> float:
    """Tek koşullu kısmi korelasyon ρ(i,j|k)."""
    den = math.sqrt(max((1 - rik * rik) * (1 - rjk * rjk), 1e-12))
    return (rij - rik * rjk) / den


def fisher_z_pvalue(r: float, n: int, k: int) -> float:
    """Fisher-z iki yönlü p-değeri."""
    dof = n - k - 3
    if dof <= 0:
        return 1.0
    rr = min(max(r, -0.999999), 0.999999)
    z = 0.5 * math.log((1 + rr) / (1 - rr))
    return 2.0 * (1.0 - norm.cdf(math.sqrt(dof) * abs(z)))


# ═══════════════════════════════════════════════════════════════════
# PC ALGORİTMASI (İSKELET AŞAMASI)
# ═══════════════════════════════════════════════════════════════════

def pc_skeleton(X: np.ndarray, alpha: float = 0.05, max_cond: int = 1) -> np.ndarray:
    """
    PC algoritması iskelet aşaması (Spirtes 2000).
    Fisher-z koşullu bağımsızlık testiyle.
    Yalnızca order ≤ max_cond (şimdilik 0 ve 1).
    Dönüş: (p,p) simetrik bool, köşegen False.
    """
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    n, p = X.shape
    if p < 2 or n < 30:
        return np.zeros((p, p), dtype=bool)

    C = np.nan_to_num(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(C, 1.0)
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)

    # Order 0
    for i, j in itertools.combinations(range(p), 2):
        if fisher_z_pvalue(C[i, j], n, 0) > alpha:
            adj[i, j] = adj[j, i] = False

    # Order 1
    if max_cond >= 1:
        for i, j in itertools.combinations(range(p), 2):
            if not adj[i, j]:
                continue
            for k in range(p):
                if k in (i, j) or not adj[i, k]:
                    continue
                r = partial_corr(C[i, j], C[i, k], C[j, k])
                if fisher_z_pvalue(r, n, 1) > alpha:
                    adj[i, j] = adj[j, i] = False
                    break
    return adj


# ═══════════════════════════════════════════════════════════════════
# BFS MESAFESİ
# ═══════════════════════════════════════════════════════════════════

def _bfs_dist(adj: np.ndarray, src: int, dst: int) -> int:
    """Çizgede kısa yol uzunluğu; ulaşılmazsa -1."""
    p = adj.shape[0]
    dist = [-1] * p
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return dist[u]
        for v in np.nonzero(adj[u])[0]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return -1


# ═══════════════════════════════════════════════════════════════════
# BACKDOOR ETKİ TAHMİNİ (Linear OLS)
# ═══════════════════════════════════════════════════════════════════

def ols_coef(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Sabitli OLS. Dönüş: [sabit, β_1, ..., β_p]."""
    A = np.column_stack([np.ones(len(y)), X])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta
    except Exception:
        return np.zeros(X.shape[1] + 1)


def backdoor_effect(y: np.ndarray, t: np.ndarray, Z: np.ndarray) -> float:
    """Arka-kapı ayarlamalı doğrusal etki: y ~ t + Z."""
    if Z.size == 0:
        X = t.reshape(-1, 1)
    else:
        X = np.column_stack([t, Z])
    beta = ols_coef(y, X)
    return float(beta[1]) if len(beta) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════
# CAUSAL FEATURES (tek pencere)
# ═══════════════════════════════════════════════════════════════════

def causal_window_features(stock_ret: np.ndarray, lag_vars: np.ndarray,
                           alpha: float = 0.05) -> Dict[str, float]:
    """
    Tek pencere için causal özellikler.
    stock_ret: (n,) — sonuç (t anı)
    lag_vars: (n, m) — gecikmeli değişkenler (t-1)
    Yönlendirme zaman sırasından gelir: gecikmeli → mevcut.
    """
    X = np.column_stack([stock_ret, lag_vars])
    adj = pc_skeleton(X, alpha)
    m = lag_vars.shape[1]

    # Causal parent: doğrudan bağlı sayısı
    parent = int(adj[0, 1:].sum())

    # Causal path: kısa yol mesafesi
    d = _bfs_dist(adj, 0, 1)
    path = 1.0 / (1.0 + d) if d >= 0 else 0.0

    # Confounder: hem i hem j'ye bağlı
    conf = int(sum(1 for j in range(2, m + 1) if adj[0, j] and adj[1, j]))

    # D-separation: bağlı olmayanlar
    dsep = int(sum(1 for j in range(1, m + 1) if not adj[0, j]))

    # Backdoor etki
    Z = lag_vars[:, 1:] if m > 1 else np.zeros((len(stock_ret), 0))
    beta = backdoor_effect(stock_ret, lag_vars[:, 0], Z)
    sd_mkt = float(np.std(lag_vars[:, 0], ddof=1)) if len(lag_vars) > 1 else 0.0

    return {
        "causal_parent": float(parent),
        "causal_child": 0.0,
        "causal_path": path,
        "do_effect": beta,
        "counterfactual": -beta * sd_mkt,
        "mediation": 0.0,
        "instrument": 0.0,
        "confounder": float(conf),
        "collider": 0.0,
        "d_separation": float(dsep),
    }


# ═══════════════════════════════════════════════════════════════════
# CAUSAL BLOCK (panel)
# ═══════════════════════════════════════════════════════════════════

def build_causal_block(store, every: int = 20, win: int = 250,
                       alpha: float = 0.05) -> np.ndarray:
    """
    Her `every` barda pencere için causal özellikler; arada ileri doldurma.
    store.feat içindeki causal bloğu standardize ederek yazar.
    Dönüş: (S, 10, T) ham değerler.
    """
    S, T = store.C.shape
    # Piyasa getirisi
    cnt = np.maximum(store.mask.sum(axis=0), 1)
    mkt = np.where(store.mask, store.LR, 0.0).sum(axis=0) / cnt

    # Sektör ortalama
    sec = np.zeros((S, T))
    for g in np.unique(store.sector_id):
        sel = store.sector_id == g
        m = store.mask & sel[:, None]
        cnt_g = np.maximum(m.sum(axis=0), 1)
        sec[sel, :] = (np.where(m, store.LR, 0.0)).sum(axis=0) / cnt_g

    # Makro sütunları
    macro_idx = [store.stock_idx.get(k, 0) for k in MACRO_KEYS]
    # Fallback: makro bloklar zaten feat içinde var, ilk satırdan oku
    from v7_config import FEATURE_NAMES
    mac_start = FEATURE_NAMES.index(f"M_{MACRO_KEYS[0]}_z")
    macro = store.feat[0, mac_start:mac_start + len(MACRO_KEYS), :].astype(np.float64)

    raw = np.zeros((S, len(CAUSAL_NAMES), T))
    last = np.zeros((S, len(CAUSAL_NAMES)))

    for t in range(T):
        if t >= win and (t - win) % every == 0:
            lo = t - win + 1
            tau = np.arange(lo, t + 1)
            tau = tau[tau >= 1]
            for s in range(S):
                if not store.mask[s, t]:
                    last[s] = 0.0
                    continue
                y = store.LR[s, tau]
                lags = np.column_stack([
                    mkt[tau - 1],
                    sec[s, tau - 1],
                ] + [macro[k, tau - 1] for k in range(macro.shape[0])])
                ok = np.isfinite(y) & np.all(np.isfinite(lags), axis=1)
                if ok.sum() < 60:
                    last[s] = 0.0
                    continue
                f = causal_window_features(y[ok], lags[ok], alpha)
                last[s] = [f[n] for n in CAUSAL_NAMES]
        raw[:, :, t] = last

    # Standardize (yalnızca train)
    from v7_config import FEATURE_NAMES
    c0 = FEATURE_NAMES.index(CAUSAL_NAMES[0])
    from v7_config import WARMUP_BARS
    tr = slice(WARMUP_BARS, store.train_end)
    for k in range(len(CAUSAL_NAMES)):
        seg = raw[:, k, tr]
        mu = float(np.mean(seg)) if seg.size else 0.0
        sd = float(np.std(seg)) if seg.size and np.std(seg) > 1e-9 else 1.0
        z = np.clip((raw[:, k, :] - mu) / sd, -5.0, 5.0)
        store.feat[:, c0 + k, :] = (z * store.mask).astype(np.float32)

    return raw


# ═══════════════════════════════════════════════════════════════════
# BASİT SCM (Structural Causal Model)
# ═══════════════════════════════════════════════════════════════════

class SimpleSCM:
    """
    Basit doğrusal SCM. T→Y, X→T, X→Y yapısı.
    y = β_0 + β_t·t + β_x·x + ε
    t = γ_0 + γ_x·x + ε_t
    """

    def __init__(self) -> None:
        self.beta = None
        self.gamma = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """X → T → Y modelini fit eder."""
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # t = γ_0 + γ_x·x
        self.gamma = ols_coef(T, X.reshape(-1, 1))
        # y = β_0 + β_t·t + β_x·x
        A = np.column_stack([T, X])
        self.beta = ols_coef(Y, A)

    def ate(self) -> float:
        """Average Treatment Effect (β_t)."""
        return float(self.beta[1]) if self.beta is not None and len(self.beta) > 1 else 0.0

    def counterfactual(self, X: np.ndarray, T_obs: np.ndarray,
                       T_cf: np.ndarray) -> np.ndarray:
        """T_obs yerine T_cf uygulanırsa Y ne olurdu?"""
        X = np.asarray(X, dtype=np.float64)
        T_obs = np.asarray(T_obs, dtype=np.float64)
        T_cf = np.asarray(T_cf, dtype=np.float64)
        if self.beta is None:
            return np.zeros_like(X)
        y_obs = self.beta[0] + self.beta[1] * T_obs + self.beta[2] * X
        y_cf = self.beta[0] + self.beta[1] * T_cf + self.beta[2] * X
        return y_cf - y_obs


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Causal Test")

    rng = np.random.default_rng(7)

    # 1) PC skeleton — x→y, z bağımsız
    n = 600
    x = rng.standard_normal(n)
    y = 2 * x + 0.5 * rng.standard_normal(n)
    z = rng.standard_normal(n)
    adj = pc_skeleton(np.column_stack([x, y, z]), alpha=0.01)
    print(f"\n✅ PC: x–y={adj[0,1]}, x–z={adj[0,2]}, y–z={adj[1,2]}")
    assert adj[0, 1] and not adj[0, 2] and not adj[1, 2]

    # 2) Ortak neden: x ← m → y → x-y koşullu bağımsız
    m = rng.standard_normal(n)
    x2 = m + 0.3 * rng.standard_normal(n)
    y2 = m + 0.3 * rng.standard_normal(n)
    adj2 = pc_skeleton(np.column_stack([x2, y2, m]), alpha=0.01)
    print(f"✅ PC ortak neden: x–y={adj2[0,1]} (False olmalı)")
    assert not adj2[0, 1]

    # 3) Backdoor etki
    c = rng.standard_normal(2000)
    t = c + 0.5 * rng.standard_normal(2000)
    yy = 2 * t + 3 * c + 0.1 * rng.standard_normal(2000)
    beta = backdoor_effect(yy, t, c.reshape(-1, 1))
    print(f"✅ Backdoor etki: {beta:.3f} (beklenen ~2.0)")
    assert abs(beta - 2.0) < 0.1

    # 4) Causal window
    ret = 0.01 * rng.standard_normal(300)
    lags = np.column_stack([
        np.roll(ret, 1) + 0.005 * rng.standard_normal(300),
        np.roll(ret, 2) + 0.003 * rng.standard_normal(300),
    ])
    f = causal_window_features(ret, lags)
    print(f"\n✅ Causal features anahtarları: {list(f.keys())}")
    assert all(k in f for k in CAUSAL_NAMES)

    # 5) SCM
    scm = SimpleSCM()
    X3 = rng.standard_normal(500)
    T3 = 0.5 * X3 + 0.3 * rng.standard_normal(500)
    Y3 = 2.0 * T3 + 1.5 * X3 + 0.1 * rng.standard_normal(500)
    scm.fit(X3, T3, Y3)
    print(f"✅ SCM ATE: {scm.ate():.3f} (beklenen ~2.0)")
    assert abs(scm.ate() - 2.0) < 0.15

    print("\n✅ Causal testi geçti")
