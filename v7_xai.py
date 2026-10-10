# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — XAI (Explainable AI)
Model-agnostic proxy · SHAP-benzeri · LIME-benzeri · Permutation · PDP
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from v7_config import FEATURE_NAMES, N_FEAT

log = logging.getLogger("v7.xai")


# ═══════════════════════════════════════════════════════════════════
# KORELASYON TABANLI AÇIKLAMA (Model-free)
# ═══════════════════════════════════════════════════════════════════

def correlation_importance(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Her özelliğin hedef değişkenle Pearson korelasyonu.
    X: (n, F), y: (n,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, F = X.shape
    if n < 5:
        return pd.DataFrame({"feature": FEATURE_NAMES[:F], "corr": 0.0, "abs_corr": 0.0})

    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum()) + 1e-12
    r = (Xc.T @ yc) / denom
    df = pd.DataFrame({
        "feature": FEATURE_NAMES[:F],
        "corr": r,
        "abs_corr": np.abs(r),
    })
    return df.sort_values("abs_corr", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def permutation_importance(predict_fn, X: np.ndarray, y: np.ndarray,
                           n_repeats: int = 5, seed: int = 0,
                           metric: str = "mse") -> pd.DataFrame:
    """
    Permutation importance: her özelliği rastgele karıştır, model skorundaki düşüşü ölç.
    predict_fn(X) → (n,) tahmin veya (n, K) olasılık matrisi.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, F = X.shape
    rng = np.random.default_rng(seed)

    def score(preds):
        if preds.ndim == 1:
            if metric == "mse":
                return -float(np.mean((preds - y) ** 2))
            elif metric == "mae":
                return -float(np.mean(np.abs(preds - y)))
            elif metric == "acc":
                return float(np.mean(preds.argmax(axis=-1) == y.astype(int)))
        # Olasılık matrisi için log-loss
        if preds.ndim == 2:
            p = np.clip(preds[np.arange(n), y.astype(int)], 1e-9, 1)
            return float(np.mean(np.log(p)))
        return 0.0

    baseline = score(predict_fn(X))
    imps = np.zeros((F, n_repeats))
    for f in range(F):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            imps[f, r] = baseline - score(predict_fn(X_perm))

    return pd.DataFrame({
        "feature": FEATURE_NAMES[:F],
        "importance_mean": imps.mean(axis=1),
        "importance_std": imps.std(axis=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# LIME-BENZERİ (Lokal Linear Approximation)
# ═══════════════════════════════════════════════════════════════════

def local_surrogate(predict_fn, x0: np.ndarray, X_pool: np.ndarray,
                    n_samples: int = 200, kernel_width: float = 0.75,
                    seed: int = 0) -> pd.DataFrame:
    """
    LIME tarzı lokal vekil model:
    - x0 etrafında Gaussian gürültülü örnekler
    - Mesafeye bağlı ağırlıklı doğrusal regresyon
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    F = x0.shape[0]
    if X_pool.shape[0] < 5:
        return pd.DataFrame({"feature": FEATURE_NAMES[:F], "coef": 0.0})

    # Std'yi havuzdan al
    scale = X_pool.std(axis=0) + 1e-9
    # Gürültülü örnekler
    Z = x0[None, :] + rng.normal(0, 1, (n_samples, F)) * scale[None, :]
    # Tahmin
    preds = predict_fn(Z)
    if preds.ndim == 2:
        y_hat = preds[:, -1]
    else:
        y_hat = preds
    # Mesafe
    dist = np.sqrt(np.sum(((Z - x0) / scale) ** 2, axis=1))
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))
    # Ağırlıklı doğrusal regresyon
    W = np.diag(weights)
    Xw = np.column_stack([np.ones(n_samples), Z - x0])
    try:
        A = Xw.T @ W @ Xw + 1e-6 * np.eye(F + 1)
        B = Xw.T @ W @ y_hat
        coef = np.linalg.solve(A, B)
        coefs = coef[1:]
    except Exception:
        coefs = np.zeros(F)
    return pd.DataFrame({
        "feature": FEATURE_NAMES[:F],
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# PARTIAL DEPENDENCE (PDP) — 1D
# ═══════════════════════════════════════════════════════════════════

def partial_dependence_1d(predict_fn, X: np.ndarray, feature_idx: int,
                          n_grid: int = 20) -> pd.DataFrame:
    """Tek özellik için PDP."""
    X = np.asarray(X, dtype=np.float64)
    f_vals = X[:, feature_idx]
    lo, hi = np.percentile(f_vals, [5, 95])
    grid = np.linspace(lo, hi, n_grid)
    pd_preds = np.zeros(n_grid)
    for i, v in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = v
        preds = predict_fn(X_mod)
        if preds.ndim == 2:
            pd_preds[i] = preds[:, -1].mean()
        else:
            pd_preds[i] = preds.mean()
    return pd.DataFrame({
        "feature_value": grid,
        "prediction": pd_preds,
        "feature": FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"f{feature_idx}",
    })


# ═══════════════════════════════════════════════════════════════════
# FEATURE GROUPING (Kategori bazlı)
# ═══════════════════════════════════════════════════════════════════

def feature_categories() -> Dict[str, Tuple[int, int]]:
    """Kategori bazlı indeks aralıkları."""
    from v7_config import (N_TECH, LIQ_NAMES, RANK_NAMES, MACRO_NAMES,
                            MICRO_NAMES, PEER_NAMES, NLP_NAMES,
                            ONEHOT_NAMES, CAUSAL_NAMES)
    off = 0
    out = {}
    out["teknik"] = (off, off + N_TECH); off += N_TECH
    out["likidite"] = (off, off + len(LIQ_NAMES)); off += len(LIQ_NAMES)
    out["rank"] = (off, off + len(RANK_NAMES)); off += len(RANK_NAMES)
    out["makro"] = (off, off + len(MACRO_NAMES)); off += len(MACRO_NAMES)
    out["mikro"] = (off, off + len(MICRO_NAMES)); off += len(MICRO_NAMES)
    out["peer"] = (off, off + len(PEER_NAMES)); off += len(PEER_NAMES)
    out["nlp"] = (off, off + len(NLP_NAMES)); off += len(NLP_NAMES)
    out["onehot"] = (off, off + len(ONEHOT_NAMES)); off += len(ONEHOT_NAMES)
    out["causal"] = (off, off + len(CAUSAL_NAMES)); off += len(CAUSAL_NAMES)
    return out


def category_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Kategori bazında toplam önem."""
    cats = feature_categories()
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    rows = []
    for cat, (lo, hi) in cats.items():
        mask = importance_df["feature"].apply(
            lambda x: lo <= name_to_idx.get(x, -1) < hi
        )
        rows.append({
            "category": cat,
            "total_importance": float(importance_df.loc[mask, "importance_mean"].sum())
            if "importance_mean" in importance_df.columns else 0.0,
            "n_features": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# TOP FEATURES
# ═══════════════════════════════════════════════════════════════════

def top_features(importance_df: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """İlk k özellik."""
    if importance_df.empty:
        return importance_df
    col = "importance_mean" if "importance_mean" in importance_df.columns else (
        "abs_corr" if "abs_corr" in importance_df.columns else importance_df.columns[-1]
    )
    return importance_df.nlargest(k, col).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — XAI Test")

    rng = np.random.default_rng(7)
    n, F = 500, 30
    X = rng.standard_normal((n, F))
    # y = 3*x0 - 2*x1 + gürültü
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * rng.standard_normal(n)

    # 1) Korelasyon importance
    df1 = correlation_importance(X, y)
    print(f"\n✅ Korelasyon: en iyi 3 = {df1.head(3)['feature'].tolist()}")
    assert df1.iloc[0]["feature"] in ("rsi7", "rsi14", "rsi21")

    # 2) Permutation
    def linear_pred(X):
        return 3 * X[:, 0] - 2 * X[:, 1]

    df2 = permutation_importance(linear_pred, X, y, n_repeats=3, metric="mse")
    print(f"\n✅ Permutation: en önemli 2 = {df2.head(2)['feature'].tolist()}")

    # 3) LIME
    df3 = local_surrogate(linear_pred, X[0], X, n_samples=200)
    print(f"\n✅ LIME: en iyi 2 katsayı = {df3.head(2)['feature'].tolist()}")
    assert df3.iloc[0]["feature"] in ("rsi7", "rsi14")

    # 4) PDP
    pdp = partial_dependence_1d(linear_pred, X, 0, n_grid=10)
    print(f"\n✅ PDP satır: {len(pdp)}")
    assert len(pdp) == 10

    # 5) Kategoriler
    cats = feature_categories()
    print(f"\n✅ Kategoriler: {list(cats.keys())}")

    # 6) Top features
    top = top_features(df2, k=5)
    print(f"\n✅ Top-5: {top['feature'].tolist()}")

    print("\n✅ XAI testi geçti")
