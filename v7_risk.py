# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — RİSK
VaR · CVaR · EVaR · Kelly · HRP · Mean-CVaR · GPD kuyruk · PSR/DSR
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform
from scipy.stats import genpareto, norm

from v7_config import CVAR_ALPHA, VAR_CONFIDENCE

log = logging.getLogger("v7.risk")


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI
# ═══════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def portfolio_cvar(daily_returns: np.ndarray, alpha: float = CVAR_ALPHA) -> Tuple[float, float]:
    """VaR (historical) ve CVaR (mean) hesaplaması. Kayıp pozitif döner."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) < 20:
        return 0.0, 0.0
    losses = -r
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


# ═══════════════════════════════════════════════════════════════════
# PORTFÖY OPTİMİZASYONU
# ═══════════════════════════════════════════════════════════════════

def mean_cvar_lp(R: np.ndarray, alpha: float = CVAR_ALPHA, max_w: float = 0.25,
                 lam: float = 1.0) -> np.ndarray:
    """
    Mean-CVaR LP (Rockafellar & Uryasev 2000):
        min  −λ·μᵀw + ζ + 1/((1−α)T) · Σ u_t
        s.t. u_t ≥ −r_tᵀw − ζ,  u_t ≥ 0,  Σw=1,  0≤w≤max_w
    """
    R = np.asarray(R, dtype=np.float64)
    T, S = R.shape
    if S * max_w < 1.0 - 1e-9:
        return np.ones(S) / S

    mu = R.mean(axis=0)
    n = S + 1 + T
    c = np.zeros(n)
    c[:S] = -lam * mu
    c[S] = 1.0
    c[S + 1:] = 1.0 / ((1.0 - alpha) * T)

    A_ub = np.zeros((T, n))
    A_ub[:, :S] = -R
    A_ub[:, S] = -1.0
    A_ub[:, S + 1:] = -np.eye(T)

    A_eq = np.zeros((1, n))
    A_eq[0, :S] = 1.0

    bounds = [(0.0, max_w)] * S + [(None, None)] + [(0.0, None)] * T
    try:
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(T), A_eq=A_eq, b_eq=[1.0],
                      bounds=bounds, method="highs")
        if res.success:
            w = np.clip(res.x[:S], 0.0, None)
            if w.sum() > 1e-9:
                return w / w.sum()
    except Exception as e:
        log.warning("Mean-CVaR LP hatası: %s", e)
    return np.ones(S) / S


def cvar_score_weights(R: np.ndarray, alpha: float = CVAR_ALPHA,
                       max_w: float = 0.25) -> np.ndarray:
    """Hızlı heuristic: μ/|CVaR|."""
    R = np.asarray(R, dtype=np.float64)
    S = R.shape[1]
    mu = R.mean(axis=0)
    cv = np.zeros(S)
    for s in range(S):
        losses = -R[:, s]
        q = np.quantile(losses, alpha)
        tail = losses[losses >= q]
        cv[s] = tail.mean() if tail.size else abs(q)
    score = np.maximum(mu / (np.abs(cv) + 1e-6), 0.0)
    if score.sum() <= 0:
        return np.ones(S) / S
    w = np.minimum(score / score.sum(), max_w)
    return w / w.sum()


def _cluster_var(cov: np.ndarray, items: List[int]) -> float:
    """HRP küme varyansı."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.maximum(np.diag(sub), 1e-12)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(R: np.ndarray) -> np.ndarray:
    """Hiyerarşik risk paritesi (López de Prado 2016)."""
    R = np.asarray(R, dtype=np.float64)
    S = R.shape[1]
    if S == 1:
        return np.ones(1)

    cov = np.atleast_2d(np.cov(R, rowvar=False))
    corr = np.nan_to_num(np.atleast_2d(np.corrcoef(R, rowvar=False)))
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    link = linkage(squareform(dist, checks=False), method="single")
    order = [int(i) for i in leaves_list(link)]

    w = np.ones(S)
    clusters = [order]
    while clusters:
        nxt: List[List[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c1, c2 = c[:half], c[half:]
            v1 = _cluster_var(cov, c1)
            v2 = _cluster_var(cov, c2)
            a = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            for i in c1:
                w[i] *= a
            for i in c2:
                w[i] *= (1.0 - a)
            nxt += [c1, c2]
        clusters = nxt
    return w / w.sum()


def risk_parity_weights(R: np.ndarray, max_w: float = 0.25) -> np.ndarray:
    """1/σ ağırlıkları, max_w ile kırpılır."""
    R = np.asarray(R, dtype=np.float64)
    vol = np.maximum(R.std(axis=0, ddof=1), 1e-9)
    w = 1.0 / vol
    w = w / w.sum()
    w = np.minimum(w, max_w)
    return w / w.sum()


# ═══════════════════════════════════════════════════════════════════
# KELLY
# ═══════════════════════════════════════════════════════════════════

def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float,
                     kelly_frac: float = 0.25, uncertainty: float = 0.0,
                     unc_penalty: float = 2.0) -> float:
    """
    Kesirli Kelly (Thorp 2006) + belirsizlik cezası.
    f* = (p·b − q)/b, b = avg_win/avg_loss
    f_pen = kelly_frac · f* · exp(−unc_penalty · uncertainty)
    """
    if avg_loss <= 1e-9 or avg_win <= 1e-9:
        return 0.0
    b = avg_win / avg_loss
    p = float(win_rate)
    q = 1.0 - p
    f = (p * b - q) / b
    if f <= 0:
        return 0.0
    f_pen = kelly_frac * f * math.exp(-unc_penalty * uncertainty)
    return _clamp(f_pen, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# RİSK METRİKLERİ (22 adet)
# ═══════════════════════════════════════════════════════════════════

def sharpe(r: np.ndarray, ann: int = 252) -> float:
    """Yıllıklandırılmış Sharpe."""
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return 0.0 if sd <= 1e-12 else float(r.mean() / sd * math.sqrt(ann))


def max_drawdown(eq: np.ndarray) -> float:
    """Maksimum DD."""
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))


def risk_metrics(r: np.ndarray, bench: Optional[np.ndarray] = None,
                 alpha: float = 0.05, n_mc: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Kapsamlı risk metrikleri (22 adet)."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if n < 30:
        raise ValueError("Risk metrikleri için en az 30 gözlem")

    losses = -r
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    out: Dict[str, float] = {"n_obs": float(n)}

    # VaR
    out["var_hist"] = float(np.quantile(losses, 1 - alpha))
    out["var_param"] = float(-mu + norm.ppf(1 - alpha) * sd)
    rng = np.random.default_rng(seed)
    sim = rng.normal(mu, sd, n_mc)
    out["var_mc"] = float(-np.quantile(sim, alpha))

    # CVaR
    var_h = out["var_hist"]
    tail = losses[losses >= var_h]
    out["cvar_mean"] = float(tail.mean()) if tail.size else var_h

    srt = np.sort(losses)[::-1]
    gamma = 10.0
    w = np.exp(-gamma * (np.arange(n) + 0.5) / n)
    out["cvar_spectral"] = float(w @ srt / w.sum())

    # EVaR (Ahmadi-Javid)
    from scipy.special import logsumexp
    zs = np.logspace(-2, 1.5, 60)
    ev = [(1.0 / z) * (logsumexp(z * losses) - math.log(n) - math.log(alpha)) for z in zs]
    out["evar"] = float(min(ev))

    # Ana performans
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd_pct = 1.0 - eq / np.maximum(peak, 1e-12)

    out["sharpe"] = sharpe(r)
    out["sortino"] = float(mu * 252 / (np.sqrt(np.mean(r[r < 0] ** 2)) * np.sqrt(252) + 1e-9))
    out["max_dd"] = max_drawdown(eq)
    out["cagr"] = float(eq[-1] ** (252 / max(n, 1)) - 1) if eq[-1] > 0 else -1.0
    out["calmar"] = out["cagr"] / max(out["max_dd"], 1e-9)
    out["ulcer"] = float(np.sqrt(np.mean(dd_pct ** 2)))
    out["vol_ann"] = sd * math.sqrt(252)

    # Şekil
    m = r - mu
    s0 = r.std(ddof=0)
    if s0 > 1e-12:
        out["skew"] = float(np.mean(m ** 3) / s0 ** 3)
        out["kurt"] = float(np.mean(m ** 4) / s0 ** 4 - 3.0)
    else:
        out["skew"] = 0.0
        out["kurt"] = 0.0

    out["kelly_full"] = mu / max(sd * sd, 1e-12)

    # Beta / alfa / IR
    if bench is None:
        b = np.zeros(n)
    else:
        b = np.asarray(bench, dtype=np.float64)
    var_b = float(b.var(ddof=1))
    cov = float(np.cov(r, b, ddof=1)[0, 1]) if n > 1 else 0.0
    beta = cov / var_b if var_b > 1e-12 else 0.0
    out["beta"] = beta
    out["alpha_ann"] = float((mu - beta * float(b.mean())) * 252)
    te = float((r - b).std(ddof=1))
    out["ir"] = float((mu - float(b.mean())) / te * math.sqrt(252)) if te > 1e-12 else 0.0
    out["treynor"] = float(mu * 252 / beta) if abs(beta) > 1e-9 else 0.0

    # GPD kuyruk
    thr = float(np.quantile(losses, 0.90))
    exc = losses[losses > thr] - thr
    xi = 0.0
    if exc.size >= 30:
        try:
            xi = float(genpareto.fit(exc, floc=0.0)[0])
        except Exception as e:
            log.debug("GPD fit başarısız: %s", e)
    out["gpd_xi"] = xi
    return out


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
        raise ValueError("DSR için n_trials ≥ 2")
    gamma = 0.5772156649
    n = len(r)
    sd_sr = math.sqrt(1.0 / max(n - 1, 1))
    sr0 = sd_sr * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                   + gamma * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr0)


# ═══════════════════════════════════════════════════════════════════
# PORTFÖY RİSK ÖZETİ
# ═══════════════════════════════════════════════════════════════════

def portfolio_summary(returns: np.ndarray, weights: np.ndarray,
                      alpha: float = CVAR_ALPHA) -> Dict[str, float]:
    """Ağırlıklı portföy risk özeti."""
    R = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != len(w):
        raise ValueError("returns (T,S), weights (S,)")
    port_ret = R @ w
    var, cvar = portfolio_cvar(port_ret, alpha)
    return {
        "port_mean": float(port_ret.mean() * 252),
        "port_vol": float(port_ret.std(ddof=1) * math.sqrt(252)),
        "port_sharpe": sharpe(port_ret),
        "port_var": var,
        "port_cvar": cvar,
        "diversification": float(1.0 - (w ** 2).sum()),
    }


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Risk Test")
    rng = np.random.default_rng(42)

    # Test 1: Risk metrikleri
    r = rng.normal(0.0005, 0.015, 500)
    m = risk_metrics(r)
    print(f"\n✅ Risk metrikleri: {len(m)} adet")
    assert len(m) >= 20
    assert m["cvar_mean"] >= m["var_hist"] - 1e-12

    # Test 2: Kelly
    k0 = fractional_kelly(0.6, 1.0, 1.0, 0.25, 0.0)
    k1 = fractional_kelly(0.6, 1.0, 1.0, 0.25, 0.5)
    print(f"\n✅ Kelly: {k0:.4f} → belirsizlikle {k1:.4f}")
    assert k0 > 0 and k1 < k0

    # Test 3: Mean-CVaR LP
    R = rng.normal(0.0005, 0.015, (300, 6))
    R[:, 0] += 0.001
    w = mean_cvar_lp(R, max_w=0.4)
    print(f"\n✅ Mean-CVaR ağırlıklar: {w.round(3)}")
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.4 + 1e-6).all()

    # Test 4: HRP
    wh = hrp_weights(R)
    print(f"\n✅ HRP ağırlıklar: {wh.round(3)}")
    assert abs(wh.sum() - 1.0) < 1e-9

    # Test 5: Risk Parity
    wrp = risk_parity_weights(R, max_w=0.4)
    print(f"\n✅ Risk Parity: {wrp.round(3)}")
    assert abs(wrp.sum() - 1.0) < 1e-9

    # Test 6: Portfolio summary
    ps = portfolio_summary(R, w)
    print(f"\n✅ Portföy: Sharpe={ps['port_sharpe']:.3f}, DD-divers={ps['diversification']:.3f}")

    # Test 7: DSR
    strong = r + 0.002
    dsr = deflated_sharpe(strong, 20)
    print(f"\n✅ DSR (güçlü edge): {dsr:.3f}")
    assert dsr > 0.5

    print("\n✅ Risk modülü testi geçti")
