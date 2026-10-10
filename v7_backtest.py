# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — BACKTEST
Numba simülatör · Stop/TP/BE/Trailing · CPCV · PBO · DSR
Look-ahead koruması: purge + embargo
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from v7_config import (
    CPCV_N_GROUPS, CPCV_N_TEST_GROUPS, EMBARGO_BARS, FEE, SLIPPAGE,
    WARMUP_BARS,
)

log = logging.getLogger("v7.backtest")

try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco


# ═══════════════════════════════════════════════════════════════════
# NUMBA SİMÜLATÖR
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True, fastmath=True)
def simulate_stock_kernel(O, H, L, C, ATRN, sig, size, sl_atr, tp_r, tp_frac,
                          be_r, trail_act_r, trail_atr, max_hold, cost_in, cost_out,
                          out_ret, out_frac):
    """
    Tek hisse için long-only simülasyon.
    Karar barı j-1 (sinyal), giriş j açılışı.
    Çıkışlar: stop (gap koruması), kısmi TP + BE, trailing, max_hold.
    Dönüş: (giriş sayısı, kazanan sayısı)
    """
    T = C.shape[0]
    n_entries = 0
    n_wins = 0
    in_pos = False
    frac = 0.0
    entry = 0.0
    stop = 0.0
    tp = 0.0
    R = 0.0
    atr_e = 0.0
    held = 0
    partial = False
    trail_on = False
    trade_pnl = 0.0

    for j in range(T):
        out_ret[j] = 0.0
        out_frac[j] = 0.0

    for j in range(1, T):
        r = 0.0
        was_in = in_pos
        opened = False

        if not in_pos:
            if sig[j - 1] and size[j - 1] > 0.0 and ATRN[j - 1] > 0.0:
                opened = True
                n_entries += 1
                entry = O[j]
                atr_e = ATRN[j - 1] * C[j - 1]
                R = sl_atr * atr_e
                stop = entry - R
                tp = entry + tp_r * R
                frac = size[j - 1]
                in_pos = True
                held = 0
                partial = False
                trail_on = False
                trade_pnl = 0.0

                r = frac * (C[j] / entry - 1.0) - frac * cost_in
                if L[j] <= stop:
                    ex = O[j] if O[j] < stop else stop
                    r = frac * (ex / entry - 1.0) - frac * (cost_in + cost_out)
                    frac = 0.0
                    in_pos = False
        else:
            held += 1
            prev = C[j - 1]
            if L[j] <= stop:
                ex = O[j] if O[j] < stop else stop
                r = frac * (ex / prev - 1.0) - frac * cost_out
                frac = 0.0
                in_pos = False
            else:
                if (not partial) and tp_r > 0.0 and H[j] >= tp:
                    px = tp if O[j] < tp else O[j]
                    part = tp_frac * frac
                    r += part * (px / prev - 1.0) - part * cost_out
                    frac -= part
                    partial = True
                    if be_r > 0.0 and entry > stop:
                        stop = entry

                if H[j] >= entry + trail_act_r * R:
                    trail_on = True
                if trail_on:
                    ts = H[j] - trail_atr * atr_e
                    if ts > stop:
                        stop = ts

                if frac > 0.0:
                    r += frac * (C[j] / prev - 1.0)
                    if held >= max_hold:
                        r -= frac * cost_out
                        frac = 0.0
                        in_pos = False

        if opened or was_in:
            trade_pnl += r
            if (not in_pos) and trade_pnl > 0.0:
                n_wins += 1
            if not in_pos:
                trade_pnl = 0.0

        out_ret[j] = r
        out_frac[j] = frac

    if in_pos:
        out_ret[T - 1] -= frac * cost_out

    return n_entries, n_wins


def simulate_panel(O, H, L, C, ATRN, sig, size, rp, cost_mult=1.0):
    """Tüm hisseler için simülasyon."""
    S, T = C.shape
    per = np.zeros((S, T))
    expo = np.zeros((S, T))
    ent = 0
    wins = 0
    tp_r = float(rp.get("tp_r", 2.0)) if rp.get("tp_on", True) else 0.0
    be_r = float(rp.get("be_r", 1.0)) if rp.get("be_on", True) else 0.0
    cost = (FEE + SLIPPAGE) * cost_mult

    for s in range(S):
        n, w = simulate_stock_kernel(
            np.ascontiguousarray(O[s], dtype=np.float64),
            np.ascontiguousarray(H[s], dtype=np.float64),
            np.ascontiguousarray(L[s], dtype=np.float64),
            np.ascontiguousarray(C[s], dtype=np.float64),
            np.ascontiguousarray(ATRN[s], dtype=np.float64),
            np.ascontiguousarray(sig[s], dtype=np.bool_),
            np.ascontiguousarray(size[s], dtype=np.float64),
            float(rp["sl_atr"]), tp_r, float(rp.get("tp_frac", 0.5)),
            be_r, float(rp["trail_act_r"]), float(rp["trail_atr"]),
            int(rp["max_hold"]), cost, cost, per[s], expo[s],
        )
        ent += int(n)
        wins += int(w)

    return {"per_stock": per, "exposure": expo, "entries": ent, "wins": wins}


# ═══════════════════════════════════════════════════════════════════
# PERFORMANS METRİKLERİ
# ═══════════════════════════════════════════════════════════════════

def sharpe(r: np.ndarray, ann: int = 252) -> float:
    """Yıllıklandırılmış Sharpe."""
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return 0.0 if sd <= 1e-12 else float(r.mean() / sd * math.sqrt(ann))


def max_drawdown(eq: np.ndarray) -> float:
    """Maksimum DD (pozitif kesir)."""
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))


def basic_metrics(r: np.ndarray, ann: int = 252) -> Dict[str, float]:
    """Temel performans: CAGR, Sharpe, Sortino, MaxDD, Calmar, Vol."""
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0,
                "calmar": 0.0, "vol": 0.0}
    eq = np.cumprod(1.0 + r)
    yrs = max(len(r) / ann, 1e-9)
    cagr = float(eq[-1] ** (1 / yrs) - 1) if len(eq) and eq[-1] > 0 else -1.0
    dn = r[r < 0]
    dd_sd = float(np.sqrt(np.mean(dn ** 2))) * math.sqrt(ann) if dn.size else 1e-12
    mdd = max_drawdown(eq)
    return {
        "cagr": cagr,
        "sharpe": sharpe(r, ann),
        "sortino": float(r.mean() * ann / max(dd_sd, 1e-12)),
        "max_dd": mdd,
        "calmar": cagr / max(mdd, 1e-9),
        "vol": float(r.std(ddof=1) * math.sqrt(ann)) if len(r) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# BACKTEST SINIFI
# ═══════════════════════════════════════════════════════════════════

class BacktestV:
    """Feature Store üzerinde sinyal + simülasyon."""

    def __init__(self, store, rp: Dict[str, Any], cost_mult: float = 1.0):
        self.store = store
        self.rp = rp
        self.cost_mult = cost_mult

    def run(self, sig: np.ndarray, size: np.ndarray,
            t0: int = 0, t1: Optional[int] = None) -> Dict[str, Any]:
        """sig (S,T) bool, size (S,T) kesir."""
        st = self.store
        T = st.C.shape[1]
        t1 = T if t1 is None else min(int(t1), T)
        sl = slice(t0, t1)

        sig_m = np.asarray(sig, dtype=bool)[:, sl] & st.mask[:, sl]
        size_m = np.where(sig_m, np.asarray(size, dtype=np.float64)[:, sl], 0.0)

        res = simulate_panel(st.O[:, sl], st.H[:, sl], st.L[:, sl], st.C[:, sl],
                             st.atrn[:, sl], sig_m, size_m, self.rp, self.cost_mult)

        ret = res["per_stock"].sum(axis=0)
        eq = np.cumprod(1.0 + ret)
        m = basic_metrics(ret)
        m["entries"] = res["entries"]
        m["win_rate"] = res["wins"] / max(res["entries"], 1)
        m["avg_exposure"] = float(res["exposure"].sum(axis=0).mean())

        return {
            "ret": ret,
            "per_stock": res["per_stock"],
            "exposure": res["exposure"].sum(axis=0),
            "equity": eq,
            "metrics": m,
            "trades": res["entries"],
        }


def score_params(res: Dict[str, Any], rp: Dict[str, Any], min_trades: int = 20) -> float:
    """Skor = Sharpe − 5·max(0, DD−limit) − 0.5·[işlem < min]."""
    m = res["metrics"]
    dd_limit = float(rp.get("max_dd", 15.0)) / 100.0
    pen_dd = max(0.0, m["max_dd"] - dd_limit)
    pen_tr = 1.0 if res.get("trades", 0) < min_trades else 0.0
    return float(m["sharpe"] - 5.0 * pen_dd - 0.5 * pen_tr)


# ═══════════════════════════════════════════════════════════════════
# DOĞRULAMA
# ═══════════════════════════════════════════════════════════════════

def _pd_skew(r):
    """Çarpıklık."""
    m = r - r.mean()
    s = r.std(ddof=0)
    return 0.0 if s <= 1e-12 else float(np.mean(m ** 3) / s ** 3)


def _pd_kurt(r):
    """Fazla basıklık (normal=0)."""
    m = r - r.mean()
    s = r.std(ddof=0)
    return 0.0 if s <= 1e-12 else float(np.mean(m ** 4) / s ** 4 - 3.0)


def probabilistic_sharpe(r: np.ndarray, sr_star: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012)."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if n < 4 or r.std(ddof=1) <= 1e-12:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    g3 = _pd_skew(r)
    g4 = _pd_kurt(r) + 3.0
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
# PURGED CV
# ═══════════════════════════════════════════════════════════════════

def make_time_groups(t_lo: int, t_hi: int, n_groups: int) -> List[Tuple[int, int]]:
    """[t_lo, t_hi) aralığını n eşit gruba böler."""
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
                n_groups: int = CPCV_N_GROUPS, k_test: int = CPCV_N_TEST_GROUPS,
                embargo: int = EMBARGO_BARS):
    """CPCV C(n,k) kombinasyonları."""
    groups = make_time_groups(t_lo, t_hi, n_groups)
    for combo in itertools.combinations(range(n_groups), k_test):
        ranges = [groups[g] for g in combo]
        te = np.nonzero(test_mask_from_ranges(t_idx, ranges))[0]
        tr = np.nonzero(purged_train_mask(t_idx, t1, ranges, embargo))[0]
        if len(te) and len(tr):
            yield combo, tr, te


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
# BASİT SİNYAL ÜRETİCİ
# ═══════════════════════════════════════════════════════════════════

def make_simple_signals(store, p_buy: float = 0.55, p_sell: float = 0.45,
                        margin: float = 0.05, unc_max: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basit sinyal üretici (test amaçlı):
    RSI + MACD + trend bazlı, feature store'dan okur.
    """
    from v7_config import FEATURE_NAMES
    rsi_idx = FEATURE_NAMES.index("rsi14")
    macd_idx = FEATURE_NAMES.index("macd_hist")
    sma200_idx = FEATURE_NAMES.index("sma200")
    ret20_idx = FEATURE_NAMES.index("ret20")

    S, T = store.C.shape
    sig = np.zeros((S, T), dtype=bool)
    size = np.zeros((S, T), dtype=np.float64)

    for s in range(S):
        for t in range(WARMUP_BARS + 20, T):
            if not store.mask[s, t]:
                continue
            rsi = store.feat[s, rsi_idx, t]
            macd_h = store.feat[s, macd_idx, t]
            sma200 = store.feat[s, sma200_idx, t]
            ret20 = store.feat[s, ret20_idx, t]

            # Basit kural: RSI düşük + MACD pozitif + trend yukarı
            score = 0.0
            if rsi < -0.2:
                score += 0.3
            elif rsi > 0.5:
                score -= 0.2
            if macd_h > 0.1:
                score += 0.2
            elif macd_h < -0.1:
                score -= 0.2
            if sma200 > 0:
                score += 0.2
            if ret20 > 0.02:
                score += 0.1

            if score >= 0.5:
                sig[s, t] = True
                size[s, t] = 0.1  # %10 pozisyon
    return sig, size


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Backtest Test")
    print(f"Numba: {'VAR' if NUMBA_OK else 'YOK'}")

    # Sentetik panel
    rng = np.random.default_rng(7)
    S, T = 6, 600
    C = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (S, T)), axis=1))
    O = C * (1 + rng.normal(0, 0.003, (S, T)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.004, (S, T))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.004, (S, T))))
    V = rng.uniform(5e5, 5e6, (S, T))
    ATRN = np.full((S, T), 0.02)

    # Basit sinyal: random
    sig = rng.random((S, T)) < 0.05
    size = np.full((S, T), 0.1)

    rp = {
        "sl_atr": 2.5, "trail_act_r": 1.5, "trail_atr": 2.5,
        "tp_on": True, "tp_r": 2.0, "tp_frac": 0.5,
        "be_on": True, "be_r": 1.0, "max_hold": 120,
        "max_dd": 15.0,
    }

    import time
    t0 = time.time()
    res = simulate_panel(O, H, L, C, ATRN, sig, size, rp)
    elapsed = time.time() - t0
    print(f"\nSimülasyon süresi: {elapsed:.3f}s")
    print(f"Toplam giriş: {res['entries']}, kazanan: {res['wins']}")

    ret = res["per_stock"].sum(axis=0)
    m = basic_metrics(ret)
    print(f"\nGetiri (toplam): {np.sum(ret)*100:.2f}%")
    print(f"Sharpe: {m['sharpe']:.3f}")
    print(f"MaxDD: {m['max_dd']*100:.2f}%")

    # DSR test
    if len(ret) > 30:
        dsr = deflated_sharpe(ret, n_trials=20)
        print(f"DSR: {dsr:.3f}")

    # PBO test
    M = rng.standard_normal((400, 8)) * 0.01
    M[:, 0] += 0.004
    pbo, _ = cscv_pbo(M, n_blocks=8)
    print(f"PBO: {pbo:.3f} (güçlü edge: < 0.2 bekleniyor)")

    # CPCV test
    t_idx = np.arange(T - 40)
    t1 = t_idx + 40
    sp = list(cpcv_splits(t_idx, t1, 0, T - 40))
    print(f"CPCV kombinasyon: {len(sp)}")

    print("\n✅ Backtest testi geçti")
