# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — FEATURE STORE
306 özellik · RobustScaler · Triple-Barrier · Benzersizlik Ağırlıkları
Look-ahead koruması: scaler train-only, purge/embargo
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from v7_config import (
    CAUSAL_NAMES, CORR_WINDOW, EMBARGO_BARS, FEATURE_NAMES, HORIZON, HORIZON_LONG,
    LIQ_NAMES, LOG_ADV_NORM, MACRO_KEYS, MACRO_NAMES, MICRO_NAMES, N_FEAT, N_TECH,
    NLP_NAMES, ONEHOT_NAMES, PEER_NAMES, PEER_K, RANK_NAMES, REGIMES, SECTORS,
    STOCK_LIST, TB_K, TB_K_LONG, TECH_ALL_NAMES, TIME_DECAY_HALF_LIFE, WARMUP_BARS,
)
from v7_numba_micro import compute_micro_panel
from v7_numba_tech import compute_tech_panel

log = logging.getLogger("v7.features")


# ═══════════════════════════════════════════════════════════════════
# ROBUST SCALER (median / IQR, train-only fit)
# ═══════════════════════════════════════════════════════════════════

class RobustScaler:
    """Median/IQR tabanlı ölçekleme. Yalnızca eğitim bölümünde fit."""

    def __init__(self, clip: float = 5.0, no_scale: Optional[np.ndarray] = None) -> None:
        self.clip = float(clip)
        self.no_scale = no_scale
        self.med: Optional[np.ndarray] = None
        self.iqr: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    def fit(self, feat: np.ndarray, valid: np.ndarray, t_lo: int, t_hi: int,
            max_samples: int = 200_000, seed: int = 0) -> "RobustScaler":
        """feat (S,F,T), valid (S,T) bool. [t_lo, t_hi) aralığında fit eder."""
        with self._lock:
            S, F, T = feat.shape
            t_hi = min(t_hi, T)
            sub = feat[:, :, t_lo:t_hi]
            m = valid[:, t_lo:t_hi]
            s_idx, t_idx = np.nonzero(m)
            if len(s_idx) == 0:
                raise ValueError("Scaler fit için geçerli örnek yok")
            if len(s_idx) > max_samples:
                rng = np.random.default_rng(seed)
                pick = rng.choice(len(s_idx), max_samples, replace=False)
                s_idx, t_idx = s_idx[pick], t_idx[pick]
            X = sub[s_idx, :, t_idx].astype(np.float64)
            med = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr < 1e-8, 1.0, iqr)
            if self.no_scale is not None:
                med = np.where(self.no_scale, 0.0, med)
                iqr = np.where(self.no_scale, 1.0, iqr)
            self.med = med.astype(np.float32)
            self.iqr = iqr.astype(np.float32)
            log.info("RobustScaler fit: %d örnek, %d özellik", X.shape[0], F)
            return self

    def transform_inplace(self, feat: np.ndarray) -> np.ndarray:
        """(S,F,T) tensörünü yerinde ölçekler."""
        if self.med is None or self.iqr is None:
            raise RuntimeError("Scaler fit edilmedi")
        feat -= self.med[None, :, None]
        feat /= self.iqr[None, :, None]
        np.clip(feat, -self.clip, self.clip, out=feat)
        return feat

    def save(self, path: str) -> None:
        """npz olarak kaydeder."""
        np.savez(path, med=self.med, iqr=self.iqr, clip=self.clip)

    @classmethod
    def load(cls, path: str) -> "RobustScaler":
        """npz'den yükler."""
        z = np.load(path)
        sc = cls(clip=float(z["clip"]))
        sc.med = z["med"]
        sc.iqr = z["iqr"]
        return sc


# ═══════════════════════════════════════════════════════════════════
# TRIPLE-BARRIER (López de Prado 2018)
# ═══════════════════════════════════════════════════════════════════

def triple_barrier(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   ATRN: np.ndarray, horizon: int, k_pt: float, k_sl: float):
    """
    Triple-barrier etiketleme. Giriş: t+1 AÇILIŞI.
    Etiket: 0 yatay, 1 yukarı, 2 aşağı, -1 etiketsiz.
    Dönüş: (labels, t1, rets) — hepsi (S,T)
    """
    O = np.ascontiguousarray(O, dtype=np.float64)
    H = np.ascontiguousarray(H, dtype=np.float64)
    L = np.ascontiguousarray(L, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    ATRN = np.ascontiguousarray(ATRN, dtype=np.float64)

    S, T = C.shape
    labels = np.full((S, T), -1, dtype=np.int64)
    t1_arr = np.full((S, T), -1, dtype=np.int64)
    rets = np.zeros((S, T), dtype=np.float64)

    for s in range(S):
        for t in range(T):
            if t + 1 >= T:
                continue
            a = ATRN[s, t]
            if not (a > 0):
                continue
            entry = O[s, t + 1]
            if not (entry > 0):
                continue
            last = t + horizon
            if last >= T:
                continue
            atr = a * C[s, t]
            pt = entry + k_pt * atr
            sl = entry - k_sl * atr
            lab = 0
            ex = C[s, last]
            j_ex = last

            for j in range(t + 1, last + 1):
                # Gap kontrolü önce
                if j > t + 1:
                    if O[s, j] <= sl:
                        lab = 2; ex = O[s, j]; j_ex = j; break
                    if O[s, j] >= pt:
                        lab = 1; ex = O[s, j]; j_ex = j; break
                # Intrabar
                if L[s, j] <= sl:
                    lab = 2; ex = min(sl, O[s, j]); j_ex = j; break
                if H[s, j] >= pt:
                    lab = 1; ex = max(pt, O[s, j]); j_ex = j; break

            labels[s, t] = lab
            t1_arr[s, t] = j_ex
            rets[s, t] = ex / entry - 1.0

    return labels, t1_arr, rets


def uniqueness_weights(lab: np.ndarray, t1: np.ndarray,
                       half_life: float = TIME_DECAY_HALF_LIFE) -> np.ndarray:
    """
    López de Prado benzersizlik ağırlıkları:
    - Eşzamanlılık (concurrency)
    - Benzersizlik (uniqueness)
    - Zaman azalması (time decay)
    Dönüş: (S,T) float32, etiketli örnekler ortalaması 1.
    """
    lab = np.ascontiguousarray(lab, dtype=np.int64)
    t1 = np.ascontiguousarray(t1, dtype=np.int64)
    S, T = lab.shape
    uq = np.zeros((S, T), dtype=np.float64)

    for s in range(S):
        diff = np.zeros(T + 2, dtype=np.float64)
        for t in range(T):
            if lab[s, t] >= 0 and t1[s, t] >= t:
                diff[t] += 1.0
                diff[min(t1[s, t] + 1, T)] -= 1.0
        conc = np.zeros(T, dtype=np.float64)
        run = 0.0
        for t in range(T):
            run += diff[t]
            conc[t] = run
        pref = np.zeros(T + 1, dtype=np.float64)
        for t in range(T):
            pref[t + 1] = pref[t] + (1.0 / conc[t] if conc[t] > 0 else 0.0)
        for t in range(T):
            if lab[s, t] >= 0 and t1[s, t] >= t:
                n = t1[s, t] - t + 1
                uq[s, t] = (pref[t1[s, t] + 1] - pref[t]) / n
            else:
                uq[s, t] = 0.0

    decay = np.power(0.5, (T - 1 - np.arange(T)) / max(half_life, 1.0))
    w = uq * decay[None, :]
    m = lab >= 0
    if m.any():
        mean = w[m].mean()
        if mean > 1e-12:
            w = w / mean
    w[~m] = 0.0
    return w.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# ÖZELLİK BLOKLARI
# ═══════════════════════════════════════════════════════════════════

def _tf(tech: np.ndarray, name: str) -> np.ndarray:
    """Teknik tensörden (S,T) özellik çeker."""
    return tech[:, TECH_ALL_NAMES.index(name), :].astype(np.float64)


def _roll_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Kayan korelasyon (S,T)."""
    a = pd.DataFrame(x.T)
    b = pd.DataFrame(y.T)
    r = a.rolling(w, min_periods=max(5, w // 4)).corr(b).values.T
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _cs_rank(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank (S,T)."""
    df = pd.DataFrame(X.T).where(pd.DataFrame(mask.T))
    r = df.rank(axis=1, pct=True).values.T
    return np.nan_to_num(r, nan=0.5)


def _sector_mean(X: np.ndarray, sector_id: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sektör ortalaması (S,T)."""
    out = np.zeros_like(X, dtype=np.float64)
    for g in np.unique(sector_id):
        sel = sector_id == g
        m = mask & sel[:, None]
        cnt = np.maximum(m.sum(axis=0), 1)
        mean = np.where(m, X, 0.0).sum(axis=0) / cnt
        out[sel, :] = mean[None, :]
    return out


def default_sector_ids(n: int) -> np.ndarray:
    """Varsayılan sektör ataması: 8'erli üç grup."""
    return (np.arange(n) // 8).astype(np.int64).clip(0, 2)


def build_liquidity_block(C, V, LR, ADV, tech, micro) -> np.ndarray:
    """Likidite (15) bloğu."""
    S, T = C.shape
    mi = lambda n: micro[:, MICRO_NAMES.index(n), :].astype(np.float64)
    out = np.zeros((S, len(LIQ_NAMES), T))

    adv = np.nan_to_num(ADV, nan=0.0)
    cv = C * V
    out[:, 0] = np.log1p(adv) / LOG_ADV_NORM
    out[:, 1] = np.clip(cv / (adv + 1.0), 0, 20)
    out[:, 2] = mi("depth_proxy") * mi("queue_bid")
    out[:, 3] = mi("depth_proxy") * mi("queue_ask")

    lag = np.concatenate([np.zeros((S, 1)), LR[:, :-1]], axis=1)
    out[:, 4] = -_roll_corr(LR, lag, 20)
    out[:, 5] = 1.0 / (1.0 + np.abs(mi("spread_quoted")) * 100.0)
    out[:, 6] = np.abs(LR) / (np.abs(_tf(tech, "vol_rel")) + 0.1)
    out[:, 7] = mi("kyle_lambda_ext")
    out[:, 8] = _tf(tech, "amihud")

    dc = np.diff(C, axis=1, prepend=C[:, :1])
    dcl = np.concatenate([np.zeros((S, 1)), dc[:, :-1]], axis=1)
    cov = (pd.DataFrame((dc * dcl).T).rolling(20, min_periods=10).mean().values.T
           - pd.DataFrame(dc.T).rolling(20, min_periods=10).mean().values.T
           * pd.DataFrame(dcl.T).rolling(20, min_periods=10).mean().values.T)
    out[:, 9] = 2.0 * np.sqrt(np.maximum(-np.nan_to_num(cov), 0.0)) / np.maximum(C, 1e-9)
    out[:, 10] = pd.DataFrame(_tf(tech, "cs_spread").T).rolling(5, min_periods=1).mean().values.T
    out[:, 11] = _roll_corr(LR, mi("trade_sign"), 20) ** 2
    out[:, 12] = pd.DataFrame(np.abs(mi("order_imb")).T).rolling(20, min_periods=5).mean().values.T
    out[:, 13] = _roll_corr(V, np.abs(LR), 20)
    out[:, 14] = 0.01 / np.maximum(C, 1e-9)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_macro_block(macro_df: Optional[pd.DataFrame], dates: pd.DatetimeIndex) -> np.ndarray:
    """Makro (30,T) bloğu. Yayın gecikmesi uygulanır."""
    T = len(dates)
    out = np.zeros((len(MACRO_NAMES), T))
    if macro_df is None or len(macro_df) == 0:
        return out

    def on_dates(s: pd.Series, lag: int) -> pd.Series:
        s = s.dropna().copy()
        s.index = s.index + pd.Timedelta(days=lag)
        u = s.index.union(dates)
        return s.reindex(u).ffill().reindex(dates)

    row = 0
    for k in MACRO_KEYS:
        if k in macro_df.columns:
            s = on_dates(macro_df[k].astype(float), 1)
            lv = np.log(s.where(s > 0))
            mu = lv.rolling(60, min_periods=20).mean()
            sd = lv.rolling(60, min_periods=20).std()
            out[row] = ((lv - mu) / sd.replace(0, np.nan)).fillna(0).values
            out[row + 1] = lv.diff(20).fillna(0).values
            out[row + 2] = lv.diff(60).fillna(0).values
        row += 3
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_peer_block(tech, LR, C, corr, mask, sector_id) -> np.ndarray:
    """Peer (25) bloğu (basitleştirilmiş)."""
    S, T = LR.shape
    out = np.zeros((S, len(PEER_NAMES), T))

    ret5 = _tf(tech, "ret5")
    rv30 = _tf(tech, "rv30")
    rsi14 = _tf(tech, "rsi14")
    macd_v = _tf(tech, "macd")
    sec_lr = _sector_mean(LR, sector_id, mask)

    # Peer ret_w = sektör ret
    out[:, 0] = sec_lr
    out[:, 1] = ret5 - _sector_mean(ret5, sector_id, mask)
    out[:, 4] = sec_lr
    out[:, 6] = LR - sec_lr
    out[:, 8] = ret5  # basit peer
    out[:, 11] = rsi14
    out[:, 12] = macd_v
    out[:, 13] = _sector_mean(ret5, sector_id, mask)
    out[:, 14] = _sector_mean(_tf(tech, "ret20"), sector_id, mask)
    out[:, 15] = _sector_mean(rv30, sector_id, mask)
    out[:, 16] = _sector_mean((LR > 0).astype(np.float64), sector_id, mask)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_rank_block(tech, liq, C, mask, ADV) -> np.ndarray:
    """Rank (20) bloğu."""
    S, T = C.shape
    ret5 = _tf(tech, "ret5")
    ret20 = _tf(tech, "ret20")
    ret60 = np.zeros((S, T))
    if T > 60:
        ret60[:, 60:] = C[:, 60:] / np.maximum(C[:, :-60], 1e-9) - 1.0

    cols = [
        ret5, ret20, ret60,
        _tf(tech, "rv30"), _tf(tech, "vol_rel"),
        _tf(tech, "rsi14"), _tf(tech, "macd"), _tf(tech, "bb_pct"),
        _tf(tech, "amihud"), liq[:, 1],
    ]
    out = np.full((S, len(RANK_NAMES), T), 0.5)
    for i, x in enumerate(cols):
        if i < len(RANK_NAMES):
            out[:, i] = _cs_rank(x, mask)
    return out


def build_onehot_block(codes, sector_id, regime, dates) -> np.ndarray:
    """One-hot (40) bloğu."""
    S, T = len(codes), len(dates)
    out = np.zeros((S, len(ONEHOT_NAMES), T))
    n_stk = len(STOCK_LIST)
    for s, c in enumerate(codes):
        j = STOCK_LIST.index(c) if c in STOCK_LIST else s
        if j < n_stk:
            out[s, j] = 1.0
    off = n_stk
    for s in range(S):
        out[s, off + min(int(sector_id[s]), len(SECTORS) - 1)] = 1.0
    off += len(SECTORS)
    for t in range(T):
        out[:, off + int(regime[t]), t] = 1.0
    off += len(REGIMES)
    # cap (small=mid, mid=1)
    out[:, off + 1] = 1.0
    off += 3
    # time_close = 1 (günlük bar)
    out[:, off + 2] = 1.0
    return out


def compute_regime(LR: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nedensel piyasa rejimi (T,)."""
    cnt = np.maximum(mask.sum(axis=0), 1)
    mkt = np.where(mask, LR, 0.0).sum(axis=0) / cnt
    s = pd.Series(mkt)
    rv = s.rolling(20, min_periods=10).std() * math.sqrt(252)
    med = rv.rolling(250, min_periods=30).median()
    r60 = s.rolling(60, min_periods=20).sum()
    reg = np.full(len(s), 3, dtype=np.int64)
    reg[(r60 > 0.05).values] = 0  # BULL
    reg[(r60 < -0.05).values] = 1  # BEAR
    reg[(rv > 1.5 * med).fillna(False).values] = 2  # VOL
    return reg


# ═══════════════════════════════════════════════════════════════════
# FEATURE STORE SINIFI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureStore:
    codes: List[str]
    dates: pd.DatetimeIndex
    valid: np.ndarray
    mask: np.ndarray
    feat: np.ndarray  # (S, N_FEAT, T)
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    LR: np.ndarray
    ADV: np.ndarray
    atrn: np.ndarray
    micro_raw: np.ndarray
    corr: np.ndarray
    sector_id: np.ndarray
    n_sector: int
    regime: np.ndarray
    lab_s: np.ndarray
    lab_l: np.ndarray
    t1_s: np.ndarray
    t1_l: np.ndarray
    ret_s: np.ndarray
    ret_l: np.ndarray
    w_s: np.ndarray
    w_l: np.ndarray
    scaler: Optional[RobustScaler]
    train_end: int
    stock_idx: Dict[str, int] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.C.shape[1]

    @property
    def S(self) -> int:
        return self.C.shape[0]

    def adjacency_at(self, t: int, topk: int = 4):
        """GAT komşuluk matrisi."""
        return build_adjacency(self.corr[t], self.sector_id, self.mask[:, t], topk)

    def sample_index(self, horizon: str = "long", t_lo: int = 0,
                     t_hi: Optional[int] = None):
        """Etiketli örnek indeksleri."""
        lab = self.lab_l if horizon == "long" else self.lab_s
        t1 = self.t1_l if horizon == "long" else self.t1_s
        hi = self.T if t_hi is None else t_hi
        m = (lab >= 0) & self.mask
        m[:, :t_lo] = False
        m[:, hi:] = False
        s_idx, t_idx = np.nonzero(m)
        return s_idx, t_idx, t1[s_idx, t_idx]


def build_adjacency(corr, sector_id, valid, topk=4):
    """GAT adjacency."""
    S = corr.shape[0]
    valid = np.asarray(valid, dtype=bool)
    same = (sector_id[:, None] == sector_id[None, :]) & valid[:, None] & valid[None, :]
    np.fill_diagonal(same, False)
    c = corr.astype(np.float64).copy()
    np.fill_diagonal(c, -2.0)
    c[~valid, :] = -2.0
    c[:, ~valid] = -2.0
    k = max(1, min(topk, S - 1))
    idx = np.argpartition(-c, k - 1, axis=1)[:, :k]
    knn = np.zeros((S, S), dtype=bool)
    knn[np.repeat(np.arange(S), k), idx.ravel()] = True
    knn &= (c > -1.5)
    knn = knn | knn.T
    eye = np.eye(S, dtype=bool)
    adj = same | knn | eye
    etype = same.astype(np.int8) + 2 * knn.astype(np.int8)
    etype[eye] = 4
    ew = np.where(adj, corr, 0.0).astype(np.float32)
    return adj, etype, ew


def compute_corr_tensor(C: np.ndarray, window: int = CORR_WINDOW) -> np.ndarray:
    """Kayan korelasyon tensörü (T,S,S)."""
    S, T = C.shape
    LR = np.zeros((S, T))
    for s in range(S):
        for t in range(1, T):
            if C[s, t] > 0 and C[s, t - 1] > 0:
                LR[s, t] = math.log(C[s, t] / C[s, t - 1])

    out = np.zeros((T, S, S), dtype=np.float32)
    for t in range(T):
        for a in range(S):
            for b in range(S):
                out[t, a, b] = 1.0 if a == b else 0.0
        if t < window - 1:
            continue
        lo = t - window + 1
        win = LR[:, lo:t + 1]
        mean = win.mean(axis=1, keepdims=True)
        std = win.std(axis=1)
        for a in range(S):
            for b in range(a + 1, S):
                if std[a] > 1e-9 and std[b] > 1e-9:
                    cov = np.mean((win[a] - mean[a, 0]) * (win[b] - mean[b, 0]))
                    r = cov / (std[a] * std[b])
                    out[t, a, b] = r
                    out[t, b, a] = r
    return out


# ═══════════════════════════════════════════════════════════════════
# ANA BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_feature_store(panel: Dict, macro_df: Optional[pd.DataFrame] = None,
                        train_frac: float = 0.7,
                        scaler: Optional[RobustScaler] = None,
                        sector_id: Optional[np.ndarray] = None,
                        scale: bool = True) -> FeatureStore:
    """Panelden Feature Store kurar."""
    t0 = time.time()
    O = np.ascontiguousarray(panel["O"], dtype=np.float64)
    H = np.ascontiguousarray(panel["H"], dtype=np.float64)
    L = np.ascontiguousarray(panel["L"], dtype=np.float64)
    C = np.ascontiguousarray(panel["C"], dtype=np.float64)
    V = np.ascontiguousarray(panel["V"], dtype=np.float64)
    dates = panel["dates"]
    codes = list(panel["codes"])
    avail = panel["avail"]
    S, T = C.shape

    sector_id = default_sector_ids(S) if sector_id is None else np.asarray(sector_id, dtype=np.int64)

    first = np.where(avail.any(axis=1), avail.argmax(axis=1), T)
    mask = avail & (np.arange(T)[None, :] >= (first + WARMUP_BARS)[:, None])
    valid = mask.any(axis=1)

    feat = np.zeros((S, N_FEAT, T), dtype=np.float32)

    # Teknik (0-110)
    tech = compute_tech_panel(O, H, L, C, V)
    feat[:, :N_TECH] = tech

    # Mikro (111-150)
    micro = compute_micro_panel(H, L, C, V, valid)
    micro_start = len(TECH_ALL_NAMES) + len(LIQ_NAMES) + len(RANK_NAMES) + len(MACRO_NAMES)
    feat[:, micro_start:micro_start + len(MICRO_NAMES)] = micro

    # LR + ADV
    LR = np.zeros((S, T))
    for s in range(S):
        for t in range(1, T):
            if C[s, t] > 0 and C[s, t - 1] > 0:
                LR[s, t] = math.log(C[s, t] / C[s, t - 1])
    ADV = np.full((S, T), np.nan)
    for s in range(S):
        for t in range(19, T):
            ADV[s, t] = np.mean(C[s, t - 19:t + 1] * V[s, t - 19:t + 1])

    atrn = np.ascontiguousarray(tech[:, TECH_ALL_NAMES.index("atrn14"), :])

    # Korelasyon tensörü
    corr = compute_corr_tensor(C, CORR_WINDOW)

    # Likidite
    liq = build_liquidity_block(C, V, LR, ADV, tech, micro)
    off_liq = len(TECH_ALL_NAMES)
    feat[:, off_liq:off_liq + len(LIQ_NAMES)] = liq

    # Rank
    rk = build_rank_block(tech, liq, C, mask, ADV)
    off_rank = off_liq + len(LIQ_NAMES)
    feat[:, off_rank:off_rank + len(RANK_NAMES)] = rk

    # Makro
    mac = build_macro_block(macro_df, dates)
    off_mac = off_rank + len(RANK_NAMES)
    feat[:, off_mac:off_mac + len(MACRO_NAMES)] = mac[None, :, :]

    # Peer
    peer = build_peer_block(tech, LR, C, corr, mask, sector_id)
    off_peer = micro_start + len(MICRO_NAMES)
    feat[:, off_peer:off_peer + len(PEER_NAMES)] = peer

    # NLP (şimdilik 0)
    off_nlp = off_peer + len(PEER_NAMES)

    # One-hot
    regime = compute_regime(LR, mask)
    oh = build_onehot_block(codes, sector_id, regime, dates)
    off_oh = off_nlp + len(NLP_NAMES)
    feat[:, off_oh:off_oh + len(ONEHOT_NAMES)] = oh

    # Causal (şimdilik 0)
    # Son blok zaten 0 kalacak

    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    train_end = int(T * train_frac)

    # Etiketler
    lab_s, t1_s, ret_s = triple_barrier(O, H, L, C, atrn, HORIZON, TB_K, TB_K)
    lab_l, t1_l, ret_l = triple_barrier(O, H, L, C, atrn, HORIZON_LONG, TB_K_LONG, TB_K_LONG)
    w_s = uniqueness_weights(lab_s, t1_s)
    w_l = uniqueness_weights(lab_l, t1_l)

    # Scaler
    if scale:
        no_scale = np.zeros(N_FEAT, dtype=bool)
        no_scale[off_oh:off_oh + len(ONEHOT_NAMES)] = True
        if scaler is None:
            scaler = RobustScaler(clip=5.0, no_scale=no_scale)
            scaler.fit(feat, mask, WARMUP_BARS, train_end)
        scaler.transform_inplace(feat)

    feat *= mask[:, None, :]

    store = FeatureStore(
        codes=codes, dates=dates, valid=valid, mask=mask,
        feat=np.ascontiguousarray(feat),
        O=O, H=H, L=L, C=C, V=V, LR=LR,
        ADV=np.nan_to_num(ADV, nan=0.0),
        atrn=atrn, micro_raw=micro, corr=corr,
        sector_id=sector_id, n_sector=int(sector_id.max()) + 1,
        regime=regime,
        lab_s=lab_s, lab_l=lab_l,
        t1_s=t1_s, t1_l=t1_l,
        ret_s=ret_s, ret_l=ret_l,
        w_s=w_s, w_l=w_l,
        scaler=scaler if scale else None,
        train_end=train_end,
        stock_idx={c: i for i, c in enumerate(codes)},
    )
    log.info("FeatureStore hazır: S=%d F=%d T=%d (%.1fs)", S, N_FEAT, T, time.time() - t0)
    return store


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Feature Store Test")
    print(f"Toplam özellik: {N_FEAT}")

    rng = np.random.default_rng(11)
    S, T = 6, 420
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (S, T)), axis=1))
    O = C * (1 + rng.normal(0, 0.003, (S, T)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.004, (S, T))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.004, (S, T))))
    V = rng.uniform(5e5, 5e6, (S, T))

    dates = pd.bdate_range("2023-01-02", periods=T)
    panel = {
        "O": O, "H": H, "L": L, "C": C, "V": V,
        "dates": dates,
        "codes": [f"S{i}" for i in range(S)],
        "avail": np.ones((S, T), dtype=bool),
        "valid": np.ones(S, dtype=bool),
    }

    # Sentetik makro
    cols = {k: pd.Series(50 * np.exp(np.cumsum(0.01 * rng.standard_normal(T))), index=dates)
            for k in MACRO_KEYS}
    macro = pd.DataFrame(cols)

    t0 = time.time()
    store = build_feature_store(panel, macro)
    elapsed = time.time() - t0

    print(f"\nŞekil: {store.feat.shape}")
    print(f"Süre: {elapsed:.2f}s")
    print(f"NaN/Inf: {np.isnan(store.feat).any() or np.isinf(store.feat).any()}")
    print(f"Korelasyon tensörü: {store.corr.shape}")
    print(f"Etiket aralığı: {set(np.unique(store.lab_s))}")
    print(f"Ağırlık ortalaması: {store.w_s[store.w_s > 0].mean():.3f}")

    # Sample index test
    s_i, t_i, t1_i = store.sample_index("long", WARMUP_BARS, store.train_end)
    print(f"\nÖrnek sayısı (train): {len(s_i)}")

    # Adjacency test
    adj, et, ew = store.adjacency_at(T - 1)
    print(f"Adjacency simetrik: {np.array_equal(adj, adj.T)}")
    print(f"Self-loop: {adj.diagonal().all()}")

    print("\n✅ Feature Store testi geçti")
