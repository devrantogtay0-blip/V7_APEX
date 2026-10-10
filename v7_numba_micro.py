# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — NUMBA MİKRO YAPI (40 özellik)
OFI · Kyle Lambda · Hawkes · Iceberg · Spoof · LOB · HHI
Look-ahead YOK: rolling istatistikler i-1'e kadar
"""
from __future__ import annotations

import logging
import math

import numpy as np

from v7_config import HAWKES_ALPHA, HAWKES_BETA, HAWKES_MU, MICRO_NAMES, OFI_WINDOW

log = logging.getLogger("v7.numba_micro")

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
# LOG RETİRİ (yardımcı)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def _logret(c, out):
    """Log getiri."""
    n = c.shape[0]
    out[0] = 0.0
    for i in range(1, n):
        if c[i] > 0 and c[i - 1] > 0:
            out[i] = math.log(c[i] / c[i - 1])
        else:
            out[i] = 0.0


# ═══════════════════════════════════════════════════════════════════
# ANA MİKRO KERNEL — 40 özellik
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def micro_stock_kernel(h, l, c, v, lr, cs, adv, mu, alpha, beta, out):
    """
    Tek hisse için 40 mikro yapı özelliği.
    Tüm rolling istatistikler i-1'e kadar hesaplanır → i. bar eklenir.
    out: (40, T) matrisi
    """
    n = c.shape[0]
    sqrt2 = math.sqrt(2.0)
    decay = math.exp(-beta)

    # Yardımcı bufferlar
    sv = np.zeros(n)   # signed volume
    sl_b = np.zeros(n)  # bid slope
    sl_a = np.zeros(n)  # ask slope
    ofi_hist = np.zeros(n)

    # Rolling durumları
    s1 = 0.0; s2 = 0.0
    e5 = 0.0; e20 = 0.0
    se = 0.0
    ts_prev = 1.0

    # Hawkes durumları
    lam = mu; lam_b = mu; lam_a = mu

    for i in range(n):
        # ── Rolling sigma (i-1'e kadar) ──
        cntp = i if i < OFI_WINDOW else OFI_WINDOW
        if cntp >= 5:
            m = s1 / cntp
            var = s2 / cntp - m * m
            sig = math.sqrt(var) if var > 0 else 0.0
            if sig < 1e-6:
                sig = 0.02
        else:
            sig = 0.02

        # ── OFI (z-score bazlı) ──
        z = lr[i] / sig
        ofi = math.erf(z / sqrt2)
        s1 += lr[i]
        s2 += lr[i] * lr[i]
        if i >= OFI_WINDOW:
            s1 -= lr[i - OFI_WINDOW]
            s2 -= lr[i - OFI_WINDOW] * lr[i - OFI_WINDOW]
        ofi_hist[i] = ofi

        # ── EMA'lar ──
        e5 = ofi if i == 0 else e5 + (2.0 / 6.0) * (ofi - e5)
        e20 = ofi if i == 0 else e20 + (2.0 / 21.0) * (ofi - e20)

        # ── Spread EMA ──
        sp = cs[i] if not np.isnan(cs[i]) else 0.0
        se = sp if i == 0 else se + (2.0 / 11.0) * (sp - se)

        # ── Multi-level OFI (1, 3, 5 bar) ──
        a1 = ofi
        a3 = 0.0; c3 = 0
        a5 = 0.0; c5 = 0
        for j in range(5):
            if i - j >= 0:
                a5 += ofi_hist[i - j]
                c5 += 1
                if j < 3:
                    a3 += ofi_hist[i - j]
                    c3 += 1
        ofi_l1 = a1
        ofi_l2 = a3 / c3 if c3 > 0 else 0.0
        ofi_l3 = a5 / c5 if c5 > 0 else 0.0

        # ── Yardımcı nicelikler ──
        rng = max(h[i] - l[i], 1e-9 * max(c[i], 1e-9))
        pos = min(max((c[i] - l[i]) / rng, 0.0), 1.0)
        pc = c[i - 1] if i > 0 else c[i]

        # Göreli hacim (20 bar)
        vm = 0.0; cnt = 0
        for j in range(1, 21):
            if i - j >= 0:
                vm += v[i - j]
                cnt += 1
        vr = v[i] / (vm / cnt + 1.0) if cnt > 0 else 1.0
        vrc = min(vr, 10.0)

        # Mid price
        mid = (h[i] + l[i]) / 2.0

        # ── 0-2: OFI + EMAlar ──
        out[0, i] = ofi
        out[1, i] = e5
        out[2, i] = e20

        # ── 3-5: Multi-level OFI ──
        out[3, i] = ofi_l1
        out[4, i] = ofi_l2
        out[5, i] = ofi_l3

        # ── 6-8: Queue proxy'leri ──
        out[6, i] = pos
        out[7, i] = 1.0 - pos
        out[8, i] = 1.0 - abs(2.0 * pos - 1.0)

        # ── 9-11: Iceberg detection ──
        ice = max(vrc - 1.0, 0.0) * (1.0 - min(abs(z), 1.0))
        out[9, i] = ice if lr[i] <= 0 else 0.0
        out[10, i] = ice if lr[i] > 0 else 0.0
        out[11, i] = math.tanh(ice)

        # ── 12-14: Spoof proxy ──
        lw = (min(c[i], pc) - l[i]) / rng
        uw = (h[i] - max(c[i], pc)) / rng
        sb = max(lw, 0.0) * vrc
        sa = max(uw, 0.0) * vrc
        out[12, i] = sb
        out[13, i] = sa
        out[14, i] = max(sb, sa)

        # ── 15-17: Micro-price ──
        out[15, i] = ((pos * h[i] + (1.0 - pos) * l[i]) / c[i] - 1.0) if c[i] > 0 else 0.0
        out[16, i] = ((mid + c[i]) / 2.0 / c[i] - 1.0) if c[i] > 0 else 0.0
        out[17, i] = (mid / c[i] - 1.0) if c[i] > 0 else 0.0

        # ── 18-20: Spread ──
        sq = rng / c[i] if c[i] > 0 else 0.0
        out[18, i] = 2.0 * abs(c[i] - mid) / mid if mid > 0 else 0.0
        out[19, i] = sq
        out[20, i] = se

        # ── 21: Depth proxy ──
        a_i = adv[i] if not np.isnan(adv[i]) else 0.0
        out[21, i] = math.log1p(a_i / max(sq, 1e-4))

        # ── 22: Kyle Lambda (20 bar slope) ──
        sv[i] = (1.0 if lr[i] > 0 else (-1.0 if lr[i] < 0 else 0.0)) * math.sqrt(max(v[i], 0.0))
        lo_j = i - 19 if i >= 19 else 0
        cn = i - lo_j + 1
        if cn >= 5:
            mr = 0.0; mv = 0.0
            for j in range(lo_j, i + 1):
                mr += lr[j]
                mv += sv[j]
            mr /= cn; mv /= cn
            cov = 0.0; vs = 0.0
            for j in range(lo_j, i + 1):
                cov += (lr[j] - mr) * (sv[j] - mv)
                vs += (sv[j] - mv) * (sv[j] - mv)
            out[22, i] = cov / vs * 1e3 if vs > 1e-12 else 0.0
        else:
            out[22, i] = 0.0

        # ── 23-25: Hawkes ──
        ev = 1.0 if abs(z) > 1.5 else 0.0
        lam = mu + (lam - mu) * decay + alpha * ev
        lam_b = mu + (lam_b - mu) * decay + alpha * (ev if lr[i] < 0 else 0.0)
        lam_a = mu + (lam_a - mu) * decay + alpha * (ev if lr[i] > 0 else 0.0)
        out[23, i] = lam
        out[24, i] = lam_b
        out[25, i] = lam_a

        # ── 26-28: Imbalance ──
        ti = 2.0 * pos - 1.0
        vsign = (1.0 if lr[i] > 0 else (-1.0 if lr[i] < 0 else 0.0)) * vrc / 10.0
        out[26, i] = ti
        out[27, i] = vsign
        out[28, i] = 0.5 * (ti + vsign)

        # ── 29-31: Trade sign ──
        tick = 1.0 if c[i] > pc else (-1.0 if c[i] < pc else 0.0)
        lee = 1.0 if c[i] > mid else (-1.0 if c[i] < mid else tick)
        if lee != 0.0:
            ts_prev = lee
        out[29, i] = tick
        out[30, i] = lee
        out[31, i] = ts_prev

        # ── 32-33: LOB slopes ──
        sl_b[i] = math.log1p(pos * vrc / max((c[i] - l[i]) / c[i], 1e-4)) if c[i] > 0 else 0.0
        sl_a[i] = math.log1p((1.0 - pos) * vrc / max((h[i] - c[i]) / c[i], 1e-4)) if c[i] > 0 else 0.0
        out[32, i] = sl_b[i]
        out[33, i] = sl_a[i]

        # ── 34-35: LOB curvature (2. türev) ──
        if i >= 2:
            out[34, i] = sl_b[i] - 2.0 * sl_b[i - 1] + sl_b[i - 2]
            out[35, i] = sl_a[i] - 2.0 * sl_a[i - 1] + sl_a[i - 2]
        else:
            out[34, i] = 0.0
            out[35, i] = 0.0

        # ── 36: LOB asymmetry ──
        out[36, i] = sl_b[i] - sl_a[i]

        # ── 37-39: HHI (5 bar hacim yoğunlaşması) ──
        tv = 0.0; tvb = 0.0; tva = 0.0
        for j in range(5):
            if i - j >= 0:
                tv += v[i - j]
                if lr[i - j] < 0:
                    tvb += v[i - j]
                elif lr[i - j] > 0:
                    tva += v[i - j]
        hb = 0.0; ha = 0.0; hc = 0.0
        for j in range(5):
            if i - j >= 0:
                if tv > 0:
                    hc += (v[i - j] / tv) ** 2
                if lr[i - j] < 0 and tvb > 0:
                    hb += (v[i - j] / tvb) ** 2
                elif lr[i - j] > 0 and tva > 0:
                    ha += (v[i - j] / tva) ** 2
        out[37, i] = hb
        out[38, i] = ha
        out[39, i] = hc


# ═══════════════════════════════════════════════════════════════════
# PANEL KERNEL
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def micro_panel_kernel(H, L, C, V, LR, CS, ADV, valid, mu, alpha, beta, feat):
    """Panel mikro yapı — paralel."""
    S = C.shape[0]
    for s in prange(S):
        if valid[s]:
            micro_stock_kernel(H[s], L[s], C[s], V[s], LR[s], CS[s], ADV[s],
                               mu, alpha, beta, feat[s])


# ═══════════════════════════════════════════════════════════════════
# PANEL YARDIMCILARI
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def logret_panel(C, out):
    """Panel log getiri."""
    for s in prange(C.shape[0]):
        _logret(C[s], out[s])


@njit(cache=True, parallel=True)
def adv_panel(C, V, p, out):
    """Panel ADV (20-gün ort TL hacim)."""
    for s in prange(C.shape[0]):
        for i in range(C.shape[1]):
            if i >= p - 1:
                s_v = 0.0
                for j in range(p):
                    s_v += C[s, i - j] * V[s, i - j]
                out[s, i] = s_v / p


@njit(cache=True, parallel=True)
def cs_panel(H, L, out):
    """Panel Corwin-Schultz."""
    for s in prange(H.shape[0]):
        n = H.shape[1]
        den = 3.0 - 2.0 * math.sqrt(2.0)
        for i in range(1, n):
            if H[s, i] > 0 and L[s, i] > 0 and H[s, i - 1] > 0 and L[s, i - 1] > 0:
                b = math.log(H[s, i] / L[s, i]) ** 2 + math.log(H[s, i - 1] / L[s, i - 1]) ** 2
                hh = max(H[s, i], H[s, i - 1])
                ll = min(L[s, i], L[s, i - 1])
                g = math.log(hh / ll) ** 2
                a = (math.sqrt(2.0 * b) - math.sqrt(b)) / den - math.sqrt(g / den)
                ea = math.exp(a)
                sp = 2.0 * (ea - 1.0) / (1.0 + ea)
                out[s, i] = sp if sp > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# PYTHON SARMALAYICI
# ═══════════════════════════════════════════════════════════════════

def compute_micro_panel(H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
                        valid: np.ndarray = None) -> np.ndarray:
    """
    Panel mikro yapı özellikleri (S, 40, T) float32.
    Gerekli yardımcı nicelikleri (LR, CS, ADV) içeride üretir.
    """
    H = np.ascontiguousarray(H, dtype=np.float64)
    L = np.ascontiguousarray(L, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    S, T = C.shape

    LR = np.zeros((S, T), dtype=np.float64)
    CS = np.full((S, T), np.nan, dtype=np.float64)
    ADV = np.full((S, T), np.nan, dtype=np.float64)

    logret_panel(C, LR)
    cs_panel(H, L, CS)
    adv_panel(C, V, 20, ADV)

    if valid is None:
        valid = np.ones(S, dtype=np.bool_)
    else:
        valid = np.ascontiguousarray(valid, dtype=np.bool_)

    feat = np.zeros((S, len(MICRO_NAMES), T), dtype=np.float64)
    micro_panel_kernel(H, L, C, V, LR, CS, ADV, valid,
                       HAWKES_MU, HAWKES_ALPHA, HAWKES_BETA, feat)

    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(feat, -1e6, 1e6, out=feat)
    return np.ascontiguousarray(feat.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Numba Micro Test")
    print(f"Numba: {'VAR' if NUMBA_OK else 'YOK (saf Python)'}")
    print(f"Mikro özellik sayısı: {len(MICRO_NAMES)}")

    rng = np.random.default_rng(11)
    S, T = 3, 400
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (S, T)), axis=1))
    O = C * (1 + rng.normal(0, 0.003, (S, T)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.004, (S, T))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.004, (S, T))))
    V = rng.uniform(1e5, 1e6, (S, T))

    import time
    t0 = time.time()
    mic = compute_micro_panel(H, L, C, V)
    elapsed = time.time() - t0

    print(f"\nŞekil: {mic.shape}")
    print(f"Süre: {elapsed:.3f}s")
    print(f"NaN/Inf: {np.isnan(mic).any() or np.isinf(mic).any()}")

    # Look-ahead testi (kesilmiş panel aynı sonucu vermeli)
    cut = 300
    mic_cut = compute_micro_panel(H[:, :cut], L[:, :cut], C[:, :cut], V[:, :cut])
    diff = np.abs(mic_cut - mic[:, :, :cut]).max()
    print(f"Look-ahead farkı: {diff:.2e}")
    assert diff < 1e-4, "Look-ahead ihlali!"

    # OFI test
    ofi = mic[0, 0]  # ilk özellik = ofi
    print(f"\nOFI aralık: [{ofi.min():.3f}, {ofi.max():.3f}]")
    assert np.abs(ofi).max() <= 1.0 + 1e-6

    print("\n✅ Numba Micro testi geçti")
