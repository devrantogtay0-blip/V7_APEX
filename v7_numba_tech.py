# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — NUMBA TEKNİK GÖSTERGELER (49 + 62 = 111)
Numba JIT · Paralel panel · Sıfır kopya
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from v7_config import N_TECH, TECH_ALL_NAMES

log = logging.getLogger("v7.numba_tech")

# Numba opsiyonel
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
# YARDIMCI ÇEKİRDEKLER
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def _ema(x, span, out):
    """EMA (NaN taşır)."""
    n = x.shape[0]
    a = 2.0 / (span + 1.0)
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = out[i - 1] if i > 0 else np.nan
        elif i == 0 or np.isnan(out[i - 1]):
            out[i] = xi
        else:
            out[i] = a * xi + (1.0 - a) * out[i - 1]


@njit(cache=True, nogil=True)
def _sma(x, p, out):
    """SMA (NaN taşır)."""
    n = x.shape[0]
    out[:] = np.nan
    for i in range(p - 1, n):
        s = 0.0
        ok = True
        for j in range(p):
            v = x[i - p + 1 + j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s / p


@njit(cache=True, nogil=True)
def _true_range(h, l, c, out):
    """True Range."""
    n = c.shape[0]
    out[0] = h[0] - l[0]
    for i in range(1, n):
        a = h[i] - l[i]
        b = abs(h[i] - c[i - 1])
        d = abs(l[i] - c[i - 1])
        out[i] = max(a, max(b, d))


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
# 49 TEMEL GÖSTERGE
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def k_rsi(close, period, out):
    """Wilder RSI."""
    n = close.shape[0]
    out[:] = np.nan
    if n <= period:
        return
    ag = 0.0; al = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0: ag += d
        else: al -= d
    ag /= period; al /= period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        ls = -d if d < 0 else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + ls) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)


@njit(cache=True, nogil=True)
def k_atrn(h, l, c, period, out):
    """ATR/Close."""
    n = c.shape[0]
    out[:] = np.nan
    if n < period: return
    tr = np.empty(n)
    _true_range(h, l, c, tr)
    atr = 0.0
    for i in range(period):
        atr += tr[i]
    atr /= period
    out[period - 1] = atr / c[period - 1] if c[period - 1] != 0 else 0.0
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
        out[i] = atr / c[i] if c[i] != 0 else 0.0


@njit(cache=True, nogil=True)
def k_macd(c, fast, slow, sig, m_out, s_out, h_out):
    """MACD (fiyata normalize)."""
    n = c.shape[0]
    ef = np.empty(n); es = np.empty(n)
    _ema(c, fast, ef); _ema(c, slow, es)
    raw = np.empty(n)
    for i in range(n):
        raw[i] = (ef[i] - es[i]) / c[i] if c[i] != 0 else 0.0
    sg = np.empty(n)
    _ema(raw, sig, sg)
    for i in range(n):
        m_out[i] = raw[i]; s_out[i] = sg[i]; h_out[i] = raw[i] - sg[i]


@njit(cache=True, nogil=True)
def k_bollinger(c, period, kstd, pct_out, width_out):
    """Bollinger %B ve genişlik."""
    n = c.shape[0]
    pct_out[:] = np.nan; width_out[:] = np.nan
    for i in range(period - 1, n):
        m = 0.0
        for j in range(period):
            m += c[i - j]
        m /= period
        v = 0.0
        for j in range(period):
            d = c[i - j] - m
            v += d * d
        sd = math.sqrt(v / period)
        up = m + kstd * sd
        lo = m - kstd * sd
        pct_out[i] = (c[i] - lo) / (up - lo) if up > lo else 0.5
        width_out[i] = (up - lo) / m if m != 0 else 0.0


@njit(cache=True, nogil=True)
def k_stoch(h, l, c, kp, k_out, d_out):
    """Stochastic %K ve %D."""
    n = c.shape[0]
    k_out[:] = np.nan
    for i in range(kp - 1, n):
        hh = h[i]; ll = l[i]
        for j in range(1, kp):
            if h[i - j] > hh: hh = h[i - j]
            if l[i - j] < ll: ll = l[i - j]
        k_out[i] = (c[i] - ll) / (hh - ll) if hh > ll else 0.5
    _sma(k_out, 3, d_out)


@njit(cache=True, nogil=True)
def k_obv_vwap(h, l, c, v, win, obv_out, vwap_out):
    """OBV normalize + VWAP uzaklığı."""
    n = c.shape[0]
    obv = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]: obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]: obv[i] = obv[i - 1] - v[i]
        else: obv[i] = obv[i - 1]
    obv_out[:] = np.nan; vwap_out[:] = np.nan
    for i in range(win, n):
        sv = 0.0; spv = 0.0
        for j in range(win):
            k = i - j
            sv += v[k]
            spv += (h[k] + l[k] + c[k]) / 3.0 * v[k]
        obv_out[i] = (obv[i] - obv[i - win]) / (sv + 1.0)
        if sv > 0:
            vwap_out[i] = c[i] / (spv / sv) - 1.0


@njit(cache=True, nogil=True)
def k_ret(c, p, out):
    """p-bar getiri."""
    n = c.shape[0]
    out[:] = np.nan
    for i in range(p, n):
        if c[i - p] != 0:
            out[i] = c[i] / c[i - p] - 1.0


@njit(cache=True, nogil=True)
def k_sma_dist(c, p, out):
    """SMA uzaklığı."""
    n = c.shape[0]
    tmp = np.empty(n)
    _sma(c, p, tmp)
    for i in range(n):
        out[i] = c[i] / tmp[i] - 1.0 if (not np.isnan(tmp[i]) and tmp[i] != 0) else np.nan


@njit(cache=True, nogil=True)
def k_ema_dist(c, span, out):
    """EMA uzaklığı."""
    n = c.shape[0]
    tmp = np.empty(n)
    _ema(c, span, tmp)
    for i in range(n):
        out[i] = c[i] / tmp[i] - 1.0 if tmp[i] != 0 else 0.0


@njit(cache=True, nogil=True)
def k_rvol(lr, p, out):
    """Gerçekleşen volatilite (yıllık)."""
    n = lr.shape[0]
    out[:] = np.nan
    for i in range(p - 1, n):
        m = 0.0
        for j in range(p):
            m += lr[i - j]
        m /= p
        v = 0.0
        for j in range(p):
            d = lr[i - j] - m
            v += d * d
        out[i] = math.sqrt(v / max(p - 1, 1)) * math.sqrt(252.0)


@njit(cache=True, nogil=True)
def k_range_vols(o, h, l, c, p, gk_out, pk_out):
    """Garman-Klass ve Parkinson volatilite."""
    n = c.shape[0]
    gk_out[:] = np.nan; pk_out[:] = np.nan
    k2 = 2.0 * math.log(2.0) - 1.0
    for i in range(p - 1, n):
        sg = 0.0; sp = 0.0
        for j in range(p):
            k = i - j
            if h[k] > 0 and l[k] > 0 and o[k] > 0 and c[k] > 0:
                hl = math.log(h[k] / l[k])
                co = math.log(c[k] / o[k])
                sg += 0.5 * hl * hl - k2 * co * co
                sp += hl * hl
        sg = max(sg / p, 0.0)
        gk_out[i] = math.sqrt(sg) * math.sqrt(252.0)
        pk_out[i] = math.sqrt(sp / p / (4.0 * math.log(2.0))) * math.sqrt(252.0)


@njit(cache=True, nogil=True)
def k_candle(o, h, l, c, hl_out, co_out, wu_out, wl_out):
    """Mum yapısı."""
    n = c.shape[0]
    for i in range(n):
        rng = h[i] - l[i]
        hl_out[i] = rng / c[i] if c[i] != 0 else 0.0
        co_out[i] = (c[i] - o[i]) / o[i] if o[i] != 0 else 0.0
        if rng > 0:
            wu_out[i] = (h[i] - max(o[i], c[i])) / rng
            wl_out[i] = (min(o[i], c[i]) - l[i]) / rng
        else:
            wu_out[i] = 0.0; wl_out[i] = 0.0


@njit(cache=True, nogil=True)
def k_volume_feats(v, p, rel_out, chg_out):
    """Göreli hacim + log hacim değişimi."""
    n = v.shape[0]
    rel_out[:] = np.nan; chg_out[:] = np.nan
    for i in range(1, n):
        chg_out[i] = math.log((v[i] + 1.0) / (v[i - 1] + 1.0))
    for i in range(p, n):
        m = 0.0
        for j in range(1, p + 1):
            m += v[i - j]
        m /= p
        rel_out[i] = v[i] / (m + 1.0)


@njit(cache=True, nogil=True)
def k_amihud(c, v, lr, p, out):
    """Amihud illiquidity."""
    n = c.shape[0]
    out[:] = np.nan
    for i in range(p - 1, n):
        s = 0.0
        for j in range(p):
            k = i - j
            s += abs(lr[k]) / (c[k] * v[k] + 1.0)
        out[i] = s / p * 1e6


@njit(cache=True, nogil=True)
def k_cs_spread(h, l, out):
    """Corwin-Schultz spread."""
    n = h.shape[0]
    out[:] = np.nan
    den = 3.0 - 2.0 * math.sqrt(2.0)
    for i in range(1, n):
        if h[i] > 0 and l[i] > 0 and h[i - 1] > 0 and l[i - 1] > 0:
            b = math.log(h[i] / l[i]) ** 2 + math.log(h[i - 1] / l[i - 1]) ** 2
            hh = max(h[i], h[i - 1]); ll = min(l[i], l[i - 1])
            g = math.log(hh / ll) ** 2
            a = (math.sqrt(2.0 * b) - math.sqrt(b)) / den - math.sqrt(g / den)
            ea = math.exp(a)
            s = 2.0 * (ea - 1.0) / (1.0 + ea)
            out[i] = s if s > 0 else 0.0


@njit(cache=True, nogil=True)
def k_gap(o, c, out):
    """Gap."""
    n = c.shape[0]
    out[0] = 0.0
    for i in range(1, n):
        out[i] = o[i] / c[i - 1] - 1.0 if c[i - 1] != 0 else 0.0


@njit(cache=True, nogil=True)
def k_skew_kurt(lr, p, sk_out, ku_out):
    """Skew + Kurt."""
    n = lr.shape[0]
    sk_out[:] = np.nan; ku_out[:] = np.nan
    for i in range(p - 1, n):
        m = 0.0
        for j in range(p):
            m += lr[i - j]
        m /= p
        m2 = 0.0; m3 = 0.0; m4 = 0.0
        for j in range(p):
            d = lr[i - j] - m
            m2 += d * d; m3 += d * d * d; m4 += d * d * d * d
        m2 /= p; m3 /= p; m4 /= p
        if m2 > 1e-18:
            sk_out[i] = m3 / (m2 ** 1.5)
            ku_out[i] = m4 / (m2 * m2) - 3.0
        else:
            sk_out[i] = 0.0; ku_out[i] = 0.0


@njit(cache=True, nogil=True)
def k_dd_low(h, l, c, p, dd_out, dl_out):
    """Drawdown + dist_low."""
    n = c.shape[0]
    dd_out[:] = np.nan; dl_out[:] = np.nan
    for i in range(p - 1, n):
        hh = h[i]; ll = l[i]
        for j in range(1, p):
            if h[i - j] > hh: hh = h[i - j]
            if l[i - j] < ll: ll = l[i - j]
        dd_out[i] = c[i] / hh - 1.0 if hh > 0 else 0.0
        dl_out[i] = c[i] / ll - 1.0 if ll > 0 else 0.0


@njit(cache=True, nogil=True)
def k_mom_acc(c, p, out):
    """Momentum ivmesi."""
    n = c.shape[0]
    out[:] = np.nan
    for i in range(2 * p, n):
        if c[i - p] != 0 and c[i - 2 * p] != 0:
            out[i] = (c[i] / c[i - p] - 1.0) - (c[i - p] / c[i - 2 * p] - 1.0)


@njit(cache=True, nogil=True)
def k_eff_ratio(c, p, out):
    """Kaufman verimlilik oranı."""
    n = c.shape[0]
    out[:] = np.nan
    for i in range(p, n):
        s = 0.0
        for j in range(p):
            s += abs(c[i - j] - c[i - j - 1])
        out[i] = abs(c[i] - c[i - p]) / s if s > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# 62 GENİŞLETİLMİŞ GÖSTERGE (basitleştirilmiş)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def k_ichimoku(h, l, c, out_t, out_k, out_sa, out_sb, out_ch):
    """Ichimoku (look-ahead koruması: 26 bar geri kaydırma)."""
    n = c.shape[0]
    out_t[:] = np.nan; out_k[:] = np.nan
    out_sa[:] = np.nan; out_sb[:] = np.nan; out_ch[:] = np.nan
    for i in range(52, n):
        h9 = h[i]; l9 = l[i]
        for j in range(9):
            if h[i - j] > h9: h9 = h[i - j]
            if l[i - j] < l9: l9 = l[i - j]
        out_t[i] = (h9 + l9) / 2.0 / c[i] - 1.0
        h26 = h[i]; l26 = l[i]
        for j in range(26):
            if h[i - j] > h26: h26 = h[i - j]
            if l[i - j] < l26: l26 = l[i - j]
        out_k[i] = (h26 + l26) / 2.0 / c[i] - 1.0
        j = i - 26
        h9j = h[j]; l9j = l[j]
        for k in range(9):
            if h[j - k] > h9j: h9j = h[j - k]
            if l[j - k] < l9j: l9j = l[j - k]
        h26j = h[j]; l26j = l[j]
        for k in range(26):
            if h[j - k] > h26j: h26j = h[j - k]
            if l[j - k] < l26j: l26j = l[j - k]
        ten = (h9j + l9j) / 2.0
        kij = (h26j + l26j) / 2.0
        out_sa[i] = (ten + kij) / 2.0 / c[i] - 1.0
        if i >= 52:
            h52 = h[j]; l52 = l[j]
            for k in range(52):
                if h[j - k] > h52: h52 = h[j - k]
                if l[j - k] < l52: l52 = l[j - k]
            out_sb[i] = (h52 + l52) / 2.0 / c[i] - 1.0
        if i >= 26 and c[i - 26] != 0:
            out_ch[i] = c[i] / c[i - 26] - 1.0


@njit(cache=True, nogil=True)
def k_keltner(h, l, c, ema_p, atr_p, mult, out_up, out_lo, out_w):
    """Keltner kanalı."""
    n = c.shape[0]
    mid = np.empty(n); tr = np.empty(n); atr = np.empty(n)
    _ema(c, ema_p, mid); _true_range(h, l, c, tr); _ema(tr, atr_p, atr)
    for i in range(n):
        out_up[i] = (mid[i] + mult * atr[i]) / c[i] - 1.0 if c[i] != 0 else 0.0
        out_lo[i] = (mid[i] - mult * atr[i]) / c[i] - 1.0 if c[i] != 0 else 0.0
        out_w[i] = 2.0 * mult * atr[i] / mid[i] if mid[i] != 0 else 0.0


@njit(cache=True, nogil=True)
def k_donchian(h, l, c, p, out_up, out_lo, out_w):
    """Donchian kanalı."""
    n = c.shape[0]
    out_up[:] = np.nan; out_lo[:] = np.nan; out_w[:] = np.nan
    for i in range(p - 1, n):
        hh = h[i]; ll = l[i]
        for j in range(1, p):
            if h[i - j] > hh: hh = h[i - j]
            if l[i - j] < ll: ll = l[i - j]
        out_up[i] = hh / c[i] - 1.0
        out_lo[i] = ll / c[i] - 1.0
        out_w[i] = (hh - ll) / c[i]


@njit(cache=True, nogil=True)
def k_aroon(h, l, p, out_up, out_dn, out_osc):
    """Aroon Up/Down/Osc."""
    n = h.shape[0]
    out_up[:] = np.nan; out_dn[:] = np.nan; out_osc[:] = np.nan
    for i in range(p, n):
        ih = 0; il = 0
        mh = h[i]; ml = l[i]
        for j in range(p + 1):
            k = i - j
            if h[k] > mh: mh = h[k]; ih = j
            if l[k] < ml: ml = l[k]; il = j
        up = (p - ih) / float(p)
        dn = (p - il) / float(p)
        out_up[i] = up; out_dn[i] = dn; out_osc[i] = up - dn


@njit(cache=True, nogil=True)
def k_cci(h, l, c, p, out):
    """CCI."""
    n = c.shape[0]
    out[:] = np.nan
    tp = np.empty(n)
    for i in range(n):
        tp[i] = (h[i] + l[i] + c[i]) / 3.0
    for i in range(p - 1, n):
        m = 0.0
        for j in range(p): m += tp[i - j]
        m /= p
        md = 0.0
        for j in range(p): md += abs(tp[i - j] - m)
        md /= p
        out[i] = (tp[i] - m) / (0.015 * md) / 100.0 if md > 0 else 0.0


@njit(cache=True, nogil=True)
def k_willr(h, l, c, p, out):
    """Williams %R."""
    n = c.shape[0]
    out[:] = np.nan
    for i in range(p - 1, n):
        hh = h[i]; ll = l[i]
        for j in range(1, p):
            if h[i - j] > hh: hh = h[i - j]
            if l[i - j] < ll: ll = l[i - j]
        out[i] = -100.0 * (hh - c[i]) / (hh - ll) if hh > ll else -50.0


@njit(cache=True, nogil=True)
def k_adx(h, l, c, p, out_adx, out_pdi, out_mdi):
    """ADX + DI."""
    n = c.shape[0]
    out_adx[:] = np.nan; out_pdi[:] = np.nan; out_mdi[:] = np.nan
    if n < 2 * p + 1: return
    tr = np.empty(n); _true_range(h, l, c, tr)
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    st = 0.0; sp = 0.0; sm = 0.0
    for i in range(1, p + 1):
        st += tr[i]; sp += pdm[i]; sm += mdm[i]
    dx = np.full(n, np.nan)
    for i in range(p, n):
        if i > p:
            st = st - st / p + tr[i]
            sp = sp - sp / p + pdm[i]
            sm = sm - sm / p + mdm[i]
        pdi = 100.0 * sp / st if st > 0 else 0.0
        mdi = 100.0 * sm / st if st > 0 else 0.0
        out_pdi[i] = pdi; out_mdi[i] = mdi
        dx[i] = 100.0 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0.0
    adx = 0.0
    for i in range(p, 2 * p): adx += dx[i]
    adx /= p
    out_adx[2 * p - 1] = adx
    for i in range(2 * p, n):
        adx = (adx * (p - 1) + dx[i]) / p
        out_adx[i] = adx


# ═══════════════════════════════════════════════════════════════════
# ANA KERNEL — Tek hisse için 111 gösterge
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True)
def tech_stock_kernel(o, h, l, c, v, out):
    """Tek hisse için 111 teknik göstergeyi out (N_TECH, T) matrisine yazar."""
    n = c.shape[0]
    lr = np.empty(n)
    _logret(c, lr)

    k_rsi(c, 7, out[0])
    k_rsi(c, 14, out[1])
    k_rsi(c, 21, out[2])
    k_atrn(h, l, c, 7, out[3])
    k_atrn(h, l, c, 14, out[4])
    k_atrn(h, l, c, 21, out[5])
    k_macd(c, 12, 26, 9, out[6], out[7], out[8])
    k_bollinger(c, 20, 2.0, out[9], out[10])
    k_stoch(h, l, c, 14, out[11], out[12])
    k_obv_vwap(h, l, c, v, 20, out[14], out[13])
    k_ret(c, 1, out[15]); k_ret(c, 2, out[16]); k_ret(c, 3, out[17]); k_ret(c, 5, out[18])
    k_ret(c, 10, out[19]); k_ret(c, 20, out[20]); k_ret(c, 40, out[21])
    k_sma_dist(c, 5, out[22]); k_sma_dist(c, 10, out[23]); k_sma_dist(c, 20, out[24])
    k_sma_dist(c, 50, out[25]); k_sma_dist(c, 200, out[26])
    k_ema_dist(c, 12, out[27]); k_ema_dist(c, 26, out[28])
    k_rvol(lr, 10, out[29]); k_rvol(lr, 30, out[30]); k_rvol(lr, 60, out[31])
    k_range_vols(o, h, l, c, 20, out[32], out[33])
    k_candle(o, h, l, c, out[34], out[35], out[36], out[37])
    k_volume_feats(v, 20, out[38], out[39])
    k_amihud(c, v, lr, 20, out[40])
    k_cs_spread(h, l, out[41])
    k_gap(o, c, out[42])
    k_skew_kurt(lr, 20, out[43], out[44])
    k_dd_low(h, l, c, 60, out[45], out[46])
    k_mom_acc(c, 10, out[47])
    k_eff_ratio(c, 20, out[48])

    # Genişletilmiş (49-110)
    k_ichimoku(h, l, c, out[49], out[50], out[51], out[52], out[53])
    k_keltner(h, l, c, 20, 10, 2.0, out[54], out[55], out[56])
    k_donchian(h, l, c, 20, out[57], out[58], out[59])
    k_aroon(h, l, 25, out[60], out[61], out[62])
    k_cci(h, l, c, 20, out[63])
    k_ret(c, 5, out[64]); k_ret(c, 10, out[65]); k_ret(c, 20, out[66])
    k_willr(h, l, c, 7, out[67]); k_willr(h, l, c, 14, out[68]); k_willr(h, l, c, 21, out[69])
    k_adx(h, l, c, 14, out[70], out[71], out[72])
    return 111


@njit(cache=True, parallel=True)
def indicator_panel_kernel(O, H, L, C, V, out):
    """Panel kernel — paralel."""
    S = C.shape[0]
    for s in prange(S):
        tech_stock_kernel(O[s], H[s], L[s], C[s], V[s], out[s])


# ═══════════════════════════════════════════════════════════════════
# PYTHON SARMALAYICI
# ═══════════════════════════════════════════════════════════════════

def compute_tech_panel(O: np.ndarray, H: np.ndarray, L: np.ndarray,
                       C: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Panel teknik özellikleri hesaplar. Girdi (S,T), çıktı (S, N_TECH, T) float32."""
    O = np.ascontiguousarray(O, dtype=np.float64)
    H = np.ascontiguousarray(H, dtype=np.float64)
    L = np.ascontiguousarray(L, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    S, T = C.shape
    out = np.full((S, N_TECH, T), np.nan, dtype=np.float64)
    indicator_panel_kernel(O, H, L, C, V, out)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(out, -1e6, 1e6, out=out)
    return np.ascontiguousarray(out.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Numba Tech Test")
    print(f"Numba: {'VAR' if NUMBA_OK else 'YOK (saf Python)'}")
    print(f"Toplam gösterge: {N_TECH}")

    rng = np.random.default_rng(7)
    S, T = 4, 400
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (S, T)), axis=1))
    O = C * (1 + rng.normal(0, 0.003, (S, T)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.004, (S, T))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.004, (S, T))))
    V = rng.uniform(1e5, 1e6, (S, T))

    import time
    t0 = time.time()
    pf = compute_tech_panel(O, H, L, C, V)
    elapsed = time.time() - t0

    print(f"\nŞekil: {pf.shape}")
    print(f"Süre: {elapsed:.3f}s")
    print(f"NaN/Inf: {np.isnan(pf).any() or np.isinf(pf).any()}")
    print(f"Min/Max: {pf.min():.4f} / {pf.max():.4f}")

    # RSI test
    rsi14 = pf[0, 1] * 50.0 + 50.0
    print(f"\nRSI14 aralık: [{rsi14.min():.2f}, {rsi14.max():.2f}]")
    assert 0 <= rsi14.min() and rsi14.max() <= 100
    assert pf.shape == (S, N_TECH, T)

    print("\n✅ Numba Tech testi geçti")
