# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — İCRA
Almgren-Chriss · Dinamik slippage · VWAP/TWAP · OFI gate · Kısmi dolum
Kâğıt üzerinde emir motoru (canlı emir gönderimi YOK)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from v7_config import (
    AC_ETA, DYNAMIC_SLIPPAGE, FEE, SLIPPAGE, SLIPPAGE_BASE,
)

log = logging.getLogger("v7.execution")


# ═══════════════════════════════════════════════════════════════════
# ALMGREN-CHRISS MARKET IMPACT
# ═══════════════════════════════════════════════════════════════════

def almgren_chriss_impact(order_size_tl: float, adv_tl: float, atr_pct: float,
                          sigma_daily: float = 0.02, eta: float = AC_ETA) -> float:
    """
    Almgren & Chriss (2005) market impact (kesir).
    Kalıcı: η·(Q/ADV)·σ
    Geçici: η·√(Q/ADV)·σ
    [0.0001, 0.05] aralığına kırpılır.
    """
    if adv_tl <= 0:
        return 0.01
    pv = order_size_tl / max(adv_tl, 1.0)
    permanent = eta * pv * sigma_daily
    temporary = eta * math.sqrt(max(pv, 0.0)) * sigma_daily
    total = permanent + temporary
    return min(max(total, 0.0001), 0.05)


def dynamic_slippage(price: float, volume_tl: float, atr_pct: float,
                     order_size_tl: float = 1e6) -> float:
    """
    Hacim + volatilite + Almgren-Chriss toplam slippage (kesir).
    """
    vf = min(max((1e7 / max(volume_tl, 1e5)) ** 0.5, 0.5), 3.0)
    af = min(max(1.0 + (atr_pct - 0.02) * 50.0, 0.8), 3.0)
    ac = almgren_chriss_impact(order_size_tl, volume_tl, atr_pct)
    return min(max(SLIPPAGE_BASE * vf * af + ac, 0.0002), 0.05)


# ═══════════════════════════════════════════════════════════════════
# EXECUTION ALGORITHMS
# ═══════════════════════════════════════════════════════════════════

def vwap_split(total_qty: float, volume_profile: Sequence[float]) -> np.ndarray:
    """VWAP dilimleri: her bara hacim oranında emir."""
    vp = np.asarray(volume_profile, dtype=np.float64)
    if vp.sum() <= 0:
        return np.full(len(vp), total_qty / max(len(vp), 1))
    return total_qty * vp / vp.sum()


def twap_split(total_qty: float, n_bars: int) -> np.ndarray:
    """TWAP dilimleri: her bara eşit emir."""
    return np.full(int(n_bars), total_qty / max(int(n_bars), 1))


def partial_fill_qty(qty: int, adv_tl: float, price: float,
                     max_adv_pct: float = 0.05) -> int:
    """ADV limitine göre kısmi dolum miktarı."""
    if price <= 0:
        return 0
    cap = math.floor(max_adv_pct * max(adv_tl, 0.0) / price)
    return int(max(0, min(qty, cap)))


def ofi_gate_ok(side: str, ofi: float, ofi_min: float) -> bool:
    """Alımda OFI eşiğin altındaysa engelle. Satım her zaman geçer."""
    if side.upper() == "BUY":
        return bool(ofi >= ofi_min)
    return True


# ═══════════════════════════════════════════════════════════════════
# EMİR VERİ YAPISI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FillResult:
    """Dolum sonucu."""
    side: str
    stock: str
    qty: int
    ref_price: float
    fill_price: float
    impact_bps: float
    slip_bps: float
    ofi: float
    adv_pct: float
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = time.strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════
# İCRA MOTORU (KÂĞIT ÜZERİNDE)
# ═══════════════════════════════════════════════════════════════════

class LiveExecutionEngine:
    """
    Kâğıt üzerinde emir motoru.
    Alımda OFI gate + kısmi dolum; her emirde dinamik slippage ve AC impact.
    Gerçek emir gönderimi YOKTUR.
    """

    def __init__(self, pm: Any, db: Any = None) -> None:
        self.pm = pm
        self.db = db
        self.lock = threading.RLock()
        self.fill_history: List[FillResult] = []

    def execute(self, side: str, stock: str, qty: int, ref_price: float,
                adv_tl: float, atr_pct: float, ofi: float = 0.0) -> Optional[FillResult]:
        """Emri uygular. Reddedilirse None."""
        rp = self.pm.rp
        side = side.upper()
        with self.lock:
            # OFI gate (yalnızca alım)
            if side == "BUY" and rp.get("ofi_gate", True):
                if not ofi_gate_ok(side, ofi, float(rp["ofi_min"])):
                    log.info("OFI gate reddi: %s (ofi=%.3f)", stock, ofi)
                    return None

            # Kısmi dolum (yalnızca alım)
            q = int(qty)
            if side == "BUY":
                q = partial_fill_qty(q, adv_tl, ref_price,
                                     float(rp.get("max_adv_pct", 0.05)))
            if q <= 0:
                return None

            # Slippage + impact
            order_tl = q * ref_price
            if DYNAMIC_SLIPPAGE:
                slip = dynamic_slippage(ref_price, adv_tl, atr_pct, order_tl)
            else:
                slip = SLIPPAGE_BASE
            imp = (almgren_chriss_impact(order_tl, adv_tl, atr_pct)
                   if rp.get("impact_on", True) else 0.0)

            # Fill fiyatı
            sign = 1.0 if side == "BUY" else -1.0
            fill = ref_price * (1.0 + sign * (slip + imp))

            # Kayıt
            rec = FillResult(
                side=side, stock=stock, qty=int(q),
                ref_price=float(ref_price), fill_price=float(fill),
                impact_bps=imp * 1e4, slip_bps=slip * 1e4,
                ofi=float(ofi), adv_pct=float(order_tl / max(adv_tl, 1.0)),
            )
            self.fill_history.append(rec)

            # DB
            if self.db is not None:
                try:
                    self.db.insert("execution", [
                        rec.ts, stock, side, int(q), float(ref_price),
                        float(fill), rec.impact_bps, rec.slip_bps,
                        float(ofi), rec.adv_pct,
                    ])
                except Exception as e:
                    log.warning("Execution DB kaydı hatası: %s", e)

            log.debug("Filled: %s %s x%d @ %.4f (slip=%.1fbps)",
                      side, stock, q, fill, rec.slip_bps)
            return rec

    def stats(self) -> Dict[str, float]:
        """İcra istatistikleri."""
        if not self.fill_history:
            return {}
        slips = np.array([f.slip_bps for f in self.fill_history])
        impacts = np.array([f.impact_bps for f in self.fill_history])
        return {
            "n_fills": len(self.fill_history),
            "avg_slip_bps": float(slips.mean()),
            "avg_impact_bps": float(impacts.mean()),
            "max_slip_bps": float(slips.max()),
        }


# ═══════════════════════════════════════════════════════════════════
# BASİT VWAP / TWAP YÜRÜTÜCÜ
# ═══════════════════════════════════════════════════════════════════

class VWAPExecutor:
    """Hacim profilinden VWAP dilimleri üretir."""

    def __init__(self, total_qty: int, n_bars: int = 10):
        self.total_qty = int(total_qty)
        self.n_bars = int(n_bars)

    def plan(self, volume_profile: Optional[Sequence[float]] = None) -> np.ndarray:
        """Her bar için adet dizisi."""
        if volume_profile is None:
            volume_profile = np.ones(self.n_bars)
        return vwap_split(self.total_qty, volume_profile)


class TWAPExecutor:
    """Eşit dilimli TWAP."""

    def __init__(self, total_qty: int, n_bars: int = 10):
        self.total_qty = int(total_qty)
        self.n_bars = int(n_bars)

    def plan(self) -> np.ndarray:
        return twap_split(self.total_qty, self.n_bars)


# ═══════════════════════════════════════════════════════════════════
# İMPLEMENTATION SHORTFALL
# ═══════════════════════════════════════════════════════════════════

def implementation_shortfall(decision_price: float, avg_fill_price: float,
                             side: str, qty: int) -> Dict[str, float]:
    """Implementation Shortfall (Perold 1988)."""
    if decision_price <= 0 or qty <= 0:
        return {"is_tl": 0.0, "is_bps": 0.0}
    sign = 1.0 if side.upper() == "BUY" else -1.0
    is_per_share = sign * (avg_fill_price - decision_price)
    is_tl = is_per_share * qty
    is_bps = (avg_fill_price / decision_price - 1.0) * 1e4 * sign
    return {"is_tl": float(is_tl), "is_bps": float(is_bps)}


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — İcra Test")

    # Almgren-Chriss
    a1 = almgren_chriss_impact(1e6, 1e8, 0.02)
    a2 = almgren_chriss_impact(1e7, 1e8, 0.02)
    print(f"\n✅ AC impact: küçük={a1:.5f}, büyük={a2:.5f}")
    assert a2 > a1

    # Dinamik slippage
    s = dynamic_slippage(100.0, 1e7, 0.03)
    print(f"✅ Dinamik slippage: {s:.5f}")
    assert 0.0002 <= s <= 0.05

    # VWAP / TWAP
    vs = vwap_split(100.0, [1, 2, 3, 4])
    ts = twap_split(100.0, 4)
    print(f"\n✅ VWAP dilimleri: {vs.round(2)}")
    print(f"✅ TWAP dilimleri: {ts.round(2)}")
    assert abs(vs.sum() - 100.0) < 1e-9
    assert np.allclose(ts, 25.0)

    # Kısmi dolum
    p = partial_fill_qty(1_000_000, 1e6, 10.0, 0.05)
    print(f"\n✅ Kısmi dolum: {p} adet (limit=5000)")
    assert p == 5000

    # OFI gate
    assert not ofi_gate_ok("BUY", -0.5, -0.3)
    assert ofi_gate_ok("SELL", -0.9, -0.3)
    print("✅ OFI gate: çalışıyor")

    # Execution engine (mock PM)
    class MockPM:
        rp = {
            "ofi_gate": True, "ofi_min": -0.30, "max_adv_pct": 0.05,
            "impact_on": True,
        }

    engine = LiveExecutionEngine(MockPM(), db=None)
    fill = engine.execute("BUY", "GARAN", 100, 100.0, 1e7, 0.02, ofi=0.5)
    print(f"\n✅ Dolum: {fill.side} x{fill.qty} @ {fill.fill_price:.4f}")
    print(f"   Slip: {fill.slip_bps:.1f}bps, Impact: {fill.impact_bps:.1f}bps")
    assert fill is not None

    # OFI reddi
    fill2 = engine.execute("BUY", "GARAN", 100, 100.0, 1e7, 0.02, ofi=-0.9)
    print(f"✅ OFI reddi: {fill2 is None}")
    assert fill2 is None

    # Stats
    st = engine.stats()
    print(f"\n✅ İstatistik: {st}")

    # IS
    is_res = implementation_shortfall(100.0, 100.15, "BUY", 1000)
    print(f"\n✅ IS: {is_res['is_tl']:.2f} TL ({is_res['is_bps']:.1f} bps)")

    print("\n✅ İcra modülü testi geçti")
