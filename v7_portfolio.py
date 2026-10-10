# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — PORTFÖY
Pozisyon yönetimi · Kelly boyutlandırma · Stop/TP/Trailing · Risk limitleri
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v7_config import BOT_SERMAYE, FEE, RT_COST, TB_K, get_risk_params
from v7_risk import fractional_kelly

log = logging.getLogger("v7.portfolio")


# ═══════════════════════════════════════════════════════════════════
# POZISYON
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Açık pozisyon."""
    stock: str
    sector: int
    qty: int
    entry: float
    atr_abs: float
    R: float
    stop: float
    tp: float
    entry_t: int
    held: int = 0
    partial_done: bool = False
    trail_on: bool = False
    peak: float = 0.0
    pnl_realized: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.entry) + self.pnl_realized


# ═══════════════════════════════════════════════════════════════════
# PORTFÖY YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════════════

class PortfolioManager:
    """Nakit, pozisyon, boyutlandırma, risk limitleri. Thread-safe."""

    def __init__(self, initial_capital: float = BOT_SERMAYE,
                 rp: Optional[Dict[str, Any]] = None,
                 sector_ids: Optional[Dict[str, int]] = None) -> None:
        self.initial = float(initial_capital)
        self.cash = float(initial_capital)
        self.rp = dict(rp) if rp is not None else get_risk_params()
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_hist: List[Tuple[int, float]] = []
        self.peak = self.initial
        self.loss_streak = 0
        self.cooldown: Dict[str, int] = {}
        self.day_start_equity = self.initial
        self.trades_today = 0
        self.sector_ids: Dict[str, int] = dict(sector_ids or {})
        self.lock = threading.RLock()

    # ─── Hesaplamalar ──────────────────────────────────────────────
    def equity(self, prices: Dict[str, float]) -> float:
        """Nakit + açık pozisyonlar."""
        with self.lock:
            mv = sum(p.qty * float(prices.get(s, p.entry)) for s, p in self.positions.items())
            return self.cash + mv

    def exposure(self, prices: Dict[str, float]) -> float:
        """Toplam maruziyet (özkaynak kesri)."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(p.qty * float(prices.get(s, p.entry))
                   for s, p in self.positions.items()) / eq

    def sector_exposure(self, sector: int, prices: Dict[str, float]) -> float:
        """Tek sektör maruziyeti."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(p.qty * float(prices.get(s, p.entry))
                   for s, p in self.positions.items()
                   if p.sector == sector) / eq

    def n_positions(self) -> int:
        return len(self.positions)

    # ─── Gün başı ────────────────────────────────────────────────
    def new_day(self, t: int, prices: Dict[str, float]) -> None:
        """Gün başı: referans özkaynak ve işlem sayacı."""
        with self.lock:
            self.day_start_equity = self.equity(prices)
            self.trades_today = 0

    # ─── İzin kontrolü ───────────────────────────────────────────
    def can_open(self, stock: str, t: int, prices: Dict[str, float]) -> Tuple[bool, str]:
        """Yeni pozisyon için tüm risk limitleri."""
        rp = self.rp
        with self.lock:
            eq = self.equity(prices)
            if stock in self.positions:
                return False, "zaten açık"
            if len(self.positions) >= int(rp["max_positions"]):
                return False, "maks pozisyon"
            if self.trades_today >= int(rp["max_trades_day"]):
                return False, "günlük işlem limiti"
            if self.loss_streak >= int(rp["loss_streak"]):
                return False, "ardışık zarar"
            dd = (1.0 - eq / max(self.peak, 1e-9)) * 100.0
            if dd > float(rp["max_dd"]):
                return False, "maks DD eşiği"
            day_pnl = (eq - self.day_start_equity) / max(self.day_start_equity, 1e-9) * 100.0
            if day_pnl < -float(rp["daily_loss"]):
                return False, "günlük zarar limiti"
            if self.cooldown.get(stock, -1) >= t:
                return False, "cooldown"
            if self.exposure(prices) >= float(rp["max_exposure"]):
                return False, "maruziyet tavanı"
            sec = self.sector_ids.get(stock)
            if sec is not None and self.sector_exposure(sec, prices) >= float(rp["max_sector"]):
                return False, "sektör tavanı"
            return True, "ok"

    # ─── Boyutlandırma ───────────────────────────────────────────
    def size(self, stock: str, price: float, atr_pct: float,
             p_up: float, unc: float, eq: float) -> int:
        """
        Risk tabanlı boyut:
        min(risk_bütçesi/stop_mesafesi, max_pos·özkaynak/fiyat, Kelly, nakit)
        """
        rp = self.rp
        if price <= 0 or atr_pct <= 0 or eq <= 0:
            return 0
        stop_dist = float(rp["sl_atr"]) * atr_pct * price
        if stop_dist <= 0:
            return 0

        # Risk bütçesi
        q = float(rp["risk_per_trade"]) * eq / stop_dist

        # Tek pozisyon tavanı
        q = min(q, float(rp["max_pos"]) * eq / price)

        # Güven ölçeklemesi
        if rp.get("conf_sizing", True):
            q *= max(0.25, min(1.0, (p_up - 0.33) / 0.67))

        # Kelly
        kf = fractional_kelly(p_up, TB_K * atr_pct, TB_K * atr_pct,
                              float(rp["kelly_frac"]), unc, float(rp["kelly_unc_pen"]))
        if kf <= 0:
            return 0
        q = min(q, kf * eq / price)

        # Nakit
        q = min(q, 0.98 * self.cash / (price * (1.0 + FEE)))

        return int(max(0, math.floor(q)))

    # ─── Pozisyon açma ───────────────────────────────────────────
    def open_position(self, stock: str, qty: int, fill: float,
                      atr_pct: float, t: int) -> Optional[Position]:
        """Alım sonrası pozisyon açar. Nakit ücretle birlikte düşer."""
        if qty <= 0 or fill <= 0:
            return None
        rp = self.rp
        cost = qty * fill
        if cost * (1.0 + FEE) > self.cash + 1e-9:
            return None
        with self.lock:
            self.cash -= cost * (1.0 + FEE)
            atr_abs = atr_pct * fill
            R = float(rp["sl_atr"]) * atr_abs
            tp = fill + float(rp["tp_r"]) * R if rp.get("tp_on", True) else float("inf")
            pos = Position(
                stock=stock,
                sector=int(self.sector_ids.get(stock, -1)),
                qty=int(qty),
                entry=float(fill),
                atr_abs=float(atr_abs),
                R=float(R),
                stop=float(fill - R),
                tp=float(tp),
                entry_t=int(t),
                peak=float(fill),
            )
            self.positions[stock] = pos
            self.trades_today += 1
            log.info("Açıldı: %s x%d @ %.2f (R=%.2f)", stock, qty, fill, R)
            return pos

    # ─── Bar güncelleme ──────────────────────────────────────────
    def update_bar(self, stock: str, o: float, h: float, l: float, c: float,
                   t: int) -> List[Tuple[int, float, str]]:
        """
        Bar güncellemesi. Dönüş: [(kapatılacak qty, ref fiyat, neden), ...]
        Sıra: stop (gap) → kısmi TP → BE → trailing → max_hold
        """
        pos = self.positions.get(stock)
        if pos is None:
            return []
        rp = self.rp
        events: List[Tuple[int, float, str]] = []

        # Stop (gap koruması)
        if l <= pos.stop:
            ex = o if o < pos.stop else pos.stop
            events.append((pos.qty, float(ex), "stop"))
            return events

        # Giriş barı: sadece stop kontrolü
        if pos.entry_t == t:
            return events

        pos.held += 1

        # Kısmi TP + BE
        if rp.get("tp_on", True) and not pos.partial_done and h >= pos.tp:
            px = max(pos.tp, o)
            q_part = int(math.floor(pos.qty * float(rp.get("tp_frac", 0.5))))
            if q_part > 0:
                events.append((q_part, float(px), "tp_partial"))
            pos.partial_done = True
            if rp.get("be_on", True):
                pos.stop = max(pos.stop, pos.entry)

        # Trailing
        if h >= pos.entry + float(rp["trail_act_r"]) * pos.R:
            pos.trail_on = True
        if pos.trail_on:
            pos.stop = max(pos.stop, h - float(rp["trail_atr"]) * pos.atr_abs)

        pos.peak = max(pos.peak, h)

        # Max hold
        if pos.held >= int(rp["max_hold"]) and pos.qty > 0:
            events.append((pos.qty, float(c), "max_hold"))

        return events

    # ─── Kapanış ────────────────────────────────────────────────
    def close(self, stock: str, qty: int, fill: float, reason: str,
              t: int) -> Dict[str, Any]:
        """Kısmi/tam kapanış."""
        with self.lock:
            pos = self.positions.get(stock)
            if pos is None or qty <= 0:
                return {}
            qty = min(int(qty), pos.qty)
            proceeds = qty * fill
            self.cash += proceeds * (1.0 - FEE)
            pnl = qty * (fill - pos.entry) - (qty * pos.entry + proceeds) * FEE
            rec = {
                "t": int(t), "stock": stock, "qty": qty,
                "entry": pos.entry, "exit": float(fill),
                "pnl": float(pnl), "reason": reason,
            }
            self.trades.append(rec)
            pos.qty -= qty
            pos.pnl_realized += pnl
            if pos.qty <= 0:
                del self.positions[stock]
                self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0
                cd = max(1, int(self.rp.get("cooldown_h", 24)) // 24)
                self.cooldown[stock] = int(t) + cd
            log.debug("Kapandı: %s x%d @ %.2f (%s) PnL=%.2f", stock, qty, fill, reason, pnl)
            return rec

    # ─── Snapshot ───────────────────────────────────────────────
    def snapshot(self, t: int, prices: Dict[str, float]) -> float:
        """Özkaynağı kaydeder, tepe günceller."""
        with self.lock:
            eq = self.equity(prices)
            self.peak = max(self.peak, eq)
            self.equity_hist.append((int(t), float(eq)))
            return eq

    # ─── Durum ──────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Kalıcı durum."""
        with self.lock:
            return {
                "initial": self.initial,
                "cash": self.cash,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "trades": list(self.trades),
                "equity_hist": list(self.equity_hist),
                "peak": self.peak,
                "loss_streak": self.loss_streak,
                "cooldown": dict(self.cooldown),
                "day_start_equity": self.day_start_equity,
                "trades_today": self.trades_today,
                "sector_ids": dict(self.sector_ids),
                "rp": dict(self.rp),
            }

    def set_state(self, s: Dict[str, Any]) -> None:
        """Durumu geri yükler."""
        with self.lock:
            self.initial = float(s["initial"])
            self.cash = float(s["cash"])
            self.positions = {k: Position(**v) for k, v in s["positions"].items()}
            self.trades = list(s["trades"])
            self.equity_hist = [tuple(x) for x in s["equity_hist"]]
            self.peak = float(s["peak"])
            self.loss_streak = int(s["loss_streak"])
            self.cooldown = {k: int(v) for k, v in s["cooldown"].items()}
            self.day_start_equity = float(s["day_start_equity"])
            self.trades_today = int(s["trades_today"])
            self.sector_ids = dict(s["sector_ids"])
            self.rp = dict(s["rp"])

    # ─── Özet ───────────────────────────────────────────────────
    def summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Portföy özeti."""
        with self.lock:
            eq = self.equity(prices)
            total_pnl = eq - self.initial
            n_wins = sum(1 for t in self.trades if t["pnl"] > 0)
            n_loss = sum(1 for t in self.trades if t["pnl"] < 0)
            return {
                "equity": eq,
                "cash": self.cash,
                "peak": self.peak,
                "total_pnl": total_pnl,
                "total_return_pct": total_pnl / max(self.initial, 1e-9) * 100,
                "n_positions": len(self.positions),
                "n_trades": len(self.trades),
                "n_wins": n_wins,
                "n_loss": n_loss,
                "win_rate": n_wins / max(len(self.trades), 1) * 100,
                "dd_from_peak": (1.0 - eq / max(self.peak, 1e-9)) * 100,
                "exposure": self.exposure(prices) * 100,
            }


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Portföy Test")

    rp = get_risk_params()
    sector_ids = {"GARAN": 0, "AKBNK": 0, "THYAO": 1, "TUPRS": 2}

    pm = PortfolioManager(100_000.0, rp, sector_ids)
    print(f"\nBaşlangıç: Nakit={pm.cash:.0f}, EQ={pm.equity({}):.0f}")

    # Boyutlandırma testi
    prices = {s: 100.0 for s in sector_ids}
    q = pm.size("GARAN", 100.0, 0.02, 0.6, 0.1, pm.equity(prices))
    print(f"\n✅ Boyut: {q} adet")
    stop_dist = rp["sl_atr"] * 0.02 * 100.0
    risk_amount = q * stop_dist
    max_risk = rp["risk_per_trade"] * pm.equity(prices)
    print(f"   Risk: {risk_amount:.0f} TL (max: {max_risk:.0f})")
    assert risk_amount <= max_risk * 1.01

    # Pozisyon açma
    pos = pm.open_position("GARAN", q, 100.0, 0.02, t=10)
    assert pos is not None
    print(f"\n✅ Açıldı: {pos.stock} x{pos.qty} @ {pos.entry}")

    # Bar güncelleme — stop tetiklenmeli
    ev = pm.update_bar("GARAN", o=99, h=99.5, l=90, c=95, t=11)
    print(f"\n✅ Stop event: {ev}")
    assert len(ev) == 1 and ev[0][2] == "stop"

    for qy, ref, why in ev:
        rec = pm.close("GARAN", qy, ref, why, t=11)
        print(f"✅ Kapandı: PnL={rec['pnl']:.2f}")

    # Maks pozisyon testi
    pm2 = PortfolioManager(100_000.0, {**rp, "max_positions": 1}, sector_ids)
    pm2.open_position("GARAN", 10, 100.0, 0.02, t=1)
    ok, why = pm2.can_open("AKBNK", t=2, prices={"GARAN": 100.0, "AKBNK": 100.0})
    print(f"\n✅ Maks pozisyon: {ok} ({why})")
    assert not ok

    # Günlük zarar testi
    pm3 = PortfolioManager(100_000.0, {**rp, "daily_loss": 0.0001}, sector_ids)
    pm3.day_start_equity = 200_000.0
    ok, why = pm3.can_open("GARAN", 5, prices={"GARAN": 100.0})
    print(f"✅ Günlük zarar: {ok} ({why})")
    assert not ok and why == "günlük zarar limiti"

    # State testi
    pm4 = PortfolioManager(100_000.0, rp, sector_ids)
    pm4.open_position("THYAO", 5, 200.0, 0.03, t=1)
    state = pm4.get_state()
    pm5 = PortfolioManager(100_000.0, rp, sector_ids)
    pm5.set_state(state)
    assert abs(pm5.cash - pm4.cash) < 1e-9
    assert len(pm5.positions) == len(pm4.positions)
    print(f"\n✅ State kaydet/yükle: cash={pm5.cash:.0f}")

    print("\n✅ Portföy testi geçti")
