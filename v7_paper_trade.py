# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — PAPER TRADE
Sanal portföy · Emir simülasyonu · PnL takibi · Günlük rapor
Gerçek emir gönderimi YOK
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from v7_config import BOT_SERMAYE, FEE, SLIPPAGE, get_risk_params

log = logging.getLogger("v7.paper_trade")


# ═══════════════════════════════════════════════════════════════════
# EMİR KAYDI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PaperOrder:
    """Sanal emir kaydı."""
    ts: str
    stock: str
    side: str  # BUY / SELL
    qty: int
    ref_price: float
    fill_price: float
    fee: float
    reason: str = ""

    @property
    def gross_value(self) -> float:
        return self.qty * self.fill_price


# ═══════════════════════════════════════════════════════════════════
# SANAL POZİSYON
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    """Sanal pozisyon."""
    stock: str
    qty: int
    avg_price: float
    entry_ts: str
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_price)


# ═══════════════════════════════════════════════════════════════════
# PAPER TRADE ENGINE
# ═══════════════════════════════════════════════════════════════════

class PaperTradeEngine:
    """
    Sanal portföy simülatörü.
    - Tüm emirler anında (aynı gün kapanışında) dolar
    - Komisyon + slippage uygulanır
    - Ortalama maliyet takibi (kısmi alım/satım)
    """

    def __init__(self, initial_capital: float = BOT_SERMAYE,
                 rp: Optional[Dict] = None) -> None:
        self.initial = float(initial_capital)
        self.cash = float(initial_capital)
        self.rp = dict(rp) if rp else get_risk_params()
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.equity_hist: List[Tuple[int, float]] = []
        self.peak = self.initial
        self.realized_total = 0.0
        self.unrealized = 0.0
        self.lock = threading.RLock()

    # ─── Değerleme ────────────────────────────────────────────────
    def equity(self, prices: Dict[str, float]) -> float:
        """Nakit + pozisyon değeri."""
        with self.lock:
            mv = sum(p.qty * float(prices.get(s, p.avg_price))
                     for s, p in self.positions.items())
            return self.cash + mv

    def exposure(self, prices: Dict[str, float]) -> float:
        """Maruziyet (0-1)."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        mv = sum(p.qty * float(prices.get(s, p.avg_price))
                 for s, p in self.positions.items())
        return mv / eq

    # ─── Emirler ─────────────────────────────────────────────────
    def buy(self, stock: str, qty: int, ref_price: float,
            reason: str = "", slip: Optional[float] = None) -> Optional[PaperOrder]:
        """Sanal alım."""
        if qty <= 0 or ref_price <= 0:
            return None
        slip = SLIPPAGE if slip is None else float(slip)
        fill = ref_price * (1.0 + slip)
        cost = qty * fill
        fee = cost * FEE
        total = cost + fee

        with self.lock:
            if total > self.cash + 1e-9:
                log.debug("Yetersiz nakit: %s (gereken %.0f, mevcut %.0f)",
                          stock, total, self.cash)
                return None
            self.cash -= total
            if stock in self.positions:
                p = self.positions[stock]
                new_qty = p.qty + qty
                new_avg = (p.qty * p.avg_price + qty * fill) / new_qty
                p.qty = new_qty
                p.avg_price = new_avg
            else:
                self.positions[stock] = PaperPosition(
                    stock=stock, qty=qty, avg_price=fill,
                    entry_ts=datetime.now().isoformat(timespec="seconds"))

            order = PaperOrder(
                ts=datetime.now().isoformat(timespec="seconds"),
                stock=stock, side="BUY", qty=qty,
                ref_price=ref_price, fill_price=fill,
                fee=fee, reason=reason)
            self.orders.append(order)
            return order

    def sell(self, stock: str, qty: int, ref_price: float,
             reason: str = "", slip: Optional[float] = None) -> Optional[PaperOrder]:
        """Sanal satım (long pozisyondan çıkış)."""
        if qty <= 0 or ref_price <= 0:
            return None
        with self.lock:
            p = self.positions.get(stock)
            if p is None or p.qty < qty:
                return None
            slip = SLIPPAGE if slip is None else float(slip)
            fill = ref_price * (1.0 - slip)
            proceeds = qty * fill
            fee = proceeds * FEE
            net = proceeds - fee
            pnl = qty * (fill - p.avg_price) - fee

            self.cash += net
            p.realized_pnl += pnl
            self.realized_total += pnl
            p.qty -= qty
            if p.qty <= 0:
                del self.positions[stock]

            order = PaperOrder(
                ts=datetime.now().isoformat(timespec="seconds"),
                stock=stock, side="SELL", qty=qty,
                ref_price=ref_price, fill_price=fill,
                fee=fee, reason=reason)
            self.orders.append(order)
            log.info("SAT: %s x%d @ %.2f → PnL %.0f", stock, qty, fill, pnl)
            return order

    # ─── Snapshot ────────────────────────────────────────────────
    def snapshot(self, t: int, prices: Dict[str, float]) -> float:
        """Özkaynak kaydet."""
        with self.lock:
            eq = self.equity(prices)
            self.peak = max(self.peak, eq)
            self.equity_hist.append((int(t), float(eq)))
            self.unrealized = sum(p.unrealized_pnl(prices.get(s, p.avg_price))
                                  for s, p in self.positions.items())
            return eq

    # ─── Raporlar ────────────────────────────────────────────────
    def summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Portföy özeti."""
        with self.lock:
            eq = self.equity(prices)
            pnl = eq - self.initial
            n_wins = sum(1 for o in self.orders
                         if o.side == "SELL" and o.qty > 0)  # placeholder
            return {
                "initial": self.initial,
                "equity": eq,
                "cash": self.cash,
                "realized_pnl": self.realized_total,
                "unrealized_pnl": self.unrealized,
                "total_pnl": pnl,
                "total_return_pct": pnl / max(self.initial, 1e-9) * 100,
                "peak": self.peak,
                "dd_from_peak_pct": (1.0 - eq / max(self.peak, 1e-9)) * 100,
                "n_positions": len(self.positions),
                "n_orders": len(self.orders),
                "exposure": self.exposure(prices),
            }

    def positions_df(self, prices: Dict[str, float]) -> pd.DataFrame:
        """Pozisyonlar DataFrame."""
        with self.lock:
            rows = []
            for s, p in self.positions.items():
                cur = prices.get(s, p.avg_price)
                rows.append({
                    "stock": s,
                    "qty": p.qty,
                    "avg_price": p.avg_price,
                    "current_price": cur,
                    "market_value": p.qty * cur,
                    "unrealized": p.unrealized_pnl(cur),
                    "unrealized_pct": (cur / p.avg_price - 1.0) * 100,
                    "realized": p.realized_pnl,
                })
            return pd.DataFrame(rows)

    def orders_df(self) -> pd.DataFrame:
        """Emirler DataFrame."""
        if not self.orders:
            return pd.DataFrame(columns=["ts", "stock", "side", "qty",
                                         "ref_price", "fill_price", "fee", "reason"])
        return pd.DataFrame([asdict(o) for o in self.orders])

    def daily_report(self) -> Dict[str, Any]:
        """Günlük özet."""
        if not self.equity_hist:
            return {}
        eqs = np.array([e for _, e in self.equity_hist])
        rets = np.diff(eqs) / np.maximum(eqs[:-1], 1e-9)
        return {
            "n_days": len(eqs),
            "start": float(eqs[0]),
            "end": float(eqs[-1]),
            "total_return_pct": (eqs[-1] / eqs[0] - 1.0) * 100 if eqs[0] > 0 else 0.0,
            "avg_daily_return_pct": float(rets.mean() * 100) if len(rets) else 0.0,
            "vol_daily_pct": float(rets.std() * 100) if len(rets) > 1 else 0.0,
            "sharpe": float(rets.mean() / (rets.std() + 1e-9) * math.sqrt(252)) if len(rets) > 1 else 0.0,
            "n_orders": len(self.orders),
            "n_buys": sum(1 for o in self.orders if o.side == "BUY"),
            "n_sells": sum(1 for o in self.orders if o.side == "SELL"),
        }

    # ─── State ──────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Pickle için."""
        with self.lock:
            return {
                "initial": self.initial,
                "cash": self.cash,
                "positions": {s: asdict(p) for s, p in self.positions.items()},
                "orders": [asdict(o) for o in self.orders],
                "equity_hist": list(self.equity_hist),
                "peak": self.peak,
                "realized_total": self.realized_total,
                "unrealized": self.unrealized,
                "rp": dict(self.rp),
            }

    def set_state(self, s: Dict[str, Any]) -> None:
        """State geri yükle."""
        with self.lock:
            self.initial = float(s["initial"])
            self.cash = float(s["cash"])
            self.positions = {k: PaperPosition(**v) for k, v in s["positions"].items()}
            self.orders = [PaperOrder(**o) for o in s["orders"]]
            self.equity_hist = [tuple(x) for x in s["equity_hist"]]
            self.peak = float(s["peak"])
            self.realized_total = float(s["realized_total"])
            self.unrealized = float(s["unrealized"])
            self.rp = dict(s["rp"])

    def reset(self) -> None:
        """Sıfırla."""
        with self.lock:
            self.cash = self.initial
            self.positions.clear()
            self.orders.clear()
            self.equity_hist.clear()
            self.peak = self.initial
            self.realized_total = 0.0
            self.unrealized = 0.0


# ═══════════════════════════════════════════════════════════════════
# BASİT BOT (Paper üzerinde)
# ═══════════════════════════════════════════════════════════════════

class PaperTradeBot:
    """
    Basit kural tabanlı sanal bot.
    RSI < 30 → AL
    RSI > 70 → SAT
    """

    def __init__(self, engine: PaperTradeEngine) -> None:
        self.engine = engine
        self.rp = engine.rp
        self.pending_buys: List[Dict] = []

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def decide(self, stock: str, df: pd.DataFrame, t: int,
               prices: Dict[str, float]) -> Optional[str]:
        """
        Karar ver. Dönüş: "BUY" | "SELL" | "HOLD" | None
        Kararlar t+1 açılışında uygulanır.
        """
        if len(df) < 30 or t < 20:
            return None
        cur = df.iloc[:t + 1]
        rsi = self._rsi(cur["Close"])
        price = float(cur["Close"].iloc[-1])

        if stock in self.engine.positions:
            # SAT kontrolü
            if rsi > 70:
                return "SELL"
            return "HOLD"

        # AL kontrolü
        if rsi < 30:
            # Boyutlandırma
            eq = self.engine.equity(prices)
            atr_pct = 0.02  # basit varsayım
            stop_dist = self.rp["sl_atr"] * atr_pct * price
            if stop_dist <= 0:
                return None
            qty = int(self.rp["risk_per_trade"] * eq / stop_dist)
            qty = min(qty, int(self.rp["max_pos"] * eq / price))
            qty = min(qty, int(self.engine.cash * 0.98 / (price * (1 + FEE))))
            if qty <= 0:
                return "HOLD"
            self.pending_buys.append({"stock": stock, "qty": qty,
                                      "reason": f"RSI={rsi:.0f}"})
            return "BUY"
        return "HOLD"

    def apply_pending(self, prices: Dict[str, float]) -> List[PaperOrder]:
        """Bekleyen alımları uygula."""
        out = []
        for p in self.pending_buys:
            o = self.engine.buy(p["stock"], p["qty"], prices.get(p["stock"], 0),
                                reason=p["reason"])
            if o:
                out.append(o)
        self.pending_buys.clear()
        return out

    def run_cycle(self, data: Dict[str, pd.DataFrame], t: int,
                  prices: Dict[str, float]) -> Dict[str, Any]:
        """Tek cycle."""
        # 1) Bekleyen alımlar (açılış)
        buys = self.apply_pending(prices)

        # 2) Mevcut pozisyonlar için satım kontrolü
        sells = []
        for stock in list(self.engine.positions.keys()):
            df = data.get(stock)
            if df is None or len(df) < 30:
                continue
            action = self.decide(stock, df, t, prices)
            if action == "SELL":
                qty = self.engine.positions[stock].qty
                o = self.engine.sell(stock, qty, prices.get(stock, 0),
                                     reason=f"RSI > 70")
                if o:
                    sells.append(o)

        # 3) Yeni kararlar (kapanış)
        for stock, df in data.items():
            if stock in self.engine.positions:
                continue
            if any(p["stock"] == stock for p in self.pending_buys):
                continue
            self.decide(stock, df, t, prices)

        # 4) Snapshot
        self.engine.snapshot(t, prices)

        return {
            "t": t,
            "n_buys": len(buys),
            "n_sells": len(sells),
            "pending": len(self.pending_buys),
        }


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Paper Trade Test")

    # Sentetik veri
    rng = np.random.default_rng(7)
    T = 300
    data = {}
    for i, code in enumerate(["GARAN", "AKBNK", "THYAO"]):
        C = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, T)))
        df = pd.DataFrame({
            "Open": C * (1 + rng.normal(0, 0.003, T)),
            "High": C * (1 + np.abs(rng.normal(0, 0.005, T))),
            "Low": C * (1 - np.abs(rng.normal(0, 0.005, T))),
            "Close": C,
            "Volume": rng.uniform(1e5, 1e6, T),
        }, index=pd.date_range("2023-01-02", periods=T))
        data[code] = df

    # 1) Engine
    eng = PaperTradeEngine(100_000.0)
    print(f"\n✅ Engine: nakit={eng.cash:.0f}, poz=0")

    # 2) Manuel alım
    o1 = eng.buy("GARAN", 100, 100.0, reason="test")
    print(f"✅ Alım: {o1.qty} @ {o1.fill_price:.3f}, fee={o1.fee:.2f}")
    assert o1 is not None

    prices = {"GARAN": 105.0, "AKBNK": 100.0, "THYAO": 200.0}
    eq = eng.equity(prices)
    print(f"✅ Equity: {eq:.0f}")

    # 3) Satım
    o2 = eng.sell("GARAN", 100, 105.0, reason="tp")
    print(f"✅ Satım: {o2.qty} @ {o2.fill_price:.3f}")
    assert o2 is not None
    assert "GARAN" not in eng.positions

    # 4) Summary
    summ = eng.summary(prices)
    print(f"\n✅ Summary:")
    print(f"   Equity: {summ['equity']:.0f}")
    print(f"   PnL: {summ['total_pnl']:.2f}")
    print(f"   Realized: {summ['realized_pnl']:.2f}")

    # 5) Bot cycle
    eng2 = PaperTradeEngine(100_000.0)
    bot = PaperTradeBot(eng2)

    for t in range(50, 250, 5):
        prices_t = {c: float(df["Close"].iloc[t]) for c, df in data.items()}
        res = bot.run_cycle(data, t, prices_t)
        if t % 50 == 0:
            print(f"\n✅ t={t}: buys={res['n_buys']}, sells={res['n_sells']}, pending={res['pending']}")

    # 6) Bot özeti
    final_prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}
    summ2 = eng2.summary(final_prices)
    print(f"\n✅ Bot özeti:")
    print(f"   Equity: {summ2['equity']:.0f}")
    print(f"   PnL: {summ2['total_pnl']:.2f} ({summ2['total_return_pct']:.2f}%)")
    print(f"   Pozisyon: {summ2['n_positions']}, Emir: {summ2['n_orders']}")

    # 7) Günlük rapor
    rep = eng2.daily_report()
    print(f"\n✅ Günlük rapor: Sharpe={rep['sharpe']:.2f}")

    # 8) DataFrame
    pdf = eng2.positions_df(final_prices)
    print(f"\n✅ Pozisyonlar: {len(pdf)} satır")
    odf = eng2.orders_df()
    print(f"✅ Emirler: {len(odf)} satır")

    # 9) State save/load
    state = eng2.get_state()
    eng3 = PaperTradeEngine(100_000.0)
    eng3.set_state(state)
    assert eng3.cash == eng2.cash
    print(f"\n✅ State save/load: nakit eşit={abs(eng3.cash - eng2.cash) < 1e-9}")

    print("\n✅ Paper Trade testi geçti")
