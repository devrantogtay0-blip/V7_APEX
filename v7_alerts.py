# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ALERTS
Fiyat alarmı · Sinyal alarmı · Portföy uyarıları · Konsol/Log/JSON çıkış
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from v7_config import CACHE_DIR

log = logging.getLogger("v7.alerts")

ALERTS_FILE = os.path.join(CACHE_DIR, "v7_alerts.json")


# ═══════════════════════════════════════════════════════════════════
# ALERT KAYDI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AlertRecord:
    """Tek alarm kaydı."""
    level: str  # INFO / WARNING / CRITICAL
    category: str  # PRICE / SIGNAL / RISK / SYSTEM
    stock: str
    message: str
    value: float = 0.0
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level, "category": self.category,
            "stock": self.stock, "message": self.message,
            "value": self.value, "ts": self.ts,
        }


# ═══════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═══════════════════════════════════════════════════════════════════

class AlertManager:
    """Uyarı toplama, kaydetme, izleme."""

    def __init__(self, max_alerts: int = 2000, auto_save: bool = True):
        self.max_alerts = int(max_alerts)
        self.auto_save = bool(auto_save)
        self.alerts: List[AlertRecord] = []
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[AlertRecord], None]] = []
        self._load()

    def _load(self) -> None:
        """Cache'ten yükle."""
        try:
            if os.path.exists(ALERTS_FILE):
                with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for d in data[-self.max_alerts:]:
                    self.alerts.append(AlertRecord(**d))
        except Exception as e:
            log.warning("Alerts yüklenemedi: %s", e)

    def _save(self) -> None:
        """Cache'e kaydet."""
        if not self.auto_save:
            return
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = ALERTS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in self.alerts[-self.max_alerts:]], f, ensure_ascii=False)
            os.replace(tmp, ALERTS_FILE)
        except Exception as e:
            log.warning("Alerts kaydedilemedi: %s", e)

    def register_callback(self, cb: Callable[[AlertRecord], None]) -> None:
        """Yeni alarm olduğunda çağrılacak fonksiyon."""
        self._callbacks.append(cb)

    def alert(self, level: str, category: str, stock: str, message: str,
              value: float = 0.0) -> AlertRecord:
        """Alarm ekle."""
        rec = AlertRecord(level=level, category=category, stock=stock,
                          message=message, value=value)
        with self._lock:
            self.alerts.append(rec)
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]

        # Log
        lvl = {"INFO": logging.INFO, "WARNING": logging.WARNING,
               "CRITICAL": logging.CRITICAL}.get(level, logging.INFO)
        log.log(lvl, "[%s/%s] %s: %s (%.4f)", level, category, stock, message, value)

        # Callback'ler
        for cb in self._callbacks:
            try:
                cb(rec)
            except Exception as e:
                log.warning("Callback hatası: %s", e)

        self._save()
        return rec

    def recent(self, n: int = 50, level: Optional[str] = None) -> List[Dict]:
        """Son n alarm."""
        with self._lock:
            items = self.alerts[-n:]
        if level:
            items = [a for a in items if a.level == level]
        return [a.to_dict() for a in items]

    def counts(self) -> Dict[str, int]:
        """Seviyeye göre sayı."""
        with self._lock:
            cnt = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
            for a in self.alerts:
                cnt[a.level] = cnt.get(a.level, 0) + 1
        return cnt

    def clear(self) -> None:
        """Tüm alarmları sil."""
        with self._lock:
            self.alerts = []
        self._save()


# ═══════════════════════════════════════════════════════════════════
# FİYAT ALARMLARI
# ═══════════════════════════════════════════════════════════════════

class PriceAlerts:
    """Fiyat eşiği alarmları."""

    def __init__(self, manager: AlertManager):
        self.mgr = manager
        self.rules: Dict[str, List[Dict]] = {}  # stock -> [{op, value}]

    def add(self, stock: str, op: str, value: float) -> None:
        """Kural ekle. op: above / below / cross_above / cross_below"""
        if stock not in self.rules:
            self.rules[stock] = []
        self.rules[stock].append({"op": op, "value": float(value), "last": None})

    def check(self, stock: str, price: float, prev_price: Optional[float] = None) -> List[AlertRecord]:
        """Fiyat kontrolü."""
        out = []
        for r in self.rules.get(stock, []):
            hit = False
            op = r["op"]
            v = r["value"]
            last = r.get("last")
            if op == "above" and price > v and (last is None or last <= v):
                hit = True
            elif op == "below" and price < v and (last is None or last >= v):
                hit = True
            elif op == "cross_above" and prev_price is not None and prev_price <= v < price:
                hit = True
            elif op == "cross_below" and prev_price is not None and prev_price >= v > price:
                hit = True
            r["last"] = price
            if hit:
                out.append(self.mgr.alert(
                    "WARNING", "PRICE", stock,
                    f"Fiyat {op} {v:.2f} → {price:.2f}", price))
        return out


# ═══════════════════════════════════════════════════════════════════
# SİNYAL ALARMLARI
# ═══════════════════════════════════════════════════════════════════

class SignalAlerts:
    """Sinyal bazlı alarmlar (AL, SAT, güçlü)."""

    def __init__(self, manager: AlertManager):
        self.mgr = manager
        self.last_state: Dict[str, str] = {}

    def check(self, stock: str, signal: str, p_up: float = 0.0, p_down: float = 0.0) -> Optional[AlertRecord]:
        """Sinyal değişimini kontrol et."""
        prev = self.last_state.get(stock)
        self.last_state[stock] = signal
        if prev == signal:
            return None

        if signal == "GÜÇLÜ AL":
            level = "INFO"
            msg = f"Güçlü AL sinyali (P_al={p_up*100:.0f}%)"
        elif signal == "AL":
            level = "INFO"
            msg = f"AL sinyali (P_al={p_up*100:.0f}%)"
        elif signal == "GÜÇLÜ SAT":
            level = "WARNING"
            msg = f"Güçlü SAT sinyali (P_sat={p_down*100:.0f}%)"
        elif signal == "SAT":
            level = "WARNING"
            msg = f"SAT sinyali (P_sat={p_down*100:.0f}%)"
        else:
            return None

        return self.mgr.alert(level, "SIGNAL", stock, msg, p_up if signal.startswith("G") or signal == "AL" else p_down)


# ═══════════════════════════════════════════════════════════════════
# PORTFÖY ALARMLARI
# ═══════════════════════════════════════════════════════════════════

class PortfolioAlerts:
    """Portföy risk limitleri."""

    def __init__(self, manager: AlertManager):
        self.mgr = manager
        self.last_dd_alert = 0.0
        self.last_day_alert = 0.0

    def check_dd(self, dd_pct: float, max_dd: float) -> Optional[AlertRecord]:
        """Drawdown kontrolü."""
        if dd_pct > max_dd and (dd_pct - self.last_dd_alert) > 1.0:
            self.last_dd_alert = dd_pct
            return self.mgr.alert(
                "CRITICAL", "RISK", "PORTFOLIO",
                f"Max DD aşıldı: %{dd_pct:.2f} > %{max_dd:.2f}", dd_pct)
        if dd_pct < self.last_dd_alert - 5.0:
            self.last_dd_alert = dd_pct
        return None

    def check_daily(self, daily_pnl_pct: float, daily_limit: float) -> Optional[AlertRecord]:
        """Günlük zarar kontrolü."""
        if daily_pnl_pct < -daily_limit and (self.last_day_alert - daily_pnl_pct) > 1.0:
            self.last_day_alert = daily_pnl_pct
            return self.mgr.alert(
                "CRITICAL", "RISK", "PORTFOLIO",
                f"Günlük zarar limiti: %{daily_pnl_pct:.2f} < -%{daily_limit:.2f}", daily_pnl_pct)
        return None

    def check_exposure(self, exposure: float, max_exposure: float) -> Optional[AlertRecord]:
        """Maruziyet kontrolü."""
        if exposure > max_exposure * 0.95:
            return self.mgr.alert(
                "WARNING", "RISK", "PORTFOLIO",
                f"Maruziyet yüksek: %{exposure*100:.1f} (limit %{max_exposure*100:.1f})", exposure)
        return None


# ═══════════════════════════════════════════════════════════════════
# SİSTEM ALARMLARI
# ═══════════════════════════════════════════════════════════════════

def system_alert(manager: AlertManager, message: str, level: str = "INFO") -> AlertRecord:
    """Sistem mesajı."""
    return manager.alert(level, "SYSTEM", "SYSTEM", message, 0.0)


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI
# ═══════════════════════════════════════════════════════════════════

_manager: Optional[AlertManager] = None
_manager_lock = threading.RLock()


def get_alert_manager() -> AlertManager:
    """Singleton manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = AlertManager()
        return _manager


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Alerts Test")

    # Temiz başla
    mgr = AlertManager(auto_save=False)
    print(f"\n✅ Manager: {len(mgr.alerts)} alarm")

    # 1) Basic alert
    mgr.alert("INFO", "SYSTEM", "BOT", "Sistem başladı", 1.0)
    mgr.alert("WARNING", "PRICE", "GARAN", "Fiyat arttı", 100.5)
    mgr.alert("CRITICAL", "RISK", "PORTFOLIO", "DD aşıldı", 15.5)
    print(f"✅ 3 alarm eklendi")

    cnt = mgr.counts()
    print(f"✅ Sayaçlar: {cnt}")
    assert cnt["INFO"] == 1 and cnt["WARNING"] == 1 and cnt["CRITICAL"] == 1

    # 2) Son alarmlar
    recent = mgr.recent(5)
    print(f"\n✅ Son 3 alarm:")
    for a in recent[-3:]:
        print(f"   [{a['level']}] {a['stock']}: {a['message']}")

    # 3) Fiyat alarmı
    pa = PriceAlerts(mgr)
    pa.add("GARAN", "above", 100.0)
    pa.add("GARAN", "cross_below", 95.0)

    a1 = pa.check("GARAN", 99.0)
    a2 = pa.check("GARAN", 101.0)  # tetiklenmeli
    a3 = pa.check("GARAN", 105.0)  # tetiklenmemeli (zaten üstünde)
    print(f"\n✅ Fiyat alarmı: {len(a1)}, {len(a2)}, {len(a3)} (beklenen 0,1,0)")

    # Cross test
    pa2 = PriceAlerts(mgr)
    pa2.add("AKBNK", "cross_below", 50.0)
    c1 = pa2.check("AKBNK", 51.0, prev_price=52.0)
    c2 = pa2.check("AKBNK", 49.5, prev_price=51.0)  # tetiklenmeli
    print(f"✅ Cross alarm: {len(c1)}, {len(c2)} (beklenen 0,1)")

    # 4) Sinyal alarmı
    sa = SignalAlerts(mgr)
    s1 = sa.check("THYAO", "AL", p_up=0.6)
    s2 = sa.check("THYAO", "AL", p_up=0.65)  # aynı sinyal, tetiklenmez
    s3 = sa.check("THYAO", "GÜÇLÜ SAT", p_down=0.7)  # değişim, tetiklenir
    print(f"\n✅ Sinyal alarmı: {s1 is not None}, {s2 is None}, {s3 is not None}")

    # 5) Portföy alarmları
    pav = PortfolioAlerts(mgr)
    d1 = pav.check_dd(10.0, 15.0)  # altında
    d2 = pav.check_dd(16.0, 15.0)  # üstünde, tetiklenmeli
    print(f"\n✅ DD alarm: {d1 is None}, {d2 is not None}")

    e1 = pav.check_exposure(0.90, 0.85)  # yüksek
    print(f"✅ Exposure alarm: {e1 is not None}")

    # 6) Callback
    received = []
    mgr2 = AlertManager(auto_save=False)
    mgr2.register_callback(lambda rec: received.append(rec))
    mgr2.alert("INFO", "TEST", "X", "callback testi")
    print(f"\n✅ Callback: {len(received)} alarm alındı")
    assert len(received) == 1

    # 7) Sistem alarmı
    srec = system_alert(mgr, "Test sistem mesajı", "INFO")
    print(f"✅ Sistem alarmı: {srec.category}")

    # 8) Toplam
    print(f"\n✅ Toplam alarm: {len(mgr.alerts)}")

    print("\n✅ Alerts testi geçti")
