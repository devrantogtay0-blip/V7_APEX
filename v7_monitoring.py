# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — MONITORING
Prometheus metrics · Health check · Alerts · Performance logger · Structured log
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("v7.monitoring")

# Prometheus opsiyonel
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROM_OK = True
except ImportError:
    Counter = Gauge = Histogram = None
    start_http_server = None
    PROM_OK = False


METRICS_FILE = "v7_metrics.prom"


# ═══════════════════════════════════════════════════════════════════
# PROMETHEUS METRİKLERİ
# ═══════════════════════════════════════════════════════════════════

class Metrics:
    """Prometheus metrik kayıtları (opsiyonel)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = PROM_OK and enabled
        if self.enabled:
            self.trades = Counter("v7_trades_total", "Total trades", ["stock", "action"])
            self.equity = Gauge("v7_equity", "Current equity (TL)")
            self.cash = Gauge("v7_cash", "Cash (TL)")
            self.n_pos = Gauge("v7_n_positions", "Open positions")
            self.sharpe = Gauge("v7_sharpe", "Rolling Sharpe")
            self.max_dd = Gauge("v7_max_dd_pct", "Max drawdown (%)")
            self.latency = Histogram("v7_latency_seconds", "Execution latency")
            self.pred_latency = Histogram("v7_pred_latency_seconds", "Prediction latency")
            self.data_latency = Histogram("v7_data_latency_seconds", "Data fetch latency")
            self.err = Counter("v7_errors_total", "Errors", ["component"])
        else:
            self.trades = self.equity = self.cash = self.n_pos = None
            self.sharpe = self.max_dd = self.latency = self.pred_latency = None
            self.data_latency = self.err = None

    def inc_trade(self, stock: str, action: str) -> None:
        if self.enabled and self.trades:
            self.trades.labels(stock=stock, action=action).inc()

    def set_equity(self, val: float) -> None:
        if self.enabled and self.equity:
            self.equity.set(float(val))

    def set_cash(self, val: float) -> None:
        if self.enabled and self.cash:
            self.cash.set(float(val))

    def set_npos(self, val: int) -> None:
        if self.enabled and self.n_pos:
            self.n_pos.set(int(val))

    def set_sharpe(self, val: float) -> None:
        if self.enabled and self.sharpe:
            self.sharpe.set(float(val))

    def set_max_dd(self, val: float) -> None:
        if self.enabled and self.max_dd:
            self.max_dd.set(float(val))

    def observe_latency(self, val: float) -> None:
        if self.enabled and self.latency:
            self.latency.observe(float(val))

    def inc_error(self, component: str) -> None:
        if self.enabled and self.err:
            self.err.labels(component=component).inc()


_METRICS: Optional[Metrics] = None
_METRICS_LOCK = threading.RLock()


def get_metrics() -> Metrics:
    """Singleton metrik."""
    global _METRICS
    with _METRICS_LOCK:
        if _METRICS is None:
            _METRICS = Metrics()
        return _METRICS


def start_prometheus_server(port: int = 8000) -> bool:
    """Prometheus HTTP server başlat."""
    if not PROM_OK:
        log.warning("prometheus_client yok")
        return False
    try:
        start_http_server(port)
        log.info("Prometheus server başladı: :%d", port)
        return True
    except Exception as e:
        log.error("Prometheus server hatası: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED LOGGER (JSON)
# ═══════════════════════════════════════════════════════════════════

class StructuredLogger:
    """JSON formatında yapılandırılmış log kaydı."""

    def __init__(self, path: str = "v7_structured.jsonl") -> None:
        self.path = path
        self._lock = threading.RLock()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass

    def log(self, event: str, **kwargs: Any) -> None:
        """JSON kaydı ekle."""
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **kwargs,
        }
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                log.warning("Structured logger hatası: %s", e)

    def read(self, n: int = 100) -> List[Dict]:
        """Son n kaydı oku."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-n:]
            return [json.loads(line) for line in lines if line.strip()]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HealthStatus:
    """Sağlık durumu."""
    status: str = "OK"  # OK / DEGRADED / FAIL
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now().isoformat(timespec="seconds")


class HealthChecker:
    """Sistem sağlık kontrolü."""

    def __init__(self, cache_dir: str = "cache_v7") -> None:
        self.cache_dir = cache_dir

    def check_data_freshness(self, max_age_hours: float = 24.0) -> bool:
        """Veri önbelleği taze mi?"""
        try:
            if not os.path.isdir(self.cache_dir):
                return False
            files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]
            if not files:
                return False
            latest = max(os.path.getmtime(f) for f in files)
            age_h = (time.time() - latest) / 3600.0
            return age_h < max_age_hours
        except Exception:
            return False

    def check_disk(self, min_gb: float = 0.5) -> bool:
        """Disk alanı yeterli mi?"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            return (free / (1024 ** 3)) >= min_gb
        except Exception:
            return True

    def check_memory(self, max_pct: float = 95.0) -> bool:
        """RAM kullanımı makul mü?"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.percent < max_pct
        except ImportError:
            return True
        except Exception:
            return True

    def check_modules(self) -> Dict[str, bool]:
        """Modül durumu."""
        out = {}
        try:
            import numpy, pandas, scipy
            out["numpy"] = True
            out["pandas"] = True
            out["scipy"] = True
        except ImportError:
            pass
        try:
            import numba
            out["numba"] = True
        except ImportError:
            out["numba"] = False
        try:
            import torch
            out["torch"] = True
        except ImportError:
            out["torch"] = False
        try:
            import plotly
            out["plotly"] = True
        except ImportError:
            out["plotly"] = False
        try:
            import streamlit
            out["streamlit"] = True
        except ImportError:
            out["streamlit"] = False
        return out

    def run_all(self) -> HealthStatus:
        """Tüm testler."""
        h = HealthStatus()
        h.checks["data_freshness"] = self.check_data_freshness()
        h.checks["disk"] = self.check_disk()
        h.checks["memory"] = self.check_memory()
        h.checks["modules"] = all(self.check_modules().values())

        if not h.checks["data_freshness"]:
            h.messages.append("Veri önbelleği eski")
        if not h.checks["disk"]:
            h.messages.append("Disk alanı az")
        if not h.checks["memory"]:
            h.messages.append("RAM kullanımı yüksek")

        n_fail = sum(1 for v in h.checks.values() if not v)
        if n_fail == 0:
            h.status = "OK"
        elif n_fail <= 1:
            h.status = "DEGRADED"
        else:
            h.status = "FAIL"
        return h


# ═══════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """Uyarı kaydı."""
    level: str  # INFO / WARNING / CRITICAL
    message: str
    ts: str = ""
    component: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now().isoformat(timespec="seconds")


class AlertManager:
    """Uyarı toplama ve izleme."""

    def __init__(self, max_alerts: int = 1000) -> None:
        self.max_alerts = int(max_alerts)
        self.alerts: List[Alert] = []
        self._lock = threading.RLock()

    def alert(self, level: str, message: str, component: str = "") -> None:
        """Uyarı ekle."""
        with self._lock:
            self.alerts.append(Alert(level=level, message=message, component=component))
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        log.log(
            {"INFO": logging.INFO, "WARNING": logging.WARNING, "CRITICAL": logging.CRITICAL}.get(level, logging.INFO),
            "[%s] %s: %s", level, component, message,
        )

    def recent(self, n: int = 20) -> List[Dict]:
        """Son n uyarı."""
        with self._lock:
            return [a.__dict__ for a in self.alerts[-n:]]

    def counts(self) -> Dict[str, int]:
        """Seviyeye göre uyarı sayısı."""
        with self._lock:
            cnt = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
            for a in self.alerts:
                cnt[a.level] = cnt.get(a.level, 0) + 1
        return cnt


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE MONITOR
# ═══════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """Süre ölçümü + rolling istatistik."""

    def __init__(self, window: int = 100) -> None:
        self.window = int(window)
        self.times: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def record(self, name: str, seconds: float) -> None:
        """Süre kaydı."""
        with self._lock:
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(float(seconds))
            if len(self.times[name]) > self.window:
                self.times[name] = self.times[name][-self.window:]

    def stats(self, name: str) -> Dict[str, float]:
        """İstatistik."""
        with self._lock:
            arr = self.times.get(name, [])
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "n": len(a),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        """Tüm süre istatistikleri."""
        with self._lock:
            names = list(self.times.keys())
        return {name: self.stats(name) for name in names}


# ═══════════════════════════════════════════════════════════════════
# SÜRE ÖLÇÜCÜ (context manager)
# ═══════════════════════════════════════════════════════════════════

class timer:
    """with timer("name", monitor): ..."""

    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.t0
        if self.monitor:
            self.monitor.record(self.name, elapsed)
        return False


# ═══════════════════════════════════════════════════════════════════
# SİSTEM DURUMU
# ═══════════════════════════════════════════════════════════════════

def system_summary() -> Dict[str, Any]:
    """Genel sistem özeti."""
    import platform
    out = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        out["ram_pct"] = mem.percent
        out["ram_free_gb"] = round(mem.available / (1024 ** 3), 2)
        disk = psutil.disk_usage(".")
        out["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)
    except ImportError:
        pass
    return out


def write_metrics_prom(m: Dict[str, float]) -> None:
    """Prometheus formatına yaz (dosya)."""
    try:
        tmp = METRICS_FILE + ".tmp"
        with open(tmp, "w") as f:
            for k, v in m.items():
                try:
                    f.write(f"v7_{k} {float(v):.6f}\n")
                except Exception:
                    pass
        os.replace(tmp, METRICS_FILE)
    except Exception as e:
        log.warning("Metrics write hatası: %s", e)


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Monitoring Test")
    print(f"Prometheus: {'VAR' if PROM_OK else 'YOK'}")

    # 1) Metrics
    m = get_metrics()
    m.inc_trade("GARAN", "BUY")
    m.set_equity(105_000.0)
    m.set_sharpe(1.85)
    print(f"\n✅ Metrics enabled={m.enabled}")

    # 2) Structured logger
    slog = StructuredLogger(path="tmp_v7_structured.jsonl")
    slog.log("trade", stock="GARAN", action="BUY", price=100.0, qty=100)
    slog.log("equity", value=105_000.0)
    records = slog.read(10)
    print(f"\n✅ Structured log: {len(records)} kayıt")
    assert len(records) >= 2

    # 3) Health check
    hc = HealthChecker()
    hs = hc.run_all()
    print(f"\n✅ Health: {hs.status}")
    print(f"   Checks: {hs.checks}")
    print(f"   Mesajlar: {hs.messages}")

    # 4) Alert manager
    am = AlertManager()
    am.alert("INFO", "Bot başladı", "core")
    am.alert("WARNING", "Slippage yüksek", "exec")
    am.alert("CRITICAL", "Kill-switch", "risk")
    cnt = am.counts()
    print(f"\n✅ Alerts: {cnt}")
    assert cnt["WARNING"] == 1 and cnt["CRITICAL"] == 1

    # 5) Performance monitor
    pm = PerformanceMonitor(window=20)
    for i in range(10):
        with timer("data", pm):
            time.sleep(0.001)
    for i in range(10):
        with timer("predict", pm):
            time.sleep(0.002)
    stats = pm.all_stats()
    print(f"\n✅ Performance:")
    for name, s in stats.items():
        print(f"   {name}: mean={s['mean']*1000:.2f}ms p95={s['p95']*1000:.2f}ms")

    # 6) System summary
    ss = system_summary()
    print(f"\n✅ Sistem: py={ss['python']}, cpu={ss.get('cpu_pct', '-')}%, ram={ss.get('ram_pct', '-')}%")

    # 7) Write metrics
    write_metrics_prom({"equity": 105_000.0, "sharpe": 1.85})
    print(f"\n✅ Metrics dosya yazıldı: {METRICS_FILE}")

    # Cleanup
    try:
        os.remove("tmp_v7_structured.jsonl")
        os.remove(METRICS_FILE)
    except Exception:
        pass

    print("\n✅ Monitoring testi geçti")
