# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — DATABASE
DuckDB (öncelikli) + SQLite (fallback) · 35 tablo · Thread-safe
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from v7_config import DB_FILE

# DuckDB opsiyonel
try:
    import duckdb
    DUCKDB_OK = True
except ImportError:
    duckdb = None
    DUCKDB_OK = False

log = logging.getLogger("v7.db")
if not log.handlers:
    log.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════
# TABLO ŞEMALARI (35 Tablo)
# ═══════════════════════════════════════════════════════════════════
DB_TABLES: Dict[str, str] = {
    "trades": "ts TEXT, stock TEXT, action TEXT, price REAL, qty INTEGER, pnl REAL, reason TEXT",
    "equity": "ts TEXT, value REAL, cash REAL, npos INTEGER",
    "train": "ts TEXT, step INTEGER, vl REAL, va REAL, note TEXT",
    "improve": 'ts TEXT, reason TEXT, detail TEXT, "before" REAL, "after" REAL, adopted INTEGER',
    "calibrate": "ts TEXT, n INTEGER, brier REAL, ece REAL, acc REAL, base REAL",
    "cpcv": "ts TEXT, mean REAL, std REAL, worst REAL, best REAL, n INTEGER, sharpe REAL",
    "pbo": "ts TEXT, pbo REAL, mean REAL, median REAL, n INTEGER, interp TEXT",
    "forecast": (
        "ts TEXT, stock TEXT, current_price REAL, target_price REAL, expected_return REAL, "
        "lower_bound REAL, upper_bound REAL, p_up REAL, p_down REAL, p_flat REAL, "
        "bull_price REAL, base_price REAL, bear_price REAL, confidence REAL, "
        "uncertainty REAL, epistemic REAL, aleatoric REAL, kelly_f REAL, "
        "regime TEXT, horizon INTEGER"
    ),
    "execution": (
        "ts TEXT, stock TEXT, side TEXT, qty INTEGER, ref_price REAL, fill_price REAL, "
        "impact_bps REAL, slip_bps REAL, ofi REAL, adv_pct REAL"
    ),
    "risk": "ts TEXT, var95 REAL, cvar95 REAL, gross REAL, net REAL, kelly_scale REAL, n_pos INTEGER",
    "moe": "ts TEXT, step INTEGER, lb_loss REAL, entropy REAL, max_load REAL",
    "foundation": "ts TEXT, model_name TEXT, pretrain_loss REAL, finetune_loss REAL, embedding_dim INTEGER",
    "rl": "ts TEXT, episode INTEGER, reward REAL, policy_loss REAL, value_loss REAL, entropy REAL",
    "causal": "ts TEXT, treatment TEXT, outcome TEXT, effect REAL, p_value REAL, method TEXT",
    "xai": "ts TEXT, method TEXT, feature TEXT, importance REAL, shap_value REAL",
    "uncertainty": "ts TEXT, aleatoric REAL, epistemic REAL, total REAL, method TEXT",
    "conformal": "ts TEXT, alpha REAL, coverage REAL, width REAL, method TEXT",
    "regime": "ts TEXT, regime TEXT, prob REAL, transition REAL",
    "chronos": "ts TEXT, stock TEXT, horizon INTEGER, forecast REAL, lower REAL, upper REAL",
    "world_model": "ts TEXT, latent_state BLOB, reward REAL, value REAL, done INTEGER",
    "decision_transformer": "ts TEXT, episode INTEGER, return_to_go REAL, action BLOB",
    "swag": "ts TEXT, mean BLOB, var BLOB, samples INTEGER",
    "nf": "ts TEXT, log_prob REAL, sample BLOB",
    "grad_cam": "ts TEXT, layer TEXT, importance BLOB",
    "iceberg": "ts TEXT, stock TEXT, side TEXT, confidence REAL, size INTEGER",
    "spoof": "ts TEXT, stock TEXT, side TEXT, severity REAL, timestamp TEXT",
    "hawkes": "ts TEXT, mu REAL, alpha REAL, beta REAL, intensity REAL",
    "lob": (
        "ts TEXT, stock TEXT, level INTEGER, bid_price REAL, bid_size INTEGER, "
        "ask_price REAL, ask_size INTEGER"
    ),
    "vpin": "ts TEXT, stock TEXT, vpin REAL, bucket INTEGER",
    "kyle": "ts TEXT, stock TEXT, kyle_lambda REAL, window INTEGER",
    "amihud": "ts TEXT, stock TEXT, amihud REAL, window INTEGER",
    "hasbrouck": "ts TEXT, stock TEXT, information_share REAL, window INTEGER",
    "white_rc": "ts TEXT, statistic REAL, p_value REAL, n_bootstrap INTEGER",
    "spa": "ts TEXT, statistic REAL, p_value REAL, n_bootstrap INTEGER",
    "mcs": "ts TEXT, model TEXT, mcs_prob REAL, n_bootstrap INTEGER",
    "structural_break": (
        "ts TEXT, statistic REAL, p_value REAL, break_date TEXT, method TEXT"
    ),
}


def _parse_schema(schema: str) -> List[Tuple[str, str]]:
    """'a TEXT, b REAL' şemasını (kolon, tip) listesine çevirir."""
    cols: List[Tuple[str, str]] = []
    for part in schema.split(","):
        part = part.strip()
        if not part:
            continue
        name, typ = part.split(None, 1)
        cols.append((name, typ.strip()))
    return cols


# ═══════════════════════════════════════════════════════════════════
# SeekDB — Ana Sınıf
# ═══════════════════════════════════════════════════════════════════
class SeekDB:
    """Thread-safe veritabanı katmanı: DuckDB varsa onu, yoksa SQLite'ı kullanır."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self.backend = "duckdb" if DUCKDB_OK else "sqlite"
        base = path or DB_FILE
        if self.backend == "sqlite" and base.endswith(".duckdb"):
            base = base[:-7] + ".sqlite"
        self.path = base
        try:
            if self.backend == "duckdb":
                self._con = duckdb.connect(self.path)
            else:
                self._con = sqlite3.connect(self.path, check_same_thread=False)
        except Exception as e:
            log.error("DB bağlantı hatası (%s), SQLite'a düşülüyor: %s", self.backend, e)
            self.backend = "sqlite"
            self.path = base.rsplit(".", 1)[0] + ".sqlite"
            self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._create_tables()
        log.info("DB hazır: %s (%s, %d tablo)", self.path, self.backend, len(DB_TABLES))

    def _create_tables(self) -> None:
        """Tüm tabloları (yoksa) oluşturur."""
        with self._lock:
            for name, schema in DB_TABLES.items():
                cols = ", ".join(f'"{c}" {t}' for c, t in _parse_schema(schema))
                self._con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" ({cols})')
            if self.backend == "sqlite":
                self._con.commit()

    def insert(self, table: str, values: Sequence[Any]) -> None:
        """Tek satır ekler."""
        self.insert_many(table, [values])

    def insert_many(self, table: str, rows: Sequence[Sequence[Any]]) -> None:
        """Çok satır ekler."""
        if table not in DB_TABLES:
            raise KeyError(f"Bilinmeyen tablo: {table}")
        if not rows:
            return
        n = len(_parse_schema(DB_TABLES[table]))
        ph = ", ".join("?" * n)
        with self._lock:
            try:
                for r in rows:
                    if len(r) != n:
                        raise ValueError(f"{table}: {n} kolon bekleniyordu, {len(r)} geldi")
                    self._con.execute(f'INSERT INTO "{table}" VALUES ({ph})', list(r))
                if self.backend == "sqlite":
                    self._con.commit()
            except Exception as e:
                log.error("DB yazma hatası (%s): %s", table, e)
                raise

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """SQL sorgusunu DataFrame olarak döndürür."""
        with self._lock:
            try:
                if self.backend == "duckdb":
                    return self._con.execute(sql, list(params or [])).fetchdf()
                return pd.read_sql_query(sql, self._con, params=list(params or []))
            except Exception as e:
                log.error("DB sorgu hatası: %s", e)
                return pd.DataFrame()

    def count(self, table: str) -> int:
        """Tablodaki satır sayısı."""
        df = self.query(f'SELECT COUNT(*) AS n FROM "{table}"')
        return int(df["n"].iloc[0]) if len(df) else 0

    def table_names(self) -> List[str]:
        """Şemadaki tablo isimleri."""
        return list(DB_TABLES.keys())

    def clear(self, table: Optional[str] = None) -> None:
        """Tablo(lar)ı temizler."""
        with self._lock:
            tables = [table] if table else self.table_names()
            for t in tables:
                if t in DB_TABLES:
                    self._con.execute(f'DELETE FROM "{t}"')
            if self.backend == "sqlite":
                self._con.commit()

    def close(self) -> None:
        """Bağlantıyı kapatır."""
        with self._lock:
            try:
                self._con.close()
            except Exception as e:
                log.warning("DB kapatma uyarısı: %s", e)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════
_DB_INSTANCE: Optional[SeekDB] = None
_DB_LOCK = threading.RLock()


def get_db() -> SeekDB:
    """Tek DB örneği döndürür."""
    global _DB_INSTANCE
    with _DB_LOCK:
        if _DB_INSTANCE is None:
            _DB_INSTANCE = SeekDB()
        return _DB_INSTANCE


@contextmanager
def db_session():
    """with bloğu için DB context manager."""
    db = get_db()
    try:
        yield db
    except Exception:
        raise


def _ts() -> str:
    """ISO timestamp."""
    return datetime.now().isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════════
# Kısayol fonksiyonları
# ═══════════════════════════════════════════════════════════════════
def log_trade(stock: str, action: str, price: float, qty: int, pnl: float = 0.0, reason: str = "") -> None:
    """İşlem kaydı ekler."""
    get_db().insert("trades", [_ts(), stock, action, float(price), int(qty), float(pnl), reason[:300]])


def log_equity(value: float, cash: float, npos: int) -> None:
    """Özkaynak kaydı ekler."""
    get_db().insert("equity", [_ts(), float(value), float(cash), int(npos)])


def log_train(step: int, vl: float, va: float, note: str = "") -> None:
    """Eğitim kaydı ekler."""
    get_db().insert("train", [_ts(), int(step), float(vl), float(va), note[:300]])


def log_moe(step: int, lb_loss: float, entropy: float, max_load: float) -> None:
    """MoE kaydı ekler."""
    get_db().insert("moe", [_ts(), int(step), float(lb_loss), float(entropy), float(max_load)])


def log_risk(var95: float, cvar95: float, gross: float, net: float,
             kelly_scale: float, n_pos: int) -> None:
    """Risk kaydı ekler."""
    get_db().insert("risk", [_ts(), float(var95), float(cvar95), float(gross),
                             float(net), float(kelly_scale), int(n_pos)])


def recent_trades(limit: int = 100) -> List[Dict]:
    """Son işlemleri döndürür."""
    df = get_db().query(
        "SELECT ts, stock, action, price, qty, pnl, reason FROM trades ORDER BY id DESC LIMIT ?",
        [int(limit)]
    )
    return df.to_dict("records") if not df.empty else []


def recent_equity(limit: int = 500) -> List[Dict]:
    """Son özkaynak kayıtları."""
    df = get_db().query(
        "SELECT ts, value, cash, npos FROM equity ORDER BY id DESC LIMIT ?",
        [int(limit)]
    )
    return df.to_dict("records")[::-1] if not df.empty else []


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — DB Test")
    print(f"DuckDB: {'VAR' if DUCKDB_OK else 'YOK → SQLite'}")
    print(f"Toplam tablo: {len(DB_TABLES)}")

    db = SeekDB(":memory:" if not DUCKDB_OK else None)
    print(f"\nTablo sayısı: {len(db.table_names())}")

    # Test insert
    db.insert("trades", [_ts(), "GARAN", "BUY", 100.0, 10, 0.0, "test"])
    db.insert("equity", [_ts(), 100000.0, 50000.0, 1])

    print(f"trades satır: {db.count('trades')}")
    print(f"equity satır: {db.count('equity')}")

    # Query test
    df = db.query("SELECT * FROM trades")
    print(f"\nSon eklenen işlem:\n{df.to_string(index=False)}")

    db.clear("trades")
    print(f"\nTemizleme sonrası trades: {db.count('trades')}")

    db.close()
    print("\n✅ DB testi geçti")
