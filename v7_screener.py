# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — SCREENER
BIST hisse tarayıcı · Çoklu kriter · Sinyal taraması · Watchlist
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from v7_config import STOCK_LIST, WARMUP_BARS

log = logging.getLogger("v7.screener")


# ═══════════════════════════════════════════════════════════════════
# KRİTER TANIMI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Criterion:
    """Tek tarama kriteri."""
    name: str
    column: str
    op: str  # ">", "<", ">=", "<=", "==", "between"
    value: Any
    value2: Optional[float] = None

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """DataFrame'e uygula, bool maskesi döner."""
        if self.column not in df.columns:
            return pd.Series(False, index=df.index)
        col = df[self.column]
        if self.op == ">":
            return col > self.value
        elif self.op == "<":
            return col < self.value
        elif self.op == ">=":
            return col >= self.value
        elif self.op == "<=":
            return col <= self.value
        elif self.op == "==":
            return col == self.value
        elif self.op == "between":
            return (col >= self.value) & (col <= self.value2)
        return pd.Series(True, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# SCREENER
# ═══════════════════════════════════════════════════════════════════

class Screener:
    """
    BIST hisselerini çoklu kriterle tarar.
    """

    def __init__(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """
        data_map: {"GARAN": DataFrame, ...}
        """
        self.data = data_map
        self.results: pd.DataFrame = pd.DataFrame()
        self.criteria: List[Criterion] = []
        self._lock = threading.RLock()

    def add_criterion(self, name: str, column: str, op: str, value: Any,
                      value2: Optional[float] = None) -> None:
        """Kriter ekle."""
        self.criteria.append(Criterion(name, column, op, value, value2))

    def clear_criteria(self) -> None:
        """Kriterleri temizle."""
        self.criteria = []

    def build_features(self) -> pd.DataFrame:
        """
        Her hisse için son bar özelliklerini hesaplar.
        Döner: DataFrame (satır=hisse, kolon=özellik)
        """
        rows = []
        for code, df in self.data.items():
            if df is None or len(df) < 60:
                continue
            try:
                feat = self._stock_features(df)
                feat["stock"] = code
                rows.append(feat)
            except Exception as e:
                log.debug("Feature hatası (%s): %s", code, e)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).set_index("stock")
        return out

    def _stock_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Tek hisse için tarama özellikleri."""
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        # SMA
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()

        # ATR
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()

        # Volatilite
        ret = c.pct_change()
        rv20 = ret.rolling(20).std() * np.sqrt(252)

        # Hacim
        vol_rel = v / v.rolling(20).mean()

        # Getiriler
        ret1 = c.pct_change(1).iloc[-1] * 100
        ret5 = c.pct_change(5).iloc[-1] * 100 if len(c) > 5 else 0
        ret20 = c.pct_change(20).iloc[-1] * 100 if len(c) > 20 else 0
        ret60 = c.pct_change(60).iloc[-1] * 100 if len(c) > 60 else 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        change = (last["Close"] - prev["Close"]) / prev["Close"] * 100

        return {
            "price": float(c.iloc[-1]),
            "change_pct": float(change),
            "volume": float(v.iloc[-1]),
            "volume_rel": float(vol_rel.iloc[-1]) if not pd.isna(vol_rel.iloc[-1]) else 1.0,
            "rsi14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            "macd_hist": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
            "macd_cross": float(macd.iloc[-1] - macd_sig.iloc[-1]) if not pd.isna(macd_sig.iloc[-1]) else 0.0,
            "sma20": float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else 0.0,
            "sma50": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else 0.0,
            "sma200": float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0.0,
            "above_sma20": float(c.iloc[-1] > sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else 0.0,
            "above_sma50": float(c.iloc[-1] > sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else 0.0,
            "above_sma200": float(c.iloc[-1] > sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0.0,
            "atr_pct": float(atr.iloc[-1] / c.iloc[-1] * 100) if not pd.isna(atr.iloc[-1]) else 0.0,
            "rv20": float(rv20.iloc[-1]) if not pd.isna(rv20.iloc[-1]) else 0.0,
            "ret1": float(ret1),
            "ret5": float(ret5),
            "ret20": float(ret20),
            "ret60": float(ret60),
        }

    def scan(self, top_n: int = 20) -> pd.DataFrame:
        """
        Tarama yap, kriterlere uyanları döndür.
        """
        with self._lock:
            features = self.build_features()
            if features.empty:
                self.results = features
                return features

            # Kriterleri uygula
            if self.criteria:
                mask = pd.Series(True, index=features.index)
                for crit in self.criteria:
                    mask &= crit.apply(features)
                result = features[mask].copy()
            else:
                result = features.copy()

            # Skor hesapla
            result["score"] = self._compute_score(result)
            result = result.sort_values("score", ascending=False)
            self.results = result.head(top_n)
            return self.results

    @staticmethod
    def _compute_score(df: pd.DataFrame) -> pd.Series:
        """0-100 arası puan."""
        score = pd.Series(0.0, index=df.index)

        # RSI (30 altı iyi, 70 üstü kötü)
        if "rsi14" in df.columns:
            score += np.where(df["rsi14"] < 30, 20,
                       np.where(df["rsi14"] < 45, 10,
                       np.where(df["rsi14"] > 70, -15, 0)))

        # MACD histogram pozitif
        if "macd_hist" in df.columns:
            score += np.where(df["macd_hist"] > 0, 10, -5)

        # Trend
        if "above_sma200" in df.columns:
            score += df["above_sma200"] * 15

        if "above_sma50" in df.columns:
            score += df["above_sma50"] * 5

        # Getiri
        if "ret20" in df.columns:
            score += np.clip(df["ret20"], -10, 10)

        # Hacim
        if "volume_rel" in df.columns:
            score += np.where(df["volume_rel"] > 1.5, 10, 0)

        # Volatilite (düşük = iyi)
        if "rv20" in df.columns:
            score += np.where(df["rv20"] < 30, 10, 0)

        return score.clip(0, 100)


# ═══════════════════════════════════════════════════════════════════
# HAZIR TARAMALAR
# ═══════════════════════════════════════════════════════════════════

def scan_oversold(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aşırı satım taraması."""
    s = Screener(data)
    s.add_criterion("RSI < 30", "rsi14", "<", 30)
    s.add_criterion("SMA200 üstünde", "above_sma200", "==", 1)
    return s.scan(20)


def scan_momentum(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Momentum taraması."""
    s = Screener(data)
    s.add_criterion("MACD +", "macd_hist", ">", 0)
    s.add_criterion("RSI 50-70", "rsi14", "between", 50, 70)
    s.add_criterion("Ret20 > 5", "ret20", ">", 5)
    return s.scan(20)


def scan_breakout(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Kırılım taraması."""
    s = Screener(data)
    s.add_criterion("Hacim 2x", "volume_rel", ">", 2.0)
    s.add_criterion("Getiri +3", "ret1", ">", 3)
    s.add_criterion("SMA20 üstü", "above_sma20", "==", 1)
    return s.scan(20)


def scan_reversal(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dönüş taraması."""
    s = Screener(data)
    s.add_criterion("RSI 30-45", "rsi14", "between", 30, 45)
    s.add_criterion("MACD kesişim +", "macd_cross", ">", 0)
    return s.scan(20)


def scan_volume_surge(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hacim patlaması."""
    s = Screener(data)
    s.add_criterion("Hacim 3x", "volume_rel", ">", 3.0)
    return s.scan(20)


def scan_low_vol(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Düşük volatilite."""
    s = Screener(data)
    s.add_criterion("RV20 < 20", "rv20", "<", 20)
    s.add_criterion("SMA200 üstü", "above_sma200", "==", 1)
    return s.scan(20)


# ═══════════════════════════════════════════════════════════════════
# WATCHLIST
# ═══════════════════════════════════════════════════════════════════

class Watchlist:
    """Kullanıcı watchlist yönetimi."""

    def __init__(self, max_n: int = 50) -> None:
        self.max_n = int(max_n)
        self.codes: List[str] = []
        self.notes: Dict[str, str] = {}

    def add(self, code: str, note: str = "") -> bool:
        """Hisse ekle."""
        code = code.upper()
        if code in self.codes:
            return False
        if len(self.codes) >= self.max_n:
            return False
        self.codes.append(code)
        self.notes[code] = note
        return True

    def remove(self, code: str) -> bool:
        """Hisse çıkar."""
        code = code.upper()
        if code in self.codes:
            self.codes.remove(code)
            self.notes.pop(code, None)
            return True
        return False

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame olarak döndür."""
        return pd.DataFrame({
            "stock": self.codes,
            "note": [self.notes.get(c, "") for c in self.codes],
        })

    def state(self) -> Dict[str, Any]:
        return {"codes": self.codes, "notes": self.notes}

    def set_state(self, s: Dict) -> None:
        self.codes = list(s.get("codes", []))
        self.notes = dict(s.get("notes", {}))


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Screener Test")

    # Sentetik veri
    rng = np.random.default_rng(7)
    data = {}
    for i, code in enumerate(["GARAN", "AKBNK", "THYAO", "TUPRS", "ASELS",
                              "EREGL", "BIMAS", "FROTO", "KCHOL", "SISE"]):
        T = 300
        drift = 0.0005 * (i - 5)
        C = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, T)))
        df = pd.DataFrame({
            "Open": C * (1 + rng.normal(0, 0.002, T)),
            "High": C * (1 + np.abs(rng.normal(0, 0.005, T))),
            "Low": C * (1 - np.abs(rng.normal(0, 0.005, T))),
            "Close": C,
            "Volume": rng.uniform(1e5, 1e6, T),
        }, index=pd.date_range("2023-01-02", periods=T))
        data[code] = df

    # 1) Features
    s = Screener(data)
    feats = s.build_features()
    print(f"\n✅ Features: {feats.shape}")
    print(f"   Kolonlar: {list(feats.columns)[:8]}...")

    # 2) Tam tarama
    result = s.scan(top_n=10)
    print(f"\n✅ Skor tablosu (ilk 5):")
    print(result[["price", "change_pct", "rsi14", "score"]].head(5).to_string())

    # 3) Hazır taramalar
    for name, fn in [("Aşırı satım", scan_oversold),
                     ("Momentum", scan_momentum),
                     ("Kırılım", scan_breakout),
                     ("Dönüş", scan_reversal),
                     ("Hacim", scan_volume_surge)]:
        r = fn(data)
        print(f"\n✅ {name}: {len(r)} hisse")

    # 4) Watchlist
    wl = Watchlist()
    for c in ["GARAN", "AKBNK", "THYAO"]:
        wl.add(c, note=f"Not: {c}")
    print(f"\n✅ Watchlist: {wl.codes}")

    wl.remove("AKBNK")
    print(f"✅ Silme sonrası: {wl.codes}")
    assert "AKBNK" not in wl.codes

    # 5) State
    state = wl.state()
    wl2 = Watchlist()
    wl2.set_state(state)
    assert wl2.codes == wl.codes
    print(f"✅ State kaydet/yükle: OK")

    print("\n✅ Screener testi geçti")
