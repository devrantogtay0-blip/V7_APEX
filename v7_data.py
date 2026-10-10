# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — VERİ
YFinance + KAP + FRED + Groq · Disk cache · Paralel çekim
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from v7_config import (
    CACHE_DIR, GROQ_API_KEY, GROQ_MODEL, MACRO_KEYS, STOCK_LIST, WARMUP_BARS,
)

log = logging.getLogger("v7.data")
if not log.handlers:
    log.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════
# DISK CACHE
# ═══════════════════════════════════════════════════════════════════
_CACHE_LOCK = threading.RLock()


def _cache_path(key: str) -> str:
    """Cache anahtarından güvenli dosya yolu üretir."""
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + "_" + key[:30].replace("/", "_").replace("=", "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str, ttl: float = 3600.0) -> Optional[object]:
    """TTL içindeyse cache'ten nesneyi döndürür, yoksa None."""
    p = _cache_path(key)
    with _CACHE_LOCK:
        try:
            if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < ttl:
                with open(p, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            log.warning("Cache okuma hatası (%s): %s", key, e)
    return None


def cache_put(key: str, obj: object) -> None:
    """Nesneyi diske yazar."""
    p = _cache_path(key)
    with _CACHE_LOCK:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)
        except Exception as e:
            log.warning("Cache yazma hatası (%s): %s", key, e)


# ═══════════════════════════════════════════════════════════════════
# YFINANCE VERİ ÇEKİCİ
# ═══════════════════════════════════════════════════════════════════
class YFinanceFetcher:
    """24 hisse + makro veriyi çeker, diske cache'ler."""

    COLS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, cache_ttl: float = 3600.0) -> None:
        self.cache_ttl = cache_ttl

    def fetch(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """Tek sembol için OHLCV DataFrame döndürür."""
        key = f"yf_{symbol}_{period}"
        hit = cache_get(key, self.cache_ttl)
        if hit is not None:
            return hit

        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if df is None or len(df) == 0:
                log.warning("YFinance boş veri: %s", symbol)
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[self.COLS].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")]
            cache_put(key, df)
            return df
        except Exception as e:
            log.error("YFinance hatası (%s): %s", symbol, e)
            return cache_get(key, 1e12)

    def fetch_many(self, symbols: List[str], period: str = "5y",
                   workers: int = 8) -> Dict[str, pd.DataFrame]:
        """Paralel veri çeker."""
        out: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {s: ex.submit(self.fetch, s, period) for s in symbols}
            for s, fu in futs.items():
                try:
                    df = fu.result()
                    if df is not None and len(df) >= WARMUP_BARS:
                        out[s] = df
                except Exception as e:
                    log.warning("fetch_many hatası (%s): %s", s, e)
        log.info("YFinance: %d/%d sembol alındı", len(out), len(symbols))
        return out

    def fetch_stocks(self, codes: Optional[List[str]] = None,
                     period: str = "5y") -> Dict[str, pd.DataFrame]:
        """BIST kodlarını (.IS ekleyerek) çeker."""
        codes = codes or STOCK_LIST
        raw = self.fetch_many([f"{c}.IS" for c in codes], period)
        return {k[:-3] if k.endswith(".IS") else k: v for k, v in raw.items()}

    def fetch_macro(self, period: str = "5y") -> pd.DataFrame:
        """8 makro sembolü çeker."""
        syms = {
            "usdtry": "USDTRY=X", "eurtry": "EURTRY=X",
            "gold": "GC=F", "brent": "BZ=F",
            "vix": "^VIX", "dxy": "DX-Y.NYB",
            "oil": "CL=F", "copper": "HG=F",
        }
        raw = self.fetch_many(list(syms.values()), period, workers=4)
        cols = {}
        for k, sym in syms.items():
            if sym in raw:
                cols[k] = raw[sym]["Close"]
        if not cols:
            return pd.DataFrame()
        return pd.concat(cols, axis=1).sort_index().ffill()


# ═══════════════════════════════════════════════════════════════════
# PANEL HİZALAMA
# ═══════════════════════════════════════════════════════════════════
def align_panel(dfs: Dict[str, pd.DataFrame], codes: List[str]) -> Dict[str, object]:
    """
    Hisse DataFrame'lerini ortak tarih ekseninde (S, T) paneline hizalar.
    Dönüş: O, H, L, C, V (S,T), dates, valid, avail, codes
    """
    have = [c for c in codes if c in dfs and len(dfs[c]) >= WARMUP_BARS]
    if not have:
        raise ValueError("Hizalanacak yeterli veri yok")

    idx = sorted(set().union(*[set(dfs[c].index) for c in have]))
    dates = pd.DatetimeIndex(idx)
    S, T = len(have), len(dates)

    O = np.zeros((S, T)); H = np.zeros((S, T)); L = np.zeros((S, T))
    C = np.zeros((S, T)); V = np.zeros((S, T))
    avail = np.zeros((S, T), dtype=bool)

    for s, c in enumerate(have):
        d = dfs[c].reindex(dates)
        avail[s] = d["Close"].notna().values
        px = d[["Open", "High", "Low", "Close"]].ffill(limit=5)
        O[s], H[s], L[s], C[s] = (px[k].values for k in ("Open", "High", "Low", "Close"))
        V[s] = d["Volume"].fillna(0.0).values

    for a in (O, H, L, C):
        np.nan_to_num(a, copy=False, nan=0.0)

    valid = avail.sum(axis=1) >= WARMUP_BARS

    return {
        "O": np.ascontiguousarray(O),
        "H": np.ascontiguousarray(H),
        "L": np.ascontiguousarray(L),
        "C": np.ascontiguousarray(C),
        "V": np.ascontiguousarray(V),
        "dates": dates,
        "valid": valid,
        "avail": avail,
        "codes": have,
    }


# ═══════════════════════════════════════════════════════════════════
# KAP FETCHER
# ═══════════════════════════════════════════════════════════════════
class KAPFetcher:
    """KAP bildirimlerini çeker (savunmacı parsing)."""

    KW = ["temettü", "sermaye", "birleşme", "satın alma",
          "yeni iş", "geri alım", "bedelsiz", "esas sözleşme"]
    ZERO = {"kap_count_1d": 0.0, "kap_count_7d": 0.0, "kap_impact_score": 0.0}
    URL = "https://www.kap.org.tr/tr/api/disclosures"

    def fetch(self, code: str, days: int = 7) -> Dict[str, float]:
        """Son N gün KAP istatistiği. Hata durumunda sıfır döner."""
        key = f"kap_{code}_{days}"
        hit = cache_get(key, 3600.0)
        if hit is not None:
            return hit

        try:
            import requests
            end = pd.Timestamp.now().normalize()
            body = {
                "fromDate": (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                "toDate": end.strftime("%Y-%m-%d"),
                "stockCodes": [code],
            }
            r = requests.post(self.URL, json=body, timeout=10,
                              headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            r.raise_for_status()
            items = r.json()
            if isinstance(items, dict):
                items = items.get("data", items.get("disclosures", []))

            c1 = c7 = 0
            impact = 0.0
            now = pd.Timestamp.now()
            for it in (items if isinstance(items, list) else []):
                txt = json.dumps(it, ensure_ascii=False).lower()
                ts = pd.to_datetime(it.get("publishDate") or it.get("date"), errors="coerce")
                age = (now - ts).days if pd.notna(ts) else days
                c7 += 1
                c1 += 1 if age <= 1 else 0
                impact += sum(1.0 for k in self.KW if k in txt)

            res = {
                "kap_count_1d": float(c1),
                "kap_count_7d": float(c7),
                "kap_impact_score": float(np.tanh(impact / 3.0)),
            }
            cache_put(key, res)
            return res
        except Exception as e:
            log.debug("KAP alınamadı (%s): %s", code, e)
            return dict(self.ZERO)


# ═══════════════════════════════════════════════════════════════════
# FRED (makro)
# ═══════════════════════════════════════════════════════════════════
class FREDFetcher:
    """FRED serilerini anahtarsız CSV olarak çeker."""

    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

    def fetch(self, sid: str) -> Optional[pd.Series]:
        """Tek FRED serisi."""
        key = f"fred_{sid}"
        hit = cache_get(key, 6 * 3600.0)
        if hit is not None:
            return hit
        try:
            df = pd.read_csv(self.URL.format(sid=sid))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            s = pd.to_numeric(df["value"], errors="coerce")
            s.index = df["date"]
            s = s.dropna()
            cache_put(key, s)
            return s
        except Exception as e:
            log.warning("FRED alınamadı (%s): %s", sid, e)
            return None


# ═══════════════════════════════════════════════════════════════════
# HABER (Groq LLM + Lexicon fallback)
# ═══════════════════════════════════════════════════════════════════
class NewsAnalyzer:
    """Haber sentiment analizi."""

    POS = ["rekor", "artış", "kâr", "büyüme", "temettü", "anlaşma", "yükseldi", "onay"]
    NEG = ["zarar", "düşüş", "ceza", "dava", "iflas", "geriledi", "soruşturma", "uyarı"]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None
        if GROQ_API_KEY:
            try:
                from groq import Groq
                self._client = Groq(api_key=GROQ_API_KEY)
                log.info("Groq client hazır")
            except Exception as e:
                log.warning("Groq kurulamadı: %s", e)

    def _lexicon(self, title: str) -> Tuple[float, float]:
        """Basit sözlük tabanlı sentiment."""
        t = title.lower()
        p = sum(1 for w in self.POS if w in t)
        n = sum(1 for w in self.NEG if w in t)
        if p + n == 0:
            return 0.0, 0.2
        return (p - n) / float(p + n), min(0.6, 0.3 + 0.1 * (p + n))

    def analyze(self, stock: str, title: str) -> Tuple[float, float]:
        """(sentiment, confidence) döndürür."""
        key = f"news_{stock}_{hashlib.md5(title.encode()).hexdigest()[:12]}"
        hit = cache_get(key, 6 * 3600.0)
        if hit is not None:
            return hit

        res = self._lexicon(title)

        if self._client is not None:
            try:
                msg = (
                    "Borsa İstanbul hissesi için haber başlığını değerlendir. "
                    'Yalnızca JSON döndür: {"sentiment": -1..1, "confidence": 0..1}. '
                    f"Hisse: {stock}. Başlık: {title}"
                )
                r = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": msg}],
                    temperature=0,
                    max_tokens=60,
                    response_format={"type": "json_object"},
                )
                j = json.loads(r.choices[0].message.content)
                res = (
                    float(np.clip(j.get("sentiment", 0.0), -1, 1)),
                    float(np.clip(j.get("confidence", 0.5), 0, 1)),
                )
            except Exception as e:
                log.debug("Groq analiz hatası (%s): %s", stock, e)

        cache_put(key, res)
        return res

    def headlines(self, code: str, max_n: int = 5) -> List[str]:
        """YFinance haber başlıklarını döndürür."""
        try:
            import yfinance as yf
            items = yf.Ticker(f"{code}.IS").news or []
            out: List[str] = []
            for it in items[:max_n]:
                t = it.get("title") or (it.get("content") or {}).get("title")
                if t:
                    out.append(str(t))
            return out
        except Exception as e:
            log.debug("Haber başlıkları alınamadı (%s): %s", code, e)
            return []


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI
# ═══════════════════════════════════════════════════════════════════
def clear_cache() -> None:
    """Tüm cache dosyalarını siler."""
    try:
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(CACHE_DIR, f))
                except OSError:
                    pass
        log.info("Cache temizlendi")
    except Exception as e:
        log.warning("Cache temizleme hatası: %s", e)


def fetch_all(period: str = "5y") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Tüm hisse + makro veriyi çeker."""
    yf = YFinanceFetcher()
    stocks = yf.fetch_stocks(period=period)
    macro = yf.fetch_macro(period=period)
    return stocks, macro


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Data Test")
    print(f"Hisse sayısı: {len(STOCK_LIST)}")
    print(f"Makro sayısı: {len(MACRO_KEYS)}")

    # Cache testi
    cache_put("test_key", {"a": 1})
    assert cache_get("test_key", 60) == {"a": 1}
    print("✅ Cache testi geçti")

    # Sentiment testi (offline)
    na = NewsAnalyzer()
    s1, c1 = na._lexicon("Şirket kâr açıkladı, hisse yükseldi")
    s2, c2 = na._lexicon("Şirket zarar açıkladı, düşüş")
    print(f"✅ Lexicon: pozitif={s1:.2f}, negatif={s2:.2f}")
    assert s1 > 0 and s2 < 0

    print("✅ Data modülü hazır")
