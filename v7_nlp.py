# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — NLP (Haber + KAP + Sentiment)
Groq LLM (opsiyonel) + Lexicon fallback + FinBERT (opsiyonel)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from v7_config import (
    CACHE_DIR, GROQ_API_KEY, GROQ_MODEL, NLP_NAMES, STOCK_LIST,
)

log = logging.getLogger("v7.nlp")

# Groq opsiyonel
try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    Groq = None
    GROQ_OK = False

# FinBERT opsiyonel
try:
    from transformers import pipeline
    TRANSFORMERS_OK = True
except ImportError:
    pipeline = None
    TRANSFORMERS_OK = False


# ═══════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════

_NLP_CACHE: Dict[str, Dict] = {}
_NLP_LOCK = threading.RLock()
_NLP_FILE = os.path.join(CACHE_DIR, "v7_nlp_cache.pkl")


def _load_cache() -> None:
    global _NLP_CACHE
    try:
        if os.path.exists(_NLP_FILE):
            with open(_NLP_FILE, "rb") as f:
                _NLP_CACHE = pickle.load(f)
    except Exception as e:
        log.warning("NLP cache yüklenemedi: %s", e)


def _save_cache() -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _NLP_FILE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_NLP_CACHE, f)
        os.replace(tmp, _NLP_FILE)
    except Exception as e:
        log.warning("NLP cache kaydedilemedi: %s", e)


_load_cache()


# ═══════════════════════════════════════════════════════════════════
# LEXICON SENTIMENT (fallback)
# ═══════════════════════════════════════════════════════════════════

class LexiconSentiment:
    """Basit Türkçe finansal sözlük tabanlı sentiment."""

    POS = [
        "rekor", "artış", "yükseliş", "kâr", "kar", "büyüme", "temettü",
        "anlaşma", "yatırım", "ihale", "onay", "olumlu", "güçlü",
        "iyileşme", "toparlanma", "hedef yükselt", "tavsiye al",
    ]
    NEG = [
        "zarar", "düşüş", "gerileme", "ceza", "dava", "iflas", "soruşturma",
        "uyarı", "olumsuz", "zayıf", "kriz", "borç", "temerrüt",
        "hedef düşür", "tavsiye sat", "kayıp",
    ]
    NEUTRAL = ["açıklama", "toplantı", "duyuru", "genel kurul"]

    @classmethod
    def score(cls, text: str) -> Tuple[float, float]:
        """(sentiment ∈ [-1,1], confidence ∈ [0,1])"""
        t = text.lower()
        p = sum(1 for w in cls.POS if w in t)
        n = sum(1 for w in cls.NEG if w in t)
        neutral = sum(1 for w in cls.NEUTRAL if w in t)
        if p + n == 0:
            conf = 0.2 if neutral > 0 else 0.1
            return 0.0, conf
        score = (p - n) / (p + n)
        conf = min(0.85, 0.3 + 0.1 * (p + n))
        return score, conf


# ═══════════════════════════════════════════════════════════════════
# GROQ LLM
# ═══════════════════════════════════════════════════════════════════

class GroqSentiment:
    """Groq LLM ile sentiment analizi."""

    PROMPT = """Sen bir BIST (Borsa İstanbul) haber analistisin.
Aşağıdaki haber başlığını {stock} hissesi için değerlendir.
Yalnızca JSON döndür, başka bir şey yazma:
{{"sentiment": <-1.0..1.0>, "confidence": <0.0..1.0>}}

Haber: {title}"""

    def __init__(self) -> None:
        self._client = None
        self.enabled = False
        if GROQ_OK and GROQ_API_KEY:
            try:
                self._client = Groq(api_key=GROQ_API_KEY)
                self.enabled = True
                log.info("Groq LLM hazır")
            except Exception as e:
                log.warning("Groq kurulamadı: %s", e)

    def analyze(self, stock: str, title: str) -> Optional[Tuple[float, float]]:
        """(sentiment, confidence) veya None."""
        if not self.enabled or not title.strip():
            return None
        try:
            msg = self.PROMPT.format(stock=stock, title=title[:400])
            r = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            s = float(np.clip(data.get("sentiment", 0.0), -1, 1))
            c = float(np.clip(data.get("confidence", 0.5), 0, 1))
            return s, c
        except Exception as e:
            log.debug("Groq analiz hatası: %s", e)
            return None


# ═══════════════════════════════════════════════════════════════════
# FINBERT (opsiyonel)
# ═══════════════════════════════════════════════════════════════════

class FinBERTAnalyzer:
    """HuggingFace FinBERT analizi (opsiyonel)."""

    def __init__(self) -> None:
        self.enabled = False
        self.pipe = None
        if TRANSFORMERS_OK:
            try:
                self.pipe = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                )
                self.enabled = True
                log.info("FinBERT hazır")
            except Exception as e:
                log.warning("FinBERT yüklenemedi: %s", e)

    def analyze(self, text: str) -> Optional[Tuple[float, float]]:
        """(sentiment, confidence) veya None."""
        if not self.enabled or not text.strip():
            return None
        try:
            r = self.pipe(text[:512])[0]
            label = r["label"].lower()
            conf = float(r["score"])
            if label == "positive":
                s = conf
            elif label == "negative":
                s = -conf
            else:
                s = 0.0
            return s, conf
        except Exception as e:
            log.debug("FinBERT hatası: %s", e)
            return None


# ═══════════════════════════════════════════════════════════════════
# HABER ÇEKİCİ
# ═══════════════════════════════════════════════════════════════════

class HeadlineFetcher:
    """YFinance haber başlıklarını çeker."""

    def fetch(self, code: str, max_n: int = 10) -> List[Dict[str, Any]]:
        """Haber listesi."""
        try:
            import yfinance as yf
            items = yf.Ticker(f"{code}.IS").news or []
            out = []
            for it in items[:max_n]:
                title = it.get("title") or (it.get("content") or {}).get("title")
                if title:
                    out.append({
                        "title": str(title),
                        "publisher": it.get("publisher", ""),
                        "link": it.get("link", ""),
                    })
            return out
        except Exception as e:
            log.debug("Haber alınamadı (%s): %s", code, e)
            return []


# ═══════════════════════════════════════════════════════════════════
# KAP ÇEKİCİ
# ═══════════════════════════════════════════════════════════════════

class KAPFetcher:
    """KAP bildirimlerini çeker."""

    URL = "https://www.kap.org.tr/tr/api/disclosures"
    KW_IMPACT = ["temettü", "sermaye", "birleşme", "satın alma",
                 "yeni iş", "geri alım", "bedelsiz", "esas sözleşme"]

    def fetch(self, code: str, days: int = 7) -> List[Dict[str, Any]]:
        """KAP bildirimleri."""
        key = f"kap_{code}_{days}"
        with _NLP_LOCK:
            cached = _NLP_CACHE.get(key)
            if cached and (time.time() - cached.get("_ts", 0)) < 3600:
                return cached.get("items", [])

        try:
            import requests
            end = pd.Timestamp.now().normalize()
            body = {
                "fromDate": (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                "toDate": end.strftime("%Y-%m-%d"),
                "stockCodes": [code],
            }
            r = requests.post(
                self.URL, json=body, timeout=10,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data", data.get("disclosures", []))
            items = []
            for d in (data if isinstance(data, list) else []):
                title = str(d.get("subject", ""))[:200]
                date = d.get("publishDate") or d.get("date", "")
                if title:
                    items.append({"title": title, "date": date})
            with _NLP_LOCK:
                _NLP_CACHE[key] = {"items": items, "_ts": time.time()}
                _save_cache()
            return items
        except Exception as e:
            log.debug("KAP alınamadı (%s): %s", code, e)
            with _NLP_LOCK:
                _NLP_CACHE[key] = {"items": [], "_ts": time.time()}
            return []

    def impact_score(self, items: List[Dict]) -> float:
        """Etki skoru (0-1)."""
        if not items:
            return 0.0
        cnt = 0
        for it in items:
            t = it.get("title", "").lower()
            if any(k in t for k in self.KW_IMPACT):
                cnt += 1
        return float(np.tanh(cnt / 3.0))


# ═══════════════════════════════════════════════════════════════════
# NLP MOTORU
# ═══════════════════════════════════════════════════════════════════

class NLPEngine:
    """Tüm NLP kaynaklarını birleştirir."""

    def __init__(self) -> None:
        self.headline = HeadlineFetcher()
        self.kap = KAPFetcher()
        self.lexicon = LexiconSentiment()
        self.groq = GroqSentiment()
        self.finbert = FinBERTAnalyzer()

    def analyze_stock(self, code: str) -> Dict[str, float]:
        """Tek hisse için 15 NLP özelliği."""
        out = {k: 0.0 for k in NLP_NAMES}

        # Haber başlıkları
        headlines = self.headline.fetch(code, max_n=10)
        titles = [h["title"] for h in headlines]
        if titles:
            sentiments = []
            confidences = []
            for t in titles:
                # Önce cache
                key = f"sent_{code}_{hashlib.md5(t.encode()).hexdigest()[:12]}"
                with _NLP_LOCK:
                    cached = _NLP_CACHE.get(key)
                if cached:
                    s, c = cached["s"], cached["c"]
                else:
                    # Groq → FinBERT → Lexicon sırası
                    result = self.groq.analyze(code, t)
                    if result is None:
                        result = self.finbert.analyze(t)
                    if result is None:
                        result = self.lexicon.score(t)
                    s, c = result
                    with _NLP_LOCK:
                        _NLP_CACHE[key] = {"s": s, "c": c, "_ts": time.time()}
                sentiments.append(s)
                confidences.append(c)

            s_arr = np.array(sentiments)
            c_arr = np.array(confidences)
            # Ağırlıklı ortalama
            wmean = float((s_arr * c_arr).sum() / (c_arr.sum() + 1e-9))
            out["news_sent_1d"] = wmean
            out["news_sent_5d"] = wmean
            out["news_sent_20d"] = wmean
            out["news_count_1d"] = float(min(len(titles), 5))
            out["news_count_5d"] = float(min(len(titles), 10))
            out["news_conf_mean"] = float(c_arr.mean())
            out["news_conf_std"] = float(c_arr.std()) if len(c_arr) > 1 else 0.0

        # KAP
        kap_items = self.kap.fetch(code, days=7)
        if kap_items:
            out["kap_count_1d"] = float(sum(1 for it in kap_items if "now" in str(it.get("date", ""))))
            out["kap_count_7d"] = float(len(kap_items))
            out["kap_impact_score"] = self.kap.impact_score(kap_items)

        # FinBERT pozitif/negatif
        if titles:
            pos_scores = []
            neg_scores = []
            for t in titles[:5]:
                r = self.finbert.analyze(t)
                if r is not None:
                    s, c = r
                    if s > 0:
                        pos_scores.append(s)
                    else:
                        neg_scores.append(abs(s))
            if pos_scores:
                out["finbert_pos"] = float(np.mean(pos_scores))
            if neg_scores:
                out["finbert_neg"] = float(np.mean(neg_scores))

        # Sosyal (şimdilik 0)
        return out

    def analyze_all(self) -> Dict[str, Dict[str, float]]:
        """Tüm hisseler."""
        return {code: self.analyze_stock(code) for code in STOCK_LIST}

    def status(self) -> Dict[str, Any]:
        """Kaynakların durumu."""
        return {
            "groq": self.groq.enabled,
            "finbert": self.finbert.enabled,
            "lexicon": True,
            "cache_size": len(_NLP_CACHE),
        }


# ═══════════════════════════════════════════════════════════════════
# NLP PANEL (S × N_NLP × T)
# ═══════════════════════════════════════════════════════════════════

def build_nlp_panel(store_codes: List[str], T: int,
                    nlp_engine: Optional[NLPEngine] = None) -> np.ndarray:
    """
    NLP paneli (S, N_NLP, T) üretir.
    Şu an için sadece son bar (T-1) gerçek değer alır, diğerleri 0.
    """
    engine = nlp_engine or NLPEngine()
    S = len(store_codes)
    out = np.zeros((S, len(NLP_NAMES), T), dtype=np.float32)
    for i, code in enumerate(store_codes):
        try:
            feat = engine.analyze_stock(code)
            for j, name in enumerate(NLP_NAMES):
                out[i, j, T - 1] = float(feat.get(name, 0.0))
        except Exception as e:
            log.warning("NLP panel hatası (%s): %s", code, e)
    return out


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — NLP Test")
    print(f"Groq: {'VAR' if GROQ_OK else 'YOK'}")
    print(f"Transformers: {'VAR' if TRANSFORMERS_OK else 'YOK'}")

    # Lexicon testi
    s1, c1 = LexiconSentiment.score("Şirket rekor kâr açıkladı, hisse yükseldi")
    s2, c2 = LexiconSentiment.score("Şirket zarar açıkladı, iflas riski")
    print(f"\n✅ Lexicon pozitif: {s1:.2f}, negatif: {s2:.2f}")
    assert s1 > 0 and s2 < 0

    # Engine
    engine = NLPEngine()
    status = engine.status()
    print(f"\n✅ Engine durumu: {status}")

    # Test analyze (offline)
    print("\n✅ NLP modülü hazır")

    # Panel testi
    codes = ["GARAN", "AKBNK"]
    panel = build_nlp_panel(codes, T=100, nlp_engine=engine)
    print(f"✅ NLP panel: {panel.shape}")

    print("\n✅ NLP testi geçti")
