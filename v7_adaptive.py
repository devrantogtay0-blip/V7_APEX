# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ADAPTIVE / SELF-IMPROVE
ADWIN drift · Experience Replay · Online Learning · Self-Improver · Tuner
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("v7.adaptive")


# ═══════════════════════════════════════════════════════════════════
# ADWIN (Kavram Kayması Dedektörü)
# ═══════════════════════════════════════════════════════════════════

class ADWIN:
    """
    Basit ADWIN drift dedektörü (Bifet & Gavaldà 2007).
    Pencere: son W gözlem. Alt pencereler arasında ortalama farkı eşiği aşarsa drift.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 500):
        self.delta = float(delta)
        self.max_window = int(max_window)
        self.window: deque = deque(maxlen=max_window)
        self.drift_detected = False
        self.n_detections = 0

    def add(self, x: float) -> bool:
        """Yeni gözlem ekle, drift varsa True döner."""
        self.window.append(float(x))
        self.drift_detected = self._detect()
        if self.drift_detected:
            self.n_detections += 1
            # Pencereyi yarıya indir
            keep = list(self.window)[len(self.window) // 2:]
            self.window.clear()
            self.window.extend(keep)
        return self.drift_detected

    def _detect(self) -> bool:
        n = len(self.window)
        if n < 20:
            return False
        w = np.array(self.window)
        # Farklı noktalarda böl, ortalama farkını ölç
        best_score = 0.0
        for cut in range(10, n - 10, max(1, n // 20)):
            left = w[:cut]
            right = w[cut:]
            mean_diff = abs(left.mean() - right.mean())
            var_l = left.var(ddof=1)
            var_r = right.var(ddof=1)
            m = 1.0 / (len(left) - 1) + 1.0 / (len(right) - 1)
            eps = math.sqrt(2 * m * (var_l / len(left) + var_r / len(right)) *
                            math.log(2 / self.delta))
            if mean_diff > eps:
                score = mean_diff - eps
                if score > best_score:
                    best_score = score
        return best_score > 0

    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    def std(self) -> float:
        return float(np.std(self.window, ddof=1)) if len(self.window) > 1 else 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "n": len(self.window),
            "mean": self.mean(),
            "std": self.std(),
            "n_detections": self.n_detections,
        }


# ═══════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY
# ═══════════════════════════════════════════════════════════════════

class ExperienceReplay:
    """Ring buffer — eski örnekleri tutar."""

    def __init__(self, capacity: int = 10000, n_feat: int = 306):
        self.capacity = int(capacity)
        self.n_feat = int(n_feat)
        self.X = np.zeros((capacity, n_feat), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.int64)
        self.w = np.ones(capacity, dtype=np.float32)
        self.ptr = 0
        self.n = 0

    def add(self, x: np.ndarray, y: int, w: float = 1.0) -> None:
        """Tek örnek ekle."""
        self.add_many(x[None, :], np.array([y]), np.array([w]))

    def add_many(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        """Çoklu ekleme."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        m = len(X)
        if m == 0:
            return
        if m > self.capacity:
            X, y = X[-self.capacity:], y[-self.capacity:]
            if w is not None:
                w = np.asarray(w)[-self.capacity:]
            m = self.capacity
        idx = (self.ptr + np.arange(m)) % self.capacity
        self.X[idx] = X
        self.y[idx] = y
        if w is not None:
            self.w[idx] = np.asarray(w, dtype=np.float32)
        else:
            self.w[idx] = 1.0
        self.ptr = int((self.ptr + m) % self.capacity)
        self.n = min(self.n + m, self.capacity)

    def sample(self, k: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ağırlıklı örnekleme."""
        if self.n < 8:
            return (np.zeros((0, self.n_feat), np.float32),
                    np.zeros(0, np.int64), np.zeros(0, np.float32))
        rng = np.random.default_rng(seed)
        k = min(k, self.n)
        cdf = np.cumsum(self.w[:self.n].astype(np.float64))
        idx = np.minimum(np.searchsorted(cdf, rng.random(k) * cdf[-1]), self.n - 1)
        return self.X[idx].copy(), self.y[idx].copy(), self.w[idx].copy()

    def size(self) -> int:
        return self.n

    def save(self, path: str) -> bool:
        try:
            n = self.n
            np.savez_compressed(path, X=self.X[:n], y=self.y[:n], w=self.w[:n])
            return True
        except Exception as e:
            log.warning("Replay save hatası: %s", e)
            return False

    def load(self, path: str) -> bool:
        try:
            z = np.load(path)
            self.X = z["X"][:self.capacity]
            self.y = z["y"][:self.capacity]
            self.w = z["w"][:self.capacity]
            self.n = len(self.X)
            self.ptr = self.n % self.capacity
            return True
        except Exception as e:
            log.warning("Replay load hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# ONLINE LEARNER (Basit SGD)
# ═══════════════════════════════════════════════════════════════════

class OnlineLearner:
    """
    Basit online lojistik regresyon (SGD).
    Kütüphane bağımsız, river olmadan çalışır.
    """

    def __init__(self, n_feat: int = 306, n_classes: int = 3, lr: float = 0.001):
        self.n_feat = int(n_feat)
        self.n_classes = int(n_classes)
        self.lr = float(lr)
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_feat, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.n_seen = 0

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._softmax(X @ self.W + self.b)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, lr_mult: float = 1.0) -> float:
        """Tek adım SGD."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if len(X) == 0:
            return 0.0
        p = self.predict_proba(X)
        # CE gradyanı
        y_oh = np.eye(self.n_classes, dtype=np.float32)[np.clip(y, 0, self.n_classes - 1)]
        grad = (p - y_oh) / len(X)
        self.W -= self.lr * lr_mult * (X.T @ grad)
        self.b -= self.lr * lr_mult * grad.sum(axis=0)
        # Kayıp
        loss = -np.mean(np.log(p[np.arange(len(y)), np.clip(y, 0, self.n_classes - 1)] + 1e-9))
        self.n_seen += len(X)
        return float(loss)

    def state(self) -> Dict[str, Any]:
        return {"n_seen": self.n_seen, "W_norm": float(np.linalg.norm(self.W))}


# ═══════════════════════════════════════════════════════════════════
# SHADOW MODE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ShadowDecision:
    """Gölge modu kararı."""
    t: int
    stock: str
    decision: str
    realized_ret: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0


class ShadowMode:
    """Reddedilen sinyalleri simüle eder."""

    def __init__(self, max_n: int = 500):
        self.max_n = int(max_n)
        self.decisions: List[ShadowDecision] = []

    def log(self, t: int, stock: str, decision: str) -> None:
        """Kararı kaydet."""
        self.decisions.append(ShadowDecision(t=t, stock=stock, decision=decision))
        if len(self.decisions) > self.max_n:
            self.decisions = self.decisions[-self.max_n:]

    def resolve(self, t: int, stock: str, entry: float, exit_: float) -> None:
        """Sonucu kaydet."""
        for d in reversed(self.decisions):
            if d.t == t and d.stock == stock and d.realized_ret == 0.0:
                d.entry_price = entry
                d.exit_price = exit_
                d.realized_ret = (exit_ / entry - 1.0) if entry > 0 else 0.0
                return

    def stats(self) -> Dict[str, float]:
        """Gölge karar istatistikleri."""
        if not self.decisions:
            return {"n": 0, "avg_ret": 0.0, "win_rate": 0.0}
        rets = np.array([d.realized_ret for d in self.decisions if d.realized_ret != 0.0])
        if len(rets) == 0:
            return {"n": len(self.decisions), "avg_ret": 0.0, "win_rate": 0.0}
        return {
            "n": len(rets),
            "avg_ret": float(rets.mean()),
            "win_rate": float((rets > 0).mean()),
            "total_ret": float(rets.sum()),
        }


# ═══════════════════════════════════════════════════════════════════
# TUNER (Random + Evolution Strategy)
# ═══════════════════════════════════════════════════════════════════

PARAM_SPACE: Dict[str, Tuple[float, float, bool]] = {
    "sl_atr": (1.5, 4.0, False),
    "tp_r": (1.0, 3.5, False),
    "p_buy": (0.40, 0.60, False),
    "p_sell": (0.40, 0.60, False),
    "margin": (0.0, 0.20, False),
    "max_hold": (20, 150, True),
    "trail_act_r": (0.8, 2.5, False),
    "trail_atr": (1.5, 4.0, False),
    "kelly_frac": (0.10, 0.50, False),
    "unc_max": (0.15, 0.40, False),
}


class Tuner:
    """Rastgele arama + (1+λ)-ES."""

    def __init__(self, space: Optional[Dict[str, Tuple[float, float, bool]]] = None,
                 seed: int = 0):
        self.space = dict(PARAM_SPACE if space is None else space)
        self.rng = np.random.default_rng(seed)
        self.evals = 0
        self.history: List[Dict[str, Any]] = []

    def _clip(self, name: str, v: float) -> float:
        lo, hi, is_int = self.space[name]
        v = min(max(float(v), lo), hi)
        return int(round(v)) if is_int else v

    def sample(self) -> Dict[str, float]:
        """Rastgele örnek."""
        return {k: self._clip(k, lo + (hi - lo) * self.rng.random())
                for k, (lo, hi, _) in self.space.items()}

    def mutate(self, p: Dict[str, float], sigma: float = 0.1) -> Dict[str, float]:
        """Gauss mutasyon."""
        out = {}
        for k, (lo, hi, _) in self.space.items():
            u = (float(p[k]) - lo) / (hi - lo) if hi > lo else 0.5
            u = float(np.clip(u + sigma * self.rng.standard_normal(), 0.0, 1.0))
            out[k] = self._clip(k, lo + (hi - lo) * u)
        return out

    def tune(self, base_rp: Dict[str, Any], evaluate: Callable[[Dict], float],
             budget_s: float = 30.0, n_random: int = 20, n_es: int = 40,
             lam: int = 4, now: Callable = time.time
             ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Ayarlama. Dönüş: (en iyi rp, en iyi skor, geçmiş)."""
        t0 = now()
        full = lambda p: {**base_rp, **p}
        best_p = {k: base_rp[k] for k in self.space if k in base_rp}
        best_s = float(evaluate(full(best_p)))
        self.evals += 1
        hist = [{"gen": "base", "score": best_s, **best_p}]

        # Random
        for _ in range(n_random):
            if now() - t0 > budget_s:
                break
            p = self.sample()
            s = float(evaluate(full(p)))
            self.evals += 1
            hist.append({"gen": "rand", "score": s, **p})
            if s > best_s:
                best_s, best_p = s, p

        # ES
        for g in range(n_es):
            if now() - t0 > budget_s:
                break
            for _ in range(lam):
                if now() - t0 > budget_s:
                    break
                c = self.mutate(best_p, 0.1)
                s = float(evaluate(full(c)))
                self.evals += 1
                hist.append({"gen": f"es{g}", "score": s, **c})
                if s > best_s:
                    best_s, best_p = s, c

        self.history.extend(hist)
        return full(best_p), best_s, hist


# ═══════════════════════════════════════════════════════════════════
# SELF-IMPROVER
# ═══════════════════════════════════════════════════════════════════

class SelfImprover:
    """
    Otonom parametre iyileştirme.
    Ardışık iyileşme sonrası adopt.
    """

    def __init__(self, tuner: Optional[Tuner] = None, consec_needed: int = 3):
        self.tuner = tuner or Tuner()
        self.streak = 0
        self.consec_needed = int(consec_needed)
        self.last_run = -1e18
        self.history: List[Dict[str, Any]] = []

    def due(self, rp: Dict[str, Any], now: float) -> bool:
        """Aralık doldu mu?"""
        return bool(rp.get("si_auto", True)) and (now - self.last_run) >= float(rp["si_interval_min"]) * 60.0

    def run(self, rp: Dict[str, Any],
            eval_val: Callable[[Dict], float],
            eval_test: Callable[[Dict], float],
            dsr_fn: Optional[Callable[[Dict], float]] = None,
            clock: Callable = time.time) -> Dict[str, Any]:
        """Bir iyileştirme turu."""
        now = clock()
        self.last_run = now
        best_rp, cand_val, hist = self.tuner.tune(
            rp, eval_val, budget_s=float(rp.get("si_budget_s", 30)), now=clock)
        base_val = float(eval_val(rp))
        base_test = float(eval_test(rp))
        cand_test = float(eval_test(best_rp))
        improved = cand_val > base_val + 1e-9
        test_ok = cand_test >= base_test - float(rp.get("test_score_tol", 0.15)) * abs(base_test)
        dsr = float(dsr_fn(best_rp)) if dsr_fn else 1.0
        dsr_ok = dsr >= float(rp.get("dsr_confidence", 0.90))
        self.streak = self.streak + 1 if (improved and test_ok and dsr_ok) else 0
        adopt = bool(rp.get("auto_adopt", True) and self.streak >= self.consec_needed)
        rec = {
            "ts": now, "before": base_val, "after": cand_val,
            "test_before": base_test, "test_after": cand_test,
            "dsr": dsr, "streak": self.streak, "adopted": adopt,
            "evals": len(hist),
        }
        self.history.append(rec)
        if adopt:
            self.streak = 0
        return {"record": rec, "adopt": adopt, "params": best_rp}


# ═══════════════════════════════════════════════════════════════════
# DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════════

class DriftMonitor:
    """Model hatası izleme + drift tespiti."""

    def __init__(self, delta: float = 0.002, max_window: int = 500):
        self.adwin = ADWIN(delta=delta, max_window=max_window)
        self.history: List[float] = []

    def update(self, error: float) -> bool:
        """Hata → drift kontrolü."""
        self.history.append(float(error))
        if len(self.history) > 10000:
            self.history = self.history[-5000:]
        return self.adwin.add(error)

    def status(self) -> Dict[str, Any]:
        return self.adwin.status()


# ═══════════════════════════════════════════════════════════════════
# CONTINUAL LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════

class ContinualLearningLoop:
    """Drift tespit + replay + online learner döngüsü."""

    def __init__(self, n_feat: int = 306, n_classes: int = 3):
        self.drift = DriftMonitor()
        self.replay = ExperienceReplay(capacity=10000, n_feat=n_feat)
        self.learner = OnlineLearner(n_feat=n_feat, n_classes=n_classes)
        self.shadow = ShadowMode()
        self.n_updates = 0

    def observe(self, x: np.ndarray, y: int, error: float, w: float = 1.0) -> bool:
        """Yeni örnek geldi."""
        drift = self.drift.update(error)
        self.replay.add(x, y, w)
        self.n_updates += 1
        return drift

    def update_model(self, batch_size: int = 32) -> Optional[float]:
        """Replay'den örnek al, güncelle."""
        X, y, w = self.replay.sample(batch_size, seed=self.n_updates)
        if len(X) == 0:
            return None
        return self.learner.partial_fit(X, y)

    def status(self) -> Dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "drift": self.drift.status(),
            "replay_size": self.replay.size(),
            "learner": self.learner.state(),
            "shadow": self.shadow.stats(),
        }


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Adaptive Test")
    rng = np.random.default_rng(7)

    # 1) ADWIN
    adwin = ADWIN(delta=0.01, max_window=200)
    for _ in range(100):
        adwin.add(rng.normal(0, 1))
    no_drift = adwin.n_detections
    for _ in range(100):
        adwin.add(rng.normal(5, 1))
    drift = adwin.n_detections
    print(f"\n✅ ADWIN: statik={no_drift}, drift={drift}")
    assert drift > no_drift

    # 2) Replay
    replay = ExperienceReplay(capacity=100, n_feat=10)
    for i in range(50):
        replay.add(rng.standard_normal(10), y=i % 3, w=1.0)
    print(f"\n✅ Replay: n={replay.size()}")
    X, y, w = replay.sample(8)
    assert X.shape == (8, 10)

    # 3) OnlineLearner
    learner = OnlineLearner(n_feat=10, n_classes=3)
    X_train = rng.standard_normal((200, 10))
    y_train = (X_train[:, 0] > 0).astype(int) + (X_train[:, 1] > 0).astype(int)
    for _ in range(20):
        loss = learner.partial_fit(X_train, y_train)
    print(f"✅ OnlineLearner: loss={loss:.4f}")

    # 4) Tuner
    tuner = Tuner(seed=3)
    fake_t = {"t": 0.0}
    def clock():
        fake_t["t"] += 0.01
        return fake_t["t"]
    base_rp = {"sl_atr": 2.5, "tp_r": 2.0, "p_buy": 0.45, "p_sell": 0.45,
               "margin": 0.10, "max_hold": 120, "trail_act_r": 1.5,
               "trail_atr": 2.5, "kelly_frac": 0.25, "unc_max": 0.25,
               "si_budget_s": 0.5, "si_interval_min": 30, "si_auto": True,
               "dsr_confidence": 0.5, "test_score_tol": 0.15, "auto_adopt": True}
    best, bs, hist = tuner.tune(base_rp, lambda p: -(p["sl_atr"] - 2.0) ** 2,
                                 budget_s=0.5, now=clock)
    print(f"\n✅ Tuner: {len(hist)} eval, en iyi skor={bs:.4f}")

    # 5) SelfImprover
    si = SelfImprover(Tuner(seed=5), consec_needed=1)
    res = si.run(base_rp,
                 lambda p: -(p["sl_atr"] - 2.0) ** 2,
                 lambda p: -(p["sl_atr"] - 2.0) ** 2,
                 dsr_fn=lambda p: 1.0,
                 clock=clock)
    print(f"✅ SelfImprover: streak={res['record']['streak']}, adopt={res['adopt']}")

    # 6) DriftMonitor
    dm = DriftMonitor()
    for _ in range(50):
        dm.update(0.05)
    for _ in range(50):
        dm.update(0.5)
    print(f"✅ DriftMonitor: {dm.status()}")

    # 7) ContinualLearningLoop
    cl = ContinualLearningLoop(n_feat=10, n_classes=3)
    for i in range(100):
        cl.observe(rng.standard_normal(10), y=i % 3, error=0.1 + i * 0.001)
    loss = cl.update_model(16)
    print(f"✅ CLL: {cl.status()}")
    assert loss is not None

    print("\n✅ Adaptive testi geçti")
