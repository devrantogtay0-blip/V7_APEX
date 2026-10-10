# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — REINFORCEMENT LEARNING
TradingEnv (numpy) · PPO/SAC (SB3, opsiyonel) · CQL (d3rlpy, opsiyonel) · RL veto
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v7_config import FEATURE_NAMES, FEE, SLIPPAGE

log = logging.getLogger("v7.rl")

# SB3 opsiyonel
try:
    import gymnasium as gym
    from gymnasium import spaces
    import stable_baselines3
    SB3_OK = True
except ImportError:
    gym = None
    spaces = None
    stable_baselines3 = None
    SB3_OK = False

# d3rlpy opsiyonel
try:
    import d3rlpy
    D3RLPY_OK = True
except ImportError:
    d3rlpy = None
    D3RLPY_OK = False


RL_FEATURES = ["rsi14", "ret5", "macd", "atrn14", "vol_rel"]


# ═══════════════════════════════════════════════════════════════════
# TRADING ENV (numpy — gym'siz)
# ═══════════════════════════════════════════════════════════════════

class TradingEnv:
    """
    Basit numpy ortamı.
    Gözlem: seçili özellikler (S·k) ‖ mevcut pozisyon (S)
    Aksiyon: [-1,1]^S → hedef pozisyon = a · max_pos
    Ödül: Σ w·r − maliyet·Σ|Δw|
    """

    def __init__(self, store, rp: Dict[str, Any], t_lo: int, t_hi: int,
                 feat_names: Optional[List[str]] = None,
                 cost: float = FEE + SLIPPAGE) -> None:
        self.store = store
        self.rp = rp
        self.t_lo = int(t_lo)
        self.t_hi = int(t_hi)
        self.cost = float(cost)
        self.feat_names = feat_names or RL_FEATURES
        self.fidx = [FEATURE_NAMES.index(n) for n in self.feat_names]
        self.S = store.C.shape[0]
        self.obs_dim = len(self.fidx) * self.S + self.S
        self.action_dim = self.S
        self.t = self.t_lo
        self.pos = np.zeros(self.S)
        self.reset()

    def reset(self) -> np.ndarray:
        """Başa döner."""
        self.t = self.t_lo
        self.pos = np.zeros(self.S)
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Gözlem vektörü."""
        f = self.store.feat[:, self.fidx, self.t]
        return np.concatenate([f.ravel(), self.pos]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Bir adım."""
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        target = a * float(self.rp.get("max_pos", 0.15)) * self.store.mask[:, self.t]
        C = self.store.C
        nxt = min(self.t + 1, self.t_hi - 1)
        ret = np.where(C[:, self.t] > 0,
                       C[:, nxt] / np.maximum(C[:, self.t], 1e-9) - 1.0, 0.0)
        turn = np.abs(target - self.pos)
        reward = float(np.sum(target * ret) - self.cost * np.sum(turn))
        self.pos = target
        self.t = nxt
        done = self.t >= self.t_hi - 1
        return self._obs(), reward, done, {"turnover": float(turn.sum())}


# ═══════════════════════════════════════════════════════════════════
# RL VETO
# ═══════════════════════════════════════════════════════════════════

def rl_veto(rl_action: float, rp: Dict[str, Any]) -> bool:
    """
    RL veto: RL güçlü satış (<-0.5) diyorsa ve veto açıksa alım iptal edilir.
    """
    return bool(rp.get("rl_veto", True) and float(rl_action) < -0.5)


# ═══════════════════════════════════════════════════════════════════
# PPO / SAC (SB3)
# ═══════════════════════════════════════════════════════════════════

class RLAgent:
    """PPO veya SAC sarmalayıcısı (SB3 opsiyonel)."""

    def __init__(self, algo: str = "ppo") -> None:
        self.algo = algo.lower()
        self.model = None
        self.available = SB3_OK

    @staticmethod
    def _gym_wrap(env: TradingEnv):
        """TradingEnv → gymnasium wrapper."""
        if not SB3_OK:
            return None

        class _G(gym.Env):
            def __init__(self):
                super().__init__()
                self.e = env
                self.observation_space = spaces.Box(
                    -np.inf, np.inf, (env.obs_dim,), np.float32)
                self.action_space = spaces.Box(
                    -1.0, 1.0, (env.action_dim,), np.float32)

            def reset(self, seed=None, options=None):
                return self.e.reset(), {}

            def step(self, a):
                o, r, d, info = self.e.step(a)
                return o, r, d, False, info

        return _G()

    def train(self, env: TradingEnv, steps: int = 50_000) -> bool:
        """Eğit. Başarılıysa True."""
        if not SB3_OK:
            log.warning("SB3 yok, eğitim atlandı")
            return False
        try:
            from stable_baselines3 import PPO, SAC
            genv = self._gym_wrap(env)
            cls = PPO if self.algo == "ppo" else SAC
            self.model = cls("MlpPolicy", genv, verbose=0)
            self.model.learn(total_timesteps=int(steps))
            log.info("RL (%s) %d adım eğitildi", self.algo.upper(), steps)
            return True
        except Exception as e:
            log.error("RL eğitim hatası: %s", e)
            return False

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Gözlem → aksiyon. Model yoksa boş dizi."""
        if self.model is None:
            return np.zeros(0)
        try:
            act, _ = self.model.predict(obs, deterministic=True)
            return np.asarray(act, dtype=np.float64).ravel()
        except Exception as e:
            log.debug("RL predict hatası: %s", e)
            return np.zeros(0)

    def save(self, path: str) -> None:
        """Kaydet."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> bool:
        """Yükle."""
        if not SB3_OK:
            return False
        try:
            from stable_baselines3 import PPO, SAC
            cls = PPO if self.algo == "ppo" else SAC
            self.model = cls.load(path)
            return True
        except Exception as e:
            log.warning("RL yükleme hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# OFFLINE RL — CQL (d3rlpy opsiyonel)
# ═══════════════════════════════════════════════════════════════════

def train_cql_offline(obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                      term: np.ndarray, steps: int = 10_000) -> Optional[Any]:
    """CQL (Kumar 2020) offline eğitim."""
    if not D3RLPY_OK:
        log.warning("d3rlpy yok, CQL atlandı")
        return None
    try:
        ds = d3rlpy.dataset.MDPDataset(
            observations=obs, actions=act, rewards=rew, terminals=term)
        algo = d3rlpy.algos.CQLConfig().create(device=False)
        algo.fit(ds, n_steps=int(steps),
                 n_steps_per_epoch=max(100, int(steps) // 10),
                 show_progress=False)
        return algo
    except Exception as e:
        log.error("CQL eğitim hatası: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════
# BASİT Q-LEARNING (numpy, fallback)
# ═══════════════════════════════════════════════════════════════════

class SimpleQLearning:
    """
    Basit tabular Q-learning (fallback).
    Durumlar: ayrıklaştırılmış gözlem (özet).
    Aksiyonlar: {-1, 0, 1} pozisyon değişimi (her hisse için).
    """

    def __init__(self, n_states: int = 20, n_actions: int = 3,
                 lr: float = 0.1, gamma: float = 0.95, eps: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.n_states = n_states
        self.n_actions = n_actions

    def _discretize(self, obs: np.ndarray) -> int:
        """Gözlemi tek state indeksine indir."""
        if len(obs) == 0:
            return 0
        # Basit quantile bazlı
        m = float(np.mean(obs))
        s = float(np.std(obs))
        if s < 1e-9:
            return 0
        z = m / (s + 1e-9)
        idx = int(np.clip((z + 3) / 6 * self.n_states, 0, self.n_states - 1))
        return idx

    def act(self, obs: np.ndarray) -> int:
        """ε-greedy aksiyon."""
        s = self._discretize(obs)
        if np.random.random() < self.eps:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        """Q-güncelleme."""
        s = self._discretize(obs)
        s2 = self._discretize(next_obs)
        target = reward if done else reward + self.gamma * np.max(self.Q[s2])
        self.Q[s, action] += self.lr * (target - self.Q[s, action])

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> bool:
        try:
            self.Q = np.load(path)
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — RL Test")
    print(f"SB3: {'VAR' if SB3_OK else 'YOK'}")
    print(f"d3rlpy: {'VAR' if D3RLPY_OK else 'YOK'}")

    # Sentetik store
    class MockStore:
        def __init__(self):
            rng = np.random.default_rng(7)
            S, T = 6, 300
            self.C = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (S, T)), axis=1))
            self.feat = rng.standard_normal((S, len(FEATURE_NAMES), T)).astype(np.float32)
            self.mask = np.ones((S, T), dtype=bool)

    store = MockStore()
    rp = {"max_pos": 0.15, "rl_veto": True}

    env = TradingEnv(store, rp, 60, 250)
    print(f"\n✅ Env: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    o = env.reset()
    print(f"✅ Reset gözlem: {o.shape}")

    # Adım
    o2, r, done, info = env.step(np.zeros(6))
    print(f"✅ Adım: reward={r:.4f}, done={done}")

    # Epizot
    steps = 1
    while not done:
        o2, r, done, info = env.step(rng.uniform(-1, 1, 6) if False else np.zeros(6))
        steps += 1
    print(f"✅ Epizot: {steps} adım")

    # RL veto
    assert rl_veto(-0.8, rp)
    assert not rl_veto(0.5, rp)
    assert not rl_veto(-0.8, {**rp, "rl_veto": False})
    print("✅ RL veto çalışıyor")

    # Q-learning
    ql = SimpleQLearning(n_states=20, n_actions=3)
    for ep in range(50):
        o = env.reset()
        done = False
        while not done:
            a = ql.act(o)
            o2, r, done, _ = env.step([a * 0.1] * 6)
            ql.update(o, a, r, o2, done)
            o = o2
    print(f"✅ Q-learning: Q max={ql.Q.max():.4f}")

    print("\n✅ RL testi geçti")
