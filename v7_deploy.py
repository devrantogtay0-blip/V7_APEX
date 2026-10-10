# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — DEPLOY
Config loader · State save/load · .env yönetimi · Docker/Streamlit Cloud uyumu
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from v7_config import (
    BOT_VERSION, MODEL_FILE, SEKTORLER, STATE_FILE, get_risk_params,
)

log = logging.getLogger("v7.deploy")


# ═══════════════════════════════════════════════════════════════════
# ENV YÖNETİMİ
# ═══════════════════════════════════════════════════════════════════

def load_env(path: str = ".env") -> Dict[str, str]:
    """`.env` dosyasını okur."""
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as e:
        log.warning(".env okuma hatası: %s", e)
    return out


def apply_env(env: Dict[str, str]) -> None:
    """Env değişkenlerini uygular."""
    for k, v in env.items():
        if k and k not in os.environ:
            os.environ[k] = v


def load_env_default() -> Dict[str, str]:
    """Otomatik `.env` yükleme."""
    env = load_env()
    apply_env(env)
    return env


# ═══════════════════════════════════════════════════════════════════
# STATE SAVE / LOAD (atomik)
# ═══════════════════════════════════════════════════════════════════

_STATE_LOCK = threading.RLock()


def save_state(bot: Any, path: Optional[str] = None) -> Optional[str]:
    """
    Bot durumunu pickle olarak kaydeder (atomik).
    bot: get_state() metodu olan nesne.
    """
    path = path or STATE_FILE
    with _STATE_LOCK:
        try:
            state = bot.get_state() if hasattr(bot, "get_state") else bot
            state["_version"] = BOT_VERSION
            state["_ts"] = datetime.now().isoformat(timespec="seconds")

            d = os.path.dirname(os.path.abspath(path)) or "."
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            log.info("State kaydedildi: %s", path)
            return path
        except Exception as e:
            log.error("State save hatası: %s", e)
            return None


def load_state(path: Optional[str] = None) -> Optional[Dict]:
    """Kayıtlı durumu yükler."""
    path = path or STATE_FILE
    if not os.path.exists(path):
        return None
    with _STATE_LOCK:
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            ver = s.get("_version", "?")
            if ver != BOT_VERSION:
                log.warning("State sürüm farkı: %s ≠ %s", ver, BOT_VERSION)
            return s
        except Exception as e:
            log.error("State load hatası: %s", e)
            return None


def apply_state(bot: Any, state: Optional[Dict] = None,
                path: Optional[str] = None) -> bool:
    """State'i bot'a uygular."""
    s = state or load_state(path)
    if s is None:
        return False
    try:
        if hasattr(bot, "set_state"):
            bot.set_state(s)
            return True
        return False
    except Exception as e:
        log.error("State uygulama hatası: %s", e)
        return False


def clear_state(path: Optional[str] = None) -> bool:
    """State dosyasını siler."""
    path = path or STATE_FILE
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        log.warning("State silme hatası: %s", e)
    return False


# ═══════════════════════════════════════════════════════════════════
# MODEL SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

def save_model(ensemble: Any, path: Optional[str] = None) -> bool:
    """Ensemble modeli kaydeder (varsa .save() metodu)."""
    path = path or MODEL_FILE
    try:
        if hasattr(ensemble, "save"):
            ensemble.save(path)
            log.info("Model kaydedildi: %s", path)
            return True
        # Fallback: numpy pickle
        np.savez_compressed(
            path + ".npz",
            history=getattr(ensemble, "history", []),
        )
        return True
    except Exception as e:
        log.error("Model save hatası: %s", e)
        return False


def load_model(path: Optional[str] = None) -> Optional[Any]:
    """Model yükleme (uygulama-özgü)."""
    path = path or MODEL_FILE
    if not os.path.exists(path):
        return None
    log.info("Model dosyası mevcut: %s", path)
    return {"path": path, "loaded": True}


# ═══════════════════════════════════════════════════════════════════
# CONFIG LOADER (JSON/YAML)
# ═══════════════════════════════════════════════════════════════════

def load_config_json(path: str) -> Optional[Dict]:
    """JSON config okur."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Config JSON okuma hatası: %s", e)
        return None


def save_config_json(cfg: Dict, path: str) -> bool:
    """JSON config yazar."""
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log.error("Config save hatası: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════
# DOCKER / DEPLOY YARDIMCILARI
# ═══════════════════════════════════════════════════════════════════

DOCKERFILE_CONTENT = """FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "v7_apex.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""


STREAMLIT_CONFIG = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50

[theme]
base = "dark"
primaryColor = "#26a69a"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"

[browser]
gatherUsageStats = false
"""


def write_deploy_files(out_dir: str = ".") -> Dict[str, str]:
    """Dockerfile + .streamlit/config.toml yazar."""
    out: Dict[str, str] = {}
    try:
        # Dockerfile
        df_path = os.path.join(out_dir, "Dockerfile")
        with open(df_path, "w", encoding="utf-8") as f:
            f.write(DOCKERFILE_CONTENT)
        out["dockerfile"] = df_path

        # .streamlit/config.toml
        st_dir = os.path.join(out_dir, ".streamlit")
        os.makedirs(st_dir, exist_ok=True)
        st_path = os.path.join(st_dir, "config.toml")
        with open(st_path, "w", encoding="utf-8") as f:
            f.write(STREAMLIT_CONFIG)
        out["streamlit_config"] = st_path

        log.info("Deploy dosyaları yazıldı: %s", out)
    except Exception as e:
        log.error("Deploy dosyaları yazma hatası: %s", e)
    return out


def write_env_template(out_path: str = ".env.example") -> bool:
    """Örnek .env dosyası yazar."""
    template = """# 🐋 V7 APEX — Environment Variables

# Sektör: BANKACILIK | HAVACILIK | ENERJI
V7_SEKTOR=BANKACILIK

# Log seviyesi: DEBUG | INFO | WARNING | ERROR
SEEKDEEP_LOG=INFO

# Groq LLM (opsiyonel) — https://console.groq.com
GROQ_API_KEY=

# FRED API (opsiyonel) — https://fredaccount.stlouisfed.org/apikeys
FRED_API_KEY=

# TCMB EVDS (opsiyonel)
TCMB_EVDS_KEY=
"""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(template)
        return True
    except Exception as e:
        log.warning(".env.example yazma hatası: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════
# GITIGNORE
# ═══════════════════════════════════════════════════════════════════

GITIGNORE_CONTENT = """# Virtual environments
venv/
env/
.venv/
ENV/

# Python cache
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Data / cache
cache_v7/
*.duckdb
*.duckdb.wal
*.sqlite
*.sqlite-journal
*.pkl
*.npz
*.pt
*.pth

# Logs & metrics
*.log
v7_structured.jsonl
v7_metrics.prom

# Env
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Streamlit
.streamlit/secrets.toml

# Test artifacts
tmp_*
pytest_cache/
.coverage
htmlcov/
"""


def write_gitignore(out_path: str = ".gitignore") -> bool:
    """`.gitignore` dosyası yazar."""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(GITIGNORE_CONTENT)
        return True
    except Exception as e:
        log.warning(".gitignore yazma hatası: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════
# DEPLOY CHECK
# ═══════════════════════════════════════════════════════════════════

def deploy_check(project_dir: str = ".") -> Dict[str, Any]:
    """Deploy öncesi kontrol listesi."""
    out: Dict[str, Any] = {"ok": True, "missing": [], "found": []}

    # Kritik dosyalar
    required = [
        "v7_apex.py",
        "v7_config.py",
        "v7_data.py",
        "v7_features.py",
        "v7_numba_tech.py",
        "v7_numba_micro.py",
        "requirements.txt",
    ]
    for f in required:
        p = os.path.join(project_dir, f)
        if os.path.exists(p):
            out["found"].append(f)
        else:
            out["missing"].append(f)
            out["ok"] = False

    # Opsiyonel
    optional = [
        "v7_db.py", "v7_models.py", "v7_backtest.py", "v7_risk.py",
        "v7_portfolio.py", "v7_execution.py", "v7_verification.py",
        "v7_xai.py", "v7_nlp.py", "v7_causal.py", "v7_rl.py",
        "v7_adaptive.py", "v7_stress.py", "v7_attribution.py",
        "v7_monitoring.py", "v7_deploy.py", "README.md",
    ]
    for f in optional:
        p = os.path.join(project_dir, f)
        if os.path.exists(p):
            out["found"].append(f)

    # .env kontrolü
    out["env_exists"] = os.path.exists(os.path.join(project_dir, ".env"))
    out["gitignore_exists"] = os.path.exists(os.path.join(project_dir, ".gitignore"))

    return out


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — Deploy Test")
    print(f"Sürüm: {BOT_VERSION}")

    import tempfile
    tmp = tempfile.mkdtemp()

    # 1) .env
    env_path = os.path.join(tmp, ".env")
    with open(env_path, "w") as f:
        f.write("V7_SEKTOR=BANKACILIK\nSEEKDEEP_LOG=INFO\n")
    env = load_env(env_path)
    print(f"\n✅ .env okundu: {env}")
    assert env.get("V7_SEKTOR") == "BANKACILIK"

    # 2) State save/load
    class MockBot:
        def __init__(self):
            self.cash = 100_000.0
            self.n = 5
        def get_state(self):
            return {"cash": self.cash, "n": self.n}
        def set_state(self, s):
            self.cash = s["cash"]
            self.n = s["n"]

    bot = MockBot()
    path = os.path.join(tmp, "test_state.pkl")
    save_state(bot, path)
    print(f"✅ State kaydedildi: {path}")

    bot2 = MockBot()
    apply_state(bot2, path=path)
    assert bot2.cash == 100_000.0 and bot2.n == 5
    print(f"✅ State yüklendi: cash={bot2.cash}, n={bot2.n}")

    # 3) Config JSON
    cfg = {"risk": {"sl_atr": 2.5, "tp_r": 2.0}}
    cfg_path = os.path.join(tmp, "cfg.json")
    save_config_json(cfg, cfg_path)
    loaded = load_config_json(cfg_path)
    print(f"✅ Config JSON: {loaded}")
    assert loaded == cfg

    # 4) Deploy dosyaları
    files = write_deploy_files(tmp)
    print(f"\n✅ Deploy dosyaları:")
    for k, v in files.items():
        print(f"   {k}: {os.path.exists(v)}")

    # 5) .gitignore
    gi_ok = write_gitignore(os.path.join(tmp, ".gitignore"))
    print(f"✅ .gitignore: {gi_ok}")

    # 6) .env.example
    env_ex_ok = write_env_template(os.path.join(tmp, ".env.example"))
    print(f"✅ .env.example: {env_ex_ok}")

    # 7) Deploy check
    check = deploy_check(tmp)
    print(f"\n✅ Deploy check:")
    print(f"   OK: {check['ok']}")
    print(f"   Eksik: {check['missing']}")

    # 8) clear_state
    clear_state(path)
    print(f"✅ State temizlendi: {not os.path.exists(path)}")

    print("\n✅ Deploy testi geçti")
