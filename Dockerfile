# ═══════════════════════════════════════════════════════════
# 🐋 V7 APEX — Dockerfile
# Kurumsal Algo Trading Botu · Streamlit Image
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim

# ─── Meta ────────────────────────────────────────────────
LABEL maintainer="devrantogtay0-blip"
LABEL version="1.0"
LABEL description="🐋 V7 APEX — Kurumsal Algo Trading Botu"

# ─── Ortam ──────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

# ─── Çalışma dizini ──────────────────────────────────────
WORKDIR /app

# ─── Sistem bağımlılıkları ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── Python bağımlılıkları ───────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ─── Uygulama ────────────────────────────────────────────
COPY . .

# ─── Cache klasörü ───────────────────────────────────────
RUN mkdir -p cache_v7 .streamlit

# ─── Port ────────────────────────────────────────────────
EXPOSE 8501

# ─── Sağlık kontrolü ─────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ─── Başlatma ────────────────────────────────────────────
CMD ["streamlit", "run", "v7_apex.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
