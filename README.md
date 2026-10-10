# 🐋 V7 APEX

**Kurumsal Algo Trading Botu** · BIST Hisse Analizi ve Otomatik Karar Sistemi

> "Zamanı oku, sabrı öğren, kazan."

![Version](https://img.shields.io/badge/version-1.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Özellikler

| Kategori | İçerik |
|----------|--------|
| **Özellik** | 306 (111 teknik + 40 mikro + 15 likidite + 25 peer + 20 rank + 30 makro + 15 NLP + 40 one-hot + 10 causal) |
| **Modeller** | GATv2 (Brody 2021) + TFT tarzı Transformer (Lim 2021) + MoE 16 (Shazeer 2017) |
| **Belirsizlik** | Aleatoric + Epistemic + MC Dropout + SWAG + Normalizing Flow + Conformal |
| **Risk** | VaR · CVaR · EVaR · Kelly · HRP · Mean-CVaR · GPD kuyruk |
| **Backtest** | CPCV · PBO · DSR · White RC · SPA · MCS · Conformal |
| **İcra** | Almgren-Chriss · Dinamik slippage · VWAP/TWAP · OFI gate |
| **UI** | 15 sekme (Streamlit) |
| **DB** | 35 tablo (DuckDB + SQLite fallback) |
| **Maliyet** | $0 (tamamen ücretsiz) |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- 2 GB RAM (minimum)
- İnternet bağlantısı (YFinance verisi için)

### 1) Klonla

git clone https://github.com/devrantogtay0-blip/V7_APEX.git

cd V7_APEX

### 2) Sanal Ortam

python -m venv venv

Linux/Mac: source venv/bin/activate

Windows: venv\Scripts\activate

### 3) Bağımlılıklar

pip install --upgrade pip

pip install -r requirements.txt

### Opsiyonel (derin öğrenme için):

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install transformers chronos-forecasting

pip install stable-baselines3 gymnasium

pip install groq

### 4) Env Ayarları

cp .env.example .env

---

## 🎮 Çalıştırma

### Streamlit Arayüzü

streamlit run v7_apex.py

Tarayıcıda http://localhost:8501 açılır.

### Testler

Tek modül test:

python v7_config.py

python v7_features.py

Tüm testler:

python v7_apex.py

### Docker

docker build -t v7-apex .

docker run -p 8501:8501 v7-apex

---

## 📂 Dosya Yapısı

V7_APEX/

├── v7_apex.py              # Ana uygulama (Streamlit)

├── v7_config.py            # Ayarlar, 306 özellik ismi

├── v7_db.py                # Veritabanı (35 tablo)

├── v7_data.py              # Veri çekici (YFinance)

├── v7_numba_tech.py        # 111 teknik gösterge (Numba JIT)

├── v7_numba_micro.py       # 40 mikro yapı

├── v7_features.py          # Feature Store (306 özellik)

├── v7_models.py            # GATv2 + TFT + MoE16

├── v7_backtest.py          # Numba simülatör

├── v7_risk.py              # VaR, CVaR, Kelly, HRP

├── v7_portfolio.py         # Pozisyon yönetimi

├── v7_execution.py         # Almgren-Chriss icra

├── v7_verification.py      # CPCV, PBO, DSR

├── v7_xai.py               # XAI (SHAP, LIME benzeri)

├── v7_ui.py                # Plotly grafikler

├── v7_nlp.py               # NLP (Groq + FinBERT)

├── v7_causal.py            # Causal inference

├── v7_rl.py                # RL (PPO/SAC/CQL)

├── v7_adaptive.py          # Self-improve, ADWIN

├── v7_stress.py            # Stres testi

├── v7_attribution.py       # PnL atıf

├── v7_monitoring.py        # Prometheus + Health

├── v7_deploy.py            # Deploy yardımcıları

├── requirements.txt        # Bağımlılıklar

├── Dockerfile              # Docker

├── .gitignore              # Git koruması

├── .env.example            # Env şablonu

└── README.md               # Bu dosya

---

## 🎯 Kullanım Örnekleri

### Python API

from v7_config import STOCK_LIST, get_risk_params

from v7_data import YFinanceFetcher, align_panel

from v7_features import build_feature_store

from v7_backtest import BacktestV

yf = YFinanceFetcher()

stocks = yf.fetch_stocks(STOCK_LIST, period="2y")

macro = yf.fetch_macro()

panel = align_panel(stocks, list(stocks.keys()))

store = build_feature_store(panel, macro)

print(f"Özellik matrisi: {store.feat.shape}")

rp = get_risk_params()

bt = BacktestV(store, rp)

### Streamlit

streamlit run v7_apex.py

### 15 Sekme:

1. 📈 Canlı — Fiyat grafiği

2. 🎯 Karar — Sinyal tablosu

3. 💼 Portföy — Pozisyonlar

4. 🔮 Tahmin — Forecast

5. 🧠 Beyin — Model durumu

6. 🧬 Öz-Gelişim — Self-improve

7. 🔬 Doğrulama — CPCV, PBO

8. 📊 Analiz — Sharpe, DD

9. 💬 Log — Kayıtlar

10. 🤖 RL — Reinforcement Learning

11. 🔥 Stres — Senaryolar

12. 📉 Atıf — PnL attribution

13. 🔗 Causal — Nedensellik

14. 🔍 XAI — Açıklanabilirlik

15. ⚙️ Meta — Sistem durumu

---

## 📊 Performans Hedefleri

| Metrik | Hedef |
|--------|-------|
| Yıllık Getiri | %20-40 |
| Sharpe | 1.5-2.5 |
| Max DD | %10-20 |
| Kazanma | %55-65 |
| Aylık İşlem | 20-50 |

NOT: Bu hedefler teoriktir, gerçek performans piyasa koşullarına bağlıdır.

---

## 🧪 Test

Her modülün kendi testi:

python v7_config.py

python v7_db.py

python v7_data.py

python v7_numba_tech.py

python v7_numba_micro.py

python v7_features.py

python v7_models.py

python v7_backtest.py

python v7_risk.py

python v7_portfolio.py

python v7_execution.py

python v7_verification.py

python v7_xai.py

python v7_ui.py

---

## 📖 Akademik Referanslar

GATv2: Brody et al. 2021

TFT: Lim et al. 2021

MoE: Shazeer 2017

Heteroscedastic: Kendall & Gal 2017

Almgren-Chriss: Almgren & Chriss 2005

Mean-CVaR: Rockafellar & Uryasev 2000

HRP: López de Prado 2016

CPCV: López de Prado 2018

PBO: Bailey et al. 2015

DSR: Bailey & López de Prado 2014

Triple Barrier: López de Prado 2018

Hawkes: Hawkes 1971

OFI: Cont et al. 2014

Conformal: Vovk 2005

---

## ⚠️ Yasal Uyarı

Bu yazılım yalnızca eğitim ve araştırma amaçlıdır.

Yatırım tavsiyesi DEĞİLDİR.

Geçmiş performans gelecek getiriyi garanti etmez.

Canlı para ile kullanmadan önce uzun süreli paper-trade yapın.

Kodu kullanmadan önce riskleri anladığınızdan emin olun.

---

## 📜 Lisans

MIT License. Detaylar için LICENSE dosyasına bakın.

---

## 🤝 Katkıda Bulunma

1. Fork edin

2. Branch açın: git checkout -b feature/yeniozellik

3. Commit edin: git commit -m 'Yeni özellik eklendi'

4. Push edin: git push origin feature/yeniozellik

5. Pull Request açın

---

## 📧 İletişim

GitHub: https://github.com/devrantogtay0-blip

---

🐋 V7 APEX — Kurumsal Algo Trading Botu

Zamanı oku, sabrı öğren, kazan.
