# Changelog

All notable changes to V7 APEX will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

---

## [1.0.0] - 2025-01-XX

### Added

**Core Foundation**
- 306 özellik (111 teknik + 40 mikro + 15 likidite + 25 peer + 20 rank + 30 makro + 15 NLP + 40 one-hot + 10 causal)
- Numba JIT hızlandırılmış 111 teknik gösterge
- 40 mikro yapı proxy (OFI, Kyle Lambda, Hawkes, Iceberg, Spoof)
- Feature Store (RobustScaler, train-only fit)
- DuckDB (fallback SQLite) · 35 tablo
- YFinance veri çekici (24 hisse + 8 makro)

**Modeller**
- GATv2 (Brody 2021) — dinamik graph attention
- TFT tarzı Transformer (Lim 2021)
- MoE 16 expert (Shazeer 2017) — Load Balancing Loss
- Bagged Ensemble (5 model)

**Belirsizlik**
- Aleatoric (Heteroscedastic CE, Kendall & Gal 2017)
- Epistemic (Deep Ensemble std)
- MC Dropout
- SWAG (diagonal)
- Normalizing Flow (RealNVP)
- Conformal Prediction (Vovk 2005)

**Backtest & Doğrulama**
- Numba paralel simülatör (stop, TP, BE, trailing, max_hold)
- Triple-barrier etiketleme (López de Prado 2018)
- Benzersizlik ağırlıkları (López de Prado 2018)
- Purged CV + Embargo
- CPCV C(8,4)=70 kombinasyon
- PBO (CSCV, Bailey et al. 2015)
- DSR (Bailey & López de Prado 2014)
- White RC (White 2000)
- Hansen SPA (Hansen 2005)
- MCS (Hansen et al. 2011)
- Conformal (split + LAC)

**Risk & Portföy**
- VaR (hist, param, MC)
- CVaR (mean, spectral)
- EVaR (Ahmadi-Javid)
- GPD kuyruk şekli
- Sharpe, Sortino, Calmar, Ulcer
- Beta, Alpha, IR, Treynor
- Fractional Kelly (Thorp 2006 + unc cezası)
- HRP (López de Prado 2016)
- Risk Parity
- Mean-CVaR LP (Rockafellar & Uryasev 2000)

**İcra**
- Almgren-Chriss market impact (2005)
- Dinamik slippage
- VWAP/TWAP split
- Kısmi dolum (ADV limiti)
- OFI gate
- Implementation Shortfall (Perold 1988)

**NLP**
- Groq LLM (llama-3.3-70b)
- FinBERT (ProsusAI)
- Lexicon (Türkçe)
- KAP fetcher
- YFinance haber başlıkları

**Causal**
- PC skeleton (Spirtes 2000)
- Backdoor etki (OLS)
- Simple SCM
- Counterfactual

**RL**
- TradingEnv (numpy)
- PPO/SAC (Stable-Baselines3)
- CQL (d3rlpy)
- Simple Q-Learning (fallback)
- RL veto

**Adaptive**
- ADWIN drift detector
- Experience Replay (ring buffer)
- Online Learner (SGD)
- Shadow Mode
- Tuner (Random + ES)
- Self-Improver
- Continual Learning Loop

**XAI**
- Correlation importance
- Permutation importance
- LIME-benzeri lokal surrogate
- PDP (Partial Dependence)
- Kategori importance

**Stres & Atıf**
- 6 tarihsel senaryo
- Monte Carlo simülasyon
- Korelasyon şoku
- Likidite şoku
- Ters stres
- PnL atıf (hisse, neden, sektör, zaman)
- Brinson atıf (1985)
- Faktör atıf
- Drawdown atıf

**Monitoring**
- Prometheus metrics
- Health check
- Alert manager
- Structured logger (JSON)
- Performance monitor
- System summary

**Deploy**
- State save/load (atomik)
- Config JSON loader
- Docker deploy
- .env yönetimi
- .gitignore otomatik
- Deploy check

**UI**
- 15 sekme (Streamlit)
- Plotly grafikler (mum, equity, DD, uncertainty, MoE, XAI)
- Sidebar (periyot, cache, bilgi)
- Türkçe arayüz

**Test**
- Her modül için kendi öz-test
- Sentetik veri üreticileri
- Assert bazlı doğrulama

### Technical Details

- **Toplam dosya:** 36
- **Toplam satır:** ~18.000
- **Özellik:** 306
- **Model:** 8 (GATv2, TFT, MoE16, Chronos, Bagging, PPO, CQL, Ensemble)
- **Belirsizlik:** 6 boyut
- **Risk metriği:** 20+
- **Doğrulama testi:** 15+
- **DB tablo:** 35
- **UI sekme:** 15

### Dependencies

- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- numba >= 0.57
- duckdb >= 0.9
- yfinance >= 0.2
- streamlit >= 1.30
- plotly >= 5.18
- torch >= 2.0 (opsiyonel)
- transformers >= 4.35 (opsiyonel)
- chronos-forecasting >= 1.0 (opsiyonel)
- stable-baselines3 >= 2.2 (opsiyonel)
- groq >= 0.4 (opsiyonel)

### Notes

- Bu ilk sürümdür
- Sıfır bütçe ile geliştirilmiştir
- Sadece ücretsiz araçlar kullanılmıştır
- Canlı emir gönderimi YOKTUR
- Kâğıt işlem (paper trade) desteklenir
- Yatırım tavsiyesi DEĞİLDİR

---

## [Unreleased]

### Planned

- Gerçek broker entegrasyonu (Alpaca, IBKR, Matriks)
- L2/L3 emir defteri analizi
- Dakika seviyesi veri desteği
- Multi-asset (kripto, forex, emtia)
- Reinforcement learning iyileştirmeleri
- Webhook entegrasyonları
- Cloud deployment (AWS, GCP)
- Model serving (TensorRT, ONNX)
- A/B testing framework

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0.0 | 2025-01 | İlk sürüm (V7 APEX) |

---

## License

MIT License — Detaylar için LICENSE dosyasına bakın.
