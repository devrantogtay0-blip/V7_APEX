# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ANA UYGULAMA (TAM SÜRÜM)
Streamlit UI · Paper Trade · Backtest · Model Eğitimi · Kaydet/Yükle

Çalıştırma: streamlit run v7_apex.py
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import streamlit as st

from v7_config import (
    AKTIF_SEKTOR, BOT_NAME, BOT_SLOGAN, BOT_VERSION, BOT_SERMAYE,
    FEATURE_NAMES, MACRO_KEYS, SEKTORLER, STOCK_LIST, WARMUP_BARS,
    STATE_FILE, MODEL_FILE, CACHE_DIR, get_risk_params,
)

# UI modülü
from v7_ui import (
    PLOTLY_OK, attribution_table, decisions_table, drawdown_chart,
    equity_chart, importance_chart, moe_chart, price_chart,
    sector_heatmap, stress_table, trades_table, uncertainty_chart,
)

# Veri + özellik
from v7_data import YFinanceFetcher, align_panel, clear_cache
from v7_numba_tech import compute_tech_panel
from v7_features import build_feature_store

# Backtest / risk
from v7_backtest import BacktestV, basic_metrics, score_params, make_simple_signals

# Paper Trade
from v7_paper_trade import PaperTradeEngine, PaperTradeBot

# Screener
from v7_screener import (
    Screener, scan_oversold, scan_momentum, scan_breakout,
    scan_reversal, scan_volume_surge, scan_low_vol,
)

# Alerts
from v7_alerts import get_alert_manager

# Notifications
from v7_notifications import get_notifier

# Deploy
from v7_deploy import save_state as save_bot_state, load_state as load_bot_state

# Monitoring
from v7_monitoring import get_metrics, HealthChecker, PerformanceMonitor, system_summary

# Stress
from v7_stress import full_stress_report

# Attribution
from v7_attribution import full_attribution_report

# XAI
from v7_xai import correlation_importance, permutation_importance, local_surrogate, top_features

# Adaptive
from v7_adaptive import Tuner, SelfImprover, DriftMonitor

# Model (torch opsiyonel)
from v7_models import TORCH_OK, DEVICE

# RL
from v7_rl import RLAgent, SB3_OK

# NLP
from v7_nlp import NLPEngine

# Causal
from v7_causal import causal_window_features

log = logging.getLogger("v7.apex")

# ═══════════════════════════════════════════════════════════════════
# YARDIMCI: RSI / MACD
# ═══════════════════════════════════════════════════════════════════

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/MACD/BB/SMA ekler."""
    out = df.copy()
    c = out["Close"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - 100 / (1 + rs)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIG"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIG"]
    sma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["BB_UP"] = sma20 + 2 * sd20
    out["BB_LO"] = sma20 - 2 * sd20
    out["BB_MID"] = sma20
    for p in (50, 200):
        out[f"SMA{p}"] = c.rolling(p).mean()
    return out


def simple_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """RSI + MACD + trend bazlı sinyal."""
    if df.empty or len(df) < 30:
        return {"sinyal": "VERİ YOK", "puan": 0, "sebep": "-", "p_up": 0.33, "p_down": 0.33}
    last = df.iloc[-1]
    puan = 0
    sebepler = []

    rsi_v = float(last.get("RSI", 50) or 50)
    if rsi_v < 30:
        puan += 2; sebepler.append(f"RSI {rsi_v:.0f}")
    elif rsi_v < 45:
        puan += 1
    elif rsi_v > 70:
        puan -= 2; sebepler.append(f"RSI {rsi_v:.0f}")
    elif rsi_v > 55:
        puan -= 1

    macd_h = float(last.get("MACD_HIST", 0) or 0)
    if macd_h > 0:
        puan += 1; sebepler.append("MACD+")
    elif macd_h < 0:
        puan -= 1; sebepler.append("MACD-")

    price = float(last["Close"])
    sma200 = last.get("SMA200")
    if sma200 is not None and not pd.isna(sma200):
        if price > float(sma200):
            puan += 1; sebepler.append("SMA200↑")
        else:
            puan -= 1; sebepler.append("SMA200↓")

    if puan >= 3:
        sinyal = "GÜÇLÜ AL"
    elif puan >= 1:
        sinyal = "AL"
    elif puan <= -3:
        sinyal = "GÜÇLÜ SAT"
    elif puan <= -1:
        sinyal = "SAT"
    else:
        sinyal = "TUT"

    p_up = min(0.85, 0.33 + puan * 0.08)
    p_down = min(0.85, 0.33 - puan * 0.05)
    return {
        "sinyal": sinyal, "puan": puan,
        "sebep": " · ".join(sebepler) or "-",
        "p_up": max(0.05, p_up), "p_down": max(0.05, p_down),
    }


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """Session state ilklendirme."""
    defaults = {
        "pt_engine": None,           # PaperTradeEngine
        "pt_bot": None,              # PaperTradeBot
        "pt_active": False,          # Paper trade aktif mi?
        "pt_last_t": 0,
        "last_save_ts": 0.0,
        "tuner": Tuner(),
        "si": SelfImprover(),
        "drift": DriftMonitor(),
        "rl_agent": RLAgent("ppo"),
        "nlp_engine": NLPEngine(),
        "alert_mgr": get_alert_manager(),
        "notifier": get_notifier(),
        "perf": PerformanceMonitor(),
        "log_lines": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_paper_engine() -> PaperTradeEngine:
    """Singleton paper trade engine."""
    if st.session_state.pt_engine is None:
        st.session_state.pt_engine = PaperTradeEngine(BOT_SERMAYE, get_risk_params())
        st.session_state.pt_bot = PaperTradeBot(st.session_state.pt_engine)
    return st.session_state.pt_engine


def log_event(msg: str) -> None:
    """Log satırı ekle."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {msg}")
    if len(st.session_state.log_lines) > 500:
        st.session_state.log_lines = st.session_state.log_lines[-250:]


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(period: str = "2y") -> Dict[str, pd.DataFrame]:
    yf = YFinanceFetcher(cache_ttl=3600)
    return yf.fetch_stocks(STOCK_LIST, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_data(period: str = "2y") -> pd.DataFrame:
    yf = YFinanceFetcher(cache_ttl=3600)
    try:
        return yf.fetch_macro(period=period)
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Sidebar paneli. Kontroller."""
    out = {}
    with st.sidebar:
        st.markdown(f"## ⚙️ {BOT_NAME}")
        st.caption(f"{BOT_VERSION} · {AKTIF_SEKTOR}")
        st.divider()

        # Veri
        st.markdown("### 📊 Veri")
        out["period"] = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y"], index=2, key="sb_period")
        if st.button("🔄 Cache Temizle", use_container_width=True):
            clear_cache()
            load_stock_data.clear()
            load_macro_data.clear()
            log_event("Cache temizlendi")
            st.success("Temizlendi")
            st.rerun()

        st.divider()

        # Paper Trade
        st.markdown("### 💼 Paper Trade")
        engine = get_paper_engine()
        eq = engine.equity({c: float(df["Close"].iloc[-1]) for c, df in data.items()})
        st.metric("Özkaynak", f"₺{eq:,.0f}", f"{eq - engine.initial:+,.0f}")
        st.caption(f"Pozisyon: {len(engine.positions)} · Emir: {len(engine.orders)}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Döngü", use_container_width=True, key="btn_cycle"):
                out["do_cycle"] = True
        with c2:
            if st.button("▶️ Başlat", use_container_width=True, key="btn_start"):
                st.session_state.pt_active = True
                log_event("Paper trade başlatıldı")
                st.rerun()

        if st.session_state.pt_active:
            if st.button("⏸️ Durdur", use_container_width=True, key="btn_stop"):
                st.session_state.pt_active = False
                log_event("Paper trade durduruldu")
                st.rerun()
            st.caption("🟢 Aktif")

        st.divider()

        # Kaydet/Yükle
        st.markdown("### 💾 Durum")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Kaydet", use_container_width=True, key="btn_save"):
                out["do_save"] = True
        with c2:
            if st.button("📂 Yükle", use_container_width=True, key="btn_load"):
                out["do_load"] = True

        if st.button("🔄 Sıfırla", use_container_width=True, key="btn_reset"):
            st.session_state.pt_engine = None
            st.session_state.pt_bot = None
            st.session_state.pt_active = False
            log_event("Portföy sıfırlandı")
            st.rerun()

        st.divider()

        # Backtest
        st.markdown("### 📊 Backtest")
        if st.button("🧪 Backtest Çalıştır", use_container_width=True, key="btn_bt"):
            out["do_backtest"] = True

        st.divider()

        # Model
        st.markdown("### 🧠 Model")
        st.caption(f"Torch: {'✅' if TORCH_OK else '❌'}")
        if TORCH_OK:
            if st.button("🚀 Model Eğit", use_container_width=True, key="btn_train"):
                out["do_train"] = True
        else:
            st.caption("⚠️ PyTorch yok")

        st.divider()
        st.caption("⚠️ Yatırım tavsiyesi değildir.")

    return out


# ═══════════════════════════════════════════════════════════════════
# SEKME: Canlı
# ═══════════════════════════════════════════════════════════════════

def tab_live(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📈 Fiyat Grafiği")
    if not data:
        st.warning("Veri yok.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        stock = st.selectbox("Hisse", list(data.keys()), key="live_stock")
    with c2:
        n_bars = st.slider("Bar", 60, 300, 180, 20, key="live_bars")
    with c3:
        show_bb = st.checkbox("Bollinger", value=True, key="live_bb")

    df = enrich_dataframe(data[stock])
    fig = price_chart(df, stock, n_bars, show_bb=show_bb, show_sma=True)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SEKME: Karar
# ═══════════════════════════════════════════════════════════════════

def tab_decisions(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🎯 Karar Tablosu")
    if not data:
        st.warning("Veri yok.")
        return
    decisions = []
    for code, df_raw in data.items():
        df = enrich_dataframe(df_raw)
        s = simple_signal(df)
        decisions.append({
            "stock": code, "sinyal": s["sinyal"], "puan": s["puan"],
            "p_up": s["p_up"], "p_down": s["p_down"], "sebep": s["sebep"],
        })
    st.dataframe(decisions_table(decisions), use_container_width=True)

    if PLOTLY_OK:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["stock"] for d in decisions],
            y=[d["p_up"] * 100 for d in decisions],
            name="P(AL) %", marker_color="#26a69a"))
        fig.add_trace(go.Bar(
            x=[d["stock"] for d in decisions],
            y=[d["p_down"] * 100 for d in decisions],
            name="P(SAT) %", marker_color="#ef5350"))
        fig.update_layout(template="plotly_dark", barmode="group", height=400,
                          title="Sinyal Olasılıkları")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SEKME: Portföy
# ═══════════════════════════════════════════════════════════════════

def tab_portfolio(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("💼 Sanal Portföy")
    engine = get_paper_engine()
    prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}

    summ = engine.summary(prices)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Özkaynak", f"₺{summ['equity']:,.0f}",
              f"{summ['total_return_pct']:+.2f}%")
    c2.metric("Nakit", f"₺{summ['cash']:,.0f}")
    c3.metric("Realize PnL", f"₺{summ['realized_pnl']:+,.0f}")
    c4.metric("Maruziyet", f"%{summ['exposure']*100:.1f}")

    st.divider()

    # Pozisyonlar
    pos_df = engine.positions_df(prices)
    if not pos_df.empty:
        st.subheader("Açık Pozisyonlar")
        st.dataframe(pos_df.round(2), use_container_width=True)
    else:
        st.info("Açık pozisyon yok. '⚡ Döngü' veya '▶️ Başlat' bas.")

    # Emirler
    orders_df = engine.orders_df()
    if not orders_df.empty:
        st.subheader("Emir Geçmişi")
        st.dataframe(orders_df.tail(30), use_container_width=True)

    # Equity curve
    if engine.equity_hist:
        fig = equity_chart(engine.equity_hist, "Özkaynak Eğrisi")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # Günlük rapor
    rep = engine.daily_report()
    if rep:
        st.subheader("Günlük Rapor")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", f"{rep['sharpe']:.2f}")
        c2.metric("Gün", rep["n_days"])
        c3.metric("Emir", rep["n_orders"])


# ═══════════════════════════════════════════════════════════════════
# SEKME: Tahmin
# ═══════════════════════════════════════════════════════════════════

def tab_forecast(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔮 Tahmin")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="fc_stock")
    df = enrich_dataframe(data[stock])
    s = simple_signal(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fiyat", f"₺{df['Close'].iloc[-1]:.2f}")
    c2.metric("Sinyal", s["sinyal"])
    c3.metric("P(AL)", f"%{s['p_up']*100:.1f}")
    c4.metric("P(SAT)", f"%{s['p_down']*100:.1f}")
    st.caption(f"Sebep: {s['sebep']}")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Beyin
# ═══════════════════════════════════════════════════════════════════

def tab_brain() -> None:
    st.subheader("🧠 Beyin")
    c1, c2, c3 = st.columns(3)
    c1.metric("Torch", "✅" if TORCH_OK else "❌")
    c2.metric("Device", DEVICE if TORCH_OK else "CPU")
    c3.metric("Feature", len(FEATURE_NAMES))
    st.markdown("""
    **Mimari:**
    - GATv2 (Graph Attention v2) — hisseler arası
    - TFT tarzı Transformer — zaman serisi
    - MoE 16 expert
    - Bagged Ensemble (5 model)
    """)


# ═══════════════════════════════════════════════════════════════════
# SEKME: Öz-Gelişim
# ═══════════════════════════════════════════════════════════════════

def tab_selfimprove() -> None:
    st.subheader("🧬 Öz-Gelişim")
    si = st.session_state.si
    if si.history:
        st.dataframe(pd.DataFrame(si.history), use_container_width=True)
    else:
        st.info("Henüz çalışmadı.")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Doğrulama
# ═══════════════════════════════════════════════════════════════════

def tab_verification(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔬 Doğrulama")
    from v7_verification import deflated_sharpe, probabilistic_sharpe
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.015, 250)
    c1, c2 = st.columns(2)
    c1.metric("PSR", f"{probabilistic_sharpe(r):.3f}")
    c2.metric("DSR", f"{deflated_sharpe(r, 20):.3f}")
    st.caption("Örnek metrikler. Gerçek için backtest gerekir.")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Analiz
# ═══════════════════════════════════════════════════════════════════

def tab_analysis(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📊 Analiz")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="an_stock")
    ret = data[stock]["Close"].pct_change().dropna()
    m = basic_metrics(ret.values)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"%{m['cagr']*100:.2f}")
    c2.metric("Sharpe", f"{m['sharpe']:.2f}")
    c3.metric("Max DD", f"%{m['max_dd']*100:.2f}")
    c4.metric("Vol", f"%{m['vol']*100:.2f}")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Log
# ═══════════════════════════════════════════════════════════════════

def tab_log() -> None:
    st.subheader("💬 Log")
    if st.button("🗑️ Temizle", key="btn_log_clear"):
        st.session_state.log_lines = []
        st.rerun()
    for line in reversed(st.session_state.log_lines[-200:]):
        st.text(line)


# ═══════════════════════════════════════════════════════════════════
# SEKME: RL
# ═══════════════════════════════════════════════════════════════════

def tab_rl() -> None:
    st.subheader("🤖 RL")
    st.caption(f"SB3: {'✅' if SB3_OK else '❌'}")
    rl = st.session_state.rl_agent
    st.caption(f"Model: {'✅' if rl.model is not None else '❌'}")
    if SB3_OK and st.button("PPO Eğit (10k adım)", key="btn_rl"):
        st.info("RL eğitimi ağır — sonra ekleriz.")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Stres
# ═══════════════════════════════════════════════════════════════════

def tab_stress(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔥 Stres Testi")
    engine = get_paper_engine()
    prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}
    eq = engine.equity(prices)
    expo = engine.exposure(prices)
    rets = np.array([df["Close"].pct_change().dropna().values for df in data.values()]).flatten()
    rep = full_stress_report(eq, expo, rets)
    if "historical" in rep:
        st.dataframe(rep["historical"].round(2), use_container_width=True)
    if "monte_carlo" in rep:
        mc = rep["monte_carlo"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Medyan", f"₺{mc['median']:,.0f}")
        c2.metric("VaR95", f"₺{mc['var_95']:,.0f}")
        c3.metric("Kayıp Olasılığı", f"%{mc['prob_loss']*100:.1f}")


# ═══════════════════════════════════════════════════════════════════
# SEKME: Atıf
# ═══════════════════════════════════════════════════════════════════

def tab_attribution(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📉 Atıf")
    engine = get_paper_engine()
    if not engine.orders:
        st.info("Henüz emir yok.")
        return
    trades = []
    for o in engine.orders:
        if o.side == "SELL":
            trades.append({"t": 0, "stock": o.stock, "qty": o.qty,
                           "entry": o.fill_price, "exit": o.fill_price,
                           "pnl": 0.0, "reason": o.reason})
    rep = full_attribution_report(trades)
    if "by_stock" in rep and not rep["by_stock"].empty:
        st.dataframe(rep["by_stock"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SEKME: Causal
# ═══════════════════════════════════════════════════════════════════

def tab_causal(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔗 Causal")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="cau_stock")
    df = data[stock]
    if len(df) < 100:
        st.warning("Yeterli veri yok.")
        return
    ret = df["Close"].pct_change().dropna().values[-250:]
    lag1 = np.roll(ret, 1)[-250:]
    lag2 = np.roll(ret, 2)[-250:]
    lags = np.column_stack([lag1, lag2])
    try:
        f = causal_window_features(ret, lags)
        st.json({k: round(v, 4) for k, v in f.items()})
    except Exception as e:
        st.error(f"Hata: {e}")


# ═══════════════════════════════════════════════════════════════════
# SEKME: XAI
# ═══════════════════════════════════════════════════════════════════

def tab_xai(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔍 XAI")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="xai_stock")
    df = enrich_dataframe(data[stock])
    feats = ["RSI", "MACD", "MACD_HIST", "SMA50", "SMA200"]
    X = df[feats].dropna().values
    y = df["Close"].pct_change().dropna().values[-len(X):]
    if len(X) < 30:
        st.warning("Veri yetersiz.")
        return
    df_imp = correlation_importance(X, y)
    st.dataframe(df_imp.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SEKME: Meta
# ═══════════════════════════════════════════════════════════════════

def tab_meta(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("⚙️ Sistem Durumu")
    from v7_numba_tech import NUMBA_OK as NB_TECH
    from v7_numba_micro import NUMBA_OK as NB_MICRO
    info = {
        "Sürüm": BOT_VERSION,
        "Sektör": AKTIF_SEKTOR,
        "Hisse": len(data),
        "Özellik": len(FEATURE_NAMES),
        "Numba Teknik": "✅" if NB_TECH else "❌",
        "Numba Mikro": "✅" if NB_MICRO else "❌",
        "Torch": "✅" if TORCH_OK else "❌",
        "Plotly": "✅" if PLOTLY_OK else "❌",
        "SB3": "✅" if SB3_OK else "❌",
    }
    for k, v in info.items():
        st.write(f"**{k}:** {v}")

    st.divider()
    st.subheader("Sistem Bilgisi")
    st.json(system_summary())


# ═══════════════════════════════════════════════════════════════════
# ANA
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=f"{BOT_NAME} {BOT_VERSION}",
        page_icon="🐋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Başlık
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
                padding: 18px 24px; border-radius: 12px; margin-bottom: 16px;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">{BOT_NAME}</h1>
        <p style="color: #b0bec5; margin: 6px 0 0 0; font-size: 0.95rem;">
            {BOT_VERSION} · {AKTIF_SEKTOR} · {len(STOCK_LIST)} hisse
        </p>
        <p style="color: #78909c; margin: 4px 0 0 0; font-size: 0.8rem; font-style: italic;">
            "{BOT_SLOGAN}"
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Veri yükle
    with st.spinner("📥 Veri yükleniyor..."):
        data = load_stock_data(st.session_state.get("sb_period", "2y"))
        macro = load_macro_data(st.session_state.get("sb_period", "2y"))

    if not data:
        st.error("❌ Veri yüklenemedi.")
        st.stop()

    st.success(f"✅ {len(data)} hisse yüklendi")

    # Sidebar
    actions = render_sidebar(data)

    # ─── AKSİYONLAR ─────────────────────────────────────────────
    prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}

    # Döngü
    if actions.get("do_cycle"):
        with st.spinner("⚡ Döngü çalışıyor..."):
            bot = st.session_state.pt_bot
            t = len(list(data.values())[0]) - 1
            res = bot.run_cycle(data, t, prices)
        log_event(f"Döngü: buys={res['n_buys']}, sells={res['n_sells']}, pending={res['pending']}")
        st.success(f"✅ Döngü tamamlandı: {res['n_buys']} alım, {res['n_sells']} satım")

    # Kaydet
    if actions.get("do_save"):
        try:
            engine = get_paper_engine()
            state = engine.get_state()
            import pickle, tempfile
            fd, tmp = tempfile.mkstemp(suffix=".pkl")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp, STATE_FILE)
            log_event(f"State kaydedildi: {STATE_FILE}")
            st.success(f"✅ Kaydedildi: {STATE_FILE}")
        except Exception as e:
            log_event(f"Kayıt hatası: {e}")
            st.error(f"❌ Kayıt hatası: {e}")

    # Yükle
    if actions.get("do_load"):
        try:
            if os.path.exists(STATE_FILE):
                import pickle
                with open(STATE_FILE, "rb") as f:
                    state = pickle.load(f)
                engine = get_paper_engine()
                engine.set_state(state)
                log_event("State yüklendi")
                st.success("✅ State yüklendi")
            else:
                st.warning("⚠️ Kayıtlı state yok")
        except Exception as e:
            st.error(f"❌ Yükleme hatası: {e}")

    # Backtest
    if actions.get("do_backtest"):
        with st.spinner("🧪 Backtest çalışıyor..."):
            try:
                panel = align_panel(data, list(data.keys()))
                store = build_feature_store(panel, macro if not macro.empty else None, scale=True)
                sig, size = make_simple_signals(store)
                rp = get_risk_params()
                bt = BacktestV(store, rp)
                res = bt.run(sig, size, WARMUP_BARS, store.train_end)
                st.success("✅ Backtest tamamlandı")
                st.json({k: round(v, 4) for k, v in res["metrics"].items()})
                log_event(f"Backtest: Sharpe={res['metrics']['sharpe']:.2f}")
            except Exception as e:
                st.error(f"❌ Backtest hatası: {e}")
                log_event(f"Backtest hatası: {e}")

    # Model Eğitimi
    if actions.get("do_train"):
        st.info("🚀 Model eğitimi ağır (1-3 saat). Şimdilik placeholder.")
        log_event("Model eğitimi istendi (placeholder)")

    # ─── SEKMELER ────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Canlı", "🎯 Karar", "💼 Portföy", "🔮 Tahmin", "🧠 Beyin",
        "🧬 Öz-Gelişim", "🔬 Doğrulama", "📊 Analiz", "💬 Log",
        "🤖 RL", "🔥 Stres", "📉 Atıf", "🔗 Causal", "🔍 XAI", "⚙️ Meta",
    ])

    with tabs[0]: tab_live(data)
    with tabs[1]: tab_decisions(data)
    with tabs[2]: tab_portfolio(data)
    with tabs[3]: tab_forecast(data)
    with tabs[4]: tab_brain()
    with tabs[5]: tab_selfimprove()
    with tabs[6]: tab_verification(data)
    with tabs[7]: tab_analysis(data)
    with tabs[8]: tab_log()
    with tabs[9]: tab_rl()
    with tabs[10]: tab_stress(data)
    with tabs[11]: tab_attribution(data)
    with tabs[12]: tab_causal(data)
    with tabs[13]: tab_xai(data)
    with tabs[14]: tab_meta(data)

    st.caption("⚠️ Eğitim amaçlıdır. Yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    if ST_OK if (ST_OK := True) else True:
        main()
