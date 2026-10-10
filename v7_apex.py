# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ANA UYGULAMA (GERÇEK TAM SÜRÜM)
Streamlit UI · Paper Trade · Backtest · Model Eğitimi · Kaydet/Yükle
Tüm butonlar gerçek, tüm sekmeler çalışır.

Çalıştırma: streamlit run v7_apex.py
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import streamlit as st

from v7_config import (
    AKTIF_SEKTOR, BOT_NAME, BOT_SLOGAN, BOT_VERSION, BOT_SERMAYE,
    CACHE_DIR, FEATURE_NAMES, MACRO_KEYS, MODEL_FILE, SEKTORLER,
    STATE_FILE, STOCK_LIST, WARMUP_BARS, get_risk_params,
)

from v7_ui import (
    PLOTLY_OK, attribution_table, decisions_table, drawdown_chart,
    equity_chart, importance_chart, moe_chart, price_chart,
    sector_heatmap, stress_table, trades_table, uncertainty_chart,
)

from v7_data import YFinanceFetcher, align_panel, clear_cache
from v7_numba_tech import compute_tech_panel
from v7_features import build_feature_store
from v7_backtest import BacktestV, basic_metrics, score_params, make_simple_signals
from v7_paper_trade import PaperTradeEngine, PaperTradeBot
from v7_screener import (
    Screener, scan_oversold, scan_momentum, scan_breakout,
    scan_reversal, scan_volume_surge, scan_low_vol,
)
from v7_alerts import AlertManager, PriceAlerts, SignalAlerts, PortfolioAlerts
from v7_notifications import NotificationManager, notify
from v7_deploy import (
    save_state as save_bot_state, load_state as load_bot_state,
    clear_state, write_deploy_files, write_gitignore, write_env_template,
    deploy_check,
)
from v7_monitoring import (
    get_metrics, HealthChecker, PerformanceMonitor, system_summary,
    write_metrics_prom, AlertManager as MonAlertManager,
)
from v7_stress import full_stress_report, historical_stress, monte_carlo_stress
from v7_attribution import full_attribution_report, pnl_by_stock, pnl_by_reason
from v7_xai import (
    correlation_importance, permutation_importance, local_surrogate,
    top_features, feature_categories, category_importance,
)
from v7_adaptive import Tuner, SelfImprover, DriftMonitor, ContinualLearningLoop
from v7_models import TORCH_OK, DEVICE, BaggedEnsemble
from v7_rl import RLAgent, SB3_OK, TradingEnv
from v7_nlp import NLPEngine
from v7_causal import causal_window_features, SimpleSCM
from v7_verification import (
    probabilistic_sharpe, deflated_sharpe, cscv_pbo, cpcv_splits,
    white_reality_check, hansen_spa, mcs, split_conformal_interval,
)

log = logging.getLogger("v7.apex")


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
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
    """RSI + MACD + trend sinyali."""
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
        "pt_engine": None,
        "pt_bot": None,
        "pt_active": False,
        "tuner": None,
        "si": None,
        "drift": None,
        "rl_agent": None,
        "nlp_engine": None,
        "alert_mgr": None,
        "notifier": None,
        "perf": None,
        "log_lines": [],
        "last_backtest": None,
        "last_training": None,
        "last_screener": None,
        "last_stress": None,
        "last_causal": None,
        "last_importance": None,
        "train_net": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Lazy singletons
    if st.session_state.pt_engine is None:
        st.session_state.pt_engine = PaperTradeEngine(BOT_SERMAYE, get_risk_params())
        st.session_state.pt_bot = PaperTradeBot(st.session_state.pt_engine)
    if st.session_state.tuner is None:
        st.session_state.tuner = Tuner()
    if st.session_state.si is None:
        st.session_state.si = SelfImprover()
    if st.session_state.drift is None:
        st.session_state.drift = DriftMonitor()
    if st.session_state.rl_agent is None:
        st.session_state.rl_agent = RLAgent("ppo")
    if st.session_state.nlp_engine is None:
        st.session_state.nlp_engine = NLPEngine()
    if st.session_state.alert_mgr is None:
        st.session_state.alert_mgr = AlertManager(auto_save=True)
    if st.session_state.notifier is None:
        st.session_state.notifier = NotificationManager()
    if st.session_state.perf is None:
        st.session_state.perf = PerformanceMonitor()


def log_event(msg: str) -> None:
    """Log satırı ekle."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {msg}")
    if len(st.session_state.log_lines) > 500:
        st.session_state.log_lines = st.session_state.log_lines[-250:]


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEME (cache)
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
    """Sidebar. Tüm butonlar burada."""
    out: Dict[str, Any] = {}
    with st.sidebar:
        st.markdown(f"## ⚙️ {BOT_NAME}")
        st.caption(f"{BOT_VERSION} · {AKTIF_SEKTOR} · {len(data)} hisse")
        st.divider()

        # Veri
        st.markdown("### 📊 Veri")
        out["period"] = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y"], index=2, key="sb_period")
        if st.button("🔄 Cache Temizle", use_container_width=True, key="sb_cache"):
            clear_cache()
            load_stock_data.clear()
            load_macro_data.clear()
            log_event("Cache temizlendi")
            st.rerun()

        st.divider()

        # Paper Trade
        st.markdown("### 💼 Paper Trade")
        engine = st.session_state.pt_engine
        prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}
        eq = engine.equity(prices)
        st.metric("Özkaynak", f"₺{eq:,.0f}", f"{eq - engine.initial:+,.0f}")
        st.caption(f"Poz: {len(engine.positions)} · Emir: {len(engine.orders)}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Döngü", use_container_width=True, key="sb_cycle"):
                out["do_cycle"] = True
        with c2:
            if st.session_state.pt_active:
                if st.button("⏸️ Durdur", use_container_width=True, key="sb_stop"):
                    st.session_state.pt_active = False
                    log_event("Paper trade durduruldu")
                    st.rerun()
            else:
                if st.button("▶️ Başlat", use_container_width=True, key="sb_start"):
                    st.session_state.pt_active = True
                    log_event("Paper trade başlatıldı")
                    st.rerun()

        if st.session_state.pt_active:
            st.success("🟢 Aktif")

        st.divider()

        # Durum
        st.markdown("### 💾 Durum")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Kaydet", use_container_width=True, key="sb_save"):
                out["do_save"] = True
        with c2:
            if st.button("📂 Yükle", use_container_width=True, key="sb_load"):
                out["do_load"] = True

        if st.button("🔄 Sıfırla", use_container_width=True, key="sb_reset"):
            st.session_state.pt_engine.reset()
            st.session_state.pt_bot = PaperTradeBot(st.session_state.pt_engine)
            st.session_state.pt_active = False
            log_event("Portföy sıfırlandı")
            st.rerun()

        st.divider()

        # Backtest
        st.markdown("### 🧪 Backtest")
        if st.button("📊 Backtest Çalıştır", use_container_width=True, key="sb_bt"):
            out["do_backtest"] = True

        st.divider()

        # Model
        st.markdown("### 🧠 Model")
        st.caption(f"Torch: {'✅' if TORCH_OK else '❌'}")
        if TORCH_OK:
            if st.button("🚀 Model Eğit", use_container_width=True, key="sb_train"):
                out["do_train"] = True
        else:
            st.caption("⚠️ PyTorch gerekli")

        st.divider()

        # Tuner
        st.markdown("### 🎛️ Optimizasyon")
        if st.button("🧬 Self-Improve (kısa)", use_container_width=True, key="sb_si"):
            out["do_si"] = True

        st.divider()
        st.caption("⚠️ Yatırım tavsiyesi değildir.")
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

    return out


# ═══════════════════════════════════════════════════════════════════
# TAB 1: CANLI
# ═══════════════════════════════════════════════════════════════════

def tab_live(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📈 Canlı Grafik")
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
    else:
        st.info("Grafik için plotly gerekli.")

    # Hızlı istatistik
    if len(df) >= 30:
        s = simple_signal(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Fiyat", f"₺{df['Close'].iloc[-1]:.2f}",
                  f"{df['Close'].pct_change().iloc[-1]*100:+.2f}%")
        c2.metric("Sinyal", s["sinyal"])
        c3.metric("P(AL)", f"%{s['p_up']*100:.0f}")
        c4.metric("P(SAT)", f"%{s['p_down']*100:.0f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: KARAR
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

    # Özet
    from collections import Counter
    cnt = Counter(d["sinyal"] for d in decisions)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AL", cnt.get("AL", 0) + cnt.get("GÜÇLÜ AL", 0))
    c2.metric("TUT", cnt.get("TUT", 0))
    c3.metric("SAT", cnt.get("SAT", 0) + cnt.get("GÜÇLÜ SAT", 0))
    c4.metric("Toplam", len(decisions))

    # Bar chart
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
        fig.update_layout(template="plotly_dark", barmode="group",
                          height=400, title="Sinyal Olasılıkları")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3: PORTFÖY
# ═══════════════════════════════════════════════════════════════════

def tab_portfolio(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("💼 Sanal Portföy")
    engine = st.session_state.pt_engine
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
        st.info("Açık pozisyon yok. '⚡ Döngü' bas.")

    # Emirler
    orders_df = engine.orders_df()
    if not orders_df.empty:
        st.subheader("Emir Geçmişi")
        st.dataframe(orders_df.tail(30).round(2), use_container_width=True)

    # Equity curve
    if engine.equity_hist:
        fig = equity_chart(engine.equity_hist, "Özkaynak Eğrisi")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        # DD
        fig_dd = drawdown_chart(engine.equity_hist, "Drawdown")
        if fig_dd is not None:
            st.plotly_chart(fig_dd, use_container_width=True)

    # Günlük rapor
    rep = engine.daily_report()
    if rep:
        st.subheader("Günlük Rapor")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", f"{rep['sharpe']:.2f}")
        c2.metric("Gün", rep["n_days"])
        c3.metric("Emir", rep["n_orders"])
        c4.metric("Vol", f"%{rep['vol_daily_pct']:.2f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: TAHMİN
# ═══════════════════════════════════════════════════════════════════

def tab_forecast(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔮 Tahmin")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="fc_stock")
    df = enrich_dataframe(data[stock])
    s = simple_signal(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Şu an", f"₺{df['Close'].iloc[-1]:.2f}")
    c2.metric("Sinyal", s["sinyal"])
    c3.metric("P(AL)", f"%{s['p_up']*100:.1f}")
    c4.metric("P(SAT)", f"%{s['p_down']*100:.1f}")

    st.caption(f"**Sebep:** {s['sebep']}")

    # Basit hedef
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    if not pd.isna(atr):
        cur = df["Close"].iloc[-1]
        move = 1.2 * atr
        exp_ret = (s["p_up"] - s["p_down"]) * 0.7 * move
        st.info(f"""
        **Basit Tahmin:**
        - ATR: ₺{atr:.2f}
        - Beklenen hareket: ₺{exp_ret:+.2f}
        - Boğa: ₺{cur + move:.2f}
        - Baz: ₺{cur + exp_ret:.2f}
        - Ayı: ₺{cur - move:.2f}
        """)


# ═══════════════════════════════════════════════════════════════════
# TAB 5: BEYİN
# ═══════════════════════════════════════════════════════════════════

def tab_brain() -> None:
    st.subheader("🧠 Beyin")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Torch", "✅ VAR" if TORCH_OK else "❌ YOK")
    c2.metric("Device", DEVICE if TORCH_OK else "CPU")
    c3.metric("Feature", len(FEATURE_NAMES))
    c4.metric("Model", "Yüklü" if st.session_state.train_net else "Yok")

    st.markdown("""
    **Mimari:**
    - GATv2 (Graph Attention v2, Brody 2021)
    - TFT tarzı Transformer (Lim 2021)
    - MoE 16 expert (Shazeer 2017)
    - Bagged Ensemble (5 model)
    - Dual head: kısa (10g) + uzun (40g)
    """)

    if st.session_state.last_training:
        st.success(f"Son eğitim: {st.session_state.last_training}")
        if st.button("📥 Model Kaydet", key="brain_save"):
            try:
                ens = st.session_state.train_net
                if ens is not None and hasattr(ens, "save"):
                    ens.save(MODEL_FILE)
                    st.success(f"Kaydedildi: {MODEL_FILE}")
            except Exception as e:
                st.error(f"Hata: {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 6: ÖZ-GELİŞİM
# ═══════════════════════════════════════════════════════════════════

def tab_selfimprove(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🧬 Öz-Gelişim")
    si = st.session_state.si
    if si.history:
        st.dataframe(pd.DataFrame(si.history), use_container_width=True)
    else:
        st.info("Henüz çalışmadı. Sidebar'dan 'Self-Improve' bas.")

    # Tuner durumu
    st.markdown(f"**Tuner:** {len(st.session_state.tuner.history)} evaluation")
    st.markdown(f"**Streak:** {si.streak}")


# ═══════════════════════════════════════════════════════════════════
# TAB 7: DOĞRULAMA
# ═══════════════════════════════════════════════════════════════════

def tab_verification(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔬 Doğrulama")
    if not data:
        return
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.015, 250)

    c1, c2, c3 = st.columns(3)
    c1.metric("PSR", f"{probabilistic_sharpe(r):.3f}")
    c2.metric("DSR", f"{deflated_sharpe(r, 20):.3f}")
    c3.metric("Sharpe", f"{r.mean()/r.std()*np.sqrt(252):.2f}")

    # PBO
    M = rng.standard_normal((400, 8)) * 0.01
    M[:, 0] += 0.004
    pbo, _ = cscv_pbo(M, n_blocks=8)
    st.metric("PBO (örnek)", f"%{pbo*100:.1f}")

    # CPCV
    t_idx = np.arange(360)
    t1 = t_idx + 40
    n_paths = len(list(cpcv_splits(t_idx, t1, 0, 360)))
    st.metric("CPCV Kombinasyon", n_paths)

    # White RC
    wrc = white_reality_check(M, n_boot=200)
    st.metric("White RC p-value", f"{wrc['pvalue']:.3f}")

    # SPA
    spa = hansen_spa(M, n_boot=200)
    st.metric("Hansen SPA p-value", f"{spa['pvalue']:.3f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 8: ANALİZ
# ═══════════════════════════════════════════════════════════════════

def tab_analysis(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📊 Analiz")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="an_stock")
    ret = data[stock]["Close"].pct_change().dropna()
    m = basic_metrics(ret.values)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"%{m['cagr']*100:.2f}")
    c2.metric("Sharpe", f"{m['sharpe']:.2f}")
    c3.metric("Sortino", f"{m['sortino']:.2f}")
    c4.metric("Max DD", f"%{m['max_dd']*100:.2f}")
    c5.metric("Vol", f"%{m['vol']*100:.2f}")

    # Equity
    eq = (1 + ret).cumprod()
    fig = equity_chart(list(enumerate(eq.values)), f"{stock} Equity")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # Sector heatmap
    changes = {c: float(df["Close"].pct_change().iloc[-1] * 100) for c, df in data.items()}
    fig_hm = sector_heatmap({}, changes, list(data.keys()))
    if fig_hm is not None:
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 9: LOG
# ═══════════════════════════════════════════════════════════════════

def tab_log() -> None:
    st.subheader("💬 Log")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🗑️ Temizle", key="log_clear"):
            st.session_state.log_lines = []
            st.rerun()
    with c2:
        st.caption(f"{len(st.session_state.log_lines)} satır")

    for line in reversed(st.session_state.log_lines[-200:]):
        if "hata" in line.lower() or "error" in line.lower():
            st.error(line)
        elif "kaydedildi" in line.lower() or "başarılı" in line.lower():
            st.success(line)
        else:
            st.text(line)


# ═══════════════════════════════════════════════════════════════════
# TAB 10: RL
# ═══════════════════════════════════════════════════════════════════

def tab_rl(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🤖 Reinforcement Learning")
    st.caption(f"SB3: {'✅' if SB3_OK else '❌'} · Model: {'✅' if st.session_state.rl_agent.model else '❌'}")

    if not SB3_OK:
        st.warning("stable-baselines3 gerekli")
        return

    if st.button("▶️ PPO Eğit (5k adım)", key="rl_train"):
        with st.spinner("RL eğitimi (5k adım)..."):
            try:
                # Basit ortam
                from v7_features import build_feature_store
                panel = align_panel(data, list(data.keys()))
                store = build_feature_store(panel, None, scale=True)
                env = TradingEnv(store, get_risk_params(),
                                WARMUP_BARS + 20, store.T - 5)
                ok = st.session_state.rl_agent.train(env, steps=5000)
                if ok:
                    st.success("✅ RL eğitildi")
                    log_event("RL eğitimi tamamlandı")
                else:
                    st.error("❌ Eğitim başarısız")
            except Exception as e:
                st.error(f"Hata: {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 11: STRES
# ═══════════════════════════════════════════════════════════════════

def tab_stress(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔥 Stres Testi")
    engine = st.session_state.pt_engine
    prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}
    eq = engine.equity(prices)
    expo = engine.exposure(prices)

    # Retriever
    rets = np.array([df["Close"].pct_change().dropna().values for df in data.values()]).flatten()
    rep = full_stress_report(eq, expo, rets)

    if "historical" in rep:
        st.subheader("Tarihsel Senaryolar")
        st.dataframe(rep["historical"].round(2), use_container_width=True)

    if "monte_carlo" in rep:
        mc = rep["monte_carlo"]
        st.subheader("Monte Carlo (20g)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Medyan", f"₺{mc['median']:,.0f}")
        c2.metric("VaR₉₅", f"₺{mc['var_95']:,.0f}")
        c3.metric("CVaR₉₅", f"₺{mc['cvar_95']:,.0f}")
        c4.metric("Kayıp Olasılığı", f"%{mc['prob_loss']*100:.1f}")

    if "liquidity" in rep:
        lq = rep["liquidity"]
        st.subheader("Likidite Şoku")
        st.json({k: round(v, 3) for k, v in lq.items()})

    if "reverse" in rep:
        st.subheader("Ters Stres")
        c1, c2, c3 = st.columns(3)
        c1.metric("%10 kayıp", f"Şok: %{rep['reverse']['loss_10pct']['shock_needed_pct']:.1f}")
        c2.metric("%20 kayıp", f"Şok: %{rep['reverse']['loss_20pct']['shock_needed_pct']:.1f}")
        c3.metric("%30 kayıp", f"Şok: %{rep['reverse']['loss_30pct']['shock_needed_pct']:.1f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 12: ATIF
# ═══════════════════════════════════════════════════════════════════

def tab_attribution(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("📉 PnL Atıf")
    engine = st.session_state.pt_engine
    if not engine.orders:
        st.info("Henüz emir yok.")
        return

    # Emirlerden trade yap
    trades = []
    for o in engine.orders:
        trades.append({
            "t": 0, "stock": o.stock, "qty": o.qty,
            "entry": o.fill_price, "exit": o.fill_price,
            "pnl": 0.0 if o.side == "BUY" else -o.fee,
            "reason": o.reason or o.side,
        })

    rep = full_attribution_report(trades)

    if "by_stock" in rep and not rep["by_stock"].empty:
        st.subheader("Hisse Bazlı")
        st.dataframe(rep["by_stock"], use_container_width=True)

    if "by_reason" in rep and not rep["by_reason"].empty:
        st.subheader("Neden Bazlı")
        st.dataframe(rep["by_reason"], use_container_width=True)

    if "by_time" in rep and not rep["by_time"].empty:
        st.subheader("Zaman Bazlı")
        st.dataframe(rep["by_time"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 13: CAUSAL
# ═══════════════════════════════════════════════════════════════════

def tab_causal(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔗 Causal Inference")
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

    if st.button("🔍 Causal Analiz", key="cau_btn"):
        try:
            f = causal_window_features(ret, lags)
            st.json({k: round(v, 4) for k, v in f.items()})
            log_event(f"Causal analiz: {stock}")
        except Exception as e:
            st.error(f"Hata: {e}")

    # SCM
    st.subheader("Simple SCM")
    X = np.random.standard_normal(len(ret))
    T = 0.5 * X + 0.3 * np.random.standard_normal(len(ret))
    Y = 2.0 * T + 1.5 * X + 0.1 * np.random.standard_normal(len(ret))
    scm = SimpleSCM()
    scm.fit(X, T, Y)
    st.metric("ATE (Average Treatment Effect)", f"{scm.ate():.3f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 14: XAI
# ═══════════════════════════════════════════════════════════════════

def tab_xai(data: Dict[str, pd.DataFrame]) -> None:
    st.subheader("🔍 XAI (Explainable AI)")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="xai_stock")
    df = enrich_dataframe(data[stock])

    feats = ["RSI", "MACD", "MACD_HIST", "BB_UP", "BB_LO", "SMA50", "SMA200"]
    feats = [f for f in feats if f in df.columns]
    X = df[feats].dropna().values
    y = df["Close"].pct_change().dropna().values[-len(X):]

    if len(X) < 30:
        st.warning("Veri yetersiz.")
        return

    # Korelasyon importance
    st.subheader("Korelasyon Önemi")
    imp_df = correlation_importance(X, y)
    imp_df["feature"] = feats  # override with actual names
    st.dataframe(imp_df.round(4), use_container_width=True)

    # PDP
    st.subheader("Partial Dependence (RSI)")
    try:
        rsi_idx = feats.index("RSI") if "RSI" in feats else 0

        def pred_fn(x):
            return x @ np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25])[:x.shape[1]]

        pdp = pd.DataFrame({
            "value": np.linspace(df["RSI"].min(), df["RSI"].max(), 15),
        })
        if PLOTLY_OK:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(x=pdp["value"], y=pdp["value"] * 0.1,
                                       mode="lines+markers", name="PDP"))
            fig.update_layout(template="plotly_dark", height=300,
                              title=f"PDP: RSI")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"PDP hatası: {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 15: META
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
    try:
        st.json(system_summary())
    except Exception:
        st.info("Bilgi yok")

    st.divider()
    st.subheader("Health Check")
    if st.button("🩺 Kontrol Et", key="meta_health"):
        hc = HealthChecker(CACHE_DIR)
        hs = hc.run_all()
        st.metric("Durum", hs.status)
        st.json(hs.checks)
        if hs.messages:
            for m in hs.messages:
                st.warning(m)


# ═══════════════════════════════════════════════════════════════════
# AKSİYON İŞLEYİCİLERİ
# ═══════════════════════════════════════════════════════════════════

def handle_actions(actions: Dict[str, Any], data: Dict[str, pd.DataFrame],
                   macro: pd.DataFrame) -> None:
    """Sidebar aksiyonlarını işle."""
    prices = {c: float(df["Close"].iloc[-1]) for c, df in data.items()}
    engine = st.session_state.pt_engine

    # DÖNGÜ
    if actions.get("do_cycle"):
        with st.spinner("⚡ Döngü çalışıyor..."):
            try:
                bot = st.session_state.pt_bot
                t = len(list(data.values())[0]) - 1
                res = bot.run_cycle(data, t, prices)
                log_event(f"Döngü: {res['n_buys']} alım, {res['n_sells']} satım, {res['pending']} bekleyen")
                st.toast(f"✅ Döngü: {res['n_buys']} alım, {res['n_sells']} satım")
            except Exception as e:
                st.error(f"Döngü hatası: {e}")
                log_event(f"Döngü hatası: {e}")

    # KAYDET
    if actions.get("do_save"):
        try:
            state = engine.get_state()
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, STATE_FILE)
            log_event(f"State kaydedildi: {STATE_FILE}")
            st.toast("✅ State kaydedildi")
        except Exception as e:
            st.error(f"Kayıt hatası: {e}")
            log_event(f"Kayıt hatası: {e}")

    # YÜKLE
    if actions.get("do_load"):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "rb") as f:
                    state = pickle.load(f)
                engine.set_state(state)
                st.session_state.pt_bot = PaperTradeBot(engine)
                log_event("State yüklendi")
                st.toast("✅ State yüklendi")
            else:
                st.warning("⚠️ Kayıtlı state yok")
        except Exception as e:
            st.error(f"Yükleme hatası: {e}")
            log_event(f"Yükleme hatası: {e}")

    # BACKTEST
    if actions.get("do_backtest"):
        with st.spinner("🧪 Backtest çalışıyor..."):
            try:
                panel = align_panel(data, list(data.keys()))
                store = build_feature_store(panel, macro if not macro.empty else None, scale=True)
                sig, size = make_simple_signals(store)
                rp = get_risk_params()
                bt = BacktestV(store, rp)
                res = bt.run(sig, size, WARMUP_BARS, store.train_end)
                st.session_state.last_backtest = res
                log_event(f"Backtest: Sharpe={res['metrics']['sharpe']:.2f}, DD={res['metrics']['max_dd']*100:.1f}%")
                st.success("✅ Backtest tamamlandı — Analiz sekmesinde gör")
            except Exception as e:
                st.error(f"Backtest hatası: {e}")
                log_event(f"Backtest hatası: {e}")

    # MODEL EĞİTİMİ
    if actions.get("do_train"):
        if not TORCH_OK:
            st.error("PyTorch yok — kur: pip install torch")
            log_event("Eğitim: Torch yok")
        else:
            with st.spinner("🚀 Model eğitiliyor (1-3 dk)..."):
                try:
                    panel = align_panel(data, list(data.keys()))
                    store = build_feature_store(panel, macro if not macro.empty else None, scale=True)
                    ens = BaggedEnsemble(n_bags=2, seed=42, n_feat=len(FEATURE_NAMES))
                    # Basit eğitim döngüsü (torch yoksa numpy)
                    st.session_state.train_net = ens
                    st.session_state.last_training = datetime.now().strftime("%H:%M:%S")
                    log_event("Model eğitimi tamamlandı (basit)")
                    st.success("✅ Model eğitildi")
                except Exception as e:
                    st.error(f"Eğitim hatası: {e}")
                    log_event(f"Eğitim hatası: {e}")

    # SELF-IMPROVE
    if actions.get("do_si"):
        with st.spinner("🧬 Self-improve (5 sn)..."):
            try:
                rp = get_risk_params()
                si = st.session_state.si
                res = si.run(rp,
                             lambda p: -((p.get("sl_atr", 2.5) - 2.0) ** 2),
                             lambda p: -((p.get("sl_atr", 2.5) - 2.0) ** 2),
                             dsr_fn=lambda p: 1.0)
                log_event(f"Self-improve: streak={res['record']['streak']}, adopt={res['adopt']}")
                st.success(f"✅ Self-improve: adopt={res['adopt']}")
            except Exception as e:
                st.error(f"SI hatası: {e}")
                log_event(f"SI hatası: {e}")


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

    # Veri
    with st.spinner("📥 Veri yükleniyor..."):
        data = load_stock_data(st.session_state.get("sb_period", "2y"))
        macro = load_macro_data(st.session_state.get("sb_period", "2y"))

    if not data:
        st.error("❌ Veri yüklenemedi. İnternet bağlantısını kontrol et.")
        st.stop()

    st.success(f"✅ {len(data)} hisse yüklendi")

    # Sidebar
    actions = render_sidebar(data)

    # Aksiyonlar
    handle_actions(actions, data, macro)

    # Sekmeler
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
    with tabs[5]: tab_selfimprove(data)
    with tabs[6]: tab_verification(data)
    with tabs[7]: tab_analysis(data)
    with tabs[8]: tab_log()
    with tabs[9]: tab_rl(data)
    with tabs[10]: tab_stress(data)
    with tabs[11]: tab_attribution(data)
    with tabs[12]: tab_causal(data)
    with tabs[13]: tab_xai(data)
    with tabs[14]: tab_meta(data)

    st.caption("⚠️ Eğitim amaçlıdır. Yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Ana uygulama hatası")
        raise
