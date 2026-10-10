# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — ANA UYGULAMA (Streamlit)
Çalıştırma: streamlit run v7_apex.py
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from v7_config import (
    AKTIF_SEKTOR, BOT_NAME, BOT_SLOGAN, BOT_VERSION, BOT_SERMAYE,
    FEATURE_NAMES, MACRO_KEYS, SEKTORLER, STOCK_LIST, WARMUP_BARS,
    get_risk_params,
)

# Streamlit kontrol
try:
    import streamlit as st
    ST_OK = True
except ImportError:
    ST_OK = False
    st = None

# UI modülü
from v7_ui import (
    PLOTLY_OK, attribution_table, decisions_table, drawdown_chart,
    equity_chart, importance_chart, moe_chart, price_chart,
    sector_heatmap, stress_table, trades_table, uncertainty_chart,
)

# Veri katmanı
from v7_data import YFinanceFetcher, align_panel, clear_cache

# Göstergeler
from v7_numba_tech import compute_tech_panel

# Feature store
from v7_features import build_feature_store

# Backtest / risk
from v7_backtest import BacktestV, basic_metrics, score_params

# ML (torch opsiyonel)
from v7_models import TORCH_OK, DEVICE, BaggedEnsemble

log = logging.getLogger("v7.apex")


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI: RSI/MACD (gösterge kolonu için)
# ═══════════════════════════════════════════════════════════════════

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'e RSI/MACD/BB/SMA ekler."""
    out = df.copy()
    c = out["Close"]
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIG"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIG"]
    # Bollinger
    sma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["BB_UP"] = sma20 + 2 * sd20
    out["BB_LO"] = sma20 - 2 * sd20
    out["BB_MID"] = sma20
    # SMA
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
        puan += 2; sebepler.append(f"RSI {rsi_v:.0f} (aşırı satım)")
    elif rsi_v < 45:
        puan += 1
    elif rsi_v > 70:
        puan -= 2; sebepler.append(f"RSI {rsi_v:.0f} (aşırı alım)")
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

    # Basit olasılık tahmini
    p_up = min(0.85, 0.33 + puan * 0.08)
    p_down = min(0.85, 0.33 - puan * 0.05)
    p_up = max(0.05, p_up)
    p_down = max(0.05, p_down)

    return {
        "sinyal": sinyal, "puan": puan, "sebep": " · ".join(sebepler) or "-",
        "p_up": p_up, "p_down": p_down,
    }


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEME (cache)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(period: str = "2y") -> Dict[str, pd.DataFrame]:
    """24 hisse verisini çeker."""
    yf = YFinanceFetcher(cache_ttl=3600)
    return yf.fetch_stocks(STOCK_LIST, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_data(period: str = "2y") -> pd.DataFrame:
    """Makro veriyi çeker."""
    yf = YFinanceFetcher(cache_ttl=3600)
    try:
        return yf.fetch_macro(period=period)
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# SEKMELER
# ═══════════════════════════════════════════════════════════════════

def tab_live(data: Dict[str, pd.DataFrame]) -> None:
    """Canlı grafik."""
    st.subheader("📈 Fiyat Grafiği")
    if not data:
        st.warning("Veri yüklenmedi.")
        return
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stock = st.selectbox("Hisse", list(data.keys()), key="live_stock")
    with col2:
        n_bars = st.slider("Bar sayısı", 60, 300, 180, 20, key="live_bars")
    with col3:
        show_bb = st.checkbox("Bollinger", value=True, key="live_bb")

    df = enrich_dataframe(data[stock])
    fig = price_chart(df, stock, n_bars, show_bb=show_bb, show_sma=True)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Grafik için plotly gerekli.")


def tab_decisions(data: Dict[str, pd.DataFrame]) -> None:
    """Karar tablosu."""
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
    df_dec = decisions_table(decisions)
    st.dataframe(df_dec, use_container_width=True)

    # Bar chart
    if PLOTLY_OK:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["stock"] for d in decisions],
            y=[d["p_up"] * 100 for d in decisions],
            name="P(AL) %", marker_color="#26a69a",
        ))
        fig.add_trace(go.Bar(
            x=[d["stock"] for d in decisions],
            y=[d["p_down"] * 100 for d in decisions],
            name="P(SAT) %", marker_color="#ef5350",
        ))
        fig.update_layout(template="plotly_dark", barmode="group", height=400,
                          title="Sinyal Olasılıkları",
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def tab_portfolio() -> None:
    """Portföy (sanal)."""
    st.subheader("💼 Sanal Portföy")
    st.info("Portföy simülasyonu için Backtest sekmesini kullan.")
    st.metric("Başlangıç Sermaye", f"₺{BOT_SERMAYE:,.0f}")


def tab_forecast(data: Dict[str, pd.DataFrame]) -> None:
    """Tahmin özeti."""
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


def tab_brain() -> None:
    """Model durumu."""
    st.subheader("🧠 Beyin")
    c1, c2, c3 = st.columns(3)
    c1.metric("Torch", "VAR" if TORCH_OK else "YOK")
    c2.metric("Device", DEVICE if TORCH_OK else "CPU")
    c3.metric("Ensemble", "Hazır" if TORCH_OK else "Numpy")
    st.markdown("""
    **Mimari:**
    - GATv2 (Graph Attention v2) — hisseler arası ilişkiler
    - TFT tarzı Transformer — zaman serisi
    - MoE 16 expert — uzmanlaşma
    - Bagged Ensemble — belirsizlik
    """)


def tab_selfimprove() -> None:
    """Öz-gelişim."""
    st.subheader("🧬 Öz-Gelişim")
    st.info("Self-improve modülü Faz 2'de eklenecek.")


def tab_verification(data: Dict[str, pd.DataFrame]) -> None:
    """Doğrulama."""
    st.subheader("🔬 Doğrulama")
    st.info("CPCV, PBO, DSR testleri Backtest sekmesinde.")
    if data and len(data) > 5:
        from v7_verification import deflated_sharpe, probabilistic_sharpe
        # Sentetik getiri
        rng = np.random.default_rng(0)
        r = rng.normal(0.0003, 0.015, 250)
        st.metric("PSR (örnek)", f"{probabilistic_sharpe(r):.3f}")
        st.metric("DSR (örnek)", f"{deflated_sharpe(r, 20):.3f}")


def tab_analysis(data: Dict[str, pd.DataFrame]) -> None:
    """Analiz."""
    st.subheader("📊 Analiz")
    if not data:
        return
    stock = st.selectbox("Hisse", list(data.keys()), key="an_stock")
    df = data[stock]
    ret = df["Close"].pct_change().dropna()
    m = basic_metrics(ret.values)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"%{m['cagr']*100:.2f}")
    c2.metric("Sharpe", f"{m['sharpe']:.2f}")
    c3.metric("Max DD", f"%{m['max_dd']*100:.2f}")
    c4.metric("Vol", f"%{m['vol']*100:.2f}")


def tab_log(data: Dict[str, pd.DataFrame]) -> None:
    """Log."""
    st.subheader("💬 Log")
    st.caption(f"Son güncelleme: {time.strftime('%H:%M:%S')}")
    if data:
        for code, df in list(data.items())[:5]:
            st.write(f"- {code}: {len(df)} bar, son fiyat ₺{df['Close'].iloc[-1]:.2f}")


def tab_rl() -> None:
    """RL."""
    st.subheader("🤖 RL")
    st.info("PPO/SAC eğitimi için `v7_rl.py` modülü gerekli. Faz 3'te eklenecek.")


def tab_stress(data: Dict[str, pd.DataFrame]) -> None:
    """Stres testi."""
    st.subheader("🔥 Stres Testi")
    scenarios = [
        {"senaryo": "2008 benzeri", "şok %": -35, "etki": "Yüksek"},
        {"senaryo": "2020 Mart benzeri", "şok %": -30, "etki": "Yüksek"},
        {"senaryo": "TL devalüasyonu", "şok %": -15, "etki": "Orta"},
        {"senaryo": "Sert düzeltme", "şok %": -10, "etki": "Düşük"},
    ]
    st.dataframe(pd.DataFrame(scenarios), use_container_width=True)


def tab_attribution(data: Dict[str, pd.DataFrame]) -> None:
    """Atıf."""
    st.subheader("📉 Atıf")
    st.info("İşlem atıfı için portföy simülasyonu gerekli.")


def tab_causal(data: Dict[str, pd.DataFrame]) -> None:
    """Causal."""
    st.subheader("🔗 Causal")
    st.info("DoWhy + PC analizi Faz 4'te eklenecek.")


def tab_xai(data: Dict[str, pd.DataFrame]) -> None:
    """XAI."""
    st.subheader("🔍 XAI")
    st.info("SHAP/LIME analizi için model eğitimi gerekli.")


def tab_meta(data: Dict[str, pd.DataFrame]) -> None:
    """Meta."""
    st.subheader("⚙️ Sistem Durumu")
    from v7_models import TORCH_OK
    from v7_numba_tech import NUMBA_OK as NB_TECH
    from v7_numba_micro import NUMBA_OK as NB_MICRO
    from v7_data import YFinanceFetcher
    info = {
        "Sürüm": BOT_VERSION,
        "Sektör": AKTIF_SEKTOR,
        "Hisse sayısı": len(STOCK_LIST),
        "Özellik sayısı": len(FEATURE_NAMES),
        "Numba (teknik)": "✅" if NB_TECH else "❌",
        "Numba (mikro)": "✅" if NB_MICRO else "❌",
        "Torch": "✅" if TORCH_OK else "❌",
        "Plotly": "✅" if PLOTLY_OK else "❌",
    }
    for k, v in info.items():
        st.write(f"**{k}:** {v}")


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

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")
        period = st.selectbox("Veri periyodu", ["6mo", "1y", "2y", "5y"], index=2, key="period")
        if st.button("🔄 Cache Temizle", use_container_width=True):
            clear_cache()
            load_stock_data.clear()
            load_macro_data.clear()
            st.success("Cache temizlendi")
            st.rerun()
        st.divider()
        st.caption(f"**Sürüm:** {BOT_VERSION}")
        st.caption(f"**Sektör:** {AKTIF_SEKTOR}")
        st.caption("⚠️ Yatırım tavsiyesi değildir.")

    # Veri yükle
    with st.spinner("📥 Veri yükleniyor (YFinance)..."):
        data = load_stock_data(period)
        macro = load_macro_data(period)

    if not data:
        st.error("❌ Veri yüklenemedi. İnternet bağlantısını kontrol edin.")
        st.stop()

    st.success(f"✅ {len(data)} hisse yüklendi")

    # Sekmeler
    tabs = st.tabs([
        "📈 Canlı", "🎯 Karar", "💼 Portföy", "🔮 Tahmin", "🧠 Beyin",
        "🧬 Öz-Gelişim", "🔬 Doğrulama", "📊 Analiz", "💬 Log",
        "🤖 RL", "🔥 Stres", "📉 Atıf", "🔗 Causal", "🔍 XAI", "⚙️ Meta",
    ])

    with tabs[0]: tab_live(data)
    with tabs[1]: tab_decisions(data)
    with tabs[2]: tab_portfolio()
    with tabs[3]: tab_forecast(data)
    with tabs[4]: tab_brain()
    with tabs[5]: tab_selfimprove()
    with tabs[6]: tab_verification(data)
    with tabs[7]: tab_analysis(data)
    with tabs[8]: tab_log(data)
    with tabs[9]: tab_rl()
    with tabs[10]: tab_stress(data)
    with tabs[11]: tab_attribution(data)
    with tabs[12]: tab_causal(data)
    with tabs[13]: tab_xai(data)
    with tabs[14]: tab_meta(data)

    st.caption("⚠️ Eğitim amaçlıdır. Yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    if ST_OK:
        main()
    else:
        print(f"🐋 {BOT_NAME} {BOT_VERSION}")
        print("Streamlit yok. Çalıştır: streamlit run v7_apex.py")
        print("Test: SEEKDEEP_SELFTEST=1 python v7_apex.py")
