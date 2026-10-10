# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — UI
Plotly grafikler + 15 sekme + Streamlit
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from v7_config import (
    FEATURE_NAMES, MACRO_KEYS, MICRO_NAMES, REGIMES, STOCK_LIST,
)

log = logging.getLogger("v7.ui")

# Opsiyonel import'lar
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    go = None
    make_subplots = None
    PLOTLY_OK = False

try:
    import streamlit as st
    STREAMLIT_OK = True
except ImportError:
    st = None
    STREAMLIT_OK = False

DARK = "plotly_dark"
GRID = "rgba(255,255,255,0.05)"


# ═══════════════════════════════════════════════════════════════════
# FİYAT GRAFİĞİ
# ═══════════════════════════════════════════════════════════════════

def price_chart(df: pd.DataFrame, title: str, n_bars: int = 180,
                show_bb: bool = True, show_sma: bool = True) -> Any:
    """3 panelli fiyat grafiği: mum + Bollinger + RSI + MACD."""
    if not PLOTLY_OK:
        return None

    d = df.tail(n_bars).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(f"{title} — Fiyat", "RSI (14)", "MACD"),
    )

    # Mum
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Fiyat",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Bollinger
    if show_bb and "BB_UP" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_UP"], mode="lines",
                                 name="BB üst", line=dict(color="rgba(150,150,150,0.5)", width=1)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_LO"], mode="lines",
                                 name="BB alt", line=dict(color="rgba(150,150,150,0.5)", width=1),
                                 fill="tonexty", fillcolor="rgba(150,150,150,0.08)"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_MID"], mode="lines",
                                 name="SMA20", line=dict(color="#29b6f6", width=1.2)),
                      row=1, col=1)

    # SMA
    if show_sma:
        for col, color in [("SMA50", "#ffa726"), ("SMA200", "#ef5350")]:
            if col in d.columns:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines",
                                         name=col, line=dict(color=color, width=1.2)),
                              row=1, col=1)

    # RSI
    if "RSI" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], mode="lines",
                                 name="RSI", line=dict(color="#ff7043", width=1.5)),
                      row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # MACD
    if "MACD" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], mode="lines",
                                 name="MACD", line=dict(color="#42a5f5", width=1.5)),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIG"], mode="lines",
                                 name="Sinyal", line=dict(color="#ffa726", width=1.5)),
                      row=3, col=1)
        if "MACD_HIST" in d.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in d["MACD_HIST"]]
            fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], name="Hist",
                                 marker_color=colors, opacity=0.6),
                          row=3, col=1)

    fig.update_layout(
        template=DARK, height=780,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


# ═══════════════════════════════════════════════════════════════════
# EQUITY GRAFİĞİ
# ═══════════════════════════════════════════════════════════════════

def equity_chart(equity_hist: List, title: str = "Özkaynak") -> Any:
    """Özkaynak eğrisi."""
    if not PLOTLY_OK or not equity_hist:
        return None
    xs = [t for t, _ in equity_hist]
    ys = [e for _, e in equity_hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Equity",
                             line=dict(color="#26a69a", width=2),
                             fill="tozeroy", fillcolor="rgba(38,166,154,0.1)"))
    fig.add_hline(y=ys[0] if ys else 0, line_dash="dash", line_color="gray")
    fig.update_layout(template=DARK, height=380, title=title,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


def equity_curve_chart(returns: pd.Series, title: str = "Kümülatif Getiri") -> Any:
    """Kümülatif getiri eğrisi."""
    if not PLOTLY_OK:
        return None
    eq = (1 + returns.fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="CumRet",
                             line=dict(color="#26a69a", width=2),
                             fill="tozeroy", fillcolor="rgba(38,166,154,0.1)"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(template=DARK, height=350, title=title,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# BELİRSİZLİK GRAFİĞİ
# ═══════════════════════════════════════════════════════════════════

def uncertainty_chart(predictions: Dict[str, np.ndarray], stock_names: List[str]) -> Any:
    """Hisse başına aleatorik / epistemik / toplam belirsizlik."""
    if not PLOTLY_OK:
        return None
    alea = np.asarray(predictions.get("alea_short", []))
    epi = np.asarray(predictions.get("epi_short", []))
    if len(alea) == 0 or len(epi) == 0:
        return None
    total = np.sqrt(alea ** 2 + epi ** 2)
    fig = go.Figure()
    fig.add_bar(x=stock_names[:len(alea)], y=alea, name="Aleatorik",
                marker_color="#42a5f5")
    fig.add_bar(x=stock_names[:len(epi)], y=epi, name="Epistemik",
                marker_color="#ffa726")
    fig.add_trace(go.Scatter(x=stock_names[:len(total)], y=total, mode="markers+lines",
                             name="Toplam", marker=dict(color="#ef5350", size=10)))
    fig.update_layout(template=DARK, barmode="group", height=400,
                      title="Belirsizlik Ayrıştırması",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# MOE / ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

def moe_chart(history: List[Dict[str, Any]]) -> Any:
    """MoE eğitim geçmişi."""
    if not PLOTLY_OK or not history:
        return None
    fig = go.Figure(go.Scatter(
        x=[h.get("bag", i) for i, h in enumerate(history)],
        y=[h.get("best_val", 0.0) for h in history],
        mode="lines+markers", name="Val Loss",
        line=dict(color="#ab47bc", width=2),
    ))
    fig.update_layout(template=DARK, height=300, title="MoE Eğitim",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# RİSK / DD GRAFİĞİ
# ═══════════════════════════════════════════════════════════════════

def drawdown_chart(equity: List, title: str = "Drawdown") -> Any:
    """Drawdown eğrisi."""
    if not PLOTLY_OK or not equity:
        return None
    ys = np.array([e for _, e in equity], dtype=np.float64)
    peak = np.maximum.accumulate(ys)
    dd = (ys / np.maximum(peak, 1e-9) - 1.0) * 100
    xs = [t for t, _ in equity]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=dd, mode="lines", name="DD %",
                             line=dict(color="#ef5350", width=2),
                             fill="tozeroy", fillcolor="rgba(239,83,80,0.15)"))
    fig.update_layout(template=DARK, height=300, title=title,
                      margin=dict(l=20, r=20, t=40, b=20),
                      yaxis_title="DD (%)")
    return fig


# ═══════════════════════════════════════════════════════════════════
# SEKTÖR ISIL HARİTA
# ═══════════════════════════════════════════════════════════════════

def sector_heatmap(prices: Dict[str, float], changes: Dict[str, float],
                   codes: List[str]) -> Any:
    """Değişim yüzdesine göre renkli bar."""
    if not PLOTLY_OK:
        return None
    vals = [changes.get(c, 0.0) for c in codes]
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in vals]
    fig = go.Figure(go.Bar(x=codes, y=vals, marker_color=colors))
    fig.update_layout(template=DARK, height=340,
                      title="Günlük Değişim (%)",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI TABLOLAR
# ═══════════════════════════════════════════════════════════════════

def decisions_table(decisions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Karar tablosu."""
    if not decisions:
        return pd.DataFrame(columns=["Hisse", "Sinyal", "P(AL)", "P(SAT)", "Puan"])
    rows = []
    for d in decisions:
        rows.append({
            "Hisse": d.get("stock", ""),
            "Sinyal": d.get("sinyal", "TUT"),
            "P(AL)": f"{d.get('p_up', 0) * 100:.1f}%",
            "P(SAT)": f"{d.get('p_down', 0) * 100:.1f}%",
            "Puan": d.get("puan", 0),
            "Sebep": d.get("sebep", ""),
        })
    return pd.DataFrame(rows)


def stress_table(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    """Stres testi tablosu."""
    return pd.DataFrame(scenarios)


def attribution_table(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """İşlem atıf tablosu."""
    if not trades:
        return pd.DataFrame(columns=["Hisse", "PnL", "İşlem"])
    df = pd.DataFrame(trades)
    g = df.groupby("stock").agg({"pnl": "sum", "qty": "count"}).reset_index()
    g.columns = ["Hisse", "PnL (TL)", "İşlem"]
    return g.sort_values("PnL (TL)", ascending=False).reset_index(drop=True)


def trades_table(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """İşlem geçmişi."""
    if not trades:
        return pd.DataFrame(columns=["Tarih", "Hisse", "Adet", "Giriş", "Çıkış", "PnL", "Neden"])
    df = pd.DataFrame(trades)
    return df.rename(columns={
        "t": "Tarih", "stock": "Hisse", "qty": "Adet",
        "entry": "Giriş", "exit": "Çıkış", "pnl": "PnL", "reason": "Neden",
    })[["Tarih", "Hisse", "Adet", "Giriş", "Çıkış", "PnL", "Neden"]]


# ═══════════════════════════════════════════════════════════════════
# XAI GRAFİĞİ
# ═══════════════════════════════════════════════════════════════════

def importance_chart(importance_df: pd.DataFrame, top_k: int = 15) -> Any:
    """Özellik önemi bar grafiği."""
    if not PLOTLY_OK or importance_df.empty:
        return None
    df = importance_df.head(top_k)
    col = "importance_mean" if "importance_mean" in df.columns else (
        "abs_corr" if "abs_corr" in df.columns else df.columns[-1]
    )
    fig = go.Figure(go.Bar(x=df[col], y=df["feature"], orientation="h",
                           marker_color="#ab47bc"))
    fig.update_layout(template=DARK, height=400, title=f"Top {top_k} Özellik",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🐋 V7 APEX — UI Test")
    print(f"Plotly: {'VAR' if PLOTLY_OK else 'YOK'}")
    print(f"Streamlit: {'VAR' if STREAMLIT_OK else 'YOK'}")

    if not PLOTLY_OK:
        print("⚠️ Plotly yok — sadece yardımcı tablolar test edilecek")

    # Sentetik DataFrame
    rng = np.random.default_rng(7)
    T = 250
    C = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, T)))
    df = pd.DataFrame({
        "Open": C * (1 + rng.normal(0, 0.002, T)),
        "High": C * (1 + np.abs(rng.normal(0, 0.005, T))),
        "Low": C * (1 - np.abs(rng.normal(0, 0.005, T))),
        "Close": C,
        "Volume": rng.uniform(1e5, 1e6, T),
    }, index=pd.date_range("2023-01-02", periods=T))

    # RSI/MACD ekle
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["MACD_SIG"] = df["MACD"].ewm(span=9).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    if PLOTLY_OK:
        fig = price_chart(df, "GARAN")
        print(f"\n✅ Fiyat grafiği: {'OK' if fig is not None else 'None'}")

        eq_hist = [(i, 100000 + i * 50 + rng.normal(0, 100)) for i in range(100)]
        fig2 = equity_chart(eq_hist)
        print(f"✅ Equity grafiği: {'OK' if fig2 is not None else 'None'}")

        fig3 = drawdown_chart(eq_hist)
        print(f"✅ DD grafiği: {'OK' if fig3 is not None else 'None'}")

    # Tablolar
    dec = [
        {"stock": "GARAN", "sinyal": "AL", "p_up": 0.65, "p_down": 0.20, "puan": 3, "sebep": "RSI düşük"},
        {"stock": "AKBNK", "sinyal": "TUT", "p_up": 0.45, "p_down": 0.35, "puan": 0, "sebep": "belirsiz"},
    ]
    t1 = decisions_table(dec)
    print(f"\n✅ Karar tablosu: {len(t1)} satır")

    trades = [
        {"t": 10, "stock": "GARAN", "qty": 100, "entry": 100.0, "exit": 105.0, "pnl": 500, "reason": "TP"},
        {"t": 20, "stock": "AKBNK", "qty": 50, "entry": 200.0, "exit": 195.0, "pnl": -250, "reason": "stop"},
    ]
    t2 = trades_table(trades)
    print(f"✅ İşlem tablosu: {len(t2)} satır")

    t3 = attribution_table(trades)
    print(f"✅ Atıf tablosu: {len(t3)} satır")

    print("\n✅ UI testi geçti")
