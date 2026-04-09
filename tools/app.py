"""Trading View Analyst v2 — Streamlit Dashboard.

A scientific trading analysis tool with anti-bias features,
news sentiment integration, and optimized UI.

Run with:
    cd B:/PlanB/Projekte/TradingViewAnalyst
    venv/Scripts/streamlit.exe run tools/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_market_data import fetch_ohlcv, fetch_multi_timeframe, get_ticker_info
from calculate_indicators import calculate_all_indicators, get_latest_values
from generate_signal import generate_signal, get_direction_label
from calculate_risk import calculate_risk
from forced_disconfirmation import get_disconfirmation
from fetch_sentiment import fetch_news_sentiment, get_sentiment_color
from config import (
    DEFAULT_CAPITAL, DEFAULT_MAX_RISK_PCT,
    DEFAULT_WIN_RATE, DEFAULT_RR_RATIO,
    ATR_LENGTH, RSI_LENGTH, EMA_SHORT, EMA_MID, EMA_LONG,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)
from ict_dashboard import render_ict_tab, render_tracker_tab, TIMEFRAME_OPTIONS

# === Watchlist persistence ===
WATCHLIST_FILE = os.path.join(PROJECT_ROOT, ".tmp", "watchlist.json")
HISTORY_FILE = os.path.join(PROJECT_ROOT, ".tmp", "signal_history.json")


def load_watchlist() -> list:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]


def save_watchlist(wl: list):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f)


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_signal_history(entry: dict):
    history = load_history()
    history.insert(0, entry)
    history = history[:500]  # Keep last 500
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, default=str)


# === PAGE CONFIG ===
st.set_page_config(
    page_title="Trading View Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Readable dark theme — higher contrast */
    .stApp { background-color: #1e2128; color: #e8eaed; }

    .signal-banner {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.5rem 0 1rem 0;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: #282c34;
        border: 1px solid #3d4250;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h4 {
        color: #b0b5bf;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    .metric-card .value {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #f0f1f3;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    .sentiment-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .counter-arg {
        background-color: #2a2035;
        border-left: 3px solid #ff6b6b;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
        color: #e8eaed;
    }
    .bull-arg {
        background-color: #1f3520;
        border-left: 3px solid #69f0ae;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
        color: #e8eaed;
    }

    .news-item {
        background: #282c34;
        border: 1px solid #3d4250;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: #e8eaed;
    }
    .news-item .source {
        color: #9ca3af;
        font-size: 0.75rem;
    }

    .disclaimer {
        background-color: #282c34;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        color: #9ca3af;
        text-align: center;
        margin-top: 1rem;
    }

    /* Readable list items */
    .stMarkdown li, .stMarkdown ul, .stMarkdown ol { font-size: 1.05rem; line-height: 1.7; }
    .bull-arg, .counter-arg { font-size: 1rem; line-height: 1.6; }

    /* Compact sidebar */
    section[data-testid="stSidebar"] .stDivider { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


def build_gauge(score: int, max_val: int = 100) -> go.Figure:
    """Build a semicircular gauge chart for the signal score."""
    # Determine color zones
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": "white"}, "suffix": ""},
        gauge={
            "axis": {"range": [-max_val, max_val], "tickwidth": 1, "tickcolor": "#333"},
            "bar": {"color": "#42A5F5", "thickness": 0.3},
            "bgcolor": "#1a1d23",
            "borderwidth": 0,
            "steps": [
                {"range": [-100, -40], "color": "#FF1744"},
                {"range": [-40, -20], "color": "#FF8A80"},
                {"range": [-20, 20], "color": "#FFD54F"},
                {"range": [20, 40], "color": "#69F0AE"},
                {"range": [40, 100], "color": "#00C853"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def build_sentiment_gauge(score: float) -> go.Figure:
    """Build a mini gauge for sentiment score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 32, "color": "white"}, "valueformat": "+.2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": get_sentiment_color(score), "thickness": 0.3},
            "bgcolor": "#1a1d23",
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.4], "color": "#FF1744"},
                {"range": [-0.4, -0.15], "color": "#FF8A80"},
                {"range": [-0.15, 0.15], "color": "#FFD54F"},
                {"range": [0.15, 0.4], "color": "#69F0AE"},
                {"range": [0.4, 1], "color": "#00C853"},
            ],
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def render_metric_card(label: str, value: str, sub: str = "", color: str = "white") -> str:
    return f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <div class="value" style="color: {color};">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


# === SIDEBAR ===
with st.sidebar:
    st.title("📊 Trading View Analyst")
    st.caption("v2.0 · Wissenschaftlich fundiert")

    st.divider()

    # Watchlist
    watchlist = load_watchlist()
    st.subheader("Watchlist")

    # Add ticker to watchlist
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_ticker = st.text_input("Ticker hinzufuegen", label_visibility="collapsed", placeholder="z.B. TSLA")
    with col_add2:
        if st.button("+", use_container_width=True) and new_ticker:
            t = new_ticker.upper().strip()
            if t not in watchlist:
                watchlist.append(t)
                save_watchlist(watchlist)
                st.rerun()

    # Watchlist buttons
    selected_ticker = None
    wl_cols = st.columns(3)
    for i, t in enumerate(watchlist):
        with wl_cols[i % 3]:
            if st.button(t, key=f"wl_{t}", use_container_width=True):
                selected_ticker = t

    # Remove from watchlist
    if watchlist:
        rm_ticker = st.selectbox("Entfernen:", ["---"] + watchlist, label_visibility="collapsed")
        if rm_ticker != "---":
            watchlist.remove(rm_ticker)
            save_watchlist(watchlist)
            st.rerun()

    st.divider()

    # Ticker input
    ticker = st.text_input(
        "Ticker Symbol",
        value=selected_ticker or "AAPL",
        help="z.B. AAPL, MSFT, BTC-USD, ETH-USD, TSLA",
    ).upper().strip()

    timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAME_OPTIONS.keys()),
        index=list(TIMEFRAME_OPTIONS.keys()).index("Daily"),
    )

    st.divider()
    st.subheader("Risiko-Parameter")

    capital = st.number_input("Kapital", min_value=100, max_value=10_000_000, value=DEFAULT_CAPITAL, step=1000)
    max_risk_pct = st.slider("Max. Risiko/Trade (%)", 0.5, 5.0, float(DEFAULT_MAX_RISK_PCT), 0.5)
    win_rate = st.slider("Win-Rate (%)", 20, 80, DEFAULT_WIN_RATE, 5)
    rr_ratio = st.slider("Reward/Risk", 0.5, 5.0, float(DEFAULT_RR_RATIO), 0.25)

    st.divider()
    analysis_mode = st.radio(
        "Analyse-Modus",
        ["ICT Unicorn", "Klassische Analyse", "Signal Tracker"],
        index=0,
        help="ICT Unicorn: Smart Money Konzepte | Klassisch: Indikator-Scoring | Tracker: Prognosen verfolgen",
    )

    st.divider()
    analyze_btn = st.button("Analyse starten", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        '<div class="disclaimer">Keine Anlageberatung. '
        'Dieses Tool dient der technischen Analyse. '
        'Alle Handelsentscheidungen liegen beim Nutzer.</div>',
        unsafe_allow_html=True,
    )


# === MAIN CONTENT ===
if analysis_mode == "ICT Unicorn":
    if analyze_btn and ticker:
        render_ict_tab(ticker, timeframe, capital, max_risk_pct)
    else:
        st.title("ICT Unicorn Model")
        st.markdown("""
        ### Smart Money Konzepte als Signalgeber

        Das ICT Unicorn Model erkennt institutionelle Preismuster:
        - **Liquidity Sweeps** — Stop Hunts an Swing Levels
        - **Displacement + MSS** — Aggressive Trendwechsel
        - **Fair Value Gaps** — Preis-Ineffizienzen
        - **Breaker Blocks** — Invalidierte Order Blocks
        - **Unicorn Zone** — FVG + BB Konfluenz = Entry

        Plus: **SMT Divergence** (SPY vs QQQ) und **VIX Regime Filter**

        Waehle einen Ticker und Timeframe in der Sidebar und klicke **Analyse starten**.
        """)

elif analysis_mode == "Signal Tracker":
    render_tracker_tab()

elif analyze_btn and ticker:
    with st.spinner(f"Analysiere {ticker}..."):
        try:
            # --- Fetch Data ---
            info = get_ticker_info(ticker)
            _period, _interval = TIMEFRAME_OPTIONS.get(timeframe, ("1y", "1d"))
            period = _period
            interval = _interval
            df = fetch_ohlcv(ticker, period=period, interval=interval)
            df = calculate_all_indicators(df)

            # Fetch other timeframe
            other_interval = "1wk" if interval in ("1d", "1h", "5m", "15m") else "1d"
            other_period = "2y" if other_interval == "1wk" else "1y"
            try:
                df_other = fetch_ohlcv(ticker, period=other_period, interval=other_interval)
                df_other = calculate_all_indicators(df_other)
                signal_other = generate_signal(df_other)
            except Exception:
                df_other = None
                signal_other = None

            # Fetch sentiment
            sentiment = fetch_news_sentiment(ticker)

            # --- Generate Signal ---
            signal = generate_signal(df)
            label, color = get_direction_label(signal["direction"])

            # --- Risk ---
            atr_col = f"ATR_{ATR_LENGTH}"
            atr_val = df.iloc[-1].get(atr_col, 0)
            if pd.isna(atr_val) or atr_val == 0:
                atr_val = df["Close"].iloc[-5:].std()

            risk = calculate_risk(
                signal=signal,
                current_price=df.iloc[-1]["Close"],
                atr=atr_val,
                capital=capital,
                win_rate=win_rate,
                rr_ratio=rr_ratio,
                max_risk_pct=max_risk_pct,
            )

            # --- Disconfirmation ---
            disc = get_disconfirmation(signal)

            # Save to history
            save_signal_history({
                "ticker": ticker,
                "datetime": datetime.now().isoformat(),
                "direction": signal["direction"],
                "score": signal["score"],
                "confidence": signal["confidence"],
                "price": df.iloc[-1]["Close"],
                "sentiment": sentiment["score"] if sentiment["available"] else None,
            })

            # ==========================================
            # DISPLAY
            # ==========================================

            # === HEADER WITH SIGNAL BANNER ===
            st.markdown(f"### {info['name']} ({ticker})")
            st.caption(f"{info['exchange']} · {info['currency']} · {info['type']}")

            st.markdown(
                f'<div class="signal-banner" style="background: {color}18; '
                f'border: 2px solid {color}; color: {color};">'
                f'{label} · Konfidenz: {signal["confidence"]}%'
                f'</div>',
                unsafe_allow_html=True,
            )

            # === TOP ROW: Gauge + Metrics + Sentiment ===
            col_gauge, col_metrics, col_sent = st.columns([1, 1.5, 1])

            with col_gauge:
                st.plotly_chart(build_gauge(signal["score"]), use_container_width=True, key="gauge")

            with col_metrics:
                # Price metrics as cards
                current_price = df.iloc[-1]["Close"]
                prev_close = df.iloc[-2]["Close"] if len(df) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                change_color = "#00C853" if change_pct >= 0 else "#FF1744"

                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(render_metric_card(
                        "Kurs", f"{current_price:,.2f}",
                        f"{change_pct:+.2f}%", change_color
                    ), unsafe_allow_html=True)
                with m2:
                    st.markdown(render_metric_card(
                        "Volumen", f"{df.iloc[-1]['Volume']:,.0f}",
                        "letzte Kerze"
                    ), unsafe_allow_html=True)

                m3, m4 = st.columns(2)
                with m3:
                    # Multi-timeframe alignment
                    if signal_other:
                        other_label, other_color = get_direction_label(signal_other["direction"])
                        other_tf = "W" if other_interval == "1wk" else "D"
                        aligned = signal["direction"] == signal_other["direction"]
                        align_icon = "OK" if aligned else "KONFLIKT"
                        align_color = "#00C853" if aligned else "#FF1744"
                        st.markdown(render_metric_card(
                            f"MTF ({other_tf})", align_icon,
                            f"{other_label}", align_color
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(render_metric_card("MTF", "N/A", ""), unsafe_allow_html=True)
                with m4:
                    st.markdown(render_metric_card(
                        "ATR", f"{atr_val:.2f}",
                        f"{(atr_val / current_price * 100):.1f}% vom Kurs"
                    ), unsafe_allow_html=True)

            with col_sent:
                if sentiment["available"]:
                    st.plotly_chart(build_sentiment_gauge(sentiment["score"]), use_container_width=True, key="sent_gauge")
                    st.caption(f"News: {sentiment['label']} ({sentiment['article_count']} Artikel)")
                else:
                    st.info("News-Sentiment nicht verfuegbar")
                    st.caption(sentiment.get("reason", ""))

            # === CHART ===
            st.divider()

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=("", "MACD", "RSI"),
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="OHLC",
                ),
                row=1, col=1,
            )

            # EMAs
            for ema_len, ema_color, dash in [
                (EMA_SHORT, "#FFD54F", None),
                (EMA_MID, "#42A5F5", None),
                (EMA_LONG, "#EF5350", "dot"),
            ]:
                col_name = f"EMA_{ema_len}"
                if col_name in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df[col_name],
                            name=f"EMA {ema_len}",
                            line=dict(color=ema_color, width=1, dash=dash),
                        ),
                        row=1, col=1,
                    )

            # Bollinger Bands
            bb_upper = "BBU_20_2.0"
            bb_lower = "BBL_20_2.0"
            if bb_upper in df.columns and bb_lower in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[bb_upper], name="BB Upper",
                        line=dict(color="rgba(150,150,150,0.3)", width=1),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[bb_lower], name="BB Lower",
                        line=dict(color="rgba(150,150,150,0.3)", width=1),
                        fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
                    ),
                    row=1, col=1,
                )

            # Stop-Loss / Take-Profit lines
            if risk["stop_loss"] > 0:
                fig.add_hline(
                    y=risk["stop_loss"], line_dash="dash",
                    line_color="#FF1744", annotation_text=f"SL: {risk['stop_loss']:.2f}",
                    row=1, col=1,
                )
            if risk["take_profit"] > 0:
                fig.add_hline(
                    y=risk["take_profit"], line_dash="dash",
                    line_color="#00C853", annotation_text=f"TP: {risk['take_profit']:.2f}",
                    row=1, col=1,
                )

            # MACD
            macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
            macd_sig_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
            macd_hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
            if macd_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[macd_col], name="MACD",
                               line=dict(color="#42A5F5", width=1)),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[macd_sig_col], name="Signal",
                               line=dict(color="#EF5350", width=1)),
                    row=2, col=1,
                )
                hist_colors = ["#69F0AE" if v >= 0 else "#FF8A80" for v in df[macd_hist_col].fillna(0)]
                fig.add_trace(
                    go.Bar(x=df.index, y=df[macd_hist_col], name="Histogram",
                           marker_color=hist_colors),
                    row=2, col=1,
                )

            # RSI
            rsi_col = f"RSI_{RSI_LENGTH}"
            if rsi_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[rsi_col], name="RSI",
                               line=dict(color="#AB47BC", width=1.5)),
                    row=3, col=1,
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.5, row=3, col=1)

            fig.update_layout(
                height=650,
                template="plotly_dark",
                showlegend=False,
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=50, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e2128",
            )
            st.plotly_chart(fig, use_container_width=True)

            # === ANTI-BIAS CHECK (Before Position Sizing!) ===
            st.divider()
            st.subheader("Anti-Bias Check")
            st.info(disc["bias_check"])

            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.markdown("**Dafuer (Bull Case)**")
                for r in signal["reasons"]:
                    st.markdown(f'<div class="bull-arg">{r}</div>', unsafe_allow_html=True)
                if not signal["reasons"]:
                    st.caption("Keine bullishen Argumente.")

            with col_bear:
                st.markdown("**Dagegen (Bear Case)**")
                for r in signal["contra_reasons"]:
                    st.markdown(f'<div class="counter-arg">{r}</div>', unsafe_allow_html=True)
                if not signal["contra_reasons"]:
                    st.caption("Keine bearishen Argumente.")

            if disc["counter_arguments"]:
                with st.expander("Gegenargumente (Forced Disconfirmation)", expanded=True):
                    for i, c in enumerate(disc["counter_arguments"], 1):
                        st.markdown(f"**{i}.** {c}")

            with st.expander("Allgemeine Warnungen"):
                for w in disc["warnings"]:
                    st.warning(w)

            # === NEWS SENTIMENT DETAILS ===
            if sentiment["available"] and sentiment["articles"]:
                st.divider()
                st.subheader("News & Sentiment")

                sent_col1, sent_col2 = st.columns([1, 2])
                with sent_col1:
                    st.markdown(render_metric_card(
                        "Bullish", f"{sentiment['bullish_pct']}%", "", "#00C853"
                    ), unsafe_allow_html=True)
                    st.markdown(render_metric_card(
                        "Bearish", f"{sentiment['bearish_pct']}%", "", "#FF1744"
                    ), unsafe_allow_html=True)

                with sent_col2:
                    for article in sentiment["articles"][:8]:
                        st.markdown(
                            f'<div class="news-item">'
                            f'{article["title"]}<br>'
                            f'<span class="source">{article["source"]} · {article["datetime"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # === POSITION SIZING (After Anti-Bias!) ===
            st.divider()
            st.subheader("Position Sizing & Risikomanagement")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(render_metric_card(
                    "Entry", f"{risk['entry_price']:,.2f}",
                    info['currency']
                ), unsafe_allow_html=True)
            with r2:
                st.markdown(render_metric_card(
                    "Stop-Loss", f"{risk['stop_loss']:,.2f}",
                    f"-{risk['risk_per_share']:,.2f}", "#FF1744"
                ), unsafe_allow_html=True)
            with r3:
                st.markdown(render_metric_card(
                    "Take-Profit", f"{risk['take_profit']:,.2f}",
                    f"+{abs(risk['take_profit'] - risk['entry_price']):,.2f}", "#00C853"
                ), unsafe_allow_html=True)
            with r4:
                st.markdown(render_metric_card(
                    "R/R Ratio", f"{rr_ratio:.1f}x", ""
                ), unsafe_allow_html=True)

            p1, p2, p3 = st.columns(3)
            with p1:
                st.markdown(render_metric_card(
                    "Position", f"{risk['position_size_units']} Stueck",
                    f"= {risk['position_size_value']:,.2f} EUR"
                ), unsafe_allow_html=True)
            with p2:
                risk_color = "#FF1744" if risk['risk_pct_of_capital'] > 3 else "#FFD54F" if risk['risk_pct_of_capital'] > 1.5 else "#00C853"
                st.markdown(render_metric_card(
                    "Risiko", f"{risk['risk_amount']:,.2f} EUR",
                    f"{risk['risk_pct_of_capital']:.1f}% vom Kapital", risk_color
                ), unsafe_allow_html=True)
            with p3:
                st.markdown(render_metric_card(
                    "Chance", f"{risk['reward_amount']:,.2f} EUR",
                    f"Kelly (1/4): {risk['position_size_pct']:.1f}%", "#00C853"
                ), unsafe_allow_html=True)

            if risk["message"]:
                st.info(risk["message"])

            # === SCORING BREAKDOWN ===
            st.divider()
            st.subheader("Scoring-Breakdown")

            components = signal.get("components", {})
            if components:
                score_df = pd.DataFrame([
                    {"Kategorie": "Trend (40%)", "Score": components.get("trend", 0), "Max": 40},
                    {"Kategorie": "Momentum (30%)", "Score": components.get("momentum", 0), "Max": 30},
                    {"Kategorie": "Volumen (15%)", "Score": components.get("volume", 0), "Max": 15},
                    {"Kategorie": "Volatilitaet (15%)", "Score": components.get("volatility", 0), "Max": 15},
                ])

                fig_score = go.Figure()
                colors_score = ["#69F0AE" if s >= 0 else "#FF8A80" for s in score_df["Score"]]
                fig_score.add_trace(go.Bar(
                    x=score_df["Kategorie"],
                    y=score_df["Score"],
                    marker_color=colors_score,
                    text=[f"{s:+d}" for s in score_df["Score"]],
                    textposition="outside",
                    textfont=dict(color="white"),
                ))
                fig_score.update_layout(
                    height=280,
                    template="plotly_dark",
                    yaxis=dict(range=[-50, 50], title="Score"),
                    margin=dict(l=50, r=50, t=20, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#1e2128",
                )
                st.plotly_chart(fig_score, use_container_width=True)

            # === INDICATOR TABLE ===
            with st.expander("Alle Indikator-Werte"):
                latest = get_latest_values(df)
                indicator_keys = [k for k in latest.keys() if k not in ["Open", "High", "Low", "Close", "Volume"]]
                if indicator_keys:
                    ind_df = pd.DataFrame([
                        {"Indikator": k, "Wert": f"{latest[k]:.4f}" if isinstance(latest[k], float) else str(latest[k])}
                        for k in sorted(indicator_keys)
                    ])
                    st.dataframe(ind_df, use_container_width=True, hide_index=True)

        except ValueError as e:
            st.error(f"Fehler: {e}")
        except Exception as e:
            st.error(f"Unerwarteter Fehler: {e}")
            st.exception(e)

elif analysis_mode == "Klassische Analyse" and not analyze_btn:
    # === LANDING PAGE ===
    st.title("📊 Trading View Analyst")
    st.markdown("""
    ### Wissenschaftlich fundierte Trading-Analyse

    Dieses Tool kombiniert **8 technisch validierte Indikatoren** mit **Anti-Bias-Mechanismen**
    und **News-Sentiment**, um emotionales Trading zu reduzieren.
    """)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        **Technische Analyse**
        - Multi-Indikator Scoring
        - MACD, DMI/ADX, RSI, KST
        - EMA, ATR, OBV, Bollinger
        - Multi-Timeframe Vergleich
        """)
    with col_f2:
        st.markdown("""
        **Anti-Bias**
        - Forced Disconfirmation
        - Gegenargumente vor Positionierung
        - Overconfidence-Warnung
        - Action-Bias Erkennung
        """)
    with col_f3:
        st.markdown("""
        **Risikomanagement**
        - ATR-basierter Stop-Loss
        - Quarter Kelly Criterion
        - Position Sizing (5% Hard Cap)
        - News Sentiment Integration
        """)

    st.divider()

    # Show signal history if exists
    history = load_history()
    if history:
        st.subheader("Letzte Analysen")
        hist_df = pd.DataFrame(history[:20])
        if not hist_df.empty and "datetime" in hist_df.columns:
            hist_df["datetime"] = pd.to_datetime(hist_df["datetime"]).dt.strftime("%d.%m %H:%M")
            display_cols = [c for c in ["ticker", "datetime", "direction", "score", "confidence", "price"] if c in hist_df.columns]
            st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Trading View Analyst v2.0 · Streamlit + yfinance + pandas-ta + Finnhub")
