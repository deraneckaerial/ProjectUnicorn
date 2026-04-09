"""ICT Unicorn Dashboard — Rendering-Modul fuer Streamlit.

Enthaelt alle Darstellungsfunktionen fuer die ICT-Analyse:
- Candlestick-Chart mit Swing Levels, FVGs, Breaker Blocks, Unicorn Zone
- Signal-Banner mit Zusammenfassung
- Schritt-fuer-Schritt-Begruendung
- VIX Regime + SMT Divergence Status
- Signal Tracker Tabelle + Statistiken

Wird von app.py importiert und im ICT-Tab aufgerufen.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tools.ict.signal_engine import analyze
from tools.ict.swing_levels import get_swing_levels
from tools.ict.fvg import detect_fvgs
from tools.ict.displacement import detect_displacement, calculate_atr
from tools.ict.breaker_block import detect_breaker_blocks
from tools.ict.liquidity_sweep import detect_liquidity_sweeps
from tools.ict.displacement import detect_mss
from tools.ict.smt_divergence import get_correlated_ticker, is_inverse_pair
from tools.ict.config import (
    SWING_LOOKBACK_BY_TIMEFRAME, SWING_LOOKBACK,
    DISPLACEMENT_MULT_BY_TIMEFRAME, ATR_DISPLACEMENT_MULT,
)
from tools.signal_tracker import (
    save_signal, get_statistics, get_open_signals, get_history,
    check_outcome, mark_expired,
)


# Timeframe mapping fuer yfinance
TIMEFRAME_OPTIONS = {
    "5 Minuten": ("5d", "5m"),
    "15 Minuten": ("60d", "15m"),
    "1 Stunde": ("60d", "1h"),
    "4 Stunden": ("60d", "4h"),  # yfinance doesn't support 4h, use 1h
    "Daily": ("1y", "1d"),
    "Weekly": ("2y", "1wk"),
}


def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def render_ict_tab(ticker: str, timeframe_label: str, capital: float, max_risk_pct: float):
    """Rendert den kompletten ICT Unicorn Tab."""

    period, interval = TIMEFRAME_OPTIONS.get(timeframe_label, ("1y", "1d"))

    with st.spinner(f"ICT Analyse fuer {ticker} ({timeframe_label})..."):
        # --- Daten laden ---
        try:
            df = flatten_columns(
                yf.download(ticker, period=period, interval=interval, progress=False)
            )
        except Exception as e:
            st.error(f"Fehler beim Laden von {ticker}: {e}")
            return

        if df.empty or len(df) < 30:
            st.warning(f"Zu wenig Daten fuer {ticker} ({len(df)} Kerzen). Mindestens 30 benoetigt.")
            return

        # Korreliertes Asset + VIX laden
        corr_ticker = get_correlated_ticker(ticker)
        df_corr = None
        if corr_ticker:
            try:
                df_corr = flatten_columns(
                    yf.download(corr_ticker, period=period, interval=interval, progress=False)
                )
            except Exception:
                df_corr = None

        df_vix = None
        try:
            df_vix = flatten_columns(
                yf.download("^VIX", period="60d", interval="1d", progress=False)
            )
        except Exception:
            pass

        # --- ICT Analyse ---
        result = analyze(
            df,
            capital=capital,
            max_risk_pct=max_risk_pct,
            timeframe=interval,
            ticker=ticker,
            df_correlated=df_corr,
            correlated_ticker=corr_ticker or "",
            df_vix=df_vix,
        )

        # --- Signal Banner ---
        _render_signal_banner(result)

        # --- Konfluenz-Status (VIX + SMT) ---
        _render_confluence_status(result)

        # --- Chart ---
        _render_ict_chart(df, result, interval)

        # --- Begruendung ---
        _render_explanation(result)

        # --- Warnungen ---
        if result["warnungen"]:
            st.divider()
            for w in result["warnungen"]:
                st.warning(w)

        # --- Signal speichern ---
        if result["status"] == "SIGNAL":
            st.divider()
            if st.button("Signal speichern & tracken", type="primary", use_container_width=True):
                sig_id = save_signal(result, instrument=ticker, timeframe=interval)
                if sig_id:
                    st.success(f"Signal gespeichert: {sig_id}")
                else:
                    st.info("Kein Signal zum Speichern.")

        # --- Detail-Statistiken ---
        with st.expander("Detaillierte Zwischenergebnisse"):
            d = result["details"]
            cols = st.columns(4)
            stats = [
                ("Swing Levels", d.get("swing_level_count", 0)),
                ("Sweeps", d.get("sweep_count", 0)),
                ("Displacements", d.get("displacement_count", 0)),
                ("MSS Events", d.get("mss_count", 0)),
            ]
            for i, (label, val) in enumerate(stats):
                cols[i].metric(label, val)

            cols2 = st.columns(4)
            stats2 = [
                ("FVGs", d.get("fvg_count", 0)),
                ("Breaker Blocks", d.get("bb_count", 0)),
                ("Unicorn Zones", d.get("unicorn_zone_count", 0)),
                ("Timeframe", interval),
            ]
            for i, (label, val) in enumerate(stats2):
                cols2[i].metric(label, val)


def render_tracker_tab():
    """Rendert den Signal Tracker Tab."""

    st.subheader("Signal Tracker")

    # Statistiken
    stats = get_statistics()

    if stats["total_signals"] == 0:
        st.info(
            "Noch keine Signale gespeichert. Fuehre eine ICT-Analyse durch "
            "und klicke 'Signal speichern' um mit dem Tracking zu beginnen."
        )
        return

    # Metriken
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gesamt", stats["total_signals"])
    c2.metric("Offen", stats["open_signals"])
    c3.metric("TP Hits", stats["tp_hits"])
    c4.metric("SL Hits", stats["sl_hits"])
    c5.metric(
        "Win Rate",
        f"{stats['win_rate_pct']:.0f}%" if stats["win_rate_pct"] is not None else "—"
    )

    if stats.get("profit_factor") is not None:
        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    if stats.get("bewertung"):
        st.info(stats["bewertung"])

    st.divider()

    # Offene Signale
    open_sigs = get_open_signals()
    if open_sigs:
        st.subheader(f"Offene Signale ({len(open_sigs)})")

        for sig in open_sigs:
            direction_icon = "BUY" if sig["direction"] == "bullish" else "SELL"
            direction_color = "#00C853" if sig["direction"] == "bullish" else "#FF1744"

            with st.container():
                cols = st.columns([1, 2, 1, 1, 1])
                cols[0].markdown(
                    f'<span style="color:{direction_color};font-weight:700;">'
                    f'{direction_icon}</span> {sig["instrument"]}',
                    unsafe_allow_html=True,
                )
                cols[1].caption(
                    f'Entry: {sig["entry_zone_low"]:.2f}-{sig["entry_zone_high"]:.2f} | '
                    f'SL: {sig["stop_loss"]:.2f} | TP: {sig["take_profit"]:.2f}'
                )
                cols[2].caption(f'R:R 1:{sig["rr_ratio"]}')
                cols[3].caption(sig["timestamp_utc"][:16])

                col_check, col_expire = cols[4].columns(2)
                # Outcome-Check wuerde aktuelle Preisdaten brauchen
                if st.button("Ablaufen", key=f"exp_{sig['id']}", help="Signal als abgelaufen markieren"):
                    mark_expired(sig["id"], "Manuell abgelaufen")
                    st.rerun()

    st.divider()

    # Vollstaendige Historie
    history = get_history()
    if history:
        st.subheader("Komplette Historie")

        hist_data = []
        for h in history[-50:]:  # Letzte 50
            outcome_display = h.get("outcome") or "OFFEN"
            outcome_color = {
                "TP_HIT": "#00C853",
                "SL_HIT": "#FF1744",
                "EXPIRED": "#9CA3AF",
                "INVALIDATED": "#FFD54F",
                "OFFEN": "#42A5F5",
            }.get(outcome_display, "#9CA3AF")

            hist_data.append({
                "Datum": h["timestamp_utc"][:10],
                "Instrument": h["instrument"],
                "Richtung": h["direction_display"],
                "Entry": f'{h["entry_optimal"]:.2f}',
                "SL": f'{h["stop_loss"]:.2f}',
                "TP": f'{h["take_profit"]:.2f}',
                "R:R": f'1:{h["rr_ratio"]}',
                "Ergebnis": outcome_display,
            })

        if hist_data:
            st.dataframe(
                pd.DataFrame(hist_data),
                use_container_width=True,
                hide_index=True,
            )


# === Private Rendering-Funktionen ===

def _render_signal_banner(result):
    """Rendert das Signal-Banner oben."""
    status = result["status"]

    if status == "SIGNAL" and result["signal"]:
        sig = result["signal"]
        if sig["direction"] == "bullish":
            color = "#00C853"
            bg = "rgba(0, 200, 83, 0.1)"
            border = "#00C853"
        else:
            color = "#FF1744"
            bg = "rgba(255, 23, 68, 0.1)"
            border = "#FF1744"

        st.markdown(
            f'<div style="padding:1rem 1.5rem;border-radius:12px;text-align:center;'
            f'font-size:1.5rem;font-weight:700;margin:0.5rem 0 1rem 0;'
            f'background:{bg};border:2px solid {border};color:{color};">'
            f'{sig["direction_display"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Kompakte Signal-Details
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f'{sig["entry_optimal"]:.2f}',
                   f'{sig["entry_zone_low"]:.2f} - {sig["entry_zone_high"]:.2f}')
        c2.metric("Stop-Loss", f'{sig["stop_loss"]:.2f}')
        c3.metric("Take-Profit", f'{sig["take_profit"]:.2f}')
        c4.metric("R:R", f'1:{sig["rr_ratio"]}',
                   f'Risiko: {sig["risk_amount"]:.0f} EUR')

    else:
        st.markdown(
            '<div style="padding:1rem 1.5rem;border-radius:12px;text-align:center;'
            'font-size:1.3rem;font-weight:600;margin:0.5rem 0 1rem 0;'
            'background:rgba(150,150,150,0.1);border:2px solid #555;color:#9CA3AF;">'
            'KEIN SIGNAL'
            '</div>',
            unsafe_allow_html=True,
        )
        st.info(result["zusammenfassung"])


def _render_confluence_status(result):
    """Rendert VIX Regime und SMT Divergence als Statuszeile."""
    details = result["details"]
    cols = st.columns(2)

    # VIX Regime
    with cols[0]:
        regime = details.get("regime")
        if regime:
            regime_colors = {
                "LOW_VOL": "#69F0AE",
                "NORMAL": "#42A5F5",
                "HIGH_VOL": "#FFD54F",
                "EXTREME": "#FF1744",
            }
            r_color = regime_colors.get(regime["regime"], "#9CA3AF")
            st.markdown(
                f'<div style="background:#282c34;border:1px solid {r_color};'
                f'border-radius:8px;padding:0.6rem 1rem;">'
                f'<span style="color:{r_color};font-weight:700;">VIX {regime["regime"]}</span>'
                f' &middot; {regime["vix_current"]} '
                f'<span style="color:#9CA3AF;">({regime["vix_trend_label"]})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("VIX: Nicht geladen")

    # SMT Divergence
    with cols[1]:
        smt = details.get("smt_divergence")
        if smt:
            if smt.get("has_divergence"):
                smt_type = smt["divergence_type"]
                smt_color = "#00C853" if smt_type == "bullish" else "#FF1744"
                smt_label = "BULLISH" if smt_type == "bullish" else "BEARISH"

                # Pruefen ob aligned mit Signal
                sig = result.get("signal")
                if sig:
                    aligned = sig.get("smt_aligned", False)
                    align_text = "bestaetigt Signal" if aligned else "WARNUNG: widerspricht Signal"
                    align_color = "#00C853" if aligned else "#FFD54F"
                else:
                    align_text = ""
                    align_color = smt_color

                st.markdown(
                    f'<div style="background:#282c34;border:1px solid {smt_color};'
                    f'border-radius:8px;padding:0.6rem 1rem;">'
                    f'<span style="color:{smt_color};font-weight:700;">SMT {smt_label}</span>'
                    f' <span style="color:{align_color};">{align_text}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#282c34;border:1px solid #555;'
                    'border-radius:8px;padding:0.6rem 1rem;">'
                    '<span style="color:#9CA3AF;">SMT: Keine Divergenz</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("SMT: Kein korreliertes Asset")


def _render_ict_chart(df, result, interval):
    """Rendert den Candlestick-Chart mit ICT-Markierungen."""
    st.divider()

    # Nur die letzten N Kerzen fuer Uebersichtlichkeit
    chart_len = min(200, len(df))
    df_chart = df.iloc[-chart_len:]

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"],
        high=df_chart["High"],
        low=df_chart["Low"],
        close=df_chart["Close"],
        name="OHLC",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
    ))

    # Swing Levels als horizontale Linien
    lookback = SWING_LOOKBACK_BY_TIMEFRAME.get(interval, SWING_LOOKBACK)
    swings = get_swing_levels(df_chart, lookback)

    if not swings.empty:
        for _, sw in swings.iterrows():
            if sw["index"] in df_chart.index:
                color = "rgba(255,215,0,0.4)" if sw["type"] == "high" else "rgba(100,149,237,0.4)"
                fig.add_hline(
                    y=sw["price"],
                    line_dash="dot",
                    line_color=color,
                    line_width=1,
                    annotation_text=f'{"SH" if sw["type"] == "high" else "SL"} {sw["price"]:.2f}',
                    annotation_font_color=color,
                    annotation_font_size=9,
                )

    # FVGs als farbige Rechtecke
    details = result["details"]
    fvgs = details.get("fvgs", [])
    for fvg in fvgs[-10:]:  # Letzte 10
        fvg_idx = fvg.get("fvg_index")
        if fvg_idx is not None and fvg_idx in df_chart.index:
            fvg_color = "rgba(0,200,83,0.12)" if fvg["direction"] == "bullish" else "rgba(255,23,68,0.12)"
            border_color = "rgba(0,200,83,0.4)" if fvg["direction"] == "bullish" else "rgba(255,23,68,0.4)"

            # Rechteck ueber die Breite von 3 Kerzen
            idx_pos = df_chart.index.get_loc(fvg_idx)
            x0 = df_chart.index[max(0, idx_pos - 1)]
            x1 = df_chart.index[min(len(df_chart) - 1, idx_pos + 1)]

            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=fvg["fvg_low"], y1=fvg["fvg_high"],
                fillcolor=fvg_color,
                line=dict(color=border_color, width=1),
            )

    # Unicorn Zone hervorheben
    unicorn_zones = details.get("unicorn_zones", [])
    for uz in unicorn_zones[:3]:  # Top 3
        uz_color = "rgba(255,215,0,0.2)" if uz["direction"] == "bullish" else "rgba(186,85,211,0.2)"
        fig.add_hrect(
            y0=uz["zone_low"], y1=uz["zone_high"],
            fillcolor=uz_color,
            line=dict(color="rgba(255,215,0,0.6)", width=2),
            annotation_text=f'UNICORN {uz["zone_low"]:.2f}-{uz["zone_high"]:.2f}',
            annotation_font_color="gold",
            annotation_font_size=10,
            annotation_position="top left",
        )

    # SL + TP Linien wenn Signal vorhanden
    sig = result.get("signal")
    if sig:
        fig.add_hline(
            y=sig["stop_loss"], line_dash="dash",
            line_color="#FF1744", line_width=1.5,
            annotation_text=f'SL: {sig["stop_loss"]:.2f}',
            annotation_font_color="#FF1744",
        )
        fig.add_hline(
            y=sig["take_profit"], line_dash="dash",
            line_color="#00C853", line_width=1.5,
            annotation_text=f'TP: {sig["take_profit"]:.2f}',
            annotation_font_color="#00C853",
        )
        fig.add_hline(
            y=sig["entry_optimal"], line_dash="dot",
            line_color="#42A5F5", line_width=1,
            annotation_text=f'Entry: {sig["entry_optimal"]:.2f}',
            annotation_font_color="#42A5F5",
        )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e2128",
        yaxis=dict(title="Preis"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_explanation(result):
    """Rendert die Schritt-fuer-Schritt-Begruendung."""
    if not result["begruendung"]:
        return

    st.divider()
    st.subheader("Begruendung")

    for step in result["begruendung"]:
        # Extrahiere Nummer und Titel
        if step[0].isdigit():
            parts = step.split(":", 1)
            if len(parts) == 2:
                header = parts[0].strip()
                body = parts[1].strip()

                # Farbkodierung basierend auf Typ
                if "WARNUNG" in header.upper():
                    st.warning(f"**{header}:** {body}")
                elif "BESTAETIGT" in step.upper():
                    st.success(f"**{header}:** {body}")
                else:
                    with st.expander(header, expanded=False):
                        st.write(body)
            else:
                st.write(step)
        else:
            st.write(step)
