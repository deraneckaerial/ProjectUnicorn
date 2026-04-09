"""VIX-basierter Marktregime-Filter.

Statt eines teuren LLM-Aufrufs nutzen wir den VIX (CBOE Volatility Index)
als deterministischen Regime-Indikator. Der VIX misst die erwartete
30-Tage-Volatilitaet des S&P 500 — er ist der "Angstindex" der Maerkte.

Regime-Klassifikation (feste Schwellenwerte, kein ML noetig):

    VIX < 15:    LOW_VOL    — Ruhiger Markt, Trends laufen sauber
    VIX 15-25:   NORMAL     — Normales Umfeld, alle Strategien aktiv
    VIX 25-35:   HIGH_VOL   — Erhoehte Angst, Vorsicht bei Reversals
    VIX > 35:    EXTREME    — Panik/Crash, NUR mit reduziertem Risiko

Zusaetzlich: VIX-Trend
    VIX faellt → "Risk-On" (Anleger werden mutiger)
    VIX steigt → "Risk-Off" (Anleger fluechten in Sicherheit)

Warum kein LLM:
    Der VIX kodiert BEREITS das kollektive Sentiment des Marktes.
    Ein LLM wuerde nur die gleichen Daten (Nachrichten, Stimmung)
    langsamer und teurer verarbeiten als der Optionsmarkt selbst.
    Der VIX ist der effizienteste Sentiment-Indikator der Welt.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


# VIX Regime-Schwellenwerte
REGIME_THRESHOLDS = {
    "LOW_VOL": (0, 15),
    "NORMAL": (15, 25),
    "HIGH_VOL": (25, 35),
    "EXTREME": (35, float("inf")),
}

# VIX-Trend-Perioden (in Handelstagen)
VIX_TREND_PERIOD = 5  # 1 Woche
VIX_TREND_LONG = 20   # 1 Monat


def classify_regime(
    vix_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Klassifiziert das aktuelle Marktregime basierend auf VIX-Daten.

    Args:
        vix_data: OHLCV DataFrame des VIX (Ticker: ^VIX).
            Mindestens 20 Kerzen fuer stabile Trend-Berechnung.

    Returns:
        Dict mit:
            'regime': 'LOW_VOL', 'NORMAL', 'HIGH_VOL', oder 'EXTREME',
            'vix_current': Aktueller VIX-Wert,
            'vix_trend': 'falling' (Risk-On) oder 'rising' (Risk-Off),
            'vix_change_5d': VIX-Aenderung ueber 5 Tage in Prozent,
            'vix_change_20d': VIX-Aenderung ueber 20 Tage in Prozent,
            'trade_allowed': Boolean — ob Trading empfohlen wird,
            'risk_adjustment': Multiplikator fuer Position Sizing (0.0-1.0),
            'erklarung': Verstaendliche Erklaerung,
    """
    if vix_data.empty or len(vix_data) < 5:
        return {
            "regime": "UNKNOWN",
            "vix_current": None,
            "vix_trend": None,
            "trade_allowed": True,
            "risk_adjustment": 0.5,
            "erklarung": "VIX-Daten nicht verfuegbar. Standardmaessig wird "
                         "mit reduziertem Risiko gehandelt.",
        }

    # Aktueller VIX (letzter Close)
    vix_current = float(vix_data["Close"].iloc[-1])

    # Regime bestimmen
    regime = "NORMAL"
    for name, (low, high) in REGIME_THRESHOLDS.items():
        if low <= vix_current < high:
            regime = name
            break

    # VIX-Trend (5 Tage und 20 Tage)
    vix_5d_ago = float(vix_data["Close"].iloc[-min(VIX_TREND_PERIOD, len(vix_data))])
    vix_change_5d = ((vix_current - vix_5d_ago) / vix_5d_ago * 100) if vix_5d_ago != 0 else 0

    vix_20d_ago = float(vix_data["Close"].iloc[-min(VIX_TREND_LONG, len(vix_data))])
    vix_change_20d = ((vix_current - vix_20d_ago) / vix_20d_ago * 100) if vix_20d_ago != 0 else 0

    vix_trend = "falling" if vix_change_5d < -2 else ("rising" if vix_change_5d > 2 else "flat")

    # Trading-Empfehlung und Risiko-Anpassung
    trade_allowed, risk_adj, regime_erklarung = _regime_assessment(
        regime, vix_current, vix_trend, vix_change_5d
    )

    return {
        "regime": regime,
        "vix_current": round(vix_current, 2),
        "vix_trend": vix_trend,
        "vix_trend_label": _trend_label(vix_trend),
        "vix_change_5d": round(vix_change_5d, 1),
        "vix_change_20d": round(vix_change_20d, 1),
        "trade_allowed": trade_allowed,
        "risk_adjustment": risk_adj,
        "erklarung": regime_erklarung,
    }


def _regime_assessment(
    regime: str,
    vix: float,
    trend: str,
    change_5d: float,
) -> tuple:
    """Bestimmt Trading-Empfehlung basierend auf Regime und Trend."""

    if regime == "LOW_VOL":
        return (
            True,
            1.0,
            f"Marktregime: NIEDRIGE VOLATILITAET (VIX: {vix:.1f}). "
            f"Der Markt ist ruhig und Trends laufen tendenziell sauber. "
            f"Unicorn-Setups koennen mit vollem Risiko gehandelt werden. "
            f"Achtung: In sehr ruhigen Maerkten sind Displacement-Kerzen "
            f"seltener — weniger Setups, aber hoehere Qualitaet."
        )

    elif regime == "NORMAL":
        return (
            True,
            1.0,
            f"Marktregime: NORMAL (VIX: {vix:.1f}). "
            f"Standardmaessiges Marktumfeld. Alle ICT-Strategien sind aktiv. "
            f"Keine Einschraenkungen."
        )

    elif regime == "HIGH_VOL":
        if trend == "falling":
            return (
                True,
                0.75,
                f"Marktregime: ERHOEHTE VOLATILITAET (VIX: {vix:.1f}), "
                f"aber FALLEND ({change_5d:+.1f}% in 5 Tagen). "
                f"Die Angst laesst nach — der Markt erholt sich. "
                f"Trades sind moeglich, aber mit 75% der normalen "
                f"Positionsgroesse. Bullische Setups werden bevorzugt."
            )
        else:
            return (
                True,
                0.5,
                f"Marktregime: ERHOEHTE VOLATILITAET (VIX: {vix:.1f}), "
                f"STEIGEND ({change_5d:+.1f}% in 5 Tagen). "
                f"Der Markt wird nervoeser. Trades nur mit 50% der "
                f"normalen Positionsgroesse. Stop-Losses werden haeufiger "
                f"getriggert durch erhoehte Schwankungen."
            )

    else:  # EXTREME
        if trend == "falling":
            return (
                True,
                0.25,
                f"WARNUNG — EXTREMES REGIME (VIX: {vix:.1f}), "
                f"aber FALLEND ({change_5d:+.1f}% in 5 Tagen). "
                f"Potentieller Boden. Wenn gehandelt wird, dann NUR "
                f"mit 25% Positionsgroesse und nur bullische Setups. "
                f"Risiko von Flash-Crashes und Liquiditaetsluecken "
                f"ist weiterhin erhoeht."
            )
        else:
            return (
                False,
                0.0,
                f"HANDEL AUSGESETZT — PANIK-REGIME (VIX: {vix:.1f}, "
                f"{change_5d:+.1f}% in 5 Tagen). "
                f"In Panikphasen funktionieren Price-Action-Strategien "
                f"nicht zuverlaessig. Der Markt wird von Angst und "
                f"Liquiditaetskrisen getrieben, nicht von technischer "
                f"Struktur. Kein Trade bis VIX unter 35 faellt."
            )


def _trend_label(trend: str) -> str:
    """Menschenlesbares Label fuer den VIX-Trend."""
    labels = {
        "falling": "Risk-On (Angst laesst nach)",
        "rising": "Risk-Off (Angst steigt)",
        "flat": "Neutral (stabil)",
    }
    return labels.get(trend, trend)
