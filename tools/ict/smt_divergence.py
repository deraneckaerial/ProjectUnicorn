"""SMT Divergence Detection (Smart Money Technique).

SMT Divergence ist einer der staerksten Konfluenzfilter im ICT-Framework.
Er vergleicht zwei korrelierte Maerkte (z.B. NQ und ES, oder QQQ und SPY)
und prueft, ob sie am gleichen Preis-Extrempunkt unterschiedliche
Strukturen zeigen.

Beispiel (Bullish SMT Divergence):
    NQ macht ein neues tieferes Tief (Lower Low)
    ES macht gleichzeitig ein hoeheres Tief (Higher Low)
    → ES weigert sich mitzuziehen → institutionelle Akkumulation in ES
    → Starkes Zeichen fuer eine bevorstehende Aufwaertsbewegung

Beispiel (Bearish SMT Divergence):
    NQ macht ein neues hoeheres Hoch (Higher High)
    ES macht gleichzeitig ein tieferes Hoch (Lower High)
    → ES kann nicht mithalten → institutionelle Distribution
    → Starkes Zeichen fuer eine bevorstehende Abwaertsbewegung

Warum das funktioniert:
    Wenn zwei hoch korrelierte Assets divergieren, zeigt das, dass
    einer der Maerkte von institutionellem Kapital gesteuert wird,
    waehrend der andere dem Retail-Flow folgt. Die Divergenz ist
    der "Fingerabdruck" des Smart Money.

Korrelierte Paare:
    - NQ (Nasdaq 100 Futures) ↔ ES (S&P 500 Futures)
    - QQQ (Nasdaq ETF) ↔ SPY (S&P 500 ETF)
    - DXY (Dollar Index) ↔ EURUSD (invers korreliert)
    - GC (Gold Futures) ↔ DXY (invers korreliert)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from tools.ict.swing_levels import detect_swing_highs, detect_swing_lows
from tools.ict.config import SWING_LOOKBACK, SWING_LOOKBACK_BY_TIMEFRAME


# Vordefinierte korrelierte Paare fuer yfinance
CORRELATED_PAIRS = {
    # Primaer-Asset → Vergleichs-Asset
    "NQ=F": "ES=F",
    "ES=F": "NQ=F",
    "QQQ": "SPY",
    "SPY": "QQQ",
    "TQQQ": "SPY",
    # Forex (invers korreliert — Divergenz-Logik wird umgekehrt)
    "EURUSD=X": "DX-Y.NYB",
    "DX-Y.NYB": "EURUSD=X",
}

# Assets die INVERS korreliert sind (Divergenz = Konvergenz)
INVERSE_PAIRS = {"EURUSD=X", "DX-Y.NYB", "GC=F"}


def detect_smt_divergence(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    primary_ticker: str = "",
    secondary_ticker: str = "",
    lookback: int = SWING_LOOKBACK,
    inverse: bool = False,
) -> Dict[str, Any]:
    """Erkennt SMT Divergence zwischen zwei korrelierten Assets.

    Vergleicht die letzten Swing Highs und Swing Lows beider Assets.
    Wenn das primaere Asset ein neues Extrem macht, aber das sekundaere
    Asset dies nicht bestaetigt, liegt eine Divergenz vor.

    Args:
        df_primary: OHLCV DataFrame des primaeren Assets.
        df_secondary: OHLCV DataFrame des sekundaeren Assets.
        primary_ticker: Ticker des primaeren Assets (fuer Erklaerung).
        secondary_ticker: Ticker des sekundaeren Assets (fuer Erklaerung).
        lookback: Swing Level Lookback.
        inverse: True wenn die Assets invers korreliert sind.

    Returns:
        Dict mit:
            'has_divergence': Boolean,
            'divergence_type': 'bullish', 'bearish', oder None,
            'confidence': 0-100 (wie stark die Divergenz ist),
            'details': Detaillierte Informationen,
            'erklarung': Verstaendliche Erklaerung,
    """
    result = {
        "has_divergence": False,
        "divergence_type": None,
        "confidence": 0,
        "details": {},
        "erklarung": "",
    }

    # Beide DataFrames muessen ueberlappende Zeitraeume haben
    common_start = max(df_primary.index[0], df_secondary.index[0])
    common_end = min(df_primary.index[-1], df_secondary.index[-1])

    df_p = df_primary.loc[common_start:common_end]
    df_s = df_secondary.loc[common_start:common_end]

    if len(df_p) < lookback * 3 or len(df_s) < lookback * 3:
        result["erklarung"] = (
            "Zu wenig ueberlappende Daten fuer SMT-Divergenz-Analyse."
        )
        return result

    # Swing Levels fuer beide Assets erkennen
    p_swing_highs = detect_swing_highs(df_p, lookback)
    p_swing_lows = detect_swing_lows(df_p, lookback)
    s_swing_highs = detect_swing_highs(df_s, lookback)
    s_swing_lows = detect_swing_lows(df_s, lookback)

    # Finde die letzten 2 Swing Highs und Lows in beiden Assets
    p_recent_highs = _get_recent_swings(df_p, p_swing_highs, "High", count=2)
    p_recent_lows = _get_recent_swings(df_p, p_swing_lows, "Low", count=2)
    s_recent_highs = _get_recent_swings(df_s, s_swing_highs, "High", count=2)
    s_recent_lows = _get_recent_swings(df_s, s_swing_lows, "Low", count=2)

    if not p_recent_highs or not p_recent_lows or not s_recent_highs or not s_recent_lows:
        result["erklarung"] = (
            "Nicht genug Swing Levels in einem der Assets fuer SMT-Analyse."
        )
        return result

    # --- Bullish SMT Divergence ---
    # Primary macht Lower Low, Secondary macht Higher Low
    bullish_div = _check_low_divergence(
        p_recent_lows, s_recent_lows, inverse
    )

    # --- Bearish SMT Divergence ---
    # Primary macht Higher High, Secondary macht Lower High
    bearish_div = _check_high_divergence(
        p_recent_highs, s_recent_highs, inverse
    )

    p_label = primary_ticker or "Asset A"
    s_label = secondary_ticker or "Asset B"

    if bullish_div["detected"]:
        result["has_divergence"] = True
        result["divergence_type"] = "bullish"
        result["confidence"] = bullish_div["confidence"]
        result["details"] = bullish_div
        result["erklarung"] = (
            f"BULLISH SMT DIVERGENCE erkannt: "
            f"{p_label} bildete ein tieferes Tief "
            f"({bullish_div['primary_prev']:.2f} -> {bullish_div['primary_last']:.2f}), "
            f"aber {s_label} bildete ein hoeheres Tief "
            f"({bullish_div['secondary_prev']:.2f} -> {bullish_div['secondary_last']:.2f}). "
            f"Das zeigt: {s_label} weigert sich, dem Abwaertsdruck zu folgen. "
            f"Institutionelle Kaeufer akkumulieren bereits in {s_label}. "
            f"In Kombination mit einem bullischen Unicorn-Setup ist dies "
            f"ein sehr starkes Kaufsignal."
        )

    elif bearish_div["detected"]:
        result["has_divergence"] = True
        result["divergence_type"] = "bearish"
        result["confidence"] = bearish_div["confidence"]
        result["details"] = bearish_div
        result["erklarung"] = (
            f"BEARISH SMT DIVERGENCE erkannt: "
            f"{p_label} bildete ein hoeheres Hoch "
            f"({bearish_div['primary_prev']:.2f} -> {bearish_div['primary_last']:.2f}), "
            f"aber {s_label} bildete ein tieferes Hoch "
            f"({bearish_div['secondary_prev']:.2f} -> {bearish_div['secondary_last']:.2f}). "
            f"Das zeigt: {s_label} kann dem Aufwaertsdruck nicht folgen. "
            f"Institutionelle Verkaeufer distribuieren bereits in {s_label}. "
            f"In Kombination mit einem bearischen Unicorn-Setup ist dies "
            f"ein sehr starkes Verkaufssignal."
        )

    else:
        result["erklarung"] = (
            f"Keine SMT-Divergenz zwischen {p_label} und {s_label}. "
            f"Beide Assets bestaetigen die gleiche Marktstruktur. "
            f"Das ist neutral — weder bullish noch bearish."
        )

    return result


def get_correlated_ticker(ticker: str) -> Optional[str]:
    """Gibt den korrelierten Ticker fuer SMT-Analyse zurueck.

    Args:
        ticker: Das primaere Instrument.

    Returns:
        Korrelierter Ticker oder None wenn kein Paar definiert ist.
    """
    return CORRELATED_PAIRS.get(ticker)


def is_inverse_pair(ticker: str) -> bool:
    """Prueft ob das Ticker-Paar invers korreliert ist."""
    return ticker in INVERSE_PAIRS


def _get_recent_swings(
    df: pd.DataFrame,
    swing_mask: pd.Series,
    price_col: str,
    count: int = 2,
) -> List[Tuple[Any, float]]:
    """Gibt die letzten N Swing-Punkte als (Index, Preis) zurueck."""
    indices = df.index[swing_mask]
    if len(indices) < count:
        return []

    recent = indices[-count:]
    return [(idx, float(df.loc[idx, price_col])) for idx in recent]


def _check_low_divergence(
    primary_lows: List[Tuple[Any, float]],
    secondary_lows: List[Tuple[Any, float]],
    inverse: bool,
) -> Dict[str, Any]:
    """Prueft auf Bullish SMT Divergence (Lower Low vs Higher Low).

    Primary macht Lower Low, Secondary macht Higher Low.
    Bei inversen Paaren wird die Logik umgekehrt.
    """
    if len(primary_lows) < 2 or len(secondary_lows) < 2:
        return {"detected": False}

    p_prev, p_last = primary_lows[0][1], primary_lows[1][1]
    s_prev, s_last = secondary_lows[0][1], secondary_lows[1][1]

    # Primary: Lower Low (neuestes Tief < vorheriges Tief)
    primary_lower_low = p_last < p_prev

    if inverse:
        # Bei inversen Paaren: Secondary macht AUCH Lower Low = Divergenz
        # (weil invers: wenn Primary faellt und Secondary auch faellt,
        # ist das eigentlich gegenlaefig)
        secondary_higher_low = s_last < s_prev
    else:
        # Normal: Secondary macht Higher Low
        secondary_higher_low = s_last > s_prev

    if primary_lower_low and secondary_higher_low:
        # Confidence basiert auf der Staerke der Divergenz
        p_drop = (p_prev - p_last) / p_prev * 100 if p_prev != 0 else 0
        s_rise = (s_last - s_prev) / s_prev * 100 if s_prev != 0 else 0
        confidence = min(100, int((abs(p_drop) + abs(s_rise)) * 10))

        return {
            "detected": True,
            "confidence": confidence,
            "primary_prev": p_prev,
            "primary_last": p_last,
            "secondary_prev": s_prev,
            "secondary_last": s_last,
        }

    return {"detected": False}


def _check_high_divergence(
    primary_highs: List[Tuple[Any, float]],
    secondary_highs: List[Tuple[Any, float]],
    inverse: bool,
) -> Dict[str, Any]:
    """Prueft auf Bearish SMT Divergence (Higher High vs Lower High).

    Primary macht Higher High, Secondary macht Lower High.
    """
    if len(primary_highs) < 2 or len(secondary_highs) < 2:
        return {"detected": False}

    p_prev, p_last = primary_highs[0][1], primary_highs[1][1]
    s_prev, s_last = secondary_highs[0][1], secondary_highs[1][1]

    # Primary: Higher High
    primary_higher_high = p_last > p_prev

    if inverse:
        secondary_lower_high = s_last > s_prev
    else:
        # Normal: Secondary macht Lower High
        secondary_lower_high = s_last < s_prev

    if primary_higher_high and secondary_lower_high:
        p_rise = (p_last - p_prev) / p_prev * 100 if p_prev != 0 else 0
        s_drop = (s_prev - s_last) / s_prev * 100 if s_prev != 0 else 0
        confidence = min(100, int((abs(p_rise) + abs(s_drop)) * 10))

        return {
            "detected": True,
            "confidence": confidence,
            "primary_prev": p_prev,
            "primary_last": p_last,
            "secondary_prev": s_prev,
            "secondary_last": s_last,
        }

    return {"detected": False}
