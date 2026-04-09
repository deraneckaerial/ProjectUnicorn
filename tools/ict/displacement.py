"""Displacement und Market Structure Shift (MSS) Detection.

Displacement ist eine aggressive, impulsive Preisbewegung die zeigt,
dass institutionelles Kapital den Preis aktiv neu bewertet. Es manifestiert
sich durch grosse Kerzenkoerper mit minimalen Dochten.

Algorithmische Quantifizierung:
    Displacement liegt vor wenn:
        abs(close - open) > ATR_DISPLACEMENT_MULT * ATR(ATR_LENGTH)

    Ein Market Structure Shift (MSS) wird bestaetigt wenn:
        Die Displacement-Kerze mit ihrem Close das letzte gegenueberliegende
        Swing Level durchbricht.

    Bullish MSS: Displacement close > letztes Swing High
    Bearish MSS: Displacement close < letztes Swing Low

Das Displacement ist der mathematische Beweis dafuer, dass institutionelles
Kapital aktiv eingreift. Ohne Displacement kein valides Setup.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from tools.ict.config import ATR_DISPLACEMENT_MULT, ATR_LENGTH
from tools.ict.swing_levels import get_swing_levels


def calculate_atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    """Berechnet die Average True Range (ATR).

    Die ATR misst die durchschnittliche Volatilitaet ueber die letzten
    `length` Perioden. Sie wird als Referenzwert genutzt, um zu beurteilen
    ob eine Kerze ungewoehnlich gross (= Displacement) ist.

    True Range = max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close)
    )

    Args:
        df: OHLCV DataFrame.
        length: Anzahl Perioden fuer den Durchschnitt.

    Returns:
        Series mit ATR-Werten.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=length, min_periods=length).mean()

    return atr


def detect_displacement(
    df: pd.DataFrame,
    atr_mult: float = ATR_DISPLACEMENT_MULT,
    atr_length: int = ATR_LENGTH,
) -> pd.DataFrame:
    """Erkennt Displacement-Kerzen basierend auf ATR-Multiplikator.

    Eine Kerze ist ein Displacement, wenn ihr Koerper (|close - open|)
    groesser ist als atr_mult * ATR. Das filtert normales Kursrauschen
    heraus und identifiziert nur wirklich aggressive Bewegungen.

    Args:
        df: OHLCV DataFrame.
        atr_mult: Mindest-Multiplikator des ATR fuer Displacement.
        atr_length: Periode fuer ATR-Berechnung.

    Returns:
        DataFrame mit Spalten:
            'is_displacement': Boolean — ist diese Kerze ein Displacement?
            'displacement_direction': 'bullish', 'bearish', oder None
            'body_size': Absolute Groesse des Kerzenkoerpers
            'atr': ATR-Wert an dieser Stelle
            'body_atr_ratio': Verhaeltnis Koerpergroesse zu ATR
    """
    atr = calculate_atr(df, atr_length)
    body = (df["Close"] - df["Open"]).abs()
    direction = np.where(
        df["Close"] > df["Open"], "bullish",
        np.where(df["Close"] < df["Open"], "bearish", None)
    )

    threshold = atr_mult * atr
    is_displacement = body > threshold

    # Bei NaN-ATR (erste Kerzen) kann kein Displacement erkannt werden
    is_displacement = is_displacement.fillna(False)

    result = pd.DataFrame({
        "is_displacement": is_displacement,
        "displacement_direction": np.where(is_displacement, direction, None),
        "body_size": body,
        "atr": atr,
        "body_atr_ratio": np.where(atr > 0, body / atr, 0.0),
    }, index=df.index)

    return result


def detect_mss(
    df: pd.DataFrame,
    displacement_df: pd.DataFrame = None,
    swing_levels: pd.DataFrame = None,
    atr_mult: float = ATR_DISPLACEMENT_MULT,
    atr_length: int = ATR_LENGTH,
) -> List[Dict[str, Any]]:
    """Erkennt Market Structure Shifts (MSS).

    Ein MSS tritt auf, wenn eine Displacement-Kerze das letzte
    gegenueberliegende Swing Level mit ihrem Close durchbricht.

    Bullish MSS: Nach einem bearish Sweep dreht der Markt —
        eine bullische Displacement-Kerze schliesst ueber dem letzten Swing High.
    Bearish MSS: Nach einem bullish Sweep dreht der Markt —
        eine bearische Displacement-Kerze schliesst unter dem letzten Swing Low.

    Args:
        df: OHLCV DataFrame.
        displacement_df: Vorab berechnete Displacement-Daten.
        swing_levels: Vorab berechnete Swing Levels.
        atr_mult: ATR-Multiplikator fuer Displacement.
        atr_length: ATR-Periode.

    Returns:
        Liste von MSS-Events:
        {
            'mss_index': Index der Displacement-Kerze,
            'direction': 'bullish' oder 'bearish',
            'broken_level_price': Preis des durchbrochenen Swing Levels,
            'broken_level_index': Index des durchbrochenen Levels,
            'displacement_close': Close der Displacement-Kerze,
            'body_atr_ratio': Wie stark das Displacement relativ zur ATR war,
            'erklarung': Verstaendliche Erklaerung,
        }
    """
    if displacement_df is None:
        displacement_df = detect_displacement(df, atr_mult, atr_length)

    if swing_levels is None:
        swing_levels = get_swing_levels(df)

    if swing_levels.empty:
        return []

    mss_events = []
    df_indices = df.index.tolist()
    index_to_pos = {idx: pos for pos, idx in enumerate(df_indices)}

    closes = df["Close"].values

    # Fuer jede Displacement-Kerze pruefen ob sie ein Swing Level bricht
    disp_mask = displacement_df["is_displacement"].values
    disp_dirs = displacement_df["displacement_direction"].values
    body_ratios = displacement_df["body_atr_ratio"].values

    for pos in range(len(df_indices)):
        if not disp_mask[pos]:
            continue

        candle_idx = df_indices[pos]
        disp_dir = disp_dirs[pos]

        if disp_dir == "bullish":
            # Suche das letzte Swing High VOR dieser Kerze
            prior_highs = swing_levels[
                (swing_levels["type"] == "high")
                & (swing_levels["index"] < candle_idx)
            ]
            if prior_highs.empty:
                continue

            last_high = prior_highs.iloc[-1]
            # MSS: Close bricht ueber das letzte Swing High
            if closes[pos] > last_high["price"]:
                mss_events.append({
                    "mss_index": candle_idx,
                    "direction": "bullish",
                    "broken_level_price": float(last_high["price"]),
                    "broken_level_index": last_high["index"],
                    "displacement_close": float(closes[pos]),
                    "body_atr_ratio": float(body_ratios[pos]),
                    "erklarung": (
                        f"Bullish Market Structure Shift: Eine aggressive "
                        f"Aufwaertskerze (Koerper = {body_ratios[pos]:.1f}x ATR) "
                        f"schloss bei {closes[pos]:.2f} und durchbrach damit das "
                        f"vorherige Swing High bei {last_high['price']:.2f}. "
                        f"Das signalisiert: Die Marktstruktur hat sich von "
                        f"abwaerts zu aufwaerts verschoben. Institutionelle "
                        f"Kaeufer haben die Kontrolle uebernommen."
                    ),
                })

        elif disp_dir == "bearish":
            # Suche das letzte Swing Low VOR dieser Kerze
            prior_lows = swing_levels[
                (swing_levels["type"] == "low")
                & (swing_levels["index"] < candle_idx)
            ]
            if prior_lows.empty:
                continue

            last_low = prior_lows.iloc[-1]
            # MSS: Close bricht unter das letzte Swing Low
            if closes[pos] < last_low["price"]:
                mss_events.append({
                    "mss_index": candle_idx,
                    "direction": "bearish",
                    "broken_level_price": float(last_low["price"]),
                    "broken_level_index": last_low["index"],
                    "displacement_close": float(closes[pos]),
                    "body_atr_ratio": float(body_ratios[pos]),
                    "erklarung": (
                        f"Bearish Market Structure Shift: Eine aggressive "
                        f"Abwaertskerze (Koerper = {body_ratios[pos]:.1f}x ATR) "
                        f"schloss bei {closes[pos]:.2f} und durchbrach damit das "
                        f"vorherige Swing Low bei {last_low['price']:.2f}. "
                        f"Das signalisiert: Die Marktstruktur hat sich von "
                        f"aufwaerts zu abwaerts verschoben. Institutionelle "
                        f"Verkaeufer haben die Kontrolle uebernommen."
                    ),
                })

    mss_events.sort(key=lambda e: e["mss_index"])
    return mss_events
