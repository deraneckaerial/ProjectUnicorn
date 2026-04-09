"""Swing Level Detection mittels Rolling-Window-Ansatz.

Ein Swing High ist ein lokales Maximum, das hoeher ist als alle
Nachbarkerzen innerhalb des Lookback-Fensters. Analog fuer Swing Lows.

Diese Levels repraesentieren sichtbare Liquiditaets-Pools, an denen
Stop-Loss-Orders und Breakout-Entries von Retail-Tradern ruhen.
Institutionelle Akteure nutzen diese Levels gezielt als Gegenpartei.

Mathematische Definition:
    Swing High(n) bei Index i: high[i] == max(high[i-n : i+n+1])
    Swing Low(n) bei Index i:  low[i]  == min(low[i-n : i+n+1])
"""

import pandas as pd
import numpy as np

from tools.ict.config import SWING_LOOKBACK


def detect_swing_highs(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.Series:
    """Erkennt Swing Highs im OHLCV-DataFrame.

    Ein Swing High bei Index i liegt vor, wenn high[i] strikt das
    Maximum aller Highs im Fenster [i-lookback, i+lookback] ist.

    Args:
        df: DataFrame mit mindestens einer 'High'-Spalte.
        lookback: Anzahl Kerzen links UND rechts fuer die Pruefung.

    Returns:
        Boolean Series — True an jedem Index der ein Swing High ist.
    """
    highs = df["High"].values
    n = len(highs)
    result = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = highs[i - lookback: i + lookback + 1]
        # Strikt: das Hoch muss einzigartig das Maximum sein.
        # Bei Gleichstand (z.B. Double Top) gilt es NICHT als Swing High,
        # da die Liquiditaet dort bereits geteilt ist.
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            result[i] = True

    return pd.Series(result, index=df.index, name="swing_high")


def detect_swing_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.Series:
    """Erkennt Swing Lows im OHLCV-DataFrame.

    Ein Swing Low bei Index i liegt vor, wenn low[i] strikt das
    Minimum aller Lows im Fenster [i-lookback, i+lookback] ist.

    Args:
        df: DataFrame mit mindestens einer 'Low'-Spalte.
        lookback: Anzahl Kerzen links UND rechts fuer die Pruefung.

    Returns:
        Boolean Series — True an jedem Index der ein Swing Low ist.
    """
    lows = df["Low"].values
    n = len(lows)
    result = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            result[i] = True

    return pd.Series(result, index=df.index, name="swing_low")


def get_swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """Gibt alle erkannten Swing Levels als strukturierten DataFrame zurueck.

    Jeder Eintrag enthaelt den Index, den Preis und den Typ (high/low).
    Die Levels sind chronologisch sortiert.

    Args:
        df: OHLCV DataFrame.
        lookback: Fenstergroesse fuer die Detection.

    Returns:
        DataFrame mit Spalten: ['index', 'price', 'type']
        - index: Original-Index aus dem Input-DataFrame
        - price: Der Preis des Swing Levels (High- oder Low-Wert)
        - type: 'high' oder 'low'
    """
    swing_highs = detect_swing_highs(df, lookback)
    swing_lows = detect_swing_lows(df, lookback)

    levels = []

    for idx in df.index[swing_highs]:
        levels.append({
            "index": idx,
            "price": df.loc[idx, "High"],
            "type": "high",
        })

    for idx in df.index[swing_lows]:
        levels.append({
            "index": idx,
            "price": df.loc[idx, "Low"],
            "type": "low",
        })

    if not levels:
        return pd.DataFrame(columns=["index", "price", "type"])

    result = pd.DataFrame(levels)
    result = result.sort_values("index").reset_index(drop=True)
    return result
