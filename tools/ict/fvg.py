"""Fair Value Gap (FVG) Detection.

Ein Fair Value Gap ist eine Preis-Ineffizienz, die durch drei
aufeinanderfolgende Kerzen entsteht. Die mittlere Kerze bewegt sich
so aggressiv, dass zwischen der ersten und dritten Kerze eine Luecke
im Preisraum entsteht — ein Bereich in dem kaum gehandelt wurde.

Der Markt tendiert dazu, diese Luecken spaeter zu schliessen (zu "fuellen"),
was sie zu starken Anziehungspunkten fuer kuenftige Preisbewegungen macht.

Algorithmische Definition:
    Bullish FVG bei Index i (mittlere Kerze):
        low[i+1] > high[i-1]
        UND close[i] > open[i]  (mittlere Kerze ist bullish)
        → Gap = [high[i-1], low[i+1]]

    Bearish FVG bei Index i (mittlere Kerze):
        high[i+1] < low[i-1]
        UND close[i] < open[i]  (mittlere Kerze ist bearish)
        → Gap = [high[i+1], low[i-1]]
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from tools.ict.config import FVG_MIN_ATR_RATIO, ATR_LENGTH
from tools.ict.displacement import calculate_atr


def detect_fvgs(
    df: pd.DataFrame,
    min_atr_ratio: float = FVG_MIN_ATR_RATIO,
    atr_length: int = ATR_LENGTH,
) -> List[Dict[str, Any]]:
    """Erkennt Fair Value Gaps im OHLCV-DataFrame.

    Prueft fuer jede Dreier-Gruppe aufeinanderfolgender Kerzen, ob eine
    Luecke zwischen Kerze 1 und Kerze 3 besteht. Die mittlere Kerze
    muss die Bewegungsrichtung bestaetigen.

    Zu kleine Gaps (< min_atr_ratio * ATR) werden ignoriert, da sie
    bedeutungslos sind und Fehlsignale erzeugen.

    Args:
        df: OHLCV DataFrame.
        min_atr_ratio: Minimale Gap-Groesse relativ zur ATR.
        atr_length: ATR-Periode.

    Returns:
        Liste von FVG-Events, chronologisch sortiert:
        {
            'fvg_index': Index der mittleren Kerze (Verursacher),
            'direction': 'bullish' oder 'bearish',
            'fvg_high': Oberkante des Gaps,
            'fvg_low': Unterkante des Gaps,
            'gap_size': Absolute Groesse des Gaps,
            'gap_atr_ratio': Gap-Groesse relativ zur ATR,
            'candle1_index': Index der ersten Kerze,
            'candle3_index': Index der dritten Kerze,
            'erklarung': Verstaendliche Erklaerung,
        }
    """
    atr = calculate_atr(df, atr_length)

    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    closes = df["Close"].values
    df_indices = df.index.tolist()

    fvgs = []

    for i in range(1, len(df_indices) - 1):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # Bullish FVG: Luecke zwischen High von Kerze 1 und Low von Kerze 3
        # Die mittlere Kerze muss bullish sein
        if lows[i + 1] > highs[i - 1] and closes[i] > opens[i]:
            gap_low = highs[i - 1]
            gap_high = lows[i + 1]
            gap_size = gap_high - gap_low

            if gap_size / atr_val >= min_atr_ratio:
                fvgs.append({
                    "fvg_index": df_indices[i],
                    "direction": "bullish",
                    "fvg_high": float(gap_high),
                    "fvg_low": float(gap_low),
                    "gap_size": float(gap_size),
                    "gap_atr_ratio": float(gap_size / atr_val),
                    "candle1_index": df_indices[i - 1],
                    "candle3_index": df_indices[i + 1],
                    "erklarung": (
                        f"Bullish Fair Value Gap: Zwischen den Kerzen bei "
                        f"{df_indices[i-1]} und {df_indices[i+1]} entstand eine "
                        f"Preisluecke von {gap_low:.2f} bis {gap_high:.2f} "
                        f"({gap_size:.2f} Punkte, {gap_size/atr_val:.1f}x ATR). "
                        f"Die mittlere Kerze bewegte sich so aggressiv nach oben, "
                        f"dass in diesem Preisbereich kaum gehandelt wurde. "
                        f"Der Markt wird voraussichtlich in diese Zone "
                        f"zurueckkehren, um die Ineffizienz auszugleichen."
                    ),
                })

        # Bearish FVG: Luecke zwischen Low von Kerze 1 und High von Kerze 3
        # Die mittlere Kerze muss bearish sein
        if highs[i + 1] < lows[i - 1] and closes[i] < opens[i]:
            gap_high = lows[i - 1]
            gap_low = highs[i + 1]
            gap_size = gap_high - gap_low

            if gap_size / atr_val >= min_atr_ratio:
                fvgs.append({
                    "fvg_index": df_indices[i],
                    "direction": "bearish",
                    "fvg_high": float(gap_high),
                    "fvg_low": float(gap_low),
                    "gap_size": float(gap_size),
                    "gap_atr_ratio": float(gap_size / atr_val),
                    "candle1_index": df_indices[i - 1],
                    "candle3_index": df_indices[i + 1],
                    "erklarung": (
                        f"Bearish Fair Value Gap: Zwischen den Kerzen bei "
                        f"{df_indices[i-1]} und {df_indices[i+1]} entstand eine "
                        f"Preisluecke von {gap_low:.2f} bis {gap_high:.2f} "
                        f"({gap_size:.2f} Punkte, {gap_size/atr_val:.1f}x ATR). "
                        f"Die mittlere Kerze bewegte sich so aggressiv nach unten, "
                        f"dass in diesem Preisbereich kaum gehandelt wurde. "
                        f"Der Markt wird voraussichtlich in diese Zone "
                        f"zurueckkehren, um die Ineffizienz auszugleichen."
                    ),
                })

    fvgs.sort(key=lambda f: f["fvg_index"])
    return fvgs
