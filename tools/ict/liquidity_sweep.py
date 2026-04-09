"""Liquidity Sweep (Stop Hunt) Detection.

Ein Liquidity Sweep liegt vor, wenn der Preis ein bekanntes Swing Level
durchbricht (per Wick/Docht), aber der Schlusskurs sich sofort zurueckzieht.
Das signalisiert: Institutionelle Akteure haben die ruhenden Stop-Loss-Orders
und Breakout-Entries an diesem Level absorbiert, ohne das Preisniveau
nachhaltig zu akzeptieren.

Algorithmische Definition:
    Bearish Sweep (BSL): high[i] > swing_high AND close[i] < swing_high
        → Preis hat Buy-Side Liquidity abgefischt, Wick ueber dem Level
    Bullish Sweep (SSL): low[i] < swing_low AND close[i] > swing_low
        → Preis hat Sell-Side Liquidity abgefischt, Wick unter dem Level

Ohne vorherigen Sweep ist ein Unicorn-Setup strukturell wertlos,
da die institutionelle Triebkraft fehlt.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from tools.ict.swing_levels import get_swing_levels
from tools.ict.config import SWING_LOOKBACK


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_lookback: int = SWING_LOOKBACK,
    swing_levels: pd.DataFrame = None,
) -> List[Dict[str, Any]]:
    """Erkennt Liquidity Sweeps an bekannten Swing Levels.

    Fuer jedes Swing Level wird geprueft, ob eine spaetere Kerze das Level
    per Wick durchbricht, aber per Close zurueckkehrt. Jedes Level kann
    nur einmal gesweept werden (das erste Mal zaehlt).

    Args:
        df: OHLCV DataFrame.
        swing_lookback: Lookback fuer Swing Level Detection.
        swing_levels: Optional vorab berechnete Swing Levels.
            Wenn None, werden sie intern berechnet.

    Returns:
        Liste von Sweep-Events, chronologisch sortiert. Jedes Event:
        {
            'sweep_index': Index der Kerze die den Sweep ausfuehrt,
            'swept_level_index': Index des gesweepten Swing Levels,
            'swept_price': Preis des gesweepten Levels,
            'sweep_type': 'bullish' (SSL sweep) oder 'bearish' (BSL sweep),
            'wick_extreme': Wie weit der Wick ueber/unter das Level ging,
            'close_distance': Abstand des Close zum gesweepten Level,
            'erklarung': Verstaendliche Erklaerung des Events,
        }
    """
    if swing_levels is None:
        swing_levels = get_swing_levels(df, swing_lookback)

    if swing_levels.empty:
        return []

    sweeps = []
    # Track welche Levels bereits gesweept wurden
    swept_level_indices = set()

    # Verwende positional indexing fuer Performance
    df_indices = df.index.tolist()
    index_to_pos = {idx: pos for pos, idx in enumerate(df_indices)}

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    for _, level in swing_levels.iterrows():
        level_idx = level["index"]
        level_price = level["price"]
        level_type = level["type"]

        if level_idx in swept_level_indices:
            continue

        level_pos = index_to_pos.get(level_idx)
        if level_pos is None:
            continue

        # Suche nach Sweep NACH dem Swing Level
        # (mindestens 1 Kerze spaeter)
        for pos in range(level_pos + 1, len(df_indices)):
            candle_idx = df_indices[pos]

            if level_type == "high":
                # Bearish Sweep: Wick ueber Swing High, Close darunter
                if highs[pos] > level_price and closes[pos] < level_price:
                    swept_level_indices.add(level_idx)
                    sweeps.append({
                        "sweep_index": candle_idx,
                        "swept_level_index": level_idx,
                        "swept_price": level_price,
                        "sweep_type": "bearish",
                        "wick_extreme": float(highs[pos]),
                        "close_distance": float(level_price - closes[pos]),
                        "erklarung": (
                            f"Bearish Sweep (Stop Hunt): Der Preis durchbrach das "
                            f"Swing High bei {level_price:.2f} per Docht "
                            f"(Hoch: {highs[pos]:.2f}), schloss aber bei "
                            f"{closes[pos]:.2f} wieder darunter. Das bedeutet: "
                            f"Kauforders oberhalb des Levels (Buy-Side Liquidity) "
                            f"wurden von institutionellen Akteuren absorbiert, "
                            f"ohne dass der Preis nachhaltig steigen konnte."
                        ),
                    })
                    break

            elif level_type == "low":
                # Bullish Sweep: Wick unter Swing Low, Close darueber
                if lows[pos] < level_price and closes[pos] > level_price:
                    swept_level_indices.add(level_idx)
                    sweeps.append({
                        "sweep_index": candle_idx,
                        "swept_level_index": level_idx,
                        "swept_price": level_price,
                        "sweep_type": "bullish",
                        "wick_extreme": float(lows[pos]),
                        "close_distance": float(closes[pos] - level_price),
                        "erklarung": (
                            f"Bullish Sweep (Stop Hunt): Der Preis durchbrach das "
                            f"Swing Low bei {level_price:.2f} per Docht "
                            f"(Tief: {lows[pos]:.2f}), schloss aber bei "
                            f"{closes[pos]:.2f} wieder darueber. Das bedeutet: "
                            f"Verkaufsorders unterhalb des Levels (Sell-Side "
                            f"Liquidity) wurden von institutionellen Akteuren "
                            f"absorbiert — ein Zeichen fuer eine moegliche "
                            f"Trendumkehr nach oben."
                        ),
                    })
                    break

    # Chronologisch sortieren
    sweeps.sort(key=lambda s: s["sweep_index"])
    return sweeps
