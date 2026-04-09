"""Breaker Block Detection.

Ein Breaker Block entsteht aus einem invalidierten Order Block.
Im ICT-Kontext: Wenn der Preis ein Swing Level sweept und dann
aggressiv in die entgegengesetzte Richtung dreht (Displacement),
wird die letzte Kerze(n) vor der Sweep-Bewegung zum Breaker Block.

Bullish Breaker Block:
    1. Preis sweept ein Swing Low (SSL) nach unten
    2. Preis dreht aggressiv nach oben (Displacement + MSS)
    3. Die letzte(n) bearische(n) Kerze(n) VOR der Abwaertsbewegung
       zum Sweep-Low werden zum Breaker Block
    → Diese Zone wirkt nun als Unterstuetzung (Demand Zone)

Bearish Breaker Block:
    1. Preis sweept ein Swing High (BSL) nach oben
    2. Preis dreht aggressiv nach unten (Displacement + MSS)
    3. Die letzte(n) bullische(n) Kerze(n) VOR der Aufwaertsbewegung
       zum Sweep-High werden zum Breaker Block
    → Diese Zone wirkt nun als Widerstand (Supply Zone)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from tools.ict.config import SWING_LOOKBACK


def detect_breaker_blocks(
    df: pd.DataFrame,
    sweeps: List[Dict[str, Any]],
    mss_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Erkennt Breaker Blocks basierend auf vorherigen Sweeps und MSS-Events.

    Fuer jeden Sweep wird geprueft, ob danach ein MSS in entgegengesetzter
    Richtung stattfand. Wenn ja, wird die letzte gegenlaefige Kerze vor
    dem Sweep als Breaker Block identifiziert.

    Args:
        df: OHLCV DataFrame.
        sweeps: Liste von Sweep-Events (aus liquidity_sweep.py).
        mss_events: Liste von MSS-Events (aus displacement.py).

    Returns:
        Liste von Breaker Blocks:
        {
            'bb_index': Index der Breaker-Block-Kerze,
            'direction': 'bullish' (Demand) oder 'bearish' (Supply),
            'bb_high': Oberkante des Breaker Blocks,
            'bb_low': Unterkante des Breaker Blocks,
            'related_sweep_index': Index des zugehoerigen Sweeps,
            'related_mss_index': Index des zugehoerigen MSS,
            'swept_price': Preis des gesweepten Levels,
            'erklarung': Verstaendliche Erklaerung,
        }
    """
    if not sweeps or not mss_events:
        return []

    breaker_blocks = []
    df_indices = df.index.tolist()
    index_to_pos = {idx: pos for pos, idx in enumerate(df_indices)}

    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    for sweep in sweeps:
        sweep_idx = sweep["sweep_index"]
        sweep_pos = index_to_pos.get(sweep_idx)
        if sweep_pos is None:
            continue

        # Finde einen MSS NACH diesem Sweep in der entgegengesetzten Richtung
        matching_mss = None
        for mss in mss_events:
            mss_pos = index_to_pos.get(mss["mss_index"])
            if mss_pos is None:
                continue

            # MSS muss NACH dem Sweep kommen
            if mss_pos <= sweep_pos:
                continue

            # Bullish Sweep → Bullish MSS (Markt dreht nach oben)
            # Bearish Sweep → Bearish MSS (Markt dreht nach unten)
            if sweep["sweep_type"] == "bullish" and mss["direction"] == "bullish":
                matching_mss = mss
                break
            elif sweep["sweep_type"] == "bearish" and mss["direction"] == "bearish":
                matching_mss = mss
                break

        if matching_mss is None:
            continue

        mss_pos = index_to_pos[matching_mss["mss_index"]]

        # Finde den Breaker Block: die letzte(n) gegenlaefige(n) Kerze(n)
        # VOR der Sweep-Bewegung

        if sweep["sweep_type"] == "bullish":
            # Bullish Sweep = Preis ging runter zum Swing Low
            # Suche die letzte bearische Kerze(n) VOR dem Sweep,
            # die den Anfang der Abwaertsbewegung markiert
            bb = _find_opposing_candles(
                opens, closes, highs, lows, df_indices,
                sweep_pos, direction="bearish"
            )
            if bb is not None:
                breaker_blocks.append({
                    "bb_index": bb["index"],
                    "direction": "bullish",
                    "bb_high": bb["high"],
                    "bb_low": bb["low"],
                    "related_sweep_index": sweep_idx,
                    "related_mss_index": matching_mss["mss_index"],
                    "swept_price": sweep["swept_price"],
                    "erklarung": (
                        f"Bullish Breaker Block (Unterstuetzungszone): "
                        f"Preisspanne {bb['low']:.2f} – {bb['high']:.2f}. "
                        f"Diese Zone war urspruenglich ein Angebotsbereich "
                        f"(bearische Kerze), der den Preis nach unten zum "
                        f"Sweep bei {sweep['swept_price']:.2f} trieb. "
                        f"Nachdem institutionelle Kaeufer dort Liquiditaet "
                        f"absorbierten und der Preis aggressiv nach oben drehte, "
                        f"wurde dieser Bereich zu einer starken "
                        f"Unterstuetzungszone. Wenn der Preis hierhin "
                        f"zurueckkehrt, ist mit erneutem Kaufdruck zu rechnen."
                    ),
                })

        elif sweep["sweep_type"] == "bearish":
            # Bearish Sweep = Preis ging hoch zum Swing High
            # Suche die letzte bullische Kerze(n) VOR dem Sweep
            bb = _find_opposing_candles(
                opens, closes, highs, lows, df_indices,
                sweep_pos, direction="bullish"
            )
            if bb is not None:
                breaker_blocks.append({
                    "bb_index": bb["index"],
                    "direction": "bearish",
                    "bb_high": bb["high"],
                    "bb_low": bb["low"],
                    "related_sweep_index": sweep_idx,
                    "related_mss_index": matching_mss["mss_index"],
                    "swept_price": sweep["swept_price"],
                    "erklarung": (
                        f"Bearish Breaker Block (Widerstandszone): "
                        f"Preisspanne {bb['low']:.2f} – {bb['high']:.2f}. "
                        f"Diese Zone war urspruenglich ein Nachfragebereich "
                        f"(bullische Kerze), der den Preis nach oben zum "
                        f"Sweep bei {sweep['swept_price']:.2f} trieb. "
                        f"Nachdem institutionelle Verkaeufer dort Liquiditaet "
                        f"absorbierten und der Preis aggressiv nach unten "
                        f"drehte, wurde dieser Bereich zu einer starken "
                        f"Widerstandszone. Wenn der Preis hierhin "
                        f"zurueckkehrt, ist mit erneutem Verkaufsdruck "
                        f"zu rechnen."
                    ),
                })

    return breaker_blocks


def _find_opposing_candles(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    df_indices: list,
    sweep_pos: int,
    direction: str,
    max_lookback: int = 20,
) -> Optional[Dict[str, Any]]:
    """Findet die letzte gegenlaefige Kerze(n) vor dem Sweep.

    Sucht rueckwaerts vom Sweep-Index nach der letzten zusammenhaengenden
    Gruppe von Kerzen in der angegebenen Richtung. Bei mehreren
    aufeinanderfolgenden Kerzen gleicher Richtung wird die gesamte
    Gruppe als ein Block behandelt.

    Args:
        opens, closes, highs, lows: Preis-Arrays.
        df_indices: Index-Liste des DataFrames.
        sweep_pos: Position des Sweeps im Array.
        direction: 'bullish' oder 'bearish' — welche Kerzenrichtung gesucht wird.
        max_lookback: Maximale Anzahl Kerzen rueckwaerts.

    Returns:
        Dict mit 'index', 'high', 'low' des Breaker Blocks, oder None.
    """
    start = max(0, sweep_pos - max_lookback)

    # Suche rueckwaerts die letzte Kerze der gesuchten Richtung
    bb_candles = []
    found_start = False

    for pos in range(sweep_pos - 1, start - 1, -1):
        is_target = (
            (direction == "bearish" and closes[pos] < opens[pos])
            or (direction == "bullish" and closes[pos] > opens[pos])
        )

        if is_target:
            bb_candles.append(pos)
            found_start = True
        elif found_start:
            # Erste Kerze die NICHT in der Zielrichtung ist → Block-Ende
            break

    if not bb_candles:
        return None

    # Block-Grenzen aus allen zusammenhaengenden Kerzen
    bb_high = max(highs[pos] for pos in bb_candles)
    bb_low = min(lows[pos] for pos in bb_candles)
    # Index der ersten (aeltesten) Kerze im Block
    first_pos = min(bb_candles)

    return {
        "index": df_indices[first_pos],
        "high": float(bb_high),
        "low": float(bb_low),
    }
