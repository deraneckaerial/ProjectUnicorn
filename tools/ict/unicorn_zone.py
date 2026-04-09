"""Unicorn Zone Berechnung — Konfluenz von FVG und Breaker Block.

Die Unicorn Zone ist das Herzstuck des ICT Unicorn Models.
Sie entsteht ausschliesslich dort, wo ein Fair Value Gap (FVG)
geometrisch mit einem Breaker Block (BB) ueberlappt.

Diese doppelte Konfluenz erzeugt eine Zone doppelten institutionellen
Drucks: Der Markt will sowohl die Preis-Ineffizienz (FVG) schliessen
als auch die strukturelle Unterstuetzung/Widerstand (BB) respektieren.

Mathematische Validierung:
    FVG-Intervall:  [fvg_low, fvg_high]
    BB-Intervall:   [bb_low, bb_high]

    Ueberlappung existiert wenn:
        max(fvg_low, bb_low) < min(fvg_high, bb_high)

    Unicorn Zone = [max(fvg_low, bb_low), min(fvg_high, bb_high)]

WICHTIG:
    - Raeumliche Naehe (Proximity) ist NICHT ausreichend.
      Die Zonen muessen sich tatsaechlich ueberlappen.
    - FVG und BB muessen aus der GLEICHEN Preissequenz stammen
      (zeitlich nahe beieinander, nicht willkuerlich kombiniert).
"""

from typing import List, Dict, Any

from tools.ict.config import UNICORN_MIN_OVERLAP_PCT, UNICORN_MAX_CANDLE_DISTANCE


def calculate_unicorn_zones(
    fvgs: List[Dict[str, Any]],
    breaker_blocks: List[Dict[str, Any]],
    min_overlap_pct: float = UNICORN_MIN_OVERLAP_PCT,
    max_candle_distance: int = UNICORN_MAX_CANDLE_DISTANCE,
) -> List[Dict[str, Any]]:
    """Berechnet Unicorn Zones aus FVGs und Breaker Blocks.

    Fuer jede Kombination aus FVG und BB gleicher Richtung wird geprueft:
    1. Gleiche Richtung (bullish/bearish)
    2. FVG entstand NACH dem BB (chronologisch)
    3. FVG und BB liegen zeitlich nah beieinander (gleiche Sequenz)
    4. Geometrische Ueberlappung existiert
    5. Ueberlappung ist gross genug (min_overlap_pct)

    Jeder BB wird nur EINMAL dem besten passenden FVG zugeordnet,
    und umgekehrt. Verhindert kombinatorische Explosion.

    Args:
        fvgs: Liste von FVG-Events (aus fvg.py).
        breaker_blocks: Liste von Breaker Blocks (aus breaker_block.py).
        min_overlap_pct: Mindest-Ueberlappung in Prozent.
        max_candle_distance: Maximale zeitliche Distanz zwischen FVG und BB.

    Returns:
        Liste von Unicorn Zones, nach Relevanz sortiert (neueste zuerst).
    """
    if not fvgs or not breaker_blocks:
        return []

    zones = []
    used_fvg_indices = set()
    used_bb_indices = set()

    # Sortiere BBs nach Zeitpunkt (neueste zuerst) — priorisiere aktuelle Setups
    sorted_bbs = sorted(breaker_blocks, key=lambda b: b["bb_index"], reverse=True)

    for bb in sorted_bbs:
        bb_id = id(bb)
        if bb_id in used_bb_indices:
            continue

        best_match = None
        best_overlap_pct = 0

        for fvg in fvgs:
            fvg_id = id(fvg)
            if fvg_id in used_fvg_indices:
                continue

            # Richtung muss uebereinstimmen
            if fvg["direction"] != bb["direction"]:
                continue

            # FVG muss NACH dem BB entstanden sein
            if fvg["fvg_index"] <= bb["bb_index"]:
                continue

            # Temporale Naehe: FVG und BB muessen aus der gleichen
            # Preissequenz stammen. Ein FVG von vor 500 Kerzen hat
            # nichts mit einem aktuellen BB zu tun.
            # Wir nutzen den MSS-Index des BBs als Referenz.
            mss_idx = bb.get("related_mss_index")
            if mss_idx is not None:
                # FVG sollte nahe am MSS liegen (das FVG entsteht
                # WAEHREND des Displacements, das den MSS verursacht)
                try:
                    # Versuche numerischen Vergleich fuer verschiedene Index-Typen
                    if hasattr(fvg["fvg_index"], 'timestamp') and hasattr(mss_idx, 'timestamp'):
                        fvg_ts = fvg["fvg_index"].timestamp()
                        mss_ts = mss_idx.timestamp()
                        bb_ts = bb["bb_index"].timestamp()
                        # FVG sollte innerhalb von max_candle_distance Perioden
                        # nach dem BB liegen
                        if abs(fvg_ts - mss_ts) > abs(mss_ts - bb_ts) * 3:
                            continue
                    else:
                        # Integer-Index: direkter Abstand
                        if abs(fvg["fvg_index"] - bb["bb_index"]) > max_candle_distance:
                            continue
                except (TypeError, AttributeError):
                    pass

            # Geometrische Ueberlappung berechnen
            overlap_low = max(fvg["fvg_low"], bb["bb_low"])
            overlap_high = min(fvg["fvg_high"], bb["bb_high"])

            if overlap_low >= overlap_high:
                continue

            overlap_size = overlap_high - overlap_low
            fvg_size = fvg["fvg_high"] - fvg["fvg_low"]
            bb_size = bb["bb_high"] - bb["bb_low"]
            smaller_zone = min(fvg_size, bb_size)

            if smaller_zone <= 0:
                continue

            overlap_pct = (overlap_size / smaller_zone) * 100

            if overlap_pct < min_overlap_pct:
                continue

            # Bestes Match fuer diesen BB: hoechste Ueberlappung
            if overlap_pct > best_overlap_pct:
                best_overlap_pct = overlap_pct
                best_match = (fvg, overlap_low, overlap_high, overlap_pct)

        if best_match is None:
            continue

        fvg, overlap_low, overlap_high, overlap_pct = best_match
        used_bb_indices.add(bb_id)
        used_fvg_indices.add(id(fvg))

        midpoint = (overlap_low + overlap_high) / 2
        direction_de = "aufwaerts" if fvg["direction"] == "bullish" else "abwaerts"
        action_de = "Kauf" if fvg["direction"] == "bullish" else "Verkauf"

        zones.append({
            "direction": fvg["direction"],
            "zone_high": float(overlap_high),
            "zone_low": float(overlap_low),
            "zone_midpoint": float(midpoint),
            "zone_size": float(overlap_high - overlap_low),
            "fvg": fvg,
            "breaker_block": bb,
            "overlap_pct": float(overlap_pct),
            "erklarung": (
                f"Unicorn Zone ({fvg['direction']}): "
                f"Preisspanne {overlap_low:.2f} – {overlap_high:.2f} "
                f"(Mitte: {midpoint:.2f}). "
                f"In diesem Bereich ueberlagern sich zwei starke Signale: "
                f"Ein Fair Value Gap (Preisluecke von {fvg['fvg_low']:.2f} "
                f"bis {fvg['fvg_high']:.2f}) und ein Breaker Block "
                f"(Strukturzone von {bb['bb_low']:.2f} bis "
                f"{bb['bb_high']:.2f}). Die Ueberlappung betraegt "
                f"{overlap_pct:.0f}%. "
                f"Diese doppelte Konfluenz bedeutet: Der Markt hat sowohl "
                f"einen Grund die Preisluecke zu schliessen, als auch "
                f"strukturelle {action_de}-Unterstuetzung in genau diesem "
                f"Bereich. Wenn der Preis in diese Zone zurueckkehrt, ist "
                f"eine {direction_de}-Reaktion wahrscheinlich."
            ),
        })

    # Sortiere nach Zeitpunkt des FVG (neueste zuerst)
    zones.sort(key=lambda z: z["fvg"]["fvg_index"], reverse=True)
    return zones
