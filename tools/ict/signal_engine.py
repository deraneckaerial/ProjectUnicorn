"""ICT Unicorn Signal Engine — Orchestriert alle Detection-Module.

Diese Engine fuehrt die komplette ICT-Analyse durch:
    1. Swing Levels erkennen (Liquiditaets-Pools)
    2. Liquidity Sweeps identifizieren (Stop Hunts)
    3. Displacement + Market Structure Shift pruefen
    4. Fair Value Gaps finden (Preis-Ineffizienzen)
    5. Breaker Blocks bestimmen (invalidierte Order Blocks)
    6. Unicorn Zones berechnen (FVG-BB-Konfluenz)
    7. Risk Management anwenden (SL, TP, R:R, Position Size)
    8. Optional: Kill Zone / Macro Filter

Jedes Signal enthaelt eine vollstaendige Begruendung, die erklaert
WARUM die Empfehlung ausgesprochen wird — verstaendlich fuer
Nicht-Experten und pruefbar fuer Trading-Spezialisten.

WICHTIG: Dieses Modul erfindet KEINE Daten und gibt KEINE Empfehlung
wenn die Bedingungen nicht vollstaendig erfuellt sind. Im Zweifel
wird "KEIN SIGNAL" ausgegeben.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from tools.ict.config import (
    SWING_LOOKBACK, ATR_DISPLACEMENT_MULT, ATR_LENGTH,
    MIN_RR_RATIO, SL_BUFFER_ATR_MULT, MAX_RISK_PCT,
    DISPLACEMENT_MULT_BY_TIMEFRAME, SWING_LOOKBACK_BY_TIMEFRAME,
)
from tools.ict.swing_levels import get_swing_levels
from tools.ict.liquidity_sweep import detect_liquidity_sweeps
from tools.ict.displacement import detect_displacement, detect_mss, calculate_atr
from tools.ict.fvg import detect_fvgs
from tools.ict.breaker_block import detect_breaker_blocks
from tools.ict.unicorn_zone import calculate_unicorn_zones
from tools.ict.killzone import is_in_killzone, is_in_macro
from tools.ict.smt_divergence import detect_smt_divergence, get_correlated_ticker, is_inverse_pair
from tools.ict.regime_filter import classify_regime


def analyze(
    df: pd.DataFrame,
    capital: float = 10000.0,
    max_risk_pct: float = MAX_RISK_PCT,
    timeframe: str = "1d",
    ticker: str = "",
    check_killzone: bool = False,
    current_time_est: Optional[datetime] = None,
    df_correlated: Optional[pd.DataFrame] = None,
    correlated_ticker: str = "",
    df_vix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Fuehrt eine vollstaendige ICT Unicorn Model Analyse durch.

    Args:
        df: OHLCV DataFrame (mindestens 30 Kerzen fuer stabile ATR).
        capital: Gesamtkapital fuer Position Sizing.
        max_risk_pct: Maximales Risiko pro Trade in Prozent.
        timeframe: Zeitrahmen der Daten (z.B. "5m", "1h", "1d").
            Beeinflusst den ATR-Displacement-Schwellenwert.
        ticker: Ticker-Symbol des analysierten Assets.
        check_killzone: Wenn True, wird geprueft ob die aktuelle Zeit
            in einer Kill Zone liegt.
        current_time_est: Aktuelle Zeit in EST (fuer Kill Zone Check).
        df_correlated: Optional — OHLCV DataFrame des korrelierten Assets
            fuer SMT Divergence. Wenn None, wird SMT uebersprungen.
        correlated_ticker: Ticker des korrelierten Assets.
        df_vix: Optional — OHLCV DataFrame des VIX (^VIX) fuer
            Regime-Klassifikation. Wenn None, wird Regime uebersprungen.

    Returns:
        Vollstaendiges Analyse-Ergebnis mit allen Zwischenschritten,
        Signal und Begruendung.
    """
    # Zeitrahmen-spezifischen Displacement-Schwellenwert waehlen
    atr_disp_mult = DISPLACEMENT_MULT_BY_TIMEFRAME.get(timeframe, ATR_DISPLACEMENT_MULT)

    result = {
        "status": "KEIN_SIGNAL",
        "signal": None,
        "zusammenfassung": "",
        "begruendung": [],
        "warnungen": [],
        "details": {},
    }

    # --- Validierung ---
    if len(df) < 30:
        result["warnungen"].append(
            "Zu wenig Daten: Mindestens 30 Kerzen benoetigt fuer stabile Analyse."
        )
        result["zusammenfassung"] = (
            "Analyse nicht moeglich — zu wenig historische Daten vorhanden."
        )
        return result

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        result["warnungen"].append(f"Fehlende Spalten: {missing}")
        result["zusammenfassung"] = "Analyse nicht moeglich — Datenformat ungueltig."
        return result

    # --- Schritt 1: Swing Levels ---
    swing_lookback = SWING_LOOKBACK_BY_TIMEFRAME.get(timeframe, SWING_LOOKBACK)
    swing_levels = get_swing_levels(df, swing_lookback)
    result["details"]["swing_levels"] = swing_levels.to_dict("records") if not swing_levels.empty else []
    result["details"]["swing_level_count"] = len(swing_levels)
    result["details"]["timeframe"] = timeframe
    result["details"]["atr_displacement_mult"] = atr_disp_mult
    result["details"]["swing_lookback"] = swing_lookback

    if swing_levels.empty:
        result["zusammenfassung"] = (
            "Keine Swing Levels erkannt. Der Markt zeigt keine klare Struktur."
        )
        return result

    # --- Schritt 2: Liquidity Sweeps ---
    sweeps = detect_liquidity_sweeps(df, swing_lookback, swing_levels)
    result["details"]["sweeps"] = sweeps
    result["details"]["sweep_count"] = len(sweeps)

    if not sweeps:
        result["zusammenfassung"] = (
            "Keine Liquidity Sweeps erkannt. Ohne vorherigen Stop Hunt fehlt "
            "die institutionelle Triebkraft fuer ein Unicorn Setup."
        )
        return result

    # --- Schritt 3: Displacement + MSS ---
    displacement_df = detect_displacement(df, atr_disp_mult, ATR_LENGTH)
    mss_events = detect_mss(df, displacement_df, swing_levels, atr_disp_mult, ATR_LENGTH)
    result["details"]["displacement_count"] = int(displacement_df["is_displacement"].sum())
    result["details"]["mss_events"] = mss_events
    result["details"]["mss_count"] = len(mss_events)

    if not mss_events:
        result["zusammenfassung"] = (
            "Sweeps erkannt, aber kein Market Structure Shift danach. "
            "Der Markt hat nach dem Stop Hunt nicht aggressiv gedreht — "
            "kein Beweis fuer institutionelle Intervention."
        )
        return result

    # --- Schritt 4: Fair Value Gaps ---
    fvgs = detect_fvgs(df)
    result["details"]["fvgs"] = fvgs
    result["details"]["fvg_count"] = len(fvgs)

    if not fvgs:
        result["zusammenfassung"] = (
            "MSS erkannt, aber keine Fair Value Gaps vorhanden. "
            "Die Preisbewegung war nicht impulsiv genug um eine "
            "Preis-Ineffizienz zu hinterlassen."
        )
        return result

    # --- Schritt 5: Breaker Blocks ---
    breaker_blocks = detect_breaker_blocks(df, sweeps, mss_events)
    result["details"]["breaker_blocks"] = breaker_blocks
    result["details"]["bb_count"] = len(breaker_blocks)

    if not breaker_blocks:
        result["zusammenfassung"] = (
            "FVGs erkannt, aber keine validen Breaker Blocks gefunden. "
            "Es fehlt die strukturelle Unterstuetzung fuer ein Unicorn Setup."
        )
        return result

    # --- Schritt 6: Unicorn Zones ---
    unicorn_zones = calculate_unicorn_zones(fvgs, breaker_blocks)
    result["details"]["unicorn_zones"] = _sanitize_zones(unicorn_zones)
    result["details"]["unicorn_zone_count"] = len(unicorn_zones)

    if not unicorn_zones:
        result["zusammenfassung"] = (
            "FVGs und Breaker Blocks erkannt, aber keine Ueberlappung gefunden. "
            "Die Zonen liegen nicht im gleichen Preisbereich — kein Unicorn Setup."
        )
        return result

    # --- Schritt 7: Bestes Setup waehlen + Risk Management ---
    atr = calculate_atr(df, ATR_LENGTH)
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None

    if current_atr is None or current_atr <= 0:
        result["warnungen"].append("ATR konnte nicht berechnet werden.")
        result["zusammenfassung"] = "Unicorn Zone gefunden, aber ATR nicht verfuegbar fuer Risk Management."
        return result

    best_signal = _select_best_zone(unicorn_zones, current_price, current_atr)

    if best_signal is None:
        result["zusammenfassung"] = (
            "Unicorn Zone(n) gefunden, aber keine ist aktuell handelbar "
            "(Preis zu weit entfernt oder R:R ungenuegend)."
        )
        return result

    # Risk Management berechnen
    risk = _calculate_risk(
        best_signal, current_price, current_atr, capital, max_risk_pct
    )

    if risk is None:
        result["zusammenfassung"] = (
            "Unicorn Zone gefunden, aber das Chance-Risiko-Verhaeltnis "
            f"liegt unter dem Minimum von 1:{MIN_RR_RATIO:.0f}. "
            "Kein Signal — das Risiko ist zu hoch relativ zum Gewinnpotential."
        )
        return result

    # --- Schritt 8: Kill Zone Check (optional) ---
    killzone_info = None
    macro_info = None
    if check_killzone and current_time_est:
        killzone_info = is_in_killzone(current_time_est)
        macro_info = is_in_macro(current_time_est)

        if not killzone_info["in_killzone"]:
            result["warnungen"].append(
                "WARNUNG: Kein aktives Kill-Zone-Fenster. "
                "Setups ausserhalb der Kill Zones haben eine niedrigere "
                "Erfolgswahrscheinlichkeit."
            )

    # --- Schritt 9: VIX Regime Filter (optional) ---
    regime_info = None
    if df_vix is not None and not df_vix.empty:
        regime_info = classify_regime(df_vix)
        result["details"]["regime"] = regime_info

        if not regime_info["trade_allowed"]:
            result["warnungen"].append(regime_info["erklarung"])
            result["zusammenfassung"] = (
                "Unicorn Zone gefunden, aber HANDEL AUSGESETZT wegen "
                f"extremem Marktregime (VIX: {regime_info['vix_current']})."
            )
            return result

        # Risiko-Anpassung bei hoher Volatilitaet
        if regime_info["risk_adjustment"] < 1.0:
            adj = regime_info["risk_adjustment"]
            risk["risk_amount"] = round(risk["risk_amount"] * adj, 2)
            risk["position_size_units"] = round(risk["position_size_units"] * adj, 4)
            risk["position_size_value"] = round(risk["position_size_value"] * adj, 2)
            result["warnungen"].append(
                f"Positionsgroesse auf {adj*100:.0f}% reduziert wegen "
                f"erhoehter Volatilitaet (VIX: {regime_info['vix_current']})."
            )

    # --- Schritt 10: SMT Divergence (optional) ---
    smt_info = None
    if df_correlated is not None and not df_correlated.empty:
        smt_lookback = SWING_LOOKBACK_BY_TIMEFRAME.get(timeframe, SWING_LOOKBACK)
        smt_info = detect_smt_divergence(
            df, df_correlated,
            primary_ticker=ticker,
            secondary_ticker=correlated_ticker,
            lookback=smt_lookback,
            inverse=is_inverse_pair(ticker),
        )
        result["details"]["smt_divergence"] = smt_info

    # --- Ergebnis zusammenbauen ---
    direction = best_signal["direction"]
    direction_de = "LONG (Kauf)" if direction == "bullish" else "SHORT (Verkauf)"

    result["status"] = "SIGNAL"
    result["signal"] = {
        "direction": direction,
        "direction_display": direction_de,
        "entry_zone_low": best_signal["zone_low"],
        "entry_zone_high": best_signal["zone_high"],
        "entry_optimal": best_signal["zone_midpoint"],
        "stop_loss": risk["stop_loss"],
        "take_profit": risk["take_profit"],
        "rr_ratio": risk["rr_ratio"],
        "position_size_units": risk["position_size_units"],
        "position_size_value": risk["position_size_value"],
        "risk_amount": risk["risk_amount"],
        "current_price": current_price,
        "current_atr": current_atr,
        "smt_aligned": _check_smt_alignment(smt_info, direction) if smt_info else None,
        "regime": regime_info["regime"] if regime_info else None,
    }

    # Begruendung zusammenstellen
    result["begruendung"] = _build_explanation(
        best_signal, risk, sweeps, mss_events,
        current_price, current_atr, killzone_info, macro_info,
        smt_info, regime_info,
    )

    result["zusammenfassung"] = _build_summary(
        direction_de, best_signal, risk, current_price
    )

    if killzone_info:
        result["details"]["killzone"] = killzone_info
    if macro_info:
        result["details"]["macro"] = macro_info

    return result


def _check_smt_alignment(smt_info: Dict[str, Any], direction: str) -> bool:
    """Prueft ob die SMT Divergence die Signal-Richtung bestaetigt."""
    if not smt_info or not smt_info.get("has_divergence"):
        return False
    return smt_info["divergence_type"] == ("bullish" if direction == "bullish" else "bearish")


def _select_best_zone(
    zones: List[Dict[str, Any]],
    current_price: float,
    current_atr: float,
) -> Optional[Dict[str, Any]]:
    """Waehlt die relevanteste Unicorn Zone aus.

    Kriterien (in Reihenfolge):
    1. Zone muss in der Naehe des aktuellen Preises liegen (innerhalb 5x ATR)
    2. Hoehere Ueberlappung (overlap_pct) wird bevorzugt
    3. Neuere Zones werden bevorzugt (bereits nach fvg_index sortiert)
    """
    max_distance = 5 * current_atr
    candidates = []

    for zone in zones:
        # Distanz vom aktuellen Preis zur Zone
        if current_price > zone["zone_high"]:
            distance = current_price - zone["zone_high"]
        elif current_price < zone["zone_low"]:
            distance = zone["zone_low"] - current_price
        else:
            distance = 0  # Preis ist IN der Zone

        if distance <= max_distance:
            candidates.append((zone, distance))

    if not candidates:
        return None

    # Sortiere: kleinste Distanz zuerst, bei Gleichstand hoechste Ueberlappung
    candidates.sort(key=lambda c: (c[1], -c[0]["overlap_pct"]))
    return candidates[0][0]


def _calculate_risk(
    zone: Dict[str, Any],
    current_price: float,
    current_atr: float,
    capital: float,
    max_risk_pct: float,
) -> Optional[Dict[str, Any]]:
    """Berechnet Stop-Loss, Take-Profit, R:R und Position Size.

    Stop-Loss: Hinter der Unicorn Zone + ATR-Puffer.
    Take-Profit: Naechstgelegenes Liquiditaets-Ziel (vereinfacht:
        symmetrisch zum Entry basierend auf MIN_RR_RATIO).
    """
    entry = zone["zone_midpoint"]
    sl_buffer = SL_BUFFER_ATR_MULT * current_atr

    if zone["direction"] == "bullish":
        stop_loss = zone["zone_low"] - sl_buffer
        risk_per_unit = entry - stop_loss
        take_profit = entry + (risk_per_unit * MIN_RR_RATIO)
    else:
        stop_loss = zone["zone_high"] + sl_buffer
        risk_per_unit = stop_loss - entry
        take_profit = entry - (risk_per_unit * MIN_RR_RATIO)

    if risk_per_unit <= 0:
        return None

    # R:R berechnen
    reward_per_unit = abs(take_profit - entry)
    rr_ratio = reward_per_unit / risk_per_unit

    if rr_ratio < MIN_RR_RATIO:
        return None

    # Position Sizing: max_risk_pct des Kapitals
    risk_amount = capital * (max_risk_pct / 100)
    position_size_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    position_size_value = position_size_units * entry

    return {
        "stop_loss": round(float(stop_loss), 5),
        "take_profit": round(float(take_profit), 5),
        "risk_per_unit": round(float(risk_per_unit), 5),
        "reward_per_unit": round(float(reward_per_unit), 5),
        "rr_ratio": round(float(rr_ratio), 2),
        "risk_amount": round(float(risk_amount), 2),
        "position_size_units": round(float(position_size_units), 4),
        "position_size_value": round(float(position_size_value), 2),
    }


def _build_explanation(
    zone: Dict[str, Any],
    risk: Dict[str, Any],
    sweeps: list,
    mss_events: list,
    current_price: float,
    current_atr: float,
    killzone_info: Optional[dict],
    macro_info: Optional[dict],
    smt_info: Optional[dict] = None,
    regime_info: Optional[dict] = None,
) -> List[str]:
    """Baut eine verstaendliche Schritt-fuer-Schritt-Begruendung auf."""
    explanation = []

    direction_de = "aufwaerts" if zone["direction"] == "bullish" else "abwaerts"
    action_de = "Kauf" if zone["direction"] == "bullish" else "Verkauf"

    # 1. Liquiditaetssweep
    sweep = zone.get("breaker_block", {})
    related_sweep_idx = sweep.get("related_sweep_index", "unbekannt")
    swept_price = sweep.get("swept_price", 0)
    sweep_type = "bearish" if zone["direction"] == "bullish" else "bullish"

    explanation.append(
        f"1. LIQUIDITAETSSWEEP: Der Preis hat ein signifikantes Swing Level "
        f"bei {swept_price:.2f} durchbrochen und sich sofort zurueckgezogen. "
        f"Das zeigt: Institutionelle Haendler haben die Stop-Loss-Orders "
        f"anderer Marktteilnehmer an diesem Preispunkt gezielt ausgeloest, "
        f"um Gegenpartei-Liquiditaet fuer ihre eigene Position zu beschaffen."
    )

    # 2. Market Structure Shift
    explanation.append(
        f"2. TRENDWECHSEL (MSS): Nach dem Liquiditaetssweep bewegte sich "
        f"der Preis aggressiv in die {direction_de}-Richtung und durchbrach "
        f"dabei die vorherige Marktstruktur. Diese Kerze hatte einen Koerper "
        f"der deutlich groesser war als die durchschnittliche Volatilitaet "
        f"(ATR: {current_atr:.2f}). Das ist der mathematische Beweis, "
        f"dass institutionelles Kapital aktiv den Preis neu bewertet."
    )

    # 3. Fair Value Gap
    fvg = zone.get("fvg", {})
    explanation.append(
        f"3. PREISLUECKE (FVG): Die aggressive Bewegung hinterliess eine "
        f"Luecke im Preisraum von {fvg.get('fvg_low', 0):.2f} bis "
        f"{fvg.get('fvg_high', 0):.2f}. In diesem Bereich wurde kaum "
        f"gehandelt — der Markt wird voraussichtlich zurueckkehren, "
        f"um dieses Ungleichgewicht auszugleichen."
    )

    # 4. Breaker Block
    bb = zone.get("breaker_block", {})
    explanation.append(
        f"4. STRUKTURZONE (Breaker Block): Im Bereich von "
        f"{bb.get('bb_low', 0):.2f} bis {bb.get('bb_high', 0):.2f} "
        f"befindet sich ein ehemaliger Order Block, der nach dem Sweep "
        f"seine Funktion gewechselt hat. Was vorher Widerstand/Unterstuetzung "
        f"war, wirkt jetzt als {action_de}-Zone."
    )

    # 5. Unicorn Konfluenz
    explanation.append(
        f"5. UNICORN KONFLUENZ: Die Preisluecke und die Strukturzone "
        f"ueberlagern sich im Bereich {zone['zone_low']:.2f} – "
        f"{zone['zone_high']:.2f} (Ueberlappung: {zone['overlap_pct']:.0f}%). "
        f"Diese doppelte Konfluenz verdoppelt die Wahrscheinlichkeit "
        f"einer Preisreaktion an diesem Level."
    )

    # 6. Risk Management
    explanation.append(
        f"6. RISIKOMANAGEMENT: Stop-Loss bei {risk['stop_loss']:.2f} "
        f"(hinter der Struktur + Sicherheitspuffer). "
        f"Take-Profit bei {risk['take_profit']:.2f}. "
        f"Chance-Risiko-Verhaeltnis: 1:{risk['rr_ratio']:.1f}. "
        f"Risiko pro Trade: {risk['risk_amount']:.2f} EUR "
        f"({MAX_RISK_PCT}% des Kapitals)."
    )

    # 7. Kill Zone (wenn geprueft)
    if killzone_info:
        if killzone_info["in_killzone"]:
            explanation.append(
                f"7. ZEITFENSTER: {killzone_info['erklarung']}"
            )
        else:
            explanation.append(
                f"7. WARNUNG ZEITFENSTER: {killzone_info['erklarung']}"
            )

    if macro_info and macro_info["in_macro"]:
        explanation.append(
            f"   MACRO: {macro_info['erklarung']}"
        )

    # 8. VIX Regime
    step = 8
    if regime_info:
        explanation.append(
            f"{step}. MARKTREGIME (VIX): {regime_info['erklarung']}"
        )
        step += 1

    # 9. SMT Divergence
    if smt_info:
        if smt_info.get("has_divergence"):
            aligned = smt_info["divergence_type"] == (
                "bullish" if zone["direction"] == "bullish" else "bearish"
            )
            if aligned:
                explanation.append(
                    f"{step}. SMT DIVERGENCE (BESTAETIGT): "
                    f"{smt_info['erklarung']} "
                    f"Die Divergenz bestaetigt die Richtung des Unicorn-Signals "
                    f"— das ist eine der staerksten Konfluenzen im ICT-Framework."
                )
            else:
                explanation.append(
                    f"{step}. SMT DIVERGENCE (WARNUNG): "
                    f"{smt_info['erklarung']} "
                    f"ACHTUNG: Die Divergenz zeigt in die ENTGEGENGESETZTE "
                    f"Richtung des Unicorn-Signals. Das schwaecht die "
                    f"Zuverlaessigkeit des Setups deutlich."
                )
        else:
            explanation.append(
                f"{step}. SMT DIVERGENCE: {smt_info['erklarung']}"
            )

    return explanation


def _build_summary(
    direction_de: str,
    zone: Dict[str, Any],
    risk: Dict[str, Any],
    current_price: float,
) -> str:
    """Baut eine kompakte Zusammenfassung des Signals."""
    return (
        f"ICT UNICORN SIGNAL: {direction_de}\n"
        f"Entry-Zone: {zone['zone_low']:.2f} – {zone['zone_high']:.2f} "
        f"(optimal: {zone['zone_midpoint']:.2f})\n"
        f"Stop-Loss: {risk['stop_loss']:.2f} | "
        f"Take-Profit: {risk['take_profit']:.2f}\n"
        f"R:R = 1:{risk['rr_ratio']:.1f} | "
        f"Risiko: {risk['risk_amount']:.2f} EUR | "
        f"Position: {risk['position_size_units']:.4f} Einheiten\n"
        f"Aktueller Preis: {current_price:.2f}"
    )


def _sanitize_zones(zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entfernt verschachtelte Objekte fuer saubere JSON-Serialisierung."""
    sanitized = []
    for z in zones:
        clean = {
            "direction": z["direction"],
            "zone_high": z["zone_high"],
            "zone_low": z["zone_low"],
            "zone_midpoint": z["zone_midpoint"],
            "zone_size": z["zone_size"],
            "overlap_pct": z["overlap_pct"],
            "erklarung": z["erklarung"],
        }
        sanitized.append(clean)
    return sanitized
