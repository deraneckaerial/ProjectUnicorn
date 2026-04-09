"""Signal Tracker — Speichert Prognosen und prueft Outcomes.

Jedes ICT-Signal wird als Prediction gespeichert. Spaeter kann geprueft
werden, ob der Take-Profit oder Stop-Loss getroffen wurde. So entsteht
ueber Wochen eine echte Datenbasis zur Validierung der Strategie.

Speicherformat: JSON-Datei in .tmp/ict_signal_history.json
Jeder Eintrag ist ein eigenstaendiger Record mit eindeutiger ID.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".tmp", "ict_signal_history.json"
)


def save_signal(analysis_result: Dict[str, Any], instrument: str, timeframe: str) -> Optional[str]:
    """Speichert ein Signal als Prediction im Tracker.

    Args:
        analysis_result: Das vollstaendige Ergebnis von signal_engine.analyze().
        instrument: Ticker-Symbol (z.B. "NQ=F", "EURUSD").
        timeframe: Zeitrahmen (z.B. "5min", "1h", "1d").

    Returns:
        Die Signal-ID oder None wenn kein Signal vorhanden.
    """
    if analysis_result["status"] != "SIGNAL" or analysis_result["signal"] is None:
        return None

    signal = analysis_result["signal"]
    now = datetime.utcnow()

    signal_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{instrument}_{signal['direction']}"

    record = {
        "id": signal_id,
        "timestamp_utc": now.isoformat() + "Z",
        "instrument": instrument,
        "timeframe": timeframe,
        "direction": signal["direction"],
        "direction_display": signal["direction_display"],
        "entry_zone_low": signal["entry_zone_low"],
        "entry_zone_high": signal["entry_zone_high"],
        "entry_optimal": signal["entry_optimal"],
        "stop_loss": signal["stop_loss"],
        "take_profit": signal["take_profit"],
        "rr_ratio": signal["rr_ratio"],
        "risk_amount": signal["risk_amount"],
        "position_size_units": signal["position_size_units"],
        "current_price_at_signal": signal["current_price"],
        "current_atr_at_signal": signal["current_atr"],
        "zusammenfassung": analysis_result["zusammenfassung"],
        "begruendung": analysis_result["begruendung"],
        "warnungen": analysis_result["warnungen"],
        "outcome": None,
        "outcome_timestamp_utc": None,
        "outcome_price": None,
        "outcome_notes": None,
    }

    history = _load_history()
    history.append(record)
    _save_history(history)

    return signal_id


def check_outcome(
    signal_id: str,
    current_high: float,
    current_low: float,
) -> Optional[Dict[str, Any]]:
    """Prueft ob ein gespeichertes Signal sein TP oder SL erreicht hat.

    Vergleicht die aktuellen High/Low-Werte mit den SL/TP-Levels
    des Signals. Aktualisiert den Record wenn ein Outcome feststeht.

    Args:
        signal_id: Die ID des zu pruefenden Signals.
        current_high: Hoechster Preis seit dem Signal.
        current_low: Niedrigster Preis seit dem Signal.

    Returns:
        Dict mit Outcome-Details oder None wenn Signal nicht gefunden.
    """
    history = _load_history()
    record = None
    record_idx = None

    for i, r in enumerate(history):
        if r["id"] == signal_id:
            record = r
            record_idx = i
            break

    if record is None:
        return None

    # Bereits abgeschlossen
    if record["outcome"] is not None:
        return {"status": "already_resolved", "outcome": record["outcome"]}

    sl = record["stop_loss"]
    tp = record["take_profit"]
    direction = record["direction"]

    outcome = None

    if direction == "bullish":
        if current_low <= sl:
            outcome = "SL_HIT"
        elif current_high >= tp:
            outcome = "TP_HIT"
    elif direction == "bearish":
        if current_high >= sl:
            outcome = "SL_HIT"
        elif current_low <= tp:
            outcome = "TP_HIT"

    if outcome:
        now = datetime.utcnow()
        history[record_idx]["outcome"] = outcome
        history[record_idx]["outcome_timestamp_utc"] = now.isoformat() + "Z"
        if outcome == "TP_HIT":
            history[record_idx]["outcome_price"] = tp
        else:
            history[record_idx]["outcome_price"] = sl
        _save_history(history)

    return {
        "signal_id": signal_id,
        "outcome": outcome,
        "still_open": outcome is None,
        "direction": direction,
        "sl": sl,
        "tp": tp,
    }


def mark_expired(signal_id: str, notes: str = "") -> bool:
    """Markiert ein Signal als abgelaufen (weder TP noch SL erreicht).

    Args:
        signal_id: Die ID des Signals.
        notes: Optionale Anmerkungen.

    Returns:
        True wenn erfolgreich, False wenn Signal nicht gefunden.
    """
    history = _load_history()

    for i, r in enumerate(history):
        if r["id"] == signal_id:
            if r["outcome"] is not None:
                return False  # Bereits abgeschlossen
            history[i]["outcome"] = "EXPIRED"
            history[i]["outcome_timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
            history[i]["outcome_notes"] = notes
            _save_history(history)
            return True

    return False


def mark_invalidated(signal_id: str, reason: str) -> bool:
    """Markiert ein Signal als invalidiert (Setup-Bedingungen verletzt).

    Args:
        signal_id: Die ID des Signals.
        reason: Grund der Invalidierung.

    Returns:
        True wenn erfolgreich.
    """
    history = _load_history()

    for i, r in enumerate(history):
        if r["id"] == signal_id:
            if r["outcome"] is not None:
                return False
            history[i]["outcome"] = "INVALIDATED"
            history[i]["outcome_timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
            history[i]["outcome_notes"] = reason
            _save_history(history)
            return True

    return False


def get_statistics() -> Dict[str, Any]:
    """Berechnet Performance-Statistiken ueber alle gespeicherten Signale.

    Returns:
        Dict mit Statistiken: total, resolved, win_rate, etc.
    """
    history = _load_history()

    if not history:
        return {
            "total_signals": 0,
            "hinweis": "Noch keine Signale gespeichert. Die Statistiken "
                       "werden aussagekraeftig sobald genuegend Daten "
                       "gesammelt wurden.",
        }

    total = len(history)
    resolved = [r for r in history if r["outcome"] is not None]
    open_signals = [r for r in history if r["outcome"] is None]
    tp_hits = [r for r in resolved if r["outcome"] == "TP_HIT"]
    sl_hits = [r for r in resolved if r["outcome"] == "SL_HIT"]
    expired = [r for r in resolved if r["outcome"] == "EXPIRED"]
    invalidated = [r for r in resolved if r["outcome"] == "INVALIDATED"]

    # Win Rate nur auf TP/SL berechnen (nicht Expired/Invalidated)
    decided = len(tp_hits) + len(sl_hits)
    win_rate = (len(tp_hits) / decided * 100) if decided > 0 else None

    # Profit Factor (vereinfacht: Anzahl Wins * avg R:R / Anzahl Losses)
    total_reward = sum(r["rr_ratio"] for r in tp_hits) if tp_hits else 0
    profit_factor = total_reward / len(sl_hits) if sl_hits else None

    stats = {
        "total_signals": total,
        "open_signals": len(open_signals),
        "resolved": len(resolved),
        "tp_hits": len(tp_hits),
        "sl_hits": len(sl_hits),
        "expired": len(expired),
        "invalidated": len(invalidated),
        "win_rate_pct": round(win_rate, 1) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
    }

    # Verstaendliche Einordnung
    if win_rate is not None and decided >= 10:
        if win_rate >= 55 and profit_factor and profit_factor >= 1.5:
            stats["bewertung"] = (
                f"Positive Edge: {win_rate:.0f}% Win Rate mit Profit Factor "
                f"{profit_factor:.1f} nach {decided} Trades. Die Strategie "
                f"zeigt einen statistischen Vorteil."
            )
        elif win_rate >= 40 and profit_factor and profit_factor >= 1.0:
            stats["bewertung"] = (
                f"Neutral: {win_rate:.0f}% Win Rate mit Profit Factor "
                f"{profit_factor:.1f} nach {decided} Trades. Noch kein "
                f"klarer Vorteil erkennbar — mehr Daten noetig."
            )
        else:
            stats["bewertung"] = (
                f"Negativ: {win_rate:.0f}% Win Rate mit Profit Factor "
                f"{profit_factor:.1f} nach {decided} Trades. Die Strategie "
                f"zeigt aktuell keinen Vorteil. Parameter ueberpruefen."
            )
    elif decided > 0:
        stats["bewertung"] = (
            f"Zu wenig Daten: Erst {decided} abgeschlossene Trades. "
            f"Mindestens 20-30 Trades noetig fuer belastbare Statistiken."
        )
    else:
        stats["bewertung"] = (
            "Noch keine abgeschlossenen Trades. Signale sammeln und "
            "Outcomes pruefen um die Strategie zu validieren."
        )

    return stats


def get_open_signals() -> List[Dict[str, Any]]:
    """Gibt alle noch offenen Signale zurueck (ohne Outcome)."""
    history = _load_history()
    return [r for r in history if r["outcome"] is None]


def get_history() -> List[Dict[str, Any]]:
    """Gibt die komplette Signal-Historie zurueck."""
    return _load_history()


def _load_history() -> List[Dict[str, Any]]:
    """Laedt die Signal-Historie aus der JSON-Datei."""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, IOError):
        return []


def _save_history(history: List[Dict[str, Any]]) -> None:
    """Speichert die Signal-Historie in die JSON-Datei."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
