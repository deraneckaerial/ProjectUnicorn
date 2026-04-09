"""Forced Disconfirmation: Generate counter-arguments for every trading signal.

This is an anti-bias feature designed to counter Confirmation Bias.
For every bullish argument, a bearish counter is shown — and vice versa.

Counter-arguments are HARD-CODED per indicator signal. No LLM needed.

Usage:
    from tools.forced_disconfirmation import get_disconfirmation
    counters = get_disconfirmation(signal)
"""


# Static counter-arguments for each type of bullish/bearish signal
COUNTER_ARGUMENTS = {
    # === TREND ===
    "MACD über Signal-Linie und steigend": [
        "MACD ist ein nachlaufender Indikator — die Bewegung könnte bereits eingepreist sein.",
        "In Seitwärtsmärkten liefert MACD häufig Fehlsignale (Whipsaws).",
    ],
    "MACD über Signal-Linie": [
        "MACD über Signal ohne steigende Tendenz zeigt nachlassendes Momentum.",
        "Ein MACD-Crossover allein hat historisch eine Trefferquote unter 50%.",
    ],
    "MACD unter Signal-Linie und fallend": [
        "Überverkaufte Bedingungen können schnelle Gegenbewegungen auslösen.",
        "Bearish MACD in der Nähe von Support-Levels kann ein Fehlsignal sein.",
    ],
    "MACD unter Signal-Linie": [
        "MACD kann in Seitwärtsphasen lange unter der Signal-Linie bleiben ohne weitere Verluste.",
    ],

    "Starker Trend": [
        "Starke Trends enden oft abrupt. Je stärker der ADX, desto näher könnte die Erschöpfung sein.",
        "ADX über 40 zeigt oft einen überhitzten Trend, der anfällig für Mean-Reversion ist.",
    ],
    "Schwacher Trend": [
        "Schwache Trends können in beide Richtungen ausbrechen — Vorsicht vor falscher Sicherheit.",
    ],

    "Bullish EMA-Alignment": [
        "Perfektes EMA-Alignment tritt oft am Ende eines Trends auf, nicht am Anfang.",
        "Wenn alle EMAs aligned sind, ist der Move oft schon fortgeschritten — Late Entry Risiko.",
    ],
    "Bearish EMA-Alignment": [
        "Bearish EMA-Alignment in der Nähe von langfristigen Support-Levels kann eine Bärenfalle sein.",
    ],

    # === MOMENTUM ===
    "RSI überverkauft": [
        "RSI kann in starken Abwärtstrends wochenlang überverkauft bleiben.",
        "'Überverkauft' heißt nicht 'muss steigen'. Es heißt 'fällt schnell'.",
    ],
    "RSI überkauft": [
        "In starken Aufwärtstrends kann RSI monatelang über 70 bleiben.",
        "RSI > 70 in einem Bullenmarkt ist oft Stärke, nicht Schwäche.",
    ],
    "RSI neutral": [
        "Neutraler RSI gibt keine Richtung vor — andere Indikatoren könnten wichtiger sein.",
    ],

    "KST über Signal-Linie": [
        "KST reagiert sehr langsam. Die Trendwende könnte bereits passiert sein.",
        "KST-Crossovers in Seitwärtsmärkten führen zu häufigen Fehlsignalen.",
    ],
    "KST unter Signal-Linie": [
        "KST ist ein stark geglätteter Indikator — kurze Gegenbewegungen werden nicht erfasst.",
    ],

    # === VOLUME ===
    "Volumen bestätigt Aufwärtsbewegung": [
        "Hohes Volumen bei steigenden Kursen kann auch Distribution sein (Smart Money verkauft).",
    ],
    "Volumen bestätigt Abwärtsbewegung": [
        "Hohes Verkaufsvolumen nahe Support kann Kapitulation signalisieren — Wendepunkt möglich.",
    ],
    "Preis steigt, aber Volumen fällt": [
        "Steigender Preis ohne Volumen ist ernst zu nehmen — die Bewegung hat keine Kraft.",
    ],
    "mögliche Akkumulation": [
        "Akkumulation ist schwer von normalem Rauschen zu unterscheiden.",
    ],

    # === VOLATILITY ===
    "Kurs über oberem Bollinger Band": [
        "In starken Trends 'reiten' Kurse oft entlang des oberen Bands — kein automatisches Verkaufssignal.",
    ],
    "Kurs unter unterem Bollinger Band": [
        "Kurse können das untere Band durchbrechen und weiter fallen (Trend-Kontinuation).",
        "Mean-Reversion funktioniert nur in Range-Märkten, nicht in Trendmärkten.",
    ],
    "Erschöpfung möglich": [
        "Trend-Erschöpfungssignale haben eine hohe Fehlalarmrate.",
    ],
}

# General warnings that always apply
GENERAL_WARNINGS = [
    "Technische Analyse allein kann Makro-Events (Zinsentscheide, Earnings, geopolitische Krisen) nicht vorhersagen.",
    "Die vergangene Performance eines Indikators garantiert keine zukünftigen Ergebnisse.",
    "Dieses Scoring-Modell wurde NICHT historisch backtested (kommt in Phase 2).",
]


def get_disconfirmation(signal: dict) -> dict:
    """Generate counter-arguments for a trading signal.

    Args:
        signal: Signal dict from generate_signal with 'reasons' and 'contra_reasons'.

    Returns:
        Dict with:
            counter_arguments: List of counter-arguments to the signal direction
            warnings: General warnings that always apply
            bias_check: Specific bias warning based on signal strength
    """
    direction = signal.get("direction", "neutral")
    reasons = signal.get("reasons", [])
    contra_reasons = signal.get("contra_reasons", [])
    score = signal.get("score", 0)

    counter_arguments = []

    # For each bullish reason, find matching counter-arguments
    all_reasons = reasons + contra_reasons
    for reason in all_reasons:
        for key, counters in COUNTER_ARGUMENTS.items():
            if key.lower() in reason.lower():
                counter_arguments.extend(counters)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_counters = []
    for c in counter_arguments:
        if c not in seen:
            seen.add(c)
            unique_counters.append(c)

    # Bias-specific warnings
    bias_check = _get_bias_warning(direction, score)

    return {
        "counter_arguments": unique_counters,
        "warnings": GENERAL_WARNINGS,
        "bias_check": bias_check,
    }


def _get_bias_warning(direction: str, score: int) -> str:
    """Generate specific bias warning based on signal strength."""
    if direction == "neutral":
        return (
            "🧘 Das Signal ist neutral. Kein Trade ist auch eine Position. "
            "Widersetze dich dem Drang, 'etwas tun zu müssen' (Action Bias)."
        )
    elif "strong" in direction:
        return (
            f"⚠️ Starkes Signal (Score: {score}). Achtung vor Overconfidence! "
            f"Starke Signale fühlen sich 'sicher' an, aber kein Signal ist garantiert. "
            f"Halte dich strikt an die berechnete Position Size."
        )
    elif "long" in direction:
        return (
            "📌 Moderates Long-Signal. Bevor du kaufst: Hast du dir die Gegenargumente "
            "durchgelesen? Suchst du nur nach Bestätigung deiner Meinung (Confirmation Bias)?"
        )
    elif "short" in direction:
        return (
            "📌 Moderates Short-Signal. Shorting fühlt sich oft 'mutiger' an als Kaufen. "
            "Prüfe ob die Evidenz wirklich für Short spricht oder ob du gegen den Markt wetten willst."
        )
    return ""


if __name__ == "__main__":
    # Test with mock signal
    test_signal = {
        "direction": "strong_long",
        "score": 55,
        "reasons": [
            "MACD über Signal-Linie und steigend",
            "Starker Trend (ADX: 32.5)",
            "Bullish EMA-Alignment (Kurs > EMA20 > EMA50 > EMA200)",
            "KST über Signal-Linie — langfristiges Momentum bullish",
            "Volumen bestätigt Aufwärtsbewegung (OBV steigend)",
        ],
        "contra_reasons": [],
    }

    result = get_disconfirmation(test_signal)
    print("=== FORCED DISCONFIRMATION ===\n")
    print(f"Bias Check: {result['bias_check']}\n")
    print("Counter-Arguments:")
    for i, c in enumerate(result["counter_arguments"], 1):
        print(f"  {i}. {c}")
    print(f"\nGeneral Warnings:")
    for w in result["warnings"]:
        print(f"  ⚠️ {w}")
