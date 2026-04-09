"""Feste Konfiguration fuer die ICT Unicorn Detection Engine.

WICHTIG: Diese Parameter sind FEST und duerfen NICHT optimiert/getunt werden.
Tuning fuehrt zu Overfitting auf historische Daten. Die Werte basieren auf
den mechanischen Regeln des ICT Unicorn Models (Michael Huddleston).
"""

# --- Swing Level Detection ---
# Anzahl Kerzen links UND rechts, die niedriger/hoeher sein muessen.
# 5 bedeutet: das Hoch muss das hoechste der letzten 11 Kerzen sein (5+1+5).
SWING_LOOKBACK = 5

# Zeitrahmen-spezifische Swing Lookbacks.
# Auf 5min braucht man groessere Fenster, weil kleinere Swings
# nur Rauschen sind. Auf Daily reicht ein kleineres Fenster.
SWING_LOOKBACK_BY_TIMEFRAME = {
    "1m": 20,    # ~40min Signifikanz
    "5m": 12,    # ~2h Signifikanz (signifikante Intraday-Levels)
    "15m": 8,    # ~4h Signifikanz
    "1h": 6,     # ~12h Signifikanz
    "4h": 5,
    "1d": 5,
    "1wk": 3,
}

# --- Displacement / Market Structure Shift ---
# Eine Kerze gilt als Displacement, wenn ihr Koerper (|close - open|)
# groesser ist als dieser Faktor multipliziert mit der ATR.
#
# WICHTIG: Der Schwellenwert haengt vom Zeitrahmen ab.
# Auf 5min-Charts (ICT-Standard) ist 2.5x passend.
# Auf Daily-Charts normalisiert die ATR bereits fuer die Tagesrange,
# daher ist 1.5x bereits aussergewoehnlich (ca. 1 von 100 Kerzen).
#
# Diagnose SPY Daily 6M: Max Body/ATR = 2.05x, nur 1 Kerze >= 1.5x.
ATR_DISPLACEMENT_MULT = 2.5  # Default fuer 5min (ICT-Standard)

# Zeitrahmen-spezifische Anpassungen.
# Die Signal Engine waehlt automatisch den passenden Wert.
DISPLACEMENT_MULT_BY_TIMEFRAME = {
    "1m": 3.0,    # Sehr streng — Minutenchart hat viel Rauschen
    "5m": 2.0,    # Intraday: 2.0x filtert Top ~1% der Kerzen
    "15m": 1.8,
    "1h": 1.8,
    "4h": 1.5,
    "1d": 1.3,    # Daily: 1.3x ATR ist bereits eine grosse Tageskerze
    "1wk": 1.2,   # Weekly: Noch konservativer
}

# ATR-Periode fuer Volatilitaetsmessung
ATR_LENGTH = 14

# --- Fair Value Gap ---
# Minimale Groesse eines FVG relativ zur ATR.
# Zu kleine Gaps sind bedeutungslos und erzeugen Fehlsignale.
FVG_MIN_ATR_RATIO = 0.1

# --- Unicorn Zone ---
# Minimale Ueberlappung zwischen FVG und Breaker Block in Prozent
# der kleineren Zone. Verhindert, dass winzige Schnittmengen als
# valide Unicorn Zone gewertet werden.
UNICORN_MIN_OVERLAP_PCT = 10.0

# Maximale Distanz (in Kerzen) zwischen FVG und BB.
# FVG und BB muessen aus der GLEICHEN Preissequenz stammen.
# Ein FVG von vor 500 Kerzen hat nichts mit einem aktuellen BB zu tun.
UNICORN_MAX_CANDLE_DISTANCE = 30

# --- Risk Management ---
# Minimales Chance-Risiko-Verhaeltnis. Unter diesem Wert wird
# kein Signal generiert, egal wie sauber das Setup ist.
MIN_RR_RATIO = 2.0

# Stop-Loss-Puffer in Preis-Einheiten hinter der Struktur.
# Bei Forex in Pips, bei Futures in Punkten. Wird relativ zur ATR
# berechnet wenn SL_BUFFER_ATR_MULT > 0.
SL_BUFFER_ATR_MULT = 0.2

# Maximales Risiko pro Trade in Prozent des Gesamtkapitals.
MAX_RISK_PCT = 1.0

# Maximale Anzahl Trades pro Session (verhindert Overtrading).
MAX_TRADES_PER_SESSION = 1

# --- Kill Zones (EST / New York Time) ---
# Format: (start_hour, start_minute, end_hour, end_minute)
KILLZONES = {
    "asian": (19, 0, 22, 0),
    "london_open": (2, 0, 5, 0),
    "new_york": (7, 0, 10, 0),
    "london_close": (10, 0, 12, 0),
}

# --- ICT Macros (EST / New York Time) ---
# Praezise Zeitfenster fuer institutionelle Preisauslieferung.
MACROS = {
    "london_1": (2, 33, 3, 0),
    "london_2": (4, 3, 4, 30),
    "ny_am_1": (8, 50, 9, 10),
    "ny_am_2": (9, 50, 10, 10),
    "ny_am_3": (10, 50, 11, 10),
    "ny_lunch": (11, 50, 12, 10),
    "ny_pm_1": (13, 10, 13, 40),
    "ny_last_hour": (15, 15, 15, 45),
}
