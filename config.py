"""Default configuration for Trading View Analyst."""

# User Defaults (overridable in UI)
DEFAULT_CAPITAL = 10000
DEFAULT_MAX_RISK_PCT = 2  # Max risk per trade in %
DEFAULT_WIN_RATE = 50  # Historical win rate in %
DEFAULT_RR_RATIO = 2.0  # Average reward/risk ratio

# Kelly Criterion
KELLY_FRACTION = 0.25  # Quarter Kelly
MAX_POSITION_PCT = 5  # Hard cap: never more than 5% of capital

# ATR Stop-Loss
ATR_MULTIPLIER = 2.0  # Stop-Loss = Entry ± (ATR * multiplier)

# Scoring Thresholds
STRONG_LONG_THRESHOLD = 40
LEAN_LONG_THRESHOLD = 20
LEAN_SHORT_THRESHOLD = -20
STRONG_SHORT_THRESHOLD = -40

# Indicator Parameters (FIXED - no tuning to prevent overfitting)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
DMI_LENGTH = 14
ADX_LENGTH = 14
RSI_LENGTH = 14
ATR_LENGTH = 14
EMA_SHORT = 20
EMA_MID = 50
EMA_LONG = 200
BB_LENGTH = 20
BB_STD = 2.0

# API Defaults
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
