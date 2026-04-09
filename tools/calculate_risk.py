"""Risk management: ATR-based Stop-Loss + Quarter Kelly position sizing.

Usage:
    from tools.calculate_risk import calculate_risk
    risk = calculate_risk(signal, current_price, atr, capital=10000)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KELLY_FRACTION, MAX_POSITION_PCT, ATR_MULTIPLIER,
    DEFAULT_CAPITAL, DEFAULT_MAX_RISK_PCT,
    DEFAULT_WIN_RATE, DEFAULT_RR_RATIO,
)


def kelly_criterion(win_rate: float, rr_ratio: float, fraction: float = KELLY_FRACTION) -> float:
    """Calculate fractional Kelly Criterion position size.

    Args:
        win_rate: Historical win rate (0-100, e.g., 55 for 55%)
        rr_ratio: Average reward/risk ratio (e.g., 2.0 means avg win is 2x avg loss)
        fraction: Kelly fraction (default: 0.25 = Quarter Kelly)

    Returns:
        Optimal position size as fraction of capital (0 to MAX_POSITION_PCT/100)
    """
    w = win_rate / 100.0
    r = rr_ratio

    if r <= 0 or w <= 0 or w >= 1:
        return 0.0

    # Kelly formula: f* = (b*p - q) / b where b=rr_ratio, p=win_rate, q=1-win_rate
    kelly = (r * w - (1 - w)) / r

    # Apply fraction (Quarter Kelly)
    kelly *= fraction

    # Clamp between 0 and max
    kelly = max(0.0, min(kelly, MAX_POSITION_PCT / 100.0))

    return kelly


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    direction: str,
    multiplier: float = ATR_MULTIPLIER,
) -> float:
    """Calculate stop-loss price based on ATR.

    Args:
        entry_price: Current/entry price
        atr: Average True Range value
        direction: "long" or "short"
        multiplier: ATR multiplier (default: 2.0)

    Returns:
        Stop-loss price
    """
    distance = atr * multiplier

    if "long" in direction:
        return entry_price - distance
    elif "short" in direction:
        return entry_price + distance
    else:
        return entry_price  # Neutral — no stop-loss


def calculate_take_profit(
    entry_price: float,
    atr: float,
    direction: str,
    rr_ratio: float = DEFAULT_RR_RATIO,
    multiplier: float = ATR_MULTIPLIER,
) -> float:
    """Calculate take-profit price based on ATR and R/R ratio.

    Take-profit = entry ± (ATR * multiplier * rr_ratio)
    """
    distance = atr * multiplier * rr_ratio

    if "long" in direction:
        return entry_price + distance
    elif "short" in direction:
        return entry_price - distance
    else:
        return entry_price


def calculate_risk(
    signal: dict,
    current_price: float,
    atr: float,
    capital: float = DEFAULT_CAPITAL,
    win_rate: float = DEFAULT_WIN_RATE,
    rr_ratio: float = DEFAULT_RR_RATIO,
    max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
) -> dict:
    """Full risk calculation for a trade signal.

    Args:
        signal: Signal dict from generate_signal (needs 'direction')
        current_price: Current asset price
        atr: ATR value
        capital: Total trading capital
        win_rate: Historical win rate (0-100)
        rr_ratio: Average reward/risk ratio
        max_risk_pct: Maximum risk per trade in %

    Returns:
        Dict with all risk metrics
    """
    direction = signal.get("direction", "neutral")

    if direction == "neutral" or current_price <= 0 or atr <= 0:
        return {
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": 0,
            "take_profit": 0,
            "risk_per_share": 0,
            "position_size_pct": 0,
            "position_size_units": 0,
            "position_size_value": 0,
            "risk_amount": 0,
            "reward_amount": 0,
            "risk_pct_of_capital": 0,
            "kelly_raw": 0,
            "kelly_fraction": KELLY_FRACTION,
            "message": "Kein Trade — Signal ist neutral.",
        }

    # Stop-Loss & Take-Profit
    stop_loss = calculate_stop_loss(current_price, atr, direction)
    take_profit = calculate_take_profit(current_price, atr, direction, rr_ratio)

    # Risk per share
    risk_per_share = abs(current_price - stop_loss)
    if risk_per_share == 0:
        risk_per_share = atr  # Fallback

    # Kelly position sizing
    kelly_pct = kelly_criterion(win_rate, rr_ratio)

    # Also calculate max-risk-based position size
    max_risk_amount = capital * (max_risk_pct / 100.0)
    risk_based_units = max_risk_amount / risk_per_share if risk_per_share > 0 else 0

    # Kelly-based position size
    kelly_value = capital * kelly_pct
    kelly_units = kelly_value / current_price if current_price > 0 else 0

    # Use the MORE CONSERVATIVE of the two
    position_units = min(risk_based_units, kelly_units)
    position_units = max(0, int(position_units))  # Round down to whole units

    position_value = position_units * current_price
    risk_amount = position_units * risk_per_share
    reward_amount = position_units * abs(take_profit - current_price)

    # Safety check: never exceed MAX_POSITION_PCT
    max_position_value = capital * (MAX_POSITION_PCT / 100.0)
    if position_value > max_position_value:
        position_units = int(max_position_value / current_price)
        position_value = position_units * current_price
        risk_amount = position_units * risk_per_share
        reward_amount = position_units * abs(take_profit - current_price)

    risk_pct = (risk_amount / capital * 100) if capital > 0 else 0

    return {
        "direction": direction,
        "entry_price": round(current_price, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_per_share": round(risk_per_share, 2),
        "position_size_pct": round(kelly_pct * 100, 2),
        "position_size_units": position_units,
        "position_size_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(reward_amount, 2),
        "risk_pct_of_capital": round(risk_pct, 2),
        "kelly_raw": round(kelly_pct * 100 / KELLY_FRACTION, 2) if KELLY_FRACTION > 0 else 0,
        "kelly_fraction": KELLY_FRACTION,
        "message": _build_message(direction, position_units, position_value, risk_amount, capital),
    }


def _build_message(direction: str, units: int, value: float, risk: float, capital: float) -> str:
    """Build human-readable position sizing message."""
    if units <= 0:
        return "Kelly Criterion empfiehlt keinen Trade bei diesen Parametern."

    dir_label = "LONG" if "long" in direction else "SHORT"
    return (
        f"📊 {dir_label}: {units} Stück für {value:,.2f} €\n"
        f"⚠️ Wenn der Trade scheitert, verlierst du {risk:,.2f} € "
        f"({risk/capital*100:.1f}% deines Kapitals)"
    )


if __name__ == "__main__":
    # Quick test
    test_signal = {"direction": "lean_long", "score": 30}
    risk = calculate_risk(
        signal=test_signal,
        current_price=150.0,
        atr=3.5,
        capital=10000,
        win_rate=55,
        rr_ratio=2.0,
    )
    for k, v in risk.items():
        print(f"  {k}: {v}")
