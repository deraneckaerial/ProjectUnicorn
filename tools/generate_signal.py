"""Multi-indicator scoring model to generate trading signals.

Scoring range: -100 to +100
  > 40:  Strong Long
  20-40: Lean Long
  -20-20: Neutral (no trade)
  -40--20: Lean Short
  < -40: Strong Short

All parameters are FIXED to prevent overfitting.

Usage:
    from tools.generate_signal import generate_signal
    signal = generate_signal(df_with_indicators)
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STRONG_LONG_THRESHOLD, LEAN_LONG_THRESHOLD,
    LEAN_SHORT_THRESHOLD, STRONG_SHORT_THRESHOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_LENGTH, RSI_LENGTH, ATR_LENGTH,
    EMA_SHORT, EMA_MID, EMA_LONG,
    BB_LENGTH, BB_STD,
)


def generate_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal from indicator DataFrame.

    Args:
        df: DataFrame with OHLCV + all indicator columns.

    Returns:
        Dict with keys:
            direction: "strong_long" | "lean_long" | "neutral" | "lean_short" | "strong_short"
            score: int (-100 to 100)
            confidence: int (0 to 100)
            reasons: list of bullish reasons
            contra_reasons: list of bearish reasons
            components: dict of individual score components
    """
    if df.empty or len(df) < 50:
        return {
            "direction": "neutral",
            "score": 0,
            "confidence": 0,
            "reasons": ["Insufficient data for analysis"],
            "contra_reasons": [],
            "components": {},
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    score = 0
    reasons = []
    contra_reasons = []
    components = {}

    # === TREND (Weight: 40%) ===
    trend_score = 0

    # MACD above signal line and rising (+20)
    macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macd_signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macd_hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    if all(c in df.columns for c in [macd_col, macd_signal_col, macd_hist_col]):
        macd_val = latest.get(macd_col, 0)
        macd_sig = latest.get(macd_signal_col, 0)
        macd_hist = latest.get(macd_hist_col, 0)
        macd_hist_prev = prev.get(macd_hist_col, 0)

        if pd.notna(macd_val) and pd.notna(macd_sig):
            if macd_val > macd_sig and macd_hist > macd_hist_prev:
                trend_score += 20
                reasons.append("MACD über Signal-Linie und steigend")
            elif macd_val > macd_sig:
                trend_score += 10
                reasons.append("MACD über Signal-Linie")
            elif macd_val < macd_sig and macd_hist < macd_hist_prev:
                trend_score -= 20
                contra_reasons.append("MACD unter Signal-Linie und fallend")
            elif macd_val < macd_sig:
                trend_score -= 10
                contra_reasons.append("MACD unter Signal-Linie")

    # ADX > 25 = strong trend (+10)
    adx_col = f"ADX_{ADX_LENGTH}"
    if adx_col in df.columns and pd.notna(latest.get(adx_col)):
        adx_val = latest[adx_col]
        if adx_val > 25:
            # ADX shows trend strength but not direction — add points in trend direction
            trend_score += 10 if trend_score > 0 else -10 if trend_score < 0 else 0
            reasons.append(f"Starker Trend (ADX: {adx_val:.1f})")
        else:
            contra_reasons.append(f"Schwacher Trend (ADX: {adx_val:.1f}) — Seitwärtsmarkt möglich")

    # EMA alignment (+10)
    if "EMA_bullish" in df.columns and pd.notna(latest.get("EMA_bullish")):
        if latest["EMA_bullish"] == 1:
            trend_score += 10
            reasons.append(f"Bullish EMA-Alignment (Kurs > EMA{EMA_SHORT} > EMA{EMA_MID} > EMA{EMA_LONG})")
        elif "EMA_bearish" in df.columns and latest.get("EMA_bearish") == 1:
            trend_score -= 10
            contra_reasons.append(f"Bearish EMA-Alignment (Kurs < EMA{EMA_SHORT} < EMA{EMA_MID} < EMA{EMA_LONG})")

    components["trend"] = trend_score
    score += trend_score

    # === MOMENTUM (Weight: 30%) ===
    momentum_score = 0

    # RSI
    rsi_col = f"RSI_{RSI_LENGTH}"
    if rsi_col in df.columns and pd.notna(latest.get(rsi_col)):
        rsi_val = latest[rsi_col]
        if rsi_val < 30:
            momentum_score += 15
            reasons.append(f"RSI überverkauft ({rsi_val:.1f}) — Erholungspotenzial")
        elif rsi_val > 70:
            momentum_score -= 15
            contra_reasons.append(f"RSI überkauft ({rsi_val:.1f}) — Korrekturrisiko")
        elif 40 <= rsi_val <= 60:
            # Neutral RSI in trend direction adds small points
            momentum_score += 5 if trend_score > 0 else -5 if trend_score < 0 else 0
            reasons.append(f"RSI neutral ({rsi_val:.1f})")

    # KST
    kst_col = [c for c in df.columns if c.startswith("KST_")]
    kst_sig_col = [c for c in df.columns if c.startswith("KSTs_")]
    if kst_col and kst_sig_col:
        kst_val = latest.get(kst_col[0])
        kst_sig = latest.get(kst_sig_col[0])
        if pd.notna(kst_val) and pd.notna(kst_sig):
            if kst_val > kst_sig:
                momentum_score += 15
                reasons.append("KST über Signal-Linie — langfristiges Momentum bullish")
            else:
                momentum_score -= 15
                contra_reasons.append("KST unter Signal-Linie — langfristiges Momentum bearish")

    components["momentum"] = momentum_score
    score += momentum_score

    # === VOLUME (Weight: 15%) ===
    volume_score = 0

    if "OBV_rising" in df.columns and pd.notna(latest.get("OBV_rising")):
        price_rising = latest["Close"] > prev["Close"]
        obv_rising = latest["OBV_rising"] == 1

        if price_rising and obv_rising:
            volume_score += 15
            reasons.append("Volumen bestätigt Aufwärtsbewegung (OBV steigend)")
        elif not price_rising and not obv_rising:
            volume_score -= 15
            contra_reasons.append("Volumen bestätigt Abwärtsbewegung (OBV fallend)")
        elif price_rising and not obv_rising:
            volume_score -= 5
            contra_reasons.append("Preis steigt, aber Volumen fällt — schwache Bewegung")
        elif not price_rising and obv_rising:
            volume_score += 5
            reasons.append("Preis fällt, aber Volumen steigt — mögliche Akkumulation")

    components["volume"] = volume_score
    score += volume_score

    # === VOLATILITY (Weight: 15%) ===
    volatility_score = 0

    bb_upper = f"BBU_{BB_LENGTH}_{BB_STD}"
    bb_lower = f"BBL_{BB_LENGTH}_{BB_STD}"
    bb_mid = f"BBM_{BB_LENGTH}_{BB_STD}"

    if all(c in df.columns for c in [bb_upper, bb_lower]):
        close = latest["Close"]
        upper = latest.get(bb_upper)
        lower = latest.get(bb_lower)

        if pd.notna(upper) and pd.notna(lower):
            if close > upper:
                adx_val = latest.get(adx_col, 0) if adx_col in df.columns else 0
                if pd.notna(adx_val) and adx_val > 30:
                    volatility_score -= 10
                    contra_reasons.append("Kurs über oberem Bollinger Band bei starkem Trend — Erschöpfung möglich")
                else:
                    volatility_score -= 15
                    contra_reasons.append("Kurs über oberem Bollinger Band — überdehnt")
            elif close < lower:
                volatility_score += 10
                reasons.append("Kurs unter unterem Bollinger Band — Mean-Reversion möglich")
            else:
                volatility_score += 5
                # Within bands is neutral-positive

    components["volatility"] = volatility_score
    score += volatility_score

    # === DETERMINE DIRECTION ===
    if score > STRONG_LONG_THRESHOLD:
        direction = "strong_long"
    elif score > LEAN_LONG_THRESHOLD:
        direction = "lean_long"
    elif score < STRONG_SHORT_THRESHOLD:
        direction = "strong_short"
    elif score < LEAN_SHORT_THRESHOLD:
        direction = "lean_short"
    else:
        direction = "neutral"

    # Confidence: abs(score) mapped to 0-100
    confidence = min(abs(score), 100)

    return {
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "contra_reasons": contra_reasons,
        "components": components,
    }


def get_direction_label(direction: str) -> tuple[str, str]:
    """Return human-readable label and color for a direction.

    Returns:
        Tuple of (label, color)
    """
    labels = {
        "strong_long": ("⬆️ STRONG LONG", "#00C853"),
        "lean_long": ("↗️ Lean Long", "#69F0AE"),
        "neutral": ("➡️ Neutral", "#FFD54F"),
        "lean_short": ("↘️ Lean Short", "#FF8A80"),
        "strong_short": ("⬇️ STRONG SHORT", "#FF1744"),
    }
    return labels.get(direction, ("❓ Unknown", "#9E9E9E"))


if __name__ == "__main__":
    from fetch_market_data import fetch_ohlcv
    from calculate_indicators import calculate_all_indicators

    df = fetch_ohlcv("AAPL", period="1y")
    df = calculate_all_indicators(df)
    signal = generate_signal(df)

    label, color = get_direction_label(signal["direction"])
    print(f"\n{'='*50}")
    print(f"AAPL Signal: {label}")
    print(f"Score: {signal['score']} | Confidence: {signal['confidence']}%")
    print(f"\nBullish reasons:")
    for r in signal["reasons"]:
        print(f"  ✅ {r}")
    print(f"\nBearish reasons:")
    for r in signal["contra_reasons"]:
        print(f"  ❌ {r}")
    print(f"\nComponents: {signal['components']}")
