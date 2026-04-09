"""Calculate technical indicators using pandas-ta.

All parameters are FIXED to prevent overfitting. See config.py for values.

Usage:
    from tools.calculate_indicators import calculate_all_indicators
    df_with_indicators = calculate_all_indicators(ohlcv_df)
"""

import pandas as pd
import pandas_ta as ta
import sys
import os

# Add parent dir to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    DMI_LENGTH, ADX_LENGTH, RSI_LENGTH,
    ATR_LENGTH, EMA_SHORT, EMA_MID, EMA_LONG,
    BB_LENGTH, BB_STD,
)


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators on OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with original OHLCV + all indicator columns.
    """
    df = df.copy()

    # === TREND INDICATORS ===

    # MACD (12, 26, 9)
    macd = df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # DMI / ADX (14)
    dmi = df.ta.dm(length=DMI_LENGTH)
    if dmi is not None:
        df = pd.concat([df, dmi], axis=1)

    adx = df.ta.adx(length=ADX_LENGTH)
    if adx is not None:
        # Avoid duplicate columns
        for col in adx.columns:
            if col not in df.columns:
                df[col] = adx[col]

    # EMA (20, 50, 200) — safe assignment handles insufficient data
    for ema_len in [EMA_SHORT, EMA_MID, EMA_LONG]:
        col_name = f"EMA_{ema_len}"
        if len(df) >= ema_len:
            result = df.ta.ema(length=ema_len)
            if isinstance(result, pd.Series):
                df[col_name] = result
            elif isinstance(result, pd.DataFrame) and not result.empty:
                df[col_name] = result.iloc[:, 0]
            else:
                df[col_name] = pd.NA
        else:
            df[col_name] = pd.NA

    # === MOMENTUM INDICATORS ===

    # RSI (14)
    rsi_result = df.ta.rsi(length=RSI_LENGTH)
    df[f"RSI_{RSI_LENGTH}"] = rsi_result.iloc[:, 0] if isinstance(rsi_result, pd.DataFrame) else rsi_result

    # KST (Know Sure Thing)
    kst = df.ta.kst()
    if kst is not None:
        df = pd.concat([df, kst], axis=1)

    # === VOLATILITY ===

    # ATR (14)
    atr_result = df.ta.atr(length=ATR_LENGTH)
    df[f"ATR_{ATR_LENGTH}"] = atr_result.iloc[:, 0] if isinstance(atr_result, pd.DataFrame) else atr_result

    # Bollinger Bands (20, 2)
    bb = df.ta.bbands(length=BB_LENGTH, std=BB_STD)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)

    # === VOLUME ===

    # OBV (On Balance Volume)
    obv_result = df.ta.obv()
    df["OBV"] = obv_result.iloc[:, 0] if isinstance(obv_result, pd.DataFrame) else obv_result

    # === DERIVED SIGNALS ===

    # MACD direction (rising/falling)
    macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macd_signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macd_hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    if macd_col in df.columns:
        df["MACD_bullish"] = (df[macd_col] > df[macd_signal_col]).astype(int)
        df["MACD_rising"] = (df[macd_hist_col] > df[macd_hist_col].shift(1)).astype(int)

    # EMA alignment (bullish: price > EMA20 > EMA50 > EMA200)
    ema_s = f"EMA_{EMA_SHORT}"
    ema_m = f"EMA_{EMA_MID}"
    ema_l = f"EMA_{EMA_LONG}"
    if all(c in df.columns for c in [ema_s, ema_m, ema_l]):
        # fillna to avoid NA boolean comparison errors (e.g. EMA_200 with <200 rows)
        es = pd.to_numeric(df[ema_s], errors="coerce").fillna(0)
        em = pd.to_numeric(df[ema_m], errors="coerce").fillna(0)
        el = pd.to_numeric(df[ema_l], errors="coerce").fillna(0)
        has_all = df[ema_s].notna() & df[ema_m].notna() & df[ema_l].notna()
        df["EMA_bullish"] = (
            has_all &
            (df["Close"] > es) &
            (es > em) &
            (em > el)
        ).astype(int)
        df["EMA_bearish"] = (
            has_all &
            (df["Close"] < es) &
            (es < em) &
            (em < el)
        ).astype(int)

    # OBV direction
    if "OBV" in df.columns:
        df["OBV_rising"] = (df["OBV"] > df["OBV"].shift(5)).astype(int)

    return df


def get_latest_values(df: pd.DataFrame) -> dict:
    """Extract the latest indicator values as a flat dictionary.

    Useful for the scoring model and UI display.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    result = {}

    for col in df.columns:
        val = latest[col]
        if pd.notna(val):
            result[col] = float(val) if not isinstance(val, str) else val

    return result


if __name__ == "__main__":
    from fetch_market_data import fetch_ohlcv

    df = fetch_ohlcv("AAPL", period="1y")
    df = calculate_all_indicators(df)
    print(f"Columns: {list(df.columns)}")
    print(f"\nLatest values:")
    for k, v in get_latest_values(df).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
