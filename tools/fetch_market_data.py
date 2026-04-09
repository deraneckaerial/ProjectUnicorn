"""Fetch OHLCV market data via yfinance.

Usage:
    from tools.fetch_market_data import fetch_ohlcv
    df = fetch_ohlcv("AAPL", period="1y", interval="1d")
"""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for a given ticker.

    Args:
        ticker: Stock/crypto symbol (e.g., "AAPL", "BTC-USD")
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex

    Raises:
        ValueError: If ticker not found or no data returned.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for '{ticker}': {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check if the symbol is correct."
        )

    # Keep only OHLCV columns, drop Dividends/Stock Splits if present
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in ohlcv_cols if c in df.columns]
    df = df[available].copy()

    # Drop rows with NaN
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"Data for '{ticker}' is empty after cleaning.")

    return df


def fetch_multi_timeframe(
    ticker: str,
) -> dict[str, pd.DataFrame]:
    """Fetch daily and weekly data for multi-timeframe analysis.

    Returns:
        Dict with keys 'daily' and 'weekly', each containing OHLCV DataFrame.
    """
    daily = fetch_ohlcv(ticker, period="1y", interval="1d")
    weekly = fetch_ohlcv(ticker, period="2y", interval="1wk")
    return {"daily": daily, "weekly": weekly}


def get_ticker_info(ticker: str) -> dict:
    """Get basic info about a ticker (name, currency, exchange)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName") or info.get("shortName", ticker),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "Unknown"),
            "type": info.get("quoteType", "Unknown"),
        }
    except Exception:
        return {"name": ticker, "currency": "USD", "exchange": "Unknown", "type": "Unknown"}


if __name__ == "__main__":
    # Quick test
    df = fetch_ohlcv("AAPL", period="3mo")
    print(f"Fetched {len(df)} rows for AAPL")
    print(df.tail())
