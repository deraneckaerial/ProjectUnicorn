"""Fetch news sentiment data from Finnhub API.

Provides market sentiment scores from recent news articles
to complement technical analysis with fundamental context.

Usage:
    from tools.fetch_sentiment import fetch_news_sentiment
    sentiment = fetch_news_sentiment("AAPL")
"""

import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _get_finnhub_client():
    """Get Finnhub client with API key from .env or environment."""
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FINNHUB_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        return None
    if HAS_FINNHUB:
        return finnhub.Client(api_key=api_key)
    return api_key


def fetch_news_sentiment(ticker: str, days_back: int = 7) -> dict:
    """Fetch news sentiment for a ticker.

    Args:
        ticker: Stock ticker (e.g., "AAPL"). For crypto, use base symbol.
        days_back: Number of days to look back for news.

    Returns:
        Dict with:
            score: float (-1 to 1, negative=bearish, positive=bullish)
            label: str ("Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish")
            article_count: int
            buzz: float (social media buzz score)
            articles: list of dicts with title, source, sentiment, datetime
            available: bool (whether sentiment data was found)
    """
    client = _get_finnhub_client()

    if client is None:
        return _empty_sentiment("Kein Finnhub API Key konfiguriert")

    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    try:
        if HAS_FINNHUB and hasattr(client, "company_news"):
            # Use finnhub SDK
            news = client.company_news(ticker, _from=from_date, to=to_date)
            sentiment_data = client.news_sentiment(ticker)
        elif HAS_REQUESTS:
            # Fallback to direct API calls
            base_url = "https://finnhub.io/api/v1"
            headers = {"X-Finnhub-Token": client}

            news_resp = requests.get(
                f"{base_url}/company-news",
                params={"symbol": ticker, "from": from_date, "to": to_date},
                headers=headers,
                timeout=10,
            )
            news = news_resp.json() if news_resp.status_code == 200 else []

            sent_resp = requests.get(
                f"{base_url}/news-sentiment",
                params={"symbol": ticker},
                headers=headers,
                timeout=10,
            )
            sentiment_data = sent_resp.json() if sent_resp.status_code == 200 else {}
        else:
            return _empty_sentiment("Weder finnhub noch requests installiert")

    except Exception as e:
        return _empty_sentiment(f"API-Fehler: {str(e)}")

    # Parse sentiment data
    sentiment_obj = sentiment_data.get("sentiment", {}) if isinstance(sentiment_data, dict) else {}
    buzz_obj = sentiment_data.get("buzz", {}) if isinstance(sentiment_data, dict) else {}

    # Calculate aggregate sentiment
    bullish = sentiment_obj.get("bullishPercent", 0.5)
    bearish = sentiment_obj.get("bearishPercent", 0.5)
    score = bullish - bearish  # Range: -1 to 1

    # Parse articles
    articles = []
    if isinstance(news, list):
        for article in news[:20]:  # Limit to 20 most recent
            articles.append({
                "title": article.get("headline", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "summary": article.get("summary", "")[:200],
            })

    label = _score_to_label(score)
    buzz_score = buzz_obj.get("buzz", 0)
    article_count = buzz_obj.get("articlesInLastWeek", len(articles))

    return {
        "score": round(score, 3),
        "label": label,
        "article_count": article_count,
        "buzz": round(buzz_score, 2),
        "articles": articles,
        "available": True,
        "bullish_pct": round(bullish * 100, 1),
        "bearish_pct": round(bearish * 100, 1),
    }


def _score_to_label(score: float) -> str:
    """Convert sentiment score to label."""
    if score > 0.4:
        return "Very Bullish"
    elif score > 0.15:
        return "Bullish"
    elif score > -0.15:
        return "Neutral"
    elif score > -0.4:
        return "Bearish"
    else:
        return "Very Bearish"


def _empty_sentiment(reason: str) -> dict:
    """Return empty sentiment result."""
    return {
        "score": 0,
        "label": "Nicht verfuegbar",
        "article_count": 0,
        "buzz": 0,
        "articles": [],
        "available": False,
        "bullish_pct": 50,
        "bearish_pct": 50,
        "reason": reason,
    }


def get_sentiment_color(score: float) -> str:
    """Get color for sentiment score."""
    if score > 0.15:
        return "#00C853"
    elif score > -0.15:
        return "#FFD54F"
    else:
        return "#FF1744"


if __name__ == "__main__":
    result = fetch_news_sentiment("AAPL")
    print(f"Sentiment: {result['label']} (Score: {result['score']})")
    print(f"Articles: {result['article_count']}")
    print(f"Bullish: {result.get('bullish_pct', 'N/A')}% | Bearish: {result.get('bearish_pct', 'N/A')}%")
    if result["articles"]:
        print(f"\nLatest headlines:")
        for a in result["articles"][:5]:
            print(f"  - [{a['source']}] {a['title']}")
