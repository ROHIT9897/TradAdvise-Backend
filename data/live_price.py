import yfinance as yf
import httpx
import logging
from cache.redis_client import cache
from config import settings
from data.fetcher import format_indian_ticker

logger = logging.getLogger(__name__)

async def get_live_price(ticker: str) -> dict:
    """
    Get current stock price with 30-second cache.
    Returns price, change, change_pct, volume, high, low
    """
    cache_key = f"live_price:{ticker}"

    # Check cache first (30 second TTL)
    cached = await cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Fetch fresh
    try:
        result = await _fetch_live_yfinance(ticker)
    except Exception as e:
        logger.warning(f"Live price fetch failed: {e}")
        # Return last cached value if available, even if expired
        stale = await cache.get(f"live_price_stale:{ticker}")
        if stale:
            return {**stale, "cached": True, "stale": True}
        raise

    # Store with short TTL
    await cache.set(cache_key, result, settings.LIVE_PRICE_TTL)

    # Also store stale backup (longer TTL, for fallback)
    await cache.set(f"live_price_stale:{ticker}", result, 3600)

    return result

async def _fetch_live_yfinance(ticker: str) -> dict:
    import asyncio

    formatted = format_indian_ticker(ticker)

    def _fetch():
        stock = yf.Ticker(formatted)
        info = stock.fast_info   # faster than .info

        price = info.last_price
        prev_close = info.previous_close

        if not price:
            raise ValueError(f"No price data for {formatted}")

        change = price - prev_close
        change_pct = (change / prev_close) * 100

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(info.day_high or 0, 2),
            "day_low": round(info.day_low or 0, 2),
            "volume": int(info.shares or 0),
            "market_cap": int(info.market_cap or 0),
            "cached": False
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)

async def get_top_gainers_losers() -> dict:
    """
    Fetch top gainers and losers from NSE.
    Uses yfinance for a predefined universe of stocks.
    In production, replace with NSE official API.
    """
    NIFTY50_TICKERS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
        "WIPRO", "ULTRACEMCO", "TITAN", "BAJFINANCE", "NESTLEIND"
    ]

    cache_key = "top_gainers_losers"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    results = []
    for ticker in NIFTY50_TICKERS:
        try:
            data = await get_live_price(ticker)
            results.append(data)
        except:
            continue

    results.sort(key=lambda x: x.get("change_pct", 0))

    output = {
        "top_gainers": results[-5:][::-1],   # Top 5 gainers
        "top_losers": results[:5],            # Top 5 losers
    }

    await cache.set(cache_key, output, 300)   # 5 minute cache
    return output