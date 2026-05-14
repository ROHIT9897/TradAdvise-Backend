# data/live_price.py
import httpx
import logging
import asyncio
import os
from cache.redis_client import cache
from config import settings
from data.fetcher import format_ticker, IS_CLOUD

logger = logging.getLogger(__name__)

async def get_live_price(ticker: str) -> dict:
    cache_key = f"live_price:{ticker}"

    cached = await cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Try multiple methods
    result = None

    if IS_CLOUD:
        # On cloud — use Alpha Vantage quote or Stooq
        result = await _fetch_price_alpha_vantage(ticker)
        if not result:
            result = await _fetch_price_from_history(ticker)
    else:
        # Local — use yfinance fast_info
        result = await _fetch_price_yfinance(ticker)
        if not result:
            result = await _fetch_price_from_history(ticker)

    if not result:
        raise Exception(f"Could not fetch live price for {ticker}")

    await cache.set(cache_key, result, settings.LIVE_PRICE_TTL)
    await cache.set(f"live_price_stale:{ticker}", result, 3600)
    return result

async def _fetch_price_yfinance(ticker: str) -> dict:
    try:
        import yfinance as yf
        formatted = format_ticker(ticker)

        def _fetch():
            stock    = yf.Ticker(formatted)
            info     = stock.fast_info
            price    = info.last_price
            prev     = info.previous_close
            if not price:
                return None
            change     = price - prev
            change_pct = (change / prev) * 100
            return {
                "ticker":         ticker,
                "price":          round(float(price), 2),
                "previous_close": round(float(prev), 2),
                "change":         round(float(change), 2),
                "change_pct":     round(float(change_pct), 2),
                "day_high":       round(float(info.day_high or price), 2),
                "day_low":        round(float(info.day_low or price), 2),
                "volume":         int(info.shares or 0),
                "cached":         False
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.warning(f"yfinance price failed for {ticker}: {e}")
        return None

async def _fetch_price_alpha_vantage(ticker: str) -> dict:
    try:
        if not settings.ALPHA_VANTAGE_KEY:
            return None

        clean = ticker.replace(".NS", "").replace(".BO", "")
        url   = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol":   f"NSE:{clean}",
            "apikey":   settings.ALPHA_VANTAGE_KEY
        }

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            data     = response.json()

        quote = data.get("Global Quote", {})
        if not quote or not quote.get("05. price"):
            return None

        price      = float(quote["05. price"])
        prev       = float(quote["08. previous close"])
        change     = float(quote["09. change"])
        change_pct = float(quote["10. change percent"].replace("%", ""))

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "day_high":       round(float(quote.get("03. high", price)), 2),
            "day_low":        round(float(quote.get("04. low", price)), 2),
            "volume":         int(quote.get("06. volume", 0)),
            "cached":         False
        }
    except Exception as e:
        logger.warning(f"Alpha Vantage price failed for {ticker}: {e}")
        return None

async def _fetch_price_from_history(ticker: str) -> dict:
    """
    Fallback — get latest price from historical data.
    Not real-time but better than nothing.
    """
    try:
        from data.fetcher import get_historical_data
        df = await get_historical_data(ticker, period="5d", use_cache=False)
        if df.empty:
            return None

        latest   = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        price    = float(latest["close"])
        prev     = float(previous["close"])
        change   = price - prev
        change_pct = (change / prev) * 100

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "day_high":       round(float(latest["high"]), 2),
            "day_low":        round(float(latest["low"]), 2),
            "volume":         int(latest["volume"]),
            "cached":         False,
            "note":           "Delayed data"
        }
    except Exception as e:
        logger.warning(f"History price fallback failed for {ticker}: {e}")
        return None

async def get_top_gainers_losers() -> dict:
    NIFTY50 = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "BHARTIARTL", "WIPRO", "LT", "AXISBANK",
        "KOTAKBANK", "HINDUNILVR", "ITC", "SUNPHARMA", "MARUTI",
        "BAJFINANCE", "TITAN", "ASIANPAINT", "NESTLEIND", "PNB"
    ]

    cache_key = "top_gainers_losers"
    cached    = await cache.get(cache_key)
    if cached:
        return cached

    results = []
    for ticker in NIFTY50:
        try:
            data = await get_live_price(ticker)
            results.append(data)
        except:
            continue

    results.sort(key=lambda x: x.get("change_pct", 0))

    output = {
        "top_gainers": results[-5:][::-1],
        "top_losers":  results[:5],
    }

    await cache.set(cache_key, output, 300)
    return output