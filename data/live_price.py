# data/live_price.py
import httpx
import logging
import asyncio
from cache.redis_client import cache
from config import settings
from data.fetcher import NSE_HEADERS

logger = logging.getLogger(__name__)

async def get_live_price(ticker: str) -> dict:
    cache_key = f"live_price:{ticker}"

    cached = await cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Try sources in order
    result = (
        await _price_from_nse_quote(ticker) or
        await _price_from_yahoo_proxy(ticker) or
        await _price_from_history(ticker)
    )

    if not result:
        raise Exception(f"All price sources failed for {ticker}")

    await cache.set(cache_key, result, settings.LIVE_PRICE_TTL)
    await cache.set(f"live_price_stale:{ticker}", result, 3600)
    return result

async def _price_from_nse_quote(ticker: str) -> dict:
    """NSE India real-time quote — best source for Indian stocks."""
    try:
        clean = ticker.replace(".NS", "").replace(".BO", "").upper()
        url   = f"https://www.nseindia.com/api/quote-equity?symbol={clean}"

        async with httpx.AsyncClient(
            headers          = NSE_HEADERS,
            timeout          = 15,
            follow_redirects = True
        ) as client:
            await client.get("https://www.nseindia.com", timeout=10)
            response = await client.get(url)

        if response.status_code != 200:
            return None

        data  = response.json()
        price_data = data.get("priceInfo", {})

        price      = float(price_data.get("lastPrice", 0))
        prev_close = float(price_data.get("previousClose", price))
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        if not price:
            return None

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev_close, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "day_high":       round(float(price_data.get("intraDayHighLow", {}).get("max", price)), 2),
            "day_low":        round(float(price_data.get("intraDayHighLow", {}).get("min", price)), 2),
            "volume":         int(data.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("totalTradedVolume", 0)),
            "cached":         False
        }

    except Exception as e:
        logger.warning(f"NSE quote failed for {ticker}: {e}")
        return None

async def _price_from_yahoo_proxy(ticker: str) -> dict:
    """Yahoo Finance API direct — works without yfinance library."""
    try:
        clean     = ticker.replace(".NS", "").replace(".BO", "")
        yf_ticker = f"{clean}.NS"
        url       = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer":    "https://finance.yahoo.com/",
        }

        async with httpx.AsyncClient(headers=headers, timeout=15) as client:
            response = await client.get(url, params={"range": "1d", "interval": "1m"})

        data   = response.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        meta       = result[0].get("meta", {})
        price      = float(meta.get("regularMarketPrice", 0))
        prev_close = float(meta.get("chartPreviousClose", price))
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        if not price:
            return None

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev_close, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "day_high":       round(float(meta.get("regularMarketDayHigh", price)), 2),
            "day_low":        round(float(meta.get("regularMarketDayLow", price)), 2),
            "volume":         int(meta.get("regularMarketVolume", 0)),
            "cached":         False
        }

    except Exception as e:
        logger.warning(f"Yahoo proxy price failed for {ticker}: {e}")
        return None

async def _price_from_history(ticker: str) -> dict:
    """Fallback — get latest close from historical data."""
    try:
        from data.fetcher import get_historical_data
        df = await get_historical_data(ticker, period="7d", use_cache=False)
        if df.empty:
            return None

        latest   = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        price    = float(latest["close"])
        prev     = float(previous["close"])
        change   = price - prev
        pct      = (change / prev * 100) if prev else 0

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev, 2),
            "change":         round(change, 2),
            "change_pct":     round(pct, 2),
            "day_high":       round(float(latest["high"]), 2),
            "day_low":        round(float(latest["low"]), 2),
            "volume":         int(latest["volume"]),
            "cached":         False,
            "note":           "Delayed — live price unavailable"
        }
    except Exception as e:
        logger.warning(f"History fallback failed for {ticker}: {e}")
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
    for t in NIFTY50:
        try:
            results.append(await get_live_price(t))
        except:
            continue

    results.sort(key=lambda x: x.get("change_pct", 0))

    output = {
        "top_gainers": results[-5:][::-1],
        "top_losers":  results[:5],
    }
    await cache.set(cache_key, output, 300)
    return output