# data/live_price.py
import httpx
import logging
import asyncio
from cache.redis_client import cache
from config import settings

logger = logging.getLogger(__name__)

# ── Live price — main entry point ─────────────────────────

async def get_live_price(ticker: str) -> dict:
    cache_key = f"live_price:{ticker}"

    cached = await cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Try sources in order — first success wins
    result = None

    result = await _price_from_yahoo_proxy(ticker)
    if not result:
        result = await _price_from_yfinance(ticker)
    if not result:
        result = await _price_from_history(ticker)

    if not result:
        raise Exception(f"All price sources failed for {ticker}")

    await cache.set(cache_key, result, settings.LIVE_PRICE_TTL)
    await cache.set(f"live_price_stale:{ticker}", result, 3600)
    return result

# ── Source 1 — Yahoo Finance direct API ──────────────────

async def _price_from_yahoo_proxy(ticker: str) -> dict:
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
            response = await client.get(
                url,
                params={"range": "1d", "interval": "1m"}
            )

        if response.status_code != 200:
            return None

        data   = response.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        meta       = result[0].get("meta", {})
        price      = float(meta.get("regularMarketPrice", 0))
        prev_close = float(meta.get("chartPreviousClose", price))

        if not price:
            return None

        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "ticker":         ticker,
            "price":          round(price, 2),
            "previous_close": round(prev_close, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "day_high":       round(float(meta.get("regularMarketDayHigh", price)), 2),
            "day_low":        round(float(meta.get("regularMarketDayLow",  price)), 2),
            "volume":         int(meta.get("regularMarketVolume", 0)),
            "cached":         False
        }

    except Exception as e:
        logger.warning(f"Yahoo proxy price failed for {ticker}: {e}")
        return None

# ── Source 2 — yfinance fast_info ────────────────────────

async def _price_from_yfinance(ticker: str) -> dict:
    try:
        import yfinance as yf
        from data.fetcher import format_indian_ticker
        formatted = format_indian_ticker(ticker)

        def _fetch():
            stock = yf.Ticker(formatted)
            info  = stock.fast_info
            price = info.last_price
            prev  = info.previous_close
            if not price:
                return None
            change     = price - prev
            change_pct = (change / prev * 100) if prev else 0
            return {
                "ticker":         ticker,
                "price":          round(float(price), 2),
                "previous_close": round(float(prev),  2),
                "change":         round(float(change), 2),
                "change_pct":     round(float(change_pct), 2),
                "day_high":       round(float(info.day_high or price), 2),
                "day_low":        round(float(info.day_low  or price), 2),
                "volume":         int(info.shares or 0),
                "cached":         False
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)

    except Exception as e:
        logger.warning(f"yfinance price failed for {ticker}: {e}")
        return None

# ── Source 3 — fallback from historical data ─────────────

async def _price_from_history(ticker: str) -> dict:
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
            "day_low":        round(float(latest["low"]),  2),
            "volume":         int(latest["volume"]),
            "cached":         False,
            "note":           "Delayed data"
        }

    except Exception as e:
        logger.warning(f"History price fallback failed for {ticker}: {e}")
        return None

# ── Top gainers and losers ────────────────────────────────

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
            data = await get_live_price(t)
            results.append(data)
        except Exception as e:
            logger.warning(f"Skipping {t}: {e}")
            continue

    if not results:
        return {"top_gainers": [], "top_losers": []}

    results.sort(key=lambda x: x.get("change_pct", 0))

    output = {
        "top_gainers": results[-5:][::-1],
        "top_losers":  results[:5],
    }

    await cache.set(cache_key, output, 300)
    return output