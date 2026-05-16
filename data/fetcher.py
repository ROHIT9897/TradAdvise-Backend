# data/fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import httpx
import logging
import asyncio
import pytz
from datetime import datetime
from cache.redis_client import cache
from config import settings

logger = logging.getLogger(__name__)

# ── Shared headers for NSE requests ──────────────────────
NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

# ── Ticker utilities ──────────────────────────────────────

def format_indian_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker

def is_market_open() -> bool:
    ist   = pytz.timezone("Asia/Kolkata")
    now   = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=15, second=0)
    close_t = now.replace(hour=15, minute=30, second=0)
    return open_t <= now <= close_t

# ── Source 1 — yfinance (primary) ────────────────────────

async def fetch_from_yfinance(
    ticker:   str,
    period:   str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    formatted = format_indian_ticker(ticker)
    logger.info(f"Fetching {formatted} from yfinance")

    def _fetch():
        stock = yf.Ticker(formatted)
        df    = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {formatted}")

        df.columns = [c.lower() for c in df.columns]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna()
        logger.info(f"Fetched {len(df)} rows for {formatted}")
        return df

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)

# ── Source 2 — Yahoo proxy (cloud fallback) ───────────────

async def fetch_from_yahoo_proxy(
    ticker: str,
    period: str = "2y"
) -> pd.DataFrame:
    clean     = ticker.replace(".NS", "").replace(".BO", "")
    yf_ticker = f"{clean}.NS"

    # Map all possible period strings
    period_map = {
        "1W":  ("5d",  "1d"),
        "1M":  ("1mo", "1d"),
        "3M":  ("3mo", "1d"),
        "6M":  ("6mo", "1d"),
        "1Y":  ("1y",  "1d"),
        "1mo": ("1mo", "1d"),
        "3mo": ("3mo", "1d"),
        "6mo": ("6mo", "1d"),
        "1y":  ("1y",  "1d"),
        "2y":  ("2y",  "1d"),
        "5y":  ("5y",  "1wk"),
        "7d":  ("5d",  "1d"),
        "5d":  ("5d",  "1d"),
    }
    yf_range, interval = period_map.get(period, ("2y", "1d"))

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
    params = {
        "range":    yf_range,
        "interval": interval,
        "events":   "history",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer":    "https://finance.yahoo.com/",
    }

    async with httpx.AsyncClient(
        headers=headers, timeout=30, follow_redirects=True
    ) as client:
        response = await client.get(url, params=params)

    if response.status_code != 200:
        raise ValueError(f"Yahoo proxy returned {response.status_code}")

    data   = response.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise ValueError(f"No data from Yahoo proxy for {yf_ticker}")

    r          = result[0]
    timestamps = r.get("timestamp", [])
    ohlcv      = r.get("indicators", {}).get("quote", [{}])[0]

    if not timestamps:
        raise ValueError(f"Empty timestamps for {yf_ticker}")

    df = pd.DataFrame({
        "date":   pd.to_datetime(timestamps, unit="s"),
        "open":   ohlcv.get("open",   []),
        "high":   ohlcv.get("high",   []),
        "low":    ohlcv.get("low",    []),
        "close":  ohlcv.get("close",  []),
        "volume": ohlcv.get("volume", []),
    })
    df = df.set_index("date")
    df.index = df.index.tz_localize(None)
    df = df.dropna()
    df = df.sort_index()

    logger.info(f"Yahoo proxy OK: {len(df)} rows for {yf_ticker}")
    return df

# ── Main fetcher with fallback chain ──────────────────────

async def get_historical_data(
    ticker:    str,
    period:    str  = "2y",
    interval:  str  = "1d",
    use_cache: bool = True
) -> pd.DataFrame:

    cache_key = f"historical:{ticker}:{period}:{interval}"

    # Check cache
    if use_cache:
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for {ticker}")
            df       = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

    # Try sources in order
    sources = [
        ("yfinance",     lambda: fetch_from_yfinance(ticker, period, interval)),
        ("yahoo_proxy",  lambda: fetch_from_yahoo_proxy(ticker, period)),
    ]

    last_error = None
    for source_name, fetch_fn in sources:
        try:
            df = await fetch_fn()
            if df is not None and len(df) >= 5:
                logger.info(f"✓ {ticker} from {source_name}: {len(df)} rows")

                # Save to cache — convert index to string first
                if use_cache:
                    df_cache       = df.copy()
                    df_cache.index = df_cache.index.astype(str)
                    await cache.set(
                        cache_key,
                        df_cache.to_dict(),
                        settings.HISTORICAL_TTL
                    )
                return df

        except Exception as e:
            logger.warning(f"✗ {source_name} failed for {ticker}: {e}")
            last_error = e
            continue

    raise Exception(
        f"All data sources failed for {ticker}. "
        f"Last error: {last_error}"
    )