import yfinance as yf
import pandas as pd
import numpy as np
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional
from cache.redis_client import cache
from config import settings
from datetime import timezone
import pytz
import asyncio



logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# TICKER UTILITIES
# ─────────────────────────────────────────

def format_indian_ticker(ticker: str) -> str:
    """
    Convert plain ticker to yfinance format.
    RELIANCE → RELIANCE.NS
    RELIANCE.NS → RELIANCE.NS (no change)
    """
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker

def is_market_open() -> bool:
    """
    NSE market hours: 9:15 AM to 3:30 PM IST, Mon-Fri
    """

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    market_open  = now.replace(hour=9,  minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)

    return market_open <= now <= market_close

# ─────────────────────────────────────────
# SOURCE 1 — yfinance (primary, free)
# ─────────────────────────────────────────
async def fetch_from_yfinance(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
) -> pd.DataFrame:
    formatted = format_indian_ticker(ticker)
    logger.info(f"Fetching {formatted} from yfinance")

    import asyncio
    loop = asyncio.get_event_loop()

    def _fetch():
        stock = yf.Ticker(formatted)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {formatted}")

        # Lowercase all columns
        df.columns = [c.lower() for c in df.columns]

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        # DROP extra columns — keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Remove timezone from index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna()
        logger.info(f"Fetched {len(df)} rows for {formatted}")
        return df

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)
# ─────────────────────────────────────────
# MAIN FETCHER with fallback chain
# ─────────────────────────────────────────

async def get_historical_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True
) -> pd.DataFrame:

    cache_key = f"historical:{ticker}:{period}:{interval}"

    if use_cache:
        cached = await cache.get(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

    sources = [
    ("yfinance", lambda: fetch_from_yfinance(ticker, period, interval)),
    ]

    last_error = None
    for source_name, fetch_fn in sources:
        try:
            df = await fetch_fn()
            logger.info(f"Successfully fetched {ticker} from {source_name}")

            if use_cache:
                # ── FIX: convert index to string before caching ──
                df_to_cache = df.copy()
                df_to_cache.index = df_to_cache.index.astype(str)
                await cache.set(
                    cache_key,
                    df_to_cache.to_dict(),
                    settings.HISTORICAL_TTL
                )

            return df

        except Exception as e:
            logger.warning(f"{source_name} failed for {ticker}: {e}")
            last_error = e
            continue

    raise Exception(f"All data sources failed for {ticker}. Last error: {last_error}")