# data/fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import httpx
import logging
import os
import asyncio
from typing import Optional
from cache.redis_client import cache
from config import settings

logger = logging.getLogger(__name__)

# Detect if running on cloud server
IS_CLOUD = os.environ.get("RENDER", False) or os.environ.get("RAILWAY", False)

def format_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker

def is_market_open() -> bool:
    import pytz
    from datetime import datetime
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=15, second=0)
    close_time = now.replace(hour=15, minute=30, second=0)
    return open_time <= now <= close_time

# ── SOURCE 1 — Alpha Vantage (works on cloud) ─────────────

async def fetch_from_alpha_vantage(ticker: str) -> pd.DataFrame:
    if not settings.ALPHA_VANTAGE_KEY:
        raise ValueError("Alpha Vantage key not set")

    clean = ticker.replace(".NS", "").replace(".BO", "")

    # Try BSE first for Indian stocks, then NSE format
    for symbol in [f"BSE:{clean}", f"NSE:{clean}", clean]:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":   symbol,
                "outputsize": "full",
                "apikey":   settings.ALPHA_VANTAGE_KEY
            }

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                data     = response.json()

            if "Time Series (Daily)" not in data:
                error_msg = data.get("Note") or data.get("Information") or "No data"
                logger.warning(f"Alpha Vantage {symbol}: {error_msg}")
                continue

            ts = data["Time Series (Daily)"]
            df = pd.DataFrame(ts).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                "1. open":             "open",
                "2. high":             "high",
                "3. low":              "low",
                "4. close":            "close",
                "5. adjusted close":   "adj_close",
                "6. volume":           "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.dropna()

            logger.info(f"Alpha Vantage OK for {symbol}: {len(df)} rows")
            return df

        except Exception as e:
            logger.warning(f"Alpha Vantage {symbol} failed: {e}")
            continue

    raise ValueError(f"Alpha Vantage failed for all symbol formats of {ticker}")

# ── SOURCE 2 — yfinance with headers (works locally) ──────

async def fetch_from_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    formatted = format_ticker(ticker)
    logger.info(f"Fetching {formatted} from yfinance")

    def _fetch():
        # Add headers to avoid Yahoo blocking
        stock = yf.Ticker(formatted)

        # Try with explicit headers
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        stock = yf.Ticker(formatted, session=session)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {formatted}")

        df.columns = [c.lower() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna()
        logger.info(f"yfinance OK: {len(df)} rows for {formatted}")
        return df

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)

# ── SOURCE 3 — Stooq (free, no API key, works on cloud) ───

async def fetch_from_stooq(ticker: str) -> pd.DataFrame:
    """
    Stooq is a free data source that works on cloud servers.
    No API key needed.
    """
    clean = ticker.replace(".NS", "").replace(".BO", "")

    # Stooq format for Indian NSE stocks
    stooq_symbol = f"{clean.lower()}.ns"

    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

    async with httpx.AsyncClient(
        timeout = 30,
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        }
    ) as client:
        response = await client.get(url)

    if response.status_code != 200:
        raise ValueError(f"Stooq returned {response.status_code} for {stooq_symbol}")

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No data from Stooq for {stooq_symbol}")

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    df = df.dropna()

    logger.info(f"Stooq OK: {len(df)} rows for {stooq_symbol}")
    return df

# ── MAIN FETCHER — smart source selection ──────────────────

async def get_historical_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True
) -> pd.DataFrame:
    cache_key = f"historical:{ticker}:{period}:{interval}"

    # Check cache first
    if use_cache:
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for {ticker}")
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

    # Choose source order based on environment
    if IS_CLOUD:
        # On Render — Alpha Vantage and Stooq work, yfinance often blocked
        sources = [
            ("stooq",         lambda: fetch_from_stooq(ticker)),
            ("alpha_vantage", lambda: fetch_from_alpha_vantage(ticker)),
            ("yfinance",      lambda: fetch_from_yfinance(ticker, period, interval)),
        ]
    else:
        # Local — yfinance is fastest
        sources = [
            ("yfinance",      lambda: fetch_from_yfinance(ticker, period, interval)),
            ("stooq",         lambda: fetch_from_stooq(ticker)),
            ("alpha_vantage", lambda: fetch_from_alpha_vantage(ticker)),
        ]

    last_error = None
    for source_name, fetch_fn in sources:
        try:
            df = await fetch_fn()
            if df is not None and len(df) > 10:
                logger.info(f"✓ {ticker} fetched from {source_name}: {len(df)} rows")

                # Cache the result
                if use_cache:
                    await cache.set(
                        cache_key,
                        df.to_dict(),
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