# data/fetcher.py
import pandas as pd
import numpy as np
import httpx
import logging
import asyncio
import os
import json
from typing import Optional
from datetime import datetime, timedelta
from cache.redis_client import cache
from config import settings

logger = logging.getLogger(__name__)

IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("RAILWAY"))

def format_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker

# ── SOURCE 1 — NSE India Official API (FREE, no key) ──────

NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

async def fetch_from_nse(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """
    NSE India official historical data API.
    Free, no key needed, works on cloud servers.
    """
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    start_str = start_date.strftime("%d-%m-%Y")
    end_str   = end_date.strftime("%d-%m-%Y")

    url = (
        f"https://www.nseindia.com/api/historical/cm/equity"
        f"?symbol={clean}"
        f"&series=[%22EQ%22]"
        f"&from={start_str}"
        f"&to={end_str}"
    )

    async with httpx.AsyncClient(
        headers  = NSE_HEADERS,
        timeout  = 30,
        follow_redirects = True
    ) as client:
        # NSE requires a session cookie — visit homepage first
        await client.get("https://www.nseindia.com", timeout=15)
        response = await client.get(url)

    if response.status_code != 200:
        raise ValueError(f"NSE returned {response.status_code}")

    data = response.json()
    records = data.get("data", [])

    if not records:
        raise ValueError(f"No records from NSE for {clean}")

    rows = []
    for r in records:
        try:
            rows.append({
                "date":   pd.to_datetime(r["CH_TIMESTAMP"]),
                "open":   float(r["CH_OPENING_PRICE"]),
                "high":   float(r["CH_TRADE_HIGH_PRICE"]),
                "low":    float(r["CH_TRADE_LOW_PRICE"]),
                "close":  float(r["CH_CLOSING_PRICE"]),
                "volume": int(r["CH_TOT_TRADED_QTY"]),
            })
        except (KeyError, ValueError):
            continue

    df = pd.DataFrame(rows)
    df = df.set_index("date")
    df = df.sort_index()
    df = df.dropna()

    logger.info(f"NSE OK: {len(df)} rows for {clean}")
    return df

# ── SOURCE 2 — Yahoo Finance via RapidAPI proxy ────────────

async def fetch_from_yahoo_proxy(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Yahoo Finance via a public proxy that bypasses IP blocking.
    No API key needed.
    """
    clean     = ticker.replace(".NS", "").replace(".BO", "")
    yf_ticker = f"{clean}.NS"

    # Use query1 endpoint which is less rate-limited
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"

    period_map = {
        "1mo":  ("1mo",  "1d"),
        "3mo":  ("3mo",  "1d"),
        "6mo":  ("6mo",  "1d"),
        "1y":   ("1y",   "1d"),
        "2y":   ("2y",   "1d"),
        "5y":   ("5y",   "1wk"),
        "7d":   ("5d",   "1d"),
        "5d":   ("5d",   "1d"),
    }

    yf_range, interval = period_map.get(period, ("2y", "1d"))

    params = {
        "range":    yf_range,
        "interval": interval,
        "events":   "history",
    }

    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://finance.yahoo.com",
        "Referer":         "https://finance.yahoo.com/",
    }

    async with httpx.AsyncClient(
        headers          = headers,
        timeout          = 30,
        follow_redirects = True
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

# ── SOURCE 3 — yfinance (works locally) ───────────────────

async def fetch_from_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    import yfinance as yf
    formatted = format_ticker(ticker)
    logger.info(f"Fetching {formatted} from yfinance")

    def _fetch():
        stock = yf.Ticker(formatted)
        df    = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {formatted}")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.dropna()
        return df

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)

# ── MAIN FETCHER ───────────────────────────────────────────

def _period_to_days(period: str) -> int:
    mapping = {
        "7d": 10, "1mo": 35, "3mo": 100,
        "6mo": 190, "1y": 370, "2y": 740, "5y": 1850
    }
    return mapping.get(period, 740)

async def get_historical_data(
    ticker:    str,
    period:    str  = "2y",
    interval:  str  = "1d",
    use_cache: bool = True
) -> pd.DataFrame:

    cache_key = f"historical:{ticker}:{period}:{interval}"

    if use_cache:
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for {ticker}")
            df       = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

    period_days = _period_to_days(period)

    # Source priority — NSE + Yahoo proxy work on cloud
    sources = [
        ("yahoo_proxy", lambda: fetch_from_yahoo_proxy(ticker, period)),
        ("nse_india",   lambda: fetch_from_nse(ticker, period_days)),
        ("yfinance",    lambda: fetch_from_yfinance(ticker, period, interval)),
    ]

    last_error = None
    for source_name, fetch_fn in sources:
        try:
            df = await fetch_fn()
            if df is not None and len(df) >= 10:
                logger.info(f"✓ {ticker} from {source_name}: {len(df)} rows")
                if use_cache:
                    await cache.set(cache_key, df.to_dict(), settings.HISTORICAL_TTL)
                return df
        except Exception as e:
            logger.warning(f"✗ {source_name} failed for {ticker}: {e}")
            last_error = e
            continue

    raise Exception(
        f"All data sources failed for {ticker}. Last error: {last_error}"
    )