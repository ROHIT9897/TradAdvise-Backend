# api/routes.py
import asyncio
import traceback
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from ml.predictor import get_full_prediction, train_model_for_ticker
from data.live_price import get_live_price, get_top_gainers_losers
from data.news_fetcher import get_stock_news
from data.fetcher import get_historical_data
from cache.redis_client import cache
from ml.horizon_predictor import get_horizon_prediction
from ml.horizon_predictor import get_target_prediction

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# ── Full AI prediction ────────────────────────────────────

@router.get("/analyze/{ticker}")
@limiter.limit("15/minute")
async def analyze(ticker: str, request: Request):
    try:
        result = await get_full_prediction(ticker.upper())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No model for {ticker}. Call POST /train/{ticker} first."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Live price ────────────────────────────────────────────

@router.get("/price/{ticker}")
@limiter.limit("30/minute")
async def live_price(ticker: str, request: Request):
    try:
        return await get_live_price(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── News + sentiment ──────────────────────────────────────

@router.get("/news/{ticker}")
@limiter.limit("10/minute")
async def news(ticker: str, request: Request):
    try:
        return await get_stock_news(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Market movers ─────────────────────────────────────────

@router.get("/market/movers")
@limiter.limit("10/minute")
async def market_movers(request: Request):
    try:
        return await get_top_gainers_losers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Chart data ────────────────────────────────────────────

@router.get("/chart/{ticker}")
@limiter.limit("20/minute")
async def get_chart_data(
    ticker:  str,
    request: Request,
    period:  str = "1M",        # default 1M
):
    cache_key = f"chart:{ticker}:{period}"
    cached    = await cache.get(cache_key)
    if cached:
        return cached

    try:
        # Maps EXACTLY what Android sends → yfinance period
        period_map = {
            "1W": ("7d",  "1d"),
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "6M": ("6mo", "1d"),
            "1Y": ("1y",  "1wk"),
        }
        yf_period, interval = period_map.get(
            period.upper(),
            ("1mo", "1d")       # fallback if unknown period sent
        )

        df = await get_historical_data(
            ticker,
            period    = yf_period,
            interval  = interval,
            use_cache = False
        )

        prices = []
        for date, row in df.iterrows():
            prices.append({
                "date":   str(date)[:10],
                "open":   round(float(row["open"]),   2),
                "high":   round(float(row["high"]),   2),
                "low":    round(float(row["low"]),    2),
                "close":  round(float(row["close"]),  2),
                "volume": int(row["volume"])
            })

        result = {
            "ticker": ticker,
            "period": period,
            "count":  len(prices),
            "prices": prices
        }

        ttl = 300 if period == "1W" else 3600
        await cache.set(cache_key, result, ttl)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Train model ───────────────────────────────────────────

@router.post("/train/{ticker}")
async def train(ticker: str):
    try:
        result = await train_model_for_ticker(ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Health check ──────────────────────────────────────────

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "redis":  await cache.ping()
    }

# ── Cache flush (dev only) ────────────────────────────────

@router.delete("/cache/flush")
async def flush_cache():
    await cache.client.flushall()
    return {"status": "cache cleared"}

@router.get("/predict/{ticker}")
@limiter.limit("10/minute")
async def predict_horizon(
    ticker:   str,
    request:  Request,
    days:     int = 30,
    strategy: str = "hold"
):
    try:
        result = await get_horizon_prediction(
            ticker.upper(), days, strategy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/target/{ticker}")
@limiter.limit("10/minute")
async def predict_target(
    ticker:       str,
    request:      Request,
    target_price: float,    # user ka goal
    strategy:     str = "hold"
):
    """
    User apna target price set karta hai.
    App batata hai kitna time lagega aur chances kitne hain.
    """
    try:
        result = await get_target_prediction(
            ticker.upper(), target_price, strategy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))