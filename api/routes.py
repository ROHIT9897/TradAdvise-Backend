from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from ml.predictor import get_full_prediction, train_model_for_ticker
from data.live_price import get_live_price, get_top_gainers_losers
from data.news_fetcher import get_stock_news
from data.fetcher import get_historical_data          # ADD THIS
from cache.redis_client import cache                  # ADD THIS

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/analyze/{ticker}")
@limiter.limit("15/minute")
async def analyze(ticker: str, request: Request):
    import traceback

    try:
        result = await get_full_prediction(ticker.upper())
        return result

    except FileNotFoundError as e:
        # Model not trained yet
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Model not trained",
                "message": f"No model found for {ticker}. Call POST /train/{ticker} first.",
                "fix": f"Run: curl -X POST http://localhost:8000/api/v1/train/{ticker}"
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Data error",
                "message": str(e)
            }
        )
    except Exception as e:
        # Print full traceback to your terminal
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "hint": "Check uvicorn terminal for full traceback"
            }
        )

@router.get("/price/{ticker}")
@limiter.limit("30/minute")
async def live_price(ticker: str, request: Request):
    try:
        return await get_live_price(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/{ticker}")
@limiter.limit("10/minute")
async def news(ticker: str, request: Request):
    try:
        return await get_stock_news(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/movers")
@limiter.limit("10/minute")
async def market_movers(request: Request):
    try:
        return await get_top_gainers_losers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/{ticker}")
async def train(ticker: str):
    try:
        result = await train_model_for_ticker(ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    from cache.redis_client import cache
    return {
        "status": "ok",
        "redis": await cache.ping()
    }

# Add this to api/routes.py temporarily
@router.delete("/cache/flush")
async def flush_cache():
    from cache.redis_client import cache
    await cache.client.flushall()
    return {"status": "cache cleared"}

@router.get("/chart/{ticker}")
@limiter.limit("20/minute")
async def get_chart_data(
    ticker: str,
    period: str = "1mo",
    request: Request = None
):
    """
    Returns OHLCV price data for charting.
    period options: 7d, 1mo, 3mo, 6mo, 1y
    """
    cache_key = f"chart:{ticker}:{period}"
    cached    = await cache.get(cache_key)
    if cached:
        return cached

    try:
        # Map frontend period to yfinance period
        period_map = {
            "1W": ("7d",  "1d"),
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "6M": ("6mo", "1d"),
            "1Y": ("1y",  "1wk"),
        }
        yf_period, interval = period_map.get(period, ("1mo", "1d"))

        df = await get_historical_data(
            ticker,
            period   = yf_period,
            interval = interval,
            use_cache = False
        )

        # Build response
        prices = []
        for date, row in df.iterrows():
            prices.append({
                "date":   str(date)[:10],   # YYYY-MM-DD only
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

        # Cache based on period
        ttl = 300 if period == "1W" else 3600
        await cache.set(cache_key, result, ttl)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))