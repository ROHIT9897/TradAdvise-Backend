# train_all.py
import httpx
import asyncio

TICKERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "WIPRO", "LT", "AXISBANK",
    "KOTAKBANK", "HINDUNILVR", "ITC", "SUNPHARMA", "MARUTI",
    "BAJFINANCE", "TITAN", "ASIANPAINT", "NESTLEIND", "PNB"
]

async def train_all():
    async with httpx.AsyncClient(timeout=120) as client:
        for ticker in TICKERS:
            try:
                r = await client.post(
                    f"http://localhost:8000/api/v1/train/{ticker}"
                )
                data = r.json()
                print(f"✓ {ticker}: {data.get('cv_accuracy')}% accuracy")
            except Exception as e:
                print(f"✗ {ticker}: {e}")
            await asyncio.sleep(2)

asyncio.run(train_all())