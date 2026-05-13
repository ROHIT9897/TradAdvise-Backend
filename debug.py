# debug.py — run with: python debug.py

import asyncio
import sys

async def run_diagnostics():
    print("\n" + "="*50)
    print("STOCKAI DIAGNOSTICS")
    print("="*50)

    # Test 1: Redis
    print("\n[1/5] Testing Redis...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("     ✅ Redis connected")
    except Exception as e:
        print(f"     ❌ Redis FAILED: {e}")
        print("     Fix: docker run -d -p 6379:6379 redis:7-alpine")

    # Test 2: yfinance + RELIANCE.NS
    print("\n[2/5] Testing yfinance with RELIANCE.NS...")
    try:
        import yfinance as yf
        df = yf.Ticker("RELIANCE.NS").history(period="6mo")
        if df.empty:
            print("     ❌ yfinance returned empty data")
            print("     Fix: pip install --upgrade yfinance")
        else:
            print(f"     ✅ yfinance OK — {len(df)} rows fetched")
            print(f"     Latest close: ₹{df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"     ❌ yfinance FAILED: {e}")

    # Test 3: Feature engineering
    print("\n[3/5] Testing feature engineering...")
    try:
        import yfinance as yf
        from ml.features import build_features
        df = yf.Ticker("RELIANCE.NS").history(period="2y")
        df.columns = [c.lower() for c in df.columns]
        features = build_features(df)
        print(f"     ✅ Features OK — {features.shape[1]} features, {len(features)} rows")
    except Exception as e:
        print(f"     ❌ Feature engineering FAILED: {e}")

    # Test 4: Model training
    print("\n[4/5] Testing model train + predict...")
    try:
        import yfinance as yf
        from ml.features import build_features
        from ml.model import StockModel
        df = yf.Ticker("RELIANCE.NS").history(period="2y")
        df.columns = [c.lower() for c in df.columns]
        features = build_features(df)
        model = StockModel()
        model.train(features)
        result = model.predict(features.drop("target", axis=1).iloc[[-1]])
        print(f"     ✅ Model OK — Signal: {result['signal']}, Confidence: {result['confidence']}%")
    except Exception as e:
        print(f"     ❌ Model FAILED: {e}")

    # Test 5: NewsAPI
    print("\n[5/5] Testing NewsAPI...")
    try:
        import httpx
        from dotenv import load_dotenv
        import os
        load_dotenv()
        key = os.getenv("NEWS_API_KEY", "")

        if not key or key == "your_newsapi_key_here":
            print("     ❌ NEWS_API_KEY not set in .env file")
        else:
            r = httpx.get(
                "https://newsapi.org/v2/everything",
                params={"q": "Infosys stock", "pageSize": 3, "apiKey": key}
            )
            data = r.json()
            if data.get("status") == "ok":
                count = len(data.get("articles", []))
                print(f"     ✅ NewsAPI OK — {count} articles found")
            else:
                print(f"     ❌ NewsAPI error: {data.get('message')}")
    except Exception as e:
        print(f"     ❌ NewsAPI FAILED: {e}")

    print("\n" + "="*50)
    print("Done. Fix any ❌ above before running the server.")
    print("="*50 + "\n")

asyncio.run(run_diagnostics())