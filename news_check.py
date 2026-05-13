# test_pipeline.py
import asyncio
import sys

async def test():
    print("\n=== STEP 1: Testing yfinance ===")
    from data.fetcher import get_historical_data
    try:
        df = await get_historical_data("RELIANCE", period="2y")
        print(f"✓ Data fetched: {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df.index[0]} → {df.index[-1]}")
        print(f"  Sample close: ₹{df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return

    print("\n=== STEP 2: Testing feature engineering ===")
    from ml.features import build_features
    try:
        features = build_features(df)
        print(f"✓ Features built: {features.shape}")
        dist = features['target'].value_counts()
        print(f"  BUY:{dist.get(1,0)} HOLD:{dist.get(0,0)} SELL:{dist.get(-1,0)}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return

    print("\n=== STEP 3: Testing model training ===")
    from ml.model import StockModel
    try:
        model = StockModel()
        result = model.train(features)
        print(f"✓ Model trained")
        print(f"  Accuracy: {result['cv_accuracy']}%")
        print(f"  Trimmed:  {result['trimmed_accuracy']}%")
        print(f"  Folds:    {result['fold_scores']}")
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)       # stop here so we don't get confusing errors below

    print("\n=== STEP 4: Testing prediction ===")
    try:
        # Drop target before predicting
        feat_X  = features.drop("target", axis=1)
        latest  = feat_X.iloc[[-1]]
        pred    = model.predict(latest)
        print(f"✓ Prediction: {pred['signal']} ({pred['confidence']}% confidence)")
        print(f"  Probabilities: {pred['probabilities']}")
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()

    print("\n=== STEP 5: Testing Redis ===")
    from cache.redis_client import cache
    try:
        ok = await cache.ping()
        print(f"{'✓' if ok else '✗'} Redis: {'connected' if ok else 'NOT connected — start Docker Redis'}")
    except Exception as e:
        print(f"✗ Redis error: {e}")

    print("\n=== STEP 6: Testing news ===")
    from data.news_fetcher import get_stock_news
    try:
        news = await get_stock_news("RELIANCE")
        print(f"✓ News fetched")
        print(f"  Sentiment: {news['overall_sentiment']}")
        print(f"  Articles: {len(news['articles'])}")
        if news['articles']:
            print(f"  First: {news['articles'][0]['title'][:60]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n=== ALL TESTS DONE ===\n")

asyncio.run(test())