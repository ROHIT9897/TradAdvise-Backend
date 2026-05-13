# ml/predictor.py
import os
import logging
import numpy as np
import pandas as pd

from ml.model         import StockModel
from ml.features      import build_features
from ml.signal_engine import compute_signal
from data.fetcher     import get_historical_data
from data.news_fetcher import get_stock_news
from cache.redis_client import cache
from config           import settings

logger = logging.getLogger(__name__)

# ── In-memory model registry ──────────────────────────────
_model_registry: dict[str, StockModel] = {}


# ─────────────────────────────────────────────────────────
# MODEL MANAGEMENT
# ─────────────────────────────────────────────────────────

def get_model(ticker: str) -> StockModel:
    """Return cached model or load from disk."""
    if ticker in _model_registry:
        return _model_registry[ticker]

    path = f"models/{ticker}.pkl"
    if os.path.exists(path):
        model = StockModel.load(path)
        _model_registry[ticker] = model
        return model

    raise FileNotFoundError(f"No model for {ticker}. Call /train/{ticker} first.")


async def train_model_for_ticker(ticker: str) -> dict:
    """Fetch data, build features, train and save model."""
    logger.info(f"Training model for {ticker}")

    df = await get_historical_data(ticker, period="2y")
    logger.info(f"Data shape: {df.shape}")

    features = build_features(df)
    logger.info(f"Feature matrix: {features.shape}")

    if len(features) < 100:
        raise ValueError(f"Not enough data: {len(features)} rows (need 100+)")

    model = StockModel()
    result = model.train(features)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{ticker}.pkl")
    _model_registry[ticker] = model

    return {"ticker": ticker, "status": "trained", **result}


# ─────────────────────────────────────────────────────────
# MAIN PREDICTION PIPELINE
# Uses signal_engine (indicator scoring) as primary signal
# ML model confidence used as secondary confirmation
# ─────────────────────────────────────────────────────────

async def get_full_prediction(ticker: str) -> dict:
    cache_key = f"prediction_v2:{ticker}"

    cached = await cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # ── Step 1: Fetch data ────────────────────────────
    df = await get_historical_data(ticker, period="2y")

    # ── Step 2: Signal engine (primary signal) ────────
    signal_result = compute_signal(df)

    # ── Step 3: ML model confidence (secondary) ───────
    ml_confidence = None
    try:
        model    = get_model(ticker)
        features = build_features(df)
        feat_X   = features.drop("target", axis=1)
        latest   = feat_X.iloc[[-1]]
        ml_pred  = model.predict(latest)
        ml_confidence = ml_pred["confidence"]

        # If ML and signal engine agree — boost confidence
        # If they disagree — reduce confidence slightly
        if ml_pred["signal"] == signal_result["signal"]:
            signal_result["confidence"] = min(
                100,
                signal_result["confidence"] + 5
            )
            logger.info(f"ML confirms signal engine: {ml_pred['signal']}")
        else:
            signal_result["confidence"] = max(
                0,
                signal_result["confidence"] - 5
            )
            logger.info(
                f"ML disagrees: engine={signal_result['signal']} "
                f"ml={ml_pred['signal']}"
            )

    except FileNotFoundError:
        logger.info(f"No ML model for {ticker} — using signal engine only")
    except Exception as e:
        logger.warning(f"ML prediction failed: {e} — using signal engine only")

    # ── Step 4: Validate signal ───────────────────────
    validation = _validate(signal_result)

    # ── Step 5: News + sentiment ──────────────────────
    news      = await get_stock_news(ticker)
    sentiment = news["overall_sentiment"]

    # ── Step 6: Sentiment adjustment ──────────────────
    score = signal_result["score"]
    note  = ""

    if signal_result["signal"] == "BUY" and sentiment == "BULLISH":
        score = min(100, score + 5)
        note  = "News sentiment confirms bullish signal"
    elif signal_result["signal"] == "BUY" and sentiment == "BEARISH":
        score = max(0, score - 8)
        note  = "Warning: bearish news contradicts BUY signal"
    elif signal_result["signal"] == "SELL" and sentiment == "BEARISH":
        score = min(100, score + 5)
        note  = "News sentiment confirms bearish signal"
    elif signal_result["signal"] == "SELL" and sentiment == "BULLISH":
        score = max(0, score - 8)
        note  = "Warning: bullish news contradicts SELL signal"

    # ── Step 7: Explanation ───────────────────────────
    explanation = _generate_explanation(
        signal_result["signal"],
        signal_result["raw_indicators"],
        signal_result["individual_signals"]
    )

    # ── Step 8: Build final response ──────────────────
    final = _sanitize({
        "signal":             signal_result["signal"],
        "score":              round(score, 1),
        "confidence":         round(signal_result["confidence"], 1),
        "agreement":          signal_result["agreement"],
        "bull_indicators":    signal_result["bull_indicators"],
        "bear_indicators":    signal_result["bear_indicators"],
        "ml_confirmation":    ml_confidence,
        "validation":         validation,
        "sentiment":          sentiment,
        "sentiment_score":    news["sentiment_score"],
        "signal_note":        note,
        "explanation":        explanation,
        "indicators":         signal_result["raw_indicators"],
        "individual_signals": signal_result["individual_signals"],
        "news":               news,
        "cached":             False,
    })

    await cache.set(cache_key, final, settings.PREDICTION_TTL)
    return final


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _validate(signal_result: dict) -> dict:
    signal     = signal_result["signal"]
    confidence = signal_result["confidence"]
    rsi        = signal_result["raw_indicators"].get("rsi", 50)
    vol_ratio  = signal_result["raw_indicators"].get("vol_ratio", 1.0)

    blocks = []
    passes = []

    # Confidence gate
    if confidence < 58:
        blocks.append(f"Signal confidence too low ({confidence:.0f}%)")
    else:
        passes.append(f"Confidence OK ({confidence:.0f}%)")

    # RSI extreme check
    if signal == "BUY" and rsi > 78:
        blocks.append(f"RSI overbought ({rsi:.0f}) — risky BUY entry")
    elif signal == "SELL" and rsi < 22:
        blocks.append(f"RSI oversold ({rsi:.0f}) — risky SELL entry")
    else:
        passes.append(f"RSI in acceptable range ({rsi:.0f})")

    # Volume check
    if vol_ratio < 0.5:
        blocks.append(f"Very low volume ({vol_ratio:.1f}x avg) — weak signal")
    else:
        passes.append(f"Volume normal ({vol_ratio:.1f}x avg)")

    is_valid     = len(blocks) == 0
    final_signal = signal if is_valid else "HOLD"

    return {
        "final_signal":   final_signal,
        "is_validated":   is_valid,
        "blocked_reasons": blocks,
        "passed_checks":  passes,
    }


def _generate_explanation(signal: str, indicators: dict,
                           individual: dict) -> list:
    reasons = []
    rsi       = indicators.get("rsi", 50)
    vol_ratio = indicators.get("vol_ratio", 1.0)
    bb_pos    = indicators.get("bb_position", 0.5)
    ma_slope  = indicators.get("ma20_slope", 0)

    trend_score   = individual.get("trend", 0)
    momentum_score = individual.get("momentum", 0)
    macd_score    = individual.get("macd", 0)
    band_score    = individual.get("bands", 0)
    vol_score     = individual.get("volume", 0)

    if signal == "BUY":
        if rsi < 40:
            reasons.append(f"RSI at {rsi:.0f} — oversold, potential reversal")
        elif rsi < 50:
            reasons.append(f"RSI at {rsi:.0f} — room to move upward")
        if macd_score > 0:
            reasons.append("MACD above signal line — bullish momentum building")
        if bb_pos < 0.3:
            reasons.append("Price near lower Bollinger Band — historical bounce zone")
        if vol_ratio > 1.3:
            reasons.append(f"Volume {vol_ratio:.1f}x above average — buyer interest")
        if trend_score > 0:
            reasons.append("Short-term MA above long-term MA — uptrend confirmed")
        if ma_slope > 0:
            reasons.append("20-day MA trending upward — positive momentum")

    elif signal == "SELL":
        if rsi > 65:
            reasons.append(f"RSI at {rsi:.0f} — overbought, correction likely")
        if macd_score < 0:
            reasons.append("MACD below signal line — bearish momentum")
        if bb_pos > 0.75:
            reasons.append("Price near upper Bollinger Band — resistance zone")
        if vol_ratio > 1.3:
            reasons.append(f"Volume {vol_ratio:.1f}x above average — selling pressure")
        if trend_score < 0:
            reasons.append("Short-term MA below long-term MA — downtrend signal")

    elif signal == "HOLD":
        reasons.append("Mixed signals across indicators — safer to wait")
        if rsi > 45 and rsi < 55:
            reasons.append(f"RSI at {rsi:.0f} — neutral momentum zone")
        reasons.append("No strong directional confirmation from multiple indicators")

    # Cap at 4 reasons — keep it readable
    return reasons[:4]


def _sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj