# ml/horizon_predictor.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from data.fetcher import get_historical_data

logger = logging.getLogger(__name__)

async def get_horizon_prediction(
    ticker:   str,
    days:     int,
    strategy: str = "hold"
) -> dict:
    """
    Predict target price, stop loss, and expected date
    for a given investment horizon.
    """
    df = await get_historical_data(ticker, period="2y")

    if len(df) < 50:
        raise ValueError(f"Not enough data for {ticker}")

    current_price = float(df["close"].iloc[-1])
    analysis      = _analyze_horizon(df, days, current_price)

    # Calculate dates
    today        = datetime.now()
    notify_date  = today + timedelta(days=int(days * 0.8))
    target_date  = today + timedelta(days=days)

    # Adjust based on strategy
    if strategy == "sell":
        # Short position — targets reversed
        target_price = current_price * (1 - analysis["upside_pct"] / 100)
        stop_loss    = current_price * (1 + analysis["downside_pct"] / 100)
    else:
        target_price = current_price * (1 + analysis["upside_pct"] / 100)
        stop_loss    = current_price * (1 - analysis["downside_pct"] / 100)

    return {
        "ticker":         ticker,
        "strategy":       strategy,
        "days":           days,
        "current_price":  round(current_price, 2),

        # Prediction
        "target_price":   round(target_price, 2),
        "stop_loss":      round(stop_loss, 2),
        "upside_pct":     round(analysis["upside_pct"], 2),
        "downside_pct":   round(analysis["downside_pct"], 2),
        "confidence":     analysis["confidence"],
        "signal":         analysis["signal"],

        # Dates
        "entry_date":     today.strftime("%d %b %Y"),
        "notify_date":    notify_date.strftime("%d %b %Y"),
        "notify_date_ts": notify_date.strftime("%Y-%m-%d"),
        "target_date":    target_date.strftime("%d %b %Y"),
        "target_date_ts": target_date.strftime("%Y-%m-%d"),

        # Context
        "horizon_type":   _get_horizon_type(days),
        "reasoning":      analysis["reasoning"],
        "risk_level":     analysis["risk_level"],
    }


def _analyze_horizon(
    df:            pd.DataFrame,
    days:          int,
    current_price: float
) -> dict:
    close   = df["close"]
    volume  = df["volume"]
    high    = df["high"]
    low     = df["low"]

    # ── Trend indicators ──────────────────────────────────
    ma20    = close.rolling(20).mean().iloc[-1]
    ma50    = close.rolling(50).mean().iloc[-1]
    ma200   = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50

    # ── RSI ───────────────────────────────────────────────
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss
    rsi     = float(100 - (100 / (1 + rs.iloc[-1])))

    # ── Volatility ────────────────────────────────────────
    returns     = close.pct_change().dropna()
    volatility  = float(returns.std() * np.sqrt(252))  # annualized

    # ── 52-week range ─────────────────────────────────────
    week52_high = float(high.tail(252).max())
    week52_low  = float(low.tail(252).min())
    range_pos   = (current_price - week52_low) / (week52_high - week52_low)

    # ── Historical returns for this horizon ───────────────
    past_returns = []
    step = max(1, days // 5)
    for i in range(100, len(df) - days, step):
        entry  = float(close.iloc[i])
        exit_p = float(close.iloc[i + days])
        past_returns.append((exit_p - entry) / entry * 100)

    avg_return   = float(np.mean(past_returns)) if past_returns else 5.0
    win_rate     = float(np.mean([r > 0 for r in past_returns])) if past_returns else 0.55

    # ── Score calculation ─────────────────────────────────
    score = 50

    # Trend
    if current_price > ma20:  score += 8
    if current_price > ma50:  score += 7
    if current_price > ma200: score += 10
    if ma20 > ma50:           score += 5

    # RSI
    if days <= 30:
        if 40 < rsi < 60:  score += 8
        if rsi < 35:       score += 12   # oversold — good entry
        if rsi > 70:       score -= 10
    else:
        if rsi < 50:       score += 5    # room to grow
        if rsi > 75:       score -= 5

    # 52-week position
    if range_pos < 0.3:    score += 10   # near 52-week low — good buy
    if range_pos > 0.85:   score -= 8    # near 52-week high — risky

    # Volume trend
    vol_avg   = float(volume.tail(20).mean())
    vol_today = float(volume.iloc[-1])
    if vol_today > vol_avg * 1.5:  score += 5

    score = max(20, min(90, score))

    # ── Upside/downside based on volatility + horizon ─────
    daily_vol    = float(returns.std())
    horizon_vol  = daily_vol * np.sqrt(days)

    upside_pct   = max(
        abs(avg_return) * 0.6,
        horizon_vol * 100 * 1.2
    )
    downside_pct = max(
        upside_pct * 0.5,
        horizon_vol * 100 * 0.8
    )

    # Cap reasonable values
    upside_pct   = min(upside_pct, 50 if days <= 30 else 80)
    downside_pct = min(downside_pct, 20 if days <= 30 else 35)

    # ── Signal ────────────────────────────────────────────
    if score >= 65:   signal = "BUY"
    elif score <= 40: signal = "WAIT"
    else:             signal = "HOLD"

    # ── Confidence ────────────────────────────────────────
    confidence = int(
        win_rate * 50 +
        (score / 100) * 30 +
        (1 - volatility) * 20
    )
    confidence = max(45, min(82, confidence))

    # ── Risk level ────────────────────────────────────────
    if volatility > 0.4:    risk = "High"
    elif volatility > 0.25: risk = "Medium"
    else:                   risk = "Low"

    # ── Reasoning ─────────────────────────────────────────
    # ── Reasoning ─────────────────────────────────────
    reasoning = []

    # Trend
    if current_price > ma200:
        reasoning.append("Price above 200-day average — strong uptrend")
    elif current_price > ma50:
        reasoning.append("Price above 50-day average — uptrend intact")
    elif current_price < ma50:
        reasoning.append("Price below 50-day average — caution advised")

    # RSI
    if rsi < 35:
        reasoning.append(f"RSI {rsi:.0f} — stock is oversold, good entry point")
    elif rsi < 50:
        reasoning.append(f"RSI {rsi:.0f} — neutral zone, room to grow")
    elif rsi > 70:
        reasoning.append(f"RSI {rsi:.0f} — overbought, wait for pullback")

    # 52-week position
    if range_pos < 0.25:
        reasoning.append("Near 52-week low — limited downside risk")
    elif range_pos > 0.85:
        reasoning.append("Near 52-week high — momentum strong but risky")
    elif 0.4 < range_pos < 0.6:
        reasoning.append("Mid-range position — balanced risk/reward")

    # Historical performance
    if past_returns and win_rate > 0.6:
        reasoning.append(
            f"Profitable {win_rate*100:.0f}% of past {days}-day periods"
        )
    elif past_returns and win_rate > 0.5:
        reasoning.append(
            f"Positive returns {win_rate*100:.0f}% of past {days}-day periods"
        )

    # Volatility warning
    if volatility > 0.35:
        reasoning.append("High volatility — use strict stop loss")
    elif volatility < 0.2:
        reasoning.append("Low volatility — stable stock, good for long term")

    # Volume
    if vol_today > vol_avg * 1.5:
        reasoning.append("High volume today — strong market interest")

    return{
        "upside_pct":   upside_pct,
        "downside_pct": downside_pct,
        "confidence":   confidence,
        "signal":       signal,
        "risk_level":   risk,
        "reasoning":    reasoning[:4],  # max 4 points
    }

def _get_horizon_type(days: int) -> str:
    if days <= 7:    return "Very Short Term"
    if days <= 30:   return "Short Term"
    if days <= 90:   return "Medium Term"
    if days <= 180:  return "Long Term"
    return "Very Long Term"