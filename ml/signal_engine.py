# ml/signal_engine.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_signal(df: pd.DataFrame) -> dict:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    signals = {}

    # ── Pre-compute all MAs first ─────────────────────
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # ── 1. Trend signals ──────────────────────────────
    signals["price_vs_ma20"]  = 1 if close.iloc[-1] > ma20.iloc[-1] else -1
    signals["ma_cross"]       = 1 if ma10.iloc[-1]  > ma20.iloc[-1] else -1
    signals["long_trend"]     = 1 if ma20.iloc[-1]  > ma50.iloc[-1] else -1

    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / (ma20.iloc[-5] + 1e-9)
    signals["ma20_trending"]  = 1 if ma20_slope > 0 else -1

    # ── 2. RSI ────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = 100 - (100 / (1 + gain / (loss + 1e-9)))
    rsi_now  = float(rsi.iloc[-1])

    if rsi_now < 30:   signals["rsi"] = 2
    elif rsi_now < 45: signals["rsi"] = 1
    elif rsi_now > 70: signals["rsi"] = -2
    elif rsi_now > 55: signals["rsi"] = -1
    else:              signals["rsi"] = 0

    # ── 3. MACD ───────────────────────────────────────
    ema12        = close.ewm(span=12).mean()
    ema26        = close.ewm(span=26).mean()
    macd         = ema12 - ema26
    macd_signal  = macd.ewm(span=9).mean()
    macd_hist    = macd - macd_signal

    signals["macd_cross"]  = 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1
    signals["macd_rising"] = 1 if macd_hist.iloc[-1] > macd_hist.iloc[-2] else -1

    # ── 4. Bollinger Bands ────────────────────────────
    std20    = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_pos   = (close.iloc[-1] - bb_lower.iloc[-1]) / (
        bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9
    )

    if bb_pos < 0.2:   signals["bb"] =  2
    elif bb_pos < 0.4: signals["bb"] =  1
    elif bb_pos > 0.8: signals["bb"] = -2
    elif bb_pos > 0.6: signals["bb"] = -1
    else:              signals["bb"] =  0

    # ── 5. Volume ─────────────────────────────────────
    vol_ma    = volume.rolling(20).mean()
    vol_ratio = float(volume.iloc[-1] / (vol_ma.iloc[-1] + 1))
    price_up  = close.iloc[-1] > close.iloc[-2]

    if price_up and vol_ratio > 1.5:      signals["volume"] =  2
    elif price_up and vol_ratio > 1.0:    signals["volume"] =  1
    elif not price_up and vol_ratio > 1.5: signals["volume"] = -2
    elif not price_up and vol_ratio > 1.0: signals["volume"] = -1
    else:                                  signals["volume"] =  0

    # ── 6. Supertrend ─────────────────────────────────
    atr        = (high - low).rolling(14).mean()
    lower_band = ((high + low) / 2) - (3 * atr)
    signals["supertrend"] = 1 if close.iloc[-1] > lower_band.iloc[-1] else -1

    # ── Compute weighted score ────────────────────────
    weights = {
        "price_vs_ma20": 2,
        "ma_cross":      2,
        "long_trend":    1,
        "ma20_trending": 1,
        "rsi":           3,
        "macd_cross":    2,
        "macd_rising":   1,
        "bb":            2,
        "volume":        2,
        "supertrend":    2,
    }

    raw_score = sum(signals[k] * weights[k] for k in signals)

    # Max possible score (all signals at max positive)
    max_possible = sum(weights[k] * (2 if k in ["rsi", "bb", "volume"] else 1)
                       for k in weights)

    # Normalize to 0-100
    score = (raw_score / (max_possible + 1e-9) + 1) / 2 * 100
    score = round(float(np.clip(score, 0, 100)), 1)

    # ── Determine signal ──────────────────────────────
    if score >= 62:
        signal     = "BUY"
        confidence = score
    elif score <= 38:
        signal     = "SELL"
        confidence = 100 - score
    else:
        signal     = "HOLD"
        confidence = 50 + abs(score - 50)

    bull_count = sum(1 for v in signals.values() if v > 0)
    bear_count = sum(1 for v in signals.values() if v < 0)
    total      = len(signals)

    return {
        "signal":             signal,
        "score":              score,
        "confidence":         round(float(confidence), 1),
        "bull_indicators":    bull_count,
        "bear_indicators":    bear_count,
        "total_indicators":   total,
        "agreement":          f"{max(bull_count, bear_count)}/{total} indicators agree",
        "individual_signals": {
            "trend":    _avg([signals["price_vs_ma20"], signals["ma_cross"],
                              signals["long_trend"],    signals["ma20_trending"]]),
            "momentum": signals["rsi"],
            "macd":     _avg([signals["macd_cross"], signals["macd_rising"]]),
            "bands":    signals["bb"],
            "volume":   signals["volume"],
        },
        "raw_indicators": {
            "rsi":         round(rsi_now, 1),
            "bb_position": round(float(bb_pos), 2),
            "vol_ratio":   round(vol_ratio, 2),
            "ma20_slope":  round(float(ma20_slope * 100), 3),
        }
    }


def _avg(values: list) -> float:
    return round(sum(values) / len(values), 1)