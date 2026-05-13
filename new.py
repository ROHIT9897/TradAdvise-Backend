python -c "
content = '''import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60:
        raise ValueError(f\"Need at least 60 rows. Got {len(df)}\")

    f      = pd.DataFrame(index=df.index)
    close  = df[\"close\"]
    high   = df[\"high\"]
    low    = df[\"low\"]
    volume = df[\"volume\"]

    # Pre-compute MAs first so all sections can use them
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # 1. Price returns
    for w in [1, 2, 3, 5, 10, 20]:
        f[f\"ret_{w}d\"] = close.pct_change(w)

    # 2. Volatility-normalized momentum
    rolling_std = close.pct_change().rolling(10).std()
    f[\"momentum_norm\"] = close.pct_change(5) / (rolling_std + 1e-9)

    # 3. MA ratios
    f[\"price_ma_5\"]  = close / (ma5  + 1e-9) - 1
    f[\"price_ma_10\"] = close / (ma10 + 1e-9) - 1
    f[\"price_ma_20\"] = close / (ma20 + 1e-9) - 1
    f[\"price_ma_50\"] = close / (ma50 + 1e-9) - 1

    # 4. MA slopes and crosses
    f[\"ma20_slope\"] = ma20.pct_change(5)
    f[\"ma50_slope\"] = ma50.pct_change(10)
    f[\"ma10_vs_20\"] = ma10 / (ma20 + 1e-9) - 1
    f[\"ma20_vs_50\"] = ma20 / (ma50 + 1e-9) - 1

    # 5. RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    f[\"rsi\"]       = rsi / 100
    f[\"rsi_slope\"] = rsi.diff(3) / 100

    # 6. MACD
    ema12       = close.ewm(span=12).mean()
    ema26       = close.ewm(span=26).mean()
    macd        = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    f[\"macd_norm\"]  = macd / (close + 1e-9)
    f[\"macd_hist\"]  = (macd - macd_signal) / (close + 1e-9)
    f[\"macd_cross\"] = np.where(macd > macd_signal, 1, -1)

    # 7. Bollinger Bands — uses ma20 defined above
    std20    = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_range = bb_upper - bb_lower + 1e-9
    f[\"bb_pos\"]   = (close - bb_lower) / bb_range
    f[\"bb_width\"] = bb_range / (ma20 + 1e-9)

    # 8. Volume
    vol_ma = volume.rolling(20).mean()
    f[\"vol_ratio\"] = volume / (vol_ma + 1)
    f[\"vol_slope\"] = vol_ma.pct_change(5)

    # 9. Range and candle
    daily_range    = (high - low) / (close + 1e-9)
    f[\"day_range\"] = daily_range
    f[\"body_size\"] = abs(close - df[\"open\"]) / (daily_range + 1e-9)
    f[\"is_bull\"]   = (close > df[\"open\"]).astype(int)

    # TARGET — equal thirds, always balanced
    future_ret  = close.shift(-3) / close - 1
    buy_thresh  = future_ret.quantile(0.67)
    sell_thresh = future_ret.quantile(0.33)

    f[\"target\"] = np.where(future_ret >= buy_thresh,   1,
                  np.where(future_ret <= sell_thresh, -1, 0))

    f = f.iloc[:-3]
    return f.dropna()
'''

with open('ml/features.py', 'w') as file:
    file.write(content)
print('Done — features.py written cleanly')
"