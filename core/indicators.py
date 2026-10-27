"""
VectorTrade - Quantitative Indicators Engine
Computes 130+ technical and contextual features from OHLCV and market indices.
Vectorized across NumPy and Pandas with strict lookahead bias prevention,
division-by-zero safeguards, and standardized snake_case column names.
"""

from __future__ import annotations
import logging
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("VectorTrade.Indicators")


def safe_divide(numerator: Union[pd.Series, np.ndarray, float],
                denominator: Union[pd.Series, np.ndarray, float],
                fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """
    Safely divides numerator by denominator, replacing 0-division, infinities, and NaNs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        num_arr = np.asarray(numerator)
        den_arr = np.asarray(denominator)
        result = np.where(den_arr != 0, num_arr / den_arr, fill_value)
        result = np.where(np.isfinite(result), result, fill_value)
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Vectorized Relative Strength Index (RSI).
    Zero lookahead: uses backward-looking rolling window.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    
    rs = safe_divide(gain, loss, fill_value=0.0)
    rsi = 100.0 - safe_divide(100.0, 1.0 + rs, fill_value=50.0)
    rsi = np.where((loss == 0) & (gain > 0), 100.0, rsi)
    rsi = np.where((gain == 0) & (loss > 0), 0.0, rsi)
    return pd.Series(rsi, index=series.index)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Vectorized Average True Range (ATR).
    TR = max(High - Low, abs(High - Close[t-1]), abs(Low - Close[t-1]))
    """
    high_low = df["high"] - df["low"]
    prev_close = df["close"].shift(1)
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_indicators(
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes the complete 130+ technical feature set for a single stock's daily OHLCV series.
    Input df must contain: ['date', 'open', 'high', 'low', 'close', 'volume'] (case-insensitive).
    Output contains clean, leak-free, snake_case feature columns matching production specs.
    """
    res = df.copy()

    # Normalize core OHLCV column names to lowercase
    rename_map = {c: c.lower() for c in res.columns}
    if "adj close" in rename_map.values():
        rename_map["adj close"] = "adjusted_close"
    res = res.rename(columns=rename_map)

    if "adjusted_close" not in res.columns:
        res["adjusted_close"] = res["close"]

    # Ensure chronological sorting to strictly prevent lookahead bias
    if "date" in res.columns:
        res["date"] = pd.to_datetime(res["date"])
        res = res.sort_values("date", ascending=True).reset_index(drop=True)

    # Convert numeric fields
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").ffill().bfill()

    close = res["close"]
    high = res["high"]
    low = res["low"]
    open_p = res["open"]
    vol = res["volume"]

    features: Dict[str, pd.Series] = {}

    # 1. Moving Averages, Slopes, and Cross Ratios
    ma_periods = [15, 30, 50, 100, 150, 200]
    for ma in ma_periods:
        ma_col = close.rolling(window=ma, min_periods=ma).mean()
        features[f"ma{ma}"] = ma_col
        prev_ma = ma_col.shift(1)
        features[f"ma{ma}_slope"] = safe_divide(ma_col.diff(1), prev_ma, fill_value=0.0) * 100.0
        features[f"close_ma{ma}_ratio"] = safe_divide(close, ma_col, fill_value=1.0)

    features["ma15_ma30_ratio"] = safe_divide(features["ma15"], features["ma30"], fill_value=1.0)
    features["ma30_ma50_ratio"] = safe_divide(features["ma30"], features["ma50"], fill_value=1.0)
    features["ma50_ma100_ratio"] = safe_divide(features["ma50"], features["ma100"], fill_value=1.0)
    features["ma50_ma150_ratio"] = safe_divide(features["ma50"], features["ma150"], fill_value=1.0)
    features["ma100_ma200_ratio"] = safe_divide(features["ma100"], features["ma200"], fill_value=1.0)

    # 2. Bollinger Bands (20 period, 2 std)
    bb_window = 20
    bb_mid = close.rolling(window=bb_window, min_periods=bb_window).mean()
    bb_std = close.rolling(window=bb_window, min_periods=bb_window).std()

    features["bb_middle"] = bb_mid
    features["bb_upper"] = bb_mid + (bb_std * 2.0)
    features["bb_lower"] = bb_mid - (bb_std * 2.0)

    bb_width = features["bb_upper"] - features["bb_lower"]
    features["bb_width"] = bb_width
    features["bb_width_pct"] = safe_divide(bb_width, bb_mid, fill_value=0.0)
    features["bb_percent_b"] = safe_divide(close - features["bb_lower"], bb_width, fill_value=0.5)

    features["close_bb_upper_dist"] = close - features["bb_upper"]
    features["close_bb_middle_dist"] = close - features["bb_middle"]
    features["close_bb_lower_dist"] = close - features["bb_lower"]
    features["bb_width_change"] = bb_width.diff().fillna(0.0)

    # 3. RSI Indicators (7, 14, 21) & Momentum
    for p in [7, 14, 21]:
        features[f"rsi{p}"] = calc_rsi(close, period=p)

    features["rsi_slope"] = features["rsi14"].diff().fillna(0.0)
    features["rsi_change_1d"] = features["rsi14"].diff(1).fillna(0.0)
    features["rsi_change_3d"] = features["rsi14"].diff(3).fillna(0.0)
    features["rsi_change_5d"] = features["rsi14"].diff(5).fillna(0.0)

    # 4. Price Returns
    for d in [1, 3, 5, 7, 10, 20, 50]:
        prev_c = close.shift(d)
        features[f"return_{d}d"] = safe_divide(close - prev_c, prev_c, fill_value=0.0) * 100.0

    # 5. Price Extremes & Range Distances
    for d in [5, 10, 20, 50]:
        features[f"high_{d}d"] = high.rolling(window=d, min_periods=d).max()
        features[f"low_{d}d"] = low.rolling(window=d, min_periods=d).min()

    features["dist_high_5d"] = safe_divide(close, features["high_5d"], fill_value=1.0) - 1.0
    features["dist_low_5d"] = safe_divide(close, features["low_5d"], fill_value=1.0) - 1.0
    features["dist_high_20d"] = safe_divide(close, features["high_20d"], fill_value=1.0) - 1.0
    features["dist_low_20d"] = safe_divide(close, features["low_20d"], fill_value=1.0) - 1.0

    high_52w = high.rolling(window=252, min_periods=20).max()
    low_52w = low.rolling(window=252, min_periods=20).min()
    features["dist_high_52w"] = safe_divide(close, high_52w, fill_value=1.0) - 1.0
    features["dist_low_52w"] = safe_divide(close, low_52w, fill_value=1.0) - 1.0

    # 6. Candlestick Structural Characteristics & Patterns
    c_body = (close - open_p).abs()
    c_range = high - low
    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low

    features["candle_body"] = c_body
    features["candle_range"] = c_range
    features["upper_wick"] = upper_wick
    features["lower_wick"] = lower_wick

    features["body_range_ratio"] = safe_divide(c_body, c_range, fill_value=0.0)
    features["upper_wick_range_ratio"] = safe_divide(upper_wick, c_range, fill_value=0.0)
    features["lower_wick_range_ratio"] = safe_divide(lower_wick, c_range, fill_value=0.0)
    features["close_position"] = safe_divide(close - low, c_range, fill_value=0.5)

    prev_c1 = close.shift(1)
    features["gap_pct"] = safe_divide(open_p - prev_c1, prev_c1, fill_value=0.0) * 100.0

    features["is_bullish"] = (close > open_p).astype(int)
    features["is_doji"] = (features["body_range_ratio"] < 0.1).astype(int)
    features["is_hammer"] = (
        (features["lower_wick_range_ratio"] > 0.6) &
        (features["upper_wick_range_ratio"] < 0.1) &
        (features["body_range_ratio"] < 0.3)
    ).astype(int)
    features["is_shooting_star"] = (
        (features["upper_wick_range_ratio"] > 0.6) &
        (features["lower_wick_range_ratio"] < 0.1) &
        (features["body_range_ratio"] < 0.3)
    ).astype(int)

    prev_body = c_body.shift(1)
    prev_bullish = close.shift(1) > open_p.shift(1)
    prev_bearish = close.shift(1) < open_p.shift(1)
    curr_bullish = close > open_p
    curr_bearish = close < open_p

    features["is_engulfing"] = (
        (c_body > prev_body) &
        (
            (curr_bullish & prev_bearish & (close > open_p.shift(1)) & (open_p < close.shift(1))) |
            (curr_bearish & prev_bullish & (close < open_p.shift(1)) & (open_p > close.shift(1)))
        )
    ).astype(int)

    features["is_inside_candle"] = (
        (high < high.shift(1)) & (low > low.shift(1))
    ).astype(int)
    features["is_outside_candle"] = (
        (high > high.shift(1)) & (low < low.shift(1))
    ).astype(int)

    # 7. Volume Dynamics & OBV
    for d in [5, 20, 50]:
        features[f"volume_avg_{d}d"] = vol.rolling(window=d, min_periods=d).mean()

    features["volume_ratio_20d"] = safe_divide(vol, features["volume_avg_20d"], fill_value=1.0)
    features["volume_ratio_50d"] = safe_divide(vol, features["volume_avg_50d"], fill_value=1.0)

    prev_v1 = vol.shift(1)
    prev_v5 = vol.shift(5)
    features["volume_change_1d"] = safe_divide(vol - prev_v1, prev_v1, fill_value=0.0)
    features["volume_change_5d"] = safe_divide(vol - prev_v5, prev_v5, fill_value=0.0)
    features["volume_spike"] = (vol > (features["volume_avg_20d"] * 2.0)).astype(int)

    price_diff_sign = np.sign(close.diff().fillna(0.0))
    features["obv"] = (price_diff_sign * vol).cumsum()
    features["obv_slope"] = features["obv"].diff().fillna(0.0)

    # 8. Volatility (ATR & Rolling Stdev)
    features["atr14"] = calc_atr(res, 14)
    features["atr20"] = calc_atr(res, 20)
    features["atr_pct"] = safe_divide(features["atr14"], close, fill_value=0.0) * 100.0

    ret_1d_raw = features["return_1d"]
    for d in [5, 10, 20, 50]:
        features[f"volatility_{d}d"] = ret_1d_raw.rolling(window=d, min_periods=d).std().fillna(0.0)

    features["volatility_change"] = features["volatility_10d"].diff().fillna(0.0)

    # 9. Moving Average Structure & Crossovers
    features["ma15_above_ma30"] = (features["ma15"] > features["ma30"]).astype(int)
    features["ma30_above_ma50"] = (features["ma30"] > features["ma50"]).astype(int)
    features["ma50_above_ma100"] = (features["ma50"] > features["ma100"]).astype(int)
    features["ma50_above_ma150"] = (features["ma50"] > features["ma150"]).astype(int)
    features["ma100_above_ma200"] = (features["ma100"] > features["ma200"]).astype(int)
    features["ma150_above_ma200"] = (features["ma150"] > features["ma200"]).astype(int)

    features["bullish_ma_structure"] = (features["ma15_above_ma30"] & features["ma30_above_ma50"]).astype(int)
    features["bearish_ma_structure"] = ((features["ma15"] < features["ma30"]) & (features["ma30"] < features["ma50"])).astype(int)

    ma_cross = (features["ma15"].shift(1) <= features["ma50"].shift(1)) & (features["ma15"] > features["ma50"])
    features["ma_crossover"] = ma_cross.astype(int)
    crossover_id = features["ma_crossover"].cumsum()
    features["days_since_ma_crossover"] = pd.Series(crossover_id).groupby(crossover_id).cumcount()

    # 10. Support & Resistance Proximity
    features["support_distance"] = features["dist_low_20d"]
    features["resistance_distance"] = features["dist_high_20d"]
    features["support_touches"] = (low <= (features["low_20d"] * 1.01)).rolling(20).sum().fillna(0)
    features["resistance_touches"] = (high >= (features["high_20d"] * 0.99)).rolling(20).sum().fillna(0)

    # 11. Market Context (NIFTY 50 and Sector)
    if nifty_df is not None and not nifty_df.empty:
        n_close = nifty_df["close"] if "close" in nifty_df.columns else nifty_df["Close"]
        for d in [1, 5, 7, 20]:
            prev_nc = n_close.shift(d)
            features[f"nifty_return_{d}d"] = safe_divide(n_close - prev_nc, prev_nc, fill_value=0.0) * 100.0

        nifty_ma50 = n_close.rolling(50, min_periods=50).mean()
        nifty_ma200 = n_close.rolling(200, min_periods=200).mean()
        features["nifty_ma50"] = nifty_ma50
        features["nifty_ma200"] = nifty_ma200
        features["nifty_ma50_distance"] = safe_divide(n_close, nifty_ma50, fill_value=1.0) - 1.0
        features["nifty_ma200_distance"] = safe_divide(n_close, nifty_ma200, fill_value=1.0) - 1.0
        features["nifty_rsi"] = calc_rsi(n_close, 14)
        features["nifty_trend"] = (nifty_ma50 > nifty_ma200).astype(int)

        for d in [5, 7, 20]:
            features[f"relative_strength_{d}d"] = features[f"return_{d}d"] - features[f"nifty_return_{d}d"]
    else:
        for d in [1, 5, 7, 20]:
            features[f"nifty_return_{d}d"] = pd.Series(0.0, index=res.index)
        features["nifty_ma50"] = pd.Series(0.0, index=res.index)
        features["nifty_ma200"] = pd.Series(0.0, index=res.index)
        features["nifty_ma50_distance"] = pd.Series(0.0, index=res.index)
        features["nifty_ma200_distance"] = pd.Series(0.0, index=res.index)
        features["nifty_rsi"] = pd.Series(50.0, index=res.index)
        features["nifty_trend"] = pd.Series(0, index=res.index)
        for d in [5, 7, 20]:
            features[f"relative_strength_{d}d"] = features[f"return_{d}d"]

    if sector_df is not None and not sector_df.empty:
        s_close = sector_df["close"] if "close" in sector_df.columns else sector_df["Close"]
        for d in [1, 5, 7, 20]:
            prev_sc = s_close.shift(d)
            features[f"sector_return_{d}d"] = safe_divide(s_close - prev_sc, prev_sc, fill_value=0.0) * 100.0

        sec_ma50 = s_close.rolling(50, min_periods=50).mean()
        sec_ma200 = s_close.rolling(200, min_periods=200).mean()
        features["sector_ma50_distance"] = safe_divide(s_close, sec_ma50, fill_value=1.0) - 1.0
        features["sector_rsi"] = calc_rsi(s_close, 14)
        features["sector_trend"] = (sec_ma50 > sec_ma200).astype(int)

        features["sector_relative_strength_5d"] = features["return_5d"] - features["sector_return_5d"]
        features["sector_relative_strength_20d"] = features["return_20d"] - features["sector_return_20d"]
    else:
        for d in [1, 5, 7, 20]:
            features[f"sector_return_{d}d"] = pd.Series(0.0, index=res.index)
        features["sector_ma50_distance"] = pd.Series(0.0, index=res.index)
        features["sector_rsi"] = pd.Series(50.0, index=res.index)
        features["sector_trend"] = pd.Series(0, index=res.index)
        features["sector_relative_strength_5d"] = features["return_5d"]
        features["sector_relative_strength_20d"] = features["return_20d"]

    # Combine all feature columns efficiently in a single operation
    feat_df = pd.DataFrame(features, index=res.index)
    output_df = pd.concat([res, feat_df], axis=1)

    # Clean any remaining NaNs or Infinities safely
    output_df = output_df.replace([np.inf, -np.inf], 0.0)
    return output_df
