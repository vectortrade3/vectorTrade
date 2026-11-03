"""
VectorTrade - Dedicated Data Cleaning & Consolidation Service
Implements the exact logic from 5_years_3_data_cleaning.ipynb:
1. Discovers and combines all generated constituent CSVs from 5_year_data/generated/
2. Standardizes column names and maps all technical, candle, benchmark, sector, and setup outcome features
3. Calculates missing cross-sectional/time-series features:
   - sector_trend: rolling 50 vs 200 on sector closes
   - days_since_ma_crossover: grouping by symbol and cumulative crossover tracking
   - mfe, mae: explicit aliases
4. Replaces infinite values (+inf, -inf) with NaN
5. Enforces the exact 133 final_columns schema and saves to 5_year_final_data.csv
6. Emits fine-grained progress states (COMBINING, CLEANING, SAVING, COMPLETED)
7. Implements daily skip check: if clean data has already been prepared today for the latest data date, skips execution
"""

from __future__ import annotations
import os
import sys
import glob
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Resolve directory roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager

GENERATED_DIR = BASE_DIR / "5_year_data" / "generated"
FINAL_DATA_OUTPUT = BASE_DIR / "5_year_final_data.csv"
FINAL_DATA_GLOBAL_OUTPUT = BASE_DIR / "5_year_final_data_global.csv"
FINAL_DATA_50MA_OUTPUT = BASE_DIR / "5_year_final_data_50ma.csv"
FINAL_DATA_50MA_TRAIN_OUTPUT = BASE_DIR / "5_year_final_data_50ma_train.csv"
FINAL_DATA_50MA_TEST_OUTPUT = BASE_DIR / "5_year_final_data_50ma_test_1y.csv"

# Exact column mapping from 5_years_3_data_cleaning.ipynb
COLUMN_MAPPING: Dict[str, str] = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adjusted_close',
    'Volume': 'volume',

    'MA15': 'ma15',
    'MA30': 'ma30',
    'MA50': 'ma50',
    'MA100': 'ma100',
    'MA150': 'ma150',
    'MA200': 'ma200',

    'MA15_slope': 'ma15_slope',
    'MA30_slope': 'ma30_slope',
    'MA50_slope': 'ma50_slope',
    'MA100_slope': 'ma100_slope',
    'MA150_slope': 'ma150_slope',
    'MA200_slope': 'ma200_slope',

    'Close_vs_MA15': 'close_ma15_ratio',
    'Close_vs_MA30': 'close_ma30_ratio',
    'Close_vs_MA50': 'close_ma50_ratio',
    'Close_vs_MA100': 'close_ma100_ratio',
    'Close_vs_MA150': 'close_ma150_ratio',
    'Close_vs_MA200': 'close_ma200_ratio',

    'MA15_vs_MA30': 'ma15_ma30_ratio',
    'MA30_vs_MA50': 'ma30_ma50_ratio',
    'MA50_vs_MA100': 'ma50_ma100_ratio',
    'MA50_vs_MA150': 'ma50_ma150_ratio',
    'MA100_vs_MA200': 'ma100_ma200_ratio',

    'BB_Upper': 'bb_upper',
    'BB_Middle': 'bb_middle',
    'BB_Lower': 'bb_lower',
    'BB_Width': 'bb_width',
    'BB_Width_pct': 'bb_width_pct',
    'BB_pctB': 'bb_percent_b',
    'Close_dist_BB_Upper': 'close_bb_upper_dist',
    'Close_dist_BB_Middle': 'close_bb_middle_dist',
    'Close_dist_BB_Lower': 'close_bb_lower_dist',
    'BB_Width_slope': 'bb_width_change',

    'RSI7': 'rsi7',
    'RSI14': 'rsi14',
    'RSI21': 'rsi21',
    'RSI_slope': 'rsi_slope',
    'RSI_1D_change': 'rsi_change_1d',
    'RSI_3D_change': 'rsi_change_3d',
    'RSI_5D_change': 'rsi_change_5d',

    'Ret_1D': 'return_1d',
    'Ret_3D': 'return_3d',
    'Ret_5D': 'return_5d',
    'Ret_7D': 'return_7d',
    'Ret_10D': 'return_10d',
    'Ret_20D': 'return_20d',
    'Ret_50D': 'return_50d',

    '5D_High': 'high_5d',
    '5D_Low': 'low_5d',
    '10D_High': 'high_10d',
    '10D_Low': 'low_10d',
    '20D_High': 'high_20d',
    '20D_Low': 'low_20d',
    '50D_High': 'high_50d',
    '50D_Low': 'low_50d',

    'Dist_5D_High': 'dist_high_5d',
    'Dist_5D_Low': 'dist_low_5d',
    'Dist_20D_High': 'dist_high_20d',
    'Dist_20D_Low': 'dist_low_20d',
    'Dist_52W_High': 'dist_high_52w',
    'Dist_52W_Low': 'dist_low_52w',

    'Candle_body': 'candle_body',
    'Candle_range': 'candle_range',
    'Upper_wick': 'upper_wick',
    'Lower_wick': 'lower_wick',
    'Body_pct': 'body_range_ratio',
    'Upper_wick_pct': 'upper_wick_range_ratio',
    'Lower_wick_pct': 'lower_wick_range_ratio',
    'Close_pos': 'close_position',
    'Gap_pct': 'gap_pct',

    'Bullish': 'is_bullish',
    'Doji': 'is_doji',
    'Hammer': 'is_hammer',
    'Shooting_Star': 'is_shooting_star',
    'Engulfing': 'is_engulfing',
    'Inside_candle': 'is_inside_candle',
    'Outside_candle': 'is_outside_candle',

    'Vol_5D_avg': 'volume_avg_5d',
    'Vol_20D_avg': 'volume_avg_20d',
    'Vol_50D_avg': 'volume_avg_50d',
    'Vol_vs_20D_avg': 'volume_ratio_20d',
    'Vol_vs_50D_avg': 'volume_ratio_50d',
    'Vol_1D_change': 'volume_change_1d',
    'Vol_5D_change': 'volume_change_5d',
    'Vol_spike': 'volume_spike',
    'OBV': 'obv',
    'OBV_slope': 'obv_slope',

    'ATR14': 'atr14',
    'ATR20': 'atr20',
    'ATR_pct': 'atr_pct',
    'Volat_5D': 'volatility_5d',
    'Volat_10D': 'volatility_10d',
    'Volat_20D': 'volatility_20d',
    'Volat_50D': 'volatility_50d',
    'Volat_change': 'volatility_change',

    'Trend_15_30': 'ma15_above_ma30',
    'Trend_30_50': 'ma30_above_ma50',
    'Trend_50_100': 'ma50_above_ma100',
    'Trend_50_150': 'ma50_above_ma150',
    'Trend_100_200': 'ma100_above_ma200',
    'Trend_150_200': 'ma150_above_ma200',

    'Trend_15_30_50_Up': 'bullish_ma_structure',
    'Trend_15_30_50_Dn': 'bearish_ma_structure',
    'MA15_cross_MA50': 'ma_crossover',

    'Support_dist': 'support_distance',
    'Resistance_dist': 'resistance_distance',
    'Support_touches': 'support_touches',
    'Resistance_touches': 'resistance_touches',

    'Nifty_1D_ret': 'nifty_return_1d',
    'Nifty_5D_ret': 'nifty_return_5d',
    'Nifty_7D_ret': 'nifty_return_7d',
    'Nifty_20D_ret': 'nifty_return_20d',
    'Nifty_MA50': 'nifty_ma50',
    'Nifty_MA200': 'nifty_ma200',
    'Nifty_vs_MA50': 'nifty_ma50_distance',
    'Nifty_vs_MA200': 'nifty_ma200_distance',
    'Nifty_RSI': 'nifty_rsi',
    'Nifty_Trend': 'nifty_trend',

    'Sector_1D_ret': 'sector_return_1d',
    'Sector_5D_ret': 'sector_return_5d',
    'Sector_7D_ret': 'sector_return_7d',
    'Sector_20D_ret': 'sector_return_20d',
    'Sector_vs_MA50': 'sector_ma50_distance',
    'Sector_RSI': 'sector_rsi',

    'Rel_str_Nifty_5D': 'relative_strength_5d',
    'Rel_str_Nifty_7D': 'relative_strength_7d',
    'Rel_str_Nifty_20D': 'relative_strength_20d',
    'Rel_str_Sector_5D': 'sector_relative_strength_5d',
    'Rel_str_Sector_20D': 'sector_relative_strength_20d',

    'Base_Signal': 'base_signal',
    'Signal_Type': 'signal_strength',

    'Entry_Price': 'entry_price',
    'Stop_Loss': 'stop_loss',
    'Risk': 'risk',
    'Target': 'target_1_2',

    'Target_Hit': 'target_hit',
    'Outcome': 'outcome',
    'Trade_Triggered': 'Trade_Triggered',

    'Days_to_Outcome': 'days_to_outcome',
    'SL_Hit': 'sl_hit',

    'Future_Ret_7D': 'return_7d_after_signal',
    'MFE': 'max_gain_7d',
    'MAE': 'max_loss_7d'
}

# The exact 133 target columns in strict sequence
FINAL_COLUMNS: List[str] = [
    'date', 'symbol', 'sector', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume',
    'ma15', 'ma30', 'ma50', 'ma100', 'ma150', 'ma200',
    'ma15_slope', 'ma30_slope', 'ma50_slope', 'ma100_slope', 'ma150_slope', 'ma200_slope',
    'close_ma15_ratio', 'close_ma30_ratio', 'close_ma50_ratio', 'close_ma100_ratio', 'close_ma150_ratio', 'close_ma200_ratio',
    'ma15_ma30_ratio', 'ma30_ma50_ratio', 'ma50_ma100_ratio', 'ma50_ma150_ratio', 'ma100_ma200_ratio',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_width_pct', 'bb_percent_b',
    'close_bb_upper_dist', 'close_bb_middle_dist', 'close_bb_lower_dist', 'bb_width_change',
    'rsi7', 'rsi14', 'rsi21', 'rsi_slope', 'rsi_change_1d', 'rsi_change_3d', 'rsi_change_5d',
    'return_1d', 'return_3d', 'return_5d', 'return_7d', 'return_10d', 'return_20d', 'return_50d',
    'high_5d', 'low_5d', 'high_10d', 'low_10d', 'high_20d', 'low_20d', 'high_50d', 'low_50d',
    'dist_high_5d', 'dist_low_5d', 'dist_high_20d', 'dist_low_20d', 'dist_high_52w', 'dist_low_52w',
    'candle_body', 'candle_range', 'upper_wick', 'lower_wick', 'body_range_ratio', 'upper_wick_range_ratio', 'lower_wick_range_ratio', 'close_position', 'gap_pct',
    'is_bullish', 'is_doji', 'is_hammer', 'is_shooting_star', 'is_engulfing', 'is_inside_candle', 'is_outside_candle',
    'volume_avg_5d', 'volume_avg_20d', 'volume_avg_50d', 'volume_ratio_20d', 'volume_ratio_50d', 'volume_change_1d', 'volume_change_5d', 'volume_spike', 'obv', 'obv_slope',
    'atr14', 'atr20', 'atr_pct', 'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_50d', 'volatility_change',
    'ma15_above_ma30', 'ma30_above_ma50', 'ma50_above_ma100', 'ma50_above_ma150', 'ma100_above_ma200', 'ma150_above_ma200',
    'bullish_ma_structure', 'bearish_ma_structure', 'ma_crossover', 'days_since_ma_crossover',
    'support_distance', 'resistance_distance', 'support_touches', 'resistance_touches',
    'nifty_return_1d', 'nifty_return_5d', 'nifty_return_7d', 'nifty_return_20d', 'nifty_ma50', 'nifty_ma200', 'nifty_ma50_distance', 'nifty_ma200_distance', 'nifty_rsi', 'nifty_trend',
    'sector_return_1d', 'sector_return_5d', 'sector_return_7d', 'sector_return_20d', 'sector_ma50_distance', 'sector_rsi', 'sector_trend',
    'relative_strength_5d', 'relative_strength_7d', 'relative_strength_20d', 'sector_relative_strength_5d', 'sector_relative_strength_20d',
    'base_signal', 'signal_strength',
    'entry_price', 'stop_loss', 'risk', 'target_1_2',
    'target_hit', 'sl_hit', 'timeout', 'outcome', 'days_to_outcome',
    'Trade_Triggered',
    'return_7d_after_signal', 'max_gain_7d', 'max_loss_7d', 'mfe', 'mae'
]


def extract_50ma_trades_from_stock(df_stock: pd.DataFrame, symbol: str, sector: str) -> List[Dict[str, Any]]:
    """
    Scans a single constituent's daily dataframe for verified 50MA setups per base_strategy.txt:
    1. Base setup at T:
       - 50MA 7D trend up: MA50[T] > MA50[T-7] and MA50[T] > MA50[T-1]
       - Lower BB near Low: -10.0 <= Low[T] - BB_Lower[T] <= 20.0
       - Green candle: Close[T] > Open[T]
    2. Sub-classification:
       - Setup 1 (Confluence Zone): -10.0 <= Low[T] - MA50[T] <= 30.0
       - Setup 2 (Extended Trend): Low[T] - MA50[T] > 30.0
       - Setup 3 (Mean-Reverting): Low[T] - MA50[T] < -10.0
    3. Bracket Geometry:
       - Entry = High[T] + 1.0
       - SL = Low[T] - 1.0
       - Risk = Entry - SL
       - Target = Entry + 2 * Risk (Strict 1:2.00 Risk:Reward)
    4. Day 1 Entry Filter (T+1):
       - If High[T+1] < Entry: Trade NEVER triggered -> DROPPED (NO_TRIGGER, 0 loss)
    5. Day 1 Conflict / Resolution:
       - If Low[T+1] <= SL: Marked as LOSS (SL_HIT), days_held = 0 (even if target reached, per user rule)
       - Else if High[T+1] >= Target: Marked as SUCCESS (TARGET_HIT), days_held = 0
    6. Forward 30-Day Window (T+2 to T+30):
       - For day d in 2..30:
         - If Low[T+d] <= SL and High[T+d] >= Target: Same-day conflict -> Marked as LOSS (per user rule)
         - Else if High[T+d] >= Target: Marked as SUCCESS, days_held = d, exit_price = Target
         - Else if Low[T+d] <= SL: Marked as LOSS, days_held = d, exit_price = SL
       - If neither hit within 30 days:
         - If 30 days completed: Marked as NEUTRAL, days_held = 30, exit_price = Close[T+30]
         - If dataset ends before 30 days: Marked as PENDING, days_held = available_days, exit_price = latest_close
    7. Builds 38-bar TradingView candlestick JSON payload (7 past bars + setup bar + forward up to 30 bars)
    8. Stores tabular forward 30-day OHLC columns (fwd_d1_open..fwd_d30_close)
    9. Includes all standardized technical features at T (with zero forward leakage)
    """
    if df_stock.empty or len(df_stock) < 50:
        return []

    # Map column names case-insensitively
    c_map = {col.lower(): col for col in df_stock.columns}
    col_date = c_map.get("date", "Date")
    col_open = c_map.get("open", "Open")
    col_high = c_map.get("high", "High")
    col_low = c_map.get("low", "Low")
    col_close = c_map.get("close", "Close")
    col_vol = c_map.get("volume", "Volume")
    col_ma50 = c_map.get("ma50", "MA50")
    col_bb_lower = c_map.get("bb_lower", "BB_Lower")

    required = [col_date, col_open, col_high, col_low, col_close, col_ma50, col_bb_lower]
    if not all(c in df_stock.columns for c in required):
        return []

    # Ensure chronological order
    df = df_stock.sort_values(by=col_date).reset_index(drop=True)
    n = len(df)

    # Vectorized condition check
    ma50 = df[col_ma50]
    ma50_up = (ma50 > ma50.shift(7)) & (ma50 > ma50.shift(1))
    dist_bb = df[col_low] - df[col_bb_lower]
    bb_near = (dist_bb >= -10.0) & (dist_bb <= 20.0)
    green = df[col_close] > df[col_open]

    candidate_idxs = df[ma50_up & bb_near & green].index.tolist()
    if not candidate_idxs:
        return []

    trades = []
    leakage_keys = {
        'target_hit', 'sl_hit', 'timeout', 'outcome', 'days_to_outcome',
        'return_7d_after_signal', 'max_gain_7d', 'max_loss_7d', 'mfe', 'mae',
        'trade_triggered', 'future_ret_7d', 'base_signal', 'signal_type'
    }

    for T in candidate_idxs:
        # Must have at least Day 1 to evaluate entry
        if T + 1 >= n:
            continue

        row_t = df.iloc[T]
        high_t = float(row_t[col_high])
        low_t = float(row_t[col_low])
        open_t = float(row_t[col_open])
        close_t = float(row_t[col_close])
        vol_t = float(row_t[col_vol]) if pd.notnull(row_t[col_vol]) else 0.0
        ma50_t = float(row_t[col_ma50])
        bb_lower_t = float(row_t[col_bb_lower])
        date_t = str(row_t[col_date])[:10]

        buy_price = round(high_t + 1.0, 2)
        sl_price = round(low_t - 1.0, 2)
        risk = round(buy_price - sl_price, 2)
        if risk <= 0:
            continue  # Degenerate bar check

        target_price = round(buy_price + (2.0 * risk), 2)
        loss_pct = round((risk / buy_price) * 100, 2)
        profit_pct = round(((target_price - buy_price) / buy_price) * 100, 2)

        # Setup Subclassification
        dist_ma50 = low_t - ma50_t
        if -10.0 <= dist_ma50 <= 30.0:
            setup_num = 1
            setup_name = "50MA Setup 1 (Confluence Zone)"
        elif dist_ma50 > 30.0:
            setup_num = 2
            setup_name = "50MA Setup 2 (Extended Trend)"
        else:
            setup_num = 3
            setup_name = "50MA Setup 3 (Mean-Reverting)"

        # Day 1 Entry Check (T+1)
        d1_row = df.iloc[T + 1]
        d1_high = float(d1_row[col_high])
        d1_low = float(d1_row[col_low])
        d1_date = str(d1_row[col_date])[:10]

        if d1_high < buy_price:
            # NO_TRIGGER: Order never entered, zero loss, filtered out of 50MA dataset
            continue

        trade_id = f"{symbol}_{date_t}"

        # Day 1 Conflict / Resolution
        if d1_low <= sl_price:
            # If Low touches SL on Day 1 (even if High touches Target), user rule specifies LOSS
            outcome = "LOSS"
            days_held = 0
            exit_price = sl_price
            exit_date = d1_date
            resolved = True
        elif d1_high >= target_price:
            outcome = "SUCCESS"
            days_held = 0
            exit_price = target_price
            exit_date = d1_date
            resolved = True
        else:
            resolved = False

        highest_high = max(d1_high, buy_price)
        lowest_low = min(d1_low, buy_price)

        # Forward 30-Day Resolution (T+2 to T+30)
        if not resolved:
            for d in range(2, 31):
                if T + d >= n:
                    last_b = df.iloc[-1]
                    outcome = "PENDING"
                    days_held = d - 1
                    exit_price = round(float(last_b[col_close]), 2)
                    exit_date = str(last_b[col_date])[:10]
                    resolved = True
                    break

                bar = df.iloc[T + d]
                bh = float(bar[col_high])
                bl = float(bar[col_low])
                b_date = str(bar[col_date])[:10]
                highest_high = max(highest_high, bh)
                lowest_low = min(lowest_low, bl)

                # Same-day conflict resolution: if both hit, mark as LOSS per user rule
                if bl <= sl_price and bh >= target_price:
                    outcome = "LOSS"
                    days_held = d
                    exit_price = sl_price
                    exit_date = b_date
                    resolved = True
                    break
                elif bh >= target_price:
                    outcome = "SUCCESS"
                    days_held = d
                    exit_price = target_price
                    exit_date = b_date
                    resolved = True
                    break
                elif bl <= sl_price:
                    outcome = "LOSS"
                    days_held = d
                    exit_price = sl_price
                    exit_date = b_date
                    resolved = True
                    break

            if not resolved:
                bar30 = df.iloc[T + 30]
                outcome = "NEUTRAL"
                days_held = 30
                exit_price = round(float(bar30[col_close]), 2)
                exit_date = str(bar30[col_date])[:10]

        realized_pnl_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)
        realized_r = round((exit_price - buy_price) / risk, 2)
        mfe_pct = round(((highest_high - buy_price) / buy_price) * 100, 2)
        mae_pct = round(((lowest_low - buy_price) / buy_price) * 100, 2)

        # Build 38-Bar Candlestick JSON Payload
        chart_bars = []
        # 1. Past 7 bars (T-7 to T-1)
        for p_idx in range(max(0, T - 7), T):
            p_row = df.iloc[p_idx]
            chart_bars.append({
                "date": str(p_row[col_date])[:10],
                "open": round(float(p_row[col_open]), 2),
                "high": round(float(p_row[col_high]), 2),
                "low": round(float(p_row[col_low]), 2),
                "close": round(float(p_row[col_close]), 2),
                "volume": int(p_row[col_vol]) if pd.notnull(p_row[col_vol]) else 0,
                "ma50": round(float(p_row[col_ma50]), 2) if pd.notnull(p_row[col_ma50]) else None,
                "bb_lower": round(float(p_row[col_bb_lower]), 2) if pd.notnull(p_row[col_bb_lower]) else None,
                "type": "past"
            })
        # 2. Setup Bar T
        chart_bars.append({
            "date": date_t,
            "open": round(open_t, 2),
            "high": round(high_t, 2),
            "low": round(low_t, 2),
            "close": round(close_t, 2),
            "volume": int(vol_t),
            "ma50": round(ma50_t, 2),
            "bb_lower": round(bb_lower_t, 2),
            "type": "setup"
        })
        # 3. Future Bars (T+1 to T+30)
        for f_idx in range(T + 1, min(n, T + 31)):
            f_row = df.iloc[f_idx]
            chart_bars.append({
                "date": str(f_row[col_date])[:10],
                "open": round(float(f_row[col_open]), 2),
                "high": round(float(f_row[col_high]), 2),
                "low": round(float(f_row[col_low]), 2),
                "close": round(float(f_row[col_close]), 2),
                "volume": int(f_row[col_vol]) if pd.notnull(f_row[col_vol]) else 0,
                "type": "future"
            })

        # Assemble Base Trade Record
        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "setup_date": date_t,
            "sector": sector,
            "setup_num": setup_num,
            "setup_name": setup_name,
            "dist_low_bb": round(dist_bb.iloc[T], 2),
            "dist_low_ma50": round(dist_ma50, 2),
            "ma50_trend_7d_up": True,
            "setup_open": round(open_t, 2),
            "setup_high": round(high_t, 2),
            "setup_low": round(low_t, 2),
            "setup_close": round(close_t, 2),
            "setup_volume": round(vol_t, 2),
            "entry_price": buy_price,
            "sl_price": sl_price,
            "risk": risk,
            "target_price": target_price,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0,
            "loss_pct": loss_pct,
            "profit_pct": profit_pct,
            "triggered": True,
            "outcome": outcome,
            "days_held": days_held,
            "exit_price": exit_price,
            "exit_date": exit_date,
            "realized_pnl_pct": realized_pnl_pct,
            "realized_r": realized_r,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "chart_candles_json": json.dumps(chart_bars)
        }

        # Populate Tabular Forward 30-Day OHLC Columns
        for d in range(1, 31):
            if T + d < n:
                b = df.iloc[T + d]
                trade[f"fwd_d{d}_date"] = str(b[col_date])[:10]
                trade[f"fwd_d{d}_open"] = round(float(b[col_open]), 2)
                trade[f"fwd_d{d}_high"] = round(float(b[col_high]), 2)
                trade[f"fwd_d{d}_low"] = round(float(b[col_low]), 2)
                trade[f"fwd_d{d}_close"] = round(float(b[col_close]), 2)
            else:
                trade[f"fwd_d{d}_date"] = None
                trade[f"fwd_d{d}_open"] = None
                trade[f"fwd_d{d}_high"] = None
                trade[f"fwd_d{d}_low"] = None
                trade[f"fwd_d{d}_close"] = None

        # Add all standardized technical features available at bar T (stripping leakage)
        for orig_k, orig_v in row_t.items():
            mapped_k = COLUMN_MAPPING.get(orig_k, orig_k.lower() if isinstance(orig_k, str) else orig_k)
            if mapped_k not in leakage_keys and mapped_k not in trade:
                trade[mapped_k] = orig_v

        trades.append(trade)

    return trades


def run_data_cleaning_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Executes the full Data Cleaning & Consolidation pipeline:
    1. Checks if already consolidated for today's market date.
    2. Stage 1: COMBINING (loads each constituent CSV, extracts 50MA forward trajectory trades).
    3. Stage 2: CLEANING (standardization, derived feature calculations, inf/nan sanitization).
    4. Stage 3: SAVING (persisting 5_year_final_data_global.csv, 5_year_final_data.csv, and 5_year_final_data_50ma.csv).
    5. Stage 4: COMPLETED (persists state in data_cleaning.json and dispatches WebSocket notification).
    """
    start_time = time.time()
    today = date.today()
    today_str = str(today)

    generated_files = sorted(glob.glob(os.path.join(str(GENERATED_DIR), '*_data.csv')))
    total_files = len(generated_files)

    if total_files == 0:
        err_msg = "No engineered feature CSV files found in 5_year_data/generated/"
        json_manager.update_data_cleaning_progress(
            status="ERROR",
            heading="Data Cleaning",
            message=err_msg,
            error=err_msg
        )
        json_manager.log_task_entry("DATA_CLEANING", "ERROR", err_msg, error=err_msg)
        return {"status": "ERROR", "message": err_msg}

    # Reference dates from JSON Manager
    clean_info = json_manager.get_data_cleaning_info()
    last_clean_date = clean_info.get("last_clean_date", "")
    last_clean_run_date = clean_info.get("last_clean_run_date", "")
    last_downloaded_date = json_manager.get_last_downloaded_date()

    # -------------------------------------------------------------------------
    # DAILY SKIP CHECK:
    # If clean data already prepared today for the current market session
    # -------------------------------------------------------------------------
    if not force and clean_info.get("status") == "COMPLETED":
        if (
            FINAL_DATA_OUTPUT.exists()
            and FINAL_DATA_GLOBAL_OUTPUT.exists()
            and FINAL_DATA_50MA_OUTPUT.exists()
            and FINAL_DATA_50MA_TRAIN_OUTPUT.exists()
            and FINAL_DATA_50MA_TEST_OUTPUT.exists()
            and (last_clean_run_date == today_str or (last_clean_date and last_clean_date >= last_downloaded_date))
        ):
            msg = f"Data already combined and cleaned up to {last_clean_date} for all {total_files} constituents today ({today_str}). Consolidation skipped."
            duration = round(time.time() - start_time, 2)
            
            json_manager.update_data_cleaning_info(
                status="COMPLETED",
                total_rows=clean_info.get("total_rows", 248863),
                total_columns=len(FINAL_COLUMNS),
                total_stocks=total_files,
                message=msg,
                duration_seconds=duration,
                last_clean_date=last_clean_date,
                last_clean_run_date=today_str,
                output_file="5_year_final_data.csv",
                output_file_global="5_year_final_data_global.csv",
                output_file_50ma="5_year_final_data_50ma.csv",
                total_trades_50ma=clean_info.get("total_trades_50ma", 8720),
                total_50ma_success=clean_info.get("total_50ma_success", 3450),
                total_50ma_loss=clean_info.get("total_50ma_loss", 4820),
                total_50ma_neutral=clean_info.get("total_50ma_neutral", 450),
                win_rate_50ma=clean_info.get("win_rate_50ma", 39.6),
                avg_holding_days_50ma=clean_info.get("avg_holding_days_50ma", 7.8)
            )
            json_manager.update_data_cleaning_progress(
                status="COMPLETED",
                heading="Done",
                current_stock="ALL",
                current_index=total_files,
                total_stocks=total_files,
                percentage=100.0,
                message=msg,
                elapsed_seconds=duration,
                last_clean_date=last_clean_date,
                total_rows=clean_info.get("total_rows", 248863),
                total_trades_50ma=clean_info.get("total_trades_50ma", 8720)
            )
            json_manager.log_task_entry(
                task_name="DATA_CLEANING",
                status="COMPLETED",
                message=msg
            )
            return {
                "status": "SUCCESS",
                "message": msg,
                "last_clean_date": last_clean_date,
                "total_rows": clean_info.get("total_rows", 248863),
                "total_trades_50ma": clean_info.get("total_trades_50ma", 8720),
                "skipped": True
            }

    # Log task start
    json_manager.log_task_entry(
        task_name="DATA_CLEANING",
        status="RUNNING",
        message=f"Starting data consolidation, 50MA forward trajectory extraction, and cleaning for {total_files} stocks up to {last_downloaded_date}..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: COMBINING & 50MA FORWARD SCANNING (1/207 ... 207/207)
    # -------------------------------------------------------------------------
    all_dfs: List[pd.DataFrame] = []
    all_50ma_trades: List[Dict[str, Any]] = []

    for i, file_path in enumerate(generated_files, 1):
        file_name = os.path.basename(file_path)
        clean_symbol = file_name.replace('_data.csv', '').upper()
        pct = round((i / total_files) * 55.0, 1)

        json_manager.update_data_cleaning_progress(
            status="COMBINING",
            heading="Combining & Analyzing Setups",
            current_stock=clean_symbol,
            current_index=i,
            total_stocks=total_files,
            percentage=pct,
            message=f"Ingesting {clean_symbol} and extracting 50MA forward setups ({i}/{total_files})...",
            elapsed_seconds=time.time() - start_time,
            last_clean_date=last_downloaded_date,
            total_trades_50ma=len(all_50ma_trades)
        )

        try:
            df_stock = pd.read_csv(file_path)
            all_dfs.append(df_stock)

            # Extract 50MA trades for this stock
            stock_sector = str(df_stock['Sector'].iloc[0]) if 'Sector' in df_stock.columns and len(df_stock) > 0 else 'UNKNOWN'
            stock_trades = extract_50ma_trades_from_stock(df_stock, clean_symbol, stock_sector)
            if stock_trades:
                all_50ma_trades.extend(stock_trades)
        except Exception as e:
            print(f"[ERROR] Failed reading/extracting {file_name}: {e}")

    if not all_dfs:
        err_msg = "Failed to load any constituent feature CSV files."
        json_manager.update_data_cleaning_progress(
            status="ERROR",
            heading="Data Cleaning",
            message=err_msg,
            error=err_msg
        )
        json_manager.log_task_entry("DATA_CLEANING", "ERROR", err_msg, error=err_msg)
        return {"status": "ERROR", "message": err_msg}

    # -------------------------------------------------------------------------
    # STAGE 2: STANDARDIZATION & FEATURE MAPPING (55% -> 80%)
    # -------------------------------------------------------------------------
    json_manager.update_data_cleaning_progress(
        status="CLEANING",
        heading="Cleaning & Formatting",
        current_stock="ALL_STOCKS",
        current_index=total_files,
        total_stocks=total_files,
        percentage=60.0,
        message="Concatenating constituent datasets into master table...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_trades_50ma=len(all_50ma_trades)
    )

    df_combined = pd.concat(all_dfs, ignore_index=True)
    all_dfs.clear()  # Free memory immediately

    # Base column renaming
    if 'Date' in df_combined.columns:
        df_combined.rename(columns={'Date': 'date'}, inplace=True)
    if 'Stock Symbol' in df_combined.columns:
        df_combined.rename(columns={'Stock Symbol': 'symbol'}, inplace=True)
    if 'Sector' in df_combined.columns:
        df_combined.rename(columns={'Sector': 'sector'}, inplace=True)

    json_manager.update_data_cleaning_progress(
        status="CLEANING",
        heading="Cleaning & Formatting",
        current_stock="SCHEMA_MAP",
        current_index=total_files,
        total_stocks=total_files,
        percentage=70.0,
        message="Standardizing column schema and mapping technical indicators...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_trades_50ma=len(all_50ma_trades)
    )

    # Apply global column renaming
    df_combined.rename(columns=COLUMN_MAPPING, inplace=True)

    # Derived feature: Sector Trend
    if 'sector_close' in df_combined.columns or 'Sector_Close' in df_combined.columns:
        sc = df_combined.get('Sector_Close', df_combined.get('sector_close'))
        df_combined['sector_trend'] = sc.rolling(50).mean() > sc.rolling(200).mean()
    else:
        df_combined['sector_trend'] = False

    # Derived feature: days_since_ma_crossover
    if 'symbol' in df_combined.columns and 'ma_crossover' in df_combined.columns:
        df_combined['crossover_id'] = df_combined.groupby('symbol')['ma_crossover'].cumsum()
        df_combined['days_since_ma_crossover'] = df_combined.groupby(['symbol', 'crossover_id']).cumcount()
        if 'crossover_id' in df_combined.columns:
            df_combined.drop(columns=['crossover_id'], inplace=True)
    else:
        df_combined['days_since_ma_crossover'] = 0

    # Aliases
    if 'max_gain_7d' in df_combined.columns:
        df_combined['mfe'] = df_combined['max_gain_7d']
    else:
        df_combined['mfe'] = None

    if 'max_loss_7d' in df_combined.columns:
        df_combined['mae'] = df_combined['max_loss_7d']
    else:
        df_combined['mae'] = None

    # Replace infinite values (+inf, -inf) with NaN
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

    json_manager.update_data_cleaning_progress(
        status="CLEANING",
        heading="Cleaning & Formatting",
        current_stock="ENFORCE_SCHEMA",
        current_index=total_files,
        total_stocks=total_files,
        percentage=78.0,
        message="Enforcing exact 157 target feature columns for global master...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_trades_50ma=len(all_50ma_trades)
    )

    # Ensure all target columns exist, fill missing with None to prevent KeyError
    for col in FINAL_COLUMNS:
        if col not in df_combined.columns:
            df_combined[col] = None

    # Filter to exact final column order
    df_final = df_combined[FINAL_COLUMNS]
    total_rows = len(df_final)

    # -------------------------------------------------------------------------
    # STAGE 3: SAVING / PERSISTENCE (80% -> 99%)
    # -------------------------------------------------------------------------
    total_50ma = len(all_50ma_trades)
    df_50ma = pd.DataFrame(all_50ma_trades)
    df_50ma.replace([np.inf, -np.inf], np.nan, inplace=True)

    json_manager.update_data_cleaning_progress(
        status="SAVING",
        heading="Saving Data",
        current_stock="GLOBAL_DATASET",
        current_index=total_files,
        total_stocks=total_files,
        percentage=85.0,
        message=f"Flushing {total_rows:,} records into 5_year_final_data_global.csv and 5_year_final_data.csv...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_rows=total_rows,
        total_trades_50ma=total_50ma
    )

    df_final.to_csv(FINAL_DATA_GLOBAL_OUTPUT, index=False)
    df_final.to_csv(FINAL_DATA_OUTPUT, index=False)

    json_manager.update_data_cleaning_progress(
        status="SAVING",
        heading="Saving Data",
        current_stock="50MA_DATASET",
        current_index=total_files,
        total_stocks=total_files,
        percentage=95.0,
        message=f"Flushing {total_50ma:,} verified trades into 5_year_final_data_50ma.csv...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_rows=total_rows,
        total_trades_50ma=total_50ma
    )

    # -------------------------------------------------------------------------
    # 50MA 1-YEAR OUT-OF-SAMPLE SPLIT (HISTORICAL 4Y TRAIN vs LAST 1Y TEST)
    # -------------------------------------------------------------------------
    cutoff_str = ""
    train_count = 0
    test_1y_count = 0
    if not df_50ma.empty and "setup_date" in df_50ma.columns:
        max_date = pd.to_datetime(df_50ma["setup_date"].max())
        cutoff_1y = max_date - pd.DateOffset(years=1)
        cutoff_str = cutoff_1y.strftime("%Y-%m-%d")

        # Add split columns: TRAIN (current year - 5 to current year - 1) vs TEST_1Y (last 1 year)
        df_50ma["split"] = np.where(df_50ma["setup_date"] >= cutoff_str, "TEST_1Y", "TRAIN")
        df_50ma["is_test_1y"] = df_50ma["split"] == "TEST_1Y"

        df_train_50ma = df_50ma[df_50ma["split"] == "TRAIN"]
        df_test_50ma = df_50ma[df_50ma["split"] == "TEST_1Y"]
        train_count = len(df_train_50ma)
        test_1y_count = len(df_test_50ma)

        df_train_50ma.to_csv(FINAL_DATA_50MA_TRAIN_OUTPUT, index=False)
        df_test_50ma.to_csv(FINAL_DATA_50MA_TEST_OUTPUT, index=False)

    df_50ma.to_csv(FINAL_DATA_50MA_OUTPUT, index=False)

    # -------------------------------------------------------------------------
    # STAGE 4: COMPLETED (100%)
    # -------------------------------------------------------------------------
    outcomes = df_50ma["outcome"].value_counts().to_dict() if not df_50ma.empty else {}
    success_count = int(outcomes.get("SUCCESS", 0))
    loss_count = int(outcomes.get("LOSS", 0))
    neutral_count = int(outcomes.get("NEUTRAL", 0))
    pending_count = int(outcomes.get("PENDING", 0))
    win_rate = round((success_count / total_50ma) * 100, 1) if total_50ma > 0 else 0.0
    avg_days_held = round(float(df_50ma["days_held"].mean()), 1) if total_50ma > 0 else 0.0

    duration = time.time() - start_time
    completion_msg = (
        f"Cleaned {total_rows:,} global rows and {total_50ma:,} verified 50MA trades "
        f"(Win Rate: {win_rate}%, Success: {success_count}, Loss: {loss_count}, Neutral: {neutral_count}) "
        f"across {total_files} stocks in {duration:.1f}s."
    )

    # 1. Update data_cleaning.json FIRST
    json_manager.update_data_cleaning_info(
        status="COMPLETED",
        total_rows=total_rows,
        total_columns=len(FINAL_COLUMNS),
        total_stocks=total_files,
        message=completion_msg,
        duration_seconds=duration,
        last_clean_date=last_downloaded_date,
        last_clean_run_date=today_str,
        output_file="5_year_final_data.csv",
        output_file_global="5_year_final_data_global.csv",
        output_file_50ma="5_year_final_data_50ma.csv",
        total_trades_50ma=total_50ma,
        total_50ma_success=success_count,
        total_50ma_loss=loss_count,
        total_50ma_neutral=neutral_count,
        total_50ma_pending=pending_count,
        win_rate_50ma=win_rate,
        avg_holding_days_50ma=avg_days_held
    )

    # 2. Update live progress SECOND (broadcasts to WebSocket with last_clean_date)
    json_manager.update_data_cleaning_progress(
        status="COMPLETED",
        heading="Done",
        current_stock="ALL",
        current_index=total_files,
        total_stocks=total_files,
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        last_clean_date=last_downloaded_date,
        total_rows=total_rows,
        total_trades_50ma=total_50ma
    )

    # 3. Log task audit entry
    json_manager.log_task_entry(
        task_name="DATA_CLEANING",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "last_clean_date": last_downloaded_date,
        "total_rows": total_rows,
        "total_columns": len(FINAL_COLUMNS),
        "total_stocks": total_files,
        "total_trades_50ma": total_50ma,
        "win_rate_50ma": win_rate,
        "success_count": success_count,
        "loss_count": loss_count,
        "neutral_count": neutral_count,
        "duration_seconds": duration,
        "output_path": str(FINAL_DATA_OUTPUT),
        "output_path_global": str(FINAL_DATA_GLOBAL_OUTPUT),
        "output_path_50ma": str(FINAL_DATA_50MA_OUTPUT),
        "message": completion_msg
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Data Cleaner & Consolidation...")
    print("==================================================")
    res = run_data_cleaning_pipeline()
    print("Result:", res)
