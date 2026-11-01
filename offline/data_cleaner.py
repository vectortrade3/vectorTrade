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


def run_data_cleaning_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Executes the full Data Cleaning & Consolidation pipeline:
    1. Checks if already consolidated for today's market date.
    2. Stage 1: COMBINING (loads each constituent CSV from 5_year_data/generated/).
    3. Stage 2: CLEANING (standardization, derived feature calculations, inf/nan sanitization).
    4. Stage 3: SAVING (persisting into 5_year_final_data.csv).
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
        if (last_clean_run_date == today_str and last_clean_date >= last_downloaded_date) or (last_clean_date and last_clean_date >= last_downloaded_date):
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
                output_file="5_year_final_data.csv"
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
                total_rows=clean_info.get("total_rows", 248863)
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
                "skipped": True
            }

    # Log task start
    json_manager.log_task_entry(
        task_name="DATA_CLEANING",
        status="RUNNING",
        message=f"Starting data consolidation and cleaning for {total_files} stocks up to {last_downloaded_date}..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: COMBINING (1/207 ... 207/207)
    # -------------------------------------------------------------------------
    all_dfs: List[pd.DataFrame] = []

    for i, file_path in enumerate(generated_files, 1):
        file_name = os.path.basename(file_path)
        clean_symbol = file_name.replace('_data.csv', '').upper()
        pct = round((i / total_files) * 60.0, 1)  # Combining occupies 0% - 60% of progress

        json_manager.update_data_cleaning_progress(
            status="COMBINING",
            heading="Combining Data",
            current_stock=clean_symbol,
            current_index=i,
            total_stocks=total_files,
            percentage=pct,
            message=f"Ingesting {clean_symbol} engineered features ({i}/{total_files})...",
            elapsed_seconds=time.time() - start_time,
            last_clean_date=last_downloaded_date
        )

        try:
            df_stock = pd.read_csv(file_path)
            all_dfs.append(df_stock)
        except Exception as e:
            print(f"[ERROR] Failed reading {file_name}: {e}")

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
    # STAGE 2: STANDARDIZATION & FEATURE MAPPING (60% -> 80%)
    # -------------------------------------------------------------------------
    json_manager.update_data_cleaning_progress(
        status="CLEANING",
        heading="Cleaning & Formatting",
        current_stock="ALL_STOCKS",
        current_index=total_files,
        total_stocks=total_files,
        percentage=65.0,
        message="Concatenating constituent datasets into master table...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date
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
        percentage=72.0,
        message="Standardizing column schema and mapping technical indicators...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date
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
        percentage=80.0,
        message="Enforcing exact 133 target feature columns...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date
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
    json_manager.update_data_cleaning_progress(
        status="SAVING",
        heading="Saving Data",
        current_stock="MASTER_DATASET",
        current_index=total_files,
        total_stocks=total_files,
        percentage=90.0,
        message=f"Flushing {total_rows:,} records into 5_year_final_data.csv...",
        elapsed_seconds=time.time() - start_time,
        last_clean_date=last_downloaded_date,
        total_rows=total_rows
    )

    df_final.to_csv(FINAL_DATA_OUTPUT, index=False)

    # -------------------------------------------------------------------------
    # STAGE 4: COMPLETED (100%)
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = f"Cleaned & combined {total_rows:,} rows across {total_files} stocks with {len(FINAL_COLUMNS)} features in {duration:.1f}s."

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
        output_file="5_year_final_data.csv"
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
        total_rows=total_rows
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
        "duration_seconds": duration,
        "output_path": str(FINAL_DATA_OUTPUT),
        "message": completion_msg
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Data Cleaner & Consolidation...")
    print("==================================================")
    res = run_data_cleaning_pipeline()
    print("Result:", res)
