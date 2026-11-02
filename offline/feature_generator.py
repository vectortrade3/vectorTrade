"""
VectorTrade - Feature Generation Engine
Implements the exact technical indicator algorithms and quantitative trade setup
forward simulation from 5_years_2_feature_generation.ipynb:
- Moving Averages (MA15, MA30, MA50, MA100, MA150, MA200, slopes, ratios)
- Bollinger Bands (Upper, Middle, Lower, Width, %B, distances, slope)
- Multi-period RSI (7, 14, 21), slopes, thresholds (30, 40, 50, 70), crossovers
- Multi-horizon Price Returns (1D, 3D, 5D, 7D, 10D, 20D, 50D)
- Multi-window Price Structures & Distances (5D, 10D, 20D, 50D, 52W Highs/Lows)
- Candle Dynamics (Body %, Upper/Lower Wick %, Position, Gap, Doji, Hammer, Engulfing)
- Volume Profiles (5D, 20D, 50D averages, ratios, changes, spikes, OBV & slope)
- Volatility Models (ATR14, ATR20, ATR %, Historical Volatility across horizons)
- Trend Structure & Composite Bullish Alignment Counts
- Support & Resistance Distance & Touch Frequency (20D window)
- NIFTY 50 Benchmark Returns, Moving Averages, RSI, Relative Strength (5D, 7D, 20D)
- Sector Cross-Sectional Indexing, Relative Strength vs Sector
- Base 50MA Pullback Quantitative Trade Setup Forward Simulation (Day 1 trigger, Day 2-7 MFE/MAE/SL/Target)
- Output Destination: 5_year_data/generated/{symbol}_data.csv
"""

from __future__ import annotations
import os
import sys
import glob
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Resolve directory roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager

RAW_DIR = BASE_DIR / "5_year_data" / "raw"
GENERATED_DIR = BASE_DIR / "5_year_data" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. CORE TECHNICAL INDICATOR ALGORITHMS
# ==============================================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes relative strength index across specified lookback window."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes standard average true range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def calculate_sector_metrics(raw_dir: Path | str) -> Optional[pd.DataFrame]:
    """Pre-calculates cross-sectional sector mean closes across all raw constituent CSVs."""
    files = glob.glob(os.path.join(str(raw_dir), '*_data.csv'))
    sector_data = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['Date', 'Sector', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            sector_data.append(df)
        except Exception:
            pass
    if not sector_data:
        return None

    all_data = pd.concat(sector_data, ignore_index=True)
    sector_mean = all_data.groupby(['Date', 'Sector'])['Close'].mean().reset_index()
    sector_pivot = sector_mean.pivot(index='Date', columns='Sector', values='Close')
    return sector_pivot


# ==============================================================================
# 2. FULL FEATURE GENERATION & FORWARD SIMULATION
# ==============================================================================

def calculate_features(df: pd.DataFrame, sector_pivot: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Computes all quantitative features and forward-simulates setup outcomes.
    Exact implementation from 5_years_2_feature_generation.ipynb.
    """
    df = df.copy()

    # 1. Moving Averages
    mas = [15, 30, 50, 100, 150, 200]
    for ma in mas:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'MA{ma}_slope'] = df[f'MA{ma}'].diff(1) / df[f'MA{ma}'].shift(1) * 100
        df[f'Close_vs_MA{ma}'] = df['Close'] / df[f'MA{ma}']

    df['MA15_vs_MA30'] = df['MA15'] / df['MA30']
    df['MA30_vs_MA50'] = df['MA30'] / df['MA50']
    df['MA50_vs_MA100'] = df['MA50'] / df['MA100']
    df['MA50_vs_MA150'] = df['MA50'] / df['MA150']
    df['MA100_vs_MA200'] = df['MA100'] / df['MA200']

    # 2. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Width_pct'] = df['BB_Width'] / df['BB_Middle']
    df['BB_pctB'] = (df['Close'] - df['BB_Lower']) / df['BB_Width'].replace(0, np.nan)

    df['Close_dist_BB_Upper'] = df['Close'] - df['BB_Upper']
    df['Close_dist_BB_Middle'] = df['Close'] - df['BB_Middle']
    df['Close_dist_BB_Lower'] = df['Close'] - df['BB_Lower']
    df['BB_Width_slope'] = df['BB_Width'].diff()

    # 3. RSI / Momentum
    for p in [7, 14, 21]:
        df[f'RSI{p}'] = calc_rsi(df['Close'], period=p)

    df['RSI_slope'] = df['RSI14'].diff()
    df['RSI_1D_change'] = df['RSI14'].diff(1)
    df['RSI_3D_change'] = df['RSI14'].diff(3)
    df['RSI_5D_change'] = df['RSI14'].diff(5)

    df['RSI_lt_30'] = df['RSI14'] < 30
    df['RSI_lt_40'] = df['RSI14'] < 40
    df['RSI_gt_50'] = df['RSI14'] > 50
    df['RSI_gt_70'] = df['RSI14'] > 70

    df['RSI_cross_30'] = (df['RSI14'].shift(1) < 30) & (df['RSI14'] >= 30)
    df['RSI_cross_50'] = (df['RSI14'].shift(1) < 50) & (df['RSI14'] >= 50)
    df['RSI_cross_70'] = (df['RSI14'].shift(1) < 70) & (df['RSI14'] >= 70)

    # 4. Price Returns
    for d in [1, 3, 5, 7, 10, 20, 50]:
        df[f'Ret_{d}D'] = df['Close'].pct_change(d) * 100

    # 5. Recent Price Structure
    for d in [5, 10, 20, 50, 252]:
        label = f'{d}D' if d != 252 else '52W'
        df[f'{label}_High'] = df['High'].rolling(d).max()
        df[f'{label}_Low'] = df['Low'].rolling(d).min()
        df[f'Dist_{label}_High'] = df['Close'] / df[f'{label}_High'] - 1
        df[f'Dist_{label}_Low'] = df['Close'] / df[f'{label}_Low'] - 1

    # 6. Candle Features
    df['Candle_body'] = np.abs(df['Close'] - df['Open'])
    df['Candle_range'] = df['High'] - df['Low']
    df['Upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    cr_safe = df['Candle_range'].replace(0, np.nan)
    df['Body_pct'] = df['Candle_body'] / cr_safe
    df['Upper_wick_pct'] = df['Upper_wick'] / cr_safe
    df['Lower_wick_pct'] = df['Lower_wick'] / cr_safe
    df['Close_pos'] = (df['Close'] - df['Low']) / cr_safe

    df['Gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan) * 100
    df['Bullish'] = df['Close'] > df['Open']
    df['Bearish'] = df['Close'] < df['Open']

    df['Doji'] = df['Body_pct'] < 0.1
    df['Hammer'] = (df['Lower_wick_pct'] > 0.6) & (df['Upper_wick_pct'] < 0.1) & (df['Body_pct'] < 0.3)
    df['Shooting_Star'] = (df['Upper_wick_pct'] > 0.6) & (df['Lower_wick_pct'] < 0.1) & (df['Body_pct'] < 0.3)

    prev_body = df['Candle_body'].shift(1)
    df['Engulfing'] = (df['Candle_body'] > prev_body) & \
                      ((df['Bullish'] & df['Bearish'].shift(1) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))) | \
                       (df['Bearish'] & df['Bullish'].shift(1) & (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))))

    df['Inside_candle'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    df['Outside_candle'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))

    # 7. Volume
    for d in [5, 20, 50]:
        df[f'Vol_{d}D_avg'] = df['Volume'].rolling(d).mean()

    df['Vol_vs_20D_avg'] = df['Volume'] / df['Vol_20D_avg'].replace(0, np.nan)
    df['Vol_vs_50D_avg'] = df['Volume'] / df['Vol_50D_avg'].replace(0, np.nan)
    df['Vol_1D_change'] = df['Volume'].pct_change(1)
    df['Vol_5D_change'] = df['Volume'].pct_change(5)
    df['Vol_spike'] = df['Volume'] > df['Vol_20D_avg'] * 2

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_slope'] = df['OBV'].diff()

    # 8. Volatility
    df['ATR14'] = calc_atr(df, 14)
    df['ATR20'] = calc_atr(df, 20)
    df['ATR_pct'] = df['ATR14'] / df['Close'].replace(0, np.nan) * 100

    for d in [5, 10, 20, 50]:
        df[f'Volat_{d}D'] = df['Ret_1D'].rolling(d).std()

    df['Volat_change'] = df['Volat_10D'].diff()

    # 9. Trend Structure
    df['Trend_15_30'] = df['MA15'] > df['MA30']
    df['Trend_30_50'] = df['MA30'] > df['MA50']
    df['Trend_50_100'] = df['MA50'] > df['MA100']
    df['Trend_50_150'] = df['MA50'] > df['MA150']
    df['Trend_100_200'] = df['MA100'] > df['MA200']
    df['Trend_150_200'] = df['MA150'] > df['MA200']

    df['Trend_15_30_50_Up'] = df['Trend_15_30'] & df['Trend_30_50']
    df['Trend_15_30_50_Dn'] = (df['MA15'] < df['MA30']) & (df['MA30'] < df['MA50'])

    df['Bullish_MA_count'] = df[['Trend_15_30', 'Trend_30_50', 'Trend_50_100', 'Trend_100_200']].sum(axis=1)
    df['MA15_cross_MA50'] = (df['MA15'].shift(1) <= df['MA50'].shift(1)) & (df['MA15'] > df['MA50'])

    # 10. Support/Resistance
    df['Support_dist'] = df['Dist_20D_Low']
    df['Resistance_dist'] = df['Dist_20D_High']
    df['Support_touches'] = (df['Low'] <= (df['20D_Low'] * 1.01)).rolling(20).sum()
    df['Resistance_touches'] = (df['High'] >= (df['20D_High'] * 0.99)).rolling(20).sum()

    # 11. Market Context
    if 'Nifty50_Close' in df.columns:
        for d in [1, 5, 7, 20]:
            df[f'Nifty_{d}D_ret'] = df['Nifty50_Close'].pct_change(d) * 100

        df['Nifty_MA50'] = df['Nifty50_Close'].rolling(50).mean()
        df['Nifty_MA200'] = df['Nifty50_Close'].rolling(200).mean()
        df['Nifty_vs_MA50'] = df['Nifty50_Close'] / df['Nifty_MA50'].replace(0, np.nan)
        df['Nifty_vs_MA200'] = df['Nifty50_Close'] / df['Nifty_MA200'].replace(0, np.nan)
        df['Nifty_RSI'] = calc_rsi(df['Nifty50_Close'])
        df['Nifty_Trend'] = df['Nifty_MA50'] > df['Nifty_MA200']

        for d in [5, 7, 20]:
            df[f'Rel_str_Nifty_{d}D'] = df[f'Ret_{d}D'] - df[f'Nifty_{d}D_ret']

    symbol_sector = df['Sector'].iloc[-1] if 'Sector' in df.columns else 'Unknown'
    if sector_pivot is not None and symbol_sector in sector_pivot.columns:
        sec_close = sector_pivot[symbol_sector]
        df['Sector_Close'] = sec_close.reindex(df.index)
        for d in [1, 5, 7, 20]:
            df[f'Sector_{d}D_ret'] = df['Sector_Close'].pct_change(d) * 100
            if f'Ret_{d}D' in df.columns:
                df[f'Rel_str_Sector_{d}D'] = df[f'Ret_{d}D'] - df[f'Sector_{d}D_ret']

        df['Sector_MA50'] = df['Sector_Close'].rolling(50).mean()
        df['Sector_vs_MA50'] = df['Sector_Close'] / df['Sector_MA50'].replace(0, np.nan)
        df['Sector_RSI'] = calc_rsi(df['Sector_Close'])

    # 12. Base Strategy Logic
    df['MA50_trend_7D_up'] = (df['MA50'] > df['MA50'].shift(7)) & (df['MA50'] > df['MA50'].shift(1))

    df['Dist_Low_BB'] = df['Low'] - df['BB_Lower']
    df['BB_near_Low'] = (df['Dist_Low_BB'] >= -10) & (df['Dist_Low_BB'] <= 20)

    df['Candle_Green'] = df['Close'] > df['Open']
    df['Dist_Low_MA50'] = df['Low'] - df['MA50']

    df['Setup_1'] = (df['Dist_Low_MA50'] >= -10) & (df['Dist_Low_MA50'] <= 30)
    df['Setup_2'] = df['Dist_Low_MA50'] > 30
    df['Setup_3'] = df['Dist_Low_MA50'] < -10

    df['Base_Signal'] = df['MA50_trend_7D_up'] & df['BB_near_Low'] & df['Candle_Green'] & (df['Setup_1'] | df['Setup_2'] | df['Setup_3'])

    df['Signal_Type'] = 0
    df.loc[df['Base_Signal'] & df['Setup_1'], 'Signal_Type'] = 1
    df.loc[df['Base_Signal'] & df['Setup_2'], 'Signal_Type'] = 2
    df.loc[df['Base_Signal'] & df['Setup_3'], 'Signal_Type'] = 3

    df['Entry_Price'] = np.where(df['Base_Signal'], df['High'] + 1, np.nan)
    df['Stop_Loss'] = np.where(df['Base_Signal'], df['Low'] - 1, np.nan)
    df['Risk'] = df['Entry_Price'] - df['Stop_Loss']
    df['Target'] = df['Entry_Price'] + (2 * df['Risk'])

    # Chronological OHLC forward trade simulation
    df['Target_Hit'] = False
    df['SL_Hit'] = False
    df['Outcome'] = 'None'
    df['Days_to_Outcome'] = np.nan
    df['Trade_Triggered'] = False
    df['MFE'] = np.nan
    df['MAE'] = np.nan
    df['Future_Ret_7D'] = df['Close'].shift(-7) / df['Close'].replace(0, np.nan) - 1

    signal_indices = np.where(df['Base_Signal'])[0]

    for i in signal_indices:
        if i + 1 >= len(df):
            df.iloc[i, df.columns.get_loc('Outcome')] = 'Pending'
            continue

        buy_price = df['Entry_Price'].iloc[i]
        sl_price = df['Stop_Loss'].iloc[i]
        target = df['Target'].iloc[i]

        day1 = df.iloc[i+1]
        if day1['High'] < buy_price:
            df.iloc[i, df.columns.get_loc('Outcome')] = 'No_Trigger'
            continue

        df.iloc[i, df.columns.get_loc('Trade_Triggered')] = True

        resolved = False
        mfe_val = day1['High'] - buy_price if day1['High'] > buy_price else 0
        mae_val = buy_price - day1['Low'] if day1['Low'] < buy_price else 0

        day1_sl_hit = False
        day1_target_hit = False

        if day1['Low'] <= sl_price:
            if day1['Open'] > sl_price:
                if day1['Open'] >= buy_price:
                    day1_sl_hit = True
                else:
                    if day1['High'] >= buy_price:
                        day1_sl_hit = True
            else:
                if day1['Close'] <= sl_price:
                    day1_sl_hit = True

        if day1['High'] >= target and not day1_sl_hit:
            day1_target_hit = True

        if day1_sl_hit:
            df.iloc[i, df.columns.get_loc('SL_Hit')] = True
            df.iloc[i, df.columns.get_loc('Outcome')] = 'SL'
            df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = 1
            df.iloc[i, df.columns.get_loc('MFE')] = mfe_val
            df.iloc[i, df.columns.get_loc('MAE')] = mae_val
            continue
        elif day1_target_hit:
            df.iloc[i, df.columns.get_loc('Target_Hit')] = True
            df.iloc[i, df.columns.get_loc('Outcome')] = 'Target'
            df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = 1
            df.iloc[i, df.columns.get_loc('MFE')] = mfe_val
            df.iloc[i, df.columns.get_loc('MAE')] = mae_val
            continue

        for d in range(2, 8):
            if i + d >= len(df):
                df.iloc[i, df.columns.get_loc('Outcome')] = 'Pending'
                resolved = True
                break

            day_d = df.iloc[i+d]

            if (day_d['High'] - buy_price) > mfe_val:
                mfe_val = day_d['High'] - buy_price
            if (buy_price - day_d['Low']) > mae_val:
                mae_val = buy_price - day_d['Low']

            sl_hit_d = day_d['Low'] <= sl_price
            target_hit_d = day_d['High'] >= target

            if sl_hit_d and target_hit_d:
                df.iloc[i, df.columns.get_loc('SL_Hit')] = True
                df.iloc[i, df.columns.get_loc('Outcome')] = 'SL'
                df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = d
                resolved = True
                break
            elif sl_hit_d:
                df.iloc[i, df.columns.get_loc('SL_Hit')] = True
                df.iloc[i, df.columns.get_loc('Outcome')] = 'SL'
                df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = d
                resolved = True
                break
            elif target_hit_d:
                df.iloc[i, df.columns.get_loc('Target_Hit')] = True
                df.iloc[i, df.columns.get_loc('Outcome')] = 'Target'
                df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = d
                resolved = True
                break

        if not resolved:
            df.iloc[i, df.columns.get_loc('Outcome')] = 'Timeout'
            df.iloc[i, df.columns.get_loc('Days_to_Outcome')] = 7

        df.iloc[i, df.columns.get_loc('MFE')] = mfe_val
        df.iloc[i, df.columns.get_loc('MAE')] = mae_val

    return df


# ==============================================================================
# 3. PIPELINE ORCHESTRATION WITH REAL-TIME PROGRESS
# ==============================================================================

def run_feature_generation_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Executes the full Feature Generation pipeline across all constituent raw CSVs:
    1. Checks if features have already been computed for today or latest raw data date.
    2. Pre-calculates sector metrics
    3. Stage: GENERATING (1/207 ... 207/207 with stock name, % and progress line)
    4. Stage: SAVING (Heading = 'Saving Data', no progress line)
    5. Stage: COMPLETED (Heading = 'Done', persists metadata and logs)
    """
    start_time = time.time()
    today = datetime.now().date()
    today_str = str(today)
    raw_files = sorted(glob.glob(os.path.join(str(RAW_DIR), '*_data.csv')))
    total_files = len(raw_files)

    if total_files == 0:
        err_msg = "No raw stock CSV files found in 5_year_data/raw"
        json_manager.update_feature_gen_progress(
            status="ERROR",
            heading="Features Generating",
            message=err_msg,
            error=err_msg
        )
        json_manager.log_task_entry("FEATURE_GENERATION", "ERROR", err_msg, error=err_msg)
        return {"status": "ERROR", "message": err_msg}

    last_downloaded_date = json_manager.get_last_downloaded_date()
    feat_info = json_manager.get_feature_gen_info()
    last_feat_date = feat_info.get("last_feature_gen_date", "")
    last_run_date = feat_info.get("last_feature_run_date", "")

    # DAILY SKIP CHECK:
    # If already generated for latest market data or already run today
    if not force and feat_info.get("status") == "COMPLETED":
        if (last_run_date == today_str and last_feat_date >= last_downloaded_date) or (last_feat_date and last_feat_date >= last_downloaded_date):
            msg = f"Features already computed up to {last_feat_date} for all {total_files} stocks. Generation skipped."
            duration = round(time.time() - start_time, 2)
            json_manager.update_feature_gen_info(
                status="COMPLETED",
                total_stocks=total_files,
                message=msg,
                duration_seconds=duration,
                last_feature_gen_date=last_feat_date,
                last_feature_run_date=today_str
            )
            json_manager.update_feature_gen_progress(
                status="COMPLETED",
                heading="Done",
                current_stock="ALL",
                current_index=total_files,
                total_stocks=total_files,
                percentage=100.0,
                message=msg,
                elapsed_seconds=duration,
                last_feature_gen_date=last_feat_date
            )
            json_manager.log_task_entry(
                task_name="FEATURE_GENERATION",
                status="COMPLETED",
                message=msg
            )
            return {
                "status": "SUCCESS",
                "message": msg,
                "last_feature_gen_date": last_feat_date,
                "skipped": True
            }

    # Log task start
    json_manager.log_task_entry(
        task_name="FEATURE_GENERATION",
        status="RUNNING",
        message=f"Starting feature generation for {total_files} stocks up to {last_downloaded_date}..."
    )

    # Pre-calculate sector cross-sectional metrics
    json_manager.update_feature_gen_progress(
        status="GENERATING",
        heading="Features Generating",
        current_stock="SECTOR_METRICS",
        current_index=0,
        total_stocks=total_files,
        percentage=0.0,
        message="Pre-calculating cross-sectional sector metrics..."
    )
    sector_pivot = calculate_sector_metrics(RAW_DIR)

    # Process all constituents in memory
    generated_results: Dict[str, pd.DataFrame] = {}

    for i, file_path in enumerate(raw_files, 1):
        file_name = os.path.basename(file_path)
        clean_symbol = file_name.replace('_data.csv', '').upper()
        pct = round((i / total_files) * 90.0, 1)

        # Update progress for UI (0% -> 90% during computation)
        json_manager.update_feature_gen_progress(
            status="GENERATING",
            heading="Features Generating",
            current_stock=clean_symbol,
            current_index=i,
            total_stocks=total_files,
            percentage=pct,
            message=f"Computing technical indicators & strategy features for {clean_symbol}...",
            elapsed_seconds=time.time() - start_time
        )

        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            df_features = calculate_features(df, sector_pivot)
            generated_results[file_name] = df_features

        except Exception as e:
            print(f"[ERROR] Failed feature computation for {clean_symbol}: {e}")

    # -------------------------------------------------------------------------
    # STAGE 2: SAVING (90% -> 100% while flushing CSVs to disk)
    # -------------------------------------------------------------------------
    total_saved = len(generated_results)
    for idx, (file_name, df_feat) in enumerate(generated_results.items(), 1):
        clean_symbol = file_name.replace('_data.csv', '').replace('_generated.csv', '').upper()
        out_path = GENERATED_DIR / file_name
        try:
            df_feat.to_csv(out_path)
        except Exception as e:
            print(f"[ERROR] Writing {out_path}: {e}")

        if idx % 3 == 0 or idx == total_saved:
            save_pct = round(90.0 + (idx / max(1, total_saved)) * 10.0, 1)
            json_manager.update_feature_gen_progress(
                status="SAVING",
                heading="Saving Data",
                current_stock=clean_symbol,
                current_index=idx,
                total_stocks=total_saved,
                percentage=min(100.0, save_pct),
                message=f"Flushing engineered CSV {idx}/{total_saved}: {file_name} to disk...",
                elapsed_seconds=time.time() - start_time
            )

    # -------------------------------------------------------------------------
    # STAGE 3: COMPLETED
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = f"Completed. Generated features for {total_files} stocks up to {last_downloaded_date} in {duration:.1f}s."

    # 1. Update feature_gen.json FIRST
    json_manager.update_feature_gen_info(
        status="COMPLETED",
        total_stocks=total_files,
        message=completion_msg,
        duration_seconds=duration,
        last_feature_gen_date=last_downloaded_date,
        last_feature_run_date=today_str
    )

    # 2. Update live progress SECOND (broadcasts to WebSocket with last_feature_gen_date)
    json_manager.update_feature_gen_progress(
        status="COMPLETED",
        heading="Done",
        current_stock="ALL",
        current_index=total_files,
        total_stocks=total_files,
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        last_feature_gen_date=last_downloaded_date
    )

    # 3. Log task audit entry
    json_manager.log_task_entry(
        task_name="FEATURE_GENERATION",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "last_feature_gen_date": last_downloaded_date,
        "total_stocks": total_files,
        "duration_seconds": duration,
        "message": completion_msg
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Feature Generation Engine...")
    print("==================================================")
    res = run_feature_generation_pipeline()
    print("Result:", res)
