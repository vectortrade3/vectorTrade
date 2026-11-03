"""
VectorTrade - Daily Market Analysis & AI Trade Finalizer Engine
Executes end-to-end scanning across all 207 Nifty constituents:
1. Market Timing & Data Check:
   - If market closed (holiday or weekday > 3:30 PM IST), ensures today's EOD is captured.
   - If market open / pre-3:30 PM, uses previous completed trading session's close.
   - If data is already up-to-date, skips heavy download and executes prediction pipeline.
2. Feature Assembly:
   - Reads latest engineered indicator vector (137 alpha indicators).
3. Dual-Model Machine Learning Inference:
   - Strategy Model (50 MA Setup): Predicts payoff multiple (R).
   - Global Regime Models: Predicts expected forward 7-day max gain and max drop.
4. Trade Synthesis & Risk Engineering:
   - Computes Entry, Target, Stop Loss (SL), Profit %, Loss %, and Risk-Reward (R:R).
   - Categorizes strategy: '50 MA Setup', 'Global Regime', '50 MA + Global Regime', extensible for future strategies.
   - Filters and finalizes highest-conviction trade setups with positive expected value.
5. Atomic Persistence:
   - Stores finalized trade recommendations in data_store/daily_analysis.json.
   - Broadcasts real-time progress via WebSocket listeners.
"""

from __future__ import annotations
import os
import sys
import time
import glob
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Directory paths resolution
BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from offline import json_manager
from offline.data_cleaner import COLUMN_MAPPING
from offline.downloader import run_stock_download_pipeline
from offline.feature_generator import run_feature_generation_pipeline

logger = logging.getLogger("VectorTrade.DailyAnalyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = BASE_DIR / "models"
FINAL_DATA_FILE = BASE_DIR / "5_year_final_data.csv"
GENERATED_DIR = BASE_DIR / "5_year_data" / "generated"
RAW_DIR = BASE_DIR / "5_year_data" / "raw"
STOCK_NAMES_FILE = BASE_DIR / "stock_names.json"


# ==============================================================================
# 1. MARKET SCHEDULE & SESSION EVALUATION HELPER
# ==============================================================================

# Common NSE Trading Holidays (Format YYYY-MM-DD)
NSE_HOLIDAYS_KNOWN = {
    "2026-01-26", # Republic Day
    "2026-03-06", # Mahashivratri
    "2026-03-25", # Holi
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-08-15", # Independence Day
    "2026-10-02", # Mahatma Gandhi Jayanti
    "2026-10-20", # Dussehra
    "2026-11-09", # Diwali / Laxmi Pujan
    "2026-11-10", # Diwali Balipratipada
    "2026-12-25", # Christmas
}


def get_market_schedule_info(ref_dt: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Evaluates current time against NSE market schedule rules:
    - Trading hours: 09:15 to 15:30 IST on Monday through Friday.
    - If today is Weekend or Holiday: Market is closed -> target date is previous trading day.
    - If today is normal trading day and time >= 15:30 IST: Market is closed -> target date is today.
    - If today is normal trading day and time < 15:30 IST: Market is in session ->
      Do NOT attempt downloading incomplete intraday data as EOD! Target date is yesterday (or last trading day).
    """
    if ref_dt is None:
        ref_dt = datetime.now().astimezone()

    curr_date = ref_dt.date()
    curr_date_str = str(curr_date)
    weekday = curr_date.weekday() # 0 = Monday, 6 = Sunday
    hour = ref_dt.hour
    minute = ref_dt.minute

    is_weekend = (weekday in [5, 6])
    is_holiday = (curr_date_str in NSE_HOLIDAYS_KNOWN)

    if is_weekend:
        # Weekend: market closed, last session was Friday
        days_back = 1 if weekday == 5 else 2
        target_date = curr_date - timedelta(days=days_back)
        return {
            "is_market_closed": True,
            "target_date": str(target_date),
            "session_type": "WEEKEND",
            "market_status": f"Weekend Closed | Using Last Trading Session ({target_date})",
            "allow_today_download": False,
            "evaluated_at": ref_dt.strftime("%Y-%m-%d %H:%M:%S")
        }

    if is_holiday:
        # Holiday: market closed, find previous weekday
        step = 1
        cand = curr_date - timedelta(days=step)
        while cand.weekday() in [5, 6] or str(cand) in NSE_HOLIDAYS_KNOWN:
            step += 1
            cand = curr_date - timedelta(days=step)
        return {
            "is_market_closed": True,
            "target_date": str(cand),
            "session_type": "HOLIDAY",
            "market_status": f"Exchange Holiday ({curr_date_str}) | Using Last Trading Session ({cand})",
            "allow_today_download": False,
            "evaluated_at": ref_dt.strftime("%Y-%m-%d %H:%M:%S")
        }

    # Weekday check: Cutoff at 15:30 IST (3:30 PM)
    is_post_close = (hour > 15) or (hour == 15 and minute >= 30)

    if is_post_close:
        # Market closed for today, full EOD available
        return {
            "is_market_closed": True,
            "target_date": curr_date_str,
            "session_type": "POST_CLOSE",
            "market_status": f"Market Closed (Past 3:30 PM IST) | Today's EOD Finalized ({curr_date_str})",
            "allow_today_download": True,
            "evaluated_at": ref_dt.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        # Market still in session (< 3:30 PM), use previous trading session
        days_back = 3 if weekday == 0 else 1
        target_date = curr_date - timedelta(days=days_back)
        return {
            "is_market_closed": False,
            "target_date": str(target_date),
            "session_type": "IN_SESSION",
            "market_status": f"Market In Session (< 3:30 PM IST) | Using Previous Closed Session ({target_date})",
            "allow_today_download": False,
            "evaluated_at": ref_dt.strftime("%Y-%m-%d %H:%M:%S")
        }


# ==============================================================================
# 2. MODEL ASSETS LOADER
# ==============================================================================

def load_prediction_models() -> Dict[str, Any]:
    """
    Loads all calibrated strategy and global models along with their scalers,
    imputers, and verified feature column contracts.
    """
    # 1. Strategy Model
    strat_model_path = MODELS_DIR / "5_year_xgboost_model.pkl"
    strat_scaler_path = MODELS_DIR / "5_year_scaler.pkl"
    strat_imputer_path = MODELS_DIR / "5_year_imputer.pkl"
    strat_feats_path = BASE_DIR / "final_features.txt"

    strat_model = joblib.load(strat_model_path)
    strat_scaler = joblib.load(strat_scaler_path)
    strat_imputer = joblib.load(strat_imputer_path)
    with open(strat_feats_path, "r", encoding="utf-8") as f:
        strat_features = [line.strip() for line in f if line.strip()]

    # 2. Global Regime Models
    gain_model_path = MODELS_DIR / "5_year_global_gain_model.pkl"
    drop_model_path = MODELS_DIR / "5_year_global_drop_model.pkl"
    glob_scaler_path = MODELS_DIR / "5_year_global_scaler.pkl"
    glob_imputer_path = MODELS_DIR / "5_year_global_imputer.pkl"
    glob_feats_path = BASE_DIR / "global_features.txt"

    gain_model = joblib.load(gain_model_path)
    drop_model = joblib.load(drop_model_path)
    glob_scaler = joblib.load(glob_scaler_path)
    glob_imputer = joblib.load(glob_imputer_path)
    with open(glob_feats_path, "r", encoding="utf-8") as f:
        glob_features = [line.strip() for line in f if line.strip()]

    # 3. Stock metadata
    stock_meta = {}
    if STOCK_NAMES_FILE.exists():
        stock_meta = json_manager.read_json_file(STOCK_NAMES_FILE, {})

    return {
        "strat_model": strat_model,
        "strat_scaler": strat_scaler,
        "strat_imputer": strat_imputer,
        "strat_features": strat_features,
        "gain_model": gain_model,
        "drop_model": drop_model,
        "glob_scaler": glob_scaler,
        "glob_imputer": glob_imputer,
        "glob_features": glob_features,
        "stock_meta": stock_meta
    }


# ==============================================================================
# 3. CONSTITUENT DATA RETRIEVER
# ==============================================================================

def extract_latest_feature_rows() -> pd.DataFrame:
    """
    Extracts the latest indicator row for all 207 constituents.
    Prioritizes reading from 5_year_final_data.csv for high-speed indexing,
    falling back to 5_year_data/generated/*.csv if necessary.
    """
    if FINAL_DATA_FILE.exists() and FINAL_DATA_FILE.stat().st_size > 1000:
        try:
            df = pd.read_csv(FINAL_DATA_FILE)
            if not df.empty and 'symbol' in df.columns:
                latest_df = df.groupby('symbol').last().reset_index()
                if len(latest_df) >= 150:
                    return latest_df
        except Exception as e:
            logger.warning(f"Could not load final dataset: {e}. Fallback to individual generated files.")

    # Fallback to generated CSV files
    generated_files = sorted(glob.glob(os.path.join(str(GENERATED_DIR), "*_data.csv")))
    if not generated_files:
        generated_files = sorted(glob.glob(os.path.join(str(GENERATED_DIR), "*_generated.csv")))

    rows = []
    for fpath in generated_files:
        sym = os.path.basename(fpath).replace('_data.csv', '').replace('_generated.csv', '')
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            df.rename(columns=COLUMN_MAPPING, inplace=True)
            for col in df.columns:
                if col not in COLUMN_MAPPING.values() and col != 'Date':
                    df.rename(columns={col: col.lower()}, inplace=True)
            if 'stock symbol' in df.columns:
                df.rename(columns={'stock symbol': 'symbol'}, inplace=True)
            if 'adjusted_close' not in df.columns and 'close' in df.columns:
                df['adjusted_close'] = df['close']
            if 'symbol' not in df.columns:
                df['symbol'] = sym
            if 'ma_crossover' in df.columns:
                cid = df['ma_crossover'].cumsum()
                df['days_since_ma_crossover'] = df.groupby(cid).cumcount()
            rows.append(df.iloc[[-1]])
        except Exception:
            continue

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# ==============================================================================
# 4. TRADE EVALUATION & STRATEGY SYNTHESIS
# ==============================================================================

def synthesize_trade_opportunity(
    row: pd.Series,
    pred_r: float,
    pred_gain_pts: float,
    pred_drop_pts: float,
    stock_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether a constituent meets criteria for a long setup,
    determines the strategy tag, and calculates precise Entry, SL, Target, and R:R.
    """
    sym = str(row.get('symbol', ''))
    meta = stock_meta.get(sym, {})
    company_name = meta.get('name', sym)
    sector = meta.get('sector', 'General')

    close = float(row.get('close', 0.0))
    if close <= 0 or np.isnan(close):
        return None

    high = float(row.get('high', close))
    low = float(row.get('low', close))
    open_p = float(row.get('open', close))
    atr = float(row.get('atr14', close * 0.015))
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.015

    ma50 = float(row.get('ma50', close))
    dist_ma50 = float(row.get('dist_low_ma50', 0.0))
    rsi = float(row.get('rsi14', 50.0))
    is_base = bool(row.get('base_signal', False))

    pred_gain_pts = max(0.5, float(pred_gain_pts))
    pred_drop_pts = max(0.5, float(pred_drop_pts))
    p_gain_pct = round((pred_gain_pts / close) * 100, 2)
    p_drop_pct = round((pred_drop_pts / close) * 100, 2)

    # -------------------------------------------------------------------------
    # STRATEGY 1: 50 MA Setup Strategy
    # Conditions:
    # 1. Price is near 50 MA support (-10 to +35 pts) with green candle or base signal.
    # 2. Strategy XGBoost model predicts positive payoff multiple (>= 0.75 R).
    # -------------------------------------------------------------------------
    strat_bullish = is_base or (dist_ma50 >= -10 and dist_ma50 <= 35 and close > open_p)
    strat_qualified = strat_bullish and (pred_r >= 0.75)

    # -------------------------------------------------------------------------
    # STRATEGY 2: Global Regime Momentum Strategy
    # Conditions:
    # 1. 7-day predicted gain >= 2.0%
    # 2. Expected upside significantly exceeds downside (Gain >= 1.25 * Drop).
    # -------------------------------------------------------------------------
    glob_qualified = (p_gain_pct >= 2.0) and (p_gain_pct >= 1.25 * p_drop_pct)

    if not (strat_qualified or glob_qualified):
        return None

    # -------------------------------------------------------------------------
    # Strategy Classification & Rationale
    # -------------------------------------------------------------------------
    if strat_qualified and glob_qualified:
        strategy = "50 MA + Global Regime"
        action = "STRONG BUY"
        risk_amount = max(close - low, 1.5 * atr, pred_drop_pts, close * 0.015)
        target_pts = max(pred_gain_pts, 2.0 * risk_amount, close * 0.035)
        rationale = (
            f"Dual Alpha Confirmation: 50 MA setup with {pred_r:.2f}R payoff multiple "
            f"combined with +{p_gain_pct:.1f}% 7-day upward momentum."
        )
    elif strat_qualified:
        strategy = "50 MA Setup"
        action = "STRONG BUY" if pred_r >= 1.25 else "BUY"
        risk_amount = max(close - low, 1.5 * atr, close * 0.015)
        target_pts = max(2.0 * risk_amount, close * (pred_r * 0.025), close * 0.03)
        rationale = (
            f"50 MA Support Retest with bullish candle confirmation. "
            f"AI Strategy Model projects {pred_r:.2f}R expected payoff."
        )
    else:
        strategy = "Global Regime"
        action = "STRONG BUY" if p_gain_pct >= 4.0 else "BUY"
        risk_amount = max(pred_drop_pts, 1.5 * atr, close * 0.015)
        target_pts = max(pred_gain_pts, 2.0 * risk_amount, close * 0.03)
        rationale = (
            f"Strong multi-day upward momentum (+{p_gain_pct:.1f}% expected gain vs "
            f"{p_drop_pct:.1f}% expected drop). Favorable risk-reward regime."
        )

    # -------------------------------------------------------------------------
    # Trade Engineering: Entry, SL, Target, Loss %, Profit %, R:R
    # -------------------------------------------------------------------------
    entry_price = round(close, 2)
    sl_price = round(max(0.1, close - risk_amount), 2)
    target_price = round(close + target_pts, 2)

    loss_pct = round(((entry_price - sl_price) / entry_price) * 100, 2)
    profit_pct = round(((target_price - entry_price) / entry_price) * 100, 2)
    rr = round(profit_pct / max(loss_pct, 0.1), 2)

    # Strict Quality Filter: Positive edge and sensible risk parameters
    if profit_pct < 2.0 or loss_pct > 6.5 or rr < 1.4:
        return None

    return {
        "symbol": sym,
        "name": company_name,
        "sector": sector,
        "strategy": strategy,
        "action": action,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "target_price": target_price,
        "loss_pct": loss_pct,
        "profit_pct": profit_pct,
        "risk_reward": f"1 : {rr:.2f}",
        "risk_reward_num": rr,
        "predicted_r": round(pred_r, 2),
        "predicted_gain_pct": p_gain_pct,
        "predicted_drop_pct": p_drop_pct,
        "rsi": round(rsi, 1),
        "atr": round(atr, 2),
        "ma50": round(ma50, 2),
        "rationale": rationale
    }


# ==============================================================================
# 5. MAIN END-TO-END PIPELINE RUNNER
# ==============================================================================

def run_daily_analysis_pipeline(force_download: bool = False) -> Dict[str, Any]:
    """
    Executes the comprehensive Daily Market Analysis & AI Trade Finalizer:
    1. Evaluates market schedule (post 3:30 PM vs pre 3:30 PM).
    2. Downloads / synchronizes market data if needed.
    3. Verifies and cleans latest feature vectors for 207 constituents.
    4. Evaluates real Machine Learning models on all constituents.
    5. Finalizes high-conviction trades with Entry, SL, Target, Profit %, Loss %, and Strategy tags.
    6. Persists results atomically in daily_analysis.json and broadcasts progress.
    """
    start_time = time.time()
    logger.info("Starting Daily Market Analysis & AI Trade Finalizer Pipeline...")

    # Stage 1: Market schedule assessment
    json_manager.update_daily_analysis_progress(
        status="RUNNING",
        heading="Schedule Check",
        phase="Evaluating Market Session",
        current_stock="ALL_STOCKS",
        current_index=0,
        total_stocks=207,
        percentage=10.0,
        message="Checking NSE trading hours and EOD session readiness...",
        elapsed_seconds=round(time.time() - start_time, 1)
    )

    sched = get_market_schedule_info()
    market_session_date = sched.get("target_date", "2026-10-06")
    market_status_desc = sched.get("market_status", "Market Closed")
    allow_today_download = sched.get("allow_today_download", False)

    download_info = json_manager.get_stocks_download_info()
    last_downloaded_date = download_info.get("last_downloaded_date", "2026-10-30")

    # Check whether download is needed
    need_download = force_download or (allow_today_download and last_downloaded_date < market_session_date)

    if need_download:
        json_manager.update_daily_analysis_progress(
            status="RUNNING",
            heading="Syncing Market Data",
            phase="Downloading Daily Bars",
            current_stock="ALL_STOCKS",
            current_index=20,
            total_stocks=207,
            percentage=25.0,
            message=f"Syncing market sessions up to {market_session_date} across 207 stocks...",
            elapsed_seconds=round(time.time() - start_time, 1)
        )
        try:
            run_stock_download_pipeline(force=force_download)
            run_feature_generation_pipeline(force=force_download)
        except Exception as e:
            logger.warning(f"Incremental sync warning: {e}. Proceeding with latest cached features.")
    else:
        logger.info(f"Market data is up-to-date ({last_downloaded_date}). Skipping download.")

    # Stage 2: Feature extraction
    json_manager.update_daily_analysis_progress(
        status="RUNNING",
        heading="Feature Assembly",
        phase="Extracting Constituent Indicators",
        current_stock="ALL_STOCKS",
        current_index=50,
        total_stocks=207,
        percentage=45.0,
        message="Loading 137 alpha indicators for all 207 constituents...",
        elapsed_seconds=round(time.time() - start_time, 1)
    )

    latest_df = extract_latest_feature_rows()
    if latest_df.empty:
        err_msg = "No constituent feature records found to analyze."
        json_manager.update_daily_analysis_progress(
            status="ERROR",
            heading="Analysis Error",
            message=err_msg,
            error=err_msg,
            elapsed_seconds=round(time.time() - start_time, 1)
        )
        return {"status": "ERROR", "message": err_msg}

    # Stage 3: Model Loading & Batch Prediction
    json_manager.update_daily_analysis_progress(
        status="RUNNING",
        heading="ML Model Inference",
        phase="Evaluating AI Regressors",
        current_stock="ALL_STOCKS",
        current_index=120,
        total_stocks=207,
        percentage=70.0,
        message="Running Strategy and Global Regime XGBoost predictors...",
        elapsed_seconds=round(time.time() - start_time, 1)
    )

    models = load_prediction_models()
    strat_m = models["strat_model"]
    strat_s = models["strat_scaler"]
    strat_i = models["strat_imputer"]
    strat_f = models["strat_features"]

    gain_m = models["gain_model"]
    drop_m = models["drop_model"]
    glob_s = models["glob_scaler"]
    glob_i = models["glob_imputer"]
    glob_f = models["glob_features"]
    stock_meta = models["stock_meta"]

    # Align columns
    for c in strat_f:
        if c not in latest_df.columns:
            latest_df[c] = np.nan
    for c in glob_f:
        if c not in latest_df.columns:
            latest_df[c] = np.nan

    # Batch Predict Strategy Payoff R
    Xs = strat_s.transform(strat_i.transform(latest_df[strat_f]))
    preds_r = strat_m.predict(Xs)

    # Batch Predict Global Gain & Drop Points
    Xg = glob_s.transform(glob_i.transform(latest_df[glob_f]))
    preds_gain = gain_m.predict(Xg)
    preds_drop = drop_m.predict(Xg)

    # Stage 4: Trade Synthesis & Finalization
    json_manager.update_daily_analysis_progress(
        status="RUNNING",
        heading="Trade Finalization",
        phase="Synthesizing Setups",
        current_stock="ALL_STOCKS",
        current_index=180,
        total_stocks=207,
        percentage=90.0,
        message="Engineering Entry, Target, and Stop Loss parameters...",
        elapsed_seconds=round(time.time() - start_time, 1)
    )

    finalized_trades = []
    strategy_counts = {
        "50 MA Setup": 0,
        "Global Regime": 0,
        "50 MA + Global Regime": 0
    }

    for i, row in latest_df.iterrows():
        trade = synthesize_trade_opportunity(
            row=row,
            pred_r=float(preds_r[i]),
            pred_gain_pts=float(preds_gain[i]),
            pred_drop_pts=float(preds_drop[i]),
            stock_meta=stock_meta
        )
        if trade:
            finalized_trades.append(trade)
            stg = trade.get("strategy", "Other")
            strategy_counts[stg] = strategy_counts.get(stg, 0) + 1

    # Sort finalized trades by conviction:
    # 1. 50 MA + Global Regime first, then 50 MA Setup, then Global Regime
    # 2. Risk:Reward ratio descending
    tier_order = {"50 MA + Global Regime": 0, "50 MA Setup": 1, "Global Regime": 2}
    finalized_trades.sort(
        key=lambda t: (tier_order.get(t["strategy"], 99), -t["risk_reward_num"], -t["profit_pct"])
    )

    total_duration = time.time() - start_time
    completion_msg = (
        f"Analyzed {len(latest_df)} stocks. Finalized {len(finalized_trades)} high-conviction "
        f"trade setups across {market_status_desc} in {total_duration:.1f}s."
    )

    # Stage 5: Save State
    analysis_record = json_manager.update_daily_analysis_info(
        status="COMPLETED",
        market_session_date=market_session_date,
        market_status=market_status_desc,
        total_stocks_analyzed=len(latest_df),
        finalized_trades=finalized_trades,
        strategies_breakdown=strategy_counts,
        message=completion_msg,
        duration_seconds=total_duration
    )

    json_manager.update_daily_analysis_progress(
        status="COMPLETED",
        heading="Analysis Complete",
        phase="Done",
        current_stock="ALL_STOCKS",
        current_index=len(latest_df),
        total_stocks=len(latest_df),
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=total_duration,
        finalized_count=len(finalized_trades)
    )

    json_manager.log_task_entry(
        task_name="DAILY_MARKET_ANALYSIS",
        status="COMPLETED",
        message=completion_msg
    )

    logger.info(f"SUCCESS: {completion_msg}")
    return {
        "status": "SUCCESS",
        "market_session_date": market_session_date,
        "market_status": market_status_desc,
        "total_analyzed": len(latest_df),
        "finalized_count": len(finalized_trades),
        "finalized_trades": finalized_trades,
        "strategies_breakdown": strategy_counts,
        "duration_seconds": round(total_duration, 2),
        "message": completion_msg
    }


if __name__ == "__main__":
    print("Testing Daily Analyzer...")
    res = run_daily_analysis_pipeline()
    print("Result:", res)
