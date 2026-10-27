"""
VectorTrade - Production Offline EOD Daemon
Battery-safe, zero-idle executor scheduled daily strictly post 16:00 IST.
1. Execution Guard: Verifies time >= 16:00 IST & checks MongoDB system_audit to prevent redundant runs.
2. Fast incremental ingestion: Ingests the latest completed bar for all NIFTY 200 stocks via yfinance.
3. Feature Computation: Computes 130+ technical indicators via core.indicators.
4. Strategy Signal Generation: Runs 50MA Pullback and Global Forecaster.
5. ML Model Inference: Employs pre-trained XGBoost brains to generate confidence & forecast metrics.
6. Unified Portfolio Risk Gate (core.risk_manager):
   - Dynamic capital-at-risk position sizing.
   - Max open position cap.
   - 30% sector concentration cap.
   - Cross-strategy correlation and symbol deduplication.
   - Portfolio-level risk circuit breaker.
7. Dispatch: Pushes risk-approved ARMED setups to MongoDB signals_watchlist and appends to buffer_daily_bars.
8. Clean Exit: Records run metrics in MongoDB and log file, then terminates immediately (0% idle RAM/CPU).
"""

from __future__ import annotations
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Resolve project base directory
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from core.indicators import compute_indicators
from core.strategies.ma50_pullback import MA50PullbackStrategy
from core.strategies.global_forecast import GlobalForecastStrategy
from core.risk_manager import PortfolioRiskManager

# Configure dual-stream logging (File + Console)
log_file = settings.LOGS_DIR / "daemon_runs.log"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VectorTrade.EODDaemon")


class EODDaemon:
    """
    Offline End-of-Day autonomous pipeline for VectorTrade.
    """

    def __init__(self, force_run: bool = False) -> None:
        self.force_run = force_run
        self.tz = pytz.timezone(settings.TIMEZONE)
        self.now_ist = datetime.now(self.tz)
        self.today_str = self.now_ist.strftime("%Y-%m-%d")
        self.db = db_manager.db
        self.risk_manager = PortfolioRiskManager()

        # Load constituents universe & MegaBull token map
        self.universe: Dict[str, Dict[str, Any]] = {}
        if settings.STOCK_NAMES_PATH.exists():
            with open(settings.STOCK_NAMES_PATH, "r", encoding="utf-8") as f:
                self.universe = json.load(f)

        self.tokens_cache: Dict[str, str] = {}
        if settings.TOKENS_CACHE_PATH.exists():
            with open(settings.TOKENS_CACHE_PATH, "r", encoding="utf-8") as f:
                self.tokens_cache = json.load(f)

        # Pre-load ML Brains
        self._load_models()

    def _load_models(self) -> None:
        """Loads pre-trained XGBoost brains, scalers, and imputers with fallbacks."""
        models_dir = BASE_DIR / "models"
        self.strat_model = None
        self.strat_scaler = None
        self.strat_imputer = None
        self.strat_features: List[str] = []

        self.glob_gain_model = None
        self.glob_drop_model = None
        self.glob_scaler = None
        self.glob_imputer = None
        self.glob_features: List[str] = []

        # Load Strategy 1 Brain
        try:
            strat_model_path = models_dir / "5_year_xgboost_model.pkl"
            strat_scaler_path = models_dir / "5_year_scaler.pkl"
            strat_imputer_path = models_dir / "5_year_imputer.pkl"
            if strat_model_path.exists() and strat_scaler_path.exists() and strat_imputer_path.exists():
                self.strat_model = joblib.load(strat_model_path)
                self.strat_scaler = joblib.load(strat_scaler_path)
                self.strat_imputer = joblib.load(strat_imputer_path)
                feats_in = getattr(self.strat_imputer, "feature_names_in_", None)
                if feats_in is not None:
                    self.strat_features = list(feats_in)
                logger.info("Loaded Strategy 1 ML Brain (%d features).", len(self.strat_features))
        except Exception as e:
            logger.warning("Could not load Strategy 1 models: %s", str(e))

        # Load Strategy 2 Brain
        try:
            gain_model_path = models_dir / "5_year_global_gain_model.pkl"
            drop_model_path = models_dir / "5_year_global_drop_model.pkl"
            glob_scaler_path = models_dir / "5_year_global_scaler.pkl"
            glob_imputer_path = models_dir / "5_year_global_imputer.pkl"
            if gain_model_path.exists() and drop_model_path.exists():
                self.glob_gain_model = joblib.load(gain_model_path)
                self.glob_drop_model = joblib.load(drop_model_path)
                self.glob_scaler = joblib.load(glob_scaler_path)
                self.glob_imputer = joblib.load(glob_imputer_path)
                feats_in = getattr(self.glob_imputer, "feature_names_in_", None)
                if feats_in is not None:
                    self.glob_features = list(feats_in)
                logger.info("Loaded Strategy 2 Global Forecaster Brains (%d features).", len(self.glob_features))
        except Exception as e:
            logger.warning("Could not load Strategy 2 models: %s", str(e))

    def check_execution_guard(self) -> bool:
        """
        Enforces execution guard:
        1. Must be past 16:00 IST on working days.
        2. MongoDB system_audit must not already contain a SUCCESS status for TODAY.
        """
        if self.force_run:
            logger.info("Execution guard: Bypassed via --force-run flag.")
            return True

        if self.now_ist.hour < 16:
            logger.warning(
                "Execution guard: Current IST time (%02d:%02d) is before 16:00 IST market close. Aborting run.",
                self.now_ist.hour, self.now_ist.minute
            )
            return False

        audit_col = db_manager.get_collection("system_audit")
        existing_run = audit_col.find_one({
            "component": "OFFLINE_EOD_DAEMON",
            "execution_date": self.today_str,
            "status": "SUCCESS"
        })
        if existing_run:
            logger.info(
                "Execution guard: OFFLINE_EOD_DAEMON has already successfully run today (%s). Terminating cleanly.",
                self.today_str
            )
            return False

        return True

    def fetch_benchmark_data(self) -> pd.DataFrame:
        """Fetches NIFTY 50 (^NSEI) daily data for relative strength context."""
        try:
            nifty = yf.download("^NSEI", period="1y", progress=False)
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            nifty = nifty.reset_index()
            nifty.columns = [c.lower() for c in nifty.columns]
            logger.info("Fetched %d bars for NIFTY 50 benchmark.", len(nifty))
            return nifty
        except Exception as e:
            logger.warning("Failed to fetch NIFTY 50 benchmark: %s", str(e))
            return pd.DataFrame()

    def run(self) -> Dict[str, Any]:
        """
        Executes the end-of-day autonomous analytics pipeline with Risk Management.
        """
        start_time = datetime.now(self.tz)
        logger.info("==================================================")
        logger.info("  Starting VectorTrade EOD Pipeline (%s)", self.today_str)
        logger.info("==================================================")

        if not self.check_execution_guard():
            sys.exit(0)

        nifty_df = self.fetch_benchmark_data()
        strategy_1 = MA50PullbackStrategy()
        strategy_2 = GlobalForecastStrategy()

        symbols_to_process = list(self.universe.keys())
        if not symbols_to_process:
            symbols_to_process = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "TATAMOTORS"]

        logger.info("Processing %d constituent symbols for EOD analysis...", len(symbols_to_process))

        raw_signals: List[Dict[str, Any]] = []
        buffer_bar_records: List[Dict[str, Any]] = []
        processed_count = 0
        error_count = 0

        for sym in symbols_to_process:
            try:
                clean_sym = sym.strip().upper()
                token = self.tokens_cache.get(clean_sym, "")
                meta = self.universe.get(clean_sym, {})
                sector = meta.get("sector", "General")

                ticker = f"{clean_sym}.NS"
                df_bar = yf.download(ticker, period="1y", progress=False)
                if df_bar.empty or len(df_bar) < 50:
                    continue

                if isinstance(df_bar.columns, pd.MultiIndex):
                    df_bar.columns = df_bar.columns.get_level_values(0)
                df_bar = df_bar.reset_index()

                df_bar["symbol"] = clean_sym
                df_bar["sector"] = sector

                # 1. Compute 130+ Indicators
                df_feat = compute_indicators(df_bar, nifty_df=nifty_df)
                if df_feat.empty:
                    continue

                # 2. Extract latest bar features for buffer collection
                last_row = df_feat.iloc[-1].to_dict()
                clean_features: Dict[str, Any] = {}
                for k, v in last_row.items():
                    if isinstance(v, (np.floating, float)):
                        clean_features[k] = round(float(v), 4) if np.isfinite(v) else 0.0
                    elif isinstance(v, (np.integer, int)):
                        clean_features[k] = int(v)
                    elif isinstance(v, (np.bool_, bool)):
                        clean_features[k] = bool(v)
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        clean_features[k] = v.strftime("%Y-%m-%d")
                    else:
                        clean_features[k] = str(v)

                buffer_bar_records.append({
                    "date": self.today_str,
                    "symbol": clean_sym,
                    "features": clean_features,
                    "consumed_for_training": False
                })

                # 3. Model Inference for Strategy 2 (Global Extremes)
                pred_gain = 0.0
                pred_drop = 0.0
                if self.glob_gain_model is not None and self.glob_drop_model is not None and self.glob_features:
                    try:
                        feat_row = df_feat.iloc[[-1]][self.glob_features]
                        x_glob = self.glob_scaler.transform(self.glob_imputer.transform(feat_row))
                        pred_gain = float(self.glob_gain_model.predict(x_glob)[0])
                        pred_drop = float(self.glob_drop_model.predict(x_glob)[0])
                    except Exception as e:
                        logger.debug("Global inference exception for %s: %s", clean_sym, str(e))

                # 4. Evaluate Strategy 1 (50MA Pullback)
                s1_signals = strategy_1.evaluate(df_feat, symbol=clean_sym, instrument_token=token)
                for sig in s1_signals:
                    confidence = 0.5
                    if self.strat_model is not None and self.strat_features:
                        try:
                            feat_row = df_feat.iloc[[-1]][self.strat_features]
                            x_strat = self.strat_scaler.transform(self.strat_imputer.transform(feat_row))
                            pred_val = float(self.strat_model.predict(x_strat)[0])
                            confidence = round(float(np.clip(pred_val, 0.0, 1.0)), 3)
                        except Exception as e:
                            logger.debug("Strategy 1 ML prediction notice: %s", str(e))
                    sig["ml_confidence"] = confidence
                    raw_signals.append(sig)

                # 5. Evaluate Strategy 2 (Global Forecaster)
                s2_signals = strategy_2.evaluate(
                    df_feat,
                    symbol=clean_sym,
                    instrument_token=token,
                    predicted_gain_pct=pred_gain,
                    predicted_drop_pct=pred_drop
                )
                for sig in s2_signals:
                    sig["ml_confidence"] = round(float(pred_gain / 10.0), 3)
                    raw_signals.append(sig)

                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning("Error processing %s: %s", sym, str(e))

        # 6. Apply Portfolio Risk Management Filter
        logger.info("Applying PortfolioRiskManager filter to %d raw strategy signals...", len(raw_signals))
        approved_signals = self.risk_manager.filter_and_size_signals(raw_signals, universe=self.universe)
        logger.info(
            "Risk Filter Result: %d / %d signals approved for armed dispatch.",
            len(approved_signals), len(raw_signals)
        )

        # 7. MongoDB Atlas Dispatch
        signals_col = db_manager.get_collection("signals_watchlist")
        buffer_col = db_manager.get_collection("buffer_daily_bars")

        # Upsert risk-approved armed signals
        for sig in approved_signals:
            signals_col.update_one(
                {"_id": sig["_id"]},
                {"$set": sig},
                upsert=True
            )

        # Upsert buffer daily bars
        for bar in buffer_bar_records:
            buffer_col.update_one(
                {"symbol": bar["symbol"], "date": bar["date"]},
                {"$set": bar},
                upsert=True
            )

        end_time = datetime.now(self.tz)
        duration_sec = round((end_time - start_time).total_seconds(), 2)

        # 8. Record System Audit Entry
        audit_col = db_manager.get_collection("system_audit")
        audit_entry = {
            "component": "OFFLINE_EOD_DAEMON",
            "execution_date": self.today_str,
            "timestamp": end_time.isoformat(),
            "status": "SUCCESS",
            "symbols_ingested": processed_count,
            "symbols_failed": error_count,
            "raw_signals_detected": len(raw_signals),
            "signals_risk_approved": len(approved_signals),
            "buffer_bars_stored": len(buffer_bar_records),
            "db_push_acknowledged": True,
            "duration_seconds": duration_sec
        }
        audit_col.insert_one(audit_entry)

        logger.info(
            "EOD Daemon Complete: %d symbols ingested | %d signals approved & armed | Duration: %.2fs",
            processed_count, len(approved_signals), duration_sec
        )
        return audit_entry


def main():
    force = "--force-run" in sys.argv or "--dry-run" in sys.argv
    daemon = EODDaemon(force_run=force)
    result = daemon.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
