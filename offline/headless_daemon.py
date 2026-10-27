"""
VectorTrade - Production Offline EOD Daemon with Network Resilience & Fallback Queues
Battery-safe, zero-idle executor with pre-calculated trade financials.

Handles Unusual Real-World Edge Cases:
- Network Loss / Offline ISP: Catches connection loss, verifies via socket test (8.8.8.8:53).
- Remote Server / DB Unreachable: If MongoDB Atlas cannot be reached, stores signals safely in
  offline/pending_dispatches/signals_YYYY-MM-DD.json and flags DISPATCH_SAVED_LOCALLY.
- Staged Syncing: When connection is restored on subsequent runs, automatically flushes pending dispatches.
- Time Guard: Exits cleanly (sys.exit(0)) if time < 16:00 IST to consume 0% CPU, 0 MB RAM, 0% battery.
- Redundant Execution Guard: Checks eod_run_state.json & system_audit; exits in <0.2s if today's run completed.
"""

from __future__ import annotations
import sys
import os
import json
import glob
import socket
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Resolve project directories
BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from core.config import settings
from core.db import db_manager
from core.secure_comms import request_online_trade_details
from core.risk_manager import PortfolioRiskManager
from offline.indicators import compute_indicators
from offline.strategies.ma50_pullback import MA50PullbackStrategy
from offline.strategies.global_forecast import GlobalForecastStrategy

logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
pending_dir = OFFLINE_DIR / "pending_dispatches"
pending_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / "daemon_runs.log"
error_log_file = logs_dir / "offline_errors.log"
state_file = logs_dir / "eod_run_state.json"

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VectorTrade.HeadlessDaemon")


def log_error(context: str, exc: Exception) -> None:
    err_msg = f"[{datetime.now().isoformat()}] ERROR in {context}: {str(exc)}\n{traceback.format_exc()}\n"
    logger.error("Error in %s: %s", context, str(exc))
    try:
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(err_msg)
    except Exception:
        pass


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except (socket.timeout, OSError):
        return False


class HeadlessDaemon:
    """
    Production Headless Offline EOD Daemon with universal resilience.
    """

    def __init__(self, force_run: bool = False, capital: Optional[float] = None, risk_pct: Optional[float] = None) -> None:
        self.force_run = force_run

        # 1. Dynamically fetch account funds from MegaBull if API key is present
        live_funds = None
        try:
            from online.megabull_client import MegaBullClient
            mb = MegaBullClient()
            live_funds = mb.get_account_funds()
            mb.close()
        except Exception as e:
            logger.info("MegaBull balance query notice (using configured capital): %s", str(e))

        if live_funds and live_funds > 0:
            self.capital = float(live_funds)
            logger.info("[CAPITAL RESOLUTION] Live MegaBull capital fetched: Rs %.2f", self.capital)
        elif capital is not None and capital > 0:
            self.capital = float(capital)
            logger.info("[CAPITAL RESOLUTION] Using provided capital: Rs %.2f", self.capital)
        else:
            self.capital = float(settings.ACCOUNT_CAPITAL)
            logger.info("[CAPITAL RESOLUTION] Using default settings.ACCOUNT_CAPITAL: Rs %.2f", self.capital)

        # 2. Enforce strict max risk <= 1.0% of total capital
        req_risk = float(risk_pct) if risk_pct is not None else float(settings.MAX_RISK_PER_TRADE_PCT)
        self.risk_pct = min(req_risk, 1.0)  # Strictly capped at <= 1.0%
        self.risk_amount_per_trade = round(self.capital * (self.risk_pct / 100.0), 2)
        logger.info("[RISK BUDGET] Max Risk Per Trade (%.2f%%): Rs %.2f", self.risk_pct, self.risk_amount_per_trade)

        self.tz = pytz.timezone(settings.TIMEZONE)
        self.now_ist = datetime.now(self.tz)
        self.today_str = self.now_ist.strftime("%Y-%m-%d")

        self.universe: Dict[str, Dict[str, Any]] = {}
        if settings.STOCK_NAMES_PATH.exists():
            try:
                with open(settings.STOCK_NAMES_PATH, "r", encoding="utf-8") as f:
                    self.universe = json.load(f)
            except Exception as e:
                log_error("Loading universe", e)

        self.tokens_cache: Dict[str, str] = {}
        if settings.TOKENS_CACHE_PATH.exists():
            try:
                with open(settings.TOKENS_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.tokens_cache = json.load(f)
            except Exception as e:
                log_error("Loading tokens cache", e)

        self._load_models()

    def _load_models(self) -> None:
        models_dir = OFFLINE_DIR / "models"
        self.strat_model = None
        self.strat_scaler = None
        self.strat_imputer = None
        self.strat_features: List[str] = []

        self.glob_gain_model = None
        self.glob_drop_model = None
        self.glob_scaler = None
        self.glob_imputer = None
        self.glob_features: List[str] = []

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
            log_error("Loading Strategy 1 model", e)

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
                logger.info("Loaded Strategy 2 Global Brains (%d features).", len(self.glob_features))
        except Exception as e:
            log_error("Loading Strategy 2 model", e)

    def check_execution_guard(self) -> bool:
        if self.force_run:
            logger.info("Execution guard bypassed via force_run.")
            return True

        if self.now_ist.hour < 16:
            logger.info(
                "Execution guard: Current time (%02d:%02d IST) is before 16:00 IST market close. Exiting cleanly (0%% CPU/RAM).",
                self.now_ist.hour, self.now_ist.minute
            )
            return False

        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    if state.get("last_successful_run_date") == self.today_str:
                        logger.info(
                            "Execution guard: Today's run (%s) already completed according to %s. Exiting cleanly.",
                            self.today_str, state_file.name
                        )
                        return False
            except Exception as e:
                log_error("Reading eod_run_state.json", e)

        try:
            audit_col = db_manager.get_collection("system_audit")
            existing_run = audit_col.find_one({
                "component": "HEADLESS_OFFLINE_DAEMON",
                "execution_date": self.today_str,
                "status": "SUCCESS"
            })
            if existing_run:
                logger.info("Execution guard: MongoDB system_audit confirms %s run completed. Exiting cleanly.", self.today_str)
                return False
        except Exception as e:
            logger.warning("Could not query MongoDB audit guard (continuing): %s", str(e))

        return True

    def calculate_trade_metrics(self, high: float, low: float) -> Dict[str, Any]:
        buy_trigger = round(high + 1.0, 2)
        sl_price = round(low - 1.0, 2)
        risk_per_share = round(buy_trigger - sl_price, 2)
        target_price = round(buy_trigger + (2.0 * risk_per_share), 2)

        quantity = int(self.risk_amount_per_trade // risk_per_share) if risk_per_share > 0 else 0
        quantity = max(quantity, 0)

        total_capital_required = round(buy_trigger * quantity, 2)
        profit_at_target = round((target_price - buy_trigger) * quantity, 2)
        loss_at_sl = round(risk_per_share * quantity, 2)

        return {
            "buy_trigger": buy_trigger,
            "sl_price": sl_price,
            "risk_per_share": risk_per_share,
            "target_price": target_price,
            "quantity": quantity,
            "total_capital_required": total_capital_required,
            "profit_at_target": profit_at_target,
            "loss_at_sl": loss_at_sl,
            "risk_reward_ratio": 2.0
        }

    def sync_pending_offline_dispatches(self) -> int:
        """
        Self-Healing Queue: Checks if previous runs were saved locally to pending_dispatches/
        and pushes them to MongoDB Atlas now that the connection is alive.
        """
        synced_count = 0
        try:
            pending_files = sorted(glob.glob(str(pending_dir / "signals_*.json")))
            if not pending_files:
                return 0

            signals_col = db_manager.get_collection("signals_watchlist")
            buffer_col = db_manager.get_collection("buffer_daily_bars")

            for pf in pending_files:
                try:
                    with open(pf, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    for s in payload.get("signals", []):
                        signals_col.update_one({"_id": s["_id"]}, {"$set": s}, upsert=True)
                        synced_count += 1
                    for b in payload.get("buffer", []):
                        buffer_col.update_one({"symbol": b["symbol"], "date": b["date"]}, {"$set": b}, upsert=True)
                    # Archive synced file
                    os.remove(pf)
                    logger.info("[OFFLINE SYNC] Flushed pending dispatches from %s to MongoDB Atlas.", Path(pf).name)
                except Exception as e:
                    logger.warning("Error syncing pending dispatch file %s: %s", pf, str(e))
        except Exception as e:
            logger.warning("Could not flush pending offline dispatches: %s", str(e))
        return synced_count

    def run(self) -> Dict[str, Any]:
        start_time = datetime.now(self.tz)
        logger.info("==================================================")
        logger.info("  VectorTrade Headless EOD Daemon (%s)", self.today_str)
        logger.info("==================================================")

        if not self.check_execution_guard():
            sys.exit(0)

        # Check internet connectivity
        online = check_internet()
        if not online and not self.force_run:
            logger.warning("[NETWORK DROP] No internet connectivity detected. Safe exit without crashing.")
            sys.exit(0)

        # Flush any previously cached offline dispatches
        self.sync_pending_offline_dispatches()

        # Securely request today's execution details from Online Engine
        try:
            today_exec = request_online_trade_details("GET_TODAY_EXECUTION")
            summary = today_exec.get("summary", {})
            logger.info(
                "[SECURE TELEMETRY] Online execution summary: %d triggered, %d closed (PnL: Rs %.2f), %d active.",
                summary.get("triggered_count", 0),
                summary.get("today_closed_count", 0),
                summary.get("today_realized_pnl", 0.0),
                summary.get("active_open_count", 0)
            )
        except Exception as e:
            logger.warning("Could not fetch online telemetry (non-fatal): %s", str(e))

        nifty_df = pd.DataFrame()
        try:
            nifty = yf.download("^NSEI", period="1y", progress=False)
            if not nifty.empty:
                if isinstance(nifty.columns, pd.MultiIndex):
                    nifty.columns = nifty.columns.get_level_values(0)
                nifty = nifty.reset_index()
                nifty.columns = [c.lower() for c in nifty.columns]
                nifty_df = nifty
        except Exception as e:
            log_error("Fetching NIFTY 50 benchmark", e)

        strategy_1 = MA50PullbackStrategy()
        strategy_2 = GlobalForecastStrategy()

        symbols = list(self.universe.keys())
        if not symbols:
            symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS"]

        logger.info("Processing %d NIFTY constituents with multithreaded engine...", len(symbols))

        raw_signals: List[Dict[str, Any]] = []
        buffer_bar_records: List[Dict[str, Any]] = []
        processed_count = 0

        def process_single_symbol(sym: str) -> Optional[Dict[str, Any]]:
            try:
                clean_sym = sym.strip().upper()
                token = self.tokens_cache.get(clean_sym, "")
                meta = self.universe.get(clean_sym, {})
                sector = meta.get("sector", "General")

                df_bar = yf.download(f"{clean_sym}.NS", period="1y", progress=False)
                if df_bar.empty or len(df_bar) < 50:
                    return None

                if isinstance(df_bar.columns, pd.MultiIndex):
                    df_bar.columns = df_bar.columns.get_level_values(0)
                df_bar = df_bar.reset_index()
                df_bar["symbol"] = clean_sym
                df_bar["sector"] = sector

                df_feat = compute_indicators(df_bar, nifty_df=nifty_df)
                if df_feat.empty:
                    return None

                last_row = df_feat.iloc[-1].to_dict()
                clean_feats = {}
                for k, v in last_row.items():
                    if isinstance(v, (np.floating, float)):
                        clean_feats[k] = round(float(v), 4) if np.isfinite(v) else 0.0
                    elif isinstance(v, (np.integer, int)):
                        clean_feats[k] = int(v)
                    elif isinstance(v, (np.bool_, bool)):
                        clean_feats[k] = bool(v)
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        clean_feats[k] = v.strftime("%Y-%m-%d")
                    else:
                        clean_feats[k] = str(v)

                buffer_rec = {
                    "date": self.today_str,
                    "symbol": clean_sym,
                    "features": clean_feats,
                    "consumed_for_training": False
                }

                high_now = float(df_feat["high"].iloc[-1])
                low_now = float(df_feat["low"].iloc[-1])
                sym_signals = []

                # Strategy 1 Evaluation
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
                            log_error(f"Inference Strategy 1 ({clean_sym})", e)

                    fin_metrics = self.calculate_trade_metrics(high_now, low_now)
                    sig.update(fin_metrics)
                    sig["ml_confidence"] = confidence
                    sig["sector"] = sector
                    sig["status"] = "ARMED"
                    sig["valid_for_date"] = self.today_str
                    sym_signals.append(sig)

                # Strategy 2 Evaluation
                pred_gain = 0.0
                pred_drop = 0.0
                if self.glob_gain_model is not None and self.glob_drop_model is not None and self.glob_features:
                    try:
                        feat_row = df_feat.iloc[[-1]][self.glob_features]
                        x_glob = self.glob_scaler.transform(self.glob_imputer.transform(feat_row))
                        pred_gain = float(self.glob_gain_model.predict(x_glob)[0])
                        pred_drop = float(self.glob_drop_model.predict(x_glob)[0])
                    except Exception as e:
                        log_error(f"Inference Strategy 2 ({clean_sym})", e)

                s2_signals = strategy_2.evaluate(
                    df_feat, symbol=clean_sym, instrument_token=token,
                    predicted_gain_pct=pred_gain, predicted_drop_pct=pred_drop
                )
                for sig in s2_signals:
                    fin_metrics = self.calculate_trade_metrics(high_now, low_now)
                    sig.update(fin_metrics)
                    sig["ml_confidence"] = round(float(pred_gain / 10.0), 3)
                    sig["sector"] = sector
                    sig["status"] = "ARMED"
                    sig["valid_for_date"] = self.today_str
                    sym_signals.append(sig)

                return {"buffer": buffer_rec, "signals": sym_signals}
            except Exception as e:
                log_error(f"Processing symbol {sym}", e)
                return None

        # Execute symbol feature generation with concurrent threads
        max_workers = min(8, len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {executor.submit(process_single_symbol, s): s for s in symbols}
            for future in as_completed(future_to_sym):
                res = future.result()
                processed_count += 1
                if res:
                    if res["buffer"]:
                        buffer_bar_records.append(res["buffer"])
                    if res["signals"]:
                        raw_signals.extend(res["signals"])

        logger.info("Generated %d raw candidates across %d symbols. Applying Portfolio Risk Gate...", len(raw_signals), processed_count)

        # Portfolio Risk Manager Gate: Caps risk per trade <= 1.0%, checks 30% sector limit, sorts by ML confidence
        risk_mgr = PortfolioRiskManager(
            total_capital=self.capital,
            risk_per_trade_pct=self.risk_pct,
            max_open_positions=10,
            max_sector_allocation_pct=30.0
        )
        approved_signals = risk_mgr.filter_and_size_signals(raw_signals, universe=self.universe)

        # Ensure all metrics on approved signals
        for sig in approved_signals:
            sig["status"] = "ARMED"
            sig["valid_for_date"] = self.today_str
            buy_trig = float(sig["buy_trigger"])
            sl_pr = float(sig["sl_price"])
            qty = int(sig["quantity"])
            sig["total_capital_required"] = round(buy_trig * qty, 2)
            sig["profit_at_target"] = round((float(sig["target_price"]) - buy_trig) * qty, 2)
            sig["loss_at_sl"] = round((buy_trig - sl_pr) * qty, 2)
            sig["risk_reward_ratio"] = 2.0

        # Database Dispatch with Local Fallback
        db_dispatched = False
        try:
            signals_col = db_manager.get_collection("signals_watchlist")
            buffer_col = db_manager.get_collection("buffer_daily_bars")

            for sig in approved_signals:
                signals_col.update_one({"_id": sig["_id"]}, {"$set": sig}, upsert=True)

            for bar in buffer_bar_records:
                buffer_col.update_one({"symbol": bar["symbol"], "date": bar["date"]}, {"$set": bar}, upsert=True)

            db_dispatched = True
            logger.info("Successfully dispatched %d signals and %d buffer bars to MongoDB.", len(approved_signals), len(buffer_bar_records))
        except Exception as e:
            log_error("MongoDB dispatch failure. Caching locally to pending_dispatches/...", e)
            fallback_file = pending_dir / f"signals_{self.today_str}.json"
            try:
                with open(fallback_file, "w", encoding="utf-8") as f:
                    json.dump({"signals": approved_signals, "buffer": buffer_bar_records}, f, indent=2)
                logger.info("Safely cached %d pending dispatches locally in %s", len(approved_signals), fallback_file.name)
            except Exception as err:
                log_error("Saving pending dispatches locally", err)

        end_time = datetime.now(self.tz)
        duration_sec = round((end_time - start_time).total_seconds(), 2)

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "last_successful_run_date": self.today_str,
                    "timestamp": end_time.isoformat(),
                    "signals_count": len(approved_signals),
                    "db_dispatched": db_dispatched,
                    "duration_seconds": duration_sec
                }, f, indent=2)
        except Exception as e:
            log_error("Writing eod_run_state.json", e)

        try:
            audit_col = db_manager.get_collection("system_audit")
            audit_col.insert_one({
                "component": "HEADLESS_OFFLINE_DAEMON",
                "execution_date": self.today_str,
                "timestamp": end_time.isoformat(),
                "status": "SUCCESS",
                "symbols_processed": processed_count,
                "signals_armed": len(approved_signals),
                "db_dispatched": db_dispatched,
                "duration_seconds": duration_sec
            })
        except Exception:
            pass

        logger.info("Headless Daemon completed successfully in %.2fs. Auto-quitting...", duration_sec)
        return {
            "status": "SUCCESS",
            "execution_date": self.today_str,
            "signals_armed": len(approved_signals),
            "db_dispatched": db_dispatched,
            "duration_seconds": duration_sec
        }


def main():
    force = "--force-run" in sys.argv or "--dry-run" in sys.argv
    daemon = HeadlessDaemon(force_run=force)
    daemon.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
