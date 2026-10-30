"""
VectorTrade - JSON Operations & Telemetry Logger
Persists structured operational audit logs in JSON format:
logs/operations_history.json

Tracks exact timestamps, status, duration, and metrics for:
1. Market Data Downloads (when last time downloaded, date range, stocks appended)
2. Machine Learning Model Training (when last time retrained, models, duration)
3. Stock Analysis & Predictions (when last time analyzed, 207 stocks, armed signals)
"""

from __future__ import annotations
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = LOGS_DIR / "operations_history.json"

logger = logging.getLogger("VectorTrade.OperationsLogger")
_lock = threading.Lock()


class OperationsLogger:
    """Manages persistent JSON history for all core vectorTrade operations."""

    def __init__(self, log_path: Path = HISTORY_FILE) -> None:
        self.log_path = log_path
        self.tz = pytz.timezone("Asia/Kolkata")
        self._ensure_file()

    def _get_now_ist(self) -> datetime:
        return datetime.now(self.tz)

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": "1.0.0",
            "last_updated": self._get_now_ist().isoformat(),
            "last_data_download": {
                "timestamp": None,
                "date": None,
                "status": "NEVER_RUN",
                "stocks_updated": 0,
                "last_date_in_dataset": None,
                "duration_seconds": 0.0,
                "message": "No download recorded yet."
            },
            "last_model_training": {
                "timestamp": None,
                "date": None,
                "status": "NEVER_RUN",
                "models_updated": [],
                "duration_seconds": 0.0,
                "message": "No model retraining recorded yet."
            },
            "last_stocks_analysis": {
                "timestamp": None,
                "date": None,
                "status": "NEVER_RUN",
                "symbols_processed": 0,
                "signals_armed": 0,
                "duration_seconds": 0.0,
                "message": "No stock analysis recorded yet."
            },
            "history": []
        }

    def _ensure_file(self) -> None:
        with _lock:
            if not self.log_path.exists():
                state = self._default_state()
                # If existing eod_run_state.json exists, seed initial analysis record
                eod_state_file = LOGS_DIR / "eod_run_state.json"
                if eod_state_file.exists():
                    try:
                        with open(eod_state_file, "r", encoding="utf-8") as f:
                            eod = json.load(f)
                            if eod.get("last_run_status") == "SUCCESS":
                                state["last_stocks_analysis"] = {
                                    "timestamp": eod.get("last_run_timestamp"),
                                    "date": eod.get("last_run_date"),
                                    "status": "SUCCESS",
                                    "symbols_processed": 207,
                                    "signals_armed": eod.get("signals_armed", 0),
                                    "duration_seconds": eod.get("last_run_duration_sec", 0.0),
                                    "message": f"Completed analysis for 207 stocks ({eod.get('signals_armed', 0)} signals armed)."
                                }
                    except Exception:
                        pass
                self._save_raw(state)

    def _save_raw(self, state: Dict[str, Any]) -> None:
        try:
            state["last_updated"] = self._get_now_ist().isoformat()
            temp_path = self.log_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            temp_path.replace(self.log_path)
        except Exception as e:
            logger.warning("Could not persist operations_history.json: %s", str(e))

    def load_log(self) -> Dict[str, Any]:
        with _lock:
            if not self.log_path.exists():
                return self._default_state()
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read operations_history.json: %s", str(e))
                return self._default_state()

    def log_data_download(
        self,
        status: str,
        stocks_updated: int,
        last_date: Optional[str] = None,
        duration_seconds: float = 0.0,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Records market data download and append operations."""
        now = self._get_now_ist()
        entry = {
            "id": f"DOWNLOAD_{int(now.timestamp())}",
            "type": "DATA_DOWNLOAD",
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "date": now.strftime("%Y-%m-%d"),
            "status": status.upper(),
            "stocks_updated": stocks_updated,
            "last_date_in_dataset": last_date,
            "duration_seconds": round(duration_seconds, 2),
            "message": message,
            "details": details or {}
        }

        with _lock:
            state = self.load_log()
            state["last_data_download"] = {
                "timestamp": entry["timestamp"],
                "formatted_time": entry["formatted_time"],
                "date": entry["date"],
                "status": entry["status"],
                "stocks_updated": stocks_updated,
                "last_date_in_dataset": last_date,
                "duration_seconds": round(duration_seconds, 2),
                "message": message
            }
            state["history"].insert(0, entry)
            state["history"] = state["history"][:100]  # retain 100 most recent operations
            self._save_raw(state)

        logger.info("[OPERATIONS_LOG] Data Download logged: %s (%d stocks, %.2fs)", status, stocks_updated, duration_seconds)
        return entry

    def log_model_training(
        self,
        status: str,
        models_updated: List[str],
        duration_seconds: float = 0.0,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Records machine learning model training operations."""
        now = self._get_now_ist()
        entry = {
            "id": f"TRAIN_{int(now.timestamp())}",
            "type": "MODEL_RETRAIN",
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "date": now.strftime("%Y-%m-%d"),
            "status": status.upper(),
            "models_updated": models_updated,
            "duration_seconds": round(duration_seconds, 2),
            "message": message,
            "details": details or {}
        }

        with _lock:
            state = self.load_log()
            state["last_model_training"] = {
                "timestamp": entry["timestamp"],
                "formatted_time": entry["formatted_time"],
                "date": entry["date"],
                "status": entry["status"],
                "models_updated": models_updated,
                "duration_seconds": round(duration_seconds, 2),
                "message": message
            }
            state["history"].insert(0, entry)
            state["history"] = state["history"][:100]
            self._save_raw(state)

        logger.info("[OPERATIONS_LOG] Model Retrain logged: %s (%s, %.2fs)", status, models_updated, duration_seconds)
        return entry

    def log_stocks_analysis(
        self,
        status: str,
        symbols_processed: int,
        signals_armed: int,
        duration_seconds: float = 0.0,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Records stock analysis and ML prediction runs."""
        now = self._get_now_ist()
        entry = {
            "id": f"ANALYSIS_{int(now.timestamp())}",
            "type": "STOCKS_ANALYSIS",
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "date": now.strftime("%Y-%m-%d"),
            "status": status.upper(),
            "symbols_processed": symbols_processed,
            "signals_armed": signals_armed,
            "duration_seconds": round(duration_seconds, 2),
            "message": message,
            "details": details or {}
        }

        with _lock:
            state = self.load_log()
            state["last_stocks_analysis"] = {
                "timestamp": entry["timestamp"],
                "formatted_time": entry["formatted_time"],
                "date": entry["date"],
                "status": entry["status"],
                "symbols_processed": symbols_processed,
                "signals_armed": signals_armed,
                "duration_seconds": round(duration_seconds, 2),
                "message": message
            }
            state["history"].insert(0, entry)
            state["history"] = state["history"][:100]
            self._save_raw(state)

        logger.info("[OPERATIONS_LOG] Stock Analysis logged: %s (%d processed, %d armed, %.2fs)",
                    status, symbols_processed, signals_armed, duration_seconds)
        return entry


# Global singleton instance
operations_logger = OperationsLogger()
