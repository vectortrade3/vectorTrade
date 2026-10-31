"""
VectorTrade - Offline Analysis Progress Tracker & State Engine
Provides thread-safe multi-stage progress tracking across background EOD runs,
persists progress to disk, and ensures seamless recovery across browser refreshes.
"""

from __future__ import annotations
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz

from core.config import settings

logger = logging.getLogger("VectorTrade.ProgressTracker")

STATE_FILE = settings.LOGS_DIR / "analysis_progress.json"


class AnalysisProgressTracker:
    """
    Thread-safe singleton managing the live state and progress bars of the offline analysis pipeline.
    """
    _instance: Optional[AnalysisProgressTracker] = None
    _lock = threading.Lock()

    def __new__(cls) -> AnalysisProgressTracker:
        if cls._instance is None:
            cls._instance = super(AnalysisProgressTracker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.tz = pytz.timezone(settings.TIMEZONE)
        self._last_save_time: float = 0.0
        self.state: Dict[str, Any] = {
            "is_running": False,
            "stage": "IDLE",           # IDLE, DOWNLOADING, CALCULATING, PREDICTING, UPLOADING, COMPLETED, ERROR
            "stage_index": 0,          # 0 to 4
            "total_stages": 4,
            "stage_name": "Idle",
            "current_symbol": "",
            "message": "Ready to run analysis.",
            "download": {
                "current": 0,
                "total": 207,
                "percentage": 0.0,
                "symbol": ""
            },
            "calculate": {
                "current": 0,
                "total": 207,
                "percentage": 0.0,
                "symbol": ""
            },
            "predict": {
                "current": 0,
                "total": 207,
                "percentage": 0.0,
                "symbol": ""
            },
            "upload": {
                "status": "PENDING",    # PENDING, IN_PROGRESS, DISPATCHED, CACHED_LOCALLY
                "signals_count": 0,
                "message": ""
            },
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
            "signals": [],
            "error": None
        }
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    # If server was restarted while running, mark as interrupted
                    if saved.get("is_running"):
                        saved["is_running"] = False
                        saved["stage"] = "INTERRUPTED"
                        saved["stage_name"] = "Analysis Interrupted"
                        saved["message"] = "Previous run was interrupted by system shutdown."
                    self.state.update(saved)
            except Exception as e:
                logger.warning("Could not read analysis_progress.json: %s", str(e))

    def _save_to_disk(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_save_time < 0.25):
            return
        self._last_save_time = now
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            temp_file = STATE_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, default=str)
            for _ in range(3):
                try:
                    temp_file.replace(STATE_FILE)
                    break
                except Exception:
                    time.sleep(0.05)
        except Exception as e:
            logger.debug("Could not persist analysis_progress.json: %s", str(e))

    def start_analysis(self, total_symbols: int) -> None:
        with self._lock:
            self.state["is_running"] = True
            self.state["stage"] = "DOWNLOADING"
            self.state["stage_index"] = 1
            self.state["stage_name"] = "Stage 1: Downloading Market Data"
            self.state["started_at"] = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S IST")
            self.state["completed_at"] = None
            self.state["duration_seconds"] = 0.0
            self.state["error"] = None
            self.state["message"] = f"Initializing download for {total_symbols} stocks..."
            self.state["download"] = {"current": 0, "total": total_symbols, "percentage": 0.0, "symbol": ""}
            self.state["calculate"] = {"current": 0, "total": total_symbols, "percentage": 0.0, "symbol": ""}
            self.state["predict"] = {"current": 0, "total": total_symbols, "percentage": 0.0, "symbol": ""}
            self.state["upload"] = {"status": "PENDING", "signals_count": 0, "message": "Awaiting signals"}
            self.state["signals"] = []
            self._save_to_disk(force=True)

    def update_download_progress(self, current: int, total: int, symbol: str) -> None:
        with self._lock:
            pct = round((current / total) * 100.0, 1) if total > 0 else 0.0
            self.state["stage"] = "DOWNLOADING"
            self.state["stage_index"] = 1
            self.state["stage_name"] = "Stage 1: Downloading Market Data"
            self.state["download"] = {
                "current": current,
                "total": total,
                "percentage": pct,
                "symbol": symbol
            }
            self.state["current_symbol"] = symbol
            self.state["message"] = f"Downloaded {current}/{total} stocks ({pct}%). Latest: {symbol}"
            self._save_to_disk()

    def start_calculating(self, total_symbols: int) -> None:
        with self._lock:
            self.state["stage"] = "CALCULATING"
            self.state["stage_index"] = 2
            self.state["stage_name"] = "Stage 2: Computing Features & Cleaning Data"
            self.state["calculate"] = {"current": 0, "total": total_symbols, "percentage": 0.0, "symbol": ""}
            self.state["message"] = f"Computing indicators & cleaning data for {total_symbols} stocks..."
            self._save_to_disk(force=True)

    def update_calculate_progress(self, current: int, total: int, symbol: str) -> None:
        with self._lock:
            pct = round((current / total) * 100.0, 1) if total > 0 else 0.0
            self.state["stage"] = "CALCULATING"
            self.state["stage_index"] = 2
            self.state["stage_name"] = "Stage 2: Computing Features & Cleaning Data"
            self.state["calculate"] = {
                "current": current,
                "total": total,
                "percentage": pct,
                "symbol": symbol
            }
            self.state["current_symbol"] = symbol
            self.state["message"] = f"Processed {current}/{total} indicators ({pct}%). Latest: {symbol}"
            self._save_to_disk()

    def start_predicting(self, total_symbols: int) -> None:
        with self._lock:
            self.state["stage"] = "PREDICTING"
            self.state["stage_index"] = 3
            self.state["stage_name"] = "Stage 3: Strategy & ML Brain Inference"
            self.state["predict"] = {"current": 0, "total": total_symbols, "percentage": 0.0, "symbol": ""}
            self.state["message"] = "Evaluating 50MA Pullback and Global Forecast XGBoost Brains..."
            self._save_to_disk(force=True)

    def update_predict_progress(self, current: int, total: int, symbol: str) -> None:
        with self._lock:
            pct = round((current / total) * 100.0, 1) if total > 0 else 0.0
            self.state["stage"] = "PREDICTING"
            self.state["stage_index"] = 3
            self.state["stage_name"] = "Stage 3: Strategy & ML Brain Inference"
            self.state["predict"] = {
                "current": current,
                "total": total,
                "percentage": pct,
                "symbol": symbol
            }
            self.state["current_symbol"] = symbol
            self.state["message"] = f"Evaluated ML Brains on {current}/{total} stocks ({pct}%). Latest: {symbol}"
            self._save_to_disk()

    def start_uploading(self, signals_count: int) -> None:
        with self._lock:
            self.state["stage"] = "UPLOADING"
            self.state["stage_index"] = 4
            self.state["stage_name"] = "Stage 4: 1% Risk Sizing & MongoDB Upload"
            self.state["upload"] = {
                "status": "IN_PROGRESS",
                "signals_count": signals_count,
                "message": f"Applying 1% capital risk gate & dispatching {signals_count} signals..."
            }
            self.state["message"] = f"Sizing positions and uploading {signals_count} approved signals to MongoDB Atlas..."
            self._save_to_disk(force=True)

    def complete_analysis(self, approved_signals: List[Dict[str, Any]], duration_seconds: float, db_dispatched: bool) -> None:
        with self._lock:
            self.state["is_running"] = False
            self.state["stage"] = "COMPLETED"
            self.state["stage_index"] = 4
            self.state["stage_name"] = "Analysis Completed Successfully"
            self.state["completed_at"] = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S IST")
            self.state["duration_seconds"] = round(duration_seconds, 2)
            self.state["upload"] = {
                "status": "DISPATCHED" if db_dispatched else "CACHED_LOCALLY",
                "signals_count": len(approved_signals),
                "message": f"Successfully uploaded {len(approved_signals)} signals to MongoDB Atlas" if db_dispatched else f"Buffered {len(approved_signals)} signals locally"
            }
            self.state["signals"] = approved_signals
            self.state["message"] = f"Completed analysis in {duration_seconds:.1f}s. Generated {len(approved_signals)} buy orders."
            self._save_to_disk(force=True)

    def fail_analysis(self, error_message: str) -> None:
        with self._lock:
            self.state["is_running"] = False
            self.state["stage"] = "ERROR"
            self.state["stage_name"] = "Analysis Failed"
            self.state["error"] = error_message
            self.state["message"] = f"Error during analysis: {error_message}"
            self._save_to_disk(force=True)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.state)


progress_tracker = AnalysisProgressTracker()
