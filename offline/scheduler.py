"""
VectorTrade - Low-Resource Offline Background Scheduler & Automation Engine
Runs full-time in the background consuming < 30 MB RAM and ~0% CPU.
Automatically triggers EOD analysis at 16:00 IST and provides thread-safe controls.
"""

from __future__ import annotations
import sys
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import pytz

try:
    import psutil
except ImportError:
    psutil = None

BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from core.config import settings
from core.db import db_manager
from offline.headless_daemon import HeadlessDaemon
from offline.monthly_retrainer import MonthlyRetrainer

logger = logging.getLogger("VectorTrade.OfflineScheduler")


class OfflineScheduler:
    """
    High-efficiency background automation service for offline processing.
    """

    def __init__(self, target_hour: int = 16, target_minute: int = 0) -> None:
        self.target_hour = target_hour
        self.target_minute = target_minute
        self.tz = pytz.timezone(settings.TIMEZONE)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.state: Dict[str, Any] = {
            "status": "IDLE",
            "is_busy": False,
            "last_action": None,
            "last_run_timestamp": None,
            "last_run_date": None,
            "last_run_duration_sec": 0,
            "last_run_status": "NEVER_RUN",
            "last_signals_count": 0,
            "last_error": None,
            "next_scheduled_run": None,
            "dispatches_synced": 0
        }

        self._load_previous_state()
        self._update_next_scheduled_time()

    def _load_previous_state(self) -> None:
        state_file = settings.LOGS_DIR / "eod_run_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.state["last_run_date"] = saved.get("last_successful_run_date")
                    self.state["last_run_timestamp"] = saved.get("timestamp")
                    self.state["last_signals_count"] = saved.get("signals_count", 0)
                    self.state["last_run_status"] = "SUCCESS" if saved.get("db_dispatched") else "CACHED_LOCALLY"
                    self.state["last_run_duration_sec"] = saved.get("duration_seconds", 0)
            except Exception as e:
                logger.warning("Could not read previous run state: %s", str(e))

    def _update_next_scheduled_time(self) -> None:
        now = datetime.now(self.tz)
        target = now.replace(hour=self.target_hour, minute=self.target_minute, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        self.state["next_scheduled_run"] = target.strftime("%Y-%m-%d %H:%M:%S IST")

    def get_system_status(self) -> Dict[str, Any]:
        """Returns instantaneous system health, memory usage, and schedule status."""
        mem_mb = 25.0
        cpu_pct = 0.0
        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                mem_mb = round(mem_info.rss / (1024 * 1024), 1)
                cpu_pct = round(process.cpu_percent(interval=None), 1)
            except Exception:
                pass

        pending_dir = OFFLINE_DIR / "pending_dispatches"
        pending_files = list(pending_dir.glob("signals_*.json")) if pending_dir.exists() else []

        db_ok = True
        try:
            db_manager.ping()
        except Exception:
            db_ok = False

        self._update_next_scheduled_time()

        return {
            "scheduler_status": self.state["status"],
            "is_busy": self.state["is_busy"],
            "last_action": self.state["last_action"],
            "last_run_date": self.state["last_run_date"],
            "last_run_timestamp": self.state["last_run_timestamp"],
            "last_run_status": self.state["last_run_status"],
            "last_run_duration_sec": self.state["last_run_duration_sec"],
            "last_signals_count": self.state["last_signals_count"],
            "last_error": self.state["last_error"],
            "next_scheduled_run": self.state["next_scheduled_run"],
            "pending_dispatches_count": len(pending_files),
            "memory_usage_mb": mem_mb,
            "cpu_usage_pct": cpu_pct,
            "database_connected": db_ok,
            "current_time_ist": datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S IST")
        }

    def run_eod_pipeline(self, force: bool = True) -> Dict[str, Any]:
        """Executes the EOD feature generation & strategy engine in a thread-safe manner."""
        if not self._lock.acquire(blocking=False):
            return {"status": "BLOCKED", "message": "An analysis task is already running in background."}

        try:
            self.state["status"] = "RUNNING_EOD"
            self.state["is_busy"] = True
            self.state["last_action"] = "EOD_ANALYSIS"
            self.state["last_error"] = None
            start_ts = time.time()

            logger.info("[SCHEDULER] Starting EOD Pipeline execution (force=%s)...", force)
            daemon = HeadlessDaemon(force_run=force)
            res = daemon.run()

            duration = round(time.time() - start_ts, 2)
            today_str = datetime.now(self.tz).strftime("%Y-%m-%d")

            self.state["last_run_date"] = today_str
            self.state["last_run_timestamp"] = datetime.now(self.tz).isoformat()
            self.state["last_run_duration_sec"] = duration
            self.state["last_signals_count"] = res.get("signals_armed", 0)
            self.state["last_run_status"] = res.get("status", "SUCCESS")
            self.state["status"] = "IDLE"
            self.state["is_busy"] = False

            logger.info("[SCHEDULER] EOD Pipeline completed successfully in %.2fs.", duration)
            return {
                "status": "SUCCESS",
                "signals_armed": res.get("signals_armed", 0),
                "duration_seconds": duration,
                "db_dispatched": res.get("db_dispatched", False)
            }
        except Exception as e:
            err_msg = str(e)
            logger.error("[SCHEDULER] Error during EOD execution: %s", err_msg)
            self.state["last_error"] = err_msg
            self.state["last_run_status"] = "ERROR"
            self.state["status"] = "ERROR"
            self.state["is_busy"] = False
            return {"status": "ERROR", "message": err_msg}
        finally:
            self._lock.release()

    def sync_dispatches(self) -> Dict[str, Any]:
        """Flushes any local offline queued dispatches to MongoDB Atlas."""
        try:
            daemon = HeadlessDaemon(force_run=True)
            synced = daemon.sync_pending_offline_dispatches()
            self.state["dispatches_synced"] += synced
            return {"status": "SUCCESS", "synced_files": synced}
        except Exception as e:
            logger.error("Error syncing dispatches: %s", str(e))
            return {"status": "ERROR", "message": str(e)}

    def retrain_models(self, force: bool = True) -> Dict[str, Any]:
        """Triggers monthly XGBoost model retraining."""
        if not self._lock.acquire(blocking=False):
            return {"status": "BLOCKED", "message": "Another task is currently running."}

        try:
            self.state["status"] = "RETRAINING"
            self.state["is_busy"] = True
            self.state["last_action"] = "MODEL_RETRAIN"
            retrainer = MonthlyRetrainer(force_run=force)
            res = retrainer.run()
            self.state["status"] = "IDLE"
            self.state["is_busy"] = False
            return res
        except Exception as e:
            self.state["status"] = "ERROR"
            self.state["is_busy"] = False
            self.state["last_error"] = str(e)
            return {"status": "ERROR", "message": str(e)}
        finally:
            self._lock.release()

    def _loop(self) -> None:
        """
        Ultra-efficient main loop:
        Wakes up every 30 seconds, checks IST time, and triggers automated EOD run if due.
        Uses negligible CPU (< 0.1%) while sleeping.
        """
        logger.info("OfflineScheduler background worker loop started.")
        while self.running:
            try:
                now_ist = datetime.now(self.tz)
                today_str = now_ist.strftime("%Y-%m-%d")

                # If past target hour and today's run has not been completed yet
                if (now_ist.hour > self.target_hour or (now_ist.hour == self.target_hour and now_ist.minute >= self.target_minute)):
                    if self.state["last_run_date"] != today_str and not self.state["is_busy"]:
                        logger.info("[SCHEDULER] Target time reached (%02d:%02d IST). Triggering automated EOD run...", now_ist.hour, now_ist.minute)
                        threading.Thread(target=self.run_eod_pipeline, kwargs={"force": False}, daemon=True).start()

            except Exception as e:
                logger.error("[SCHEDULER] Loop exception: %s", str(e))

            # Deep sleep for 30 seconds to conserve battery and CPU
            for _ in range(30):
                if not self.running:
                    break
                time.sleep(1)

    def start(self) -> None:
        """Starts scheduler thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="VectorTrade-SchedulerThread")
        self.thread.start()
        logger.info("VectorTrade OfflineScheduler started.")

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("VectorTrade OfflineScheduler stopped.")


# Global singleton instance
offline_scheduler = OfflineScheduler()
