"""
VectorTrade - Offline Daily 4:00 PM Automated Scheduler & Self-Closer
Continuously monitors system time, internet connectivity, and market status:
1. 00:00 AM - 03:59 PM IST: Sits idle, checks every 1-2 minutes without heavy work.
2. 04:00 PM - 11:59 PM IST (Auto-run window):
   - First checks if today's analysis and MongoDB Atlas upload were already completed.
     If yes, initiates graceful self-close (os._exit(0)).
   - Checks if user laptop has active internet connectivity.
     If no, logs status and waits for connection.
     If yes, executes daily stock analysis for all enabled/checked strategies,
     uploads results to MongoDB Atlas (signals_watchlist), updates daily_analysis.json,
     and gracefully self-closes once completed.
"""

from __future__ import annotations
import os
import sys
import time
import socket
import logging
import threading
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytz

# Setup base logger
logger = logging.getLogger("VectorTrade.Scheduler")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [Scheduler] %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Base directories resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from strategies.manager import StrategyManager

# Timezone & Schedule Constants
KOLKATA_TZ = pytz.timezone("Asia/Kolkata")
AUTO_RUN_START_HOUR = 16  # 4:00 PM IST
AUTO_RUN_END_HOUR = 23    # 11:59 PM IST
DEFAULT_TICK_INTERVAL_SECONDS = 60  # Check every 1 minute


def check_internet_connection(timeout: float = 3.0) -> bool:
    """
    Checks if the local system has active internet connectivity
    using fast DNS socket probes with HTTP fallback.
    """
    probe_hosts = [
        ("8.8.8.8", 53),          # Google DNS
        ("1.1.1.1", 53),          # Cloudflare DNS
        ("208.67.222.222", 53),   # OpenDNS
    ]
    for host, port in probe_hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            continue

    # Fallback HTTP probe
    try:
        import urllib.request
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False


def show_windows_close_prompt(
    title: str = "VectorTrade - Daily Analysis Completed",
    message: str = (
        "Daily analysis for today has already completed and auto-closed.\n\n"
        "Do you want to close VectorTrade now?\n\n"
        "- Click YES to close the application.\n"
        "- Click NO to keep it running so you can view the dashboard and orders."
    )
) -> str:
    """
    Displays a native Windows modal prompt with Yes/No buttons.
    Returns:
        "CLOSE" if user clicks Yes (IDYES = 6).
        "KEEP_RUNNING" if user clicks No (IDNO = 7).
    Supports headless/unit-test mocking via show_windows_close_prompt.mock_response.
    """
    mock = getattr(show_windows_close_prompt, "mock_response", None)
    if mock is not None:
        logger.info(f"[TEST_MODE] show_windows_close_prompt mocked response: {mock}")
        return mock

    try:
        import ctypes
        # MB_YESNO (0x04) | MB_ICONQUESTION (0x20) | MB_TOPMOST (0x40000) | MB_SETFOREGROUND (0x10000)
        flags = 0x04 | 0x20 | 0x40000 | 0x10000
        # IDYES = 6, IDNO = 7
        response = ctypes.windll.user32.MessageBoxW(0, message, title, flags)
        if response == 6:
            return "CLOSE"
        else:
            return "KEEP_RUNNING"
    except Exception as e:
        logger.error(f"Error displaying Windows MessageBox: {e}")
        return "KEEP_RUNNING"


def trigger_self_close(delay_seconds: float = 3.0, reason: str = "") -> None:
    """
    Triggers a graceful delayed application self-close.
    Can be disabled during automated unit testing by setting trigger_self_close.disabled = True.
    """
    if getattr(trigger_self_close, "disabled", False):
        logger.info(f"[TEST_MODE] Self-close intercepted (Reason: {reason})")
        return

    def _worker():
        logger.info(f"Self-close scheduled in {delay_seconds:.1f}s. Reason: {reason}")
        time.sleep(delay_seconds)
        logger.info("Gracefully self-closing application process now (Exit Code 0).")
        try:
            json_manager.log_app_event(
                event="APP_SELF_CLOSED",
                status="Success",
                message=f"Application self-closed automatically: {reason}"
            )
        except Exception:
            pass
        try:
            from tray_app import stop_tray_app
            stop_tray_app()
        except Exception:
            pass
        try:
            import subprocess
            subprocess.Popen(f"taskkill /F /T /PID {os.getpid()}", shell=True)
        except Exception:
            pass
        os._exit(0)

    close_thread = threading.Thread(target=_worker, daemon=True, name="VectorTradeSelfCloseWorker")
    close_thread.start()


class OfflineDailyScheduler:
    """
    Thread-safe 4:00 PM Automated Daily Scheduler & Self-Closer.
    Monitors IST time every 1-2 minutes and manages auto-run and self-closing lifecycles.
    """

    def __init__(self, check_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS):
        self.check_interval_seconds = max(10, check_interval_seconds)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_executing_pipeline = False
        self._self_close_scheduled = False
        self._last_tick_time: Optional[str] = None
        self._last_internet_status: Optional[bool] = None
        self._user_opted_to_keep_running_for_date: Optional[str] = None
        self._status_message = "Scheduler initialized. Waiting for first tick."

    def get_time_window_info(self) -> Dict[str, Any]:
        """
        Determines current IST time, active time window, and countdown to 4:00 PM.
        """
        now_ist = datetime.now(KOLKATA_TZ)
        today_str = now_ist.strftime("%Y-%m-%d")
        current_hour = now_ist.hour
        current_minute = now_ist.minute
        time_formatted = now_ist.strftime("%H:%M:%S")

        # 00:00 to 15:59: Idle window
        # 16:00 to 23:59: Auto-run window
        if current_hour >= AUTO_RUN_START_HOUR:
            window = "AUTORUN_WINDOW"
            window_label = "Auto-Run Window (04:00 PM - 11:59 PM IST)"
            time_until_4pm = "0m (Active Window)"
            is_active_window = True
        else:
            window = "IDLE_WAITING_FOR_4PM"
            window_label = "Idle Window (00:00 AM - 03:59 PM IST)"
            target_4pm = now_ist.replace(hour=16, minute=0, second=0, microsecond=0)
            diff_seconds = int((target_4pm - now_ist).total_seconds())
            hours_left = diff_seconds // 3600
            minutes_left = (diff_seconds % 3600) // 60
            time_until_4pm = f"{hours_left}h {minutes_left}m"
            is_active_window = False

        return {
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "today_date": today_str,
            "current_hour": current_hour,
            "current_minute": current_minute,
            "time_formatted": time_formatted,
            "window": window,
            "window_label": window_label,
            "time_until_4pm": time_until_4pm,
            "is_active_window": is_active_window
        }

    def check_today_completed(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Checks daily_analysis.json to see if today's analysis and MongoDB sync are already finished.
        """
        return json_manager.is_daily_analysis_and_sync_done_for_today(target_date=target_date)

    def run_tick_once(self, force_execution: bool = False, allow_interactive_prompt: bool = True) -> Dict[str, Any]:
        """
        Executes a single evaluation tick:
        1. Evaluates IST time and active window.
        2. Evaluates daily_analysis.json completion state for today.
        3. If 4:00 PM window and completed today -> triggers self-close (or prompts if reopened).
        4. If 4:00 PM window and NOT completed -> checks internet; if connected, runs pipeline and self-closes.
        5. If idle window (00:00 - 15:59) -> stays idle.
        """
        with self._lock:
            win_info = self.get_time_window_info()
            today_str = win_info["today_date"]
            self._last_tick_time = win_info["current_time_ist"]

            # Query enabled strategies
            strat_mgr = StrategyManager()
            active_list = strat_mgr.get_enabled_strategies()
            enabled_strategies = [s.get("name", s.get("strategy_id")) for s in active_list]
            enabled_strat_ids = [s.get("strategy_id", s.get("id")) for s in active_list]

            # Check if today's analysis + mongo sync is already completed
            done_info = self.check_today_completed(today_str)
            is_today_done = done_info["is_done"]

            # Case 1: In Auto-Run window and today is ALREADY done
            if win_info["is_active_window"] and is_today_done and not force_execution:
                # Check if user already dismissed/opted to keep running today
                if self._user_opted_to_keep_running_for_date == today_str:
                    msg = f"Today's analysis ({today_str}) completed. User opted to keep application running for this session."
                    self._status_message = msg
                    return {
                        "action": "KEEP_RUNNING_USER_OPTED",
                        "window": win_info["window"],
                        "is_today_done": True,
                        "self_close_scheduled": False,
                        "message": msg,
                        "window_info": win_info,
                        "enabled_strategies": enabled_strategies
                    }

                # Check how many times the application has already auto-closed today
                auto_close_count = json_manager.get_auto_close_count_for_date(today_str)

                if auto_close_count >= 1:
                    # If this tick was triggered by the background loop, do NOT show modal prompt
                    if not allow_interactive_prompt:
                        msg = f"Today's analysis ({today_str}) was completed and auto-closed {auto_close_count} time(s). Background scheduler is idle."
                        self._status_message = msg
                        return {
                            "action": "ALREADY_COMPLETED_BACKGROUND_IDLE",
                            "window": win_info["window"],
                            "is_today_done": True,
                            "self_close_scheduled": False,
                            "message": msg,
                            "window_info": win_info,
                            "enabled_strategies": enabled_strategies
                        }

                    # User reopened the app after it had already auto-closed once today.
                    # Prompt user via native Windows modal dialog: Close or Keep Running.
                    prompt_title = f"VectorTrade - Daily Analysis Completed ({today_str})"
                    prompt_msg = (
                        f"Daily analysis and MongoDB Atlas sync for today ({today_str}) have already completed.\n"
                        f"The application has previously auto-closed ({auto_close_count} time{'s' if auto_close_count > 1 else ''}).\n\n"
                        f"Do you want to close VectorTrade now?\n\n"
                        f"- Click YES to close the application.\n"
                        f"- Click NO to keep it running until tomorrow so you can view the dashboard and trade book."
                    )
                    user_choice = show_windows_close_prompt(title=prompt_title, message=prompt_msg)

                    if user_choice == "CLOSE":
                        msg = f"User chose YES to close VectorTrade after prior auto-close on {today_str}."
                        self._status_message = msg
                        logger.info(msg)
                        json_manager.record_auto_close_event(today_str, reason=msg)
                        self._self_close_scheduled = True
                        trigger_self_close(delay_seconds=0.5, reason=msg)
                        return {
                            "action": "SELF_CLOSE_USER_CONFIRMED",
                            "window": win_info["window"],
                            "is_today_done": True,
                            "self_close_scheduled": True,
                            "message": msg,
                            "window_info": win_info,
                            "enabled_strategies": enabled_strategies
                        }
                    else:
                        # User clicked NO -> Keep running until tomorrow!
                        self._user_opted_to_keep_running_for_date = today_str
                        msg = f"User clicked NO (keep running) for {today_str}. VectorTrade will stay running until tomorrow."
                        self._status_message = msg
                        logger.info(msg)
                        try:
                            json_manager.log_app_event(
                                event="USER_KEPT_RUNNING",
                                status="Success",
                                message=msg,
                                extra={"target_date": today_str}
                            )
                        except Exception:
                            pass
                        return {
                            "action": "KEEP_RUNNING_USER_REQUESTED",
                            "window": win_info["window"],
                            "is_today_done": True,
                            "self_close_scheduled": False,
                            "message": msg,
                            "window_info": win_info,
                            "enabled_strategies": enabled_strategies
                        }
                else:
                    # First time auto-closing today (auto_close_count == 0)
                    msg = f"Today's analysis ({today_str}) & MongoDB sync are already COMPLETED. Auto-closing application for today."
                    self._status_message = msg
                    logger.info(msg)
                    json_manager.record_auto_close_event(today_str, reason=msg)
                    if not self._self_close_scheduled:
                        self._self_close_scheduled = True
                        trigger_self_close(delay_seconds=3.0, reason=msg)
                    return {
                        "action": "SELF_CLOSE_TRIGGERED",
                        "window": win_info["window"],
                        "is_today_done": True,
                        "self_close_scheduled": True,
                        "message": msg,
                        "window_info": win_info,
                        "enabled_strategies": enabled_strategies
                    }

            # Case 2: In Idle window (00:00 - 15:59) -> Sit idle
            if not win_info["is_active_window"] and not force_execution:
                msg = f"Idle mode: {win_info['time_formatted']} IST. Waiting for 4:00 PM (starts in {win_info['time_until_4pm']})."
                self._status_message = msg
                return {
                    "action": "IDLE_WAITING",
                    "window": win_info["window"],
                    "is_today_done": is_today_done,
                    "self_close_scheduled": False,
                    "message": msg,
                    "window_info": win_info,
                    "enabled_strategies": enabled_strategies
                }

            # Case 3: In Auto-Run window (or force_execution) and NOT yet done today
            # Check internet connectivity first
            has_internet = check_internet_connection(timeout=3.0)
            self._last_internet_status = has_internet

            if not has_internet:
                msg = f"Auto-run window active ({win_info['time_formatted']} IST), but NO INTERNET connection detected. Waiting for connection."
                self._status_message = msg
                logger.warning(msg)
                return {
                    "action": "WAITING_FOR_INTERNET",
                    "window": win_info["window"],
                    "internet_connected": False,
                    "is_today_done": False,
                    "self_close_scheduled": False,
                    "message": msg,
                    "window_info": win_info,
                    "enabled_strategies": enabled_strategies
                }

            # Case 4: In Auto-run window + Internet is available + Not done -> RUN PIPELINE!
            if self._is_executing_pipeline:
                return {
                    "action": "PIPELINE_ALREADY_RUNNING",
                    "window": win_info["window"],
                    "internet_connected": True,
                    "is_today_done": False,
                    "self_close_scheduled": False,
                    "message": "Analysis pipeline currently in progress...",
                    "window_info": win_info,
                    "enabled_strategies": enabled_strategies
                }

            self._is_executing_pipeline = True
            msg = f"Auto-run active & internet connected. Running daily analysis for {len(enabled_strategies)} enabled strategies: {enabled_strategies}..."
            self._status_message = msg
            logger.info(msg)

            try:
                from offline.daily_analyzer import run_daily_analysis_pipeline
                result = run_daily_analysis_pipeline(
                    force_download=False,
                    execution_trigger="AUTO_4PM_TIMER"
                )

                mongo_res = result.get("mongo_sync", {})
                mongo_status = mongo_res.get("status", "")
                finalized_count = result.get("finalized_count", 0)

                success_msg = (
                    f"Daily analysis finished successfully. Finalized {finalized_count} setups. "
                    f"MongoDB sync: {mongo_status}. Initiating application self-close."
                )
                self._status_message = success_msg
                logger.info(success_msg)

                # Record the auto-close event so logs have the lifecycle entry
                json_manager.record_auto_close_event(today_str, reason=success_msg)

                # Trigger self-close now that analysis + upload is completed
                if not self._self_close_scheduled:
                    self._self_close_scheduled = True
                    trigger_self_close(delay_seconds=4.0, reason=success_msg)

                return {
                    "action": "PIPELINE_COMPLETED_SELF_CLOSING",
                    "window": win_info["window"],
                    "internet_connected": True,
                    "is_today_done": True,
                    "self_close_scheduled": True,
                    "finalized_count": finalized_count,
                    "mongo_status": mongo_status,
                    "message": success_msg,
                    "window_info": win_info,
                    "enabled_strategies": enabled_strategies
                }

            except Exception as e:
                err_msg = f"Daily analysis pipeline failed: {e}"
                self._status_message = err_msg
                logger.error(err_msg, exc_info=True)
                return {
                    "action": "PIPELINE_ERROR",
                    "window": win_info["window"],
                    "internet_connected": True,
                    "is_today_done": False,
                    "self_close_scheduled": False,
                    "error": str(e),
                    "message": err_msg,
                    "window_info": win_info,
                    "enabled_strategies": enabled_strategies
                }
            finally:
                self._is_executing_pipeline = False

    def _scheduler_loop(self):
        """Continuous background scheduling loop running every 1-2 minutes."""
        logger.info(f"OfflineDailyScheduler background thread started (Tick interval: {self.check_interval_seconds}s).")
        while self._running:
            try:
                self.run_tick_once(force_execution=False, allow_interactive_prompt=False)
            except Exception as e:
                logger.error(f"Error during scheduler tick: {e}", exc_info=True)

            # Sleep in 1-second slices so thread can terminate cleanly when stopped
            for _ in range(self.check_interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("OfflineDailyScheduler background thread stopped.")

    def start(self):
        """Starts the scheduler in a background daemon thread."""
        with self._lock:
            if self._running:
                logger.warning("Scheduler is already running.")
                return
            self._running = True
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="VectorTradeDailyScheduler")
            self._thread.start()
            logger.info("Scheduler started successfully.")

    def stop(self):
        """Stops the scheduler loop."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Scheduler stopped.")

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive real-time status of the scheduler."""
        win_info = self.get_time_window_info()
        strat_mgr = StrategyManager()
        enabled_strats = [s.get("name", s.get("strategy_id")) for s in strat_mgr.get_enabled_strategies()]
        done_info = self.check_today_completed(win_info["today_date"])

        return {
            "status": "RUNNING" if self._running else "STOPPED",
            "is_running": self._running,
            "window": win_info["window"],
            "window_label": win_info["window_label"],
            "is_active_window": win_info["is_active_window"],
            "current_time_ist": win_info["current_time_ist"],
            "time_until_4pm": win_info["time_until_4pm"],
            "today_date": win_info["today_date"],
            "today_analysis_done": done_info["is_done"],
            "daily_analysis_status": done_info.get("status", "UNKNOWN"),
            "mongo_synced": done_info.get("mongo_synced", False),
            "mongo_synced_at": done_info.get("mongo_synced_at", ""),
            "last_run_timestamp": done_info.get("last_run_timestamp", ""),
            "internet_connected": self._last_internet_status if self._last_internet_status is not None else check_internet_connection(timeout=1.5),
            "enabled_strategies": enabled_strats,
            "auto_close_count": json_manager.get_auto_close_count_for_date(win_info["today_date"]),
            "user_opted_to_keep_running": self._user_opted_to_keep_running_for_date == win_info["today_date"],
            "self_close_scheduled": self._self_close_scheduled,
            "is_executing_pipeline": self._is_executing_pipeline,
            "last_tick_time": self._last_tick_time,
            "tick_interval_seconds": self.check_interval_seconds,
            "message": self._status_message
        }


# Global singleton instance for the process
_scheduler_instance: Optional[OfflineDailyScheduler] = None
_scheduler_lock = threading.Lock()


def get_daily_scheduler(check_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS) -> OfflineDailyScheduler:
    """Returns or initializes the process-wide OfflineDailyScheduler instance."""
    global _scheduler_instance
    with _scheduler_lock:
        if _scheduler_instance is None:
            _scheduler_instance = OfflineDailyScheduler(check_interval_seconds=check_interval_seconds)
        return _scheduler_instance
