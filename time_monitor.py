"""
VectorTrade - Ultra-Lightweight 2-Second Time Monitor Daemon
Monitors system clock every 2 seconds with near-zero CPU and RAM footprint.
When current time is 4:00 PM (16:00) or later and today's session has not run:
- Launches run_headless_daily_pipeline.py (download -> analyse -> upload -> update logs).
- Runs cleanly as a batch process, updates state, and returns to silent sleep.
- No background web server or port binding is created or sustained.
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_STORE_DIR = BASE_DIR / "data_store"
DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_STORE_DIR / "time_monitor_state.json"
LOG_FILE = DATA_STORE_DIR / "time_monitor.log"
PIPELINE_SCRIPT = BASE_DIR / "run_headless_daily_pipeline.py"


def log_msg(msg: str) -> None:
    """Safe logger that writes to both time_monitor.log and stdout (if available)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

    if sys.stdout is not None:
        try:
            print(formatted)
        except Exception:
            pass


def load_monitor_state() -> dict:
    """Reads persisted monitor state to avoid re-triggering on same date."""
    if not STATE_FILE.exists():
        return {
            "last_run_date": None,
            "last_run_time": None,
            "status": "INITIALIZED",
            "last_exit_code": 0
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "last_run_date": None,
            "last_run_time": None,
            "status": "INITIALIZED",
            "last_exit_code": 0
        }


def save_monitor_state(state: dict) -> None:
    """Atomically persists state to disk."""
    temp_file = STATE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        temp_file.replace(STATE_FILE)
    except Exception as e:
        log_msg(f"Warning: Could not save state: {e}")


def should_trigger_pipeline(
    now: datetime,
    last_run_date: str | None,
    target_hour: int = 16,
    force_now: bool = False
) -> bool:
    """
    Evaluates whether the daily pipeline should be triggered:
    1. If force_now is True -> True
    2. If current local hour >= target_hour (4 PM IST) and today hasn't executed -> True
    3. Otherwise -> False
    """
    if force_now:
        return True

    today_str = now.strftime("%Y-%m-%d")
    if now.hour >= target_hour:
        if last_run_date != today_str:
            return True

    return False


def execute_headless_pipeline(force: bool = False) -> int:
    """
    Executes run_headless_daily_pipeline.py via subprocess.
    The child process handles download, analysis, MongoDB upload, and log updates,
    then terminates completely.
    """
    cmd = [sys.executable, str(PIPELINE_SCRIPT), "--trigger", "TIME_MONITOR_SCHEDULE"]
    if force:
        cmd.append("--force")

    log_msg(f"Launching VectorTrade Headless Daily Pipeline: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=False
        )
        return proc.returncode
    except Exception as exc:
        log_msg(f"[ERROR] Failed to spawn pipeline process: {exc}")
        return 1


def run_monitor_loop(
    interval: float = 2.0,
    target_hour: int = 16,
    run_now: bool = False,
    once: bool = False,
    verbose: bool = False
) -> None:
    """
    Core monitoring loop ticking every `interval` seconds (default: 2.0s).
    Consumes ~0% CPU via kernel thread sleep.
    """
    state = load_monitor_state()
    last_run_date = state.get("last_run_date")

    log_msg(f"VectorTrade Time Monitor started. Interval: {interval}s, Target: {target_hour}:00, Last Run: {last_run_date}")

    retry_delay_seconds = 60.0
    last_error_time = 0.0

    while True:
        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            trigger = should_trigger_pipeline(
                now=now,
                last_run_date=last_run_date,
                target_hour=target_hour,
                force_now=run_now
            )

            # Prevent immediate rapid retries if an error occurred recently
            if trigger and (time.time() - last_error_time < retry_delay_seconds) and not run_now:
                trigger = False

            if trigger:
                log_msg(f"Time condition met (Hour: {now.hour} >= {target_hour}). Triggering EOD pipeline...")
                exit_code = execute_headless_pipeline()

                if exit_code == 0:
                    last_run_date = today_str
                    state["last_run_date"] = today_str
                    state["last_run_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    state["status"] = "SUCCESS"
                    state["last_exit_code"] = 0
                    save_monitor_state(state)
                    log_msg("Pipeline completed successfully. Resuming silent sleep.")
                else:
                    last_error_time = time.time()
                    state["status"] = "ERROR"
                    state["last_exit_code"] = exit_code
                    state["last_attempt_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    save_monitor_state(state)
                    log_msg(f"Pipeline exited with error code {exit_code}. Backing off {retry_delay_seconds}s.")

                if run_now:
                    run_now = False

            if once:
                break

            # Ultra-low resource sleep: 0% CPU, OS kernel yields execution thread
            time.sleep(interval)

        except KeyboardInterrupt:
            log_msg("Stopped by user (KeyboardInterrupt). Exiting cleanly.")
            break
        except Exception as e:
            log_msg(f"Loop exception: {e}\n{traceback.format_exc()}")
            time.sleep(interval)


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="VectorTrade 2-Second Time Monitor Daemon")
        parser.add_argument("--interval", type=float, default=2.0, help="Check interval in seconds (default: 2.0)")
        parser.add_argument("--target-hour", type=int, default=16, help="Hour to trigger daily pipeline in 24h format (default: 16 = 4:00 PM)")
        parser.add_argument("--run-now", action="store_true", help="Force immediate execution of daily pipeline")
        parser.add_argument("--once", action="store_true", help="Perform one check iteration and exit")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose tick logging")
        args = parser.parse_args()

        run_monitor_loop(
            interval=args.interval,
            target_hour=args.target_hour,
            run_now=args.run_now,
            once=args.once,
            verbose=args.verbose
        )
    except Exception as exc:
        log_msg(f"Fatal error in main: {exc}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
