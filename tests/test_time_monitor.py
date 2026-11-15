"""
VectorTrade - Unit Tests for 2-Second Time Monitor Daemon
Tests:
1. Time evaluation logic: before 16:00 vs >= 16:00.
2. Once-per-day idempotency: skips execution if today already ran.
3. State persistence: load and atomic save to time_monitor_state.json.
4. Process spawning: subprocess invocation of run_headless_daily_pipeline.py.
5. CLI loop control: once=True terminates cleanly in 1 iteration.
"""

import sys
import unittest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import time_monitor
from time_monitor import (
    should_trigger_pipeline,
    load_monitor_state,
    save_monitor_state,
    execute_headless_pipeline,
    run_monitor_loop,
    STATE_FILE
)


class TestTimeMonitor(unittest.TestCase):
    """Test suite for the ultra-lightweight time monitor daemon."""

    def setUp(self):
        self.test_state_file = BASE_DIR / "data_store" / "test_time_monitor_state.json"
        time_monitor.STATE_FILE = self.test_state_file
        if self.test_state_file.exists():
            self.test_state_file.unlink()

    def tearDown(self):
        if self.test_state_file.exists():
            self.test_state_file.unlink()
        time_monitor.STATE_FILE = STATE_FILE

    def test_01_time_trigger_before_16_returns_false(self):
        """Verifies trigger is False before 4:00 PM (16:00)."""
        dt_morning = datetime(2026, 11, 15, 10, 30, 0)
        dt_pre_close = datetime(2026, 11, 15, 15, 59, 59)

        self.assertFalse(should_trigger_pipeline(dt_morning, last_run_date=None))
        self.assertFalse(should_trigger_pipeline(dt_pre_close, last_run_date=None))

    def test_02_time_trigger_at_or_after_16_returns_true_if_not_run(self):
        """Verifies trigger is True at or after 16:00 if not run today."""
        dt_at_4pm = datetime(2026, 11, 15, 16, 0, 0)
        dt_evening = datetime(2026, 11, 15, 18, 30, 0)

        # Has not run today
        self.assertTrue(should_trigger_pipeline(dt_at_4pm, last_run_date="2026-11-14"))
        self.assertTrue(should_trigger_pipeline(dt_evening, last_run_date=None))

    def test_03_time_trigger_at_or_after_16_returns_false_if_already_run(self):
        """Verifies trigger is False if today has already run successfully."""
        dt_at_4pm = datetime(2026, 11, 15, 16, 0, 0)
        dt_evening = datetime(2026, 11, 15, 19, 0, 0)

        # Already ran today
        self.assertFalse(should_trigger_pipeline(dt_at_4pm, last_run_date="2026-11-15"))
        self.assertFalse(should_trigger_pipeline(dt_evening, last_run_date="2026-11-15"))

    def test_04_force_now_overrides_time_and_date(self):
        """Verifies force_now forces trigger regardless of time or date."""
        dt_morning = datetime(2026, 11, 15, 9, 0, 0)
        self.assertTrue(should_trigger_pipeline(dt_morning, last_run_date="2026-11-15", force_now=True))

    def test_05_state_persistence_load_and_save(self):
        """Verifies monitor state persistence and round-trip integrity."""
        # Initial state when file does not exist
        init_state = load_monitor_state()
        self.assertEqual(init_state.get("status"), "INITIALIZED")
        self.assertIsNone(init_state.get("last_run_date"))

        # Save updated state
        new_state = {
            "last_run_date": "2026-11-15",
            "last_run_time": "2026-11-15 16:00:02",
            "status": "SUCCESS",
            "last_exit_code": 0
        }
        save_monitor_state(new_state)

        # Load back
        loaded = load_monitor_state()
        self.assertEqual(loaded.get("last_run_date"), "2026-11-15")
        self.assertEqual(loaded.get("status"), "SUCCESS")
        self.assertEqual(loaded.get("last_exit_code"), 0)

    @patch("subprocess.run")
    def test_06_execute_headless_pipeline_spawns_process(self, mock_subprocess):
        """Verifies execute_headless_pipeline invokes subprocess with Python executable and script."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_subprocess.return_value = mock_proc

        exit_code = execute_headless_pipeline(force=False)
        self.assertEqual(exit_code, 0)

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("run_headless_daily_pipeline.py", cmd[1])
        self.assertIn("--trigger", cmd)
        self.assertIn("TIME_MONITOR_SCHEDULE", cmd)
        self.assertEqual(kwargs.get("cwd"), str(BASE_DIR))

    @patch("time_monitor.execute_headless_pipeline", return_value=0)
    def test_07_run_monitor_loop_once_with_run_now(self, mock_execute):
        """Verifies run_monitor_loop executes pipeline and updates state when run_now=True and once=True."""
        today_str = datetime.now().strftime("%Y-%m-%d")

        run_monitor_loop(
            interval=0.01,
            target_hour=16,
            run_now=True,
            once=True,
            verbose=False
        )

        mock_execute.assert_called_once()
        state = load_monitor_state()
        self.assertEqual(state.get("last_run_date"), today_str)
        self.assertEqual(state.get("status"), "SUCCESS")

    def test_08_default_check_interval_is_120_seconds(self):
        """Verifies default interval is set to 120.0 seconds (2 minutes)."""
        self.assertEqual(time_monitor.DEFAULT_CHECK_INTERVAL, 120.0)


if __name__ == "__main__":
    unittest.main()
