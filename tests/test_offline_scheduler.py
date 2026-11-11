"""
VectorTrade - Unit & Integration Tests for Offline Daily 4:00 PM Scheduler & Self-Closer
Verifies:
1. Time window evaluation (00:00 - 15:59 Idle vs 16:00 - 23:59 Auto-run in IST).
2. Internet connectivity detection and graceful handling when offline.
3. Idle window behavior (sits idle, does not analyze, does not self-close).
4. Auto-run window behavior when already completed today (triggers self-close).
5. Auto-run window behavior when offline (waits for internet, does not crash).
6. Auto-run window execution with internet (runs pipeline, uploads to Mongo, self-closes).
7. REST APIs (/api/scheduler/status, /api/scheduler/tick).
8. JSON state persistence of mongo_synced, timestamps, and strategies.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.scheduler import (
    OfflineDailyScheduler,
    check_internet_connection,
    trigger_self_close,
    get_daily_scheduler
)
from offline.app import app
from strategies.manager import StrategyManager

KOLKATA_TZ = pytz.timezone("Asia/Kolkata")


class TestOfflineDailyScheduler(unittest.TestCase):
    """Test suite for the 4:00 PM Automated Scheduler and Self-Closer."""

    def setUp(self):
        # ALWAYS disable actual process exit during tests
        trigger_self_close.disabled = True
        self.scheduler = OfflineDailyScheduler(check_interval_seconds=60)
        self.client = app.test_client()
        # Backup daily_analysis.json state
        self._daily_analysis_backup = json_manager.get_daily_analysis_info()

    def tearDown(self):
        self.scheduler.stop()
        # Restore daily_analysis.json state
        if hasattr(self, "_daily_analysis_backup") and self._daily_analysis_backup:
            json_manager.write_json_file(json_manager.DAILY_ANALYSIS_FILE, self._daily_analysis_backup)

    def test_01_time_window_evaluation_idle_vs_autorun(self):
        """Verifies correct classification of 00:00-15:59 (Idle) and 16:00-23:59 (Auto-run) in IST."""
        # Case A: Morning 10:30 AM IST
        mock_morning = datetime(2026, 11, 11, 10, 30, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_morning
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            win = self.scheduler.get_time_window_info()
            self.assertEqual(win["window"], "IDLE_WAITING_FOR_4PM")
            self.assertFalse(win["is_active_window"])
            self.assertIn("Idle Window", win["window_label"])
            self.assertIn("5h 30m", win["time_until_4pm"])

        # Case B: Afternoon 3:59 PM IST (Still idle)
        mock_359pm = datetime(2026, 11, 11, 15, 59, 30, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_359pm
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            win = self.scheduler.get_time_window_info()
            self.assertEqual(win["window"], "IDLE_WAITING_FOR_4PM")
            self.assertFalse(win["is_active_window"])

        # Case C: 4:00 PM IST (Auto-run window starts)
        mock_400pm = datetime(2026, 11, 11, 16, 0, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_400pm
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            win = self.scheduler.get_time_window_info()
            self.assertEqual(win["window"], "AUTORUN_WINDOW")
            self.assertTrue(win["is_active_window"])
            self.assertIn("Auto-Run Window", win["window_label"])

        # Case D: Evening 8:15 PM IST (Auto-run window active)
        mock_815pm = datetime(2026, 11, 11, 20, 15, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_815pm
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            win = self.scheduler.get_time_window_info()
            self.assertEqual(win["window"], "AUTORUN_WINDOW")
            self.assertTrue(win["is_active_window"])

    def test_02_internet_connectivity_check(self):
        """Verifies internet check returns boolean and handles connection failures gracefully."""
        # Simulated offline
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.connect.side_effect = OSError("Network unreachable")
            with patch("urllib.request.urlopen", side_effect=Exception("No route to host")):
                self.assertFalse(check_internet_connection(timeout=0.1))

        # Simulated online
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.connect.return_value = None
            self.assertTrue(check_internet_connection(timeout=0.1))

    def test_03_idle_window_tick_does_not_execute_or_close(self):
        """During idle hours (00:00 - 15:59 IST), scheduler sits idle and does not self-close."""
        mock_morning = datetime(2026, 11, 11, 11, 0, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_morning
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            res = self.scheduler.run_tick_once()

            self.assertEqual(res["action"], "IDLE_WAITING")
            self.assertEqual(res["window"], "IDLE_WAITING_FOR_4PM")
            self.assertFalse(res["self_close_scheduled"])
            self.assertIn("Idle mode", res["message"])

    def test_04_autorun_window_already_completed_triggers_self_close(self):
        """If already completed today in 4PM window, scheduler detects it and triggers self-close."""
        mock_evening = datetime(2026, 11, 11, 17, 30, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_evening
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Mock daily_analysis check to return already completed today
            with patch.object(self.scheduler, "check_today_completed") as mock_check:
                mock_check.return_value = {
                    "is_done": True,
                    "today_date": "2026-11-11",
                    "status": "COMPLETED",
                    "mongo_synced": True
                }

                res = self.scheduler.run_tick_once()
                self.assertEqual(res["action"], "SELF_CLOSE_TRIGGERED")
                self.assertTrue(res["self_close_scheduled"])
                self.assertIn("already COMPLETED", res["message"])

    def test_05_autorun_window_no_internet_waits_without_crashing(self):
        """In 4PM window with no internet, scheduler reports WAITING_FOR_INTERNET and does not crash."""
        mock_4pm = datetime(2026, 11, 11, 16, 5, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_4pm
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(self.scheduler, "check_today_completed") as mock_check:
                mock_check.return_value = {"is_done": False}

                with patch("offline.scheduler.check_internet_connection", return_value=False):
                    res = self.scheduler.run_tick_once()
                    self.assertEqual(res["action"], "WAITING_FOR_INTERNET")
                    self.assertFalse(res["internet_connected"])
                    self.assertFalse(res["self_close_scheduled"])
                    self.assertIn("NO INTERNET", res["message"])

    def test_06_autorun_window_with_internet_executes_pipeline_and_self_closes(self):
        """In 4PM window with internet and not completed today: runs pipeline and triggers self-close."""
        mock_4pm = datetime(2026, 11, 11, 16, 15, 0, tzinfo=KOLKATA_TZ)
        with patch("offline.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_4pm
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(self.scheduler, "check_today_completed") as mock_check:
                mock_check.return_value = {"is_done": False}

                with patch("offline.scheduler.check_internet_connection", return_value=True):
                    with patch("offline.daily_analyzer.run_daily_analysis_pipeline") as mock_pipe:
                        mock_pipe.return_value = {
                            "status": "SUCCESS",
                            "finalized_count": 12,
                            "mongo_sync": {"status": "SUCCESS", "synced_count": 12}
                        }

                        res = self.scheduler.run_tick_once()
                        self.assertEqual(res["action"], "PIPELINE_COMPLETED_SELF_CLOSING")
                        self.assertTrue(res["self_close_scheduled"])
                        self.assertEqual(res["finalized_count"], 12)
                        self.assertEqual(res["mongo_status"], "SUCCESS")
                        mock_pipe.assert_called_once_with(
                            force_download=False,
                            execution_trigger="AUTO_4PM_TIMER"
                        )

    def test_07_api_scheduler_status_and_tick_endpoints(self):
        """Verifies GET /api/scheduler/status and POST /api/scheduler/tick endpoints."""
        # 1. GET /api/scheduler/status
        res = self.client.get("/api/scheduler/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        s = data["scheduler"]
        self.assertIn("window", s)
        self.assertIn("current_time_ist", s)
        self.assertIn("time_until_4pm", s)
        self.assertIn("enabled_strategies", s)

        # 2. POST /api/scheduler/tick (mock pipeline execution if tick runs)
        with patch("offline.daily_analyzer.run_daily_analysis_pipeline") as mock_pipe:
            mock_pipe.return_value = {
                "status": "SUCCESS",
                "finalized_count": 5,
                "mongo_sync": {"status": "SUCCESS"}
            }
            res_tick = self.client.post("/api/scheduler/tick", json={"force": False})
            self.assertEqual(res_tick.status_code, 200)
            data_tick = res_tick.get_json()
            self.assertEqual(data_tick["status"], "SUCCESS")
            self.assertIn("action", data_tick["result"])

    def test_08_json_manager_daily_analysis_sync_tracking(self):
        """Verifies json_manager atomically tracks mongo_synced, timestamps, and strategies."""
        # Update daily analysis with mongo sync data
        info = json_manager.update_daily_analysis_info(
            status="COMPLETED",
            market_session_date="2026-11-11",
            market_status="Market Closed",
            total_stocks_analyzed=207,
            finalized_trades=[{"symbol": "INFY", "action": "BUY"}],
            strategies_breakdown={"50 MA Setup": 1},
            strategies_run=["ma50_setup"],
            mongo_synced=True,
            mongo_synced_at="2026-11-11T16:10:00+05:30",
            mongo_sync_details={"status": "SUCCESS", "synced_count": 1},
            execution_trigger="AUTO_4PM_TIMER",
            message="Test auto run completed."
        )

        self.assertEqual(info["status"], "COMPLETED")
        self.assertTrue(info["mongo_synced"])
        self.assertEqual(info["mongo_synced_at"], "2026-11-11T16:10:00+05:30")
        self.assertEqual(info["strategies_run"], ["ma50_setup"])
        self.assertEqual(info["execution_trigger"], "AUTO_4PM_TIMER")

        # Test is_daily_analysis_and_sync_done_for_today
        check_res = json_manager.is_daily_analysis_and_sync_done_for_today("2026-11-11")
        self.assertTrue(check_res["is_done"])
        self.assertTrue(check_res["mongo_synced"])

        # Check different date returns False
        check_diff = json_manager.is_daily_analysis_and_sync_done_for_today("2026-12-25")
        self.assertFalse(check_diff["is_done"])


if __name__ == "__main__":
    unittest.main()
