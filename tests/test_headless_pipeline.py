"""
VectorTrade - Unit Tests for Headless Daily Pipeline Runner
Verifies:
1. Argument parsing (--force, --trigger).
2. Invocation of run_daily_analysis_pipeline with correct parameters.
3. Proper exit code 0 upon SUCCESS status.
4. Proper exit code 1 upon ERROR status or raised exception.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import run_headless_daily_pipeline


class TestHeadlessPipeline(unittest.TestCase):
    """Test suite for the standalone headless daily pipeline runner."""

    @patch("run_headless_daily_pipeline.get_market_schedule_info")
    @patch("run_headless_daily_pipeline.run_daily_analysis_pipeline")
    def test_01_main_success_returns_zero(self, mock_run_pipeline, mock_sched):
        """Verifies successful pipeline execution returns exit code 0."""
        mock_sched.return_value = {
            "market_status": "Market Closed",
            "target_date": "2026-11-15"
        }
        mock_run_pipeline.return_value = {
            "status": "SUCCESS",
            "finalized_count": 13,
            "mongo_synced": True,
            "message": "Analyzed 207 stocks."
        }

        with patch("sys.argv", ["run_headless_daily_pipeline.py", "--trigger", "TEST_TRIGGER"]):
            ret = run_headless_daily_pipeline.main()
            self.assertEqual(ret, 0)
            mock_run_pipeline.assert_called_once_with(
                force_download=False,
                execution_trigger="TEST_TRIGGER"
            )

    @patch("run_headless_daily_pipeline.get_market_schedule_info")
    @patch("run_headless_daily_pipeline.run_daily_analysis_pipeline")
    def test_02_main_force_flag_passed(self, mock_run_pipeline, mock_sched):
        """Verifies --force flag passes force_download=True to pipeline."""
        mock_sched.return_value = {"market_status": "OK", "target_date": "2026-11-15"}
        mock_run_pipeline.return_value = {"status": "SUCCESS"}

        with patch("sys.argv", ["run_headless_daily_pipeline.py", "--force"]):
            ret = run_headless_daily_pipeline.main()
            self.assertEqual(ret, 0)
            mock_run_pipeline.assert_called_once_with(
                force_download=True,
                execution_trigger="HEADLESS_DAILY_PIPELINE"
            )

    @patch("run_headless_daily_pipeline.get_market_schedule_info")
    @patch("run_headless_daily_pipeline.run_daily_analysis_pipeline")
    def test_03_main_error_status_returns_one(self, mock_run_pipeline, mock_sched):
        """Verifies pipeline returning non-SUCCESS status exits with code 1."""
        mock_sched.return_value = {"market_status": "OK", "target_date": "2026-11-15"}
        mock_run_pipeline.return_value = {
            "status": "ERROR",
            "message": "Data download failed"
        }

        with patch("sys.argv", ["run_headless_daily_pipeline.py"]):
            ret = run_headless_daily_pipeline.main()
            self.assertEqual(ret, 1)

    @patch("run_headless_daily_pipeline.get_market_schedule_info")
    @patch("run_headless_daily_pipeline.run_daily_analysis_pipeline")
    @patch("run_headless_daily_pipeline.json_manager.log_app_event")
    def test_04_main_exception_returns_one(self, mock_log_event, mock_run_pipeline, mock_sched):
        """Verifies uncaught exception logs event and returns code 1."""
        mock_sched.return_value = {"market_status": "OK", "target_date": "2026-11-15"}
        mock_run_pipeline.side_effect = RuntimeError("Network timeout")

        with patch("sys.argv", ["run_headless_daily_pipeline.py"]):
            ret = run_headless_daily_pipeline.main()
            self.assertEqual(ret, 1)
            mock_log_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()
