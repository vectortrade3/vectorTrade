"""
Tests for VectorTrade JSON Manager & State Persistence
Verifies thread-safe atomic reading/writing, progress reporting, listener dispatch, and task logs.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager


class TestJsonManagerPipeline(unittest.TestCase):

    def setUp(self):
        self.dispatched_events = []
        def listener(event_type, data):
            self.dispatched_events.append((event_type, data))
        self.test_listener = listener
        json_manager.register_progress_listener(self.test_listener)

        # Snapshot progress files so unit tests never leave dirty state on disk
        self._dl_snapshot = json_manager.get_download_progress()
        self._fg_snapshot = json_manager.get_feature_gen_progress()
        self._dc_snapshot = json_manager.get_data_cleaning_progress()

    def tearDown(self):
        # Deregister listener
        if hasattr(self, 'test_listener') and self.test_listener in json_manager._progress_listeners:
            json_manager._progress_listeners.remove(self.test_listener)
        # Restore clean progress states
        if hasattr(self, '_dl_snapshot') and self._dl_snapshot:
            json_manager.write_json_file(json_manager.DOWNLOAD_PROGRESS_FILE, self._dl_snapshot)
        if hasattr(self, '_fg_snapshot') and self._fg_snapshot:
            json_manager.write_json_file(json_manager.FEATURE_GEN_PROGRESS_FILE, self._fg_snapshot)
        if hasattr(self, '_dc_snapshot') and self._dc_snapshot:
            json_manager.write_json_file(json_manager.DATA_CLEANING_PROGRESS_FILE, self._dc_snapshot)

    def test_stocks_download_state(self):
        info = json_manager.get_stocks_download_info()
        self.assertIn("last_downloaded_date", info)
        self.assertIn("status", info)
        self.assertIn("total_stocks", info)
        self.assertEqual(info["total_stocks"], 207)

    def test_feature_gen_state(self):
        info = json_manager.get_feature_gen_info()
        self.assertIn("last_feature_gen_date", info)
        self.assertIn("status", info)
        self.assertIn("total_stocks", info)

    def test_data_cleaning_state(self):
        info = json_manager.get_data_cleaning_info()
        self.assertIn("last_clean_date", info)
        self.assertIn("status", info)
        self.assertIn("total_rows", info)
        self.assertEqual(info["total_columns"], 157)
        self.assertEqual(info["total_stocks"], 207)

    def test_progress_dispatch_and_listener(self):
        self.dispatched_events.clear()
        
        # Test download progress dispatch
        json_manager.update_download_progress(
            status="CHECKING",
            heading="Checking",
            percentage=10.0,
            message="Test dispatch message"
        )
        self.assertTrue(any(e[0] == "DOWNLOAD_PROGRESS" for e in self.dispatched_events))

        # Test feature gen progress dispatch
        json_manager.update_feature_gen_progress(
            status="GENERATING",
            heading="Features Generating",
            percentage=25.0,
            message="Test feature progress"
        )
        self.assertTrue(any(e[0] == "FEATURE_GEN_PROGRESS" for e in self.dispatched_events))

        # Test data clean progress dispatch
        json_manager.update_data_cleaning_progress(
            status="COMBINING",
            heading="Combining Data",
            percentage=50.0,
            message="Test clean progress"
        )
        self.assertTrue(any(e[0] == "DATA_CLEAN_PROGRESS" for e in self.dispatched_events))

    def test_task_logs_logging(self):
        entry = json_manager.log_task_entry(
            task_name="UNIT_TEST_TASK",
            status="COMPLETED",
            message="Unit test executed successfully."
        )
        self.assertEqual(entry["task_name"], "UNIT_TEST_TASK")
        self.assertEqual(entry["status"], "COMPLETED")
        
        logs = json_manager.get_task_logs(limit=10)
        self.assertTrue(any(l["task_name"] == "UNIT_TEST_TASK" for l in logs))

    def test_app_logs_logging(self):
        entry = json_manager.log_app_event(
            event="TEST_EVENT",
            status="Everything fine",
            message="Event logged cleanly."
        )
        self.assertEqual(entry["event"], "TEST_EVENT")
        app_logs = json_manager.get_app_logs(limit=10)
        self.assertTrue(any(l["event"] == "TEST_EVENT" for l in app_logs))


if __name__ == "__main__":
    unittest.main()
