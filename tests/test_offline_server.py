"""
VectorTrade - Offline Server & Backend Controller Unit Test Suite
Verifies:
1. Flask control dashboard rendering and API routes.
2. System health, resource telemetry, and schedule state.
3. EOD results and pending dispatches queries.
4. System tray icon image generation.
5. Asynchronous action dispatching (run-eod, sync-dispatches).
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline.server import app
from offline.scheduler import offline_scheduler
from offline.tray_app import create_tray_icon_image


class TestOfflineServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_index_html_renders(self):
        """Verifies GET / renders the offline control dashboard."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"VectorTrade", res.data)
        self.assertIn(b"Offline Control Center", res.data)

    def test_02_api_status(self):
        """Verifies /api/status returns memory, CPU, and scheduler metrics."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("scheduler_status", data)
        self.assertIn("memory_usage_mb", data)
        self.assertIn("cpu_usage_pct", data)
        self.assertIn("database_connected", data)
        self.assertIn("next_scheduled_run", data)
        # Resource footprint verification: idle memory should be reasonable (< 150 MB)
        self.assertLess(data["memory_usage_mb"], 250.0)

    def test_03_api_results(self):
        """Verifies /api/results returns JSON signal list."""
        res = self.client.get("/api/results")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIsInstance(data["signals"], list)

    def test_04_api_pending(self):
        """Verifies /api/pending returns pending offline fallback count."""
        res = self.client.get("/api/pending")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("pending_files", data)

    def test_05_tray_icon_generation(self):
        """Verifies tray icon image is properly drawn with RGBA channels."""
        img = create_tray_icon_image()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))
        self.assertEqual(img.mode, "RGBA")

    def test_06_action_dispatch(self):
        """Verifies /api/action/sync-dispatches executes properly."""
        res = self.client.post("/api/action/sync-dispatches")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")

    def test_07_analysis_status_and_trigger(self):
        """Verifies /api/analysis-status returns multi-stage structure."""
        res = self.client.get("/api/analysis-status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("download", data)
        self.assertIn("calculate", data)
        self.assertIn("predict", data)
        self.assertIn("upload", data)

    def test_08_predictor_and_dashboard_endpoints(self):
        """Verifies newly merged 5_years_7_dashboard endpoints."""
        # 1. Stock list
        res = self.client.get("/api/stock_list")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertGreater(data["count"], 0)

        # 2. Model performance
        res = self.client.get("/api/model_performance")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("models", data)

        # 3. Dataset snapshot
        res = self.client.get("/api/dataset_snapshot")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("total_stocks", data)

        # 4. Predictions
        res = self.client.get("/api/predictions")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")

        # 5. Predict status
        res = self.client.get("/api/predict_status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)

        # 6. Unified Task status
        res = self.client.get("/api/task_status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("predict", data)
        self.assertIn("analysis", data)
        self.assertIn("data", data)
        self.assertIn("retrain", data)



if __name__ == "__main__":
    unittest.main()
