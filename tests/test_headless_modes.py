"""
VectorTrade - Unit & Integration Tests for Offline and Online Headless Modes
Verifies headless diagnostics, JSON and text output generation, and CLI options.
"""

import unittest
from unittest.mock import patch, MagicMock
from offline.app import run_offline_headless
from online.app import run_online_headless


class TestHeadlessModes(unittest.TestCase):
    """Tests for headless verification and diagnostic routines."""

    def test_01_offline_headless_json_mode(self):
        """Verifies run_offline_headless returns structured dictionary in JSON mode."""
        res = run_offline_headless(output_format="json")
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("engine"), "VectorTrade Offline Engine")
        self.assertIn("components", res)
        self.assertIn("data_store", res["components"])
        self.assertIn("models", res["components"])
        self.assertIn("ma50_strategy", res["components"])
        self.assertIn("market_schedule", res["components"])
        self.assertIn("mongo_syncer", res["components"])
        self.assertTrue(res.get("success"))

    def test_02_offline_headless_text_mode(self):
        """Verifies run_offline_headless succeeds in text mode."""
        res = run_offline_headless(output_format="text")
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))

    def test_03_online_headless_json_mode(self):
        """Verifies run_online_headless returns structured dictionary in JSON mode."""
        res = run_online_headless(output_format="json")
        self.assertIsInstance(res, dict)
        self.assertIn("components", res)
        self.assertIn("database", res["components"])
        self.assertIn("orders_service", res["components"])
        self.assertIn("megabull_broker", res["components"])
        self.assertIn("routes", res)
        self.assertTrue(len(res["routes"]) >= 15)
        # Verify all routes passed
        all_passed = all(r.get("passed") for r in res["routes"])
        self.assertTrue(all_passed)
        self.assertTrue(res.get("success"))

    def test_04_online_headless_text_mode(self):
        """Verifies run_online_headless succeeds in text mode."""
        res = run_online_headless(output_format="text")
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))


if __name__ == "__main__":
    unittest.main()
