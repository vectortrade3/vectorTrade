"""
Unit & Integration Test Suite for UI Dashboard Integration & Daily Automation (Phase 4)
Tests: tests/test_pair_phase4_integration.py

Verifies:
1. Daily PairInferenceService integration in headless daily pipeline runner.
2. /strategies page rendering with Model Diagnostics subtab and KPI cards.
3. /api/pairs/model-metrics endpoint returning trained model parameters and feature importances.
4. /api/pairs/trades endpoint supporting temporal filtering (ALL, TRAIN, TEST) and pagination limits.
5. End-to-end integration across both offline Flask and online FastAPI applications.
"""

import os
import json
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from online.app import app as online_app
from offline.app import app as offline_app
from online.services.pair_inference_service import PairInferenceService


class TestPairPhase4Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.online_client = TestClient(online_app)
        cls.offline_client = offline_app.test_client()

    def test_01_pair_inference_service_daily_scan(self):
        """Verifies daily scanning evaluates all 156 pairs and saves cache."""
        res = PairInferenceService.evaluate_daily_pairs()
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("evaluated_pairs_count"), 156)
        self.assertIn("qualified_signals", res)
        self.assertIn("decision_threshold", res)

    def test_02_offline_strategies_page_model_diagnostics(self):
        """Verifies offline /strategies renders model diagnostics and backtest KPIs."""
        res = self.offline_client.get("/strategies")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("subtab-pairs-model", html)
        self.assertIn("Model Diagnostics", html)
        self.assertIn("1-Year Holdout Test Comparison", html)
        self.assertIn("Profit Factor", html)
        self.assertIn("Top Predictive Features", html)

    def test_03_online_strategies_page_model_diagnostics(self):
        """Verifies online /strategies renders model diagnostics and backtest KPIs."""
        res = self.online_client.get("/strategies")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn("subtab-pairs-model", html)
        self.assertIn("Model Diagnostics", html)
        self.assertIn("1-Year Holdout Test Comparison", html)
        self.assertIn("Top Predictive Features", html)

    def test_04_offline_model_metrics_api(self):
        """Verifies GET /api/pairs/model-metrics returns model parameters and ranked features."""
        res = self.offline_client.get("/api/pairs/model-metrics")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload.get("status"), "SUCCESS")
        self.assertIn("model_metrics", payload)
        self.assertIn("analytics_report", payload)
        self.assertIn("metadata", payload)
        ranked = payload["model_metrics"].get("ranked_feature_importances", [])
        self.assertEqual(len(ranked), 23)

    def test_05_online_model_metrics_api(self):
        """Verifies GET /api/pairs/model-metrics on online app."""
        res = self.online_client.get("/api/pairs/model-metrics")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload.get("status"), "SUCCESS")
        self.assertIn("model_metrics", payload)
        self.assertIn("analytics_report", payload)

    def test_06_offline_trades_api_filtering(self):
        """Verifies GET /api/pairs/trades supports split filtering and limits."""
        # ALL
        res_all = self.offline_client.get("/api/pairs/trades?limit=10")
        self.assertEqual(res_all.status_code, 200)
        data_all = res_all.get_json()
        self.assertEqual(data_all["returned_count"], 10)
        self.assertEqual(data_all["total_count"], 1676)

        # TRAIN
        res_train = self.offline_client.get("/api/pairs/trades?split=TRAIN&limit=5")
        self.assertEqual(res_train.status_code, 200)
        data_train = res_train.get_json()
        self.assertEqual(data_train["returned_count"], 5)
        self.assertTrue(all(t["split_type"] == "TRAIN" for t in data_train["trades"]))

        # TEST
        res_test = self.offline_client.get("/api/pairs/trades?split=TEST&limit=5")
        self.assertEqual(res_test.status_code, 200)
        data_test = res_test.get_json()
        self.assertEqual(data_test["returned_count"], 5)
        self.assertTrue(all(t["split_type"] == "TEST" for t in data_test["trades"]))

    def test_07_online_trades_api_filtering(self):
        """Verifies GET /api/pairs/trades on online app."""
        res = self.online_client.get("/api/pairs/trades?split=TEST&limit=15")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload.get("status"), "SUCCESS")
        self.assertEqual(payload["returned_count"], 15)
        self.assertTrue(all(t["split_type"] == "TEST" for t in payload["trades"]))


if __name__ == "__main__":
    unittest.main()
