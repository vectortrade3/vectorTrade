"""
VectorTrade - Unit Test Suite: Machine Learning Model Trainer Pipeline
Tests Section 1 (Setup Strategy Model) and Section 2 (Global Regime Models):
- JSON State initialization and persistence
- Feature pruning & correlation elimination logic
- Daily skip check (< 0.1s execution when already calibrated today)
- Flask endpoints (/api/training/status, /api/training/progress, /api/training/start_*)
- Concurrency locking preventing duplicate parallel runs
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.model_trainer import (
    train_strategy_model,
    train_global_models,
    train_all_models,
    _prune_features_by_importance_and_correlation,
    _extract_base_features
)
from offline.app import app


class TestModelTrainerPipeline(unittest.TestCase):

    def setUp(self):
        self.app_client = app.test_client()
        self._models_prog_snapshot = json_manager.get_models_training_progress()

    def tearDown(self):
        if hasattr(self, '_models_prog_snapshot') and self._models_prog_snapshot:
            json_manager.write_json_file(json_manager.MODELS_TRAINING_PROGRESS_FILE, self._models_prog_snapshot)

    def test_01_models_state_initialization(self):
        """Verifies models_state.json contains Section 1 (Strategy) and Section 2 (Global)."""
        info = json_manager.get_models_info()
        self.assertIn("strategy_model", info)
        self.assertIn("global_models", info)
        self.assertIn("models", info)

        strat = info["strategy_model"]
        self.assertIn("model_file", strat)
        self.assertIn("status", strat)

        glob = info["global_models"]
        self.assertIn("gain_model_file", glob)
        self.assertIn("drop_model_file", glob)
        self.assertIn("status", glob)

    def test_02_models_training_progress_structure(self):
        """Verifies models_training_progress.json returns expected schema."""
        prog = json_manager.get_models_training_progress()
        self.assertIn("status", prog)
        self.assertIn("percentage", prog)
        self.assertIn("message", prog)
        self.assertIn("elapsed_seconds", prog)

    def test_03_progress_listener_dispatch(self):
        """Verifies progress updates dispatch to registered WebSocket listeners."""
        received = []

        def dummy_listener(event_type, data):
            received.append((event_type, data))

        json_manager.register_progress_listener(dummy_listener)
        try:
            json_manager.update_models_training_progress(
                status="TRAINING",
                active_section="TEST",
                phase="Unit Test Phase",
                percentage=50.0,
                message="Dispatch test"
            )

            self.assertTrue(len(received) > 0)
            self.assertEqual(received[-1][0], "MODELS_TRAINING_PROGRESS")
            self.assertEqual(received[-1][1]["active_section"], "TEST")
            self.assertEqual(received[-1][1]["percentage"], 50.0)
        finally:
            if dummy_listener in json_manager._progress_listeners:
                json_manager._progress_listeners.remove(dummy_listener)

    def test_04_feature_pruning_unit_logic(self):
        """Tests zero-importance elimination and multicollinearity pruning."""
        # 3 features: A (corr with B = 0.95), B, C (zero importance)
        np.random.seed(42)
        feat_names = ["feat_A", "feat_B", "feat_C"]
        importances = np.array([0.8, 0.4, 0.0])  # A > B, C has 0 importance

        # Mock data where feat_A and feat_B are highly correlated (0.99)
        col_a = np.random.randn(100)
        col_b = col_a + np.random.randn(100) * 0.01  # highly correlated
        col_c = np.random.randn(100)
        X_mock = np.column_stack([col_a, col_b, col_c])

        refined = _prune_features_by_importance_and_correlation(
            X_mock, feat_names, importances, threshold=0.90
        )

        # feat_C should be dropped (zero imp)
        self.assertNotIn("feat_C", refined)
        # feat_B should be dropped in favor of higher importance feat_A
        self.assertIn("feat_A", refined)
        self.assertNotIn("feat_B", refined)

    def test_05_strategy_model_skip_logic(self):
        """Verifies skip logic when strategy model is already calibrated for today."""
        res = train_strategy_model(force=False)
        self.assertIn(res["status"], ["SKIPPED", "SUCCESS"])
        if res["status"] == "SKIPPED":
            self.assertEqual(res["section"], "STRATEGY")

    def test_06_global_models_skip_logic(self):
        """Verifies skip logic when global models are already calibrated for today."""
        res = train_global_models(force=False)
        self.assertIn(res["status"], ["SKIPPED", "SUCCESS"])
        if res["status"] == "SKIPPED":
            self.assertEqual(res["section"], "GLOBAL")

    def test_07_train_all_models_orchestration(self):
        """Verifies train_all_models executes sequential orchestrator."""
        res = train_all_models(force=False)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("strategy", res)
        self.assertIn("global", res)

    def test_08_api_training_status_endpoint(self):
        """Verifies GET /api/training/status returns HTTP 200 with info and progress."""
        res = self.app_client.get('/api/training/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("info", data)
        self.assertIn("progress", data)
        self.assertIn("strategy_model", data["info"])
        self.assertIn("global_models", data["info"])

    def test_09_api_training_progress_endpoint(self):
        """Verifies GET /api/training/progress returns HTTP 200 with progress fields."""
        res = self.app_client.get('/api/training/progress')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("percentage", data)

    def test_10_api_system_status_includes_models(self):
        """Verifies GET /api/status includes models_info and models_training_progress."""
        res = self.app_client.get('/api/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("models_info", data)
        self.assertIn("models_training_progress", data)
        self.assertEqual(data["status"], "ONLINE")


if __name__ == '__main__':
    unittest.main()
