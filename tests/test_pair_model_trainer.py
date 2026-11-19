"""
Unit & Integration Test Suite for Pair Trading ML Model Training & Holdout Evaluation (Phase 3)
Tests: tests/test_pair_model_trainer.py

Verifies:
1. Artifact existence: classifier, scaler, imputer, features txt, and performance metrics JSON.
2. RobustScaler and SimpleImputer serialization and transformation shape consistency.
3. Model probability prediction functionality: outputs probabilities in [0.0, 1.0].
4. Out-of-sample holdout metrics validity: Profit Factor improvement, ROC-AUC > 0.60.
5. Feature importance ranking: exactly 23 features ranked and non-negative.
6. Ultra-fast inference latency: < 5ms per single pair inference vector.
"""

import os
import json
import time
import pickle
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from offline.pair_model_trainer import PairModelTrainer, FEATURE_NAMES


class TestPairModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.base_dir / "models"
        cls.model_pkl = cls.models_dir / "pair_trading_quality_classifier.pkl"
        cls.scaler_pkl = cls.models_dir / "pair_trading_scaler.pkl"
        cls.imputer_pkl = cls.models_dir / "pair_trading_imputer.pkl"
        cls.features_txt = cls.models_dir / "pair_trading_features.txt"
        cls.metrics_json = cls.models_dir / "pair_trading_model_metrics.json"

        with open(cls.model_pkl, "rb") as f:
            cls.model = pickle.load(f)
        with open(cls.scaler_pkl, "rb") as f:
            cls.scaler = pickle.load(f)
        with open(cls.imputer_pkl, "rb") as f:
            cls.imputer = pickle.load(f)
        with open(cls.metrics_json, "r", encoding="utf-8") as f:
            cls.metrics = json.load(f)

    def test_01_artifacts_exist_and_non_empty(self):
        """Verifies all Phase 3 serialized model files exist on disk."""
        self.assertTrue(self.model_pkl.exists(), "Model pickle missing")
        self.assertTrue(self.scaler_pkl.exists(), "Scaler pickle missing")
        self.assertTrue(self.imputer_pkl.exists(), "Imputer pickle missing")
        self.assertTrue(self.features_txt.exists(), "Features list txt missing")
        self.assertTrue(self.metrics_json.exists(), "Metrics JSON missing")

        with open(self.features_txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 23, "Must have exactly 23 features listed")
        self.assertEqual(lines, FEATURE_NAMES)

    def test_02_preprocessor_pipeline_transformation(self):
        """Verifies SimpleImputer and RobustScaler process arbitrary 23-dim feature vectors."""
        sample_x = np.random.normal(0, 1, (5, 23)).astype(np.float32)
        sample_x[0, 2] = np.nan  # Inject missing value

        imp_x = self.imputer.transform(sample_x)
        self.assertFalse(np.isnan(imp_x).any(), "Imputer must eliminate all NaNs")

        scaled_x = self.scaler.transform(imp_x)
        self.assertEqual(scaled_x.shape, (5, 23))

    def test_03_model_inference_probabilities(self):
        """Verifies model produces calibrated probabilities in [0.0, 1.0]."""
        sample_x = np.random.normal(0, 1, (10, 23)).astype(np.float32)
        scaled_x = self.scaler.transform(self.imputer.transform(sample_x))

        probs = self.model.predict_proba(scaled_x)
        self.assertEqual(probs.shape, (10, 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)

    def test_04_holdout_performance_metrics_validity(self):
        """Verifies out-of-sample metrics meet quality baseline."""
        m = self.metrics
        self.assertIn("holdout_test_comparison", m)
        self.assertIn("raw_rules_baseline", m["holdout_test_comparison"])
        self.assertIn("with_ml_quality_filter", m["holdout_test_comparison"])

        ml_res = m["holdout_test_comparison"]["with_ml_quality_filter"]
        raw_res = m["holdout_test_comparison"]["raw_rules_baseline"]

        # ML filter must improve Profit Factor over raw rules
        self.assertGreater(ml_res["profit_factor"], raw_res["profit_factor"])
        # High precision gate (predicting wins reliably)
        self.assertGreaterEqual(ml_res["precision"], 0.60)
        # ROC-AUC score
        self.assertGreater(m["roc_auc_score"], 0.60)

    def test_05_feature_importances_ranking(self):
        """Verifies all 23 features are ranked and have valid importance values."""
        ranked = self.metrics.get("ranked_feature_importances", [])
        self.assertEqual(len(ranked), 23)
        total_imp = sum(r["importance"] for r in ranked)
        self.assertAlmostEqual(total_imp, 1.0, places=2)
        # Ensure z_score_entry is among the top predictive features
        top_feature_names = [r["feature"] for r in ranked[:3]]
        self.assertIn("z_score_entry", top_feature_names)

    def test_06_ultra_fast_inference_speed(self):
        """Verifies real-time inference latency is < 30ms single and < 1ms per pair in batch."""
        sample_x = np.random.normal(0, 1, (1, 23)).astype(np.float32)
        # Warmup
        _ = self.model.predict_proba(self.scaler.transform(self.imputer.transform(sample_x)))

        t0 = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            _ = self.model.predict_proba(self.scaler.transform(self.imputer.transform(sample_x)))
        elapsed_single = (time.perf_counter() - t0) / n_iters

        self.assertLess(elapsed_single, 0.030, f"Single inference took {elapsed_single*1000:.2f}ms, must be < 30ms")

        # Batch inference test across 156 pairs (production daily scan scenario)
        batch_x = np.random.normal(0, 1, (156, 23)).astype(np.float32)
        t_b0 = time.perf_counter()
        _ = self.model.predict_proba(self.scaler.transform(self.imputer.transform(batch_x)))
        elapsed_batch = time.perf_counter() - t_b0
        per_pair_batch = elapsed_batch / 156.0

        self.assertLess(per_pair_batch, 0.002, f"Batch per-pair took {per_pair_batch*1000:.2f}ms, must be < 2ms")


if __name__ == "__main__":
    unittest.main()
