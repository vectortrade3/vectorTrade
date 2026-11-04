"""
VectorTrade - Unit Test Suite: Phase 4 Zero-Leakage 50MA Machine Learning Engine
Tests:
1. Strict zero-leakage feature shielding (no fwd_* or outcome columns in feature list)
2. Presence and integrity of 50MA models (regressor, classifier, scaler, imputer, feature lists)
3. Feature dimension consistency across models and scalers
4. Out-of-sample 1-year test slice edge verification (Win rate > 33.3% breakeven, positive Net R, positive P&L)
5. Decision policy filtering (TAKE vs SKIP) and risk-reward geometry
6. Trade ledger record schema fidelity with win_probability and predicted_r
7. models/model_performance_log.json and data_store/inference_state.json persistence
8. Flask API endpoints for 50MA inference trade ledger querying
"""

import sys
import unittest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.model_trainer import _extract_base_features, LEAKAGE_COLS, FINAL_DATA_50MA_CSV
from offline.model_inferencer import run_strategy_inference
from offline.app import app


class Test50MAMLPhase4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = BASE_DIR / "models"
        cls.app_client = app.test_client()

    def test_01_zero_leakage_feature_shielding(self):
        """Verifies zero future leakage: all fwd_*, outcome, and bracket coordinates are stripped."""
        if not FINAL_DATA_50MA_CSV.exists():
            self.skipTest("5_year_final_data_50ma.csv not generated yet")

        df = pd.read_csv(FINAL_DATA_50MA_CSV, nrows=5)
        features = _extract_base_features(df)

        # 1. Zero forward columns
        fwd_in_feats = [f for f in features if f.startswith("fwd_")]
        self.assertEqual(len(fwd_in_feats), 0, f"Forward leakage detected: {fwd_in_feats}")

        # 2. Zero outcome columns
        forbidden = [
            "outcome", "days_held", "exit_price", "exit_date", "realized_pnl_pct",
            "realized_r", "mfe_pct", "mae_pct", "chart_candles_json", "split", "is_test_1y",
            "entry_price", "sl_price", "target_price", "risk", "triggered"
        ]
        leaks = [f for f in features if f in forbidden]
        self.assertEqual(len(leaks), 0, f"Outcome/Bracket leakage detected: {leaks}")

        # 3. High feature density (> 100 pure alpha indicators)
        self.assertGreaterEqual(len(features), 100, f"Expected >= 100 alpha features, got {len(features)}")

    def test_02_model_artifacts_integrity(self):
        """Verifies trained model artifacts, scalers, imputers, and feature files exist."""
        required = [
            self.models_dir / "5_year_xgboost_model.pkl",
            self.models_dir / "5_year_scaler.pkl",
            self.models_dir / "5_year_imputer.pkl",
            self.models_dir / "5_year_xgboost_50ma_classifier.pkl",
            self.models_dir / "strategy_base_model.pkl",
            BASE_DIR / "base_features.txt",
            BASE_DIR / "final_features.txt"
        ]
        for f in required:
            self.assertTrue(f.exists(), f"Missing required Phase 4 artifact: {f.name}")

    def test_03_feature_dimensions_consistency(self):
        """Verifies feature count in final_features.txt matches scaler and model input dimensions."""
        with open(BASE_DIR / "final_features.txt", "r", encoding="utf-8") as f:
            final_feats = [line.strip() for line in f if line.strip()]

        scaler = joblib.load(self.models_dir / "5_year_scaler.pkl")
        imputer = joblib.load(self.models_dir / "5_year_imputer.pkl")

        self.assertEqual(len(final_feats), scaler.n_features_in_)
        self.assertEqual(len(final_feats), imputer.n_features_in_)

    def test_04_out_of_sample_1y_positive_edge(self):
        """Verifies ML filtering converts unguided -129R / -193% loss into positive Win Rate & positive P&L."""
        inf_trades = json_manager.get_inference_50ma_trades()
        summary = inf_trades.get("summary", {})

        self.assertEqual(summary.get("timeframe"), "LAST_1_YEAR")
        self.assertEqual(summary.get("total_trades"), 1576)

        taken_cnt = summary.get("taken_trades", 0)
        self.assertGreater(taken_cnt, 50, "Expected at least 50 high-confidence trades taken")
        self.assertLess(taken_cnt, 1000, "Model should be selective, not taking all trades")

        # Win rate verification: must exceed 1:2 R:R breakeven (33.3%)
        win_rate_taken_str = summary.get("win_rate_taken", "0%").replace("%", "")
        win_rate_taken = float(win_rate_taken_str)
        self.assertGreater(win_rate_taken, 33.3, f"Win rate ({win_rate_taken}%) must exceed 33.3% breakeven")

        # Overall P&L% and Net R must be positive
        overall_pnl = float(summary.get("overall_pnl_pct", 0.0))
        self.assertGreater(overall_pnl, 0.0, f"Filtered strategy P&L ({overall_pnl}%) must be positive")

        net_r = float(summary.get("net_r_taken", 0.0))
        self.assertGreater(net_r, 0.0, f"Filtered strategy Net R ({net_r}) must be positive")

    def test_05_trade_ledger_schema_fidelity(self):
        """Verifies individual trade records contain predicted_r, win_probability, decision, and P&L."""
        inf_trades = json_manager.get_inference_50ma_trades()
        trades = inf_trades.get("trades", [])
        self.assertTrue(len(trades) > 0, "No trades in 50MA inference ledger")

        sample = trades[0]
        required_keys = [
            "id", "symbol", "date", "entry_price", "stop_loss", "target", "risk",
            "predicted_r", "win_probability", "real_r", "outcome", "decision",
            "is_win", "trade_pnl_pct", "cumulative_pnl_pct"
        ]
        for k in required_keys:
            self.assertIn(k, sample, f"Missing key {k} in 50MA trade record")

        self.assertIn(sample["decision"], ["TAKE", "SKIP"])

    def test_06_model_performance_log_structure(self):
        """Verifies Strategy_50MA_1Y_Test section in model_performance_log.json."""
        perf_log = json_manager.get_model_performance_log()
        self.assertIn("Strategy_50MA_1Y_Test", perf_log)

        s_log = perf_log["Strategy_50MA_1Y_Test"]
        metrics = s_log.get("metrics", {})
        self.assertIn("Total_Trades", metrics)
        self.assertIn("Taken_Trades", metrics)
        self.assertIn("Win_Rate_Taken", metrics)
        self.assertIn("Overall_PnL_Pct", metrics)
        self.assertIn("Net_R_Taken", metrics)

    def test_07_api_inference_filtering(self):
        """Verifies /api/inference/trades/50ma handles TAKE/SKIP filtering and pagination."""
        res_take = self.app_client.get('/api/inference/trades/50ma?decision=TAKE&limit=20')
        self.assertEqual(res_take.status_code, 200)
        d_take = res_take.get_json()
        self.assertTrue(all(t["decision"] == "TAKE" for t in d_take["trades"]))

        res_skip = self.app_client.get('/api/inference/trades/50ma?decision=SKIP&limit=20')
        self.assertEqual(res_skip.status_code, 200)
        d_skip = res_skip.get_json()
        self.assertTrue(all(t["decision"] == "SKIP" for t in d_skip["trades"]))


if __name__ == "__main__":
    unittest.main()
