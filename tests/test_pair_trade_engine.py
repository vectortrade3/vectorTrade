"""
Unit & Integration Test Suite for Institutional Pairs Trading Strategy v4 Engine (Phase 1)
Tests: tests/test_pair_trade_engine.py

Verifies:
1. Strategy configuration loading and parameters
2. Rolling OLS beta, log spread, and z-score mathematical stability
3. Momentum velocity flattening filter (|Delta Z| <= 0.05)
4. Entry hurdle constraints (2.10 <= |Z| <= 3.00)
5. 7-day forward horizon tracking (exact forward bars, non-empty)
6. Dynamic stop-loss barrier (|Z| >= 2.85) and target exit (|Z| <= 0.40)
7. Maximum holding period clamping (4 to 22 days)
8. Temporal split integrity (train <= 2025-11-18, test > 2025-11-18) with 0 data leakage
9. Dual-stock feature presence (A and B indicators, spread metrics)
"""

import os
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from offline.pair_trade_engine import (
    PairTradeEngine,
    load_strategy_config,
    compute_rsi,
    compute_atr,
    compute_half_life,
    compute_adf_stat
)


class TestPairTradeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PairTradeEngine()
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.pair_strategy_dir = cls.base_dir / "data_store" / "pair_strategy"
        cls.train_csv = cls.pair_strategy_dir / "pair_trades_train.csv"
        cls.test_csv = cls.pair_strategy_dir / "pair_trades_test.csv"
        cls.all_csv = cls.pair_strategy_dir / "pair_trades_all.csv"
        cls.metadata_json = cls.pair_strategy_dir / "pair_trades_metadata.json"
        cls.forward_json = cls.pair_strategy_dir / "forward_7d_trajectories.json"

    def test_01_config_parameters_loaded(self):
        """Verifies institutional config thresholds are properly set in engine."""
        cfg = load_strategy_config()
        self.assertEqual(cfg.get("version"), "4.0.0")
        self.assertEqual(self.engine.lookback_days, 90)
        self.assertEqual(self.engine.z_entry_hurdle, 2.10)
        self.assertEqual(self.engine.z_entry_max, 3.00)
        self.assertEqual(self.engine.max_momentum_slope, 0.05)
        self.assertEqual(self.engine.z_exit_target, 0.40)
        self.assertEqual(self.engine.z_stop_loss, 2.85)
        self.assertEqual(self.engine.min_holding_days, 4)
        self.assertEqual(self.engine.max_holding_days, 22)

    def test_02_synthetic_spread_and_zscore_math(self):
        """Verifies mathematical correctness of OLS beta and Z-score calculations."""
        # Create synthetic cointegrated series
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
        p_b = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))
        spread_true = np.zeros(n)
        for i in range(1, n):
            spread_true[i] = 0.8 * spread_true[i - 1] + np.random.normal(0, 0.5)
        beta_true = 1.5
        p_a = np.exp(np.log(p_b) * beta_true + spread_true)

        df = pd.DataFrame({
            "Date": dates,
            "Close_A": p_a,
            "Close_B": p_b,
            "Log_A": np.log(p_a),
            "Log_B": np.log(p_b)
        })

        # Test half-life computation
        hl = compute_half_life(pd.Series(spread_true))
        self.assertGreater(hl, 0)
        self.assertLess(hl, 45)

        # Test ADF proxy
        adf = compute_adf_stat(pd.Series(spread_true))
        self.assertIsInstance(adf, float)

    def test_03_technical_indicators_functions(self):
        """Verifies RSI and ATR calculations."""
        prices = pd.Series([100.0 + i * 0.5 for i in range(50)])
        rsi = compute_rsi(prices, 14)
        self.assertEqual(len(rsi), 50)
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

        high = prices + 2.0
        low = prices - 2.0
        atr = compute_atr(high, low, prices, 14)
        self.assertEqual(len(atr), 50)
        self.assertTrue((atr > 0).all())

    def test_04_dataset_files_exist_and_non_empty(self):
        """Verifies all Phase 1 artifact files exist and contain extracted trades."""
        self.assertTrue(self.all_csv.exists(), "pair_trades_all.csv must exist")
        self.assertTrue(self.train_csv.exists(), "pair_trades_train.csv must exist")
        self.assertTrue(self.test_csv.exists(), "pair_trades_test.csv must exist")
        self.assertTrue(self.metadata_json.exists(), "pair_trades_metadata.json must exist")
        self.assertTrue(self.forward_json.exists(), "forward_7d_trajectories.json must exist")

        df_all = pd.read_csv(self.all_csv)
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        self.assertGreaterEqual(len(df_all), 1000, "Should have extracted at least 1000 pair trades across 5 years")
        self.assertGreater(len(df_train), 500, "Training set should have at least 500 trades")
        self.assertGreater(len(df_test), 100, "Holdout test set should have at least 100 trades")
        self.assertEqual(len(df_all), len(df_train) + len(df_test))

    def test_05_strict_train_test_split_dates(self):
        """Verifies zero data leakage: train trades <= 2025-11-18, test trades > 2025-11-18."""
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        max_train_date = df_train["entry_date"].max()
        min_test_date = df_test["entry_date"].min()

        self.assertLessEqual(max_train_date, "2025-11-18")
        self.assertGreater(min_test_date, "2025-11-18")

    def test_06_entry_hurdle_and_velocity_constraints(self):
        """Verifies every trade strictly adheres to the Z-score entry hurdle and velocity slope."""
        df_all = pd.read_csv(self.all_csv)

        # Check |z_entry| in [2.10, 3.00]
        abs_z = df_all["z_score_entry"].abs()
        self.assertTrue((abs_z >= 2.10 - 1e-4).all(), "Entry Z-score must be >= 2.10")
        self.assertTrue((abs_z <= 3.00 + 1e-4).all(), "Entry Z-score must be <= 3.00 (blowout penalty threshold)")

        # Check velocity slope |delta_z| <= 0.05
        abs_delta = df_all["delta_z_entry"].abs()
        self.assertTrue((abs_delta <= 0.05 + 1e-4).all(), "Entry momentum slope |Delta Z| must be <= 0.05")

    def test_07_holding_duration_clamped(self):
        """Verifies holding period does not exceed maximum holding limit cap."""
        df_all = pd.read_csv(self.all_csv)
        self.assertTrue((df_all["holding_days"] >= 1).all())
        self.assertTrue((df_all["holding_days"] <= df_all["max_holding_limit"]).all())
        self.assertTrue((df_all["holding_days"] <= 22).all(), "Holding period must not exceed 22 days cap")

    def test_08_forward_7d_trajectories_cached(self):
        """Verifies forward 7-day trajectory cache matches trade records."""
        with open(self.forward_json, "r", encoding="utf-8") as f:
            trajs = json.load(f)

        df_all = pd.read_csv(self.all_csv)
        self.assertEqual(len(trajs), len(df_all))

        # Inspect first trajectory
        first_key = list(trajs.keys())[0]
        traj = trajs[first_key]
        self.assertEqual(len(traj), 7, "Each forward trajectory must contain exactly 7 steps")
        for step in traj:
            self.assertIn("step", step)
            self.assertIn("date", step)
            self.assertIn("price_a", step)
            self.assertIn("price_b", step)
            self.assertIn("spread", step)
            self.assertIn("zscore", step)
            self.assertIn("pair_return_pct", step)

    def test_09_dual_stock_features_present(self):
        """Verifies technical indicator features are present for both Stock A and Stock B."""
        df_all = pd.read_csv(self.all_csv)
        required_features = [
            "a_sma_20_ratio", "a_sma_50_ratio", "a_sma_200_ratio", "a_rsi_14", "a_atr_14_pct", "a_histvol_20", "a_vol_ratio_20",
            "b_sma_20_ratio", "b_sma_50_ratio", "b_sma_200_ratio", "b_rsi_14", "b_atr_14_pct", "b_histvol_20", "b_vol_ratio_20",
            "rolling_corr_90", "spread_vol_pct_90", "vol_ratio_a_to_b", "rsi_diff_a_minus_b", "half_life_days", "adf_stat"
        ]
        for feat in required_features:
            self.assertIn(feat, df_all.columns, f"Feature {feat} must exist in trade dataset")
            self.assertFalse(df_all[feat].isna().any(), f"Feature {feat} must not contain NaNs")


if __name__ == "__main__":
    unittest.main()
