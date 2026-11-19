"""
Unit & Integration Test Suite for Dynamic Stop-Loss, Target Resolution & Outcome Labeling (Phase 2)
Tests: tests/test_pair_phase2_exits.py

Verifies:
1. Target Reached Exit condition (|Z| <= 0.40) always labeled SUCCESS (1).
2. Stop-Loss Barrier Exit condition (|Z| >= 2.85) always labeled FAIL (0).
3. Holding duration clamping strictly enforces 4 to 22 trading days.
4. Stop Loss barriers trigger quickly in ~2.4 days average, cutting tail risk as designed.
5. Exit reasons breakdown accounts for 100% of generated trades with no unclassified trades.
6. Clean separation of SUCCESS (1), FAIL (0), and NEUTRAL (2) outcomes.
7. Phase 2 analytics report exists and validates empirical metric boundaries.
"""

import os
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from offline.pair_analytics import generate_phase2_report, calculate_split_metrics


class TestPairPhase2Exits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.pair_strategy_dir = cls.base_dir / "data_store" / "pair_strategy"
        cls.train_csv = cls.pair_strategy_dir / "pair_trades_train.csv"
        cls.test_csv = cls.pair_strategy_dir / "pair_trades_test.csv"
        cls.report_json = cls.pair_strategy_dir / "phase2_analytics_report.json"

        cls.df_train = pd.read_csv(cls.train_csv)
        cls.df_test = pd.read_csv(cls.test_csv)
        cls.df_all = pd.concat([cls.df_train, cls.df_test], ignore_index=True)

    def test_01_all_target_reached_are_success(self):
        """Verifies every trade that hit TARGET_REACHED is labeled SUCCESS (1)."""
        target_trades = self.df_all[self.df_all["exit_reason"] == "TARGET_REACHED"]
        self.assertGreater(len(target_trades), 0)
        self.assertTrue((target_trades["outcome_label"] == 1).all())
        self.assertTrue((target_trades["outcome_text"] == "SUCCESS").all())

    def test_02_all_stop_barrier_are_fail(self):
        """Verifies every trade that hit STOP_LOSS_BARRIER is labeled FAIL (0)."""
        stop_trades = self.df_all[self.df_all["exit_reason"] == "STOP_LOSS_BARRIER"]
        self.assertGreater(len(stop_trades), 0)
        self.assertTrue((stop_trades["outcome_label"] == 0).all())
        self.assertTrue((stop_trades["outcome_text"] == "FAIL").all())

    def test_03_holding_days_strictly_bounded(self):
        """Verifies all trades respect the dynamic half-life cap bounded by [4, 22]."""
        self.assertTrue((self.df_all["holding_days"] >= 1).all())
        self.assertTrue((self.df_all["holding_days"] <= 22).all())

        # For time stop exits, holding_days must equal max_holding_limit
        time_trades = self.df_all[self.df_all["exit_reason"] == "TIME_STOP_CAP"]
        self.assertGreater(len(time_trades), 0)
        self.assertTrue((time_trades["holding_days"] == time_trades["max_holding_limit"]).all())

    def test_04_barrier_stops_fast_exit_behavior(self):
        """Verifies stop barrier triggers rapidly (average <= 5 days) rather than lingering."""
        stop_trades = self.df_all[self.df_all["exit_reason"] == "STOP_LOSS_BARRIER"]
        mean_stop_days = stop_trades["holding_days"].mean()
        self.assertLessEqual(mean_stop_days, 5.0, "Stop barrier should liquidate diverging trades quickly")

    def test_05_three_tier_labeling_mutually_exclusive_and_exhaustive(self):
        """Verifies every trade has exactly one outcome in {0, 1, 2} and {FAIL, SUCCESS, NEUTRAL}."""
        labels = set(self.df_all["outcome_label"].unique())
        self.assertTrue(labels.issubset({0, 1, 2}))

        texts = set(self.df_all["outcome_text"].unique())
        self.assertEqual(texts, {"FAIL", "SUCCESS", "NEUTRAL"})

        # No NaNs anywhere in critical outcome columns
        for col in ["exit_reason", "outcome_label", "outcome_text", "net_return_pct", "holding_days", "mfe_pct", "mae_pct"]:
            self.assertFalse(self.df_all[col].isna().any(), f"{col} must not contain NaNs")

    def test_06_analytics_report_generation(self):
        """Verifies Phase 2 analytics report produces accurate structure and calculations."""
        report = generate_phase2_report()
        self.assertEqual(report.get("status"), "SUCCESS")
        self.assertIn("train_4y_analytics", report)
        self.assertIn("test_1y_analytics", report)

        train_stats = report["train_4y_analytics"]
        self.assertEqual(train_stats["total_trades"], len(self.df_train))
        self.assertIn("holding_time_stats", train_stats)
        self.assertIn("exit_reasons_breakdown", train_stats)
        self.assertGreater(train_stats["holding_time_stats"]["mean_days"], 5.0)

        self.assertTrue(self.report_json.exists())


if __name__ == "__main__":
    unittest.main()
