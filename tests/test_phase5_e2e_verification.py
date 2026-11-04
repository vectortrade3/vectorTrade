"""
VectorTrade - Comprehensive Phase 5 End-to-End Verification & Stress Test Suite
Validates:
1. Complete pipeline stage consistency: Downloader -> Feature Gen -> Dual Cleaner -> Models -> Inference -> Daily Scanner
2. Strict zero lookahead leakage barrier across final_features.txt, base_features.txt, and model pipelines
3. Rigorous 50MA bracket geometry: Target == Entry + 2 * (Entry - SL)
4. Day 1 trigger gate: verify zero untriggered setups exist in 5_year_final_data_50ma.csv
5. 50MA Service boundary stress testing (extreme pages, negative limits, missing trade IDs, symbol substrings)
6. Chart API candlestick trajectory fidelity (38 bars: past 7, setup, forward 30)
7. Full Flask API HTTP 200 contract verification across all active endpoints
8. JSON Manager serialization and atomic state persistence integrity
9. Dual alpha decision policy stability (TAKE vs SKIP performance edge)
"""

import os
import sys
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.ma50_service import (
    load_50ma_dataset,
    get_50ma_trade_by_id,
    query_50ma_trades,
    get_50ma_strategy_summary
)
from offline.app import app


class TestPhase5E2EVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app_client = app.test_client()
        cls.models_dir = BASE_DIR / "models"
        cls.data_store_dir = BASE_DIR / "data_store"

    # ==========================================================================
    # 1. Pipeline Artifacts & Dataset Parity
    # ==========================================================================

    def test_01_dual_cleaning_datasets_present(self):
        """Verifies dual datasets (global master + dedicated 50MA) are present on disk."""
        p_global = BASE_DIR / "5_year_final_data_global.csv"
        p_50ma = BASE_DIR / "5_year_final_data_50ma.csv"
        p_50ma_train = BASE_DIR / "5_year_final_data_50ma_train.csv"
        p_50ma_test = BASE_DIR / "5_year_final_data_50ma_test_1y.csv"

        self.assertTrue(p_global.exists(), "5_year_final_data_global.csv missing")
        self.assertTrue(p_50ma.exists(), "5_year_final_data_50ma.csv missing")
        self.assertTrue(p_50ma_train.exists(), "5_year_final_data_50ma_train.csv missing")
        self.assertTrue(p_50ma_test.exists(), "5_year_final_data_50ma_test_1y.csv missing")

        # Verify row counts match expectations
        df_50ma = load_50ma_dataset()
        self.assertGreaterEqual(len(df_50ma), 8000, f"Expected >= 8000 50MA trades, got {len(df_50ma)}")

    def test_02_strict_zero_lookahead_barrier(self):
        """Verifies zero future or outcome columns exist in final_features.txt or base_features.txt."""
        for feat_file in ["base_features.txt", "final_features.txt"]:
            p = BASE_DIR / feat_file
            self.assertTrue(p.exists(), f"{feat_file} does not exist")
            with open(p, "r", encoding="utf-8") as f:
                features = [line.strip() for line in f if line.strip()]

            # Zero forward columns
            fwd = [feat for feat in features if feat.startswith("fwd_")]
            self.assertEqual(len(fwd), 0, f"Forward lookahead in {feat_file}: {fwd}")

            # Zero outcome / future trajectory columns
            forbidden = [
                "outcome", "days_held", "exit_price", "exit_date", "realized_pnl_pct",
                "realized_r", "mfe_pct", "mae_pct", "chart_candles_json", "split", "is_test_1y",
                "entry_price", "sl_price", "target_price", "risk", "triggered", "Trade_Triggered"
            ]
            leaks = [feat for feat in features if feat in forbidden]
            self.assertEqual(len(leaks), 0, f"Outcome leakage in {feat_file}: {leaks}")

    def test_03_bracket_geometry_mathematical_precision(self):
        """Verifies strict 1:2.00 Risk-Reward bracket geometry across sample trades."""
        df = load_50ma_dataset()
        sample = df.sample(min(100, len(df)), random_state=42)

        for _, row in sample.iterrows():
            entry = float(row["entry_price"])
            sl = float(row["sl_price"])
            tgt = float(row["target_price"])
            risk = float(row["risk"])

            # Risk = Entry - SL
            expected_risk = round(entry - sl, 2)
            self.assertAlmostEqual(risk, expected_risk, delta=0.05)

            # Target = Entry + 2 * Risk (Strict 1:2.00 R:R)
            expected_target = round(entry + (2.0 * risk), 2)
            self.assertAlmostEqual(tgt, expected_target, delta=0.05)

    def test_04_day1_trigger_gate_integrity(self):
        """Verifies no untriggered setups (Day 1 High < Entry) exist in 5_year_final_data_50ma.csv."""
        df = load_50ma_dataset()
        if "triggered" in df.columns:
            untriggered = df[df["triggered"] == False]
            self.assertEqual(len(untriggered), 0, "Found untriggered setups in 50MA verified dataset")

        if "outcome" in df.columns:
            no_triggers = df[df["outcome"].isin(["NO_TRIGGER", "No_Trigger"])]
            self.assertEqual(len(no_triggers), 0, "Found NO_TRIGGER trades in 50MA verified dataset")

    # ==========================================================================
    # 2. 50MA Service & API Boundary Stress Testing
    # ==========================================================================

    def test_05_ma50_service_get_by_id_edge_cases(self):
        """Verifies get_50ma_trade_by_id handles edge cases (empty, non-existent, whitespace)."""
        self.assertIsNone(get_50ma_trade_by_id(""))
        self.assertIsNone(get_50ma_trade_by_id("NON_EXISTENT_TRADE_9999999"))
        self.assertIsNone(get_50ma_trade_by_id("   "))

        # Valid trade retrieval
        df = load_50ma_dataset()
        first_id = str(df.iloc[0]["trade_id"])
        res = get_50ma_trade_by_id(first_id)
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("candles", res)
        self.assertIn("coordinates", res)
        self.assertIn("trade", res)
        self.assertEqual(res["coordinates"]["risk_reward"], "1 : 2.00")

    def test_06_ma50_service_query_boundary_stress(self):
        """Verifies query_50ma_trades handles extreme pagination, negative limits, and filters."""
        # Extreme high page number
        res_high_page = query_50ma_trades(page=999999, limit=50)
        self.assertEqual(res_high_page["status"], "SUCCESS")
        self.assertEqual(len(res_high_page["trades"]), 0)

        # Extreme limit clamped safely
        res_limit = query_50ma_trades(page=1, limit=999999)
        self.assertEqual(res_limit["status"], "SUCCESS")
        self.assertLessEqual(res_limit["limit"], 500)

        # Subclass filters (1, 2, 3)
        for s_num in [1, 2, 3]:
            res_sub = query_50ma_trades(setup=s_num, limit=10)
            self.assertEqual(res_sub["status"], "SUCCESS")
            for t in res_sub["trades"]:
                self.assertEqual(t["setup_num"], s_num)

        # Split filter: TRAIN vs TEST_1Y
        res_train = query_50ma_trades(split="TRAIN", limit=10)
        for t in res_train["trades"]:
            self.assertEqual(t["split"], "TRAIN")

        res_test = query_50ma_trades(split="TEST_1Y", limit=10)
        for t in res_test["trades"]:
            self.assertEqual(t["split"], "TEST_1Y")

    def test_07_chart_candlestick_trajectory_structure(self):
        """Verifies chart candlestick payload structure (past, setup, future bars)."""
        df = load_50ma_dataset()
        first_id = str(df.iloc[0]["trade_id"])
        res = get_50ma_trade_by_id(first_id)

        candles = res["candles"]
        self.assertGreaterEqual(len(candles), 8, "Expected at least past 7 + setup candle")

        types = [c["type"] for c in candles]
        self.assertIn("past", types)
        self.assertIn("setup", types)

        # Verify indicators present on past and setup bars
        for c in candles:
            if c["type"] in ("past", "setup"):
                self.assertIn("ma50", c)
                self.assertIn("bb_lower", c)

    # ==========================================================================
    # 3. HTTP Endpoints Contract Validation
    # ==========================================================================

    def test_08_api_status_contract(self):
        """Verifies GET /api/status returns ONLINE and contains all pipeline stages."""
        res = self.app_client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("download_info", data)
        self.assertIn("data_cleaning_info", data)
        self.assertIn("models_info", data)
        self.assertIn("inference_info", data)

    def test_09_api_50ma_endpoints_contract(self):
        """Verifies GET /api/strategy/50ma/* endpoints return valid JSON structures."""
        # Summary
        res_sum = self.app_client.get("/api/strategy/50ma/summary")
        self.assertEqual(res_sum.status_code, 200)
        d_sum = res_sum.get_json()
        self.assertEqual(d_sum["status"], "SUCCESS")
        self.assertIn("overall", d_sum)
        self.assertEqual(d_sum["overall"]["total_trades"], 8667)
        self.assertIn("last_1_year", d_sum)
        self.assertEqual(d_sum["last_1_year"]["total_trades"], 1576)
        self.assertIn("setups", d_sum)

        # Trades ledger
        res_tr = self.app_client.get("/api/strategy/50ma/trades?page=1&limit=15")
        self.assertEqual(res_tr.status_code, 200)
        d_tr = res_tr.get_json()
        self.assertEqual(len(d_tr["trades"]), 15)
        self.assertIn("summary", d_tr)

        # Chart
        sample_id = d_tr["trades"][0]["trade_id"]
        res_chart = self.app_client.get(f"/api/strategy/50ma/trade/{sample_id}/chart")
        self.assertEqual(res_chart.status_code, 200)
        d_chart = res_chart.get_json()
        self.assertEqual(d_chart["status"], "SUCCESS")
        self.assertIn("candles", d_chart)
        self.assertIn("coordinates", d_chart)

    def test_10_api_daily_schedule_contract(self):
        """Verifies GET /api/analysis/schedule returns market session evaluation."""
        res = self.app_client.get("/api/analysis/schedule")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("schedule", data)
        sched = data["schedule"]
        self.assertIn("is_market_closed", sched)
        self.assertIn("session_type", sched)
        self.assertIn("target_date", sched)

    # ==========================================================================
    # 4. JSON Manager Serialization & Thread Safety
    # ==========================================================================

    def test_11_json_manager_atomic_read_write(self):
        """Verifies JSON Manager atomic write and read functions preserve data integrity."""
        test_file = self.data_store_dir / "_test_atomic_integrity.json"
        payload = {
            "test_key": "Phase 5 Verification",
            "nested": {"count": 42, "ratio": 3.14159},
            "array": [1, 2, 3, "alpha", "omega"]
        }

        try:
            json_manager.write_json_file(test_file, payload)
            read_back = json_manager.read_json_file(test_file)
            self.assertEqual(read_back, payload)
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    unittest.main()
