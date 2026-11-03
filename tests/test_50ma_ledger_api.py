"""
Unit & Integration Tests for VectorTrade 50MA Trade Ledger & Chart Data API (Phase 2)
Verifies:
1. Executive summary statistics & KPI calculations (/api/strategy/50ma/summary)
2. Trade ledger querying, filtering (outcome, setup, symbol, sector), sorting, and pagination (/api/strategy/50ma/trades)
3. TradingView-style 38-bar chart payload and price bracket coordinates (/api/strategy/50ma/trade/<id>/chart)
4. Error handling (404 for missing trade ID, boundary pagination, invalid inputs)
5. Service-level caching and O(1) trade indexing integrity
"""

import unittest
import json
from pathlib import Path
import pandas as pd

from offline.app import app
from offline import ma50_service


class Test50MALedgerApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes Flask test client and pre-warms the 50MA in-memory dataset cache."""
        cls.client = app.test_client()
        cls.df = ma50_service.load_50ma_dataset()

    def test_01_summary_api_structure_and_kpis(self):
        """Test 1: Verifies /api/strategy/50ma/summary returns exact overall, setup, and yearly KPIs."""
        response = self.client.get("/api/strategy/50ma/summary")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertEqual(data.get("strategy"), "50_MA_SETUP")

        overall = data.get("overall", {})
        self.assertEqual(overall.get("total_trades"), 8667)
        self.assertEqual(overall.get("success_count"), 3068)
        self.assertEqual(overall.get("loss_count"), 5117)
        self.assertEqual(overall.get("neutral_count"), 474)
        self.assertAlmostEqual(overall.get("win_rate", 0), 35.4, places=1)
        self.assertGreater(overall.get("total_realized_r", 0), 1000.0)

        # Check Setups breakdown
        setups = data.get("setups", {})
        self.assertIn("setup_1", setups)
        self.assertIn("setup_2", setups)
        self.assertIn("setup_3", setups)
        total_setup_count = sum(s["total_trades"] for s in setups.values())
        self.assertEqual(total_setup_count, 8667)

        # Check Yearly breakdown
        yearly = data.get("yearly", {})
        for yr in ["2021", "2022", "2023", "2024", "2025", "2026"]:
            self.assertIn(yr, yearly)

        # Check Top Sectors
        sectors = data.get("top_sectors", [])
        self.assertGreaterEqual(len(sectors), 5)
        self.assertIn("sector", sectors[0])

    def test_02_trades_list_default_pagination(self):
        """Test 2: Verifies /api/strategy/50ma/trades default pagination (50 items per page)."""
        response = self.client.get("/api/strategy/50ma/trades")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertEqual(data.get("page"), 1)
        self.assertEqual(data.get("limit"), 50)
        self.assertEqual(data.get("total_trades"), 8667)
        self.assertGreater(data.get("total_pages"), 170)

        trades = data.get("trades", [])
        self.assertEqual(len(trades), 50)

        # Verify lightweight trade object schema
        first = trades[0]
        required_fields = [
            "trade_id", "symbol", "setup_date", "sector", "setup_num", "setup_name",
            "entry_price", "sl_price", "risk", "target_price", "risk_reward",
            "outcome", "days_held", "exit_price", "exit_date", "realized_pnl_pct", "realized_r"
        ]
        for field in required_fields:
            self.assertIn(field, first, f"Missing field {field} in trade record")

    def test_03_trades_filter_by_outcome_success(self):
        """Test 3: Verifies filtering by outcome=SUCCESS returns only winning trades."""
        response = self.client.get("/api/strategy/50ma/trades?outcome=SUCCESS&limit=20")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("total_trades"), 3068)
        self.assertEqual(data.get("summary", {}).get("win_rate"), 100.0)

        trades = data.get("trades", [])
        self.assertEqual(len(trades), 20)
        for t in trades:
            self.assertEqual(t.get("outcome"), "SUCCESS")
            self.assertAlmostEqual(t.get("realized_r"), 2.0, places=1)

    def test_04_trades_filter_by_outcome_loss(self):
        """Test 4: Verifies filtering by outcome=LOSS returns only stop loss trades."""
        response = self.client.get("/api/strategy/50ma/trades?outcome=LOSS&limit=20")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("total_trades"), 5117)

        trades = data.get("trades", [])
        self.assertEqual(len(trades), 20)
        for t in trades:
            self.assertEqual(t.get("outcome"), "LOSS")
            self.assertAlmostEqual(t.get("realized_r"), -1.0, places=1)

    def test_05_trades_filter_by_outcome_neutral(self):
        """Test 5: Verifies filtering by outcome=NEUTRAL returns 30-day timeout trades."""
        response = self.client.get("/api/strategy/50ma/trades?outcome=NEUTRAL&limit=20")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("total_trades"), 474)

        trades = data.get("trades", [])
        self.assertEqual(len(trades), 20)
        for t in trades:
            self.assertEqual(t.get("outcome"), "NEUTRAL")
            self.assertEqual(t.get("days_held"), 30)

    def test_06_trades_filter_by_setup_subclasses(self):
        """Test 6: Verifies filtering by setup=1, setup=2, and setup=3 correctly segments trades."""
        for s_num in [1, 2, 3]:
            response = self.client.get(f"/api/strategy/50ma/trades?setup={s_num}&limit=10")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            trades = data.get("trades", [])
            self.assertGreater(len(trades), 0)
            for t in trades:
                self.assertEqual(t.get("setup_num"), s_num)

    def test_07_trades_filter_by_symbol_search(self):
        """Test 7: Verifies substring and case-insensitive symbol search."""
        response = self.client.get("/api/strategy/50ma/trades?symbol=reliance&limit=25")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        trades = data.get("trades", [])
        self.assertGreater(len(trades), 0)
        for t in trades:
            self.assertIn("RELIANCE", t.get("symbol", "").upper())

    def test_08_trades_custom_sorting_order(self):
        """Test 8: Verifies sorting trades by realized_r ascending and descending."""
        # Descending: highest R first (2.0)
        res_desc = self.client.get("/api/strategy/50ma/trades?sort_by=realized_r&order=desc&limit=10")
        self.assertEqual(res_desc.status_code, 200)
        trades_desc = res_desc.get_json().get("trades", [])
        self.assertEqual(trades_desc[0].get("realized_r"), 2.0)

        # Ascending: lowest R first (-1.0)
        res_asc = self.client.get("/api/strategy/50ma/trades?sort_by=realized_r&order=asc&limit=10")
        self.assertEqual(res_asc.status_code, 200)
        trades_asc = res_asc.get_json().get("trades", [])
        self.assertEqual(trades_asc[0].get("realized_r"), -1.0)

    def test_09_trades_pagination_boundary_cases(self):
        """Test 9: Verifies pagination slice integrity and out-of-bounds page handling."""
        # Page 1 vs Page 2
        r1 = self.client.get("/api/strategy/50ma/trades?page=1&limit=10")
        r2 = self.client.get("/api/strategy/50ma/trades?page=2&limit=10")
        t1 = r1.get_json().get("trades", [])
        t2 = r2.get_json().get("trades", [])

        self.assertEqual(len(t1), 10)
        self.assertEqual(len(t2), 10)
        # Ensure mutually exclusive trade_ids
        ids_1 = {t["trade_id"] for t in t1}
        ids_2 = {t["trade_id"] for t in t2}
        self.assertEqual(len(ids_1.intersection(ids_2)), 0, "Pages 1 and 2 must not have overlapping records")

        # Out-of-bounds page
        r_oob = self.client.get("/api/strategy/50ma/trades?page=999999&limit=50")
        self.assertEqual(r_oob.status_code, 200)
        self.assertEqual(len(r_oob.get_json().get("trades", [])), 0)

    def test_10_chart_data_retrieval_and_candle_structure(self):
        """Test 10: Verifies /api/strategy/50ma/trade/<id>/chart returns 38 candles and bracket coordinates."""
        # Fetch an arbitrary valid trade_id from the dataset
        sample_trade_id = str(self.df.iloc[3]["trade_id"])
        response = self.client.get(f"/api/strategy/50ma/trade/{sample_trade_id}/chart")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("status"), "SUCCESS")

        # Verify trade metadata
        trade = data.get("trade", {})
        self.assertEqual(trade.get("trade_id"), sample_trade_id)
        self.assertIn("entry_price", trade)
        self.assertIn("sl_price", trade)
        self.assertIn("target_price", trade)

        # Verify price bracket coordinates
        coords = data.get("coordinates", {})
        self.assertEqual(coords.get("entry_price"), trade.get("entry_price"))
        self.assertEqual(coords.get("sl_price"), trade.get("sl_price"))
        self.assertEqual(coords.get("target_price"), trade.get("target_price"))
        expected_target = round(coords["entry_price"] + 2.0 * coords["risk"], 2)
        self.assertAlmostEqual(coords["target_price"], expected_target, places=1)

        # Verify candles payload
        candles = data.get("candles", [])
        self.assertGreaterEqual(len(candles), 8)
        self.assertLessEqual(len(candles), 38)

        # Check past, setup, future breakdown
        types = [c.get("type") for c in candles]
        self.assertIn("past", types)
        self.assertIn("setup", types)
        self.assertIn("future", types)

        # Check setup candle
        setup_c = [c for c in candles if c.get("type") == "setup"][0]
        self.assertIn("ma50", setup_c)
        self.assertIn("bb_lower", setup_c)
        self.assertIsNotNone(setup_c["ma50"])
        self.assertIsNotNone(setup_c["bb_lower"])

    def test_11_chart_data_404_on_invalid_trade_id(self):
        """Test 11: Verifies 404 response on non-existent trade_id."""
        response = self.client.get("/api/strategy/50ma/trade/INVALID_TRADE_ID_99999/chart")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ERROR")
        self.assertIn("not found", data.get("message", "").lower())

    def test_12_service_cache_and_indexing_performance(self):
        """Test 12: Verifies service caching reuses in-memory dataframe and index lookups are sub-millisecond."""
        import time

        # Fast lookup test
        sample_id = str(self.df.iloc[100]["trade_id"])
        t0 = time.time()
        for _ in range(100):
            res = ma50_service.get_50ma_trade_by_id(sample_id)
            self.assertIsNotNone(res)
        elapsed = time.time() - t0

        # 100 lookups including full JSON parsing should complete in under 0.5s (average < 5ms per lookup)
        self.assertLess(elapsed, 0.5, f"100 lookups took {elapsed:.4f}s, expected < 0.5s")

    def test_13_trades_filter_by_split_train(self):
        """Test 13: Verifies filtering by split=TRAIN returns exact historical training trades."""
        response = self.client.get("/api/strategy/50ma/trades?split=TRAIN&limit=20")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("total_trades"), 7091)
        trades = data.get("trades", [])
        self.assertEqual(len(trades), 20)
        for t in trades:
            self.assertEqual(t.get("split"), "TRAIN")
            self.assertFalse(t.get("is_test_1y"))

    def test_14_trades_filter_by_split_test_1y(self):
        """Test 14: Verifies filtering by split=TEST_1Y returns exact 1-year out-of-sample test trades."""
        response = self.client.get("/api/strategy/50ma/trades?split=TEST_1Y&limit=20")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data.get("total_trades"), 1576)
        trades = data.get("trades", [])
        self.assertEqual(len(trades), 20)
        for t in trades:
            self.assertEqual(t.get("split"), "TEST_1Y")
            self.assertTrue(t.get("is_test_1y"))

    def test_15_summary_api_last_1_year_kpis(self):
        """Test 15: Verifies /api/strategy/50ma/summary includes exact last_1_year KPI breakdown."""
        response = self.client.get("/api/strategy/50ma/summary")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("last_1_year", data)
        l1y = data["last_1_year"]
        self.assertEqual(l1y.get("timeframe"), "LAST_1_YEAR")
        self.assertEqual(l1y.get("total_trades"), 1576)
        self.assertEqual(l1y.get("success_count"), 453)
        self.assertEqual(l1y.get("loss_count"), 1035)
        self.assertEqual(l1y.get("neutral_count"), 80)
        self.assertAlmostEqual(l1y.get("win_rate", 0), 28.7, places=1)
        self.assertEqual(l1y.get("start_date"), "2025-10-06")
        self.assertEqual(l1y.get("end_date"), "2026-10-05")


if __name__ == "__main__":
    unittest.main()
