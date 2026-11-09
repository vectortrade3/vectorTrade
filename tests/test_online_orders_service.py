"""
VectorTrade - Unit & Domain Tests for Online OrdersService
Verifies legacy document normalization, multi-parameter filtering,
search, and executive statistics aggregation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.services.orders_service import OrdersService
from online.db import db_manager


class TestOnlineOrdersService(unittest.TestCase):

    def setUp(self):
        self.service = OrdersService()

    def tearDown(self):
        db_manager.reset_client()

    def test_01_normalize_order_handles_legacy_schema(self):
        """Verifies normalization seamlessly fills missing fields for legacy signals."""
        legacy_doc = {
            "_id": "SIG_HDFC_001",
            "symbol": "HDFCBANK",
            "instrument_token": "341249",
            "strategy_id": "GLOBAL_FORECAST",
            "buy_trigger": 1600.0,
            "sl_price": 1550.0,
            "target_price": 1700.0,
            "quantity": 15,
            "status": "ARMED",
            "valid_for_date": "2026-10-29"
        }

        norm = self.service.normalize_order(legacy_doc)
        self.assertEqual(norm["id"], "SIG_HDFC_001")
        self.assertEqual(norm["symbol"], "HDFCBANK")
        self.assertEqual(norm["strategy"], "Global Forecast")
        self.assertEqual(norm["action"], "BUY")
        self.assertEqual(norm["buy_trigger"], 1600.0)
        self.assertEqual(norm["entry_price"], 1600.0)
        self.assertEqual(norm["sl_price"], 1550.0)
        self.assertEqual(norm["target_price"], 1700.0)
        self.assertEqual(norm["risk_per_share"], 50.0)
        self.assertEqual(norm["quantity"], 15)
        self.assertEqual(norm["status"], "ARMED")
        self.assertIn("Armed for Execution", norm["outcome_detail"])

    def test_02_normalize_order_modern_schema(self):
        """Verifies full modern schema normalization with all 24 trade parameters."""
        modern_doc = {
            "_id": "SIG_50_MA_SETUP_TCS_2026-11-03",
            "symbol": "TCS",
            "name": "Tata Consultancy Services Ltd",
            "sector": "Information Technology",
            "strategy": "50 MA Setup",
            "setup_subclass": "50MA Setup 2 (Extended Trend)",
            "setup_num": 2,
            "action": "STRONG BUY",
            "entry_price": 4200.0,
            "buy_trigger": 4200.0,
            "sl_price": 4120.0,
            "target_price": 4360.0,
            "risk_per_share": 80.0,
            "quantity": 125,
            "total_capital_required": 525000.0,
            "loss_at_sl": 10000.0,
            "profit_at_target": 20000.0,
            "loss_pct": 1.9,
            "profit_pct": 3.81,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0,
            "predicted_r": 1.82,
            "ml_confidence": 0.91,
            "status": "ARMED",
            "outcome_detail": "Armed for Execution: Buy Trigger ₹4200.00, SL ₹4120.00, Target ₹4360.00.",
            "rationale": "Extended trend above 50MA with high momentum.",
            "rsi": 62.4,
            "atr": 52.0,
            "ma50": 4100.0,
            "dist_low_bb": 12.0,
            "dist_low_ma50": 35.0,
            "is_latest": True,
            "valid_for_date": "2026-11-03",
            "synced_at": "2026-11-03T15:35:00+05:30"
        }

        norm = self.service.normalize_order(modern_doc)
        self.assertEqual(norm["symbol"], "TCS")
        self.assertEqual(norm["name"], "Tata Consultancy Services Ltd")
        self.assertEqual(norm["sector"], "Information Technology")
        self.assertEqual(norm["setup_num"], 2)
        self.assertEqual(norm["setup_subclass"], "50MA Setup 2 (Extended Trend)")
        self.assertEqual(norm["action"], "STRONG BUY")
        self.assertEqual(norm["predicted_r"], 1.82)
        self.assertEqual(norm["rsi"], 62.4)
        self.assertEqual(norm["atr"], 52.0)

    def test_03_get_orders_status_filtering(self):
        """Verifies filtering by status on mock signals collection."""
        mock_col = MagicMock()
        mock_docs = [
            {"_id": "1", "symbol": "INFY", "status": "ARMED", "buy_trigger": 1800.0, "predicted_r": 1.5},
            {"_id": "2", "symbol": "WIPRO", "status": "ACTIVE", "buy_trigger": 500.0, "predicted_r": 1.2},
            {"_id": "3", "symbol": "HCLTECH", "status": "TARGET_HIT", "buy_trigger": 1600.0, "predicted_r": 1.8}
        ]

        # When querying ARMED only
        mock_col.find.return_value.limit.return_value = [mock_docs[0]]
        db_manager.set_mock_db({"signals_watchlist": mock_col})

        orders = self.service.get_orders(status="ARMED")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "INFY")
        self.assertEqual(orders[0]["status"], "ARMED")

    def test_04_get_orders_strategy_and_search_query_building(self):
        """Verifies query builder merges search regex and strategy accurately."""
        mock_col = MagicMock()
        mock_col.find.return_value.limit.return_value = []
        db_manager.set_mock_db({"signals_watchlist": mock_col})

        self.service.get_orders(strategy="50 MA Setup", search="TATA", status="ARMED")
        # Verify find was called with expected query shape
        call_args = mock_col.find.call_args[0][0]
        self.assertIn("$and", call_args)
        self.assertEqual(call_args["$and"][0]["status"], "ARMED")

    def test_05_get_order_by_id(self):
        """Verifies finding order by exact primary key ID."""
        mock_col = MagicMock()
        mock_col.find_one.return_value = {
            "_id": "SIG_TEST_01",
            "symbol": "SBIN",
            "status": "ARMED",
            "buy_trigger": 850.0
        }
        db_manager.set_mock_db({"signals_watchlist": mock_col})

        order = self.service.get_order_by_id("SIG_TEST_01")
        self.assertIsNotNone(order)
        self.assertEqual(order["symbol"], "SBIN")
        mock_col.find_one.assert_called_with({"_id": "SIG_TEST_01"})

    def test_06_get_order_summary_stats(self):
        """Verifies KPI aggregation across armed orders."""
        mock_col = MagicMock()
        mock_docs = [
            {
                "_id": "1", "symbol": "RELIANCE", "status": "ARMED",
                "predicted_r": 1.6, "profit_pct": 3.5, "loss_pct": 1.75,
                "total_capital_required": 50000.0, "loss_at_sl": 1000.0,
                "strategy": "50 MA Setup", "setup_subclass": "Setup 1",
                "valid_for_date": "2026-11-03", "synced_at": "2026-11-03T15:00:00"
            },
            {
                "_id": "2", "symbol": "TCS", "status": "ARMED",
                "predicted_r": 1.8, "profit_pct": 4.5, "loss_pct": 2.25,
                "total_capital_required": 60000.0, "loss_at_sl": 1200.0,
                "strategy": "50 MA Setup", "setup_subclass": "Setup 2",
                "valid_for_date": "2026-11-03", "synced_at": "2026-11-03T15:01:00"
            },
            {
                "_id": "3", "symbol": "INFY", "status": "TARGET_HIT",
                "predicted_r": 1.4, "profit_pct": 3.0, "loss_pct": 1.5,
                "total_capital_required": 40000.0, "loss_at_sl": 800.0,
                "strategy": "Global Forecast", "setup_subclass": "Setup 1",
                "valid_for_date": "2026-11-02", "synced_at": "2026-11-02T15:00:00"
            }
        ]
        mock_col.find.return_value = mock_docs
        db_manager.set_mock_db({"signals_watchlist": mock_col})

        stats = self.service.get_order_summary_stats()
        self.assertEqual(stats["total_orders"], 3)
        self.assertEqual(stats["armed_orders_count"], 2)
        self.assertEqual(stats["total_capital_required"], 110000.0)
        self.assertEqual(stats["total_risk_budget"], 2200.0)
        # Avg predicted_r for ARMED: (1.6 + 1.8) / 2 = 1.7
        self.assertEqual(stats["avg_predicted_r"], 1.7)
        self.assertEqual(stats["latest_date"], "2026-11-03")


if __name__ == "__main__":
    unittest.main()
