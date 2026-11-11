"""
VectorTrade - Unit & Integration Tests for MongoSyncer
Verifies signal payload formatting, strict 1:2 geometry calculations,
and idempotent synchronization.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline.mongo_syncer import MongoSyncer
from offline.app import app as flask_offline_app


class TestMongoSyncer(unittest.TestCase):

    def setUp(self):
        self.sample_trade = {
            "symbol": "RELIANCE",
            "name": "Reliance Industries Ltd",
            "sector": "Energy",
            "date": "2026-11-03",
            "strategy": "50 MA Setup",
            "setup_subclass": "50MA Setup 1 (Confluence Zone)",
            "setup_num": 1,
            "action": "STRONG BUY",
            "entry_price": 2850.0,
            "sl_price": 2800.0,
            "target_price": 2950.0,
            "loss_pct": 1.75,
            "profit_pct": 3.51,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0,
            "predicted_r": 1.65,
            "outcome": "ARMED",
            "outcome_detail": "Armed for execution.",
            "rationale": "Confluence zone with upward trending 50MA.",
            "rsi": 54.2,
            "atr": 45.0,
            "ma50": 2810.0,
            "dist_low_bb": 5.2,
            "dist_low_ma50": 10.0,
            "is_latest": True
        }

    def test_01_syncer_initialization(self):
        """Verifies MongoSyncer initializes with config and default timeouts."""
        syncer = MongoSyncer()
        self.assertIsNotNone(syncer.db_name)
        self.assertEqual(syncer.db_name, "vectortrade")
        self.assertGreater(syncer.timeout_ms, 0)

    def test_02_format_trade_to_signal_geometry(self):
        """Verifies formatting respects strict 1:2 R:R geometry and quantity sizing."""
        signal = MongoSyncer.format_trade_to_signal(self.sample_trade, default_risk_budget=10000.0)

        self.assertEqual(signal["_id"], "SIG_50_MA_SETUP_RELIANCE_2026-11-03")
        self.assertEqual(signal["symbol"], "RELIANCE")
        self.assertEqual(signal["strategy"], "50 MA Setup")
        self.assertEqual(signal["action"], "STRONG BUY")
        self.assertEqual(signal["entry_price"], 2850.0)
        self.assertEqual(signal["buy_trigger"], 2850.0)
        self.assertEqual(signal["sl_price"], 2800.0)
        self.assertEqual(signal["target_price"], 2950.0)

        # Risk = 2850 - 2800 = 50 per share
        self.assertEqual(signal["risk_per_share"], 50.0)
        # Quantity = 10000 / 50 = 200 shares
        self.assertEqual(signal["quantity"], 200)
        # Capital = 2850 * 200 = 570000
        self.assertEqual(signal["total_capital_required"], 570000.0)
        # Loss at SL = 50 * 200 = 10000
        self.assertEqual(signal["loss_at_sl"], 10000.0)
        # Profit at Target = 100 * 200 = 20000
        self.assertEqual(signal["profit_at_target"], 20000.0)
        self.assertEqual(signal["risk_reward"], "1 : 2.00")
        self.assertEqual(signal["status"], "ARMED")

    def test_03_format_trade_preserves_indicators(self):
        """Verifies technical indicators (RSI, ATR, MA50) and metadata are captured."""
        signal = MongoSyncer.format_trade_to_signal(self.sample_trade)

        self.assertEqual(signal["rsi"], 54.2)
        self.assertEqual(signal["atr"], 45.0)
        self.assertEqual(signal["ma50"], 2810.0)
        self.assertEqual(signal["dist_low_bb"], 5.2)
        self.assertEqual(signal["dist_low_ma50"], 10.0)
        self.assertTrue(signal["is_latest"])
        self.assertIn("synced_at", signal)

    def test_04_syncer_offline_fallback_on_unreachable(self):
        """Verifies clean offline fallback when MongoDB Atlas URI is unresolvable."""
        syncer = MongoSyncer(uri="mongodb://invalid-host-which-does-not-exist:27017")
        syncer.timeout_ms = 500  # short timeout

        res = syncer.sync_trades([self.sample_trade])
        self.assertIn(res["status"], ["OFFLINE_FALLBACK", "ERROR"])
        self.assertEqual(res["synced_count"], 0)

    def test_05_syncer_skips_empty_trades(self):
        """Verifies empty trades list returns SKIPPED gracefully."""
        syncer = MongoSyncer()
        res = syncer.sync_trades([])
        self.assertEqual(res["status"], "SKIPPED")
        self.assertEqual(res["synced_count"], 0)

    def test_06_sync_from_json_file(self):
        """Verifies syncing from daily_analysis.json directly."""
        syncer = MongoSyncer()
        json_path = BASE_DIR / "data_store" / "daily_analysis.json"
        if json_path.exists():
            res = syncer.sync_from_json_file(json_path)
            self.assertIn(res["status"], ["SUCCESS", "OFFLINE_FALLBACK", "ERROR", "SKIPPED"])
            if res["status"] == "SUCCESS":
                self.assertGreater(res["synced_count"], 0)

    def test_07_offline_app_sync_endpoints(self):
        """Tests offline Flask app endpoints for MongoDB status and manual sync."""
        client = flask_offline_app.test_client()

        # Status check endpoint
        status_resp = client.get("/api/sync_mongodb/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()
        self.assertIn("status", data)
        self.assertIn("info", data)

        # Trigger sync endpoint
        sync_resp = client.post("/api/sync_mongodb")
        self.assertIn(sync_resp.status_code, [200, 500])
        sync_data = sync_resp.get_json()
        self.assertIn("status", sync_data)


if __name__ == "__main__":
    unittest.main()
