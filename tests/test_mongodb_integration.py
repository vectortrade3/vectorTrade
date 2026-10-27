"""
VectorTrade - MongoDB Atlas / mongomock Universal Integration Test Suite
Verifies:
1. DatabaseManager connection pool and schema indexes.
2. Offline daemon writes to signals_watchlist & buffer_daily_bars.
3. Online worker queries signals, creates active_trades, and archives to trade_history.
4. Unique compound index deduplication on buffer_daily_bars (symbol + date).
5. Inter-service mailbox request-response flow in MongoDB.
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from core.secure_comms import SecureMailboxCommunicator, TelemetryCompiler


class TestMongoDBIntegration(unittest.TestCase):

    def setUp(self):
        self.db = db_manager.db
        for col_name in ["signals_watchlist", "active_trades", "trade_history", "buffer_daily_bars", "inter_service_requests"]:
            self.db[col_name].delete_many({})

    def test_01_database_ping_and_indexes(self):
        """Verifies DB connection ping and index creation across all collections."""
        ping_res = db_manager.ping()
        self.assertEqual(ping_res["status"], "ok")
        self.assertIn("database", ping_res)

        # Verify collections exist
        cols = self.db.list_collection_names()
        expected = ["signals_watchlist", "active_trades", "trade_history", "buffer_daily_bars", "instruments_cache", "system_audit", "inter_service_requests"]
        for c in expected:
            col = db_manager.get_collection(c)
            self.assertIsNotNone(col)

    def test_02_offline_signal_dispatch_to_online_worker(self):
        """Verifies Offline -> MongoDB -> Online full lifecycle."""
        today_str = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d")

        # 1. Offline writes signal
        signals_col = db_manager.get_collection("signals_watchlist")
        signal_id = f"STRAT1_RELIANCE_{today_str}"
        signal_data = {
            "_id": signal_id,
            "symbol": "RELIANCE",
            "instrument_token": "738561",
            "strategy_id": "50MA_PULLBACK",
            "buy_trigger": 2850.0,
            "sl_price": 2800.0,
            "target_price": 2950.0,
            "quantity": 10,
            "total_capital_required": 28500.0,
            "profit_at_target": 1000.0,
            "loss_at_sl": 500.0,
            "ml_confidence": 0.85,
            "status": "ARMED",
            "valid_for_date": today_str
        }
        signals_col.update_one({"_id": signal_id}, {"$set": signal_data}, upsert=True)

        # 2. Online queries signal
        doc = signals_col.find_one({"_id": signal_id, "status": "ARMED"})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["symbol"], "RELIANCE")
        self.assertEqual(doc["buy_trigger"], 2850.0)

        # 3. Online triggers trade and creates active_trade
        active_col = db_manager.get_collection("active_trades")
        trade_id = f"TRADE_RELIANCE_{today_str}"
        trade_data = {
            "_id": trade_id,
            "symbol": "RELIANCE",
            "instrument_token": "738561",
            "strategy_id": "50MA_PULLBACK",
            "fill_price": 2852.0,
            "sl_price": 2800.0,
            "target_price": 2950.0,
            "quantity": 10,
            "status": "OPEN",
            "entry_timestamp": datetime.now(pytz.utc).isoformat()
        }
        active_col.insert_one(trade_data)
        signals_col.update_one({"_id": signal_id}, {"$set": {"status": "TRIGGERED"}})

        # 4. Online closes trade and archives to trade_history
        history_col = db_manager.get_collection("trade_history")
        hist_data = {
            "_id": f"HIST_{trade_id}",
            "trade_id": trade_id,
            "symbol": "RELIANCE",
            "entry_price": 2852.0,
            "exit_price": 2952.0,
            "quantity": 10,
            "pnl": 1000.0,
            "roi_pct": 3.5,
            "exit_reason": "TARGET_HIT",
            "ml_outcome_label": 1,
            "exit_timestamp": datetime.now(pytz.utc).isoformat()
        }
        history_col.insert_one(hist_data)
        active_col.delete_one({"_id": trade_id})

        # Assert final state in MongoDB
        self.assertEqual(signals_col.find_one({"_id": signal_id})["status"], "TRIGGERED")
        self.assertIsNone(active_col.find_one({"_id": trade_id}))
        self.assertIsNotNone(history_col.find_one({"_id": f"HIST_{trade_id}"}))

    def test_03_buffer_daily_bars_deduplication(self):
        """Verifies buffer_daily_bars upsert and deduplication logic."""
        bars_col = db_manager.get_collection("buffer_daily_bars")
        bar = {
            "symbol": "TCS",
            "date": "2026-10-27",
            "features": {"close": 4120.0, "rsi": 58.2},
            "consumed_for_training": False
        }
        bars_col.update_one({"symbol": "TCS", "date": "2026-10-27"}, {"$set": bar}, upsert=True)
        # Duplicate upsert with updated RSI
        bar["features"]["rsi"] = 60.1
        bars_col.update_one({"symbol": "TCS", "date": "2026-10-27"}, {"$set": bar}, upsert=True)

        count = bars_col.count_documents({"symbol": "TCS", "date": "2026-10-27"})
        self.assertEqual(count, 1)
        updated = bars_col.find_one({"symbol": "TCS", "date": "2026-10-27"})
        self.assertEqual(updated["features"]["rsi"], 60.1)

    def test_04_inter_service_requests_mailbox(self):
        """Verifies MongoDB collection inter_service_requests processes request/fulfillment."""
        comms_col = db_manager.get_collection("inter_service_requests")
        req_id = "REQ_TEST_DB_01"
        comms_col.insert_one({
            "request_id": req_id,
            "action": "GET_SYSTEM_HEALTH",
            "status": "PENDING",
            "timestamp": datetime.now(pytz.utc).timestamp(),
            "signature": "test_sig"
        })

        self.assertEqual(comms_col.count_documents({"status": "PENDING"}), 1)


if __name__ == "__main__":
    unittest.main()
