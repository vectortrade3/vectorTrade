"""
Verification Test Suite: WebSocket Live Stream, Order Cancellation, and 1% Capital Risk Sizing.
"""

import sys
import os
import unittest
from pathlib import Path

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from core.config import settings
from core.db import db_manager
from core.risk_manager import PortfolioRiskManager
from online.app import app
from online.execution_worker import execution_worker


class TestWebSocketAndRisk(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.signals_col = db_manager.get_collection("signals_watchlist")
        self.active_col = db_manager.get_collection("active_trades")
        self.history_col = db_manager.get_collection("trade_history")

    def test_risk_manager_strict_one_percent_cap(self):
        """
        Verify that risk per trade is strictly <= 1.0% of total capital,
        and quantity matches: floor(Risk_Amount / (Buy - SL)).
        """
        capital = 100000.0  # Rs 1 Lakh
        risk_pct = 1.0      # 1%
        risk_mgr = PortfolioRiskManager(
            total_capital=capital,
            risk_per_trade_pct=risk_pct,
            max_open_positions=5,
            max_sector_allocation_pct=30.0
        )

        # Expected risk budget = 1000.0
        self.assertEqual(risk_mgr.risk_amount_per_trade, 1000.0)

        # Scenario 1: Entry = 100, SL = 90 -> Risk per share = 10
        # Expected quantity = 1000 // 10 = 100 shares
        qty_1 = risk_mgr.calculate_position_size(buy_trigger=100.0, sl_price=90.0)
        self.assertEqual(qty_1, 100)
        actual_risk_1 = (100.0 - 90.0) * qty_1
        self.assertLessEqual(actual_risk_1, 1000.0)

        # Scenario 2: Entry = 2500, SL = 2450 -> Risk per share = 50
        # Expected quantity = 1000 // 50 = 20 shares
        qty_2 = risk_mgr.calculate_position_size(buy_trigger=2500.0, sl_price=2450.0)
        self.assertEqual(qty_2, 20)
        actual_risk_2 = (2500.0 - 2450.0) * qty_2
        self.assertLessEqual(actual_risk_2, 1000.0)

        # Scenario 3: High priced stock where single share risk > risk budget (e.g. SL distance 1500 > 1000)
        # Expected quantity = 0 (discarded, zero risk allowed to breach 1%)
        qty_3 = risk_mgr.calculate_position_size(buy_trigger=10000.0, sl_price=8500.0)
        self.assertEqual(qty_3, 0)

        print("\n[PASS] Risk sizing strictly adheres to <= 1.0% capital risk rule.")

    def test_cancel_armed_signal_via_api(self):
        """
        Verify cancelling an ARMED signal via POST /api/order/cancel:
        - Updates MongoDB signals_watchlist to CANCELLED
        - Removes from execution_worker.armed_signals
        """
        test_sig_id = "SIG_TEST_CANCEL_001"
        self.signals_col.delete_one({"_id": test_sig_id})

        signal_doc = {
            "_id": test_sig_id,
            "symbol": "TATAMOTORS",
            "instrument_token": "884737",
            "strategy_id": "50MA_PULLBACK",
            "buy_trigger": 950.0,
            "sl_price": 935.0,
            "target_price": 980.0,
            "quantity": 66,
            "status": "ARMED"
        }
        self.signals_col.insert_one(signal_doc)
        execution_worker.armed_signals[test_sig_id] = signal_doc

        # Dispatch cancel request
        response = self.client.post("/api/order/cancel", json={"id": test_sig_id, "type": "SIGNAL"})
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "SUCCESS")

        # Verify DB status
        updated_sig = self.signals_col.find_one({"_id": test_sig_id})
        self.assertIsNotNone(updated_sig)
        self.assertEqual(updated_sig["status"], "CANCELLED")
        self.assertIn("cancelled_at", updated_sig)

        # Verify removed from in-memory worker
        self.assertNotIn(test_sig_id, execution_worker.armed_signals)

        # Clean up
        self.signals_col.delete_one({"_id": test_sig_id})
        print("[PASS] ARMED signal cancellation via API verified.")

    def test_close_active_trade_via_api(self):
        """
        Verify emergency closing an active OPEN trade via POST /api/order/cancel:
        - Removes from active_trades collection
        - Inserts into trade_history collection with realized PnL
        - Removes from execution_worker.active_trades
        """
        test_trade_id = "TRADE_TEST_CLOSE_001"
        self.active_col.delete_one({"_id": test_trade_id})
        self.history_col.delete_many({"trade_id": test_trade_id})

        trade_doc = {
            "_id": test_trade_id,
            "symbol": "INFY",
            "instrument_token": "408065",
            "strategy_id": "50MA_PULLBACK",
            "megabull_order_id": "ORD-MOCK-1234",
            "fill_price": 1850.0,
            "sl_price": 1820.0,
            "target_price": 1910.0,
            "quantity": 33,
            "status": "OPEN"
        }
        self.active_col.insert_one(trade_doc)
        execution_worker.active_trades[test_trade_id] = trade_doc

        response = self.client.post("/api/order/cancel", json={"id": test_trade_id, "type": "TRADE"})
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "SUCCESS")

        # Verify removed from active_trades
        active_check = self.active_col.find_one({"_id": test_trade_id})
        self.assertIsNone(active_check)
        self.assertNotIn(test_trade_id, execution_worker.active_trades)

        # Verify inserted into trade_history
        hist_check = self.history_col.find_one({"trade_id": test_trade_id})
        self.assertIsNotNone(hist_check)
        self.assertEqual(hist_check["symbol"], "INFY")
        self.assertEqual(hist_check["exit_reason"], "MANUAL_USER_CANCEL")
        self.assertIn("pnl", hist_check)

        # Clean up
        self.history_col.delete_many({"trade_id": test_trade_id})
        print("[PASS] Active trade emergency closure via API verified.")

    def test_websocket_live_stream_connect_and_cancel(self):
        """
        Verify WebSocket /ws/live endpoint:
        - Client connects successfully and immediately receives full system state snapshot
        - Client sends CANCEL_ORDER action over WebSocket
        - Server processes cancellation and responds with ACTION_RESPONSE
        """
        test_sig_id = "SIG_WS_CANCEL_002"
        self.signals_col.delete_one({"_id": test_sig_id})

        sig_doc = {
            "_id": test_sig_id,
            "symbol": "RELIANCE",
            "instrument_token": "738561",
            "strategy_id": "GLOBAL_FORECAST",
            "buy_trigger": 2980.0,
            "sl_price": 2950.0,
            "target_price": 3040.0,
            "quantity": 33,
            "status": "ARMED"
        }
        self.signals_col.insert_one(sig_doc)
        execution_worker.armed_signals[test_sig_id] = sig_doc

        with self.client.websocket_connect("/ws/live") as websocket:
            # 1. Verify initial snapshot
            initial_data = websocket.receive_json()
            self.assertIn("signals_watchlist", initial_data)
            self.assertIn("active_trades", initial_data)
            self.assertIn("system_status", initial_data)

            # 2. Send CANCEL_ORDER action over WebSocket
            websocket.send_json({
                "action": "CANCEL_ORDER",
                "id": test_sig_id,
                "type": "SIGNAL"
            })

            # 3. Receive action confirmation
            action_res = websocket.receive_json()
            self.assertEqual(action_res.get("type"), "ACTION_RESPONSE")
            self.assertEqual(action_res.get("result", {}).get("status"), "SUCCESS")

        # Verify DB updated
        updated_sig = self.signals_col.find_one({"_id": test_sig_id})
        self.assertEqual(updated_sig["status"], "CANCELLED")

        # Clean up
        self.signals_col.delete_one({"_id": test_sig_id})
        print("[PASS] WebSocket live connection and real-time cancellation verified.")


if __name__ == "__main__":
    unittest.main()
