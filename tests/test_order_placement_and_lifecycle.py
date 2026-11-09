"""
VectorTrade - Unit and Integration Tests for Order Placement & Bracket Lifecycle
Verifies:
1. Duplicate Position Prevention (holdings & open orders)
2. Off-Market Staging when market is closed
3. Live order placement on MegaBull buysell API
4. Bracket order placement (Target & SL) on execution
5. Daily bracket re-arming and trade completion detection
6. FastAPI endpoints for order placement, bracket sync, and portfolio symbols
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.services.megabull_service import MegaBullService, megabull_service


class TestOrderPlacementAndLifecycle(unittest.TestCase):

    def setUp(self):
        self.service = MegaBullService(api_key="test_api_key")
        self.client = TestClient(app, follow_redirects=False)
        self.liq_patch = patch.object(self.service, "get_liquidation_status", return_value={"armed": False, "status": "IDLE"})
        self.liq_patch.start()
        self.app_liq_patch = patch("online.app.megabull_service.get_liquidation_status", return_value={"armed": False, "status": "IDLE"})
        self.app_liq_patch.start()

    def tearDown(self):
        self.liq_patch.stop()
        self.app_liq_patch.stop()

    def test_01_duplicate_rejected_when_active_holding_exists(self):
        """Verifies duplicate buy/sell is rejected if stock is already held with qty != 0."""
        with patch.object(self.service, "get_holdings") as mock_holdings, \
             patch.object(self.service, "get_orders") as mock_orders:
            mock_holdings.return_value = [
                {"symbol": "EICHERMOT", "qty": 10, "avg_buy_price": 4500.0, "instrument_token": "232961"}
            ]
            mock_orders.return_value = {"open": [], "executed": []}

            # Attempt to place order for EICHERMOT
            res = self.service.place_order(
                symbol="EICHERMOT",
                action="BUY",
                quantity=5,
                price=4550.0,
                enforce_duplicate_check=True
            )

            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["code"], "DUPLICATE_POSITION")
            self.assertIn("already exists as an active holding", res["error"])

    def test_02_duplicate_rejected_when_pending_order_exists(self):
        """Verifies duplicate order is rejected if an open order for the same symbol is pending."""
        with patch.object(self.service, "get_holdings") as mock_holdings, \
             patch.object(self.service, "get_orders") as mock_orders:
            mock_holdings.return_value = []
            mock_orders.return_value = {
                "open": [{"symbol": "INFY", "instrumentName": "INFY", "qty": 10, "price": 1750.0}],
                "executed": []
            }

            res = self.service.place_order(
                symbol="INFY",
                action="BUY",
                quantity=10,
                price=1750.0,
                enforce_duplicate_check=True
            )

            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["code"], "DUPLICATE_POSITION")
            self.assertIn("already has a pending order", res["error"])

    def test_03_order_placed_stages_offmarket_when_market_closed(self):
        """Verifies order is safely staged off-market in MongoDB Atlas when broker returns Market Closed."""
        with patch.object(self.service, "is_symbol_in_portfolio") as mock_dup, \
             patch.object(self.service, "_persist_staged_order") as mock_persist, \
             patch("requests.post") as mock_post:
            mock_dup.return_value = {"exists": False}
            
            # Broker responds Market Closed
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.json.return_value = {"error": "ErrorMsgException", "message": ["Market Closed!"]}
            mock_post.return_value = mock_resp

            res = self.service.place_order(
                symbol="TCS",
                action="BUY",
                quantity=2,
                price=3800.0,
                target_price=3950.0,
                sl_price=3725.0,
                strategy="50 MA Setup"
            )

            self.assertEqual(res["status"], "STAGED_OFFMARKET")
            self.assertIn("Market Closed", res["message"])
            self.assertEqual(res["order"]["status"], "STAGED_OFFMARKET")
            self.assertEqual(res["order"]["symbol"], "TCS")
            self.assertEqual(res["order"]["price"], 3800.0)
            mock_persist.assert_called_once()

    def test_04_order_placed_executes_when_market_open(self):
        """Verifies order is sent to MegaBull /api/order/buysell and marked PLACED on HTTP 200."""
        with patch.object(self.service, "is_symbol_in_portfolio") as mock_dup, \
             patch.object(self.service, "_persist_staged_order") as mock_persist, \
             patch("requests.post") as mock_post:
            mock_dup.return_value = {"exists": False}

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"orderId": "MB_ORD_999", "status": "PENDING"}
            mock_post.return_value = mock_resp

            res = self.service.place_order(
                symbol="SBIN",
                action="BUY",
                quantity=15,
                price=820.0,
                strategy="50 MA Setup"
            )

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["order"]["status"], "PLACED")
            self.assertEqual(res["order"]["order_id"], "MB_ORD_999")
            mock_persist.assert_called_once()

    def test_05_place_bracket_orders_sends_target_and_sl(self):
        """Verifies place_bracket_orders dispatches Target and SL limit orders to MegaBull."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"orderId": "MB_BRACKET_1"}
            mock_post.return_value = mock_resp

            holding = {"qty": 10, "instrument_token": "408065", "symbol": "INFY"}
            res = self.service.place_bracket_orders(
                symbol="INFY",
                holding=holding,
                target_price=1850.0,
                sl_price=1700.0
            )

            self.assertEqual(res["status"], "PLACED")
            self.assertIsNotNone(res["target_order"])
            self.assertIsNotNone(res["sl_order"])
            self.assertEqual(mock_post.call_count, 2)

    def test_06_sync_execution_places_brackets_for_newly_filled_order(self):
        """Verifies lifecycle sync detects when staged order fills into active holding and places brackets."""
        with patch.object(self.service, "get_holdings") as mock_holdings, \
             patch.object(self.service, "get_orders") as mock_orders, \
             patch.object(self.service, "place_bracket_orders") as mock_place_brackets, \
             patch("online.db.db_manager") as mock_db:
            
            mock_holdings.return_value = [
                {"symbol": "RELIANCE", "qty": 5, "avg_buy_price": 2800.0, "instrument_token": "738561"}
            ]
            mock_orders.return_value = {"open": [], "executed": []}

            # MongoDB returns an active tracked order without open brackets
            mock_col = MagicMock()
            mock_col.find.return_value = [{
                "_id": "mock_id_123",
                "symbol": "RELIANCE",
                "status": "PLACED",
                "target_price": 2950.0,
                "sl_price": 2725.0
            }]
            mock_db.staged_orders = mock_col

            report = self.service.sync_execution_and_bracket_orders()

            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["active_holdings_count"], 1)
            self.assertEqual(report["brackets_placed_count"], 1)
            mock_place_brackets.assert_called_once_with("RELIANCE", mock_holdings.return_value[0], 2950.0, 2725.0)

    def test_07_sync_execution_detects_completed_trade(self):
        """Verifies lifecycle sync detects when holding is no longer active (Target or SL hit) and marks trade COMPLETED."""
        with patch.object(self.service, "get_holdings") as mock_holdings, \
             patch.object(self.service, "get_orders") as mock_orders, \
             patch("online.db.db_manager") as mock_db:

            # Holdings is now empty (position sold/closed)
            mock_holdings.return_value = []
            mock_orders.return_value = {"open": [], "executed": []}

            # MongoDB previously had an ACTIVE order
            mock_col = MagicMock()
            mock_col.find.return_value = [{
                "_id": "mock_id_456",
                "symbol": "TVSMOTOR",
                "status": "ACTIVE",
                "target_price": 2500.0,
                "sl_price": 2300.0
            }]
            mock_db.staged_orders = mock_col

            report = self.service.sync_execution_and_bracket_orders()

            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["completed_count"], 1)
            mock_col.update_one.assert_called_once()
            args, kwargs = mock_col.update_one.call_args
            self.assertEqual(args[1]["$set"]["status"], "COMPLETED")
            self.assertEqual(args[1]["$set"]["outcome"], "TARGET_OR_SL_HIT")

    def test_08_api_place_order_endpoint_handles_duplicates(self):
        """Verifies POST /api/megabull/orders/place rejects duplicate symbols with HTTP 400."""
        with patch("online.app.megabull_service.place_order") as mock_place:
            mock_place.return_value = {
                "status": "ERROR",
                "code": "DUPLICATE_POSITION",
                "error": "Cannot place order: Symbol 'EICHERMOT' already exists as an active holding."
            }

            resp = self.client.post("/api/megabull/orders/place", json={
                "symbol": "EICHERMOT",
                "action": "BUY",
                "quantity": 5,
                "price": 4500.0,
                "enforce_duplicate_check": True
            })

            self.assertEqual(resp.status_code, 400)
            self.assertIn("already exists", resp.json()["detail"])

    def test_09_api_portfolio_symbols_endpoint(self):
        """Verifies GET /api/megabull/portfolio-symbols returns held and pending lists."""
        with patch("online.app.megabull_service.get_holdings") as mock_holdings, \
             patch("online.app.megabull_service.get_orders") as mock_orders:
            mock_holdings.return_value = [
                {"symbol": "EICHERMOT", "qty": 10},
                {"symbol": "TATAELXSI", "qty": 5}
            ]
            mock_orders.return_value = {
                "open": [{"symbol": "INFY", "instrumentName": "INFY"}]
            }

            resp = self.client.get("/api/megabull/portfolio-symbols")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertIn("EICHERMOT", data["held_symbols"])
            self.assertIn("TATAELXSI", data["held_symbols"])
            self.assertIn("INFY", data["pending_symbols"])
            self.assertIn("EICHERMOT", data["all_portfolio_symbols"])
            self.assertIn("INFY", data["all_portfolio_symbols"])

    def test_10_api_sync_brackets_endpoint(self):
        """Verifies POST /api/megabull/orders/sync-brackets triggers lifecycle sync."""
        with patch("online.app.megabull_service.sync_execution_and_bracket_orders") as mock_sync:
            mock_sync.return_value = {
                "status": "SUCCESS",
                "active_holdings_count": 8,
                "synced_count": 8,
                "brackets_placed_count": 2,
                "completed_count": 1
            }

            resp = self.client.post("/api/megabull/orders/sync-brackets")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["data"]["brackets_placed_count"], 2)


if __name__ == "__main__":
    unittest.main()
