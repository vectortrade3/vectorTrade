"""
VectorTrade - Unit & Integration Tests for Emergency Liquidation Protocol
Tests:
1. Immediate square off at market price when market is open (Case 1)
2. Off-market scheduling for market open when market is closed (Case 2)
3. Persistent scheduled warning banner and cancel capability
4. Cancellation of all pending/staged buy orders and prevention of new buys
5. Automatic execution at market open via sync_execution_and_bracket_orders
6. REST API endpoints and UI template integration
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from online.app import app
from online.config import settings
from online.services.megabull_service import MegaBullService, megabull_service
from online.db import db_manager


class TestEmergencyLiquidation(unittest.TestCase):
    """Test suite for emergency square-off and liquidation workflow."""

    def setUp(self):
        self.client = TestClient(app)
        self.service = MegaBullService(api_key="test_api_key", base_url="https://api.megabull.in")

        # Mock MongoDB
        self.mock_db = MagicMock()
        self.mock_system_state = MagicMock()
        self.mock_staged_orders = MagicMock()
        self.mock_signals = MagicMock()

        self.mock_system_state.find_one.return_value = None

        cols_map = {
            "system_state": self.mock_system_state,
            "staged_orders": self.mock_staged_orders,
            "signals_watchlist": self.mock_signals,
            "active_trades": MagicMock(),
            "trade_history": MagicMock(),
            "system_audit": MagicMock()
        }
        self.mock_db.__getitem__.side_effect = lambda k: cols_map.get(k, MagicMock())
        self.mock_db.list_collection_names.return_value = ["system_state", "staged_orders", "signals_watchlist"]
        db_manager.set_mock_db(self.mock_db)

        # Authenticate session
        self.client.post(
            "/login",
            data={"username": settings.auth_username, "password": settings.auth_password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False
        )

        # Sample holdings with 1 long and 1 short
        self.sample_holdings = [
            {
                "symbol": "TITAN",
                "instrument_token": "897537",
                "qty": 12,
                "lot_size": 1,
                "type": "LONG",
                "avg_buy_price": 4379.0,
                "ltp": 4400.0,
                "invested_value": 52548.0,
                "current_value": 52800.0,
                "unrealized_pnl": 252.0,
                "charges_breakdown": {"brokerage": 40.0, "total_taxes": 15.0, "total_charges": 55.0}
            },
            {
                "symbol": "INFY",
                "instrument_token": "408065",
                "qty": -66,
                "lot_size": 1,
                "type": "SHORT",
                "avg_buy_price": 1163.0,
                "ltp": 1150.0,
                "invested_value": 76758.0,
                "current_value": 75900.0,
                "unrealized_pnl": 858.0,
                "charges_breakdown": {"brokerage": 40.0, "total_taxes": 15.0, "total_charges": 55.0}
            }
        ]

    def tearDown(self):
        db_manager.reset_client()

    def test_01_case1_immediate_liquidation_when_market_open(self):
        """Case 1: When market is open, fires MARKET exit orders immediately."""
        with patch.object(self.service, "is_market_open", return_value=True), \
             patch.object(self.service, "get_holdings", return_value=self.sample_holdings), \
             patch("requests.post") as mock_post:

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"orderId": "MB_MKT_123", "status": "COMPLETE"}
            mock_post.return_value = mock_resp

            result = self.service.liquidate_all_positions(force_market_open=True)

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["case"], "CASE_1_MARKET_OPEN")
            self.assertTrue(result["market_open"])
            self.assertEqual(result["holdings_squared_off"], 2)

            # Check that TITAN (LONG) was sold and INFY (SHORT) was bought back
            orders = result["orders_fired"]
            self.assertEqual(len(orders), 2)

            titan_order = next(o for o in orders if o["symbol"] == "TITAN")
            self.assertEqual(titan_order["action"], "SELL")
            self.assertEqual(titan_order["quantity"], 12)
            self.assertEqual(titan_order["order_type"], "MARKET")

            infy_order = next(o for o in orders if o["symbol"] == "INFY")
            self.assertEqual(infy_order["action"], "BUY")
            self.assertEqual(infy_order["quantity"], 66)
            self.assertEqual(infy_order["order_type"], "MARKET")

            # Check that staged orders were cancelled
            self.mock_staged_orders.update_many.assert_called()

    def test_02_case2_scheduled_liquidation_when_market_closed(self):
        """Case 2: When market is closed, stages emergency liquidation for market open."""
        with patch.object(self.service, "is_market_open", return_value=False), \
             patch.object(self.service, "get_holdings", return_value=self.sample_holdings):

            result = self.service.liquidate_all_positions(force_market_open=False)

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["case"], "CASE_2_MARKET_CLOSED")
            self.assertFalse(result["market_open"])
            self.assertEqual(result["holdings_count"], 2)

            # Check snapshot structure
            snapshot = result["holdings_snapshot"]
            self.assertEqual(len(snapshot), 2)
            self.assertEqual(snapshot[0]["symbol"], "TITAN")
            self.assertEqual(snapshot[0]["exit_action"], "SELL")
            self.assertEqual(snapshot[1]["symbol"], "INFY")
            self.assertEqual(snapshot[1]["exit_action"], "BUY")

            # Verify liquidation state is armed
            status = self.service.get_liquidation_status()
            self.assertTrue(status["armed"])
            self.assertEqual(status["status"], "SCHEDULED_FOR_MARKET_OPEN")

            # Verify staged buy orders were cancelled
            self.mock_staged_orders.update_many.assert_called()

    def test_03_cancel_scheduled_liquidation(self):
        """Verifies cancel_scheduled_liquidation disarms state and restores normal ops."""
        # Arm state first
        self.service._liquidation_state = {
            "armed": True,
            "status": "SCHEDULED_FOR_MARKET_OPEN",
            "holdings_count": 2
        }

        cancel_res = self.service.cancel_scheduled_liquidation()
        self.assertEqual(cancel_res["status"], "SUCCESS")

        status = self.service.get_liquidation_status()
        self.assertFalse(status["armed"])
        self.assertEqual(status["status"], "CANCELLED_BY_USER")

    def test_04_buying_blocked_while_liquidation_armed(self):
        """Verifies place_order rejects new buys while liquidation is armed for market open."""
        with patch.object(self.service, "get_liquidation_status", return_value={
            "armed": True,
            "status": "SCHEDULED_FOR_MARKET_OPEN"
        }):
            res = self.service.place_order(
                symbol="SBIN",
                action="BUY",
                quantity=10,
                price=800.0,
                enforce_duplicate_check=False
            )
            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["code"], "LIQUIDATION_ACTIVE")
            self.assertIn("Emergency liquidation is currently scheduled", res["error"])

    def test_05_market_open_triggers_scheduled_liquidation_in_sync(self):
        """Verifies sync_execution_and_bracket_orders executes liquidation when market opens."""
        # Set armed state
        self.service._liquidation_state = {
            "armed": True,
            "status": "SCHEDULED_FOR_MARKET_OPEN",
            "holdings_count": 2
        }

        with patch.object(self.service, "is_market_open", return_value=True), \
             patch.object(self.service, "liquidate_all_positions") as mock_liq:

            mock_liq.return_value = {"status": "SUCCESS", "case": "CASE_1_MARKET_OPEN"}
            sync_res = self.service.sync_execution_and_bracket_orders()

            self.assertEqual(sync_res["status"], "SUCCESS")
            self.assertEqual(sync_res["action"], "EMERGENCY_LIQUIDATION_EXECUTED")
            mock_liq.assert_called_with(force_market_open=True)

    def test_06_market_closed_waits_in_sync(self):
        """Verifies sync_execution_and_bracket_orders does not place brackets while liquidation is armed off-market."""
        self.service._liquidation_state = {
            "armed": True,
            "status": "SCHEDULED_FOR_MARKET_OPEN",
            "holdings_count": 2
        }

        with patch.object(self.service, "is_market_open", return_value=False):
            sync_res = self.service.sync_execution_and_bracket_orders()
            self.assertEqual(sync_res["status"], "WAITING_FOR_MARKET_OPEN")
            self.assertEqual(sync_res["action"], "LIQUIDATION_ARMED")

    def test_07_api_liquidation_status_endpoint(self):
        """Verifies GET /api/megabull/liquidation-status returns valid payload."""
        resp = self.client.get("/api/megabull/liquidation-status")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.json()
        self.assertEqual(json_data["status"], "SUCCESS")
        self.assertIn("armed", json_data["data"])
        self.assertIn("status", json_data["data"])
        self.assertIn("is_market_open_now", json_data["data"])

    def test_08_api_liquidate_all_endpoint(self):
        """Verifies POST /api/megabull/liquidate-all triggers liquidation."""
        with patch("online.app.megabull_service.liquidate_all_positions") as mock_liq:
            mock_liq.return_value = {
                "status": "SUCCESS",
                "case": "CASE_2_MARKET_CLOSED",
                "market_open": False,
                "holdings_count": 8
            }

            resp = self.client.post("/api/megabull/liquidate-all", json={})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["data"]["case"], "CASE_2_MARKET_CLOSED")

    def test_09_api_cancel_liquidation_endpoint(self):
        """Verifies POST /api/megabull/cancel-liquidation disarms liquidation."""
        with patch("online.app.megabull_service.cancel_scheduled_liquidation") as mock_cancel:
            mock_cancel.return_value = {
                "status": "SUCCESS",
                "message": "Scheduled liquidation cancelled."
            }

            resp = self.client.post("/api/megabull/cancel-liquidation")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")

    def test_10_orders_and_megabull_pages_render_liquidation_elements(self):
        """Verifies Square Off All button, persistent banner, and modal exist in page HTML."""
        # Orders page
        resp_orders = self.client.get("/orders")
        self.assertEqual(resp_orders.status_code, 200)
        orders_html = resp_orders.text
        self.assertIn("Square Off All", orders_html)
        self.assertIn("btnLiquidateAll", orders_html)
        self.assertIn("liquidationBanner", orders_html)
        self.assertIn("liquidateAllModal", orders_html)
        self.assertIn("btnCancelLiquidationBanner", orders_html)

        # MegaBull page
        resp_mb = self.client.get("/megabull")
        self.assertEqual(resp_mb.status_code, 200)
        mb_html = resp_mb.text
        self.assertIn("Square Off All", mb_html)
        self.assertIn("btnLiquidateAll", mb_html)
        self.assertIn("liquidationBanner", mb_html)
        self.assertIn("liquidateAllModal", mb_html)
        self.assertIn("btnCancelLiquidationBanner", mb_html)

    def test_11_multiple_clicks_idempotency_empty_portfolio_or_already_scheduled(self):
        """Verifies idempotency: clicking square off multiple times does not duplicate orders or staging."""
        # 1. When portfolio is empty (0 holdings, 0 positions)
        with patch.object(self.service, "is_market_open", return_value=True), \
             patch.object(self.service, "get_holdings", return_value=[]), \
             patch.object(self.service, "get_positions", return_value=[]):

            res1 = self.service.liquidate_all_positions(force_market_open=True)
            self.assertEqual(res1["status"], "SUCCESS")
            self.assertEqual(res1["case"], "ALREADY_EMPTY")
            self.assertEqual(res1["holdings_squared_off"], 0)
            self.assertEqual(len(res1["orders_fired"]), 0)

        # 2. When market is closed: 1st click schedules, 2nd click recognises already scheduled
        with patch.object(self.service, "is_market_open", return_value=False), \
             patch.object(self.service, "get_holdings", return_value=self.sample_holdings), \
             patch.object(self.service, "get_positions", return_value=[]):

            # First click
            res_first = self.service.liquidate_all_positions(force_market_open=False)
            self.assertEqual(res_first["status"], "SUCCESS")
            self.assertEqual(res_first["case"], "CASE_2_MARKET_CLOSED")
            self.assertEqual(self.service._liquidation_state["status"], "SCHEDULED_FOR_MARKET_OPEN")

            # Second click with same state
            res_second = self.service.liquidate_all_positions(force_market_open=False)
            self.assertEqual(res_second["status"], "SUCCESS")
            self.assertEqual(res_second["case"], "CASE_2_MARKET_CLOSED")
            self.assertIn("already armed", res_second["message"].lower())


if __name__ == "__main__":
    unittest.main()
