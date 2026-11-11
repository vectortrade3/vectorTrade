"""
VectorTrade - Unit Tests for Live Market & Order Synchronization Worker
Verifies:
1. Order Lifecycle Stage transitions:
   - Stage 1: WILL_BE_PUT_AMO (off-market staging, submission upon market open)
   - Stage 2: PENDING (order submitted, limit price awaiting trigger, fill detection)
   - Stage 3: RUNNING_BOUGHT / RUNNING_SHORT_SELLED (LTP pricing, bracket placement, daily re-arm)
   - Stage 4: COMPLETED_BOUGHT_TO_SOLD / COMPLETED_SHORT_TO_COVERED (exit detection, cancel opposing brackets)
2. Live quote fetching & caching in MongoDB Atlas
3. Portfolio live state synchronization
4. FastAPI endpoints: /api/sync/status, /api/sync/trigger, /api/sync/live-feed, /api/prices/live
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.services.live_sync_worker import live_sync_worker, LiveSyncWorker, OrderLifecycleStage
from online.services.megabull_service import MegaBullService, megabull_service


class TestLiveSyncWorker(unittest.TestCase):

    def setUp(self):
        self.worker = LiveSyncWorker(interval_seconds=5)
        self.client = TestClient(app, follow_redirects=False)
        self.patch_liq = patch.object(megabull_service, "get_liquidation_status", return_value={"armed": False, "status": "IDLE"})
        self.patch_liq.start()

    def tearDown(self):
        self.patch_liq.stop()

    def test_01_stage_1_amo_retained_when_market_closed(self):
        """Verifies staged off-market orders remain in WILL_BE_PUT_AMO stage when market is closed."""
        with patch.object(megabull_service, "is_market_open", return_value=False), \
             patch("online.services.live_sync_worker.db_manager") as mock_db, \
             patch.object(megabull_service, "get_holdings", return_value=[]), \
             patch.object(megabull_service, "get_orders", return_value={"open": [], "executed": []}), \
             patch.object(megabull_service, "get_historical_completed_trades", return_value=[]):

            mock_col = MagicMock()
            mock_col.find.return_value = [
                {"_id": "1", "symbol": "RELIANCE", "status": "STAGED_OFFMARKET", "action": "BUY", "quantity": 10, "price": 1300.0}
            ]
            mock_db.staged_orders = mock_col
            mock_db.signals_watchlist = MagicMock()
            mock_db.signals_watchlist.find.return_value = []

            res = self.worker.sync_orders_lifecycle()

            self.assertEqual(res["status"], "SUCCESS")
            self.assertFalse(res["market_open"])
            self.assertEqual(res["amo_fired_count"], 0)
            self.assertEqual(res["orders_by_stage"][OrderLifecycleStage.WILL_BE_PUT_AMO], 1)

    def test_02_stage_1_amo_submitted_when_market_opens(self):
        """Verifies staged off-market orders are submitted to MegaBull broker as soon as market opens."""
        with patch.object(megabull_service, "is_market_open", return_value=True), \
             patch("online.services.live_sync_worker.db_manager") as mock_db, \
             patch.object(megabull_service, "get_holdings", return_value=[]), \
             patch.object(megabull_service, "get_orders", return_value={"open": [], "executed": []}), \
             patch.object(megabull_service, "get_historical_completed_trades", return_value=[]), \
             patch.object(megabull_service, "place_order") as mock_place_order:

            mock_place_order.return_value = {"status": "SUCCESS", "order_id": "ORD_123", "case": "PLACED"}
            mock_col = MagicMock()
            def mock_find(query=None, *args, **kwargs):
                if query and "STAGED_OFFMARKET" in query.get("status", {}).get("$in", []):
                    return [{"_id": "1", "symbol": "INFY", "status": "STAGED_OFFMARKET", "action": "BUY", "quantity": 5, "price": 1800.0}]
                return []

            mock_col.find.side_effect = mock_find
            mock_db.staged_orders = mock_col
            mock_db.signals_watchlist = MagicMock()
            mock_db.signals_watchlist.find.return_value = []

            res = self.worker.sync_orders_lifecycle()

            self.assertEqual(res["status"], "SUCCESS")
            self.assertTrue(res["market_open"])
            self.assertEqual(res["amo_fired_count"], 1)
            mock_place_order.assert_called_once()
            mock_col.update_one.assert_called_once()

    def test_03_stage_2_pending_order_transition_on_fill(self):
        """Verifies open pending broker order transitions to RUNNING when holding appears."""
        with patch.object(megabull_service, "is_market_open", return_value=True), \
             patch("online.services.live_sync_worker.db_manager") as mock_db, \
             patch.object(megabull_service, "get_holdings") as mock_holdings, \
             patch.object(megabull_service, "get_orders") as mock_orders, \
             patch.object(megabull_service, "get_historical_completed_trades", return_value=[]), \
             patch.object(megabull_service, "place_bracket_orders") as mock_brackets:

            # Order is in open_orders AND present in active holdings (fill detected)
            mock_holdings.return_value = [
                {"symbol": "TCS", "qty": 10, "avg_buy_price": 3800.0, "ltp": 3820.0, "instrument_token": "2953213"}
            ]
            mock_orders.return_value = {
                "open": [{"orderId": "PO_1", "instrumentName": "TCS", "price": 3800.0, "qty": 10}],
                "executed": []
            }
            mock_brackets.return_value = {"target_order": "TGT_1", "sl_order": "SL_1"}
            mock_col = MagicMock()
            mock_col.find.return_value = []
            mock_db.staged_orders = mock_col
            mock_db.signals_watchlist = MagicMock()
            mock_db.signals_watchlist.find.return_value = []

            res = self.worker.sync_orders_lifecycle()

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["pending_to_running_count"], 1)
            self.assertEqual(res["orders_by_stage"][OrderLifecycleStage.RUNNING_BOUGHT], 1)

    def test_04_stage_3_bracket_order_placement_and_daily_rearm(self):
        """Verifies target and stoploss bracket orders are placed for active holdings lacking brackets."""
        with patch.object(megabull_service, "is_market_open", return_value=True), \
             patch("online.services.live_sync_worker.db_manager") as mock_db, \
             patch.object(megabull_service, "get_holdings") as mock_holdings, \
             patch.object(megabull_service, "get_orders") as mock_orders, \
             patch.object(megabull_service, "get_historical_completed_trades", return_value=[]), \
             patch.object(megabull_service, "place_bracket_orders") as mock_brackets:

            mock_holdings.return_value = [
                {"symbol": "HDFCBANK", "qty": 20, "avg_buy_price": 1600.0, "ltp": 1610.0, "instrument_token": "341249"}
            ]
            # No open brackets for HDFCBANK
            mock_orders.return_value = {"open": [], "executed": []}
            mock_brackets.return_value = {"target_order": "TGT_HDFC", "sl_order": "SL_HDFC"}

            mock_col = MagicMock()
            mock_col.find.return_value = []
            mock_db.staged_orders = mock_col
            mock_db.signals_watchlist = MagicMock()
            mock_db.signals_watchlist.find.return_value = [
                {"symbol": "HDFCBANK", "target_price": 1650.0, "sl_price": 1580.0}
            ]

            res = self.worker.sync_orders_lifecycle()

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["brackets_placed_count"], 1)
            mock_brackets.assert_called_once()
            args, kwargs = mock_brackets.call_args
            self.assertEqual(args[0], "HDFCBANK")
            self.assertEqual(args[2], 1650.0) # Target price
            self.assertEqual(args[3], 1580.0) # SL price

    def test_05_stage_4_completed_trade_detection_and_bracket_cancel(self):
        """Verifies that when an active holding is closed, opposing brackets are cancelled and trade is COMPLETED."""
        with patch.object(megabull_service, "is_market_open", return_value=True), \
             patch("online.services.live_sync_worker.db_manager") as mock_db, \
             patch.object(megabull_service, "get_holdings", return_value=[]), \
             patch.object(megabull_service, "get_orders", return_value={"open": [], "executed": []}), \
             patch.object(megabull_service, "get_historical_completed_trades", return_value=[]), \
             patch.object(megabull_service, "cancel_open_orders") as mock_cancel:

            # Document was previously ACTIVE in MongoDB, but no longer in holdings
            mock_col = MagicMock()
            mock_col.find.return_value = [
                {"_id": "active_doc_1", "symbol": "SBIN", "status": "ACTIVE", "action": "BUY"}
            ]
            mock_db.staged_orders = mock_col
            mock_db.signals_watchlist = MagicMock()
            mock_db.signals_watchlist.find.return_value = []

            res = self.worker.sync_orders_lifecycle()

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["completed_count"], 1)
            mock_cancel.assert_called_once_with("SBIN")
            mock_col.update_one.assert_called_once()
            update_call = mock_col.update_one.call_args[0]
            self.assertEqual(update_call[1]["$set"]["status"], "COMPLETED")
            self.assertEqual(update_call[1]["$set"]["lifecycle_stage"], OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD)

    def test_06_live_quotes_fetching_and_cache(self):
        """Verifies price fetching gathers symbols and populates price cache."""
        with patch.object(megabull_service, "fetch_live_quotes") as mock_fetch:
            mock_fetch.return_value = {
                "RELIANCE": {"symbol": "RELIANCE", "ltp": 1324.4, "market_status": "CLOSED"},
                "INFY": {"symbol": "INFY", "ltp": 1154.1, "market_status": "CLOSED"}
            }

            res = self.worker.fetch_and_update_prices(["RELIANCE", "INFY"])

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["updated_count"], 2)
            self.assertIn("RELIANCE", self.worker.price_cache)
            self.assertEqual(self.worker.price_cache["RELIANCE"]["ltp"], 1324.4)

    def test_07_portfolio_live_state_sync_to_mongodb(self):
        """Verifies portfolio state document is synced to MongoDB Atlas system_state collection."""
        with patch.object(megabull_service, "get_portfolio_summary") as mock_summary, \
             patch("online.services.live_sync_worker.db_manager") as mock_db:

            mock_summary.return_value = {
                "total_capital_had": 501800.0,
                "capital_put_in_stocks": 474278.8,
                "capital_remaining": 27521.2,
                "total_invested": 474278.8,
                "total_current": 475000.0,
                "net_worth": 502521.2,
                "total_pnl": 721.2,
                "total_pnl_pct": 0.15,
                "total_charges_paid": 187.48,
                "running_orders_count": 8,
                "pending_orders_count": 8,
                "amo_orders_count": 8,
                "completed_orders_count": 1,
                "orders_by_stage": {"RUNNING_BOUGHT": 7, "RUNNING_SHORT_SELLED": 1}
            }
            mock_sys = MagicMock()
            mock_db.system_state = mock_sys

            state = self.worker.sync_portfolio_state()

            self.assertEqual(state["_id"], "portfolio_live_state")
            self.assertEqual(state["net_worth"], 502521.2)
            self.assertEqual(state["running_orders_count"], 8)
            mock_sys.update_one.assert_called_once()

    def test_08_api_sync_status(self):
        """Verifies GET /api/sync/status endpoint returns live operational metadata."""
        with patch.object(live_sync_worker, "get_status") as mock_status:
            mock_status.return_value = {
                "is_running": True,
                "interval_seconds": 5,
                "last_sync_time": "2026-11-10 19:50:00",
                "sync_count": 42,
                "cached_symbols_count": 15,
                "monitored_symbols_count": 15,
                "last_error": None,
                "last_summary": {"status": "SUCCESS"},
                "market_open": False
            }

            resp = self.client.get("/api/sync/status")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["is_running"])
            self.assertEqual(data["sync_count"], 42)

    def test_09_api_sync_live_feed(self):
        """Verifies GET /api/sync/live-feed returns real-time bundle with prices, stages, and metrics."""
        with patch.object(megabull_service, "get_portfolio_summary") as mock_summary:
            mock_summary.return_value = {
                "valid_orders": [
                    {"order_id": "1", "symbol": "TCS", "lifecycle_stage": "RUNNING_BOUGHT", "ltp": 3800.0}
                ],
                "orders_by_stage": {"RUNNING_BOUGHT": 1},
                "running_orders_count": 1,
                "pending_orders_count": 0,
                "amo_orders_count": 0,
                "completed_orders_count": 0,
                "net_worth": 500000.0,
                "total_invested": 38000.0,
                "total_current": 38500.0,
                "total_pnl": 500.0,
                "total_pnl_pct": 1.31
            }

            resp = self.client.get("/api/sync/live-feed")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(len(data["valid_orders"]), 1)
            self.assertEqual(data["valid_orders"][0]["lifecycle_stage"], "RUNNING_BOUGHT")

    def test_10_api_prices_live(self):
        """Verifies GET /api/prices/live returns real-time quote map."""
        with patch.object(megabull_service, "fetch_live_quotes") as mock_quotes:
            mock_quotes.return_value = {
                "INFY": {"symbol": "INFY", "ltp": 1750.5, "market_status": "CLOSED"}
            }

            resp = self.client.get("/api/prices/live?symbols=INFY")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertIn("INFY", data["quotes"])
            self.assertEqual(data["quotes"]["INFY"]["ltp"], 1750.5)


if __name__ == "__main__":
    unittest.main()
