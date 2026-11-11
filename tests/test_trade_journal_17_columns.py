"""
VectorTrade - Unit and Integration Tests for 17-Column Trade Journal
Verifies:
1. Dynamic calculation of all 17 columns from the 4 MongoDB geometry parameters (Stock Name, Entry, Stoploss, Target):
   Stock Name, Buy Date, Strategy Name, TF, QTY, Entry, Stoploss, RISK per share, Sale Date,
   Target 1:2, Target 1:3, AMT. Invested, EXTRA P&L (Charges / dividend), AMT. On Exit,
   P/L Per Share, P/L Per Share %, NET P/L
2. Running trades: Sale Date, AMT. On Exit, P/L Per Share, P/L Per Share %, and NET P/L are blank (None).
3. Completed trades: Sale Date and exit parameters are populated with realized values.
4. Short sell handling (inverse target and SL direction).
5. REST API endpoint GET /api/trades/journal and UI rendering.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.services.orders_service import OrdersService, orders_service


class TestTradeJournal17Columns(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app, follow_redirects=False)

    def test_01_running_long_trade_calculation(self):
        """Verifies 17 columns for running long trade: exit fields must be blank (None)."""
        raw_trade = {
            "stock_name": "EICHERMOT",
            "buy_date": "2026-11-01",
            "strategy": "50 MA Setup",
            "tf": "Daily",
            "quantity": 20,
            "entry": 7219.50,
            "stoploss": 7075.11,
            "target": 7508.28,
            "status": "RUNNING",
            "total_charges": 377.57
        }

        row = OrdersService.calculate_trade_journal_row(raw_trade)

        self.assertEqual(row["stock_name"], "EICHERMOT")
        self.assertEqual(row["buy_date"], "2026-11-01")
        self.assertEqual(row["strategy_name"], "50 MA Setup")
        self.assertEqual(row["tf"], "Daily")
        self.assertEqual(row["qty"], 20)
        self.assertEqual(row["entry"], 7219.50)
        self.assertEqual(row["stoploss"], 7075.11)
        self.assertEqual(row["risk_per_share"], 144.39)
        self.assertEqual(row["target_1_2"], 7508.28)
        self.assertEqual(row["target_1_3"], 7652.67)
        self.assertEqual(row["amt_invested"], 144390.00)
        self.assertEqual(row["extra_pnl"], -377.57)

        # Blank for running trades
        self.assertEqual(row["sale_date"], "")
        self.assertIsNone(row["amt_on_exit"])
        self.assertIsNone(row["pnl_per_share"])
        self.assertIsNone(row["pnl_per_share_pct"])
        self.assertIsNone(row["net_pnl"])
        self.assertFalse(row["is_completed"])

    def test_02_running_short_trade_calculation(self):
        """Verifies short sell trade: stoploss is above entry, targets are below entry."""
        raw_trade = {
            "symbol": "INFY",
            "buy_date": "2026-11-01",
            "strategy": "50 MA Setup",
            "quantity": -66,
            "action": "SELL (SHORT)",
            "entry": 1163.00,
            "stoploss": 1186.26,
            "status": "RUNNING",
            "total_charges": 74.72
        }

        row = OrdersService.calculate_trade_journal_row(raw_trade)

        self.assertEqual(row["stock_name"], "INFY")
        self.assertEqual(row["qty"], 66)
        self.assertEqual(row["entry"], 1163.00)
        self.assertEqual(row["stoploss"], 1186.26)
        self.assertEqual(row["risk_per_share"], 23.26)

        # For Short: Target 1:2 = 1163 - (2 * 23.26) = 1116.48
        self.assertEqual(row["target_1_2"], 1116.48)
        # Target 1:3 = 1163 - (3 * 23.26) = 1093.22
        self.assertEqual(row["target_1_3"], 1093.22)
        self.assertEqual(row["amt_invested"], 76758.00)
        self.assertEqual(row["extra_pnl"], -74.72)

        # Blank for running trades
        self.assertEqual(row["sale_date"], "")
        self.assertIsNone(row["amt_on_exit"])
        self.assertIsNone(row["pnl_per_share"])
        self.assertIsNone(row["pnl_per_share_pct"])
        self.assertIsNone(row["net_pnl"])

    def test_03_completed_trade_calculation(self):
        """Verifies completed trade: exit fields are fully populated with realized values."""
        raw_trade = {
            "symbol": "TVSMOTOR",
            "buy_date": "2026-09-29",
            "strategy": "50 MA Setup",
            "quantity": 15,
            "entry": 4135.10,
            "stoploss": 4036.20,
            "status": "COMPLETED",
            "closed_date": "2026-10-29",
            "exit_price": 4267.60,
            "total_charges": 187.48
        }

        row = OrdersService.calculate_trade_journal_row(raw_trade)

        self.assertEqual(row["stock_name"], "TVSMOTOR")
        self.assertEqual(row["buy_date"], "2026-09-29")
        self.assertEqual(row["qty"], 15)
        self.assertEqual(row["entry"], 4135.10)
        self.assertEqual(row["stoploss"], 4036.20)
        self.assertEqual(row["risk_per_share"], 98.90)
        self.assertEqual(row["target_1_2"], 4332.90)
        self.assertEqual(row["target_1_3"], 4431.80)
        self.assertEqual(row["amt_invested"], 62026.50)
        self.assertEqual(row["extra_pnl"], -187.48)

        # Populated for completed trades
        self.assertEqual(row["sale_date"], "2026-10-29")
        self.assertEqual(row["amt_on_exit"], 64014.00)
        self.assertEqual(row["pnl_per_share"], 132.50)
        self.assertEqual(row["pnl_per_share_pct"], 3.20)
        self.assertEqual(row["net_pnl"], 1800.02)
        self.assertTrue(row["is_completed"])

    def test_04_get_trade_journal_returns_live_and_historical(self):
        """Verifies get_trade_journal aggregates active holdings and completed trades."""
        journal = orders_service.get_trade_journal()
        self.assertGreaterEqual(len(journal), 8)

        running_trades = [r for r in journal if not r["is_completed"]]
        completed_trades = [r for r in journal if r["is_completed"]]

        self.assertGreaterEqual(len(running_trades), 7)
        self.assertGreaterEqual(len(completed_trades), 1)
        self.assertTrue(any(t["stock_name"] == "TVSMOTOR" for t in completed_trades))

    def test_05_api_trades_journal_endpoint(self):
        """Verifies GET /api/trades/journal returns 200 with complete 17 columns."""
        resp = self.client.get("/api/trades/journal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("trades", data)
        self.assertGreaterEqual(data["count"], 8)

        # Check required columns in first trade
        first = data["trades"][0]
        required_keys = [
            "stock_name", "buy_date", "strategy_name", "tf", "qty",
            "entry", "stoploss", "risk_per_share", "sale_date",
            "target_1_2", "target_1_3", "amt_invested", "extra_pnl",
            "amt_on_exit", "pnl_per_share", "pnl_per_share_pct", "net_pnl"
        ]
        for k in required_keys:
            self.assertIn(k, first)

    def test_06_authenticated_orders_page_renders_17_columns(self):
        """Verifies GET /orders renders the 17-column Trade Book table."""
        self.client.post(
            "/login",
            data="username=root&password=root",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        resp = self.client.get("/orders")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Verify all 17 column header names appear in HTML
        headers = [
            "Stock Name", "Buy Date", "Strategy Name", "TF", "QTY",
            "Entry", "Stoploss", "RISK per share", "Sale Date",
            "Target 1:2", "Target 1:3", "AMT. Invested",
            "EXTRA P&amp;L (Charges / dividend)", "AMT. On Exit",
            "P/L Per Share", "P/L Per Share %", "NET P/L"
        ]
        for h in headers:
            self.assertIn(h, html)

    def test_07_csv_generation_and_export_endpoint(self):
        """Verifies CSV generation and GET /api/trades/export/csv returns valid CSV attachment."""
        csv_data = orders_service.generate_trade_journal_csv()
        self.assertIn("Stock Name,Buy Date,Strategy Name,TF,QTY,Entry,Stoploss", csv_data)
        self.assertIn("NET P/L,Lifecycle Status", csv_data)

        resp = self.client.get("/api/trades/export/csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers.get("content-type", ""))
        self.assertIn("attachment; filename=", resp.headers.get("content-disposition", ""))
        self.assertIn("Stock Name,Buy Date", resp.text)

    def test_08_login_screen_has_no_credentials_or_root_placeholder(self):
        """Verifies login page does not display root credentials hint, placeholder, or pre-filled value."""
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Ensure no root placeholder or default credentials hint in HTML
        self.assertNotIn("Enter username (root)", html)
        self.assertNotIn("Enter password (root)", html)
        self.assertNotIn("Default: root / root", html)
        self.assertNotIn('value="root"', html)
        self.assertIn('placeholder="Enter username"', html)
        self.assertIn('placeholder="Enter password"', html)

    def test_09_offline_app_orders_and_csv_export(self):
        """Verifies offline Flask app serves /orders, /api/trades/journal, and /api/trades/export/csv."""
        from offline.app import app as offline_app
        offline_client = offline_app.test_client()

        # 1. Orders HTML page
        resp_page = offline_client.get("/orders")
        self.assertEqual(resp_page.status_code, 200)
        self.assertIn("Trade Book", resp_page.get_data(as_text=True))
        self.assertIn("Export CSV", resp_page.get_data(as_text=True))

        # 2. Trades Journal API
        resp_api = offline_client.get("/api/trades/journal")
        self.assertEqual(resp_api.status_code, 200)
        data = resp_api.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("trades", data)

        # 3. CSV Export API
        resp_csv = offline_client.get("/api/trades/export/csv")
        self.assertEqual(resp_csv.status_code, 200)
        self.assertIn("text/csv", resp_csv.content_type)
        self.assertIn("attachment; filename=", resp_csv.headers.get("Content-Disposition", ""))
        self.assertIn("Stock Name,Buy Date", resp_csv.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
