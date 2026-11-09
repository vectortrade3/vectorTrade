"""
VectorTrade - Integration Tests for MegaBull Web Page and REST API
Verifies session authentication protection, Coming Soon landing page,
and MegaBull portfolio telemetry.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.services.megabull_service import megabull_service


class TestMegaBullAPIAndPages(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app, follow_redirects=False)

    def test_01_unauthenticated_home_redirects_to_login(self):
        """Verifies GET / without session redirects to /login?next=/."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?next=/", resp.headers["location"])

    def test_02_unauthenticated_megabull_redirects_to_login(self):
        """Verifies GET /megabull without session redirects to /login?next=/megabull."""
        resp = self.client.get("/megabull")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?next=/megabull", resp.headers["location"])

    def test_03_authenticated_home_renders_coming_soon(self):
        """Verifies GET / for authenticated user renders Coming Soon landing page."""
        # Authenticate
        self.client.post(
            "/login",
            data="username=root&password=root",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        html = resp.text
        self.assertIn("VectorTrade Quantum", html)
        self.assertIn("Coming Soon", html)
        self.assertIn("MongoDB Orders", html)
        self.assertIn("MegaBull Account", html)
        self.assertIn("/orders", html)
        self.assertIn("/megabull", html)

    def test_04_authenticated_megabull_page_renders_portfolio(self):
        """Verifies GET /megabull for authenticated user renders live portfolio metrics."""
        # Authenticate
        self.client.post(
            "/login",
            data="username=root&password=root",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        resp = self.client.get("/megabull")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        html = resp.text
        self.assertIn("MegaBull", html)
        self.assertIn("Virtual Capital", html)
        self.assertIn("Blocked Margin", html)
        self.assertIn("Free Cash Margin", html)
        self.assertIn("Total Capital We Had", html)
        self.assertIn("Capital Put in Stocks", html)
        self.assertIn("Capital Remaining", html)
        self.assertIn("Running (Bought, Not Sold)", html)
        self.assertIn("Equity Portfolio Holdings", html)
        self.assertIn("Transactions (Ledger)", html)
        self.assertIn("Fee Breakdown", html)

    def test_05_api_megabull_account_endpoint(self):
        """Verifies GET /api/megabull/account returns structured account metadata."""
        resp = self.client.get("/api/megabull/account")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        self.assertIn("virtual_money", data["data"])
        self.assertIn("virtual_money_blocked", data["data"])
        self.assertIn("virtual_money_left", data["data"])

    def test_06_api_megabull_holdings_endpoint(self):
        """Verifies GET /api/megabull/holdings returns enriched holdings."""
        resp = self.client.get("/api/megabull/holdings")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        self.assertIn("count", data)
        self.assertIsInstance(data["data"], list)

    def test_07_api_megabull_positions_endpoint(self):
        """Verifies GET /api/megabull/positions returns positions list."""
        resp = self.client.get("/api/megabull/positions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)

    def test_08_api_megabull_orders_endpoint(self):
        """Verifies GET /api/megabull/orders returns orders dictionary."""
        resp = self.client.get("/api/megabull/orders")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        self.assertIn("open", data["data"])
        self.assertIn("executed", data["data"])

    def test_09_api_megabull_summary_endpoint(self):
        """Verifies GET /api/megabull/summary returns complete portfolio summary."""
        resp = self.client.get("/api/megabull/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        summary = data["data"]
        self.assertIn("account", summary)
        self.assertIn("holdings", summary)
        self.assertIn("total_invested", summary)
        self.assertIn("total_current", summary)
        self.assertIn("net_worth", summary)
        self.assertIn("total_capital_had", summary)
        self.assertIn("capital_put_in_stocks", summary)
        self.assertIn("capital_remaining", summary)
        self.assertIn("valid_orders", summary)

    def test_10_api_megabull_valid_orders_endpoint(self):
        """Verifies GET /api/megabull/orders/valid returns running, pending, and completed orders, excluding cancelled."""
        resp = self.client.get("/api/megabull/orders/valid")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("count", data)
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)

        # Ensure no cancelled or failed orders exist
        for o in data["data"]:
            self.assertNotIn(o.get("status", "").upper(), ["CANCELLED", "CANCELED", "REJECTED", "FAILED"])
            self.assertIn(o.get("category"), ["RUNNING", "PENDING", "COMPLETED"])

    def test_11_api_megabull_transactions_endpoint(self):
        """Verifies GET /api/megabull/transactions returns ledger and fee totals."""
        resp = self.client.get("/api/megabull/transactions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("count", data)
        self.assertIn("data", data)
        self.assertIn("total_brokerage", data)
        self.assertIn("total_taxes", data)
        self.assertIn("gross_realized", data)
        self.assertIn("net_realized", data)
        self.assertIsInstance(data["data"], list)
        self.assertGreaterEqual(len(data["data"]), 2)


if __name__ == "__main__":
    unittest.main()
