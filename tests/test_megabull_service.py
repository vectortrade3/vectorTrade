"""
VectorTrade - Unit Tests for MegaBull Broker Service
Verifies header formatting, margin balance calculations, holdings P&L math,
LTP enrichment, caching, and graceful offline fallback.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.services.megabull_service import MegaBullService


class TestMegaBullService(unittest.TestCase):

    def setUp(self):
        self.service = MegaBullService(
            api_key="test_api_key_123",
            user_id="test_user_456",
            base_url="https://api.megabull.in",
            cache_ttl_seconds=10
        )

    def test_01_headers_contain_api_key_and_user_id(self):
        """Verifies headers strictly follow MegaBull specification."""
        headers = self.service._get_headers()
        self.assertEqual(headers["api-key"], "test_api_key_123")
        self.assertEqual(headers["user-id"], "test_user_456")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_02_account_details_parsing(self):
        """Verifies parsing of /api/user/my into structured margin balances."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "firstName": "Vector",
            "lastName": "Trade",
            "emailId": "trader@vectortrade.in",
            "virtualMoney": 500000.0,
            "virtualMoneyBlocked": 450000.0,
            "virtualMoneyLeft": 50000.0,
            "brokerage": 20.0,
            "premiumType": "FREE"
        }

        with patch("requests.get", return_value=mock_resp):
            acct = self.service.get_account_details()
            self.assertEqual(acct["status"], "SUCCESS")
            self.assertEqual(acct["full_name"], "Vector Trade")
            self.assertEqual(acct["email"], "trader@vectortrade.in")
            self.assertEqual(acct["virtual_money"], 500000.0)
            self.assertEqual(acct["virtual_money_blocked"], 450000.0)
            self.assertEqual(acct["virtual_money_left"], 50000.0)
            # Utilization: 450,000 / 500,000 = 90%
            self.assertEqual(acct["margin_utilization_pct"], 90.0)

    def test_03_holdings_parsing_and_pnl_math(self):
        """Verifies invested value, current market value, and unrealized P&L calculations."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "instrumentToken": "1001",
                "instrumentName": "RELIANCE",
                "qty": 10,
                "lotSize": 1,
                "type": "LONG",
                "priceAvg": 2800.0
            },
            {
                "instrumentToken": "1002",
                "instrumentName": "INFY",
                "qty": 20,
                "lotSize": 1,
                "type": "LONG",
                "priceAvg": 1800.0
            }
        ]

        # Mock LTP responses: RELIANCE = 2900 (+100/sh), INFY = 1750 (-50/sh)
        def mock_ltp(token):
            return 2900.0 if token == "1001" else 1750.0

        with patch("requests.get", return_value=mock_resp):
            with patch.object(self.service, "get_ltp", side_effect=mock_ltp):
                holdings = self.service.get_holdings(enrich_ltp=True)
                self.assertEqual(len(holdings), 2)

                rel = [h for h in holdings if h["symbol"] == "RELIANCE"][0]
                self.assertEqual(rel["invested_value"], 28000.0)
                self.assertEqual(rel["current_value"], 29000.0)
                self.assertEqual(rel["unrealized_pnl"], 1000.0)
                self.assertAlmostEqual(rel["pnl_pct"], 3.57, places=2)
                self.assertTrue(rel["is_profit"])

                infy = [h for h in holdings if h["symbol"] == "INFY"][0]
                self.assertEqual(infy["invested_value"], 36000.0)
                self.assertEqual(infy["current_value"], 35000.0)
                self.assertEqual(infy["unrealized_pnl"], -1000.0)
                self.assertAlmostEqual(infy["pnl_pct"], -2.78, places=2)
                self.assertFalse(infy["is_profit"])

    def test_04_ltp_fetching(self):
        """Verifies /api/marketwatch/ltp/{token} parsing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "instrumentToken": "900609",
            "ltp": 4762.2
        }

        with patch("requests.get", return_value=mock_resp):
            ltp = self.service.get_ltp("900609")
            self.assertEqual(ltp, 4762.2)

    def test_05_caching_within_ttl(self):
        """Verifies repeated calls within TTL hit in-memory cache without repeating HTTP calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"virtualMoney": 500000.0}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            res1 = self.service.get_account_details()
            res2 = self.service.get_account_details()
            # Should have called requests.get exactly once
            mock_get.assert_called_once()
            self.assertEqual(res1["virtual_money"], 500000.0)
            self.assertEqual(res2["virtual_money"], 500000.0)

    def test_06_unconfigured_api_key_graceful_fallback(self):
        """Verifies empty API key returns clean NOT_CONFIGURED state without raising exceptions."""
        unconfigured_svc = MegaBullService(api_key="")
        acct = unconfigured_svc.get_account_details()
        self.assertEqual(acct["status"], "NOT_CONFIGURED")
        self.assertEqual(acct["virtual_money"], 0.0)

        holdings = unconfigured_svc.get_holdings()
        self.assertEqual(holdings, [])

    def test_07_api_error_returns_clean_fallback(self):
        """Verifies HTTP 500 returns error state without unhandled exception."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.get", return_value=mock_resp):
            acct = self.service.get_account_details()
            self.assertEqual(acct["status"], "ERROR")
            self.assertIn("HTTP 500", acct["error"])

    def test_08_portfolio_summary_aggregation(self):
        """Verifies aggregation of net worth, total P&L, and portfolio metrics."""
        mock_acct = {
            "status": "SUCCESS",
            "virtual_money": 500000.0,
            "virtual_money_left": 100000.0,
            "virtual_money_blocked": 400000.0
        }
        mock_holdings = [
            {"symbol": "TCS", "invested_value": 400000.0, "current_value": 420000.0, "unrealized_pnl": 20000.0}
        ]

        with patch.object(self.service, "get_account_details", return_value=mock_acct):
            with patch.object(self.service, "get_holdings", return_value=mock_holdings):
                with patch.object(self.service, "get_positions", return_value=[]):
                    with patch.object(self.service, "get_orders", return_value={"open": [], "executed": []}):
                        summary = self.service.get_portfolio_summary()
                        self.assertEqual(summary["total_invested"], 400000.0)
                        self.assertEqual(summary["total_current"], 420000.0)
                        self.assertEqual(summary["total_pnl"], 20000.0)
                        self.assertEqual(summary["total_pnl_pct"], 5.0)
                        # Net Worth = 100,000 cash + 420,000 holdings = 520,000
                        self.assertEqual(summary["net_worth"], 520000.0)
                        self.assertEqual(summary["holdings_count"], 1)


if __name__ == "__main__":
    unittest.main()
