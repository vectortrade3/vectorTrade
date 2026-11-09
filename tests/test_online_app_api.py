"""
VectorTrade - Unit & Integration Tests for Online FastAPI Application
Verifies REST API endpoints (/api/health, /api/stats, /api/orders)
and Apple HIG HTML template rendering.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.db import db_manager


class TestOnlineAppAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Verifies GET /api/health returns system and database status."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("database", data)
        self.assertIn("app_name", data)

    def test_02_stats_endpoint(self):
        """Verifies GET /api/stats returns executive summary metrics."""
        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("data", data)
        self.assertIn("armed_orders_count", data["data"])
        self.assertIn("total_orders", data["data"])

    def test_03_orders_endpoint_default(self):
        """Verifies GET /api/orders returns list of orders."""
        resp = self.client.get("/api/orders")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("orders", data)
        self.assertIn("count", data)
        self.assertIsInstance(data["orders"], list)

    def test_04_orders_endpoint_status_filter(self):
        """Verifies GET /api/orders?status=ARMED returns armed orders only."""
        resp = self.client.get("/api/orders?status=ARMED")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        for order in data["orders"]:
            self.assertEqual(order["status"], "ARMED")

    def test_05_orders_endpoint_search(self):
        """Verifies GET /api/orders?search=... performs keyword matching."""
        resp = self.client.get("/api/orders?search=APL")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")

    def test_06_order_by_id_not_found(self):
        """Verifies GET /api/orders/{id} with unknown ID returns 404."""
        resp = self.client.get("/api/orders/NON_EXISTENT_ORDER_ID_XYZ")
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("detail", data)

    def test_07_order_by_id_found(self):
        """Verifies GET /api/orders/{id} returns single order detail."""
        # Find first existing order ID
        orders_resp = self.client.get("/api/orders?limit=1")
        orders = orders_resp.json().get("orders", [])
        if orders:
            order_id = orders[0]["id"]
            resp = self.client.get(f"/api/orders/{order_id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["order"]["id"], order_id)

    def test_08_html_dashboard_page(self):
        """Verifies GET / serves full Apple HIG HTML orders dashboard."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        html = resp.text
        self.assertIn("VectorTrade", html)
        self.assertIn("Live Stock Orders", html)
        self.assertIn("Armed for Execution", html)
        self.assertIn("Buy Trigger", html)
        self.assertIn("Stop Loss", html)
        self.assertIn("Target (1:2 R:R)", html)

    def test_09_html_orders_alias(self):
        """Verifies GET /orders serves identical dashboard."""
        resp = self.client.get("/orders")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("VectorTrade", resp.text)


if __name__ == "__main__":
    unittest.main()
