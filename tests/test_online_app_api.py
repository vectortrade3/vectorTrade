"""
VectorTrade - Unit & Integration Tests for Online FastAPI Application
Verifies Authentication, Session management, REST API endpoints (/api/health, /api/stats, /api/orders),
Overview dashboard, and the dedicated MongoDB Synced Orders Page.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from online.app import app
from online.config import settings
from online.db import db_manager


class TestOnlineAppAPI(unittest.TestCase):

    def setUp(self):
        # Fresh client per test to handle session state independently
        self.client = TestClient(app, follow_redirects=False)

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
        orders_resp = self.client.get("/api/orders?limit=1")
        orders = orders_resp.json().get("orders", [])
        if orders:
            order_id = orders[0]["id"]
            resp = self.client.get(f"/api/orders/{order_id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["order"]["id"], order_id)

    # --------------------------------------------------------------------------
    # Authentication & Session Tests
    # --------------------------------------------------------------------------

    def test_08_unauthenticated_overview_redirects_to_login(self):
        """Verifies GET / without session redirects to /login?next=/."""
        resp = self.client.get("/", follow_redirects=False)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?next=/", resp.headers["location"])

    def test_09_unauthenticated_orders_redirects_to_login(self):
        """Verifies GET /orders without session redirects to /login?next=/orders."""
        resp = self.client.get("/orders", follow_redirects=False)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?next=/orders", resp.headers["location"])

    def test_10_login_page_renders_cleanly(self):
        """Verifies GET /login displays sign in form with root credentials hint."""
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        html = resp.text
        self.assertIn("Sign In", html)
        self.assertIn("VectorTrade Online", html)
        self.assertIn("root", html)

    def test_11_login_post_invalid_credentials_fails(self):
        """Verifies POST /login with bad credentials returns 400 and error banner."""
        resp = self.client.post(
            "/login",
            data="username=wrong_user&password=wrong_pass",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid username or password", resp.text)

    def test_12_login_post_valid_credentials_sets_session(self):
        """Verifies POST /login with root/root succeeds and redirects to orders."""
        resp = self.client.post(
            "/login",
            data="username=root&password=root&next=%2Forders",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False
        )
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/orders")
        self.assertIn("set-cookie", resp.headers)

    def test_13_authenticated_user_can_access_overview_and_orders(self):
        """Verifies authenticated session can view both Overview and Orders pages."""
        # Step 1: Login
        login_resp = self.client.post(
            "/login",
            data="username=root&password=root&next=%2F",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False
        )
        self.assertEqual(login_resp.status_code, 303)

        # Step 2: Access Home Page (Coming Soon)
        home_resp = self.client.get("/")
        self.assertEqual(home_resp.status_code, 200)
        self.assertIn("VectorTrade Quantum", home_resp.text)
        self.assertIn("Coming Soon", home_resp.text)
        self.assertIn("root", home_resp.text)

        # Step 3: Access Dedicated MongoDB Orders Page
        orders_resp = self.client.get("/orders")
        self.assertEqual(orders_resp.status_code, 200)
        self.assertIn("Live Stock Orders", orders_resp.text)
        self.assertIn("Armed for Execution", orders_resp.text)
        self.assertIn("Orders", orders_resp.text)

    def test_14_logout_clears_session(self):
        """Verifies GET /logout clears session and redirects to login."""
        # Login first
        self.client.post(
            "/login",
            data="username=root&password=root",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # Logout
        logout_resp = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(logout_resp.status_code, 303)
        self.assertEqual(logout_resp.headers["location"], "/login")

        # Subsequent request to /orders should now redirect to login again
        follow_resp = self.client.get("/orders", follow_redirects=False)
        self.assertEqual(follow_resp.status_code, 303)
        self.assertIn("/login", follow_resp.headers["location"])

    def test_15_api_auth_json_login(self):
        """Verifies headless JSON client can authenticate via JSON payload."""
        json_resp = self.client.post(
            "/login",
            json={"username": "root", "password": "root", "next": "/orders"},
            headers={"Accept": "application/json"}
        )
        self.assertEqual(json_resp.status_code, 200)
        data = json_resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["user"], "root")


if __name__ == "__main__":
    unittest.main()
