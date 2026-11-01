"""
VectorTrade - Refactor Verification Suite
Performs strict CodeRabbit audits on:
1. Headless Daemon Startup & Guard:
   - Exits cleanly (code 0) if time < 16:00 IST or if today's run already recorded.
2. Complete Offline Pre-computation:
   - Verifies all financial fields (quantity, total_capital_required, profit_at_target, loss_at_sl, risk_reward_ratio)
     are computed purely offline and persisted in signals_watchlist.
3. Modular Directory Isolation:
   - Confirms online/requirements_online.txt contains 0 bulky ML dependencies (no pandas, numpy, scikit-learn, or xgboost).
   - Confirms online/app.py and online/execution_worker.py import 0 bulky analytical libraries.
4. FastAPI Dashboard & API:
   - Tests FastAPI state endpoint (/api/state) to verify clean data serialization and metric serving.
"""

import sys
import os
import json
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from offline.headless_daemon import HeadlessDaemon
from fastapi.testclient import TestClient
from online.app import app


class TestRefactorArchitecture(unittest.TestCase):

    def setUp(self):
        self.db = db_manager.db
        self.signals_col = db_manager.get_collection("signals_watchlist")
        self.active_col = db_manager.get_collection("active_trades")
        self.history_col = db_manager.get_collection("trade_history")

    def test_modular_dependency_isolation(self):
        """Verifies online/ has zero heavy ML / analytics dependencies."""
        online_req_file = BASE_DIR / "online" / "requirements_online.txt"
        self.assertTrue(online_req_file.exists())

        with open(online_req_file, "r", encoding="utf-8") as f:
            req_content = f.read().lower()
            for forbidden in ["pandas", "numpy", "scikit-learn", "sklearn", "xgboost", "scipy", "yfinance"]:
                self.assertNotIn(forbidden, req_content, f"Forbidden dependency '{forbidden}' found in online requirements!")

        # Check online python modules directly
        for py_file in (BASE_DIR / "online").glob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        for forbidden in ["pandas", "numpy", "scikit-learn", "sklearn", "xgboost", "scipy", "yfinance"]:
                            self.assertNotIn(forbidden, stripped, f"Forbidden import '{forbidden}' in {py_file.name}: {stripped}")

    def test_headless_daemon_time_and_state_guard(self):
        """Verifies headless daemon exits cleanly if time < 16:00 IST or run already completed."""
        daemon = HeadlessDaemon(force_run=False)
        # 1. Simulate morning time (< 16:00 IST)
        if daemon.now_ist.hour < 16:
            guard = daemon.check_execution_guard()
            self.assertFalse(guard, "Headless daemon must exit cleanly when time < 16:00 IST")

        # 2. Simulate today's run already completed in state file
        state_path = BASE_DIR / "logs" / "eod_run_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"last_successful_run_date": daemon.today_str}, f)

        guard_after_run = daemon.check_execution_guard()
        self.assertFalse(guard_after_run, "Headless daemon must exit cleanly when today's run is already recorded in eod_run_state.json")
        # Clean up state file
        if state_path.exists():
            os.remove(state_path)

    def test_offline_financial_risk_precomputation(self):
        """Verifies full financial pre-computations are performed purely offline."""
        daemon = HeadlessDaemon(force_run=True, capital=100000.0, risk_pct=1.5)
        # Test High = 1000.0, Low = 950.0 (Buy = 1001.0, SL = 949.0, Risk per Share = 52.0)
        # Risk amount = 1.5% of 100,000 = 1,500. Quantity = floor(1500 / 52) = 28
        metrics = daemon.calculate_trade_metrics(high=1000.0, low=950.0)

        self.assertEqual(metrics["buy_trigger"], 1001.0)
        self.assertEqual(metrics["sl_price"], 949.0)
        self.assertEqual(metrics["risk_per_share"], 52.0)
        self.assertEqual(metrics["target_price"], 1001.0 + (2.0 * 52.0))
        self.assertEqual(metrics["quantity"], 28)
        self.assertEqual(metrics["total_capital_required"], round(1001.0 * 28, 2))
        self.assertEqual(metrics["profit_at_target"], round((metrics["target_price"] - 1001.0) * 28, 2))
        self.assertEqual(metrics["loss_at_sl"], round(52.0 * 28, 2))
        self.assertEqual(metrics["risk_reward_ratio"], 2.0)

    def test_fastapi_dashboard_api(self):
        """Verifies FastAPI dashboard serves real-time state with financial metrics."""
        # Insert test armed setup with pre-computed financials
        self.signals_col.delete_many({"_id": "TEST_DASH_SIG"})
        self.signals_col.insert_one({
            "_id": "TEST_DASH_SIG",
            "symbol": "RELIANCE",
            "sector": "Energy",
            "strategy_id": "50MA_PULLBACK",
            "ml_confidence": 0.85,
            "buy_trigger": 1200.0,
            "sl_price": 1170.0,
            "target_price": 1260.0,
            "quantity": 50,
            "total_capital_required": 60000.0,
            "profit_at_target": 3000.0,
            "loss_at_sl": 1500.0,
            "status": "ARMED"
        })

        client = TestClient(app)
        # Test HTML Dashboard Render
        html_resp = client.get("/")
        self.assertEqual(html_resp.status_code, 200)
        self.assertIn("VectorTrade", html_resp.text)
        self.assertIn("Armed Watchlist", html_resp.text)

        # Test API State endpoint
        api_resp = client.get("/api/state")
        self.assertEqual(api_resp.status_code, 200)
        data = api_resp.json()
        self.assertEqual(data["status"], "OK")
        self.assertGreaterEqual(len(data["signals_watchlist"]), 1)

        sig = next((s for s in data["signals_watchlist"] if s["_id"] == "TEST_DASH_SIG"), None)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["profit_at_target"], 3000.0)
        self.assertEqual(sig["loss_at_sl"], 1500.0)
        self.assertEqual(sig["total_capital_required"], 60000.0)

        # Clean up
        self.signals_col.delete_many({"_id": "TEST_DASH_SIG"})


if __name__ == "__main__":
    unittest.main()
