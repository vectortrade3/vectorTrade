"""
VectorTrade - Unit & Integration Tests for Strategy Selection & Dynamic Adaptability
Tests:
1. Dynamic discovery and default enabled state of modular strategies
2. Toggling strategies: enabling/disabling persists and updates active registry
3. Query matching by ID, display name, and aliases (e.g. "50 MA Setup", "Global Regime")
4. Offline Daily Analyzer skips analyzing/qualifying stocks for deselected strategies
5. All strategies deselected produces 0 trades without crashing
6. Prevention of trade execution: orders for deselected strategies are blocked
7. Extensibility: future strategies placed in strategies/ are automatically discovered and toggleable
8. Offline and Online REST APIs: GET /api/strategies, POST /api/strategies/toggle, POST /api/strategies/select
9. UI Template rendering: Dashboard and Daily Analysis strategy selection elements present in HTML
"""

import os
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from fastapi.testclient import TestClient

from strategies.manager import StrategyManager, STRATEGIES_DIR
from offline.app import app as offline_flask_app
from online.app import app as online_fastapi_app
from online.services.megabull_service import megabull_service
from offline.daily_analyzer import synthesize_trade_opportunity, scan_all_constituents


class TestStrategySelection(unittest.TestCase):
    """Test suite for strategy selection, deselection, and future strategy adaptability."""

    def setUp(self):
        self.offline_client = offline_flask_app.test_client()
        self.online_client = TestClient(online_fastapi_app)
        # Ensure strategies are freshly initialized with both active
        StrategyManager.initialize(force_reload=True)
        StrategyManager.set_strategy_enabled("ma50_setup", True)
        StrategyManager.set_strategy_enabled("global_regime", True)

    def tearDown(self):
        # Reset default state: both enabled
        StrategyManager.set_strategy_enabled("ma50_setup", True)
        StrategyManager.set_strategy_enabled("global_regime", True)
        StrategyManager.initialize(force_reload=True)

    def test_01_discovery_and_default_enabled(self):
        """Verifies both ma50_setup and global_regime are discovered and active by default."""
        all_strats = StrategyManager.get_all_strategies()
        active_strats = StrategyManager.get_active_strategies()

        self.assertIn("ma50_setup", all_strats)
        self.assertIn("global_regime", all_strats)
        self.assertIn("ma50_setup", active_strats)
        self.assertIn("global_regime", active_strats)

        self.assertTrue(StrategyManager.is_strategy_enabled("ma50_setup"))
        self.assertTrue(StrategyManager.is_strategy_enabled("global_regime"))
        self.assertTrue(StrategyManager.is_strategy_enabled("50 MA Setup"))
        self.assertTrue(StrategyManager.is_strategy_enabled("Global Regime"))

    def test_02_toggle_strategy_selection(self):
        """Verifies disabling a strategy removes it from active registry and sets enabled=False."""
        # Deselect Global Regime
        success = StrategyManager.set_strategy_enabled("global_regime", False)
        self.assertTrue(success)
        self.assertFalse(StrategyManager.is_strategy_enabled("global_regime"))
        self.assertFalse(StrategyManager.is_strategy_enabled("Global Regime"))

        active_strats = StrategyManager.get_active_strategies()
        self.assertNotIn("global_regime", active_strats)
        self.assertIn("ma50_setup", active_strats)

        # Re-enable Global Regime
        StrategyManager.set_strategy_enabled("global_regime", True)
        self.assertTrue(StrategyManager.is_strategy_enabled("global_regime"))
        self.assertIn("global_regime", StrategyManager.get_active_strategies())

    def test_03_alias_and_combo_resolution(self):
        """Verifies flexible name matching and dual strategy combo resolution."""
        # When both enabled, combo is enabled
        self.assertTrue(StrategyManager.is_strategy_enabled("50 MA + Global Regime"))

        # When one is disabled, combo becomes disabled
        StrategyManager.set_strategy_enabled("global_regime", False)
        self.assertFalse(StrategyManager.is_strategy_enabled("50 MA + Global Regime"))
        self.assertTrue(StrategyManager.is_strategy_enabled("50 MA Setup"))
        self.assertFalse(StrategyManager.is_strategy_enabled("Global Regime"))

        # Deselect 50MA as well
        StrategyManager.set_strategy_enabled("ma50_setup", False)
        self.assertFalse(StrategyManager.is_strategy_enabled("50 MA Setup"))
        self.assertFalse(StrategyManager.is_strategy_enabled("50 MA + Global Regime"))

    def test_04_daily_analyzer_skips_deselected_strategy_evaluation(self):
        """Verifies evaluate_single_setup_row does NOT qualify trades for deselected strategies."""
        mock_row = {
            "symbol": "TCS",
            "close": 3500.0,
            "open": 3480.0,
            "high": 3520.0,
            "low": 3470.0,
            "ma50": 3465.0,
            "bb_lower": 3465.0,
            "base_signal": True
        }
        mock_meta = {"TCS": {"name": "Tata Consultancy Services", "sector": "IT"}}

        # Case A: Both strategies enabled -> qualifies
        StrategyManager.set_strategy_enabled("ma50_setup", True)
        StrategyManager.set_strategy_enabled("global_regime", True)
        res_both = synthesize_trade_opportunity(
            mock_row, pred_r=1.2, pred_gain_pts=180.0, pred_drop_pts=30.0, stock_meta=mock_meta
        )
        self.assertIsNotNone(res_both)
        self.assertEqual(res_both["symbol"], "TCS")

        # Case B: Deselect 50MA Setup -> does NOT qualify as 50MA
        StrategyManager.set_strategy_enabled("ma50_setup", False)
        StrategyManager.set_strategy_enabled("global_regime", True)
        # Even with strong base signal, if 50MA is deselected and global criteria fail:
        res_no_ma = synthesize_trade_opportunity(
            mock_row, pred_r=1.5, pred_gain_pts=5.0, pred_drop_pts=50.0, stock_meta=mock_meta
        )
        self.assertIsNone(res_no_ma)

        # Case C: Deselect Global Regime -> does NOT qualify as Global Regime
        StrategyManager.set_strategy_enabled("ma50_setup", False)
        StrategyManager.set_strategy_enabled("global_regime", False)
        res_none = synthesize_trade_opportunity(
            mock_row, pred_r=1.5, pred_gain_pts=200.0, pred_drop_pts=20.0, stock_meta=mock_meta
        )
        self.assertIsNone(res_none)

    def test_05_all_strategies_deselected_returns_zero_trades(self):
        """Verifies that when all strategies are deselected, scan_all_constituents returns empty list."""
        StrategyManager.set_strategy_enabled("ma50_setup", False)
        StrategyManager.set_strategy_enabled("global_regime", False)

        models_dict = {
            "xgb_model": MagicMock(),
            "xgb_scaler": MagicMock(),
            "xgb_imputer": MagicMock(),
            "xgb_features": ["f1"],
            "gain_model": MagicMock(),
            "drop_model": MagicMock(),
            "glob_scaler": MagicMock(),
            "glob_imputer": MagicMock(),
            "glob_features": ["f1"],
            "stock_meta": {}
        }

        trades = scan_all_constituents(models_dict, recent_sessions_count=3)
        self.assertEqual(trades, [])

    def test_06_future_strategy_adaptability(self):
        """
        Verifies the system is fully adaptive:
        Creating a new strategy directory dynamically appears in StrategyManager,
        is toggleable, and participates in registry.
        """
        mock_strat_dir = STRATEGIES_DIR / "temp_breakout_future"
        mock_strat_dir.mkdir(exist_ok=True)
        cfg_file = mock_strat_dir / "config.json"

        mock_config = {
            "strategy_id": "temp_breakout_future",
            "name": "Breakout Momentum Strategy",
            "display_name": "20-Day Breakout Momentum",
            "enabled": True,
            "type": "breakout_engine",
            "version": "1.0.0",
            "description": "Adaptive future strategy test."
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(mock_config, f, indent=2)

        try:
            StrategyManager.initialize(force_reload=True)
            self.assertIn("temp_breakout_future", StrategyManager.get_all_strategies())
            self.assertTrue(StrategyManager.is_strategy_enabled("temp_breakout_future"))
            self.assertTrue(StrategyManager.is_strategy_enabled("Breakout Momentum Strategy"))

            # Toggle future strategy off
            StrategyManager.set_strategy_enabled("temp_breakout_future", False)
            self.assertFalse(StrategyManager.is_strategy_enabled("temp_breakout_future"))
            self.assertNotIn("temp_breakout_future", StrategyManager.get_active_strategies())

            # Verify config.json was updated with enabled: false
            with open(cfg_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertFalse(saved["enabled"])
        finally:
            if mock_strat_dir.exists():
                shutil.rmtree(mock_strat_dir)
            StrategyManager.initialize(force_reload=True)

    def test_07_trade_execution_blocked_for_deselected_strategy(self):
        """Verifies place_order rejects orders associated with a deselected strategy."""
        # Deselect Global Regime
        StrategyManager.set_strategy_enabled("global_regime", False)

        with patch.object(megabull_service, "get_liquidation_status", return_value={"armed": False}), \
             patch.object(megabull_service, "is_symbol_in_portfolio", return_value={"exists": False}):

            res = megabull_service.place_order(
                symbol="RELIANCE",
                action="BUY",
                quantity=10,
                price=1300.0,
                strategy="Global Regime"
            )
            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["code"], "STRATEGY_DISABLED")
            self.assertIn("deselected", res["error"].lower())

    def test_08_offline_flask_strategy_endpoints(self):
        """Verifies GET /api/strategies, POST /api/strategies/toggle, and POST /api/strategies/select on offline app."""
        # 1. GET /api/strategies
        resp_get = self.offline_client.get("/api/strategies")
        self.assertEqual(resp_get.status_code, 200)
        data_get = resp_get.get_json()
        self.assertEqual(data_get["status"], "SUCCESS")
        self.assertGreaterEqual(data_get["total_count"], 2)
        strat_ids = [s["strategy_id"] for s in data_get["strategies"]]
        self.assertIn("ma50_setup", strat_ids)
        self.assertIn("global_regime", strat_ids)

        # 2. POST /api/strategies/toggle
        resp_toggle = self.offline_client.post(
            "/api/strategies/toggle",
            json={"strategy_id": "global_regime", "enabled": False}
        )
        self.assertEqual(resp_toggle.status_code, 200)
        data_toggle = resp_toggle.get_json()
        self.assertEqual(data_toggle["status"], "SUCCESS")
        self.assertFalse(data_toggle["enabled"])
        self.assertFalse(StrategyManager.is_strategy_enabled("global_regime"))

        # 3. POST /api/strategies/select (Batch select)
        resp_batch = self.offline_client.post(
            "/api/strategies/select",
            json={"selected_strategy_ids": ["ma50_setup", "global_regime"]}
        )
        self.assertEqual(resp_batch.status_code, 200)
        data_batch = resp_batch.get_json()
        self.assertEqual(data_batch["status"], "SUCCESS")
        self.assertTrue(StrategyManager.is_strategy_enabled("ma50_setup"))
        self.assertTrue(StrategyManager.is_strategy_enabled("global_regime"))

    def test_09_online_fastapi_strategy_endpoints(self):
        """Verifies GET /api/strategies and POST /api/strategies/toggle on online FastAPI app."""
        resp_get = self.online_client.get("/api/strategies")
        self.assertEqual(resp_get.status_code, 200)
        data = resp_get.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertGreaterEqual(data["total_count"], 2)

        resp_toggle = self.online_client.post(
            "/api/strategies/toggle",
            json={"strategy_id": "ma50_setup", "enabled": False}
        )
        self.assertEqual(resp_toggle.status_code, 200)
        data_toggle = resp_toggle.json()
        self.assertEqual(data_toggle["status"], "SUCCESS")
        self.assertFalse(data_toggle["enabled"])
        self.assertFalse(StrategyManager.is_strategy_enabled("ma50_setup"))

    def test_10_html_templates_render_strategy_elements(self):
        """Verifies Dashboard and Daily Analysis views in index.html contain strategy selection components."""
        resp = self.offline_client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Dashboard strategy panel
        self.assertIn("dashboardStrategiesContainer", html)
        self.assertIn("Strategy Model Selection", html)
        self.assertIn("dashActiveStrategiesCountBadge", html)

        # Daily Analysis strategy selector bar
        self.assertIn("analysisStrategiesSelectorBar", html)
        self.assertIn("analysisAllDisabledWarning", html)
        self.assertIn("analysisActiveStrategiesSummaryBadge", html)


if __name__ == "__main__":
    unittest.main()
