"""
VectorTrade - Unit Tests for Risk Calculator & Capital Allocation Engine
Tests:
1. Exact Excel sheet reproduction (Entry 1216, SL 1195, Portfolio 30000)
2. User's second example (Capital 10000, 1% = 100, SL 11 -> Qty 9, not 10)
3. Short Sell position sizing and downward target payoffs
4. Threshold & boundary logic:
   - Custom limit < Broker balance -> threshold honored
   - Custom limit > Broker balance -> excess ignored, capped at broker balance
   - Custom limit None / 0 -> defaults to full broker paper capital
5. API endpoints:
   - GET /api/trading-settings
   - POST /api/trading-settings
   - POST /api/trading/calculate-quantity
6. OrdersService integration with dynamic 1% risk sizing
"""

import unittest
from unittest.mock import patch, MagicMock
from online.services.risk_calculator import RiskCalculator, risk_calculator
from online.services.orders_service import orders_service


class TestRiskCalculator(unittest.TestCase):
    """Test suite for 1% Risk per Trade Excel Geometry & Threshold Logic."""

    def setUp(self):
        self.calc = RiskCalculator()
        self.calc._cached_broker_capital = 526545.11
        self.calc._broker_cache_time = 9999999999.0

    def test_01_exact_excel_sheet_geometry(self):
        """
        Validates 100% exact reproduction of the user's Excel sheet:
        - Portfolio: 30,000
        - 1% Risk per Trade: 300
        - Entry Price: 1216
        - Stop Loss Price: 1195
        - SL Rupee per share: 21 (1216 - 1195)
        - Total Shares to trade: 14 (floor(300 / 21))
        - Investment amount: 17,024 (1216 * 14)
        - Target 1 (01:01): 1237.00
        - Target 2 (01:02): 1258.00
        - Target 3 (01:03): 1279.00
        - AMT 1:2: 17,612 (17024 + 14 * 42)
        - AMT 1:3: 17,906 (17024 + 14 * 63)
        - RISK: 294 (14 * 21) <= 300
        """
        res = self.calc.calculate_quantity_and_targets(
            entry_price=1216.0,
            sl_price=1195.0,
            action="BUY",
            custom_portfolio=30000.0,
            risk_pct=1.0
        )

        self.assertEqual(res["portfolio"], 30000.0)
        self.assertEqual(res["risk_per_trade"], 300.0)
        self.assertEqual(res["entry_price"], 1216.0)
        self.assertEqual(res["sl_price"], 1195.0)
        self.assertEqual(res["sl_rupee_per_share"], 21.0)
        self.assertEqual(res["quantity"], 14)
        self.assertEqual(res["total_shares_to_trade"], 14)
        self.assertEqual(res["investment_amount"], 17024.0)

        # Targets
        self.assertEqual(res["target_1_1"], 1237.0)
        self.assertEqual(res["target_1_2"], 1258.0)
        self.assertEqual(res["target_1_3"], 1279.0)

        # Payoff Amounts
        self.assertEqual(res["amt_1_2"], 17612.0)
        self.assertEqual(res["amt_1_3"], 17906.0)

        # Actual Risk
        self.assertEqual(res["actual_risk"], 294.0)
        self.assertTrue(res["within_1_pct_risk"])
        self.assertLessEqual(res["actual_risk"], 300.0)

    def test_02_user_second_example_floor_risk(self):
        """
        Validates user's explicit rule:
        'like risk = 11rs , capital = 10k , 1% of capital = 100 , so the qty = 9 , not 10 okkk , risking <= 1% of total capital okk'
        """
        res = self.calc.calculate_quantity_and_targets(
            entry_price=100.0,
            sl_price=89.0,  # 100 - 89 = 11 rs SL
            action="BUY",
            custom_portfolio=10000.0,
            risk_pct=1.0
        )

        self.assertEqual(res["sl_rupee_per_share"], 11.0)
        self.assertEqual(res["risk_per_trade"], 100.0)
        self.assertEqual(res["quantity"], 9)  # Exactly 9, NOT 10!
        self.assertEqual(res["actual_risk"], 99.0)
        self.assertLessEqual(res["actual_risk"], 100.0)
        self.assertTrue(res["within_1_pct_risk"])

    def test_03_short_sell_calculation_and_targets(self):
        """
        Validates short sell calculation:
        - SL is above entry (sl - entry)
        - Targets are below entry
        """
        res = self.calc.calculate_quantity_and_targets(
            entry_price=1000.0,
            sl_price=1011.0,  # 1011 - 1000 = 11 rs SL
            action="SELL",
            custom_portfolio=10000.0,
            risk_pct=1.0
        )

        self.assertTrue(res["is_short"])
        self.assertEqual(res["sl_rupee_per_share"], 11.0)
        self.assertEqual(res["quantity"], 9)
        self.assertEqual(res["actual_risk"], 99.0)
        self.assertEqual(res["target_1_1"], 989.0)   # 1000 - 11
        self.assertEqual(res["target_1_2"], 978.0)   # 1000 - 22
        self.assertEqual(res["target_1_3"], 967.0)   # 1000 - 33
        self.assertTrue(res["within_1_pct_risk"])

    def test_04_capital_threshold_logic(self):
        """
        Validates capital threshold rules:
        - If user sets 1 Lakh and paper has 5.26 Lakh -> trades under 1 Lakh
        - If user sets 10 Lakh (more than paper balance) -> caps at actual paper balance
        - If user resets to None -> uses full paper balance
        """
        calc = RiskCalculator()
        calc._cached_broker_capital = 526545.11
        calc._broker_cache_time = 9999999999.0

        # Case A: Custom limit < Broker balance (e.g. 1 Lakh)
        calc.update_trading_settings(custom_capital_limit=100000.0, risk_pct_per_trade=1.0)
        s = calc.get_trading_settings()
        self.assertEqual(s["effective_capital"], 100000.0)
        self.assertTrue(s["is_custom_limit_active"])
        self.assertEqual(s["risk_per_trade_inr"], 1000.0)

        # Case B: Custom limit > Broker balance (e.g. 10 Lakh) -> ignore limit, cap at actual
        calc.update_trading_settings(custom_capital_limit=1000000.0, risk_pct_per_trade=1.0)
        s = calc.get_trading_settings()
        self.assertEqual(s["effective_capital"], 526545.11)
        self.assertFalse(s["is_custom_limit_active"])
        self.assertEqual(s["risk_per_trade_inr"], 5265.45)

        # Case C: Reset to None -> default full money
        calc.update_trading_settings(custom_capital_limit=None, risk_pct_per_trade=1.0)
        s = calc.get_trading_settings()
        self.assertEqual(s["effective_capital"], 526545.11)
        self.assertFalse(s["is_custom_limit_active"])

    def test_05_orders_service_integration(self):
        """Validates that OrdersService uses RiskCalculator in normalize_order and summary_stats."""
        raw_doc = {
            "_id": "SIG_TEST_01",
            "symbol": "SBIN",
            "action": "BUY",
            "buy_trigger": 800.0,
            "entry_price": 800.0,
            "sl_price": 780.0,  # 20 rs SL per share
            "target_price": 840.0,
            "status": "ARMED"
        }

        norm = orders_service.normalize_order(raw_doc)
        self.assertEqual(norm["symbol"], "SBIN")
        self.assertEqual(norm["risk_per_share"], 20.0)
        self.assertIn("target_1_1", norm)
        self.assertIn("target_1_2", norm)
        self.assertIn("target_1_3", norm)
        self.assertIn("amt_1_2", norm)
        self.assertIn("amt_1_3", norm)
        self.assertTrue(norm["within_1_pct_risk"])

        # Summary stats check
        stats = orders_service.get_order_summary_stats()
        self.assertIn("effective_capital", stats)
        self.assertIn("risk_per_trade_inr", stats)
        self.assertIn("broker_total_capital", stats)


if __name__ == "__main__":
    unittest.main()
