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
        self.assertIn("broker_used_capital", stats)
        self.assertIn("allocated_remaining_capital", stats)

    @patch("online.services.megabull_service.megabull_service.get_account_details")
    def test_06_allocated_remaining_capital_user_scenario(self, mock_acc):
        """
        Validates user's exact scenario:
        - Overall broker capital = 5 Lakh (500,000)
        - Remaining broker free cash = 4.10 Lakh (410,000)
        - Broker used capital = 500,000 - 410,000 = 90,000
        - User threshold = 1 Lakh (100,000)
        - Remaining allocated capital = 100,000 - 90,000 = 10,000 (10k only!)
        """
        mock_acc.return_value = {
            "status": "SUCCESS",
            "virtual_money": 500000.00,
            "virtual_money_left": 410000.00,
            "virtual_money_blocked": 90000.00
        }

        calc = RiskCalculator()
        calc._cached_broker_capital = 500000.00
        calc._broker_cache_time = 9999999999.0

        calc.update_trading_settings(custom_capital_limit=100000.0, risk_pct_per_trade=1.0)
        settings = calc.get_trading_settings()

        self.assertEqual(settings["broker_total_capital"], 500000.00)
        self.assertEqual(settings["broker_available_balance"], 410000.00)
        self.assertEqual(settings["broker_used_capital"], 90000.00)
        self.assertEqual(settings["effective_capital"], 100000.00)
        self.assertTrue(settings["is_custom_limit_active"])
        # Exactly 10k remaining!
        self.assertEqual(settings["allocated_remaining_capital"], 10000.00)
        self.assertEqual(settings["allocated_used_capital"], 90000.00)
        self.assertEqual(settings["allocated_used_pct"], 90.0)
        self.assertTrue(settings["can_trade"])

    @patch("online.services.megabull_service.megabull_service.get_account_details")
    def test_07_check_capital_for_trade_boundaries(self, mock_acc):
        """
        Validates trade affordability under active allocated remaining capital:
        - Remaining: 10,000
        - Trade 1: Qty 10 @ 800 = 8,000 <= 10,000 -> Allowed (True)
        - Trade 2: Qty 10 @ 1200 = 12,000 > 10,000 -> Rejected (False, shortfall 2,000)
        - Trade 3: SELL (square off) Qty 10 @ 1200 -> Allowed (True, frees capital)
        """
        mock_acc.return_value = {
            "status": "SUCCESS",
            "virtual_money": 500000.00,
            "virtual_money_left": 410000.00,
            "virtual_money_blocked": 90000.00
        }

        calc = RiskCalculator()
        calc._cached_broker_capital = 500000.00
        calc._broker_cache_time = 9999999999.0
        calc.update_trading_settings(custom_capital_limit=100000.0, risk_pct_per_trade=1.0)

        # 1. Affordable trade: 8,000 <= 10,000
        check1 = calc.check_capital_for_trade(quantity=10, entry_price=800.0, action="BUY")
        self.assertTrue(check1["allowed"])
        self.assertEqual(check1["trade_amount"], 8000.00)
        self.assertEqual(check1["allocated_remaining_capital"], 10000.00)
        self.assertEqual(check1["shortfall"], 0.0)

        # 2. Exceeding trade: 12,000 > 10,000 -> strictly rejected
        check2 = calc.check_capital_for_trade(quantity=10, entry_price=1200.0, action="BUY")
        self.assertFalse(check2["allowed"])
        self.assertEqual(check2["trade_amount"], 12000.00)
        self.assertEqual(check2["allocated_remaining_capital"], 10000.00)
        self.assertEqual(check2["shortfall"], 2000.00)
        self.assertIn("exceeds allocated remaining capital", check2["reason"])

        # 3. Square off / exit order (SELL): always allowed
        check3 = calc.check_capital_for_trade(quantity=10, entry_price=1200.0, action="SELL")
        self.assertTrue(check3["allowed"])

    @patch("online.services.megabull_service.megabull_service.is_symbol_in_portfolio")
    @patch("online.services.megabull_service.megabull_service.get_liquidation_status")
    @patch("online.services.risk_calculator.risk_calculator.check_capital_for_trade")
    def test_08_megabull_place_order_insufficient_allocated_capital_guard(
        self,
        mock_cap_check,
        mock_liq,
        mock_dup
    ):
        """
        Validates that megabull_service.place_order strictly blocks buy orders
        when trade amount exceeds remaining allocated capital.
        """
        from online.services.megabull_service import megabull_service

        mock_dup.return_value = {"exists": False}
        mock_liq.return_value = {"armed": False, "status": "NONE"}
        mock_cap_check.return_value = {
            "allowed": False,
            "trade_amount": 15000.00,
            "allocated_remaining_capital": 10000.00,
            "shortfall": 5000.00,
            "reason": "Trade amount ₹15,000 exceeds allocated remaining capital ₹10,000. Order cannot be placed."
        }

        res = megabull_service.place_order(
            symbol="TITAN",
            action="BUY",
            quantity=5,
            price=3000.0,
            strategy="50 MA Setup"
        )

        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["code"], "INSUFFICIENT_ALLOCATED_CAPITAL")
        self.assertEqual(res["shortfall"], 5000.00)
        self.assertIn("exceeds allocated remaining capital", res["error"])

    def test_09_candidate_sorting_best_orders_on_top(self):
        """Verifies candidate setups are sorted with ARMED on top and highest predicted_r descending."""
        raw_candidates = [
            {"symbol": "ACTIVE_HIGH_R", "status": "ACTIVE", "predicted_r": 3.10, "profit_pct": 5.0, "entry_price": 500, "sl_price": 490},
            {"symbol": "ARMED_MED_R", "status": "ARMED", "predicted_r": 2.27, "profit_pct": 4.0, "entry_price": 600, "sl_price": 585},
            {"symbol": "ARMED_HIGH_R", "status": "ARMED", "predicted_r": 2.48, "profit_pct": 4.5, "entry_price": 700, "sl_price": 680},
            {"symbol": "ARMED_LOW_R", "status": "ARMED", "predicted_r": 1.85, "profit_pct": 3.0, "entry_price": 400, "sl_price": 390},
        ]
        calc = RiskCalculator()
        with patch.object(calc, "get_candidate_setups", return_value=raw_candidates):
            plan = calc.evaluate_and_allocate_candidate_orders(custom_capital_limit=100000.0, auto_stage_confirmed=False)
            all_ranked = plan["confirmed_orders"] + plan["skipped_orders"]
            all_ranked.sort(key=lambda x: x["priority_rank"])
            symbols_in_order = [x["symbol"] for x in all_ranked]

            # Highest ARMED first (2.48R > 2.27R > 1.85R), then non-ARMED
            self.assertEqual(symbols_in_order[0], "ARMED_HIGH_R")
            self.assertEqual(symbols_in_order[1], "ARMED_MED_R")
            self.assertEqual(symbols_in_order[2], "ARMED_LOW_R")
            self.assertEqual(symbols_in_order[3], "ACTIVE_HIGH_R")

    def test_10_capital_allocation_greedy_budget_knapsack(self):
        """
        Validates sequential capital allocation:
        - Portfolio: 30,000 (1% Risk = 300)
        - Ord 1: ALPHA (Entry 1216, SL 1195 -> Diff 21, Qty 14, Inv 17,024) -> Fits (Remaining 12,976)
        - Ord 2: BETA (Entry 1500, SL 1470 -> Diff 30, Qty 10, Inv 15,000) -> Exceeds 12,976 -> SKIPPED
        - Ord 3: GAMMA (Entry 400, SL 390 -> Diff 10, Qty 30, Inv 12,000) -> Fits 12,976 -> CONFIRMED (Remaining 976)
        """
        raw_candidates = [
            {"symbol": "ALPHA", "status": "ARMED", "predicted_r": 2.50, "profit_pct": 4.0, "entry_price": 1216.0, "sl_price": 1195.0},
            {"symbol": "BETA", "status": "ARMED", "predicted_r": 2.40, "profit_pct": 3.5, "entry_price": 1500.0, "sl_price": 1470.0},
            {"symbol": "GAMMA", "status": "ARMED", "predicted_r": 2.30, "profit_pct": 3.0, "entry_price": 400.0, "sl_price": 390.0},
        ]
        calc = RiskCalculator()
        calc._cached_broker_capital = 500000.0
        calc._broker_cache_time = 9999999999.0

        mock_settings = {
            "broker_total_capital": 500000.0,
            "broker_available_balance": 500000.0,
            "broker_used_capital": 0.0,
            "effective_capital": 30000.0,
            "risk_pct_per_trade": 1.0
        }

        with patch.object(calc, "get_candidate_setups", return_value=raw_candidates):
            with patch.object(calc, "get_trading_settings", return_value=mock_settings):
                plan = calc.evaluate_and_allocate_candidate_orders(
                    custom_capital_limit=30000.0,
                    risk_pct=1.0,
                    auto_stage_confirmed=False
                )

                self.assertEqual(plan["effective_capital"], 30000.0)
                self.assertEqual(plan["risk_per_trade"], 300.0)
                self.assertEqual(plan["confirmed_orders_count"], 2)
                self.assertEqual(plan["skipped_orders_count"], 1)

            # Confirmed orders
            conf_syms = [x["symbol"] for x in plan["confirmed_orders"]]
            self.assertEqual(conf_syms, ["ALPHA", "GAMMA"])

            # ALPHA checks
            alpha = plan["confirmed_orders"][0]
            self.assertEqual(alpha["quantity"], 14)
            self.assertEqual(alpha["investment_amount"], 17024.0)
            self.assertEqual(alpha["actual_risk"], 294.0)
            self.assertEqual(alpha["remaining_budget_after"], 12976.0)

            # GAMMA checks
            gamma = plan["confirmed_orders"][1]
            self.assertEqual(gamma["quantity"], 30)
            self.assertEqual(gamma["investment_amount"], 12000.0)
            self.assertEqual(gamma["actual_risk"], 300.0)
            self.assertEqual(gamma["remaining_budget_after"], 976.0)

            # Skipped order: BETA
            skip_syms = [x["symbol"] for x in plan["skipped_orders"]]
            self.assertEqual(skip_syms, ["BETA"])
            beta = plan["skipped_orders"][0]
            self.assertEqual(beta["quantity"], 10)
            self.assertEqual(beta["investment_amount"], 15000.0)
            self.assertEqual(beta["shortfall"], 2024.0)
            self.assertIn("Insufficient remaining capital", beta["reason"])

            # Totals
            self.assertEqual(plan["total_committed_capital"], 29024.0)
            self.assertEqual(plan["remaining_unallocated_cash"], 976.0)
            self.assertEqual(plan["total_risk_committed"], 594.0)

    def test_11_allocation_plan_auto_stages_to_staged_orders(self):
        """Verifies confirmed orders are staged off-market (AMO) into MongoDB staged_orders."""
        from unittest.mock import PropertyMock
        raw_candidates = [
            {"symbol": "INFY", "status": "ARMED", "predicted_r": 2.2, "profit_pct": 3.0, "entry_price": 1500.0, "sl_price": 1470.0}
        ]
        calc = RiskCalculator()
        mock_staged_col = MagicMock()
        mock_system_state = MagicMock()

        mock_settings = {
            "broker_total_capital": 500000.0,
            "broker_available_balance": 500000.0,
            "broker_used_capital": 0.0,
            "effective_capital": 50000.0,
            "risk_pct_per_trade": 1.0
        }

        with patch.object(calc, "get_candidate_setups", return_value=raw_candidates):
            with patch.object(calc, "get_trading_settings", return_value=mock_settings):
                with patch("online.db.DatabaseManager.staged_orders", new_callable=PropertyMock, return_value=mock_staged_col):
                    with patch("online.db.DatabaseManager.system_state", new_callable=PropertyMock, return_value=mock_system_state):
                        plan = calc.evaluate_and_allocate_candidate_orders(
                            custom_capital_limit=50000.0,
                            auto_stage_confirmed=True
                        )
                        self.assertEqual(plan["confirmed_orders_count"], 1)
                        mock_staged_col.delete_many.assert_called_once_with({
                            "is_auto_allocated": True,
                            "status": "STAGED_OFFMARKET"
                        })
                        mock_staged_col.update_one.assert_called()

    def test_12_api_trading_settings_allocation_plan_endpoints(self):
        """Verifies GET /api/trading-settings/allocation-plan and POST /api/trading-settings/allocate."""
        from offline.app import app as flask_app
        client = flask_app.test_client()

        # 1. GET /api/trading-settings/allocation-plan
        with patch("online.services.risk_calculator.risk_calculator.get_allocation_plan") as mock_get_plan:
            mock_get_plan.return_value = {
                "effective_capital": 100000.0,
                "confirmed_orders_count": 3,
                "confirmed_orders": []
            }
            res = client.get("/api/trading-settings/allocation-plan")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["plan"]["effective_capital"], 100000.0)

        # 2. POST /api/trading-settings/allocate
        with patch("online.services.risk_calculator.risk_calculator.evaluate_and_allocate_candidate_orders") as mock_eval:
            mock_eval.return_value = {
                "effective_capital": 80000.0,
                "confirmed_orders_count": 2,
                "confirmed_orders": []
            }
            res_post = client.post("/api/trading-settings/allocate", json={"custom_capital_limit": 80000.0})
            self.assertEqual(res_post.status_code, 200)
            data_post = res_post.get_json()
            self.assertEqual(data_post["status"], "SUCCESS")
            self.assertEqual(data_post["plan"]["confirmed_orders_count"], 2)

    def test_13_ten_percent_position_cap_on_one_lakh_account(self):
        """
        Validates user requirement:
        'make sure that we dont put more than 10% of the total capital in a single stock ,
        and like for now the total capital is 1Lakh ,10% is 10k , so we have to things to calcuilate the qty on ,
        based on risk , and if qty based on risk is high , then we just lower the qty until its under 10% of the overall amt'
        """
        calc = RiskCalculator()

        # 1. PNB test: Entry 125.10, SL 121.51 (Diff 3.59), Portfolio 100,000
        # Risk 1% = 1000 -> Qty by risk = 278 (would be 34.7k without cap!)
        # 10% cap = 10,000 -> Qty by cap = floor(10,000 / 125.10) = 79
        # Final Qty must be 79
        pnb_res = calc.calculate_quantity_and_targets(
            entry_price=125.10,
            sl_price=121.51,
            action="BUY",
            custom_portfolio=100000.0,
            risk_pct=1.0
        )
        self.assertEqual(pnb_res["quantity"], 79)
        self.assertLessEqual(pnb_res["investment_amount"], 10000.0)
        self.assertEqual(pnb_res["investment_amount"], 9882.90)
        self.assertLessEqual(pnb_res["actual_risk"], 1000.0)

        # 2. INDUSTOWER test: Entry 384.35, SL 378.20 (Diff 6.15), Portfolio 100,000
        # Risk 1% = 1000 -> Qty by risk = 162 (would be 62.2k without cap!)
        # 10% cap = 10,000 -> Qty by cap = floor(10,000 / 384.35) = 26
        # Final Qty must be 26
        indus_res = calc.calculate_quantity_and_targets(
            entry_price=384.35,
            sl_price=378.20,
            action="BUY",
            custom_portfolio=100000.0,
            risk_pct=1.0
        )
        self.assertEqual(indus_res["quantity"], 26)
        self.assertLessEqual(indus_res["investment_amount"], 10000.0)
        self.assertEqual(indus_res["investment_amount"], 9993.10)
        self.assertLessEqual(indus_res["actual_risk"], 1000.0)

    def test_14_confluence_priority_and_alphabetical_tie_breaker(self):
        """
        Validates user requirement:
        'priority is -> a stock selected by more than 1 strategy, next sort filter -> alphabetically'
        """
        raw_candidates = [
            {"symbol": "ZEE", "status": "ARMED", "strategy": "50 MA Setup", "predicted_r": 2.50, "entry_price": 100.0, "sl_price": 95.0},
            {"symbol": "TATA_MULTI", "status": "ARMED", "strategy": "50 MA + Global Regime", "predicted_r": 1.80, "entry_price": 200.0, "sl_price": 190.0},
            {"symbol": "ADANI_MULTI", "status": "ARMED", "strategy": "50 MA + Global Regime", "predicted_r": 1.90, "entry_price": 300.0, "sl_price": 285.0},
            {"symbol": "AIRTEL", "status": "ARMED", "strategy": "50 MA Setup", "predicted_r": 2.60, "entry_price": 400.0, "sl_price": 380.0},
        ]
        calc = RiskCalculator()
        with patch.object(calc, "get_candidate_setups", return_value=raw_candidates):
            plan = calc.evaluate_and_allocate_candidate_orders(custom_capital_limit=100000.0, auto_stage_confirmed=False)
            all_ranked = plan["confirmed_orders"] + plan["skipped_orders"]
            all_ranked.sort(key=lambda x: x["priority_rank"])
            ordered_syms = [x["symbol"] for x in all_ranked]

            # 1. Multi-strategy confluence candidates come FIRST, ordered alphabetically:
            # ADANI_MULTI before TATA_MULTI
            self.assertEqual(ordered_syms[0], "ADANI_MULTI")
            self.assertEqual(ordered_syms[1], "TATA_MULTI")

            # 2. Single-strategy candidates come next (sorted by expectancy / tie-breakers):
            self.assertIn("AIRTEL", ordered_syms[2:])
            self.assertIn("ZEE", ordered_syms[2:])


if __name__ == "__main__":
    unittest.main()
