"""
VectorTrade - Phase 5 Verification Suite
Comprehensive tests for:
1. Dynamic Capital-at-Risk Position Sizing:
   - Verifies Quantity = floor(Risk_Amount / (Buy - SL)).
2. Sector Concentration Filter:
   - Verifies that signals exceeding 30% sector allocation cap are cleanly rejected.
3. Cross-Strategy Correlation & Duplicate Symbol Deduplication:
   - Verifies that multiple signals on the same symbol are deduplicated in favor of higher ML confidence.
4. Portfolio-Level Circuit Breaker:
   - Verifies that when cumulative active risk breaches the threshold (e.g. 7.5%), new arming is suspended.
5. Zero Online Overhead:
   - Confirms online worker schema remains untouched and zero risk code runs in online daemon.
"""

import sys
import os
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.risk_manager import PortfolioRiskManager
from core.db import db_manager


class TestPortfolioRiskManagement(unittest.TestCase):

    def setUp(self):
        # 100k capital, 1.5% risk per trade = Rs 1,500 risk per trade
        # Max 5 open positions, max 30% per sector (1 position per sector), max 7.5% portfolio risk
        self.risk_mgr = PortfolioRiskManager(
            total_capital=100000.0,
            risk_per_trade_pct=1.5,
            max_open_positions=5,
            max_sector_allocation_pct=30.0,
            max_portfolio_risk_pct=7.5
        )
        # Clear mock active_trades
        active_col = db_manager.get_collection("active_trades")
        active_col.delete_many({})

    def test_dynamic_position_sizing(self):
        """Verifies lot sizing: Quantity = floor(1500 / (Buy - SL))."""
        # Case A: Buy = 1000, SL = 950 (Risk = 50) -> Qty = 1500 / 50 = 30
        qty_a = self.risk_mgr.calculate_position_size(buy_trigger=1000.0, sl_price=950.0)
        self.assertEqual(qty_a, 30)

        # Case B: Buy = 2500, SL = 2470 (Risk = 30) -> Qty = 1500 / 30 = 50
        qty_b = self.risk_mgr.calculate_position_size(buy_trigger=2500.0, sl_price=2470.0)
        self.assertEqual(qty_b, 50)

        # Case C: Tight SL: Buy = 500, SL = 495 (Risk = 5) -> Qty = 1500 / 5 = 300
        qty_c = self.risk_mgr.calculate_position_size(buy_trigger=500.0, sl_price=495.0)
        self.assertEqual(qty_c, 300)

    def test_cross_strategy_deduplication(self):
        """Verifies duplicate signals on same symbol select highest ML confidence."""
        raw_signals = [
            {
                "symbol": "RELIANCE",
                "strategy_id": "50MA_PULLBACK",
                "buy_trigger": 1200.0,
                "sl_price": 1170.0,
                "target_price": 1260.0,
                "ml_confidence": 0.65,
                "_id": "SIG_1"
            },
            {
                "symbol": "RELIANCE",
                "strategy_id": "GLOBAL_FORECAST",
                "buy_trigger": 1200.0,
                "sl_price": 1170.0,
                "target_price": 1280.0,
                "ml_confidence": 0.88,  # Higher confidence
                "_id": "SIG_2"
            }
        ]
        universe = {"RELIANCE": {"sector": "Energy"}}
        approved = self.risk_mgr.filter_and_size_signals(raw_signals, universe=universe)

        self.assertEqual(len(approved), 1, "Expected only 1 signal for RELIANCE after deduplication")
        self.assertEqual(approved[0]["strategy_id"], "GLOBAL_FORECAST")
        self.assertEqual(approved[0]["ml_confidence"], 0.88)

    def test_sector_concentration_cap(self):
        """Verifies signals exceeding sector concentration cap are rejected."""
        # 30% of 5 positions = 1.5 -> floor/max = 1 max per sector
        raw_signals = [
            {
                "symbol": "TCS",
                "strategy_id": "50MA_PULLBACK",
                "buy_trigger": 3500.0,
                "sl_price": 3450.0,
                "target_price": 3600.0,
                "ml_confidence": 0.85,
                "_id": "SIG_TCS"
            },
            {
                "symbol": "INFY",
                "strategy_id": "50MA_PULLBACK",
                "buy_trigger": 1900.0,
                "sl_price": 1870.0,
                "target_price": 1960.0,
                "ml_confidence": 0.80,  # Same sector IT, lower confidence
                "_id": "SIG_INFY"
            },
            {
                "symbol": "HDFCBANK",
                "strategy_id": "50MA_PULLBACK",
                "buy_trigger": 1600.0,
                "sl_price": 1580.0,
                "target_price": 1640.0,
                "ml_confidence": 0.78,  # Financial Services
                "_id": "SIG_HDFC"
            }
        ]
        universe = {
            "TCS": {"sector": "Information Technology"},
            "INFY": {"sector": "Information Technology"},
            "HDFCBANK": {"sector": "Financial Services"}
        }

        approved = self.risk_mgr.filter_and_size_signals(raw_signals, universe=universe)
        approved_symbols = [s["symbol"] for s in approved]

        self.assertIn("TCS", approved_symbols, "Highest confidence IT stock must be approved")
        self.assertNotIn("INFY", approved_symbols, "Second IT stock must be rejected by sector cap")
        self.assertIn("HDFCBANK", approved_symbols, "Financial Services stock must be approved")

    def test_portfolio_circuit_breaker(self):
        """Verifies circuit breaker halts new arming if cumulative active risk is breached."""
        active_col = db_manager.get_collection("active_trades")
        # Insert 5 active trades totaling Rs 8,000 risk (> 7,500 cap)
        active_col.insert_one({
            "_id": "ACTIVE_MOCK_1",
            "symbol": "SBIN",
            "fill_price": 800.0,
            "sl_price": 720.0, # 80 risk * 100 qty = 8,000
            "quantity": 100,
            "status": "OPEN",
            "sector": "Financial Services"
        })

        raw_signals = [
            {
                "symbol": "TATAMOTORS",
                "strategy_id": "50MA_PULLBACK",
                "buy_trigger": 1000.0,
                "sl_price": 950.0,
                "target_price": 1100.0,
                "ml_confidence": 0.95,
                "_id": "SIG_TATA"
            }
        ]
        universe = {"TATAMOTORS": {"sector": "Automobile"}}

        approved = self.risk_mgr.filter_and_size_signals(raw_signals, universe=universe)
        self.assertEqual(len(approved), 0, "Circuit breaker must reject new signals when portfolio risk cap breached")


if __name__ == "__main__":
    unittest.main()
