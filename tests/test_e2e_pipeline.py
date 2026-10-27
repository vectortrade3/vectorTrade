"""
VectorTrade - End-to-End Simulation Harness (Phase 6)
Conducts a zero-network integration test across the entire trading lifecycle:
1. Simulates Setups 1, 2, and 3 through Feature Engine and Strategy Engine.
2. Applies PortfolioRiskManager: dynamic sizing, deduplication, and sector constraints.
3. Dispatches ARMED records into MongoDB signals_watchlist.
4. Executes simulated micro-worker tick runs testing:
   - Scenario A: Stock opens, drops to SL, then rallies -> Invalidation triggered, NO BUY order.
   - Scenario B: Stock triggers Buy, then reaches Target -> Buy executed, Target exit executed, P&L recorded.
   - Scenario C: Stock triggers Buy, then drops to SL -> Buy executed, SL exit executed, Loss recorded.
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from core.indicators import compute_indicators
from core.strategies.ma50_pullback import MA50PullbackStrategy
from core.risk_manager import PortfolioRiskManager
from online.execution_daemon import OnlineExecutionDaemon


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.db = db_manager.db
        self.signals_col = db_manager.get_collection("signals_watchlist")
        self.active_col = db_manager.get_collection("active_trades")
        self.history_col = db_manager.get_collection("trade_history")

        # Clean test state
        self.signals_col.delete_many({})
        self.active_col.delete_many({})
        self.history_col.delete_many({})

    def generate_synthetic_series(self, n_bars: int = 220) -> pd.DataFrame:
        """Generates historical OHLCV series with guaranteed upward MA50."""
        dates = pd.date_range("2026-01-01", periods=n_bars, freq="B")
        base = np.linspace(500.0, 1000.0, n_bars)
        close = base + np.random.normal(0, 1.0, n_bars)
        high = close + 5.0
        low = close - 5.0
        open_p = low + 2.0
        volume = np.full(n_bars, 1000000.0)

        return pd.DataFrame({
            "date": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": volume,
            "symbol": "TEST"
        })

    def test_full_e2e_scenarios(self):
        """Simulates all 3 execution lifecycles: Invalidation, Target Win, and SL Loss."""
        # 1. Craft Setup 1 (Confluence), Setup 2 (Extended), and Setup 3 (Mean-reverting)
        df_base = self.generate_synthetic_series()
        ind_df = compute_indicators(df_base)

        strategy = MA50PullbackStrategy()
        risk_mgr = PortfolioRiskManager(total_capital=100000.0, risk_per_trade_pct=1.5)

        # Craft Candidate A (Scenario A: Will be Invalidated)
        df_a = ind_df.copy()
        idx_a = df_a.index[-1]
        df_a.loc[idx_a, "ma50"] = 800.0
        df_a.loc[df_a.index[-2], "ma50"] = 799.0
        df_a.loc[df_a.index[-8], "ma50"] = 795.0
        df_a.loc[idx_a, "bb_lower"] = 802.0
        df_a.loc[idx_a, "low"] = 805.0       # Low - BB = 3.0 (in range)
        df_a.loc[idx_a, "open"] = 808.0
        df_a.loc[idx_a, "close"] = 815.0     # Green candle
        df_a.loc[idx_a, "high"] = 820.0      # Buy = 821, SL = 804, Risk = 17, Target = 855

        sigs_a = strategy.evaluate(df_a, symbol="STOCK_A", instrument_token="1001")
        self.assertEqual(len(sigs_a), 1)

        # Craft Candidate B (Scenario B: Will reach Target)
        df_b = ind_df.copy()
        idx_b = df_b.index[-1]
        df_b.loc[idx_b, "ma50"] = 600.0      # Setup 2: Low - MA50 = 640 - 600 = 40 (> 30)
        df_b.loc[df_b.index[-2], "ma50"] = 599.0
        df_b.loc[df_b.index[-8], "ma50"] = 595.0
        df_b.loc[idx_b, "bb_lower"] = 638.0
        df_b.loc[idx_b, "low"] = 640.0
        df_b.loc[idx_b, "open"] = 645.0
        df_b.loc[idx_b, "close"] = 655.0
        df_b.loc[idx_b, "high"] = 660.0      # Buy = 661, SL = 639, Risk = 22, Target = 705

        sigs_b = strategy.evaluate(df_b, symbol="STOCK_B", instrument_token="1002")
        self.assertEqual(len(sigs_b), 1)

        # Craft Candidate C (Scenario C: Will hit SL after entry)
        df_c = ind_df.copy()
        idx_c = df_c.index[-1]
        df_c.loc[idx_c, "ma50"] = 920.0      # Setup 3: Low - MA50 = 905 - 920 = -15 (< -10)
        df_c.loc[df_c.index[-2], "ma50"] = 919.0
        df_c.loc[df_c.index[-8], "ma50"] = 915.0
        df_c.loc[idx_c, "bb_lower"] = 903.0
        df_c.loc[idx_c, "low"] = 905.0
        df_c.loc[idx_c, "open"] = 910.0
        df_c.loc[idx_c, "close"] = 918.0
        df_c.loc[idx_c, "high"] = 925.0      # Buy = 926, SL = 904, Risk = 22, Target = 970

        sigs_c = strategy.evaluate(df_c, symbol="STOCK_C", instrument_token="1003")
        self.assertEqual(len(sigs_c), 1)

        raw_signals = sigs_a + sigs_b + sigs_c
        universe = {
            "STOCK_A": {"sector": "Energy"},
            "STOCK_B": {"sector": "Automobile"},
            "STOCK_C": {"sector": "Metals"}
        }

        # 2. Pass through Risk Manager
        approved = risk_mgr.filter_and_size_signals(raw_signals, universe=universe)
        self.assertEqual(len(approved), 3, "All 3 non-correlated signals should be approved and sized")

        # Persist ARMED signals to MongoDB
        for s in approved:
            self.signals_col.insert_one(s)

        # 3. Initialize Online Execution Worker
        daemon = OnlineExecutionDaemon(dry_run=True)
        daemon.load_armed_signals()
        self.assertEqual(len(daemon.armed_signals), 3)

        # -------------------------------------------------------------
        # Scenario A Execution: Dropping to SL first -> Invalidation
        # -------------------------------------------------------------
        # STOCK_A: Buy = 821, SL = 804. Tick arrives at 802 (SL breached before Buy)
        daemon.handle_invalidation_and_entry(ltp_override={"1001": 802.0})

        doc_a = self.signals_col.find_one({"symbol": "STOCK_A"})
        self.assertEqual(doc_a["status"], "INVALIDATED")
        # Ensure NO buy order was entered
        self.assertIsNone(self.active_col.find_one({"symbol": "STOCK_A"}))

        # -------------------------------------------------------------
        # Scenario B Execution: Breaks Buy (661) -> OPEN, then reaches Target (705) -> EXIT WIN
        # -------------------------------------------------------------
        # Tick 1: Reaches 662 (Buy trigger)
        daemon.handle_invalidation_and_entry(ltp_override={"1002": 662.0})
        doc_b = self.signals_col.find_one({"symbol": "STOCK_B"})
        self.assertEqual(doc_b["status"], "TRIGGERED")
        active_b = self.active_col.find_one({"symbol": "STOCK_B", "status": "OPEN"})
        self.assertIsNotNone(active_b)
        self.assertEqual(active_b["fill_price"], 662.0)

        # Tick 2: Reaches 708 (Target hit: 708 >= 705)
        daemon.handle_position_exits(ltp_override={"1002": 708.0})
        self.assertIsNone(self.active_col.find_one({"symbol": "STOCK_B", "status": "OPEN"}))

        hist_b = self.history_col.find_one({"symbol": "STOCK_B"})
        self.assertIsNotNone(hist_b)
        self.assertEqual(hist_b["exit_reason"], "TARGET_HIT")
        self.assertGreater(hist_b["pnl"], 0.0)
        self.assertEqual(hist_b["ml_outcome_label"], 1)

        # -------------------------------------------------------------
        # Scenario C Execution: Breaks Buy (926) -> OPEN, then drops to SL (904) -> EXIT LOSS
        # -------------------------------------------------------------
        # Tick 1: Reaches 928 (Buy trigger)
        daemon.handle_invalidation_and_entry(ltp_override={"1003": 928.0})
        active_c = self.active_col.find_one({"symbol": "STOCK_C", "status": "OPEN"})
        self.assertIsNotNone(active_c)
        self.assertEqual(active_c["fill_price"], 928.0)

        # Tick 2: Drops to 902 (SL breached: 902 <= 904)
        daemon.handle_position_exits(ltp_override={"1003": 902.0})
        self.assertIsNone(self.active_col.find_one({"symbol": "STOCK_C", "status": "OPEN"}))

        hist_c = self.history_col.find_one({"symbol": "STOCK_C"})
        self.assertIsNotNone(hist_c)
        self.assertEqual(hist_c["exit_reason"], "SL_HIT")
        self.assertLess(hist_c["pnl"], 0.0)
        self.assertEqual(hist_c["ml_outcome_label"], 0)

        print("\nAll 3 End-to-End Execution Scenarios Verified Successfully!")


if __name__ == "__main__":
    unittest.main()
