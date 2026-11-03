"""
Tests for VectorTrade 50MA Strategy Dataset Cleaning & Trajectory Resolution Engine
Verifies Day 1 entry gate, Day 1 conflict resolution, 30-day forward trajectory tracking,
mathematical bracket integrity, and candlestick JSON payload format.
"""

import unittest
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import data_cleaner, json_manager


class Test50MADatasetCleaning(unittest.TestCase):

    def setUp(self):
        self.csv_50ma = BASE_DIR / "5_year_final_data_50ma.csv"
        self.csv_global = BASE_DIR / "5_year_final_data_global.csv"
        self.csv_master = BASE_DIR / "5_year_final_data.csv"

    def test_output_csv_files_exist_and_non_empty(self):
        """Verifies that all 3 output files exist on disk and are populated."""
        self.assertTrue(self.csv_50ma.exists(), "5_year_final_data_50ma.csv does not exist")
        self.assertTrue(self.csv_global.exists(), "5_year_final_data_global.csv does not exist")
        self.assertTrue(self.csv_master.exists(), "5_year_final_data.csv does not exist")

        self.assertGreater(self.csv_50ma.stat().st_size, 1_000_000, "50MA dataset is too small")
        self.assertGreater(self.csv_global.stat().st_size, 100_000_000, "Global dataset is too small")
        self.assertGreater(self.csv_master.stat().st_size, 100_000_000, "Master dataset is too small")

    def test_50ma_math_integrity_and_bracket_geometry(self):
        """Verifies strict 1:2.00 Risk:Reward bracket geometry across all 50MA trades."""
        df = pd.read_csv(self.csv_50ma)
        self.assertGreater(len(df), 5000, "Expected at least 5000 verified 50MA trades")

        # 1. Entry price = setup_high + 1.0
        expected_entry = (df['setup_high'] + 1.0).round(2)
        diff_entry = (df['entry_price'] - expected_entry).abs().max()
        self.assertAlmostEqual(diff_entry, 0.0, places=2, msg="Entry price mismatch")

        # 2. SL price = setup_low - 1.0
        expected_sl = (df['setup_low'] - 1.0).round(2)
        diff_sl = (df['sl_price'] - expected_sl).abs().max()
        self.assertAlmostEqual(diff_sl, 0.0, places=2, msg="SL price mismatch")

        # 3. Risk = Entry - SL
        expected_risk = (df['entry_price'] - df['sl_price']).round(2)
        diff_risk = (df['risk'] - expected_risk).abs().max()
        self.assertAlmostEqual(diff_risk, 0.0, places=2, msg="Risk mismatch")

        # 4. Target = Entry + 2 * Risk (Strict 1:2.00 R:R)
        expected_target = (df['entry_price'] + 2.0 * df['risk']).round(2)
        diff_target = (df['target_price'] - expected_target).abs().max()
        self.assertAlmostEqual(diff_target, 0.0, places=2, msg="Target price mismatch")

        # 5. All trades must be triggered
        self.assertTrue(df['triggered'].all(), "Found un-triggered trade in 50MA dataset")

    def test_outcome_r_multiples(self):
        """Verifies that SUCCESS trades have realized_r == 2.0 and standard LOSS trades have realized_r == -1.0."""
        df = pd.read_csv(self.csv_50ma)

        success_trades = df[df['outcome'] == 'SUCCESS']
        self.assertGreater(len(success_trades), 2000, "Insufficient SUCCESS trades")
        diff_r_success = (success_trades['realized_r'] - 2.0).abs().max()
        self.assertAlmostEqual(diff_r_success, 0.0, places=1, msg="SUCCESS trades should have 2.0R payoff")

        loss_trades = df[df['outcome'] == 'LOSS']
        self.assertGreater(len(loss_trades), 3000, "Insufficient LOSS trades")
        diff_r_loss = (loss_trades['realized_r'] - (-1.0)).abs().max()
        self.assertAlmostEqual(diff_r_loss, 0.0, places=1, msg="LOSS trades should have -1.0R loss")

    def test_chart_candles_json_format(self):
        """Verifies structure and keys in embedded chart_candles_json column."""
        df = pd.read_csv(self.csv_50ma, nrows=50)

        for _, row in df.iterrows():
            raw_json = row['chart_candles_json']
            self.assertIsInstance(raw_json, str)
            candles = json.loads(raw_json)
            self.assertGreaterEqual(len(candles), 8, "Expected at least 8 candles (past 7 + setup)")
            self.assertLessEqual(len(candles), 38, "Max 38 candles expected (7 past + 1 setup + 30 future)")

            # Check setup candle
            setup_candles = [c for c in candles if c['type'] == 'setup']
            self.assertEqual(len(setup_candles), 1, "Exactly 1 setup candle expected")
            sc = setup_candles[0]
            self.assertIn('ma50', sc)
            self.assertIn('bb_lower', sc)
            self.assertEqual(sc['high'], row['setup_high'])
            self.assertEqual(sc['low'], row['setup_low'])

            # Check future candles
            future_candles = [c for c in candles if c['type'] == 'future']
            for fc in future_candles:
                self.assertIn('date', fc)
                self.assertIn('open', fc)
                self.assertIn('high', fc)
                self.assertIn('low', fc)
                self.assertIn('close', fc)

    def test_forward_tabular_columns(self):
        """Verifies that fwd_d1 through fwd_d30 OHLC columns exist in dataset."""
        df = pd.read_csv(self.csv_50ma, nrows=10)

        for d in range(1, 31):
            self.assertIn(f"fwd_d{d}_date", df.columns)
            self.assertIn(f"fwd_d{d}_open", df.columns)
            self.assertIn(f"fwd_d{d}_high", df.columns)
            self.assertIn(f"fwd_d{d}_low", df.columns)
            self.assertIn(f"fwd_d{d}_close", df.columns)

    def test_synthetic_day1_no_trigger_filtering(self):
        """Verifies synthetic case: Day 1 High < Buy Price is dropped from trades."""
        # Create 60 days of synthetic data
        dates = pd.date_range("2023-01-01", periods=60)
        df_synthetic = pd.DataFrame({
            "Date": dates.astype(str),
            "Open": [100.0] * 60,
            "High": [105.0] * 60,
            "Low": [95.0] * 60,
            "Close": [102.0] * 60,
            "Volume": [100000] * 60,
            "MA50": [90.0 + i * 0.2 for i in range(60)],  # trending up
            "BB_Lower": [94.0] * 60                       # dist_bb = 95 - 94 = 1.0 (in [-10, 20])
        })
        # At T=50: High=105, Low=95 -> Buy=106, SL=94.
        # Set Day 1 (T=51) High=104 (< 106) -> should NOT trigger
        df_synthetic.loc[51, 'High'] = 104.0
        df_synthetic.loc[51, 'Low'] = 98.0

        trades = data_cleaner.extract_50ma_trades_from_stock(df_synthetic, "TEST_SYM", "IT")
        # Ensure T=50 is NOT in trades
        setup_dates = [t['setup_date'] for t in trades]
        self.assertNotIn(str(dates[50])[:10], setup_dates, "No-trigger setup must be dropped")

    def test_synthetic_day1_sl_conflict_marked_loss(self):
        """Verifies synthetic case: Day 1 hits both Entry and SL -> marked as LOSS (SL_HIT)."""
        dates = pd.date_range("2023-01-01", periods=60)
        df_synthetic = pd.DataFrame({
            "Date": dates.astype(str),
            "Open": [100.0] * 60,
            "High": [105.0] * 60,
            "Low": [95.0] * 60,
            "Close": [102.0] * 60,
            "Volume": [100000] * 60,
            "MA50": [90.0 + i * 0.2 for i in range(60)],
            "BB_Lower": [94.0] * 60
        })
        # At T=50: High=105, Low=95 -> Buy=106, SL=94, Target=130
        # Set Day 1 (T=51) High=135 (>= Target), Low=90 (<= SL) -> Conflict on Day 1!
        df_synthetic.loc[51, 'High'] = 135.0
        df_synthetic.loc[51, 'Low'] = 90.0

        trades = data_cleaner.extract_50ma_trades_from_stock(df_synthetic, "TEST_SYM", "IT")
        trade_50 = [t for t in trades if t['setup_date'] == str(dates[50])[:10]][0]

        self.assertEqual(trade_50['outcome'], 'LOSS', "Day 1 conflict must be marked as LOSS")
        self.assertEqual(trade_50['days_held'], 0, "Day 1 conflict days_held should be 0")
        self.assertEqual(trade_50['realized_r'], -1.0)

    def test_synthetic_holding_target_hit(self):
        """Verifies synthetic case: Enters on Day 1, reaches Target on Day 4 -> SUCCESS."""
        dates = pd.date_range("2023-01-01", periods=60)
        df_synthetic = pd.DataFrame({
            "Date": dates.astype(str),
            "Open": [100.0] * 60,
            "High": [105.0] * 60,
            "Low": [95.0] * 60,
            "Close": [102.0] * 60,
            "Volume": [100000] * 60,
            "MA50": [90.0 + i * 0.2 for i in range(60)],
            "BB_Lower": [94.0] * 60
        })
        # T=50: High=105, Low=95 -> Buy=106, SL=94, Target=130
        # Day 1 (T=51): High=107 (entered), Low=98 (SL not hit)
        df_synthetic.loc[51, 'High'] = 107.0
        df_synthetic.loc[51, 'Low'] = 98.0
        # Days 2 & 3: stays between 100 and 115
        df_synthetic.loc[52, 'High'] = 110.0
        df_synthetic.loc[52, 'Low'] = 100.0
        df_synthetic.loc[53, 'High'] = 115.0
        df_synthetic.loc[53, 'Low'] = 101.0
        # Day 4 (T=54): High=132 (>= 130 Target), Low=105 (> SL)
        df_synthetic.loc[54, 'High'] = 132.0
        df_synthetic.loc[54, 'Low'] = 105.0

        trades = data_cleaner.extract_50ma_trades_from_stock(df_synthetic, "TEST_SYM", "IT")
        trade_50 = [t for t in trades if t['setup_date'] == str(dates[50])[:10]][0]

        self.assertEqual(trade_50['outcome'], 'SUCCESS')
        self.assertEqual(trade_50['days_held'], 4)
        self.assertEqual(trade_50['realized_r'], 2.0)
        self.assertEqual(trade_50['exit_price'], 130.0)

    def test_synthetic_same_day_target_sl_conflict_in_holding(self):
        """Verifies synthetic case: On Day 5 both Target and SL touched -> marked as LOSS per user rule."""
        dates = pd.date_range("2023-01-01", periods=60)
        df_synthetic = pd.DataFrame({
            "Date": dates.astype(str),
            "Open": [100.0] * 60,
            "High": [105.0] * 60,
            "Low": [95.0] * 60,
            "Close": [102.0] * 60,
            "Volume": [100000] * 60,
            "MA50": [90.0 + i * 0.2 for i in range(60)],
            "BB_Lower": [94.0] * 60
        })
        # T=50: High=105, Low=95 -> Buy=106, SL=94, Target=130
        df_synthetic.loc[51, 'High'] = 107.0
        df_synthetic.loc[51, 'Low'] = 98.0
        df_synthetic.loc[52, 'High'] = 110.0
        df_synthetic.loc[52, 'Low'] = 100.0
        df_synthetic.loc[53, 'High'] = 115.0
        df_synthetic.loc[53, 'Low'] = 101.0
        df_synthetic.loc[54, 'High'] = 120.0
        df_synthetic.loc[54, 'Low'] = 102.0
        # Day 5 (T=55): High=135 (>= Target), Low=92 (<= SL) -> Conflict!
        df_synthetic.loc[55, 'High'] = 135.0
        df_synthetic.loc[55, 'Low'] = 92.0

        trades = data_cleaner.extract_50ma_trades_from_stock(df_synthetic, "TEST_SYM", "IT")
        trade_50 = [t for t in trades if t['setup_date'] == str(dates[50])[:10]][0]

        self.assertEqual(trade_50['outcome'], 'LOSS', "Same-day conflict in holding must default to LOSS")
        self.assertEqual(trade_50['days_held'], 5)
        self.assertEqual(trade_50['realized_r'], -1.0)
        self.assertEqual(trade_50['exit_price'], 94.0)


if __name__ == "__main__":
    unittest.main()
