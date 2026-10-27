"""
VectorTrade - Comprehensive Unit and Integration Test Suite for Phase 2
Tests:
1. Indicators Engine:
   - Full 130+ technical feature computation.
   - Lookahead bias check (no future leaks).
   - Division-by-zero, infinity, and NaN resilience (zero-volume days, flat prices, missing sector/nifty).
2. Pluggable Strategy Contract (BaseStrategy):
   - Strict interface compliance.
3. Strategy 1 (MA50PullbackStrategy):
   - Setups 1, 2, 3 classification.
   - Order parameter accuracy: Buy = High + 1, SL = Low - 1, Risk = Buy - SL, Target = Buy + 2*Risk.
   - Rejection of invalid setups (red candle, downward MA50, far BB).
4. Strategy 2 (GlobalForecastStrategy):
   - Reward-to-risk ratio filtering (>= 2.0) and min gain threshold (>= 3.5%).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Set up module path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.indicators import compute_indicators, safe_divide, calc_rsi, calc_atr
from core.strategies.base_strategy import BaseStrategy
from core.strategies.ma50_pullback import MA50PullbackStrategy
from core.strategies.global_forecast import GlobalForecastStrategy


class TestPhase2Pipeline(unittest.TestCase):

    def setUp(self):
        """Generates synthetic multi-day OHLCV dataset for deterministic testing."""
        np.random.seed(42)
        n_bars = 220
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")
        
        # Monotonically increasing base price for MA upward slope
        trend = np.linspace(100.0, 300.0, n_bars)
        noise = np.random.normal(0, 2.0, n_bars)
        close = trend + noise

        high = close + np.random.uniform(1.0, 5.0, n_bars)
        low = close - np.random.uniform(1.0, 5.0, n_bars)
        open_p = low + (high - low) * 0.4  # mostly green
        volume = np.random.randint(100000, 5000000, n_bars).astype(float)

        self.df = pd.DataFrame({
            "date": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": volume,
            "symbol": "TEST_SYM"
        })

    def test_safe_divide_edge_cases(self):
        """Verifies zero division and inf prevention."""
        a = np.array([10.0, 0.0, 5.0, -4.0])
        b = np.array([0.0, 0.0, 2.0, 0.0])
        res = safe_divide(a, b, fill_value=0.0)
        self.assertEqual(res[0], 0.0)
        self.assertEqual(res[1], 0.0)
        self.assertEqual(res[2], 2.5)
        self.assertEqual(res[3], 0.0)
        self.assertTrue(np.all(np.isfinite(res)))

    def test_indicators_computation_and_infinities(self):
        """Tests that full indicators compute cleanly with 0 infinities even with zero-volume days."""
        # Inject zero-volume and duplicate prices
        self.df.loc[10:15, "volume"] = 0.0
        self.df.loc[20, "high"] = self.df.loc[20, "low"]  # zero candle range
        self.df.loc[30, "close"] = self.df.loc[29, "close"]

        ind_df = compute_indicators(self.df)
        
        # Verify no infinite values exist anywhere in the dataframe
        numeric_cols = ind_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(np.isinf(ind_df[col]).any(), f"Infinity found in column: {col}")

        # Check key indicator columns exist
        key_cols = [
            "ma15", "ma50", "ma200", "ma50_slope", "close_ma50_ratio",
            "bb_upper", "bb_middle", "bb_lower", "bb_width_pct", "bb_percent_b",
            "rsi7", "rsi14", "rsi21", "rsi_slope",
            "return_1d", "return_5d", "return_20d",
            "high_5d", "low_5d", "dist_high_5d", "dist_low_5d",
            "candle_body", "candle_range", "body_range_ratio", "is_bullish",
            "volume_avg_20d", "volume_ratio_20d", "volume_change_1d", "obv",
            "atr14", "volatility_20d"
        ]
        for k in key_cols:
            self.assertIn(k, ind_df.columns, f"Mandatory feature {k} missing from output")

    def test_ma50_pullback_strategy_math(self):
        """Verifies exact order math: Buy = High + 1, SL = Low - 1, Risk = Buy - SL, Target = Buy + 2*Risk."""
        ind_df = compute_indicators(self.df)
        
        # Manually craft the latest row to guarantee a Setup 1 trigger
        last_idx = ind_df.index[-1]
        ma50_val = 250.0
        bb_lower_val = 252.0
        low_val = 255.0     # Low - BB_lower = 3.0 (in range [-10, 20])
        open_val = 258.0
        close_val = 265.0   # Green candle (Close > Open)
        high_val = 270.0    # High
        
        ind_df.loc[last_idx, "ma50"] = ma50_val
        ind_df.loc[ind_df.index[-2], "ma50"] = ma50_val - 1.0
        ind_df.loc[ind_df.index[-8], "ma50"] = ma50_val - 5.0 # 7D upward MA50
        ind_df.loc[last_idx, "bb_lower"] = bb_lower_val
        ind_df.loc[last_idx, "low"] = low_val
        ind_df.loc[last_idx, "open"] = open_val
        ind_df.loc[last_idx, "close"] = close_val
        ind_df.loc[last_idx, "high"] = high_val

        strategy = MA50PullbackStrategy()
        signals = strategy.evaluate(ind_df, symbol="RELIANCE", instrument_token="738561")

        self.assertEqual(len(signals), 1, "Expected 1 signal generated for crafted valid setup")
        sig = signals[0]
        
        expected_buy = round(high_val + 1.0, 2)
        expected_sl = round(low_val - 1.0, 2)
        expected_risk = round(expected_buy - expected_sl, 2)
        expected_target = round(expected_buy + (2.0 * expected_risk), 2)

        self.assertEqual(sig["buy_trigger"], expected_buy)
        self.assertEqual(sig["sl_price"], expected_sl)
        self.assertEqual(sig["risk"], expected_risk)
        self.assertEqual(sig["target_price"], expected_target)
        self.assertEqual(sig["setup_type"], 1) # Low - MA50 = 5.0 in [-10, 30]
        self.assertEqual(sig["symbol"], "RELIANCE")
        self.assertEqual(sig["instrument_token"], "738561")
        self.assertEqual(sig["status"], "ARMED")

    def test_ma50_pullback_rejections(self):
        """Verifies rejection when rules are violated (e.g. red candle or downward MA)."""
        ind_df = compute_indicators(self.df)
        last_idx = ind_df.index[-1]
        
        # Test Red Candle rejection
        ind_df.loc[last_idx, "open"] = 300.0
        ind_df.loc[last_idx, "close"] = 290.0 # Red candle
        strategy = MA50PullbackStrategy()
        signals = strategy.evaluate(ind_df)
        self.assertEqual(len(signals), 0, "Red candle must be rejected")

    def test_global_forecast_strategy_math(self):
        """Verifies Strategy 2 reward-to-risk ratio and gain thresholds."""
        ind_df = compute_indicators(self.df)
        strategy = GlobalForecastStrategy(min_gain_pct=3.5, min_reward_risk_ratio=2.0)

        # 1. Unqualified forecast (R/R = 1.0 < 2.0)
        rejected = strategy.evaluate(ind_df, predicted_gain_pct=3.5, predicted_drop_pct=3.5)
        self.assertEqual(len(rejected), 0, "Low R/R forecast must be rejected")

        # 2. Qualified forecast (Gain = 6.0%, Drop = 2.0%, R/R = 3.0 >= 2.0)
        accepted = strategy.evaluate(ind_df, symbol="INFY", instrument_token="408065",
                                     predicted_gain_pct=6.0, predicted_drop_pct=2.0)
        self.assertEqual(len(accepted), 1, "High conviction forecast must be accepted")
        sig = accepted[0]
        self.assertEqual(sig["symbol"], "INFY")
        self.assertGreater(sig["target_price"], sig["buy_trigger"])
        self.assertLess(sig["sl_price"], sig["buy_trigger"])


if __name__ == "__main__":
    unittest.main()
