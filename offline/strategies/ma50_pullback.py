"""
VectorTrade - Strategy 1: 50MA Pullback Engine ("Sniper")
Executes the base algorithmic trading rules:
1. 50MA strictly upward trending for at least 7 days.
2. Price proximity to Lower Bollinger Band (-10 <= Low - BB_Lower <= 20).
3. Bullish candle structure (Close > Open).
4. Sub-classification into Setups 1, 2, and 3 based on 50MA relationship.
5. Strict Bracket Geometry: Buy = High + 1, SL = Low - 1, Risk = Buy - SL, Target = Buy + 2*Risk.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("VectorTrade.Strategy.MA50Pullback")


class MA50PullbackStrategy(BaseStrategy):
    """
    50MA Pullback ("Sniper") algorithmic signal generator.
    """

    def __init__(self) -> None:
        super().__init__(strategy_id="50MA_PULLBACK", name="50MA Bollinger Pullback Sniper")

    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Scans OHLCV indicators dataframe and extracts armed candidate setups for the latest bar.
        """
        if df.empty or len(df) < 50:
            return []

        # Ensure required indicator columns are present
        required_cols = ["open", "high", "low", "close", "ma50", "bb_lower"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing mandatory indicator '{col}' for strategy evaluation.")

        if "symbol" in kwargs and kwargs["symbol"]:
            symbol = str(kwargs["symbol"])
        elif "symbol" in df.columns and len(df["symbol"]) > 0:
            symbol = str(df["symbol"].iloc[-1])
        else:
            symbol = "UNKNOWN"

        instrument_token = str(kwargs.get("instrument_token", ""))

        # We evaluate the latest completed bar (index -1)
        # 1. 50MA trend up for at least 7 days: MA50[t] > MA50[t-7] and MA50[t] > MA50[t-1]
        ma50_now = float(df["ma50"].iloc[-1])
        ma50_prev1 = float(df["ma50"].iloc[-2]) if len(df) >= 2 else ma50_now
        ma50_prev7 = float(df["ma50"].iloc[-8]) if len(df) >= 8 else ma50_prev1
        is_ma50_up_7d = (ma50_now > ma50_prev7) and (ma50_now > ma50_prev1)

        # 2. Lower BB is near low of candle: diff is -10 to 20 (Low - BB_Lower)
        low_now = float(df["low"].iloc[-1])
        bb_lower_now = float(df["bb_lower"].iloc[-1])
        dist_low_bb = low_now - bb_lower_now
        is_bb_near_low = (-10.0 <= dist_low_bb <= 20.0)

        # 3. Candle is green (Close > Open)
        close_now = float(df["close"].iloc[-1])
        open_now = float(df["open"].iloc[-1])
        is_green = (close_now > open_now)

        # Base signal composite gate
        is_base_signal = is_ma50_up_7d and is_bb_near_low and is_green

        if not is_base_signal:
            return []

        # 4. Sub-classification into 3 setups based on distance: (Low - MA50)
        dist_low_ma50 = low_now - ma50_now
        setup_type = 0
        if -10.0 <= dist_low_ma50 <= 30.0:
            setup_type = 1  # Confluence Zone
        elif dist_low_ma50 > 30.0:
            setup_type = 2  # Extended Trend
        elif dist_low_ma50 < -10.0:
            setup_type = 3  # Mean-Reverting under 50MA

        if setup_type == 0:
            return []

        # 5. Strict Bracket Calculations
        high_now = float(df["high"].iloc[-1])
        buy_trigger = round(high_now + 1.0, 2)
        sl_price = round(low_now - 1.0, 2)
        risk = round(buy_trigger - sl_price, 2)
        target_price = round(buy_trigger + (2.0 * risk), 2)

        # Signal date
        valid_date = str(df["date"].iloc[-1])[:10] if "date" in df.columns else ""

        # Extract features dictionary for ML brain inference
        feature_row = df.iloc[-1].to_dict()

        signal_doc = {
            "_id": f"{symbol}_{valid_date.replace('-', '')}_{self.strategy_id}",
            "symbol": symbol,
            "instrument_token": instrument_token,
            "strategy_id": self.strategy_id,
            "setup_type": setup_type,
            "action": "BUY",
            "buy_trigger": buy_trigger,
            "sl_price": sl_price,
            "risk": risk,
            "target_price": target_price,
            "quantity": 1,  # Default unit position; overridden by capital manager
            "max_hold_days": 7,
            "status": "ARMED",
            "valid_for_date": valid_date,
            "metrics": {
                "ma50": ma50_now,
                "bb_lower": bb_lower_now,
                "dist_low_bb": dist_low_bb,
                "dist_low_ma50": dist_low_ma50,
                "candle_body": abs(close_now - open_now),
                "is_green": is_green
            },
            "features": feature_row
        }

        logger.info(
            "[%s] Signal detected for %s (Setup %d) | Buy: %.2f, SL: %.2f, Target: %.2f",
            self.strategy_id, symbol, setup_type, buy_trigger, sl_price, target_price
        )
        return [signal_doc]
