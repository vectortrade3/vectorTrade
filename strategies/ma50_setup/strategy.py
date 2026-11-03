"""
VectorTrade - 50MA Setup Strategy Engine
Implements:
1. 50MA trend = up (at least 7 days: MA50[t] > MA50[t-7] and MA50[t] > MA50[t-1])
2. Lower BB near low of candle (-10.0 <= Low - BB_Lower <= 20.0)
3. Candle is green (Close > Open)
4. Sub-classification into Setups 1, 2, and 3
5. Bracket Geometry: Buy = High + 1, SL = Low - 1, Target = Buy + 2*Risk (Strict 1:2.00 R:R)
6. Forward execution checking.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger("VectorTrade.Strategies.MA50Setup")


class MA50SetupStrategy(BaseStrategy):
    """50MA Setup Strategy engine executing base algorithmic trading rules."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.triggers = config.get("triggers", {})
        self.bracket_cfg = self.triggers.get("bracket_geometry", {})

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters master dataset to ONLY trades matching the 50MA setup.
        Used for strategy model training and evaluation.
        """
        if df.empty:
            return df

        # If precomputed base_signal column exists
        if "base_signal" in df.columns:
            df_filtered = df[df["base_signal"] == True].copy()
        else:
            # Dynamically compute from OHLC and indicators
            req = ["close", "open", "low", "high", "ma50", "bb_lower"]
            if not all(col in df.columns for col in req):
                return pd.DataFrame()

            ma50 = df["ma50"]
            prev1 = ma50.shift(1)
            prev7 = ma50.shift(7)
            ma50_up = (ma50 > prev7) & (ma50 > prev1)

            dist_bb = df["low"] - df["bb_lower"]
            bb_near = (dist_bb >= -10.0) & (dist_bb <= 20.0)
            green = df["close"] > df["open"]

            dist_ma = df["low"] - ma50
            setup1 = (dist_ma >= -10.0) & (dist_ma <= 30.0)
            setup2 = dist_ma > 30.0
            setup3 = dist_ma < -10.0

            df_filtered = df[ma50_up & bb_near & green & (setup1 | setup2 | setup3)].copy()

        # Filter out unresolved / pending / no_trigger if outcome column exists
        if "outcome" in df_filtered.columns:
            df_filtered = df_filtered[
                ~df_filtered["outcome"].isin(["No_Trigger", "Pending", "Timeout", np.nan])
            ]

        # Ensure risk > 0 and compute max_r_achieved if mfe is present
        if "risk" in df_filtered.columns and "mfe" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["risk"] > 0]
            df_filtered["max_r_achieved"] = df_filtered["mfe"] / df_filtered["risk"]
            df_filtered = df_filtered.dropna(subset=["max_r_achieved"])

        if "date" in df_filtered.columns:
            df_filtered["date"] = pd.to_datetime(df_filtered["date"])
            df_filtered = df_filtered.sort_values("date").reset_index(drop=True)

        return df_filtered

    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Scans OHLC indicators dataframe for 50MA setups on recent bar(s).
        """
        if df.empty or len(df) < 50:
            return []

        col_map = {c.lower(): c for c in df.columns}
        c_close = col_map.get("close", "Close")
        c_open = col_map.get("open", "Open")
        c_high = col_map.get("high", "High")
        c_low = col_map.get("low", "Low")
        c_ma50 = col_map.get("ma50", "MA50")
        c_bb_lower = col_map.get("bb_lower", "BB_Lower")

        symbol = str(kwargs.get("symbol", df.get("symbol", ["UNKNOWN"])[-1] if "symbol" in df.columns else "UNKNOWN"))
        n = len(df)
        results = []

        # Check last bar
        i = n - 1
        row = df.iloc[i]
        prev1 = df.iloc[i - 1] if i >= 1 else row
        prev7 = df.iloc[i - 7] if i >= 7 else prev1

        ma50_now = float(row[c_ma50])
        ma50_up_7d = (ma50_now > float(prev7[c_ma50])) and (ma50_now > float(prev1[c_ma50]))

        low = float(row[c_low])
        high = float(row[c_high])
        close = float(row[c_close])
        open_p = float(row[c_open])
        bb_lower = float(row[c_bb_lower])

        dist_low_bb = low - bb_lower
        bb_near_low = (-10.0 <= dist_low_bb <= 20.0)
        is_green = (close > open_p)

        if not (ma50_up_7d and bb_near_low and is_green):
            return []

        # Sub-classification into 3 setups
        dist_ma50 = low - ma50_now
        if -10.0 <= dist_ma50 <= 30.0:
            setup_num = 1
            setup_label = "50MA Setup 1 (Confluence Zone)"
        elif dist_ma50 > 30.0:
            setup_num = 2
            setup_label = "50MA Setup 2 (Extended Trend)"
        else:
            setup_num = 3
            setup_label = "50MA Setup 3 (Mean-Reverting)"

        buy_price = round(high + 1.0, 2)
        sl_price = round(low - 1.0, 2)
        risk = round(buy_price - sl_price, 2)
        target_price = round(buy_price + (2.0 * risk), 2)
        loss_pct = round((risk / buy_price) * 100, 2)
        profit_pct = round(((target_price - buy_price) / buy_price) * 100, 2)

        results.append({
            "symbol": symbol,
            "strategy": self.name,
            "strategy_id": self.strategy_id,
            "setup_num": setup_num,
            "setup_subclass": setup_label,
            "action": "BUY",
            "entry_price": buy_price,
            "sl_price": sl_price,
            "target_price": target_price,
            "risk": risk,
            "loss_pct": loss_pct,
            "profit_pct": profit_pct,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0,
            "dist_low_bb": round(dist_low_bb, 2),
            "dist_low_ma50": round(dist_ma50, 2),
            "ma50": round(ma50_now, 2),
            "date": str(row.get("date", row.get("Date", "")))[:10]
        })

        return results
