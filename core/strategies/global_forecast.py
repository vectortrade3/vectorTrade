"""
VectorTrade - Strategy 2: Global Forecaster Engine
Evaluates market-wide constituent stocks against 7-day high/low forecasts.
Triggers high-conviction signals when predicted gain substantially exceeds predicted drop.
Criteria:
- G_max / abs(D_max) >= 2.0 (Reward-to-Risk ratio >= 2.0)
- G_max >= min_gain_pct (Default: >= 3.5%)
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("VectorTrade.Strategy.GlobalForecast")


class GlobalForecastStrategy(BaseStrategy):
    """
    Global 7-Day High/Low Forecaster Strategy Engine.
    """

    def __init__(self, min_gain_pct: float = 3.5, min_reward_risk_ratio: float = 2.0) -> None:
        super().__init__(strategy_id="GLOBAL_FORECAST", name="Global 7-Day Extremes Forecaster")
        self.min_gain_pct = min_gain_pct
        self.min_reward_risk_ratio = min_reward_risk_ratio

    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Evaluates indicator dataframe and model forecast targets.
        Optional kwargs:
            predicted_gain_pct (float): Predicted 7-day max percentage gain.
            predicted_drop_pct (float): Predicted 7-day max percentage drop.
        """
        if df.empty or len(df) < 20:
            return []

        if "symbol" in kwargs and kwargs["symbol"]:
            symbol = str(kwargs["symbol"])
        elif "symbol" in df.columns and len(df["symbol"]) > 0:
            symbol = str(df["symbol"].iloc[-1])
        else:
            symbol = "UNKNOWN"

        instrument_token = str(kwargs.get("instrument_token", ""))
        valid_date = str(df["date"].iloc[-1])[:10] if "date" in df.columns else ""

        # Fetch model predictions from kwargs if provided
        pred_gain = kwargs.get("predicted_gain_pct")
        pred_drop = kwargs.get("predicted_drop_pct")

        if pred_gain is None or pred_drop is None:
            return []

        pred_gain = float(pred_gain)
        pred_drop = abs(float(pred_drop))

        reward_risk = (pred_gain / pred_drop) if pred_drop > 0 else 999.0

        if pred_gain < self.min_gain_pct or reward_risk < self.min_reward_risk_ratio:
            return []

        # Pricing bracket estimation based on forecasts
        close_now = float(df["close"].iloc[-1])
        high_now = float(df["high"].iloc[-1])
        low_now = float(df["low"].iloc[-1])

        # Entry trigger is break of latest high
        buy_trigger = round(high_now + 1.0, 2)
        # SL positioned at predicted drop buffer or previous bar low
        est_sl = round(close_now * (1.0 - (pred_drop / 100.0)), 2)
        sl_price = min(round(low_now - 1.0, 2), est_sl)
        risk = round(buy_trigger - sl_price, 2)
        target_price = round(buy_trigger + (close_now * (pred_gain / 100.0)), 2)

        feature_row = df.iloc[-1].to_dict()

        signal_doc = {
            "_id": f"{symbol}_{valid_date.replace('-', '')}_{self.strategy_id}",
            "symbol": symbol,
            "instrument_token": instrument_token,
            "strategy_id": self.strategy_id,
            "setup_type": 1,
            "action": "BUY",
            "buy_trigger": buy_trigger,
            "sl_price": sl_price,
            "risk": risk,
            "target_price": target_price,
            "quantity": 1,
            "max_hold_days": 7,
            "status": "ARMED",
            "valid_for_date": valid_date,
            "metrics": {
                "predicted_gain_pct": pred_gain,
                "predicted_drop_pct": pred_drop,
                "reward_risk_ratio": reward_risk,
                "close": close_now
            },
            "features": feature_row
        }

        logger.info(
            "[%s] Signal detected for %s | Gain: +%.2f%%, Drop: -%.2f%%, R/R: %.2f | Target: %.2f",
            self.strategy_id, symbol, pred_gain, pred_drop, reward_risk, target_price
        )
        return [signal_doc]
