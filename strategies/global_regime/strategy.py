"""
VectorTrade - Global Market Regime Strategy Engine
Evaluates market-wide 7-day maximum gain and maximum drop asymmetric forecasts.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger("VectorTrade.Strategies.GlobalRegime")


class GlobalRegimeStrategy(BaseStrategy):
    """Global Market Regime Forecaster evaluating 7-day asymmetric upside."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.triggers = config.get("triggers", {})
        self.min_gain_pct = float(self.triggers.get("min_gain_pct", 4.0))
        self.max_drop_pct = float(self.triggers.get("max_drop_pct", 4.0))
        self.gain_drop_ratio = float(self.triggers.get("gain_drop_ratio", 2.0))

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For Global Regime, returns all market data (no setup filter),
        ensuring chronological sort and dropna on target columns if present.
        """
        if df.empty:
            return df

        df_out = df.copy()
        target_gain = self.training_config.get("target_gain", "max_gain_7d")
        target_drop = self.training_config.get("target_drop", "max_loss_7d")

        if target_gain in df_out.columns and target_drop in df_out.columns:
            df_out = df_out.dropna(subset=[target_gain, target_drop])

        if "date" in df_out.columns:
            df_out["date"] = pd.to_datetime(df_out["date"])
            df_out = df_out.sort_values("date").reset_index(drop=True)

        return df_out

    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Evaluates indicator row and model forecasts for favorable 7-day asymmetric upside.
        """
        if df.empty or len(df) < 20:
            return []

        symbol = str(kwargs.get("symbol", df.get("symbol", ["UNKNOWN"])[-1] if "symbol" in df.columns else "UNKNOWN"))
        pred_gain = kwargs.get("predicted_gain_pct")
        pred_drop = kwargs.get("predicted_drop_pct")

        if pred_gain is None or pred_drop is None:
            return []

        pred_gain = float(pred_gain)
        pred_drop = abs(float(pred_drop))

        if pred_gain < self.min_gain_pct or pred_drop > self.max_drop_pct:
            return []

        ratio = (pred_gain / pred_drop) if pred_drop > 0 else 99.0
        if ratio < self.gain_drop_ratio:
            return []

        row = df.iloc[-1]
        col_map = {c.lower(): c for c in df.columns}
        c_close = col_map.get("close", "Close")
        c_high = col_map.get("high", "High")
        c_low = col_map.get("low", "Low")

        close = float(row[c_close])
        high = float(row[c_high])
        low = float(row[c_low])

        risk_amount = max(close * (pred_drop / 100.0), close * 0.015)
        entry_price = round(close, 2)
        sl_price = round(max(0.1, entry_price - risk_amount), 2)
        target_price = round(entry_price + (2.0 * risk_amount), 2)

        loss_pct = round(((entry_price - sl_price) / entry_price) * 100, 2)
        profit_pct = round(((target_price - entry_price) / entry_price) * 100, 2)

        return [{
            "symbol": symbol,
            "strategy": self.name,
            "strategy_id": self.strategy_id,
            "setup_num": 0,
            "setup_subclass": "7-Day High Momentum Breakout",
            "action": "STRONG BUY" if pred_gain >= 6.0 else "BUY",
            "entry_price": entry_price,
            "sl_price": sl_price,
            "target_price": target_price,
            "risk": round(risk_amount, 2),
            "loss_pct": loss_pct,
            "profit_pct": profit_pct,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0,
            "predicted_gain_pct": round(pred_gain, 2),
            "predicted_drop_pct": round(pred_drop, 2),
            "date": str(row.get("date", row.get("Date", "")))[:10]
        }]
