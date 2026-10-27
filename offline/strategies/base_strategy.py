"""
VectorTrade - Abstract Strategy Interface
Defines the standardized contract for pluggable trading strategies.
All concrete strategies must inherit from BaseStrategy and implement .evaluate(df).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract Base Class for VectorTrade Strategy Engines.
    Provides uniform execution, risk bracket calculation, and audit serialization.
    """

    def __init__(self, strategy_id: str, name: str) -> None:
        self.strategy_id = strategy_id
        self.name = name

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Evaluates market data for candidate trade setups.

        Args:
            df: Historical daily OHLCV dataframe with computed technical indicators.
            **kwargs: Optional context (e.g. instrument token, current symbol).

        Returns:
            List of dictionary setup payloads ready for MongoDB 'signals_watchlist'.
            Schema:
            {
                "symbol": str,
                "strategy_id": str,
                "setup_type": int,
                "action": "BUY" | "SELL",
                "buy_trigger": float,
                "sl_price": float,
                "risk": float,
                "target_price": float,
                "valid_for_date": str (YYYY-MM-DD),
                "features": dict,
                "metadata": dict
            }
        """
        pass

    def get_info(self) -> Dict[str, str]:
        """Returns metadata about the strategy engine."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name
        }
