"""
VectorTrade - Base Strategy Abstract Interface
Defines the uniform protocol for all algorithmic strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """Abstract interface implemented by all strategy engines in the strategies/ catalog."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.strategy_id = config.get("strategy_id", "unknown_strategy")
        self.name = config.get("name", self.strategy_id)
        self.display_name = config.get("display_name", self.name)
        self.enabled = config.get("enabled", True)
        self.strategy_type = config.get("type", "setup_filter")
        self.triggers = config.get("triggers", {})
        self.training_config = config.get("training", {})
        self.inference_config = config.get("inference", {})
        self.artifacts = config.get("artifacts", {})

    @abstractmethod
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters input dataset to rows matching this strategy's specific setup criteria."""
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        """Evaluates latest market bar(s) for actionable setups according to strategy triggers."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata and trigger specification for this strategy."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "type": self.strategy_type,
            "triggers": self.triggers,
            "artifacts": self.artifacts
        }
