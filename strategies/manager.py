"""
VectorTrade - Strategy Manager & Discovery Engine
Discovers all strategy folders in strategies/, reads their config.json,
skips folders without config or marked disabled, and exposes dynamic
filtering, training, and inference interfaces.
"""

from __future__ import annotations
import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger("VectorTrade.Strategies.Manager")

STRATEGIES_DIR = Path(__file__).resolve().parent


class StrategyManager:
    """Dynamic discovery, registry, and execution manager for trading strategies."""

    _registry: Dict[str, BaseStrategy] = {}
    _configs: Dict[str, Dict[str, Any]] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, force_reload: bool = False) -> None:
        """Discovers and initializes all valid strategy folders."""
        if cls._initialized and not force_reload:
            return

        cls._registry.clear()
        cls._configs.clear()

        if not STRATEGIES_DIR.exists():
            logger.warning(f"Strategies directory {STRATEGIES_DIR} does not exist.")
            cls._initialized = True
            return

        for item in sorted(STRATEGIES_DIR.iterdir()):
            if not item.is_dir() or item.name.startswith("__") or item.name.startswith("."):
                continue

            config_file = item / "config.json"
            if not config_file.exists():
                logger.info(f"Skipping directory '{item.name}': No config.json found.")
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Skipping strategy '{item.name}': Corrupt config.json ({e}).")
                continue

            if not config.get("enabled", True):
                logger.info(f"Strategy '{item.name}' is disabled in config.json. Skipping.")
                continue

            strat_id = config.get("strategy_id", item.name)
            cls._configs[strat_id] = config

            # Dynamically import strategy.py if available
            strategy_py = item / "strategy.py"
            instance = None

            if strategy_py.exists():
                try:
                    module_name = f"strategies.{item.name}.strategy"
                    spec = importlib.util.spec_from_file_location(module_name, strategy_py)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = mod
                        spec.loader.exec_module(mod)

                        # Find subclass of BaseStrategy
                        for attr_name in dir(mod):
                            attr = getattr(mod, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseStrategy)
                                and attr is not BaseStrategy
                            ):
                                instance = attr(config)
                                break
                except Exception as e:
                    logger.warning(f"Could not load custom strategy.py for '{strat_id}': {e}. Using fallback.")

            if instance is None:
                # Built-in fallback based on strategy_id or type
                strat_type = config.get("type", "setup_filter")
                if "50ma" in strat_id.lower() or strat_type == "setup_filter":
                    from strategies.ma50_setup.strategy import MA50SetupStrategy
                    instance = MA50SetupStrategy(config)
                else:
                    from strategies.global_regime.strategy import GlobalRegimeStrategy
                    instance = GlobalRegimeStrategy(config)

            cls._registry[strat_id] = instance
            logger.info(f"Registered active strategy: '{strat_id}' ({config.get('name')})")

        cls._initialized = True

    @classmethod
    def get_active_strategies(cls) -> Dict[str, BaseStrategy]:
        """Returns map of all active strategies: {strategy_id: strategy_instance}."""
        cls.initialize()
        return cls._registry

    @classmethod
    def get_strategy(cls, strategy_id: str) -> Optional[BaseStrategy]:
        """Returns strategy instance by ID, or None if not found."""
        cls.initialize()
        return cls._registry.get(strategy_id)

    @classmethod
    def get_config(cls, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Returns strategy configuration dictionary by ID."""
        cls.initialize()
        return cls._configs.get(strategy_id)

    @classmethod
    def list_strategies(cls) -> List[Dict[str, Any]]:
        """Returns list of metadata summaries for all active strategies."""
        cls.initialize()
        return [strat.get_info() for strat in cls._registry.values()]

    @classmethod
    def filter_dataset(cls, strategy_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a dataset specifically for the given strategy.
        If strategy requires setup filtering (e.g. 50MA), filters to strictly
        those triggered setups. If strategy requires all data, returns full dataset.
        """
        cls.initialize()
        strat = cls._registry.get(strategy_id)
        if strat is None:
            logger.warning(f"Strategy '{strategy_id}' not found in registry. Returning full dataframe.")
            return df
        return strat.filter_data(df)
