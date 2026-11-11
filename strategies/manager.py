"""
VectorTrade - Strategy Manager & Discovery Engine
Discovers all strategy folders in strategies/, reads their config.json,
tracks active and deselected states, and exposes dynamic filtering,
training, evaluation, and selection interfaces.
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
    """Dynamic discovery, registry, selection, and execution manager for trading strategies."""

    _all_strategies: Dict[str, BaseStrategy] = {}
    _registry: Dict[str, BaseStrategy] = {}  # Active / enabled strategies only
    _configs: Dict[str, Dict[str, Any]] = {}
    _folders: Dict[str, Path] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, force_reload: bool = False) -> None:
        """Discovers and initializes all valid strategy folders."""
        if cls._initialized and not force_reload:
            return

        cls._all_strategies.clear()
        cls._registry.clear()
        cls._configs.clear()
        cls._folders.clear()

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

            strat_id = config.get("strategy_id", item.name)
            cls._configs[strat_id] = config
            cls._folders[strat_id] = item
            is_enabled = bool(config.get("enabled", True))

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
                elif "global" in strat_id.lower() or strat_type == "global_regime":
                    from strategies.global_regime.strategy import GlobalRegimeStrategy
                    instance = GlobalRegimeStrategy(config)
                else:
                    # Adaptive generic wrapper for future custom strategies
                    instance = _GenericCustomStrategy(config)

            instance.enabled = is_enabled
            cls._all_strategies[strat_id] = instance

            if is_enabled:
                cls._registry[strat_id] = instance
                logger.info(f"Registered active strategy: '{strat_id}' ({config.get('name')})")
            else:
                logger.info(f"Discovered strategy (deselected): '{strat_id}' ({config.get('name')})")

        cls._initialized = True

    @classmethod
    def get_active_strategies(cls) -> Dict[str, BaseStrategy]:
        """Returns map of all currently active (selected) strategies: {strategy_id: strategy_instance}."""
        cls.initialize()
        return {sid: strat for sid, strat in cls._all_strategies.items() if strat.enabled}

    @classmethod
    def get_all_strategies(cls) -> Dict[str, BaseStrategy]:
        """Returns map of all discovered strategies (both selected and deselected)."""
        cls.initialize()
        return cls._all_strategies

    @classmethod
    def get_strategy(cls, strategy_id_or_name: str) -> Optional[BaseStrategy]:
        """Returns strategy instance by ID or query name, or None if not found."""
        cls.initialize()
        sid = cls.resolve_strategy_id(strategy_id_or_name)
        if sid and sid in cls._all_strategies:
            return cls._all_strategies[sid]
        return cls._all_strategies.get(strategy_id_or_name)

    @classmethod
    def get_config(cls, strategy_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Returns strategy configuration dictionary by ID or name."""
        cls.initialize()
        sid = cls.resolve_strategy_id(strategy_id_or_name)
        if sid and sid in cls._configs:
            return cls._configs[sid]
        return cls._configs.get(strategy_id_or_name)

    @classmethod
    def resolve_strategy_id(cls, query: str) -> Optional[str]:
        """
        Resolves a user-friendly name, alias, or display name to the canonical strategy_id.
        Examples:
          '50 MA Setup' -> 'ma50_setup'
          '50ma' -> 'ma50_setup'
          'Global Regime' -> 'global_regime'
        """
        cls.initialize()
        if not query:
            return None
        q = str(query).strip().lower()

        # Direct match on strategy_id
        if q in cls._all_strategies:
            return q

        # Case-insensitive direct match
        for s_id in cls._all_strategies:
            if s_id.lower() == q:
                return s_id

        # Heuristic aliases for existing strategies
        if "50" in q and ("ma" in q or "setup" in q or "moving" in q):
            if "ma50_setup" in cls._all_strategies:
                return "ma50_setup"
        if "global" in q or "regime" in q:
            if "global_regime" in cls._all_strategies:
                return "global_regime"

        # Match by strategy name or display_name
        for s_id, strat in cls._all_strategies.items():
            if q == strat.name.lower() or q == strat.display_name.lower():
                return s_id
            if q in strat.name.lower() or q in strat.display_name.lower():
                return s_id

        return None

    @classmethod
    def is_strategy_enabled(cls, strategy_id_or_name: str) -> bool:
        """
        Returns True if the strategy is registered and currently selected/enabled,
        False otherwise. Supports dual combinations like '50 MA + Global Regime'.
        """
        cls.initialize()
        if not strategy_id_or_name:
            return False

        q_clean = str(strategy_id_or_name).strip()

        # Combo strategy check (e.g. "50 MA + Global Regime")
        if "+" in q_clean:
            parts = [p.strip() for p in q_clean.split("+")]
            # Active if ALL participating strategies are enabled
            return all(cls.is_strategy_enabled(p) for p in parts)

        sid = cls.resolve_strategy_id(q_clean)
        if sid and sid in cls._all_strategies:
            return bool(cls._all_strategies[sid].enabled)

        # Fallback query matching
        for s_id, strat in cls._all_strategies.items():
            if (
                s_id.lower() == q_clean.lower()
                or strat.name.lower() == q_clean.lower()
                or strat.display_name.lower() == q_clean.lower()
            ):
                return bool(strat.enabled)

        return False

    @classmethod
    def set_strategy_enabled(cls, strategy_id_or_name: str, enabled: bool) -> bool:
        """
        Toggles a strategy's selected state:
        - Updates in-memory registry.
        - Persists to that strategy's config.json.
        - Persists to data_store/strategy_settings.json.
        - Returns True if found and updated, False if strategy not found.
        """
        cls.initialize()
        sid = cls.resolve_strategy_id(strategy_id_or_name)
        if not sid or sid not in cls._all_strategies:
            logger.warning(f"Cannot toggle strategy '{strategy_id_or_name}': Strategy not registered.")
            return False

        strat = cls._all_strategies[sid]
        strat.enabled = bool(enabled)
        if sid in cls._configs:
            cls._configs[sid]["enabled"] = bool(enabled)

        if bool(enabled):
            cls._registry[sid] = strat
        else:
            cls._registry.pop(sid, None)

        # 1. Update config.json in strategy folder
        folder = cls._folders.get(sid)
        if folder and folder.exists():
            cfg_file = folder / "config.json"
            try:
                if cfg_file.exists():
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["enabled"] = bool(enabled)
                    with open(cfg_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to persist enabled state in {cfg_file}: {e}")

        # 2. Update data_store/strategy_settings.json
        try:
            settings_file = STRATEGIES_DIR.parent / "data_store" / "strategy_settings.json"
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, "w", encoding="utf-8") as sf:
                json.dump({s_id: s.enabled for s_id, s in cls._all_strategies.items()}, sf, indent=2)
        except Exception as e:
            logger.debug(f"Could not write strategy_settings.json: {e}")

        logger.info(f"Strategy '{sid}' ({strat.name}) state updated: enabled={enabled}")
        return True

    @classmethod
    def list_strategies(cls, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """
        Returns list of metadata summaries for strategies.
        If include_disabled=True, returns all discovered strategies with their active status.
        If include_disabled=False, returns only active strategies.
        """
        cls.initialize()
        source = cls._all_strategies if include_disabled else cls._registry
        results = []
        for strat in source.values():
            info = strat.get_info()
            cfg = cls._configs.get(strat.strategy_id, {})
            info["version"] = cfg.get("version", "1.0.0")
            info["description"] = cfg.get("description", "")
            results.append(info)
        return results

    @classmethod
    def filter_dataset(cls, strategy_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a dataset specifically for the given strategy.
        If strategy requires setup filtering (e.g. 50MA), filters to strictly
        those triggered setups. If strategy requires all data, returns full dataset.
        """
        cls.initialize()
        sid = cls.resolve_strategy_id(strategy_id) or strategy_id
        strat = cls._all_strategies.get(sid) or cls._registry.get(sid)
        if strat is None:
            logger.warning(f"Strategy '{strategy_id}' not found in registry. Returning full dataframe.")
            return df
        return strat.filter_data(df)


class _GenericCustomStrategy(BaseStrategy):
    """
    Adaptive fallback engine for new/future strategies added to strategies/
    that do not define their own custom class.
    """

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.training_config.get("target_column")
        if target_col and target_col in df.columns:
            return df.dropna(subset=[target_col])
        return df

    def evaluate(self, df: pd.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        # Future custom strategy evaluate hook
        return []
