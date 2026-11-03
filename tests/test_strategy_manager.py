"""
Unit and integration tests for VectorTrade Strategy Manager & Modular Architecture.
Validates:
1. StrategyManager discovery: finds active strategies with valid config.json.
2. Skipping logic: skips folders without config.json or marked disabled.
3. 50MA setup data isolation: filter_dataset('ma50_setup', df) isolates ONLY 50MA setup trades.
4. Global regime data retention: filter_dataset('global_regime', df) keeps all market data.
5. Multi-strategy extensibility: dynamic loading of new strategies.
"""

import os
import json
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from strategies.manager import StrategyManager, STRATEGIES_DIR
from strategies.base import BaseStrategy


def test_strategy_discovery():
    """StrategyManager discovers both ma50_setup and global_regime."""
    StrategyManager.initialize(force_reload=True)
    strats = StrategyManager.get_active_strategies()
    assert "ma50_setup" in strats
    assert "global_regime" in strats

    ma50 = StrategyManager.get_strategy("ma50_setup")
    assert ma50 is not None
    assert ma50.strategy_type == "setup_filter"
    assert "triggers" in ma50.config
    assert "bracket_geometry" in ma50.config["triggers"]

    glob = StrategyManager.get_strategy("global_regime")
    assert glob is not None
    assert glob.strategy_type == "global_regime"


def test_strategy_skip_if_no_config():
    """A directory without config.json is gracefully skipped without error."""
    test_dir = STRATEGIES_DIR / "temp_unconfigured_strategy"
    test_dir.mkdir(exist_ok=True)
    try:
        StrategyManager.initialize(force_reload=True)
        strats = StrategyManager.get_active_strategies()
        assert "temp_unconfigured_strategy" not in strats
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_strategy_skip_if_disabled():
    """A strategy marked enabled: false in config.json is gracefully skipped."""
    test_dir = STRATEGIES_DIR / "temp_disabled_strategy"
    test_dir.mkdir(exist_ok=True)
    cfg_file = test_dir / "config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump({"strategy_id": "temp_disabled_strategy", "name": "Disabled", "enabled": False}, f)

    try:
        StrategyManager.initialize(force_reload=True)
        strats = StrategyManager.get_active_strategies()
        assert "temp_disabled_strategy" not in strats
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_ma50_dataset_filtering():
    """StrategyManager.filter_dataset('ma50_setup', df) isolates ONLY 50MA setup trades."""
    mock_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "base_signal": [True, False, True, True],
        "outcome": ["Target", "Target", "No_Trigger", "SL"],
        "risk": [10.0, 5.0, 10.0, 15.0],
        "mfe": [20.0, 10.0, 0.0, -15.0],
        "close": [100.0, 200.0, 300.0, 400.0],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    })

    filtered = StrategyManager.filter_dataset("ma50_setup", mock_df)
    # Row B rejected (base_signal == False)
    # Row C rejected (outcome == 'No_Trigger')
    # Rows A and D accepted
    assert len(filtered) == 2
    assert list(filtered["symbol"]) == ["A", "D"]
    assert "max_r_achieved" in filtered.columns


def test_global_regime_dataset_filtering():
    """StrategyManager.filter_dataset('global_regime', df) retains all rows."""
    mock_df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "base_signal": [True, False, False],
        "max_gain_7d": [5.0, 2.0, 8.0],
        "max_loss_7d": [1.0, 3.0, 2.0],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"]
    })

    filtered = StrategyManager.filter_dataset("global_regime", mock_df)
    assert len(filtered) == 3
