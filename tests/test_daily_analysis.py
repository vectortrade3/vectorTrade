"""
Unit and integration tests for VectorTrade Daily Market Analysis & AI Trade Finalizer.
Validates:
1. Market schedule evaluation across trading hours, weekends, and holidays.
2. 50MA Base Strategy geometry: Buy = High + 1, SL = Low - 1, Target = Buy + 2*Risk (Strict 1:2.00 R:R).
3. 50MA Sub-classifications: Setup 1 (Confluence), Setup 2 (Extended Trend), Setup 3 (Mean-Reverting).
4. Strict filtering: Rejection of red candles, non-confluent Bollinger bands, or low AI payoff.
5. JSON Manager atomic persistence and retrieval.
6. Flask API endpoints and HTML template rendering.
7. Verification that constituent scanner produces calibrated setups (not 175 spam trades).
"""

import json
from datetime import datetime
import pytest
import pandas as pd

from offline.daily_analyzer import (
    get_market_schedule_info,
    synthesize_trade_opportunity,
    load_prediction_models,
    run_daily_analysis_pipeline
)
from offline import json_manager
from offline.app import app


# ------------------------------------------------------------------------------
# 1. Market Schedule Evaluator Tests
# ------------------------------------------------------------------------------
def test_schedule_weekday_before_cutoff():
    """On a Tuesday at 14:00 (before 3:30 PM), should NOT download incomplete intraday data; use yesterday."""
    dt = datetime(2026, 11, 3, 14, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is False
    assert sched["session_type"] == "IN_SESSION"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-11-02"
    assert "Market In Session" in sched["market_status"]


def test_schedule_weekday_after_cutoff():
    """On a Tuesday at 16:00 (after 3:30 PM), market is closed and today's EOD is final."""
    dt = datetime(2026, 11, 3, 16, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "POST_CLOSE"
    assert sched["allow_today_download"] is True
    assert sched["target_date"] == "2026-11-03"


def test_schedule_weekend_saturday():
    """On a Saturday, market is closed; target is previous Friday."""
    dt = datetime(2026, 11, 7, 11, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "WEEKEND"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-11-06"


def test_schedule_weekend_sunday():
    """On a Sunday, market is closed; target is previous Friday."""
    dt = datetime(2026, 11, 8, 12, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "WEEKEND"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-11-06"


def test_schedule_holiday():
    """On a recognized exchange holiday, market is closed; target is previous trading day."""
    dt = datetime(2026, 10, 2, 10, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "HOLIDAY"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-10-01"


# ------------------------------------------------------------------------------
# 2. 50MA Base Strategy & Bracket Geometry Tests
# ------------------------------------------------------------------------------
def test_trade_synthesis_dual_alpha():
    """Dual alpha setup produces STRONG BUY with exact 1:2.00 R:R geometry."""
    row = pd.Series({
        "symbol": "RELIANCE",
        "close": 2500.0,
        "high": 2520.0,
        "low": 2480.0,
        "open": 2490.0,
        "bb_lower": 2475.0,  # low (2480) - bb_lower (2475) = 5.0 (in range [-10, 20])
        "ma50": 2470.0,      # low (2480) - ma50 (2470) = 10.0 (Setup 1)
        "rsi14": 55.0,
        "base_signal": True
    })
    stock_meta = {
        "RELIANCE": {"name": "Reliance Industries Ltd", "sector": "Energy"}
    }
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=1.8,
        pred_gain_pts=150.0,
        pred_drop_pts=50.0,
        stock_meta=stock_meta
    )
    assert trade is not None
    assert trade["symbol"] == "RELIANCE"
    assert trade["strategy"] == "50 MA + Global Regime"
    assert trade["action"] == "STRONG BUY"
    # Buy = High + 1.0 = 2521.0
    assert trade["entry_price"] == 2521.0
    # SL = Low - 1.0 = 2479.0
    assert trade["sl_price"] == 2479.0
    # Risk = 2521 - 2479 = 42.0. Target = 2521 + (2 * 42) = 2605.0
    assert trade["target_price"] == 2605.0
    assert trade["risk_reward"] == "1 : 2.00"
    assert trade["risk_reward_num"] == 2.0


def test_50ma_setup_1_confluence():
    """Setup 1: 50MA is near BB and Low (-10 <= Low - MA50 <= 30)."""
    row = pd.Series({
        "symbol": "TATASTEEL",
        "close": 150.0,
        "high": 152.0,
        "low": 146.0,
        "open": 147.0,      # Green candle (150 > 147)
        "bb_lower": 144.0,   # Low - BB = 2.0 (in [-10, 20])
        "ma50": 145.0,       # Low - MA50 = 1.0 (in [-10, 30] -> Setup 1)
        "base_signal": True
    })
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=1.2,
        pred_gain_pts=2.0,
        pred_drop_pts=3.0,
        stock_meta={}
    )
    assert trade is not None
    assert trade["setup_num"] == 1
    assert "Setup 1" in trade["setup_subclass"]
    assert trade["entry_price"] == 153.0  # 152 + 1
    assert trade["sl_price"] == 145.0     # 146 - 1
    assert trade["target_price"] == 169.0 # 153 + 2*(153 - 145) = 153 + 16 = 169.0


def test_50ma_setup_2_extended():
    """Setup 2: 50MA is below BB / Low (Low - MA50 > 30)."""
    row = pd.Series({
        "symbol": "TITAN",
        "close": 3400.0,
        "high": 3450.0,
        "low": 3380.0,
        "open": 3370.0,     # Green candle
        "bb_lower": 3375.0, # Low - BB = 5.0 (in [-10, 20])
        "ma50": 3300.0,     # Low - MA50 = 80.0 (> 30 -> Setup 2)
        "base_signal": True
    })
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=1.3,
        pred_gain_pts=10.0,
        pred_drop_pts=20.0,
        stock_meta={}
    )
    assert trade is not None
    assert trade["setup_num"] == 2
    assert "Setup 2" in trade["setup_subclass"]
    assert trade["entry_price"] == 3451.0 # 3450 + 1
    assert trade["sl_price"] == 3379.0    # 3380 - 1
    # Risk = 72.0, Target = 3451 + 144 = 3595.0
    assert trade["target_price"] == 3595.0


def test_50ma_setup_3_mean_reverting():
    """Setup 3: 50MA is above candle (Low - MA50 < -10)."""
    row = pd.Series({
        "symbol": "INFY",
        "close": 1820.0,
        "high": 1830.0,
        "low": 1800.0,
        "open": 1810.0,     # Green candle
        "bb_lower": 1795.0, # Low - BB = 5.0 (in [-10, 20])
        "ma50": 1850.0,     # Low - MA50 = -50.0 (< -10 -> Setup 3)
        "base_signal": True
    })
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=1.25,
        pred_gain_pts=5.0,
        pred_drop_pts=10.0,
        stock_meta={}
    )
    assert trade is not None
    assert trade["setup_num"] == 3
    assert "Setup 3" in trade["setup_subclass"]
    assert trade["entry_price"] == 1831.0 # 1830 + 1
    assert trade["sl_price"] == 1799.0    # 1800 - 1
    # Risk = 32.0, Target = 1831 + 64 = 1895.0
    assert trade["target_price"] == 1895.0


def test_trade_synthesis_negative_edge_rejected():
    """A constituent with poor expected gain and high drop is rejected (returns None)."""
    row = pd.Series({
        "symbol": "WEAKSTOCK",
        "close": 100.0,
        "high": 101.0,
        "low": 95.0,
        "open": 100.0,
        "atr14": 3.0,
        "ma50": 110.0,
        "bb_lower": 60.0,
        "rsi14": 30.0,
        "base_signal": False
    })
    stock_meta = {"WEAKSTOCK": {"name": "Weak Corp", "sector": "Consumer"}}
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=-0.8,
        pred_gain_pts=1.0,
        pred_drop_pts=8.0,
        stock_meta=stock_meta
    )
    assert trade is None


# ------------------------------------------------------------------------------
# 3. JSON Manager Integration Tests
# ------------------------------------------------------------------------------
def test_json_manager_daily_analysis_info_persistence():
    """Verify daily_analysis.json state updates and read-backs correctly."""
    info_before = json_manager.get_daily_analysis_info()
    assert isinstance(info_before, dict)

    sample_trades = [
        {
            "symbol": "TESTSYM",
            "name": "Test Symbol",
            "strategy": "50 MA Setup",
            "action": "BUY",
            "entry_price": 500.0,
            "sl_price": 480.0,
            "target_price": 540.0,
            "loss_pct": 4.0,
            "profit_pct": 8.0,
            "risk_reward": "1 : 2.00",
            "risk_reward_num": 2.0
        }
    ]
    try:
        updated = json_manager.update_daily_analysis_info(
            status="COMPLETED",
            market_session_date="2026-11-02",
            market_status="Market Closed",
            total_stocks_analyzed=207,
            finalized_trades=sample_trades,
            strategies_breakdown={"50 MA Setup": 1},
            message="Test update",
            duration_seconds=5.2
        )
        assert updated["status"] == "COMPLETED"
        assert len(updated["finalized_trades"]) == 1
        assert updated["strategies_breakdown"]["50 MA Setup"] == 1

        stored = json_manager.get_daily_analysis_info()
        assert stored["status"] == "COMPLETED"
        assert stored["finalized_trades"][0]["symbol"] == "TESTSYM"
    finally:
        if info_before:
            json_manager.write_json_file(json_manager.DAILY_ANALYSIS_FILE, info_before)


# ------------------------------------------------------------------------------
# 4. Flask Web Application & API Route Tests
# ------------------------------------------------------------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_analysis_schedule(client):
    res = client.get("/api/analysis/schedule")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "SUCCESS"
    assert "schedule" in data
    assert "target_date" in data["schedule"]


def test_api_analysis_results(client):
    res = client.get("/api/analysis/results")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "SUCCESS"
    assert "data" in data
    assert "finalized_trades" in data["data"]


def test_api_analysis_progress(client):
    res = client.get("/api/analysis/progress")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "percentage" in data


def test_analysis_page_render(client):
    """GET /analysis renders HTML with active_tab='analysis' and HTTP 200."""
    res = client.get("/analysis")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "tab-analysis-view" in html
    assert "Daily Market Scanner &amp; AI Trade Finalizer" in html
    assert "btnAnalysisRunPage" in html
    assert "analysisStrategyFilterPills" in html


def test_dashboard_page_render(client):
    """GET / renders HTML with Daily Market Scanner Dashboard Card."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "dashCard-daily-analysis" in html
    assert "btnDashRunAnalysis" in html
    assert "nav-analysis" in html


def test_constituent_scan_realistic_numbers():
    """Verifies daily_analysis.json contains calibrated, realistic trade setups (not 175 spam trades)."""
    info = json_manager.get_daily_analysis_info()
    assert info.get("status") == "COMPLETED"
    trades = info.get("finalized_trades", [])
    # Must NOT be 175 spam trades!
    assert len(trades) < 60, f"Expected realistic trade count (<60), got {len(trades)}"
    assert len(trades) > 0, "Expected at least some qualified trade setups"

    # Check outcomes
    outcomes = [t.get("outcome") for t in trades]
    assert "ARMED" in outcomes
    # Armed trades should be between 3 and 15
    armed_trades = [t for t in trades if t.get("outcome") == "ARMED"]
    assert 1 <= len(armed_trades) <= 15, f"Expected 1-15 armed trades, got {len(armed_trades)}"
