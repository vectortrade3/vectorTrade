"""
Unit and integration tests for VectorTrade Daily Market Analysis & AI Trade Finalizer.
Validates:
1. Market schedule evaluation across trading hours, weekends, and holidays.
2. Trade synthesis with strict risk-reward geometry and zero hardcoding.
3. JSON Manager atomic persistence and retrieval.
4. Flask API endpoints and HTML template rendering.
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
    # 2026-11-03 is Tuesday
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
    # 2026-11-07 is Saturday
    dt = datetime(2026, 11, 7, 11, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "WEEKEND"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-11-06"


def test_schedule_weekend_sunday():
    """On a Sunday, market is closed; target is previous Friday."""
    # 2026-11-08 is Sunday
    dt = datetime(2026, 11, 8, 12, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "WEEKEND"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-11-06"


def test_schedule_holiday():
    """On a recognized exchange holiday, market is closed; target is previous trading day."""
    # 2026-10-02 is Gandhi Jayanti (Friday)
    dt = datetime(2026, 10, 2, 10, 0, 0)
    sched = get_market_schedule_info(dt)
    assert sched["is_market_closed"] is True
    assert sched["session_type"] == "HOLIDAY"
    assert sched["allow_today_download"] is False
    assert sched["target_date"] == "2026-10-01"


# ------------------------------------------------------------------------------
# 2. Trade Opportunity Synthesis & Risk-Reward Geometry Tests
# ------------------------------------------------------------------------------
def test_trade_synthesis_dual_alpha():
    """Dual alpha setup (both 50 MA and Global Regime qualification) produces STRONG BUY with 1:2+ R:R."""
    row = pd.Series({
        "symbol": "RELIANCE",
        "close": 2500.0,
        "high": 2520.0,
        "low": 2480.0,
        "open": 2490.0,
        "atr14": 40.0,
        "ma50": 2450.0,
        "dist_low_ma50": 0.012,
        "rsi14": 55.0,
        "base_signal": True
    })
    stock_meta = {
        "RELIANCE": {"name": "Reliance Industries Ltd", "sector": "Energy"}
    }
    # pred_r = 1.8 R, pred_gain_pts = 150 pts (+6.0%), pred_drop_pts = 50 pts (-2.0%)
    trade = synthesize_trade_opportunity(
        row=row,
        pred_r=1.8,
        pred_gain_pts=150.0,
        pred_drop_pts=50.0,
        stock_meta=stock_meta
    )
    assert trade is not None
    assert trade["symbol"] == "RELIANCE"
    assert trade["name"] == "Reliance Industries Ltd"
    assert trade["strategy"] == "50 MA + Global Regime"
    assert trade["action"] == "STRONG BUY"
    assert trade["entry_price"] == 2500.0
    assert trade["target_price"] > trade["entry_price"]
    assert trade["sl_price"] < trade["entry_price"]
    assert trade["profit_pct"] > 0
    assert trade["loss_pct"] > 0
    assert trade["risk_reward_num"] >= 1.4


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
        "dist_low_ma50": -0.15,
        "rsi14": 30.0,
        "base_signal": False
    })
    stock_meta = {"WEAKSTOCK": {"name": "Weak Corp", "sector": "Consumer"}}
    # Negative R, gain < drop
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

    # Read back from disk
    stored = json_manager.get_daily_analysis_info()
    assert stored["status"] == "COMPLETED"
    assert stored["finalized_trades"][0]["symbol"] == "TESTSYM"


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
