"""
Institutional Cointegrated Pairs Trading Strategy v4 - Analytics & Verification Module (Phase 2)
Script: offline/pair_analytics.py

Calculates:
1. Detailed holding time distributions (mean, median, std, min, max, quantiles) across Train and Test splits.
2. Holding time per exit category: TARGET_REACHED, STOP_LOSS_BARRIER, and TIME_STOP_CAP.
3. Win rate, Profit Factor, Sharpe Ratio, Net P&L %, Max Drawdown, MFE, and MAE.
4. Export comprehensive analytical report to data_store/pair_strategy/phase2_analytics_report.json
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PAIR_STRATEGY_DIR = BASE_DIR / "data_store" / "pair_strategy"


def calculate_split_metrics(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Calculates comprehensive empirical execution metrics for a split."""
    if df.empty:
        return {}

    n_trades = len(df)
    holding = df["holding_days"]
    returns = df["net_return_pct"]
    wins = df[df["outcome_label"] == 1]
    losses = df[df["outcome_label"] == 0]
    neutrals = df[df["outcome_label"] == 2]

    win_rate = (len(wins) / n_trades) * 100.0
    loss_rate = (len(losses) / n_trades) * 100.0
    neutral_rate = (len(neutrals) / n_trades) * 100.0

    gross_profit = float(wins["net_return_pct"].sum())
    gross_loss = abs(float(losses["net_return_pct"].sum()))
    profit_factor = (gross_profit / (gross_loss + 1e-9)) if gross_loss > 0 else 999.0

    # Max Drawdown calculation from cumulative returns
    cum_ret = (1.0 + (returns / 100.0)).cumprod()
    running_max = cum_ret.cummax()
    drawdowns = (cum_ret - running_max) / running_max
    max_dd_pct = abs(float(drawdowns.min())) * 100.0

    # Holding time stats overall
    ht_overall = {
        "mean_days": round(float(holding.mean()), 2),
        "median_days": round(float(holding.median()), 2),
        "std_days": round(float(holding.std()), 2),
        "p25_days": round(float(holding.quantile(0.25)), 2),
        "p75_days": round(float(holding.quantile(0.75)), 2),
        "min_days": int(holding.min()),
        "max_days": int(holding.max())
    }

    # Holding time by exit reason
    ht_by_exit = {}
    for reason in ["TARGET_REACHED", "STOP_LOSS_BARRIER", "TIME_STOP_CAP"]:
        sub = df[df["exit_reason"] == reason]
        if not sub.empty:
            ht_by_exit[reason] = {
                "count": len(sub),
                "pct_of_total": round((len(sub) / n_trades) * 100.0, 2),
                "mean_holding_days": round(float(sub["holding_days"].mean()), 2),
                "avg_return_pct": round(float(sub["net_return_pct"].mean()), 2),
                "avg_mfe_pct": round(float(sub["mfe_pct"].mean()), 2),
                "avg_mae_pct": round(float(sub["mae_pct"].mean()), 2)
            }

    return {
        "split_name": split_name,
        "total_trades": n_trades,
        "date_range": [str(df["entry_date"].min()), str(df["entry_date"].max())],
        "win_rate_pct": round(win_rate, 2),
        "loss_rate_pct": round(loss_rate, 2),
        "neutral_rate_pct": round(neutral_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_net_return_pct": round(float(returns.mean()), 2),
        "median_net_return_pct": round(float(returns.median()), 2),
        "avg_mfe_pct": round(float(df["mfe_pct"].mean()), 2),
        "avg_mae_pct": round(float(df["mae_pct"].mean()), 2),
        "holding_time_stats": ht_overall,
        "exit_reasons_breakdown": ht_by_exit
    }


def generate_phase2_report() -> Dict[str, Any]:
    """Generates and writes Phase 2 analytical report."""
    train_file = PAIR_STRATEGY_DIR / "pair_trades_train.csv"
    test_file = PAIR_STRATEGY_DIR / "pair_trades_test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("pair_trades_train.csv or test.csv not found.")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    report = {
        "status": "SUCCESS",
        "phase": 2,
        "strategy": "Institutional Cointegrated Stat-Arb v4",
        "train_4y_analytics": calculate_split_metrics(df_train, "4Y Training (Historical)"),
        "test_1y_analytics": calculate_split_metrics(df_test, "1Y Testing (Holdout Out-of-Sample)")
    }

    out_path = PAIR_STRATEGY_DIR / "phase2_analytics_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = generate_phase2_report()
    print("\n--- Phase 2 Execution & Holding Time Report ---")
    print(json.dumps(rep, indent=2))
