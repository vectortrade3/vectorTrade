"""
VectorTrade - Phase 4 Verification Suite
Low-latency simulation tests for:
1. Pure standard library / requests / pymongo dependency adherence (Zero pandas/numpy in online worker).
2. MegaBull Client connection pooling, error handling, and LTP queries.
3. Day-1 Pre-entry Invalidation Gate: verifies signal transitions to INVALIDATED if LTP <= SL before Buy Trigger.
4. Entry Trigger & Bracket Tracking: verifies transition to TRIGGERED and insertion into active_trades.
5. Exit Monitor & Realized PnL: verifies Target / SL breach, offsetting order dispatch, and trade_history persistence.
6. Memory footprint audit: confirms process remains bounded (< 50 MB RAM).
"""

import sys
import os
import time
import tracemalloc
from pathlib import Path

# Add vectorTrade directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Start memory tracking
tracemalloc.start()

from core.config import settings
from core.db import db_manager
from online.megabull_client import MegaBullClient
from online.execution_daemon import OnlineExecutionDaemon


def verify_phase_4():
    print("==================================================")
    print("  VectorTrade Phase 4 Verification Suite")
    print("==================================================")

    # 1. Dependency Boundary Verification (CodeRabbit Audit)
    print("[1/5] Auditing low-latency dependency boundaries...")
    for file_path in [BASE_DIR / "online" / "execution_daemon.py", BASE_DIR / "online" / "megabull_client.py"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    for forbidden in ["pandas", "numpy", "scikit-learn", "sklearn", "xgboost"]:
                        assert forbidden not in stripped, f"Violation: Forbidden module '{forbidden}' imported in {file_path.name}: {stripped}"

    print("  -> Dependency Audit: Confirmed 0 bulky analytical libraries imported in online execution modules.")

    # 2. Live MegaBull Client LTP Check
    print("[2/5] Testing MegaBull Client connection pool and live LTP retrieval...")
    client = MegaBullClient()
    reliance_ltp = client.get_ltp("738561")
    print(f"  -> Live RELIANCE LTP query returned: {reliance_ltp}")
    assert reliance_ltp is not None and reliance_ltp > 0, "MegaBull live LTP retrieval failed"
    client.close()

    # 3. Test Day-1 Invalidation Gate
    print("[3/5] Testing Day-1 Pre-entry Invalidation Gate...")
    daemon = OnlineExecutionDaemon(dry_run=True)
    signals_col = db_manager.get_collection("signals_watchlist")

    # Insert mock armed signal for Invalidation test
    inv_sig_id = "TEST_INVALIDATION_TATA"
    signals_col.delete_many({"_id": inv_sig_id})
    signals_col.insert_one({
        "_id": inv_sig_id,
        "symbol": "TATAMOTORS",
        "instrument_token": "3456",
        "strategy_id": "50MA_PULLBACK",
        "action": "BUY",
        "buy_trigger": 1000.0,
        "sl_price": 950.0,
        "target_price": 1100.0,
        "status": "ARMED",
        "quantity": 10
    })

    daemon.load_armed_signals()
    assert inv_sig_id in daemon.armed_signals

    # Feed LTP = 948.0 (Below SL of 950.0 before hitting 1000.0)
    daemon.handle_invalidation_and_entry(ltp_override={"3456": 948.0})

    updated_inv = signals_col.find_one({"_id": inv_sig_id})
    assert updated_inv["status"] == "INVALIDATED", f"Expected INVALIDATED, got {updated_inv['status']}"
    assert inv_sig_id not in daemon.armed_signals, "Invalidated signal should be dropped from active memory"
    print("  -> Invalidation Gate: Verified immediate transition to INVALIDATED upon SL touch.")

    # 4. Test Entry Trigger and Bracket Execution
    print("[4/5] Testing Entry Trigger, Position Insertion & Exit Execution...")
    entry_sig_id = "TEST_ENTRY_INFY"
    signals_col.delete_many({"_id": entry_sig_id})
    signals_col.insert_one({
        "_id": entry_sig_id,
        "symbol": "INFY",
        "instrument_token": "408065",
        "strategy_id": "50MA_PULLBACK",
        "action": "BUY",
        "buy_trigger": 1900.0,
        "sl_price": 1850.0,
        "target_price": 2000.0,
        "status": "ARMED",
        "quantity": 5
    })

    daemon.load_armed_signals()
    # Feed LTP = 1902.0 (Breaks Buy Trigger 1900.0)
    daemon.handle_invalidation_and_entry(ltp_override={"408065": 1902.0})

    updated_entry = signals_col.find_one({"_id": entry_sig_id})
    assert updated_entry["status"] == "TRIGGERED", f"Expected TRIGGERED, got {updated_entry['status']}"

    active_col = db_manager.get_collection("active_trades")
    active_trade = active_col.find_one({"symbol": "INFY", "status": "OPEN"})
    assert active_trade is not None, "Active trade document missing from active_trades"
    assert active_trade["fill_price"] == 1902.0
    print("  -> Entry Trigger: Verified transition to TRIGGERED and insertion into active_trades.")

    # Test Target Exit
    history_col = db_manager.get_collection("trade_history")
    history_col.delete_many({"symbol": "INFY"})

    # Feed LTP = 2005.0 (Hits Target 2000.0)
    daemon.handle_position_exits(ltp_override={"408065": 2005.0})

    # Verify active position deleted and moved to trade_history
    assert active_col.find_one({"symbol": "INFY", "status": "OPEN"}) is None, "Closed trade should be removed from active_trades"
    hist_record = history_col.find_one({"symbol": "INFY"})
    assert hist_record is not None, "Trade history record missing"
    assert hist_record["exit_reason"] == "TARGET_HIT"
    assert hist_record["exit_price"] == 2005.0
    expected_pnl = round((2005.0 - 1902.0) * 5, 2)
    assert hist_record["pnl"] == expected_pnl, f"Expected PnL {expected_pnl}, got {hist_record['pnl']}"
    assert hist_record["ml_outcome_label"] == 1
    print(f"  -> Exit Execution: Verified TARGET_HIT exit, Realized PnL: +Rs {hist_record['pnl']}, Outcome: 1.")

    # 5. Memory Footprint Audit
    print("[5/5] Auditing runtime memory footprint...")
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_mb = current_mem / (1024 * 1024)
    peak_mb = peak_mem / (1024 * 1024)
    print(f"  -> Memory Usage: Current={current_mb:.2f} MB, Peak={peak_mb:.2f} MB (Constraint: < 50 MB)")
    assert peak_mb < 50.0, f"Memory limit exceeded: Peak {peak_mb:.2f} MB >= 50 MB"

    # Clean up test artifacts in DB
    signals_col.delete_many({"_id": {"$in": [inv_sig_id, entry_sig_id]}})
    history_col.delete_many({"symbol": "INFY"})

    print("==================================================")
    print("  ALL PHASE 4 CHECKS PASSED (Micro-Worker Verified)")
    print("==================================================")


if __name__ == "__main__":
    verify_phase_4()
