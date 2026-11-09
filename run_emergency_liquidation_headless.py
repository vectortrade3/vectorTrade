#!/usr/bin/env python3
"""
VectorTrade - Headless Emergency Portfolio Liquidation Runner
Executes square off of all active portfolio holdings headlessly via MegaBull API
and MongoDB Atlas lifecycle orchestration.
"""

import sys
import json
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from online.services.megabull_service import megabull_service
from online.db import db_manager


def run_emergency_liquidation_headless(force_live: bool = False) -> dict:
    """
    Executes emergency square off of all active portfolio holdings headlessly.
    Case 1: Market Open -> Sends live MARKET exit orders to MegaBull.
    Case 2: Market Closed -> Stages all 8 positions for market open (09:15 IST),
            cancels all buy orders, and arms persistent warning banner.
    """
    print("\n" + "=" * 76)
    print("      VectorTrade Online -- Headless Emergency Portfolio Liquidation")
    print("=" * 76)

    # 1. Fetch live MegaBull account details
    account = megabull_service.get_account_details()
    print(f" [*] MegaBull Account  : {account.get('full_name')} ({account.get('email')})")
    print(f" [*] Virtual Capital   : INR {account.get('virtual_money', 0.0):,.2f}")
    print(f" [*] Margin Blocked    : INR {account.get('virtual_money_blocked', 0.0):,.2f}")
    print(f" [*] Free Balance      : INR {account.get('virtual_money_left', 0.0):,.2f}")

    # 2. Fetch active holdings
    holdings = megabull_service.get_holdings(enrich_ltp=True)
    active = [h for h in holdings if int(h.get("qty", 0)) != 0]
    print(f" [*] Active Holdings   : {len(active)} stocks currently held in portfolio")
    print("-" * 76)

    for idx, h in enumerate(active, start=1):
        qty = int(h.get("qty", 0))
        is_long = qty > 0
        action = "SELL" if is_long else "BUY (COVER)"
        ltp = float(h.get("ltp", h.get("avg_buy_price", 0.0)))
        val = float(h.get("current_value", 0.0))
        pnl = float(h.get("unrealized_pnl", 0.0))
        pnl_sign = "+" if pnl >= 0 else ""
        print(f"  {idx:2d}. {h['symbol']:<12} | {h['type']:<5} | Qty: {abs(qty):3d} | LTP: INR {ltp:8.2f} | Action: {action:<11} | P&L: {pnl_sign}{pnl:8.2f}")

    print("-" * 76)

    # 3. Check market trading hours
    is_open = megabull_service.is_market_open() if not force_live else True
    print(f" [*] Market Status     : {'OPEN (Live Trading)' if is_open else 'CLOSED (Outside 09:15 - 15:30 IST)'}")

    # 4. Execute Liquidation
    if not is_open:
        print(" [!] Executing CASE 2 (Market Closed Protocol):")
        print("     -> Attempting broker transmission & staging orders in MongoDB Atlas for market open...")
    else:
        print(" [!] Executing CASE 1 (Live Market Open Protocol):")
        print("     -> Sending immediate MARKET exit orders to MegaBull...")

    result = megabull_service.liquidate_all_positions(force_market_open=True if force_live else None)

    print("-" * 76)
    print(f" [*] Result Status     : {result.get('status')}")
    print(f" [*] Execution Case    : {result.get('case')}")
    print(f" [*] Summary Message   : {result.get('message')}")

    if result.get("case") == "CASE_2_MARKET_CLOSED":
        print("\n [OK] Case 2 Staged Schedule Details:")
        print(f"      - Holdings Scheduled: {result.get('holdings_count')} stocks")
        print("      - Target Execution  : Market Open (09:15 IST, Next Trading Session)")
        print("      - All Buy Orders    : CANCELLED in MongoDB Atlas")
        print("      - New Buying        : HALTED (Protected by Liquidation Guard)")
        print("      - Safety Banner     : ARMED on /orders and /megabull with 'Cancel Scheduled Exit' button")

    # 5. Check Database Persistence
    try:
        col_state = db_manager.system_state
        if col_state is not None:
            saved_state = col_state.find_one({"_id": "emergency_liquidation"})
            print(f" [*] MongoDB Atlas State: _id='emergency_liquidation' status='{saved_state.get('status') if saved_state else 'N/A'}'")
    except Exception as e:
        print(f" [!] DB check notice   : {e}")

    print("=" * 76 + "\n")
    return result


if __name__ == "__main__":
    force = "--force-market-open" in sys.argv or "--force" in sys.argv
    run_emergency_liquidation_headless(force_live=force)
