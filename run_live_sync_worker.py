#!/usr/bin/env python3
"""
VectorTrade - Standalone Live Market & Order Synchronization Worker
Runs full-time to continuously:
1. Fetch latest prices (LTP) from MegaBull and update MongoDB `live_quotes`.
2. Sync order statuses across the 4 key stages:
   - WILL_BE_PUT_AMO: Off-market staged orders for market open (09:15 IST).
   - PENDING: Submitted to MegaBull broker, waiting for price trigger.
   - RUNNING: Bought (Long) or Short Selled positions actively held.
   - COMPLETED: Closed round-trip trades (Bought -> Sold, or Short Sell -> Covered).
3. Place & manage Target & SL bracket orders.
4. Live sync portfolio balance and P&L into MongoDB Atlas.

Usage:
  python run_live_sync_worker.py               # Runs continuously every 5 seconds
  python run_live_sync_worker.py --interval 10 # Polls every 10 seconds
  python run_live_sync_worker.py --once        # Runs a single verification cycle and exits
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online.services.live_sync_worker import live_sync_worker, OrderLifecycleStage
from online.services.megabull_service import megabull_service


def parse_args():
    parser = argparse.ArgumentParser(description="VectorTrade Live Market & Order Synchronization Worker")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds (default: 5)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for headless test/verification)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def print_banner(interval: int, run_once: bool):
    market_open = megabull_service.is_market_open()
    status_str = "🟢 OPEN (09:15-15:30 IST)" if market_open else "🔴 CLOSED (Off-Market)"
    print("\n" + "=" * 76)
    print("  🚀 VECTORTRADE - LIVE MARKET & ORDER SYNCHRONIZATION WORKER")
    print("=" * 76)
    print(f"  • Indian Market Status : {status_str}")
    print(f"  • Polling Frequency   : Every {interval} seconds")
    print(f"  • Execution Mode       : {'Single-Pass Verification (--once)' if run_once else 'Continuous Full-Time Daemon'}")
    print(f"  • Monitored Lifecycle  : [AMO] ➔ [PENDING] ➔ [RUNNING] ➔ [COMPLETED]")
    print("=" * 76 + "\n")


def print_tick_summary(res: dict):
    if res.get("status") == "ERROR":
        print(f"❌ [Tick #{res.get('tick_number')}] Error: {res.get('error')}")
        return

    stages = res.get("orders_by_stage", {})
    amo_cnt = stages.get(OrderLifecycleStage.WILL_BE_PUT_AMO, 0)
    pending_cnt = stages.get(OrderLifecycleStage.PENDING, 0)
    bought_cnt = stages.get(OrderLifecycleStage.RUNNING_BOUGHT, 0)
    short_cnt = stages.get(OrderLifecycleStage.RUNNING_SHORT_SELLED, 0)
    completed_b2s = stages.get(OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD, 0)
    completed_s2c = stages.get(OrderLifecycleStage.COMPLETED_SHORT_TO_COVERED, 0)
    total_completed = completed_b2s + completed_s2c

    m_status = "OPEN" if res.get("market_open") else "CLOSED"
    print(
        f"⏱️ [{res.get('timestamp')}] Tick #{res.get('tick_number')} ({res.get('elapsed_seconds')}s) | "
        f"Mkt: {m_status} | Quotes: {res.get('prices_updated', 0)} | "
        f"AMO: {amo_cnt} | Pending: {pending_cnt} | "
        f"Running Bought: {bought_cnt} | Short: {short_cnt} | "
        f"Completed: {total_completed} | Net Worth: ₹{res.get('portfolio_net_worth', 0.0):,.2f}"
    )


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner(args.interval, args.once)

    if args.once:
        print("▶ Executing single sync pass...")
        res = live_sync_worker.tick()
        print_tick_summary(res)
        print("\n✅ Single-pass sync completed successfully.")
        sys.exit(0)

    print("▶ Starting continuous background sync loop (Press Ctrl+C to terminate)...")
    try:
        live_sync_worker.is_running = True
        while not live_sync_worker._stop_requested:
            res = live_sync_worker.tick()
            print_tick_summary(res)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Caught KeyboardInterrupt. Shutting down LiveSyncWorker cleanly...")
    finally:
        live_sync_worker.stop()
        print("👋 LiveSyncWorker terminated cleanly.")


if __name__ == "__main__":
    main()
