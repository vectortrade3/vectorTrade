"""
VectorTrade - Automated 4:00 PM Daily Market Analysis & Self-Closer CLI Runner
Usage:
    python run_offline_auto.py          # Runs automated loop (ticks every 60s, executes at 4:00 PM, self-closes)
    python run_offline_auto.py --once   # Runs a single tick evaluation
    python run_offline_auto.py --force  # Forces immediate analysis pipeline run now
"""

import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from offline.scheduler import get_daily_scheduler, trigger_self_close, check_internet_connection
from strategies.manager import StrategyManager


def main():
    parser = argparse.ArgumentParser(description="VectorTrade 4:00 PM Daily Market Analysis & Self-Closer")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation tick and exit")
    parser.add_argument("--force", action="store_true", help="Force analysis execution immediately")
    parser.add_argument("--interval", type=int, default=60, help="Tick interval in seconds (default: 60)")
    parser.add_argument("--no-close", action="store_true", help="Disable process self-closing for debugging")
    args = parser.parse_args()

    if args.no_close:
        trigger_self_close.disabled = True

    scheduler = get_daily_scheduler(check_interval_seconds=args.interval)
    win_info = scheduler.get_time_window_info()
    strat_mgr = StrategyManager()
    enabled_strats = [s["name"] for s in strat_mgr.get_enabled_strategies()]

    print("=" * 70)
    print("  VectorTrade - Daily 4:00 PM Automated Analysis & Self-Closer")
    print(f"  Time (IST)          : {win_info['current_time_ist']}")
    print(f"  Active Window       : {win_info['window_label']}")
    print(f"  Time until 4:00 PM  : {win_info['time_until_4pm']}")
    print(f"  Enabled Strategies  : {', '.join(enabled_strats) if enabled_strats else 'None (All deselected)'}")
    print("=" * 70)

    if args.force:
        print("\n[*] Forcing immediate execution tick...")
        res = scheduler.run_tick_once(force_execution=True)
        print(f"[*] Result: {res.get('action')} - {res.get('message')}")
        return

    if args.once:
        print("\n[*] Evaluating single scheduler tick...")
        res = scheduler.run_tick_once(force_execution=False)
        print(f"[*] Result: {res.get('action')} - {res.get('message')}")
        return

    # Continuous loop
    print(f"\n[*] Starting scheduler loop (Interval: {args.interval}s). Press Ctrl+C to terminate.\n")
    try:
        while True:
            res = scheduler.run_tick_once()
            action = res.get("action")
            msg = res.get("message")
            t_now = res.get("window_info", {}).get("time_formatted", "")
            print(f"[{t_now} IST] [{action}] {msg}")

            if action in ("SELF_CLOSE_TRIGGERED", "PIPELINE_COMPLETED_SELF_CLOSING"):
                print("[*] Self-close has been scheduled. Waiting for process exit...")
                time.sleep(5)
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[!] Scheduler loop stopped by user.")


if __name__ == "__main__":
    main()
