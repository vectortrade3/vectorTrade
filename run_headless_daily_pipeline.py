"""
VectorTrade - Standalone Headless Daily Pipeline Runner
Executes the complete EOD market pipeline:
1. Downloads updated stock market data (download)
2. Scans & generates trade setups with calibrated AI models (analyse)
3. Synchronizes trades and pre-market capital allocations to MongoDB Atlas (upload)
4. Persists state atomically and updates JSON progress logs (update_logs)

Completely headless, no web server, exits with code 0 upon completion.
"""

from __future__ import annotations
import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Ensure vectorTrade directory is in Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.daily_analyzer import run_daily_analysis_pipeline, get_market_schedule_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VectorTrade Headless Daily Pipeline (Download -> Analyse -> Upload -> Update Logs)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if market session is marked complete or holiday"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="HEADLESS_DAILY_PIPELINE",
        help="Identifier for the pipeline trigger source (default: HEADLESS_DAILY_PIPELINE)"
    )
    args = parser.parse_args()

    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("  VectorTrade Headless Daily Pipeline Runner")
    print(f"  Started at : {now_str}")
    print(f"  Trigger    : {args.trigger}")
    print(f"  Force Mode : {args.force}")
    print("=" * 70)

    try:
        # Evaluate current market schedule info for reporting
        sched = get_market_schedule_info()
        print(f"[*] Market Schedule  : {sched.get('market_status')}")
        print(f"[*] Target Session   : {sched.get('target_date')}")

        # Execute the unified 4-step pipeline: download, analyse, upload, update logs
        print("[*] Launching analysis pipeline...")
        result = run_daily_analysis_pipeline(
            force_download=args.force,
            execution_trigger=args.trigger
        )

        elapsed = time.time() - start_time
        status = result.get("status", "UNKNOWN")

        if status == "SUCCESS":
            finalized_count = result.get("finalized_count", 0)
            mongo_sync_obj = result.get("mongo_sync", {})
            mongo_synced = (mongo_sync_obj.get("status") == "SUCCESS") or result.get("mongo_synced", False)

            # Daily Cointegrated Stat-Arb Pairs Strategy scan
            pair_scan_info = "Skipped"
            try:
                print("[*] Running Daily Pair Trading Model Evaluation across 156 pairs...")
                from online.services.pair_inference_service import PairInferenceService
                pair_res = PairInferenceService.evaluate_daily_pairs()
                if pair_res.get("status") == "SUCCESS":
                    q_count = pair_res.get("qualified_signals_count", 0)
                    c_count = pair_res.get("candidates_found_count", 0)
                    pair_scan_info = f"{q_count} ML-qualified signals from {c_count} hurdle candidates"
                    print(f"[*] Pair Trading Engine: {pair_scan_info}")
            except Exception as p_err:
                print(f"[!] Pair Trading scan notice: {p_err}")

            print("-" * 70)
            print("  PIPELINE EXECUTION SUCCESSFUL")
            print(f"  Total Duration  : {elapsed:.2f} seconds")
            print(f"  Finalized Setups: {finalized_count}")
            print(f"  Pairs Qualified : {pair_scan_info}")
            print(f"  MongoDB Synced  : {mongo_synced}")
            if mongo_synced:
                print(f"  MongoDB Message : {mongo_sync_obj.get('message', 'Synchronized successfully')}")
            print(f"  Summary Message : {result.get('message', 'Completed')}")
            print("=" * 70)
            return 0
        else:
            print("-" * 70)
            print(f"  PIPELINE FINISHED WITH NON-SUCCESS STATUS: {status}")
            print(f"  Message: {result.get('message', 'No details available')}")
            print("=" * 70)
            return 1

    except Exception as exc:
        elapsed = time.time() - start_time
        print("-" * 70)
        print(f"  [ERROR] Pipeline execution failed: {exc}")
        print(f"  Elapsed before crash: {elapsed:.2f} seconds")
        print("=" * 70)
        json_manager.log_app_event(
            event="HEADLESS_DAILY_PIPELINE_ERROR",
            status="Error",
            message=f"Headless daily pipeline failed: {str(exc)}",
            error=str(exc)
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
