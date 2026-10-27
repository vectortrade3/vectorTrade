"""
VectorTrade - Production Online Execution Daemon
Lightweight, stateful paper trading executor operating strictly between 09:17 and 15:32 IST.
Constraints:
- Absolute dependency limit: Standard library, requests, and pymongo ONLY.
- Under NO circumstances import pandas, numpy, scikit-learn, or xgboost.
- Memory consumption strictly bounded under 50 MB RAM.

Lifecycle:
1. 09:17 IST: Pull all ARMED setups for today from MongoDB signals_watchlist.
2. 2-3s Polling Loop:
   - Invalidation Gate: If LTP <= sl_price BEFORE LTP >= buy_trigger, transition to INVALIDATED.
   - Entry Trigger: If LTP >= buy_trigger, place BUY order, insert into active_trades (OPEN).
   - Exit Monitor: If LTP >= target_price (TARGET_HIT) or LTP <= sl_price (SL_HIT),
     fire SELL order, calculate realized PnL, archive into trade_history, remove from active_trades.
3. 15:32 IST: Expire remaining untriggered ARMED signals and terminate cleanly.
"""

from __future__ import annotations
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz

# Setup path to import core config and db modules
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from online.megabull_client import MegaBullClient

# Configure logging
log_file = settings.LOGS_DIR / "online_execution.log"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VectorTrade.OnlineDaemon")


class OnlineExecutionDaemon:
    """
    Micro-worker for live order execution, invalidation, and bracket tracking.
    """

    def __init__(self, dry_run: bool = False, poll_interval: float = 2.5) -> None:
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.tz = pytz.timezone(settings.TIMEZONE)
        self.mb_client = MegaBullClient()
        self.running = True

        # In-memory tracking state (minimal dictionary footprint < 1 MB)
        self.armed_signals: Dict[str, Dict[str, Any]] = {}
        self.active_trades: Dict[str, Dict[str, Any]] = {}

    def get_current_ist_time(self) -> datetime:
        return datetime.now(self.tz)

    def load_armed_signals(self) -> None:
        """Pulls ARMED signals for today from MongoDB."""
        signals_col = db_manager.get_collection("signals_watchlist")
        today_str = self.get_current_ist_time().strftime("%Y-%m-%d")
        
        # Query for today's ARMED signals (or any ARMED in dry_run mode)
        query = {"status": "ARMED"}
        if not self.dry_run:
            query["valid_for_date"] = today_str

        cursor = signals_col.find(query)
        self.armed_signals.clear()
        for doc in cursor:
            # We track by instrument_token or document ID
            sig_id = str(doc["_id"])
            self.armed_signals[sig_id] = doc

        logger.info("Loaded %d ARMED signal(s) from signals_watchlist.", len(self.armed_signals))

    def load_active_trades(self) -> None:
        """Pulls existing OPEN trades from MongoDB active_trades."""
        active_col = db_manager.get_collection("active_trades")
        cursor = active_col.find({"status": "OPEN"})
        self.active_trades.clear()
        for doc in cursor:
            trade_id = str(doc["_id"])
            self.active_trades[trade_id] = doc
        logger.info("Loaded %d OPEN trade(s) from active_trades.", len(self.active_trades))

    def handle_invalidation_and_entry(self, ltp_override: Optional[Dict[str, float]] = None) -> None:
        """
        Evaluates ARMED signals against current LTP.
        - If LTP <= SL: Signal is INVALIDATED (dropped immediately).
        - If LTP >= Buy Trigger: Executes BUY, transitions to active_trades.
        """
        signals_col = db_manager.get_collection("signals_watchlist")
        active_col = db_manager.get_collection("active_trades")

        tokens_to_remove: List[str] = []

        for sig_id, sig in list(self.armed_signals.items()):
            token = str(sig.get("instrument_token", "")).strip()
            if not token:
                continue

            # Fetch live LTP or use mock override
            ltp = ltp_override.get(token) if ltp_override else self.mb_client.get_ltp(token)
            if ltp is None or ltp <= 0:
                continue

            buy_trigger = float(sig["buy_trigger"])
            sl_price = float(sig["sl_price"])
            symbol = sig.get("symbol", "UNKNOWN")

            # Priority Rule 1: Day-1 Pre-entry Invalidation
            if ltp <= sl_price:
                logger.warning(
                    "[INVALIDATION] %s hit SL (LTP: %.2f <= SL: %.2f) before Buy Trigger (%.2f). Discarding signal.",
                    symbol, ltp, sl_price, buy_trigger
                )
                signals_col.update_one(
                    {"_id": sig["_id"]},
                    {"$set": {
                        "status": "INVALIDATED",
                        "invalidation_price": ltp,
                        "invalidated_at": datetime.now(pytz.utc).isoformat()
                    }}
                )
                tokens_to_remove.append(sig_id)
                continue

            # Priority Rule 2: Entry Trigger
            if ltp >= buy_trigger:
                logger.info(
                    "[ENTRY TRIGGER] %s crossed Buy Trigger (LTP: %.2f >= Buy: %.2f)! Firing BUY order...",
                    symbol, ltp, buy_trigger
                )
                qty = int(sig.get("quantity", 1))

                # Dispatch BUY order to MegaBull
                order_res = self.mb_client.place_order(
                    instrument_token=token,
                    action="BUY",
                    quantity=qty,
                    order_type="LIMIT",
                    price=ltp,
                    duration="CNC"
                )

                order_id = order_res.get("order_id", f"MOCK-BUY-{int(time.time())}")
                now_iso = datetime.now(pytz.utc).isoformat()

                # Update signals_watchlist to TRIGGERED
                signals_col.update_one(
                    {"_id": sig["_id"]},
                    {"$set": {
                        "status": "TRIGGERED",
                        "triggered_at": now_iso,
                        "fill_price": ltp,
                        "megabull_order_id": order_id
                    }}
                )

                # Insert into active_trades
                active_trade_doc = {
                    "_id": f"TRADE_{symbol}_{int(time.time())}",
                    "symbol": symbol,
                    "instrument_token": token,
                    "strategy_id": sig.get("strategy_id", "50MA_PULLBACK"),
                    "megabull_order_id": order_id,
                    "fill_price": ltp,
                    "sl_price": sl_price,
                    "target_price": float(sig["target_price"]),
                    "quantity": qty,
                    "days_active": 1,
                    "max_hold_days": int(sig.get("max_hold_days", 7)),
                    "entry_timestamp": now_iso,
                    "status": "OPEN",
                    "highest_ltp": ltp,
                    "lowest_ltp": ltp
                }
                active_col.insert_one(active_trade_doc)
                self.active_trades[active_trade_doc["_id"]] = active_trade_doc

                tokens_to_remove.append(sig_id)

        for sig_id in tokens_to_remove:
            self.armed_signals.pop(sig_id, None)

    def handle_position_exits(self, ltp_override: Optional[Dict[str, float]] = None) -> None:
        """
        Monitors active positions against target and stop loss.
        Dispatches offsetting SELL orders upon breach and commits realized PnL to trade_history.
        """
        active_col = db_manager.get_collection("active_trades")
        history_col = db_manager.get_collection("trade_history")

        trades_to_close: List[str] = []

        for trade_id, trade in list(self.active_trades.items()):
            token = str(trade.get("instrument_token", "")).strip()
            ltp = ltp_override.get(token) if ltp_override else self.mb_client.get_ltp(token)
            if ltp is None or ltp <= 0:
                continue

            # Update MFE / MAE tracking metrics
            fill_price = float(trade["fill_price"])
            highest_ltp = max(float(trade.get("highest_ltp", fill_price)), ltp)
            lowest_ltp = min(float(trade.get("lowest_ltp", fill_price)), ltp)
            trade["highest_ltp"] = highest_ltp
            trade["lowest_ltp"] = lowest_ltp

            target_price = float(trade["target_price"])
            sl_price = float(trade["sl_price"])
            symbol = trade.get("symbol", "UNKNOWN")
            qty = int(trade.get("quantity", 1))

            exit_reason: Optional[str] = None
            if ltp >= target_price:
                exit_reason = "TARGET_HIT"
            elif ltp <= sl_price:
                exit_reason = "SL_HIT"

            if exit_reason:
                logger.info(
                    "[EXIT TRIGGER] %s position triggered %s (LTP: %.2f | Target: %.2f | SL: %.2f). Firing SELL...",
                    symbol, exit_reason, ltp, target_price, sl_price
                )

                # Dispatch offsetting SELL order to MegaBull
                order_res = self.mb_client.place_order(
                    instrument_token=token,
                    action="SELL",
                    quantity=qty,
                    order_type="LIMIT",
                    price=ltp,
                    duration="CNC"
                )

                realized_pnl = round((ltp - fill_price) * qty, 2)
                roi_pct = round(((ltp - fill_price) / fill_price) * 100.0, 2)
                now_iso = datetime.now(pytz.utc).isoformat()

                trade_history_doc = {
                    "_id": f"HIST_{trade_id}",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "instrument_token": token,
                    "strategy_id": trade.get("strategy_id", "50MA_PULLBACK"),
                    "entry_price": fill_price,
                    "exit_price": ltp,
                    "quantity": qty,
                    "pnl": realized_pnl,
                    "roi_pct": roi_pct,
                    "exit_reason": exit_reason,
                    "entry_timestamp": trade.get("entry_timestamp", now_iso),
                    "exit_timestamp": now_iso,
                    "mfe": round(highest_ltp - fill_price, 2),
                    "mae": round(fill_price - lowest_ltp, 2),
                    "ml_outcome_label": 1 if exit_reason == "TARGET_HIT" else 0
                }

                # Persist to trade_history and remove from active_trades
                history_col.insert_one(trade_history_doc)
                active_col.delete_one({"_id": trade["_id"]})
                trades_to_close.append(trade_id)

        for trade_id in trades_to_close:
            self.active_trades.pop(trade_id, None)

    def expire_untriggered_signals(self) -> None:
        """At 15:32 IST, marks remaining untriggered ARMED signals as EXPIRED."""
        signals_col = db_manager.get_collection("signals_watchlist")
        now_iso = datetime.now(pytz.utc).isoformat()
        
        expired_count = 0
        for sig_id, sig in list(self.armed_signals.items()):
            signals_col.update_one(
                {"_id": sig["_id"], "status": "ARMED"},
                {"$set": {"status": "EXPIRED", "expired_at": now_iso}}
            )
            expired_count += 1

        self.armed_signals.clear()
        logger.info("Market session concluded. Expired %d untriggered ARMED signals.", expired_count)

    def run_market_session(self, max_cycles: Optional[int] = None) -> None:
        """
        Main execution loop operating between 09:17 and 15:32 IST.
        """
        logger.info("Starting VectorTrade Online Execution Micro-Worker...")
        self.load_armed_signals()
        self.load_active_trades()

        cycle = 0
        while self.running:
            now_ist = self.get_current_ist_time()
            hour, minute = now_ist.hour, now_ist.minute

            # If before 09:17 IST and not dry_run, sleep until market open
            if not self.dry_run and (hour < 9 or (hour == 9 and minute < 17)):
                logger.info("Current time %02d:%02d is before 09:17 IST market open. Standing by...", hour, minute)
                time.sleep(30)
                continue

            # If past 15:32 IST, wrap up and exit
            if not self.dry_run and (hour > 15 or (hour == 15 and minute >= 32)):
                logger.info("Current time %02d:%02d is past 15:32 IST market close. Concluding session.", hour, minute)
                self.expire_untriggered_signals()
                break

            # Core polling loop
            self.handle_invalidation_and_entry()
            self.handle_position_exits()

            cycle += 1
            if max_cycles and cycle >= max_cycles:
                logger.info("Reached maximum requested test cycles (%d). Halting loop.", max_cycles)
                break

            time.sleep(self.poll_interval)

        self.mb_client.close()
        logger.info("Online execution daemon closed cleanly. Memory released.")


def main():
    dry_run = "--dry-run" in sys.argv
    daemon = OnlineExecutionDaemon(dry_run=dry_run)
    daemon.run_market_session()
    sys.exit(0)


if __name__ == "__main__":
    main()
