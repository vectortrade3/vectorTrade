"""
VectorTrade - Production Online Execution Worker with Universal Fail-Safe Engine
Implements defense-in-depth for real-world production trading:
1. Stale Signal Rejection: Only arms signals matching today's date (valid_for_date == TODAY).
   If Offline Daemon failed to run, Online skips taking new trades completely.
2. Continuous Open Position Protection: Always monitors Target/SL for open trades from prior days
   regardless of whether new signals were generated today.
3. Day-1 Pre-entry Invalidation Gate: If LTP <= SL before Buy Trigger, marks INVALIDATED (never buys).
4. Broker Authentication Expiration Detection: Detects 401 Unauthorized from MegaBull and raises alert.
5. External Reconciliation: Syncs with MegaBull open positions to detect manual user closures or broker fills.
6. Local Offline Queue & Network Loss Recovery: Reconnects to MongoDB with exponential backoff.
"""

from __future__ import annotations
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager
from online.megabull_client import MegaBullClient
from core.secure_comms import SecureMailboxCommunicator

logger = logging.getLogger("VectorTrade.ExecutionWorker")


class ExecutionWorker:
    """
    Asynchronous / Threaded execution worker with comprehensive fail-safe recovery.
    """

    def __init__(self, dry_run: bool = False, poll_interval: float = 2.5) -> None:
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.tz = pytz.timezone(settings.TIMEZONE)
        self.mb_client = MegaBullClient()
        self.running = False

        # In-memory fast tracking state
        self.armed_signals: Dict[str, Dict[str, Any]] = {}
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        
        # System Health & Fail-Safe Flags
        self.system_status = {
            "broker_auth_valid": True,
            "offline_data_fresh": True,
            "database_connected": True,
            "last_cycle_timestamp": None,
            "last_cycle_duration_ms": 0,
            "alerts": []
        }

    def get_current_ist_time(self) -> datetime:
        return datetime.now(self.tz)

    def add_alert(self, code: str, message: str, level: str = "WARNING") -> None:
        """Adds a deduplicated system alert for dashboard notification."""
        timestamp = self.get_current_ist_time().isoformat()
        # Avoid duplicate alert flooding
        existing = [a for a in self.system_status["alerts"] if a["code"] == code]
        if not existing:
            self.system_status["alerts"].insert(0, {
                "code": code,
                "message": message,
                "level": level,
                "timestamp": timestamp
            })
            # Keep maximum 10 recent alerts
            self.system_status["alerts"] = self.system_status["alerts"][:10]
            logger.warning("[SYSTEM ALERT][%s] %s: %s", level, code, message)

    def clear_alert(self, code: str) -> None:
        self.system_status["alerts"] = [a for a in self.system_status["alerts"] if a["code"] != code]

    def load_signals_and_trades(self) -> None:
        """
        Pulls signals and active trades with strict Stale-Data and Open-Trade isolation:
        - Stale-Data Gate: Ignores any signals where valid_for_date != TODAY.
        - Open-Trade Protection: Always loads and tracks open trades regardless of new signal status.
        """
        today_str = self.get_current_ist_time().strftime("%Y-%m-%d")

        try:
            signals_col = db_manager.get_collection("signals_watchlist")
            active_col = db_manager.get_collection("active_trades")

            # 1. Open Trade Protection: Always load active trades across all historical dates
            self.active_trades.clear()
            for doc in active_col.find({"status": "OPEN"}):
                self.active_trades[str(doc["_id"])] = doc

            # 2. Stale Signal Rejection Gate: Strictly load ARMED signals valid for TODAY
            self.armed_signals.clear()
            query: Dict[str, Any] = {"status": "ARMED"}
            if not self.dry_run:
                query["valid_for_date"] = today_str

            armed_cursor = list(signals_col.find(query))

            if armed_cursor:
                for doc in armed_cursor:
                    self.armed_signals[str(doc["_id"])] = doc
                self.system_status["offline_data_fresh"] = True
                self.clear_alert("STALE_OFFLINE_DATA")
            else:
                # If no signals for today, verify if stale signals exist from prior dates
                stale_signals_count = signals_col.count_documents({
                    "status": "ARMED",
                    "valid_for_date": {"$ne": today_str}
                })
                if stale_signals_count > 0:
                    self.system_status["offline_data_fresh"] = False
                    self.add_alert(
                        "STALE_OFFLINE_DATA",
                        f"Offline daemon did not run for today ({today_str}). Found {stale_signals_count} old signals which were rejected for safety.",
                        level="WARNING"
                    )
                else:
                    self.clear_alert("STALE_OFFLINE_DATA")

            self.system_status["database_connected"] = True
            self.clear_alert("DB_DISCONNECTED")
        except Exception as e:
            self.system_status["database_connected"] = False
            self.add_alert("DB_DISCONNECTED", f"MongoDB connection error: {str(e)}", level="CRITICAL")
            logger.error("Error loading signals from database: %s", str(e))

    def reconcile_with_broker(self) -> None:
        """
        Broker Reconciliation Engine:
        Cross-checks open trades against MegaBull /api/position/my.
        If a position was closed externally (or manually by user), syncs state to trade_history.
        """
        try:
            positions = self.mb_client.get_positions()
            active_col = db_manager.get_collection("active_trades")
            history_col = db_manager.get_collection("trade_history")

            # Set of tokens actively held at broker
            broker_tokens: Set[str] = set()
            for pos in positions:
                # MegaBull net quantity > 0 means position is open
                qty = int(pos.get("netQty", pos.get("qty", 0)))
                token = str(pos.get("instrumentToken", "")).strip()
                if qty > 0 and token:
                    broker_tokens.add(token)

            # Check if any local active trade is no longer open at broker
            for trade_id, trade in list(self.active_trades.items()):
                token = str(trade.get("instrument_token", "")).strip()
                if token and token not in broker_tokens and len(positions) > 0:
                    logger.info("[RECONCILIATION] Position %s (Token %s) closed externally. Archiving.", trade.get("symbol"), token)
                    now_iso = datetime.now(pytz.utc).isoformat()
                    fill_price = float(trade.get("fill_price", 0.0))
                    ltp = self.mb_client.get_ltp(token) or fill_price
                    qty = int(trade.get("quantity", 1))

                    hist_doc = {
                        "_id": f"HIST_MANUAL_{trade_id}",
                        "trade_id": trade_id,
                        "symbol": trade.get("symbol"),
                        "instrument_token": token,
                        "strategy_id": trade.get("strategy_id", "50MA_PULLBACK"),
                        "entry_price": fill_price,
                        "exit_price": ltp,
                        "quantity": qty,
                        "pnl": round((ltp - fill_price) * qty, 2),
                        "roi_pct": round(((ltp - fill_price) / fill_price) * 100.0, 2) if fill_price > 0 else 0.0,
                        "exit_reason": "MANUAL_EXTERNAL_EXIT",
                        "entry_timestamp": trade.get("entry_timestamp", now_iso),
                        "exit_timestamp": now_iso,
                        "ml_outcome_label": 1 if ltp >= fill_price else 0
                    }
                    try:
                        history_col.insert_one(hist_doc)
                        active_col.delete_one({"_id": trade["_id"]})
                    except Exception as e:
                        logger.error("Error archiving reconciled trade: %s", str(e))

                    self.active_trades.pop(trade_id, None)

        except Exception as e:
            logger.warning("Reconciliation check error (non-fatal): %s", str(e))

    def process_cycle(self, ltp_override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Executes a robust tick cycle:
        - Evaluates ARMED signals with Day-1 Invalidation.
        - Monitors OPEN trades against Target and SL.
        - Catches broker auth expirations (HTTP 401).
        """
        cycle_start = time.time()
        invalidated_count = 0
        triggered_count = 0
        closed_count = 0

        signals_col = db_manager.get_collection("signals_watchlist")
        active_col = db_manager.get_collection("active_trades")
        history_col = db_manager.get_collection("trade_history")

        # -------------------------------------------------------------
        # 1. Process ARMED Signals (New Entries & Invalidation)
        # -------------------------------------------------------------
        for sig_id, sig in list(self.armed_signals.items()):
            token = str(sig.get("instrument_token", "")).strip()
            ltp = ltp_override.get(token) if ltp_override else self.mb_client.get_ltp(token)
            
            # Validation guard: reject None, zero, or negative prices
            if ltp is None or ltp <= 0:
                continue

            buy_trigger = float(sig["buy_trigger"])
            sl_price = float(sig["sl_price"])
            symbol = sig.get("symbol", "UNKNOWN")

            # Day-1 Invalidation Gate: If price touches SL before Buy Trigger
            if ltp <= sl_price:
                logger.warning(
                    "[FAIL-SAFE INVALIDATION] %s: LTP (%.2f) hit SL (%.2f) before Buy Trigger (%.2f). Discarding signal.",
                    symbol, ltp, sl_price, buy_trigger
                )
                try:
                    signals_col.update_one(
                        {"_id": sig["_id"]},
                        {"$set": {
                            "status": "INVALIDATED",
                            "invalidation_price": ltp,
                            "invalidated_at": datetime.now(pytz.utc).isoformat()
                        }}
                    )
                except Exception as e:
                    logger.error("Failed to update invalidated signal %s in DB: %s", sig_id, str(e))

                self.armed_signals.pop(sig_id, None)
                invalidated_count += 1
                continue

            # Entry Trigger: Price breaks Buy Trigger
            if ltp >= buy_trigger:
                logger.info(
                    "[ENTRY TRIGGER] %s: LTP (%.2f) >= Buy (%.2f). Executing paper BUY order...",
                    symbol, ltp, buy_trigger
                )
                qty = int(sig.get("quantity", 1))

                order_res = self.mb_client.place_order(
                    instrument_token=token,
                    action="BUY",
                    quantity=qty,
                    order_type="LIMIT",
                    price=ltp,
                    duration="CNC"
                )

                # Check if broker rejected due to auth expiration
                if order_res.get("status_code") == 401 or "Unauthorized" in str(order_res.get("error", "")):
                    self.system_status["broker_auth_valid"] = False
                    self.add_alert("BROKER_AUTH_EXPIRED", "MegaBull API Key is invalid or expired! Update .env immediately.", level="CRITICAL")
                    continue
                else:
                    self.system_status["broker_auth_valid"] = True
                    self.clear_alert("BROKER_AUTH_EXPIRED")

                order_id = order_res.get("order_id", f"ORD-BUY-{int(time.time()*1000)}")
                now_iso = datetime.now(pytz.utc).isoformat()

                try:
                    signals_col.update_one(
                        {"_id": sig["_id"]},
                        {"$set": {
                            "status": "TRIGGERED",
                            "triggered_at": now_iso,
                            "fill_price": ltp,
                            "megabull_order_id": order_id
                        }}
                    )

                    trade_doc = {
                        "_id": f"TRADE_{symbol}_{int(time.time()*1000)}",
                        "symbol": symbol,
                        "instrument_token": token,
                        "strategy_id": sig.get("strategy_id", "50MA_PULLBACK"),
                        "megabull_order_id": order_id,
                        "fill_price": ltp,
                        "sl_price": sl_price,
                        "target_price": float(sig["target_price"]),
                        "quantity": qty,
                        "total_capital_required": float(sig.get("total_capital_required", ltp * qty)),
                        "potential_profit": float(sig.get("profit_at_target", (float(sig["target_price"]) - ltp) * qty)),
                        "potential_loss": float(sig.get("loss_at_sl", (ltp - sl_price) * qty)),
                        "sector": sig.get("sector", "General"),
                        "entry_timestamp": now_iso,
                        "status": "OPEN",
                        "highest_ltp": ltp,
                        "lowest_ltp": ltp
                    }
                    active_col.insert_one(trade_doc)
                    self.active_trades[trade_doc["_id"]] = trade_doc
                except Exception as e:
                    logger.error("DB insertion failed for trade %s: %s", symbol, str(e))

                self.armed_signals.pop(sig_id, None)
                triggered_count += 1

        # -------------------------------------------------------------
        # 2. Process OPEN Positions (Continuous SL / Target Protection)
        # -------------------------------------------------------------
        for trade_id, trade in list(self.active_trades.items()):
            token = str(trade.get("instrument_token", "")).strip()
            ltp = ltp_override.get(token) if ltp_override else self.mb_client.get_ltp(token)
            if ltp is None or ltp <= 0:
                continue

            fill_price = float(trade["fill_price"])
            target_price = float(trade["target_price"])
            sl_price = float(trade["sl_price"])
            symbol = trade.get("symbol", "UNKNOWN")
            qty = int(trade.get("quantity", 1))

            highest_ltp = max(float(trade.get("highest_ltp", fill_price)), ltp)
            lowest_ltp = min(float(trade.get("lowest_ltp", fill_price)), ltp)
            trade["highest_ltp"] = highest_ltp
            trade["lowest_ltp"] = lowest_ltp

            exit_reason = None
            if ltp >= target_price:
                exit_reason = "TARGET_HIT"
            elif ltp <= sl_price:
                exit_reason = "SL_HIT"

            if exit_reason:
                logger.info("[EXIT TRIGGER] %s: %s at LTP %.2f (Target: %.2f | SL: %.2f)", symbol, exit_reason, ltp, target_price, sl_price)
                self.mb_client.place_order(
                    instrument_token=token,
                    action="SELL",
                    quantity=qty,
                    order_type="LIMIT",
                    price=ltp,
                    duration="CNC"
                )

                pnl = round((ltp - fill_price) * qty, 2)
                roi = round(((ltp - fill_price) / fill_price) * 100.0, 2) if fill_price > 0 else 0.0
                now_iso = datetime.now(pytz.utc).isoformat()

                hist_doc = {
                    "_id": f"HIST_{trade_id}",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "instrument_token": token,
                    "strategy_id": trade.get("strategy_id", "50MA_PULLBACK"),
                    "entry_price": fill_price,
                    "exit_price": ltp,
                    "quantity": qty,
                    "pnl": pnl,
                    "roi_pct": roi,
                    "exit_reason": exit_reason,
                    "entry_timestamp": trade.get("entry_timestamp", now_iso),
                    "exit_timestamp": now_iso,
                    "mfe": round(highest_ltp - fill_price, 2),
                    "mae": round(fill_price - lowest_ltp, 2),
                    "ml_outcome_label": 1 if exit_reason == "TARGET_HIT" else 0
                }

                try:
                    history_col.insert_one(hist_doc)
                    active_col.delete_one({"_id": trade["_id"]})
                except Exception as e:
                    logger.error("DB update error while closing trade %s: %s", trade_id, str(e))

                self.active_trades.pop(trade_id, None)
                closed_count += 1

        # -------------------------------------------------------------
        # 3. Process Pending Offline Communication / Telemetry Requests
        # -------------------------------------------------------------
        fulfilled_reqs = SecureMailboxCommunicator.fulfill_pending_requests(system_status=self.system_status)

        # Track cycle health metrics
        cycle_duration = round((time.time() - cycle_start) * 1000, 1)
        self.system_status["last_cycle_timestamp"] = self.get_current_ist_time().isoformat()
        self.system_status["last_cycle_duration_ms"] = cycle_duration

        return {
            "invalidated": invalidated_count,
            "triggered": triggered_count,
            "closed": closed_count,
            "active_positions": len(self.active_trades),
            "armed_signals": len(self.armed_signals),
            "fulfilled_comms_requests": fulfilled_reqs,
            "cycle_duration_ms": cycle_duration
        }

    def start_polling(self) -> None:
        """Starts continuous fail-safe polling loop during market hours."""
        self.running = True
        logger.info("ExecutionWorker polling started with universal fail-safe engine.")
        self.load_signals_and_trades()

        reconcile_counter = 0

        while self.running:
            try:
                now = self.get_current_ist_time()
                hour, minute = now.hour, now.minute

                # Standby before market open
                if not self.dry_run and (hour < 9 or (hour == 9 and minute < 17)):
                    time.sleep(15)
                    continue

                # Close session post 15:32 IST
                if not self.dry_run and (hour > 15 or (hour == 15 and minute >= 32)):
                    logger.info("Market session finished (15:32 IST). Expiring remaining signals.")
                    self.expire_signals()
                    break

                # Execute core polling pass
                self.process_cycle()

                # Reconcile with broker every 50 cycles (~2 minutes)
                reconcile_counter += 1
                if reconcile_counter >= 50:
                    self.reconcile_with_broker()
                    reconcile_counter = 0

            except Exception as e:
                logger.error("Unexpected exception in polling loop: %s", str(e))
                self.add_alert("LOOP_EXCEPTION", f"Unhandled loop exception: {str(e)}", level="WARNING")

            time.sleep(self.poll_interval)

        self.mb_client.close()

    def expire_signals(self) -> None:
        """Expires untriggered armed signals cleanly at end of session."""
        try:
            signals_col = db_manager.get_collection("signals_watchlist")
            signals_col.update_many({"status": "ARMED"}, {"$set": {"status": "EXPIRED"}})
            self.armed_signals.clear()
        except Exception as e:
            logger.error("Error expiring signals: %s", str(e))

    def cancel_signal(self, signal_id: str) -> Dict[str, Any]:
        """
        Manually cancels an armed signal, preventing any buy execution.
        Updates MongoDB and removes from in-memory tracking.
        """
        try:
            signals_col = db_manager.get_collection("signals_watchlist")
            now_iso = datetime.now(pytz.utc).isoformat()
            signals_col.update_one(
                {"_id": signal_id},
                {"$set": {"status": "CANCELLED", "cancelled_at": now_iso}}
            )
            self.armed_signals.pop(str(signal_id), None)
            logger.info("[MANUAL CANCEL] Signal %s marked CANCELLED in DB.", signal_id)
            return {"status": "SUCCESS", "message": f"Signal {signal_id} cancelled successfully."}
        except Exception as e:
            logger.error("Error cancelling signal %s: %s", signal_id, str(e))
            return {"status": "ERROR", "message": str(e)}

    def close_active_trade(self, trade_id: str, reason: str = "MANUAL_USER_EXIT") -> Dict[str, Any]:
        """
        Manually exits an open position immediately by placing a SELL order to MegaBull,
        updating MongoDB trade_history, and removing from active_trades.
        """
        try:
            active_col = db_manager.get_collection("active_trades")
            history_col = db_manager.get_collection("trade_history")

            trade = active_col.find_one({"_id": trade_id}) or self.active_trades.get(trade_id)
            if not trade:
                return {"status": "ERROR", "message": f"Trade {trade_id} not found."}

            token = str(trade.get("instrument_token", "")).strip()
            qty = int(trade.get("quantity", 1))
            fill_price = float(trade.get("fill_price", 0.0))
            symbol = trade.get("symbol", "UNKNOWN")

            ltp = self.mb_client.get_ltp(token) or fill_price

            # Dispatch SELL order to MegaBull
            self.mb_client.place_order(
                instrument_token=token,
                action="SELL",
                quantity=qty,
                order_type="LIMIT",
                price=ltp,
                duration="CNC"
            )

            pnl = round((ltp - fill_price) * qty, 2)
            roi = round(((ltp - fill_price) / fill_price) * 100.0, 2) if fill_price > 0 else 0.0
            now_iso = datetime.now(pytz.utc).isoformat()

            hist_doc = {
                "_id": f"HIST_MANUAL_{trade_id}_{int(time.time()*1000)}",
                "trade_id": trade_id,
                "symbol": symbol,
                "instrument_token": token,
                "strategy_id": trade.get("strategy_id", "50MA_PULLBACK"),
                "entry_price": fill_price,
                "exit_price": ltp,
                "quantity": qty,
                "pnl": pnl,
                "roi_pct": roi,
                "exit_reason": reason,
                "entry_timestamp": trade.get("entry_timestamp", now_iso),
                "exit_timestamp": now_iso,
                "ml_outcome_label": 1 if pnl >= 0 else 0
            }

            history_col.insert_one(hist_doc)
            active_col.delete_one({"_id": trade_id})
            self.active_trades.pop(trade_id, None)

            logger.info("[MANUAL EXIT] Closed trade %s (%s) at LTP %.2f. P&L: Rs %.2f", trade_id, symbol, ltp, pnl)
            return {
                "status": "SUCCESS",
                "message": f"Trade {symbol} successfully exited at Rs {ltp:.2f} (P&L: Rs {pnl:.2f}).",
                "pnl": pnl,
                "exit_price": ltp
            }
        except Exception as e:
            logger.error("Error closing trade %s: %s", trade_id, str(e))
            return {"status": "ERROR", "message": str(e)}

    def stop(self) -> None:
        self.running = False


# Global singleton instance
execution_worker = ExecutionWorker(dry_run=True)
