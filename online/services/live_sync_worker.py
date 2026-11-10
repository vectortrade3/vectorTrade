"""
VectorTrade - Live Full-Time Market & Order Synchronization Engine
Continuously fetches real-time prices (LTP), synchronizes order lifecycle statuses
(AMO, Pending, Running Bought/Short, Completed), places automated bracket orders (Target & SL),
and updates MongoDB Atlas live state collections.
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

logger = logging.getLogger("VectorTrade.LiveSyncWorker")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [LiveSync] %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

from online.services.megabull_service import megabull_service
from online.services.orders_service import orders_service
from online.db import db_manager


class OrderLifecycleStage:
    """Explicit lifecycle stages for VectorTrade orders."""
    WILL_BE_PUT_AMO = "WILL_BE_PUT_AMO"                     # Queued off-market for 09:15 IST
    PENDING = "PENDING"                                     # Submitted to MegaBull, awaiting price trigger
    RUNNING_BOUGHT = "RUNNING_BOUGHT"                       # Filled long position active in portfolio
    RUNNING_SHORT_SELLED = "RUNNING_SHORT_SELLED"           # Filled short position active in portfolio
    COMPLETED_BOUGHT_TO_SOLD = "COMPLETED_BOUGHT_TO_SOLD"   # Round-trip closed: Bought -> Sold
    COMPLETED_SHORT_TO_COVERED = "COMPLETED_SHORT_TO_COVERED" # Round-trip closed: Short -> Covered


class LiveSyncWorker:
    """
    Full-time worker daemon that:
    1. Fetches and caches live prices (LTP) from MegaBull.
    2. Syncs orders lifecycle across the 4 key stages:
       - Will be put AMO (staged off-market)
       - Pending (order submitted to MegaBull, limit price not yet hit)
       - Order bought or short selled (running holding)
       - Completed (bought -> sold, or short sell -> covered)
    3. Automates Bracket Orders (Target & Stoploss) upon execution.
    4. Handles 1-day expiry by re-arming Target & SL next morning.
    5. Syncs live portfolio balance and P&L state to MongoDB Atlas.
    """

    def __init__(self, interval_seconds: int = 5):
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._stop_requested = False
        self.last_sync_time: Optional[str] = None
        self.sync_count: int = 0
        self.last_error: Optional[str] = None
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.last_summary: Dict[str, Any] = {}
        self._monitored_symbols: Set[str] = set()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the live sync worker."""
        return {
            "is_running": self.is_running,
            "interval_seconds": self.interval_seconds,
            "last_sync_time": self.last_sync_time,
            "sync_count": self.sync_count,
            "cached_symbols_count": len(self.price_cache),
            "monitored_symbols_count": len(self._monitored_symbols),
            "last_error": self.last_error,
            "last_summary": self.last_summary,
            "market_open": megabull_service.is_market_open()
        }

    def stop(self):
        """Requests graceful shutdown of the worker loop."""
        self._stop_requested = True
        self.is_running = False
        logger.info("LiveSyncWorker stop requested.")

    def fetch_and_update_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Gathers symbols from active holdings, pending orders, staged orders, and MongoDB watchlist,
        fetches their latest traded prices (LTP) from MegaBull, and updates the MongoDB `live_quotes` collection.
        """
        target_symbols: Set[str] = set(symbols) if symbols else set()

        if not target_symbols:
            # 1. From active holdings
            try:
                for h in megabull_service.get_holdings(enrich_ltp=False):
                    sym = str(h.get("symbol", "")).strip().upper()
                    if sym:
                        target_symbols.add(sym)
            except Exception as e:
                logger.debug(f"Could not load holdings for price fetch: {e}")

            # 2. From open orders
            try:
                for po in megabull_service.get_orders().get("open", []):
                    sym = str(po.get("instrumentName", po.get("symbol", ""))).strip().upper()
                    if sym:
                        target_symbols.add(sym)
            except Exception as e:
                logger.debug(f"Could not load open orders for price fetch: {e}")

            # 3. From MongoDB staged orders
            try:
                if db_manager.staged_orders is not None:
                    for doc in db_manager.staged_orders.find({"status": {"$in": ["STAGED_OFFMARKET", "PENDING", "ACTIVE"]}}):
                        sym = str(doc.get("symbol", "")).strip().upper()
                        if sym:
                            target_symbols.add(sym)
            except Exception as e:
                logger.debug(f"Could not load staged orders for price fetch: {e}")

            # 4. From MongoDB signals watchlist
            try:
                if db_manager.signals_watchlist is not None:
                    for doc in db_manager.signals_watchlist.find({}, {"symbol": 1}).limit(20):
                        sym = str(doc.get("symbol", "")).strip().upper()
                        if sym:
                            target_symbols.add(sym)
            except Exception as e:
                logger.debug(f"Could not load signals watchlist for price fetch: {e}")

        # Baseline fallback list if nothing active
        if not target_symbols:
            target_symbols = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "TATAMOTORS"}

        self._monitored_symbols = target_symbols
        quotes = megabull_service.fetch_live_quotes(list(target_symbols))

        # Update in-memory price cache
        for sym, q in quotes.items():
            self.price_cache[sym] = q

        return {
            "status": "SUCCESS",
            "updated_count": len(quotes),
            "quotes": quotes
        }

    def sync_orders_lifecycle(self) -> Dict[str, Any]:
        """
        Synchronizes all order lifecycle states between MegaBull broker and MongoDB Atlas:
        1. WILL_BE_PUT_AMO: Check if market opened; if open, submits orders or triggers liquidation.
        2. PENDING: Checks if open broker order was filled on MegaBull; transitions to RUNNING.
        3. RUNNING (BOUGHT / SHORT SELLED): Calculates live P&L, places/re-arms Target & SL brackets.
        4. COMPLETED: Trade exited; cancels opposing brackets, records completion metadata & Net P&L.
        """
        market_open = megabull_service.is_market_open()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Pull live broker state
        raw_holdings = megabull_service.get_holdings(enrich_ltp=True)
        raw_orders = megabull_service.get_orders()
        open_orders = raw_orders.get("open", [])
        executed_orders = raw_orders.get("executed", [])

        active_symbols_map = {
            str(h.get("symbol", "")).strip().upper(): h
            for h in raw_holdings if int(h.get("qty", 0)) != 0
        }

        # Track transitions
        amo_fired_count = 0
        pending_to_running_count = 0
        brackets_placed_count = 0
        completed_count = 0
        orders_by_stage = {
            OrderLifecycleStage.WILL_BE_PUT_AMO: 0,
            OrderLifecycleStage.PENDING: 0,
            OrderLifecycleStage.RUNNING_BOUGHT: 0,
            OrderLifecycleStage.RUNNING_SHORT_SELLED: 0,
            OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD: 0,
            OrderLifecycleStage.COMPLETED_SHORT_TO_COVERED: 0
        }

        # ------------------------------------------------------------------
        # STAGE 1: WILL_BE_PUT_AMO (Staged Off-Market Orders)
        # ------------------------------------------------------------------
        try:
            col_staged = db_manager.staged_orders
            if col_staged is not None:
                staged_docs = list(col_staged.find({"status": {"$in": ["STAGED_OFFMARKET", "AMO"]}}))
                for so in staged_docs:
                    s_id = so.get("_id")
                    s_sym = str(so.get("symbol", "")).strip().upper()
                    s_qty = int(so.get("quantity", so.get("qty", 1)))
                    s_action = str(so.get("action", so.get("type", "BUY"))).upper()
                    s_price = float(so.get("price", so.get("entry_price", 0.0)))
                    s_token = str(so.get("instrument_token", megabull_service.get_instrument_token(s_sym)))
                    is_liq = bool(so.get("is_liquidation_order", False))

                    orders_by_stage[OrderLifecycleStage.WILL_BE_PUT_AMO] += 1

                    # If market is OPEN, execute staged AMO order
                    if market_open:
                        if is_liq:
                            logger.info(f"Market is OPEN: Executing emergency liquidation market exit for {s_sym}.")
                            liq_res = megabull_service.liquidate_all_positions(force_market_open=True)
                            col_staged.update_one(
                                {"_id": s_id},
                                {"$set": {"status": "EXECUTED", "executed_at": timestamp_str, "result": liq_res}}
                            )
                            amo_fired_count += 1
                        else:
                            logger.info(f"Market is OPEN: Submitting staged AMO entry order for {s_sym} ({s_action} {s_qty} @ {s_price}).")
                            order_res = megabull_service.place_order(
                                symbol=s_sym,
                                action=s_action,
                                quantity=s_qty,
                                price=s_price,
                                target_price=so.get("target_price"),
                                sl_price=so.get("sl_price"),
                                strategy=so.get("strategy", "50 MA Setup"),
                                product=so.get("product", "CNC"),
                                order_type=so.get("order_type", "LIMIT"),
                                enforce_duplicate_check=False
                            )
                            if order_res.get("status") == "SUCCESS":
                                col_staged.update_one(
                                    {"_id": s_id},
                                    {"$set": {
                                        "status": "PENDING" if order_res.get("case") != "FILLED" else "RUNNING",
                                        "broker_order_id": order_res.get("order_id"),
                                        "submitted_at": timestamp_str,
                                        "order_response": order_res
                                    }}
                                )
                                amo_fired_count += 1
        except Exception as e:
            logger.error(f"Error handling STAGE 1 (WILL_BE_PUT_AMO): {e}")

        # ------------------------------------------------------------------
        # STAGE 2: PENDING (Submitted to MegaBull, Awaiting Price Trigger)
        # ------------------------------------------------------------------
        for po in open_orders:
            po_sym = str(po.get("instrumentName", po.get("symbol", ""))).strip().upper()
            po_price = float(po.get("price", po.get("triggerPrice", 0.0)))
            po_id = str(po.get("orderId", po.get("id", "")))
            orders_by_stage[OrderLifecycleStage.PENDING] += 1

            # Check if this pending order was executed (holding is now present in active holdings)
            if po_sym in active_symbols_map:
                logger.info(f"Pending order {po_id} for {po_sym} executed! Transitioning to RUNNING.")
                pending_to_running_count += 1
                try:
                    if db_manager.staged_orders is not None:
                        db_manager.staged_orders.update_one(
                            {"symbol": po_sym, "status": "PENDING"},
                            {"$set": {"status": "ACTIVE", "executed_at": timestamp_str, "fill_price": po_price}}
                        )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # STAGE 3: RUNNING (Order Bought or Short Selled)
        # ------------------------------------------------------------------
        signals_map = {}
        try:
            if db_manager.signals_watchlist is not None:
                for doc in db_manager.signals_watchlist.find({}):
                    sym = str(doc.get("symbol", "")).strip().upper()
                    if sym:
                        signals_map[sym] = doc
        except Exception:
            pass

        for sym, holding in active_symbols_map.items():
            qty = int(holding.get("qty", 1))
            is_long = qty > 0
            if is_long:
                orders_by_stage[OrderLifecycleStage.RUNNING_BOUGHT] += 1
            else:
                orders_by_stage[OrderLifecycleStage.RUNNING_SHORT_SELLED] += 1

            # Bracket Order Automation: Check if Target & SL are active on MegaBull
            sig = signals_map.get(sym, {})
            target_px = float(sig.get("target_price", holding.get("target_price", 0.0)))
            sl_px = float(sig.get("sl_price", holding.get("sl_price", 0.0)))

            # If target/SL not in signal, generate from standard 1:2 risk/reward geometry
            entry_px = float(holding.get("avg_buy_price", 0.0))
            if (target_px <= 0 or sl_px <= 0) and entry_px > 0:
                if is_long:
                    sl_px = round(entry_px * 0.98, 2)
                    target_px = round(entry_px * 1.04, 2)
                else:
                    sl_px = round(entry_px * 1.02, 2)
                    target_px = round(entry_px * 0.96, 2)

            has_open_bracket = any(
                str(bo.get("instrumentName", bo.get("symbol", ""))).strip().upper() == sym
                for bo in open_orders
            )

            # If market is open and brackets are not currently placed, place them automatically!
            if market_open and not has_open_bracket and target_px > 0 and sl_px > 0:
                logger.info(f"Automating Bracket Orders for running holding {sym}: Target @ {target_px}, SL @ {sl_px}")
                bracket_res = megabull_service.place_bracket_orders(sym, holding, target_px, sl_px)
                brackets_placed_count += 1
                try:
                    if db_manager.staged_orders is not None:
                        db_manager.staged_orders.update_one(
                            {"symbol": sym},
                            {"$set": {
                                "brackets_placed": True,
                                "target_order": bracket_res.get("target_order"),
                                "sl_order": bracket_res.get("sl_order"),
                                "last_bracket_sync": timestamp_str
                            }},
                            upsert=True
                        )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # STAGE 4: COMPLETED (Bought -> Sold, or Short Sell -> Covered)
        # ------------------------------------------------------------------
        # Detect trades that were previously tracked as ACTIVE in MongoDB but are no longer in active holdings
        try:
            col_staged = db_manager.staged_orders
            if col_staged is not None:
                active_tracked = list(col_staged.find({"status": "ACTIVE"}))
                for tracked in active_tracked:
                    t_sym = str(tracked.get("symbol", "")).strip().upper()
                    if t_sym not in active_symbols_map:
                        # Holding has been closed (Target hit, SL hit, or squared off)
                        logger.info(f"Holding {t_sym} exited! Trade completed. Cancelling opposing open brackets.")
                        megabull_service.cancel_open_orders(t_sym)

                        t_action = str(tracked.get("action", "BUY")).upper()
                        is_short_roundtrip = "SELL" in t_action or "SHORT" in t_action
                        completion_stage = (
                            OrderLifecycleStage.COMPLETED_SHORT_TO_COVERED
                            if is_short_roundtrip
                            else OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD
                        )
                        orders_by_stage[completion_stage] += 1
                        completed_count += 1

                        col_staged.update_one(
                            {"_id": tracked["_id"]},
                            {"$set": {
                                "status": "COMPLETED",
                                "lifecycle_stage": completion_stage,
                                "outcome": "TARGET_OR_SL_HIT",
                                "completed_at": timestamp_str
                            }}
                        )
        except Exception as e:
            logger.error(f"Error handling STAGE 4 (COMPLETED): {e}")

        # Account for historical completed trades (e.g. TVSMOTOR)
        historical_closed = megabull_service.get_historical_completed_trades()
        for hc in historical_closed:
            stg = hc.get("lifecycle_stage", OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD)
            if stg in orders_by_stage:
                orders_by_stage[stg] += 1

        return {
            "status": "SUCCESS",
            "market_open": market_open,
            "timestamp": timestamp_str,
            "amo_fired_count": amo_fired_count,
            "pending_to_running_count": pending_to_running_count,
            "brackets_placed_count": brackets_placed_count,
            "completed_count": completed_count,
            "orders_by_stage": orders_by_stage
        }

    def sync_portfolio_state(self) -> Dict[str, Any]:
        """
        Synchronizes overall account balance, portfolio valuation, unrealized/realized P&L,
        and saves a unified snapshot into MongoDB `system_state` collection (`_id: "portfolio_live_state"`).
        """
        summary = megabull_service.get_portfolio_summary()
        state_doc = {
            "_id": "portfolio_live_state",
            "account_balance": summary.get("total_capital_had", 0.0),
            "capital_put_in_stocks": summary.get("capital_put_in_stocks", 0.0),
            "capital_remaining": summary.get("capital_remaining", 0.0),
            "total_invested": summary.get("total_invested", 0.0),
            "current_valuation": summary.get("total_current", 0.0),
            "net_worth": summary.get("net_worth", 0.0),
            "total_pnl": summary.get("total_pnl", 0.0),
            "total_pnl_pct": summary.get("total_pnl_pct", 0.0),
            "total_charges_paid": summary.get("total_charges_paid", 0.0),
            "running_orders_count": summary.get("running_orders_count", 0),
            "pending_orders_count": summary.get("pending_orders_count", 0),
            "amo_orders_count": summary.get("amo_orders_count", 0),
            "completed_orders_count": summary.get("completed_orders_count", 0),
            "orders_by_stage": summary.get("orders_by_stage", {}),
            "market_open": megabull_service.is_market_open(),
            "last_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            if db_manager.system_state is not None:
                db_manager.system_state.update_one(
                    {"_id": "portfolio_live_state"},
                    {"$set": state_doc},
                    upsert=True
                )
        except Exception as e:
            logger.debug(f"Could not persist portfolio_live_state: {e}")

        # Automatically update persistent trades record CSV on disk
        try:
            from online.services.orders_service import orders_service
            orders_service.save_trades_record_csv()
        except Exception as e:
            logger.debug(f"Could not auto-save trades_record.csv: {e}")

        return state_doc

    def tick(self) -> Dict[str, Any]:
        """
        Executes one complete live synchronization cycle:
        1. Fetches and caches live prices (LTP).
        2. Reconciles and manages order lifecycle (AMO, Pending, Running, Completed).
        3. Syncs portfolio state to MongoDB Atlas.
        """
        start_t = time.time()
        self.sync_count += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_sync_time = now_str

        try:
            prices_res = self.fetch_and_update_prices()
            lifecycle_res = self.sync_orders_lifecycle()
            portfolio_res = self.sync_portfolio_state()

            elapsed = round(time.time() - start_t, 3)
            summary = {
                "status": "SUCCESS",
                "tick_number": self.sync_count,
                "timestamp": now_str,
                "elapsed_seconds": elapsed,
                "market_open": lifecycle_res.get("market_open", False),
                "prices_updated": prices_res.get("updated_count", 0),
                "amo_fired": lifecycle_res.get("amo_fired_count", 0),
                "brackets_placed": lifecycle_res.get("brackets_placed_count", 0),
                "completed_count": lifecycle_res.get("completed_count", 0),
                "orders_by_stage": lifecycle_res.get("orders_by_stage", {}),
                "portfolio_net_worth": portfolio_res.get("net_worth", 0.0),
                "total_pnl": portfolio_res.get("total_pnl", 0.0)
            }
            self.last_summary = summary
            self.last_error = None
            return summary
        except Exception as e:
            logger.error(f"Error during LiveSyncWorker tick: {e}", exc_info=True)
            self.last_error = str(e)
            return {
                "status": "ERROR",
                "tick_number": self.sync_count,
                "timestamp": now_str,
                "error": str(e)
            }

    def start_sync_loop(self, interval_seconds: Optional[int] = None, run_once: bool = False):
        """
        Runs the live synchronization loop continuously (for standalone CLI daemon execution).
        """
        if interval_seconds:
            self.interval_seconds = interval_seconds

        self.is_running = True
        self._stop_requested = False
        logger.info(f"LiveSyncWorker started. Poll interval: {self.interval_seconds}s (run_once={run_once})")

        try:
            while not self._stop_requested:
                res = self.tick()
                logger.info(
                    f"Tick #{res.get('tick_number')} | Market: {'OPEN' if res.get('market_open') else 'CLOSED'} | "
                    f"Prices: {res.get('prices_updated', 0)} | "
                    f"AMO: {res.get('orders_by_stage', {}).get(OrderLifecycleStage.WILL_BE_PUT_AMO, 0)} | "
                    f"Pending: {res.get('orders_by_stage', {}).get(OrderLifecycleStage.PENDING, 0)} | "
                    f"Bought: {res.get('orders_by_stage', {}).get(OrderLifecycleStage.RUNNING_BOUGHT, 0)} | "
                    f"Short: {res.get('orders_by_stage', {}).get(OrderLifecycleStage.RUNNING_SHORT_SELLED, 0)} | "
                    f"Done: {res.get('orders_by_stage', {}).get(OrderLifecycleStage.COMPLETED_BOUGHT_TO_SOLD, 0) + res.get('orders_by_stage', {}).get(OrderLifecycleStage.COMPLETED_SHORT_TO_COVERED, 0)}"
                )
                if run_once:
                    break
                time.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            logger.info("LiveSyncWorker received KeyboardInterrupt. Exiting cleanly.")
        finally:
            self.is_running = False

    async def run_async_loop(self, interval_seconds: Optional[int] = None):
        """
        Runs the live synchronization loop asynchronously (for FastAPI lifespan background task).
        """
        if interval_seconds:
            self.interval_seconds = interval_seconds

        self.is_running = True
        self._stop_requested = False
        logger.info(f"Async LiveSyncWorker started. Poll interval: {self.interval_seconds}s")

        try:
            while not self._stop_requested:
                # Run the synchronous tick in a thread pool to avoid blocking the async event loop
                await asyncio.to_thread(self.tick)
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            logger.info("Async LiveSyncWorker task cancelled cleanly.")
        finally:
            self.is_running = False


# Export singleton instance
live_sync_worker = LiveSyncWorker()
