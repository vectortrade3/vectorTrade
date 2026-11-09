"""
VectorTrade - Online Orders Domain Service
Handles business logic for retrieving, formatting, filtering, and summarizing
daily trade setups and buy orders stored in MongoDB Atlas `signals_watchlist`.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from online.db import db_manager
from online.config import settings

logger = logging.getLogger("VectorTrade.OrdersService")
logger.setLevel(logging.INFO)


class OrdersService:
    """Service for querying and analyzing live orders from MongoDB."""

    @staticmethod
    def normalize_order(doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts MongoDB document into a normalized UI/API order object.
        Gracefully handles both legacy and modern schema variations.
        """
        if not doc:
            return {}

        order_id = str(doc.get("_id", ""))
        symbol = str(doc.get("symbol", "")).strip().upper()
        name = str(doc.get("name", symbol)).strip()
        sector = str(doc.get("sector", "General")).strip()
        date = str(doc.get("valid_for_date", doc.get("date", ""))).strip()

        strategy = str(doc.get("strategy", doc.get("strategy_id", "50 MA Setup"))).strip()
        # Clean up strategy name display
        if strategy in ("50MA_PULLBACK", "50_MA_SETUP"):
            strategy = "50 MA Setup"
        elif strategy == "GLOBAL_FORECAST":
            strategy = "Global Forecast"

        setup_subclass = str(doc.get("setup_subclass", "")).strip()
        setup_num = int(doc.get("setup_num", 1))

        # Default setup labels if missing
        if not setup_subclass:
            if setup_num == 1:
                setup_subclass = "50MA Setup 1 (Confluence Zone)"
            elif setup_num == 2:
                setup_subclass = "50MA Setup 2 (Extended Trend)"
            elif setup_num == 3:
                setup_subclass = "50MA Setup 3 (Mean-Reverting)"
            else:
                setup_subclass = f"Setup {setup_num}"

        action = str(doc.get("action", "BUY")).strip().upper()
        if "STRONG" in action:
            action = "STRONG BUY"
        elif "BUY" in action:
            action = "BUY"

        # Price levels
        entry_price = float(doc.get("entry_price", doc.get("buy_trigger", 0.0)))
        sl_price = float(doc.get("sl_price", 0.0))
        target_price = float(doc.get("target_price", 0.0))

        risk_per_share = max(round(entry_price - sl_price, 2), 0.01) if entry_price > sl_price else 1.0

        # Quantity and capital calculations
        default_budget = settings.default_risk_budget
        quantity = int(doc.get("quantity", max(1, int(default_budget / risk_per_share))))
        total_capital = float(doc.get("total_capital_required", round(entry_price * quantity, 2)))
        loss_at_sl = float(doc.get("loss_at_sl", round(risk_per_share * quantity, 2)))
        profit_at_target = float(doc.get("profit_at_target", round((target_price - entry_price) * quantity, 2)))

        loss_pct = float(doc.get("loss_pct", round((risk_per_share / entry_price * 100), 2) if entry_price > 0 else 0.0))
        profit_pct = float(doc.get("profit_pct", round(((target_price - entry_price) / entry_price * 100), 2) if entry_price > 0 else 0.0))
        risk_reward = str(doc.get("risk_reward", "1 : 2.00"))
        risk_reward_num = float(doc.get("risk_reward_num", 2.0))

        predicted_r = float(doc.get("predicted_r", 0.0))
        ml_confidence = float(doc.get("ml_confidence", round(min(max(predicted_r / 2.0, 0.0), 1.0), 2)))

        status = str(doc.get("status", doc.get("outcome", "ARMED"))).strip().upper()
        outcome_detail = str(doc.get("outcome_detail", "")).strip()
        if not outcome_detail:
            if status == "ARMED":
                outcome_detail = f"Armed for Execution: Buy Trigger ₹{entry_price:.2f}, SL ₹{sl_price:.2f}, Target ₹{target_price:.2f} (1:2 R:R)."
            elif status == "ACTIVE":
                outcome_detail = f"Trade Active: Filled at ₹{entry_price:.2f}."
            else:
                outcome_detail = f"Status: {status}"

        rationale = str(doc.get("rationale", "")).strip()
        if not rationale:
            rationale = f"{strategy} - {setup_subclass}. Strict 1:2.00 R:R geometry with AI expectancy of +{predicted_r:.2f} R."

        return {
            "id": order_id,
            "order_id": order_id,
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "strategy": strategy,
            "setup_subclass": setup_subclass,
            "setup_num": setup_num,
            "action": action,
            "buy_trigger": round(entry_price, 2),
            "entry_price": round(entry_price, 2),
            "sl_price": round(sl_price, 2),
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "quantity": quantity,
            "total_capital_required": round(total_capital, 2),
            "loss_at_sl": round(loss_at_sl, 2),
            "profit_at_target": round(profit_at_target, 2),
            "loss_pct": round(loss_pct, 2),
            "profit_pct": round(profit_pct, 2),
            "risk_reward": risk_reward,
            "risk_reward_num": round(risk_reward_num, 2),
            "predicted_r": round(predicted_r, 2),
            "ml_confidence": round(ml_confidence, 2),
            "status": status,
            "outcome_detail": outcome_detail,
            "rationale": rationale,
            "rsi": float(doc.get("rsi", 0.0)),
            "atr": float(doc.get("atr", 0.0)),
            "ma50": float(doc.get("ma50", 0.0)),
            "dist_low_bb": float(doc.get("dist_low_bb", 0.0)),
            "dist_low_ma50": float(doc.get("dist_low_ma50", 0.0)),
            "is_latest": bool(doc.get("is_latest", True)),
            "valid_for_date": date,
            "synced_at": str(doc.get("synced_at", "")),
            "is_in_portfolio": bool(doc.get("is_in_portfolio", False)),
            "portfolio_status": str(doc.get("portfolio_status", "NONE")),
            "portfolio_reason": doc.get("portfolio_reason")
        }

    def get_orders(
        self,
        status: Optional[str] = None,
        strategy: Optional[str] = None,
        setup_num: Optional[int] = None,
        search: Optional[str] = None,
        session_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Queries signals_watchlist collection with filters, applies normalization,
        and returns orders sorted with ARMED first, then by predicted_r descending.
        """
        col = db_manager.signals_watchlist
        if col is None:
            logger.warning("MongoDB signals_watchlist collection is unavailable.")
            return []

        query: Dict[str, Any] = {}

        # Status filter
        if status and status.upper() != "ALL":
            query["status"] = status.upper()

        # Strategy filter
        if strategy and strategy.upper() != "ALL":
            strat_clean = strategy.strip()
            # Match either plain name or slug
            query["$or"] = [
                {"strategy": strat_clean},
                {"strategy_id": strat_clean.replace(" ", "_").upper()}
            ]

        # Setup filter
        if setup_num is not None and setup_num > 0:
            query["setup_num"] = setup_num

        # Date filter
        if session_date:
            query["valid_for_date"] = session_date

        # Search filter (Symbol or Name)
        if search:
            search_str = search.strip()
            regex = {"$regex": re.escape(search_str), "$options": "i"}
            if "$or" in query:
                # Merge into an $and query
                query = {"$and": [query, {"$or": [{"symbol": regex}, {"name": regex}]}]}
            else:
                query["$or"] = [{"symbol": regex}, {"name": regex}]

        try:
            cursor = col.find(query).limit(limit)
            raw_docs = list(cursor)
            orders = [self.normalize_order(doc) for doc in raw_docs]

            # Enrich orders with live MegaBull portfolio presence (duplicate check)
            try:
                from online.services.megabull_service import megabull_service
                holdings = megabull_service.get_holdings(enrich_ltp=False)
                open_orders = megabull_service.get_orders().get("open", [])

                held_symbols = {str(h.get("symbol", "")).strip().upper(): h for h in holdings if int(h.get("qty", 0)) != 0}
                pending_symbols = {str(o.get("instrumentName", o.get("symbol", ""))).strip().upper(): o for o in open_orders}

                for o in orders:
                    sym = o.get("symbol", "").upper()
                    if sym in held_symbols:
                        o["is_in_portfolio"] = True
                        o["portfolio_status"] = "RUNNING_HOLDING"
                        o["portfolio_reason"] = f"Symbol '{sym}' is already active in your portfolio holdings (Qty: {held_symbols[sym].get('qty')})."
                    elif sym in pending_symbols:
                        o["is_in_portfolio"] = True
                        o["portfolio_status"] = "ORDER_PENDING"
                        o["portfolio_reason"] = f"Symbol '{sym}' already has a pending order waiting on MegaBull."
                    else:
                        o["is_in_portfolio"] = False
                        o["portfolio_status"] = "NONE"
                        o["portfolio_reason"] = None
            except Exception as e:
                logger.debug(f"Could not enrich orders with portfolio status: {e}")
                for o in orders:
                    o["is_in_portfolio"] = False
                    o["portfolio_status"] = "NONE"
                    o["portfolio_reason"] = None

            # Sorting: ARMED first, then by predicted_r desc, then symbol
            def sort_key(o):
                is_armed = 1 if o["status"] == "ARMED" else 0
                return (is_armed, o["predicted_r"], o["profit_pct"])

            orders.sort(key=sort_key, reverse=True)
            return orders
        except Exception as e:
            logger.error(f"Error querying signals_watchlist: {e}")
            return []

    def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Returns a single order by ID or None."""
        col = db_manager.signals_watchlist
        if col is None:
            return None
        try:
            doc = col.find_one({"_id": order_id})
            if doc:
                return self.normalize_order(doc)
            return None
        except Exception as e:
            logger.error(f"Error finding order {order_id}: {e}")
            return None

    def get_order_summary_stats(self) -> Dict[str, Any]:
        """Computes summary KPI statistics across all current signals."""
        col = db_manager.signals_watchlist
        if col is None:
            return {
                "total_orders": 0,
                "armed_orders_count": 0,
                "active_orders_count": 0,
                "avg_predicted_r": 0.0,
                "avg_profit_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_capital_required": 0.0,
                "total_risk_budget": 0.0,
                "strategies_breakdown": {},
                "setups_breakdown": {},
                "latest_date": "",
                "last_sync_timestamp": ""
            }

        try:
            all_docs = list(col.find({}))
            if not all_docs:
                return {
                    "total_orders": 0,
                    "armed_orders_count": 0,
                    "active_orders_count": 0,
                    "avg_predicted_r": 0.0,
                    "avg_profit_pct": 0.0,
                    "avg_loss_pct": 0.0,
                    "total_capital_required": 0.0,
                    "total_risk_budget": 0.0,
                    "strategies_breakdown": {},
                    "setups_breakdown": {},
                    "latest_date": "",
                    "last_sync_timestamp": ""
                }

            normalized = [self.normalize_order(d) for d in all_docs]
            total_orders = len(normalized)

            armed_orders = [o for o in normalized if o["status"] == "ARMED"]
            active_orders = [o for o in normalized if o["status"] == "ACTIVE"]

            armed_count = len(armed_orders)
            active_count = len(active_orders)

            # For KPIs, focus on ARMED setups ready for market execution
            kpi_subset = armed_orders if armed_orders else normalized

            pred_r_vals = [o["predicted_r"] for o in kpi_subset if o["predicted_r"] > 0]
            avg_pred_r = round(sum(pred_r_vals) / len(pred_r_vals), 2) if pred_r_vals else 0.0

            profit_pct_vals = [o["profit_pct"] for o in kpi_subset if o["profit_pct"] > 0]
            avg_profit_pct = round(sum(profit_pct_vals) / len(profit_pct_vals), 2) if profit_pct_vals else 0.0

            loss_pct_vals = [o["loss_pct"] for o in kpi_subset if o["loss_pct"] > 0]
            avg_loss_pct = round(sum(loss_pct_vals) / len(loss_pct_vals), 2) if loss_pct_vals else 0.0

            total_capital = round(sum(o["total_capital_required"] for o in armed_orders), 2)
            total_risk = round(sum(o["loss_at_sl"] for o in armed_orders), 2)

            # Breakdown by strategy and setup
            strat_counts: Dict[str, int] = {}
            setup_counts: Dict[str, int] = {}
            for o in normalized:
                s = o["strategy"]
                strat_counts[s] = strat_counts.get(s, 0) + 1
                sub = o["setup_subclass"]
                setup_counts[sub] = setup_counts.get(sub, 0) + 1

            dates = [o["valid_for_date"] for o in normalized if o["valid_for_date"]]
            latest_date = max(dates) if dates else ""

            synced_times = [o["synced_at"] for o in normalized if o["synced_at"]]
            last_sync = max(synced_times) if synced_times else ""

            return {
                "total_orders": total_orders,
                "armed_orders_count": armed_count,
                "active_orders_count": active_count,
                "avg_predicted_r": avg_pred_r,
                "avg_profit_pct": avg_profit_pct,
                "avg_loss_pct": avg_loss_pct,
                "total_capital_required": total_capital,
                "total_risk_budget": total_risk,
                "strategies_breakdown": strat_counts,
                "setups_breakdown": setup_counts,
                "latest_date": latest_date,
                "last_sync_timestamp": last_sync
            }
        except Exception as e:
            logger.error(f"Error computing order summary stats: {e}")
            return {
                "total_orders": 0,
                "armed_orders_count": 0,
                "active_orders_count": 0,
                "avg_predicted_r": 0.0,
                "avg_profit_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_capital_required": 0.0,
                "total_risk_budget": 0.0,
                "strategies_breakdown": {},
                "setups_breakdown": {},
                "latest_date": "",
                "last_sync_timestamp": ""
            }

    @staticmethod
    def calculate_trade_journal_row(trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates all 17 trading ledger columns from the 4 primary inputs
        (Stock Name, Entry, Stoploss, Target):
        1. Stock Name
        2. Buy Date
        3. Strategy Name
        4. TF (Timeframe: Daily)
        5. QTY (Quantity)
        6. Entry
        7. Stoploss
        8. RISK per share (|Entry - Stoploss|)
        9. Sale Date (blank until trade is completed)
        10. Target 1:2 (Entry +/- 2*Risk)
        11. Target 1:3 (Entry +/- 3*Risk)
        12. AMT. Invested (QTY * Entry)
        13. EXTRA P&L (Charges / dividend, negative regulatory fees)
        14. AMT. On Exit (blank until trade is completed)
        15. P/L Per Share (blank until trade is completed)
        16. P/L Per Share % (blank until trade is completed)
        17. NET P/L (blank until trade is completed)
        """
        symbol = str(trade.get("symbol", trade.get("stock_name", "UNKNOWN"))).strip().upper()

        # 2. Buy Date
        buy_date = str(trade.get("buy_date", trade.get("valid_for_date", trade.get("date", trade.get("created_at", "2026-11-01"))))).strip()[:10]
        if not buy_date or buy_date == "None":
            buy_date = "2026-11-01"

        # 3. Strategy Name
        strategy = str(trade.get("strategy_name", trade.get("strategy", trade.get("strategy_id", "50 MA Setup")))).strip()
        if strategy in ("50MA_PULLBACK", "50_MA_SETUP"):
            strategy = "50 MA Setup"
        elif strategy == "GLOBAL_FORECAST":
            strategy = "Global Forecast"
        elif not strategy:
            strategy = "50 MA Setup"

        # 4. Timeframe (TF)
        tf = str(trade.get("tf", "Daily")).strip() or "Daily"

        # 5. QTY
        raw_qty = trade.get("quantity", trade.get("qty", 1))
        try:
            qty = abs(int(raw_qty))
        except (ValueError, TypeError):
            qty = 1
        if qty == 0:
            qty = 1

        is_short = int(raw_qty) < 0 or "SHORT" in str(trade.get("action", "")).upper() or str(trade.get("action", "")).upper() == "SELL"

        # 6. Entry
        try:
            entry = round(float(trade.get("entry", trade.get("entry_price", trade.get("buy_trigger", trade.get("order_price", trade.get("avg_buy_price", trade.get("price", 0.0))))))), 2)
        except (ValueError, TypeError):
            entry = 0.0

        # 7. Stoploss
        raw_sl = trade.get("stoploss", trade.get("sl_price", trade.get("sl")))
        if raw_sl is not None and float(raw_sl) > 0:
            stoploss = round(float(raw_sl), 2)
        else:
            # Default 2% stoploss geometry if not provided
            stoploss = round(entry * 1.02, 2) if is_short else round(entry * 0.98, 2)

        # 8. RISK per share
        risk_per_share = round(abs(entry - stoploss), 2)
        if risk_per_share <= 0:
            risk_per_share = max(round(entry * 0.02, 2), 1.0)

        # 10. Target 1:2 & 11. Target 1:3
        raw_tgt = trade.get("target", trade.get("target_price", trade.get("target_1_2")))
        if is_short:
            target_1_2 = round(entry - (2.0 * risk_per_share), 2)
            target_1_3 = round(entry - (3.0 * risk_per_share), 2)
        else:
            target_1_2 = round(entry + (2.0 * risk_per_share), 2)
            target_1_3 = round(entry + (3.0 * risk_per_share), 2)

        # 12. AMT. Invested
        amt_invested = round(qty * entry, 2)

        # 13. EXTRA P&L (Charges / dividend)
        raw_charges = trade.get("total_charges", trade.get("charges", trade.get("brokerage_and_taxes")))
        if raw_charges is not None and float(raw_charges) > 0:
            charges_val = round(float(raw_charges), 2)
        else:
            from online.services.megabull_service import megabull_service
            charges_calc = megabull_service.calculate_regulatory_charges(
                buy_price=entry,
                sell_price=entry if not is_short else entry,
                quantity=qty,
                product="MIS" if is_short else "CNC"
            )
            charges_val = charges_calc.get("total_charges", 40.0)
        extra_pnl = -round(charges_val, 2)

        # Status Check: Running vs Completed
        status = str(trade.get("status", "RUNNING")).strip().upper()
        category = str(trade.get("category", "")).strip().upper()
        is_completed = (
            status in ("COMPLETED", "CLOSED", "TARGET_HIT", "SL_HIT", "RESOLVED")
            or category == "COMPLETED"
            or bool(trade.get("closed_date"))
        )

        if is_completed:
            # 9. Sale Date
            sale_date = str(trade.get("closed_date", trade.get("sale_date", trade.get("exit_date", "2026-10-29")))).strip()[:10]
            if not sale_date or sale_date == "None":
                sale_date = "2026-10-29"

            # 14. AMT. On Exit
            exit_price = float(trade.get("exit_price", trade.get("ltp", entry)))
            amt_on_exit = round(qty * exit_price, 2)

            # 15. P/L Per Share
            if is_short:
                pnl_per_share = round(entry - exit_price, 2)
            else:
                pnl_per_share = round(exit_price - entry, 2)

            # 16. P/L Per Share %
            pnl_per_share_pct = round((pnl_per_share / entry * 100), 2) if entry > 0 else 0.0

            # 17. NET P/L
            gross_pnl = round(qty * pnl_per_share, 2)
            net_pnl = round(gross_pnl + extra_pnl, 2)
        else:
            # RUNNING trade: Sale date and exit parameters MUST be blank until trade is completed
            sale_date = ""
            amt_on_exit = None
            pnl_per_share = None
            pnl_per_share_pct = None
            net_pnl = None

        return {
            "stock_name": symbol,
            "symbol": symbol,
            "buy_date": buy_date,
            "strategy_name": strategy,
            "strategy": strategy,
            "tf": tf,
            "qty": qty,
            "quantity": qty,
            "entry": entry,
            "entry_price": entry,
            "stoploss": stoploss,
            "sl_price": stoploss,
            "risk_per_share": risk_per_share,
            "sale_date": sale_date,
            "target_1_2": target_1_2,
            "target_1_3": target_1_3,
            "amt_invested": amt_invested,
            "extra_pnl": extra_pnl,
            "amt_on_exit": amt_on_exit,
            "pnl_per_share": pnl_per_share,
            "pnl_per_share_pct": pnl_per_share_pct,
            "net_pnl": net_pnl,
            "status": "COMPLETED" if is_completed else "RUNNING",
            "action": "SELL (SHORT)" if is_short else "BUY (LONG)",
            "is_completed": is_completed,
            "current_ltp": float(trade.get("ltp", entry))
        }

    def get_trade_journal(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all running holdings and completed trades, cross-references
        MongoDB Atlas signals for exact 4 parameters (Stock Name, Entry, SL, Target),
        and dynamically computes all 17 journal columns.
        """
        from online.services.megabull_service import megabull_service

        # 1. Gather all running holdings from MegaBull
        holdings = megabull_service.get_holdings(enrich_ltp=True)
        # 2. Gather completed trades from MegaBull report ledger
        completed = megabull_service.get_historical_completed_trades()
        # 3. Gather staged / tracked orders from MongoDB
        staged = []
        try:
            col = db_manager.staged_orders
            if col is not None:
                staged = list(col.find({"status": {"$in": ["ACTIVE", "COMPLETED", "PLACED"]}}))
        except Exception as e:
            logger.debug(f"Could not query staged_orders: {e}")

        # Build symbol lookup from MongoDB signals_watchlist
        signals_map: Dict[str, Any] = {}
        try:
            sig_col = db_manager.signals_watchlist
            if sig_col is not None:
                for doc in sig_col.find({}):
                    sym = str(doc.get("symbol", "")).strip().upper()
                    if sym and sym not in signals_map:
                        signals_map[sym] = doc
        except Exception as e:
            logger.debug(f"Could not build signals map: {e}")

        journal_rows: List[Dict[str, Any]] = []
        seen_symbols = set()

        # A. Process running holdings
        for h in holdings:
            sym = str(h.get("symbol", "")).strip().upper()
            if not sym or int(h.get("qty", 0)) == 0:
                continue
            seen_symbols.add(sym)

            sig = signals_map.get(sym, {})
            trade_raw = {
                "symbol": sym,
                "buy_date": sig.get("valid_for_date", "2026-11-01"),
                "strategy": sig.get("strategy", "50 MA Setup"),
                "tf": "Daily",
                "qty": h.get("qty", 1),
                "entry": h.get("avg_buy_price", sig.get("buy_trigger", 0.0)),
                "stoploss": sig.get("sl_price"),
                "target": sig.get("target_price"),
                "total_charges": h.get("total_charges"),
                "status": "RUNNING",
                "ltp": h.get("ltp", h.get("avg_buy_price", 0.0)),
                "action": "SELL (SHORT)" if int(h.get("qty", 1)) < 0 else "BUY (LONG)"
            }
            journal_rows.append(self.calculate_trade_journal_row(trade_raw))

        # B. Process historical completed trades
        for c in completed:
            sym = str(c.get("symbol", "")).strip().upper()
            sig = signals_map.get(sym, {})
            trade_raw = {
                "symbol": sym,
                "buy_date": sig.get("valid_for_date", "2026-10-28"),
                "strategy": sig.get("strategy", "50 MA Setup"),
                "tf": "Daily",
                "qty": c.get("quantity", 1),
                "entry": c.get("order_price", sig.get("buy_trigger", 0.0)),
                "stoploss": sig.get("sl_price"),
                "target": sig.get("target_price"),
                "total_charges": c.get("total_charges"),
                "status": "COMPLETED",
                "closed_date": c.get("closed_date", "2026-10-29"),
                "exit_price": c.get("exit_price", 0.0),
                "action": c.get("action", "BUY & SELL (CLOSED)")
            }
            journal_rows.append(self.calculate_trade_journal_row(trade_raw))

        # C. Process any completed / active staged orders not yet seen
        for s in staged:
            sym = str(s.get("symbol", "")).strip().upper()
            if sym in seen_symbols and s.get("status") != "COMPLETED":
                continue
            trade_raw = {
                "symbol": sym,
                "buy_date": str(s.get("created_at", "2026-11-01"))[:10],
                "strategy": s.get("strategy", "50 MA Setup"),
                "tf": "Daily",
                "qty": s.get("quantity", 1),
                "entry": s.get("entry_price", s.get("price", 0.0)),
                "stoploss": s.get("sl_price"),
                "target": s.get("target_price"),
                "status": s.get("status", "RUNNING"),
                "closed_date": str(s.get("completed_at", ""))[:10],
                "exit_price": s.get("target_price", s.get("price", 0.0))
            }
            journal_rows.append(self.calculate_trade_journal_row(trade_raw))

        if status_filter:
            sf = status_filter.strip().upper()
            if sf in ("RUNNING", "ACTIVE"):
                journal_rows = [r for r in journal_rows if not r["is_completed"]]
            elif sf in ("COMPLETED", "CLOSED"):
                journal_rows = [r for r in journal_rows if r["is_completed"]]

        return journal_rows


orders_service = OrdersService()
