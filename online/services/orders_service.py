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
            "synced_at": str(doc.get("synced_at", ""))
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


orders_service = OrdersService()
