"""
VectorTrade - Risk Management & Dynamic Position Sizing Calculator
Implements the 1% Risk-per-Trade Excel geometry model with MongoDB persistence:
- Configurable trading capital allocation (custom threshold vs paper broker balance).
- Upper bound guard: min(custom_capital_limit, broker_balance) if limit set, else broker_balance.
- Strict 1% risk ceiling per trade: Quantity = floor(Risk_per_trade / SL_rupee_per_share).
- Mathematical guarantee: Actual Trade Risk <= 1.0% of Capital Base.
- Calculates Target 1 (1:1), Target 2 (1:2), Target 3 (1:3), AMT 1:2, AMT 1:3, and Investment.
"""

from __future__ import annotations
import math
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from online.db import db_manager

logger = logging.getLogger("VectorTrade.RiskCalculator")
logger.setLevel(logging.INFO)

SETTINGS_DOC_ID = "global_trading_settings"
DEFAULT_BASE_CAPITAL = 500000.00
DEFAULT_RISK_PCT = 1.0  # 1% per trade


class RiskCalculator:
    """Dynamic position sizing and capital allocation manager."""

    def __init__(self):
        self._cached_broker_capital: Optional[float] = None
        self._broker_cache_time: float = 0.0
        self._cached_trading_settings: Optional[Dict[str, Any]] = None
        self._settings_cache_time: float = 0.0
        self.settings_cache_ttl: float = 5.0

    def _fetch_broker_total_capital(self) -> float:
        """Pulls total capital from MegaBull paper account with 10s caching."""
        now = time.time()
        if self._cached_broker_capital is not None and (now - self._broker_cache_time) < 10.0:
            return self._cached_broker_capital

        try:
            from online.services.megabull_service import megabull_service
            acc = megabull_service.get_account_details()
            virtual_money = float(acc.get("virtual_money", 0.0))
            if virtual_money > 0:
                self._cached_broker_capital = round(virtual_money, 2)
                self._broker_cache_time = now
                return self._cached_broker_capital
        except Exception as e:
            logger.debug(f"Could not fetch broker account details: {e}")

        # Fallback to default if broker offline
        return DEFAULT_BASE_CAPITAL

    def get_trading_settings(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves current trading capital settings from MongoDB Atlas
        and computes effective capital against the paper trading broker balance.
        Cached in-memory with a 5-second TTL to eliminate latency spikes.
        """
        now = time.time()
        if not force_refresh and self._cached_trading_settings is not None and (now - self._settings_cache_time) < self.settings_cache_ttl:
            return self._cached_trading_settings

        custom_capital_limit: Optional[float] = None
        risk_pct_per_trade: float = DEFAULT_RISK_PCT
        updated_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Fetch saved preference from MongoDB
        try:
            col = db_manager.trading_settings
            if col is not None:
                doc = col.find_one({"_id": SETTINGS_DOC_ID})
                if doc:
                    raw_limit = doc.get("custom_capital_limit")
                    if raw_limit is not None and float(raw_limit) > 0:
                        custom_capital_limit = round(float(raw_limit), 2)
                    risk_pct_per_trade = float(doc.get("risk_pct_per_trade", DEFAULT_RISK_PCT))
                    updated_at = str(doc.get("updated_at", updated_at))
        except Exception as e:
            logger.warning(f"Failed to read trading_settings from MongoDB: {e}")

        # 2. Fetch broker paper money
        broker_total_capital = self._fetch_broker_total_capital()

        # Fetch free available balance and blocked margin if possible
        broker_available = broker_total_capital
        broker_blocked = 0.0
        try:
            from online.services.megabull_service import megabull_service
            acc = megabull_service.get_account_details()
            vm_left = float(acc.get("virtual_money_left", 0.0))
            vm_blocked = float(acc.get("virtual_money_blocked", 0.0))
            if vm_left > 0:
                broker_available = round(vm_left, 2)
            if vm_blocked > 0:
                broker_blocked = round(vm_blocked, 2)
        except Exception:
            pass

        # Calculate actual capital currently blocked / used in positions
        calc_used = round(max(0.0, broker_total_capital - broker_available), 2)
        broker_used_capital = round(max(calc_used, broker_blocked), 2)

        # 3. Apply Threshold Logic:
        # - If user sets a limit less than broker balance, that limit is the active threshold.
        # - If user sets a limit greater than broker balance, ignore user limit and use money we actually have.
        # - If no limit is set (None), by default use total paper money.
        if custom_capital_limit is not None and custom_capital_limit > 0:
            if custom_capital_limit < broker_total_capital:
                effective_capital = custom_capital_limit
                is_custom_active = True
                limit_status = "CUSTOM_THRESHOLD_APPLIED"
            else:
                effective_capital = broker_total_capital
                is_custom_active = False
                limit_status = "EXCEEDS_BROKER_CAPITAL_FALLBACK_TO_ACTUAL"
        else:
            effective_capital = broker_total_capital
            is_custom_active = False
            limit_status = "DEFAULT_FULL_PAPER_CAPITAL"

        # 4. Allocated Capital Remaining Calculation:
        # User Rule:
        # If overall broker capital = 5 Lakh, remaining free cash = 4.10 Lakh (Used = 90k).
        # Threshold set = 1 Lakh.
        # Allocated remaining capital to trade = Threshold - Used = 1 Lakh - 90k = 10k only!
        # If trade amount (quantity * entry) exceeds allocated remaining, do not place trade.
        if is_custom_active:
            allocated_remaining_capital = round(max(0.0, min(effective_capital - broker_used_capital, broker_available)), 2)
            allocated_used_capital = round(min(broker_used_capital, effective_capital), 2)
        else:
            allocated_remaining_capital = round(max(0.0, broker_available), 2)
            allocated_used_capital = broker_used_capital

        allocated_used_pct = round((allocated_used_capital / effective_capital * 100.0), 2) if effective_capital > 0 else 0.0

        risk_per_trade_inr = round(effective_capital * (risk_pct_per_trade / 100.0), 2)

        res = {
            "custom_capital_limit": custom_capital_limit,
            "broker_total_capital": broker_total_capital,
            "broker_available_balance": broker_available,
            "broker_used_capital": broker_used_capital,
            "effective_capital": round(effective_capital, 2),
            "allocated_used_capital": allocated_used_capital,
            "allocated_used_pct": allocated_used_pct,
            "allocated_remaining_capital": allocated_remaining_capital,
            "can_trade": allocated_remaining_capital > 0,
            "risk_pct_per_trade": round(risk_pct_per_trade, 2),
            "risk_per_trade_inr": risk_per_trade_inr,
            "is_custom_limit_active": is_custom_active,
            "limit_status": limit_status,
            "updated_at": updated_at
        }
        self._cached_trading_settings = res
        self._settings_cache_time = now
        return res

    def update_trading_settings(
        self,
        custom_capital_limit: Optional[float] = None,
        risk_pct_per_trade: float = 1.0
    ) -> Dict[str, Any]:
        """
        Updates and persists trading capital settings in MongoDB Atlas.
        Passing custom_capital_limit <= 0 or None resets to full broker paper account balance.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cleaned_limit: Optional[float] = None
        if custom_capital_limit is not None and float(custom_capital_limit) > 0:
            cleaned_limit = round(float(custom_capital_limit), 2)

        cleaned_risk_pct = max(0.1, min(10.0, float(risk_pct_per_trade)))

        doc = {
            "_id": SETTINGS_DOC_ID,
            "custom_capital_limit": cleaned_limit,
            "risk_pct_per_trade": cleaned_risk_pct,
            "updated_at": now_str
        }

        try:
            col = db_manager.trading_settings
            if col is not None:
                col.update_one(
                    {"_id": SETTINGS_DOC_ID},
                    {"$set": doc},
                    upsert=True
                )
                logger.info(
                    f"Trading settings persisted to MongoDB: custom_limit={cleaned_limit}, "
                    f"risk_pct={cleaned_risk_pct}%"
                )
        except Exception as e:
            logger.error(f"Failed to persist trading settings to MongoDB: {e}")

        # Automatically evaluate and allocate candidate orders with the updated capital limit
        try:
            self.evaluate_and_allocate_candidate_orders(
                custom_capital_limit=cleaned_limit,
                risk_pct=cleaned_risk_pct,
                auto_stage_confirmed=True
            )
        except Exception as alloc_err:
            logger.warning(f"Error auto-allocating candidate orders on settings update: {alloc_err}")

        self._cached_trading_settings = None
        self._settings_cache_time = 0.0
        return self.get_trading_settings(force_refresh=True)

    def get_effective_capital(self) -> float:
        """Returns the effective trading capital (in INR) taking into account custom limits."""
        settings_info = self.get_trading_settings()
        return float(settings_info.get("effective_capital", DEFAULT_BASE_CAPITAL))

    def get_allocated_remaining_capital(self) -> float:
        """Returns the remaining allocated capital available for taking trades (in INR)."""
        settings_info = self.get_trading_settings()
        return float(settings_info.get("allocated_remaining_capital", 0.0))

    def check_capital_for_trade(
        self,
        quantity: int,
        entry_price: float,
        action: str = "BUY",
        trading_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validates whether a trade is affordable under the active capital threshold.
        Rule:
        - If overall broker capital = 5 Lakh, remaining free cash = 4.10 Lakh (Used = 90k).
        - If threshold = 1 Lakh -> Allocated Remaining Capital = 1 Lakh - 90k = 10k only!
        - If trade amount (quantity * entry_price) exceeds allocated remaining capital,
          WE DO NOT PUT THAT TRADE.
        - Exits / Square offs (selling existing long holding or covering short) do not consume
          buying allocation and are always allowed.
        """
        act = str(action).strip().upper()
        is_closing_or_exit = (act == "SELL" or "SQUARE" in act or "COVER" in act)

        try:
            qty = abs(int(quantity))
        except (ValueError, TypeError):
            qty = 0

        try:
            px = max(0.0, float(entry_price))
        except (ValueError, TypeError):
            px = 0.0

        trade_amount = round(qty * px, 2)
        settings = trading_settings if trading_settings is not None else self.get_trading_settings()
        allocated_remaining = float(settings.get("allocated_remaining_capital", 0.0))
        effective_capital = float(settings.get("effective_capital", 0.0))
        broker_used = float(settings.get("broker_used_capital", 0.0))

        # Exits and standard long square-offs always permitted since they release capital
        if is_closing_or_exit and not act.startswith("SHORT"):
            return {
                "allowed": True,
                "trade_amount": trade_amount,
                "allocated_remaining_capital": allocated_remaining,
                "shortfall": 0.0,
                "effective_capital": effective_capital,
                "broker_used_capital": broker_used,
                "reason": "Square off / exit order does not consume capital allocation."
            }

        # New trade entry: verify affordability
        if trade_amount <= allocated_remaining:
            return {
                "allowed": True,
                "trade_amount": trade_amount,
                "allocated_remaining_capital": allocated_remaining,
                "shortfall": 0.0,
                "effective_capital": effective_capital,
                "broker_used_capital": broker_used,
                "reason": f"Trade amount ₹{trade_amount:,.2f} is within allocated remaining capital ₹{allocated_remaining:,.2f}."
            }
        else:
            shortfall = round(trade_amount - allocated_remaining, 2)
            return {
                "allowed": False,
                "trade_amount": trade_amount,
                "allocated_remaining_capital": allocated_remaining,
                "shortfall": shortfall,
                "effective_capital": effective_capital,
                "broker_used_capital": broker_used,
                "reason": (
                    f"Trade amount ₹{trade_amount:,.2f} (Qty: {qty} @ ₹{px:,.2f}) exceeds allocated "
                    f"remaining capital ₹{allocated_remaining:,.2f} (Threshold: ₹{effective_capital:,.2f}, "
                    f"Used: ₹{broker_used:,.2f}). Shortfall: ₹{shortfall:,.2f}. Order cannot be placed."
                )
            }

    def calculate_quantity_and_targets(
        self,
        entry_price: float,
        sl_price: float,
        action: str = "BUY",
        custom_portfolio: Optional[float] = None,
        risk_pct: float = 1.0,
        max_position_pct: Optional[float] = None,
        trading_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates position quantity and multi-target payoffs exactly matching the Excel model:
        1. Portfolio = Effective Capital (or custom_portfolio if explicitly passed)
        2. 1% Risk per Trade = Portfolio * 1%
        3. SL Rupee per share = |Entry - StopLoss| (Entry - SL for BUY, SL - Entry for SHORT SELL)
        4. Total shares to trade (Quantity) = floor(Risk per trade / SL Rupee per share)
           * Subject to: Quantity <= floor(Portfolio / Entry) to prevent over-allocation.
           * Mathematical guarantee: Actual Risk <= 1% of Portfolio.
        5. Investment amount = Entry * Quantity
        6. Target 1 (1:1) = Entry +/- SL Rupee per share
        7. Target 2 (1:2) = Entry +/- 2 * SL Rupee per share
        8. Target 3 (1:3) = Entry +/- 3 * SL Rupee per share
        9. AMT 1:2 = Investment + (Quantity * 2 * SL Rupee per share)
        10. AMT 1:3 = Investment + (Quantity * 3 * SL Rupee per share)
        11. Actual RISK = Quantity * SL Rupee per share
        """
        try:
            entry = round(float(entry_price), 2)
        except (ValueError, TypeError):
            entry = 0.0

        try:
            sl = round(float(sl_price), 2)
        except (ValueError, TypeError):
            sl = 0.0

        act = str(action).strip().upper()
        is_short = ("SHORT" in act) or (act == "SELL")

        # 1. Determine Portfolio Base Capital
        if custom_portfolio is not None and float(custom_portfolio) > 0:
            portfolio = round(float(custom_portfolio), 2)
        elif trading_settings is not None:
            portfolio = float(trading_settings.get("effective_capital", DEFAULT_BASE_CAPITAL))
        else:
            portfolio = self.get_effective_capital()

        # 2. Risk per Trade (Default 1.0%)
        risk_pct_clean = max(0.01, float(risk_pct))
        risk_per_trade = round(portfolio * (risk_pct_clean / 100.0), 2)

        # 3. SL Rupee per Share
        if is_short:
            # For Short Sell: StopLoss is above Entry
            raw_sl_diff = sl - entry
        else:
            # For Buy: StopLoss is below Entry
            raw_sl_diff = entry - sl

        if raw_sl_diff <= 0:
            # Fallback to standard 2% stop loss buffer if entry and SL are inverted or equal
            sl_rupee_per_share = max(round(entry * 0.02, 2), 0.05) if entry > 0 else 1.0
            if is_short:
                sl = round(entry + sl_rupee_per_share, 2)
            else:
                sl = round(max(0.05, entry - sl_rupee_per_share), 2)
        else:
            sl_rupee_per_share = round(raw_sl_diff, 2)

        # 4. Total Shares to Trade (Quantity)
        # Quantity = floor(Risk per trade / SL per share)
        if sl_rupee_per_share > 0:
            qty_by_risk = int(math.floor(risk_per_trade / sl_rupee_per_share))
        else:
            qty_by_risk = 0

        # Boundary guard: Capital ceiling & Max Position Size Constraint
        # Professional Risk Parity rule: Cap any single stock at max_position_pct (e.g. 15% of portfolio)
        max_pos_pct = max_position_pct
        if max_pos_pct is None and trading_settings:
            max_pos_pct = trading_settings.get("max_position_pct")

        if max_pos_pct is not None and max_pos_pct > 0 and max_pos_pct < 100:
            max_stock_capital = portfolio * (float(max_pos_pct) / 100.0)
            qty_by_capital = int(math.floor(max_stock_capital / entry)) if entry > 0 else 0
        elif entry > 0:
            qty_by_capital = int(math.floor(portfolio / entry))
        else:
            qty_by_capital = 0

        quantity = max(0, min(qty_by_risk, qty_by_capital))

        # 5. Investment Amount
        investment_amount = round(entry * quantity, 2)

        # 6. Actual Risk (Quantity * SL per share)
        # Strictly <= risk_per_trade
        actual_risk = round(quantity * sl_rupee_per_share, 2)

        # 7. Targets (Risk to Reward 1:1, 1:2, 1:3)
        if is_short:
            target_1_1 = round(entry - sl_rupee_per_share, 2)
            target_1_2 = round(entry - (2.0 * sl_rupee_per_share), 2)
            target_1_3 = round(entry - (3.0 * sl_rupee_per_share), 2)
        else:
            target_1_1 = round(entry + sl_rupee_per_share, 2)
            target_1_2 = round(entry + (2.0 * sl_rupee_per_share), 2)
            target_1_3 = round(entry + (3.0 * sl_rupee_per_share), 2)

        # 8. Amounts at Targets (Investment + Payoff)
        amt_1_1 = round(investment_amount + (quantity * sl_rupee_per_share), 2)
        amt_1_2 = round(investment_amount + (quantity * 2.0 * sl_rupee_per_share), 2)
        amt_1_3 = round(investment_amount + (quantity * 3.0 * sl_rupee_per_share), 2)

        actual_risk_pct = round((actual_risk / portfolio * 100.0), 2) if portfolio > 0 else 0.0

        return {
            "portfolio": portfolio,
            "total_capital": portfolio,
            "risk_pct_per_trade": risk_pct_clean,
            "risk_per_trade": risk_per_trade,
            "entry_price": entry,
            "sl_price": sl,
            "stop_loss_price": sl,
            "action": "SELL" if is_short else "BUY",
            "is_short": is_short,
            "sl_rupee_per_share": sl_rupee_per_share,
            "quantity": quantity,
            "total_shares_to_trade": quantity,
            "investment_amount": investment_amount,
            "actual_risk": actual_risk,
            "actual_risk_pct": actual_risk_pct,
            "within_1_pct_risk": actual_risk <= risk_per_trade,
            "target_1_1": target_1_1,
            "target_1_2": target_1_2,
            "target_1_3": target_1_3,
            "amt_1_1": amt_1_1,
            "amt_1_2": amt_1_2,
            "amt_1_3": amt_1_3
        }

    def get_candidate_setups(self) -> List[Dict[str, Any]]:
        """
        Retrieves candidate setups from MongoDB signals_watchlist,
        with fallback to local daily_analysis.json.
        Filters by enabled strategies and removes active holdings.
        """
        candidates: List[Dict[str, Any]] = []

        # 1. Try reading from db_manager.signals_watchlist
        try:
            col = db_manager.signals_watchlist
            if col is not None:
                docs = list(col.find({"$or": [{"status": {"$in": ["ARMED", "ACTIVE", "READY"]}}, {"status": {"$exists": False}}]}))
                if not docs:
                    docs = list(col.find({}))
                for d in docs:
                    candidates.append(d)
        except Exception as e:
            logger.debug(f"Could not load candidates from MongoDB signals_watchlist: {e}")

        # 2. Fallback to local daily_analysis.json if candidates empty
        if not candidates:
            try:
                from offline import json_manager
                info = json_manager.get_daily_analysis_info()
                finalized = info.get("finalized_trades") or []
                for f in finalized:
                    candidates.append(f)
            except Exception as e:
                logger.debug(f"Could not load fallback candidates from daily_analysis.json: {e}")

        # 3. Filter by enabled strategies
        try:
            from strategies.manager import StrategyManager
            strat_mgr = StrategyManager()
            active_strats = strat_mgr.get_enabled_strategies()
            enabled_ids = {str(s.get("strategy_id", s.get("id"))).strip().lower() for s in active_strats}
            enabled_names = {str(s.get("name", "")).strip().lower() for s in active_strats}

            if enabled_ids or enabled_names:
                filtered = []
                for c in candidates:
                    s_id = str(c.get("strategy_id", "")).strip().lower()
                    s_name = str(c.get("strategy", "")).strip().lower()
                    if (s_id in enabled_ids) or (s_name in enabled_names) or ("50" in s_name and any("50" in x for x in enabled_names)):
                        filtered.append(c)
                candidates = filtered
        except Exception as e:
            logger.debug(f"Could not filter candidates by enabled strategies: {e}")

        # 4. Filter out active holdings (already owned stocks on MegaBull)
        try:
            from online.services.megabull_service import megabull_service
            holdings = megabull_service.get_holdings(enrich_ltp=False)
            held_symbols = {str(h.get("symbol", "")).strip().upper() for h in holdings if int(h.get("qty", 0)) != 0}
            candidates = [c for c in candidates if str(c.get("symbol", "")).strip().upper() not in held_symbols]
        except Exception as e:
            logger.debug(f"Could not filter active holdings from candidates: {e}")

        return candidates

    def evaluate_and_allocate_candidate_orders(
        self,
        custom_capital_limit: Optional[float] = None,
        risk_pct: float = 1.0,
        max_position_pct: Optional[float] = None,
        max_daily_budget_pct: Optional[float] = None,
        max_daily_trades: Optional[int] = None,
        auto_stage_confirmed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates all candidate setups from enabled strategies, ranks them with the best orders
        on top (ARMED status, highest predicted_r, highest profit_pct), and iterates down the list,
        allocating capital according to institutional risk budgeting rules:
        1. 1% Risk per Trade: Absolute loss at SL is guaranteed <= 1% of portfolio.
        2. 15% Max Position Cap: No single stock can take more than 15% of portfolio capital.
        3. 50% Daily Deployment Cap: Never deploy 100% of money in one day; preserves dry powder for future days.
        4. Max 3 New Trades / Day: Confirms the top 3 highest AI expectancy setups.
        Confirmed orders are automatically staged off-market (AMO) in MongoDB for 09:15 AM execution.
        """
        candidates = self.get_candidate_setups()

        # Sort best orders on top:
        # 1. ARMED / READY status on top
        # 2. Predicted R multiple descending (AI expectancy)
        # 3. Profit percentage descending
        # 4. ML confidence descending
        def _candidate_sort_key(c):
            st = str(c.get("status", "ARMED")).upper()
            is_armed = 1 if st in ("ARMED", "READY") else 0
            pred_r = float(c.get("predicted_r", 0.0))
            profit = float(c.get("profit_pct", 0.0))
            conf = float(c.get("ml_confidence", 0.0))
            return (is_armed, pred_r, profit, conf)

        sorted_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)

        # Settings resolution
        settings = self.get_trading_settings()
        broker_total = float(settings.get("broker_total_capital", DEFAULT_BASE_CAPITAL))
        if custom_capital_limit is not None and float(custom_capital_limit) > 0:
            if float(custom_capital_limit) < broker_total:
                effective_cap = round(float(custom_capital_limit), 2)
            else:
                effective_cap = broker_total
        else:
            effective_cap = float(settings.get("effective_capital", broker_total))

        broker_used = float(settings.get("broker_used_capital", 0.0))
        broker_avail = float(settings.get("broker_available_balance", effective_cap))

        # Remaining capital available to allocate
        allocated_remaining = round(max(0.0, min(effective_cap - broker_used, broker_avail)), 2)

        risk_pct_clean = max(0.1, min(10.0, float(risk_pct or settings.get("risk_pct_per_trade", 1.0))))
        risk_per_trade = round(effective_cap * (risk_pct_clean / 100.0), 2)

        # Institutional Portfolio Sizing Constraints
        # 1. Position size ceiling (15% max per stock when effective_cap >= 50k)
        pos_cap_pct = max_position_pct
        if pos_cap_pct is None:
            pos_cap_pct = settings.get("max_position_pct", 15.0 if effective_cap >= 50000.0 else None)

        # 2. Daily capital deployment ceiling (50% max new capital today to keep dry powder)
        daily_budget_pct = max_daily_budget_pct
        if daily_budget_pct is None:
            daily_budget_pct = settings.get("max_daily_budget_pct", 50.0 if effective_cap >= 50000.0 else None)

        # 3. Maximum concurrent new trades per session (top 3 trades)
        trades_limit = max_daily_trades
        if trades_limit is None:
            trades_limit = int(settings.get("max_daily_trades", 3 if effective_cap >= 50000.0 else 50))

        if daily_budget_pct is not None and daily_budget_pct > 0 and daily_budget_pct < 100:
            daily_deployment_limit = round(effective_cap * (float(daily_budget_pct) / 100.0), 2)
            budget_remaining = round(min(allocated_remaining, daily_deployment_limit), 2)
        else:
            budget_remaining = allocated_remaining

        total_committed_capital = 0.0
        total_risk_committed = 0.0
        confirmed_orders: List[Dict[str, Any]] = []
        skipped_orders: List[Dict[str, Any]] = []
        seen_symbols: set = set()

        for idx, cand in enumerate(sorted_candidates, start=1):
            sym = str(cand.get("symbol", "")).strip().upper()
            if not sym or sym in seen_symbols:
                continue
            seen_symbols.add(sym)

            entry_px = float(cand.get("entry_price") or cand.get("buy_trigger") or 0.0)
            sl_px = float(cand.get("sl_price") or 0.0)
            act = str(cand.get("action") or "BUY").upper()

            calc = self.calculate_quantity_and_targets(
                entry_price=entry_px,
                sl_price=sl_px,
                action=act,
                custom_portfolio=effective_cap,
                risk_pct=risk_pct_clean,
                max_position_pct=pos_cap_pct
            )

            qty = calc["quantity"]
            investment = calc["investment_amount"]
            actual_risk = calc["actual_risk"]

            if qty < 1 or investment <= 0:
                skipped_orders.append({
                    "priority_rank": idx,
                    "symbol": sym,
                    "name": cand.get("name", sym),
                    "strategy": cand.get("strategy", cand.get("strategy_id", "50 MA Setup")),
                    "action": act,
                    "entry_price": entry_px,
                    "sl_price": sl_px,
                    "quantity": qty,
                    "investment_amount": investment,
                    "actual_risk": actual_risk,
                    "predicted_r": float(cand.get("predicted_r", 0.0)),
                    "profit_pct": float(cand.get("profit_pct", 0.0)),
                    "allocation_status": "SKIPPED_INVALID_PRICING",
                    "can_take": False,
                    "reason": "Calculated quantity is 0 or invalid entry/SL pricing."
                })
                continue

            if len(confirmed_orders) < trades_limit and investment <= budget_remaining:
                # CONFIRMED!
                budget_remaining = round(budget_remaining - investment, 2)
                total_committed_capital = round(total_committed_capital + investment, 2)
                total_risk_committed = round(total_risk_committed + actual_risk, 2)

                order_doc = {
                    "priority_rank": idx,
                    "symbol": sym,
                    "name": cand.get("name", sym),
                    "strategy": cand.get("strategy", cand.get("strategy_id", "50 MA Setup")),
                    "action": act,
                    "entry_price": entry_px,
                    "sl_price": sl_px,
                    "target_1_1": calc["target_1_1"],
                    "target_1_2": calc["target_1_2"],
                    "target_1_3": calc["target_1_3"],
                    "amt_1_2": calc["amt_1_2"],
                    "amt_1_3": calc["amt_1_3"],
                    "sl_rupee_per_share": calc["sl_rupee_per_share"],
                    "quantity": qty,
                    "investment_amount": investment,
                    "actual_risk": actual_risk,
                    "risk_pct_per_trade": risk_pct_clean,
                    "predicted_r": float(cand.get("predicted_r", 0.0)),
                    "profit_pct": float(cand.get("profit_pct", 0.0)),
                    "allocation_status": "CONFIRMED",
                    "can_take": True,
                    "remaining_budget_after": budget_remaining,
                    "reason": f"Confirmed: ₹{investment:,.2f} allocated (Qty: {qty} @ ₹{entry_px:.2f}). Max 15% pos cap & 1% risk enforced."
                }
                confirmed_orders.append(order_doc)
            else:
                # SKIPPED DUE TO INSUFFICIENT BUDGET OR DAILY LIMIT
                shortfall = round(max(0.0, investment - budget_remaining), 2)
                if len(confirmed_orders) >= trades_limit:
                    skip_reason = f"Daily trade limit reached ({trades_limit} top setups confirmed today; cash buffer reserved for future days)."
                    skip_status = "SKIPPED_DAILY_LIMIT"
                else:
                    skip_reason = f"Insufficient remaining capital (Requires ₹{investment:,.2f}, only ₹{budget_remaining:,.2f} available). Shortfall: ₹{shortfall:,.2f}."
                    skip_status = "SKIPPED_INSUFFICIENT_BUDGET"

                skipped_orders.append({
                    "priority_rank": idx,
                    "symbol": sym,
                    "name": cand.get("name", sym),
                    "strategy": cand.get("strategy", cand.get("strategy_id", "50 MA Setup")),
                    "action": act,
                    "entry_price": entry_px,
                    "sl_price": sl_px,
                    "target_1_2": calc["target_1_2"],
                    "quantity": qty,
                    "investment_amount": investment,
                    "actual_risk": actual_risk,
                    "predicted_r": float(cand.get("predicted_r", 0.0)),
                    "profit_pct": float(cand.get("profit_pct", 0.0)),
                    "allocation_status": skip_status,
                    "can_take": False,
                    "shortfall": shortfall,
                    "reason": skip_reason
                })

        # Auto-stage confirmed orders into db_manager.staged_orders if requested
        if auto_stage_confirmed and db_manager.staged_orders is not None:
            try:
                from online.services.megabull_service import megabull_service
                is_mkt_open = megabull_service.is_market_open()
                acct = megabull_service.get_account_details()
                is_free_tier = acct.get("premium_type", "FREE").upper() == "FREE"
                chosen_order_type = "MKT" if is_free_tier else "LIMIT"

                # Clean up previous unexecuted auto-allocated orders
                db_manager.staged_orders.delete_many({
                    "is_auto_allocated": True,
                    "status": "STAGED_OFFMARKET"
                })
                now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                today_tag = datetime.now().strftime("%Y%m%d")
                staged_docs = []
                for o in confirmed_orders:
                    raw_act = str(o["action"]).upper()
                    norm_act = "SELL" if "SELL" in raw_act or "SHORT" in raw_act else "BUY"
                    sym = o["symbol"]
                    tok = megabull_service.get_instrument_token(sym)
                    staged_docs.append({
                        "_id": f"STAGED_{sym}_{today_tag}",
                        "order_id": f"STAGED_{sym}_{today_tag}",
                        "symbol": sym,
                        "name": o["name"],
                        "action": norm_act,
                        "quantity": o["quantity"],
                        "price": o["entry_price"],
                        "entry_price": o["entry_price"],
                        "sl_price": o["sl_price"],
                        "target_price": o["target_1_2"],
                        "target_1_1": o["target_1_1"],
                        "target_1_2": o["target_1_2"],
                        "target_1_3": o["target_1_3"],
                        "amt_1_2": o["amt_1_2"],
                        "amt_1_3": o["amt_1_3"],
                        "strategy": o["strategy"],
                        "predicted_r": o["predicted_r"],
                        "profit_pct": o["profit_pct"],
                        "investment_amount": o["investment_amount"],
                        "actual_risk": o["actual_risk"],
                        "instrument_token": tok,
                        "product": "CNC",
                        "order_type": chosen_order_type,
                        "status": "STAGED_OFFMARKET",
                        "is_auto_allocated": True,
                        "created_at": now_ts
                    })
                if staged_docs:
                    for doc in staged_docs:
                        db_manager.staged_orders.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
                    logger.info(f"Auto-staged {len(staged_docs)} confirmed orders in MongoDB staged_orders.")

                    # If market is actively open, immediately execute confirmed staged orders on MegaBull
                    if is_mkt_open:
                        logger.info("Market is actively OPEN: Immediately auto-submitting confirmed staged orders to MegaBull.")
                        for doc in staged_docs:
                            try:
                                o_res = megabull_service.place_order(
                                    symbol=doc["symbol"],
                                    action=doc["action"],
                                    quantity=doc["quantity"],
                                    price=doc["entry_price"],
                                    target_price=doc["target_price"],
                                    sl_price=doc["sl_price"],
                                    strategy=doc["strategy"],
                                    product="CNC",
                                    order_type="MKT" if is_free_tier else "LIMIT",
                                    enforce_duplicate_check=False
                                )
                                if o_res.get("status") == "SUCCESS":
                                    is_filled = o_res.get("case") == "FILLED" or o_res.get("order", {}).get("status") == "FILLED"
                                    db_manager.staged_orders.update_one(
                                        {"_id": doc["_id"]},
                                        {"$set": {
                                            "status": "RUNNING" if is_filled else "PENDING",
                                            "broker_order_id": o_res.get("order_id") or o_res.get("order", {}).get("order_id"),
                                            "submitted_at": now_ts,
                                            "fill_price": o_res.get("fill_price") or o_res.get("order", {}).get("fill_price", doc["entry_price"]),
                                            "order_response": o_res
                                        }}
                                    )
                                    logger.info(f"Instant market execution for {doc['symbol']}: {o_res.get('message')}")
                            except Exception as exec_err:
                                logger.error(f"Error executing auto-staged order for {doc['symbol']}: {exec_err}")

                    try:
                        from online.services.orders_service import orders_service
                        orders_service.invalidate_cache()
                    except Exception:
                        pass
            except Exception as stage_e:
                logger.error(f"Failed to auto-stage confirmed orders: {stage_e}")

        plan_summary = {
            "effective_capital": effective_cap,
            "allocated_remaining_capital": allocated_remaining,
            "risk_pct_per_trade": risk_pct_clean,
            "risk_per_trade": risk_per_trade,
            "total_committed_capital": total_committed_capital,
            "total_risk_committed": total_risk_committed,
            "remaining_unallocated_cash": budget_remaining,
            "confirmed_orders_count": len(confirmed_orders),
            "skipped_orders_count": len(skipped_orders),
            "total_candidates_evaluated": len(sorted_candidates),
            "confirmed_orders": confirmed_orders,
            "skipped_orders": skipped_orders,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Persist plan in system_state
        try:
            if db_manager.system_state is not None:
                db_manager.system_state.update_one(
                    {"_id": "portfolio_allocation_plan"},
                    {"$set": plan_summary},
                    upsert=True
                )
        except Exception as save_e:
            logger.debug(f"Could not persist allocation plan: {save_e}")

        return plan_summary

    def get_allocation_plan(self) -> Dict[str, Any]:
        """Returns the latest persisted allocation plan or computes one on the fly."""
        try:
            if db_manager.system_state is not None:
                doc = db_manager.system_state.find_one({"_id": "portfolio_allocation_plan"})
                if doc:
                    doc.pop("_id", None)
                    return doc
        except Exception:
            pass
        return self.evaluate_and_allocate_candidate_orders()


# Global singleton instance
risk_calculator = RiskCalculator()
