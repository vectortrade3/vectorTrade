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

    def get_trading_settings(self) -> Dict[str, Any]:
        """
        Retrieves current trading capital settings from MongoDB Atlas
        and computes effective capital against the paper trading broker balance.
        """
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

        # Fetch free available balance if possible
        broker_available = broker_total_capital
        try:
            from online.services.megabull_service import megabull_service
            acc = megabull_service.get_account_details()
            vm_left = float(acc.get("virtual_money_left", 0.0))
            if vm_left > 0:
                broker_available = round(vm_left, 2)
        except Exception:
            pass

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

        risk_per_trade_inr = round(effective_capital * (risk_pct_per_trade / 100.0), 2)

        return {
            "custom_capital_limit": custom_capital_limit,
            "broker_total_capital": broker_total_capital,
            "broker_available_balance": broker_available,
            "effective_capital": round(effective_capital, 2),
            "risk_pct_per_trade": round(risk_pct_per_trade, 2),
            "risk_per_trade_inr": risk_per_trade_inr,
            "is_custom_limit_active": is_custom_active,
            "limit_status": limit_status,
            "updated_at": updated_at
        }

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

        return self.get_trading_settings()

    def get_effective_capital(self) -> float:
        """Returns the effective trading capital (in INR) taking into account custom limits."""
        settings_info = self.get_trading_settings()
        return float(settings_info.get("effective_capital", DEFAULT_BASE_CAPITAL))

    def calculate_quantity_and_targets(
        self,
        entry_price: float,
        sl_price: float,
        action: str = "BUY",
        custom_portfolio: Optional[float] = None,
        risk_pct: float = 1.0
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

        # Boundary guard: Capital ceiling (Total investment cannot exceed Portfolio capital)
        if entry > 0:
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


# Global singleton instance
risk_calculator = RiskCalculator()
