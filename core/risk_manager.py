"""
VectorTrade - Quantitative Portfolio Risk Manager
Enforces institutional-grade risk constraints, dynamic position sizing,
sector concentration limits, cross-strategy correlation deduplication,
and portfolio-level circuit breakers prior to MongoDB dispatch.
Runs strictly within the Offline pipeline, adding ZERO latency to the Online worker.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set

from core.config import settings
from core.db import db_manager

logger = logging.getLogger("VectorTrade.RiskManager")


class PortfolioRiskManager:
    """
    Evaluates prospective signals from all strategies and applies:
    1. Capital-at-Risk Position Sizing: Quantity = floor(Risk_Per_Trade / (Buy - SL)).
    2. Concurrent Open Position Cap (e.g. max 5 simultaneous trades across portfolio).
    3. Sector Concentration Cap (e.g. maximum 30% of open trades in any single sector).
    4. Cross-Strategy Deduplication: Discards lower-confidence duplicate signals on same symbol.
    5. Portfolio Circuit Breaker: Suspends new arming if cumulative open risk exceeds threshold.
    """

    def __init__(
        self,
        total_capital: float = 100000.0,
        risk_per_trade_pct: float = 1.5,      # 1.5% of total capital per trade
        max_open_positions: int = 5,           # Max 5 concurrent positions
        max_sector_allocation_pct: float = 30.0, # Max 30% of total positions per sector
        max_portfolio_risk_pct: float = 7.5    # Circuit breaker: max 7.5% total active risk
    ) -> None:
        self.total_capital = float(total_capital)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.max_open_positions = int(max_open_positions)
        self.max_sector_allocation_pct = float(max_sector_allocation_pct)
        self.max_portfolio_risk_pct = float(max_portfolio_risk_pct)

        # Risk amount in currency per trade
        self.risk_amount_per_trade = self.total_capital * (self.risk_per_trade_pct / 100.0)
        self.max_portfolio_risk_currency = self.total_capital * (self.max_portfolio_risk_pct / 100.0)

    def calculate_position_size(self, buy_trigger: float, sl_price: float) -> int:
        """
        Dynamic lot sizing based on stop loss distance:
        Quantity = floor(Risk_Amount / (Buy - SL))
        Strictly guarantees risk <= risk_amount_per_trade (<= 1.0% capital).
        If 1 share exceeds the risk budget, returns 0.
        """
        risk_per_share = buy_trigger - sl_price
        if risk_per_share <= 0:
            return 0
        quantity = int(self.risk_amount_per_trade // risk_per_share)
        return max(quantity, 0)

    def get_current_portfolio_state(self) -> Dict[str, Any]:
        """
        Inspects MongoDB active_trades to determine current open exposure.
        """
        active_col = db_manager.get_collection("active_trades")
        active_trades = list(active_col.find({"status": "OPEN"}))

        sector_counts: Dict[str, int] = {}
        total_open_risk = 0.0

        for trade in active_trades:
            # Query sector from universe or trade doc
            sector = str(trade.get("sector", "General"))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

            buy = float(trade.get("fill_price", trade.get("buy_trigger", 0.0)))
            sl = float(trade.get("sl_price", 0.0))
            qty = int(trade.get("quantity", 1))
            risk_per_share = max(buy - sl, 0.0)
            total_open_risk += (risk_per_share * qty)

        return {
            "active_count": len(active_trades),
            "sector_counts": sector_counts,
            "total_open_risk": total_open_risk,
            "active_symbols": {t["symbol"] for t in active_trades if "symbol" in t}
        }

    def filter_and_size_signals(
        self,
        raw_signals: List[Dict[str, Any]],
        universe: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters, sizes, and ranks raw strategy signals through multi-tiered risk gates.
        """
        if not raw_signals:
            return []

        universe = universe or {}
        portfolio = self.get_current_portfolio_state()
        current_active = portfolio["active_count"]
        sector_counts = dict(portfolio["sector_counts"])
        current_open_risk = portfolio["total_open_risk"]
        active_symbols = set(portfolio["active_symbols"])

        # 1. Portfolio Circuit Breaker Check
        if current_open_risk >= self.max_portfolio_risk_currency:
            logger.warning(
                "[CIRCUIT BREAKER] Cumulative active risk (%.2f) exceeds threshold (%.2f). Halting new signals.",
                current_open_risk, self.max_portfolio_risk_currency
            )
            return []

        slots_available = max(self.max_open_positions - current_active, 0)
        if slots_available <= 0:
            logger.info(
                "[MAX POSITIONS CAP] Already holding %d open positions (Max: %d). No new slots available.",
                current_active, self.max_open_positions
            )
            return []

        # 2. Cross-Strategy Correlation & Duplicate Symbol Deduplication
        # If multiple strategies signal on the same symbol, retain the highest ML confidence
        grouped_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for sig in raw_signals:
            sym = sig["symbol"]
            grouped_by_symbol.setdefault(sym, []).append(sig)

        deduplicated_candidates: List[Dict[str, Any]] = []
        for sym, sigs in grouped_by_symbol.items():
            if sym in active_symbols:
                logger.info("[CORRELATION FILTER] Symbol %s is already OPEN in active_trades. Skipping.", sym)
                continue

            # Sort by ML confidence descending
            sigs.sort(key=lambda x: float(x.get("ml_confidence", 0.5)), reverse=True)
            chosen_sig = sigs[0]
            if len(sigs) > 1:
                logger.info(
                    "[DEDUPLICATION] Multiple signals on %s. Selected Strategy '%s' (Conf: %.3f) over %d other(s).",
                    sym, chosen_sig["strategy_id"], chosen_sig.get("ml_confidence", 0.5), len(sigs) - 1
                )
            deduplicated_candidates.append(chosen_sig)

        # 3. Sort all candidates across the universe by conviction (ML confidence)
        deduplicated_candidates.sort(key=lambda x: float(x.get("ml_confidence", 0.5)), reverse=True)

        approved_signals: List[Dict[str, Any]] = []
        accumulated_risk = current_open_risk

        # Max allowed positions per sector
        max_per_sector = max(int(self.max_open_positions * (self.max_sector_allocation_pct / 100.0)), 1)

        for sig in deduplicated_candidates:
            if len(approved_signals) >= slots_available:
                logger.info("[CAP REACHED] Allocated all %d available portfolio slots.", slots_available)
                break

            sym = sig["symbol"]
            sector = universe.get(sym, {}).get("sector", "General")
            sig["sector"] = sector

            # Sector Concentration Filter
            current_sector_alloc = sector_counts.get(sector, 0)
            if current_sector_alloc >= max_per_sector:
                logger.warning(
                    "[SECTOR FILTER] Rejecting %s. Sector '%s' reached limit (%d/%d positions).",
                    sym, sector, current_sector_alloc, max_per_sector
                )
                continue

            # Position Sizing based on Stop Loss Distance
            buy_trigger = float(sig["buy_trigger"])
            sl_price = float(sig["sl_price"])
            quantity = self.calculate_position_size(buy_trigger, sl_price)

            if quantity <= 0:
                logger.warning("[SIZING ERROR] Invalid quantity for %s (Buy: %.2f, SL: %.2f). Discarding.", sym, buy_trigger, sl_price)
                continue

            proposed_risk = (buy_trigger - sl_price) * quantity
            if (accumulated_risk + proposed_risk) > self.max_portfolio_risk_currency:
                logger.warning(
                    "[CIRCUIT BREAKER] Adding %s risk (%.2f) would breach total portfolio risk cap (%.2f). Skipping.",
                    sym, proposed_risk, self.max_portfolio_risk_currency
                )
                continue

            # Update signal payload with risk-sized parameters
            sig["quantity"] = quantity
            sig["risk_allocated"] = round(proposed_risk, 2)
            sig["capital_allocated"] = round(buy_trigger * quantity, 2)

            # Update tracker state
            sector_counts[sector] = current_sector_alloc + 1
            accumulated_risk += proposed_risk
            approved_signals.append(sig)

            logger.info(
                "[RISK APPROVED] %s (%s) | Qty: %d, Allocated Risk: Rs %.2f, Capital: Rs %.2f (Conf: %.3f)",
                sym, sig["strategy_id"], quantity, proposed_risk, sig["capital_allocated"], sig.get("ml_confidence", 0.5)
            )

        return approved_signals
