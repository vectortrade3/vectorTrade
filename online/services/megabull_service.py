"""
VectorTrade - MegaBull Paper Trading REST Integration Service
READ-ONLY integration: Pulls user account details, margin allocations,
portfolio holdings with live LTP, active positions, and trade logs.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import requests

from online.config import settings

logger = logging.getLogger("VectorTrade.MegaBullService")
logger.setLevel(logging.INFO)


class MegaBullService:
    """Service for querying account details, holdings, and positions from MegaBull API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_ttl_seconds: int = 10
    ):
        self.api_key = api_key if api_key is not None else settings.megabull_api_key
        self.user_id = user_id if user_id is not None else settings.megabull_user_id
        self.base_url = (base_url if base_url is not None else settings.megabull_base_url).rstrip("/")
        self.cache_ttl = cache_ttl_seconds
        self.timeout = 6.0

        # In-memory cache to prevent hammering broker rate limits
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}

    def _get_headers(self) -> Dict[str, str]:
        """MegaBull API expects 'api-key' and optional 'user-id' headers."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        if self.user_id:
            headers["user-id"] = self.user_id
        return headers

    def _get_cached(self, key: str) -> Optional[Any]:
        """Retrieves cached item if within TTL."""
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < self.cache_ttl:
                return self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Stores item in cache with current timestamp."""
        self._cache[key] = value
        self._cache_time[key] = time.time()

    def get_account_details(self) -> Dict[str, Any]:
        """
        Fetches user profile, virtual capital, blocked margins, and brokerage tier.
        Endpoint: GET /api/user/my
        """
        cached = self._get_cached("account_details")
        if cached is not None:
            return cached

        if not self.api_key:
            return {
                "status": "NOT_CONFIGURED",
                "error": "MEGABULL_API_KEY is not configured in .env",
                "full_name": "MegaBull Account",
                "email": "N/A",
                "virtual_money": 0.0,
                "virtual_money_blocked": 0.0,
                "virtual_money_left": 0.0,
                "brokerage": 20.0,
                "premium_type": "FREE",
                "margin_utilization_pct": 0.0
            }

        url = f"{self.base_url}/api/user/my"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                first = data.get("firstName") or ""
                last = data.get("lastName") or ""
                full_name = f"{first} {last}".strip() or "Vector Trade"

                virtual_money = round(float(data.get("virtualMoney", 0.0)), 2)
                blocked = round(float(data.get("virtualMoneyBlocked", 0.0)), 2)
                left = round(float(data.get("virtualMoneyLeft", 0.0)), 2)
                brokerage = round(float(data.get("brokerage", 20.0)), 2)
                premium_type = data.get("premiumType", "FREE")
                expiry = data.get("premiumExpiry", "")

                util_pct = round((blocked / virtual_money * 100), 2) if virtual_money > 0 else 0.0

                result = {
                    "status": "SUCCESS",
                    "full_name": full_name,
                    "first_name": first,
                    "last_name": last,
                    "email": data.get("emailId", ""),
                    "phone": data.get("phoneNumber"),
                    "virtual_money": virtual_money,
                    "virtual_money_blocked": blocked,
                    "virtual_money_left": left,
                    "brokerage": brokerage,
                    "premium_type": premium_type,
                    "premium_expiry": expiry,
                    "margin_utilization_pct": util_pct
                }
                self._set_cached("account_details", result)
                return result
            else:
                logger.warning(f"MegaBull /api/user/my returned HTTP {resp.status_code}: {resp.text[:100]}")
                return {
                    "status": "ERROR",
                    "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
                    "full_name": "MegaBull User",
                    "virtual_money": 0.0,
                    "virtual_money_blocked": 0.0,
                    "virtual_money_left": 0.0,
                    "brokerage": 20.0,
                    "premium_type": "FREE",
                    "margin_utilization_pct": 0.0
                }
        except Exception as e:
            logger.error(f"Error fetching MegaBull account details: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "full_name": "MegaBull (Offline)",
                "virtual_money": 0.0,
                "virtual_money_blocked": 0.0,
                "virtual_money_left": 0.0,
                "brokerage": 20.0,
                "premium_type": "FREE",
                "margin_utilization_pct": 0.0
            }

    def get_ltp(self, token: str) -> Optional[float]:
        """Fetches latest traded price for a given instrument token."""
        cache_key = f"ltp_{token}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/api/marketwatch/ltp/{token}"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                ltp = float(data.get("ltp", 0.0))
                self._set_cached(cache_key, ltp)
                return ltp
        except Exception as e:
            logger.debug(f"Failed to fetch LTP for token {token}: {e}")
        return None

    def get_holdings(self, enrich_ltp: bool = True) -> List[Dict[str, Any]]:
        """
        Fetches current equity holdings and enriches with live market pricing and P&L.
        Endpoint: GET /api/holding/my
        """
        cached = self._get_cached("holdings")
        if cached is not None:
            return cached

        if not self.api_key:
            return []

        url = f"{self.base_url}/api/holding/my"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"MegaBull /api/holding/my returned HTTP {resp.status_code}")
                return []

            raw_holdings = resp.json()
            if not isinstance(raw_holdings, list):
                return []

            enriched = []
            for item in raw_holdings:
                token = str(item.get("instrumentToken", ""))
                symbol = str(item.get("instrumentName", "")).strip().upper()
                qty = int(item.get("qty", 0))
                lot_size = int(item.get("lotSize", 1))
                h_type = str(item.get("type", "LONG")).upper()
                avg_buy_price = round(float(item.get("priceAvg", item.get("avgBuyPrice", 0.0))), 2)

                invested_val = round(avg_buy_price * abs(qty), 2)

                # Fetch live LTP if requested
                ltp = self.get_ltp(token) if enrich_ltp and token else None
                if ltp is None or ltp <= 0:
                    ltp = avg_buy_price

                current_val = round(ltp * abs(qty), 2)
                if qty >= 0:
                    unrealized_pnl = round(current_val - invested_val, 2)
                else:
                    unrealized_pnl = round(invested_val - current_val, 2)

                pnl_pct = round((unrealized_pnl / invested_val * 100), 2) if invested_val > 0 else 0.0

                enriched.append({
                    "instrument_token": token,
                    "symbol": symbol,
                    "qty": qty,
                    "lot_size": lot_size,
                    "type": h_type,
                    "avg_buy_price": avg_buy_price,
                    "ltp": round(ltp, 2),
                    "invested_value": invested_val,
                    "current_value": current_val,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_pct": pnl_pct,
                    "is_profit": unrealized_pnl >= 0
                })

            # Sort holdings by current value descending
            enriched.sort(key=lambda x: abs(x["current_value"]), reverse=True)
            self._set_cached("holdings", enriched)
            return enriched
        except Exception as e:
            logger.error(f"Error fetching MegaBull holdings: {e}")
            return []

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Fetches active intraday and delivery positions.
        Endpoint: GET /api/position/my
        """
        cached = self._get_cached("positions")
        if cached is not None:
            return cached

        if not self.api_key:
            return []

        url = f"{self.base_url}/api/position/my"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                positions = data if isinstance(data, list) else []
                self._set_cached("positions", positions)
                return positions
            return []
        except Exception as e:
            logger.error(f"Error fetching MegaBull positions: {e}")
            return []

    def get_orders(self) -> Dict[str, Any]:
        """
        Fetches open and executed order logs, excluding any cancelled, rejected, or failed orders.
        Endpoint: GET /api/order/my
        """
        cached = self._get_cached("orders")
        if cached is not None:
            return cached

        if not self.api_key:
            return {"open": [], "executed": []}

        url = f"{self.base_url}/api/order/my"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    # Filter out failed and cancelled orders
                    def is_valid_order(o: Dict[str, Any]) -> bool:
                        st = str(o.get("status", "")).upper()
                        return st not in ("CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED")

                    open_orders = [o for o in data.get("open", []) if is_valid_order(o)]
                    executed_orders = [o for o in data.get("executed", []) if is_valid_order(o)]

                    filtered = {"open": open_orders, "executed": executed_orders}
                    self._set_cached("orders", filtered)
                    return filtered
            return {"open": [], "executed": []}
        except Exception as e:
            logger.error(f"Error fetching MegaBull orders: {e}")
            return {"open": [], "executed": []}

    def get_all_valid_orders(self, holdings: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Builds a comprehensive stream of all valid orders:
        1. Running Orders: Stocks bought and currently held (bought but not sold yet).
        2. Pending Orders: Open orders waiting to execute on MegaBull.
        3. Completed Orders: Executed trade logs.
        Strictly excludes any failed, rejected, or cancelled orders.
        """
        if holdings is None:
            holdings = self.get_holdings(enrich_ltp=True)

        raw_orders = self.get_orders()
        valid_orders: List[Dict[str, Any]] = []

        # 1. Running Orders (Bought but not sold yet - active portfolio holdings)
        for h in holdings:
            token = str(h.get("instrument_token", ""))
            symbol = str(h.get("symbol", ""))
            qty = int(h.get("qty", 1))
            is_long = qty >= 0
            avg_buy_price = float(h.get("avg_buy_price", 0.0))
            ltp = float(h.get("ltp", avg_buy_price))
            invested_val = float(h.get("invested_value", round(avg_buy_price * abs(qty), 2)))
            current_val = float(h.get("current_value", round(ltp * abs(qty), 2)))
            unrealized_pnl = float(h.get("unrealized_pnl", round(current_val - invested_val, 2)))
            pnl_pct = float(h.get("pnl_pct", 0.0))
            is_profit = h.get("is_profit", unrealized_pnl >= 0)

            valid_orders.append({
                "order_id": f"HOLD_{token}" if token else f"HOLD_{symbol}",
                "symbol": symbol,
                "instrument_token": token,
                "category": "RUNNING",
                "category_label": "Running (Bought, Not Sold)",
                "action": "BUY" if is_long else "SELL (SHORT)",
                "quantity": abs(qty),
                "order_price": avg_buy_price,
                "ltp": ltp,
                "capital_put": invested_val,
                "current_valuation": current_val,
                "status": "RUNNING",
                "status_badge": "RUNNING",
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct,
                "is_profit": is_profit,
                "product": "CNC"
            })

        # 2. Pending Orders (Open on MegaBull)
        for po in raw_orders.get("open", []):
            st = str(po.get("status", "PENDING")).upper()
            if st in ("CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED"):
                continue
            symbol = str(po.get("instrumentName", po.get("symbol", ""))).upper()
            qty = int(po.get("qty", po.get("quantity", 0)))
            price = float(po.get("price", po.get("triggerPrice", 0.0)))
            valid_orders.append({
                "order_id": str(po.get("orderId", po.get("id", ""))),
                "symbol": symbol,
                "instrument_token": str(po.get("instrumentToken", "")),
                "category": "PENDING",
                "category_label": "Pending Order",
                "action": str(po.get("action", po.get("type", "BUY"))).upper(),
                "quantity": qty,
                "order_price": price,
                "ltp": price,
                "capital_put": round(price * qty, 2),
                "current_valuation": round(price * qty, 2),
                "status": "PENDING",
                "status_badge": "PENDING",
                "unrealized_pnl": 0.0,
                "pnl_pct": 0.0,
                "is_profit": True,
                "product": str(po.get("product", "CNC")).upper()
            })

        # 3. Completed Orders (Executed on MegaBull)
        for eo in raw_orders.get("executed", []):
            st = str(eo.get("status", "EXECUTED")).upper()
            if st in ("CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED"):
                continue
            symbol = str(eo.get("instrumentName", eo.get("symbol", ""))).upper()
            qty = int(eo.get("qty", eo.get("quantity", 0)))
            price = float(eo.get("price", eo.get("priceAvg", 0.0)))
            valid_orders.append({
                "order_id": str(eo.get("orderId", eo.get("id", ""))),
                "symbol": symbol,
                "instrument_token": str(eo.get("instrumentToken", "")),
                "category": "COMPLETED",
                "category_label": "Completed Order",
                "action": str(eo.get("action", eo.get("type", "BUY"))).upper(),
                "quantity": qty,
                "order_price": price,
                "ltp": price,
                "capital_put": round(price * qty, 2),
                "current_valuation": round(price * qty, 2),
                "status": "COMPLETED",
                "status_badge": "EXECUTED",
                "unrealized_pnl": 0.0,
                "pnl_pct": 0.0,
                "is_profit": True,
                "product": str(eo.get("product", "CNC")).upper()
            })

        return valid_orders

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Aggregates account details, holdings valuation, and positions into unified payload."""
        account = self.get_account_details()
        holdings = self.get_holdings(enrich_ltp=True)
        positions = self.get_positions()
        orders = self.get_orders()

        # Build comprehensive valid orders stream (running, pending, completed)
        valid_orders = self.get_all_valid_orders(holdings=holdings)

        running_orders = [o for o in valid_orders if o["category"] == "RUNNING"]
        pending_orders = [o for o in valid_orders if o["category"] == "PENDING"]
        completed_orders = [o for o in valid_orders if o["category"] == "COMPLETED"]

        total_invested = round(sum(h["invested_value"] for h in holdings), 2)
        total_current = round(sum(h["current_value"] for h in holdings), 2)
        total_pnl = round(sum(h["unrealized_pnl"] for h in holdings), 2)
        total_pnl_pct = round((total_pnl / total_invested * 100), 2) if total_invested > 0 else 0.0

        total_capital_had = account.get("virtual_money", 0.0)
        capital_put_in_stocks = account.get("virtual_money_blocked", 0.0)
        capital_remaining = account.get("virtual_money_left", 0.0)
        net_worth = round(capital_remaining + total_current, 2)

        return {
            "status": account.get("status", "SUCCESS"),
            "account": account,
            "total_capital_had": total_capital_had,
            "capital_put_in_stocks": capital_put_in_stocks,
            "capital_remaining": capital_remaining,
            "holdings": holdings,
            "holdings_count": len(holdings),
            "total_invested": total_invested,
            "total_current": total_current,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "net_worth": net_worth,
            "positions": positions,
            "positions_count": len(positions),
            "orders": orders,
            "valid_orders": valid_orders,
            "valid_orders_count": len(valid_orders),
            "running_orders_count": len(running_orders),
            "pending_orders_count": len(pending_orders),
            "completed_orders_count": len(completed_orders),
            "last_synced": time.strftime("%Y-%m-%d %H:%M:%S")
        }


megabull_service = MegaBullService()
