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
        Fetches open and executed order logs.
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
                    self._set_cached("orders", data)
                    return data
            return {"open": [], "executed": []}
        except Exception as e:
            logger.error(f"Error fetching MegaBull orders: {e}")
            return {"open": [], "executed": []}

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Aggregates account details, holdings valuation, and positions into unified payload."""
        account = self.get_account_details()
        holdings = self.get_holdings(enrich_ltp=True)
        positions = self.get_positions()
        orders = self.get_orders()

        total_invested = round(sum(h["invested_value"] for h in holdings), 2)
        total_current = round(sum(h["current_value"] for h in holdings), 2)
        total_pnl = round(sum(h["unrealized_pnl"] for h in holdings), 2)
        total_pnl_pct = round((total_pnl / total_invested * 100), 2) if total_invested > 0 else 0.0

        cash_left = account.get("virtual_money_left", 0.0)
        net_worth = round(cash_left + total_current, 2)

        return {
            "status": account.get("status", "SUCCESS"),
            "account": account,
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
            "open_orders_count": len(orders.get("open", [])),
            "executed_orders_count": len(orders.get("executed", [])),
            "last_synced": time.strftime("%Y-%m-%d %H:%M:%S")
        }


megabull_service = MegaBullService()
