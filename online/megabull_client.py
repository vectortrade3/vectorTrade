"""
VectorTrade - Production MegaBull REST API Client
Ultra-lightweight HTTP client utilizing requests.Session with connection pooling,
rate-limit resilience, exponential backoff, and strict timeout enforcement.
Strict constraint: NO pandas, numpy, scikit-learn, or xgboost.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config import settings

logger = logging.getLogger("VectorTrade.MegaBullClient")


class MegaBullClient:
    """
    Stateful REST Client for MegaBull Paper Trading Simulator.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = (api_key or settings.MEGABULL_API_KEY).strip().strip("'\"")
        self.base_url = (base_url or settings.MEGABULL_BASE_URL).rstrip("/")
        self.timeout = settings.HTTP_TIMEOUT_SECONDS
        self.max_retries = settings.HTTP_MAX_RETRIES

        if not self.api_key:
            raise ValueError("MEGABULL_API_KEY must be provided or configured in .env")

        # Initialize persistent connection pool with automated backoff for 429/500/502/503/504
        self.session = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"]
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.headers = {
            "api-key": self.api_key,
            "User-Agent": f"{settings.APP_NAME}/1.0 (Low-Latency Online Engine)",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if settings.MEGABULL_USER_ID:
            self.headers["user-id"] = str(settings.MEGABULL_USER_ID)

    def get_ltp(self, instrument_token: str) -> Optional[float]:
        """
        Queries GET /api/marketwatch/ltp/{instrumentToken}.
        Returns float price or None if query fails.
        """
        token = str(instrument_token).strip()
        url = f"{self.base_url}/api/marketwatch/ltp/{token}"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("ltp")
                return float(price) if price is not None else None
            logger.warning("GET LTP %s returned HTTP %d: %s", token, resp.status_code, resp.text[:100])
        except Exception as e:
            logger.error("Exception fetching LTP for token %s: %s", token, str(e))
        return None

    def place_order(
        self,
        instrument_token: str,
        action: str,
        quantity: int,
        order_type: str = "LIMIT",
        price: float = 0.0,
        trigger_price: float = 0.0,
        duration: str = "CNC"
    ) -> Dict[str, Any]:
        """
        Dispatches POST /api/order/buysell.
        Payload format:
        {
            "instrumentToken": "738561",
            "qty": 50,
            "type": "BUY",
            "duration": "CNC",
            "orderType": "LIMIT",
            "price": 1190.0,
            "triggerPrice": 0.0
        }
        """
        url = f"{self.base_url}/api/order/buysell"
        payload = {
            "instrumentToken": str(instrument_token),
            "qty": int(quantity),
            "type": action.upper(),
            "duration": duration.upper(),
            "orderType": order_type.upper(),
            "price": float(price),
            "triggerPrice": float(trigger_price)
        }

        try:
            logger.info("Dispatching %s order on token %s (Qty: %d, Price: %.2f)...", action, instrument_token, quantity, price)
            resp = self.session.post(url, headers=self.headers, json=payload, timeout=self.timeout)
            
            if resp.status_code in (200, 201):
                res_data = resp.json()
                logger.info("Order executed successfully: %s", res_data)
                return {
                    "success": True,
                    "status_code": resp.status_code,
                    "data": res_data,
                    "order_id": res_data.get("orderId") or res_data.get("id") or f"MB-{int(time.time()*1000)}"
                }
            
            logger.error("Order rejected with HTTP %d: %s", resp.status_code, resp.text)
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text
            }
        except Exception as e:
            logger.error("Failed to place order on token %s: %s", instrument_token, str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def get_orders(self) -> List[Dict[str, Any]]:
        """Queries GET /api/order/my."""
        url = f"{self.base_url}/api/order/my"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("orders", [])
        except Exception as e:
            logger.error("Error fetching open orders: %s", str(e))
        return []

    def get_positions(self) -> List[Dict[str, Any]]:
        """Queries GET /api/position/my."""
        url = f"{self.base_url}/api/position/my"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("positions", [])
        except Exception as e:
            logger.error("Error fetching open positions: %s", str(e))
        return []

    def get_account_funds(self) -> Optional[float]:
        """
        Queries MegaBull API to fetch live account balance / available cash.
        Attempts standard MegaBull endpoints: /api/user/profile, /api/user/funds, /api/funds/my.
        Returns float available funds or None if query fails or is unsupported.
        """
        endpoints = [
            "/api/user/profile",
            "/api/user/funds",
            "/api/funds/my",
            "/api/user/my",
            "/api/account/balance"
        ]
        for ep in endpoints:
            try:
                url = f"{self.base_url}{ep}"
                resp = self.session.get(url, headers=self.headers, timeout=min(self.timeout, 2.0))
                if resp.status_code == 200:
                    data = resp.json()
                    # Check direct keys
                    for key in ["availableCash", "cash", "margin", "balance", "net", "fund", "availableMargin", "wallet"]:
                        if key in data and data[key] is not None:
                            val = float(data[key])
                            if val > 0:
                                logger.info("[MEGABULL] Retrieved account funds from %s: Rs %.2f", ep, val)
                                return val
                    # Check nested under 'data'
                    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                        d = data["data"]
                        for key in ["availableCash", "cash", "margin", "balance", "net", "fund"]:
                            if key in d and d[key] is not None:
                                val = float(d[key])
                                if val > 0:
                                    logger.info("[MEGABULL] Retrieved account funds from %s: Rs %.2f", ep, val)
                                    return val
            except Exception:
                continue
        return None

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Dispatches order cancellation to MegaBull API.
        Attempts POST /api/order/cancel/{orderId} or DELETE /api/order/{orderId}.
        """
        url = f"{self.base_url}/api/order/cancel/{order_id}"
        try:
            resp = self.session.post(url, headers=self.headers, json={"orderId": order_id}, timeout=self.timeout)
            if resp.status_code in (200, 204):
                return {"success": True, "message": f"Order {order_id} cancelled successfully."}
            # Fallback DELETE
            del_url = f"{self.base_url}/api/order/{order_id}"
            del_resp = self.session.delete(del_url, headers=self.headers, timeout=self.timeout)
            if del_resp.status_code in (200, 204):
                return {"success": True, "message": f"Order {order_id} cancelled via DELETE."}
        except Exception as e:
            logger.error("Failed to cancel order %s at MegaBull: %s", order_id, str(e))
        return {"success": False, "message": f"Broker order cancellation not acknowledged for {order_id}."}

    def close(self) -> None:
        """Closes the underlying HTTP session connection pool."""
        self.session.close()
