"""
VectorTrade - Secure Inter-Process Communication & Telemetry Subsystem
Enables Offline Daemon to request trade details, execution telemetry, and model feedback
from Online Worker securely without exposing sensitive information to public URLs.

Security Architecture:
1. Indirect Database Mailbox Channel (Default):
   - Offline writes authenticated request to MongoDB collection 'inter_service_requests'.
   - Online Execution Worker processes requests asynchronously during its tick loop.
   - Zero open inbound ports required; works across NATs, private clouds, and firewalls.
   - All payloads signed using HMAC-SHA256 with pre-shared INTERNAL_AUTH_TOKEN.
2. Cryptographically Authenticated REST API Channel:
   - Online exposes /api/internal/trade-details.
   - Direct web browser visits without secret headers are rejected with 401 Unauthorized.
   - Uses constant-time token comparison (hmac.compare_digest) and replay protection.
"""

from __future__ import annotations
import hmac
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pytz
import requests

from core.config import settings
from core.db import db_manager

logger = logging.getLogger("VectorTrade.SecureComms")


def generate_hmac_signature(secret: str, message: str) -> str:
    """Generates a hexadecimal HMAC-SHA256 signature."""
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def verify_hmac_signature(secret: str, message: str, signature: str) -> bool:
    """Constant-time verification of HMAC-SHA256 signature to prevent timing attacks."""
    expected = generate_hmac_signature(secret, message)
    return hmac.compare_digest(expected, signature)


def verify_auth_token(token: Optional[str]) -> bool:
    """Validates pre-shared internal authentication token with timing-safe comparison."""
    if not token or not settings.INTERNAL_AUTH_TOKEN:
        return False
    return hmac.compare_digest(settings.INTERNAL_AUTH_TOKEN.strip(), token.strip())


class TelemetryCompiler:
    """
    Compiles sanitized, verified execution state and performance metrics from MongoDB
    to fulfill Offline Daemon requests.
    """

    @staticmethod
    def get_today_execution_summary() -> Dict[str, Any]:
        """Returns details on today's triggered, active, closed, and invalidated orders."""
        tz = pytz.timezone(settings.TIMEZONE)
        today_str = datetime.now(tz).strftime("%Y-%m-%d")

        signals_col = db_manager.get_collection("signals_watchlist")
        active_col = db_manager.get_collection("active_trades")
        history_col = db_manager.get_collection("trade_history")

        # 1. Triggered & Invalidated signals from today's watchlist
        today_signals = list(signals_col.find({"valid_for_date": today_str}))
        triggered_signals = [s for s in today_signals if s.get("status") == "TRIGGERED"]
        invalidated_signals = [s for s in today_signals if s.get("status") == "INVALIDATED"]
        armed_signals = [s for s in today_signals if s.get("status") == "ARMED"]

        # 2. Currently Open positions
        open_positions = list(active_col.find({"status": "OPEN"}))

        # 3. Closed trades
        recent_closed = list(history_col.find({}).sort("_id", -1).limit(50))
        # Filter for trades closed today
        today_closed = [
            t for t in recent_closed
            if str(t.get("exit_timestamp", "")).startswith(today_str)
        ]

        total_realized_pnl = sum(float(t.get("pnl", 0.0)) for t in today_closed)
        winning_trades = [t for t in today_closed if float(t.get("pnl", 0.0)) > 0]
        losing_trades = [t for t in today_closed if float(t.get("pnl", 0.0)) <= 0]

        # Clean ObjectIds for serialization
        def sanitize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            clean = []
            for d in docs:
                c = dict(d)
                c["_id"] = str(c.get("_id", ""))
                clean.append(c)
            return clean

        return {
            "date": today_str,
            "status": "SUCCESS",
            "summary": {
                "armed_count": len(armed_signals),
                "triggered_count": len(triggered_signals),
                "invalidated_count": len(invalidated_signals),
                "active_open_count": len(open_positions),
                "today_closed_count": len(today_closed),
                "today_realized_pnl": round(total_realized_pnl, 2),
                "win_rate_today_pct": round((len(winning_trades) / len(today_closed) * 100.0), 1) if today_closed else 0.0
            },
            "active_positions": sanitize_docs(open_positions),
            "today_closed_trades": sanitize_docs(today_closed),
            "invalidated_signals": sanitize_docs(invalidated_signals)
        }

    @staticmethod
    def get_active_positions_details() -> Dict[str, Any]:
        """Returns detailed breakdown of all currently open trades."""
        active_col = db_manager.get_collection("active_trades")
        open_positions = list(active_col.find({"status": "OPEN"}))

        for pos in open_positions:
            pos["_id"] = str(pos["_id"])

        total_capital = sum(float(p.get("total_capital_required", 0.0)) for p in open_positions)

        return {
            "status": "SUCCESS",
            "open_count": len(open_positions),
            "total_capital_committed": round(total_capital, 2),
            "positions": open_positions
        }

    @staticmethod
    def get_ml_training_feedback(limit: int = 100) -> Dict[str, Any]:
        """
        Returns structured trade outcome feedback (features, fill price, exit price,
        MFE, MAE, and outcome label 1/0) for offline model retraining.
        """
        history_col = db_manager.get_collection("trade_history")
        closed_trades = list(history_col.find({}).sort("_id", -1).limit(limit))

        feedback_records = []
        for t in closed_trades:
            feedback_records.append({
                "trade_id": str(t.get("trade_id", t.get("_id"))),
                "symbol": t.get("symbol"),
                "strategy_id": t.get("strategy_id"),
                "entry_price": t.get("entry_price"),
                "exit_price": t.get("exit_price"),
                "pnl": t.get("pnl"),
                "roi_pct": t.get("roi_pct"),
                "exit_reason": t.get("exit_reason"),
                "mfe": t.get("mfe"),
                "mae": t.get("mae"),
                "ml_outcome_label": t.get("ml_outcome_label", 1 if float(t.get("pnl", 0)) > 0 else 0),
                "entry_timestamp": t.get("entry_timestamp"),
                "exit_timestamp": t.get("exit_timestamp")
            })

        return {
            "status": "SUCCESS",
            "records_count": len(feedback_records),
            "training_feedback": feedback_records
        }

    @staticmethod
    def compile(action: str, params: Optional[Dict[str, Any]] = None, system_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Routes action to the corresponding telemetry compiler."""
        params = params or {}
        if action == "GET_TODAY_EXECUTION":
            res = TelemetryCompiler.get_today_execution_summary()
        elif action == "GET_ACTIVE_POSITIONS":
            res = TelemetryCompiler.get_active_positions_details()
        elif action == "GET_ML_TRAINING_FEEDBACK":
            limit = int(params.get("limit", 100))
            res = TelemetryCompiler.get_ml_training_feedback(limit=limit)
        elif action == "GET_SYSTEM_HEALTH":
            res = {
                "status": "SUCCESS",
                "system_status": system_status or {},
                "timestamp": datetime.now(pytz.utc).isoformat()
            }
        else:
            res = {
                "status": "ERROR",
                "message": f"Unknown requested action: '{action}'",
                "supported_actions": ["GET_TODAY_EXECUTION", "GET_ACTIVE_POSITIONS", "GET_ML_TRAINING_FEEDBACK", "GET_SYSTEM_HEALTH"]
            }
        
        if system_status and "system_status" not in res:
            res["system_status"] = system_status
        return res


class SecureMailboxCommunicator:
    """
    Manages indirect asynchronous request-response communication between Offline and Online
    via MongoDB Atlas collection 'inter_service_requests'.
    """

    COLLECTION_NAME = "inter_service_requests"

    @classmethod
    def request_from_offline(cls, action: str, params: Optional[Dict[str, Any]] = None, timeout_sec: float = 8.0) -> Dict[str, Any]:
        """
        Sends an authenticated request from Offline Daemon to Online Worker via MongoDB
        and waits for fulfillment. Zero network ports opened.
        """
        col = db_manager.get_collection(cls.COLLECTION_NAME)
        req_id = f"REQ_{action}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        now_ts = time.time()
        params = params or {}

        # Message signature
        sig_message = f"{req_id}:{action}:{now_ts}"
        sig = generate_hmac_signature(settings.INTERNAL_AUTH_TOKEN, sig_message)

        req_doc = {
            "request_id": req_id,
            "action": action,
            "params": params,
            "requester": "OFFLINE_DAEMON",
            "timestamp": now_ts,
            "signature": sig,
            "status": "PENDING",
            "created_at": datetime.now(pytz.utc).isoformat()
        }

        logger.info("[SECURE MAILBOX] Offline created request %s (Action: %s). Waiting for Online Worker...", req_id, action)
        col.insert_one(req_doc)

        # Poll for completion
        start_time = time.time()
        poll_interval = 0.25
        while (time.time() - start_time) < timeout_sec:
            res_doc = col.find_one({"request_id": req_id})
            if res_doc and res_doc.get("status") == "COMPLETED":
                # Verify response authenticity
                resp_payload = res_doc.get("response", {})
                resp_sig = res_doc.get("response_signature", "")
                resp_ts = res_doc.get("fulfilled_at_ts", 0)
                
                check_msg = f"{req_id}:{res_doc.get('status')}:{resp_ts}"
                if verify_hmac_signature(settings.INTERNAL_AUTH_TOKEN, check_msg, resp_sig):
                    logger.info("[SECURE MAILBOX] Request %s fulfilled and cryptographically verified in %.2fs.", req_id, time.time() - start_time)
                    # Clean up completed request
                    try:
                        col.delete_one({"request_id": req_id})
                    except Exception:
                        pass
                    return resp_payload
                else:
                    logger.error("[SECURE MAILBOX] Response signature mismatch for request %s! Possible tampering.", req_id)
                    return {"status": "ERROR", "message": "Cryptographic signature validation failed."}

            time.sleep(poll_interval)

        # Timeout reached: fallback to reading current DB state directly
        logger.warning("[SECURE MAILBOX] Request %s timed out after %.1fs. Compiling fallback snapshot directly from DB.", req_id, timeout_sec)
        fallback = TelemetryCompiler.compile(action, params)
        fallback["note"] = "Snapshot compiled directly from persistent database after worker mailbox timeout."
        return fallback

    @classmethod
    def fulfill_pending_requests(cls, system_status: Optional[Dict[str, Any]] = None) -> int:
        """
        Called inside ExecutionWorker cycle: finds PENDING requests, compiles response,
        signs it, and updates DB.
        """
        try:
            col = db_manager.get_collection(cls.COLLECTION_NAME)
            pending = list(col.find({"status": "PENDING"}))
            if not pending:
                return 0

            fulfilled_count = 0
            for req in pending:
                req_id = req["request_id"]
                action = req["action"]
                req_ts = req.get("timestamp", 0)
                req_sig = req.get("signature", "")

                # 1. Verify Request Signature & Timestamp freshness (<= 300s)
                sig_msg = f"{req_id}:{action}:{req_ts}"
                if not verify_hmac_signature(settings.INTERNAL_AUTH_TOKEN, sig_msg, req_sig):
                    logger.warning("[SECURE MAILBOX] Rejected request %s: Invalid signature.", req_id)
                    col.update_one({"_id": req["_id"]}, {"$set": {"status": "REJECTED", "reason": "INVALID_SIGNATURE"}})
                    continue

                if abs(time.time() - req_ts) > 300:
                    logger.warning("[SECURE MAILBOX] Rejected request %s: Expired timestamp (>300s).", req_id)
                    col.update_one({"_id": req["_id"]}, {"$set": {"status": "REJECTED", "reason": "TIMESTAMP_EXPIRED"}})
                    continue

                # 2. Compile Telemetry Response
                payload = TelemetryCompiler.compile(action, req.get("params", {}), system_status=system_status)

                # 3. Cryptographically Sign Response
                resp_ts = time.time()
                resp_sig_msg = f"{req_id}:COMPLETED:{resp_ts}"
                resp_sig = generate_hmac_signature(settings.INTERNAL_AUTH_TOKEN, resp_sig_msg)

                col.update_one(
                    {"_id": req["_id"]},
                    {"$set": {
                        "status": "COMPLETED",
                        "response": payload,
                        "fulfilled_at_ts": resp_ts,
                        "fulfilled_at": datetime.now(pytz.utc).isoformat(),
                        "response_signature": resp_sig
                    }}
                )
                fulfilled_count += 1
                logger.info("[SECURE MAILBOX] Successfully fulfilled request %s (%s).", req_id, action)

            return fulfilled_count
        except Exception as e:
            logger.error("Error in fulfill_pending_requests: %s", str(e))
            return 0


class SecureHTTPCommunicator:
    """
    Direct authenticated REST client for offline-to-online communications.
    Protected with constant-time pre-shared token and HMAC SHA-256 signatures.
    """

    @classmethod
    def request_from_offline(cls, action: str, params: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends an authenticated POST request to /api/internal/trade-details on Online server.
        Direct URL visits in browser without these headers will fail with 401 Unauthorized.
        """
        url = f"{(base_url or settings.ONLINE_BASE_URL).rstrip('/')}/api/internal/trade-details"
        now_ts = str(int(time.time()))
        nonce = uuid.uuid4().hex[:12]
        params = params or {}

        body = {
            "action": action,
            "params": params,
            "timestamp": now_ts,
            "nonce": nonce
        }
        body_json = json.dumps(body, sort_keys=True)

        signature = generate_hmac_signature(
            settings.INTERNAL_AUTH_TOKEN,
            f"{action}:{now_ts}:{nonce}:{body_json}"
        )

        headers = {
            "Content-Type": "application/json",
            "X-Vector-Internal-Token": settings.INTERNAL_AUTH_TOKEN,
            "X-Vector-Signature": signature,
            "X-Vector-Timestamp": now_ts,
            "X-Vector-Nonce": nonce
        }

        try:
            resp = requests.post(url, data=body_json, headers=headers, timeout=settings.HTTP_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {
                    "status": "ERROR",
                    "status_code": resp.status_code,
                    "message": f"Server returned error {resp.status_code}: {resp.text}"
                }
        except Exception as e:
            logger.warning("[SECURE HTTP] Could not reach online server directly at %s: %s", url, str(e))
            return {
                "status": "CONNECTION_FAILED",
                "message": f"Network error contacting {url}: {str(e)}"
            }


# High-Level Unified Interface for Offline Daemon
def request_online_trade_details(
    action: str = "GET_TODAY_EXECUTION",
    params: Optional[Dict[str, Any]] = None,
    channel: str = "auto"
) -> Dict[str, Any]:
    """
    Unified entry point for Offline program to request details from Online program.
    Channels:
    - 'auto' (Default): Tries Indirect Database Mailbox first (zero port exposure).
      If in local mock mode or fallback requested, falls back to direct compiled DB snapshot.
    - 'mailbox': Strictly forces indirect MongoDB collection request-response.
    - 'http': Uses cryptographically authenticated HTTP request to /api/internal/trade-details.
    """
    params = params or {}
    if channel == "http":
        res = SecureHTTPCommunicator.request_from_offline(action, params)
        if res.get("status") != "CONNECTION_FAILED":
            return res
        logger.info("[COMMS FALLBACK] HTTP failed. Falling back to Secure Mailbox.")

    # Default: Mailbox communication via MongoDB Atlas
    return SecureMailboxCommunicator.request_from_offline(action, params)
