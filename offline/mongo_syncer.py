"""
VectorTrade - Offline to MongoDB Atlas Synchronization Engine
Syncs daily scanned setups, entry triggers, stop-losses, and profit targets
to MongoDB Atlas (`signals_watchlist` collection) for the online application.
"""

import os
import sys
import logging
import pytz
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Optimize DNS resolution for MongoDB Atlas SRV on machines with virtual adapters (WSL/Docker/VPN)
try:
    import dns.resolver
    _default_resolver = dns.resolver.get_default_resolver()
    if _default_resolver and hasattr(_default_resolver, "nameservers"):
        # Prioritize Google/Cloudflare public DNS while keeping existing nameservers as fallback
        public_dns = ["8.8.8.8", "1.1.1.1"]
        _default_resolver.nameservers = [ns for ns in public_dns if ns not in _default_resolver.nameservers] + _default_resolver.nameservers
        _default_resolver.lifetime = 6.0
except Exception:
    pass

import pymongo
from pymongo import UpdateOne
from dotenv import load_dotenv

logger = logging.getLogger("VectorTrade.MongoSyncer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class MongoSyncer:
    """
    Manages robust synchronization between local offline analysis results
    and the MongoDB Atlas database used by the online application.
    """

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None, timeout_ms: int = 4000):
        self.uri = uri or os.getenv("MONGODB_URI")
        self.db_name = db_name or os.getenv("MONGODB_DATABASE", "vectortrade")
        self.timeout_ms = timeout_ms
        self._client: Optional[pymongo.MongoClient] = None
        self._db = None

    def get_client(self) -> Optional[pymongo.MongoClient]:
        """Returns or creates a MongoClient with short timeout for responsiveness."""
        if not self.uri:
            logger.warning("MONGODB_URI not set in environment or .env file.")
            return None

        if self._client is None:
            try:
                self._client = pymongo.MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=self.timeout_ms,
                    connectTimeoutMS=self.timeout_ms,
                    socketTimeoutMS=self.timeout_ms
                )
            except Exception as e:
                logger.error(f"Failed to create MongoClient: {e}")
                self._client = None
        return self._client

    def get_db(self):
        """Returns the database instance or None if connection fails."""
        client = self.get_client()
        if client is None:
            return None
        return client[self.db_name]

    def test_connection(self) -> Dict[str, Any]:
        """Tests MongoDB connectivity and returns status."""
        try:
            client = self.get_client()
            if not client:
                return {"connected": False, "error": "MONGODB_URI not configured"}
            # Send a ping
            client.admin.command("ping")
            db = self.get_db()
            cols = db.list_collection_names()
            return {
                "connected": True,
                "database": self.db_name,
                "collections": cols,
                "signals_count": db.signals_watchlist.count_documents({}) if "signals_watchlist" in cols else 0
            }
        except Exception as e:
            logger.warning(f"MongoDB connection test failed: {e}")
            return {"connected": False, "error": str(e)}

    @staticmethod
    def format_trade_to_signal(trade: Dict[str, Any], default_risk_budget: float = 10000.0) -> Dict[str, Any]:
        """
        Normalizes a trade dictionary from offline scanner into standard
        MongoDB signal document compatible with both legacy and new online specs.
        """
        symbol = str(trade.get("symbol", "")).strip().upper()
        name = str(trade.get("name", symbol)).strip()
        sector = str(trade.get("sector", "")).strip()
        date = str(trade.get("date", datetime.now().strftime("%Y-%m-%d")))
        strategy = str(trade.get("strategy", "50 MA Setup")).strip()
        setup_subclass = str(trade.get("setup_subclass", "")).strip()
        setup_num = int(trade.get("setup_num", 1))
        action = str(trade.get("action", "STRONG BUY")).strip()

        entry_price = round(float(trade.get("entry_price", trade.get("buy_trigger", 0.0))), 2)
        sl_price = round(float(trade.get("sl_price", 0.0)), 2)
        target_price = round(float(trade.get("target_price", 0.0)), 2)

        risk_per_share = max(round(entry_price - sl_price, 2), 0.01)
        quantity = int(trade.get("quantity", max(1, int(default_risk_budget / risk_per_share))))
        total_capital = round(entry_price * quantity, 2)
        loss_at_sl = round(risk_per_share * quantity, 2)
        profit_at_target = round((target_price - entry_price) * quantity, 2)

        loss_pct = round(float(trade.get("loss_pct", (risk_per_share / entry_price * 100) if entry_price > 0 else 0.0)), 2)
        profit_pct = round(float(trade.get("profit_pct", ((target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0)), 2)
        risk_reward = str(trade.get("risk_reward", "1 : 2.00"))
        risk_reward_num = round(float(trade.get("risk_reward_num", 2.0)), 2)

        predicted_r = round(float(trade.get("predicted_r", 0.0)), 2)
        ml_confidence = round(float(trade.get("ml_confidence", min(max(predicted_r / 2.0, 0.0), 1.0))), 2)

        outcome = str(trade.get("outcome", trade.get("status", "ARMED"))).strip()
        status = outcome  # For backward-compatibility with online runner
        outcome_detail = str(trade.get("outcome_detail", "")).strip()
        rationale = str(trade.get("rationale", "")).strip()

        kolkata_tz = pytz.timezone("Asia/Kolkata")
        synced_at = datetime.now(kolkata_tz).isoformat()

        strat_slug = strategy.replace(" ", "_").replace("-", "_").upper()
        doc_id = f"SIG_{strat_slug}_{symbol}_{date}"

        return {
            "_id": doc_id,
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "strategy": strategy,
            "strategy_id": strat_slug,
            "setup_subclass": setup_subclass,
            "setup_num": setup_num,
            "action": action,
            "buy_trigger": entry_price,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "target_price": target_price,
            "risk_per_share": risk_per_share,
            "quantity": quantity,
            "total_capital_required": total_capital,
            "loss_at_sl": loss_at_sl,
            "profit_at_target": profit_at_target,
            "loss_pct": loss_pct,
            "profit_pct": profit_pct,
            "risk_reward": risk_reward,
            "risk_reward_num": risk_reward_num,
            "predicted_r": predicted_r,
            "ml_confidence": ml_confidence,
            "outcome": outcome,
            "status": status,
            "outcome_detail": outcome_detail,
            "rationale": rationale,
            "rsi": float(trade.get("rsi", 0.0)),
            "atr": float(trade.get("atr", 0.0)),
            "ma50": float(trade.get("ma50", 0.0)),
            "dist_low_bb": float(trade.get("dist_low_bb", 0.0)),
            "dist_low_ma50": float(trade.get("dist_low_ma50", 0.0)),
            "is_latest": bool(trade.get("is_latest", True)),
            "valid_for_date": date,
            "synced_at": synced_at
        }

    def sync_trades(self, trades: List[Dict[str, Any]], session_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Upserts the list of finalized trades into MongoDB Atlas signals_watchlist collection.
        Returns a summary report of operations.
        """
        if not trades:
            logger.info("No trades to sync to MongoDB.")
            return {"status": "SKIPPED", "synced_count": 0, "message": "No trades provided"}

        db = self.get_db()
        if db is None:
            msg = "Could not connect to MongoDB Atlas. Sync skipped (offline mode)."
            logger.warning(msg)
            return {"status": "OFFLINE_FALLBACK", "synced_count": 0, "message": msg}

        try:
            signals_col = db["signals_watchlist"]
            
            # Ensure indexes exist
            signals_col.create_index([("valid_for_date", pymongo.DESCENDING)])
            signals_col.create_index([("status", pymongo.ASCENDING)])
            signals_col.create_index([("symbol", pymongo.ASCENDING)])
            signals_col.create_index([("strategy", pymongo.ASCENDING)])

            operations = []
            armed_count = 0
            for trade in trades:
                doc = self.format_trade_to_signal(trade)
                if doc["status"] == "ARMED":
                    armed_count += 1
                operations.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

            if operations:
                result = signals_col.bulk_write(operations, ordered=False)
                upserted = len(result.upserted_ids) if result.upserted_ids else 0
                modified = result.modified_count
                total_processed = len(operations)

                # Record sync telemetry in system_audit collection
                try:
                    audit_col = db["system_audit"]
                    audit_col.insert_one({
                        "event": "OFFLINE_SIGNALS_SYNC",
                        "session_date": session_date or datetime.now().strftime("%Y-%m-%d"),
                        "total_synced": total_processed,
                        "armed_count": armed_count,
                        "upserted_count": upserted,
                        "modified_count": modified,
                        "timestamp": datetime.now(pytz.utc).isoformat()
                    })
                except Exception as audit_err:
                    logger.debug(f"Audit log write failed (non-critical): {audit_err}")

                msg = f"Successfully synced {total_processed} setups to MongoDB Atlas (Armed: {armed_count}, Upserted: {upserted}, Modified: {modified})."
                logger.info(msg)
                return {
                    "status": "SUCCESS",
                    "synced_count": total_processed,
                    "armed_count": armed_count,
                    "upserted_count": upserted,
                    "modified_count": modified,
                    "message": msg
                }
            return {"status": "SUCCESS", "synced_count": 0, "message": "Zero operations executed"}

        except Exception as e:
            err_msg = f"Error during MongoDB sync: {e}"
            logger.error(err_msg)
            return {"status": "ERROR", "synced_count": 0, "message": err_msg}

    def sync_from_json_file(self, json_path: Optional[Path] = None) -> Dict[str, Any]:
        """Reads trades from daily_analysis.json and syncs them to MongoDB."""
        import json
        if json_path is None:
            json_path = PROJECT_ROOT / "data_store" / "daily_analysis.json"

        if not json_path.exists():
            return {"status": "ERROR", "message": f"File not found: {json_path}", "synced_count": 0}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            trades = data.get("finalized_trades", [])
            session_date = data.get("market_session_date", "")
            return self.sync_trades(trades, session_date=session_date)
        except Exception as e:
            return {"status": "ERROR", "message": f"Failed reading {json_path}: {e}", "synced_count": 0}


if __name__ == "__main__":
    syncer = MongoSyncer()
    print("Testing MongoDB connection...")
    info = syncer.test_connection()
    print("Connection info:", info)
    print("Syncing existing daily_analysis.json...")
    res = syncer.sync_from_json_file()
    print("Sync result:", res)
