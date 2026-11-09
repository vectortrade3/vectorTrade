"""
VectorTrade - Online Database Manager
Provides robust MongoDB Atlas connection pooling, schema collections,
and health monitoring with DNS optimization and fallback resiliency.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# Optimize DNS resolution for MongoDB Atlas SRV on machines with virtual adapters (WSL/Docker/VPN)
try:
    import dns.resolver
    _default_resolver = dns.resolver.get_default_resolver()
    if _default_resolver and hasattr(_default_resolver, "nameservers"):
        public_dns = ["8.8.8.8", "1.1.1.1"]
        _default_resolver.nameservers = [ns for ns in public_dns if ns not in _default_resolver.nameservers] + _default_resolver.nameservers
        _default_resolver.lifetime = 6.0
except Exception:
    pass

import pymongo
from pymongo import MongoClient
from online.config import settings

logger = logging.getLogger("VectorTrade.OnlineDB")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class DatabaseManager:
    """Singleton MongoDB Atlas connection manager for online services."""

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or settings.mongodb_uri
        self.db_name = db_name or settings.mongodb_database
        self._client: Optional[MongoClient] = None
        self._db = None
        self._mock_mode = False

    @property
    def client(self) -> Optional[MongoClient]:
        """Lazy-loaded MongoClient with connection pooling and timeouts."""
        if self._mock_mode:
            return None

        if self._client is None:
            if not self.uri:
                logger.warning("No MONGODB_URI configured for DatabaseManager.")
                return None
            try:
                self._client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=settings.mongo_timeout_ms,
                    connectTimeoutMS=settings.mongo_timeout_ms,
                    socketTimeoutMS=settings.mongo_timeout_ms,
                    maxPoolSize=20,
                    minPoolSize=2
                )
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB Atlas: {e}")
                self._client = None
        return self._client

    @property
    def db(self):
        """Returns the database instance."""
        if self._mock_mode:
            return self._db
        if self._db is None:
            client = self.client
            if client is not None:
                self._db = client[self.db_name]
        return self._db

    def set_mock_db(self, mock_db_instance):
        """Allows injecting mongomock or in-memory dict for isolated testing."""
        self._db = mock_db_instance
        self._mock_mode = True

    def reset_client(self):
        """Closes client and resets connection state."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._db = None
        self._mock_mode = False

    def ping(self) -> Dict[str, Any]:
        """Pings the database and returns health metadata."""
        if self._mock_mode:
            return {
                "status": "ok",
                "mode": "mock",
                "database": self.db_name,
                "latency_ms": 0.0
            }

        start = time.time()
        client = self.client
        if client is None:
            return {
                "status": "error",
                "message": "MongoClient not initialized (check MONGODB_URI)",
                "latency_ms": -1
            }

        try:
            client.admin.command("ping")
            latency = round((time.time() - start) * 1000, 2)
            db = self.db
            cols = db.list_collection_names() if db is not None else []
            signals_count = db.signals_watchlist.count_documents({}) if db is not None and "signals_watchlist" in cols else 0

            return {
                "status": "ok",
                "mode": "live_atlas",
                "database": self.db_name,
                "latency_ms": latency,
                "collections_count": len(cols),
                "signals_count": signals_count
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "latency_ms": round((time.time() - start) * 1000, 2)
            }

    def get_collection(self, name: str):
        """Returns collection helper."""
        database = self.db
        if database is None:
            return None
        return database[name]

    @property
    def signals_watchlist(self):
        return self.get_collection("signals_watchlist")

    @property
    def active_trades(self):
        return self.get_collection("active_trades")

    @property
    def trade_history(self):
        return self.get_collection("trade_history")

    @property
    def staged_orders(self):
        return self.get_collection("staged_orders")

    @property
    def system_state(self):
        return self.get_collection("system_state")

    @property
    def system_audit(self):
        return self.get_collection("system_audit")


# Export singleton instance
db_manager = DatabaseManager()
