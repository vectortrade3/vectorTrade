"""
VectorTrade - Production MongoDB Connection Pool & Resilience Engine
Provides automatic reconnection retry logic with exponential backoff for intermittent network drops,
compound and unique index initialization, and seamless mongomock fallback.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional, Union
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, AutoReconnect

from core.config import settings

logger = logging.getLogger("VectorTrade.DB")


class DatabaseManager:
    """
    Singleton connection pool and schema manager for MongoDB Atlas / mongomock.
    Includes automated reconnection retry logic with exponential backoff.
    """
    _instance: Optional[DatabaseManager] = None
    _client: Optional[Union[MongoClient, Any]] = None
    _db: Optional[Union[Database, Any]] = None
    _is_mock: bool = False

    def __new__(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance

    def _initialize_connection(self) -> None:
        """
        Initializes MongoDB connection pool or in-memory mock with retry handling.
        """
        if settings.is_mongo_mock:
            try:
                import mongomock
                logger.warning("MongoDB URI not provided or set to 'mock'. Initializing in-memory mongomock database.")
                self._client = mongomock.MongoClient()
                self._db = self._client[settings.MONGODB_DB_NAME]
                self._is_mock = True
            except ImportError:
                raise RuntimeError("mongomock package is required for offline mock execution. Run: pip install mongomock")
        else:
            logger.info("Connecting to MongoDB Atlas cluster at: %s", settings.MONGODB_URI.split("@")[-1])
            self._client = self._connect_with_retry()
            self._db = self._client[settings.MONGODB_DB_NAME]
            self._is_mock = False

        self._ensure_indexes()

    def _connect_with_retry(self, max_attempts: int = 4) -> MongoClient:
        """Connects to MongoDB Atlas with exponential backoff on transient errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                client = MongoClient(
                    settings.MONGODB_URI,
                    maxPoolSize=50,
                    minPoolSize=5,
                    serverSelectionTimeoutMS=int(settings.HTTP_TIMEOUT_SECONDS * 1000),
                    connectTimeoutMS=int(settings.HTTP_TIMEOUT_SECONDS * 1000),
                    socketTimeoutMS=int(settings.HTTP_TIMEOUT_SECONDS * 1000),
                    retryWrites=True,
                    w="majority"
                )
                client.admin.command("ping")
                logger.info("Successfully established connection to MongoDB Atlas.")
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError, AutoReconnect) as e:
                backoff = 2 ** attempt
                logger.warning("MongoDB connection attempt %d/%d failed: %s. Retrying in %ds...", attempt, max_attempts, str(e), backoff)
                if attempt < max_attempts:
                    time.sleep(backoff)
                else:
                    logger.error("Failed to connect to MongoDB Atlas after %d attempts.", max_attempts)
                    raise ConnectionError(f"Could not connect to MongoDB Atlas: {str(e)}") from e

    def _ensure_indexes(self) -> None:
        """
        Creates mandatory compound and unique indexes across all required collections.
        """
        try:
            signals = self.get_collection("signals_watchlist")
            signals.create_index([("symbol", ASCENDING), ("valid_for_date", ASCENDING)], name="idx_symbol_date")
            signals.create_index([("status", ASCENDING), ("valid_for_date", ASCENDING)], name="idx_status_date")
            signals.create_index([("instrument_token", ASCENDING)], name="idx_token")

            active = self.get_collection("active_trades")
            active.create_index([("symbol", ASCENDING)], name="idx_active_symbol")
            active.create_index([("status", ASCENDING)], name="idx_active_status")
            active.create_index([("instrument_token", ASCENDING)], name="idx_active_token")

            history = self.get_collection("trade_history")
            history.create_index([("symbol", ASCENDING), ("exit_date", DESCENDING)], name="idx_hist_symbol_exit")
            history.create_index([("strategy_id", ASCENDING), ("exit_date", DESCENDING)], name="idx_hist_strat_date")

            bars = self.get_collection("buffer_daily_bars")
            bars.create_index([("symbol", ASCENDING), ("date", ASCENDING)], unique=True, name="idx_bars_unique_sym_date")
            bars.create_index([("consumed_for_training", ASCENDING)], name="idx_bars_consumed")

            inst = self.get_collection("instruments_cache")
            inst.create_index([("symbol", ASCENDING)], unique=True, name="idx_inst_symbol_unique")
            inst.create_index([("instrument_token", ASCENDING)], name="idx_inst_token")

            audit = self.get_collection("system_audit")
            audit.create_index([("component", ASCENDING), ("execution_date", DESCENDING)], name="idx_audit_comp_date")
            audit.create_index([("timestamp", DESCENDING)], name="idx_audit_timestamp")

            comms = self.get_collection("inter_service_requests")
            comms.create_index([("status", ASCENDING), ("created_at", ASCENDING)], name="idx_comms_status_created")
            comms.create_index([("request_id", ASCENDING)], unique=True, name="idx_comms_request_id")

            logger.info("Successfully ensured compound indexes across all collections.")
        except Exception as e:
            logger.warning("Index creation notice: %s", str(e))

    @property
    def db(self) -> Database:
        if self._db is None:
            self._initialize_connection()
        return self._db

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def get_collection(self, name: str) -> Collection:
        """Retrieve collection with reconnection recovery."""
        try:
            return self.db[name]
        except (AutoReconnect, ConnectionFailure):
            logger.warning("Lost MongoDB connection. Re-initializing pool...")
            self._initialize_connection()
            return self.db[name]

    def ping(self) -> Dict[str, Any]:
        """Verifies connectivity with health check."""
        if self._is_mock:
            return {"status": "ok", "mode": "mock_in_memory", "database": settings.MONGODB_DB_NAME}
        try:
            self._client.admin.command("ping")
            return {"status": "ok", "mode": "atlas_cluster", "database": settings.MONGODB_DB_NAME}
        except Exception as e:
            logger.error("MongoDB ping failed: %s", str(e))
            raise ConnectionError(f"Failed to reach MongoDB server: {str(e)}") from e

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB client pool closed.")


# Global instance
db_manager = DatabaseManager()
get_collection = db_manager.get_collection
