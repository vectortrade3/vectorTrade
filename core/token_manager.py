"""
VectorTrade - Instrument Token Ingestion & Mapping Pipeline
Fetches all tradable instrument definitions from MegaBull API, parses the CSV payload,
extracts NIFTY 200 constituents, and synchronizes the normalized symbol-to-token map
into MongoDB collection 'instruments_cache' and local cache 'tokens_cache.json'.
Features strict timeout (10s), exponential backoff retry logic, and zero credential leakage.
"""

from __future__ import annotations
import csv
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional
import requests

from core.config import settings
from core.db import db_manager

# Configure structured logging
logger = logging.getLogger("VectorTrade.TokenManager")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


class TokenManager:
    """
    Manages the lifecycle of MegaBull instrument tokens for NIFTY 200 stocks.
    """

    def __init__(self) -> None:
        self.api_key = settings.MEGABULL_API_KEY
        self.base_url = settings.MEGABULL_BASE_URL.rstrip("/")
        self.timeout = settings.HTTP_TIMEOUT_SECONDS
        self.max_retries = settings.HTTP_MAX_RETRIES

    def _get_headers(self) -> Dict[str, str]:
        """Constructs secure request headers with API key validation."""
        if not self.api_key:
            raise ValueError("MEGABULL_API_KEY is not set in environment or .env file.")
        headers = {
            "api-key": self.api_key,
            "User-Agent": f"{settings.APP_NAME}/1.0 (Automated Ingestion)",
            "Accept": "application/json"
        }
        if settings.MEGABULL_USER_ID:
            headers["user-id"] = settings.MEGABULL_USER_ID
        return headers

    def fetch_instruments_csv_url(self) -> str:
        """
        Calls GET /api/marketwatch/instruments with exponential backoff.
        Returns the AWS S3 CSV download URL.
        """
        endpoint = f"{self.base_url}/api/marketwatch/instruments"
        headers = self._get_headers()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Fetching instruments manifest from MegaBull (attempt %d/%d)...", attempt, self.max_retries)
                response = requests.get(endpoint, headers=headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    payload = response.json()
                    download_url = payload.get("downloadUrl")
                    if not download_url:
                        raise ValueError(f"Missing 'downloadUrl' in MegaBull response payload: {payload}")
                    logger.info("Retrieved CSV download URL successfully.")
                    return str(download_url)
                
                if response.status_code in (401, 403):
                    raise PermissionError(f"MegaBull API authentication failed (HTTP {response.status_code}). Check API key validity.")

                logger.warning("MegaBull API returned HTTP %d: %s", response.status_code, response.text[:150])
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning("Network error on attempt %d: %s", attempt, str(e))

            if attempt < self.max_retries:
                backoff_time = 2 ** attempt
                logger.info("Backing off for %d seconds before retrying...", backoff_time)
                time.sleep(backoff_time)

        raise RuntimeError(f"Failed to fetch instruments manifest from MegaBull after {self.max_retries} attempts.")

    def download_and_parse_csv(self, download_url: str) -> Dict[str, str]:
        """
        Downloads instrument definitions CSV and extracts tradingSymbol -> instrumentToken mapping.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Downloading instruments CSV payload (attempt %d/%d)...", attempt, self.max_retries)
                response = requests.get(
                    download_url,
                    headers={"User-Agent": f"{settings.APP_NAME}/1.0"},
                    timeout=self.timeout * 2  # Payload can be a few megabytes
                )
                if response.status_code == 200:
                    content = response.text
                    reader = csv.DictReader(io.StringIO(content))
                    symbol_map: Dict[str, str] = {}

                    for row in reader:
                        sym = row.get("tradingSymbol", "").strip().upper()
                        token = row.get("instrumentToken", "").strip()
                        if sym and token:
                            symbol_map[sym] = token

                    logger.info("Parsed %d total instrument definitions from MegaBull CSV.", len(symbol_map))
                    return symbol_map

                logger.warning("CSV download returned HTTP %d", response.status_code)
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning("CSV download error on attempt %d: %s", attempt, str(e))

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to download instruments CSV payload from {download_url}")

    def load_target_universe(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads NIFTY 200 constituent symbols from stock_names.json.
        """
        if not settings.STOCK_NAMES_PATH.exists():
            raise FileNotFoundError(f"Universe definitions missing at: {settings.STOCK_NAMES_PATH}")

        with open(settings.STOCK_NAMES_PATH, "r", encoding="utf-8") as f:
            universe: Dict[str, Dict[str, Any]] = json.load(f)

        logger.info("Loaded %d target constituent stocks from %s.", len(universe), settings.STOCK_NAMES_PATH.name)
        return universe

    def sync_tokens(self) -> Dict[str, Any]:
        """
        Executes full token synchronization:
        1. Fetch CSV URL from MegaBull API.
        2. Download & parse CSV into symbol map.
        3. Match against all 200 NIFTY constituent symbols (handling formatting variants).
        4. Upsert records into MongoDB collection 'instruments_cache'.
        5. Save local backup to 'tokens_cache.json'.
        """
        download_url = self.fetch_instruments_csv_url()
        all_instruments = self.download_and_parse_csv(download_url)
        universe = self.load_target_universe()

        records_to_upsert: List[Dict[str, Any]] = []
        tokens_cache: Dict[str, str] = {}
        missing_symbols: List[str] = []

        for symbol, metadata in universe.items():
            clean_sym = symbol.strip().upper()
            token: Optional[str] = None

            # Primary exact match
            if clean_sym in all_instruments:
                token = all_instruments[clean_sym]
            else:
                # Secondary normalized match (e.g. BAJAJ-AUTO -> BAJAJAUTO, M&M -> MM)
                sanitized = clean_sym.replace("&", "").replace("-", "")
                if sanitized in all_instruments:
                    token = all_instruments[sanitized]

            if token:
                tokens_cache[clean_sym] = token
                doc = {
                    "symbol": clean_sym,
                    "instrument_token": str(token),
                    "name": metadata.get("name", clean_sym),
                    "sector": metadata.get("sector", "Unknown"),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "active": True
                }
                records_to_upsert.append(doc)
            else:
                missing_symbols.append(clean_sym)

        if missing_symbols:
            logger.warning("Could not locate tokens for %d symbols: %s", len(missing_symbols), missing_symbols)

        # Upsert into MongoDB
        inst_collection = db_manager.get_collection("instruments_cache")
        upsert_count = 0
        for doc in records_to_upsert:
            inst_collection.update_one(
                {"symbol": doc["symbol"]},
                {"$set": doc},
                upsert=True
            )
            upsert_count += 1

        # Save local JSON cache
        with open(settings.TOKENS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(tokens_cache, f, indent=2)

        summary = {
            "status": "SUCCESS",
            "total_universe": len(universe),
            "matched_tokens": len(records_to_upsert),
            "missing_tokens": len(missing_symbols),
            "missing_symbols": missing_symbols,
            "mongodb_upserts": upsert_count,
            "cache_file": str(settings.TOKENS_CACHE_PATH)
        }
        logger.info(
            "Token Sync Complete: %d/%d constituents matched and persisted to MongoDB and tokens_cache.json.",
            len(records_to_upsert), len(universe)
        )
        return summary


def run_token_sync() -> Dict[str, Any]:
    """Helper entry point for CLI and automated pipelines."""
    manager = TokenManager()
    return manager.sync_tokens()


if __name__ == "__main__":
    result = run_token_sync()
    print("\n--- Token Synchronization Summary ---")
    print(json.dumps(result, indent=2))
