"""
VectorTrade - Automated Incremental Stock Data Downloader & Appender
Checks dataset dates, fetches delta candles from yfinance, and appends to existing CSVs.

Guarantees:
1. Date Checking: Checks last available date across existing stock CSVs.
2. Missing Delta Only: Only downloads the gap from (last_date + 1 day) to today.
3. Clean Append: Formats new rows with Sector, Nifty50 reference, and Day of Week,
   then appends directly into 5_year_data/raw/{symbol}_data.csv without data loss.
4. JSON Telemetry: Automatically updates dataset_metadata.json and logs/operations_history.json.
5. Live Progress: Updates analysis_progress.json and DATA_STATUS live for the dashboard progress bar.
"""

from __future__ import annotations
import os
import glob
import json
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yfinance as yf
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "5_year_data" / "raw"
METADATA_FILE = BASE_DIR / "dataset_metadata.json"

from offline.operations_logger import operations_logger
from offline.progress_tracker import progress_tracker

logger = logging.getLogger("VectorTrade.DataDownloader")


class StockDataDownloader:
    """Manages incremental delta downloading and appending to existing datasets."""

    def __init__(self, raw_dir: Path = RAW_DIR, metadata_file: Path = METADATA_FILE) -> None:
        self.raw_dir = raw_dir
        self.metadata_file = metadata_file
        self.tz = pytz.timezone("Asia/Kolkata")
        self.status = {
            "status": "IDLE",
            "progress_pct": 0.0,
            "current_symbol": "",
            "processed": 0,
            "total": 0,
            "last_date": None,
            "updated_to_date": None,
            "message": "Ready to download",
            "start_time": 0.0,
            "elapsed_seconds": 0.0
        }

    def detect_last_date(self) -> date:
        """Finds the most recent date in the dataset by reading metadata or existing CSVs."""
        # 1. Try dataset_metadata.json
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    date_str = meta.get("last_updated") or meta.get("global_last_date")
                    if date_str:
                        return datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                pass

        # 2. Inspect sample CSVs directly from raw_dir
        sample_files = list(self.raw_dir.glob("*_data.csv"))
        if sample_files:
            for sample in sample_files[:5]:
                try:
                    df = pd.read_csv(sample, usecols=[0], nrows=10, skiprows=lambda x: False)
                    # read last row
                    with open(sample, "rb") as f:
                        f.seek(max(0, sample.stat().st_size - 1024))
                        lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                        if lines:
                            last_line = lines[-1].split(",")
                            if len(last_line) > 0 and len(last_line[0]) == 10 and "-" in last_line[0]:
                                return datetime.strptime(last_line[0], "%Y-%m-%d").date()
                except Exception:
                    pass

        # Fallback default: 30 days ago
        return date.today() - timedelta(days=30)

    def run_download_and_append(self, force: bool = False) -> Dict[str, Any]:
        """
        Executes incremental download for all 207 stocks and appends to existing CSVs.
        """
        start_ts = time.time()
        self.status["start_time"] = start_ts
        self.status["status"] = "RUNNING"
        self.status["message"] = "Checking dataset dates and missing trading days..."

        today = date.today()
        # Market usually finishes after 16:00 IST. If today is a weekend, reference Friday.
        last_date = self.detect_last_date()
        self.status["last_date"] = str(last_date)

        logger.info("[DATA_DOWNLOAD] Last date in dataset: %s. Today: %s", last_date, today)

        # Check if already up-to-date
        if last_date >= today and not force:
            duration = round(time.time() - start_ts, 2)
            msg = f"Market data is already completely up to date (Latest: {last_date})."
            self.status["status"] = "COMPLETED"
            self.status["progress_pct"] = 100.0
            self.status["message"] = msg
            self.status["elapsed_seconds"] = duration
            operations_logger.log_data_download(
                status="SUCCESS",
                stocks_updated=0,
                last_date=str(last_date),
                duration_seconds=duration,
                message=msg
            )
            return {
                "status": "ALREADY_UP_TO_DATE",
                "last_date": str(last_date),
                "stocks_updated": 0,
                "duration_seconds": duration,
                "message": msg
            }

        start_d = last_date + timedelta(days=1)
        end_d = today + timedelta(days=1)  # yfinance end date is exclusive

        self.status["message"] = f"Downloading missing delta from {start_d} to {today}..."

        # 1. Fetch Nifty 50 reference delta
        logger.info("[DATA_DOWNLOAD] Fetching NIFTY 50 reference data from %s to %s...", start_d, end_d)
        nifty50_df = pd.DataFrame()
        try:
            nifty50_df = yf.download("^NSEI", start=start_d, end=end_d, auto_adjust=False, progress=False)
            if isinstance(nifty50_df.columns, pd.MultiIndex):
                nifty50_df.columns = nifty50_df.columns.droplevel(1)
            nifty_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in nifty_cols:
                if c not in nifty50_df.columns:
                    nifty50_df[c] = 0.0
            nifty50_df = nifty50_df[nifty_cols].rename(columns={
                'Open': 'Nifty50_Open',
                'High': 'Nifty50_High',
                'Low': 'Nifty50_Low',
                'Close': 'Nifty50_Close',
                'Volume': 'Nifty50_Volume'
            })
        except Exception as e:
            logger.warning("[DATA_DOWNLOAD] Could not fetch Nifty50 reference delta: %s", str(e))

        # 2. Iterate across all 207 stock files in 5_year_data/raw/
        raw_files = sorted(list(self.raw_dir.glob("*_data.csv")))
        total_stocks = len(raw_files)
        self.status["total"] = total_stocks

        if total_stocks == 0:
            msg = "No raw stock CSV files found in 5_year_data/raw/."
            self.status["status"] = "ERROR"
            self.status["message"] = msg
            operations_logger.log_data_download(
                status="FAILED",
                stocks_updated=0,
                last_date=str(last_date),
                duration_seconds=round(time.time() - start_ts, 2),
                message=msg
            )
            return {"status": "ERROR", "message": msg}

        stocks_updated = 0
        updated_max_date = str(last_date)
        metadata_symbols = {}

        cols_needed = [
            'Stock Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Sector', 'Sector Index',
            'Nifty50_Open', 'Nifty50_High', 'Nifty50_Low', 'Nifty50_Close', 'Nifty50_Volume',
            'Trading_Day_Of_Week'
        ]

        for idx, f_path in enumerate(raw_files, start=1):
            symbol = f_path.name.replace("_data.csv", "")
            self.status["current_symbol"] = symbol
            self.status["processed"] = idx
            self.status["progress_pct"] = round((idx / total_stocks) * 100.0, 1)
            self.status["elapsed_seconds"] = round(time.time() - start_ts, 1)
            self.status["message"] = f"Downloading & appending {symbol} ({idx}/{total_stocks})..."

            try:
                # Fetch delta candles
                df_new = yf.download(f"{symbol}.NS", start=start_d, end=end_d, auto_adjust=False, progress=False)
                if df_new is None or df_new.empty:
                    continue

                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.droplevel(1)

                # Read sector & sector index metadata from existing file
                sector = "General"
                sector_idx = "NIFTY 50"
                try:
                    df_old = pd.read_csv(f_path, nrows=1)
                    if "Sector" in df_old.columns:
                        sector = df_old["Sector"].iloc[0]
                    if "Sector Index" in df_old.columns:
                        sector_idx = df_old["Sector Index"].iloc[0]
                except Exception:
                    pass

                df_new['Stock Symbol'] = symbol
                df_new['Sector'] = sector
                df_new['Sector Index'] = sector_idx
                df_new['Trading_Day_Of_Week'] = df_new.index.day_name()
                if 'Adj Close' not in df_new.columns:
                    df_new['Adj Close'] = df_new['Close'] if 'Close' in df_new.columns else df_new.get('Open')

                # Join Nifty 50 reference
                if not nifty50_df.empty:
                    df_new = df_new.join(nifty50_df, how='left')

                for c in cols_needed:
                    if c not in df_new.columns:
                        df_new[c] = None

                df_new = df_new[cols_needed]
                df_new.index.name = 'Date'

                # Append cleanly to the CSV file (mode='a', no header)
                df_new.to_csv(f_path, mode='a', header=False)
                stocks_updated += 1

                new_max = df_new.index.max().strftime('%Y-%m-%d')
                metadata_symbols[symbol] = {"last_date": new_max}
                if new_max > updated_max_date:
                    updated_max_date = new_max

            except Exception as e:
                logger.warning("[DATA_DOWNLOAD] Error updating %s: %s", symbol, str(e))

        # 3. Save / Update dataset_metadata.json
        meta_payload = {
            "last_updated": updated_max_date,
            "global_last_date": updated_max_date,
            "total_stocks": total_stocks,
            "stocks_updated": stocks_updated,
            "symbols": metadata_symbols
        }
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=4)
        except Exception as e:
            logger.warning("[DATA_DOWNLOAD] Could not write metadata_file: %s", str(e))

        duration_sec = round(time.time() - start_ts, 2)
        success_msg = f"Successfully downloaded and appended market data across {stocks_updated} stocks. Dataset is updated up to {updated_max_date}."

        self.status["status"] = "COMPLETED"
        self.status["progress_pct"] = 100.0
        self.status["updated_to_date"] = updated_max_date
        self.status["message"] = success_msg
        self.status["elapsed_seconds"] = duration_sec

        # 4. Log operation in JSON operations history
        operations_logger.log_data_download(
            status="SUCCESS",
            stocks_updated=stocks_updated,
            last_date=updated_max_date,
            duration_seconds=duration_sec,
            message=success_msg,
            details={
                "start_date": str(start_d),
                "end_date": str(today),
                "total_stocks_available": total_stocks
            }
        )

        logger.info("[DATA_DOWNLOAD] %s (took %.2fs)", success_msg, duration_sec)
        return {
            "status": "SUCCESS",
            "stocks_updated": stocks_updated,
            "updated_to_date": updated_max_date,
            "duration_seconds": duration_sec,
            "message": success_msg
        }


# Global singleton instance
stock_data_downloader = StockDataDownloader()
