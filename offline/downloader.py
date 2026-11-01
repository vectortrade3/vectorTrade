"""
VectorTrade - Dedicated Stock Downloader & Appender Service
Implements the exact logic from 5_years_1_download.ipynb with incremental sync:
1. Reads stocks from stocks_list.txt (207 constituents)
2. Queries json_manager for last_downloaded_date (e.g. 2026-10-06)
3. Computes from_date = last_date + 1 day till today
4. Three distinct progress states:
   - CHECKING: Validates dates and fetches NIFTY 50 benchmark
   - DOWNLOADING: Fetches price bars for each constituent (1/207 ... 207/207)
   - APPENDING: Formats columns and appends new rows into 5_year_data/raw/{symbol}_data.csv
5. Updates stocks_download.json and app_logs.json upon completion
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

# Resolve directory roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager

RAW_DATA_DIR = BASE_DIR / "5_year_data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_sector_index(sector: str) -> str:
    """Maps Yahoo Finance sectors to approximate NIFTY Sector Indices."""
    mapping = {
        "Financial Services": "NIFTY FINANCIAL SERVICES",
        "Information Technology": "NIFTY IT",
        "Healthcare": "NIFTY PHARMA",
        "Consumer Defensive": "NIFTY FMCG",
        "Consumer Cyclical": "NIFTY AUTO",
        "Basic Materials": "NIFTY METAL",
        "Energy": "NIFTY ENERGY",
        "Industrials": "NIFTY INFRA",
        "Communication Services": "NIFTY MEDIA",
        "Utilities": "NIFTY ENERGY",
        "Real Estate": "NIFTY REALTY"
    }
    return mapping.get(sector, "Unknown")


def run_stock_download_pipeline() -> Dict[str, Any]:
    """
    Executes the full 3-stage download & append pipeline:
    Checking -> Downloading -> Appending -> Completed
    """
    start_time = time.time()
    today = date.today()
    symbols = json_manager.get_stocks_list()
    total_symbols = len(symbols)

    # -------------------------------------------------------------------------
    # STAGE 1: CHECKING
    # -------------------------------------------------------------------------
    last_date_str = json_manager.get_last_downloaded_date()
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception:
        last_date = today - timedelta(days=30)

    from_date = last_date + timedelta(days=1)
    
    json_manager.log_task_entry(
        task_name="STOCKS_DOWNLOAD",
        status="RUNNING",
        message=f"Starting market data sync for {total_symbols} stocks from {from_date} to {today}..."
    )

    json_manager.update_download_progress(
        status="CHECKING",
        heading="Checking",
        total_stocks=total_symbols,
        percentage=0.0,
        from_date=str(from_date),
        to_date=str(today),
        message=f"Checking latest trading days from {from_date} to {today}..."
    )

    # If already up-to-date
    if from_date >= today:
        msg = f"Dataset is already up to date ({last_date_str}). No new trading day to download."
        json_manager.update_download_progress(
            status="COMPLETED",
            heading="DOWNLOADING STOCKS",
            total_stocks=total_symbols,
            percentage=100.0,
            from_date=str(from_date),
            to_date=str(today),
            message=msg
        )
        json_manager.log_task_entry(
            task_name="STOCKS_DOWNLOAD",
            status="COMPLETED",
            message=msg
        )
        return {"status": "SUCCESS", "message": msg, "stocks_updated": 0, "date": last_date_str}

    # Load universe metadata for sectors
    universe_metadata = {}
    universe_file = BASE_DIR / "stock_names.json"
    if universe_file.exists():
        universe_metadata = json_manager.read_json_file(universe_file, {})

    # Fetch NIFTY 50 Benchmark
    nifty50_df = pd.DataFrame()
    try:
        # End is exclusive in yfinance, so today + 1 day covers today's potential close
        nifty = yf.download("^NSEI", start=from_date, end=today + timedelta(days=1), auto_adjust=False, progress=False)
        if not nifty.empty:
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.droplevel(1)
            # Filter up to today
            nifty = nifty[nifty.index.date <= today]
            nifty_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in nifty_cols:
                if c not in nifty.columns:
                    nifty[c] = 0
            nifty50_df = nifty[nifty_cols].rename(columns={
                'Open': 'Nifty50_Open',
                'High': 'Nifty50_High',
                'Low': 'Nifty50_Low',
                'Close': 'Nifty50_Close',
                'Volume': 'Nifty50_Volume'
            })
    except Exception as e:
        print(f"[WARNING] Could not fetch NIFTY 50 benchmark: {e}")

    # -------------------------------------------------------------------------
    # STAGE 2: DOWNLOADING (1 to 207)
    # -------------------------------------------------------------------------
    downloaded_dfs: Dict[str, pd.DataFrame] = {}
    
    for idx, sym in enumerate(symbols, 1):
        clean_sym = sym.strip().upper()
        pct = round((idx / total_symbols) * 100.0, 1)
        
        json_manager.update_download_progress(
            status="DOWNLOADING",
            heading="Downloading",
            current_stock=clean_sym,
            current_index=idx,
            total_stocks=total_symbols,
            percentage=pct,
            from_date=str(from_date),
            to_date=str(today),
            message=f"Downloading {clean_sym} market bars ({idx}/{total_symbols})...",
            elapsed_seconds=time.time() - start_time
        )

        try:
            df = yf.download(
                f"{clean_sym}.NS",
                start=from_date,
                end=today + timedelta(days=1),
                auto_adjust=False,
                progress=False
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                # Filter up to today
                df = df[df.index.date <= today]
                if not df.empty:
                    downloaded_dfs[clean_sym] = df
        except Exception as e:
            print(f"[ERROR] Failed downloading {clean_sym}: {e}")

        # Micro-sleep to prevent rate limiting
        time.sleep(0.04)

    # -------------------------------------------------------------------------
    # STAGE 3: APPENDING / UPDATING (1 to 207)
    # -------------------------------------------------------------------------
    latest_seen_date = last_date_str
    total_rows_appended = 0
    cols_needed = [
        'Stock Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        'Sector', 'Sector Index', 
        'Nifty50_Open', 'Nifty50_High', 'Nifty50_Low', 'Nifty50_Close', 'Nifty50_Volume',
        'Trading_Day_Of_Week'
    ]

    for idx, sym in enumerate(symbols, 1):
        clean_sym = sym.strip().upper()
        pct = round((idx / total_symbols) * 100.0, 1)

        json_manager.update_download_progress(
            status="APPENDING",
            heading="Appending Data",
            current_stock=clean_sym,
            current_index=idx,
            total_stocks=total_symbols,
            percentage=pct,
            from_date=str(from_date),
            to_date=str(today),
            message=f"Appending new records to {clean_sym}_data.csv ({idx}/{total_symbols})...",
            elapsed_seconds=time.time() - start_time
        )

        df = downloaded_dfs.get(clean_sym)
        csv_path = RAW_DATA_DIR / f"{clean_sym}_data.csv"

        if df is not None and not df.empty:
            try:
                # Sector resolution
                meta = universe_metadata.get(clean_sym, {})
                sector = meta.get("sector", "Unknown")
                if sector == "Unknown":
                    try:
                        t = yf.Ticker(f"{clean_sym}.NS")
                        sector = t.info.get("sector", "Unknown")
                    except Exception:
                        sector = "Unknown"

                sector_index = get_sector_index(sector)

                df_final = df.copy()
                df_final['Stock Symbol'] = clean_sym
                df_final['Sector'] = sector
                df_final['Sector Index'] = sector_index
                df_final['Trading_Day_Of_Week'] = df_final.index.day_name()

                if 'Adj Close' not in df_final.columns:
                    df_final['Adj Close'] = df_final['Close'] if 'Close' in df_final.columns else None

                # Join Nifty 50 columns
                if not nifty50_df.empty:
                    df_final = df_final.join(nifty50_df, how='left')
                else:
                    for col in ['Nifty50_Open', 'Nifty50_High', 'Nifty50_Low', 'Nifty50_Close', 'Nifty50_Volume']:
                        df_final[col] = None

                for c in cols_needed:
                    if c not in df_final.columns:
                        df_final[c] = None

                df_final = df_final[cols_needed]

                # Track latest date
                max_dt = df_final.index.max()
                if pd.notnull(max_dt):
                    max_dt_str = max_dt.strftime("%Y-%m-%d")
                    if max_dt_str > latest_seen_date:
                        latest_seen_date = max_dt_str

                # Read existing dates to prevent duplicate rows
                existing_dates = set()
                if csv_path.exists():
                    try:
                        prev = pd.read_csv(csv_path, usecols=['Date'])
                        existing_dates = set(prev['Date'].dropna().astype(str).tolist())
                    except Exception:
                        existing_dates = set()

                # Filter out rows already in CSV
                df_to_append = df_final[~df_final.index.strftime("%Y-%m-%d").isin(existing_dates)]

                if not df_to_append.empty:
                    total_rows_appended += len(df_to_append)
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    if csv_path.exists():
                        df_to_append.to_csv(csv_path, mode='a', header=False, index_label='Date')
                    else:
                        df_to_append.to_csv(csv_path, mode='w', header=True, index_label='Date')

            except Exception as e:
                print(f"[ERROR] Failed appending CSV for {clean_sym}: {e}")

        time.sleep(0.01)

    # -------------------------------------------------------------------------
    # STAGE 4: COMPLETED
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    if total_rows_appended > 0:
        completion_message = f"Successfully appended {total_rows_appended} new market bars across {total_symbols} stocks up to {latest_seen_date}."
    else:
        completion_message = f"All {total_symbols} stocks already up-to-date with latest market session ({latest_seen_date})."

    json_manager.update_download_progress(
        status="COMPLETED",
        heading="DOWNLOADING STOCKS",
        current_stock="ALL",
        current_index=total_symbols,
        total_stocks=total_symbols,
        percentage=100.0,
        from_date=str(from_date),
        to_date=str(today),
        message=completion_message,
        elapsed_seconds=duration
    )

    json_manager.update_stocks_download_info(
        last_date=latest_seen_date,
        status="COMPLETED",
        stocks_updated=total_symbols,
        message=completion_message,
        duration_seconds=duration
    )

    json_manager.log_app_event(
        event="STOCKS_DOWNLOAD_COMPLETED",
        status="Everything fine",
        message=completion_message
    )

    json_manager.log_task_entry(
        task_name="STOCKS_DOWNLOAD",
        status="COMPLETED",
        message=completion_message
    )

    return {
        "status": "SUCCESS",
        "last_downloaded_date": latest_seen_date,
        "stocks_updated": total_symbols,
        "rows_appended": total_rows_appended,
        "duration_seconds": duration,
        "message": completion_message
    }


def reset_raw_dataset_to_date(cutoff_date: str = "2026-10-20") -> Dict[str, Any]:
    """
    Rolls back constituent CSV records and download state back to a cutoff date.
    Used for testing the end-to-end download, appending, and real-time badge updates.
    """
    total_trimmed = 0
    total_rows_removed = 0
    for f in RAW_DATA_DIR.glob("*_data.csv"):
        try:
            df = pd.read_csv(f)
            if 'Date' in df.columns:
                filtered = df[df['Date'] <= cutoff_date]
                removed = len(df) - len(filtered)
                if removed > 0:
                    filtered.to_csv(f, index=False)
                    total_trimmed += 1
                    total_rows_removed += removed
        except Exception as e:
            print(f"Error resetting {f.name}: {e}")

    json_manager.update_stocks_download_info(
        last_date=cutoff_date,
        status="COMPLETED",
        stocks_updated=len(symbols) if 'symbols' in locals() else 207,
        message=f"Reset dataset to test baseline {cutoff_date}. Trimmed {total_rows_removed} rows.",
        duration_seconds=0.1
    )
    json_manager.update_download_progress(
        status="IDLE",
        heading="Ready",
        percentage=0.0,
        message=f"Dataset reset to {cutoff_date}. Ready to test auto-append."
    )
    return {
        "cutoff_date": cutoff_date,
        "files_trimmed": total_trimmed,
        "rows_removed": total_rows_removed,
        "status": "SUCCESS"
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Standalone Downloader...")
    print("==================================================")
    res = run_stock_download_pipeline()
    print("Result:", res)
