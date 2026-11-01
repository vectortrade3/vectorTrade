"""
VectorTrade - Dedicated Stock Downloader & Appender Service
Implements the exact logic from 5_years_1_download.ipynb with incremental sync:
1. Reads stocks from stocks_list.txt (207 constituents)
2. Queries json_manager for last_downloaded_date, last_append_date, and last_download_run_date
3. Daily check: If already downloaded and appended today, skips heavy downloading & appending
4. Three distinct progress states when execution is required:
   - CHECKING: Validates dates and fetches NIFTY 50 benchmark
   - DOWNLOADING: Fetches price bars for each constituent (1/207 ... 207/207)
   - APPENDING: Formats columns and appends new rows into 5_year_data/raw/{symbol}_data.csv
5. At the exact moment appending completes, updates stocks_download.json and broadcasts to UI
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
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


def get_latest_market_trading_date(ref_datetime: Optional[datetime] = None) -> date:
    """
    Returns the date of the latest completed NSE market trading session.
    NSE trading days: Monday (0) through Friday (4).
    Trading hours: 09:15 to 15:30 IST. EOD bhavcopy finalized by ~16:00 IST.
    If Saturday or Sunday -> previous Friday.
    If Monday before 16:00 -> previous Friday.
    If Tuesday-Friday before 16:00 -> previous day.
    """
    if ref_datetime is None:
        ref_datetime = datetime.now()
    
    current_date = ref_datetime.date()
    weekday = current_date.weekday()
    hour = ref_datetime.hour

    if weekday == 5:
        return current_date - timedelta(days=1)
    elif weekday == 6:
        return current_date - timedelta(days=2)
    else:
        if hour < 16:
            if weekday == 0:
                return current_date - timedelta(days=3)
            else:
                return current_date - timedelta(days=1)
        else:
            return current_date


def run_stock_download_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Executes the stock download & incremental append pipeline.
    Checks json_manager first:
    - If last_downloaded_date == today's date (or already run today with no new trading day),
      skips downloading.
    - If last_append_date == today's date, skips appending.
    - Otherwise runs Checking -> Downloading -> Appending -> Completed.
    - Updates stocks_download.json at the exact moment appending is completed.
    """
    start_time = time.time()
    today = date.today()
    today_str = str(today)
    symbols = json_manager.get_stocks_list()
    total_symbols = len(symbols)

    download_info = json_manager.get_stocks_download_info()
    last_date_str = download_info.get("last_downloaded_date", "2026-10-30")
    last_append_date = download_info.get("last_append_date", "")
    last_run_date = download_info.get("last_download_run_date", "")

    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception:
        last_date = today - timedelta(days=30)

    # Determine latest completed market trading session on the exchange
    latest_market_date = get_latest_market_trading_date()
    latest_market_date_str = str(latest_market_date)

    # -------------------------------------------------------------------------
    # DAILY SKIP CHECK:
    # 1. If last_downloaded_date == today's date or >= latest market trading session
    # 2. If last_append_date == today's date
    # 3. If already executed today and current with market sessions -> skip both
    # -------------------------------------------------------------------------
    is_downloaded_today = (last_date_str >= latest_market_date_str) or (last_date_str == today_str) or (last_run_date == today_str and download_info.get("status") == "COMPLETED")
    is_appended_today = (last_append_date == today_str) or (last_date_str >= latest_market_date_str and last_append_date != "")

    if not force and is_downloaded_today and is_appended_today:
        msg = f"Market data already downloaded and appended for latest market session ({last_date_str}). Current as of today ({today_str})."
        duration = round(time.time() - start_time, 2)
        
        # Exact moment completion: write JSON first, then broadcast
        json_manager.update_stocks_download_info(
            last_date=last_date_str,
            status="COMPLETED",
            stocks_updated=total_symbols,
            message=msg,
            duration_seconds=duration,
            last_append_date=today_str,
            last_download_run_date=today_str
        )
        json_manager.update_download_progress(
            status="COMPLETED",
            heading="Done",
            current_stock="ALL",
            current_index=total_symbols,
            total_stocks=total_symbols,
            percentage=100.0,
            from_date=last_date_str,
            to_date=today_str,
            message=msg,
            elapsed_seconds=duration,
            last_downloaded_date=last_date_str,
            last_append_date=today_str
        )
        json_manager.log_task_entry(
            task_name="STOCKS_DOWNLOAD",
            status="COMPLETED",
            message=msg
        )
        return {
            "status": "SUCCESS",
            "message": msg,
            "last_downloaded_date": last_date_str,
            "stocks_updated": total_symbols,
            "rows_appended": 0,
            "skipped": True
        }

    from_date = last_date + timedelta(days=1)

    # If from_date is in the future
    if from_date > today and not force:
        msg = f"Dataset is already up-to-date ({last_date_str}). No new trading day to download."
        duration = round(time.time() - start_time, 2)
        json_manager.update_stocks_download_info(
            last_date=last_date_str,
            status="COMPLETED",
            stocks_updated=total_symbols,
            message=msg,
            duration_seconds=duration,
            last_append_date=today_str,
            last_download_run_date=today_str
        )
        json_manager.update_download_progress(
            status="COMPLETED",
            heading="Done",
            current_stock="ALL",
            current_index=total_symbols,
            total_stocks=total_symbols,
            percentage=100.0,
            from_date=str(from_date),
            to_date=today_str,
            message=msg,
            elapsed_seconds=duration,
            last_downloaded_date=last_date_str,
            last_append_date=today_str
        )
        json_manager.log_task_entry(
            task_name="STOCKS_DOWNLOAD",
            status="COMPLETED",
            message=msg
        )
        return {"status": "SUCCESS", "message": msg, "stocks_updated": 0, "date": last_date_str}

    # -------------------------------------------------------------------------
    # STAGE 1: CHECKING
    # -------------------------------------------------------------------------
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
        message=f"Checking latest trading days from {from_date} to {today}...",
        last_downloaded_date=last_date_str
    )

    # Load universe metadata for sectors
    universe_metadata = {}
    universe_file = BASE_DIR / "stock_names.json"
    if universe_file.exists():
        universe_metadata = json_manager.read_json_file(universe_file, {})

    # Fetch NIFTY 50 Benchmark
    nifty50_df = pd.DataFrame()
    try:
        nifty = yf.download("^NSEI", start=from_date, end=today + timedelta(days=1), auto_adjust=False, progress=False)
        if not nifty.empty:
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.droplevel(1)
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
            elapsed_seconds=time.time() - start_time,
            last_downloaded_date=last_date_str
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
                df = df[df.index.date <= today]
                if not df.empty:
                    downloaded_dfs[clean_sym] = df
        except Exception as e:
            print(f"[ERROR] Failed downloading {clean_sym}: {e}")

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
            elapsed_seconds=time.time() - start_time,
            last_downloaded_date=latest_seen_date
        )

        df = downloaded_dfs.get(clean_sym)
        csv_path = RAW_DATA_DIR / f"{clean_sym}_data.csv"

        if df is not None and not df.empty:
            try:
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

                if not nifty50_df.empty:
                    df_final = df_final.join(nifty50_df, how='left')
                else:
                    for col in ['Nifty50_Open', 'Nifty50_High', 'Nifty50_Low', 'Nifty50_Close', 'Nifty50_Volume']:
                        df_final[col] = None

                for c in cols_needed:
                    if c not in df_final.columns:
                        df_final[c] = None

                df_final = df_final[cols_needed]

                max_dt = df_final.index.max()
                if pd.notnull(max_dt):
                    max_dt_str = max_dt.strftime("%Y-%m-%d")
                    if max_dt_str > latest_seen_date:
                        latest_seen_date = max_dt_str

                existing_dates = set()
                if csv_path.exists():
                    try:
                        prev = pd.read_csv(csv_path, usecols=['Date'])
                        existing_dates = set(prev['Date'].dropna().astype(str).tolist())
                    except Exception:
                        existing_dates = set()

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
    # STAGE 4: EXACT MOMENT APPENDING IS COMPLETED
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    effective_last_date = latest_seen_date if latest_seen_date > last_date_str else last_date_str

    if total_rows_appended > 0:
        completion_message = f"Successfully appended {total_rows_appended} new market bars across {total_symbols} stocks up to {effective_last_date}."
    else:
        completion_message = f"All {total_symbols} stocks already up-to-date with latest market session ({effective_last_date})."

    # 1. Update stocks_download.json FIRST
    json_manager.update_stocks_download_info(
        last_date=effective_last_date,
        status="COMPLETED",
        stocks_updated=total_symbols,
        message=completion_message,
        duration_seconds=duration,
        last_append_date=today_str,
        last_download_run_date=today_str
    )

    # 2. Update live download progress (broadcasts to WebSocket with latest date synchronously)
    json_manager.update_download_progress(
        status="COMPLETED",
        heading="Done",
        current_stock="ALL",
        current_index=total_symbols,
        total_stocks=total_symbols,
        percentage=100.0,
        from_date=str(from_date),
        to_date=str(today),
        message=completion_message,
        elapsed_seconds=duration,
        last_downloaded_date=effective_last_date,
        last_append_date=today_str
    )

    # 3. Log app event & task audit log
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
        "last_downloaded_date": effective_last_date,
        "stocks_updated": total_symbols,
        "rows_appended": total_rows_appended,
        "duration_seconds": duration,
        "message": completion_message
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Standalone Downloader...")
    print("==================================================")
    res = run_stock_download_pipeline()
    print("Result:", res)
