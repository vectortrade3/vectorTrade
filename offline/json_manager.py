"""
VectorTrade - JSON Data & State Manager
Provides centralized, thread-safe, atomic operations for managing all project JSON files:
- Stocks download state & history (stocks_download.json)
- System application lifecycle & error logs (app_logs.json)
- AI Model metadata & training records (models_state.json)
- Live real-time download progress tracking (download_progress.json)
- Stocks universe & constituent cache (stocks_universe.json)
"""

from __future__ import annotations
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base directories resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STORE_DIR = BASE_DIR / "data_store"
DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Central JSON file locations
STOCKS_DOWNLOAD_FILE = DATA_STORE_DIR / "stocks_download.json"
APP_LOGS_FILE = DATA_STORE_DIR / "app_logs.json"
MODELS_STATE_FILE = DATA_STORE_DIR / "models_state.json"
DOWNLOAD_PROGRESS_FILE = DATA_STORE_DIR / "download_progress.json"
FEATURE_GEN_FILE = DATA_STORE_DIR / "feature_gen.json"
FEATURE_GEN_PROGRESS_FILE = DATA_STORE_DIR / "feature_gen_progress.json"
TASK_LOGS_FILE = DATA_STORE_DIR / "task_logs.json"
STOCKS_LIST_FILE = DATA_STORE_DIR / "stocks_list.txt"

# Thread safety lock
_FILE_LOCK = threading.Lock()


# ==============================================================================
# GENERIC ATOMIC FILE HELPERS
# ==============================================================================

def read_json_file(file_path: Path, default_val: Any = None) -> Any:
    """Reads a JSON file safely with a fallback default."""
    with _FILE_LOCK:
        if not file_path.exists():
            return default_val if default_val is not None else {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_val if default_val is not None else {}


def write_json_file(file_path: Path, data: Any) -> bool:
    """Writes data to a JSON file atomically using a temporary file to avoid corruption."""
    with _FILE_LOCK:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_suffix(f".tmp_{os.getpid()}_{int(time.time()*1000)}")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            
            # Atomic replacement with retries on Windows
            for _ in range(5):
                try:
                    temp_path.replace(file_path)
                    return True
                except Exception:
                    time.sleep(0.05)
            if temp_path.exists():
                os.remove(temp_path)
            return False
        except Exception:
            return False


# ==============================================================================
# 1. STOCKS DOWNLOAD STATE MANAGER
# ==============================================================================

def initialize_stocks_download_state() -> Dict[str, Any]:
    """Ensures stocks_download.json is initialized with download and append tracking."""
    default_state = {
        "last_downloaded_date": "2026-10-30",
        "last_append_date": "2026-10-30",
        "last_download_run_date": "2026-10-30",
        "status": "COMPLETED",
        "total_stocks": 207,
        "last_updated": datetime.now().astimezone().isoformat(),
        "message": "Market data synced up to 2026-10-30.",
        "download_history": []
    }
    
    if not STOCKS_DOWNLOAD_FILE.exists():
        write_json_file(STOCKS_DOWNLOAD_FILE, default_state)
        return default_state
    
    current = read_json_file(STOCKS_DOWNLOAD_FILE, default_state)
    needs_save = False
    if not current.get("last_downloaded_date"):
        current["last_downloaded_date"] = "2026-10-30"
        needs_save = True
    if not current.get("last_append_date"):
        current["last_append_date"] = current.get("last_downloaded_date", "2026-10-30")
        needs_save = True
    if not current.get("last_download_run_date"):
        current["last_download_run_date"] = current.get("last_downloaded_date", "2026-10-30")
        needs_save = True
    if needs_save:
        write_json_file(STOCKS_DOWNLOAD_FILE, current)
    return current


def get_stocks_download_info() -> Dict[str, Any]:
    """Retrieves current stocks download metadata."""
    return initialize_stocks_download_state()


def get_last_downloaded_date() -> str:
    """Returns the last market data date downloaded (e.g. '2026-10-30')."""
    info = get_stocks_download_info()
    return info.get("last_downloaded_date", "2026-10-30")


def get_last_append_date() -> str:
    """Returns the date on which records were last appended."""
    info = get_stocks_download_info()
    return info.get("last_append_date", "")


def get_last_download_run_date() -> str:
    """Returns the calendar date on which the downloader was last executed."""
    info = get_stocks_download_info()
    return info.get("last_download_run_date", "")


# ==============================================================================
# Progress Notification Listeners (Real-Time WebSockets)
# ==============================================================================

_progress_listeners = []

def register_progress_listener(listener):
    """Registers a listener function fn(event_type: str, data: dict)."""
    if listener not in _progress_listeners:
        _progress_listeners.append(listener)

def notify_progress_listeners(event_type: str, data: Dict[str, Any]):
    """Dispatches real-time event updates to all registered listeners."""
    for listener in list(_progress_listeners):
        try:
            listener(event_type, data)
        except Exception as e:
            pass


def update_stocks_download_info(
    last_date: str,
    status: str = "COMPLETED",
    stocks_updated: int = 207,
    message: str = "",
    duration_seconds: float = 0.0,
    last_append_date: Optional[str] = None,
    last_download_run_date: Optional[str] = None
) -> Dict[str, Any]:
    """Updates the stocks download state and appends to the history log."""
    current = get_stocks_download_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")
    
    append_date = last_append_date or current.get("last_append_date") or today_str
    run_date = last_download_run_date or current.get("last_download_run_date") or today_str

    history_entry = {
        "timestamp": now_str,
        "date_synced": last_date,
        "last_append_date": append_date,
        "status": status,
        "stocks_updated": stocks_updated,
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    }
    
    current["last_downloaded_date"] = last_date
    current["last_append_date"] = append_date
    current["last_download_run_date"] = run_date
    current["status"] = status
    current["total_stocks"] = stocks_updated
    current["last_updated"] = now_str
    current["message"] = message
    
    history = current.get("download_history", [])
    history.insert(0, history_entry)
    current["download_history"] = history[:100]  # keep last 100 runs
    
    write_json_file(STOCKS_DOWNLOAD_FILE, current)
    notify_progress_listeners("STOCKS_DOWNLOAD_INFO", current)
    return current


# ==============================================================================
# 2. APPLICATION LIFECYCLE & SYSTEM LOGS
# ==============================================================================

def log_app_event(
    event: str,
    status: str = "Everything fine",
    message: str = "",
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Appends an application lifecycle event to app_logs.json.
    e.g. event='APP_STARTED', status='Everything fine', time='...'
    """
    now = datetime.now().astimezone()
    log_entry = {
        "timestamp": now.isoformat(),
        "readable_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "event": event,
        "status": status,
        "message": message,
        "error": error
    }
    if extra:
        log_entry["extra"] = extra
        
    logs_data = read_json_file(APP_LOGS_FILE, {"logs": []})
    logs_list = logs_data.get("logs", [])
    logs_list.insert(0, log_entry)
    logs_data["logs"] = logs_list[:500]  # keep last 500 logs
    logs_data["last_event_time"] = log_entry["timestamp"]
    
    write_json_file(APP_LOGS_FILE, logs_data)
    return log_entry


def get_app_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns the most recent app log events."""
    logs_data = read_json_file(APP_LOGS_FILE, {"logs": []})
    return logs_data.get("logs", [])[:limit]


# ==============================================================================
# 3. LIVE DOWNLOAD PROGRESS TRACKER (FOR REAL-TIME UI POLLING)
# ==============================================================================

def get_download_progress() -> Dict[str, Any]:
    """Returns the current real-time state of the stock download process."""
    default_progress = {
        "status": "IDLE",            # IDLE, CHECKING, DOWNLOADING, APPENDING, COMPLETED, ERROR
        "heading": "Ready",
        "current_stock": "",
        "current_index": 0,
        "total_stocks": 207,
        "percentage": 0.0,
        "from_date": None,
        "to_date": None,
        "message": "Ready to download market data.",
        "elapsed_seconds": 0.0,
        "error": None
    }
    return read_json_file(DOWNLOAD_PROGRESS_FILE, default_progress)


def update_download_progress(
    status: str,
    heading: str,
    current_stock: str = "",
    current_index: int = 0,
    total_stocks: int = 207,
    percentage: float = 0.0,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    message: str = "",
    elapsed_seconds: float = 0.0,
    error: Optional[str] = None,
    last_downloaded_date: Optional[str] = None,
    last_append_date: Optional[str] = None
) -> None:
    """Updates the live progress JSON file for UI polling & WebSocket stream."""
    progress_data = {
        "status": status,
        "heading": heading,
        "current_stock": current_stock,
        "current_index": current_index,
        "total_stocks": total_stocks,
        "percentage": round(percentage, 1),
        "from_date": from_date,
        "to_date": to_date,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
        "last_downloaded_date": last_downloaded_date,
        "last_append_date": last_append_date,
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(DOWNLOAD_PROGRESS_FILE, progress_data)
    notify_progress_listeners("DOWNLOAD_PROGRESS", progress_data)


# ==============================================================================
# 4. MODELS STATE MANAGER
# ==============================================================================

def get_models_info() -> Dict[str, Any]:
    """Returns AI model training metadata."""
    default_state = {
        "last_trained_date": "2026-10-26",
        "status": "READY",
        "models": [
            "5_year_xgboost_model.pkl",
            "5_year_global_gain_model.pkl",
            "5_year_global_drop_model.pkl"
        ],
        "message": "Models trained on 5-year historical regime data."
    }
    return read_json_file(MODELS_STATE_FILE, default_state)


def update_models_info(last_trained: str, status: str, models: List[str], message: str) -> None:
    """Persists model training status."""
    data = {
        "last_trained_date": last_trained,
        "status": status,
        "models": models,
        "message": message,
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(MODELS_STATE_FILE, data)


# ==============================================================================
# 5. STOCKS LIST UTILITY
# ==============================================================================

def get_stocks_list() -> List[str]:
    """Reads the hardcoded/configured 207 stocks list from stocks_list.txt."""
    if STOCKS_LIST_FILE.exists():
        with open(STOCKS_LIST_FILE, "r", encoding="utf-8") as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
            if symbols:
                return symbols
                
    # Fallback to stock_names.json if txt missing
    stock_names_file = BASE_DIR / "stock_names.json"
    if stock_names_file.exists():
        data = read_json_file(stock_names_file, {})
        return sorted(list(data.keys()))
        
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


# ==============================================================================
# 6. FEATURE GENERATION STATE & PROGRESS MANAGER
# ==============================================================================

def initialize_feature_gen_state() -> Dict[str, Any]:
    """Ensures feature_gen.json is initialized with previous baseline info and run tracking."""
    default_state = {
        "last_feature_gen_date": "2026-10-30",
        "last_feature_run_date": "2026-10-30",
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
        "status": "COMPLETED",
        "total_stocks": 207,
        "message": "Baseline feature engineering computed for 207 stocks.",
        "history": []
    }
    if not FEATURE_GEN_FILE.exists():
        write_json_file(FEATURE_GEN_FILE, default_state)
        return default_state
    current = read_json_file(FEATURE_GEN_FILE, default_state)
    needs_save = False
    if not current.get("last_feature_gen_date"):
        current["last_feature_gen_date"] = "2026-10-30"
        needs_save = True
    if not current.get("last_feature_run_date"):
        current["last_feature_run_date"] = current.get("last_feature_gen_date", "2026-10-30")
        needs_save = True
    if needs_save:
        write_json_file(FEATURE_GEN_FILE, current)
    return current


def get_feature_gen_info() -> Dict[str, Any]:
    """Retrieves feature generation metadata including last run date."""
    return initialize_feature_gen_state()


def update_feature_gen_info(
    status: str = "COMPLETED",
    total_stocks: int = 207,
    message: str = "",
    duration_seconds: float = 0.0,
    last_feature_gen_date: Optional[str] = None,
    last_feature_run_date: Optional[str] = None
) -> Dict[str, Any]:
    """Updates feature_gen.json and appends run entry to history."""
    current = get_feature_gen_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    feat_date = last_feature_gen_date or current.get("last_feature_gen_date") or today_str
    run_date = last_feature_run_date or current.get("last_feature_run_date") or today_str

    entry = {
        "timestamp": now_str,
        "date": feat_date,
        "run_date": run_date,
        "status": status,
        "total_stocks": total_stocks,
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    }

    current["last_feature_gen_date"] = feat_date
    current["last_feature_run_date"] = run_date
    current["last_run_timestamp"] = now_str
    current["status"] = status
    current["total_stocks"] = total_stocks
    current["message"] = message

    history = current.get("history", [])
    history.insert(0, entry)
    current["history"] = history[:100]

    write_json_file(FEATURE_GEN_FILE, current)
    notify_progress_listeners("FEATURE_GEN_INFO", current)
    return current


def get_feature_gen_progress() -> Dict[str, Any]:
    """Retrieves live real-time progress for Feature Generation."""
    default_progress = {
        "status": "IDLE",            # IDLE, GENERATING, SAVING, COMPLETED, ERROR
        "heading": "Ready",
        "current_stock": "",
        "current_index": 0,
        "total_stocks": 207,
        "percentage": 0.0,
        "message": "Ready to generate features.",
        "elapsed_seconds": 0.0,
        "error": None,
        "last_feature_gen_date": None
    }
    return read_json_file(FEATURE_GEN_PROGRESS_FILE, default_progress)


def update_feature_gen_progress(
    status: str,
    heading: str,
    current_stock: str = "",
    current_index: int = 0,
    total_stocks: int = 207,
    percentage: float = 0.0,
    message: str = "",
    elapsed_seconds: float = 0.0,
    error: Optional[str] = None,
    last_feature_gen_date: Optional[str] = None
) -> None:
    """Updates live transient feature generation progress."""
    data = {
        "status": status,
        "heading": heading,
        "current_stock": current_stock,
        "current_index": current_index,
        "total_stocks": total_stocks,
        "percentage": round(percentage, 1),
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
        "last_feature_gen_date": last_feature_gen_date,
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(FEATURE_GEN_PROGRESS_FILE, data)
    notify_progress_listeners("FEATURE_GEN_PROGRESS", data)


# ==============================================================================
# 7. UNIFIED TASK LOGS (FOR COMING SOON LOGS PAGE)
# ==============================================================================

def log_task_entry(
    task_name: str,
    status: str,
    message: str = "",
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs task execution with:
    task_name (e.g. STOCKS_DOWNLOAD, FEATURE_GENERATION)
    status ('RUNNING', 'COMPLETED', 'ERROR')
    timestamp, error if any
    """
    now = datetime.now().astimezone()
    entry = {
        "task_name": task_name,
        "status": status,
        "timestamp": now.isoformat(),
        "readable_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "message": message,
        "error": error
    }

    logs_data = read_json_file(TASK_LOGS_FILE, {"task_logs": []})
    logs_list = logs_data.get("task_logs", [])
    logs_list.insert(0, entry)
    logs_data["task_logs"] = logs_list[:500]
    logs_data["last_task"] = task_name
    logs_data["last_status"] = status
    logs_data["last_updated"] = entry["timestamp"]

    write_json_file(TASK_LOGS_FILE, logs_data)
    
    # Also mirror into app_logs.json for unified visibility
    log_app_event(
        event=task_name,
        status="Everything fine" if status in ["RUNNING", "COMPLETED"] else "Error",
        message=f"[{status}] {message}",
        error=error
    )

    return entry


def get_task_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns task logs for the log page."""
    logs_data = read_json_file(TASK_LOGS_FILE, {"task_logs": []})
    return logs_data.get("task_logs", [])[:limit]


# Auto-initialize state on import
initialize_stocks_download_state()
initialize_feature_gen_state()

