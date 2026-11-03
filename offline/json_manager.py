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
MODELS_TRAINING_PROGRESS_FILE = DATA_STORE_DIR / "models_training_progress.json"
DOWNLOAD_PROGRESS_FILE = DATA_STORE_DIR / "download_progress.json"
FEATURE_GEN_FILE = DATA_STORE_DIR / "feature_gen.json"
FEATURE_GEN_PROGRESS_FILE = DATA_STORE_DIR / "feature_gen_progress.json"
DATA_CLEANING_FILE = DATA_STORE_DIR / "data_cleaning.json"
DATA_CLEANING_PROGRESS_FILE = DATA_STORE_DIR / "data_cleaning_progress.json"
TASK_LOGS_FILE = DATA_STORE_DIR / "task_logs.json"
STOCKS_LIST_FILE = DATA_STORE_DIR / "stocks_list.txt"
INFERENCE_STATE_FILE = DATA_STORE_DIR / "inference_state.json"
INFERENCE_PROGRESS_FILE = DATA_STORE_DIR / "inference_progress.json"
INFERENCE_50MA_TRADES_FILE = DATA_STORE_DIR / "inference_50ma_trades.json"
INFERENCE_GLOBAL_TRADES_FILE = DATA_STORE_DIR / "inference_global_trades.json"
DAILY_ANALYSIS_FILE = DATA_STORE_DIR / "daily_analysis.json"
DAILY_ANALYSIS_PROGRESS_FILE = DATA_STORE_DIR / "daily_analysis_progress.json"
MODEL_PERFORMANCE_LOG_FILE = BASE_DIR / "models" / "model_performance_log.json"

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
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
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
    if not current.get("last_run_timestamp"):
        current["last_run_timestamp"] = current.get("last_updated") or datetime.now().astimezone().isoformat()
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
    current["last_run_timestamp"] = now_str
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

# ==============================================================================
# 4. MODELS STATE MANAGER (SECTION 1: STRATEGY & SECTION 2: GLOBAL)
# ==============================================================================

def initialize_models_state() -> Dict[str, Any]:
    """Ensures models_state.json is initialized with Section 1 (Strategy) and Section 2 (Global) info."""
    default_state = {
        "status": "COMPLETED",
        "last_trained_date": "2026-10-30",
        "last_trained_run_date": "2026-10-30",
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
        "strategy_model": {
            "status": "COMPLETED",
            "last_trained_date": "2026-10-30",
            "last_run_date": "2026-10-30",
            "model_file": "models/5_year_xgboost_model.pkl",
            "scaler_file": "models/5_year_scaler.pkl",
            "imputer_file": "models/5_year_imputer.pkl",
            "features_file": "final_features.txt",
            "features_count": 87,
            "mae_base": 0.45,
            "mae_refined": 0.41,
            "message": "Strategy XGBoost model tuned on triggered trade setups."
        },
        "global_models": {
            "status": "COMPLETED",
            "last_trained_date": "2026-10-30",
            "last_run_date": "2026-10-30",
            "gain_model_file": "models/5_year_global_gain_model.pkl",
            "drop_model_file": "models/5_year_global_drop_model.pkl",
            "scaler_file": "models/5_year_global_scaler.pkl",
            "imputer_file": "models/5_year_global_imputer.pkl",
            "features_file": "global_features.txt",
            "features_count": 92,
            "mae_gain": 0.042,
            "mae_drop": 0.038,
            "message": "Global multi-horizon gain and drop regime predictors tuned."
        },
        "models": [
            "5_year_xgboost_model.pkl",
            "5_year_global_gain_model.pkl",
            "5_year_global_drop_model.pkl"
        ],
        "message": "All models calibrated on 5-year consolidated regime dataset.",
        "history": []
    }
    if not MODELS_STATE_FILE.exists():
        write_json_file(MODELS_STATE_FILE, default_state)
        return default_state
    current = read_json_file(MODELS_STATE_FILE, default_state)
    needs_save = False
    if "strategy_model" not in current:
        current["strategy_model"] = default_state["strategy_model"]
        needs_save = True
    if "global_models" not in current:
        current["global_models"] = default_state["global_models"]
        needs_save = True
    if not current.get("last_trained_date"):
        current["last_trained_date"] = "2026-10-30"
        needs_save = True
    if not current.get("last_trained_run_date"):
        current["last_trained_run_date"] = current.get("last_trained_date", "2026-10-30")
        needs_save = True
    if needs_save:
        write_json_file(MODELS_STATE_FILE, current)
    return current


def get_models_info() -> Dict[str, Any]:
    """Returns AI model training metadata."""
    return initialize_models_state()


def update_strategy_model_info(
    status: str = "COMPLETED",
    mae_base: float = 0.0,
    mae_refined: float = 0.0,
    features_count: int = 0,
    message: str = "",
    duration_seconds: float = 0.0,
    last_trained_date: Optional[str] = None,
    last_run_date: Optional[str] = None
) -> Dict[str, Any]:
    """Updates strategy model metadata in models_state.json."""
    current = get_models_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    trained_date = last_trained_date or current.get("strategy_model", {}).get("last_trained_date") or today_str
    run_date = last_run_date or today_str

    current["strategy_model"] = {
        "status": status,
        "last_trained_date": trained_date,
        "last_run_date": run_date,
        "model_file": "models/5_year_xgboost_model.pkl",
        "scaler_file": "models/5_year_scaler.pkl",
        "imputer_file": "models/5_year_imputer.pkl",
        "features_file": "final_features.txt",
        "features_count": features_count,
        "mae_base": round(mae_base, 3),
        "mae_refined": round(mae_refined, 3),
        "duration_seconds": round(duration_seconds, 2),
        "message": message,
        "updated_at": now_str
    }
    current["last_trained_date"] = trained_date
    current["last_trained_run_date"] = run_date
    current["last_run_timestamp"] = now_str
    current["message"] = message

    history = current.get("history", [])
    history.insert(0, {
        "timestamp": now_str,
        "section": "STRATEGY_MODEL",
        "status": status,
        "date": trained_date,
        "run_date": run_date,
        "mae_base": round(mae_base, 3),
        "mae_refined": round(mae_refined, 3),
        "features_count": features_count,
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    })
    current["history"] = history[:100]

    write_json_file(MODELS_STATE_FILE, current)
    notify_progress_listeners("MODELS_TRAINING_INFO", current)
    return current


def update_global_models_info(
    status: str = "COMPLETED",
    mae_gain: float = 0.0,
    mae_drop: float = 0.0,
    features_count: int = 0,
    message: str = "",
    duration_seconds: float = 0.0,
    last_trained_date: Optional[str] = None,
    last_run_date: Optional[str] = None
) -> Dict[str, Any]:
    """Updates global models metadata in models_state.json."""
    current = get_models_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    trained_date = last_trained_date or current.get("global_models", {}).get("last_trained_date") or today_str
    run_date = last_run_date or today_str

    current["global_models"] = {
        "status": status,
        "last_trained_date": trained_date,
        "last_run_date": run_date,
        "gain_model_file": "models/5_year_global_gain_model.pkl",
        "drop_model_file": "models/5_year_global_drop_model.pkl",
        "scaler_file": "models/5_year_global_scaler.pkl",
        "imputer_file": "models/5_year_global_imputer.pkl",
        "features_file": "global_features.txt",
        "features_count": features_count,
        "mae_gain": round(mae_gain, 4),
        "mae_drop": round(mae_drop, 4),
        "duration_seconds": round(duration_seconds, 2),
        "message": message,
        "updated_at": now_str
    }
    current["last_trained_date"] = trained_date
    current["last_trained_run_date"] = run_date
    current["last_run_timestamp"] = now_str
    current["message"] = message

    history = current.get("history", [])
    history.insert(0, {
        "timestamp": now_str,
        "section": "GLOBAL_MODELS",
        "status": status,
        "date": trained_date,
        "run_date": run_date,
        "mae_gain": round(mae_gain, 4),
        "mae_drop": round(mae_drop, 4),
        "features_count": features_count,
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    })
    current["history"] = history[:100]

    write_json_file(MODELS_STATE_FILE, current)
    notify_progress_listeners("MODELS_TRAINING_INFO", current)
    return current


def get_models_training_progress() -> Dict[str, Any]:
    """Retrieves live real-time progress for Model Training."""
    default_progress = {
        "status": "IDLE",            # IDLE, TRAINING, COMPLETED, ERROR
        "active_section": "NONE",    # STRATEGY, GLOBAL, ALL
        "phase": "Ready",
        "heading": "Ready",
        "current_model": "",
        "percentage": 0.0,
        "message": "Ready to train predictive models.",
        "elapsed_seconds": 0.0,
        "error": None,
        "metrics": {}
    }
    return read_json_file(MODELS_TRAINING_PROGRESS_FILE, default_progress)


def update_models_training_progress(
    status: str,
    active_section: str = "STRATEGY",
    phase: str = "Tuning",
    heading: str = "Model Training",
    current_model: str = "",
    percentage: float = 0.0,
    message: str = "",
    elapsed_seconds: float = 0.0,
    error: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Updates live transient model training progress and broadcasts to listeners."""
    data = {
        "status": status,
        "active_section": active_section,
        "phase": phase,
        "heading": heading,
        "current_model": current_model,
        "percentage": round(percentage, 1),
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
        "metrics": metrics or {},
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(MODELS_TRAINING_PROGRESS_FILE, data)
    notify_progress_listeners("MODELS_TRAINING_PROGRESS", data)


def update_models_info(last_trained: str, status: str, models: List[str], message: str) -> None:
    """Backward compatibility helper for persisting general model training status."""
    current = get_models_info()
    current["last_trained_date"] = last_trained
    current["status"] = status
    current["models"] = models
    current["message"] = message
    current["updated_at"] = datetime.now().astimezone().isoformat()
    write_json_file(MODELS_STATE_FILE, current)
    notify_progress_listeners("MODELS_TRAINING_INFO", current)


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


# ==============================================================================
# 8. DATA CLEANING STATE & PROGRESS MANAGER
# ==============================================================================

def initialize_data_cleaning_state() -> Dict[str, Any]:
    """Ensures data_cleaning.json is initialized with previous baseline info."""
    default_state = {
        "last_clean_date": "2026-10-30",
        "last_clean_run_date": "2026-10-30",
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
        "status": "COMPLETED",
        "total_rows": 248863,
        "total_columns": 157,
        "total_stocks": 207,
        "output_file": "5_year_final_data.csv",
        "message": "Baseline 5-year combined and cleaned dataset ready.",
        "history": []
    }
    if not DATA_CLEANING_FILE.exists():
        write_json_file(DATA_CLEANING_FILE, default_state)
        return default_state
    current = read_json_file(DATA_CLEANING_FILE, default_state)
    needs_save = False
    if not current.get("last_clean_date"):
        current["last_clean_date"] = "2026-10-30"
        needs_save = True
    if not current.get("last_clean_run_date"):
        current["last_clean_run_date"] = current.get("last_clean_date", "2026-10-30")
        needs_save = True
    if current.get("total_columns") != 157:
        current["total_columns"] = 157
        needs_save = True
    if needs_save:
        write_json_file(DATA_CLEANING_FILE, current)
    return current


def get_data_cleaning_info() -> Dict[str, Any]:
    """Retrieves data cleaning metadata."""
    return initialize_data_cleaning_state()


def update_data_cleaning_info(
    status: str = "COMPLETED",
    total_rows: int = 0,
    total_columns: int = 157,
    total_stocks: int = 207,
    message: str = "",
    duration_seconds: float = 0.0,
    last_clean_date: Optional[str] = None,
    last_clean_run_date: Optional[str] = None,
    output_file: str = "5_year_final_data.csv",
    **kwargs: Any
) -> Dict[str, Any]:
    """Updates data_cleaning.json and appends run entry to history."""
    current = get_data_cleaning_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    clean_date = last_clean_date or current.get("last_clean_date") or today_str
    run_date = last_clean_run_date or current.get("last_clean_run_date") or today_str

    entry = {
        "timestamp": now_str,
        "clean_date": clean_date,
        "run_date": run_date,
        "status": status,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_stocks": total_stocks,
        "output_file": output_file,
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    }
    entry.update(kwargs)

    current["last_clean_date"] = clean_date
    current["last_clean_run_date"] = run_date
    current["last_run_timestamp"] = now_str
    current["status"] = status
    current["total_rows"] = total_rows
    current["total_columns"] = total_columns
    current["total_stocks"] = total_stocks
    current["output_file"] = output_file
    current["message"] = message
    current.update(kwargs)

    history = current.get("history", [])
    history.insert(0, entry)
    current["history"] = history[:100]

    write_json_file(DATA_CLEANING_FILE, current)
    notify_progress_listeners("DATA_CLEAN_INFO", current)
    return current


def get_data_cleaning_progress() -> Dict[str, Any]:
    """Retrieves live real-time progress for Data Cleaning."""
    default_progress = {
        "status": "IDLE",            # IDLE, COMBINING, CLEANING, SAVING, COMPLETED, ERROR
        "heading": "Ready",
        "current_stock": "",
        "current_index": 0,
        "total_stocks": 207,
        "percentage": 0.0,
        "message": "Ready to clean and combine feature data.",
        "elapsed_seconds": 0.0,
        "error": None,
        "last_clean_date": None,
        "total_rows": 0,
        "total_trades_50ma": 0
    }
    return read_json_file(DATA_CLEANING_PROGRESS_FILE, default_progress)


def update_data_cleaning_progress(
    status: str,
    heading: str,
    current_stock: str = "",
    current_index: int = 0,
    total_stocks: int = 207,
    percentage: float = 0.0,
    message: str = "",
    elapsed_seconds: float = 0.0,
    error: Optional[str] = None,
    last_clean_date: Optional[str] = None,
    total_rows: int = 0,
    total_trades_50ma: int = 0
) -> None:
    """Updates live transient data cleaning progress and broadcasts to listeners."""
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
        "last_clean_date": last_clean_date,
        "total_rows": total_rows,
        "total_trades_50ma": total_trades_50ma,
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(DATA_CLEANING_PROGRESS_FILE, data)
    notify_progress_listeners("DATA_CLEAN_PROGRESS", data)


# ==============================================================================
# 9. INFERENCE & MODEL EVALUATION STATE MANAGER (PART 5 & 5.2)
# ==============================================================================

def initialize_inference_state() -> Dict[str, Any]:
    """Ensures inference_state.json is initialized with Section 1 (Strategy) and Section 2 (Global) inference results."""
    default_state = {
        "status": "COMPLETED",
        "last_inference_date": "2026-11-01",
        "last_inference_run_date": "2026-11-01",
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
        "strategy_inference": {
            "status": "COMPLETED",
            "last_run_date": "2026-11-01",
            "accuracy_base": "44.6%",
            "mae_base": 0.886,
            "accuracy_refined": "43.9%",
            "win_rate_recall": "77.3%",
            "mae_refined": 0.886,
            "base_features_count": 137,
            "refined_features_count": 97,
            "pruned_features_count": 40,
            "duration_seconds": 0.0,
            "message": "Strategy models evaluated on historical out-of-sample test set."
        },
        "global_inference": {
            "status": "COMPLETED",
            "last_run_date": "2026-11-01",
            "mae_gain_base": 15.58,
            "mae_drop_base": 9.77,
            "mae_gain_refined": 15.93,
            "mae_drop_refined": 9.92,
            "directional_accuracy": "48.8%",
            "base_features_count": 137,
            "refined_features_count": 91,
            "pruned_features_count": 46,
            "duration_seconds": 0.0,
            "message": "Global multi-horizon gain and drop regime predictors evaluated."
        },
        "message": "All models evaluated against historical test splits.",
        "history": []
    }
    if not INFERENCE_STATE_FILE.exists():
        write_json_file(INFERENCE_STATE_FILE, default_state)
        return default_state

    current = read_json_file(INFERENCE_STATE_FILE, default_state)
    needs_save = False
    if "strategy_inference" not in current:
        current["strategy_inference"] = default_state["strategy_inference"]
        needs_save = True
    if "global_inference" not in current:
        current["global_inference"] = default_state["global_inference"]
        needs_save = True
    if not current.get("last_inference_date"):
        current["last_inference_date"] = "2026-11-01"
        needs_save = True
    if not current.get("last_inference_run_date"):
        current["last_inference_run_date"] = current.get("last_inference_date", "2026-11-01")
        needs_save = True
    if needs_save:
        write_json_file(INFERENCE_STATE_FILE, current)
    return current


def get_inference_info() -> Dict[str, Any]:
    """Returns inference state and evaluation metrics."""
    return initialize_inference_state()


def update_strategy_inference_info(
    status: str = "COMPLETED",
    accuracy_base: str = "0.0%",
    mae_base: float = 0.0,
    accuracy_refined: str = "0.0%",
    win_rate_recall: str = "0.0%",
    mae_refined: float = 0.0,
    base_features_count: int = 137,
    refined_features_count: int = 97,
    message: str = "",
    duration_seconds: float = 0.0,
    last_inference_date: Optional[str] = None,
    last_run_date: Optional[str] = None,
    win_rate_taken: str = "0.0%",
    win_rate_all: str = "0.0%",
    overall_pnl_pct: float = 0.0,
    net_r_taken: float = 0.0,
    total_trades: int = 0,
    taken_trades: int = 0,
    wins_count: int = 0,
    losses_count: int = 0
) -> Dict[str, Any]:
    """Updates Section 1 (Strategy) inference results including trade win rate and P&L metrics."""
    current = get_inference_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    eff_inf_date = last_inference_date or current.get("last_inference_date") or today_str
    eff_run_date = last_run_date or today_str
    pruned = max(0, base_features_count - refined_features_count)

    strat_info = {
        "status": status,
        "last_run_date": eff_run_date,
        "accuracy_base": accuracy_base,
        "mae_base": round(mae_base, 3),
        "accuracy_refined": accuracy_refined,
        "win_rate_recall": win_rate_recall,
        "mae_refined": round(mae_refined, 3),
        "win_rate_taken": win_rate_taken,
        "win_rate_all": win_rate_all,
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "net_r_taken": round(net_r_taken, 2),
        "total_trades": total_trades,
        "taken_trades": taken_trades,
        "wins_count": wins_count,
        "losses_count": losses_count,
        "base_features_count": base_features_count,
        "refined_features_count": refined_features_count,
        "pruned_features_count": pruned,
        "duration_seconds": round(duration_seconds, 2),
        "message": message,
        "updated_at": now_str
    }
    current["strategy_inference"] = strat_info
    current["last_inference_date"] = eff_inf_date
    current["last_inference_run_date"] = eff_run_date
    current["last_run_timestamp"] = now_str
    current["status"] = status
    current["message"] = message

    history = current.get("history", [])
    history.insert(0, {
        "timestamp": now_str,
        "section": "STRATEGY_INFERENCE",
        "status": status,
        "date": eff_inf_date,
        "run_date": eff_run_date,
        "accuracy_refined": accuracy_refined,
        "win_rate_taken": win_rate_taken,
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "win_rate_recall": win_rate_recall,
        "mae_refined": round(mae_refined, 3),
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    })
    current["history"] = history[:100]

    write_json_file(INFERENCE_STATE_FILE, current)
    notify_progress_listeners("INFERENCE_INFO", current)
    return current


def update_global_inference_info(
    status: str = "COMPLETED",
    mae_gain_base: float = 0.0,
    mae_drop_base: float = 0.0,
    mae_gain_refined: float = 0.0,
    mae_drop_refined: float = 0.0,
    directional_accuracy: str = "0.0%",
    base_features_count: int = 137,
    refined_features_count: int = 91,
    message: str = "",
    duration_seconds: float = 0.0,
    last_inference_date: Optional[str] = None,
    last_run_date: Optional[str] = None,
    win_rate_pct: str = "0.0%",
    overall_pnl_pct: float = 0.0,
    total_regimes: int = 0,
    long_regimes: int = 0,
    best_trade_pct: float = 0.0,
    worst_trade_pct: float = 0.0
) -> Dict[str, Any]:
    """Updates Section 2 (Global) inference results including directional accuracy and regime P&L."""
    current = get_inference_info()
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    today_str = now_dt.strftime("%Y-%m-%d")

    eff_inf_date = last_inference_date or current.get("last_inference_date") or today_str
    eff_run_date = last_run_date or today_str
    pruned = max(0, base_features_count - refined_features_count)

    global_info = {
        "status": status,
        "last_run_date": eff_run_date,
        "mae_gain_base": round(mae_gain_base, 3),
        "mae_drop_base": round(mae_drop_base, 3),
        "mae_gain_refined": round(mae_gain_refined, 3),
        "mae_drop_refined": round(mae_drop_refined, 3),
        "directional_accuracy": directional_accuracy,
        "win_rate_pct": win_rate_pct,
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "total_regimes": total_regimes,
        "long_regimes": long_regimes,
        "best_trade_pct": round(best_trade_pct, 2),
        "worst_trade_pct": round(worst_trade_pct, 2),
        "base_features_count": base_features_count,
        "refined_features_count": refined_features_count,
        "pruned_features_count": pruned,
        "duration_seconds": round(duration_seconds, 2),
        "message": message,
        "updated_at": now_str
    }
    current["global_inference"] = global_info
    current["last_inference_date"] = eff_inf_date
    current["last_inference_run_date"] = eff_run_date
    current["last_run_timestamp"] = now_str
    current["status"] = status
    current["message"] = message

    history = current.get("history", [])
    history.insert(0, {
        "timestamp": now_str,
        "section": "GLOBAL_INFERENCE",
        "status": status,
        "date": eff_inf_date,
        "run_date": eff_run_date,
        "mae_gain_refined": round(mae_gain_refined, 3),
        "mae_drop_refined": round(mae_drop_refined, 3),
        "directional_accuracy": directional_accuracy,
        "win_rate_pct": win_rate_pct,
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "duration_seconds": round(duration_seconds, 2),
        "message": message
    })
    current["history"] = history[:100]

    write_json_file(INFERENCE_STATE_FILE, current)
    notify_progress_listeners("INFERENCE_INFO", current)
    return current


# ==============================================================================
# TRADE LEDGERS: 50 MA STRATEGY & GLOBAL REGIME TRADES
# ==============================================================================

def save_inference_50ma_trades(trades: List[Dict[str, Any]], summary: Dict[str, Any]) -> bool:
    """Persists 50 MA test trades ledger with summary statistics atomically."""
    payload = {
        "strategy": "50_MA_SETUP",
        "updated_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "total_trades": len(trades),
        "trades": trades
    }
    return write_json_file(INFERENCE_50MA_TRADES_FILE, payload)


def get_inference_50ma_trades() -> Dict[str, Any]:
    """Returns the recorded 50 MA test trades ledger and summary."""
    default_data = {
        "strategy": "50_MA_SETUP",
        "summary": {},
        "total_trades": 0,
        "trades": []
    }
    return read_json_file(INFERENCE_50MA_TRADES_FILE, default_data)


def save_inference_global_trades(regimes: List[Dict[str, Any]], summary: Dict[str, Any]) -> bool:
    """Persists Global regime test trades ledger with summary statistics atomically."""
    payload = {
        "strategy": "GLOBAL_REGIME",
        "updated_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "total_regimes": len(regimes),
        "regimes": regimes
    }
    return write_json_file(INFERENCE_GLOBAL_TRADES_FILE, payload)


def get_inference_global_trades() -> Dict[str, Any]:
    """Returns the recorded Global regime test trades ledger and summary."""
    default_data = {
        "strategy": "GLOBAL_REGIME",
        "summary": {},
        "total_regimes": 0,
        "regimes": []
    }
    return read_json_file(INFERENCE_GLOBAL_TRADES_FILE, default_data)


def get_inference_progress() -> Dict[str, Any]:
    """Returns the current real-time inference progress."""
    default_progress = {
        "status": "COMPLETED",
        "active_section": "ALL",
        "phase": "Done",
        "heading": "Done",
        "current_model": "",
        "percentage": 100.0,
        "message": "Inference evaluation ready.",
        "elapsed_seconds": 0.0,
        "error": None,
        "metrics": {}
    }
    return read_json_file(INFERENCE_PROGRESS_FILE, default_progress)


def update_inference_progress(
    status: str,
    active_section: str = "ALL",
    phase: str = "Initializing",
    heading: str = "Evaluating",
    current_model: str = "",
    percentage: float = 0.0,
    message: str = "",
    elapsed_seconds: float = 0.0,
    error: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Updates transient real-time progress for UI polling and WebSockets."""
    data = {
        "status": status,
        "active_section": active_section,
        "phase": phase,
        "heading": heading,
        "current_model": current_model,
        "percentage": round(percentage, 1),
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
        "metrics": metrics or {},
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(INFERENCE_PROGRESS_FILE, data)
    notify_progress_listeners("INFERENCE_PROGRESS", data)


def get_model_performance_log() -> Dict[str, Any]:
    """Reads models/model_performance_log.json safely."""
    return read_json_file(MODEL_PERFORMANCE_LOG_FILE, {})


# ==============================================================================
# 11. DAILY MARKET ANALYSIS & TRADE FINALIZER STATE MANAGER
# ==============================================================================

def initialize_daily_analysis_state() -> Dict[str, Any]:
    """Ensures daily_analysis.json is initialized with structured trade recommendations."""
    default_state = {
        "status": "IDLE",
        "market_session_date": "2026-10-06",
        "market_status": "Market Closed",
        "last_run_timestamp": datetime.now().astimezone().isoformat(),
        "total_stocks_analyzed": 207,
        "total_bullish_candidates": 0,
        "finalized_trades_count": 0,
        "strategies_breakdown": {
            "50 MA Setup": 0,
            "Global Regime": 0,
            "50 MA + Global Regime": 0
        },
        "finalized_trades": [],
        "message": "Ready to run daily market analysis and trade finalizer."
    }
    if not DAILY_ANALYSIS_FILE.exists():
        write_json_file(DAILY_ANALYSIS_FILE, default_state)
        return default_state
    current = read_json_file(DAILY_ANALYSIS_FILE, default_state)
    needs_save = False
    if "finalized_trades" not in current:
        current["finalized_trades"] = []
        needs_save = True
    if not current.get("last_run_timestamp"):
        current["last_run_timestamp"] = datetime.now().astimezone().isoformat()
        needs_save = True
    if needs_save:
        write_json_file(DAILY_ANALYSIS_FILE, current)
    return current


def get_daily_analysis_info() -> Dict[str, Any]:
    """Retrieves current daily analysis metadata and finalized trades."""
    return initialize_daily_analysis_state()


def update_daily_analysis_info(
    status: str = "COMPLETED",
    market_session_date: str = "2026-10-06",
    market_status: str = "Market Closed",
    total_stocks_analyzed: int = 207,
    finalized_trades: Optional[List[Dict[str, Any]]] = None,
    strategies_breakdown: Optional[Dict[str, int]] = None,
    message: str = "",
    duration_seconds: float = 0.0
) -> Dict[str, Any]:
    """Atomically persists daily trade analysis results and notifies WebSocket listeners."""
    now_dt = datetime.now().astimezone()
    now_str = now_dt.isoformat()
    trades = finalized_trades or []
    breakdown = strategies_breakdown or {}

    data = {
        "status": status,
        "market_session_date": market_session_date,
        "market_status": market_status,
        "last_run_timestamp": now_str,
        "total_stocks_analyzed": total_stocks_analyzed,
        "total_bullish_candidates": len(trades),
        "finalized_trades_count": len(trades),
        "strategies_breakdown": breakdown,
        "finalized_trades": trades,
        "duration_seconds": round(duration_seconds, 2),
        "message": message,
        "updated_at": now_str
    }

    write_json_file(DAILY_ANALYSIS_FILE, data)
    notify_progress_listeners("DAILY_ANALYSIS_INFO", data)
    return data


def get_daily_analysis_progress() -> Dict[str, Any]:
    """Retrieves transient live progress of daily market analysis."""
    default_progress = {
        "status": "IDLE",            # IDLE, SYNCING, COMPUTING, PREDICTING, FINALIZING, COMPLETED, ERROR
        "heading": "Ready",
        "phase": "Ready",
        "current_stock": "",
        "current_index": 0,
        "total_stocks": 207,
        "percentage": 0.0,
        "message": "Ready to scan today's market for trade signals.",
        "elapsed_seconds": 0.0,
        "finalized_count": 0,
        "error": None
    }
    return read_json_file(DAILY_ANALYSIS_PROGRESS_FILE, default_progress)


def update_daily_analysis_progress(
    status: str,
    heading: str = "Analyzing Market",
    phase: str = "Scanning",
    current_stock: str = "",
    current_index: int = 0,
    total_stocks: int = 207,
    percentage: float = 0.0,
    message: str = "",
    elapsed_seconds: float = 0.0,
    finalized_count: int = 0,
    error: Optional[str] = None
) -> None:
    """Updates transient live analysis progress and broadcasts to UI."""
    data = {
        "status": status,
        "heading": heading,
        "phase": phase,
        "current_stock": current_stock,
        "current_index": current_index,
        "total_stocks": total_stocks,
        "percentage": round(percentage, 1),
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "finalized_count": finalized_count,
        "error": error,
        "updated_at": datetime.now().astimezone().isoformat()
    }
    write_json_file(DAILY_ANALYSIS_PROGRESS_FILE, data)
    notify_progress_listeners("DAILY_ANALYSIS_PROGRESS", data)


# Auto-initialize state on import
initialize_stocks_download_state()
initialize_feature_gen_state()
initialize_data_cleaning_state()
initialize_models_state()
initialize_inference_state()
initialize_daily_analysis_state()

