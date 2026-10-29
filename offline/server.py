"""
VectorTrade - Production Offline Flask Control Server & Frontend Hub
Serves the offline desktop UI on http://127.0.0.1:5000 and provides execution controls.
Ultra-lightweight: zero unnecessary dependencies, negligible CPU & battery consumption.
"""

from __future__ import annotations
import sys
import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List
from flask import Flask, render_template, jsonify, request
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from core.config import settings
from core.db import db_manager
from core.secure_comms import request_online_trade_details
from offline.scheduler import offline_scheduler
from offline.progress_tracker import progress_tracker

logger = logging.getLogger("VectorTrade.OfflineServer")

# Create Flask application
app = Flask(
    __name__,
    template_folder=str(OFFLINE_DIR / "templates"),
    static_folder=str(OFFLINE_DIR / "static")
)
app.config["SECRET_KEY"] = settings.INTERNAL_AUTH_TOKEN

# Global in-process task state trackers
PREDICT_STATUS: Dict[str, Any] = {
    "status": "Idle",
    "target": "",
    "start_time": 0.0,
    "elapsed_seconds": 0,
    "last_result": None,
    "error": None
}

DATA_STATUS: Dict[str, Any] = {
    "status": "Idle",
    "start_time": 0.0,
    "elapsed_seconds": 0,
    "error": None
}

RETRAIN_STATUS: Dict[str, str] = {
    "strategy": "Idle",
    "global": "Idle"
}


def get_live_predictions_file() -> Path:
    candidates = [
        BASE_DIR / "live_predictions.json",
        OFFLINE_DIR / "live_predictions.json",
        Path("live_predictions.json")
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE_DIR / "live_predictions.json"


def get_stock_names_file() -> Path:
    candidates = [
        BASE_DIR / "stock_names.json",
        OFFLINE_DIR / "stock_names.json",
        Path("stock_names.json")
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE_DIR / "stock_names.json"


def get_model_performance_file() -> Path:
    candidates = [
        BASE_DIR / "offline" / "models" / "model_performance_log.json",
        BASE_DIR / "models" / "model_performance_log.json",
        OFFLINE_DIR / "models" / "model_performance_log.json",
        Path("models/model_performance_log.json")
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE_DIR / "models" / "model_performance_log.json"


@app.route("/")
def index():
    """Renders the offline unified control center dashboard."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns real-time status of the offline scheduler, resource usage, and DB state."""
    status = offline_scheduler.get_system_status()
    return jsonify(status)


@app.route("/api/results", methods=["GET"])
def get_results():
    """Returns today's generated EOD signals and pre-calculated financial metrics."""
    tz = pytz.timezone(settings.TIMEZONE)
    from datetime import datetime
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    signals_col = db_manager.get_collection("signals_watchlist")
    query = {"valid_for_date": today_str}
    docs = []
    try:
        docs = list(signals_col.find(query).sort("_id", -1))
    except Exception:
        pass

    # If DB has 0 for today, check progress tracker state
    if not docs:
        snapshot = progress_tracker.get_snapshot()
        docs = list(snapshot.get("signals", []))

    # If still 0, check pending_dispatches locally
    if not docs:
        pending_file = OFFLINE_DIR / "pending_dispatches" / f"signals_{today_str}.json"
        if pending_file.exists():
            try:
                with open(pending_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    docs = cached.get("signals", [])
            except Exception:
                pass

    # Sanitize ObjectIds
    for d in docs:
        if "_id" in d:
            d["_id"] = str(d["_id"])

    return jsonify({
        "status": "SUCCESS",
        "date": today_str,
        "count": len(docs),
        "signals": docs
    })


@app.route("/api/analysis-status", methods=["GET"])
def get_analysis_status():
    """Returns real-time multi-stage progress for the stock analysis pipeline."""
    snap = progress_tracker.get_snapshot()
    return jsonify({
        "status": "SUCCESS",
        **snap
    })


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Returns all single-stock live predictions with target/SL/scenario figures."""
    pred_file = get_live_predictions_file()
    predictions = {}
    timestamp = "Never"
    if pred_file.exists():
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                predictions = data.get("predictions", {})
                raw_ts = data.get("timestamp", "")
                if raw_ts:
                    try:
                        from datetime import datetime
                        ts = datetime.fromisoformat(raw_ts)
                        timestamp = ts.strftime("%d %b %Y, %I:%M %p")
                    except Exception:
                        timestamp = raw_ts
        except Exception as e:
            logger.warning("Could not read live predictions: %s", str(e))

    return jsonify({
        "status": "SUCCESS",
        "timestamp": timestamp,
        "count": len(predictions),
        "predictions": predictions
    })


@app.route("/api/predict_status", methods=["GET"])
def get_predict_status():
    """Returns the live status and elapsed seconds of the stock predictor."""
    if PREDICT_STATUS["status"].startswith("Analyzing") and PREDICT_STATUS["start_time"] > 0:
        PREDICT_STATUS["elapsed_seconds"] = int(time.time() - PREDICT_STATUS["start_time"])
    return jsonify(PREDICT_STATUS)


@app.route("/api/start_predict/<symbol>", methods=["POST"])
def start_predict(symbol: str):
    """Triggers in-process prediction for a given stock symbol in a background thread."""
    clean_sym = symbol.strip().upper().replace(".NS", "") if symbol else "RELIANCE"
    if PREDICT_STATUS["status"].startswith("Analyzing"):
        return jsonify({
            "status": "ALREADY_RUNNING",
            "message": f"Already analyzing {PREDICT_STATUS['target']}."
        }), 409

    PREDICT_STATUS["status"] = f"Analyzing {clean_sym}..."
    PREDICT_STATUS["target"] = clean_sym
    PREDICT_STATUS["start_time"] = time.time()
    PREDICT_STATUS["elapsed_seconds"] = 0
    PREDICT_STATUS["error"] = None

    def _worker():
        try:
            import live_predictor_job
            res = live_predictor_job.run_predictor(symbol_arg=clean_sym)
            PREDICT_STATUS["status"] = "Completed"
            PREDICT_STATUS["elapsed_seconds"] = int(time.time() - PREDICT_STATUS["start_time"])
            PREDICT_STATUS["last_result"] = res.get(clean_sym) if isinstance(res, dict) else None
        except Exception as e:
            logger.error("Predict worker failed for %s: %s", clean_sym, str(e))
            PREDICT_STATUS["status"] = f"Error: {str(e)[:80]}"
            PREDICT_STATUS["error"] = str(e)
            PREDICT_STATUS["elapsed_seconds"] = int(time.time() - PREDICT_STATUS["start_time"])

    threading.Thread(target=_worker, daemon=True, name="PredictorWorker").start()
    return jsonify({"status": "STARTED", "target": clean_sym})


@app.route("/api/remove_prediction/<symbol>", methods=["POST"])
def remove_prediction(symbol: str):
    """Deletes a stock's entry from live_predictions.json."""
    clean_sym = symbol.strip().upper().replace(".NS", "")
    pred_file = get_live_predictions_file()
    if pred_file.exists():
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            preds = data.get("predictions", {})
            if clean_sym in preds:
                del preds[clean_sym]
                data["predictions"] = preds
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            return jsonify({"status": "ERROR", "message": str(e)}), 500
    return jsonify({"status": "REMOVED", "symbol": clean_sym})


@app.route("/api/clear_predictions", methods=["POST"])
def clear_predictions():
    """Resets live_predictions.json to empty."""
    pred_file = get_live_predictions_file()
    try:
        from datetime import datetime
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "predictions": {}}, f, indent=2)
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500
    return jsonify({"status": "CLEARED"})


@app.route("/api/stock_list", methods=["GET"])
def get_stock_list():
    """Returns the list of stocks available for autocomplete."""
    names_file = get_stock_names_file()
    stock_meta = {}
    if names_file.exists():
        try:
            with open(names_file, "r", encoding="utf-8") as f:
                stock_meta = json.load(f)
        except Exception:
            pass

    stock_list = []
    if stock_meta:
        for s, info in stock_meta.items():
            stock_list.append({
                "symbol": s,
                "name": info.get("name", s),
                "sector": info.get("sector", "NSE")
            })
    else:
        default_syms = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "TATAMOTORS", "TRENT"]
        stock_list = [{"symbol": s, "name": s, "sector": "NSE"} for s in default_syms]

    return jsonify({
        "status": "SUCCESS",
        "count": len(stock_list),
        "stocks": stock_list
    })


@app.route("/api/model_performance", methods=["GET"])
def get_model_performance():
    """Returns real metrics from model_performance_log.json."""
    perf_file = get_model_performance_file()
    log_data = {}
    if perf_file.exists():
        try:
            with open(perf_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception:
            pass
    return jsonify({
        "status": "SUCCESS",
        "models": log_data
    })


@app.route("/api/dataset_snapshot", methods=["GET"])
def get_dataset_snapshot():
    """Returns dataset snapshot statistics."""
    meta_file = BASE_DIR / "dataset_metadata.json"
    last_updated = "Unknown"
    total_stocks = 207
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                last_updated = meta_data.get("last_updated", meta_data.get("global_last_date", "Unknown"))
                total_stocks = len(meta_data.get("symbols", {})) or 207
        except Exception:
            pass

    csv_file = BASE_DIR / "5_year_final_data.csv"
    final_csv_size = "Unknown"
    if csv_file.exists():
        try:
            size_bytes = csv_file.stat().st_size
            final_csv_size = f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            pass

    return jsonify({
        "status": "SUCCESS",
        "total_stocks": total_stocks,
        "last_updated": last_updated,
        "final_csv_size": final_csv_size
    })


@app.route("/api/start_data", methods=["POST"])
def start_data():
    """Triggers background data update job."""
    if DATA_STATUS["status"].startswith("Running"):
        return jsonify({"status": "ALREADY_RUNNING", "message": "Data job is already running."}), 409

    DATA_STATUS["status"] = "Running... (Syncing Missing Market Days)"
    DATA_STATUS["start_time"] = time.time()
    DATA_STATUS["elapsed_seconds"] = 0
    DATA_STATUS["error"] = None

    def _data_worker():
        try:
            import subprocess
            proc = subprocess.run([sys.executable, str(BASE_DIR / "5_years_6.0_data_update_job.py")],
                                  capture_output=True, text=True, timeout=900)
            if proc.returncode == 0:
                DATA_STATUS["status"] = "Completed"
            else:
                DATA_STATUS["status"] = f"Error: {proc.stderr[:100]}"
                DATA_STATUS["error"] = proc.stderr
        except Exception as e:
            DATA_STATUS["status"] = f"Error: {str(e)[:80]}"
            DATA_STATUS["error"] = str(e)
        finally:
            DATA_STATUS["elapsed_seconds"] = int(time.time() - DATA_STATUS["start_time"])

    threading.Thread(target=_data_worker, daemon=True, name="DataSyncWorker").start()
    return jsonify({"status": "STARTED"})


@app.route("/api/data_status", methods=["GET"])
def get_data_status():
    """Returns status of the dataset update job."""
    if DATA_STATUS["status"].startswith("Running") and DATA_STATUS["start_time"] > 0:
        DATA_STATUS["elapsed_seconds"] = int(time.time() - DATA_STATUS["start_time"])
    return jsonify(DATA_STATUS)


@app.route("/api/start_update/<task_key>", methods=["POST"])
def start_update(task_key: str):
    """Triggers retraining for strategy or global model."""
    if RETRAIN_STATUS.get(task_key, "").startswith("Running"):
        return jsonify({"status": "ALREADY_RUNNING"}), 409

    RETRAIN_STATUS[task_key] = "Running (Monthly Retrain Engine)..."

    def _retrain_worker():
        try:
            res = offline_scheduler.retrain_models(force=True)
            if res.get("status") == "SUCCESS":
                RETRAIN_STATUS[task_key] = "Completed"
            else:
                RETRAIN_STATUS[task_key] = f"Done: {res.get('message', 'Completed')[:60]}"
        except Exception as e:
            RETRAIN_STATUS[task_key] = f"Error: {str(e)[:80]}"

    threading.Thread(target=_retrain_worker, daemon=True, name=f"Retrain-{task_key}").start()
    return jsonify({"status": "STARTED", "task": task_key})


@app.route("/api/task_status", methods=["GET"])
def get_all_task_status():
    """Returns unified status across all background tasks for the universal progress banner."""
    if PREDICT_STATUS["status"].startswith("Analyzing") and PREDICT_STATUS["start_time"] > 0:
        PREDICT_STATUS["elapsed_seconds"] = int(time.time() - PREDICT_STATUS["start_time"])
    if DATA_STATUS["status"].startswith("Running") and DATA_STATUS["start_time"] > 0:
        DATA_STATUS["elapsed_seconds"] = int(time.time() - DATA_STATUS["start_time"])

    return jsonify({
        "predict": PREDICT_STATUS,
        "analysis": progress_tracker.get_snapshot(),
        "data": DATA_STATUS,
        "retrain": RETRAIN_STATUS,
        "scheduler": offline_scheduler.get_system_status()
    })


@app.route("/api/telemetry", methods=["GET"])
def get_online_telemetry():
    """Fetches today's live execution telemetry securely from Online Engine."""
    try:
        telemetry = request_online_trade_details("GET_TODAY_EXECUTION")
        return jsonify(telemetry)
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Could not fetch online telemetry: {str(e)}"
        }), 500


@app.route("/api/pending", methods=["GET"])
def get_pending():
    """Returns local pending dispatches awaiting sync to MongoDB Atlas."""
    pending_dir = OFFLINE_DIR / "pending_dispatches"
    files = list(pending_dir.glob("signals_*.json")) if pending_dir.exists() else []

    pending_list = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                pending_list.append({
                    "filename": f.name,
                    "signals_count": len(data.get("signals", [])),
                    "buffer_count": len(data.get("buffer", []))
                })
        except Exception:
            pass

    return jsonify({
        "status": "SUCCESS",
        "pending_files": pending_list,
        "total_files": len(pending_list)
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns past EOD execution runs from MongoDB system_audit."""
    try:
        audit_col = db_manager.get_collection("system_audit")
        records = list(audit_col.find({"component": "HEADLESS_OFFLINE_DAEMON"}).sort("timestamp", -1).limit(30))
        for r in records:
            r["_id"] = str(r["_id"])
        return jsonify({"status": "SUCCESS", "history": records})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent log records."""
    log_file = settings.LOGS_DIR / "daemon_runs.log"
    lines = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()
                lines = [l.strip() for l in all_lines[-100:]]
        except Exception:
            pass
    return jsonify({"status": "SUCCESS", "lines": lines})


@app.route("/api/action/run-analysis", methods=["POST"])
@app.route("/api/action/run-eod", methods=["POST"])
def trigger_run_eod():
    """Triggers immediate multi-stage EOD stock analysis in background."""
    snapshot = progress_tracker.get_snapshot()
    if snapshot.get("is_running") or offline_scheduler.state["is_busy"]:
        return jsonify({"status": "BLOCKED", "message": "An analysis task is already running in background."}), 409

    def _worker():
        from offline.headless_daemon import HeadlessDaemon
        try:
            daemon = HeadlessDaemon(force_run=True)
            daemon.run()
        except Exception as e:
            logger.error("Analysis background worker failed: %s", str(e))
            progress_tracker.fail_analysis(str(e))

    threading.Thread(target=_worker, daemon=True, name="EodAnalysisWorker").start()
    return jsonify({"status": "STARTED", "message": "Multi-stage stock analysis triggered in background."})


@app.route("/api/action/sync-dispatches", methods=["POST"])
def trigger_sync():
    """Synchronizes local pending dispatches to MongoDB Atlas."""
    res = offline_scheduler.sync_dispatches()
    return jsonify(res)


@app.route("/api/action/retrain", methods=["POST"])
def trigger_retrain():
    """Triggers monthly model retraining."""
    if offline_scheduler.state["is_busy"]:
        return jsonify({"status": "BLOCKED", "message": "A task is already running in background."}), 409

    threading.Thread(target=offline_scheduler.retrain_models, kwargs={"force": True}, daemon=True, name="RetrainWorker").start()
    return jsonify({"status": "STARTED", "message": "Monthly Model Retrainer triggered in background."})


def run_server(host: str = "127.0.0.1", port: int = 5000):
    """Runs the Flask web server in production-ready mode."""
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    offline_scheduler.start()
    run_server()
