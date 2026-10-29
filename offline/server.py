"""
VectorTrade - Production Offline Flask Control Server & Frontend Hub
Serves the offline desktop UI on http://127.0.0.1:5000 and provides execution controls.
Ultra-lightweight: zero unnecessary dependencies, negligible CPU & battery consumption.
"""

from __future__ import annotations
import sys
import os
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


@app.route("/")
def index():
    """Renders the offline control center dashboard."""
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

    threading.Thread(target=_worker, daemon=True).start()
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

    threading.Thread(target=offline_scheduler.retrain_models, kwargs={"force": True}, daemon=True).start()
    return jsonify({"status": "STARTED", "message": "Monthly Model Retrainer triggered in background."})


def run_server(host: str = "127.0.0.1", port: int = 5000):
    """Runs the Flask web server in production-ready mode."""
    # Suppress verbose default werkzeug logger to save IO
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    offline_scheduler.start()
    run_server()
