"""
VectorTrade - Minimalist Offline Flask Application
Ultra-low system resource consumption (0% idle CPU, minimal memory, battery-friendly).
Managed and launched silently via Windows VBS scripts.
Logs application lifecycle events to data_store/app_logs.json on boot.
"""

from __future__ import annotations
import sys
import os
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Resolve directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from offline import json_manager
from offline.downloader import run_stock_download_pipeline

# Initialize Flask
app = Flask(__name__, template_folder=str(OFFLINE_DIR / "templates"))
app.config["SECRET_KEY"] = "vectortrade-offline-secure-key"

# Configure lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VectorTrade.OfflineApp")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Concurrency lock for background downloader
_download_thread: Optional[threading.Thread] = None
_download_lock = threading.Lock()


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Renders the main control panel."""
    download_info = json_manager.get_stocks_download_info()
    models_info = json_manager.get_models_info()
    return render_template(
        "index.html",
        download_info=download_info,
        models_info=models_info
    )


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Returns combined status across storage, last download, and active progress."""
    return jsonify({
        "status": "ONLINE",
        "download_info": json_manager.get_stocks_download_info(),
        "progress": json_manager.get_download_progress(),
        "models_info": json_manager.get_models_info(),
        "stocks_count": len(json_manager.get_stocks_list()),
        "current_time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/download/start", methods=["POST"])
def start_download():
    """Triggers the stock download pipeline in a background thread."""
    global _download_thread
    with _download_lock:
        curr_progress = json_manager.get_download_progress()
        if curr_progress.get("status") in ["CHECKING", "DOWNLOADING", "APPENDING"]:
            return jsonify({
                "status": "BUSY",
                "message": "A download pipeline is already running."
            }), 409

        def _worker():
            try:
                run_stock_download_pipeline()
            except Exception as e:
                logger.error(f"Download pipeline failed: {e}", exc_info=True)
                json_manager.update_download_progress(
                    status="ERROR",
                    heading="DOWNLOADING STOCKS",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_app_event(
                    event="STOCKS_DOWNLOAD_ERROR",
                    status="Error",
                    message=f"Pipeline crashed: {str(e)}",
                    error=str(e)
                )

        _download_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-DownloaderWorker")
        _download_thread.start()
        
        json_manager.log_app_event(
            event="DOWNLOAD_TRIGGERED",
            status="Everything fine",
            message="User initiated market data synchronization."
        )

        return jsonify({
            "status": "STARTED",
            "message": "Market data download pipeline started."
        })


@app.route("/api/download/progress", methods=["GET"])
def get_progress():
    """Returns the live download progress for UI polling."""
    return jsonify(json_manager.get_download_progress())


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent application and audit logs."""
    limit = request.args.get("limit", default=30, type=int)
    return jsonify({
        "logs": json_manager.get_app_logs(limit=limit)
    })


def run_app(host: str = "127.0.0.1", port: int = 5000):
    """Starts the application and logs boot event."""
    try:
        json_manager.log_app_event(
            event="APP_STARTED",
            status="Everything fine",
            message=f"VectorTrade Offline Flask server listening on http://{host}:{port}"
        )
        logger.info(f"VectorTrade Offline App starting on http://{host}:{port}...")
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        json_manager.log_app_event(
            event="APP_CRASH",
            status="Error",
            message=f"Server startup failed: {str(e)}",
            error=str(e)
        )
        raise


if __name__ == "__main__":
    run_app()
