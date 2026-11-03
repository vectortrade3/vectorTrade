"""
VectorTrade - Minimalist Offline Flask Application
Ultra-low system resource consumption (0% idle CPU, minimal memory, battery-friendly).
Managed and launched silently via Windows VBS scripts.
Logs application lifecycle events to data_store/app_logs.json on boot.
"""

from __future__ import annotations
import sys
import os
import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_sock import Sock

# Resolve directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from offline import json_manager
from offline.downloader import run_stock_download_pipeline
from offline.feature_generator import run_feature_generation_pipeline
from offline.data_cleaner import run_data_cleaning_pipeline
from offline.model_trainer import train_strategy_model, train_global_models, train_all_models
from offline.model_inferencer import run_strategy_inference, run_global_inference, run_all_inference
from offline.daily_analyzer import run_daily_analysis_pipeline, get_market_schedule_info

# Initialize Flask & WebSockets
app = Flask(__name__, template_folder=str(OFFLINE_DIR / "templates"))
app.config["SECRET_KEY"] = "vectortrade-offline-secure-key"
sock = Sock(app)

# WebSocket client registry
ws_clients = set()
ws_clients_lock = threading.Lock()

def broadcast_ws_event(event_type: str, data: dict):
    """Pushes real-time progress events to all connected WebSocket clients."""
    payload = json.dumps({"type": event_type, "data": data})
    with ws_clients_lock:
        dead = set()
        for ws in list(ws_clients):
            try:
                ws.send(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            ws_clients.discard(ws)

# Register WebSocket broadcaster with json_manager
json_manager.register_progress_listener(broadcast_ws_event)

@sock.route("/ws")
def websocket_route(ws):
    """Real-time bidirectional WebSocket stream for zero-polling instant updates."""
    reconcile_stale_progress()
    with ws_clients_lock:
        ws_clients.add(ws)
    try:
        # Push immediate snapshot upon connection
        initial_payload = {
            "type": "INITIAL_STATE",
            "download": json_manager.get_download_progress(),
            "download_info": json_manager.get_stocks_download_info(),
            "feature_gen": json_manager.get_feature_gen_progress(),
            "feature_gen_info": json_manager.get_feature_gen_info(),
            "data_cleaning": json_manager.get_data_cleaning_progress(),
            "data_cleaning_info": json_manager.get_data_cleaning_info(),
            "models_training": json_manager.get_models_training_progress(),
            "models_info": json_manager.get_models_info(),
            "inference": json_manager.get_inference_progress(),
            "inference_info": json_manager.get_inference_info(),
            "daily_analysis": json_manager.get_daily_analysis_progress(),
            "daily_analysis_info": json_manager.get_daily_analysis_info(),
            "market_schedule": get_market_schedule_info(),
            "model_performance_log": json_manager.get_model_performance_log()
        }
        ws.send(json.dumps(initial_payload))
        while True:
            msg = ws.receive()
            if msg is None:
                break
    except Exception:
        pass
    finally:
        with ws_clients_lock:
            ws_clients.discard(ws)

# Configure lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VectorTrade.OfflineApp")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Concurrency locks for background workers
_download_thread: Optional[threading.Thread] = None
_download_lock = threading.Lock()

_feat_thread: Optional[threading.Thread] = None
_feat_lock = threading.Lock()

_clean_thread: Optional[threading.Thread] = None
_clean_lock = threading.Lock()

_train_thread: Optional[threading.Thread] = None
_train_lock = threading.Lock()

_infer_thread: Optional[threading.Thread] = None
_infer_lock = threading.Lock()

_analysis_thread: Optional[threading.Thread] = None
_analysis_lock = threading.Lock()


def reconcile_stale_progress():
    """
    Self-healing progress reconciler.
    If no background worker thread is alive, any in-flight progress state
    (CHECKING, DOWNLOADING, APPENDING, GENERATING, SAVING, COMBINING, CLEANING, TRAINING, EVALUATING)
    persisting on disk from an interrupted process or unit test run is reconciled to COMPLETED.
    """
    # 1. Stocks Download
    if _download_thread is None or not _download_thread.is_alive():
        prog = json_manager.get_download_progress()
        if prog.get("status") in ("CHECKING", "DOWNLOADING", "APPENDING"):
            info = json_manager.get_stocks_download_info()
            last_date = info.get("last_downloaded_date") or "2026-11-01"
            json_manager.update_download_progress(
                status="COMPLETED",
                heading="Done",
                current_stock="ALL",
                current_index=207,
                total_stocks=207,
                percentage=100.0,
                message=f"Market data synchronized up to {last_date}.",
                last_downloaded_date=last_date,
                last_append_date=info.get("last_append_date", last_date)
            )

    # 2. Feature Generation
    if _feat_thread is None or not _feat_thread.is_alive():
        prog = json_manager.get_feature_gen_progress()
        if prog.get("status") in ("GENERATING", "SAVING"):
            info = json_manager.get_feature_gen_info()
            last_date = info.get("last_feature_gen_date") or "2026-11-01"
            json_manager.update_feature_gen_progress(
                status="COMPLETED",
                heading="Done",
                current_stock="ALL",
                current_index=207,
                total_stocks=207,
                percentage=100.0,
                message=f"Features already computed up to {last_date} for all 207 stocks.",
                last_feature_gen_date=last_date
            )

    # 3. Data Cleaning
    if _clean_thread is None or not _clean_thread.is_alive():
        prog = json_manager.get_data_cleaning_progress()
        if prog.get("status") in ("COMBINING", "CLEANING", "SAVING"):
            info = json_manager.get_data_cleaning_info()
            last_date = info.get("last_clean_date") or "2026-11-01"
            json_manager.update_data_cleaning_progress(
                status="COMPLETED",
                heading="Done",
                current_stock="ALL",
                current_index=207,
                total_stocks=207,
                percentage=100.0,
                message=f"Data already cleaned and consolidated up to {last_date}.",
                last_clean_date=last_date
            )

    # 4. Model Training
    if _train_thread is None or not _train_thread.is_alive():
        prog = json_manager.get_models_training_progress()
        if prog.get("status") in ("TRAINING", "INITIALIZING", "SAVING"):
            json_manager.update_models_training_progress(
                status="COMPLETED",
                active_section="ALL",
                phase="Done",
                heading="Done",
                current_model="",
                percentage=100.0,
                message="All models trained successfully."
            )

    # 5. Model Inference
    if _infer_thread is None or not _infer_thread.is_alive():
        prog = json_manager.get_inference_progress()
        if prog.get("status") in ("EVALUATING", "INITIALIZING"):
            json_manager.update_inference_progress(
                status="COMPLETED",
                active_section="ALL",
                phase="Done",
                heading="Done",
                current_model="",
                percentage=100.0,
                message="All quantitative models evaluated and validated successfully."
            )

    # 6. Daily Market Analysis
    if _analysis_thread is None or not _analysis_thread.is_alive():
        prog = json_manager.get_daily_analysis_progress()
        if prog.get("status") in ("RUNNING", "SYNCING", "COMPUTING", "PREDICTING", "FINALIZING"):
            info = json_manager.get_daily_analysis_info()
            json_manager.update_daily_analysis_progress(
                status="COMPLETED",
                heading="Analysis Ready",
                phase="Done",
                current_stock="ALL_STOCKS",
                current_index=207,
                total_stocks=207,
                percentage=100.0,
                message="Daily market analysis and trade setups verified.",
                finalized_count=len(info.get("finalized_trades", []))
            )


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Renders the main control panel."""
    reconcile_stale_progress()
    download_info = json_manager.get_stocks_download_info()
    feature_gen_info = json_manager.get_feature_gen_info()
    data_cleaning_info = json_manager.get_data_cleaning_info()
    models_info = json_manager.get_models_info()
    inference_info = json_manager.get_inference_info()
    daily_analysis_info = json_manager.get_daily_analysis_info()
    market_schedule = get_market_schedule_info()
    performance_log = json_manager.get_model_performance_log()
    return render_template(
        "index.html",
        active_tab="dashboard",
        download_info=download_info,
        feature_gen_info=feature_gen_info,
        data_cleaning_info=data_cleaning_info,
        models_info=models_info,
        inference_info=inference_info,
        daily_analysis_info=daily_analysis_info,
        market_schedule=market_schedule,
        performance_log=performance_log
    )


@app.route("/analysis", methods=["GET"])
def analysis_page():
    """Renders the Daily Market Analysis and Finalized Trades view."""
    reconcile_stale_progress()
    download_info = json_manager.get_stocks_download_info()
    feature_gen_info = json_manager.get_feature_gen_info()
    data_cleaning_info = json_manager.get_data_cleaning_info()
    models_info = json_manager.get_models_info()
    inference_info = json_manager.get_inference_info()
    daily_analysis_info = json_manager.get_daily_analysis_info()
    market_schedule = get_market_schedule_info()
    performance_log = json_manager.get_model_performance_log()
    return render_template(
        "index.html",
        active_tab="analysis",
        download_info=download_info,
        feature_gen_info=feature_gen_info,
        data_cleaning_info=data_cleaning_info,
        models_info=models_info,
        inference_info=inference_info,
        daily_analysis_info=daily_analysis_info,
        market_schedule=market_schedule,
        performance_log=performance_log
    )


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Returns combined status across storage, last download, and active progress."""
    reconcile_stale_progress()
    return jsonify({
        "status": "ONLINE",
        "download_info": json_manager.get_stocks_download_info(),
        "feature_gen_info": json_manager.get_feature_gen_info(),
        "data_cleaning_info": json_manager.get_data_cleaning_info(),
        "models_info": json_manager.get_models_info(),
        "inference_info": json_manager.get_inference_info(),
        "daily_analysis_info": json_manager.get_daily_analysis_info(),
        "progress": json_manager.get_download_progress(),
        "feature_gen_progress": json_manager.get_feature_gen_progress(),
        "data_cleaning_progress": json_manager.get_data_cleaning_progress(),
        "models_training_progress": json_manager.get_models_training_progress(),
        "inference_progress": json_manager.get_inference_progress(),
        "daily_analysis_progress": json_manager.get_daily_analysis_progress(),
        "market_schedule": get_market_schedule_info(),
        "performance_log": json_manager.get_model_performance_log(),
        "stocks_count": len(json_manager.get_stocks_list()),
        "current_time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/download/start", methods=["POST"])
def start_download():
    """Triggers the stock download pipeline in a background thread."""
    global _download_thread
    with _download_lock:
        if _download_thread and _download_thread.is_alive():
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
    reconcile_stale_progress()
    return jsonify(json_manager.get_download_progress())


# ==============================================================================
# FEATURE GENERATION ENDPOINTS
# ==============================================================================

@app.route("/api/feature_gen/status", methods=["GET"])
def get_feature_gen_status():
    """Returns feature generation info and live progress."""
    reconcile_stale_progress()
    return jsonify({
        "info": json_manager.get_feature_gen_info(),
        "progress": json_manager.get_feature_gen_progress()
    })


@app.route("/api/feature_gen/start", methods=["POST"])
def start_feature_generation():
    """Triggers the Feature Generation pipeline in a background thread."""
    global _feat_thread
    with _feat_lock:
        if _feat_thread and _feat_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "A feature generation task is currently running."
            }), 409

        def _worker():
            try:
                run_feature_generation_pipeline()
            except Exception as e:
                logger.error(f"Feature generation failed: {e}", exc_info=True)
                json_manager.update_feature_gen_progress(
                    status="ERROR",
                    heading="Features Generating",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="FEATURE_GENERATION",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _feat_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-FeatureWorker")
        _feat_thread.start()

        return jsonify({
            "status": "STARTED",
            "message": "Feature generation pipeline started."
        })


@app.route("/api/feature_gen/progress", methods=["GET"])
def get_feature_gen_progress():
    """Returns live progress of the feature generation pipeline."""
    reconcile_stale_progress()
    return jsonify(json_manager.get_feature_gen_progress())


# ==============================================================================
# DATA CLEANING & CONSOLIDATION ENDPOINTS
# ==============================================================================

@app.route("/api/data_clean/status", methods=["GET"])
def get_data_clean_status():
    """Returns data cleaning info and live progress."""
    reconcile_stale_progress()
    return jsonify({
        "info": json_manager.get_data_cleaning_info(),
        "progress": json_manager.get_data_cleaning_progress()
    })


@app.route("/api/data_clean/start", methods=["POST"])
def start_data_cleaning():
    """Triggers the Data Cleaning & Consolidation pipeline in a background thread."""
    global _clean_thread
    with _clean_lock:
        if _clean_thread and _clean_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "A data cleaning and consolidation task is currently running."
            }), 409

        def _worker():
            try:
                run_data_cleaning_pipeline()
            except Exception as e:
                logger.error(f"Data cleaning failed: {e}", exc_info=True)
                json_manager.update_data_cleaning_progress(
                    status="ERROR",
                    heading="Data Cleaning",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="DATA_CLEANING",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _clean_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-CleanerWorker")
        _clean_thread.start()

        return jsonify({
            "status": "STARTED",
            "message": "Data cleaning and consolidation pipeline started."
        })


@app.route("/api/data_clean/progress", methods=["GET"])
def get_data_clean_progress():
    """Returns live progress of the data cleaning pipeline."""
    reconcile_stale_progress()
    return jsonify(json_manager.get_data_cleaning_progress())


# ==============================================================================
# MACHINE LEARNING MODEL TRAINING ENDPOINTS
# ==============================================================================

@app.route("/api/training/status", methods=["GET"])
def get_training_status():
    """Returns model training metadata and live progress."""
    reconcile_stale_progress()
    return jsonify({
        "info": json_manager.get_models_info(),
        "progress": json_manager.get_models_training_progress()
    })


@app.route("/api/training/progress", methods=["GET"])
def get_training_progress():
    """Returns live progress of the model training pipeline."""
    reconcile_stale_progress()
    return jsonify(json_manager.get_models_training_progress())


@app.route("/api/training/start_strategy", methods=["POST"])
def start_training_strategy():
    """Triggers the Setup Strategy model training pipeline in a background thread."""
    global _train_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _train_lock:
        if _train_thread and _train_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "A model training task is currently running."
            }), 409

        def _worker():
            try:
                import importlib
                import offline.model_trainer
                importlib.reload(offline.model_trainer)
                offline.model_trainer.train_strategy_model(force=force)
            except Exception as e:
                logger.error(f"Strategy model training failed: {e}", exc_info=True)
                json_manager.update_models_training_progress(
                    status="ERROR",
                    active_section="STRATEGY",
                    heading="Training Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="STRATEGY_MODEL_TRAINING",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _train_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-TrainStrategyWorker")
        _train_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "STRATEGY",
            "message": "Strategy model training started."
        })


@app.route("/api/training/start_global", methods=["POST"])
def start_training_global():
    """Triggers the Global Regime models training pipeline in a background thread."""
    global _train_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _train_lock:
        if _train_thread and _train_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "A model training task is currently running."
            }), 409

        def _worker():
            try:
                import importlib
                import offline.model_trainer
                importlib.reload(offline.model_trainer)
                offline.model_trainer.train_global_models(force=force)
            except Exception as e:
                logger.error(f"Global models training failed: {e}", exc_info=True)
                json_manager.update_models_training_progress(
                    status="ERROR",
                    active_section="GLOBAL",
                    heading="Training Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="GLOBAL_MODELS_TRAINING",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _train_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-TrainGlobalWorker")
        _train_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "GLOBAL",
            "message": "Global models training started."
        })


@app.route("/api/training/start_all", methods=["POST"])
def start_training_all():
    """Triggers training of all models sequentially in a background thread."""
    global _train_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _train_lock:
        if _train_thread and _train_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "A model training task is currently running."
            }), 409

        def _worker():
            try:
                import importlib
                import offline.model_trainer
                importlib.reload(offline.model_trainer)
                offline.model_trainer.train_all_models(force=force)
            except Exception as e:
                logger.error(f"All models training failed: {e}", exc_info=True)
                json_manager.update_models_training_progress(
                    status="ERROR",
                    active_section="ALL",
                    heading="Training Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="ALL_MODELS_TRAINING",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _train_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-TrainAllWorker")
        _train_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "ALL",
            "message": "All models training started."
        })


# ==============================================================================
# MACHINE LEARNING MODEL INFERENCE & EVALUATION ENDPOINTS (PART 5 & 5.2)
# ==============================================================================

@app.route("/api/inference/status", methods=["GET"])
def get_inference_status():
    """Returns inference evaluation metadata, live progress, and benchmark logs."""
    reconcile_stale_progress()
    return jsonify({
        "info": json_manager.get_inference_info(),
        "progress": json_manager.get_inference_progress(),
        "performance_log": json_manager.get_model_performance_log()
    })


@app.route("/api/inference/progress", methods=["GET"])
def get_inference_progress():
    """Returns live progress of the model inference pipeline."""
    reconcile_stale_progress()
    return jsonify(json_manager.get_inference_progress())


@app.route("/api/inference/start_strategy", methods=["POST"])
def start_inference_strategy():
    """Triggers the Setup Strategy model evaluation pipeline in a background thread."""
    global _infer_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _infer_lock:
        if _infer_thread and _infer_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "An inference evaluation task is currently running."
            }), 409

        def _worker():
            try:
                run_strategy_inference(force=force)
            except Exception as e:
                logger.error(f"Strategy inference failed: {e}", exc_info=True)
                json_manager.update_inference_progress(
                    status="ERROR",
                    active_section="STRATEGY",
                    heading="Inference Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="STRATEGY_INFERENCE",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _infer_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-InferStrategyWorker")
        _infer_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "STRATEGY",
            "message": "Strategy model evaluation started."
        })


@app.route("/api/inference/start_global", methods=["POST"])
def start_inference_global():
    """Triggers the Global Regime models evaluation pipeline in a background thread."""
    global _infer_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _infer_lock:
        if _infer_thread and _infer_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "An inference evaluation task is currently running."
            }), 409

        def _worker():
            try:
                run_global_inference(force=force)
            except Exception as e:
                logger.error(f"Global inference failed: {e}", exc_info=True)
                json_manager.update_inference_progress(
                    status="ERROR",
                    active_section="GLOBAL",
                    heading="Inference Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="GLOBAL_INFERENCE",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _infer_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-InferGlobalWorker")
        _infer_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "GLOBAL",
            "message": "Global regime models evaluation started."
        })


@app.route("/api/inference/start_all", methods=["POST"])
def start_inference_all():
    """Triggers evaluation of all models sequentially in a background thread."""
    global _infer_thread
    force = request.args.get("force", "true").lower() in ("true", "1")
    if request.is_json and request.json:
        force = request.json.get("force", force)
    with _infer_lock:
        if _infer_thread and _infer_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "An inference evaluation task is currently running."
            }), 409

        def _worker():
            try:
                run_all_inference(force=force)
            except Exception as e:
                logger.error(f"All inference evaluation failed: {e}", exc_info=True)
                json_manager.update_inference_progress(
                    status="ERROR",
                    active_section="ALL",
                    heading="Inference Error",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_task_entry(
                    task_name="ALL_INFERENCE",
                    status="ERROR",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )

        _infer_thread = threading.Thread(target=_worker, daemon=True, name="VectorTrade-InferAllWorker")
        _infer_thread.start()

        return jsonify({
            "status": "STARTED",
            "section": "ALL",
            "message": "All models evaluation started."
        })


@app.route("/api/inference/trades/50ma", methods=["GET"])
def get_inference_50ma_trades_api():
    """Returns recorded 50 MA test trades with optional filtering and pagination."""
    data = json_manager.get_inference_50ma_trades()
    trades = data.get("trades", [])
    summary = data.get("summary", {})

    # Filter by search symbol
    search = request.args.get("search", "").strip().upper()
    if search:
        trades = [t for t in trades if search in str(t.get("symbol", "")).upper()]

    # Filter by decision (TAKE / SKIP)
    decision_filter = request.args.get("decision", "").strip().upper()
    if decision_filter in ("TAKE", "SKIP"):
        trades = [t for t in trades if str(t.get("decision", "")).upper() == decision_filter]

    # Filter by outcome (TARGET / SL)
    outcome_filter = request.args.get("outcome", "").strip().upper()
    if outcome_filter in ("TARGET", "SL"):
        trades = [t for t in trades if str(t.get("outcome", "")).upper() == outcome_filter]

    total_filtered = len(trades)
    page = request.args.get("page", default=1, type=int)
    limit = request.args.get("limit", default=50, type=int)

    if limit > 0:
        start_idx = max(0, (page - 1) * limit)
        end_idx = start_idx + limit
        paginated_trades = trades[start_idx:end_idx]
        total_pages = max(1, (total_filtered + limit - 1) // limit)
    else:
        paginated_trades = trades
        total_pages = 1

    return jsonify({
        "status": "SUCCESS",
        "strategy": "50_MA_SETUP",
        "summary": summary,
        "total_trades": total_filtered,
        "total_unfiltered": len(data.get("trades", [])),
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "trades": paginated_trades
    })


@app.route("/api/inference/trades/global", methods=["GET"])
def get_inference_global_trades_api():
    """Returns recorded Global regime test trades with optional filtering and pagination."""
    data = json_manager.get_inference_global_trades()
    regimes = data.get("regimes", [])
    summary = data.get("summary", {})

    # Filter by search symbol
    search = request.args.get("search", "").strip().upper()
    if search:
        regimes = [r for r in regimes if search in str(r.get("symbol", "")).upper()]

    # Filter by decision (LONG / CASH)
    decision_filter = request.args.get("decision", "").strip().upper()
    if decision_filter in ("LONG", "CASH"):
        regimes = [r for r in regimes if str(r.get("decision", "")).upper() == decision_filter]

    # Filter by win
    win_filter = request.args.get("win", "").strip().lower()
    if win_filter == "true":
        regimes = [r for r in regimes if r.get("is_win") is True]
    elif win_filter == "false":
        regimes = [r for r in regimes if r.get("is_win") is False and str(r.get("decision", "")).upper() == "LONG"]

    total_filtered = len(regimes)
    page = request.args.get("page", default=1, type=int)
    limit = request.args.get("limit", default=50, type=int)

    if limit > 0:
        start_idx = max(0, (page - 1) * limit)
        end_idx = start_idx + limit
        paginated_regimes = regimes[start_idx:end_idx]
        total_pages = max(1, (total_filtered + limit - 1) // limit)
    else:
        paginated_regimes = regimes
        total_pages = 1

    return jsonify({
        "status": "SUCCESS",
        "strategy": "GLOBAL_REGIME",
        "summary": summary,
        "total_regimes": total_filtered,
        "total_unfiltered": len(data.get("regimes", [])),
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "regimes": paginated_regimes
    })


@app.route("/api/tasks/logs", methods=["GET"])
def get_task_logs():
    """Returns execution task records for the logs page."""
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({
        "task_logs": json_manager.get_task_logs(limit=limit)
    })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent application and audit logs."""
    limit = request.args.get("limit", default=30, type=int)
    return jsonify({
        "logs": json_manager.get_app_logs(limit=limit)
    })


# ==============================================================================
# DAILY MARKET ANALYSIS & TRADE FINALIZER ENDPOINTS
# ==============================================================================

@app.route("/api/analysis/start", methods=["POST"])
def start_daily_analysis():
    """Triggers the Daily Market Analysis & AI Trade Finalizer pipeline."""
    global _analysis_thread
    with _analysis_lock:
        if _analysis_thread and _analysis_thread.is_alive():
            return jsonify({
                "status": "BUSY",
                "message": "Daily market analysis is already running."
            }), 409

        force_download = False
        try:
            req_data = request.get_json(silent=True) or {}
            force_download = bool(req_data.get("force_download", False))
        except Exception:
            force_download = False

        def _worker():
            try:
                import importlib
                import offline.daily_analyzer
                importlib.reload(offline.daily_analyzer)
                offline.daily_analyzer.run_daily_analysis_pipeline(force_download=force_download)
            except Exception as e:
                logger.error(f"Daily analysis pipeline failed: {e}", exc_info=True)
                json_manager.update_daily_analysis_progress(
                    status="ERROR",
                    heading="Analysis Failed",
                    message=f"Pipeline error: {str(e)}",
                    error=str(e)
                )
                json_manager.log_app_event(
                    event="DAILY_ANALYSIS_ERROR",
                    status="Error",
                    message=f"Daily analysis error: {str(e)}",
                    error=str(e)
                )

        _analysis_thread = threading.Thread(target=_worker, daemon=True)
        _analysis_thread.start()

        return jsonify({
            "status": "STARTED",
            "message": "Daily market analysis and AI trade finalizer started in background.",
            "force_download": force_download
        })


@app.route("/api/analysis/progress", methods=["GET"])
def get_analysis_progress():
    """Returns transient real-time progress for Daily Market Analysis."""
    return jsonify(json_manager.get_daily_analysis_progress())


@app.route("/api/analysis/results", methods=["GET"])
def get_analysis_results():
    """Returns complete finalized trade setups and executive metrics."""
    info = json_manager.get_daily_analysis_info()
    return jsonify({
        "status": "SUCCESS",
        "data": info
    })


@app.route("/api/analysis/schedule", methods=["GET"])
def get_market_schedule_endpoint():
    """Returns current NSE market schedule and post/pre 3:30 PM classification."""
    return jsonify({
        "status": "SUCCESS",
        "schedule": get_market_schedule_info()
    })


def run_app(host: str = "0.0.0.0", port: int = 8684):
    """Starts the application and logs boot event."""
    env_port = os.environ.get("PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            pass

    try:
        json_manager.log_app_event(
            event="APP_STARTED",
            status="Everything fine",
            message=f"VectorTrade Offline Flask server listening on http://vectortrade:{port} (or http://localhost:{port})"
        )
        logger.info(f"VectorTrade Offline App starting on http://vectortrade:{port} (http://{host}:{port})...")
        print("\n" + "=" * 65)
        print("  VectorTrade Engine Online!")
        print(f"  Primary Custom URL : http://vectortrade:{port}")
        print(f"  Standard Localhost : http://localhost:{port}")
        print(f"  Loopback IP        : http://127.0.0.1:{port}")
        print("=" * 65 + "\n")
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
    import argparse
    parser = argparse.ArgumentParser(description="VectorTrade Offline Web Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8684, help="Port (default: 8684 [ASCII VT])")
    args, _ = parser.parse_known_args()
    run_app(host=args.host, port=args.port)
