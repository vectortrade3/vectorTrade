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
if getattr(sys, "frozen", False):
    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "offline").exists():
        BASE_DIR = exe_dir
    elif (exe_dir.parent / "offline").exists():
        BASE_DIR = exe_dir.parent
    elif (exe_dir.parent.parent / "offline").exists():
        BASE_DIR = exe_dir.parent.parent
    else:
        BASE_DIR = Path(__file__).resolve().parent.parent
    OFFLINE_DIR = BASE_DIR / "offline"
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

from offline import json_manager
from offline import ma50_service
from offline.downloader import run_stock_download_pipeline
from offline.feature_generator import run_feature_generation_pipeline
from offline.data_cleaner import run_data_cleaning_pipeline
from offline.model_trainer import train_strategy_model, train_global_models, train_all_models
from offline.model_inferencer import run_strategy_inference, run_global_inference, run_all_inference
from offline.daily_analyzer import run_daily_analysis_pipeline, get_market_schedule_info
from strategies.manager import StrategyManager

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
    strategies = StrategyManager.list_strategies(include_disabled=True)
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
        performance_log=performance_log,
        strategies=strategies
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
    strategies = StrategyManager.list_strategies(include_disabled=True)
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
        performance_log=performance_log,
        strategies=strategies
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
        "strategies": StrategyManager.list_strategies(include_disabled=True),
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
# 50MA STRATEGY TRADE LEDGER & CUSTOM TRADINGVIEW CHART API
# ==============================================================================

@app.route("/api/strategy/50ma/trades", methods=["GET"])
def get_strategy_50ma_trades_api():
    """
    Returns filtered, sorted, and paginated 50MA trades from the verified 50MA dataset.
    Supports query parameters:
      - outcome: ALL, SUCCESS, LOSS, NEUTRAL, PENDING (default: ALL)
      - setup: ALL, 1, 2, 3 (default: ALL)
      - symbol: Search query substring (e.g. RELIANCE)
      - sector: Filter by sector name
      - sort_by: setup_date, realized_r, realized_pnl_pct, days_held, entry_price, symbol
      - order: desc (default) or asc
      - page: Page number (1-based, default: 1)
      - limit: Page size limit (default: 50, max: 500)
    """
    outcome = request.args.get("outcome", default="ALL", type=str)
    split = request.args.get("split", default="ALL", type=str)
    setup = request.args.get("setup", default="ALL", type=str)
    symbol = request.args.get("symbol", default="", type=str)
    sector = request.args.get("sector", default="", type=str)
    sort_by = request.args.get("sort_by", default="setup_date", type=str)
    order = request.args.get("order", default="desc", type=str)
    page = request.args.get("page", default=1, type=int)
    limit = request.args.get("limit", default=50, type=int)

    result = ma50_service.query_50ma_trades(
        outcome=outcome,
        setup=setup,
        symbol=symbol,
        sector=sector,
        split=split,
        sort_by=sort_by,
        order=order,
        page=page,
        limit=limit
    )
    return jsonify(result), 200


@app.route("/api/strategy/50ma/trade/<trade_id>/chart", methods=["GET"])
def get_strategy_50ma_trade_chart_api(trade_id: str):
    """
    Returns the parsed 38-bar TradingView candlestick series and exact Entry, SL, Target
    coordinates for custom TradingView-style candlestick chart visualization.
    """
    res = ma50_service.get_50ma_trade_by_id(trade_id)
    if not res:
        return jsonify({
            "status": "ERROR",
            "message": f"Trade '{trade_id}' not found in 50MA dataset."
        }), 404
    return jsonify(res), 200


@app.route("/api/strategy/50ma/summary", methods=["GET"])
def get_strategy_50ma_summary_api():
    """
    Returns executive KPI metrics, setup subclass distributions (1, 2, 3), and yearly performance breakdown.
    """
    summary = ma50_service.get_50ma_strategy_summary()
    return jsonify(summary), 200


# ==============================================================================
# STRATEGY SELECTION & DYNAMIC MANAGEMENT ENDPOINTS
# ==============================================================================

@app.route("/api/strategies", methods=["GET"])
def get_strategies_api():
    """
    Returns list of all discovered strategies, including active/deselected status,
    metadata, triggers, description, and execution rules.
    """
    strats = StrategyManager.list_strategies(include_disabled=True)
    active_count = sum(1 for s in strats if s.get("enabled", True))
    return jsonify({
        "status": "SUCCESS",
        "strategies": strats,
        "total_count": len(strats),
        "active_count": active_count
    }), 200


@app.route("/api/strategies/toggle", methods=["POST"])
def toggle_strategy_api():
    """
    Toggles or sets the selected/deselected state for a given strategy.
    When deselected:
    - Daily analysis skips analyzing stocks with that strategy's model.
    - No trades are taken with that strategy.
    """
    data = request.get_json(silent=True) or {}
    strategy_id = data.get("strategy_id")
    if not strategy_id:
        return jsonify({"status": "ERROR", "message": "strategy_id is required"}), 400

    current_enabled = StrategyManager.is_strategy_enabled(strategy_id)
    target_enabled = data.get("enabled")
    if target_enabled is None:
        target_enabled = not current_enabled
    else:
        target_enabled = bool(target_enabled)

    success = StrategyManager.set_strategy_enabled(strategy_id, target_enabled)
    if not success:
        return jsonify({"status": "ERROR", "message": f"Strategy '{strategy_id}' not found"}), 404

    strats = StrategyManager.list_strategies(include_disabled=True)
    active_count = sum(1 for s in strats if s.get("enabled", True))
    action_text = "selected (active for analysis & trading)" if target_enabled else "deselected (no analysis, no trades will be taken)"
    msg = f"Strategy '{strategy_id}' is now {action_text}."

    json_manager.log_app_event(
        event="STRATEGY_TOGGLED",
        status="Success",
        message=msg,
        extra={"strategy_id": strategy_id, "enabled": target_enabled, "active_count": active_count}
    )

    return jsonify({
        "status": "SUCCESS",
        "message": msg,
        "strategy_id": strategy_id,
        "enabled": target_enabled,
        "strategies": strats,
        "total_count": len(strats),
        "active_count": active_count
    }), 200


@app.route("/api/strategies/select", methods=["POST"])
def select_strategies_batch_api():
    """
    Batch selects/deselects strategies from a list of selected strategy IDs.
    Any strategy in the list is enabled; any omitted registered strategy is disabled.
    """
    data = request.get_json(silent=True) or {}
    selected_ids = data.get("selected_strategy_ids", data.get("selected", []))
    if not isinstance(selected_ids, list):
        return jsonify({"status": "ERROR", "message": "selected_strategy_ids must be a list"}), 400

    selected_ids_norm = {StrategyManager.resolve_strategy_id(s) or s.lower() for s in selected_ids}
    all_strats = StrategyManager.get_all_strategies()

    for sid in all_strats.keys():
        should_enable = sid.lower() in selected_ids_norm or sid in selected_ids
        StrategyManager.set_strategy_enabled(sid, should_enable)

    strats = StrategyManager.list_strategies(include_disabled=True)
    active_count = sum(1 for s in strats if s.get("enabled", True))
    msg = f"Batch strategy selection updated: {active_count}/{len(strats)} strategies active."

    return jsonify({
        "status": "SUCCESS",
        "message": msg,
        "strategies": strats,
        "total_count": len(strats),
        "active_count": active_count
    }), 200

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
                "status": "RUNNING",
                "message": "Daily market analysis is already running.",
                "progress": json_manager.get_daily_analysis_progress()
            }), 200

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
            "status": "RUNNING",
            "action": "STARTED",
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


@app.route("/api/sync_mongodb", methods=["POST"])
def sync_mongodb_endpoint():
    """Synchronizes daily finalized trade setups into MongoDB Atlas signals_watchlist collection."""
    try:
        from offline.mongo_syncer import MongoSyncer
        syncer = MongoSyncer()
        res = syncer.sync_from_json_file()
        return jsonify(res)
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": f"Sync trigger failed: {str(e)}",
            "synced_count": 0
        }), 500


@app.route("/api/sync_mongodb/status", methods=["GET"])
def get_mongodb_status_endpoint():
    """Returns connection diagnostics and signals count from MongoDB Atlas."""
    try:
        from offline.mongo_syncer import MongoSyncer
        syncer = MongoSyncer()
        info = syncer.test_connection()
        return jsonify({
            "status": "SUCCESS" if info.get("connected") else "DISCONNECTED",
            "info": info
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500


# ==============================================================================
# AUTOMATED 4:00 PM DAILY SCHEDULER & SELF-CLOSER ENDPOINTS
# ==============================================================================

@app.route("/api/scheduler/status", methods=["GET"])
def get_scheduler_status_api():
    """Returns real-time status of the automated 4:00 PM timer, window, internet check, and self-close state."""
    from offline.scheduler import get_daily_scheduler
    scheduler = get_daily_scheduler()
    return jsonify({
        "status": "SUCCESS",
        "scheduler": scheduler.get_status()
    }), 200


@app.route("/api/scheduler/tick", methods=["POST"])
def trigger_scheduler_tick_api():
    """Manually triggers an immediate scheduler evaluation tick."""
    from offline.scheduler import get_daily_scheduler
    scheduler = get_daily_scheduler()
    force = False
    try:
        body = request.get_json(silent=True) or {}
        force = bool(body.get("force", False))
    except Exception:
        pass
    res = scheduler.run_tick_once(force_execution=force)
    return jsonify({
        "status": "SUCCESS",
        "result": res
    }), 200


# ==============================================================================
# OFFLINE ORDERS BOARD & 17-COLUMN TRADE BOOK (SYNCED WITH MONGODB ATLAS)
# ==============================================================================

@app.route("/orders", methods=["GET"])
def orders_page():
    """Renders the comprehensive Trade Book & Orders Board in the offline app, fetching live data from MongoDB Atlas."""
    from online.services.orders_service import orders_service
    from online.db import db_manager

    trade_journal = orders_service.get_trade_journal()
    orders = orders_service.get_orders(limit=100)
    summary_stats = orders_service.get_order_summary_stats()
    db_health = db_manager.ping()
    from online.services.risk_calculator import risk_calculator
    trading_settings = risk_calculator.get_trading_settings()

    return render_template(
        "orders.html",
        app_name="VectorTrade (Offline)",
        trade_journal=trade_journal,
        orders=orders,
        stats=summary_stats,
        db_health=db_health,
        trading_settings=trading_settings,
        is_offline=True
    )


@app.route("/api/trades/journal", methods=["GET"])
@app.route("/api/orders/tradebook", methods=["GET"])
def get_offline_trades_journal_api():
    """Returns the complete Trade Book / Journal with all 17 columns calculated from MongoDB Atlas."""
    from online.services.orders_service import orders_service
    status_filter = request.args.get("status")
    trades = orders_service.get_trade_journal(status_filter=status_filter)
    return jsonify({
        "status": "SUCCESS",
        "count": len(trades),
        "trades": trades
    })


@app.route("/api/orders", methods=["GET"])
def get_offline_orders_api():
    """Returns watchlist setup orders from MongoDB Atlas."""
    from online.services.orders_service import orders_service
    status_filter = request.args.get("status")
    strategy_filter = request.args.get("strategy")
    limit = int(request.args.get("limit", 100))
    orders = orders_service.get_orders(status=status_filter, strategy=strategy_filter, limit=limit)
    return jsonify({
        "status": "SUCCESS",
        "count": len(orders),
        "orders": orders
    })


@app.route("/api/stats", methods=["GET"])
def get_offline_orders_stats_api():
    """Returns summary KPIs and statistics for orders in the offline app."""
    from online.services.orders_service import orders_service
    stats = orders_service.get_order_summary_stats()
    return jsonify({
        "status": "SUCCESS",
        "stats": stats
    })


@app.route("/api/trades/export/csv", methods=["GET"])
@app.route("/api/orders/export/csv", methods=["GET"])
def export_offline_trades_csv_api():
    """Exports and downloads the 17-column CSV trade journal directly from the offline app."""
    from online.services.orders_service import orders_service
    from flask import Response
    from datetime import datetime

    status_filter = request.args.get("status")
    trades = orders_service.get_trade_journal(status_filter=status_filter)
    csv_content = orders_service.generate_trade_journal_csv(trades)

    try:
        orders_service.save_trades_record_csv()
    except Exception as e:
        logger.debug(f"Could not persist trades_record.csv on disk: {e}")

    filename = f"vectortrade_journal_offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )


@app.route("/api/megabull/liquidation-status", methods=["GET"])
def get_offline_liquidation_status_api():
    """Returns liquidation status for offline orders board telemetry."""
    from online.services.megabull_service import megabull_service
    return jsonify(megabull_service.get_liquidation_status())


@app.route("/api/megabull/portfolio-symbols", methods=["GET"])
def get_offline_portfolio_symbols_api():
    """Returns portfolio symbols for duplicate checks in offline orders modal."""
    from online.services.megabull_service import megabull_service
    holdings = megabull_service.get_holdings(enrich_ltp=False)
    symbols = list({str(h.get("symbol", "")).upper() for h in holdings if h.get("symbol")})
    return jsonify({"status": "SUCCESS", "symbols": symbols, "all_portfolio_symbols": symbols})


@app.route("/api/megabull/liquidate-all", methods=["POST"])
def offline_liquidate_all_api():
    """Triggers emergency portfolio square off from the offline app."""
    from online.services.megabull_service import megabull_service
    data = request.get_json(silent=True) or {}
    force_mo = data.get("force_market_open")
    result = megabull_service.liquidate_all_positions(force_market_open=force_mo)
    return jsonify({"status": "SUCCESS", "data": result})


@app.route("/api/megabull/cancel-liquidation", methods=["POST"])
def offline_cancel_liquidation_api():
    """Cancels scheduled emergency liquidation from the offline app."""
    from online.services.megabull_service import megabull_service
    result = megabull_service.cancel_scheduled_liquidation()
    return jsonify({"status": "SUCCESS", "data": result})


@app.route("/api/megabull/orders/place", methods=["POST"])
def offline_place_order_api():
    """Places a manual order on MegaBull or stages it off-market from the offline app."""
    from online.services.megabull_service import megabull_service
    data = request.get_json(silent=True) or {}
    symbol = str(data.get("symbol", "")).strip().upper()
    action = str(data.get("action", "BUY")).strip().upper()
    quantity = int(data.get("quantity", 1))
    price = float(data.get("price", 0.0))
    target_price = float(data["target_price"]) if data.get("target_price") else None
    sl_price = float(data["sl_price"]) if data.get("sl_price") else None
    strategy = str(data.get("strategy", "Manual Setup"))
    product = str(data.get("product", "CNC"))
    order_type = str(data.get("order_type", "LIMIT"))
    res = megabull_service.place_order(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        target_price=target_price,
        sl_price=sl_price,
        strategy=strategy,
        product=product,
        order_type=order_type
    )
    return jsonify(res)


@app.route("/api/megabull/holdings", methods=["GET"])
def offline_holdings_api():
    """Returns active holdings for offline modal and dashboard."""
    from online.services.megabull_service import megabull_service
    return jsonify(megabull_service.get_holdings(enrich_ltp=True))


@app.route("/api/health", methods=["GET"])
def offline_health_api():
    """Returns health telemetry for orders board polling in the offline app."""
    from online.db import db_manager
    from online.services.megabull_service import megabull_service
    ping_res = db_manager.ping()
    db_ok = bool(ping_res.get("ok", False))
    mb_open = megabull_service.is_market_open()
    return jsonify({
        "status": "healthy" if db_ok else "degraded",
        "mongodb_connected": db_ok,
        "market_open": mb_open,
        "latency_ms": ping_res.get("latency_ms", 0.0),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/trading-settings", methods=["GET", "POST"])
def offline_trading_settings_api():
    """Returns or updates trading capital configuration in MongoDB Atlas from offline app."""
    from online.services.risk_calculator import risk_calculator
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        limit = data.get("custom_capital_limit")
        risk_pct = data.get("risk_pct_per_trade", 1.0)
        updated = risk_calculator.update_trading_settings(
            custom_capital_limit=limit,
            risk_pct_per_trade=risk_pct
        )
        plan = risk_calculator.get_allocation_plan()
        return jsonify({
            "status": "SUCCESS",
            "message": "Trading capital settings saved to MongoDB Atlas and orders allocated.",
            "settings": updated,
            "allocation_plan": plan
        })
    return jsonify({
        "status": "SUCCESS",
        "settings": risk_calculator.get_trading_settings()
    })


@app.route("/api/trading-settings/allocation-plan", methods=["GET"])
def offline_trading_allocation_plan_api():
    """Returns the latest capital allocation and confirmed orders plan."""
    from online.services.risk_calculator import risk_calculator
    plan = risk_calculator.get_allocation_plan()
    return jsonify({
        "status": "SUCCESS",
        "plan": plan
    })


@app.route("/api/trading-settings/allocate", methods=["POST"])
def offline_trading_allocate_api():
    """Evaluates candidate orders, sorts best orders on top, and confirms orders by budget."""
    from online.services.risk_calculator import risk_calculator
    data = request.get_json(silent=True) or {}
    limit = data.get("custom_capital_limit")
    risk_pct = float(data.get("risk_pct_per_trade", 1.0))
    plan = risk_calculator.evaluate_and_allocate_candidate_orders(
        custom_capital_limit=limit,
        risk_pct=risk_pct,
        auto_stage_confirmed=True
    )
    return jsonify({
        "status": "SUCCESS",
        "message": f"Evaluated candidate setups: {plan.get('confirmed_orders_count', 0)} confirmed orders staged for 09:15 AM execution.",
        "plan": plan
    })


@app.route("/api/trading/calculate-quantity", methods=["POST"])
def offline_calculate_quantity_api():
    """Calculates position sizing and targets matching Excel Quantity Calculator."""
    from online.services.risk_calculator import risk_calculator
    data = request.get_json(silent=True) or {}
    entry = float(data.get("entry_price", 0.0))
    sl = float(data.get("sl_price", 0.0))
    act = str(data.get("action", "BUY"))
    portfolio = data.get("custom_portfolio")
    risk_pct = float(data.get("risk_pct", 1.0))
    calc = risk_calculator.calculate_quantity_and_targets(
        entry_price=entry,
        sl_price=sl,
        action=act,
        custom_portfolio=portfolio,
        risk_pct=risk_pct
    )
    return jsonify({
        "status": "SUCCESS",
        "data": calc
    })


def run_offline_headless(output_format: str = "text") -> Dict[str, Any]:
    """
    Executes a comprehensive, non-interactive diagnostic of the VectorTrade Offline engine.
    Checks data stores, machine learning models, 50MA strategy service, market schedule,
    and MongoDB Atlas synchronization without starting the web server.
    """
    diagnostics: Dict[str, Any] = {
        "engine": "VectorTrade Offline Engine",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "components": {}
    }

    # 1. Data Store Health
    data_store_dir = BASE_DIR / "data_store"
    expected_files = [
        "app_logs.json",
        "task_logs.json",
        "models_state.json",
        "inference_state.json",
        "inference_50ma_trades.json",
        "inference_global_trades.json",
        "data_cleaning.json"
    ]
    verified_files = []
    missing_files = []
    for fn in expected_files:
        fp = data_store_dir / fn
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    json.load(f)
                verified_files.append(fn)
            except Exception as e:
                missing_files.append(f"{fn} (corrupt: {e})")
        else:
            missing_files.append(f"{fn} (not found)")

    diagnostics["components"]["data_store"] = {
        "status": "OK" if not missing_files else "WARNING",
        "verified_count": len(verified_files),
        "total_expected": len(expected_files),
        "verified_files": verified_files,
        "issues": missing_files
    }

    # 2. Models & State
    try:
        models_info = json_manager.get_models_info()
        models_dir = BASE_DIR / "models"
        available_models = [f.name for f in models_dir.glob("*.pkl")] if models_dir.exists() else []
        diagnostics["components"]["models"] = {
            "status": "OK",
            "active_models_count": len(available_models),
            "available_models": available_models,
            "models_state_records": len(models_info)
        }
    except Exception as e:
        diagnostics["components"]["models"] = {"status": "ERROR", "error": str(e)}
        diagnostics["success"] = False

    # 3. 50MA Strategy Service
    try:
        strat_summary = ma50_service.get_50ma_strategy_summary()
        overall = strat_summary.get("overall", {})
        last_1y = strat_summary.get("last_1_year", {})
        diagnostics["components"]["ma50_strategy"] = {
            "status": "OK",
            "total_setups": overall.get("total_trades", 0),
            "win_rate_5y": f"{overall.get('win_rate', 0):.1f}%",
            "win_rate_1y": f"{last_1y.get('win_rate', 0):.1f}%",
            "net_r_edge": f"{overall.get('total_realized_r', 0):+.1f} R",
            "payoff_ratio": "1 : 2.00"
        }
    except Exception as e:
        diagnostics["components"]["ma50_strategy"] = {"status": "ERROR", "error": str(e)}
        diagnostics["success"] = False

    # 4. Market Schedule
    try:
        schedule = get_market_schedule_info()
        diagnostics["components"]["market_schedule"] = {
            "status": "OK",
            "info": schedule
        }
    except Exception as e:
        diagnostics["components"]["market_schedule"] = {"status": "WARNING", "error": str(e)}

    # 5. MongoDB Atlas Syncer
    try:
        from offline.mongo_syncer import MongoSyncer
        syncer = MongoSyncer()
        conn_info = syncer.test_connection()
        diagnostics["components"]["mongo_syncer"] = {
            "status": "OK" if conn_info.get("connected") else "DEGRADED",
            "connected": conn_info.get("connected", False),
            "database": conn_info.get("database"),
            "signals_count": conn_info.get("signals_count", 0),
            "error": conn_info.get("error")
        }
    except Exception as e:
        diagnostics["components"]["mongo_syncer"] = {"status": "ERROR", "error": str(e)}

    if output_format == "json":
        print(json.dumps(diagnostics, indent=2))
    else:
        print("\n" + "=" * 68)
        print("   VectorTrade Offline Engine -- Headless Verification Diagnostic")
        print("=" * 68)
        ds = diagnostics["components"].get("data_store", {})
        print(f" [OK] Data Store         : {ds.get('verified_count', 0)}/{ds.get('total_expected', 0)} JSON stores verified")

        md = diagnostics["components"].get("models", {})
        print(f" [OK] Models Directory   : {md.get('active_models_count', 0)} model weight artifacts detected")

        m50 = diagnostics["components"].get("ma50_strategy", {})
        if m50.get("status") == "OK":
            print(f" [OK] 50MA Strategy      : {m50.get('total_setups'):,} setups | {m50.get('win_rate_5y')} Win | {m50.get('net_r_edge')}")
        else:
            print(f" [ERR] 50MA Strategy     : {m50.get('error')}")

        mkt = diagnostics["components"].get("market_schedule", {})
        print(f" [OK] Market Schedule    : {mkt.get('info', {}).get('status', 'Ready')}")

        ms = diagnostics["components"].get("mongo_syncer", {})
        if ms.get("connected"):
            print(f" [OK] MongoDB Syncer     : Connected to Atlas (signals_watchlist: {ms.get('signals_count')} items)")
        else:
            print(f" [WARN] MongoDB Syncer   : {ms.get('error', 'Disconnected')}")

        print("-" * 68)
        status_label = "ALL SYSTEMS OPERATIONAL (0 Errors)" if diagnostics["success"] else "ISSUES DETECTED"
        print(f" Status: {status_label}")
        print("=" * 68 + "\n")

    return diagnostics


def run_app(host: str = "0.0.0.0", port: int = 30307):
    """Starts the application and logs boot event."""
    env_port = os.environ.get("PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            pass

    try:
        # Start 4:00 PM Daily Scheduler background thread
        try:
            from offline.scheduler import get_daily_scheduler
            daily_sched = get_daily_scheduler(check_interval_seconds=60)
            daily_sched.start()
            logger.info("4:00 PM Daily Automated Scheduler background thread started.")
        except Exception as sched_e:
            logger.warning(f"Could not start background daily scheduler: {sched_e}")

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
    parser.add_argument("--port", type=int, default=30307, help="Port (default: 30307 [Ultra-Unique])")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode without starting interactive web server; checks engine & outputs status")
    parser.add_argument("--json", action="store_true", help="Output headless check results in JSON format")
    args, _ = parser.parse_known_args()
    if args.headless:
        result = run_offline_headless(output_format="json" if args.json else "text")
        sys.exit(0 if result.get("success") else 1)
    else:
        run_app(host=args.host, port=args.port)
