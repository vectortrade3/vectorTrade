"""
VectorTrade - Production Online Dashboard & State API Server
Lightweight FastAPI server running concurrently with the market execution worker.
Delivers real-time state streams and fail-safe alerts to templates/dashboard.html.
Constraints: Zero heavy ML libraries (no numpy, pandas, scikit-learn, or xgboost).
"""

from __future__ import annotations
import sys
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
ONLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ONLINE_DIR))

from core.config import settings
from core.db import db_manager
from online.megabull_client import MegaBullClient
from online.execution_worker import execution_worker
from core.secure_comms import verify_auth_token, TelemetryCompiler

app = FastAPI(title="VectorTrade Online Execution Hub", version="1.0.0")

# Setup template engine
templates = Jinja2Templates(directory=str(ONLINE_DIR / "templates"))

# Mount static files if present
static_dir = ONLINE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared MegaBull client instance for fast dashboard price queries
mb_client = MegaBullClient()


@app.on_event("startup")
def startup_event():
    """Starts the execution worker polling loop in a resilient background thread."""
    worker_thread = threading.Thread(target=execution_worker.start_polling, daemon=True)
    worker_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    execution_worker.stop()
    mb_client.close()


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Renders real-time quantitative dashboard."""
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/api/state", response_class=JSONResponse)
async def get_system_state():
    """
    Returns full state payload from MongoDB with real-time fail-safe alerts:
    - signals_watchlist (ARMED & INVALIDATED setups)
    - active_trades (OPEN positions with current live LTP and unrealized PnL)
    - trade_history (Realized closed trades with audit metrics)
    - system_status (Live alerts, broker auth validity, stale data notice)
    """
    signals_col = db_manager.get_collection("signals_watchlist")
    active_col = db_manager.get_collection("active_trades")
    history_col = db_manager.get_collection("trade_history")

    signals = list(signals_col.find({"status": {"$in": ["ARMED", "TRIGGERED", "INVALIDATED"]}}).sort("_id", -1).limit(50))
    active_trades = list(active_col.find({"status": "OPEN"}))
    history = list(history_col.find({}).sort("_id", -1).limit(50))

    # Enrich active trades with live LTP
    for t in active_trades:
        token = str(t.get("instrument_token", "")).strip()
        ltp = mb_client.get_ltp(token) if token else None
        t["current_ltp"] = ltp if ltp is not None else float(t.get("fill_price", 0.0))

    # Clean ObjectIds for JSON serialization
    for doc_list in [signals, active_trades, history]:
        for d in doc_list:
            if "_id" in d:
                d["_id"] = str(d["_id"])

    return {
        "status": "OK",
        "signals_watchlist": signals,
        "active_trades": active_trades,
        "trade_history": history,
        "system_status": execution_worker.system_status
    }


@app.get("/api/internal/trade-details")
@app.post("/api/internal/trade-details")
async def get_internal_trade_details(request: Request):
    """
    Secure Inter-Process Endpoint for Offline Daemon requests.
    Zero Public Access: Rejects direct browser URL visits with 401 Unauthorized.
    Requires pre-shared cryptographic header: 'X-Vector-Internal-Token'.
    """
    token = request.headers.get("X-Vector-Internal-Token") or request.headers.get("x-vector-internal-token")
    if not token:
        # Check Bearer format
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not verify_auth_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Missing or invalid security token. Direct browser access is strictly forbidden."
        )

    # Extract action and params based on method
    action = "GET_TODAY_EXECUTION"
    params: Dict[str, Any] = {}
    if request.method == "POST":
        try:
            body = await request.json()
            action = body.get("action", "GET_TODAY_EXECUTION")
            params = body.get("params", {})
        except Exception:
            pass
    else:
        action = request.query_params.get("action", "GET_TODAY_EXECUTION")
        limit = request.query_params.get("limit")
        if limit:
            params["limit"] = int(limit)

    return TelemetryCompiler.compile(action, params, system_status=execution_worker.system_status)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("online.app:app", host="0.0.0.0", port=8000, reload=False)
