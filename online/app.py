"""
VectorTrade - Production Online Dashboard & State API Server
Lightweight FastAPI server running concurrently with the market execution worker.
Delivers real-time state streams and fail-safe alerts to templates/dashboard.html.
Constraints: Zero heavy ML libraries (no numpy, pandas, scikit-learn, or xgboost).
"""

from __future__ import annotations
import sys
import os
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import FastAPI, Request, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
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

# Active WebSocket connections registry
active_websockets: Set[WebSocket] = set()


def compile_full_system_state() -> Dict[str, Any]:
    """Compiles complete real-time snapshot of watchlist, open trades, history, and system status."""
    signals_col = db_manager.get_collection("signals_watchlist")
    active_col = db_manager.get_collection("active_trades")
    history_col = db_manager.get_collection("trade_history")

    signals = list(signals_col.find({"status": {"$in": ["ARMED", "TRIGGERED", "INVALIDATED", "CANCELLED"]}}).sort("_id", -1).limit(50))
    active_trades = list(active_col.find({"status": "OPEN"}))
    history = list(history_col.find({}).sort("_id", -1).limit(50))

    # Enrich active trades with instantaneous cached LTP (Zero latency, no blocking HTTP calls)
    for t in active_trades:
        cached = execution_worker.active_trades.get(str(t.get("_id")), {})
        ltp = cached.get("highest_ltp") or t.get("highest_ltp") or t.get("fill_price")
        t["current_ltp"] = float(ltp) if ltp is not None else float(t.get("fill_price", 0.0))

    # Clean ObjectIds for JSON serialization
    for doc_list in [signals, active_trades, history]:
        for d in doc_list:
            if "_id" in d:
                d["_id"] = str(d["_id"])

    return jsonable_encoder({
        "status": "OK",
        "signals_watchlist": signals,
        "active_trades": active_trades,
        "trade_history": history,
        "system_status": execution_worker.system_status
    })


async def broadcast_live_state() -> None:
    """Broadcasts current snapshot to all active WebSocket clients."""
    if not active_websockets:
        return
    state = compile_full_system_state()
    dead = set()
    for ws in list(active_websockets):
        try:
            await ws.send_json(state)
        except Exception:
            dead.add(ws)
    for d in dead:
        active_websockets.discard(d)


async def periodic_websocket_broadcaster() -> None:
    """Pushes live stream updates to connected dashboard WebSockets every 2.0 seconds."""
    while True:
        await asyncio.sleep(2.0)
        try:
            if os.environ.get("TESTING") == "1":
                continue
            await broadcast_live_state()
        except Exception:
            pass


@app.on_event("startup")
def startup_event():
    """Starts the execution worker and background WebSocket broadcaster."""
    if os.environ.get("TESTING") == "1":
        return
    worker_thread = threading.Thread(target=execution_worker.start_polling, daemon=True)
    worker_thread.start()
    asyncio.create_task(periodic_websocket_broadcaster())


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
    """Returns full state payload from MongoDB with real-time fail-safe alerts."""
    return compile_full_system_state()


@app.post("/api/order/cancel")
async def cancel_order_endpoint(request: Request):
    """
    Cancels an armed signal or emergency exits an active open position.
    Updates MongoDB Atlas immediately and broadcasts to connected WebSocket clients.
    """
    try:
        data = await request.json()
        item_id = str(data.get("id", "")).strip()
        item_type = str(data.get("type", "SIGNAL")).upper()

        if not item_id:
            raise HTTPException(status_code=400, detail="Missing required 'id' parameter.")

        if item_type == "TRADE" or item_id.startswith("TRADE_"):
            res = execution_worker.close_active_trade(item_id, reason="MANUAL_USER_CANCEL")
        else:
            res = execution_worker.cancel_signal(item_id)

        # Broadcast updated state over WebSockets immediately
        await broadcast_live_state()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Real-Time WebSocket Stream for VectorTrade Live Execution Dashboard.
    Pushes live ticks, armed signals, active positions, and accepts manual cancel commands.
    """
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Push initial snapshot immediately on connection (guaranteed jsonable primitives)
        initial_state = compile_full_system_state()
        await websocket.send_json(initial_state)

        while True:
            # Receive client actions (e.g. order cancel, heartbeat)
            msg = await websocket.receive_json()
            action = msg.get("action") if isinstance(msg, dict) else None
            if action == "PING":
                await websocket.send_json({"type": "PONG"})
                continue
            elif action == "CANCEL_ORDER":
                item_id = str(msg.get("id", "")).strip()
                item_type = str(msg.get("type", "SIGNAL")).upper()
                if item_type == "TRADE" or item_id.startswith("TRADE_"):
                    res = execution_worker.close_active_trade(item_id, reason="MANUAL_USER_CANCEL")
                else:
                    res = execution_worker.cancel_signal(item_id)
                await websocket.send_json({"type": "ACTION_RESPONSE", "result": res})
                await broadcast_live_state()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception as e:
        logger.warning("WebSocket client connection ended: %s", str(e))
        active_websockets.discard(websocket)
    finally:
        active_websockets.discard(websocket)


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
