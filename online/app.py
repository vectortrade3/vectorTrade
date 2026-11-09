"""
VectorTrade - Online Cloud Orders & Trading Execution Dashboard
FastAPI Application serving real-time MongoDB Atlas trade setups,
orders monitoring board, and executive analytics.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONLINE_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online.config import settings
from online.db import db_manager
from online.services.orders_service import orders_service

logger = logging.getLogger("VectorTrade.OnlineApp")
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="VectorTrade Cloud Orders Dashboard & Execution Terminal",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates configuration
static_dir = ONLINE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = ONLINE_DIR / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# ==============================================================================
# UI ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/orders", response_class=HTMLResponse)
async def orders_page(request: Request):
    """Renders the clean, restrained Apple HIG + Linear design Orders Board."""
    stats = orders_service.get_order_summary_stats()
    db_health = db_manager.ping()
    initial_orders = orders_service.get_orders(limit=100)

    return templates.TemplateResponse(
        request=request,
        name="orders.html",
        context={
            "app_name": settings.app_name,
            "stats": stats,
            "db_health": db_health,
            "orders": initial_orders,
            "env": settings.env
        }
    )


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint verifying online service and MongoDB Atlas connectivity."""
    db_ping = db_manager.ping()
    is_healthy = db_ping.get("status") == "ok"
    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "app_name": settings.app_name,
        "database": db_ping,
        "environment": settings.env
    }


@app.get("/api/stats")
async def get_stats():
    """Returns aggregated executive statistics for all orders."""
    stats = orders_service.get_order_summary_stats()
    return {
        "status": "SUCCESS",
        "data": stats
    }


@app.get("/api/orders")
async def get_orders_api(
    status: Optional[str] = Query(None, description="Filter by status: ARMED, ACTIVE, TARGET_HIT, SL_HIT, ALL"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    setup_num: Optional[int] = Query(None, description="Filter by setup number (1, 2, or 3)"),
    search: Optional[str] = Query(None, description="Search by symbol or company name"),
    date: Optional[str] = Query(None, description="Filter by valid_for_date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=500, description="Max orders to return")
):
    """
    Returns filtered, sorted list of orders/signals from MongoDB Atlas signals_watchlist.
    """
    orders = orders_service.get_orders(
        status=status,
        strategy=strategy,
        setup_num=setup_num,
        search=search,
        session_date=date,
        limit=limit
    )
    return {
        "status": "SUCCESS",
        "count": len(orders),
        "orders": orders
    }


@app.get("/api/orders/{order_id}")
async def get_order_by_id_api(order_id: str):
    """Returns detailed information for a single order."""
    order = orders_service.get_order_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return {
        "status": "SUCCESS",
        "order": order
    }


if __name__ == "__main__":
    import uvicorn
    # Local direct debugging runner only
    uvicorn.run("online.app:app", host=settings.host, port=settings.port, reload=True)
