"""
VectorTrade - Online Cloud Orders & Trading Execution Dashboard
FastAPI Application serving real-time MongoDB Atlas trade setups,
session authentication, executive overview, and dedicated orders board.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

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

# Enable signed session cookies for authentication
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    session_cookie="vt_online_session",
    max_age=settings.session_max_age_seconds
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


async def parse_request_data(request: Request) -> Dict[str, Any]:
    """Helper to parse both JSON and URL-encoded form submissions without third-party multipart."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return await request.json()
        except Exception:
            return {}
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    parsed = parse_qs(body_str)
    return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


# ==============================================================================
# AUTHENTICATION ROUTES
# ==============================================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Renders the Apple HIG login screen or redirects if already authenticated."""
    if request.session.get("user"):
        next_url = request.query_params.get("next", "/orders")
        return RedirectResponse(url=next_url, status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "app_name": settings.app_name,
            "error": None,
            "next": request.query_params.get("next", "/orders"),
            "default_user": "root"
        }
    )


@app.post("/login")
async def login_submit(request: Request):
    """Handles username & password validation against environment settings."""
    data = await parse_request_data(request)
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    next_url = str(data.get("next", "/orders")).strip()

    if not next_url or not next_url.startswith("/"):
        next_url = "/orders"

    # Authenticate against configured environment variables
    if username == settings.auth_username and password == settings.auth_password:
        request.session["user"] = username
        logger.info(f"User '{username}' authenticated successfully.")

        # If requesting JSON API
        accept_hdr = request.headers.get("accept", "")
        if "application/json" in accept_hdr and "text/html" not in accept_hdr:
            return JSONResponse({"status": "SUCCESS", "user": username, "redirect": next_url})

        return RedirectResponse(url=next_url, status_code=303)
    else:
        logger.warning(f"Failed login attempt for username: '{username}'")
        accept_hdr = request.headers.get("accept", "")
        if "application/json" in accept_hdr and "text/html" not in accept_hdr:
            return JSONResponse({"status": "ERROR", "message": "Invalid username or password"}, status_code=401)

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "app_name": settings.app_name,
                "error": "Invalid username or password. Please verify your credentials.",
                "next": next_url,
                "default_user": username
            },
            status_code=400
        )


@app.get("/logout")
@app.post("/logout")
async def logout(request: Request):
    """Clears user session and redirects to login."""
    user = request.session.get("user")
    if user:
        logger.info(f"User '{user}' signed out.")
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ==============================================================================
# UI DASHBOARD ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def overview_page(request: Request):
    """Renders the Cloud Executive Overview & Gateway Dashboard."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login?next=/", status_code=303)

    stats = orders_service.get_order_summary_stats()
    db_health = db_manager.ping()

    return templates.TemplateResponse(
        request=request,
        name="overview.html",
        context={
            "app_name": settings.app_name,
            "stats": stats,
            "db_health": db_health,
            "user": user,
            "env": settings.env
        }
    )


@app.get("/orders", response_class=HTMLResponse)
async def orders_page(request: Request):
    """Renders the dedicated MongoDB Synced Orders & Execution Board."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login?next=/orders", status_code=303)

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
            "user": user,
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
    uvicorn.run("online.app:app", host=settings.host, port=settings.port, reload=True)
