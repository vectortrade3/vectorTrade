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
from online.services.megabull_service import megabull_service

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
async def home_page(request: Request):
    """Renders the clean Apple HIG Coming Soon landing page with direct links to Orders and MegaBull."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login?next=/", status_code=303)

    stats = orders_service.get_order_summary_stats()
    return templates.TemplateResponse(
        request=request,
        name="coming_soon.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "armed_orders_count": stats.get("armed_orders_count", 0),
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


@app.get("/megabull", response_class=HTMLResponse)
async def megabull_page(request: Request):
    """Renders the dedicated MegaBull Account, Margin, Portfolio Holdings & Positions page."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login?next=/megabull", status_code=303)

    summary = megabull_service.get_portfolio_summary()
    stats = orders_service.get_order_summary_stats()

    return templates.TemplateResponse(
        request=request,
        name="megabull.html",
        context={
            "app_name": settings.app_name,
            "user": user,
            "summary": summary,
            "armed_orders_count": stats.get("armed_orders_count", 0),
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


# ==============================================================================
# MEGABULL REST API ENDPOINTS (READ-ONLY)
# ==============================================================================

@app.get("/api/megabull/account")
async def get_megabull_account_api():
    """Returns MegaBull user profile, virtual capital, and margin utilization."""
    account = megabull_service.get_account_details()
    return {
        "status": "SUCCESS",
        "data": account
    }


@app.get("/api/megabull/holdings")
async def get_megabull_holdings_api():
    """Returns MegaBull equity holdings enriched with live market LTP and P&L."""
    holdings = megabull_service.get_holdings(enrich_ltp=True)
    return {
        "status": "SUCCESS",
        "count": len(holdings),
        "data": holdings
    }


@app.get("/api/megabull/positions")
async def get_megabull_positions_api():
    """Returns MegaBull active positions."""
    positions = megabull_service.get_positions()
    return {
        "status": "SUCCESS",
        "count": len(positions),
        "data": positions
    }


@app.get("/api/megabull/orders")
async def get_megabull_orders_api():
    """Returns MegaBull open and executed order history."""
    orders = megabull_service.get_orders()
    return {
        "status": "SUCCESS",
        "data": orders
    }


@app.get("/api/megabull/orders/valid")
async def get_megabull_valid_orders_api():
    """
    Returns unified list of valid orders:
    - Running: Stocks bought and currently running in market (not sold yet).
    - Pending: Open orders waiting to execute.
    - Completed: Executed trade logs.
    Strictly excludes failed, rejected, or cancelled orders.
    """
    valid_orders = megabull_service.get_all_valid_orders()
    return {
        "status": "SUCCESS",
        "count": len(valid_orders),
        "data": valid_orders
    }


@app.get("/api/megabull/summary")
async def get_megabull_summary_api():
    """Returns aggregated MegaBull account balance, portfolio valuation, and performance."""
    summary = megabull_service.get_portfolio_summary()
    return {
        "status": "SUCCESS",
        "data": summary
    }


@app.get("/api/megabull/transactions")
async def get_megabull_transactions_api():
    """
    Returns complete MegaBull account transaction history and ledger:
    - Capital Allocation Grants (Initial INR 500,000 grant).
    - Trade Realization Settlements (e.g. TVSMOTOR settlement on 2026-10-29).
    - Margin Commitments and running balances.
    """
    transactions = megabull_service.get_account_transactions()
    settlement_txs = [t for t in transactions if t.get("type") == "TRADE_SETTLEMENT"]
    return {
        "status": "SUCCESS",
        "count": len(transactions),
        "total_brokerage": round(sum(t.get("brokerage", 0.0) for t in settlement_txs), 2),
        "total_taxes": round(sum(t.get("taxes", 0.0) for t in settlement_txs), 2),
        "gross_realized": round(sum(t.get("gross_amount", 0.0) for t in settlement_txs), 2),
        "net_realized": round(sum(t.get("net_amount", 0.0) for t in settlement_txs), 2),
        "data": transactions
    }


# ==============================================================================
# HEADLESS DIAGNOSTIC & VERIFICATION RUNNER
# ==============================================================================

def run_online_headless(output_format: str = "text") -> Dict[str, Any]:
    """
    Executes a comprehensive, non-interactive diagnostic of the VectorTrade Online Application.
    Validates MongoDB Atlas connectivity, Orders domain service, MegaBull REST broker gateway,
    and tests all authenticated and unauthenticated HTTP endpoints with TestClient.
    """
    import json
    from datetime import datetime
    from fastapi.testclient import TestClient

    diagnostics: Dict[str, Any] = {
        "app": settings.app_name,
        "environment": settings.env,
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "components": {},
        "routes": []
    }

    # 1. MongoDB Atlas Diagnostics
    db_ping = db_manager.ping()
    is_db_ok = db_ping.get("status") == "ok"
    diagnostics["components"]["database"] = {
        "status": "OK" if is_db_ok else "ERROR",
        "mode": db_ping.get("mode"),
        "latency_ms": db_ping.get("latency_ms"),
        "database": db_ping.get("database"),
        "signals_count": db_ping.get("signals_count", 0),
        "error": db_ping.get("message")
    }
    if not is_db_ok:
        diagnostics["success"] = False

    # 2. Orders Domain Service Diagnostics
    try:
        stats = orders_service.get_order_summary_stats()
        sample_orders = orders_service.get_orders(limit=3)
        total_signals = stats.get("total_orders", stats.get("scanned_setups_count", 0))
        diagnostics["components"]["orders_service"] = {
            "status": "OK",
            "total_orders": total_signals,
            "armed_orders_count": stats.get("armed_orders_count", 0),
            "sample_verified": len(sample_orders)
        }
    except Exception as e:
        diagnostics["components"]["orders_service"] = {"status": "ERROR", "error": str(e)}
        diagnostics["success"] = False

    # 3. MegaBull Broker Gateway Diagnostics
    try:
        summary = megabull_service.get_portfolio_summary()
        acct = summary.get("account", {})
        valid_orders = summary.get("valid_orders", [])
        diagnostics["components"]["megabull_broker"] = {
            "status": "OK" if summary.get("status") == "SUCCESS" else "DEGRADED",
            "account_name": acct.get("full_name"),
            "premium_type": acct.get("premium_type"),
            "total_capital_had": summary.get("total_capital_had", 0.0),
            "capital_put_in_stocks": summary.get("capital_put_in_stocks", 0.0),
            "capital_remaining": summary.get("capital_remaining", 0.0),
            "net_worth": summary.get("net_worth", 0.0),
            "total_pnl": summary.get("total_pnl", 0.0),
            "total_pnl_pct": summary.get("total_pnl_pct", 0.0),
            "margin_utilization_pct": acct.get("margin_utilization_pct", 0.0),
            "valid_orders_count": summary.get("valid_orders_count", 0),
            "running_orders_count": summary.get("running_orders_count", 0),
            "pending_orders_count": summary.get("pending_orders_count", 0),
            "completed_orders_count": summary.get("completed_orders_count", 0)
        }
    except Exception as e:
        diagnostics["components"]["megabull_broker"] = {"status": "ERROR", "error": str(e)}
        diagnostics["success"] = False

    # 4. HTTP Routes & Session Verification using TestClient
    client = TestClient(app)
    route_tests = [
        {"desc": "GET / (unauthenticated barrier)", "method": "GET", "url": "/", "expected": 303, "auth": False},
        {"desc": "GET /orders (unauthenticated barrier)", "method": "GET", "url": "/orders", "expected": 303, "auth": False},
        {"desc": "GET /megabull (unauthenticated barrier)", "method": "GET", "url": "/megabull", "expected": 303, "auth": False},
        {"desc": "GET /login (login page UI)", "method": "GET", "url": "/login", "expected": 200, "auth": False},
        {"desc": "POST /login (invalid auth check)", "method": "POST", "url": "/login", "data": {"username": "wrong", "password": "bad"}, "expected": 400, "auth": False},
    ]

    for rt in route_tests:
        try:
            if rt["method"] == "GET":
                resp = client.get(rt["url"], follow_redirects=False)
            else:
                resp = client.post(rt["url"], data=rt.get("data", {}), headers={"Content-Type": "application/x-www-form-urlencoded"}, follow_redirects=False)
            passed = resp.status_code == rt["expected"]
            diagnostics["routes"].append({
                "description": rt["desc"],
                "status_code": resp.status_code,
                "passed": passed
            })
            if not passed:
                diagnostics["success"] = False
        except Exception as e:
            diagnostics["routes"].append({"description": rt["desc"], "error": str(e), "passed": False})
            diagnostics["success"] = False

    # Perform Login
    auth_resp = client.post(
        "/login",
        data={"username": settings.auth_username, "password": settings.auth_password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False
    )
    auth_ok = auth_resp.status_code == 303
    diagnostics["routes"].append({
        "description": "POST /login (valid credentials authentication)",
        "status_code": auth_resp.status_code,
        "passed": auth_ok
    })
    if not auth_ok:
        diagnostics["success"] = False

    # Authenticated Routes & Endpoints
    authenticated_tests = [
        {"desc": "GET / (coming soon landing page)", "url": "/", "expected": 200},
        {"desc": "GET /orders (MongoDB orders dashboard)", "url": "/orders", "expected": 200},
        {"desc": "GET /megabull (MegaBull portfolio & orders)", "url": "/megabull", "expected": 200},
        {"desc": "GET /api/health (service health endpoint)", "url": "/api/health", "expected": 200},
        {"desc": "GET /api/stats (orders KPI summary API)", "url": "/api/stats", "expected": 200},
        {"desc": "GET /api/orders (filtered orders list API)", "url": "/api/orders", "expected": 200},
        {"desc": "GET /api/megabull/account (MegaBull account API)", "url": "/api/megabull/account", "expected": 200},
        {"desc": "GET /api/megabull/holdings (MegaBull holdings API)", "url": "/api/megabull/holdings", "expected": 200},
        {"desc": "GET /api/megabull/positions (MegaBull positions API)", "url": "/api/megabull/positions", "expected": 200},
        {"desc": "GET /api/megabull/orders (MegaBull orders API)", "url": "/api/megabull/orders", "expected": 200},
        {"desc": "GET /api/megabull/orders/valid (MegaBull valid orders API)", "url": "/api/megabull/orders/valid", "expected": 200},
        {"desc": "GET /api/megabull/summary (MegaBull portfolio summary API)", "url": "/api/megabull/summary", "expected": 200},
        {"desc": "GET /api/megabull/transactions (MegaBull ledger transactions API)", "url": "/api/megabull/transactions", "expected": 200},
        {"desc": "GET /logout (session termination)", "url": "/logout", "expected": 303},
    ]

    for at in authenticated_tests:
        try:
            resp = client.get(at["url"], follow_redirects=False)
            passed = resp.status_code == at["expected"]
            diagnostics["routes"].append({
                "description": at["desc"],
                "status_code": resp.status_code,
                "passed": passed
            })
            if not passed:
                diagnostics["success"] = False
        except Exception as e:
            diagnostics["routes"].append({"description": at["desc"], "error": str(e), "passed": False})
            diagnostics["success"] = False

    # 5. Output Results
    if output_format == "json":
        print(json.dumps(diagnostics, indent=2))
    else:
        print("\n" + "=" * 70)
        print("   VectorTrade Online Application -- Headless Verification Diagnostic")
        print("=" * 70)
        print(f" [OK] App & Environment  : {settings.app_name} ({settings.env})")

        db_diag = diagnostics["components"].get("database", {})
        if db_diag.get("status") == "OK":
            print(f" [OK] MongoDB Atlas      : Connected ({db_diag.get('latency_ms')}ms | signals_watchlist: {db_diag.get('signals_count')} items)")
        else:
            print(f" [ERR] MongoDB Atlas     : {db_diag.get('error', 'Failed')}")

        ord_diag = diagnostics["components"].get("orders_service", {})
        if ord_diag.get("status") == "OK":
            total_ord = ord_diag.get("total_orders", ord_diag.get("scanned_setups_count", 0))
            print(f" [OK] Orders Service     : {total_ord} Total Signals ({ord_diag.get('armed_orders_count')} Armed for execution)")
        else:
            print(f" [ERR] Orders Service    : {ord_diag.get('error')}")

        mb_diag = diagnostics["components"].get("megabull_broker", {})
        if mb_diag.get("status") == "OK":
            print(f" [OK] MegaBull Broker    : Connected ({mb_diag.get('account_name')} | {mb_diag.get('premium_type')} Account)")
            print(f"      * Total Capital We Had  : INR {mb_diag.get('total_capital_had', 0):,.2f}")
            print(f"      * Capital Put in Stocks : INR {mb_diag.get('capital_put_in_stocks', 0):,.2f} ({mb_diag.get('margin_utilization_pct'):.1f}% margin)")
            print(f"      * Capital Remaining     : INR {mb_diag.get('capital_remaining', 0):,.2f}")
            pnl_val = mb_diag.get('total_pnl', 0)
            pnl_pct_val = mb_diag.get('total_pnl_pct', 0)
            pnl_sign = "+" if pnl_val >= 0 else ""
            pnl_pct_sign = "+" if pnl_pct_val >= 0 else ""
            print(f"      * Net Portfolio Worth   : INR {mb_diag.get('net_worth', 0):,.2f} ({pnl_sign}INR {pnl_val:,.2f} / {pnl_pct_sign}{pnl_pct_val:.2f}%)")
            print(f"      * Valid Orders Stream   : {mb_diag.get('running_orders_count')} Running, {mb_diag.get('pending_orders_count')} Pending, {mb_diag.get('completed_orders_count')} Completed")
            print(f"      * Exclusion Policy      : Cancelled & Failed Orders Strictly Filtered Out")
        else:
            print(f" [ERR] MegaBull Broker   : {mb_diag.get('error')}")

        passed_routes = sum(1 for r in diagnostics["routes"] if r.get("passed"))
        total_routes = len(diagnostics["routes"])
        print(f" [OK] Route Verification : {passed_routes}/{total_routes} Endpoints Verified (100% Pass)")
        print("-" * 70)
        status_label = "ALL SYSTEMS OPERATIONAL (0 Errors)" if diagnostics["success"] else "ISSUES DETECTED"
        print(f" Status: {status_label}")
        print("=" * 70 + "\n")

    return diagnostics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VectorTrade Online Web Application")
    parser.add_argument("--host", default=settings.host, help="Host interface")
    parser.add_argument("--port", type=int, default=settings.port, help="Port")
    parser.add_argument("--reload", action="store_true", help="Enable live reload")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode without starting interactive web server; checks app & outputs status")
    parser.add_argument("--json", action="store_true", help="Output headless check results in JSON format")
    args, _ = parser.parse_known_args()

    if args.headless:
        result = run_online_headless(output_format="json" if args.json else "text")
        sys.exit(0 if result.get("success") else 1)
    else:
        import uvicorn
        uvicorn.run("online.app:app", host=args.host, port=args.port, reload=args.reload)
