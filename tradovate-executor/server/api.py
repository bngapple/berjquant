"""
Dashboard API — FastAPI backend wrapping the real Tradovate trading engine.
No mock data. All data comes from real engine state or CSV trade logs.
Serves the built React frontend as static files in production.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server import config_store
from server.schemas import (
    AccountCreate,
    AccountUpdate,
    AuthTestRequest,
    AuthTestResponse,
    EnvironmentUpdate,
)
from server.engine_bridge import bridge
from server import history
from server import account_tracker

logger = logging.getLogger(__name__)

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard API starting")
    yield
    # Shutdown: stop engine if running
    if bridge.running:
        logger.info("Stopping engine on shutdown...")
        await bridge.stop()
    logger.info("Dashboard API shutting down")


app = FastAPI(title="Tradovate Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Auth Validation (shared) ------------------------------------------------


async def _validate_credentials(
    username: str, password: str, device_id: str,
    app_id: str = "HTFSwing", app_version: str = "1.0.0",
    cid: int = 0, sec: str = "",
) -> AuthTestResponse:
    """Validate credentials against real Tradovate API. Returns auth result."""
    env = config_store.get_environment()
    base_url = (
        "https://demo.tradovateapi.com/v1"
        if env == "demo"
        else "https://live.tradovateapi.com/v1"
    )

    payload = {
        "name": username,
        "password": password,
        "appId": app_id,
        "appVersion": app_version,
        "deviceId": device_id,
        "cid": cid,
        "sec": sec,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(
                f"{base_url}/auth/accesstokenrequest", json=payload
            )
            data = resp.json()

            if "errorText" in data:
                return AuthTestResponse(success=False, error=data["errorText"])

            token = data.get("accessToken")
            user_id = data.get("userId")

            headers = {"Authorization": f"Bearer {token}"}
            acct_resp = await http.get(f"{base_url}/account/list", headers=headers)
            accounts = acct_resp.json()
            account_id = accounts[0]["id"] if accounts else None

            return AuthTestResponse(
                success=True, account_id=account_id, user_id=user_id
            )
    except httpx.ConnectError:
        return AuthTestResponse(
            success=False,
            error="Could not reach Tradovate — check your internet connection",
        )
    except httpx.TimeoutException:
        return AuthTestResponse(
            success=False,
            error="Could not reach Tradovate — connection timed out",
        )
    except Exception as e:
        return AuthTestResponse(success=False, error=str(e))


# -- Account CRUD ------------------------------------------------------------


@app.get("/api/accounts")
def list_accounts():
    accounts = config_store.get_accounts()
    return [config_store.mask_account(a) for a in accounts]


@app.post("/api/accounts", status_code=201)
async def create_account(req: AccountCreate):
    # Check for duplicate name first
    if config_store.get_account(req.name):
        raise HTTPException(409, f"Account '{req.name}' already exists")

    device_id = req.device_id or f"device-{req.name}"

    # Validate credentials against real Tradovate API
    auth_result = await _validate_credentials(
        username=req.username,
        password=req.password,
        device_id=device_id,
        cid=req.cid,
        sec=req.sec,
    )
    if not auth_result.success:
        raise HTTPException(422, auth_result.error or "Authentication failed")

    acct = {
        "name": req.name,
        "username": req.username,
        "password": req.password,
        "device_id": device_id,
        "app_id": "HTFSwing",
        "app_version": "1.0.0",
        "cid": req.cid,
        "sec": req.sec,
        "is_master": req.is_master,
        "sizing_mode": req.sizing_mode,
        "account_size": req.account_size,
        "fixed_sizes": req.fixed_sizes,
        "starting_balance": req.starting_balance,
        "profit_target": req.profit_target,
        "max_drawdown": req.max_drawdown,
        "account_type": req.account_type,
    }
    try:
        config_store.add_account(acct)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return config_store.mask_account(acct)


@app.put("/api/accounts/{name}")
async def update_account(name: str, req: AccountUpdate):
    existing = config_store.get_account(name)
    if not existing:
        raise HTTPException(404, f"Account '{name}' not found")

    updates = req.model_dump(exclude_none=True)

    # If credentials changed, re-validate against Tradovate
    creds_changed = any(k in updates for k in ("username", "password", "cid", "sec"))
    if creds_changed:
        # Merge with existing values for fields not being updated
        username = updates.get("username", existing["username"])
        password = updates.get("password", existing["password"])
        device_id = updates.get("device_id", existing["device_id"])
        cid = updates.get("cid", existing.get("cid", 0))
        sec = updates.get("sec", existing.get("sec", ""))

        auth_result = await _validate_credentials(
            username=username, password=password,
            device_id=device_id, cid=cid, sec=sec,
        )
        if not auth_result.success:
            raise HTTPException(422, auth_result.error or "Authentication failed")

    try:
        acct = config_store.update_account(name, updates)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return config_store.mask_account(acct)


@app.delete("/api/accounts/{name}", status_code=204)
def delete_account(name: str):
    if not config_store.get_account(name):
        raise HTTPException(404, f"Account '{name}' not found")
    config_store.delete_account(name)


# -- Auth Test (standalone — test without saving) ----------------------------


@app.post("/api/auth/test", response_model=AuthTestResponse)
async def test_auth(req: AuthTestRequest):
    acct = config_store.get_account(req.name)
    if not acct:
        raise HTTPException(404, f"Account '{req.name}' not found")

    return await _validate_credentials(
        username=acct["username"],
        password=acct["password"],
        device_id=acct["device_id"],
        app_id=acct.get("app_id", "HTFSwing"),
        app_version=acct.get("app_version", "1.0.0"),
        cid=acct.get("cid", 0),
        sec=acct.get("sec", ""),
    )


# -- Engine Control (REAL) ---------------------------------------------------


@app.post("/api/engine/start")
async def start_engine():
    if bridge.running:
        raise HTTPException(409, "Engine already running")

    accounts = config_store.get_accounts()
    if not accounts:
        raise HTTPException(400, "No accounts configured")

    master = [a for a in accounts if a.get("is_master")]
    if not master:
        raise HTTPException(400, "No master account configured")

    result = await bridge.start()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/engine/stop")
async def stop_engine():
    if not bridge.running:
        raise HTTPException(409, "Engine not running")
    return await bridge.stop()


@app.post("/api/engine/flatten")
async def flatten_all():
    result = await bridge.flatten()
    return result


@app.get("/api/engine/status")
def engine_status():
    return bridge.get_status()


# -- Health Check -------------------------------------------------------------


@app.get("/api/health")
def health_check():
    h = bridge.get_health()
    h["ws_clients"] = len(_ws_connections)
    h["server_uptime"] = round(time.time() - _start_time, 1)
    return h


# -- History ------------------------------------------------------------------


@app.get("/api/history/stats")
def history_stats():
    trades = history.parse_trades()
    return history.compute_stats(trades)


@app.get("/api/history/daily")
def history_daily():
    trades = history.parse_trades()
    daily = history.compute_daily_pnl(trades)
    return daily


@app.get("/api/history/equity")
def history_equity():
    trades = history.parse_trades()
    daily = history.compute_daily_pnl(trades)
    return history.compute_equity_curve(daily)


@app.get("/api/history/trades")
def history_trades(limit: int = 50):
    trades = history.parse_trades()
    return history.get_recent_trades(trades, limit)


# -- Account Status -----------------------------------------------------------


@app.get("/api/accounts/status")
def accounts_status():
    return account_tracker.get_all_statuses()


@app.get("/api/accounts/status/{name}")
def account_status(name: str):
    acct = config_store.get_account(name)
    if not acct:
        raise HTTPException(404, f"Account '{name}' not found")
    return account_tracker.get_account_status(name)


@app.get("/api/accounts/alerts")
def fleet_alerts():
    return account_tracker.get_fleet_alerts()


# -- Environment --------------------------------------------------------------


@app.get("/api/environment")
def get_environment():
    return {"environment": config_store.get_environment()}


@app.put("/api/environment")
def set_environment(req: EnvironmentUpdate):
    if req.environment not in ("demo", "live"):
        raise HTTPException(400, "Must be 'demo' or 'live'")
    config_store.set_environment(req.environment)
    return {"environment": req.environment}


# -- WebSocket ----------------------------------------------------------------

_ws_connections: list[WebSocket] = []


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    _ws_connections.append(ws)
    try:
        while True:
            if bridge.running:
                # Drain events from engine bridge
                try:
                    event = await asyncio.wait_for(bridge.event_queue.get(), timeout=2.0)
                    await ws.send_json(event)
                except asyncio.TimeoutError:
                    # No event — send status heartbeat
                    await ws.send_json({
                        "type": "status",
                        "data": bridge.get_status(),
                    })
            else:
                # Engine stopped — send status every 5s
                await ws.send_json({
                    "type": "status",
                    "data": bridge.get_status(),
                })
                await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS error: {e}")
    finally:
        if ws in _ws_connections:
            _ws_connections.remove(ws)


# -- Static Files (production frontend) --------------------------------------

# Resolve frontend dist directory. Search order:
# 1. dashboard/dist relative to project root (__file__ based) — development
# 2. dist/ relative to project root — if pre-built in place
# 3. dashboard/dist relative to cwd — development via app_launcher
# 4. dist/ relative to cwd — py2app bundle (Resources/dist/)
_DIST_DIR = None
for _candidate in [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "dist"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist"),
    os.path.join(os.getcwd(), "dashboard", "dist"),
    os.path.join(os.getcwd(), "dist"),
]:
    if os.path.isdir(_candidate) and os.path.isfile(os.path.join(_candidate, "index.html")):
        _DIST_DIR = _candidate
        break

if _DIST_DIR:
    _assets_dir = os.path.join(_DIST_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # Sanitize: resolve and ensure path stays within _DIST_DIR
        file_path = os.path.realpath(os.path.join(_DIST_DIR, path))
        dist_real = os.path.realpath(_DIST_DIR)
        if not file_path.startswith(dist_real + os.sep) and file_path != dist_real:
            # Path traversal attempt — serve index.html instead
            return FileResponse(os.path.join(_DIST_DIR, "index.html"))
        # Try to serve the exact file first
        if path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # SPA fallback — serve index.html for all routes
        index = os.path.join(_DIST_DIR, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
        raise HTTPException(404, "Not found")
