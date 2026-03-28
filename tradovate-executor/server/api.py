"""
Dashboard API -- FastAPI backend wrapping the Tradovate trading engine.
Phase 1: Account CRUD, auth testing, mock engine status, WebSocket mock data.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from server import config_store
from server.schemas import (
    AccountCreate,
    AccountUpdate,
    AuthTestRequest,
    AuthTestResponse,
    EnvironmentUpdate,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard API starting")
    yield
    logger.info("Dashboard API shutting down")


app = FastAPI(title="Tradovate Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Account CRUD ------------------------------------------------------------


@app.get("/api/accounts")
def list_accounts():
    accounts = config_store.get_accounts()
    return [config_store.mask_account(a) for a in accounts]


@app.post("/api/accounts", status_code=201)
def create_account(req: AccountCreate):
    acct = {
        "name": req.name,
        "username": req.username,
        "password": req.password,
        "device_id": req.device_id or f"device-{req.name}",
        "app_id": "HTFSwing",
        "app_version": "1.0.0",
        "cid": req.cid,
        "sec": req.sec,
        "is_master": req.is_master,
        "sizing_mode": req.sizing_mode,
        "account_size": req.account_size,
        "fixed_sizes": req.fixed_sizes,
    }
    try:
        config_store.add_account(acct)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return config_store.mask_account(acct)


@app.put("/api/accounts/{name}")
def update_account(name: str, req: AccountUpdate):
    updates = req.model_dump(exclude_none=True)
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


# -- Auth Test ----------------------------------------------------------------


@app.post("/api/auth/test", response_model=AuthTestResponse)
async def test_auth(req: AuthTestRequest):
    acct = config_store.get_account(req.name)
    if not acct:
        raise HTTPException(404, f"Account '{req.name}' not found")

    env = config_store.get_environment()
    base_url = (
        "https://demo.tradovateapi.com/v1"
        if env == "demo"
        else "https://live.tradovateapi.com/v1"
    )

    payload = {
        "name": acct["username"],
        "password": acct["password"],
        "appId": acct.get("app_id", "HTFSwing"),
        "appVersion": acct.get("app_version", "1.0.0"),
        "deviceId": acct["device_id"],
        "cid": acct.get("cid", 0),
        "sec": acct.get("sec", ""),
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
    except Exception as e:
        return AuthTestResponse(success=False, error=str(e))


# -- Engine Control (mock -- Phase 1) ----------------------------------------

_engine_state = {"running": False}


@app.post("/api/engine/start")
def start_engine():
    if _engine_state["running"]:
        raise HTTPException(409, "Engine already running")
    _engine_state["running"] = True
    return {"status": "started"}


@app.post("/api/engine/stop")
def stop_engine():
    if not _engine_state["running"]:
        raise HTTPException(409, "Engine not running")
    _engine_state["running"] = False
    return {"status": "stopped"}


@app.post("/api/engine/flatten")
def flatten_all():
    _engine_state["running"] = False
    return {"status": "flattened", "message": "All positions flattened"}


@app.get("/api/engine/status")
def engine_status():
    from server.mock_data import get_engine_status
    return get_engine_status(_engine_state["running"])


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
    tick = 0
    try:
        while True:
            from server.mock_data import generate_ws_batch
            messages = generate_ws_batch(_engine_state["running"], tick)
            for msg in messages:
                await ws.send_json(msg)
            tick += 1
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_connections:
            _ws_connections.remove(ws)
