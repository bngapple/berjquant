# Dashboard Phase 1: Backend API + Frontend Shell

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend wrapping the existing trading engine config + React frontend shell with account management, mock dashboard data, and real Tradovate auth testing.

**Architecture:** FastAPI on port 8000 serves REST API + WebSocket. Vite dev server on port 8080 proxies `/api/*` and `/ws/*` to FastAPI. Config persisted to existing `config.json` format with Fernet-encrypted passwords at rest. Dashboard displays mock trading data until engine integration in Phase 2.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, Fernet (cryptography), React 18, TypeScript, Vite, Tailwind CSS v3, React Router v6

---

## File Structure

```
tradovate-executor/
├── server/
│   ├── __init__.py            # empty
│   ├── api.py                 # FastAPI app — all routes + WebSocket
│   ├── schemas.py             # Pydantic request/response models
│   ├── config_store.py        # Account CRUD + Fernet password encryption
│   └── mock_data.py           # Mock data generators for dashboard WS
├── dashboard/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   ├── tsconfig.node.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css           # Tailwind base + dark theme
│       ├── vite-env.d.ts
│       ├── api/
│       │   └── client.ts       # fetch() wrapper for all endpoints
│       ├── types/
│       │   └── index.ts        # TypeScript interfaces matching schemas.py
│       ├── hooks/
│       │   └── useWebSocket.ts # WebSocket hook for /ws/live
│       ├── pages/
│       │   ├── Setup.tsx       # Account management + auth testing
│       │   ├── Dashboard.tsx   # Engine controls + positions + P&L + logs
│       │   └── Settings.tsx    # Strategy params (read-only)
│       └── components/
│           ├── Layout.tsx      # App shell with sidebar nav
│           └── StatusBadge.tsx # Engine status indicator
├── tests/
│   ├── __init__.py
│   └── test_api.py            # Backend endpoint tests
├── run_dashboard.py           # Dev runner — starts backend + frontend
└── requirements.txt           # Updated with FastAPI deps
```

---

### Task 1: Backend Dependencies and Schemas

**Files:**
- Modify: `requirements.txt`
- Create: `server/__init__.py`
- Create: `server/schemas.py`

- [ ] **Step 1: Add FastAPI dependencies to requirements.txt**

Append to `requirements.txt`:

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-multipart>=0.0.9
```

- [ ] **Step 2: Install new dependencies**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && pip install fastapi "uvicorn[standard]" python-multipart`

Expected: Successfully installed fastapi uvicorn ...

- [ ] **Step 3: Create server package**

Create `server/__init__.py` (empty file).

- [ ] **Step 4: Create Pydantic schemas**

Create `server/schemas.py`:

```python
from pydantic import BaseModel
from typing import Optional


class AccountCreate(BaseModel):
    name: str
    username: str
    password: str
    cid: int = 0
    sec: str = ""
    device_id: str = ""
    is_master: bool = False
    sizing_mode: str = "mirror"
    account_size: float = 150000.0
    fixed_sizes: dict[str, int] = {"RSI": 3, "IB": 3, "MOM": 3}


class AccountUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    cid: Optional[int] = None
    sec: Optional[str] = None
    device_id: Optional[str] = None
    is_master: Optional[bool] = None
    sizing_mode: Optional[str] = None
    account_size: Optional[float] = None
    fixed_sizes: Optional[dict[str, int]] = None


class AccountResponse(BaseModel):
    name: str
    username: str
    password: str
    cid: int
    sec: str
    device_id: str
    is_master: bool
    sizing_mode: str
    account_size: float
    fixed_sizes: dict[str, int]


class AuthTestRequest(BaseModel):
    name: str


class AuthTestResponse(BaseModel):
    success: bool
    account_id: Optional[int] = None
    user_id: Optional[int] = None
    error: Optional[str] = None


class EngineStatus(BaseModel):
    running: bool
    can_trade: bool
    daily_pnl: float
    monthly_pnl: float
    daily_limit: float
    monthly_limit: float
    daily_limit_hit: bool
    monthly_limit_hit: bool
    positions: dict
    pending_signals: int
    connected_accounts: list[dict]


class EnvironmentUpdate(BaseModel):
    environment: str
```

- [ ] **Step 5: Verify imports**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -c "from server.schemas import AccountCreate, AuthTestResponse; print('schemas ok')"`

Expected: `schemas ok`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt server/__init__.py server/schemas.py
git commit -m "feat: add backend dependencies and Pydantic schemas"
```

---

### Task 2: Config Store with Password Encryption

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_api.py` (config store tests only for now)
- Create: `server/config_store.py`

- [ ] **Step 1: Create tests directory and write config store tests**

Create `tests/__init__.py` (empty file).

Create `tests/test_api.py`:

```python
import json
import os
import pytest

# Point to test config file to avoid polluting real config
os.environ["CONFIG_PATH"] = "test_config.json"
os.environ["KEY_FILE"] = "test_secret_key"

from server import config_store


@pytest.fixture(autouse=True)
def clean_test_files():
    """Remove test files before and after each test."""
    for f in ["test_config.json", "test_secret_key"]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in ["test_config.json", "test_secret_key"]:
        if os.path.exists(f):
            os.remove(f)


class TestConfigStore:
    def test_load_missing_config_returns_defaults(self):
        data = config_store.load_config()
        assert data["environment"] == "demo"
        assert data["symbol"] == "MNQM6"
        assert data["accounts"] == []

    def test_add_and_get_account(self):
        acct = {
            "name": "test-1",
            "username": "user1",
            "password": "pass1",
            "device_id": "dev-1",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 123,
            "sec": "mysecret",
            "is_master": True,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        }
        config_store.add_account(acct)
        result = config_store.get_account("test-1")
        assert result is not None
        assert result["username"] == "user1"
        assert result["password"] == "pass1"  # Decrypted on read

    def test_passwords_encrypted_on_disk(self):
        config_store.add_account({
            "name": "enc-test",
            "username": "u",
            "password": "secret_password",
            "device_id": "d",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 0,
            "sec": "api_secret",
            "is_master": False,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        # Read raw JSON from disk
        with open("test_config.json") as f:
            raw = json.load(f)
        raw_acct = raw["accounts"][0]
        assert raw_acct["password"].startswith("enc:")
        assert raw_acct["sec"].startswith("enc:")
        assert raw_acct["password"] != "secret_password"

    def test_add_duplicate_name_raises(self):
        config_store.add_account({
            "name": "dup",
            "username": "u",
            "password": "p",
            "device_id": "d",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 0,
            "sec": "",
            "is_master": False,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        with pytest.raises(ValueError, match="already exists"):
            config_store.add_account({
                "name": "dup",
                "username": "u2",
                "password": "p2",
                "device_id": "d2",
                "app_id": "HTFSwing",
                "app_version": "1.0.0",
                "cid": 0,
                "sec": "",
                "is_master": False,
                "sizing_mode": "mirror",
                "account_size": 150000.0,
                "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
            })

    def test_update_account(self):
        config_store.add_account({
            "name": "upd",
            "username": "u",
            "password": "p",
            "device_id": "d",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 0,
            "sec": "",
            "is_master": False,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        result = config_store.update_account("upd", {"sizing_mode": "fixed", "account_size": 50000.0})
        assert result["sizing_mode"] == "fixed"
        assert result["account_size"] == 50000.0

    def test_update_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not found"):
            config_store.update_account("nope", {"sizing_mode": "fixed"})

    def test_delete_account(self):
        config_store.add_account({
            "name": "del-me",
            "username": "u",
            "password": "p",
            "device_id": "d",
            "app_id": "HTFSwing",
            "app_version": "1.0.0",
            "cid": 0,
            "sec": "",
            "is_master": False,
            "sizing_mode": "mirror",
            "account_size": 150000.0,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })
        config_store.delete_account("del-me")
        assert config_store.get_account("del-me") is None

    def test_mask_account(self):
        acct = {"name": "x", "password": "secret", "sec": "api_key", "username": "u"}
        masked = config_store.mask_account(acct)
        assert masked["password"] == "********"
        assert masked["sec"] == "********"
        assert masked["username"] == "u"  # Not masked

    def test_environment_get_set(self):
        assert config_store.get_environment() == "demo"
        config_store.set_environment("live")
        assert config_store.get_environment() == "live"

    def test_backwards_compat_plaintext_passwords(self):
        """Config with plaintext passwords should still load correctly."""
        with open("test_config.json", "w") as f:
            json.dump({
                "environment": "demo",
                "symbol": "MNQM6",
                "accounts": [{
                    "name": "plain",
                    "username": "u",
                    "password": "plaintext_pass",
                    "device_id": "d",
                    "app_id": "HTFSwing",
                    "app_version": "1.0.0",
                    "cid": 0,
                    "sec": "plain_sec",
                    "is_master": False,
                    "sizing_mode": "mirror",
                    "account_size": 150000.0,
                    "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
                }]
            }, f)
        acct = config_store.get_account("plain")
        assert acct["password"] == "plaintext_pass"
        assert acct["sec"] == "plain_sec"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py::TestConfigStore -v 2>&1 | head -30`

Expected: ERRORS — `ModuleNotFoundError: No module named 'server.config_store'`

- [ ] **Step 3: Implement config_store.py**

Create `server/config_store.py`:

```python
"""
Config Store — CRUD for accounts in config.json with Fernet password encryption.

Passwords and API secrets are encrypted at rest using Fernet symmetric encryption.
The encryption key is stored in .secret_key (generated on first run).
Backwards compatible: plaintext passwords (no 'enc:' prefix) are read as-is.
"""

import json
import os
from cryptography.fernet import Fernet

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")
KEY_FILE = os.environ.get("KEY_FILE", ".secret_key")

_DEFAULT_CONFIG = {
    "environment": "demo",
    "symbol": "MNQM6",
    "accounts": [],
}


def _get_fernet() -> Fernet:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read().strip()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    return Fernet(key)


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return plaintext
    return "enc:" + _get_fernet().encrypt(plaintext.encode()).decode()


def _decrypt(stored: str) -> str:
    if not stored:
        return stored
    if stored.startswith("enc:"):
        return _get_fernet().decrypt(stored[4:].encode()).decode()
    return stored  # Plaintext fallback


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return dict(_DEFAULT_CONFIG)
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    for acct in data.get("accounts", []):
        acct["password"] = _decrypt(acct.get("password", ""))
        acct["sec"] = _decrypt(acct.get("sec", ""))
    return data


def save_config(data: dict):
    save_data = json.loads(json.dumps(data))  # Deep copy
    for acct in save_data.get("accounts", []):
        pw = acct.get("password", "")
        if pw and not pw.startswith("enc:"):
            acct["password"] = _encrypt(pw)
        sec = acct.get("sec", "")
        if sec and not sec.startswith("enc:"):
            acct["sec"] = _encrypt(sec)
    with open(CONFIG_PATH, "w") as f:
        json.dump(save_data, f, indent=2)


def get_accounts() -> list[dict]:
    return load_config().get("accounts", [])


def get_account(name: str) -> dict | None:
    for acct in get_accounts():
        if acct["name"] == name:
            return acct
    return None


def add_account(acct: dict):
    config = load_config()
    for existing in config["accounts"]:
        if existing["name"] == acct["name"]:
            raise ValueError(f"Account '{acct['name']}' already exists")
    config["accounts"].append(acct)
    save_config(config)


def update_account(name: str, updates: dict) -> dict:
    config = load_config()
    for acct in config["accounts"]:
        if acct["name"] == name:
            for k, v in updates.items():
                if v is not None:
                    acct[k] = v
            save_config(config)
            return acct
    raise ValueError(f"Account '{name}' not found")


def delete_account(name: str):
    config = load_config()
    config["accounts"] = [a for a in config["accounts"] if a["name"] != name]
    save_config(config)


def mask_account(acct: dict) -> dict:
    masked = dict(acct)
    masked["password"] = "********"
    if masked.get("sec"):
        masked["sec"] = "********"
    return masked


def get_environment() -> str:
    return load_config().get("environment", "demo")


def set_environment(env: str):
    config = load_config()
    config["environment"] = env
    save_config(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py::TestConfigStore -v`

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/config_store.py tests/__init__.py tests/test_api.py
git commit -m "feat: config store with Fernet password encryption and CRUD"
```

---

### Task 3: FastAPI App with Account CRUD Endpoints

**Files:**
- Modify: `tests/test_api.py` (add API endpoint tests)
- Create: `server/api.py`

- [ ] **Step 1: Write endpoint tests**

Append to `tests/test_api.py`:

```python
from fastapi.testclient import TestClient
from server.api import app

client = TestClient(app)


class TestAccountEndpoints:
    def test_list_accounts_empty(self):
        resp = client.get("/api/accounts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_account(self):
        resp = client.post("/api/accounts", json={
            "name": "api-test-1",
            "username": "testuser",
            "password": "testpass",
            "cid": 123,
            "sec": "secret",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "api-test-1"
        assert data["password"] == "********"
        assert data["sec"] == "********"
        assert data["sizing_mode"] == "mirror"

    def test_create_duplicate_returns_409(self):
        client.post("/api/accounts", json={
            "name": "dup-api",
            "username": "u",
            "password": "p",
        })
        resp = client.post("/api/accounts", json={
            "name": "dup-api",
            "username": "u2",
            "password": "p2",
        })
        assert resp.status_code == 409

    def test_update_account(self):
        client.post("/api/accounts", json={
            "name": "upd-api",
            "username": "u",
            "password": "p",
        })
        resp = client.put("/api/accounts/upd-api", json={
            "sizing_mode": "fixed",
            "account_size": 50000.0,
        })
        assert resp.status_code == 200
        assert resp.json()["sizing_mode"] == "fixed"
        assert resp.json()["account_size"] == 50000.0

    def test_update_nonexistent_returns_404(self):
        resp = client.put("/api/accounts/ghost", json={"sizing_mode": "fixed"})
        assert resp.status_code == 404

    def test_delete_account(self):
        client.post("/api/accounts", json={
            "name": "del-api",
            "username": "u",
            "password": "p",
        })
        resp = client.delete("/api/accounts/del-api")
        assert resp.status_code == 204
        # Verify gone
        accounts = client.get("/api/accounts").json()
        assert not any(a["name"] == "del-api" for a in accounts)

    def test_delete_nonexistent_returns_404(self):
        resp = client.delete("/api/accounts/ghost")
        assert resp.status_code == 404

    def test_list_masks_passwords(self):
        client.post("/api/accounts", json={
            "name": "mask-test",
            "username": "u",
            "password": "supersecret",
            "sec": "api_key",
        })
        accounts = client.get("/api/accounts").json()
        target = [a for a in accounts if a["name"] == "mask-test"][0]
        assert target["password"] == "********"
        assert target["sec"] == "********"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py::TestAccountEndpoints -v 2>&1 | head -20`

Expected: ERRORS — `ImportError: cannot import name 'app' from 'server.api'`

- [ ] **Step 3: Implement server/api.py with account routes**

Create `server/api.py`:

```python
"""
Dashboard API — FastAPI backend wrapping the Tradovate trading engine.

Phase 1: Account CRUD, auth testing, mock engine status, WebSocket mock data.
Engine integration deferred to Phase 2.
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


# ── Account CRUD ────────────────────────────────────────────────────────


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


# ── Auth Test ───────────────────────────────────────────────────────────


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

            # Fetch account ID
            headers = {"Authorization": f"Bearer {token}"}
            acct_resp = await http.get(f"{base_url}/account/list", headers=headers)
            accounts = acct_resp.json()
            account_id = accounts[0]["id"] if accounts else None

            return AuthTestResponse(
                success=True, account_id=account_id, user_id=user_id
            )
    except Exception as e:
        return AuthTestResponse(success=False, error=str(e))


# ── Engine Control (mock — Phase 1) ────────────────────────────────────

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


# ── Environment ─────────────────────────────────────────────────────────


@app.get("/api/environment")
def get_environment():
    return {"environment": config_store.get_environment()}


@app.put("/api/environment")
def set_environment(req: EnvironmentUpdate):
    if req.environment not in ("demo", "live"):
        raise HTTPException(400, "Must be 'demo' or 'live'")
    config_store.set_environment(req.environment)
    return {"environment": req.environment}


# ── WebSocket ───────────────────────────────────────────────────────────

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
```

- [ ] **Step 4: Run endpoint tests**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py::TestAccountEndpoints -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/api.py tests/test_api.py
git commit -m "feat: FastAPI app with account CRUD, auth test, engine mock, WebSocket"
```

---

### Task 4: Mock Data Generator and Engine Endpoint Tests

**Files:**
- Create: `server/mock_data.py`
- Modify: `tests/test_api.py` (add engine + environment tests)

- [ ] **Step 1: Write engine and environment tests**

Append to `tests/test_api.py`:

```python
class TestEngineEndpoints:
    def test_status_when_stopped(self):
        resp = client.get("/api/engine/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False

    def test_start_engine(self):
        resp = client.post("/api/engine/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        # Stop it for other tests
        client.post("/api/engine/stop")

    def test_start_already_running_returns_409(self):
        client.post("/api/engine/start")
        resp = client.post("/api/engine/start")
        assert resp.status_code == 409
        client.post("/api/engine/stop")

    def test_stop_not_running_returns_409(self):
        resp = client.post("/api/engine/stop")
        assert resp.status_code == 409

    def test_flatten(self):
        client.post("/api/engine/start")
        resp = client.post("/api/engine/flatten")
        assert resp.status_code == 200
        assert resp.json()["status"] == "flattened"


class TestEnvironmentEndpoints:
    def test_get_environment(self):
        resp = client.get("/api/environment")
        assert resp.status_code == 200
        assert resp.json()["environment"] in ("demo", "live")

    def test_set_environment(self):
        resp = client.put("/api/environment", json={"environment": "live"})
        assert resp.status_code == 200
        assert resp.json()["environment"] == "live"
        # Reset
        client.put("/api/environment", json={"environment": "demo"})

    def test_set_invalid_environment(self):
        resp = client.put("/api/environment", json={"environment": "staging"})
        assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py::TestEngineEndpoints -v 2>&1 | head -20`

Expected: ERRORS — `ModuleNotFoundError: No module named 'server.mock_data'`

- [ ] **Step 3: Implement mock_data.py**

Create `server/mock_data.py`:

```python
"""
Mock data generators for Phase 1 dashboard.
Produces realistic-looking trading data when engine is "running".
Replaced by real engine data in Phase 2.
"""

import random
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# Module-level simulated state
_positions: dict[str, dict | None] = {"RSI": None, "IB": None, "MOM": None}
_daily_pnl: float = 0.0
_monthly_pnl: float = 0.0
_signals: list[dict] = []
_trades: list[dict] = []

_SL_PTS = {"RSI": 10, "IB": 10, "MOM": 15}
_TP_PTS = {"RSI": 100, "IB": 120, "MOM": 100}
_REASONS = {
    "RSI": lambda s: f"RSI(5) = {random.uniform(20, 34):.1f} < 35"
    if s == "Buy"
    else f"RSI(5) = {random.uniform(66, 80):.1f} > 65",
    "IB": lambda s: f"Close {'>' if s == 'Buy' else '<'} IB {'high' if s == 'Buy' else 'low'}",
    "MOM": lambda _: "Bar range > ATR(14), volume > SMA(20)",
}


def get_engine_status(running: bool) -> dict:
    accounts = [{"name": a["name"], "connected": running} for a in _get_configured_accounts()]
    if not accounts:
        accounts = [{"name": "No accounts configured", "connected": False}]
    return {
        "running": running,
        "can_trade": running,
        "daily_pnl": round(_daily_pnl, 2),
        "monthly_pnl": round(_monthly_pnl, 2),
        "daily_limit": -3000.0,
        "monthly_limit": -4500.0,
        "daily_limit_hit": _daily_pnl <= -3000,
        "monthly_limit_hit": _monthly_pnl <= -4500,
        "positions": {k: v for k, v in _positions.items()},
        "pending_signals": 0,
        "connected_accounts": accounts,
    }


def _get_configured_accounts() -> list[dict]:
    try:
        from server.config_store import get_accounts
        return get_accounts()
    except Exception:
        return []


def generate_ws_batch(running: bool, tick: int) -> list[dict]:
    global _daily_pnl, _monthly_pnl
    now = datetime.now(ET).isoformat()
    messages = []

    # Always send status
    messages.append({"type": "status", "data": get_engine_status(running), "timestamp": now})

    if not running:
        return messages

    base_price = 21500 + random.uniform(-50, 50)

    # Position updates every tick
    for strategy, pos in _positions.items():
        if pos is not None:
            current = base_price + random.uniform(-5, 5)
            if pos["side"] == "Buy":
                pnl = (current - pos["entry_price"]) * 2.0 * pos["contracts"]
            else:
                pnl = (pos["entry_price"] - current) * 2.0 * pos["contracts"]
            pos["current_price"] = round(current, 2)
            pos["pnl"] = round(pnl, 2)
            if tick % 5 == 0:
                pos["bars_held"] += 1
            messages.append(
                {"type": "position", "data": {**pos, "strategy": strategy}, "timestamp": now}
            )

    # Every ~16s: maybe generate a signal
    if tick % 8 == 3 and random.random() > 0.5:
        strategy = random.choice(["RSI", "IB", "MOM"])
        if _positions[strategy] is None:
            side = random.choice(["Buy", "Sell"])
            signal = {
                "strategy": strategy,
                "side": side,
                "contracts": 3,
                "reason": _REASONS[strategy](side),
                "price": round(base_price, 2),
            }
            _signals.append({**signal, "timestamp": now})
            if len(_signals) > 50:
                _signals.pop(0)
            messages.append({"type": "signal", "data": signal, "timestamp": now})

    # Every ~20s: maybe fill a signal (open position)
    if tick % 10 == 7:
        for strategy in ["RSI", "IB", "MOM"]:
            if _positions[strategy] is None and random.random() > 0.6:
                side = random.choice(["Buy", "Sell"])
                fill_price = round(base_price + random.uniform(-2, 2), 2)
                sl_pts = _SL_PTS[strategy]
                tp_pts = _TP_PTS[strategy]
                sl = fill_price - sl_pts if side == "Buy" else fill_price + sl_pts
                tp = fill_price + tp_pts if side == "Buy" else fill_price - tp_pts

                _positions[strategy] = {
                    "side": side,
                    "entry_price": fill_price,
                    "current_price": fill_price,
                    "contracts": 3,
                    "pnl": 0.0,
                    "bars_held": 0,
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                }
                fill = {
                    "strategy": strategy,
                    "side": side,
                    "contracts": 3,
                    "fill_price": fill_price,
                    "slippage": round(random.uniform(0, 1.5), 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                }
                _trades.append({**fill, "timestamp": now, "action": "entry"})
                if len(_trades) > 50:
                    _trades.pop(0)
                messages.append({"type": "fill", "data": fill, "timestamp": now})
                break

    # Every ~30s: maybe close a position
    if tick % 15 == 12:
        for strategy in ["RSI", "IB", "MOM"]:
            pos = _positions[strategy]
            if pos is not None and random.random() > 0.5:
                exit_price = round(pos["current_price"], 2)
                if pos["side"] == "Buy":
                    pnl = (exit_price - pos["entry_price"]) * 2.0 * pos["contracts"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * 2.0 * pos["contracts"]

                _daily_pnl += pnl
                _monthly_pnl += pnl

                exit_data = {
                    "strategy": strategy,
                    "side": pos["side"],
                    "contracts": pos["contracts"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": random.choice(["SL", "TP", "MaxHold"]),
                    "bars_held": pos["bars_held"],
                }
                _trades.append({**exit_data, "timestamp": now, "action": "exit"})
                if len(_trades) > 50:
                    _trades.pop(0)
                _positions[strategy] = None
                messages.append({"type": "exit", "data": exit_data, "timestamp": now})
                break

    # P&L update every tick
    messages.append({
        "type": "pnl",
        "data": {
            "daily": round(_daily_pnl, 2),
            "monthly": round(_monthly_pnl, 2),
            "daily_limit": -3000.0,
            "monthly_limit": -4500.0,
        },
        "timestamp": now,
    })

    return messages
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && python -m pytest tests/test_api.py -v`

Expected: All tests PASS (TestConfigStore: 10, TestAccountEndpoints: 8, TestEngineEndpoints: 5, TestEnvironmentEndpoints: 3 = 26 total).

- [ ] **Step 5: Verify server starts**

Run: `cd /Users/berjourlian/berjquant/tradovate-executor && source venv/bin/activate && timeout 5 python -m uvicorn server.api:app --port 8000 2>&1 || true`

Expected: Output includes `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 6: Commit**

```bash
git add server/mock_data.py tests/test_api.py
git commit -m "feat: mock data generator and engine/environment endpoint tests"
```

---

### Task 5: Frontend Scaffold (Vite + React + TypeScript + Tailwind)

**Files:**
- Create: `dashboard/` (entire Vite project)

- [ ] **Step 1: Create Vite project**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor
npm create vite@latest dashboard -- --template react-ts
```

Expected: Project created in `dashboard/`.

- [ ] **Step 2: Install dependencies**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npm install
npm install react-router-dom
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 3: Configure Tailwind**

Replace `dashboard/src/index.css` with:

```css
@import "tailwindcss";

/* Trading terminal dark theme */
body {
  @apply bg-gray-950 text-gray-100 antialiased;
  font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Monospace for numbers */
.font-mono {
  font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: rgb(17 24 39);
}

::-webkit-scrollbar-thumb {
  background: rgb(55 65 81);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgb(75 85 99);
}
```

- [ ] **Step 4: Configure Vite with proxy and Tailwind plugin**

Replace `dashboard/vite.config.ts` with:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 8080,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
});
```

- [ ] **Step 5: Clean up Vite defaults**

Delete these files if they exist:
- `dashboard/src/App.css`
- `dashboard/src/assets/react.svg`
- `dashboard/public/vite.svg`

Replace `dashboard/src/main.tsx` with:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
```

Replace `dashboard/src/App.tsx` with:

```tsx
import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Setup } from "./pages/Setup";
import { Settings } from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
```

Replace `dashboard/index.html` with:

```html
<!DOCTYPE html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Tradovate Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Create placeholder pages and Layout**

Create `dashboard/src/components/Layout.tsx`:

```tsx
import { NavLink, Outlet } from "react-router-dom";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/setup", label: "Setup" },
  { to: "/settings", label: "Settings" },
];

export function Layout() {
  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <nav className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold tracking-tight">Tradovate</h1>
          <p className="text-xs text-gray-500">HTF Swing v3</p>
        </div>
        <div className="flex-1 py-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `block px-4 py-2.5 text-sm transition-colors ${
                  isActive
                    ? "bg-gray-800 text-white border-r-2 border-blue-500"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
```

Create `dashboard/src/components/StatusBadge.tsx`:

```tsx
export function StatusBadge({
  running,
  connected,
}: {
  running: boolean;
  connected: boolean;
}) {
  if (!connected) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
        <span className="w-2 h-2 rounded-full bg-gray-600" />
        Disconnected
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs ${running ? "text-green-400" : "text-gray-400"}`}
    >
      <span
        className={`w-2 h-2 rounded-full ${running ? "bg-green-500 animate-pulse" : "bg-gray-600"}`}
      />
      {running ? "Running" : "Stopped"}
    </span>
  );
}
```

Create `dashboard/src/pages/Dashboard.tsx`:

```tsx
export function Dashboard() {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Dashboard</h2>
      <p className="text-gray-400">Loading...</p>
    </div>
  );
}
```

Create `dashboard/src/pages/Setup.tsx`:

```tsx
export function Setup() {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Account Setup</h2>
      <p className="text-gray-400">Loading...</p>
    </div>
  );
}
```

Create `dashboard/src/pages/Settings.tsx`:

```tsx
export function Settings() {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Settings</h2>
      <p className="text-gray-400">Loading...</p>
    </div>
  );
}
```

- [ ] **Step 7: Verify frontend starts**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npm run dev -- --host 2>&1 &
sleep 3
curl -s http://localhost:8080 | head -5
kill %1
```

Expected: HTML with `<div id="root">` returned.

- [ ] **Step 8: Commit**

```bash
git add dashboard/
git commit -m "feat: Vite + React + Tailwind frontend scaffold with routing"
```

---

### Task 6: TypeScript Types, API Client, and WebSocket Hook

**Files:**
- Create: `dashboard/src/types/index.ts`
- Create: `dashboard/src/api/client.ts`
- Create: `dashboard/src/hooks/useWebSocket.ts`

- [ ] **Step 1: Create TypeScript types**

Create `dashboard/src/types/index.ts`:

```typescript
export interface Account {
  name: string;
  username: string;
  password: string; // Masked "********" from API
  cid: number;
  sec: string;
  device_id: string;
  is_master: boolean;
  sizing_mode: "mirror" | "fixed" | "scaled";
  account_size: number;
  fixed_sizes: Record<string, number>;
}

export interface AccountCreate {
  name: string;
  username: string;
  password: string;
  cid?: number;
  sec?: string;
  device_id?: string;
  is_master?: boolean;
  sizing_mode?: string;
  account_size?: number;
  fixed_sizes?: Record<string, number>;
}

export interface AuthTestResult {
  success: boolean;
  account_id?: number;
  user_id?: number;
  error?: string;
}

export interface EngineStatus {
  running: boolean;
  can_trade: boolean;
  daily_pnl: number;
  monthly_pnl: number;
  daily_limit: number;
  monthly_limit: number;
  daily_limit_hit: boolean;
  monthly_limit_hit: boolean;
  positions: Record<string, Position | null>;
  pending_signals: number;
  connected_accounts: { name: string; connected: boolean }[];
}

export interface Position {
  strategy: string;
  side: "Buy" | "Sell";
  entry_price: number;
  current_price: number;
  contracts: number;
  pnl: number;
  bars_held: number;
  sl: number;
  tp: number;
}

export interface Signal {
  strategy: string;
  side: "Buy" | "Sell";
  contracts: number;
  reason: string;
  price: number;
  timestamp: string;
}

export interface Trade {
  strategy: string;
  side: "Buy" | "Sell";
  contracts: number;
  fill_price?: number;
  entry_price?: number;
  exit_price?: number;
  slippage?: number;
  pnl?: number;
  exit_reason?: string;
  bars_held?: number;
  sl?: number;
  tp?: number;
  timestamp: string;
  action: "entry" | "exit";
}

export interface PnL {
  daily: number;
  monthly: number;
  daily_limit: number;
  monthly_limit: number;
}

export interface WSMessage {
  type: "status" | "position" | "pnl" | "signal" | "fill" | "exit";
  data: unknown;
  timestamp: string;
}
```

- [ ] **Step 2: Create API client**

Create `dashboard/src/api/client.ts`:

```typescript
import type {
  Account,
  AccountCreate,
  AuthTestResult,
  EngineStatus,
} from "../types";

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(error.detail || resp.statusText);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

export const api = {
  // Accounts
  getAccounts: () => request<Account[]>("/accounts"),

  createAccount: (data: AccountCreate) =>
    request<Account>("/accounts", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  updateAccount: (name: string, data: Partial<AccountCreate>) =>
    request<Account>(`/accounts/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  deleteAccount: (name: string) =>
    request<void>(`/accounts/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  // Auth
  testAuth: (name: string) =>
    request<AuthTestResult>("/auth/test", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  // Engine
  startEngine: () =>
    request<{ status: string }>("/engine/start", { method: "POST" }),

  stopEngine: () =>
    request<{ status: string }>("/engine/stop", { method: "POST" }),

  flattenAll: () =>
    request<{ status: string }>("/engine/flatten", { method: "POST" }),

  getStatus: () => request<EngineStatus>("/engine/status"),

  // Environment
  getEnvironment: () => request<{ environment: string }>("/environment"),

  setEnvironment: (env: string) =>
    request<{ environment: string }>("/environment", {
      method: "PUT",
      body: JSON.stringify({ environment: env }),
    }),
};
```

- [ ] **Step 3: Create WebSocket hook**

Create `dashboard/src/hooks/useWebSocket.ts`:

```typescript
import { useEffect, useRef, useState, useCallback } from "react";
import type { Position, PnL, Signal, Trade, EngineStatus } from "../types";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [positions, setPositions] = useState<Record<string, Position | null>>(
    {}
  );
  const [pnl, setPnl] = useState<PnL | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case "status":
          setStatus(msg.data as EngineStatus);
          break;
        case "position":
          setPositions((prev) => ({
            ...prev,
            [(msg.data as Position).strategy]: msg.data as Position,
          }));
          break;
        case "pnl":
          setPnl(msg.data as PnL);
          break;
        case "signal":
          setSignals((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Signal), timestamp: msg.timestamp },
          ]);
          break;
        case "fill":
          setTrades((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Trade), timestamp: msg.timestamp, action: "entry" as const },
          ]);
          break;
        case "exit":
          setTrades((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Trade), timestamp: msg.timestamp, action: "exit" as const },
          ]);
          // Clear the position
          setPositions((prev) => ({
            ...prev,
            [(msg.data as { strategy: string }).strategy]: null,
          }));
          break;
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected, status, positions, pnl, signals, trades };
}
```

- [ ] **Step 4: Verify TypeScript compilation**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit 2>&1
```

Expected: No errors (or only warnings about unused vars in placeholder pages).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/types/ dashboard/src/api/ dashboard/src/hooks/
git commit -m "feat: TypeScript types, API client, and WebSocket hook"
```

---

### Task 7: Setup Page (Account Management + Auth Testing)

**Files:**
- Rewrite: `dashboard/src/pages/Setup.tsx`

- [ ] **Step 1: Implement Setup page**

Replace `dashboard/src/pages/Setup.tsx` with:

```tsx
import { useState, useEffect, type FormEvent } from "react";
import { api } from "../api/client";
import type { Account, AccountCreate, AuthTestResult } from "../types";

const EMPTY_FORM: AccountCreate = {
  name: "",
  username: "",
  password: "",
  cid: 0,
  sec: "",
  device_id: "",
  is_master: false,
  sizing_mode: "mirror",
  account_size: 150000,
};

export function Setup() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [form, setForm] = useState<AccountCreate>({ ...EMPTY_FORM });
  const [editing, setEditing] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<
    Record<string, AuthTestResult>
  >({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");

  const reload = () => {
    api.getAccounts().then(setAccounts).catch(console.error);
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  };

  useEffect(() => {
    reload();
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (editing) {
        await api.updateAccount(editing, form);
      } else {
        await api.createAccount(form);
      }
      setForm({ ...EMPTY_FORM });
      setEditing(null);
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    }
  };

  const handleEdit = (acct: Account) => {
    setEditing(acct.name);
    setForm({
      name: acct.name,
      username: acct.username,
      password: "", // Don't populate masked password
      cid: acct.cid,
      sec: "",
      device_id: acct.device_id,
      is_master: acct.is_master,
      sizing_mode: acct.sizing_mode,
      account_size: acct.account_size,
    });
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete account "${name}"?`)) return;
    await api.deleteAccount(name);
    reload();
  };

  const handleTestAuth = async (name: string) => {
    setTestingName(name);
    try {
      const result = await api.testAuth(name);
      setTestResults((prev) => ({ ...prev, [name]: result }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [name]: {
          success: false,
          error: err instanceof Error ? err.message : "Failed",
        },
      }));
    } finally {
      setTestingName(null);
    }
  };

  const toggleEnv = async () => {
    const next = environment === "demo" ? "live" : "demo";
    await api.setEnvironment(next);
    setEnvironment(next);
  };

  const inputCls =
    "w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500";

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Account Setup</h2>
        <button
          onClick={toggleEnv}
          className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider ${
            environment === "demo"
              ? "bg-yellow-900/50 text-yellow-300 border border-yellow-700"
              : "bg-red-900/50 text-red-300 border border-red-700"
          }`}
        >
          {environment}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
          <h3 className="font-semibold mb-4">
            {editing ? `Edit: ${editing}` : "Add Account"}
          </h3>
          {error && (
            <div className="mb-3 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded p-2">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              className={inputCls}
              placeholder="Account name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              disabled={!!editing}
              required
            />
            <input
              className={inputCls}
              placeholder="Tradovate username"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required
            />
            <input
              className={inputCls}
              type="password"
              placeholder={editing ? "New password (leave blank to keep)" : "Password"}
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required={!editing}
            />
            <div className="grid grid-cols-2 gap-3">
              <input
                className={inputCls}
                type="number"
                placeholder="CID"
                value={form.cid || ""}
                onChange={(e) =>
                  setForm({ ...form, cid: parseInt(e.target.value) || 0 })
                }
              />
              <input
                className={inputCls}
                type="password"
                placeholder="API Secret"
                value={form.sec}
                onChange={(e) => setForm({ ...form, sec: e.target.value })}
              />
            </div>
            <input
              className={inputCls}
              placeholder="Device ID (auto-generated if empty)"
              value={form.device_id}
              onChange={(e) => setForm({ ...form, device_id: e.target.value })}
            />
            <div className="grid grid-cols-2 gap-3">
              <select
                className={inputCls}
                value={form.sizing_mode}
                onChange={(e) =>
                  setForm({ ...form, sizing_mode: e.target.value })
                }
              >
                <option value="mirror">Mirror</option>
                <option value="fixed">Fixed</option>
                <option value="scaled">Scaled</option>
              </select>
              <input
                className={inputCls}
                type="number"
                placeholder="Account size"
                value={form.account_size}
                onChange={(e) =>
                  setForm({
                    ...form,
                    account_size: parseFloat(e.target.value) || 0,
                  })
                }
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={form.is_master}
                onChange={(e) =>
                  setForm({ ...form, is_master: e.target.checked })
                }
                className="rounded bg-gray-800 border-gray-600"
              />
              Master account
            </label>
            <div className="flex gap-2">
              <button
                type="submit"
                className="flex-1 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium py-2 rounded transition-colors"
              >
                {editing ? "Update" : "Add Account"}
              </button>
              {editing && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(null);
                    setForm({ ...EMPTY_FORM });
                  }}
                  className="px-4 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Account List */}
        <div className="space-y-3">
          {accounts.length === 0 && (
            <p className="text-gray-500 text-sm">No accounts configured.</p>
          )}
          {accounts.map((acct) => {
            const testResult = testResults[acct.name];
            const isTesting = testingName === acct.name;
            return (
              <div
                key={acct.name}
                className="bg-gray-900 rounded-lg border border-gray-800 p-4"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="font-semibold">{acct.name}</span>
                    {acct.is_master && (
                      <span className="ml-2 text-xs bg-blue-900/50 text-blue-300 px-1.5 py-0.5 rounded">
                        MASTER
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500 uppercase">
                    {acct.sizing_mode}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mb-3">
                  <span>{acct.username}</span>
                  <span className="mx-2">·</span>
                  <span className="font-mono">
                    ${acct.account_size.toLocaleString()}
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleTestAuth(acct.name)}
                    disabled={isTesting}
                    className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors disabled:opacity-50"
                  >
                    {isTesting ? "Testing..." : "Test Connection"}
                  </button>
                  <button
                    onClick={() => handleEdit(acct)}
                    className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(acct.name)}
                    className="text-xs bg-gray-800 hover:bg-red-900/50 text-gray-400 hover:text-red-300 px-3 py-1.5 rounded transition-colors"
                  >
                    Delete
                  </button>
                </div>
                {testResult && (
                  <div
                    className={`mt-2 text-xs p-2 rounded ${
                      testResult.success
                        ? "bg-green-900/20 text-green-400 border border-green-800"
                        : "bg-red-900/20 text-red-400 border border-red-800"
                    }`}
                  >
                    {testResult.success
                      ? `Connected — Account ID: ${testResult.account_id}, User ID: ${testResult.user_id}`
                      : `Failed: ${testResult.error}`}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/pages/Setup.tsx
git commit -m "feat: Setup page with account CRUD and auth testing"
```

---

### Task 8: Dashboard Page (Engine Controls + Positions + P&L + Logs)

**Files:**
- Rewrite: `dashboard/src/pages/Dashboard.tsx`

- [ ] **Step 1: Implement Dashboard page**

Replace `dashboard/src/pages/Dashboard.tsx` with:

```tsx
import { useState } from "react";
import { api } from "../api/client";
import { useWebSocket } from "../hooks/useWebSocket";
import { StatusBadge } from "../components/StatusBadge";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { connected, status, positions, pnl, signals, trades } =
    useWebSocket();
  const [loading, setLoading] = useState("");

  const running = status?.running ?? false;

  const handleAction = async (
    action: "start" | "stop" | "flatten",
    fn: () => Promise<unknown>
  ) => {
    setLoading(action);
    try {
      await fn();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading("");
    }
  };

  const formatPnl = (value: number) => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}$${value.toFixed(2)}`;
  };

  const pnlColor = (value: number) =>
    value >= 0 ? "text-green-400" : "text-red-400";

  const pnlBar = (value: number, limit: number) => {
    // Percentage of limit used (0-100+)
    const pct = Math.min(Math.abs(value / limit) * 100, 100);
    const color =
      pct > 80 ? "bg-red-500" : pct > 50 ? "bg-yellow-500" : "bg-green-500";
    return { pct, color };
  };

  return (
    <div className="space-y-6">
      {/* Engine Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <StatusBadge running={running} connected={connected} />
          <span className="text-lg font-semibold">
            {running ? "Engine Running" : "Engine Stopped"}
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => handleAction("start", api.startEngine)}
            disabled={running || loading === "start"}
            className="px-4 py-2 text-sm font-medium rounded bg-green-700 hover:bg-green-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "start" ? "Starting..." : "Start"}
          </button>
          <button
            onClick={() => handleAction("stop", api.stopEngine)}
            disabled={!running || loading === "stop"}
            className="px-4 py-2 text-sm font-medium rounded bg-gray-700 hover:bg-gray-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "stop" ? "Stopping..." : "Stop"}
          </button>
          <button
            onClick={() => handleAction("flatten", api.flattenAll)}
            disabled={loading === "flatten"}
            className="px-4 py-2 text-sm font-medium rounded bg-red-700 hover:bg-red-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "flatten" ? "Flattening..." : "Flatten All"}
          </button>
        </div>
      </div>

      {/* Positions */}
      <div className="grid grid-cols-3 gap-4">
        {(["RSI", "IB", "MOM"] as const).map((strategy) => {
          const pos = positions[strategy] as Position | null | undefined;
          return (
            <div
              key={strategy}
              className="bg-gray-900 rounded-lg border border-gray-800 p-4"
            >
              <div className="flex justify-between items-center mb-3">
                <span className="font-semibold">{strategy}</span>
                {pos ? (
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-medium ${
                      pos.side === "Buy"
                        ? "bg-green-900/50 text-green-300"
                        : "bg-red-900/50 text-red-300"
                    }`}
                  >
                    {pos.side.toUpperCase()}
                  </span>
                ) : (
                  <span className="text-xs text-gray-600 uppercase">Flat</span>
                )}
              </div>
              {pos ? (
                <div className="space-y-1.5 text-sm">
                  <Row label="Entry" value={pos.entry_price.toFixed(2)} mono />
                  <Row
                    label="Current"
                    value={pos.current_price.toFixed(2)}
                    mono
                  />
                  <Row
                    label="P&L"
                    value={formatPnl(pos.pnl)}
                    mono
                    className={pnlColor(pos.pnl)}
                  />
                  <Row label="Bars" value={String(pos.bars_held)} mono />
                  <Row
                    label="SL / TP"
                    value={`${pos.sl.toFixed(2)} / ${pos.tp.toFixed(2)}`}
                    mono
                    className="text-gray-500"
                  />
                </div>
              ) : (
                <p className="text-gray-600 text-sm">No position</p>
              )}
            </div>
          );
        })}
      </div>

      {/* P&L + Copy Accounts */}
      <div className="grid grid-cols-2 gap-4">
        {/* P&L */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-4">Profit & Loss</h3>
          <div className="space-y-4">
            <PnLBar
              label="Daily"
              value={pnl?.daily ?? 0}
              limit={pnl?.daily_limit ?? -3000}
            />
            <PnLBar
              label="Monthly"
              value={pnl?.monthly ?? 0}
              limit={pnl?.monthly_limit ?? -4500}
            />
          </div>
        </div>

        {/* Copy Accounts */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-4">Accounts</h3>
          <div className="space-y-2">
            {(status?.connected_accounts ?? []).map((acct) => (
              <div
                key={acct.name}
                className="flex items-center justify-between text-sm"
              >
                <span className="text-gray-300">{acct.name}</span>
                <span
                  className={`flex items-center gap-1.5 text-xs ${acct.connected ? "text-green-400" : "text-gray-500"}`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${acct.connected ? "bg-green-500" : "bg-gray-600"}`}
                  />
                  {acct.connected ? "Connected" : "Disconnected"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Signal + Trade Logs */}
      <div className="grid grid-cols-2 gap-4">
        {/* Signal Log */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3">Signal Log</h3>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {signals.length === 0 && (
              <p className="text-gray-600 text-xs">No signals yet</p>
            )}
            {[...signals].reverse().map((sig: Signal, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs py-1 border-b border-gray-800/50"
              >
                <span className="text-gray-500 font-mono w-16 shrink-0">
                  {sig.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span
                  className={`w-12 font-medium ${sig.side === "Buy" ? "text-green-400" : "text-red-400"}`}
                >
                  {sig.strategy}
                </span>
                <span
                  className={
                    sig.side === "Buy" ? "text-green-300" : "text-red-300"
                  }
                >
                  {sig.side}
                </span>
                <span className="text-gray-500 truncate">{sig.reason}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Trade Log */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3">Trade Log</h3>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {trades.length === 0 && (
              <p className="text-gray-600 text-xs">No trades yet</p>
            )}
            {[...trades].reverse().map((trade: Trade, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs py-1 border-b border-gray-800/50"
              >
                <span className="text-gray-500 font-mono w-16 shrink-0">
                  {trade.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span className="w-12 font-medium text-gray-300">
                  {trade.strategy}
                </span>
                <span
                  className={`w-10 ${trade.action === "entry" ? "text-blue-400" : "text-orange-400"}`}
                >
                  {trade.action === "entry" ? "ENTRY" : "EXIT"}
                </span>
                {trade.action === "entry" ? (
                  <>
                    <span className="font-mono text-gray-300">
                      @{trade.fill_price?.toFixed(2)}
                    </span>
                    <span className="text-gray-500">
                      slip: {trade.slippage?.toFixed(2)}
                    </span>
                  </>
                ) : (
                  <>
                    <span className="text-gray-500">
                      {trade.exit_reason}
                    </span>
                    <span
                      className={`font-mono ${(trade.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}
                    >
                      {formatPnl(trade.pnl ?? 0)}
                    </span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* Helper components */

function Row({
  label,
  value,
  mono,
  className,
}: {
  label: string;
  value: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-400">{label}</span>
      <span className={`${mono ? "font-mono" : ""} ${className ?? ""}`}>
        {value}
      </span>
    </div>
  );
}

function PnLBar({
  label,
  value,
  limit,
}: {
  label: string;
  value: number;
  limit: number;
}) {
  const pct = Math.min(Math.abs(value / limit) * 100, 100);
  const barColor =
    value >= 0
      ? "bg-green-500"
      : pct > 80
        ? "bg-red-500"
        : pct > 50
          ? "bg-yellow-500"
          : "bg-blue-500";
  const textColor = value >= 0 ? "text-green-400" : "text-red-400";

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <div className="flex gap-3">
          <span className={`font-mono ${textColor}`}>
            {value >= 0 ? "+" : ""}${value.toFixed(2)}
          </span>
          <span className="text-gray-600 font-mono">
            / ${limit.toFixed(0)}
          </span>
        </div>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${value >= 0 ? 0 : pct}%` }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/pages/Dashboard.tsx
git commit -m "feat: Dashboard page with positions, P&L, signal/trade logs"
```

---

### Task 9: Settings Page, Dev Runner, and Final Integration

**Files:**
- Rewrite: `dashboard/src/pages/Settings.tsx`
- Create: `run_dashboard.py`

- [ ] **Step 1: Implement Settings page**

Replace `dashboard/src/pages/Settings.tsx` with:

```tsx
import { useState, useEffect } from "react";
import { api } from "../api/client";

const STRATEGIES = [
  {
    name: "RSI Extremes",
    key: "RSI",
    params: [
      { label: "RSI Period", value: "5" },
      { label: "Oversold", value: "35" },
      { label: "Overbought", value: "65" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "10 pts" },
      { label: "Take Profit", value: "100 pts" },
      { label: "Max Hold", value: "5 bars (75 min)" },
    ],
  },
  {
    name: "IB Breakout",
    key: "IB",
    params: [
      { label: "IB Window", value: "9:30 – 10:00 ET" },
      { label: "Range Filter", value: "P25 – P75 (50 day)" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "10 pts" },
      { label: "Take Profit", value: "120 pts" },
      { label: "Max Hold", value: "15 bars (225 min)" },
      { label: "Max/Day", value: "1" },
    ],
  },
  {
    name: "Momentum Bars",
    key: "MOM",
    params: [
      { label: "ATR Period", value: "14" },
      { label: "EMA Period", value: "21" },
      { label: "Vol SMA", value: "20" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "15 pts" },
      { label: "Take Profit", value: "100 pts" },
      { label: "Max Hold", value: "5 bars (75 min)" },
    ],
  },
];

const SESSION = [
  { label: "Session Start", value: "9:30 AM ET" },
  { label: "No New Entries", value: "4:30 PM ET" },
  { label: "Flatten Time", value: "4:45 PM ET" },
  { label: "Daily Loss Limit", value: "-$3,000" },
  { label: "Monthly Loss Limit", value: "-$4,500" },
  { label: "Timezone", value: "US/Eastern" },
];

const CONTRACT = [
  { label: "Symbol", value: "MNQ (Micro Nasdaq 100)" },
  { label: "Front Month", value: "MNQM6 (June 2026)" },
  { label: "Tick Size", value: "0.25 pts = $0.50/contract" },
  { label: "Point Value", value: "$2.00/contract/point" },
];

export function Settings() {
  const [environment, setEnvironment] = useState("demo");

  useEffect(() => {
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Settings</h2>

      {/* Environment */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="font-semibold">Environment</h3>
            <p className="text-sm text-gray-400">
              Current: {environment.toUpperCase()}
            </p>
          </div>
          <span
            className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider ${
              environment === "demo"
                ? "bg-yellow-900/50 text-yellow-300 border border-yellow-700"
                : "bg-red-900/50 text-red-300 border border-red-700"
            }`}
          >
            {environment}
          </span>
        </div>
      </div>

      {/* Strategies */}
      <div>
        <h3 className="font-semibold mb-3">Strategy Parameters</h3>
        <div className="grid grid-cols-3 gap-4">
          {STRATEGIES.map((strat) => (
            <div
              key={strat.key}
              className="bg-gray-900 rounded-lg border border-gray-800 p-4"
            >
              <h4 className="font-medium text-sm mb-3">{strat.name}</h4>
              <div className="space-y-1.5">
                {strat.params.map((p) => (
                  <div
                    key={p.label}
                    className="flex justify-between text-xs"
                  >
                    <span className="text-gray-400">{p.label}</span>
                    <span className="font-mono text-gray-200">{p.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Session Config */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3 text-sm">Session Rules</h3>
          <div className="space-y-1.5">
            {SESSION.map((p) => (
              <div key={p.label} className="flex justify-between text-xs">
                <span className="text-gray-400">{p.label}</span>
                <span className="font-mono text-gray-200">{p.value}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3 text-sm">Contract Info</h3>
          <div className="space-y-1.5">
            {CONTRACT.map((p) => (
              <div key={p.label} className="flex justify-between text-xs">
                <span className="text-gray-400">{p.label}</span>
                <span className="font-mono text-gray-200">{p.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create dev runner script**

Create `run_dashboard.py`:

```python
#!/usr/bin/env python3
"""
Dev runner — starts both the FastAPI backend and Vite frontend.
Backend: uvicorn on port 8000
Frontend: Vite dev server on port 8080 (proxies /api and /ws to backend)

Usage: python run_dashboard.py
"""

import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(ROOT, "dashboard")


def main():
    procs = []

    try:
        # Start backend
        print("Starting backend on :8000 ...")
        backend = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "server.api:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--reload-dir", "server",
            ],
            cwd=ROOT,
        )
        procs.append(backend)

        # Start frontend
        print("Starting frontend on :8080 ...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host"],
            cwd=DASHBOARD,
        )
        procs.append(frontend)

        print("\n  Dashboard:  http://localhost:8080")
        print("  API:        http://localhost:8000/docs")
        print("  Press Ctrl+C to stop both.\n")

        # Wait for either to exit
        for proc in procs:
            proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for proc in procs:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add .gitignore for generated files**

Create `dashboard/.gitignore`:

```
node_modules/
dist/
```

Append to root `.gitignore` (or create if absent):

```
.secret_key
venv/
__pycache__/
*.pyc
test_config.json
test_secret_key
logs/
dashboard/node_modules/
dashboard/dist/
```

- [ ] **Step 4: Run the full stack**

Run in one terminal:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor
source venv/bin/activate
python run_dashboard.py
```

Open `http://localhost:8080` in browser.

Expected:
- Sidebar navigation (Dashboard, Setup, Settings) renders with dark theme
- Dashboard shows engine stopped, empty positions, empty logs
- Setup page has form to add accounts
- Settings page shows strategy parameters
- Adding an account via Setup persists to config.json
- Start engine button makes mock data flow to dashboard via WebSocket

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/pages/Settings.tsx run_dashboard.py dashboard/.gitignore .gitignore
git commit -m "feat: Settings page, dev runner, and .gitignore"
```

---

## Summary

| Task | What it builds | Tests |
|------|---------------|-------|
| 1 | Backend deps + Pydantic schemas | Import verification |
| 2 | Config store (CRUD + Fernet encryption) | 10 unit tests |
| 3 | FastAPI app + account CRUD endpoints | 8 endpoint tests |
| 4 | Mock data generator + engine/env endpoints | 8 endpoint tests |
| 5 | Vite + React + Tailwind scaffold + routing | Dev server starts |
| 6 | TypeScript types, API client, WS hook | tsc --noEmit |
| 7 | Setup page (account form, list, auth test) | tsc --noEmit |
| 8 | Dashboard (positions, P&L, logs, controls) | tsc --noEmit |
| 9 | Settings page, dev runner, gitignore | Full stack integration |

**Total: 9 tasks, 26 backend tests, full-stack dark-themed trading dashboard.**
