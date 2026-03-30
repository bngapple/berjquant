#!/usr/bin/env python3
"""
test_connection.py — Standalone Tradovate demo API integration test.

Sequence:
  1. Load credentials from config.json (master account)
  2. Authenticate via POST /auth/accesstokenrequest
  3. Print access token + account ID
  4. Connect to market data WebSocket, subscribe to MNQM6 quotes
  5. Print 30s of raw incoming messages
  6. Connect to order WebSocket, subscribe to user sync
  7. Print 10s of raw incoming messages
  8. Disconnect cleanly
"""

import asyncio
import json
import sys
import time

import httpx
import websockets


# ── Config ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config.json") as f:
        return json.load(f)


def get_master_account(config: dict) -> dict:
    for acct in config["accounts"]:
        if acct.get("is_master"):
            return acct
    # Fallback to first account
    return config["accounts"][0]


# ── REST Auth ───────────────────────────────────────────────────────────

async def authenticate(base_url: str, acct: dict) -> dict:
    """POST /auth/accesstokenrequest and return full response."""
    url = f"{base_url}/auth/accesstokenrequest"
    payload = {
        "name": acct["username"],
        "password": acct["password"],
        "appId": acct.get("app_id", "HTFSwing"),
        "appVersion": acct.get("app_version", "1.0.0"),
        "deviceId": acct.get("device_id", "test-connection-1"),
        "cid": acct.get("cid", 0),
        "sec": acct.get("sec", ""),
    }

    print(f"\n{'='*60}")
    print("STEP 1: Authenticating...")
    print(f"  URL: {url}")
    print(f"  User: {acct['username']}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.post(url, json=payload)
        print(f"\n  HTTP {resp.status_code}")
        data = resp.json()
        print(f"  Response keys: {list(data.keys())}")
        print(f"  Full response:\n{json.dumps(data, indent=2)}")

        if "errorText" in data:
            print(f"\n  AUTH FAILED: {data['errorText']}")
            sys.exit(1)

        resp.raise_for_status()
        return data


async def fetch_accounts(base_url: str, token: str) -> list:
    """GET /account/list to resolve numeric account ID."""
    url = f"{base_url}/account/list"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.get(url, headers=headers)
        resp.raise_for_status()
        accounts = resp.json()
        return accounts


# ── WebSocket Helpers ───────────────────────────────────────────────────

async def ws_listen(url: str, token: str, label: str, duration: float,
                    subscribe_cmds: list[tuple[str, dict]] = None):
    """
    Connect to a Tradovate WebSocket, authenticate, optionally subscribe,
    then print raw messages for `duration` seconds.
    """
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"  URL: {url}")
    print(f"  Duration: {duration}s")
    print(f"{'='*60}")

    msg_count = 0
    deadline = time.time() + duration

    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            close_timeout=5,
            max_size=2**22,
        ) as ws:
            # Wait for open frame
            first = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f"\n  [OPEN] {first}")

            # Authenticate
            auth_msg = f"authorize\n0\n\n{token}"
            await ws.send(auth_msg)
            print(f"  [SENT] authorize\\n0\\n\\n<token>")

            # Wait for auth response
            auth_resp = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f"  [AUTH RESP] {auth_resp[:500]}")

            # Send subscriptions
            if subscribe_cmds:
                for endpoint, body in subscribe_cmds:
                    body_str = json.dumps(body) if body else ""
                    msg = f"{endpoint}\n1\n\n{body_str}"
                    await ws.send(msg)
                    print(f"  [SENT] {endpoint} → {body_str[:200]}")

            # Start heartbeat in background
            async def heartbeat():
                while True:
                    await asyncio.sleep(2.5)
                    try:
                        await ws.send("[]")
                    except Exception:
                        break

            hb_task = asyncio.create_task(heartbeat())

            # Listen and print
            print(f"\n  Listening for {duration}s...\n")
            try:
                while time.time() < deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                    except asyncio.TimeoutError:
                        continue

                    msg_count += 1
                    ts = time.strftime("%H:%M:%S")

                    # Skip heartbeat frames for cleaner output (but count them)
                    if raw.strip() == "h":
                        if msg_count <= 5 or msg_count % 10 == 0:
                            print(f"  [{ts}] #{msg_count} HEARTBEAT")
                        continue

                    # Print data frames in full
                    if len(raw) > 2000:
                        print(f"  [{ts}] #{msg_count} ({len(raw)} bytes):")
                        print(f"    {raw[:2000]}...")
                    else:
                        print(f"  [{ts}] #{msg_count} ({len(raw)} bytes):")
                        print(f"    {raw}")

                    # Pretty-print JSON if it's a data frame
                    if raw.startswith("a"):
                        try:
                            parsed = json.loads(raw[1:])
                            print(f"    PARSED: {json.dumps(parsed, indent=4)}")
                        except json.JSONDecodeError:
                            pass

            finally:
                hb_task.cancel()

            print(f"\n  Done. {msg_count} messages received in {duration}s.")

    except Exception as e:
        print(f"\n  WebSocket error: {e}")
        raise


# ── Main ────────────────────────────────────────────────────────────────

async def main():
    config = load_config()
    env = config.get("environment", "demo")
    symbol = config.get("symbol", "MNQM6")
    acct = get_master_account(config)

    if env == "live":
        base_url = "https://live.tradovateapi.com/v1"
        ws_md_url = "wss://md.tradovateapi.com/v1/websocket"
        ws_order_url = "wss://live.tradovateapi.com/v1/websocket"
    else:
        base_url = "https://demo.tradovateapi.com/v1"
        ws_md_url = "wss://md-demo.tradovateapi.com/v1/websocket"
        ws_order_url = "wss://demo.tradovateapi.com/v1/websocket"

    print(f"Environment: {env.upper()}")
    print(f"Symbol: {symbol}")
    print(f"Account: {acct['name']}")

    # ── Step 1: Authenticate ──
    auth_data = await authenticate(base_url, acct)
    token = auth_data["accessToken"]

    print(f"\n  Access Token: {token[:20]}...{token[-10:]}")
    print(f"  User ID: {auth_data.get('userId')}")

    # ── Step 2: Fetch account ID ──
    print(f"\n{'='*60}")
    print("STEP 2: Fetching account list...")
    print(f"{'='*60}")

    accounts = await fetch_accounts(base_url, token)
    print(f"\n  Found {len(accounts)} account(s):")
    for a in accounts:
        print(f"    ID: {a.get('id')}  Name: {a.get('name')}  Status: {a.get('active')}")
        print(f"    Full: {json.dumps(a, indent=4)}")

    account_id = accounts[0]["id"] if accounts else None
    print(f"\n  Using account ID: {account_id}")

    # ── Step 3: Market data WebSocket — 30s of quotes ──
    await ws_listen(
        url=ws_md_url,
        token=token,
        label="Market Data WebSocket — MNQM6 quotes (30s)",
        duration=30,
        subscribe_cmds=[
            ("md/subscribeQuote", {"symbol": symbol}),
        ],
    )

    # ── Step 4: Order WebSocket — 10s of user sync ──
    await ws_listen(
        url=ws_order_url,
        token=token,
        label="Order WebSocket — user/syncrequest (10s)",
        duration=10,
        subscribe_cmds=[
            ("user/syncrequest", {"users": [auth_data.get("userId", 0)]}),
        ],
    )

    print(f"\n{'='*60}")
    print("ALL DONE — clean disconnect")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
