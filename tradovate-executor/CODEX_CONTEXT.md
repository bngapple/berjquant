# Codex Context — HTF Swing v3 Hybrid v2

**Last updated:** 2026-04-02  
**Repo:** `berjquant/tradovate-executor` (GitHub: bngapple/berjquant)  
**Working directory:** `/Users/berjourlian/berjquant/tradovate-executor`  
**Python:** 3.12, venv at `./venv`  
**OS:** macOS (Apple Silicon) — dev machine. Production: Windows VM (NinjaTrader).

---

## What This Is

A fully async Python trading engine that executes 3 strategies on 15-minute MNQ (Micro Nasdaq) futures bars, targeting LucidFlex 150K prop firm accounts.

**Flow:**
```
Tradovate WebSocket (ticks)
    → MarketDataEngine (tick → 15m bar + indicator state)
    → SignalEngine (RSI / IB Breakout / Momentum — evaluate on bar close)
    → [pending signal queue]
    → [next bar open] → execute
    → NinjaTraderBridge (Python TCP client)
        ↓ JSON over TCP
        → PythonBridge.cs (NT8 NinjaScript Strategy — TCP server on Windows VM)
            → NinjaTrader order management (brackets, OCO fills, SL/TP)
        ← fill / exit callbacks (JSON over same TCP)
    → RiskManager (trailing max drawdown, EOD flatten)
    → TradeLogger (CSV)
    → CopyEngine (mirror fills to N linked accounts) [disabled in NT mode]
```

**Operator Interface:**
```
app_launcher.py (PyWebview desktop app)
    → FastAPI server (server/api.py)
    → React dashboard (dashboard/src/)
    ← WebSocket events (live P&L, positions, signals, bars)
```

---

## Directory Structure

```
tradovate-executor/
├── app.py                      # Main TradovateExecutor class — wires everything together
├── app_launcher.py             # PyWebview desktop app launcher
├── config.py                   # All dataclasses: AppConfig, NTConfig, strategy params, etc.
├── auth_manager.py             # Tradovate OAuth token lifecycle (obtain, renew, multi-account)
├── websocket_client.py         # Tradovate WS protocol (o/h/a/c frames), reconnect, heartbeat
├── market_data.py              # Tick → 15m bar aggregation, indicator state, IB range tracking
├── signal_engine.py            # 3 strategies, per-strategy position tracking, max-hold flatten
├── order_executor.py           # Tradovate REST execution (used only if NOT in NT mode)
├── ninjatrader_bridge.py       # NT TCP bridge — drop-in replacement for OrderExecutor
├── copy_engine.py              # Mirror fills to N linked accounts (disabled in NT mode)
├── risk_manager.py             # Daily/monthly P&L limits, EOD flatten, trailing drawdown
├── trade_logger.py             # CSV trade log with entry/exit/P&L
├── position_sync.py            # Reconnect recovery — rebuild state from open positions
├── NinjaTrader/
│   └── PythonBridge.cs         # NT8 NinjaScript Strategy (TCP server, order execution)
├── server/
│   ├── api.py                  # FastAPI app — REST endpoints + WebSocket broadcast
│   ├── config_store.py         # Account CRUD with Fernet-encrypted credential storage
│   ├── engine_bridge.py        # DashboardExecutor (extends TradovateExecutor for dashboard events)
│   ├── schemas.py              # Pydantic schemas for all API request/response types
│   ├── account_tracker.py      # Per-account P&L tracking for dashboard
│   └── history.py              # Equity curve + trade history storage
├── dashboard/
│   └── src/
│       ├── App.tsx             # Router (Dashboard / Calendar / Cockpit / Setup / Settings)
│       ├── pages/
│       │   ├── Dashboard.tsx   # Main operating view
│       │   ├── Cockpit.tsx     # Execution / account overview
│       │   ├── Setup.tsx       # NT-only account and bridge setup flow
│       │   └── Settings.tsx    # Strategy params + session info
│       ├── components/
│       │   └── Layout.tsx      # Sidebar + outlet with WS context
│       ├── hooks/
│       │   └── useWebSocket.ts # WS hook — auto-reconnect, typed messages
│       └── api/
│           └── client.ts       # Typed API client (axios)
├── tests/
│   ├── test_signal_engine.py   # 896 lines — comprehensive signal logic tests
│   ├── test_risk_manager.py    # 578 lines — daily/monthly limits, resets
│   ├── test_dry_run.py         # Integration: engine startup, sizing, copy logic
│   ├── test_ninjatrader_bridge.py  # NT bridge connect/disconnect/protocol tests
│   ├── test_api.py             # FastAPI endpoint tests
│   └── test_indicators.py      # RSI, EMA, ATR, SMA math tests
├── scripts/
│   └── test_nt_connection.py   # Standalone TCP connectivity tester (stdlib only)
├── config.json                 # Runtime config — edit with your NT account names + VM IP
├── HTFExecutor.spec            # PyInstaller spec for macOS .app build
├── build_app.sh                # Build macOS .dmg
└── .github/workflows/          # GitHub Actions: Windows .exe + macOS .dmg on push to main
```

---

## The 3 Strategies

All evaluate at 15-minute bar close. Signal is queued, executed at **next bar's open** (market order).

### Strategy 1: RSI Extremes (`signal_engine.py`)
- **Indicator:** RSI(5), Wilder's smoothing
- **Long:** RSI < 35 (oversold)
- **Short:** RSI > 65 (overbought)
- **Contracts:** 3
- **SL:** 10 pts, **TP:** 100 pts
- **Max hold:** 5 bars (75 min)
- **Fires every bar** the condition holds (not crossover)
- **One position at a time** per strategy

### Strategy 2: IB Breakout (`signal_engine.py`)
- **IB window:** 9:30–10:00 ET (tracks high/low)
- **Signal:** After 10:00 ET, bar close > IB high → Long; bar close < IB low → Short
- **IB filter:** Range must be P25–P75 of last 50 trading day ranges
- **Max 1 trade per day** (`_ib_traded_today` flag, set at signal generation)
- **Contracts:** 3, **SL:** 10 pts, **TP:** 120 pts, **Max hold:** 15 bars

### Strategy 3: Momentum Bars (`signal_engine.py`)
- **Conditions:** Bar range > ATR(14) AND bar volume > SMA(volume, 20)
- **Long:** Bullish bar AND close > EMA(21)
- **Short:** Bearish bar AND close < EMA(21)
- **Contracts:** 3, **SL:** 15 pts, **TP:** 100 pts, **Max hold:** 5 bars

---

## Execution Architecture: NinjaTrader Mode (Current)

When `config.nt` is set, `app.py` instantiates a `NinjaTraderBridge` instead of `OrderExecutor`. The copy engine is **disabled** in NT mode (set to `None`).

### Python Side: `ninjatrader_bridge.py`

- Async TCP client connecting to `PythonBridge.cs` on the Windows VM
- Persistent reconnect loop with `RECONNECT_DELAY = 5s`
- Periodic `PING` every `PING_INTERVAL = 10s`
- `place_entry_with_bracket()` sends ENTRY command, awaits fill confirmation with 30s timeout
- On timeout: marks position as WORKING, uses signal price as fallback fill price
- `flatten_position(strategy)` sends FLATTEN or FLATTEN_ALL
- `_exit_callback` fires `app._on_nt_exit()` which computes P&L, logs exit, marks strategy flat

### NT Side: `NinjaTrader/PythonBridge.cs`

- NinjaScript Strategy (NT8 NinjaScript, .NET 4.8, C# 6)
- **No external dependencies** — stdlib only (Newtonsoft removed; JSON done with Regex + StringBuilder)
- Listens on `TcpPort` (default 6000), accepts one Python client at a time
- Commands processed on NT strategy thread via `ConcurrentQueue` drained in `OnBarUpdate`
- Entry: `EnterLong/EnterShort` + `SetStopLoss(Points)` + `SetProfitTarget(Points)` → bracket managed by NT
- Exit detection: `OnExecutionUpdate` — `OrderType.StopMarket` → SL, `OrderType.Limit` → TP
- EOD flatten: runs at `EodFlattenTime` (default 164500 = 4:45 PM ET)

### TCP Protocol (newline-delimited JSON)

**Python → NT:**
```json
{"cmd":"ENTRY","id":"req-000001","strategy":"RSI","side":"Buy","qty":3,"sl_pts":10.0,"tp_pts":100.0}
{"cmd":"FLATTEN","id":"req-000002","strategy":"RSI"}
{"cmd":"FLATTEN_ALL","id":"req-000003"}
{"cmd":"PING"}
```

**NT → Python:**
```json
{"type":"fill","id":"req-000001","strategy":"RSI","side":"Buy","qty":3,"fill_price":19500.25,"sl_price":19490.25,"tp_price":19600.25}
{"type":"exit","strategy":"RSI","exit_type":"SL","fill_price":19490.25,"qty":3}
{"type":"exit","strategy":"RSI","exit_type":"TP","fill_price":19600.25,"qty":3}
{"type":"exit","strategy":"RSI","exit_type":"EOD","fill_price":19490.00,"qty":0}
{"type":"ack","id":"req-000002"}
{"type":"pong"}
{"type":"error","id":"req-000001","message":"Missing strategy or side"}
```

---

## Configuration

### `config.json` (runtime, edit before running)

```json
{
  "environment": "live",
  "symbol": "MNQM6",
  "accounts": [
    {
      "name": "EXACT_TRADOVATE_ACCOUNT_NAME",
      "username": "your@email.com",
      "password": "",
      "device_id": "unique-device-id",
      "is_master": true,
      "sizing_mode": "mirror",
      "account_size": 150000,
      "monthly_loss_limit": -4500.0
    }
  ],
  "ninjatrader": {
    "accounts": {
      "EXACT_NT_ACCOUNT_NAME": {
        "host": "VM_HOST_ONLY_IPV4",
        "port": 6000
      }
    },
    "default_atm_template": "MNQ_2R",
    "order_timeout_seconds": 10,
    "status_timeout_seconds": 5,
    "reconnect_max_backoff_seconds": 30,
    "symbol": "MNQU6"
  }
}
```

**Critical:** NT account name must exactly match the NinjaTrader Accounts tab (case-sensitive). VM host is the host-only network adapter IPv4 from `ipconfig` inside the Windows VM.

### Key Config Dataclasses (`config.py`)

```python
AppConfig
  .environment        # Environment.DEMO or Environment.LIVE
  .symbol             # "MNQM6" — update quarterly
  .rsi                # RSIParams (period=5, oversold=35, overbought=65, contracts=3, sl=10, tp=100, max_hold=5)
  .ib                 # IBParams (contracts=3, sl=10, tp=120, max_hold=15)
  .mom                # MOMParams (contracts=3, sl=15, tp=100, max_hold=5)
  .session            # SessionConfig (session_start=09:30, no_new_entries_after=16:30, flatten=16:45, monthly_limit=-4500)
  .accounts           # list[AccountConfig]
  .nt                 # Optional[NTConfig] — if set, NT mode

NTConfig
  .accounts           # dict[str, NTAccountConfig] — name → host/port
  .symbol             # NT instrument symbol (different month may differ from Tradovate)

NTAccountConfig
  .host               # Windows VM IP
  .port               # TCP port (matches TcpPort in PythonBridge)

SessionConfig
  .monthly_loss_limit = -4500.0   # LucidFlex 150K trailing max drawdown
  .daily_loss_limit   = None      # None = disabled (LucidFlex has no daily limit)
```

---

## Running the System

### Engine only (headless)
```bash
cd /Users/berjourlian/berjquant/tradovate-executor
source venv/bin/activate
python app.py            # config.json environment
python app.py --demo     # force demo
python app.py --live     # force live
```

### Desktop App (full UI)
```bash
python app_launcher.py   # opens PyWebview window with React dashboard
```

### Dashboard dev mode
```bash
# Terminal 1 — FastAPI backend
python server/api.py

# Terminal 2 — Vite frontend
cd dashboard && npm run dev
# → http://localhost:5173
```

### Tests
```bash
source venv/bin/activate
python -m pytest tests/ -v
# 263 tests, 0 failures (as of 2026-04-02)
```

### NT connection test (no engine needed)
```bash
python scripts/test_nt_connection.py
# reads config.json, tests TCP connectivity to each NT account
```

### Build distributable
```bash
bash build_app.sh              # macOS .dmg
# Windows .exe built automatically via GitHub Actions on push to main
```

---

## Key Module Details

### `risk_manager.py`
- Tracks `daily_pnl`, `monthly_pnl` across all strategies
- `monthly_limit_hit`: halts trading for the month, survives daily reset
- `daily_limit_hit`: halts trading for the day (optional — disabled by default for LucidFlex)
- `can_trade()`: returns False if either limit hit OR outside session hours
- EOD flatten fires at `session.flatten_time` (16:45 ET)
- State persisted to `risk_state.json` for restart recovery

### `signal_engine.py`
- `evaluate(state: MarketState)` → `list[Signal]`
- `Signal.contracts == 0` means "flatten this strategy" (max hold reached)
- `mark_filled(strategy, side)` / `mark_flat(strategy)` track open positions
- Per-strategy `PositionState` tracks side, bars_held, bar_entered
- `_ib_traded_today` flag set at signal generation (prevents duplicate IB signals same day)

### `market_data.py`
- Aggregates ticks into 15m bars in real time
- `on_tick(price, volume, ts)` → emits `on_bar_complete(state)` on each bar close
- `ingest_historical_bar()` for seeding from Tradovate chart history on startup
- `MarketState` holds: `current_bar`, `completed_bars[]`, `rsi`, `atr`, `ema21`, `ib_high`, `ib_low`, `ib_range_history[]`

### `auth_manager.py`
- Multi-account Tradovate OAuth (username/password → JWT)
- `get_master_session()` → `AuthSession` for the `is_master=True` account
- `get_copy_sessions()` → all non-master sessions
- Tokens expire after 60 min; auto-renewal fires at 55 min via background task
- `on_token_renewed` callback → `app._on_token_renewed()` pushes fresh tokens to WS clients

### `websocket_client.py`
- Tradovate WS protocol: frames are `o` (open), `h` (heartbeat), `a[...]` (data array), `c` (close)
- Auth on connect: `authorize\n0\n\n{token}`
- Subscriptions: `endpoint\nid\n\nbody_json`
- Auto-reconnect with exponential backoff

### `order_executor.py` (Tradovate REST mode — not used in NT mode)
- `place_entry_with_bracket()` → market entry → poll fill → bracket at fill price via OCO
- `on_fill_event()` → called from `app._handle_fill()` when WS fill event arrives
- OCO linking via `orderStrategy/startorderstrategy` with existing SL/TP order IDs
- **Note:** The OCO linking has NOT been verified against real Tradovate API. May need adjustment.

### `copy_engine.py` (disabled in NT mode)
- Mirrors master fills to N linked accounts
- Per-account `OrderExecutor` instances, each with their own auth session
- Sizing modes: MIRROR (same contracts), FIXED (manual), SCALED (proportional to account_size / 150k)

### `server/config_store.py`
- Fernet-encrypted credential storage at `server/accounts.json`
- REST endpoints: GET/POST/PUT/DELETE `/api/accounts`
- `_encrypt(plaintext)` / `_decrypt(ciphertext)` — key derived from machine fingerprint

### `server/api.py` (FastAPI)
- `GET /api/status` → engine status (P&L, positions, running state)
- `GET /api/accounts` / `POST /api/accounts` / etc. → account CRUD
- `POST /api/engine/start` / `/api/engine/stop`
- `POST /api/engine/flatten`
- `GET /api/history/bars` → last 200 15m bars for chart
- `WS /ws` → real-time event stream: `bar`, `signal`, `fill`, `exit`, `pnl`, `position`, `status`

### `server/engine_bridge.py`
- `DashboardExecutor` extends `TradovateExecutor`
- Overrides `_on_market_message`, `_on_bar_complete`, `_handle_fill`, `_on_nt_exit` to push events to `event_queue`
- `_push(event)` → non-blocking `event_queue.put_nowait()`
- `get_status()` → reads monthly_limit from master account config (NOT hardcoded)
- `_build_config(raw)` → constructs `AppConfig` from the dashboard's account store

---

## Tradovate API Notes (Partially Verified)

**Market data WS message format:**
- Quote ticks arrive as: `data["entries"]["Trade"]["price"]`
- Historical bars arrive as: `data["bars"][n]` with `open/high/low/close/upVolume/downVolume`
- Timestamp format: ISO 8601 with `Z` suffix → replace with `+00:00` for `fromisoformat()`

**Order WS message format:**
- Frame type `e` field: `"order"`, `"fill"`, `"position"`
- Actual data in `d` field
- Fill data: `orderId`, `price`, `qty`
- Order data: `id`, `ordStatus` (`"Rejected"`, etc.), `text`

**⚠️ UNVERIFIED:**
- `md/subscribeQuote` and `md/getChart` exact payload format
- OCO bracket linking in `order_executor.py` (`orderStrategy/startorderstrategy`)
- Exact JSON structure of Tradovate order/fill events
- These need to be tested against the real Tradovate demo API before going live

---

## NinjaTrader Setup (Windows VM)

1. Copy `NinjaTrader/PythonBridge.cs` to `Documents\NinjaTrader 8\bin\Custom\Strategies\`
2. NT8 → Tools → Edit NinjaScript → Strategy → Compile
3. Open a chart with MNQM6, 1-min bars
4. Add strategy: PythonBridge, set `TcpPort = 6000`, `EodFlattenTime = 164500`
5. Enable automated trading (AT button in toolbar must be green)
6. Find VM's host-only network IP: run `ipconfig` in Windows, look for "Host-only Adapter" IPv4
7. Update `config.json` `ninjatrader.accounts[name].host` with that IP
8. Ensure Windows Firewall allows inbound TCP on port 6000
9. Account name in `config.json` must match NinjaTrader Accounts tab exactly (case-sensitive)

**Common issues:**
- "Automated trading disabled" in NT log → click AT button to enable
- Connection refused → check VM is running + strategy is active on chart + firewall
- Account name mismatch → warning logged by Python; check NT Accounts tab

---

## LucidFlex Prop Firm Rules

| Tier | Profit Target | Max Drawdown |
|------|--------------|--------------|
| 25K  | $1,250       | -$1,000      |
| 50K  | $3,000       | -$2,000      |
| 100K | $6,000       | -$3,000      |
| **150K** | **$9,000** | **-$4,500** |

- Session: 9:30 AM – 4:45 PM ET
- **No daily loss limit** — trailing max drawdown only
- `daily_loss_limit = None` in `SessionConfig` (disabled by default)
- 90% payout on profits

---

## Backtest Results (Why We Built This)

- **Strategies:** RSI Extremes + IB Breakout + Momentum Bars on 15-min MNQ
- **In-sample:** 2024 only
- **Y2 OOS (2025):** $7,140/month avg, 13/13 months profitable, max DD -$1,962
- **Blind OOS (2022-2024):** $3,242/month avg, 20/28 months profitable
- **Monte Carlo (10,000 runs):** 100% LucidFlex 150K eval pass rate, 0% blow-up
- **Key insight:** Strategy only works in high-vol regimes (2020+). 2018-2019 was losing. Current NQ vol regime is favorable.
- **Honest execution:** 2-tick slippage, $3.78/contract round-trip, next-bar fills, no trailing stop tricks

---

## What's Left Before Live

1. **NT setup on Windows VM** — install PythonBridge.cs, enable AT, verify TCP
2. **Demo paper trading** — 2-4 weeks to verify fills, copy accuracy, EOD behavior
3. **Tradovate WS message parsing** — test against real demo API (payloads unverified)
4. **OCO bracket linking** — verify `orderStrategy/startorderstrategy` works or switch to `placeOCO`
5. **Contract symbol rollover** — `MNQM6` expires June 2026; update to `MNQU6` (September) before expiry

---

## Test Suite (263 passing, 0 failures)

```
tests/test_signal_engine.py   — signal generation logic, position tracking, IB rules
tests/test_risk_manager.py    — limit triggers, daily reset, monthly persist, state isolation
tests/test_dry_run.py         — full engine init, sizing modes, copy math, P&L accumulation
tests/test_ninjatrader_bridge.py — TCP connect/disconnect, fill dispatch, timeout handling
tests/test_api.py             — FastAPI CRUD endpoints, engine start/stop
tests/test_indicators.py      — RSI(Wilder), EMA, ATR, SMA correctness
```

**Important:** RiskManager tests patch `_load_state` to prevent disk state contamination:
```python
with patch.object(RiskManager, '_load_state'):
    rm = RiskManager(session_config=..., on_flatten_all=None)
rm._save_state = lambda: None
```

---

## Front Month Symbol

| Period      | Symbol  |
|-------------|---------|
| Until Jun 2026 | MNQM6 |
| Jun–Sep 2026   | MNQU6 |
| Sep–Dec 2026   | MNQZ6 |

Update `config.json` `"symbol"` and `ninjatrader.symbol` before rollover.

---

## Contract Specs (MNQ)

- Tick size: 0.25 pts = $0.50/contract
- Point value: $2.00/contract
- 3 contracts = $6.00/point
- 10-point SL = $60/trade, 100-point TP = $600/trade (RSI/MOM)

---

## Dependencies

```
# Runtime
fastapi uvicorn pydantic
websockets httpx
pywebview
cryptography          # Fernet for credential encryption
pytz / zoneinfo       # Timezone handling

# Frontend
react vite tailwindcss
lightweight-charts    # Candlestick (replaces recharts for Terminal page)
axios

# Tests
pytest pytest-asyncio
```

Install: `pip install -r requirements.txt` (inside venv)

---

## Files NOT to Edit Without Understanding

- `NinjaTrader/PythonBridge.cs` — must compile in NT8 NinjaScript (C# 6, no Newtonsoft, no external deps)
- `config.py` — `NTConfig.accounts` is a `dict`, not a list. `NTAccountConfig` is frozen.
- `app.py` — `copy_engine` is `None` when in NT mode (guarded everywhere with `if self.copy_engine:`)
- `server/engine_bridge.py` — `_build_config()` reads `"ninjatrader"` key (new) with `"nt"` fallback (old)
