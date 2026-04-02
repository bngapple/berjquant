# Week Summary — March 26 – April 2, 2026

**Project:** HTF Swing v3 Hybrid v2 — `berjquant/tradovate-executor`  
**Prop Firm Target:** LucidFlex 150K ($4,500 trailing max drawdown, no daily limit, 90% payout)  
**System:** 3 strategies on 15-min MNQ bars → NinjaTrader execution → copy to N accounts

---

## What We Built

### March 26–27 — FastAPI Backend + React Frontend Scaffold

Started from scratch on the operator dashboard. Built the full stack foundation:

- **FastAPI backend** (`server/`) with account CRUD, auth testing, engine mock, and WebSocket broadcast
- **Config store** with Fernet-encrypted password storage at rest
- **Pydantic schemas** for all request/response types
- **Vite + React + TypeScript** frontend with Tailwind, routing, and a WebSocket hook
- **API client** (`src/api/client.ts`) with typed endpoints

### March 28 — Full Dashboard UI (Lucid Dark Theme)

Built every page of the dashboard in a single day:

- **Lucid Trading dark theme** — emerald accents, tight monospace layout, no noise
- **Dashboard** — stat strip (daily P&L, monthly P&L, win rate, active strategies), donut rings, equity curve with tabs, TopstepX P&L calendar, tabbed trade/signal logs
- **Cockpit** — fleet summary, per-account copy activity feed, inline fleet strip
- **Setup** — account table, slide-over form with modal, segmented controls for sizing mode
- **Settings** — strategy parameter table with grouped sections, session info strip
- **Calendar** page, sidebar with tooltips, seeded equity chart history
- Custom components: `DonutRing`, `DailyBarChart`, `EquityCurve`, `PnLCalendar`

### March 31 — Bloomberg Terminal Page + Bug Audit

- **Terminal page** — live candlestick chart (lightweight-charts), execution tape, strategy position cards. Made it the default landing view on app open.
- **11-bug audit pass** — full code review flagged and fixed 11 critical/high issues:
  - Hardcoded monthly loss limit in `engine_bridge.get_status()` (now reads from config)
  - NT bridge attribute errors in `engine_bridge` (wrong field names)
  - Position sync crash in NT mode when `copy_engine` is `None`
  - Plaintext credentials in config store (Fernet encryption applied)
  - Slippage stub returning `None` (fixed to return `0.0`)
  - StateFile integration gaps
- **5 remaining bugs** fixed in follow-up pass

### March 31 – April 1 — Desktop App + Windows Build

- **App launcher** (`app_launcher.py`) — wraps FastAPI + pywebview into a single macOS `.app`
- **PyInstaller spec** (`HTFExecutor.spec`) for bundling all assets
- **Build script** (`build_app.sh`) — one command produces a distributable macOS `.dmg`
- **GitHub Actions workflow** — cross-platform CI that builds Windows `.exe` (PyInstaller) on push to `main`
- Both targets produce self-contained executables with no Python install required

### April 2 — NinjaTrader Execution Layer + Config Refactor

This was the biggest architectural change of the week.

**Replaced Tradovate REST execution with NinjaTrader bridge:**

The original design placed bracket orders through Tradovate's REST API directly. This approach has known reliability issues (OCO linking, partial fills, order state race conditions). We rebuilt the execution layer so Python signals route to NinjaTrader 8, which handles all order management natively.

**Python side (`ninjatrader_bridge.py`):**
- TCP client connecting to the NT8 strategy
- Newline-delimited JSON protocol (ENTRY / FLATTEN / FLATTEN_ALL / PING commands)
- Async reconnect loop with exponential backoff
- Fill and exit callbacks → trade logger + copy engine
- Diagnostic connection errors (firewall, VM IP, strategy active, port match)

**NinjaScript side (`NinjaTrader/PythonBridge.cs`):**
- TCP server running inside NT8 as a NinjaScript Strategy
- Accepts JSON commands from Python, executes `EnterLong`/`EnterShort` with `SetStopLoss` + `SetProfitTarget` brackets
- Reports fills and bracket exits back via the same socket
- Runs `FlattenAll` at configured EOD time
- **Fixed compile errors for NT8.1:**
  - Removed `Newtonsoft.Json` entirely (not available in NT8's NinjaScript compiler sandbox)
  - Replaced all `JObject` usage with inline JSON helpers (`BuildJson`, `Jq`, `Jd`) using stdlib only (`System.Text.RegularExpressions`, `System.Globalization`)
  - Fixed `[NinjaScriptProperty]` and `[Display(...)]` attribute namespaces
  - Fixed `out _` C# 7 discard → named variable (NT8 compiler targets C# 6)

**Multi-account NT config (`config.py`):**
- Added `NTAccountConfig` dataclass (host + port per account)
- `NTConfig` now holds a named `accounts` dict, `default_atm_template`, timeouts, symbol
- `AppConfig.save()/load()` read/write `"ninjatrader"` key with backward-compat fallback for old `"nt"` key
- `config.json` updated with placeholder account names and setup instructions

**Risk manager additions:**
- Added `daily_loss_limit: Optional[float]` to `SessionConfig` (default `None` = disabled, per LucidFlex rules)
- Added `daily_limit_hit` flag, triggers flatten and halts trading if set
- Daily reset clears `daily_limit_hit` but preserves `monthly_limit_hit`

**Signal engine fix:**
- IB breakout `_ib_traded_today` flag now set at signal generation time, not fill time (prevents duplicate signals on the same bar)

**Test suite:**
- Fixed all pre-existing test failures (stale disk state isolation, EMA/RSI math corrections, `min_contracts=0` for floor-to-zero sizing)
- **263 tests passing, 0 failures**

**Diagnostic tooling:**
- `scripts/test_nt_connection.py` — standalone TCP connectivity tester (stdlib only, reads `config.json`, tests each NT account)

---

## Current Architecture

```
15-min MNQ bars (Tradovate WebSocket)
    → MarketDataEngine (tick aggregation, indicators)
    → SignalEngine (RSI / IB / MOM)
    → NinjaTraderBridge (Python TCP client)
        → PythonBridge.cs (NT8 NinjaScript TCP server)
            → NinjaTrader order management (brackets, OCO, fills)
        ← fill/exit callbacks
    → CopyEngine (mirror fills to N accounts)
    → RiskManager (trailing max drawdown, EOD flatten)
    → TradeLogger (CSV)
```

**Operator Interface:** PyWebview desktop app → React dashboard → FastAPI WebSocket

---

## What's Left Before Live

1. **NinjaTrader setup on Windows VM** — install PythonBridge.cs, enable automated trading, verify TCP port, confirm account name matches exactly
2. **Demo run** — 2–4 weeks paper trading to verify fill reports, copy accuracy, and EOD flatten behavior
3. **Tradovate WebSocket verification** — market data parsing against real API responses (noted as unverified in CLAUDE.md)
4. **OCO bracket confirmation** — NT8 handles brackets natively; verify SL/TP prices reported in fills match Python's expectations
5. **LucidFlex eval** — fund 150K account, run live

---

## Key Numbers (Backtest)

| Metric | Value |
|--------|-------|
| Strategies | RSI Extremes + IB Breakout + Momentum Bars |
| Bars | 15-minute MNQ, 2020–2026 (in-sample 2024) |
| Y2 OOS P&L | $7,140/month |
| Y2 OOS months profitable | 13/13 |
| Y2 OOS max drawdown | -$1,962 |
| Monte Carlo eval pass rate | 100% |
| Monte Carlo blow-up rate | 0% |
| Median days to pass LucidFlex 150K | ~25 |
