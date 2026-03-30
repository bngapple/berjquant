# Tradovate Executor — Project Context

## What This Is

An automated MNQ futures trading system (HTF Swing v3) with a web dashboard. The existing Python trading engine (`app.py`, `signal_engine.py`, `order_executor.py`, etc.) runs 3 strategies (RSI, IB Breakout, Momentum) on 15-minute bars. We built a FastAPI + React dashboard on top of it.

## Architecture

```
tradovate-executor/
├── Trading Engine (existing, untouched):
│   app.py, signal_engine.py, order_executor.py, market_data.py,
│   indicators.py, copy_engine.py, risk_manager.py, auth_manager.py,
│   websocket_client.py, position_sync.py, trade_logger.py, config.py
│
├── server/ (FastAPI backend — Phase 1):
│   api.py          — REST endpoints + WebSocket /ws/live
│   config_store.py — Account CRUD + Fernet password encryption
│   mock_data.py    — Simulated trading data for dashboard
│   schemas.py      — Pydantic request/response models
│
├── dashboard/ (React frontend — Vite + TypeScript + Tailwind v4 + recharts):
│   src/
│   ├── components/  — Layout, EquityCurve, DailyBarChart, DonutRing, PnLCalendar, FlattenModal
│   ├── pages/       — Dashboard, Calendar, Cockpit, Setup, Settings
│   ├── hooks/       — useWebSocket (single WS connection in Layout), useLayoutData (outlet context)
│   ├── types/       — All TS interfaces (Account, Position, Trade, Signal, PnL, WSData, etc.)
│   ├── api/         — REST client (fetch wrapper for all /api/* endpoints)
│   └── index.css    — Lucid Trading theme (#0d0d0d bg, #00d4aa accent)
│
├── tests/test_api.py — 26 backend tests (config store + CRUD + engine + environment)
├── run_dashboard.py  — Dev runner (starts uvicorn + vite simultaneously)
└── config.json       — Account credentials (passwords Fernet-encrypted at rest)
```

## How To Run

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
source venv/bin/activate
python run_dashboard.py
# Backend: http://localhost:8000 (API docs at /docs)
# Frontend: http://localhost:8080
# Tests: python -m pytest tests/test_api.py -v
```

## Backend API (server/api.py)

All working, 26 tests passing:
- `GET/POST/PUT/DELETE /api/accounts` — Account CRUD (passwords masked in responses)
- `POST /api/auth/test` — Test Tradovate auth for an account (hits real Tradovate demo API)
- `POST /api/engine/start|stop|flatten` — Engine controls (mock in Phase 1)
- `GET /api/engine/status` — Returns mock positions, P&L, account connections
- `GET/PUT /api/environment` — Demo/Live toggle
- `WS /ws/live` — Streams mock data every 2s (status, positions, pnl, signals, fills, exits)

## Frontend Pages (5 total)

### Dashboard (`/`)
- Top stat strip: Total P&L (hero, 32px), Win Rate + donut, Profit Factor + donut, Avg Win/Loss, Trades, Day Win %
- Equity curve (recharts AreaChart, 320px tall, time filter tabs 1W/1M/3M/ALL, seeds mock data when stopped)
- Daily P&L bar chart (recharts BarChart, 220px, emerald/red bars)
- Positions table (RSI/IB/MOM rows, active indicator, LONG/SHORT/FLAT)
- Daily + Monthly P&L limit bars
- Tabbed Signal Log / Trade Log (compact, 15 rows, white tab underline)

### Calendar (`/calendar`)
- Full-page TopstepX-style monthly P&L calendar
- Large day cells with dollar P&L values, color-coded backgrounds
- Weekly P&L totals in right column
- Month navigation with < > arrows
- Summary row: trading days, winners, losers, net P&L
- Click a day to expand and see that day's individual trades

### Cockpit (`/cockpit`)
- "LucidFlex 150K Fleet" header with inline fleet stats
- Account table: Status dot, Name, Role (Leader/Follower badges), Sizing, Position, Open P&L, Day P&L, Last Fill
- Leader row has emerald left border + elevated background
- Compact fleet summary strip: Connected, Open, Contracts, Day/Month P&L
- Copy Activity feed: scrolling log of copy operations with checkmark/X

### Setup (`/setup`)
- Account table: Name, Username, Env badge, Role badge, Sizing, Status (test result), Actions
- Master rows have emerald left border
- Hover row: X delete button fades in (opacity 0→1)
- "+ Add Account" button opens slide-over panel (400px from right)
- Form: inputs for all fields, segmented controls for Environment/Role/Sizing Mode
- "Test Connection" hits real Tradovate API, shows checkmark or X

### Settings (`/settings`)
- Session section: trading hours (9:30-4:45 ET), daily/monthly loss limits, emerald top border
- Contract info: MNQM6 details
- Strategy Parameters table: 3-column (RSI/IB/MOM) with alternating row striping

## Theme (Lucid Trading inspired)

CSS vars in `index.css`:
- `--bg: #0d0d0d` (near-black)
- `--panel: #161616` (dark gray surface)
- `--elevated: #1e1e1e` (hover/leader rows)
- `--accent: #00d4aa` (emerald green — used sparingly for P&L, equity line, connected dots, Start button)
- `--red: #ef4444` / `--amber: #f59e0b` / `--blue: #3b82f6`
- `--text: #e8e8e8` / `--text-muted: #6b7280` / `--text-dim: #3f3f46`
- `.panel` class: `background: var(--panel); border: 1px solid var(--border)`

Layout: 48px header (engine controls) + 56px icon sidebar (expands to 200px on hover) + content area. WebSocket connection lives in Layout, shared to pages via React Router Outlet context.

## Design Decisions

- **Single WS connection**: Layout.tsx calls useWebSocket(), passes data via `<Outlet context={ws} />`. Pages use `useLayoutData()` hook.
- **Mock data**: server/mock_data.py generates fake positions/trades/signals when engine is "running". Calendar and equity chart also seed client-side mock history for visual richness.
- **Password encryption**: Fernet symmetric encryption at rest in config.json. Key stored in `.secret_key`.
- **No real engine integration yet**: Engine start/stop/flatten are mock toggles. Phase 2 will wire up the real TradovateExecutor class.

## Commit History (25 commits)

```
6af9abe fix: add Calendar page, sidebar tooltips, seed equity chart
9cf81e8 fix: Dashboard stat hierarchy, chart heights, reduced green, tighter spacing
5ebb92f fix: Cockpit inline fleet strip, Setup rounded corners
5542cce feat: Setup page with account table, slide-over form, segmented controls
e37ac16 feat: Settings page with striped param table, emerald section borders
83c0b22 feat: Cockpit page with account table, copy activity feed, fleet summary
15fbe45 feat: Dashboard with stat strip, donut rings, equity curve, TopstepX calendar, tabbed logs
9c24f40 feat: DonutRing, DailyBarChart, EquityCurve with tabs, TopstepX calendar
f653fec feat: Lucid layout with header bar, 4-item sidebar, outlet WS context
92fe8e5 feat: Lucid Trading theme, WSData types, outlet context hook
--- (earlier: first redesign pass, Phase 1 build, initial scaffold) ---
4a0fecf feat: add backend dependencies and Pydantic schemas  (first commit)
```

## What Was Just Completed (Fix Pass)

Applied 3 commits addressing 7 specific problems:
1. Calendar extracted to its own `/calendar` page (full-page TopstepX style)
2. Stat strip hierarchy fixed (hero P&L, smaller donuts, varied widths)
3. Grid uniformity broken (varied padding, chart heights 320/220, compact logs)
4. Green overuse reduced (section labels gray, tab underlines white, sidebar border muted)
5. Cockpit panels consolidated (inline fleet strip, full-width copy activity)
6. Setup hover delete verified working
7. Bug fixes (seeded equity chart, Day Win % shows "--" when 0 trades, sidebar tooltips)

## What's Next

### Pending UI fix pass (user re-sent same list — these may need a second look):
- The user said "The UI looks artificial and AI-generated" again after the fixes were applied
- Some fixes may not have fully landed or may need further refinement
- Specifically re-check: stat strip appearance, calendar polish, green reduction, font weight variation

### Phase 2 (Engine Integration):
1. Wire real TradovateExecutor into server/api.py (start/stop actually runs the engine)
2. Replace mock_data.py with real WS data from the engine
3. Per-account position tracking for Cockpit
4. Real Tradovate WebSocket message parsing validation
5. OCO bracket linking verification
6. Partial-close logic for per-strategy flattening

### Phase 3 (Production):
- Real credential management (env vars or secrets manager)
- Mac desktop app packaging (Electron or Tauri)
- Logging dashboard (view log files from the UI)
- Alerting (Slack/Discord notifications on fills, limit hits)

## Key Files Quick Reference

| Need to... | File |
|---|---|
| Change theme colors | `dashboard/src/index.css` |
| Add/modify a page | `dashboard/src/pages/*.tsx` + `App.tsx` route |
| Change sidebar/header | `dashboard/src/components/Layout.tsx` |
| Modify API endpoints | `server/api.py` |
| Change mock data behavior | `server/mock_data.py` |
| Add/modify types | `dashboard/src/types/index.ts` |
| Change WS data handling | `dashboard/src/hooks/useWebSocket.ts` |
| Run backend tests | `python -m pytest tests/test_api.py -v` |
| Run TS type check | `cd dashboard && npx tsc --noEmit` |
