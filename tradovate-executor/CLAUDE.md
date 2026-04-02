# CLAUDE.md — HTF Swing v3 Hybrid v2 Tradovate Executor

## What This Is

A Python async application that auto-trades MNQ futures on Tradovate via their REST API + WebSocket. It runs 3 independent strategies on 15-minute bars and copies fills to N linked Tradovate accounts. Built for LucidFlex 150K prop firm accounts.

## Architecture

Single market data WebSocket → tick-to-15m bar aggregation → indicator calculation → signal engine (3 strategies) → order executor (master) → copy engine (N accounts). One risk manager enforces P&L limits and EOD flatten across all accounts.

```
Ticks → MarketDataEngine → SignalEngine → OrderExecutor (master) → CopyEngine (copies)
                                              ↓
                                         RiskManager (P&L limits, EOD flatten)
```

## File Map

| File | Lines | Purpose |
|---|---|---|
| `config.py` | 237 | All settings, strategy params, API URLs, account configs, sizing modes |
| `auth_manager.py` | 177 | Multi-account token lifecycle (obtain, renew, expiry) |
| `websocket_client.py` | 202 | Tradovate WS protocol (o/h/a/c frames), reconnection, heartbeat |
| `indicators.py` | 106 | RSI(5), ATR(14), EMA(21), SMA — Wilder's smoothing, matches backtest |
| `market_data.py` | 241 | Tick → 15m bar aggregation, IB range tracking, indicator state |
| `signal_engine.py` | 321 | 3 strategies, position tracking per strategy, max-hold flatten |
| `order_executor.py` | 506 | Market entry → poll fill → bracket at fill price → OCO link |
| `copy_engine.py` | 168 | Replicate fills to all linked accounts with per-account sizing |
| `risk_manager.py` | 191 | Daily/monthly P&L limits, EOD flatten at 4:45 ET, session enforcement |
| `trade_logger.py` | 199 | CSV trade log with slippage tracking |
| `position_sync.py` | 229 | Reconnect recovery — query positions/orders, rebuild state |
| `app.py` | 565 | Main orchestrator — startup, WebSocket handlers, signal→execution flow |

## The 3 Strategies

**RSI Extremes:** RSI(5) < 35 → buy, > 65 → sell. 3 contracts. SL 10pts, TP 100pts. Max hold 5 bars. Fires every bar RSI is in the zone (level check, not crossover).

**IB Breakout:** Capture 9:30-10:00 ET high/low. After 10:00: close > IB high → buy, close < IB low → sell. IB range must be P25-P75 of trailing 50-day ranges. Max 1 per day. SL 10pts, TP 120pts. Max hold 15 bars.

**Momentum Bars:** Bar range > ATR(14) AND volume > SMA(vol,20). Bullish + price > EMA(21) → buy. Bearish + price < EMA(21) → sell. SL 15pts, TP 100pts. Max hold 5 bars.

## Execution Rules (Non-Negotiable)

- Signal on bar N → entry at bar N+1 OPEN (market order)
- After fill: place bracket (SL stop + TP limit), linked as OCO
- All orders MUST have `isAutomated: true` (CME requirement)
- Max 1 position per strategy simultaneously
- No new entries after 4:30 PM ET
- Flatten ALL positions at 4:45 PM ET
- Max drawdown limit: -$4,500 (LucidFlex trailing drawdown) → stop trading for the month
- No daily loss limit — LucidFlex uses trailing max drawdown only
- Ctrl+C → flatten everything → exit

## Tradovate API

- Demo REST: `https://demo.tradovateapi.com/v1/`
- Live REST: `https://live.tradovateapi.com/v1/`
- Demo WS (orders): `wss://demo.tradovateapi.com/v1/websocket`
- Demo WS (market data): `wss://md-demo.tradovateapi.com/v1/websocket`
- Live WS (orders): `wss://live.tradovateapi.com/v1/websocket`
- Live WS (market data): `wss://md.tradovateapi.com/v1/websocket`
- Auth: POST `/auth/accesstokenrequest` → bearer token (1hr, renew via `/auth/renewaccesstoken`)
- Place order: POST `/order/placeorder` with `isAutomated: true`
- Cancel: POST `/order/cancelorder`
- Flatten: POST `/order/liquidateposition`
- WS protocol: frames are newline-delimited, first char = type (o=open, h=heartbeat, a=data array, c=close)
- WS auth: send `authorize\n0\n\n{access_token}` after receiving 'o' frame
- WS requests: `endpoint\nid\n\nbody_json`

## Contract Info

- Symbol: MNQ (Micro Nasdaq 100 futures)
- Current front month: MNQM6 (June 2026) — changes quarterly
- Tick size: 0.25 points = $0.50/contract
- Point value: $2.00/contract/point

## Copy Trading

The app mirrors master account fills to N linked Tradovate accounts. Each copy account has its own auth session and order executor. Sizing modes:
- **mirror** (default): same contracts as master (3/3/3)
- **fixed**: manually set per strategy per account
- **scaled**: proportional to account_size / 150,000 × master_size (floors to int, skips if 0)

Failed copies are logged and skipped (no retry). All accounts are Tradovate/CQG.

## LucidFlex Prop Firm Tiers

| Tier  | Profit Target | Max Drawdown |
|-------|--------------|--------------|
| 25K   | $1,250       | -$1,000      |
| 50K   | $3,000       | -$2,000      |
| 100K  | $6,000       | -$3,000      |
| 150K  | $9,000       | -$4,500      |

- Session: 9:30 AM – 4:45 PM ET
- **No daily loss limit** — trailing max drawdown only
- 90% payout

## Current State / What Needs Work

The core architecture is built and compiles clean. Key areas that need testing and iteration:

1. **Tradovate WebSocket message parsing** — the `_on_market_message` and `_on_order_message` handlers need testing against real demo API responses. Tradovate's message format varies and the exact JSON structure for quotes, fills, and position updates needs to be verified.

2. **OCO bracket linking** — `_link_oco` in order_executor.py uses `orderStrategy/startorderstrategy` with existing order IDs. This may need adjustment to match Tradovate's actual OCO API. Alternative: use `placeOCO` or manual cancel-on-fill logic.

3. **Market data subscription** — `md/subscribeQuote` and `md/getChart` payloads need verification against Tradovate docs. The chart subscription for historical bar seeding may need different params.

4. **Per-strategy position tracking with copy accounts** — currently `flatten_position` liquidates the entire position on the symbol. With 3 strategies potentially open (9 contracts total), flattening one strategy means closing only 3 of 9. This needs partial-close logic via `placeorder` with reducing qty instead of `liquidateposition`.

5. **Desktop UI** — complete. PyWebview + React dashboard. Run `python3 app_launcher.py`. Distributable via `bash build_app.sh` (macOS .dmg + Windows .exe via GitHub Actions).

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python app.py          # Uses config.json
python app.py --demo   # Force demo
python app.py --live   # Force live

# First run creates sample config.json — fill in credentials
```

## Style Notes

- Pure async Python (asyncio + websockets + httpx)
- No frameworks, no ORMs, minimal dependencies
- All indicator math must match backtest (Wilder's smoothing for RSI/ATR)
- Logging goes to stdout + daily log files in logs/
- Trade CSV format must be comparable to backtest output
