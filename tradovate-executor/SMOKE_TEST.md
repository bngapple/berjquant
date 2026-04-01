# Smoke Test Checklist

Manual test plan for the Tradovate Executor dashboard. Run after any significant changes.

## Prerequisites

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
source venv/bin/activate
python run_dashboard.py
# Open http://localhost:8080
```

## 1. Startup (No Credentials)

- [ ] Dashboard loads at http://localhost:8080 without errors
- [ ] All 5 sidebar nav links work: Dashboard, Calendar, Cockpit, Setup, Settings
- [ ] Header shows "Stopped" status with gray dot
- [ ] No console errors in browser DevTools

## 2. Dashboard Page (Engine Stopped, No History)

- [ ] Total P&L shows +$0.00
- [ ] Win Rate, Profit Factor show "--"
- [ ] Avg Win / Loss shows "--"
- [ ] Trades shows 0
- [ ] Equity curve shows "No data"
- [ ] Daily P&L bar chart shows "No data"
- [ ] Positions table shows RSI/IB/MOM all "Flat" with dashes
- [ ] Daily/Monthly limit bars show $0 / -$3000 and $0 / -$4500
- [ ] Signal Log shows "No signals yet"
- [ ] Trade Log shows "No trades yet"
- [ ] Fleet health strip is hidden (no alerts)

## 3. Calendar Page (No History)

- [ ] Monthly P/L shows $0.00
- [ ] Month/year header centered, < > arrows work
- [ ] Today button resets to current month
- [ ] Calendar grid shows Su-Sa columns with empty cells
- [ ] Week column shows dashes
- [ ] No console errors

## 4. Cockpit Page (No Accounts)

- [ ] Shows "No accounts configured" in table
- [ ] Fleet strip shows Connected: 0/0
- [ ] Copy Activity shows "No copy activity yet"

## 5. Setup Page — Add Account

- [ ] Click "+ Add Account" — slide-over panel opens from right
- [ ] Fill in: Name, Username, Password, CID, API Secret, Device ID
- [ ] Account Type segmented control: Eval / Funded
- [ ] Role segmented control: Master / Copy
- [ ] Sizing Mode: Mirror / Fixed / Scaled
- [ ] Risk Parameters section: Starting Balance, Profit Target, Max Drawdown
- [ ] Submit creates account, appears in table
- [ ] Account shows in table with correct Role badge
- [ ] Hover over row — X delete icon fades in
- [ ] Click X — confirmation modal appears
- [ ] Cancel dismisses modal, Delete removes account

## 6. Setup Page — Test Connection (Requires Real Credentials)

- [ ] Click "Test" on an account with valid Tradovate demo credentials
- [ ] Shows "Testing..." while in progress
- [ ] Shows green checkmark + "OK" on success
- [ ] Shows red X + error message on failure
- [ ] Error message is the real Tradovate error (not generic)

## 7. Engine Start (Requires Real Credentials)

- [ ] Click Start in header with a configured master account
- [ ] If auth fails: error returned to frontend, engine stays stopped
- [ ] If auth succeeds: header shows "Running" with green pulsing dot
- [ ] Cockpit account table shows green connected dots
- [ ] WebSocket starts receiving real status messages

## 8. Market Data (Requires Running Engine + Market Hours)

- [ ] If market closed: status shows "Engine Running" but no signals generate
- [ ] If market open: ticks flow in, 15m bars build
- [ ] Bar completion events appear (check browser DevTools WS frames)
- [ ] Indicators calculate (RSI, ATR, EMA values in bar events)

## 9. Signals and Fills (Requires Running Engine + Market Hours)

- [ ] When RSI/IB/MOM conditions met: signal appears in Signal Log
- [ ] Signal shows strategy, side, price, reason
- [ ] Fill appears in Trade Log with real fill price and slippage
- [ ] Position appears in Positions table with entry price, SL, TP
- [ ] P&L updates in real-time as price moves

## 10. Engine Stop

- [ ] Click Stop — engine stops, status shows "Stopped"
- [ ] All positions flattened
- [ ] Cockpit shows disconnected dots
- [ ] Dashboard reverts to historical data display

## 11. Flatten All

- [ ] Click "Flatten All" — confirmation modal appears
- [ ] Confirm — all positions liquidated on all accounts
- [ ] Engine remains running after flatten

## 12. Cockpit — Account Status

- [ ] Accounts show Day P&L, Total P&L, Trades Today, DD Used
- [ ] DD Used shows progress bar (green/yellow/red based on %)
- [ ] Click account row — detail panel expands below
- [ ] Detail shows: Balance, P&L, Drawdown bar, Remaining, Target Progress, Status badge
- [ ] Click again — detail panel collapses

## 13. Settings Page

- [ ] Session section shows trading hours, loss limits
- [ ] Contract section shows MNQM6 details
- [ ] Strategy params table shows RSI/IB/MOM columns
- [ ] Env badge (DEMO/LIVE) displays correctly

## 14. Historical Data (After Running with Real Trades)

- [ ] Stop engine — dashboard loads stats from CSV logs
- [ ] Total P&L, Win Rate, Profit Factor populated from trade history
- [ ] Equity curve shows cumulative P&L from CSV data
- [ ] Calendar shows daily P&L cells from CSV data
- [ ] Click a calendar day with trades — trade detail panel shows real trades

## 15. Health Check

- [ ] GET http://localhost:8000/api/health returns JSON
- [ ] Shows uptime, engine_running, event_queue_size, ws_clients

## 16. Backend Tests

```bash
python -m pytest tests/test_api.py -v
# All 31 tests should pass
```

## 17. Frontend Build

```bash
cd dashboard && npx tsc --noEmit  # No type errors
cd dashboard && npx vite build    # Build succeeds
```
