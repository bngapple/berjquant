# HTF Swing v3 Hybrid v2 — Tradovate Auto-Executor

Automated MNQ futures trading system running 3 independent strategies on 15-minute bars, with multi-account copy trading.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Tradovate WebSocket                        │
│              (single market data connection)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │ ticks
                       ▼
              ┌─────────────────┐
              │  Market Data    │  Aggregates ticks → 15m bars
              │    Engine       │  Calculates RSI, ATR, EMA, SMA, IB
              └────────┬────────┘
                       │ completed bar + indicators
                       ▼
              ┌─────────────────┐
              │  Signal Engine  │  RSI Extremes / IB Breakout / Momentum
              │  (3 strategies) │  Emits signals for next-bar execution
              └────────┬────────┘
                       │ signals
                       ▼
              ┌─────────────────┐     ┌──────────────────────┐
              │ Order Executor  │────▶│     Copy Engine       │
              │   (master)      │     │  (N copy accounts)    │
              └────────┬────────┘     └──────────┬───────────┘
                       │                          │
                       ▼                          ▼
              ┌─────────────────┐     ┌──────────────────────┐
              │  Risk Manager   │     │  Per-account sizing  │
              │  P&L / Limits   │     │  mirror/fixed/scaled │
              └─────────────────┘     └──────────────────────┘
```

## Modules

| File | Purpose |
|---|---|
| `config.py` | All settings: API URLs, strategy params, account configs, sizing modes |
| `auth_manager.py` | Multi-account token management with auto-renewal |
| `websocket_client.py` | Tradovate WS protocol handler with reconnection |
| `market_data.py` | Tick → 15m bar aggregation, indicator calculation, IB tracking |
| `indicators.py` | RSI(5), ATR(14), EMA(21), SMA(vol,20) — matches backtest math |
| `signal_engine.py` | 3 strategies, position tracking, max-hold flattening |
| `order_executor.py` | Market entry + bracket (SL+TP), flatten, cancel |
| `copy_engine.py` | Replicates master fills to all linked accounts |
| `risk_manager.py` | Daily/monthly P&L limits, EOD flatten, session enforcement |
| `trade_logger.py` | CSV logging with slippage tracking |
| `position_sync.py` | Reconnect recovery — syncs app state with live positions |
| `app.py` | Main orchestrator, startup/shutdown lifecycle |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Edit `config.json` with your Tradovate credentials for each account

3. Run:
   ```
   python app.py            # Uses config.json environment setting
   python app.py --demo     # Force demo mode
   python app.py --live     # Force live mode
   ```

4. Stop: `Ctrl+C` → flattens all positions, then exits

## Account Sizing Modes

- **mirror** (default): Copy accounts trade same size as master (3/3/3)
- **fixed**: Manually set contracts per strategy per account
- **scaled**: Auto-scale based on `account_size / 150000 * master_size`

## Strategy Parameters

| Strategy | Entry | SL | TP | Max Hold |
|---|---|---|---|---|
| RSI | RSI(5) < 35 buy, > 65 sell | 10 pts | 100 pts | 5 bars |
| IB | Break above/below IB range | 10 pts | 120 pts | 15 bars |
| MOM | Range > ATR, vol > SMA, EMA filter | 15 pts | 100 pts | 5 bars |

## Risk Limits

- Daily loss: -$3,000 → halt trading
- Monthly loss: -$4,500 → halt trading
- No entries after 4:30 PM ET
- Flatten all at 4:45 PM ET
- Ctrl+C → emergency flatten
