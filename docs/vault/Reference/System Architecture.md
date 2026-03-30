---
date: 2026-03-29
type: reference
tags: [berjquant, architecture, engine]
---

# System Architecture

## Core Pipeline

```
Raw Data (MotiveWave CSV / Databento / YFinance)
  → Data Pipeline (ingest, clean, resample to Parquet)
    → Vectorized Backtesting (lookahead-free, bar N+1 fills)
      → Monte Carlo Stress Testing (10K sims, bootstrap + jitter + gaps)
        → Walk-Forward Validation (IS/OOS split, overfit detection)
          → Paper/Live Trading (real-time signal engine + risk enforcement)
```

## Engine Components

| Component | File | Purpose |
|-----------|------|---------|
| Data Structures | `engine/utils.py` | ContractSpec, PropFirmRules, AccountState, Position, Trade, BacktestConfig |
| Data Pipeline | `engine/data_pipeline.py` | MotiveWave CSV ingestion, cleaning, session tagging, bar resampling |
| Data Fetcher | `engine/data_fetcher.py` | YFinance integration for NQ=F proxy data |
| Backtester | `engine/backtester.py` | VectorizedBacktester with Strategy protocol, SlippageModel |
| Risk Manager | `engine/risk_manager.py` | 7-step pre-trade validation, kill switch, event blackouts |
| Metrics | `engine/metrics.py` | Performance analytics, TradeLogger (SQLite), Sharpe/PF/DD |
| Parallel Runner | `engine/parallel_runner.py` | Multi-process batch execution, composite scoring |

## Backtester Execution Flow (per bar)

1. Day boundary? → Reset daily P&L, kill switch, trades_today
2. EOD flatten time? → Close all positions at market
3. Pending entry from previous bar? → Fill at this bar's open + slippage
4. Overnight gap check → If open gaps beyond SL, fill at open (worst case)
5. Update trailing stop → Ratchet stop if profit activation reached
6. Check SL/TP → If both hit on same bar, use open-distance heuristic
7. Check exit signals → Close on signal
8. Check entry signals → Queue for next bar (lookahead-free)
9. Update equity curve → Record intrabar unrealized P&L

## Risk Manager — 7 Pre-Trade Checks

1. **Kill switch**: At 80% of daily loss limit → block ALL entries for rest of day
2. **Session bounds**: Only 08:00–17:00 ET
3. **Event blackout**: ±15 min of NFP, CPI, FOMC, etc.
4. **Daily loss limit**: Block if daily P&L ≤ limit
5. **Max drawdown**: Block if current DD ≤ limit
6. **Max contracts**: Enforce position size caps
7. **EOD proximity**: No new entries within 5 min of 17:00 ET

## Signal Library (30+ functions)

| Category | Signals |
|----------|---------|
| Trend | EMA Crossover, EMA Slope, EMA Ribbon, Linear Regression, Heikin-Ashi, Supertrend |
| Momentum | RSI, MACD, Stochastic |
| Volatility | ATR, Bollinger Bands, Keltner Channels |
| Volume | VWAP, Volume Delta |
| Orderflow | Delta Divergence, Absorption |
| Price Action | Session Levels, Previous Day Levels |
| Time Filters | Time of Day, Day of Week |

## Validation Framework

- **Walk-Forward**: Rolling 60-day train / 20-day test. WFE = OOS Sharpe / IS Sharpe (>0.5 good, >0.7 great)
- **Regime Analysis**: Trending/Ranging/High-Vol/Low-Vol classification. Sensitivity 0-1 (lower = more robust)
- **Overfitting Detection**: Parameter sensitivity ±10%, time stability, IS/OOS degradation
- **Monte Carlo**: 10K sims, bootstrap + slippage perturbation + P&L jitter + gap injection

## Slippage Model

- Base: 2 ticks per side
- +2 ticks if volume < 100 (thin market)
- +1 tick if off-hours (before 9 AM or after 4 PM ET)

See [[BERJQUANT_STATE]] for complete details.
