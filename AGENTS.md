# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_engine.py -v

# Run a single test class or function
pytest tests/test_engine.py::TestRiskManager -v
pytest tests/test_engine.py::TestRiskManager::test_session_bounds -v

# Run a basic backtest (uses synthetic data if data/processed/ is empty)
python run_backtest.py

# Run with a specific Parquet file
python run_backtest.py data/processed/MNQ/1m/2025-01.parquet
```

## Architecture Overview

This is the **MCQ Engine** — a Monte Carlo Quantitative trading engine for NQ/MNQ futures, designed for prop firm accounts. It generates, backtests, stress-tests, and validates intraday strategies.

### Core Data Flow

```
Raw CSV (MotiveWave) → DataPipeline → Parquet → VectorizedBacktester → Monte Carlo → WalkForward
```

### Key Concepts

**Strategy Protocol** (`engine/backtester.py`)
All strategies must implement four methods: `compute_signals(data)`, `get_stop_loss()`, `get_take_profit()`, `get_position_size()`. `compute_signals` takes a `dict[str, pl.DataFrame]` keyed by timeframe (`"1m"`, `"5m"`, etc.) and must return a DataFrame with boolean columns `entry_long`, `entry_short`, `exit_long`, `exit_short`. Entries fire on bar N but fill on bar N+1 open (lookahead-free design).

**Backtester** (`engine/backtester.py` — `VectorizedBacktester`)
Walks bars chronologically. At each bar it: (1) resets daily risk tracking on new date, (2) flattens at EOD, (3) fills pending entries at the open, (4) checks overnight gap fills, (5) updates trailing stops, (6) checks SL/TP, (7) processes exit signals, (8) queues new entry signals. `RiskManager.pre_trade_check()` is always called before filling — it cannot be bypassed.

**Risk Manager** (`engine/risk_manager.py`)
Enforces prop firm rules. Checks (in order): kill switch, session bounds (`config/sessions.yaml`), event blackout windows (`config/events_calendar.yaml`), daily loss limit, max drawdown, max contracts, EOD proximity. Kill switch triggers at 80% of daily loss limit and blocks all entries for the rest of the day. Drawdown type can be `trailing`, `eod`, or `static` — this changes when `high_water_mark` is updated.

**Signal Registry & Generator** (`signals/registry.py`, `strategies/generator.py`)
Signals are registered in `SignalRegistry` with their parameter ranges, categories (`trend`, `momentum`, `volatility`, `volume`, `orderflow`, `price_action`, `time_filter`), and types (`entry`, `filter`, `indicator`). `StrategyGenerator` builds `GeneratedStrategy` objects by taking a cartesian product of entry signal combos × filter combos × exit rule variations × sizing rule variations. Entry combos enforce that each signal must come from a different category (prevents duplicate trend+trend combos). Strategies can be serialized to/from JSON via `to_dict()`/`from_dict()`.

**Monte Carlo Simulator** (`monte_carlo/simulator.py`)
Takes a list of historical trades (P&L array) and stress-tests via: bootstrap resampling of trade sequence, parameter jitter (±15%), slippage perturbation (1–6 extra ticks), and gap injection (2% chance per trade). Runs in parallel via `multiprocessing`. Key output metrics: `prop_firm_pass_rate`, `probability_of_ruin`, `median_return`, `composite_score`.

**Walk-Forward Validation** (`validation/walk_forward.py`)
Splits data into rolling train/test windows. Key metric: **Walk-Forward Efficiency = OOS Sharpe / IS Sharpe** (>0.5 = acceptable, >0.7 = strong). Also tracks `pct_profitable_windows` and `sharpe_stability`. Strategies are flagged as overfit if WFE < 0.3 or too few OOS windows are profitable.

### Data Model

- Primary data unit: Polars DataFrame with columns `timestamp, open, high, low, close, volume, tick_count`
- Processed data is stored partitioned by year-month as Parquet under `data/processed/<symbol>/<timeframe>/YYYY-MM.parquet`
- Raw CSV exports from MotiveWave go in `data/raw/` and are ingested via `MotiveWaveIngestor`
- Backtest results (runs + trades + daily summaries) are logged to SQLite at `results/trades.db`

### Config Files

- `config/prop_firms.yaml` — prop firm profiles (`topstep_50k`, `apex_50k`, `lucid_50k`, `topstep_150k`). Profile keys are passed as `BacktestConfig.prop_firm_profile`.
- `config/sessions.yaml` — active trading window, `flat_by` time, session segment definitions
- `config/events_calendar.yaml` — one-time and recurring news event blackout windows (e.g. CPI, NFP, FOMC). `blackout_buffer_minutes` controls the window around each event.

### Core Data Structures (`engine/utils.py`)

- `ContractSpec` — tick size, tick value, point value. Pre-defined as `MNQ_SPEC` and `NQ_SPEC`.
- `PropFirmRules` — frozen dataclass loaded from YAML. `kill_switch_threshold` is automatically 80% of `daily_loss_limit`.
- `AccountState` — mutable state tracking balance, daily P&L, high water mark, open position.
- `BacktestConfig` — immutable run config (symbol, prop firm profile, dates, slippage ticks).
- `BacktestResult` — trades list + equity curve. Call `calculate_metrics(result.trades, capital)` separately to populate `result.metrics`.

### `run_*.py` Scripts

The repo contains many `run_*.py` scripts at the root. These are standalone experiment scripts (backtests, evolution runs, breed/crossover experiments, HTF swing strategy variations) and are not part of the importable library. They all follow the pattern of loading config/data, constructing a `VectorizedBacktester` or specialized runner, and printing/saving results.
