# MCQ Engine — Monte Carlo Quantitative Trading Engine

Strategy research engine that generates, tests, and validates intraday trading strategies for NQ/MNQ futures using Monte Carlo simulation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run a sample backtest (once you have data)
python run_backtest.py
```

## Architecture

```
engine/          Core: data pipeline, backtester, risk manager, metrics
signals/         Signal library (Phase 2)
monte_carlo/     MC simulation engine (Phase 3)
validation/      Walk-forward, regime detection (Phase 4)
live/            Paper/live trading (Phase 5)
strategies/      Strategy definitions (JSON + Python)
config/          Prop firm rules, sessions, events calendar
data/            Raw → processed → features (Parquet)
```

## Data Pipeline

1. Export MNQ/NQ data from MotiveWave as CSV
2. Place files in `data/raw/`
3. Run ingestion pipeline to clean and convert to Parquet

## Prop Firm Profiles

Configured in `config/prop_firms.yaml`:
- **Topstep 50K** — $1,000 daily loss limit, $2,000 trailing drawdown
- **Apex 50K** — $1,100 daily loss limit, $2,500 trailing drawdown
- **Lucid 50K** — $1,000 daily loss limit, $2,000 trailing drawdown
