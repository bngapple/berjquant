# BerjQuant

BerjQuant is a futures-trading workspace with two main code areas:

1. The root research engine for backtests, Monte Carlo analysis, signal development, and validation.
2. [`tradovate-executor/`](tradovate-executor/) for the live desktop executor and NinjaTrader bridge workflow.

## Repo Layout

```text
engine/               Core research engine: pipeline, backtester, metrics, risk
signals/              Signal library used by the research stack
monte_carlo/          Monte Carlo search and validation tooling
validation/           Walk-forward, regime, and out-of-sample validation
live/                 Root-level live/paper trading experiments
strategies/           Strategy definitions and research variants
config/               Prop firm rules, sessions, and shared configuration
data/                 Raw and processed market data
reports/              Generated analysis outputs
tradovate-executor/   Desktop trading app, FastAPI backend, React UI, NT bridge
ANTHONY/              Pine and strategy-research snapshots
anthony 2/            Older mirrored research snapshot kept for reference
```

## Root Project Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v
python run_backtest.py
```

## Tradovate Executor Quick Start

```bash
cd tradovate-executor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./venv/bin/pytest -q
./venv/bin/python app_launcher.py
```

## Notes

- The root of this repo contains many research entry-point scripts (`run_*.py`). They are intentionally kept as experiment runners unless clearly superseded.
- The live executor now supports NinjaTrader-only operation through the bridge in [`tradovate-executor/NinjaTrader/`](tradovate-executor/NinjaTrader/).
- Local scratch repos and archive artifacts are ignored at the root so the main repo stays publishable.
