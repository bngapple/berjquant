"""Run a backtest with the MCQ Engine."""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from engine.utils import (
    BacktestConfig,
    MNQ_SPEC,
    load_prop_firm_rules,
    load_session_config,
    load_events_calendar,
)
from engine.data_pipeline import load_parquet
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics, print_metrics, TradeLogger
from strategies.ema_crossover import EMACrossoverStrategy

CONFIG_DIR = Path("config")
DATA_DIR = Path("data/processed")


def run_with_data(data_path: str | None = None):
    """Run backtest on real or synthetic data."""

    config = BacktestConfig(
        symbol="MNQ",
        prop_firm_profile="topstep_50k",
        start_date="2025-01-01",
        end_date="2025-12-31",
        slippage_ticks=2,
        initial_capital=50000.0,
    )

    # Load config
    prop_rules = load_prop_firm_rules(CONFIG_DIR, config.prop_firm_profile)
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)

    # Load data
    if data_path:
        df = pl.read_parquet(data_path)
    else:
        try:
            df = load_parquet(DATA_DIR, "MNQ", "1m")
        except FileNotFoundError:
            print("No data found. Generating synthetic data for demo...")
            from tests.test_engine import generate_synthetic_data
            df = generate_synthetic_data(num_bars=5000)

    print(f"Data: {len(df)} bars from {df['timestamp'][0]} to {df['timestamp'][-1]}")

    # Setup
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    backtester = VectorizedBacktester(
        data={"1m": df},
        risk_manager=rm,
        contract_spec=MNQ_SPEC,
        config=config,
    )

    strategy = EMACrossoverStrategy(
        fast_period=9,
        slow_period=21,
        stop_loss_points=4.0,
        take_profit_points=8.0,
        contracts=1,
    )

    # Run
    print(f"Running backtest: {strategy.name}")
    result = backtester.run(strategy)

    # Metrics
    metrics = calculate_metrics(result.trades, config.initial_capital)
    result.metrics = metrics
    print_metrics(metrics)

    # Log to SQLite
    logger = TradeLogger()
    run_id = logger.log_result(result)
    logger.close()
    print(f"Results logged to database (run_id: {run_id})")

    return result


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_with_data(data_path)
