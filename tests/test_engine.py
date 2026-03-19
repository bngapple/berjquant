"""Integration tests for the MCQ Engine — validates the full pipeline."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.utils import (
    BacktestConfig,
    ContractSpec,
    MNQ_SPEC,
    load_prop_firm_rules,
    load_session_config,
    load_events_calendar,
)
from engine.data_pipeline import DataCleaner, BarBuilder
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics, print_metrics
from strategies.ema_crossover import EMACrossoverStrategy

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Helpers ──────────────────────────────────────────────────────────

def generate_synthetic_data(
    num_bars: int = 2000,
    start_price: float = 18000.0,
    volatility: float = 2.0,
    start_time: datetime | None = None,
    freq_minutes: int = 1,
) -> pl.DataFrame:
    """Generate synthetic 1-minute MNQ data for testing."""
    if start_time is None:
        start_time = datetime(2025, 1, 6, 9, 30)  # Monday 9:30 AM

    np.random.seed(42)
    prices = [start_price]
    for _ in range(num_bars - 1):
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] + change)

    timestamps = []
    t = start_time
    for _ in range(num_bars):
        # Skip weekends
        while t.weekday() >= 5:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        # Skip outside trading hours (8:00-17:00)
        if t.hour >= 17:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        if t.hour < 8:
            t = t.replace(hour=9, minute=30)
        timestamps.append(t)
        t += timedelta(minutes=freq_minutes)

    # Generate OHLCV from close prices
    data = {
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "low": [p - abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "close": prices,
        "volume": [int(abs(np.random.normal(500, 200))) for _ in range(num_bars)],
        "tick_count": [int(abs(np.random.normal(50, 20))) for _ in range(num_bars)],
    }

    df = pl.DataFrame(data)

    # Fix OHLCV consistency
    df = df.with_columns([
        pl.max_horizontal("open", "high", "close").alias("high"),
        pl.min_horizontal("open", "low", "close").alias("low"),
    ])

    return df


# ── Tests ────────────────────────────────────────────────────────────

class TestDataCleaner:
    def test_validate_ohlcv(self):
        df = pl.DataFrame({
            "timestamp": [datetime(2025, 1, 6, 10, i) for i in range(5)],
            "open": [100.0, 101.0, 99.0, 102.0, 100.0],
            "high": [98.0, 103.0, 100.0, 103.0, 101.0],  # first bar high < open
            "low": [97.0, 100.0, 98.0, 101.0, 99.0],
            "close": [99.0, 102.0, 100.0, 101.0, 100.5],
            "volume": [100, 200, 150, 300, 250],
            "tick_count": [10, 20, 15, 30, 25],
        })
        cleaner = DataCleaner()
        result = cleaner.validate_ohlcv(df)
        # After fix, high should be >= max(open, close)
        highs = result["high"].to_list()
        opens = result["open"].to_list()
        closes = result["close"].to_list()
        for h, o, c in zip(highs, opens, closes):
            assert h >= max(o, c)

    def test_remove_bad_ticks(self):
        df = pl.DataFrame({
            "timestamp": [datetime(2025, 1, 6, 10, i) for i in range(5)],
            "open": [100.0, 100.5, 200.0, 101.0, 101.5],  # 200 is a bad tick
            "high": [101.0, 101.0, 201.0, 102.0, 102.0],
            "low": [99.0, 100.0, 199.0, 100.0, 101.0],
            "close": [100.5, 100.8, 200.5, 101.5, 101.8],
            "volume": [100, 100, 100, 100, 100],
            "tick_count": [10, 10, 10, 10, 10],
        })
        cleaner = DataCleaner()
        result = cleaner.remove_bad_ticks(df, max_pct_change=5.0)
        # Bad tick (200.5) and the row after it (back to 101.5) both have >5% change
        assert len(result) == 3  # bad tick and recovery row removed

    def test_tag_sessions(self):
        session_config = {
            "sessions": {
                "pre_market": {"start": "08:00", "end": "09:30"},
                "core": {"start": "09:30", "end": "16:00"},
                "post_close": {"start": "16:00", "end": "17:00"},
            }
        }
        df = pl.DataFrame({
            "timestamp": [
                datetime(2025, 1, 6, 8, 30),   # pre_market
                datetime(2025, 1, 6, 10, 0),    # core
                datetime(2025, 1, 6, 16, 30),   # post_close
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [100, 200, 150],
            "tick_count": [10, 20, 15],
        })
        cleaner = DataCleaner(session_config=session_config)
        result = cleaner.tag_sessions(df)
        sessions = result["session_segment"].to_list()
        assert sessions == ["pre_market", "core", "post_close"]


class TestBarBuilder:
    def test_resample(self):
        timestamps = [datetime(2025, 1, 6, 10, i) for i in range(10)]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [float(100 + i) for i in range(10)],
            "high": [float(101 + i) for i in range(10)],
            "low": [float(99 + i) for i in range(10)],
            "close": [float(100.5 + i) for i in range(10)],
            "volume": [100] * 10,
            "tick_count": [10] * 10,
        })
        result = BarBuilder.resample(df, freq="5m")
        assert len(result) == 2
        assert result["volume"].to_list() == [500, 500]


class TestRiskManager:
    def setup_method(self):
        self.prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
        self.session_config = load_session_config(CONFIG_DIR)
        self.events_calendar = load_events_calendar(CONFIG_DIR)
        self.rm = RiskManager(
            self.prop_rules, self.session_config,
            self.events_calendar, MNQ_SPEC,
        )

    def test_session_bounds(self):
        account = self.rm.init_account(50000)
        # Inside session
        allowed, _ = self.rm.pre_trade_check(
            datetime(2025, 1, 6, 10, 0), "long", 1, account
        )
        assert allowed

        # Outside session
        allowed, reason = self.rm.pre_trade_check(
            datetime(2025, 1, 6, 7, 0), "long", 1, account
        )
        assert not allowed
        assert reason == "outside_session"

    def test_max_contracts(self):
        account = self.rm.init_account(50000)
        # Request too many
        allowed, reason = self.rm.pre_trade_check(
            datetime(2025, 1, 6, 10, 0), "long", 20, account
        )
        assert not allowed
        assert "max_contracts" in reason

    def test_kill_switch(self):
        account = self.rm.init_account(50000)
        # Simulate hitting 80% of daily loss limit (-800)
        account.daily_pnl = -850
        allowed, reason = self.rm.pre_trade_check(
            datetime(2025, 1, 6, 10, 0), "long", 1, account
        )
        assert not allowed
        assert "kill_switch" in reason


class TestFullBacktest:
    def test_ema_crossover_runs(self):
        """End-to-end test: synthetic data → backtest → metrics."""
        df = generate_synthetic_data(num_bars=2000)

        config = BacktestConfig(
            symbol="MNQ",
            prop_firm_profile="topstep_50k",
            start_date="2025-01-06",
            end_date="2025-01-10",
            slippage_ticks=2,
            initial_capital=50000.0,
        )

        prop_rules = load_prop_firm_rules(CONFIG_DIR, config.prop_firm_profile)
        session_config = load_session_config(CONFIG_DIR)
        events_calendar = load_events_calendar(CONFIG_DIR)

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

        result = backtester.run(strategy)

        # Should have produced some trades
        assert len(result.trades) > 0

        # Calculate metrics
        metrics = calculate_metrics(result.trades, config.initial_capital)
        result.metrics = metrics

        assert metrics.total_trades > 0
        assert metrics.win_rate >= 0
        assert metrics.win_rate <= 100

        print_metrics(metrics)

    def test_risk_blocks_excessive_trades(self):
        """Verify risk manager blocks trades when limits are hit."""
        df = generate_synthetic_data(num_bars=500)

        config = BacktestConfig(
            symbol="MNQ",
            prop_firm_profile="topstep_50k",
            start_date="2025-01-06",
            end_date="2025-01-07",
            slippage_ticks=2,
            initial_capital=50000.0,
        )

        prop_rules = load_prop_firm_rules(CONFIG_DIR, config.prop_firm_profile)
        session_config = load_session_config(CONFIG_DIR)
        events_calendar = load_events_calendar(CONFIG_DIR)

        rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
        account = rm.init_account(50000)

        # Pre-damage the account to near kill switch
        account.daily_pnl = -750

        # Should still allow (80% of -1000 = -800)
        allowed, _ = rm.pre_trade_check(
            datetime(2025, 1, 6, 10, 0), "long", 1, account
        )
        assert allowed

        # Push past kill switch
        account.daily_pnl = -850
        allowed, reason = rm.pre_trade_check(
            datetime(2025, 1, 6, 10, 5), "long", 1, account
        )
        assert not allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
