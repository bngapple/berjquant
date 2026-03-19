"""Phase 4 tests — walk-forward validation, regime detection, correlation, overfitting."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.utils import (
    BacktestConfig, MNQ_SPEC, Trade,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.ema_crossover import EMACrossoverStrategy

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Helpers ──────────────────────────────────────────────────────────

def generate_synthetic_data(num_bars=2000, start_price=18000.0, volatility=2.0, seed=42):
    np.random.seed(seed)
    start_time = datetime(2025, 1, 6, 9, 30)
    prices = [start_price]
    for _ in range(num_bars - 1):
        prices.append(prices[-1] + np.random.normal(0, volatility))

    timestamps = []
    t = start_time
    for _ in range(num_bars):
        while t.weekday() >= 5:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        if t.hour >= 17:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        if t.hour < 8:
            t = t.replace(hour=9, minute=30)
        timestamps.append(t)
        t += timedelta(minutes=1)

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "low": [p - abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "close": prices,
        "volume": [int(abs(np.random.normal(500, 200))) for _ in range(num_bars)],
        "tick_count": [int(abs(np.random.normal(50, 20))) for _ in range(num_bars)],
    })
    df = df.with_columns([
        pl.max_horizontal("open", "high", "close").alias("high"),
        pl.min_horizontal("open", "low", "close").alias("low"),
    ])
    return df


def get_backtest_trades(seed=42):
    df = generate_synthetic_data(2000, seed=seed)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2025-01-06", end_date="2025-01-10",
        slippage_ticks=2, initial_capital=50000.0,
    )
    prop_rules = load_prop_firm_rules(CONFIG_DIR, config.prop_firm_profile)
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    backtester = VectorizedBacktester(
        data={"1m": df}, risk_manager=rm, contract_spec=MNQ_SPEC, config=config,
    )
    strategy = EMACrossoverStrategy(
        fast_period=9, slow_period=21,
        stop_loss_points=4.0, take_profit_points=8.0, contracts=1,
    )
    result = backtester.run(strategy)
    return result.trades, df, config, prop_rules, rm


# ── Walk-Forward Tests ───────────────────────────────────────────────

class TestWalkForward:
    def setup_method(self):
        self.trades, self.df, self.config, self.prop_rules, self.rm = get_backtest_trades()

    def test_validate_runs(self):
        from validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(
            data={"1m": self.df},
            risk_manager=self.rm,
            contract_spec=MNQ_SPEC,
            config=self.config,
        )
        strategy = EMACrossoverStrategy(
            fast_period=9, slow_period=21,
            stop_loss_points=4.0, take_profit_points=8.0, contracts=1,
        )
        result = validator.validate(
            strategy=strategy,
            train_days=3,
            test_days=1,
            step_days=1,
        )
        assert result is not None
        assert result.strategy_name is not None
        assert len(result.windows) >= 1
        assert isinstance(result.is_overfit, bool)

    def test_holdout(self):
        from validation.walk_forward import WalkForwardValidator
        # Use more data to have enough trading days for holdout
        df = generate_synthetic_data(5000, seed=42)
        validator = WalkForwardValidator(
            data={"1m": df},
            risk_manager=self.rm,
            contract_spec=MNQ_SPEC,
            config=self.config,
        )
        strategy = EMACrossoverStrategy(
            fast_period=9, slow_period=21,
            stop_loss_points=4.0, take_profit_points=8.0, contracts=1,
        )
        result = validator.holdout_test(strategy, holdout_pct=0.3)
        assert result is not None
        # Either returns metrics or an error dict for insufficient data
        assert isinstance(result, dict)


# ── Regime Detection Tests ───────────────────────────────────────────

class TestRegimeDetection:
    def setup_method(self):
        self.df = generate_synthetic_data(2000)

    def test_classify(self):
        from validation.regime import RegimeDetector
        detector = RegimeDetector()
        result = detector.classify(self.df)
        assert "regime" in result.columns
        # Should have classified some bars
        regimes = result["regime"].unique().to_list()
        assert len(regimes) >= 1

    def test_regime_summary(self):
        from validation.regime import RegimeDetector
        detector = RegimeDetector()
        classified = detector.classify(self.df)
        summary = detector.regime_summary(classified)
        assert isinstance(summary, dict)
        # Percentages should sum to ~100
        total = sum(summary.values())
        assert 95 <= total <= 105  # Allow some rounding

    def test_analyze_strategy(self):
        from validation.regime import RegimeDetector
        trades, df, _, _, _ = get_backtest_trades()
        if len(trades) == 0:
            pytest.skip("No trades generated")
        detector = RegimeDetector()
        analysis = detector.analyze_strategy(trades, df, strategy_name="test")
        assert analysis is not None
        assert analysis.strategy_name == "test"
        assert isinstance(analysis.regime_sensitivity, float)
        assert 0 <= analysis.regime_sensitivity <= 1


# ── Correlation Tests ────────────────────────────────────────────────

class TestCorrelation:
    def setup_method(self):
        self.trades_a, _, _, _, _ = get_backtest_trades(seed=42)
        self.trades_b, _, _, _, _ = get_backtest_trades(seed=99)

    def test_analyze_pair(self):
        from validation.correlation import CorrelationAnalyzer
        if len(self.trades_a) < 2 or len(self.trades_b) < 2:
            pytest.skip("Need trades for correlation")
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze_pair("strat_a", self.trades_a, "strat_b", self.trades_b)
        assert result is not None
        assert -1 <= result.return_correlation <= 1
        assert 0 <= result.trade_overlap_pct <= 100

    def test_analyze_portfolio(self):
        from validation.correlation import CorrelationAnalyzer
        if len(self.trades_a) < 2 or len(self.trades_b) < 2:
            pytest.skip("Need trades for portfolio")
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze_portfolio([
            ("strat_a", self.trades_a),
            ("strat_b", self.trades_b),
        ])
        assert result is not None
        assert result.n_strategies == 2
        assert result.correlation_matrix.shape == (2, 2)


# ── Overfit Detection Tests ──────────────────────────────────────────

class TestOverfitDetector:
    def test_analyze(self):
        from validation.correlation import OverfitDetector
        trades, df, _, _, _ = get_backtest_trades()
        if len(trades) < 5:
            pytest.skip("Need more trades")
        detector = OverfitDetector()
        report = detector.analyze(
            strategy_name="test_strat",
            trades=trades,
            data=df,
        )
        assert report is not None
        assert report.strategy_name == "test_strat"
        assert 0 <= report.overfit_probability <= 1
        assert report.verdict in ["likely_overfit", "possibly_overfit", "likely_robust"]

    def test_time_stability(self):
        from validation.correlation import OverfitDetector
        trades, _, _, _, _ = get_backtest_trades()
        if len(trades) < 8:
            pytest.skip("Need more trades for time stability")
        detector = OverfitDetector()
        report = detector.analyze(strategy_name="test", trades=trades)
        assert isinstance(report.is_time_stable, bool)
        assert len(report.period_sharpes) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
