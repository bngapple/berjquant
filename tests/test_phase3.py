"""Phase 3 tests — Monte Carlo simulation, scoring, visualization."""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.utils import (
    BacktestConfig,
    MNQ_SPEC,
    Trade,
    PerformanceMetrics,
    load_prop_firm_rules,
    load_session_config,
    load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.ema_crossover import EMACrossoverStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig, MCResult
from monte_carlo.scoring import StrategyScorer, StrategyScore

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Helpers ──────────────────────────────────────────────────────────

def generate_synthetic_data(num_bars=2000, start_price=18000.0, volatility=2.0):
    np.random.seed(42)
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


def run_backtest_get_trades():
    """Run a backtest and return the trades for MC testing."""
    df = generate_synthetic_data(2000)
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
    return result.trades, prop_rules


# ── MC Simulator Tests ───────────────────────────────────────────────

class TestMonteCarloSimulator:
    def setup_method(self):
        self.trades, self.prop_rules = run_backtest_get_trades()
        assert len(self.trades) > 0, "Need trades for MC testing"

    def test_basic_simulation(self):
        """Run a small MC simulation and verify output structure."""
        mc_config = MCConfig(
            n_simulations=100,
            initial_capital=50000.0,
            seed=42,
        )
        sim = MonteCarloSimulator(mc_config)
        result = sim.run(self.trades, strategy_name="test_ema")

        assert isinstance(result, MCResult)
        assert result.strategy_name == "test_ema"
        assert result.n_simulations == 100
        assert len(result.final_equities) == 100
        assert len(result.max_drawdowns) == 100
        assert len(result.sharpe_ratios) == 100
        assert len(result.profit_factors) == 100
        assert result.probability_of_profit >= 0
        assert result.probability_of_profit <= 1
        assert result.probability_of_ruin >= 0
        assert result.probability_of_ruin <= 1

    def test_equity_percentiles(self):
        """Verify equity percentile shape for fan chart."""
        mc_config = MCConfig(n_simulations=50, seed=42)
        sim = MonteCarloSimulator(mc_config)
        result = sim.run(self.trades, strategy_name="test")

        # Should have shape (n_trades+1, 5) for 5 percentile bands
        assert result.equity_percentiles is not None
        assert result.equity_percentiles.ndim == 2
        assert result.equity_percentiles.shape[1] == 5
        # First row should be initial capital for all percentiles
        assert np.allclose(result.equity_percentiles[0, :], 50000.0)

    def test_with_prop_firm_rules(self):
        """Test prop firm compliance checking."""
        mc_config = MCConfig(
            n_simulations=100,
            initial_capital=50000.0,
            prop_firm_rules=self.prop_rules,
            seed=42,
        )
        sim = MonteCarloSimulator(mc_config)
        result = sim.run(self.trades, strategy_name="test_prop")

        assert result.prop_firm_pass_rate >= 0
        assert result.prop_firm_pass_rate <= 1

    def test_composite_score(self):
        """Composite score should be between 0 and 100."""
        mc_config = MCConfig(n_simulations=100, seed=42)
        sim = MonteCarloSimulator(mc_config)
        result = sim.run(self.trades, strategy_name="test")

        assert 0 <= result.composite_score <= 100

    def test_reproducibility(self):
        """Same seed should produce same results."""
        mc_config = MCConfig(n_simulations=50, seed=123)

        sim1 = MonteCarloSimulator(mc_config)
        result1 = sim1.run(self.trades, strategy_name="test")

        sim2 = MonteCarloSimulator(mc_config)
        result2 = sim2.run(self.trades, strategy_name="test")

        np.testing.assert_array_almost_equal(
            result1.final_equities, result2.final_equities
        )

    def test_slippage_perturbation(self):
        """Slippage perturbation should reduce average P&L."""
        # Without extra slippage
        mc_config_no_slip = MCConfig(
            n_simulations=200, seed=42,
            slippage_perturbation=False,
            gap_injection=False,
        )
        sim1 = MonteCarloSimulator(mc_config_no_slip)
        result1 = sim1.run(self.trades, strategy_name="no_slip")

        # With extra slippage
        mc_config_slip = MCConfig(
            n_simulations=200, seed=42,
            slippage_perturbation=True,
            slippage_max_ticks=6,
            gap_injection=False,
        )
        sim2 = MonteCarloSimulator(mc_config_slip)
        result2 = sim2.run(self.trades, strategy_name="with_slip")

        # Average return should be lower with slippage
        assert result2.mean_return <= result1.mean_return


# ── Scoring Tests ────────────────────────────────────────────────────

class TestStrategyScorer:
    def setup_method(self):
        trades, prop_rules = run_backtest_get_trades()
        mc_config = MCConfig(
            n_simulations=100, seed=42,
            initial_capital=50000.0,
            prop_firm_rules=prop_rules,
        )
        sim = MonteCarloSimulator(mc_config)
        self.mc_result = sim.run(trades, strategy_name="test_strat")
        self.prop_rules = prop_rules

    def test_score_single(self):
        scorer = StrategyScorer(prop_rules=self.prop_rules)
        score = scorer.score(self.mc_result)

        assert isinstance(score, StrategyScore)
        assert score.strategy_name == "test_strat"
        assert 0 <= score.composite_score <= 100
        assert 0 <= score.sharpe_score <= 100
        assert 0 <= score.profit_factor_score <= 100
        assert 0 <= score.pass_rate_score <= 100
        assert 0 <= score.ruin_score <= 100
        assert score.grade in ["A+", "A", "B+", "B", "C", "D", "F"]

    def test_score_batch(self):
        scorer = StrategyScorer(prop_rules=self.prop_rules)
        scores = scorer.score_batch([self.mc_result, self.mc_result])
        assert len(scores) == 2

    def test_rank(self):
        scorer = StrategyScorer(prop_rules=self.prop_rules)
        scores = scorer.score_batch([self.mc_result])
        ranked = scorer.rank(scores, viable_only=False)
        assert len(ranked) >= 1

    def test_distribution_analysis(self):
        scorer = StrategyScorer(prop_rules=self.prop_rules)
        analysis = scorer.distribution_analysis(self.mc_result)
        assert "return_distribution" in analysis
        assert "drawdown_distribution" in analysis
        assert "tail_risk" in analysis

    def test_prop_firm_analysis(self):
        scorer = StrategyScorer(prop_rules=self.prop_rules)
        analysis = scorer.prop_firm_analysis(self.mc_result, self.prop_rules)
        assert "pass_rate" in analysis
        assert "failure_reasons" in analysis


# ── Visualization Tests ──────────────────────────────────────────────

class TestMCVisualizer:
    def setup_method(self):
        trades, prop_rules = run_backtest_get_trades()
        mc_config = MCConfig(
            n_simulations=50, seed=42,
            initial_capital=50000.0,
            prop_firm_rules=prop_rules,
        )
        sim = MonteCarloSimulator(mc_config)
        self.mc_result = sim.run(trades, strategy_name="viz_test")

        scorer = StrategyScorer(prop_rules=prop_rules)
        self.score = scorer.score(self.mc_result)

    def test_equity_fan_chart(self):
        from monte_carlo.visualization import MCVisualizer
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MCVisualizer(output_dir=tmpdir)
            path = viz.equity_fan_chart(self.mc_result)
            assert path is not None
            assert path.exists()

    def test_drawdown_distribution(self):
        from monte_carlo.visualization import MCVisualizer
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MCVisualizer(output_dir=tmpdir)
            path = viz.drawdown_distribution(self.mc_result)
            assert path is not None
            assert path.exists()

    def test_return_distribution(self):
        from monte_carlo.visualization import MCVisualizer
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MCVisualizer(output_dir=tmpdir)
            path = viz.return_distribution(self.mc_result)
            assert path is not None
            assert path.exists()

    def test_full_report(self):
        from monte_carlo.visualization import MCVisualizer
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MCVisualizer(output_dir=tmpdir)
            paths = viz.full_report(self.mc_result, self.score)
            assert len(paths) >= 3  # At least fan chart, DD dist, return dist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
