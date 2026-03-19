"""Phase 2 tests — signal library, strategy generator, serializer, parallel runner."""

import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.utils import (
    BacktestConfig,
    MNQ_SPEC,
    load_prop_firm_rules,
    load_session_config,
    load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Helpers ──────────────────────────────────────────────────────────

def generate_synthetic_data(
    num_bars: int = 2000,
    start_price: float = 18000.0,
    volatility: float = 2.0,
    start_time: datetime | None = None,
) -> pl.DataFrame:
    """Generate synthetic 1-minute MNQ data for testing."""
    if start_time is None:
        start_time = datetime(2025, 1, 6, 9, 30)

    np.random.seed(42)
    prices = [start_price]
    for _ in range(num_bars - 1):
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] + change)

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
    df = df.with_columns([
        pl.max_horizontal("open", "high", "close").alias("high"),
        pl.min_horizontal("open", "low", "close").alias("low"),
    ])
    return df


# ── Signal Tests ─────────────────────────────────────────────────────

class TestTrendSignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_ema_crossover(self):
        from signals.trend import ema_crossover
        result = ema_crossover(self.df, fast_period=9, slow_period=21)
        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns
        assert "entry_long_ema_cross" in result.columns
        assert "entry_short_ema_cross" in result.columns
        # Should have some crossover signals
        longs = result["entry_long_ema_cross"].sum()
        shorts = result["entry_short_ema_cross"].sum()
        assert longs + shorts > 0

    def test_ema_slope(self):
        from signals.trend import ema_slope
        result = ema_slope(self.df, period=20, slope_lookback=3)
        assert "ema_20" in result.columns
        assert "ema_slope_20" in result.columns
        assert "signal_ema_slope_up" in result.columns
        assert "signal_ema_slope_down" in result.columns

    def test_supertrend(self):
        from signals.trend import supertrend
        result = supertrend(self.df, period=10, multiplier=3.0)
        assert "supertrend" in result.columns
        assert "signal_supertrend_bullish" in result.columns
        assert "signal_supertrend_bearish" in result.columns

    def test_heikin_ashi(self):
        from signals.trend import heikin_ashi
        result = heikin_ashi(self.df)
        assert "ha_open" in result.columns
        assert "ha_close" in result.columns
        assert "signal_ha_bullish" in result.columns

    def test_linear_regression_slope(self):
        from signals.trend import linear_regression_slope
        result = linear_regression_slope(self.df, period=20)
        assert "linreg_slope_20" in result.columns
        assert "signal_linreg_up" in result.columns

    def test_ema_ribbon(self):
        from signals.trend import ema_ribbon
        result = ema_ribbon(self.df, periods=[8, 13, 21, 34, 55])
        assert "ema_8" in result.columns
        assert "ema_55" in result.columns
        assert "signal_ema_ribbon_bullish" in result.columns


class TestMomentumSignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_rsi(self):
        from signals.momentum import rsi
        result = rsi(self.df, period=14)
        assert "rsi_14" in result.columns
        assert "entry_long_rsi" in result.columns
        assert "entry_short_rsi" in result.columns
        # RSI values should be 0-100
        valid = result.filter(pl.col("rsi_14").is_not_null())
        assert valid["rsi_14"].min() >= 0
        assert valid["rsi_14"].max() <= 100

    def test_macd(self):
        from signals.momentum import macd
        result = macd(self.df)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert "entry_long_macd" in result.columns

    def test_stochastic(self):
        from signals.momentum import stochastic
        result = stochastic(self.df)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_roc(self):
        from signals.momentum import roc
        result = roc(self.df, period=10)
        assert "roc_10" in result.columns
        assert "entry_long_roc" in result.columns

    def test_cci(self):
        from signals.momentum import cci
        result = cci(self.df, period=20)
        assert "cci_20" in result.columns

    def test_williams_r(self):
        from signals.momentum import williams_r
        result = williams_r(self.df, period=14)
        assert "williams_r_14" in result.columns


class TestVolatilitySignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_atr(self):
        from signals.volatility import atr
        result = atr(self.df, period=14)
        assert "atr_14" in result.columns
        valid = result.filter(pl.col("atr_14").is_not_null())
        assert valid["atr_14"].min() >= 0

    def test_bollinger_bands(self):
        from signals.volatility import bollinger_bands
        result = bollinger_bands(self.df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_pct_b" in result.columns
        assert "entry_long_bb" in result.columns

    def test_keltner_channels(self):
        from signals.volatility import keltner_channels
        result = keltner_channels(self.df)
        assert "kc_upper" in result.columns
        assert "kc_lower" in result.columns

    def test_bollinger_keltner_squeeze(self):
        from signals.volatility import bollinger_keltner_squeeze
        result = bollinger_keltner_squeeze(self.df)
        assert "signal_squeeze_on" in result.columns
        assert "signal_squeeze_fire" in result.columns

    def test_atr_percentile(self):
        from signals.volatility import atr_percentile
        result = atr_percentile(self.df, atr_period=14, lookback=100)
        assert "atr_percentile" in result.columns

    def test_historical_volatility(self):
        from signals.volatility import historical_volatility
        result = historical_volatility(self.df, period=20)
        assert "hv_20" in result.columns


class TestVolumeSignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_vwap(self):
        from signals.volume import vwap
        result = vwap(self.df)
        assert "vwap" in result.columns
        assert "entry_long_vwap" in result.columns

    def test_volume_delta(self):
        from signals.volume import volume_delta
        result = volume_delta(self.df)
        assert "buy_volume" in result.columns
        assert "sell_volume" in result.columns
        assert "volume_delta" in result.columns
        assert "cumulative_delta" in result.columns

    def test_relative_volume(self):
        from signals.volume import relative_volume
        result = relative_volume(self.df, lookback=20)
        assert "rvol" in result.columns
        assert "signal_high_volume" in result.columns

    def test_volume_climax(self):
        from signals.volume import volume_climax
        result = volume_climax(self.df)
        assert "signal_volume_climax" in result.columns

    def test_volume_profile(self):
        from signals.volume import volume_profile
        result = volume_profile(self.df, lookback_bars=50, num_bins=20)
        assert "vpoc" in result.columns
        assert "vah" in result.columns
        assert "val" in result.columns


class TestOrderFlowSignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)
        # Pre-compute volume_delta since many orderflow signals depend on it
        from signals.volume import volume_delta
        self.df_with_delta = volume_delta(self.df)

    def test_imbalance(self):
        from signals.orderflow import imbalance
        result = imbalance(self.df)
        assert "buy_sell_ratio" in result.columns
        assert "signal_buy_imbalance" in result.columns
        assert "signal_sell_imbalance" in result.columns

    def test_absorption(self):
        from signals.orderflow import absorption
        result = absorption(self.df)
        assert "signal_absorption" in result.columns

    def test_large_trade_detection(self):
        from signals.orderflow import large_trade_detection
        result = large_trade_detection(self.df)
        assert "signal_large_trade" in result.columns
        assert "large_trade_direction" in result.columns

    def test_delta_divergence(self):
        from signals.orderflow import delta_divergence
        result = delta_divergence(self.df_with_delta, lookback=10)
        assert "signal_delta_div_bullish" in result.columns
        assert "signal_delta_div_bearish" in result.columns

    def test_exhaustion(self):
        from signals.orderflow import exhaustion
        result = exhaustion(self.df_with_delta)
        assert "signal_exhaustion_long" in result.columns
        assert "signal_exhaustion_short" in result.columns

    def test_delta_momentum(self):
        from signals.orderflow import delta_momentum
        result = delta_momentum(self.df_with_delta)
        assert "delta_ema_fast" in result.columns
        assert "entry_long_delta_momentum" in result.columns

    def test_footprint_imbalance(self):
        from signals.orderflow import footprint_imbalance
        result = footprint_imbalance(self.df)
        assert "stacked_buy_imbalance" in result.columns
        assert "signal_stacked_buy" in result.columns

    def test_trapped_traders(self):
        from signals.orderflow import trapped_traders
        result = trapped_traders(self.df)
        assert "signal_trapped_longs" in result.columns
        assert "signal_trapped_shorts" in result.columns


class TestPriceActionSignals:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_session_levels(self):
        from signals.price_action import session_levels
        result = session_levels(self.df)
        assert "session_open" in result.columns
        assert "session_high" in result.columns

    def test_previous_day_levels(self):
        from signals.price_action import previous_day_levels
        result = previous_day_levels(self.df)
        assert "prev_day_high" in result.columns
        assert "prev_day_low" in result.columns

    def test_range_breakout(self):
        from signals.price_action import range_breakout
        result = range_breakout(self.df, lookback=20)
        assert "range_high" in result.columns
        assert "entry_long_breakout" in result.columns

    def test_pivot_points(self):
        from signals.price_action import pivot_points
        result = pivot_points(self.df)
        assert "pivot" in result.columns
        assert "r1" in result.columns
        assert "s1" in result.columns

    def test_opening_range(self):
        from signals.price_action import opening_range
        result = opening_range(self.df, or_minutes=30)
        assert "or_high" in result.columns
        assert "or_low" in result.columns

    def test_candle_patterns(self):
        from signals.price_action import candle_patterns
        result = candle_patterns(self.df)
        assert "signal_engulfing_bullish" in result.columns
        assert "signal_pin_bar_bullish" in result.columns
        assert "signal_inside_bar" in result.columns


class TestTimeFilters:
    def setup_method(self):
        self.df = generate_synthetic_data(500)

    def test_session_segment(self):
        from signals.time_filters import session_segment
        result = session_segment(self.df)
        assert "signal_is_core" in result.columns
        assert "signal_is_premarket" in result.columns

    def test_time_of_day(self):
        from signals.time_filters import time_of_day
        result = time_of_day(self.df, start_hour=9, start_minute=30, end_hour=11, end_minute=0)
        assert "signal_in_time_window" in result.columns

    def test_day_of_week(self):
        from signals.time_filters import day_of_week
        result = day_of_week(self.df)
        assert "day_of_week" in result.columns
        assert "signal_day_allowed" in result.columns

    def test_first_n_minutes(self):
        from signals.time_filters import first_n_minutes
        result = first_n_minutes(self.df, n=30)
        assert "signal_first_n_minutes" in result.columns

    def test_last_n_minutes(self):
        from signals.time_filters import last_n_minutes
        result = last_n_minutes(self.df, n=30)
        assert "signal_last_n_minutes" in result.columns

    def test_london_overlap(self):
        from signals.time_filters import london_overlap
        result = london_overlap(self.df)
        assert "signal_london_overlap" in result.columns


# ── Registry Tests ───────────────────────────────────────────────────

class TestSignalRegistry:
    def test_registry_loads(self):
        registry = SignalRegistry()
        all_signals = registry.get_all()
        assert len(all_signals) >= 40  # We registered 44 signals

    def test_categories(self):
        registry = SignalRegistry()
        categories = set()
        for sig in registry.get_all().values():
            categories.add(sig.category)
        expected = {"trend", "momentum", "volatility", "volume", "orderflow", "price_action", "time_filter"}
        assert categories == expected

    def test_entry_signals(self):
        registry = SignalRegistry()
        entries = registry.list_entry_signals()
        assert len(entries) >= 10
        for sig in entries:
            assert sig.signal_type == "entry"
            assert len(sig.entry_columns) > 0

    def test_filters(self):
        registry = SignalRegistry()
        filters = registry.list_filters()
        assert len(filters) >= 10
        for sig in filters:
            assert sig.signal_type == "filter"
            assert len(sig.filter_columns) > 0

    def test_dependencies(self):
        registry = SignalRegistry()
        deps = registry.get_dependencies("delta_divergence")
        assert "volume_delta" in deps


# ── Generator Tests ──────────────────────────────────────────────────

class TestStrategyGenerator:
    def setup_method(self):
        self.registry = SignalRegistry()

    def test_generate_strategies(self):
        from strategies.generator import StrategyGenerator, ExitRules, SizingRules
        gen = StrategyGenerator(self.registry)
        strategies = gen.generate(
            max_entry_signals=1,
            max_filters=0,
            exit_variations=[ExitRules()],
            sizing_variations=[SizingRules()],
            max_strategies=20,
        )
        assert len(strategies) > 0
        assert len(strategies) <= 20
        for s in strategies:
            assert s.name
            assert len(s.entry_signals) >= 1

    def test_count_combinations(self):
        from strategies.generator import StrategyGenerator
        gen = StrategyGenerator(self.registry)
        count = gen.count_combinations(max_entry_signals=1, max_filters=0)
        assert count > 0

    def test_generated_strategy_runs_backtest(self):
        """End-to-end: generate a strategy → run backtest → get results."""
        from strategies.generator import StrategyGenerator, ExitRules, SizingRules

        gen = StrategyGenerator(self.registry)
        strategies = gen.generate(
            entry_categories=["trend"],
            max_entry_signals=1,
            max_filters=0,
            exit_variations=[ExitRules(stop_loss_value=4.0, take_profit_value=8.0)],
            sizing_variations=[SizingRules(method="fixed", fixed_contracts=1)],
            max_strategies=3,
        )
        assert len(strategies) > 0

        df = generate_synthetic_data(2000)
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

        strategy = strategies[0]
        backtester = VectorizedBacktester(
            data={"1m": df},
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            config=config,
        )
        result = backtester.run(strategy)
        # Should complete without error
        assert result is not None


# ── Serializer Tests ─────────────────────────────────────────────────

class TestStrategySerializer:
    def test_serialize_roundtrip(self):
        from strategies.generator import GeneratedStrategy, ExitRules, SizingRules

        strategy = GeneratedStrategy(
            name="TEST_STRAT",
            entry_signals=[{
                "signal_name": "ema_crossover",
                "module": "signals.trend",
                "function": "ema_crossover",
                "params": {"fast_period": 9, "slow_period": 21},
                "columns": {"long": "entry_long_ema_cross", "short": "entry_short_ema_cross"},
            }],
            entry_filters=[],
            exit_rules=ExitRules(stop_loss_value=4.0, take_profit_value=8.0),
            sizing_rules=SizingRules(method="fixed", fixed_contracts=1),
        )

        d = strategy.to_dict()
        restored = GeneratedStrategy.from_dict(d)
        assert restored.name == strategy.name
        assert len(restored.entry_signals) == 1
        assert restored.exit_rules.stop_loss_value == 4.0
        assert restored.sizing_rules.method == "fixed"

    def test_save_and_load(self):
        from strategies.generator import GeneratedStrategy, ExitRules, SizingRules
        from strategies.serializer import StrategySerializer

        with tempfile.TemporaryDirectory() as tmpdir:
            serializer = StrategySerializer(strategies_dir=tmpdir)

            strategy = GeneratedStrategy(
                name="SAVE_TEST",
                entry_signals=[{
                    "signal_name": "rsi",
                    "module": "signals.momentum",
                    "function": "rsi",
                    "params": {"period": 14},
                    "columns": {"long": "entry_long_rsi", "short": "entry_short_rsi"},
                }],
                entry_filters=[],
                exit_rules=ExitRules(),
                sizing_rules=SizingRules(),
            )

            path = serializer.save(strategy, tag="test")
            assert path.exists()

            loaded = serializer.load(path)
            assert loaded.name == "SAVE_TEST"
            assert len(loaded.entry_signals) == 1


# ── Parameter Variation Tests ────────────────────────────────────────

class TestParameterVariations:
    def test_random_variations(self):
        from strategies.generator import StrategyGenerator, GeneratedStrategy, ExitRules, SizingRules

        registry = SignalRegistry()
        gen = StrategyGenerator(registry)

        base_strategy = GeneratedStrategy(
            name="BASE",
            entry_signals=[{
                "signal_name": "ema_crossover",
                "module": "signals.trend",
                "function": "ema_crossover",
                "params": {"fast_period": 9, "slow_period": 21},
                "columns": {"long": "entry_long_ema_cross", "short": "entry_short_ema_cross"},
            }],
            entry_filters=[],
            exit_rules=ExitRules(),
            sizing_rules=SizingRules(),
        )

        variations = gen.generate_parameter_variations(
            base_strategy, num_variations=5, method="random"
        )
        assert len(variations) == 5
        # Each variation should have different params
        params_set = set()
        for v in variations:
            fp = v.entry_signals[0]["params"]["fast_period"]
            sp = v.entry_signals[0]["params"]["slow_period"]
            params_set.add((fp, sp))
        # At least some should differ (randomness, so not guaranteed all unique)
        assert len(params_set) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
