"""
Signal Registry — catalogs all available signals, parameter ranges, and categories.

The combinatorial engine uses this registry to enumerate which signals are
available for strategy generation, what parameters they accept, and what
columns they produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Signal definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalDefinition:
    """Defines a signal and its parameter space for combinatorial search."""

    name: str                          # unique identifier e.g. "ema_crossover"
    category: str                      # "trend", "momentum", "volatility", "volume", "orderflow", "price_action", "time_filter"
    signal_type: str                   # "entry" (generates entry_long/entry_short), "filter" (generates signal_* bool), "indicator" (generates continuous values)
    module: str                        # e.g. "signals.trend"
    function: str                      # function name to call
    parameters: dict                   # param_name -> {"type": "int"|"float"|"list", "min": x, "max": y, "step": z, "default": d}
    entry_columns: list[str] = field(default_factory=list)      # names of entry_long_*/entry_short_* columns produced
    filter_columns: list[str] = field(default_factory=list)     # names of signal_* columns produced
    indicator_columns: list[str] = field(default_factory=list)  # names of continuous value columns produced
    requires: list[str] = field(default_factory=list)           # other signals that must run first
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SignalRegistry:
    """Central registry of all available signals for the strategy generator."""

    def __init__(self) -> None:
        self._signals: dict[str, SignalDefinition] = {}
        self._register_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> SignalDefinition:
        """Return a signal definition by name.  Raises KeyError if not found."""
        return self._signals[name]

    def list_by_category(self, category: str) -> list[SignalDefinition]:
        """Return all signals in a given category."""
        return [s for s in self._signals.values() if s.category == category]

    def list_by_type(self, signal_type: str) -> list[SignalDefinition]:
        """Return all signals of a given type (entry / filter / indicator)."""
        return [s for s in self._signals.values() if s.signal_type == signal_type]

    def list_entry_signals(self) -> list[SignalDefinition]:
        """Return all entry-type signals."""
        return self.list_by_type("entry")

    def list_filters(self) -> list[SignalDefinition]:
        """Return all filter-type signals."""
        return self.list_by_type("filter")

    def get_all(self) -> dict[str, SignalDefinition]:
        """Return the full mapping of name -> SignalDefinition."""
        return dict(self._signals)

    def get_dependencies(self, name: str) -> list[str]:
        """Get ordered list of signals that must run before *name*.

        Performs a depth-first topological traversal of the ``requires``
        graph so that every prerequisite appears before its dependent.
        """
        visited: set[str] = set()
        order: list[str] = []

        def _visit(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            for dep in self._signals[n].requires:
                _visit(dep)
            order.append(n)

        _visit(name)
        # Remove the signal itself — the caller only wants prerequisites.
        order.remove(name)
        return order

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register(self, sig: SignalDefinition) -> None:
        self._signals[sig.name] = sig

    def _register_all(self) -> None:
        """Register every built-in signal."""
        self._register_trend_signals()
        self._register_momentum_signals()
        self._register_volatility_signals()
        self._register_volume_signals()
        self._register_orderflow_signals()
        self._register_price_action_signals()
        self._register_time_filter_signals()

    # ==================================================================
    # TREND signals  (signals.trend)
    # ==================================================================

    def _register_trend_signals(self) -> None:
        self._register(SignalDefinition(
            name="ema_crossover",
            category="trend",
            signal_type="entry",
            module="signals.trend",
            function="ema_crossover",
            parameters={
                "fast_period": {"type": "int", "min": 5, "max": 50, "step": 1, "default": 9},
                "slow_period": {"type": "int", "min": 10, "max": 100, "step": 1, "default": 21},
            },
            entry_columns=["entry_long_ema_cross", "entry_short_ema_cross"],
            indicator_columns=["ema_fast", "ema_slow"],
            description="Classic EMA crossover. Generates entry signals when fast EMA crosses above/below slow EMA.",
        ))

        self._register(SignalDefinition(
            name="ema_slope",
            category="trend",
            signal_type="filter",
            module="signals.trend",
            function="ema_slope",
            parameters={
                "period": {"type": "int", "min": 5, "max": 100, "step": 1, "default": 21},
                "slope_lookback": {"type": "int", "min": 1, "max": 10, "step": 1, "default": 3},
            },
            filter_columns=["signal_ema_slope_up", "signal_ema_slope_down"],
            indicator_columns=["ema_slope_{period}"],
            description="EMA direction/slope filter. Positive slope = uptrend, negative = downtrend.",
        ))

        self._register(SignalDefinition(
            name="ema_ribbon",
            category="trend",
            signal_type="filter",
            module="signals.trend",
            function="ema_ribbon",
            parameters={
                "periods": {"type": "list", "values": [
                    [8, 13, 21, 34, 55],
                    [5, 10, 20, 40, 80],
                    [8, 13, 21, 34],
                    [10, 20, 30, 50],
                ], "default": [8, 13, 21, 34, 55]},
            },
            filter_columns=["signal_ema_ribbon_bullish", "signal_ema_ribbon_bearish"],
            description="Multiple-EMA ribbon for trend strength. Bullish when all EMAs in ascending order (fastest on top).",
        ))

        self._register(SignalDefinition(
            name="linear_regression_slope",
            category="trend",
            signal_type="filter",
            module="signals.trend",
            function="linear_regression_slope",
            parameters={
                "period": {"type": "int", "min": 10, "max": 50, "step": 1, "default": 20},
            },
            filter_columns=["signal_linreg_up", "signal_linreg_down"],
            indicator_columns=["linreg_slope_{period}"],
            description="Rolling linear regression slope of close price. Positive slope = uptrend.",
        ))

        self._register(SignalDefinition(
            name="heikin_ashi",
            category="trend",
            signal_type="filter",
            module="signals.trend",
            function="heikin_ashi",
            parameters={},
            filter_columns=["signal_ha_bullish", "signal_ha_bearish"],
            indicator_columns=["ha_open", "ha_high", "ha_low", "ha_close"],
            description="Heikin-Ashi candle transformation. Smooths price action for clearer trend identification.",
        ))

        self._register(SignalDefinition(
            name="supertrend",
            category="trend",
            signal_type="filter",
            module="signals.trend",
            function="supertrend",
            parameters={
                "period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 10},
                "multiplier": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.5, "default": 3.0},
            },
            filter_columns=["signal_supertrend_bullish", "signal_supertrend_bearish"],
            indicator_columns=["supertrend"],
            description="ATR-based Supertrend indicator. Flips between support (bullish) and resistance (bearish).",
        ))

    # ==================================================================
    # MOMENTUM signals  (signals.momentum)
    # ==================================================================

    def _register_momentum_signals(self) -> None:
        self._register(SignalDefinition(
            name="rsi",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="rsi",
            parameters={
                "period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
                "overbought": {"type": "float", "min": 65.0, "max": 85.0, "step": 5.0, "default": 70.0},
                "oversold": {"type": "float", "min": 15.0, "max": 35.0, "step": 5.0, "default": 30.0},
            },
            entry_columns=["entry_long_rsi", "entry_short_rsi"],
            indicator_columns=["rsi_{period}"],
            description="Relative Strength Index with Wilder smoothing. Entry on crosses of overbought/oversold levels.",
        ))

        self._register(SignalDefinition(
            name="macd",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="macd",
            parameters={
                "fast": {"type": "int", "min": 8, "max": 16, "step": 1, "default": 12},
                "slow": {"type": "int", "min": 20, "max": 34, "step": 1, "default": 26},
                "signal_period": {"type": "int", "min": 5, "max": 13, "step": 1, "default": 9},
            },
            entry_columns=["entry_long_macd", "entry_short_macd"],
            indicator_columns=["macd_line", "macd_signal", "macd_histogram"],
            description="MACD histogram crossover. Entry when histogram crosses zero.",
        ))

        self._register(SignalDefinition(
            name="stochastic",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="stochastic",
            parameters={
                "k_period": {"type": "int", "min": 5, "max": 21, "step": 1, "default": 14},
                "d_period": {"type": "int", "min": 2, "max": 5, "step": 1, "default": 3},
                "overbought": {"type": "float", "min": 70.0, "max": 90.0, "step": 5.0, "default": 80.0},
                "oversold": {"type": "float", "min": 10.0, "max": 30.0, "step": 5.0, "default": 20.0},
            },
            entry_columns=["entry_long_stoch", "entry_short_stoch"],
            indicator_columns=["stoch_k", "stoch_d"],
            description="Stochastic oscillator. Entry when %K crosses %D in overbought/oversold zones.",
        ))

        self._register(SignalDefinition(
            name="roc",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="roc",
            parameters={
                "period": {"type": "int", "min": 5, "max": 20, "step": 1, "default": 10},
            },
            entry_columns=["entry_long_roc", "entry_short_roc"],
            indicator_columns=["roc_{period}"],
            description="Rate of Change (percentage). Entry on zero-line crossover.",
        ))

        self._register(SignalDefinition(
            name="cci",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="cci",
            parameters={
                "period": {"type": "int", "min": 10, "max": 30, "step": 1, "default": 20},
            },
            entry_columns=["entry_long_cci", "entry_short_cci"],
            indicator_columns=["cci_{period}"],
            description="Commodity Channel Index. Entry on crosses of +/-100 levels.",
        ))

        self._register(SignalDefinition(
            name="williams_r",
            category="momentum",
            signal_type="entry",
            module="signals.momentum",
            function="williams_r",
            parameters={
                "period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
                "overbought": {"type": "float", "min": -30.0, "max": -10.0, "step": 5.0, "default": -20.0},
                "oversold": {"type": "float", "min": -90.0, "max": -70.0, "step": 5.0, "default": -80.0},
            },
            entry_columns=["entry_long_williams", "entry_short_williams"],
            indicator_columns=["williams_r_{period}"],
            description="Williams %R oscillator. Entry on crosses of overbought/oversold levels.",
        ))

    # ==================================================================
    # VOLATILITY signals  (signals.volatility)
    # ==================================================================

    def _register_volatility_signals(self) -> None:
        self._register(SignalDefinition(
            name="atr",
            category="volatility",
            signal_type="indicator",
            module="signals.volatility",
            function="atr",
            parameters={
                "period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
            },
            indicator_columns=["atr_{period}"],
            description="Average True Range. Continuous volatility measure used for position sizing and stop placement.",
        ))

        self._register(SignalDefinition(
            name="bollinger_bands",
            category="volatility",
            signal_type="entry",
            module="signals.volatility",
            function="bollinger_bands",
            parameters={
                "period": {"type": "int", "min": 10, "max": 30, "step": 1, "default": 20},
                "std_dev": {"type": "float", "min": 1.5, "max": 3.0, "step": 0.25, "default": 2.0},
            },
            entry_columns=["entry_long_bb", "entry_short_bb"],
            indicator_columns=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b"],
            description="Bollinger Bands mean-reversion. Entry when price crosses back inside the bands.",
        ))

        self._register(SignalDefinition(
            name="keltner_channels",
            category="volatility",
            signal_type="entry",
            module="signals.volatility",
            function="keltner_channels",
            parameters={
                "ema_period": {"type": "int", "min": 10, "max": 30, "step": 1, "default": 20},
                "atr_period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
                "multiplier": {"type": "float", "min": 1.0, "max": 3.0, "step": 0.25, "default": 1.5},
            },
            entry_columns=["entry_long_kc", "entry_short_kc"],
            indicator_columns=["kc_upper", "kc_middle", "kc_lower"],
            description="Keltner Channels (EMA +/- ATR*multiplier). Entry on channel boundary crosses.",
        ))

        self._register(SignalDefinition(
            name="bollinger_keltner_squeeze",
            category="volatility",
            signal_type="filter",
            module="signals.volatility",
            function="bollinger_keltner_squeeze",
            parameters={
                "bb_period": {"type": "int", "min": 15, "max": 25, "step": 1, "default": 20},
                "bb_std": {"type": "float", "min": 1.5, "max": 2.5, "step": 0.25, "default": 2.0},
                "kc_period": {"type": "int", "min": 15, "max": 25, "step": 1, "default": 20},
                "kc_atr_period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
                "kc_mult": {"type": "float", "min": 1.0, "max": 2.0, "step": 0.25, "default": 1.5},
            },
            filter_columns=["signal_squeeze_on", "signal_squeeze_fire"],
            indicator_columns=["squeeze_momentum"],
            description="TTM Squeeze. Bollinger Bands inside Keltner = compression. Fires on expansion with momentum direction.",
        ))

        self._register(SignalDefinition(
            name="atr_percentile",
            category="volatility",
            signal_type="indicator",
            module="signals.volatility",
            function="atr_percentile",
            parameters={
                "atr_period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14},
                "lookback": {"type": "int", "min": 50, "max": 200, "step": 10, "default": 100},
            },
            indicator_columns=["atr_percentile"],
            requires=["atr"],
            description="ATR ranked against its rolling history (0-100). Useful as a volatility regime filter.",
        ))

        self._register(SignalDefinition(
            name="historical_volatility",
            category="volatility",
            signal_type="indicator",
            module="signals.volatility",
            function="historical_volatility",
            parameters={
                "period": {"type": "int", "min": 10, "max": 30, "step": 1, "default": 20},
            },
            indicator_columns=["hv_{period}"],
            description="Annualized historical volatility from log returns (assumes 252 trading days).",
        ))

    # ==================================================================
    # VOLUME signals  (signals.volume)
    # ==================================================================

    def _register_volume_signals(self) -> None:
        self._register(SignalDefinition(
            name="vwap",
            category="volume",
            signal_type="entry",
            module="signals.volume",
            function="vwap",
            parameters={},
            entry_columns=["entry_long_vwap", "entry_short_vwap"],
            indicator_columns=["vwap", "vwap_std", "vwap_upper_1", "vwap_lower_1"],
            description="Session VWAP (resets daily) with standard-deviation bands. Entry on price crossing VWAP.",
        ))

        self._register(SignalDefinition(
            name="volume_delta",
            category="volume",
            signal_type="indicator",
            module="signals.volume",
            function="volume_delta",
            parameters={},
            indicator_columns=["buy_volume", "sell_volume", "volume_delta", "cumulative_delta"],
            description="Estimate buy/sell volume split using close position within the bar. Produces cumulative delta (resets daily).",
        ))

        self._register(SignalDefinition(
            name="volume_profile",
            category="volume",
            signal_type="filter",
            module="signals.volume",
            function="volume_profile",
            parameters={
                "lookback_bars": {"type": "int", "min": 50, "max": 200, "step": 10, "default": 100},
                "num_bins": {"type": "int", "min": 25, "max": 100, "step": 25, "default": 50},
            },
            filter_columns=["signal_above_vah", "signal_below_val"],
            indicator_columns=["vpoc", "vah", "val"],
            description="Rolling volume profile with VPOC, value area high/low. Filters on price vs value area.",
        ))

        self._register(SignalDefinition(
            name="relative_volume",
            category="volume",
            signal_type="filter",
            module="signals.volume",
            function="relative_volume",
            parameters={
                "lookback": {"type": "int", "min": 10, "max": 50, "step": 5, "default": 20},
            },
            filter_columns=["signal_high_volume"],
            indicator_columns=["rvol"],
            description="Current bar volume relative to rolling average. Flags bars with rvol > 1.5.",
        ))

        self._register(SignalDefinition(
            name="volume_climax",
            category="volume",
            signal_type="filter",
            module="signals.volume",
            function="volume_climax",
            parameters={
                "lookback": {"type": "int", "min": 20, "max": 100, "step": 10, "default": 50},
                "threshold": {"type": "float", "min": 2.0, "max": 4.0, "step": 0.5, "default": 2.5},
            },
            filter_columns=["signal_volume_climax", "signal_climax_reversal"],
            description="Volume spike detection. Flags climax bars and potential exhaustion reversals.",
        ))

    # ==================================================================
    # ORDERFLOW signals  (signals.orderflow)
    # ==================================================================

    def _register_orderflow_signals(self) -> None:
        self._register(SignalDefinition(
            name="delta_divergence",
            category="orderflow",
            signal_type="entry",
            module="signals.orderflow",
            function="delta_divergence",
            parameters={
                "lookback": {"type": "int", "min": 5, "max": 20, "step": 1, "default": 10},
            },
            entry_columns=["entry_long_delta_div", "entry_short_delta_div"],
            requires=["volume_delta"],
            description="Delta divergence: price makes new high/low but cumulative delta diverges. Signals potential reversal.",
        ))

        self._register(SignalDefinition(
            name="absorption",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="absorption",
            parameters={
                "lookback": {"type": "int", "min": 3, "max": 10, "step": 1, "default": 5},
                "volume_threshold": {"type": "float", "min": 1.5, "max": 3.0, "step": 0.5, "default": 2.0},
            },
            filter_columns=["signal_absorption_bid", "signal_absorption_ask"],
            requires=["volume_delta"],
            description="Absorption detection: high volume with minimal price movement indicates large passive orders absorbing aggressive flow.",
        ))

        self._register(SignalDefinition(
            name="exhaustion",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="exhaustion",
            parameters={
                "lookback": {"type": "int", "min": 3, "max": 10, "step": 1, "default": 5},
                "delta_threshold": {"type": "float", "min": 1.5, "max": 3.0, "step": 0.5, "default": 2.0},
            },
            filter_columns=["signal_exhaustion_long", "signal_exhaustion_short"],
            requires=["volume_delta"],
            description="Exhaustion detection: extreme delta spike with price failing to follow through. Signals potential reversal.",
        ))

        self._register(SignalDefinition(
            name="imbalance",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="imbalance",
            parameters={
                "ratio_threshold": {"type": "float", "min": 2.0, "max": 5.0, "step": 0.5, "default": 3.0},
            },
            filter_columns=["signal_buy_imbalance", "signal_sell_imbalance"],
            requires=["volume_delta"],
            description="Buy/sell volume imbalance. Flags bars where one side dominates by the threshold ratio.",
        ))

        self._register(SignalDefinition(
            name="large_trade_detection",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="large_trade_detection",
            parameters={
                "lookback": {"type": "int", "min": 20, "max": 100, "step": 10, "default": 50},
                "std_mult": {"type": "float", "min": 2.0, "max": 4.0, "step": 0.5, "default": 3.0},
            },
            filter_columns=["signal_large_trade"],
            description="Detects abnormally large single-bar volume relative to rolling statistics. Proxy for institutional activity.",
        ))

        self._register(SignalDefinition(
            name="delta_momentum",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="delta_momentum",
            parameters={
                "fast_period": {"type": "int", "min": 3, "max": 10, "step": 1, "default": 5},
                "slow_period": {"type": "int", "min": 10, "max": 30, "step": 1, "default": 20},
            },
            filter_columns=["signal_delta_momentum_bull", "signal_delta_momentum_bear"],
            indicator_columns=["delta_momentum"],
            requires=["volume_delta"],
            description="Fast vs slow EMA of cumulative delta. Bullish when fast delta EMA > slow delta EMA.",
        ))

        self._register(SignalDefinition(
            name="footprint_imbalance",
            category="orderflow",
            signal_type="filter",
            module="signals.orderflow",
            function="footprint_imbalance",
            parameters={
                "lookback": {"type": "int", "min": 3, "max": 10, "step": 1, "default": 5},
                "imbalance_ratio": {"type": "float", "min": 2.0, "max": 5.0, "step": 0.5, "default": 3.0},
            },
            filter_columns=["signal_footprint_bid_imbalance", "signal_footprint_ask_imbalance"],
            requires=["volume_delta"],
            description="Consecutive bar imbalance stacking. Flags clusters of aggressive buying or selling.",
        ))

        self._register(SignalDefinition(
            name="trapped_traders",
            category="orderflow",
            signal_type="entry",
            module="signals.orderflow",
            function="trapped_traders",
            parameters={
                "lookback": {"type": "int", "min": 3, "max": 10, "step": 1, "default": 5},
                "delta_threshold": {"type": "float", "min": 1.5, "max": 3.0, "step": 0.5, "default": 2.0},
            },
            entry_columns=["entry_long_trapped", "entry_short_trapped"],
            requires=["volume_delta"],
            description="Trapped traders: aggressive buyers/sellers get stuck as price reverses against them. Entry on the reversal side.",
        ))

    # ==================================================================
    # PRICE ACTION signals  (signals.price_action)
    # ==================================================================

    def _register_price_action_signals(self) -> None:
        self._register(SignalDefinition(
            name="session_levels",
            category="price_action",
            signal_type="filter",
            module="signals.price_action",
            function="session_levels",
            parameters={},
            filter_columns=["signal_at_session_high", "signal_at_session_low"],
            indicator_columns=["session_high", "session_low"],
            description="Running session (daily) high and low levels. Flags when price is at/near session extremes.",
        ))

        self._register(SignalDefinition(
            name="previous_day_levels",
            category="price_action",
            signal_type="filter",
            module="signals.price_action",
            function="previous_day_levels",
            parameters={},
            filter_columns=["signal_above_prev_high", "signal_below_prev_low"],
            indicator_columns=["prev_day_high", "prev_day_low", "prev_day_close"],
            description="Previous day high, low, close levels. Filters on price breaking above/below prior day range.",
        ))

        self._register(SignalDefinition(
            name="range_breakout",
            category="price_action",
            signal_type="entry",
            module="signals.price_action",
            function="range_breakout",
            parameters={
                "lookback": {"type": "int", "min": 10, "max": 60, "step": 5, "default": 20},
            },
            entry_columns=["entry_long_breakout", "entry_short_breakout"],
            indicator_columns=["range_high", "range_low"],
            description="N-bar range breakout. Entry when price breaks above the lookback high or below the lookback low.",
        ))

        self._register(SignalDefinition(
            name="pivot_points",
            category="price_action",
            signal_type="filter",
            module="signals.price_action",
            function="pivot_points",
            parameters={
                "method": {"type": "list", "values": ["standard", "fibonacci", "woodie", "camarilla"], "default": "standard"},
            },
            filter_columns=["signal_above_pivot", "signal_below_pivot"],
            indicator_columns=["pivot", "r1", "r2", "r3", "s1", "s2", "s3"],
            description="Daily pivot points (standard, fibonacci, woodie, or camarilla). Filters on price relative to pivot.",
        ))

        self._register(SignalDefinition(
            name="opening_range",
            category="price_action",
            signal_type="entry",
            module="signals.price_action",
            function="opening_range",
            parameters={
                "minutes": {"type": "int", "min": 5, "max": 30, "step": 5, "default": 15},
            },
            entry_columns=["entry_long_orb", "entry_short_orb"],
            indicator_columns=["or_high", "or_low"],
            description="Opening Range Breakout. Entry on price breaking above/below the first N minutes range.",
        ))

        self._register(SignalDefinition(
            name="candle_patterns",
            category="price_action",
            signal_type="filter",
            module="signals.price_action",
            function="candle_patterns",
            parameters={},
            filter_columns=[
                "signal_hammer", "signal_shooting_star",
                "signal_engulfing_bull", "signal_engulfing_bear",
                "signal_doji", "signal_pin_bar_bull", "signal_pin_bar_bear",
            ],
            description="Classic candlestick pattern detection: hammer, shooting star, engulfing, doji, pin bars.",
        ))

    # ==================================================================
    # TIME FILTER signals  (signals.time_filters)
    # ==================================================================

    def _register_time_filter_signals(self) -> None:
        self._register(SignalDefinition(
            name="time_of_day",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="time_of_day",
            parameters={
                "start_hour": {"type": "int", "min": 0, "max": 23, "step": 1, "default": 9},
                "start_minute": {"type": "int", "min": 0, "max": 59, "step": 15, "default": 30},
                "end_hour": {"type": "int", "min": 0, "max": 23, "step": 1, "default": 16},
                "end_minute": {"type": "int", "min": 0, "max": 59, "step": 15, "default": 0},
            },
            filter_columns=["signal_time_allowed"],
            description="Time-of-day filter. Only allows trading within the specified hour:minute window (ET).",
        ))

        self._register(SignalDefinition(
            name="day_of_week",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="day_of_week",
            parameters={
                "allowed_days": {"type": "list", "values": [
                    [0, 1, 2, 3, 4],       # Mon-Fri
                    [1, 2, 3],              # Tue-Thu
                    [0, 1, 2, 3],           # Mon-Thu
                    [1, 2, 3, 4],           # Tue-Fri
                ], "default": [0, 1, 2, 3, 4]},
            },
            filter_columns=["signal_day_allowed"],
            description="Day-of-week filter. Only allows trading on specified weekdays (0=Mon, 4=Fri).",
        ))

        self._register(SignalDefinition(
            name="session_segment",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="session_segment",
            parameters={
                "segment": {"type": "list", "values": [
                    "pre_market", "open_drive", "morning", "midday", "afternoon", "close_drive",
                ], "default": "morning"},
            },
            filter_columns=["signal_in_segment"],
            description="Session segment filter. Segments: pre_market, open_drive, morning, midday, afternoon, close_drive.",
        ))

        self._register(SignalDefinition(
            name="minutes_since_open",
            category="time_filter",
            signal_type="indicator",
            module="signals.time_filters",
            function="minutes_since_open",
            parameters={
                "market_open_hour": {"type": "int", "min": 9, "max": 9, "step": 1, "default": 9},
                "market_open_minute": {"type": "int", "min": 30, "max": 30, "step": 1, "default": 30},
            },
            indicator_columns=["minutes_since_open"],
            description="Continuous value: minutes elapsed since the regular session open (9:30 ET).",
        ))

        self._register(SignalDefinition(
            name="first_n_minutes",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="first_n_minutes",
            parameters={
                "n": {"type": "int", "min": 5, "max": 60, "step": 5, "default": 30},
            },
            filter_columns=["signal_first_n_minutes"],
            description="Filter that is True only during the first N minutes of the regular session.",
        ))

        self._register(SignalDefinition(
            name="last_n_minutes",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="last_n_minutes",
            parameters={
                "n": {"type": "int", "min": 5, "max": 60, "step": 5, "default": 30},
            },
            filter_columns=["signal_last_n_minutes"],
            description="Filter that is True only during the last N minutes of the regular session.",
        ))

        self._register(SignalDefinition(
            name="london_overlap",
            category="time_filter",
            signal_type="filter",
            module="signals.time_filters",
            function="london_overlap",
            parameters={},
            filter_columns=["signal_london_overlap"],
            description="Filter for the London/US session overlap window (8:00-11:30 ET). Often sees highest NQ liquidity.",
        ))
