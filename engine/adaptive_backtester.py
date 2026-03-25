"""Adaptive backtester — extends FastBacktester with ATR-adaptive exits and regime filters.
New file — does NOT modify any existing code."""

import copy
from engine.fast_backtester import FastBacktester
from engine.adaptive import atr_stops, add_htf_trend, add_regime, add_volatility_regime
from engine.utils import BacktestConfig, ContractSpec, Position
from strategies.generator import GeneratedStrategy


class AdaptiveBacktester(FastBacktester):
    """FastBacktester + ATR-adaptive exits + regime/trend/vol filters."""

    def run(self, strategy):
        # Pre-process data with adaptive features
        key = list(self.data.keys())[0]
        df = self.data[key].clone()

        # Add adaptive infrastructure
        er = strategy.exit_rules if hasattr(strategy, 'exit_rules') else None
        atr_period = 14
        sl_mult = 1.5
        tp_mult = 1.5

        if er and hasattr(er, 'stop_loss_type') and er.stop_loss_type == "atr_adaptive":
            sl_mult = er.stop_loss_value  # Reuse value field as multiplier
            tp_mult = er.take_profit_value
            atr_period = 14

        df = atr_stops(df, sl_mult, tp_mult, atr_period)
        df = add_htf_trend(df)
        df = add_regime(df)
        df = add_volatility_regime(df, atr_period)

        # Replace data with enriched version
        enriched_data = {key: df}
        self.data = enriched_data

        # Run parent's backtest
        result = super().run(strategy)

        return result
