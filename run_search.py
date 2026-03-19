#!/usr/bin/env python3
"""
Brute-force Monte Carlo strategy search.
Keeps generating, testing, and mutating strategies until we find ones
that consistently make $2K+ over 30 days.
"""

import sys
import time
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC, NQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry
from strategies.generator import StrategyGenerator, GeneratedStrategy, ExitRules, SizingRules
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("search")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
TARGET_PNL = 2000.0


def load_data():
    df_1m = pl.read_parquet("data/processed/MNQ/1m/all.parquet")
    df_5m = pl.read_parquet("data/processed/MNQ/5m/all.parquet")
    return {"1m": df_1m, "5m": df_5m}


def make_exit_variations():
    """Generate diverse exit rule combinations."""
    exits = []
    # Fixed point stops with various R:R ratios
    for sl in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        for rr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            tp = sl * rr
            exits.append(ExitRules(
                stop_loss_type="fixed_points", stop_loss_value=sl,
                take_profit_type="fixed_points", take_profit_value=tp,
            ))
    # ATR-based
    for atr_sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for atr_tp in [2.0, 3.0, 4.0, 5.0, 6.0]:
            if atr_tp > atr_sl:
                exits.append(ExitRules(
                    stop_loss_type="atr_multiple", stop_loss_value=atr_sl,
                    take_profit_type="atr_multiple", take_profit_value=atr_tp,
                ))
    return exits


def make_sizing_variations():
    return [
        SizingRules(method="fixed", fixed_contracts=1),
        SizingRules(method="fixed", fixed_contracts=2),
        SizingRules(method="fixed", fixed_contracts=3),
        SizingRules(method="risk_pct", risk_pct=0.005),
        SizingRules(method="risk_pct", risk_pct=0.01),
        SizingRules(method="risk_pct", risk_pct=0.015),
        SizingRules(method="risk_pct", risk_pct=0.02),
    ]


def backtest_strategy(strategy, data, rm, config):
    """Run a single backtest, return (trades, metrics) or None on failure."""
    try:
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 5:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, n_sims=3000, prop_rules=None):
    """Run Monte Carlo on trades, return MCResult."""
    mc_config = MCConfig(
        n_simulations=n_sims,
        initial_capital=50000.0,
        prop_firm_rules=prop_rules,
        seed=random.randint(0, 999999),
    )
    sim = MonteCarloSimulator(mc_config)
    return sim.run(trades, strategy_name="test")


def main():
    logger.info("Loading data...")
    data = load_data()
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2025-01-01", end_date="2026-12-31",
        slippage_ticks=2, initial_capital=50000.0,
    )

    registry = SignalRegistry()
    generator = StrategyGenerator(registry)
    all_exits = make_exit_variations()
    all_sizing = make_sizing_variations()

    winners = []
    total_tested = 0
    round_num = 0
    start = time.time()

    logger.info(f"Target: ${TARGET_PNL:,.0f} in 30 days")
    logger.info(f"Exit variations: {len(all_exits)}, Sizing variations: {len(all_sizing)}")
    logger.info(f"Starting brute-force search...\n")

    # Round 1: Single entry signals with all exit/sizing combos
    round_num += 1
    logger.info(f"=== ROUND {round_num}: Single entry signals × all exits × all sizing ===")
    entry_signals = registry.list_entry_signals()
    for sig_def in entry_signals:
        for exit_rules in random.sample(all_exits, min(30, len(all_exits))):
            for sizing in random.sample(all_sizing, min(3, len(all_sizing))):
                try:
                    entry_dict = {
                        "signal_name": sig_def.name,
                        "module": sig_def.module,
                        "function": sig_def.function,
                        "params": {k: v["default"] for k, v in sig_def.parameters.items()},
                        "columns": {
                            "long": next((c for c in sig_def.entry_columns if "long" in c), ""),
                            "short": next((c for c in sig_def.entry_columns if "short" in c), ""),
                        },
                    }
                    strat = GeneratedStrategy(
                        name=f"{sig_def.name}|SL{exit_rules.stop_loss_value}_TP{exit_rules.take_profit_value}_S{sizing.method}",
                        entry_signals=[entry_dict],
                        entry_filters=[],
                        exit_rules=exit_rules,
                        sizing_rules=sizing,
                    )
                    result = backtest_strategy(strat, data, rm, config)
                    total_tested += 1
                    if result is None:
                        continue
                    trades, metrics = result
                    if metrics.total_pnl > 500:  # Promising
                        mc = mc_test(trades, n_sims=2000, prop_rules=prop_rules)
                        if mc.median_return > TARGET_PNL:
                            winners.append((strat, metrics, mc))
                            logger.info(f"  ★ WINNER: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f} | P(profit)={mc.probability_of_profit:.1%}")
                        elif mc.median_return > 0:
                            logger.info(f"  → Promising: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f}")
                except Exception:
                    total_tested += 1
                    continue

            if total_tested % 100 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{total_tested} tested | {elapsed:.0f}s | {total_tested/elapsed:.0f}/sec | {len(winners)} winners]")

    # Round 2: Two entry signals (cross-category) with promising exits
    round_num += 1
    logger.info(f"\n=== ROUND {round_num}: Two entry signals (cross-category) ===")
    categories = {}
    for sig in entry_signals:
        categories.setdefault(sig.category, []).append(sig)

    cat_names = list(categories.keys())
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            sigs_a = categories[cat_names[i]]
            sigs_b = categories[cat_names[j]]
            for sa in sigs_a:
                for sb in sigs_b:
                    for exit_rules in random.sample(all_exits, min(20, len(all_exits))):
                        for sizing in random.sample(all_sizing, min(2, len(all_sizing))):
                            try:
                                entry_a = {
                                    "signal_name": sa.name, "module": sa.module, "function": sa.function,
                                    "params": {k: v["default"] for k, v in sa.parameters.items()},
                                    "columns": {
                                        "long": next((c for c in sa.entry_columns if "long" in c), ""),
                                        "short": next((c for c in sa.entry_columns if "short" in c), ""),
                                    },
                                }
                                entry_b = {
                                    "signal_name": sb.name, "module": sb.module, "function": sb.function,
                                    "params": {k: v["default"] for k, v in sb.parameters.items()},
                                    "columns": {
                                        "long": next((c for c in sb.entry_columns if "long" in c), ""),
                                        "short": next((c for c in sb.entry_columns if "short" in c), ""),
                                    },
                                }
                                strat = GeneratedStrategy(
                                    name=f"{sa.name}+{sb.name}|SL{exit_rules.stop_loss_value}_TP{exit_rules.take_profit_value}",
                                    entry_signals=[entry_a, entry_b],
                                    entry_filters=[],
                                    exit_rules=exit_rules,
                                    sizing_rules=sizing,
                                )
                                result = backtest_strategy(strat, data, rm, config)
                                total_tested += 1
                                if result is None:
                                    continue
                                trades, metrics = result
                                if metrics.total_pnl > 500:
                                    mc = mc_test(trades, n_sims=2000, prop_rules=prop_rules)
                                    if mc.median_return > TARGET_PNL:
                                        winners.append((strat, metrics, mc))
                                        logger.info(f"  ★ WINNER: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f} | P(profit)={mc.probability_of_profit:.1%}")
                                    elif mc.median_return > 0:
                                        logger.info(f"  → Promising: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f}")
                            except Exception:
                                total_tested += 1
                                continue

                    if total_tested % 200 == 0:
                        elapsed = time.time() - start
                        logger.info(f"  [{total_tested} tested | {elapsed:.0f}s | {total_tested/elapsed:.0f}/sec | {len(winners)} winners]")

    # Round 3: Entry + filter combos
    round_num += 1
    logger.info(f"\n=== ROUND {round_num}: Entry + orderflow/volume filters ===")
    good_filters = [f for f in registry.list_filters() if f.category in ("orderflow", "volume", "volatility")]
    for sig in entry_signals:
        for filt in good_filters:
            for exit_rules in random.sample(all_exits, min(15, len(all_exits))):
                sizing = random.choice(all_sizing)
                try:
                    entry_dict = {
                        "signal_name": sig.name, "module": sig.module, "function": sig.function,
                        "params": {k: v["default"] for k, v in sig.parameters.items()},
                        "columns": {
                            "long": next((c for c in sig.entry_columns if "long" in c), ""),
                            "short": next((c for c in sig.entry_columns if "short" in c), ""),
                        },
                    }
                    filt_dict = {
                        "signal_name": filt.name, "module": filt.module, "function": filt.function,
                        "params": {k: v["default"] for k, v in filt.parameters.items()},
                        "column": filt.filter_columns[0] if filt.filter_columns else "",
                    }
                    strat = GeneratedStrategy(
                        name=f"{sig.name}|{filt.name}|SL{exit_rules.stop_loss_value}_TP{exit_rules.take_profit_value}",
                        entry_signals=[entry_dict],
                        entry_filters=[filt_dict],
                        exit_rules=exit_rules,
                        sizing_rules=sizing,
                    )
                    result = backtest_strategy(strat, data, rm, config)
                    total_tested += 1
                    if result is None:
                        continue
                    trades, metrics = result
                    if metrics.total_pnl > 500:
                        mc = mc_test(trades, n_sims=2000, prop_rules=prop_rules)
                        if mc.median_return > TARGET_PNL:
                            winners.append((strat, metrics, mc))
                            logger.info(f"  ★ WINNER: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f} | P(profit)={mc.probability_of_profit:.1%}")
                        elif mc.median_return > 0:
                            logger.info(f"  → Promising: {strat.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f}")
                except Exception:
                    total_tested += 1
                    continue

                if total_tested % 200 == 0:
                    elapsed = time.time() - start
                    logger.info(f"  [{total_tested} tested | {elapsed:.0f}s | {total_tested/elapsed:.0f}/sec | {len(winners)} winners]")

    # Round 4: Parameter variations on winners/promising
    if winners:
        round_num += 1
        logger.info(f"\n=== ROUND {round_num}: Parameter mutations on {len(winners)} winners ===")
        for strat, _, _ in list(winners):
            try:
                variations = generator.generate_parameter_variations(strat, num_variations=50, method="random")
                for var in variations:
                    result = backtest_strategy(var, data, rm, config)
                    total_tested += 1
                    if result is None:
                        continue
                    trades, metrics = result
                    if metrics.total_pnl > 500:
                        mc = mc_test(trades, n_sims=3000, prop_rules=prop_rules)
                        if mc.median_return > TARGET_PNL:
                            winners.append((var, metrics, mc))
                            logger.info(f"  ★ MUTANT WINNER: {var.name} | PnL=${metrics.total_pnl:,.0f} | MC median=${mc.median_return:,.0f} | P(profit)={mc.probability_of_profit:.1%}")
            except Exception:
                continue

    # Final report
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total strategies tested: {total_tested:,}")
    print(f"  Time elapsed:            {elapsed:.0f}s ({total_tested/max(elapsed,1):.0f} strategies/sec)")
    print(f"  Winners (MC median > ${TARGET_PNL:,.0f}): {len(winners)}")
    print()

    if winners:
        # Sort by MC median return
        winners.sort(key=lambda x: x[2].median_return, reverse=True)
        print(f"  {'Rank':<5} {'Strategy':<50} {'BT PnL':>10} {'MC Median':>10} {'P(profit)':>10} {'Trades':>7}")
        print(f"  {'-'*95}")
        for i, (strat, metrics, mc) in enumerate(winners[:20], 1):
            print(f"  {i:<5} {strat.name[:50]:<50} ${metrics.total_pnl:>9,.0f} ${mc.median_return:>9,.0f} {mc.probability_of_profit:>9.1%} {metrics.total_trades:>7}")
        print(f"  {'-'*95}")

        # Save winners
        output = []
        for strat, metrics, mc in winners:
            output.append({
                "name": strat.name,
                "strategy": strat.to_dict(),
                "backtest_pnl": metrics.total_pnl,
                "backtest_trades": metrics.total_trades,
                "backtest_win_rate": metrics.win_rate,
                "backtest_sharpe": metrics.sharpe_ratio,
                "backtest_pf": metrics.profit_factor,
                "mc_median_return": mc.median_return,
                "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin,
                "mc_composite": mc.composite_score,
            })
        out_path = Path("reports/mc_winners.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Winners saved to {out_path}")
    else:
        print("  No strategies hit the target. Need more data or signal tuning.")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
