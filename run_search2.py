#!/usr/bin/env python3
"""
Fast MC strategy search on 2 years of 1m data.
Strategy: backtest on rolling 30-day windows across 2 years,
find strategies that consistently make $2K+ per window.
"""

import json
import time
import random
import logging
from pathlib import Path

import numpy as np
import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry
from strategies.generator import StrategyGenerator, GeneratedStrategy, ExitRules, SizingRules
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("search2")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
TARGET_PNL = 2000.0


def load_data():
    return pl.read_parquet("data/processed/MNQ/1m/all.parquet")


def split_into_windows(df, window_days=30):
    """Split data into non-overlapping 30-day windows for walk-forward testing."""
    dates = df["timestamp"].cast(pl.Date).unique().sort()
    windows = []
    i = 0
    while i + window_days <= len(dates):
        start = dates[i]
        end = dates[i + window_days - 1]
        window = df.filter(
            (pl.col("timestamp").cast(pl.Date) >= start) &
            (pl.col("timestamp").cast(pl.Date) <= end)
        )
        if len(window) > 100:
            windows.append(window)
        i += window_days
    return windows


def make_exit_variations():
    exits = []
    for sl in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        for rr in [2.0, 2.5, 3.0, 4.0, 5.0]:
            exits.append(ExitRules(
                stop_loss_type="fixed_points", stop_loss_value=sl,
                take_profit_type="fixed_points", take_profit_value=sl * rr,
            ))
    for atr_sl in [1.0, 1.5, 2.0, 3.0]:
        for atr_tp in [3.0, 4.0, 5.0]:
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
    ]


def build_strategy(sig_def, exit_rules, sizing, filt_def=None):
    entry_dict = {
        "signal_name": sig_def.name, "module": sig_def.module, "function": sig_def.function,
        "params": {k: v["default"] for k, v in sig_def.parameters.items()},
        "columns": {
            "long": next((c for c in sig_def.entry_columns if "long" in c), ""),
            "short": next((c for c in sig_def.entry_columns if "short" in c), ""),
        },
    }
    filters = []
    name_parts = [sig_def.name]
    if filt_def:
        filters.append({
            "signal_name": filt_def.name, "module": filt_def.module, "function": filt_def.function,
            "params": {k: v["default"] for k, v in filt_def.parameters.items()},
            "column": filt_def.filter_columns[0] if filt_def.filter_columns else "",
        })
        name_parts.append(filt_def.name)
    name_parts.append(f"SL{exit_rules.stop_loss_value}_TP{exit_rules.take_profit_value}")
    return GeneratedStrategy(
        name="|".join(name_parts),
        entry_signals=[entry_dict],
        entry_filters=filters,
        exit_rules=exit_rules,
        sizing_rules=sizing,
    )


def test_on_window(strategy, window_df, rm, config):
    """Backtest on a single 30-day window."""
    try:
        bt = VectorizedBacktester(data={"1m": window_df}, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 3:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def test_across_windows(strategy, windows, rm, config):
    """Test strategy across all 30-day windows. Return per-window results."""
    results = []
    for window_df in windows:
        out = test_on_window(strategy, window_df, rm, config)
        if out is None:
            results.append((0, 0.0))
        else:
            trades, metrics = out
            results.append((metrics.total_trades, metrics.total_pnl))
    return results


def main():
    logger.info("Loading 2-year 1m data...")
    df = load_data()
    logger.info(f"  {len(df):,} bars | {df['timestamp'][0]} -> {df['timestamp'][-1]}")

    # Split into 30-day windows
    windows = split_into_windows(df, window_days=30)
    logger.info(f"  Split into {len(windows)} non-overlapping 30-day windows")

    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2024-01-01", end_date="2026-12-31",
        slippage_ticks=2, initial_capital=50000.0,
    )

    registry = SignalRegistry()
    all_exits = make_exit_variations()
    all_sizing = make_sizing_variations()
    entry_signals = registry.list_entry_signals()
    good_filters = [f for f in registry.list_filters()
                    if f.category in ("orderflow", "volume", "volatility", "trend")]

    # Use first window for fast screening, then validate winners across all windows
    screen_window = windows[0]
    validate_windows = windows[1:]

    winners = []
    total_tested = 0
    start = time.time()

    logger.info(f"\nTarget: ${TARGET_PNL:,.0f}/month across {len(windows)} months")
    logger.info(f"Screen on window 1, validate on windows 2-{len(windows)}")
    logger.info(f"Entries: {len(entry_signals)}, Filters: {len(good_filters)}, Exits: {len(all_exits)}, Sizing: {len(all_sizing)}")
    logger.info(f"Starting search...\n")

    # Phase 1: Screen single-signal strategies on first window
    logger.info("=== PHASE 1: Single signals — fast screen ===")
    promising = []
    for sig in entry_signals:
        for exit_r in all_exits:
            for sizing in all_sizing:
                strat = build_strategy(sig, exit_r, sizing)
                out = test_on_window(strat, screen_window, rm, config)
                total_tested += 1
                if out and out[1].total_pnl > 300:
                    promising.append((strat, out[1].total_pnl, out[0]))
                if total_tested % 500 == 0:
                    elapsed = time.time() - start
                    logger.info(f"  [{total_tested:,} tested | {elapsed:.0f}s | {total_tested/elapsed:.0f}/sec | {len(promising)} promising]")

    logger.info(f"\n  Phase 1: {len(promising)} promising from {total_tested} tested")

    # Phase 2: Screen entry+filter combos
    logger.info("\n=== PHASE 2: Entry + filter combos — fast screen ===")
    for sig in entry_signals:
        for filt in good_filters:
            for exit_r in random.sample(all_exits, min(15, len(all_exits))):
                sizing = random.choice(all_sizing)
                strat = build_strategy(sig, exit_r, sizing, filt_def=filt)
                out = test_on_window(strat, screen_window, rm, config)
                total_tested += 1
                if out and out[1].total_pnl > 300:
                    promising.append((strat, out[1].total_pnl, out[0]))
                if total_tested % 500 == 0:
                    elapsed = time.time() - start
                    logger.info(f"  [{total_tested:,} tested | {elapsed:.0f}s | {total_tested/elapsed:.0f}/sec | {len(promising)} promising]")

    logger.info(f"\n  Phase 2 done: {len(promising)} total promising from {total_tested} tested")

    # Phase 3: Validate promising across ALL windows
    logger.info(f"\n=== PHASE 3: Walk-forward validation on {len(validate_windows)} windows ===")
    promising.sort(key=lambda x: x[1], reverse=True)
    top_promising = promising[:200]  # validate top 200

    for idx, (strat, screen_pnl, screen_trades) in enumerate(top_promising):
        window_results = test_across_windows(strat, validate_windows, rm, config)
        pnls = [r[1] for r in window_results]
        trades = [r[0] for r in window_results]

        avg_pnl = np.mean(pnls) if pnls else 0
        pct_profitable = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        total_trades = sum(trades)
        all_trades_flat = []
        for w in validate_windows:
            out = test_on_window(strat, w, rm, config)
            if out:
                all_trades_flat.extend(out[0])

        if avg_pnl > TARGET_PNL * 0.5 and pct_profitable > 0.4 and total_trades > 20:
            # MC test on all trades combined
            if all_trades_flat:
                mc = MCConfig(n_simulations=2000, initial_capital=50000.0, prop_firm_rules=prop_rules, seed=42)
                sim = MonteCarloSimulator(mc)
                mc_result = sim.run(all_trades_flat, strategy_name=strat.name)

                if mc_result.median_return > TARGET_PNL:
                    winners.append({
                        "strategy": strat,
                        "screen_pnl": screen_pnl,
                        "avg_monthly_pnl": avg_pnl,
                        "pct_profitable_months": pct_profitable,
                        "monthly_pnls": pnls,
                        "total_trades": total_trades,
                        "mc_median": mc_result.median_return,
                        "mc_p_profit": mc_result.probability_of_profit,
                        "mc_p_ruin": mc_result.probability_of_ruin,
                        "mc_pass_rate": mc_result.prop_firm_pass_rate,
                        "mc_composite": mc_result.composite_score,
                    })
                    logger.info(f"  ★ WINNER: {strat.name[:50]} | Avg=${avg_pnl:,.0f}/mo | {pct_profitable:.0%} profitable | MC=${mc_result.median_return:,.0f} | P(profit)={mc_result.probability_of_profit:.1%}")
                elif mc_result.median_return > 0:
                    logger.info(f"  → Close: {strat.name[:50]} | Avg=${avg_pnl:,.0f}/mo | MC=${mc_result.median_return:,.0f}")

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start
            logger.info(f"  [{idx+1}/{len(top_promising)} validated | {elapsed:.0f}s | {len(winners)} winners]")

    # Phase 4: Parameter mutations on winners
    if winners:
        logger.info(f"\n=== PHASE 4: Parameter mutations on {len(winners)} winners ===")
        gen = StrategyGenerator(registry)
        for w in list(winners):
            try:
                variations = gen.generate_parameter_variations(w["strategy"], num_variations=30, method="random")
                for var in variations:
                    window_results = test_across_windows(var, validate_windows, rm, config)
                    pnls = [r[1] for r in window_results]
                    avg_pnl = np.mean(pnls) if pnls else 0
                    pct_prof = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
                    total_t = sum(r[0] for r in window_results)

                    if avg_pnl > TARGET_PNL * 0.5 and pct_prof > 0.4 and total_t > 20:
                        all_t = []
                        for w_df in validate_windows:
                            out = test_on_window(var, w_df, rm, config)
                            if out:
                                all_t.extend(out[0])
                        if all_t:
                            mc = MCConfig(n_simulations=2000, initial_capital=50000.0, prop_firm_rules=prop_rules, seed=42)
                            sim = MonteCarloSimulator(mc)
                            mc_result = sim.run(all_t, strategy_name=var.name)
                            if mc_result.median_return > TARGET_PNL:
                                winners.append({
                                    "strategy": var,
                                    "screen_pnl": 0,
                                    "avg_monthly_pnl": avg_pnl,
                                    "pct_profitable_months": pct_prof,
                                    "monthly_pnls": pnls,
                                    "total_trades": total_t,
                                    "mc_median": mc_result.median_return,
                                    "mc_p_profit": mc_result.probability_of_profit,
                                    "mc_p_ruin": mc_result.probability_of_ruin,
                                    "mc_pass_rate": mc_result.prop_firm_pass_rate,
                                    "mc_composite": mc_result.composite_score,
                                })
                                logger.info(f"  ★ MUTANT: {var.name[:50]} | Avg=${avg_pnl:,.0f}/mo | MC=${mc_result.median_return:,.0f} | P(profit)={mc_result.probability_of_profit:.1%}")
            except Exception:
                continue

    # Final report
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"  SEARCH COMPLETE — 2 YEARS OF 1-MINUTE NQ DATA")
    print(f"{'='*80}")
    print(f"  Total tested:    {total_tested:,} strategies")
    print(f"  Time:            {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Promising:       {len(promising)}")
    print(f"  Walk-fwd validated: {min(len(top_promising), 200)}")
    print(f"  MC Winners (>${TARGET_PNL:,.0f}/mo median): {len(winners)}")
    print()

    if winners:
        winners.sort(key=lambda w: w["mc_median"], reverse=True)
        print(f"  {'Rank':<5} {'Strategy':<45} {'Avg/Mo':>9} {'%Prof':>6} {'MC Med':>9} {'P(prof)':>8} {'Trades':>7}")
        print(f"  {'-'*92}")
        for i, w in enumerate(winners[:20], 1):
            print(f"  {i:<5} {w['strategy'].name[:45]:<45} ${w['avg_monthly_pnl']:>8,.0f} {w['pct_profitable_months']:>5.0%} ${w['mc_median']:>8,.0f} {w['mc_p_profit']:>7.1%} {w['total_trades']:>7}")
        print(f"  {'-'*92}")

        # Save
        output = []
        for w in winners:
            output.append({
                "name": w["strategy"].name,
                "strategy": w["strategy"].to_dict(),
                "avg_monthly_pnl": w["avg_monthly_pnl"],
                "pct_profitable_months": w["pct_profitable_months"],
                "monthly_pnls": w["monthly_pnls"],
                "total_trades": w["total_trades"],
                "mc_median": w["mc_median"],
                "mc_p_profit": w["mc_p_profit"],
                "mc_p_ruin": w["mc_p_ruin"],
                "mc_pass_rate": w["mc_pass_rate"],
                "mc_composite": w["mc_composite"],
            })
        with open("reports/mc_winners_2yr.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/mc_winners_2yr.json")
    else:
        print("  No strategies hit $2K/month across 2 years of walk-forward windows.")
        print("  Top promising strategies by average monthly P&L:")
        promising.sort(key=lambda x: x[1], reverse=True)
        for i, (s, pnl, trades) in enumerate(promising[:10], 1):
            print(f"    {i}. {s.name[:50]} | Screen PnL=${pnl:,.0f} | {len(trades)} trades")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
