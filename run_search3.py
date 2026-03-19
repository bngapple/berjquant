#!/usr/bin/env python3
"""
Deep MC search — find strategies profitable over 2 years.
Tests on FULL 2-year dataset, MC validates robustness.
No window splitting — let the strategy trade the entire history.
"""

import json
import time
import random
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry, SignalDefinition
from strategies.generator import StrategyGenerator, GeneratedStrategy, ExitRules, SizingRules
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("deep")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")


def build_strat(entries, filters, exit_r, sizing):
    entry_dicts = []
    name_parts = []
    for sig in entries:
        entry_dicts.append({
            "signal_name": sig.name, "module": sig.module, "function": sig.function,
            "params": {k: v["default"] for k, v in sig.parameters.items()},
            "columns": {
                "long": next((c for c in sig.entry_columns if "long" in c), ""),
                "short": next((c for c in sig.entry_columns if "short" in c), ""),
            },
        })
        name_parts.append(sig.name.upper())

    filt_dicts = []
    for f in filters:
        filt_dicts.append({
            "signal_name": f.name, "module": f.module, "function": f.function,
            "params": {k: v["default"] for k, v in f.parameters.items()},
            "column": f.filter_columns[0] if f.filter_columns else "",
        })
        name_parts.append(f.name.upper())

    name_parts.append(f"SL{exit_r.stop_loss_value}_TP{exit_r.take_profit_value}")
    if sizing.method == "fixed":
        name_parts.append(f"{sizing.fixed_contracts}ct")
    else:
        name_parts.append(f"r{sizing.risk_pct}")

    return GeneratedStrategy(
        name="|".join(name_parts),
        entry_signals=entry_dicts,
        entry_filters=filt_dicts,
        exit_rules=exit_r,
        sizing_rules=sizing,
    )


def backtest(strategy, data, rm, config):
    try:
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 10:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, prop_rules, n_sims=3000):
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=50000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def main():
    logger.info("Loading 2-year data...")
    df = pl.read_parquet("data/processed/MNQ/1m/all.parquet")
    data = {"1m": df}
    logger.info(f"  {len(df):,} bars | {df['timestamp'][0]} -> {df['timestamp'][-1]}")

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
    generator = StrategyGenerator(registry)
    entry_sigs = registry.list_entry_signals()
    all_filters = registry.list_filters()

    # Aggressive exit variations — wide range
    exits = []
    for sl in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        for rr in [1.5, 2, 2.5, 3, 4, 5, 6, 8]:
            exits.append(ExitRules(stop_loss_value=float(sl), take_profit_value=float(sl * rr)))
    for atr_sl in [0.5, 1, 1.5, 2, 2.5, 3]:
        for atr_tp in [2, 3, 4, 5, 6, 8]:
            if atr_tp > atr_sl:
                exits.append(ExitRules(
                    stop_loss_type="atr_multiple", stop_loss_value=atr_sl,
                    take_profit_type="atr_multiple", take_profit_value=atr_tp))

    sizings = [
        SizingRules(method="fixed", fixed_contracts=1),
        SizingRules(method="fixed", fixed_contracts=2),
        SizingRules(method="fixed", fixed_contracts=3),
        SizingRules(method="fixed", fixed_contracts=5),
        SizingRules(method="risk_pct", risk_pct=0.005),
        SizingRules(method="risk_pct", risk_pct=0.01),
        SizingRules(method="risk_pct", risk_pct=0.015),
        SizingRules(method="risk_pct", risk_pct=0.02),
    ]

    winners = []
    tested = 0
    start = time.time()

    logger.info(f"Entries: {len(entry_sigs)}, Filters: {len(all_filters)}, Exits: {len(exits)}, Sizing: {len(sizings)}")
    logger.info("Running full 2-year backtests...\n")

    # ── ROUND 1: Single entry, no filter ──
    logger.info("=== ROUND 1: Single entry signals ===")
    for sig in entry_sigs:
        for ex in random.sample(exits, min(40, len(exits))):
            for sz in random.sample(sizings, min(4, len(sizings))):
                strat = build_strat([sig], [], ex, sz)
                out = backtest(strat, data, rm, config)
                tested += 1
                if out:
                    trades, m = out
                    if m.total_pnl > 0 and m.profit_factor > 1.0:
                        mc = mc_test(trades, prop_rules, 2000)
                        if mc.median_return > 0 and mc.probability_of_profit > 0.6:
                            winners.append((strat, m, mc, trades))
                            logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                        elif m.total_pnl > 2000:
                            logger.info(f"  → {strat.name[:55]} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
        elapsed = time.time() - start
        logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/elapsed:.0f}/s | {len(winners)}W]")

    # ── ROUND 2: Single entry + filter ──
    logger.info("\n=== ROUND 2: Entry + filter ===")
    for sig in entry_sigs:
        for filt in all_filters:
            for ex in random.sample(exits, min(20, len(exits))):
                sz = random.choice(sizings)
                strat = build_strat([sig], [filt], ex, sz)
                out = backtest(strat, data, rm, config)
                tested += 1
                if out:
                    trades, m = out
                    if m.total_pnl > 0 and m.profit_factor > 1.0:
                        mc = mc_test(trades, prop_rules, 2000)
                        if mc.median_return > 0 and mc.probability_of_profit > 0.6:
                            winners.append((strat, m, mc, trades))
                            logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                        elif m.total_pnl > 2000:
                            logger.info(f"  → {strat.name[:55]} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
            if tested % 500 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/elapsed:.0f}/s | {len(winners)}W]")

    # ── ROUND 3: Two entries (cross-category) ──
    logger.info("\n=== ROUND 3: Two entry signals ===")
    cats = {}
    for s in entry_sigs:
        cats.setdefault(s.category, []).append(s)
    cat_list = list(cats.keys())
    for i in range(len(cat_list)):
        for j in range(i+1, len(cat_list)):
            for sa in cats[cat_list[i]]:
                for sb in cats[cat_list[j]]:
                    for ex in random.sample(exits, min(10, len(exits))):
                        sz = random.choice(sizings)
                        strat = build_strat([sa, sb], [], ex, sz)
                        out = backtest(strat, data, rm, config)
                        tested += 1
                        if out:
                            trades, m = out
                            if m.total_pnl > 0 and m.profit_factor > 1.0:
                                mc = mc_test(trades, prop_rules, 2000)
                                if mc.median_return > 0 and mc.probability_of_profit > 0.6:
                                    winners.append((strat, m, mc, trades))
                                    logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                    if tested % 500 == 0:
                        elapsed = time.time() - start
                        logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/elapsed:.0f}/s | {len(winners)}W]")

    # ── ROUND 4: Mutate winners ──
    if winners:
        logger.info(f"\n=== ROUND 4: Mutating {len(winners)} winners ===")
        for strat, m, mc, trades in list(winners):
            try:
                vars = generator.generate_parameter_variations(strat, num_variations=40, method="random")
                for v in vars:
                    out = backtest(v, data, rm, config)
                    tested += 1
                    if out:
                        t2, m2 = out
                        if m2.total_pnl > m.total_pnl * 0.8 and m2.profit_factor > 1.0:
                            mc2 = mc_test(t2, prop_rules, 2000)
                            if mc2.median_return > 0 and mc2.probability_of_profit > 0.6:
                                winners.append((v, m2, mc2, t2))
                                if mc2.median_return > mc.median_return:
                                    logger.info(f"  ★★ {v.name[:55]} | PnL=${m2.total_pnl:,.0f} | MC=${mc2.median_return:,.0f} | P={mc2.probability_of_profit:.0%}")
            except Exception:
                continue

    # ── FINAL REPORT ──
    elapsed = time.time() - start
    print(f"\n{'='*90}")
    print(f"  DEEP SEARCH COMPLETE — 2 YEARS OF 1-MINUTE NQ DATA (DATABENTO)")
    print(f"{'='*90}")
    print(f"  Strategies tested:  {tested:,}")
    print(f"  Time:               {elapsed/60:.1f} min")
    print(f"  Winners (MC prof):  {len(winners)}")
    print()

    if winners:
        winners.sort(key=lambda w: w[2].median_return, reverse=True)
        print(f"  {'#':<4} {'Strategy':<50} {'Trades':>6} {'WR':>6} {'PF':>6} {'2yr PnL':>10} {'MC Med':>10} {'P(prof)':>8}")
        print(f"  {'-'*100}")
        for i, (s, m, mc, _) in enumerate(winners[:25], 1):
            print(f"  {i:<4} {s.name[:50]:<50} {m.total_trades:>6} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f} ${m.total_pnl:>9,.0f} ${mc.median_return:>9,.0f} {mc.probability_of_profit:>7.1%}")
        print(f"  {'-'*100}")

        # Monthly breakdown for #1
        best_strat, best_m, best_mc, best_trades = winners[0]
        monthly = {}
        for t in best_trades:
            mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:7]
            monthly[mo] = monthly.get(mo, 0) + t.net_pnl
        print(f"\n  BEST STRATEGY MONTHLY P&L: {best_strat.name}")
        for mo in sorted(monthly.keys()):
            p = monthly[mo]
            bar = "█" * max(1, int(abs(p) / 300))
            print(f"    {mo}: {'+'if p>=0 else '-'}${abs(p):>8,.2f}  {bar}{'*' if p<0 else ''}")
        prof_months = sum(1 for p in monthly.values() if p > 0)
        print(f"    Profitable months: {prof_months}/{len(monthly)} ({prof_months/len(monthly):.0%})")

        # Save
        output = []
        for s, m, mc, _ in winners:
            output.append({
                "name": s.name, "strategy": s.to_dict(),
                "trades": m.total_trades, "win_rate": m.win_rate,
                "profit_factor": m.profit_factor, "sharpe": m.sharpe_ratio,
                "total_pnl_2yr": m.total_pnl, "max_drawdown": m.max_drawdown,
                "mc_median": mc.median_return, "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin, "mc_composite": mc.composite_score,
            })
        with open("reports/mc_winners_2yr.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/mc_winners_2yr.json")
    else:
        print("  No MC-validated profitable strategies found on 2 years of data.")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
