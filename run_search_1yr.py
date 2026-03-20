#!/usr/bin/env python3
"""
Full search + evolution on Year 1 (Mar 2024 – Mar 2025).
Goal: Find the most profitable strategy possible, then validate on Year 2.
"""

import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path

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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("1yr_search")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def make_exits():
    exits = []
    for sl in [3, 5, 8, 10, 12, 15, 20, 25, 30]:
        for rr in [1.5, 2, 2.5, 3, 4, 5, 6, 8]:
            exits.append(ExitRules(stop_loss_value=float(sl), take_profit_value=float(sl * rr)))
    for atr_sl in [0.5, 1, 1.5, 2, 2.5, 3]:
        for atr_tp in [2, 3, 4, 5, 6, 8]:
            if atr_tp > atr_sl:
                exits.append(ExitRules(
                    stop_loss_type="atr_multiple", stop_loss_value=atr_sl,
                    take_profit_type="atr_multiple", take_profit_value=atr_tp))
    return exits


def make_sizing():
    return [
        SizingRules(method="fixed", fixed_contracts=1),
        SizingRules(method="fixed", fixed_contracts=2),
        SizingRules(method="fixed", fixed_contracts=3),
        SizingRules(method="fixed", fixed_contracts=5),
        SizingRules(method="risk_pct", risk_pct=0.005),
        SizingRules(method="risk_pct", risk_pct=0.01),
        SizingRules(method="risk_pct", risk_pct=0.015),
        SizingRules(method="risk_pct", risk_pct=0.02),
    ]


def backtest_obj(strategy, data, rm, config):
    try:
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 10:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def backtest_dict(sd, data, rm, config):
    try:
        sd["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd)
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


def fitness(metrics, mc):
    mc_med = mc.median_return
    p_prof = mc.probability_of_profit
    pf = min(metrics.profit_factor, 10.0)
    dd = abs(metrics.max_drawdown)
    trades = metrics.total_trades
    trade_factor = min(1.0, trades / 30.0)
    dd_factor = max(0, 1.0 - (dd / 2000.0))
    return (
        mc_med * 0.4 +
        p_prof * 10000 * 0.2 +
        pf * 1000 * 0.15 +
        metrics.total_pnl * 0.15 +
        dd_factor * 5000 * 0.1
    ) * trade_factor


def mutate_params(params, intensity=0.3):
    new = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            delta = abs(v) * intensity * random.uniform(-1, 1)
            new_val = v + delta
            new[k] = max(1, int(round(new_val))) if isinstance(v, int) else round(max(0.01, new_val), 4)
        else:
            new[k] = v
    return new


def mutate_strategy(sd, intensity=0.3):
    new = copy.deepcopy(sd)
    for sig in new["entry_signals"]:
        sig["params"] = mutate_params(sig["params"], intensity)
    for filt in new.get("entry_filters", []):
        filt["params"] = mutate_params(filt["params"], intensity)
    er = new["exit_rules"]
    sl, tp = er["stop_loss_value"], er["take_profit_value"]
    er["stop_loss_value"] = round(max(1.0, sl + sl * intensity * random.uniform(-1, 1)), 1)
    er["take_profit_value"] = round(max(er["stop_loss_value"] * 1.2, tp + tp * intensity * random.uniform(-1, 1)), 1)
    sz = new["sizing_rules"]
    if sz["method"] == "fixed":
        sz["fixed_contracts"] = max(1, int(sz["fixed_contracts"] + sz["fixed_contracts"] * intensity * random.uniform(-1, 1)))
    elif sz["method"] == "risk_pct":
        sz["risk_pct"] = round(max(0.002, min(0.03, sz["risk_pct"] + sz["risk_pct"] * intensity * random.uniform(-1, 1))), 4)
    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = "|".join(parts[:-1]) if len(parts) >= 2 else parts[0]
    new["name"] = f"{base}|gen_{h}"
    return new


def crossover(a, b):
    child = copy.deepcopy(a)
    if random.random() < 0.5:
        child["exit_rules"] = copy.deepcopy(b["exit_rules"])
    if random.random() < 0.5:
        child["sizing_rules"] = copy.deepcopy(b["sizing_rules"])
    if b.get("entry_filters") and random.random() < 0.5:
        child["entry_filters"] = copy.deepcopy(b["entry_filters"])
    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = child["name"].split("|")
    base = "|".join(parts[:-1]) if len(parts) >= 2 else parts[0]
    child["name"] = f"{base}|breed_{h}"
    return child


# ══════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════

def run_search(data, rm, config, prop_rules):
    registry = SignalRegistry()
    generator = StrategyGenerator(registry)
    all_exits = make_exits()
    all_sizing = make_sizing()
    entry_sigs = registry.list_entry_signals()
    all_filters = registry.list_filters()

    winners = []  # (strat_dict, metrics, mc, trades)
    tested = 0
    start = time.time()

    logger.info(f"Signals: {len(entry_sigs)} entries, {len(all_filters)} filters, {len(all_exits)} exits")

    # ── Round 1: Single entry ──
    logger.info("=== ROUND 1: Single entry signals ===")
    for sig in entry_sigs:
        for ex in random.sample(all_exits, min(35, len(all_exits))):
            for sz in random.sample(all_sizing, min(3, len(all_sizing))):
                entry_dict = {
                    "signal_name": sig.name, "module": sig.module, "function": sig.function,
                    "params": {k: v["default"] for k, v in sig.parameters.items()},
                    "columns": {
                        "long": next((c for c in sig.entry_columns if "long" in c), ""),
                        "short": next((c for c in sig.entry_columns if "short" in c), ""),
                    },
                }
                strat = GeneratedStrategy(
                    name=f"{sig.name.upper()}|SL{ex.stop_loss_value}_TP{ex.take_profit_value}",
                    entry_signals=[entry_dict], entry_filters=[],
                    exit_rules=ex, sizing_rules=sz,
                )
                out = backtest_obj(strat, data, rm, config)
                tested += 1
                if out:
                    trades, m = out
                    if m.total_pnl > 0 and m.profit_factor > 1.0:
                        mc = mc_test(trades, prop_rules, 2000)
                        if mc.median_return > 0 and mc.probability_of_profit > 0.6:
                            winners.append((strat.to_dict(), m, mc, trades))
                            if mc.median_return > 5000:
                                logger.info(f"  ★★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                            elif mc.probability_of_profit > 0.8:
                                logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
        elapsed = time.time() - start
        logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/max(1,elapsed):.0f}/s | {len(winners)}W]")

    # ── Round 2: Entry + filter (focus on combos that worked before) ──
    logger.info("\n=== ROUND 2: Entry + filter ===")
    for sig in entry_sigs:
        for filt in all_filters:
            for ex in random.sample(all_exits, min(15, len(all_exits))):
                sz = random.choice(all_sizing)
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
                    name=f"{sig.name.upper()}|{filt.name.upper()}|SL{ex.stop_loss_value}_TP{ex.take_profit_value}",
                    entry_signals=[entry_dict], entry_filters=[filt_dict],
                    exit_rules=ex, sizing_rules=sz,
                )
                out = backtest_obj(strat, data, rm, config)
                tested += 1
                if out:
                    trades, m = out
                    if m.total_pnl > 0 and m.profit_factor > 1.0:
                        mc = mc_test(trades, prop_rules, 2000)
                        if mc.median_return > 0 and mc.probability_of_profit > 0.6:
                            winners.append((strat.to_dict(), m, mc, trades))
                            if mc.median_return > 5000:
                                logger.info(f"  ★★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                            elif mc.probability_of_profit > 0.8:
                                logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
            if tested % 500 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/max(1,elapsed):.0f}/s | {len(winners)}W]")

    elapsed = time.time() - start
    logger.info(f"\nSearch done: {tested:,} tested in {elapsed:.0f}s, {len(winners)} winners")
    return winners, tested


# ══════════════════════════════════════════════════════════════════
#  EVOLUTION
# ══════════════════════════════════════════════════════════════════

def run_evolution(seed_strats, data, rm, config, prop_rules):
    GENERATIONS = 15
    POP_SIZE = 60
    ELITE_COUNT = 10
    MUTANT_COUNT = 30
    CROSSOVER_COUNT = 15
    RANDOM_COUNT = 5

    def get_intensity(gen):
        return max(0.05, 0.5 * (1.0 - gen / GENERATIONS))

    logger.info(f"\n=== Evaluating {len(seed_strats)} seed strategies ===")
    population = []
    for sd in seed_strats:
        out = backtest_dict(sd, data, rm, config)
        if out:
            trades, m = out
            mc = mc_test(trades, prop_rules)
            score = fitness(m, mc)
            population.append((sd, m, mc, score))

    population.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  {len(population)} viable from seed")

    best_ever = None
    best_ever_score = -float("inf")
    tested = len(population)
    start = time.time()

    if population and population[0][3] > best_ever_score:
        best_ever = population[0]
        best_ever_score = population[0][3]

    for gen in range(GENERATIONS):
        gen_start = time.time()
        intensity = get_intensity(gen)
        elites = [p[0] for p in population[:ELITE_COUNT]]

        mutants = [mutate_strategy(random.choice(elites), intensity) for _ in range(MUTANT_COUNT)]
        children = []
        for _ in range(CROSSOVER_COUNT):
            if len(elites) >= 2:
                a, b = random.sample(elites, 2)
                children.append(crossover(a, b))
        immigrants = []
        if len(population) > ELITE_COUNT:
            for _ in range(RANDOM_COUNT):
                immigrants.append(mutate_strategy(random.choice([p[0] for p in population[ELITE_COUNT:]]), intensity * 2))

        new_pop = []
        for sd in elites:
            out = backtest_dict(sd, data, rm, config)
            if out:
                trades, m = out
                mc = mc_test(trades, prop_rules)
                score = fitness(m, mc)
                new_pop.append((sd, m, mc, score))

        for sd in mutants + children + immigrants:
            out = backtest_dict(sd, data, rm, config)
            tested += 1
            if out:
                trades, m = out
                if m.total_pnl > 0 and m.profit_factor > 1.0:
                    mc = mc_test(trades, prop_rules)
                    if mc.probability_of_profit > 0.5:
                        score = fitness(m, mc)
                        new_pop.append((sd, m, mc, score))

        new_pop.sort(key=lambda x: x[3], reverse=True)
        population = new_pop[:POP_SIZE]

        if population and population[0][3] > best_ever_score:
            best_ever = population[0]
            best_ever_score = population[0][3]

        gen_elapsed = time.time() - gen_start
        if population:
            s, m, mc, score = population[0]
            logger.info(
                f"  Gen {gen+1:>2}/{GENERATIONS} | int={intensity:.2f} | pop={len(population)} | "
                f"BEST: PnL=${m.total_pnl:,.0f} | PF={m.profit_factor:.2f} | "
                f"MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%} | "
                f"Sharpe={m.sharpe_ratio:.2f} | {gen_elapsed:.0f}s"
            )

    return population, best_ever, tested


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    # ── Load Year 1: Mar 2024 – Mar 2025 ──
    logger.info("Loading Year 1 data (Mar 2024 – Mar 2025)...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    trading_days = df_yr1["timestamp"].dt.date().n_unique()
    data = {"1m": df_yr1}

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     FULL SEARCH + EVOLUTION — YEAR 1 (Mar 2024 – Mar 2025)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data:       {len(df_yr1):,} 1-minute bars ({trading_days} trading days)         ║
║  Period:     Mar 19, 2024 – Mar 18, 2025                           ║
║  Account:    Topstep $50K | Slippage: 2 ticks/side                 ║
║  Phase 1:    Brute-force search (all signals × exits × sizing)     ║
║  Phase 2:    15-generation genetic evolution                       ║
║  Goal:       MAXIMIZE profitability                                ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    logger.info(f"Year 1: {len(df_yr1):,} bars | {df_yr1['timestamp'][0]} -> {df_yr1['timestamp'][-1]}")

    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2024-03-19", end_date="2025-03-18",
        slippage_ticks=2, initial_capital=50000.0,
    )

    # ── PHASE 1: Search ──
    winners, search_tested = run_search(data, rm, config, prop_rules)

    if not winners:
        print("\n  No profitable strategies found.")
        return

    winners.sort(key=lambda w: w[2].median_return, reverse=True)
    seed_strats = [w[0] for w in winners[:20]]

    search_elapsed = time.time() - total_start
    print(f"\n{'='*90}")
    print(f"  SEARCH COMPLETE — {search_tested:,} tested in {search_elapsed/60:.1f} min, {len(winners)} winners")
    print(f"{'='*90}")

    print(f"\n  Top 15 search winners:")
    print(f"  {'#':<4} {'Strategy':<50} {'Tr':>5} {'PF':>6} {'PnL':>12} {'MC Med':>12} {'P(prof)':>8}")
    print(f"  {'-'*100}")
    for i, (sd, m, mc, _) in enumerate(winners[:15], 1):
        print(f"  {i:<4} {sd['name'][:50]:<50} {m.total_trades:>5} {m.profit_factor:>5.2f} ${m.total_pnl:>11,.0f} ${mc.median_return:>11,.0f} {mc.probability_of_profit:>7.1%}")

    # ── PHASE 2: Evolution ──
    population, best_ever, evo_tested = run_evolution(seed_strats, data, rm, config, prop_rules)

    total_elapsed = time.time() - total_start
    total_tested = search_tested + evo_tested

    print(f"\n{'='*120}")
    print(f"  YEAR 1 PIPELINE COMPLETE — {total_tested:,} strategies in {total_elapsed/60:.1f} min")
    print(f"{'='*120}")

    if population:
        print(f"\n  {'#':<4} {'Strategy':<50} {'Tr':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'PnL':>12} {'MC Med':>12} {'P(prof)':>8} {'Fitness':>10}")
        print(f"  {'-'*125}")
        for i, (s, m, mc, score) in enumerate(population[:25], 1):
            print(f"  {i:<4} {s['name'][:50]:<50} {m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f} {m.sharpe_ratio:>6.2f} ${m.total_pnl:>11,.0f} ${mc.median_return:>11,.0f} {mc.probability_of_profit:>7.1%} {score:>10,.0f}")
        print(f"  {'-'*125}")

    if best_ever:
        s, m, mc, score = best_ever
        print(f"""
  ══════════════════════════════════════════════════════════════
  CHAMPION — YEAR 1 (Mar 2024 – Mar 2025)
  ══════════════════════════════════════════════════════════════
  Name:           {s['name']}
  Entry:          {', '.join(e['signal_name'] for e in s['entry_signals'])}""")
        if s.get('entry_filters'):
            print(f"  Filter:         {', '.join(f['signal_name'] for f in s['entry_filters'])}")
        print(f"""  Stop Loss:      {s['exit_rules']['stop_loss_value']}pt ({s['exit_rules']['stop_loss_type']})
  Take Profit:    {s['exit_rules']['take_profit_value']}pt ({s['exit_rules']['take_profit_type']})
  Sizing:         {s['sizing_rules']}
  ──────────────────────────────────────────────────────────────
  Trades:         {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe:         {m.sharpe_ratio:.2f}
  Net P&L:        ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC P(ruin):     {mc.probability_of_ruin:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  MC Composite:   {mc.composite_score:.1f}/100
  Fitness:        {score:,.0f}""")
        print(f"\n  Parameters:")
        for sig in s["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in s.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

    # ── Save ──
    if population:
        output = []
        for s, m, mc, score in population:
            output.append({
                "name": s["name"], "strategy": s, "fitness": score,
                "trades": m.total_trades, "win_rate": m.win_rate,
                "profit_factor": m.profit_factor, "sharpe": m.sharpe_ratio,
                "total_pnl": m.total_pnl, "max_drawdown": m.max_drawdown,
                "mc_median": mc.median_return, "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin, "mc_composite": mc.composite_score,
                "mc_5th": mc.pct_5th_return, "mc_95th": mc.pct_95th_return,
            })
        with open("reports/evolved_strategies_yr1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/evolved_strategies_yr1.json")

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
