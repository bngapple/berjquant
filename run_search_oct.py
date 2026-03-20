#!/usr/bin/env python3
"""
Full search + genetic evolution on October 2025 data.
Step 1: Brute-force search across all signal/exit/sizing combos
Step 2: Genetic evolution of winners through 15 generations
Goal: Find strategies that profit on Oct 2025 NQZ5 data.
"""

import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("oct_search")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
TARGET_PNL = 2000.0


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_oct_data():
    """Load Oct 2025 NQZ5 1m bars, session-filtered to 08:00-17:00 ET."""
    logger.info("Loading Databento NQZ5 Oct 2025 data...")
    df = pl.read_parquet("data/processed/MNQ/databento_nq_1m_raw.parquet")
    oct = df.filter(
        (pl.col("symbol") == "NQZ5") &
        (pl.col("ts_event") >= datetime(2025, 10, 1, tzinfo=timezone.utc)) &
        (pl.col("ts_event") < datetime(2025, 11, 1, tzinfo=timezone.utc)) &
        (pl.col("ts_event").dt.hour() >= 12) &  # 08:00 ET (EDT=UTC-4)
        (pl.col("ts_event").dt.hour() < 21)      # 17:00 ET
    )
    result = oct.select([
        pl.col("ts_event").dt.replace_time_zone(None).alias("timestamp"),
        pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"),
        pl.col("volume").cast(pl.Int64),
        (pl.col("volume") / 10).cast(pl.Int64).alias("tick_count"),
    ]).sort("timestamp")
    logger.info(f"  {len(result):,} bars | {result['timestamp'][0]} -> {result['timestamp'][-1]}")
    return {"1m": result}


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def make_exits():
    exits = []
    for sl in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
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


def backtest(strat_dict, data, rm, config):
    try:
        strat_dict["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(strat_dict)
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 5:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def backtest_obj(strategy, data, rm, config):
    try:
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 5:
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
    trade_factor = min(1.0, trades / 20.0)
    dd_factor = max(0, 1.0 - (dd / 2000.0))
    return (
        mc_med * 0.4 +
        p_prof * 10000 * 0.2 +
        pf * 1000 * 0.15 +
        metrics.total_pnl * 0.15 +
        dd_factor * 5000 * 0.1
    ) * trade_factor


# ══════════════════════════════════════════════════════════════════
#  MUTATION / CROSSOVER
# ══════════════════════════════════════════════════════════════════

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
#  PHASE 1: BROAD SEARCH
# ══════════════════════════════════════════════════════════════════

def run_search(data, rm, config, prop_rules):
    registry = SignalRegistry()
    generator = StrategyGenerator(registry)
    all_exits = make_exits()
    all_sizing = make_sizing()
    entry_sigs = registry.list_entry_signals()
    all_filters = registry.list_filters()

    winners = []
    tested = 0
    start = time.time()

    logger.info(f"Signals: {len(entry_sigs)} entries, {len(all_filters)} filters, {len(all_exits)} exits, {len(all_sizing)} sizing")

    # ── Round 1: Single entry ──
    logger.info("=== ROUND 1: Single entry signals ===")
    for sig in entry_sigs:
        for ex in random.sample(all_exits, min(40, len(all_exits))):
            for sz in random.sample(all_sizing, min(4, len(all_sizing))):
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
                            logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
        elapsed = time.time() - start
        logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/max(1,elapsed):.0f}/s | {len(winners)}W]")

    # ── Round 2: Entry + filter ──
    logger.info("\n=== ROUND 2: Entry + filter ===")
    for sig in entry_sigs:
        for filt in all_filters:
            for ex in random.sample(all_exits, min(20, len(all_exits))):
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
                            logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
            if tested % 500 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/max(1,elapsed):.0f}/s | {len(winners)}W]")

    # ── Round 3: Two entries (cross-category) ──
    logger.info("\n=== ROUND 3: Two entry signals ===")
    cats = {}
    for s in entry_sigs:
        cats.setdefault(s.category, []).append(s)
    cat_list = list(cats.keys())
    for i in range(len(cat_list)):
        for j in range(i + 1, len(cat_list)):
            for sa in cats[cat_list[i]]:
                for sb in cats[cat_list[j]]:
                    for ex in random.sample(all_exits, min(10, len(all_exits))):
                        sz = random.choice(all_sizing)
                        ea = {
                            "signal_name": sa.name, "module": sa.module, "function": sa.function,
                            "params": {k: v["default"] for k, v in sa.parameters.items()},
                            "columns": {
                                "long": next((c for c in sa.entry_columns if "long" in c), ""),
                                "short": next((c for c in sa.entry_columns if "short" in c), ""),
                            },
                        }
                        eb = {
                            "signal_name": sb.name, "module": sb.module, "function": sb.function,
                            "params": {k: v["default"] for k, v in sb.parameters.items()},
                            "columns": {
                                "long": next((c for c in sb.entry_columns if "long" in c), ""),
                                "short": next((c for c in sb.entry_columns if "short" in c), ""),
                            },
                        }
                        strat = GeneratedStrategy(
                            name=f"{sa.name.upper()}+{sb.name.upper()}|SL{ex.stop_loss_value}_TP{ex.take_profit_value}",
                            entry_signals=[ea, eb], entry_filters=[],
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
                                    logger.info(f"  ★ {strat.name[:55]} | {m.total_trades}t | PF={m.profit_factor:.2f} | PnL=${m.total_pnl:,.0f} | MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%}")
                    if tested % 500 == 0:
                        elapsed = time.time() - start
                        logger.info(f"  [{tested:,} | {elapsed:.0f}s | {tested/max(1,elapsed):.0f}/s | {len(winners)}W]")

    # ── Round 4: Mutate winners ──
    if winners:
        logger.info(f"\n=== ROUND 4: Mutating {len(winners)} winners ===")
        generator = StrategyGenerator(registry)
        for sd, m, mc, trades in list(winners):
            try:
                strat_obj = GeneratedStrategy.from_dict(sd)
                vars = generator.generate_parameter_variations(strat_obj, num_variations=40, method="random")
                for v in vars:
                    out = backtest_obj(v, data, rm, config)
                    tested += 1
                    if out:
                        t2, m2 = out
                        if m2.total_pnl > m.total_pnl * 0.8 and m2.profit_factor > 1.0:
                            mc2 = mc_test(t2, prop_rules, 2000)
                            if mc2.median_return > 0 and mc2.probability_of_profit > 0.6:
                                winners.append((v.to_dict(), m2, mc2, t2))
                                if mc2.median_return > mc.median_return:
                                    logger.info(f"  ★★ {v.name[:55]} | PnL=${m2.total_pnl:,.0f} | MC=${mc2.median_return:,.0f} | P={mc2.probability_of_profit:.0%}")
            except Exception:
                continue

    elapsed = time.time() - start
    logger.info(f"\nSearch complete: {tested:,} tested in {elapsed:.0f}s, {len(winners)} winners")
    return winners, tested


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: GENETIC EVOLUTION
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

    # Evaluate seed population
    logger.info(f"\n=== Evaluating {len(seed_strats)} seed strategies ===")
    population = []
    for sd in seed_strats:
        out = backtest(sd, data, rm, config)
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

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          GENETIC EVOLUTION — OCT 2025 NQZ5 DATA            ║
╠══════════════════════════════════════════════════════════════╣
║  Generations:  {GENERATIONS}                                         ║
║  Population:   {POP_SIZE} per generation                           ║
║  Operators:    mutation, crossover, elitism, immigration    ║
║  Fitness:      MC median + P(profit) + PF + drawdown        ║
╚══════════════════════════════════════════════════════════════╝
""")

    for gen in range(GENERATIONS):
        gen_start = time.time()
        intensity = get_intensity(gen)

        elites = [p[0] for p in population[:ELITE_COUNT]]

        # Mutation
        mutants = [mutate_strategy(random.choice(elites), intensity) for _ in range(MUTANT_COUNT)]

        # Crossover
        children = []
        for _ in range(CROSSOVER_COUNT):
            if len(elites) >= 2:
                a, b = random.sample(elites, 2)
                children.append(crossover(a, b))

        # Immigration
        immigrants = []
        if len(population) > ELITE_COUNT:
            for _ in range(RANDOM_COUNT):
                parent = random.choice([p[0] for p in population[ELITE_COUNT:]])
                immigrants.append(mutate_strategy(parent, intensity * 2))

        # Evaluate
        new_pop = []
        for sd in elites:
            out = backtest(sd, data, rm, config)
            if out:
                trades, m = out
                mc = mc_test(trades, prop_rules)
                score = fitness(m, mc)
                new_pop.append((sd, m, mc, score))

        for sd in mutants + children + immigrants:
            out = backtest(sd, data, rm, config)
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
                f"  Gen {gen+1:>2}/{GENERATIONS} | int={intensity:.2f} | "
                f"pop={len(population)} | tested={tested} | "
                f"BEST: {s['name'][:40]} | PnL=${m.total_pnl:,.0f} | PF={m.profit_factor:.2f} | "
                f"MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%} | "
                f"Sharpe={m.sharpe_ratio:.2f} | fit={score:,.0f} | {gen_elapsed:.0f}s"
            )
        else:
            logger.info(f"  Gen {gen+1:>2}/{GENERATIONS} | No viable strategies")

    return population, best_ever, tested


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    data = load_oct_data()
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2025-10-01", end_date="2025-10-31",
        slippage_ticks=2, initial_capital=50000.0,
    )

    bars = len(data["1m"])
    days = data["1m"]["timestamp"].dt.date().n_unique()
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║       FULL SEARCH + EVOLUTION — OCTOBER 2025 NQZ5                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data:       {bars:,} 1-minute bars ({days} trading days)                ║
║  Period:     Oct 1 – Oct 31, 2025                                  ║
║  Account:    Topstep $50K | Slippage: 2 ticks/side                 ║
║  Phase 1:    Brute-force search (signals × exits × sizing)         ║
║  Phase 2:    Genetic evolution (15 generations)                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # ── PHASE 1: Search ──
    winners, search_tested = run_search(data, rm, config, prop_rules)

    if not winners:
        print("\n  No profitable strategies found in search phase.")
        print("  Cannot proceed to evolution.")
        return

    # Deduplicate and take top 20 as seeds
    winners.sort(key=lambda w: w[2].median_return, reverse=True)
    seed_strats = [w[0] for w in winners[:20]]

    search_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SEARCH COMPLETE — {search_tested:,} tested in {search_elapsed/60:.1f} min")
    print(f"  Winners: {len(winners)} | Seeding top {len(seed_strats)} for evolution")
    print(f"{'='*70}")

    print(f"\n  Top 10 search winners:")
    print(f"  {'#':<4} {'Strategy':<50} {'Tr':>4} {'PF':>6} {'PnL':>10} {'MC Med':>10} {'P(prof)':>8}")
    print(f"  {'-'*96}")
    for i, (sd, m, mc, _) in enumerate(winners[:10], 1):
        print(f"  {i:<4} {sd['name'][:50]:<50} {m.total_trades:>4} {m.profit_factor:>5.2f} ${m.total_pnl:>9,.0f} ${mc.median_return:>9,.0f} {mc.probability_of_profit:>7.1%}")

    # ── PHASE 2: Evolution ──
    population, best_ever, evo_tested = run_evolution(seed_strats, data, rm, config, prop_rules)

    total_elapsed = time.time() - total_start
    total_tested = search_tested + evo_tested

    print(f"\n{'='*90}")
    print(f"  FULL PIPELINE COMPLETE — OCT 2025")
    print(f"{'='*90}")
    print(f"  Total tested:     {total_tested:,}")
    print(f"  Total time:       {total_elapsed/60:.1f} min")
    print(f"  Final population: {len(population)}")
    print()

    if population:
        print(f"  {'#':<4} {'Strategy':<45} {'Tr':>5} {'WR':>6} {'PF':>6} {'PnL':>10} {'Sharpe':>7} {'MC Med':>10} {'P(prof)':>8} {'Fitness':>10}")
        print(f"  {'-'*115}")
        for i, (s, m, mc, score) in enumerate(population[:25], 1):
            print(f"  {i:<4} {s['name'][:45]:<45} {m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f} ${m.total_pnl:>9,.0f} {m.sharpe_ratio:>6.2f} ${mc.median_return:>9,.0f} {mc.probability_of_profit:>7.1%} {score:>10,.0f}")
        print(f"  {'-'*115}")

    if best_ever:
        s, m, mc, score = best_ever
        print(f"""
  ══════════════════════════════════════════════════════════
  CHAMPION STRATEGY:
  ══════════════════════════════════════════════════════════
  Name:           {s['name']}
  Entry signals:  {', '.join(e['signal_name'] for e in s['entry_signals'])}""")
        if s.get('entry_filters'):
            print(f"  Filters:        {', '.join(f['signal_name'] for f in s['entry_filters'])}")
        print(f"""  Stop Loss:      {s['exit_rules']['stop_loss_value']}pt ({s['exit_rules']['stop_loss_type']})
  Take Profit:    {s['exit_rules']['take_profit_value']}pt ({s['exit_rules']['take_profit_type']})
  Sizing:         {s['sizing_rules']}
  ──────────────────────────────────────────────────────────
  Trades:         {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe Ratio:   {m.sharpe_ratio:.2f}
  Net P&L:        ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}
  Avg Winner:     ${m.avg_winner:,.2f}
  Avg Loser:      ${m.avg_loser:,.2f}
  ──────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC P(ruin):     {mc.probability_of_ruin:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  MC Composite:   {mc.composite_score:.1f}/100
  Fitness Score:  {score:,.0f}""")

        print(f"\n  Signal Parameters:")
        for sig in s["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in s.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

    # Save results
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
        with open("reports/evolved_strategies_oct2025.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/evolved_strategies_oct2025.json")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
