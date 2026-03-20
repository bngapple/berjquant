#!/usr/bin/env python3
"""
Genetic evolution of trading strategies via Monte Carlo.
Takes top winners and breeds/mutates them across generations,
getting progressively sharper until we find the best possible parameters.
"""

import json
import time
import random
import copy
import logging
import hashlib
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
from strategies.generator import GeneratedStrategy, ExitRules, SizingRules
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evolve")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")

# ── Mutation operators ──

def mutate_params(params: dict, param_specs: dict, intensity: float = 0.3) -> dict:
    """Mutate signal parameters. intensity=0.3 means ±30% variation."""
    new = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            delta = abs(v) * intensity * random.uniform(-1, 1)
            new_val = v + delta
            if isinstance(v, int):
                new_val = max(1, int(round(new_val)))
            else:
                new_val = round(max(0.01, new_val), 4)
            new[k] = new_val
        else:
            new[k] = v
    return new


def mutate_exit_rules(exit_rules: dict, intensity: float = 0.3) -> dict:
    """Mutate stop loss and take profit values."""
    new = copy.deepcopy(exit_rules)
    sl = new["stop_loss_value"]
    tp = new["take_profit_value"]

    sl_delta = sl * intensity * random.uniform(-1, 1)
    tp_delta = tp * intensity * random.uniform(-1, 1)

    new["stop_loss_value"] = round(max(1.0, sl + sl_delta), 1)
    new["take_profit_value"] = round(max(new["stop_loss_value"] * 1.2, tp + tp_delta), 1)
    return new


def mutate_sizing(sizing: dict, intensity: float = 0.3) -> dict:
    """Mutate position sizing."""
    new = copy.deepcopy(sizing)
    if new["method"] == "fixed":
        ct = new["fixed_contracts"]
        new["fixed_contracts"] = max(1, int(ct + ct * intensity * random.uniform(-1, 1)))
    elif new["method"] == "risk_pct":
        rp = new["risk_pct"]
        new["risk_pct"] = round(max(0.002, min(0.03, rp + rp * intensity * random.uniform(-1, 1))), 4)
    return new


def mutate_strategy(strat_dict: dict, intensity: float = 0.3) -> dict:
    """Full mutation of a strategy dict."""
    new = copy.deepcopy(strat_dict)

    # Mutate entry signal params
    for sig in new["entry_signals"]:
        sig["params"] = mutate_params(sig["params"], {}, intensity)

    # Mutate filter params
    for filt in new.get("entry_filters", []):
        filt["params"] = mutate_params(filt["params"], {}, intensity)

    # Mutate exits
    new["exit_rules"] = mutate_exit_rules(new["exit_rules"], intensity)

    # Mutate sizing
    new["sizing_rules"] = mutate_sizing(new["sizing_rules"], intensity)

    # Generate unique name
    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    base = new["name"].split("_")[0] if "_" in new["name"] else new["name"]
    # Keep the base strategy identity
    parts = new["name"].split("|")
    if len(parts) >= 2:
        base = "|".join(parts[:-1])
    else:
        base = parts[0]
    new["name"] = f"{base}|gen_{h}"

    return new


def crossover(parent_a: dict, parent_b: dict) -> dict:
    """Breed two strategies — take signals from A, exits from B (or vice versa)."""
    child = copy.deepcopy(parent_a)

    # 50% chance to swap exits
    if random.random() < 0.5:
        child["exit_rules"] = copy.deepcopy(parent_b["exit_rules"])

    # 50% chance to swap sizing
    if random.random() < 0.5:
        child["sizing_rules"] = copy.deepcopy(parent_b["sizing_rules"])

    # If both have filters, 50% chance to swap
    if parent_b.get("entry_filters") and random.random() < 0.5:
        child["entry_filters"] = copy.deepcopy(parent_b["entry_filters"])

    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = child["name"].split("|")
    base = "|".join(parts[:-1]) if len(parts) >= 2 else parts[0]
    child["name"] = f"{base}|breed_{h}"
    return child


def backtest_dict(strat_dict, data, rm, config):
    """Backtest a strategy from its dict representation."""
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


def mc_test(trades, prop_rules, n_sims=2000):
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=50000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def fitness(metrics, mc_result):
    """
    Fitness score combining multiple objectives.
    Prioritizes: high MC median, high P(profit), high PF, low drawdown.
    """
    mc_med = mc_result.median_return
    p_prof = mc_result.probability_of_profit
    pf = min(metrics.profit_factor, 10.0)
    dd = abs(metrics.max_drawdown)
    trades = metrics.total_trades

    # Penalize too few trades (unreliable)
    trade_factor = min(1.0, trades / 20.0)

    # Penalize excessive drawdown relative to prop firm limit
    dd_factor = max(0, 1.0 - (dd / 2000.0))

    score = (
        mc_med * 0.4 +
        p_prof * 10000 * 0.2 +
        pf * 1000 * 0.15 +
        metrics.total_pnl * 0.15 +
        dd_factor * 5000 * 0.1
    ) * trade_factor

    return score


def main():
    logger.info("Loading data...")
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

    # Load previous winners as seed population
    with open("reports/mc_winners_2yr.json") as f:
        all_winners = json.load(f)

    # Take top 20 as seed population
    seed_pop = [w["strategy"] for w in all_winners[:20]]
    logger.info(f"Seed population: {len(seed_pop)} strategies from previous search")

    # ── Evolution parameters ──
    GENERATIONS = 15
    POP_SIZE = 60          # Population per generation
    ELITE_COUNT = 10       # Best from each gen carry forward unchanged
    MUTANT_COUNT = 30      # Mutated offspring
    CROSSOVER_COUNT = 15   # Bred offspring
    RANDOM_COUNT = 5       # Fresh random immigrants

    # Intensity decreases over generations (wide search → fine tuning)
    def get_intensity(gen):
        return max(0.05, 0.5 * (1.0 - gen / GENERATIONS))

    # ── Initial population evaluation ──
    logger.info("\n=== Evaluating seed population ===")
    population = []  # (strat_dict, metrics, mc_result, fitness_score)

    for sd in seed_pop:
        out = backtest_dict(sd, data, rm, config)
        if out:
            trades, metrics = out
            mc = mc_test(trades, prop_rules)
            score = fitness(metrics, mc)
            population.append((sd, metrics, mc, score))

    population.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  {len(population)} viable from seed")

    best_ever = None
    best_ever_score = -float("inf")
    tested = len(population)
    start = time.time()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          GENETIC STRATEGY EVOLUTION — MC POWERED            ║
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

        # ── Selection: take top performers ──
        elites = [p[0] for p in population[:ELITE_COUNT]]

        # ── Mutation: mutate elites ──
        mutants = []
        for _ in range(MUTANT_COUNT):
            parent = random.choice(elites)
            child = mutate_strategy(parent, intensity)
            mutants.append(child)

        # ── Crossover: breed pairs of elites ──
        children = []
        for _ in range(CROSSOVER_COUNT):
            if len(elites) >= 2:
                a, b = random.sample(elites, 2)
                child = crossover(a, b)
                children.append(child)

        # ── Immigration: random mutations of non-elite survivors ──
        immigrants = []
        if len(population) > ELITE_COUNT:
            for _ in range(RANDOM_COUNT):
                parent = random.choice([p[0] for p in population[ELITE_COUNT:]])
                child = mutate_strategy(parent, intensity * 2)  # More aggressive
                immigrants.append(child)

        # ── Evaluate new generation ──
        candidates = mutants + children + immigrants
        new_pop = []

        # Elites carry forward (re-evaluated for consistency)
        for sd in elites:
            out = backtest_dict(sd, data, rm, config)
            if out:
                trades, metrics = out
                mc = mc_test(trades, prop_rules)
                score = fitness(metrics, mc)
                new_pop.append((sd, metrics, mc, score))

        # Evaluate new candidates
        for sd in candidates:
            out = backtest_dict(sd, data, rm, config)
            tested += 1
            if out:
                trades, metrics = out
                if metrics.total_pnl > 0 and metrics.profit_factor > 1.0:
                    mc = mc_test(trades, prop_rules)
                    if mc.probability_of_profit > 0.5:
                        score = fitness(metrics, mc)
                        new_pop.append((sd, metrics, mc, score))

        # Sort by fitness
        new_pop.sort(key=lambda x: x[3], reverse=True)
        population = new_pop[:POP_SIZE]

        # Track best ever
        if population and population[0][3] > best_ever_score:
            best_ever = population[0]
            best_ever_score = population[0][3]

        gen_elapsed = time.time() - gen_start
        top = population[0] if population else None

        if top:
            s, m, mc, score = top
            logger.info(
                f"  Gen {gen+1:>2}/{GENERATIONS} | intensity={intensity:.2f} | "
                f"pop={len(population)} | tested={tested} | "
                f"BEST: {s['name'][:40]} | PnL=${m.total_pnl:,.0f} | PF={m.profit_factor:.2f} | "
                f"MC=${mc.median_return:,.0f} | P={mc.probability_of_profit:.0%} | "
                f"fit={score:,.0f} | {gen_elapsed:.0f}s"
            )
        else:
            logger.info(f"  Gen {gen+1:>2}/{GENERATIONS} | No viable strategies")

    # ── Final report ──
    elapsed = time.time() - start
    print(f"\n{'='*90}")
    print(f"  EVOLUTION COMPLETE")
    print(f"{'='*90}")
    print(f"  Generations:      {GENERATIONS}")
    print(f"  Total tested:     {tested:,}")
    print(f"  Time:             {elapsed/60:.1f} min")
    print(f"  Final population: {len(population)}")
    print()

    if population:
        print(f"  {'#':<4} {'Strategy':<45} {'Trades':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'MC Med':>10} {'P(prof)':>8} {'Fitness':>10}")
        print(f"  {'-'*108}")
        for i, (s, m, mc, score) in enumerate(population[:25], 1):
            print(f"  {i:<4} {s['name'][:45]:<45} {m.total_trades:>6} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f} ${m.total_pnl:>9,.0f} ${mc.median_return:>9,.0f} {mc.probability_of_profit:>7.1%} {score:>10,.0f}")
        print(f"  {'-'*108}")

        # Best ever detailed breakdown
        if best_ever:
            s, m, mc, score = best_ever
            print(f"\n  ══════════════════════════════════════════════════")
            print(f"  BEST STRATEGY FOUND:")
            print(f"  ══════════════════════════════════════════════════")
            print(f"  Name:           {s['name']}")
            print(f"  Entry signals:  {', '.join(e['signal_name'] for e in s['entry_signals'])}")
            if s.get('entry_filters'):
                print(f"  Filters:        {', '.join(f['signal_name'] for f in s['entry_filters'])}")
            print(f"  Stop Loss:      {s['exit_rules']['stop_loss_value']}pt ({s['exit_rules']['stop_loss_type']})")
            print(f"  Take Profit:    {s['exit_rules']['take_profit_value']}pt ({s['exit_rules']['take_profit_type']})")
            print(f"  Sizing:         {s['sizing_rules']['method']} ({s['sizing_rules']})")
            print(f"  ──────────────────────────────────────────────────")
            print(f"  Trades:         {m.total_trades}")
            print(f"  Win Rate:       {m.win_rate:.1f}%")
            print(f"  Profit Factor:  {m.profit_factor:.2f}")
            print(f"  Sharpe Ratio:   {m.sharpe_ratio:.2f}")
            print(f"  Net P&L:        ${m.total_pnl:,.2f}")
            print(f"  Max Drawdown:   ${m.max_drawdown:,.2f}")
            print(f"  Avg Winner:     ${m.avg_winner:,.2f}")
            print(f"  Avg Loser:      ${m.avg_loser:,.2f}")
            print(f"  ──────────────────────────────────────────────────")
            print(f"  MC Median:      ${mc.median_return:,.2f}")
            print(f"  MC P(profit):   {mc.probability_of_profit:.1%}")
            print(f"  MC P(ruin):     {mc.probability_of_ruin:.1%}")
            print(f"  MC 5th pctl:    ${mc.pct_5th_return:,.2f}")
            print(f"  MC 95th pctl:   ${mc.pct_95th_return:,.2f}")
            print(f"  MC Composite:   {mc.composite_score:.1f}/100")
            print(f"  Fitness Score:  {score:,.0f}")

            # Entry signal params
            print(f"\n  Signal Parameters:")
            for sig in s["entry_signals"]:
                print(f"    {sig['signal_name']}: {sig['params']}")
            for filt in s.get("entry_filters", []):
                print(f"    [filter] {filt['signal_name']}: {filt['params']}")

        # Save
        output = []
        for s, m, mc, score in population:
            output.append({
                "name": s["name"], "strategy": s, "fitness": score,
                "trades": m.total_trades, "win_rate": m.win_rate,
                "profit_factor": m.profit_factor, "sharpe": m.sharpe_ratio,
                "total_pnl": m.total_pnl, "max_drawdown": m.max_drawdown,
                "mc_median": mc.median_return, "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin, "mc_composite": mc.composite_score,
            })
        with open("reports/evolved_strategies.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(output)} evolved strategies to reports/evolved_strategies.json")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
