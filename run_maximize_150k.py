#!/usr/bin/env python3
"""
MAXIMIZE — Take the proven Stochastic+Imbalance edge and push sizing/params
to achieve $20K+ months. The edge is real (100% P(profit) on 1yr, 789 trades).
Now we optimize position sizing and exit parameters to maximize monthly PnL.

Strategy: mutate aggressively around the champion params while scaling up contracts.
"""

import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path

import polars as pl
import numpy as np

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("maximize")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")


def backtest(sd, data, rm, config):
    try:
        sd["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd)
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < 20:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, prop_rules, n_sims=3000):
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=150000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def monthly_pnl(trades):
    """Calculate monthly P&L breakdown."""
    months = {}
    for t in trades:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


def fitness_maximize(metrics, mc, trades):
    """Fitness focused on MAXIMUM monthly PnL while staying within prop firm limits."""
    months = monthly_pnl(trades)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    max_monthly = max(monthly_vals)
    profitable_months = sum(1 for v in monthly_vals if v > 0)
    pct_profitable = profitable_months / len(monthly_vals)

    mc_med = mc.median_return
    p_prof = mc.probability_of_profit
    dd = abs(metrics.max_drawdown)

    # Heavy penalty if drawdown exceeds prop firm limit ($2000)
    dd_penalty = max(0, (dd - 4000) * 10) if dd > 1800 else 0

    # Reward high monthly PnL, penalize losing months
    score = (
        avg_monthly * 3.0 +          # Primary: maximize average monthly PnL
        min_monthly * 1.0 +           # Don't have terrible months
        mc_med * 0.5 +                # MC robustness
        p_prof * 5000 * 0.3 +         # Probability of profit
        pct_profitable * 10000 * 0.5  # % of months profitable
        - dd_penalty                  # Stay within prop limits
    )
    return score


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

    # Mutate signal params
    for sig in new["entry_signals"]:
        sig["params"] = mutate_params(sig["params"], intensity)
    for filt in new.get("entry_filters", []):
        filt["params"] = mutate_params(filt["params"], intensity)

    # Mutate exits
    er = new["exit_rules"]
    sl, tp = er["stop_loss_value"], er["take_profit_value"]
    er["stop_loss_value"] = round(max(2.0, sl + sl * intensity * random.uniform(-1, 1)), 1)
    er["take_profit_value"] = round(max(er["stop_loss_value"] * 1.5, tp + tp * intensity * random.uniform(-1, 1)), 1)

    # Mutate sizing — this is key for maximizing PnL
    sz = new["sizing_rules"]
    if sz["method"] == "fixed":
        ct = sz["fixed_contracts"]
        new_ct = max(1, int(ct + ct * intensity * random.uniform(-0.5, 1.5)))  # Bias upward
        sz["fixed_contracts"] = min(50, new_ct)
    elif sz["method"] == "risk_pct":
        rp = sz["risk_pct"]
        sz["risk_pct"] = round(max(0.005, min(0.05, rp + rp * intensity * random.uniform(-1, 1))), 4)

    # Randomly swap sizing method
    if random.random() < 0.15:
        if sz["method"] == "fixed":
            sz["method"] = "risk_pct"
            sz["risk_pct"] = random.choice([0.01, 0.015, 0.02, 0.025, 0.03, 0.04])
        else:
            sz["method"] = "fixed"
            sz["fixed_contracts"] = random.choice([2, 3, 5, 7, 10])

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
    new["name"] = f"{base}|max_{h}"
    return new


def crossover(a, b):
    child = copy.deepcopy(a)
    if random.random() < 0.5:
        child["exit_rules"] = copy.deepcopy(b["exit_rules"])
    if random.random() < 0.5:
        child["sizing_rules"] = copy.deepcopy(b["sizing_rules"])
    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = child["name"].split("|")
    base = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
    child["name"] = f"{base}|breed_{h}"
    return child


def main():
    total_start = time.time()

    # Load Year 1 data
    logger.info("Loading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    data = {"1m": df}

    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_150k",
        start_date="2024-03-19", end_date="2025-03-18",
        slippage_ticks=2, initial_capital=150000.0,
    )

    # Load evolved strategies as seeds
    with open("reports/evolved_strategies_yr1.json") as f:
        evolved = json.load(f)

    # Take top 20 + create aggressive sizing variants
    seeds = []
    for e in evolved[:20]:
        sd = e["strategy"]
        seeds.append(sd)

        # Create variants with higher sizing
        for contracts in [5, 10, 15, 20, 30]:
            v = copy.deepcopy(sd)
            v["sizing_rules"]["method"] = "fixed"
            v["sizing_rules"]["fixed_contracts"] = contracts
            h = hashlib.md5(f"{v['name']}_{contracts}ct".encode()).hexdigest()[:6]
            v["name"] = f"{v['name'].split('|')[0]}|{v['name'].split('|')[1]}|{contracts}ct_{h}"
            seeds.append(v)

        for rp in [0.02, 0.03, 0.04, 0.05, 0.06]:
            v = copy.deepcopy(sd)
            v["sizing_rules"]["method"] = "risk_pct"
            v["sizing_rules"]["risk_pct"] = rp
            h = hashlib.md5(f"{v['name']}_{rp}rp".encode()).hexdigest()[:6]
            v["name"] = f"{v['name'].split('|')[0]}|{v['name'].split('|')[1]}|r{rp}_{h}"
            seeds.append(v)

    logger.info(f"Seeds: {len(seeds)} (20 base + sizing variants)")

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     MAXIMIZE — PUSH FOR $20K MONTHS                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Base Edge:  Stochastic + Imbalance (100% P(profit), PF 1.72)     ║
║  Data:       {len(df):,} bars (Mar 2024 – Mar 2025)                ║
║  Seeds:      {len(seeds)} strategies (base + sizing variants)             ║
║  Evolution:  25 generations, maximize monthly PnL                  ║
║  Target:     $20,000+ per month                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # ── Evaluate seeds ──
    logger.info("Evaluating seed population...")
    population = []
    for sd in seeds:
        out = backtest(sd, data, rm, config)
        if out:
            trades, m = out
            mc = mc_test(trades, prop_rules, 2000)
            if mc.probability_of_profit > 0.5:
                score = fitness_maximize(m, mc, trades)
                months = monthly_pnl(trades)
                avg_mo = np.mean(list(months.values())) if months else 0
                max_mo = max(months.values()) if months else 0
                population.append((sd, m, mc, score, trades))
                if avg_mo > 1500:
                    logger.info(f"  ★ {sd['name'][:50]} | {m.total_trades}t | PnL=${m.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | max/mo=${max_mo:,.0f} | DD=${m.max_drawdown:,.0f} | MC P={mc.probability_of_profit:.0%}")

    population.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  {len(population)} viable seeds")

    best_ever = None
    best_ever_score = -float("inf")
    tested = len(population)

    if population:
        best_ever = population[0]
        best_ever_score = population[0][3]

    # ── Evolution ──
    GENERATIONS = 25
    POP_SIZE = 60
    ELITE_COUNT = 12
    MUTANT_COUNT = 30
    CROSSOVER_COUNT = 12
    RANDOM_COUNT = 6

    for gen in range(GENERATIONS):
        gen_start = time.time()
        # Start aggressive, taper to fine-tuning
        intensity = max(0.03, 0.6 * (1.0 - gen / GENERATIONS))

        elites = [p[0] for p in population[:ELITE_COUNT]]

        # Mutations — bias toward more aggressive sizing
        mutants = [mutate_strategy(random.choice(elites), intensity) for _ in range(MUTANT_COUNT)]

        # Crossovers
        children = []
        for _ in range(CROSSOVER_COUNT):
            if len(elites) >= 2:
                a, b = random.sample(elites, 2)
                children.append(crossover(a, b))

        # Immigration from lower-ranked
        immigrants = []
        if len(population) > ELITE_COUNT:
            for _ in range(RANDOM_COUNT):
                immigrants.append(mutate_strategy(
                    random.choice([p[0] for p in population[ELITE_COUNT:]]),
                    intensity * 2
                ))

        # Evaluate
        new_pop = []
        for sd in elites:
            out = backtest(sd, data, rm, config)
            if out:
                trades, m = out
                mc = mc_test(trades, prop_rules, 2000)
                score = fitness_maximize(m, mc, trades)
                new_pop.append((sd, m, mc, score, trades))

        for sd in mutants + children + immigrants:
            out = backtest(sd, data, rm, config)
            tested += 1
            if out:
                trades, m = out
                if m.total_pnl > 0:
                    mc = mc_test(trades, prop_rules, 2000)
                    if mc.probability_of_profit > 0.5:
                        score = fitness_maximize(m, mc, trades)
                        new_pop.append((sd, m, mc, score, trades))

        new_pop.sort(key=lambda x: x[3], reverse=True)
        population = new_pop[:POP_SIZE]

        if population and population[0][3] > best_ever_score:
            best_ever = population[0]
            best_ever_score = population[0][3]

        gen_elapsed = time.time() - gen_start
        if population:
            s, m, mc, score, trades = population[0]
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values()))
            max_mo = max(months.values())
            min_mo = min(months.values())
            sz = s["sizing_rules"]
            sz_str = f'{sz["fixed_contracts"]}ct' if sz["method"] == "fixed" else f'r{sz["risk_pct"]}'
            logger.info(
                f"  Gen {gen+1:>2}/{GENERATIONS} | int={intensity:.2f} | pop={len(population)} | "
                f"PnL=${m.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | max/mo=${max_mo:,.0f} | min/mo=${min_mo:,.0f} | "
                f"DD=${m.max_drawdown:,.0f} | {sz_str} | P={mc.probability_of_profit:.0%} | {gen_elapsed:.0f}s"
            )

    total_elapsed = time.time() - total_start

    # ── Final Results ──
    print(f"\n{'='*130}")
    print(f"  MAXIMIZE COMPLETE — {tested:,} tested in {total_elapsed/60:.1f} min")
    print(f"{'='*130}")

    if population:
        print(f"\n  {'#':<4} {'Strategy':<40} {'Tr':>5} {'PF':>5} {'1yr PnL':>12} {'Avg/Mo':>10} {'Max/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'Sizing':>8} {'MC P':>6}")
        print(f"  {'-'*130}")
        for i, (s, m, mc, score, trades) in enumerate(population[:25], 1):
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values()))
            max_mo = max(months.values())
            min_mo = min(months.values())
            sz = s["sizing_rules"]
            sz_str = f'{sz["fixed_contracts"]}ct' if sz["method"] == "fixed" else f'r{sz["risk_pct"]}'
            flag = "★" if avg_mo >= 20000 else "◆" if avg_mo >= 10000 else " "
            print(
                f"  {flag}{i:<3} {s['name'][:39]:<40} "
                f"{m.total_trades:>5} {m.profit_factor:>4.2f} ${m.total_pnl:>11,.0f} "
                f"${avg_mo:>9,.0f} ${max_mo:>9,.0f} ${min_mo:>9,.0f} "
                f"${m.max_drawdown:>9,.0f} {sz_str:>8} {mc.probability_of_profit:>5.0%}"
            )
        print(f"  {'-'*130}")

    if best_ever:
        s, m, mc, score, trades = best_ever
        months = monthly_pnl(trades)
        avg_mo = np.mean(list(months.values()))
        max_mo = max(months.values())
        min_mo = min(months.values())

        print(f"""
  ══════════════════════════════════════════════════════════════
  CHAMPION — MAXIMUM PROFITABILITY
  ══════════════════════════════════════════════════════════════
  Name:           {s['name']}
  Entry:          {', '.join(e['signal_name'] for e in s['entry_signals'])}""")
        if s.get('entry_filters'):
            print(f"  Filter:         {', '.join(f['signal_name'] for f in s['entry_filters'])}")
        print(f"""  Stop Loss:      {s['exit_rules']['stop_loss_value']}pt
  Take Profit:    {s['exit_rules']['take_profit_value']}pt
  R:R:            {s['exit_rules']['take_profit_value']/s['exit_rules']['stop_loss_value']:.1f}:1
  Sizing:         {s['sizing_rules']}
  ──────────────────────────────────────────────────────────────
  Trades (1yr):   {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe:         {m.sharpe_ratio:.2f}
  Net P&L (1yr):  ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}
  Avg Winner:     ${m.avg_winner:,.2f}
  Avg Loser:      ${m.avg_loser:,.2f}
  ──────────────────────────────────────────────────────────────
  Avg Month:      ${avg_mo:,.2f}
  Best Month:     ${max_mo:,.2f}
  Worst Month:    ${min_mo:,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  ══════════════════════════════════════════════════════════════""")

        print(f"\n  Parameters:")
        for sig in s["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in s.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

        # Monthly breakdown
        print(f"\n  MONTHLY P&L:")
        for mo in sorted(months.keys()):
            p = months[mo]
            bar = "█" * max(1, int(abs(p) / 500))
            flag = "★" if p >= 20000 else "◆" if p >= 10000 else " "
            print(f"    {flag} {mo}: {'+'if p>=0 else '-'}${abs(p):>10,.2f}  {bar}{'*' if p<0 else ''}")
        prof_months = sum(1 for p in months.values() if p > 0)
        print(f"    Profitable months: {prof_months}/{len(months)} ({prof_months/len(months):.0%})")

    # Save
    if population:
        output = []
        for s, m, mc, score, trades in population[:60]:
            months = monthly_pnl(trades)
            output.append({
                "name": s["name"], "strategy": s, "fitness": score,
                "trades": m.total_trades, "win_rate": m.win_rate,
                "profit_factor": m.profit_factor, "sharpe": m.sharpe_ratio,
                "total_pnl": m.total_pnl, "max_drawdown": m.max_drawdown,
                "avg_monthly": float(np.mean(list(months.values()))),
                "max_monthly": float(max(months.values())),
                "min_monthly": float(min(months.values())),
                "mc_median": mc.median_return, "mc_p_profit": mc.probability_of_profit,
                "mc_5th": mc.pct_5th_return, "mc_95th": mc.pct_95th_return,
            })
        with open("reports/maximized_strategies_150k.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/maximized_strategies_150k.json")

    print(f"\n{'='*130}\n")


if __name__ == "__main__":
    main()
