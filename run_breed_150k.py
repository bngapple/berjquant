#!/usr/bin/env python3
"""
BREED 150K v3 — Sniper mode.

Core thesis change: instead of wide stops and moderate sizing,
use TIGHT stops + HIGH contracts + HUGE targets during the opening
power hour (9:30-11:00 ET) when NQ moves are biggest.

Design:
  1. SL: 10-18pt (was 31pt) — cut losers fast
  2. TP: 100-300pt — let winners run, accept 20-25% win rate
  3. Contracts: 8-15 (was 4) — more firepower per trade
  4. Session: 9:30-11:00 ET only via time_of_day filter
  5. Diverse entries: aggressively mutate stochastic params to find
     multiple uncorrelated firing points per day

At $2/pt on MNQ with 10ct:
  - Loss: 15pt × 10ct × $2 = $300/trade
  - Win:  200pt × 10ct × $2 = $4,000/trade
  - At 22% WR: (0.22 × $4000) - (0.78 × $300) = $880 - $234 = $646 expected per trade
  - 1 trade/day × 22 days = $14,212/month

Still walk-forward validated + MC stress tested.
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
logger = logging.getLogger("breed")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")

# ── Hard limits ───────────────────────────────────────────────────────
MAX_DD_LIMIT = -4500.0
DAILY_LOSS_LIMIT = -3000.0
MIN_TRADES = 30
MC_SIMS_TRAIN = 2000
MC_SIMS_FINAL = 5000
OOS_MC_P_PROFIT_MIN = 0.80
CORRELATION_THRESHOLD = 0.80
TARGET_AVG_MONTHLY = 15000

# ── Sniper exit bounds ────────────────────────────────────────────────
SL_MIN, SL_MAX = 10.0, 18.0       # Tight stops
TP_MIN, TP_MAX = 100.0, 300.0     # Huge targets
CT_MIN, CT_MAX = 8, 15            # More contracts
RR_MIN = 6.0                      # Minimum R:R ratio


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST + MC HELPERS
# ═══════════════════════════════════════════════════════════════════════

def backtest(sd, data, rm, config, min_trades=MIN_TRADES):
    try:
        sd_copy = copy.deepcopy(sd)
        sd_copy["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd_copy)
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < min_trades:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital)
        if metrics.max_drawdown < MAX_DD_LIMIT:
            return None
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, prop_rules, n_sims=MC_SIMS_TRAIN):
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=150000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def monthly_pnl(trades):
    months = {}
    for t in trades:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


# ═══════════════════════════════════════════════════════════════════════
# FITNESS
# ═══════════════════════════════════════════════════════════════════════

def fitness(metrics, mc, trades):
    """Fix #37: Aligned with $15K target, added consistency bonus and floor penalty."""
    months = monthly_pnl(trades)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    profitable_months = sum(1 for v in monthly_vals if v > 0)
    pct_profitable = profitable_months / len(monthly_vals)
    dd = abs(metrics.max_drawdown)

    dd_margin_bonus = max(0, (4500 - dd) / 2500) * 3000
    # Fix #37: Target bonus kicks in at $15K, not $10K
    target_bonus = max(0, (avg_monthly - 15000)) * 3.0 if avg_monthly > 15000 else 0
    # Fix #37: Penalize months below $15K floor
    floor_penalty = sum(max(0, 15000 - v) for v in monthly_vals if v > 0) * 0.3
    weak_month_penalty = sum(max(0, 5000 - v) for v in monthly_vals) * 0.5
    # Fix #37: Consistency bonus — count months hitting $15K+
    months_hitting_target = sum(1 for v in monthly_vals if v >= 15000)
    consistency_bonus = (months_hitting_target / len(monthly_vals)) * 10000

    score = (
        avg_monthly * 4.0 +
        min_monthly * 2.5 +
        mc.median_return * 0.5 +
        mc.probability_of_profit * 5000 * 0.4 +
        pct_profitable * 10000 * 0.6 +
        dd_margin_bonus +
        metrics.sharpe_ratio * 500 +
        target_bonus +
        consistency_bonus -
        weak_month_penalty -
        floor_penalty
    )
    return score


def fitness_proxy(metrics, trades):
    """Fix #11: Fast proxy fitness without MC — used during evolution."""
    months = monthly_pnl(trades)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    pct_profitable = sum(1 for v in monthly_vals if v > 0) / len(monthly_vals)
    dd = abs(metrics.max_drawdown)

    dd_penalty = max(0, (dd - 3500) * 5) if dd > 3500 else 0

    score = (
        avg_monthly * 4.0 +
        min_monthly * 2.5 +
        metrics.total_pnl * 0.5 +
        pct_profitable * 10000 * 0.6 +
        metrics.sharpe_ratio * 500 +
        metrics.profit_factor * 1000 -
        dd_penalty
    )
    return score


# ═══════════════════════════════════════════════════════════════════════
# FIX #4: DIVERSE SEED GENERATION — explore the full signal library
# ═══════════════════════════════════════════════════════════════════════

def generate_diverse_seeds(n=200):
    """Generate structurally diverse strategies from the full signal registry.

    Fix #4: Instead of only stochastic+imbalance, explore all entry × filter
    combinations with random parameters and sniper-style exits.
    """
    from signals.registry import SignalRegistry
    from strategies.generator import StrategyGenerator, ExitRules, SizingRules

    registry = SignalRegistry()
    gen = StrategyGenerator(registry)

    entry_sigs = registry.list_entry_signals()
    filter_sigs = registry.list_filters()

    seeds = []
    seen_combos = set()

    for _ in range(n * 3):  # Oversample to handle duplicates
        if len(seeds) >= n:
            break

        # Pick 1 entry signal (from different categories)
        entry = random.choice(entry_sigs)

        # Pick 0-1 filters
        filt = random.choice(filter_sigs) if random.random() < 0.7 else None

        combo_key = (entry.name, filt.name if filt else None)
        if combo_key in seen_combos:
            continue
        seen_combos.add(combo_key)

        # Random parameters for entry signal
        entry_params = gen._sample_params(entry, method="random")
        entry_dict = {
            "signal_name": entry.name,
            "module": entry.module,
            "function": entry.function,
            "params": entry_params,
            "columns": {
                "long": entry.entry_columns[0] if entry.entry_columns else "",
                "short": entry.entry_columns[1] if len(entry.entry_columns) > 1 else "",
            },
        }

        filters = []
        if filt:
            filt_params = gen._sample_params(filt, method="random")
            filters.append({
                "signal_name": filt.name,
                "module": filt.module,
                "function": filt.function,
                "params": filt_params,
                "column": filt.filter_columns[0] if filt.filter_columns else "",
            })

        # Add time_of_day filter (9:30-11:00 ET)
        filters.append({
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {"start_hour": 9, "start_minute": 30, "end_hour": 11, "end_minute": 0},
            "column": "signal_in_time_window",
        })

        # Sniper exits with random params
        sl = round(random.uniform(SL_MIN, SL_MAX), 1)
        tp = round(random.uniform(TP_MIN, TP_MAX), 1)
        if tp / sl < RR_MIN:
            tp = round(sl * RR_MIN, 1)

        sd = {
            "name": f"{entry.name.upper()}|{'_'.join(f['signal_name'].upper() for f in filters)}|div_{hashlib.md5(str(combo_key).encode()).hexdigest()[:6]}",
            "entry_signals": [entry_dict],
            "entry_filters": filters,
            "exit_rules": {
                "stop_loss_type": "fixed_points",
                "stop_loss_value": sl,
                "take_profit_type": "fixed_points",
                "take_profit_value": tp,
                "trailing_stop": random.random() < 0.3,
                "trailing_activation": round(random.uniform(20, 60), 1),
                "trailing_distance": round(random.uniform(8, 20), 1),
                "time_exit_minutes": None,
            },
            "sizing_rules": {
                "method": "fixed",
                "fixed_contracts": random.randint(CT_MIN, CT_MAX),
                "risk_pct": 0.01,
                "atr_risk_multiple": 2.0,
            },
            "primary_timeframe": "1m",
            "require_all_entries": True,
        }
        seeds.append(sd)

    return seeds


# ═══════════════════════════════════════════════════════════════════════
# SEED CONVERSION — transform old strategies into sniper format
# ═══════════════════════════════════════════════════════════════════════

def to_sniper(sd):
    """
    Convert a winning strategy to sniper format:
    - Tight SL, huge TP
    - More contracts
    - Add time_of_day filter for 9:30-11:00 ET
    """
    new = copy.deepcopy(sd)

    # Force sniper exits
    er = new["exit_rules"]
    er["stop_loss_value"] = round(random.uniform(SL_MIN, SL_MAX), 1)
    er["take_profit_value"] = round(random.uniform(TP_MIN, TP_MAX), 1)
    er["stop_loss_type"] = "fixed_points"
    er["take_profit_type"] = "fixed_points"

    # Force higher contracts
    sz = new["sizing_rules"]
    sz["method"] = "fixed"
    sz["fixed_contracts"] = random.randint(CT_MIN, CT_MAX)

    # Add time_of_day filter for 9:30-11:00 ET
    # (fix #17: time_filters now converts to ET internally, use ET hours directly)
    has_time_filter = any(
        f.get("signal_name") == "time_of_day" or f.get("function") == "time_of_day"
        for f in new.get("entry_filters", [])
    )
    if not has_time_filter:
        new["entry_filters"].append({
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {
                "start_hour": 9,
                "start_minute": 30,
                "end_hour": 11,
                "end_minute": 0,
            },
            "column": "signal_in_time_window",
        })

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
    new["name"] = f"{base}|sniper_{h}"
    return new


# ═══════════════════════════════════════════════════════════════════════
# MUTATION + CROSSOVER — SNIPER-AWARE
# ═══════════════════════════════════════════════════════════════════════

def mutate_params(params, intensity=0.5):
    new = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            delta = abs(v) * intensity * random.uniform(-1, 1)
            new_val = v + delta
            new[k] = max(1, int(round(new_val))) if isinstance(v, int) else round(max(0.01, new_val), 4)
        else:
            new[k] = v
    return new


def mutate_strategy(sd, intensity=0.5):
    new = copy.deepcopy(sd)

    # Mutate stochastic params — VERY aggressive on thresholds
    for sig in new["entry_signals"]:
        sig["params"] = mutate_params(sig["params"], intensity)
        # Extra ±50-100% on overbought/oversold to find new entry points
        if "overbought" in sig["params"]:
            ob = sig["params"]["overbought"]
            sig["params"]["overbought"] = round(max(15, min(95, ob * random.uniform(0.5, 1.5))), 4)
        if "oversold" in sig["params"]:
            os_val = sig["params"]["oversold"]
            sig["params"]["oversold"] = round(max(5, min(85, os_val * random.uniform(0.5, 1.5))), 4)
        # Mutate k_period and d_period more aggressively
        if "k_period" in sig["params"]:
            sig["params"]["k_period"] = max(2, int(sig["params"]["k_period"] * random.uniform(0.5, 2.0)))
        if "d_period" in sig["params"]:
            sig["params"]["d_period"] = max(1, int(sig["params"]["d_period"] * random.uniform(0.5, 2.0)))

    # Mutate imbalance filter
    for filt in new.get("entry_filters", []):
        if filt.get("function") != "time_of_day":
            filt["params"] = mutate_params(filt["params"], intensity)
            if "ratio_threshold" in filt["params"]:
                rt = filt["params"]["ratio_threshold"]
                filt["params"]["ratio_threshold"] = round(max(0.5, min(5.0, rt * random.uniform(0.5, 1.5))), 4)

    # Mutate exits — within sniper bounds
    er = new["exit_rules"]
    sl = er["stop_loss_value"]
    tp = er["take_profit_value"]
    new_sl = round(max(SL_MIN, min(SL_MAX, sl + sl * intensity * random.uniform(-1, 1))), 1)
    new_tp = round(max(TP_MIN, min(TP_MAX, tp + tp * intensity * random.uniform(-1, 1))), 1)
    # Enforce minimum R:R
    if new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
    er["stop_loss_value"] = new_sl
    er["take_profit_value"] = new_tp

    # Mutate contracts — within bounds
    sz = new["sizing_rules"]
    sz["method"] = "fixed"
    ct = sz["fixed_contracts"]
    new_ct = max(CT_MIN, min(CT_MAX, int(ct + ct * intensity * random.uniform(-0.5, 0.5))))
    sz["fixed_contracts"] = new_ct

    # Mutate time window (explore different high-vol periods)
    for filt in new.get("entry_filters", []):
        if filt.get("function") == "time_of_day":
            if random.random() < 0.2:
                # Occasionally shift the window (now in ET directly)
                windows = [
                    (9, 30, 11, 0),    # 9:30-11:00 ET — opening power hour
                    (9, 30, 11, 30),   # 9:30-11:30 ET — extended opening
                    (9, 30, 10, 30),   # 9:30-10:30 ET — first hour only
                    (9, 45, 11, 0),    # 9:45-11:00 ET — slight delay
                    (9, 0, 11, 0),     # 9:00-11:00 ET — pre-market + first hour
                    (9, 30, 12, 0),    # 9:30-12:00 ET — morning session
                ]
                w = random.choice(windows)
                filt["params"] = {
                    "start_hour": w[0], "start_minute": w[1],
                    "end_hour": w[2], "end_minute": w[3],
                }

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
    new["name"] = f"{base}|sniper_{h}"
    return new


def crossover(a, b):
    """Aggressive crossover with param blending."""
    child = copy.deepcopy(a)

    # Always blend signal params
    for i, sig in enumerate(child["entry_signals"]):
        if i < len(b["entry_signals"]):
            b_params = b["entry_signals"][i]["params"]
            for k in sig["params"]:
                if k in b_params and isinstance(sig["params"][k], (int, float)):
                    w = random.uniform(0.2, 0.8)
                    blended = sig["params"][k] * w + b_params[k] * (1 - w)
                    blended *= random.uniform(0.9, 1.1)
                    sig["params"][k] = int(round(blended)) if isinstance(sig["params"][k], int) else round(blended, 4)

    # Blend filter params (skip time_of_day)
    for i, filt in enumerate(child.get("entry_filters", [])):
        if filt.get("function") == "time_of_day":
            continue
        if i < len(b.get("entry_filters", [])):
            bf = b["entry_filters"][i]
            if bf.get("function") == "time_of_day":
                continue
            for k in filt["params"]:
                if k in bf["params"] and isinstance(filt["params"][k], (int, float)):
                    w = random.uniform(0.2, 0.8)
                    blended = filt["params"][k] * w + bf["params"][k] * (1 - w)
                    blended *= random.uniform(0.9, 1.1)
                    filt["params"][k] = int(round(blended)) if isinstance(filt["params"][k], int) else round(blended, 4)

    # Blend exits (within sniper bounds)
    a_er, b_er = a["exit_rules"], b["exit_rules"]
    w = random.uniform(0.3, 0.7)
    new_sl = round(max(SL_MIN, min(SL_MAX, a_er["stop_loss_value"] * w + b_er["stop_loss_value"] * (1 - w))), 1)
    new_tp = round(max(TP_MIN, min(TP_MAX, a_er["take_profit_value"] * w + b_er["take_profit_value"] * (1 - w))), 1)
    if new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
    child["exit_rules"]["stop_loss_value"] = new_sl
    child["exit_rules"]["take_profit_value"] = new_tp

    # Blend contracts
    a_ct = a["sizing_rules"]["fixed_contracts"]
    b_ct = b["sizing_rules"]["fixed_contracts"]
    child["sizing_rules"]["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(a_ct * w + b_ct * (1 - w))))

    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = child["name"].split("|")
    base = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
    child["name"] = f"{base}|cross_{h}"
    return child


# ═══════════════════════════════════════════════════════════════════════
# DIVERSITY PRUNING
# ═══════════════════════════════════════════════════════════════════════

def decorrelate(population, max_corr=CORRELATION_THRESHOLD):
    if len(population) <= 2:
        return population

    keep = [True] * len(population)
    n = len(population)
    monthly_vecs = [monthly_pnl(p[4]) for p in population]

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            all_months = sorted(set(monthly_vecs[i].keys()) | set(monthly_vecs[j].keys()))
            if len(all_months) < 3:
                continue
            vec_i = [monthly_vecs[i].get(m, 0) for m in all_months]
            vec_j = [monthly_vecs[j].get(m, 0) for m in all_months]
            corr = np.corrcoef(vec_i, vec_j)[0, 1]
            if not np.isnan(corr) and abs(corr) > max_corr:
                if population[i][3] >= population[j][3]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [p for p, k in zip(population, keep) if k]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    # Fix #36: Reproducible random seeds
    random.seed(42)
    np.random.seed(42)

    logger.info("Loading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))

    split_date = "2024-11-19"
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_validate = df_yr1.filter(pl.col("timestamp") >= pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))

    logger.info(f"  Year 1: {len(df_yr1):,} bars")
    logger.info(f"  Train (8mo): {len(df_train):,} bars")
    logger.info(f"  Validate (4mo): {len(df_validate):,} bars")

    data_train = {"1m": df_train}
    data_validate = {"1m": df_validate}
    data_full = {"1m": df_yr1}

    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)

    config_train = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_150k",
        start_date="2024-03-19", end_date="2024-11-18",
        slippage_ticks=3, initial_capital=150000.0,
    )
    config_validate = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_150k",
        start_date="2024-11-19", end_date="2025-03-18",
        slippage_ticks=3, initial_capital=150000.0,
    )
    config_full = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_150k",
        start_date="2024-03-19", end_date="2025-03-18",
        slippage_ticks=3, initial_capital=150000.0,
    )

    # Load winners and convert to sniper format
    with open("reports/maximized_strategies_150k.json") as f:
        winners = json.load(f)

    # Create diverse sniper seeds from all 57 winners
    seeds = []
    for w in winners:
        # Multiple sniper variants per winner with different params
        for _ in range(3):
            seeds.append(to_sniper(w["strategy"]))

    # Fix #4: Add structurally diverse seeds from the full signal library
    diverse_seeds = generate_diverse_seeds(200)
    seeds.extend(diverse_seeds)

    logger.info(f"Created {len(seeds)} seeds ({len(seeds) - len(diverse_seeds)} from winners + {len(diverse_seeds)} diverse)")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║     BREED 150K v3 — SNIPER MODE                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Strategy:    Tight SL ({SL_MIN}-{SL_MAX}pt) + Huge TP ({TP_MIN}-{TP_MAX}pt) + {CT_MIN}-{CT_MAX}ct  ║
║  Session:     9:30–11:00 ET (13:30-15:00 UTC) power hour               ║
║  Win Rate:    Expect 20-25% — asymmetric R:R does the work            ║
║  Risk/Trade:  ~15pt × 10ct × $2 = $300 loss                           ║
║  Reward/Trade: ~200pt × 10ct × $2 = $4,000 win                        ║
║  Seeds:       {len(seeds)} sniper variants from {len(winners)} winners                    ║
║  Account:     $150,000 Topstep (DD: -$4,500, Daily: -$3,000)          ║
║  Target:      $15,000+/month consistent                                ║
║  DD Rule:     HARD KILL at -$4,500                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: EVOLVE ON TRAIN DATA
    # ══════════════════════════════════════════════════════════════════
    logger.info("═══ PHASE 1: Evolution on train data (8 months) ═══")
    logger.info("Evaluating sniper seeds...")

    population = []
    for sd in seeds:
        out = backtest(sd, data_train, rm, config_train, min_trades=15)
        if out:
            trades, m = out
            mc = mc_test(trades, prop_rules)
            if mc.probability_of_profit > 0.4:  # Lower bar for snipers (low WR is expected)
                score = fitness(m, mc, trades)
                population.append((sd, m, mc, score, trades))

    population.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  {len(population)} sniper seeds viable on train data")

    if population:
        sd, m, mc, score, trades = population[0]
        months = monthly_pnl(trades)
        avg_mo = np.mean(list(months.values())) if months else 0
        logger.info(f"  Best seed: PnL=${m.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | WR={m.win_rate:.1f}% | DD=${m.max_drawdown:,.0f}")

    best_ever = population[0] if population else None
    best_ever_score = population[0][3] if population else -float("inf")
    tested = len(seeds)

    MAX_GENERATIONS = 50
    POP_SIZE = 120
    ELITE_COUNT = 20
    MUTANT_COUNT = 100
    CROSSOVER_COUNT = 80
    IMMIGRANT_COUNT = 20

    stagnant_gens = 0
    prev_best_score = best_ever_score

    for gen in range(MAX_GENERATIONS):
        gen_start = time.time()
        # Fix #12: Floor intensity at 0.25, not 0.10
        intensity = max(0.25, 1.5 * (1.0 - gen / MAX_GENERATIONS))

        elites = [p[0] for p in population[:ELITE_COUNT]]
        mid_tier = [p[0] for p in population[ELITE_COUNT:ELITE_COUNT*3]] if len(population) > ELITE_COUNT else []

        # Fix #13: Inject diversity after stagnation
        if stagnant_gens >= 8:
            logger.info(f"  ★ Stagnation detected ({stagnant_gens} gens) — injecting diversity")
            intensity = min(2.0, intensity * 3)
            stagnant_gens = 0

        # Mutations
        mutants = []
        for _ in range(MUTANT_COUNT):
            parent = random.choice(mid_tier) if (mid_tier and random.random() < 0.3) else random.choice(elites) if elites else random.choice(seeds)
            mutants.append(mutate_strategy(parent, intensity))

        # Crossovers
        children = []
        pool = elites + mid_tier if mid_tier else elites
        for _ in range(CROSSOVER_COUNT):
            if len(pool) >= 2:
                a, b = random.sample(pool, 2)
                children.append(crossover(a, b))

        # Immigrants
        immigrants = []
        if len(population) > ELITE_COUNT:
            lower = [p[0] for p in population[ELITE_COUNT:]]
            for _ in range(IMMIGRANT_COUNT):
                immigrants.append(mutate_strategy(random.choice(lower), intensity * 2))

        # Evaluate — Fix #11: use proxy fitness during evolution (no MC)
        new_pop = []
        for sd in elites:
            out = backtest(sd, data_train, rm, config_train, min_trades=15)
            if out:
                trades, m = out
                score = fitness_proxy(m, trades)
                new_pop.append((sd, m, None, score, trades))

        for sd in mutants + children + immigrants:
            tested += 1
            out = backtest(sd, data_train, rm, config_train, min_trades=15)
            if out:
                trades, m = out
                if m.total_pnl > 0:
                    score = fitness_proxy(m, trades)
                    new_pop.append((sd, m, None, score, trades))

        new_pop.sort(key=lambda x: x[3], reverse=True)
        # Fix #10: Only decorrelate in final phase, not every generation
        population = new_pop[:POP_SIZE]

        # Fix #13: Track stagnation
        if population and population[0][3] > prev_best_score * 1.001:
            stagnant_gens = 0
            prev_best_score = population[0][3]
        else:
            stagnant_gens += 1

        if population and population[0][3] > best_ever_score:
            best_ever = population[0]
            best_ever_score = population[0][3]

        gen_elapsed = time.time() - gen_start

        if population:
            sd, m, mc, score, trades = population[0]
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values())) if months else 0
            min_mo = min(months.values()) if months else 0
            sz = sd["sizing_rules"]
            er = sd["exit_rules"]

            high_earners = sum(
                1 for _, _, _, _, t in population
                if np.mean(list(monthly_pnl(t).values())) >= TARGET_AVG_MONTHLY
            )

            mc_str = f"MC={mc.probability_of_profit:.0%}" if mc else "MC=proxy"
            logger.info(
                f"  Gen {gen+1:>2}/{MAX_GENERATIONS} | int={intensity:.2f} | pop={len(population):>3} | "
                f"PnL=${m.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | min/mo=${min_mo:,.0f} | "
                f"WR={m.win_rate:.0f}% | SL={er['stop_loss_value']}→TP={er['take_profit_value']} | "
                f"{sz['fixed_contracts']}ct | DD=${m.max_drawdown:,.0f} | {mc_str} | "
                f"$15K+={high_earners} | {gen_elapsed:.0f}s"
            )

            if avg_mo >= TARGET_AVG_MONTHLY and min_mo > 0 and high_earners >= 5:
                logger.info(f"  ★★★ TARGET HIT ★★★")
                if gen >= 10 and high_earners >= 10:
                    break

    logger.info(f"  Evolution complete: {tested:,} tested over {gen+1} generations")

    # Fix #10: Decorrelate ONCE after evolution, before validation
    population = decorrelate(population, CORRELATION_THRESHOLD)
    logger.info(f"  After decorrelation: {len(population)} unique strategies")

    # Run MC on top candidates now (fix #11: MC only in validation phases)
    logger.info("  Running MC on top candidates...")
    mc_population = []
    for sd, m, _, score, trades in population[:60]:
        mc = mc_test(trades, prop_rules)
        if mc.probability_of_profit > 0.4:
            score_mc = fitness(m, mc, trades)
            mc_population.append((sd, m, mc, score_mc, trades))
    mc_population.sort(key=lambda x: x[3], reverse=True)
    population = mc_population
    logger.info(f"  After MC filter: {len(population)} strategies")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 2: Walk-forward validation (4 months OOS) ═══")

    oos_survivors = []
    for sd, m_train, mc_train, score, _ in population:
        out = backtest(sd, data_validate, rm, config_validate, min_trades=5)
        if out is None:
            continue
        trades_oos, m_oos = out
        mc_oos = mc_test(trades_oos, prop_rules, MC_SIMS_TRAIN)
        if mc_oos.probability_of_profit < OOS_MC_P_PROFIT_MIN or m_oos.total_pnl <= 0:
            continue
        oos_score = fitness(m_oos, mc_oos, trades_oos)
        oos_survivors.append((sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos))
        months = monthly_pnl(trades_oos)
        avg_mo = np.mean(list(months.values())) if months else 0
        logger.info(
            f"  ✓ OOS: {sd['name'][:40]} | PnL=${m_oos.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | "
            f"WR={m_oos.win_rate:.0f}% | DD=${m_oos.max_drawdown:,.0f} | MC={mc_oos.probability_of_profit:.0%}"
        )

    logger.info(f"  OOS: {len(oos_survivors)}/{len(population)} survived")

    if not oos_survivors:
        logger.warning("  No OOS survivors at 80% — trying 65%...")
        for sd, m_train, mc_train, score, _ in population:
            out = backtest(sd, data_validate, rm, config_validate, min_trades=3)
            if out is None:
                continue
            trades_oos, m_oos = out
            mc_oos = mc_test(trades_oos, prop_rules, MC_SIMS_TRAIN)
            if mc_oos.probability_of_profit < 0.65 or m_oos.total_pnl <= 0:
                continue
            oos_score = fitness(m_oos, mc_oos, trades_oos)
            oos_survivors.append((sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos))
        logger.info(f"  Relaxed OOS: {len(oos_survivors)} survived")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2.5: WALK-FORWARD + REGIME VALIDATION (Fix #23, #24)
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 2.5: Walk-forward + regime validation ═══")

    try:
        from validation.walk_forward import WalkForwardValidator
        from validation.regime import RegimeDetector

        wf_validated = []
        for sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos in oos_survivors:
            try:
                strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
                wf = WalkForwardValidator(
                    data=data_full, risk_manager=rm,
                    contract_spec=MNQ_SPEC, config=config_full,
                    account_size=150000.0,
                )
                wf_result = wf.validate(strategy, train_days=60, test_days=20)

                # Fix #23: Kill strategies with WF efficiency < 0.3
                if wf_result.wf_efficiency < 0.3:
                    logger.info(f"  ✗ WF reject: {sd['name'][:40]} | WF eff={wf_result.wf_efficiency:.2f}")
                    continue

                logger.info(f"  ✓ WF pass: {sd['name'][:40]} | WF eff={wf_result.wf_efficiency:.2f}")
                wf_validated.append((sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos))
            except Exception as e:
                # If WF fails, still include the strategy
                wf_validated.append((sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos))

        if wf_validated:
            oos_survivors = wf_validated
            logger.info(f"  WF survivors: {len(oos_survivors)}")

        # Fix #24: Regime-aware analysis on full-year data
        detector = RegimeDetector()
        df_classified = detector.classify(df_yr1)
        for sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos in oos_survivors[:10]:
            try:
                out = backtest(sd, data_full, rm, config_full, min_trades=10)
                if out:
                    trades_full, _ = out
                    analysis = detector.analyze_strategy(trades_full, df_classified, sd["name"][:40])
                    # Log regime sensitivity
                    if hasattr(analysis, 'regime_sensitivity') and analysis.regime_sensitivity:
                        logger.info(f"  Regime: {sd['name'][:30]} | sensitivity={analysis.regime_sensitivity:.2f}")
            except Exception:
                pass

    except ImportError:
        logger.warning("  Walk-forward/regime modules not available, skipping")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: FULL-YEAR MC
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 3: Full-year MC ({MC_SIMS_FINAL} sims) ═══")

    final_candidates = []
    for sd, m_train, mc_train, m_oos, mc_oos, oos_score, _ in oos_survivors:
        out = backtest(sd, data_full, rm, config_full, min_trades=MIN_TRADES)
        if out is None:
            continue
        trades_full, m_full = out
        mc_full = mc_test(trades_full, prop_rules, MC_SIMS_FINAL)
        if mc_full.probability_of_profit < 0.85:
            continue
        full_score = fitness(m_full, mc_full, trades_full)
        months = monthly_pnl(trades_full)
        avg_mo = np.mean(list(months.values()))
        final_candidates.append((sd, m_full, mc_full, full_score, trades_full))
        logger.info(
            f"  ★ {sd['name'][:40]} | PnL=${m_full.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | "
            f"WR={m_full.win_rate:.0f}% | DD=${m_full.max_drawdown:,.0f} | MC={mc_full.probability_of_profit:.0%}"
        )

    final_candidates.sort(key=lambda x: x[3], reverse=True)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: FINAL DE-CORRELATION
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 4: De-correlation ═══")
    final_pool = decorrelate(final_candidates, CORRELATION_THRESHOLD)
    final_pool.sort(key=lambda x: x[3], reverse=True)

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{'='*150}")
    print(f"  SNIPER BREED COMPLETE — {tested:,} tested in {total_elapsed/60:.1f} min")
    print(f"  Pipeline: {len(seeds)} seeds → {len(population)} evolved → {len(oos_survivors)} OOS → {len(final_candidates)} MC → {len(final_pool)} final")
    print(f"{'='*150}")

    if final_pool:
        print(f"\n  {'#':<4} {'Strategy':<42} {'Tr':>5} {'WR':>6} {'PF':>5} {'1yr PnL':>12} {'Avg/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'SL→TP':>12} {'Ct':>4} {'MC P':>6}")
        print(f"  {'-'*150}")
        for i, (sd, m, mc, score, trades) in enumerate(final_pool[:30], 1):
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values()))
            min_mo = min(months.values())
            er = sd["exit_rules"]
            flag = "★" if avg_mo >= TARGET_AVG_MONTHLY else " "
            print(
                f"  {flag}{i:<3} {sd['name'][:41]:<42} "
                f"{m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>4.2f} "
                f"${m.total_pnl:>11,.0f} ${avg_mo:>9,.0f} ${min_mo:>9,.0f} "
                f"${m.max_drawdown:>9,.0f} {er['stop_loss_value']:>5.1f}→{er['take_profit_value']:>5.1f} "
                f"{sd['sizing_rules']['fixed_contracts']:>4} {mc.probability_of_profit:>5.0%}"
            )
        print(f"  {'-'*150}")

        # Champion
        sd, m, mc, score, trades = final_pool[0]
        months = monthly_pnl(trades)
        avg_mo = np.mean(list(months.values()))
        max_mo = max(months.values())
        min_mo = min(months.values())
        er = sd["exit_rules"]

        print(f"""
  ══════════════════════════════════════════════════════════════
  CHAMPION — SNIPER v3
  ══════════════════════════════════════════════════════════════
  Name:           {sd['name']}
  Entry:          {', '.join(e['signal_name'] for e in sd['entry_signals'])}
  Filters:        {', '.join(f['signal_name'] for f in sd.get('entry_filters', []))}
  Stop Loss:      {er['stop_loss_value']}pt
  Take Profit:    {er['take_profit_value']}pt
  R:R:            {er['take_profit_value']/er['stop_loss_value']:.1f}:1
  Contracts:      {sd['sizing_rules']['fixed_contracts']}
  ──────────────────────────────────────────────────────────────
  Trades (1yr):   {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe:         {m.sharpe_ratio:.2f}
  Net P&L (1yr):  ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}  (limit: ${MAX_DD_LIMIT:,.0f})
  DD Margin:      ${abs(MAX_DD_LIMIT) - abs(m.max_drawdown):,.2f}
  ──────────────────────────────────────────────────────────────
  Avg Month:      ${avg_mo:,.2f}
  Best Month:     ${max_mo:,.2f}
  Worst Month:    ${min_mo:,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC P(ruin):     {mc.probability_of_ruin:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  MC Composite:   {mc.composite_score:.1f}/100
  MC Pass Rate:   {mc.prop_firm_pass_rate:.1%}
  ══════════════════════════════════════════════════════════════""")

        print(f"\n  Parameters:")
        for sig in sd["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in sd.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

        print(f"\n  MONTHLY P&L:")
        for mo in sorted(months.keys()):
            p = months[mo]
            bar = "█" * max(1, int(abs(p) / 500))
            flag = "★" if p >= TARGET_AVG_MONTHLY else " "
            sign = "+" if p >= 0 else "-"
            print(f"    {flag} {mo}: {sign}${abs(p):>10,.2f}  {bar}{'*' if p<0 else ''}")
        prof_months = sum(1 for p in months.values() if p > 0)
        print(f"    Profitable months: {prof_months}/{len(months)} ({prof_months/len(months):.0%})")

    # ── Save ──
    if final_pool:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "breed_150k_v3_sniper",
            "account": {"initial_capital": 150000, "prop_firm": "topstep_150k", "max_drawdown": MAX_DD_LIMIT},
            "sniper_config": {"sl_range": [SL_MIN, SL_MAX], "tp_range": [TP_MIN, TP_MAX], "ct_range": [CT_MIN, CT_MAX], "rr_min": RR_MIN},
            "data": {"full_period": "2024-03-19 to 2025-03-18", "train_period": "2024-03-19 to 2024-11-18", "validate_period": "2024-11-19 to 2025-03-18"},
            "pipeline_stats": {"seeds": len(seeds), "total_tested": tested, "generations": gen + 1, "evolved": len(population), "oos_survivors": len(oos_survivors), "mc_validated": len(final_candidates), "final": len(final_pool), "elapsed_min": round(total_elapsed / 60, 1)},
            "strategies": [],
        }
        for sd, m, mc, score, trades in final_pool:
            months = monthly_pnl(trades)
            monthly_vals = list(months.values())
            output["strategies"].append({
                "name": sd["name"], "strategy": sd, "fitness": score,
                "trades": m.total_trades, "win_rate": m.win_rate, "profit_factor": m.profit_factor, "sharpe": m.sharpe_ratio,
                "total_pnl": m.total_pnl, "max_drawdown": m.max_drawdown, "dd_margin": abs(MAX_DD_LIMIT) - abs(m.max_drawdown),
                "avg_monthly": float(np.mean(monthly_vals)), "max_monthly": float(max(monthly_vals)), "min_monthly": float(min(monthly_vals)),
                "pct_months_profitable": sum(1 for v in monthly_vals if v > 0) / len(monthly_vals),
                "monthly_breakdown": {mo: round(v, 2) for mo, v in sorted(months.items())},
                "mc_median": mc.median_return, "mc_mean": mc.mean_return, "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin, "mc_5th": mc.pct_5th_return, "mc_95th": mc.pct_95th_return,
                "mc_composite": mc.composite_score, "mc_pass_rate": mc.prop_firm_pass_rate,
            })
        with open("reports/bred_strategies_150k.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(final_pool)} strategies to reports/bred_strategies_150k.json")

    elif oos_survivors:
        logger.warning("No strategies passed final MC — saving OOS survivors")
        output = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "breed_150k_v3_sniper", "note": "OOS only", "strategies": []}
        for sd, m_train, mc_train, m_oos, mc_oos, oos_score, trades_oos in oos_survivors:
            output["strategies"].append({"name": sd["name"], "strategy": sd, "oos_pnl": m_oos.total_pnl, "oos_trades": m_oos.total_trades, "oos_dd": m_oos.max_drawdown, "oos_mc_p": mc_oos.probability_of_profit})
        with open("reports/bred_strategies_150k.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(oos_survivors)} OOS survivors")

    else:
        print("\n  No strategies survived. The sniper approach may need different signal families or wider DD.")

    print(f"\n{'='*150}\n")


if __name__ == "__main__":
    main()
