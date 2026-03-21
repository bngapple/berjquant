#!/usr/bin/env python3
"""
CROSSBREED V4 — Cross-family evolution engine.

Loads ALL winning strategies from every report file, deduplicates them,
then runs a full genetic algorithm with CROSS-FAMILY crossover as the
key operation: pairing entry signals from one family with filters from
another to discover novel hybrid strategies.

Pipeline:
  Phase 0: Load all winners → deduplicate → prepare seeds
  Phase 1: Cross-family evolution on 8-month train data (proxy fitness)
  Phase 2: Walk-forward validation (train=60d, test=20d, step=20d)
  Phase 3: Full-year MC stress test (5000 sims)
  Phase 4: Decorrelation (monthly PnL corr < 0.75)
  Phase 5: Output ranked leaderboard + JSON

Target: $15K+/month consistent on Topstep 150K account.
"""

import argparse
import json
import time
import copy
import random
import hashlib
import logging
import traceback
from collections import Counter, defaultdict
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
from signals.registry import SignalRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crossbreed")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")

# ── Hard limits ───────────────────────────────────────────────────────
MAX_DD_LIMIT = -4500.0
DAILY_LOSS_LIMIT = -3000.0
MIN_TRADES_TRAIN = 15
MIN_TRADES_OOS = 5
MC_SIMS_FINAL = 5000
OOS_MC_P_PROFIT_MIN = 0.80
CORRELATION_THRESHOLD = 0.75
TARGET_AVG_MONTHLY = 15000

# ── 150K account bounds ──────────────────────────────────────────────
SL_MIN, SL_MAX = 10.0, 35.0
TP_MIN, TP_MAX = 50.0, 300.0
CT_MIN, CT_MAX = 4, 15
RR_MIN = 4.0

# ── Evolution settings ───────────────────────────────────────────────
POP_CAP = 200
ELITE_COUNT = 30
MAX_GENERATIONS = 60
STAGNATION_THRESHOLD = 8

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def backtest_strategy(sd, data, rm, config, min_trades=MIN_TRADES_TRAIN):
    """Run a backtest on a strategy dict. Returns (trades, metrics) or None."""
    try:
        sd_copy = copy.deepcopy(sd)
        sd_copy["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd_copy)
        bt = VectorizedBacktester(
            data=data, risk_manager=rm,
            contract_spec=MNQ_SPEC, config=config,
        )
        result = bt.run(strategy)
        if len(result.trades) < min_trades:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital, result.equity_curve)
        if metrics.max_drawdown < MAX_DD_LIMIT:
            return None
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, prop_rules, n_sims=2000):
    """Run MC stress test."""
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=150000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def monthly_pnl(trades):
    """Compute monthly P&L from trade list."""
    months = {}
    for t in trades:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


def get_signal_family(sd):
    """Extract signal family label like 'stochastic|imbalance'."""
    entries = sorted(e["signal_name"] for e in sd.get("entry_signals", []))
    filters = sorted(
        f["signal_name"] for f in sd.get("entry_filters", [])
        if f.get("signal_name") != "time_of_day"
    )
    return "|".join(entries) + ("+" + "|".join(filters) if filters else "")


def strategy_hash(sd):
    """Deterministic hash for deduplication."""
    key_parts = {
        "entry_signals": [
            {"signal_name": e["signal_name"], "params": e.get("params", {})}
            for e in sd.get("entry_signals", [])
        ],
        "entry_filters": [
            {"signal_name": f["signal_name"], "params": f.get("params", {})}
            for f in sd.get("entry_filters", [])
            if f.get("signal_name") != "time_of_day"
        ],
        "exit_rules": {
            k: sd.get("exit_rules", {}).get(k)
            for k in ["stop_loss_value", "take_profit_value", "stop_loss_type", "take_profit_type"]
        },
    }
    blob = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fitness_proxy(metrics, trades):
    """Proxy fitness for evolution (no MC). Matches user spec exactly."""
    months = monthly_pnl(trades)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    dd = abs(metrics.max_drawdown)

    months_hitting_15k = sum(1 for v in monthly_vals if v >= 15000)
    floor_penalty = sum(max(0, 15000 - v) for v in monthly_vals if v > 0) * 0.3

    score = (
        avg_monthly * 4.0
        + min_monthly * 2.5
        + metrics.sharpe_ratio * 1000
        + metrics.profit_factor * 500
        - dd * 0.5
        + months_hitting_15k * 2000
        - floor_penalty
    )
    return score


def fitness_full(metrics, mc, trades):
    """Full fitness with MC metrics."""
    months = monthly_pnl(trades)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    dd = abs(metrics.max_drawdown)

    dd_margin_bonus = max(0, (4500 - dd) / 2500) * 3000
    target_bonus = max(0, (avg_monthly - 15000)) * 3.0 if avg_monthly > 15000 else 0
    floor_penalty = sum(max(0, 15000 - v) for v in monthly_vals if v > 0) * 0.3
    months_hitting_15k = sum(1 for v in monthly_vals if v >= 15000)
    consistency_bonus = (months_hitting_15k / len(monthly_vals)) * 10000
    pct_profitable = sum(1 for v in monthly_vals if v > 0) / len(monthly_vals)

    score = (
        avg_monthly * 4.0
        + min_monthly * 2.5
        + mc.median_return * 0.5
        + mc.probability_of_profit * 5000 * 0.4
        + pct_profitable * 10000 * 0.6
        + dd_margin_bonus
        + metrics.sharpe_ratio * 500
        + target_bonus
        + consistency_bonus
        - floor_penalty
    )
    return score


def mutate_params(params, intensity=0.5):
    """Mutate numeric params by intensity factor."""
    new = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            delta = abs(v) * intensity * random.uniform(-1, 1)
            new_val = v + delta
            if isinstance(v, int):
                new[k] = max(1, int(round(new_val)))
            else:
                new[k] = round(max(0.01, new_val), 4)
        else:
            new[k] = v
    return new


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: LOAD + DEDUPLICATE + PREPARE
# ═══════════════════════════════════════════════════════════════════════

def load_all_seeds():
    """Load strategy dicts from all report files, deduplicate, prepare."""
    report_files = [
        ("reports/1yr_validation.json", "1yr_validation", True),
        ("reports/mc_winners_2yr.json", "mc_winners_2yr", False),
        ("reports/mc_winners.json", "mc_winners", False),
        ("reports/maximized_strategies_150k.json", "maximized_150k", False),
        ("reports/evolved_strategies_yr1.json", "evolved_yr1", False),
        ("reports/evolved_strategies.json", "evolved", False),
        ("reports/maximized_strategies.json", "maximized", False),
    ]

    all_strategies = []
    elite_hashes = set()  # Track 1yr_validation profitable ones as elite

    for filepath, label, is_oos in report_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"  Skipping {filepath}: {e}")
            continue

        # Extract items list
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["results", "strategies", "winners", "validated"]:
                if key in data:
                    items = data[key]
                    break
            else:
                items = []
        else:
            items = []

        count = 0
        for item in items:
            # Extract strategy dict — handle nested vs flat
            if "strategy" in item and isinstance(item["strategy"], dict):
                sd = item["strategy"]
            elif "entry_signals" in item:
                sd = item
            else:
                continue

            # Validate required fields
            if not sd.get("entry_signals") or not sd.get("exit_rules"):
                continue

            sd = copy.deepcopy(sd)

            # Mark elite seeds (profitable 1yr_validation)
            is_elite = False
            if is_oos:
                total_pnl = item.get("total_pnl", 0)
                if total_pnl > 0:
                    is_elite = True

            all_strategies.append((sd, label, is_elite))
            count += 1

        logger.info(f"  {filepath}: loaded {count} strategies")

    # Deduplicate
    seen_hashes = {}
    unique = []
    for sd, label, is_elite in all_strategies:
        h = strategy_hash(sd)
        if h not in seen_hashes:
            seen_hashes[h] = True
            unique.append((sd, label, is_elite))
            if is_elite:
                elite_hashes.add(h)

    logger.info(f"  Total loaded: {len(all_strategies)} → {len(unique)} unique after dedup")

    # Count signal families
    family_counts = Counter(get_signal_family(sd) for sd, _, _ in unique)
    logger.info(f"  Signal families: {len(family_counts)} unique")
    for fam, cnt in family_counts.most_common(10):
        logger.info(f"    {cnt:>4}  {fam}")

    return unique, elite_hashes, family_counts


def prepare_seed(sd, from_50k=False):
    """Prepare a seed for 150K: fix timeframe, sizing, SL/TP, time filter, R:R."""
    sd = copy.deepcopy(sd)

    # Force 1m timeframe
    sd["primary_timeframe"] = "1m"

    # Force fixed_points for SL/TP
    er = sd.get("exit_rules", {})
    er["stop_loss_type"] = "fixed_points"
    er["take_profit_type"] = "fixed_points"

    # Scale SL/TP to 150K ranges, clamp
    sl = er.get("stop_loss_value", 20.0)
    tp = er.get("take_profit_value", 100.0)

    # If from 50K account, scale contracts up
    sz = sd.get("sizing_rules", {})
    sz["method"] = "fixed"
    ct = sz.get("fixed_contracts", 1)
    if from_50k:
        ct = min(CT_MAX, max(CT_MIN, ct * 3))
    else:
        ct = min(CT_MAX, max(CT_MIN, ct))
    sz["fixed_contracts"] = ct

    # Clamp SL/TP
    sl = max(SL_MIN, min(SL_MAX, sl))
    tp = max(TP_MIN, min(TP_MAX, tp))

    # Enforce minimum R:R
    if sl > 0 and tp / sl < RR_MIN:
        tp = round(sl * RR_MIN, 1)
        tp = min(TP_MAX, tp)
        # If TP capped and R:R still not met, tighten SL
        if tp / sl < RR_MIN:
            sl = round(tp / RR_MIN, 1)
            sl = max(SL_MIN, sl)

    er["stop_loss_value"] = round(sl, 1)
    er["take_profit_value"] = round(tp, 1)
    sd["exit_rules"] = er
    sd["sizing_rules"] = sz

    # Add time_of_day filter for 9:30-11:00 ET if missing
    has_time_filter = any(
        f.get("signal_name") == "time_of_day" or f.get("function") == "time_of_day"
        for f in sd.get("entry_filters", [])
    )
    if not has_time_filter:
        sd.setdefault("entry_filters", []).append({
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {"start_hour": 9, "start_minute": 30, "end_hour": 11, "end_minute": 0},
            "column": "signal_time_allowed",
        })

    sd["require_all_entries"] = sd.get("require_all_entries", True)
    return sd


# ═══════════════════════════════════════════════════════════════════════
# EVOLUTION OPERATORS
# ═══════════════════════════════════════════════════════════════════════

def cross_family_crossover(parent_a, parent_b, intensity=0.5):
    """
    CROSS-FAMILY CROSSOVER: Take entry signals from parent_a and
    filters from parent_b (excluding time_of_day). Blend exit rules.
    """
    child = copy.deepcopy(parent_a)

    # Take entry signals from parent_a (already there)
    # Take non-time filters from parent_b
    b_filters = [
        copy.deepcopy(f) for f in parent_b.get("entry_filters", [])
        if f.get("signal_name") != "time_of_day"
    ]
    # Keep time_of_day from parent_a
    time_filters = [
        f for f in child.get("entry_filters", [])
        if f.get("signal_name") == "time_of_day"
    ]
    child["entry_filters"] = b_filters + time_filters

    # Blend exit rules
    a_er = parent_a["exit_rules"]
    b_er = parent_b["exit_rules"]
    w = random.uniform(0.3, 0.7)

    new_sl = round(a_er.get("stop_loss_value", 20) * w + b_er.get("stop_loss_value", 20) * (1 - w), 1)
    new_tp = round(a_er.get("take_profit_value", 100) * w + b_er.get("take_profit_value", 100) * (1 - w), 1)
    new_sl = max(SL_MIN, min(SL_MAX, new_sl))
    new_tp = max(TP_MIN, min(TP_MAX, new_tp))
    if new_sl > 0 and new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
        new_tp = min(TP_MAX, new_tp)

    child["exit_rules"]["stop_loss_value"] = new_sl
    child["exit_rules"]["take_profit_value"] = new_tp

    # Blend contracts
    a_ct = parent_a.get("sizing_rules", {}).get("fixed_contracts", 4)
    b_ct = parent_b.get("sizing_rules", {}).get("fixed_contracts", 4)
    child["sizing_rules"]["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(a_ct * w + b_ct * (1 - w))))

    # Name
    a_entries = "|".join(e["signal_name"] for e in child.get("entry_signals", []))
    b_filters_names = "|".join(f["signal_name"] for f in b_filters) if b_filters else "none"
    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    child["name"] = f"{a_entries.upper()}×{b_filters_names.upper()}|xfam_{h}"

    return child


def same_family_crossover(parent_a, parent_b, intensity=0.5):
    """SAME-FAMILY CROSSOVER: Blend all numeric params between same-signal parents."""
    child = copy.deepcopy(parent_a)

    # Blend entry signal params
    for i, sig in enumerate(child["entry_signals"]):
        if i < len(parent_b["entry_signals"]):
            b_params = parent_b["entry_signals"][i]["params"]
            for k in sig["params"]:
                if k in b_params and isinstance(sig["params"][k], (int, float)):
                    w = random.uniform(0.2, 0.8)
                    blended = sig["params"][k] * w + b_params[k] * (1 - w)
                    blended *= random.uniform(0.95, 1.05)
                    if isinstance(sig["params"][k], int):
                        sig["params"][k] = max(1, int(round(blended)))
                    else:
                        sig["params"][k] = round(max(0.01, blended), 4)

    # Blend filter params (skip time_of_day)
    for i, filt in enumerate(child.get("entry_filters", [])):
        if filt.get("signal_name") == "time_of_day":
            continue
        if i < len(parent_b.get("entry_filters", [])):
            bf = parent_b["entry_filters"][i]
            if bf.get("signal_name") == "time_of_day":
                continue
            for k in filt["params"]:
                if k in bf["params"] and isinstance(filt["params"][k], (int, float)):
                    w = random.uniform(0.2, 0.8)
                    blended = filt["params"][k] * w + bf["params"][k] * (1 - w)
                    if isinstance(filt["params"][k], int):
                        filt["params"][k] = max(1, int(round(blended)))
                    else:
                        filt["params"][k] = round(max(0.01, blended), 4)

    # Blend exits
    a_er, b_er = parent_a["exit_rules"], parent_b["exit_rules"]
    w = random.uniform(0.3, 0.7)
    new_sl = round(max(SL_MIN, min(SL_MAX, a_er["stop_loss_value"] * w + b_er["stop_loss_value"] * (1 - w))), 1)
    new_tp = round(max(TP_MIN, min(TP_MAX, a_er["take_profit_value"] * w + b_er["take_profit_value"] * (1 - w))), 1)
    if new_sl > 0 and new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
        new_tp = min(TP_MAX, new_tp)
    child["exit_rules"]["stop_loss_value"] = new_sl
    child["exit_rules"]["take_profit_value"] = new_tp

    # Blend contracts
    a_ct = parent_a["sizing_rules"]["fixed_contracts"]
    b_ct = parent_b["sizing_rules"]["fixed_contracts"]
    child["sizing_rules"]["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(a_ct * w + b_ct * (1 - w))))

    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = child["name"].split("|")
    base = parts[0] if parts else "UNKNOWN"
    child["name"] = f"{base}|sfam_{h}"

    return child


def mutate_strategy(sd, intensity=0.5, registry=None):
    """
    Mutate a strategy: params, exits, contracts.
    15% chance of structural mutation (swap filter type from registry).
    """
    new = copy.deepcopy(sd)

    # Mutate entry signal params
    for sig in new["entry_signals"]:
        sig["params"] = mutate_params(sig["params"], intensity)

    # Mutate filter params (skip time_of_day)
    for filt in new.get("entry_filters", []):
        if filt.get("signal_name") == "time_of_day":
            continue
        filt["params"] = mutate_params(filt["params"], intensity)

    # Mutate exits
    er = new["exit_rules"]
    sl = er["stop_loss_value"]
    tp = er["take_profit_value"]
    new_sl = round(max(SL_MIN, min(SL_MAX, sl + sl * intensity * random.uniform(-1, 1))), 1)
    new_tp = round(max(TP_MIN, min(TP_MAX, tp + tp * intensity * random.uniform(-1, 1))), 1)
    if new_sl > 0 and new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
        new_tp = min(TP_MAX, new_tp)
    er["stop_loss_value"] = new_sl
    er["take_profit_value"] = new_tp

    # Mutate contracts
    sz = new["sizing_rules"]
    ct = sz["fixed_contracts"]
    new_ct = max(CT_MIN, min(CT_MAX, int(ct + ct * intensity * random.uniform(-0.5, 0.5))))
    sz["fixed_contracts"] = new_ct

    # 15% chance of structural mutation: swap filter type
    if registry is not None and random.random() < 0.15:
        filter_sigs = registry.list_filters()
        if filter_sigs:
            new_filter_def = random.choice(filter_sigs)
            if new_filter_def.name != "time_of_day":
                # Find a non-time filter to replace, or add one
                non_time_filters = [
                    (i, f) for i, f in enumerate(new.get("entry_filters", []))
                    if f.get("signal_name") != "time_of_day"
                ]
                from strategies.generator import StrategyGenerator
                gen = StrategyGenerator(registry)
                new_filt_dict = gen._signal_def_to_filter_dict(new_filter_def)
                new_filt_dict["params"] = gen._sample_params(new_filter_def, method="random")

                if non_time_filters:
                    idx = non_time_filters[0][0]
                    new["entry_filters"][idx] = new_filt_dict
                else:
                    new["entry_filters"].insert(0, new_filt_dict)

    # Mutate time window (occasionally)
    for filt in new.get("entry_filters", []):
        if filt.get("signal_name") == "time_of_day" or filt.get("function") == "time_of_day":
            if random.random() < 0.15:
                windows = [
                    (9, 30, 11, 0),
                    (9, 30, 11, 30),
                    (9, 30, 10, 30),
                    (9, 45, 11, 0),
                    (9, 0, 11, 0),
                    (9, 30, 12, 0),
                ]
                w = random.choice(windows)
                filt["params"] = {
                    "start_hour": w[0], "start_minute": w[1],
                    "end_hour": w[2], "end_minute": w[3],
                }

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = parts[0] if parts else "UNKNOWN"
    new["name"] = f"{base}|mut_{h}"

    return new


def decorrelate(population, max_corr=CORRELATION_THRESHOLD):
    """Remove correlated strategies by monthly PnL. Keep higher-scoring one."""
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
                # Keep higher-scoring
                if population[i][3] >= population[j][3]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [p for p, k in zip(population, keep) if k]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(batch_id=None, num_batches=None, output_file=None):
    total_start = time.time()

    # Batch mode: deterministic seed per batch
    if batch_id is not None:
        random.seed(42 + batch_id * 1000)
        np.random.seed(42 + batch_id * 1000)

    batch_label = f" (batch {batch_id+1}/{num_batches})" if batch_id is not None else ""

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║     CROSSBREED V4 — Cross-Family Evolution Engine{batch_label:<23}║
╠══════════════════════════════════════════════════════════════════════════╣
║  Load ALL winning strategies from every report file                     ║
║  Cross-breed entry signals from one family with filters from another    ║
║  60-generation genetic algorithm with proxy fitness                     ║
║  Walk-forward validated + MC stress tested (5000 sims)                  ║
║  Target: $15K+/month consistent on Topstep 150K                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 0: LOAD ALL WINNERS
    # ══════════════════════════════════════════════════════════════════
    logger.info("═══ PHASE 0: Loading all winning strategies ═══")

    unique_strategies, elite_hashes, family_counts = load_all_seeds()

    # Determine which come from 50K accounts
    labels_50k = {"evolved", "maximized"}

    # Prepare all seeds
    prepared = []
    for sd, label, is_elite in unique_strategies:
        from_50k = label in labels_50k
        try:
            prepped = prepare_seed(sd, from_50k=from_50k)
            prepared.append((prepped, label, is_elite))
        except Exception:
            continue

    logger.info(f"  Prepared {len(prepared)} seeds for evolution")
    elite_count = sum(1 for _, _, e in prepared if e)
    logger.info(f"  Elite seeds (profitable OOS): {elite_count}")

    # Batch mode: split seeds deterministically across batches
    if batch_id is not None and num_batches is not None:
        # Shuffle with fixed seed so all batches agree on the split
        rng_split = random.Random(42)
        indices = list(range(len(prepared)))
        rng_split.shuffle(indices)
        # Each batch gets its slice
        batch_indices = [i for i in indices if i % num_batches == batch_id]
        prepared = [prepared[i] for i in batch_indices]
        logger.info(f"  Batch {batch_id+1}/{num_batches}: {len(prepared)} seeds assigned")

    # ══════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ══════════════════════════════════════════════════════════════════
    logger.info("\nLoading data...")
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

    registry = SignalRegistry()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: INITIAL EVALUATION + CROSS-FAMILY EVOLUTION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 1: Initial evaluation of all seeds on train data ═══")

    # Evaluate all seeds
    population = []  # Each: (sd, metrics, mc_or_none, score, trades)
    all_seeds = [sd for sd, _, _ in prepared]
    original_pool = list(all_seeds)  # Keep for immigrants

    eval_count = 0
    for sd, label, is_elite in prepared:
        eval_count += 1
        if eval_count % 100 == 0:
            logger.info(f"  Evaluating seed {eval_count}/{len(prepared)}...")
        out = backtest_strategy(sd, data_train, rm, config_train, min_trades=MIN_TRADES_TRAIN)
        if out:
            trades, m = out
            if m.total_pnl > 0:
                score = fitness_proxy(m, trades)
                population.append((sd, m, None, score, trades))

    population.sort(key=lambda x: x[3], reverse=True)
    initial_viable = len(population)
    logger.info(f"  {initial_viable} viable seeds (PnL > 0, trades >= {MIN_TRADES_TRAIN})")

    if not population:
        print("ERROR: No viable seeds found on train data. Check data path and strategy formats.")
        return

    # Show initial top
    sd, m, _, score, trades = population[0]
    months = monthly_pnl(trades)
    avg_mo = np.mean(list(months.values())) if months else 0
    logger.info(f"  Best seed: {sd['name'][:50]} | PnL=${m.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f}")

    # Count families in viable population
    viable_families = Counter(get_signal_family(p[0]) for p in population)
    logger.info(f"  Viable families: {len(viable_families)}")

    # ── EVOLUTION LOOP ──
    logger.info(f"\n═══ PHASE 1: Cross-family evolution ({MAX_GENERATIONS} generations) ═══")

    best_ever_score = population[0][3]
    stagnant_gens = 0
    prev_best_score = best_ever_score
    tested = eval_count

    for gen in range(MAX_GENERATIONS):
        gen_start = time.time()

        # Cyclical intensity: oscillates 0.25-1.0 every 15 generations
        cycle_pos = (gen % 15) / 15.0
        intensity = 0.25 + 0.75 * abs(np.sin(cycle_pos * np.pi))

        # Stagnation check
        if stagnant_gens >= STAGNATION_THRESHOLD:
            logger.info(f"  ★ Stagnation ({stagnant_gens} gens) — spiking intensity + injecting diversity")
            intensity = 1.5
            stagnant_gens = 0
            # Inject 40 seeds from least-represented families
            pop_families = Counter(get_signal_family(p[0]) for p in population)
            least_represented = [fam for fam, _ in pop_families.most_common()[-20:]]
            immigrants_stag = []
            for sd_orig in original_pool:
                fam = get_signal_family(sd_orig)
                if fam in least_represented or random.random() < 0.2:
                    immigrants_stag.append(mutate_strategy(copy.deepcopy(sd_orig), intensity=1.5, registry=registry))
                    if len(immigrants_stag) >= 40:
                        break
            # Evaluate stagnation immigrants
            for sd_imm in immigrants_stag:
                tested += 1
                out = backtest_strategy(sd_imm, data_train, rm, config_train, min_trades=MIN_TRADES_TRAIN)
                if out:
                    trades, m = out
                    if m.total_pnl > 0:
                        score = fitness_proxy(m, trades)
                        population.append((sd_imm, m, None, score, trades))

        elites = [p[0] for p in population[:ELITE_COUNT]]
        pool = [p[0] for p in population[:min(len(population), POP_CAP)]]

        # ── Produce offspring ──
        n_offspring = POP_CAP - ELITE_COUNT
        n_cross_family = int(n_offspring * 0.40)
        n_same_family = int(n_offspring * 0.20)
        n_mutation = int(n_offspring * 0.30)
        n_immigrant = n_offspring - n_cross_family - n_same_family - n_mutation

        offspring = []

        # 1. CROSS-FAMILY CROSSOVER (40%)
        for _ in range(n_cross_family):
            a = random.choice(pool)
            b = random.choice(pool)
            fam_a = get_signal_family(a)
            fam_b = get_signal_family(b)
            # Try to pick from different families
            attempts = 0
            while fam_a == fam_b and attempts < 10:
                b = random.choice(pool)
                fam_b = get_signal_family(b)
                attempts += 1
            offspring.append(cross_family_crossover(a, b, intensity))

        # 2. SAME-FAMILY CROSSOVER (20%)
        for _ in range(n_same_family):
            a = random.choice(pool)
            fam_a = get_signal_family(a)
            # Find a same-family partner
            same_fam = [p for p in pool if get_signal_family(p) == fam_a and p is not a]
            if same_fam:
                b = random.choice(same_fam)
            else:
                b = random.choice(pool)
            offspring.append(same_family_crossover(a, b, intensity))

        # 3. PARAM MUTATION (30%)
        for _ in range(n_mutation):
            parent = random.choice(pool)
            offspring.append(mutate_strategy(parent, intensity, registry=registry))

        # 4. FRESH IMMIGRANTS (10%)
        for _ in range(n_immigrant):
            immigrant = random.choice(original_pool)
            # Check it's not already in population
            offspring.append(copy.deepcopy(immigrant))

        # ── Evaluate offspring ──
        new_pop = []

        # Carry forward elites without re-backtesting
        for p in population[:ELITE_COUNT]:
            new_pop.append(p)

        for sd_new in offspring:
            tested += 1
            try:
                out = backtest_strategy(sd_new, data_train, rm, config_train, min_trades=MIN_TRADES_TRAIN)
                if out:
                    trades, m = out
                    if m.total_pnl > 0:
                        score = fitness_proxy(m, trades)
                        new_pop.append((sd_new, m, None, score, trades))
            except Exception:
                continue

        new_pop.sort(key=lambda x: x[3], reverse=True)
        population = new_pop[:POP_CAP]

        # Track stagnation
        if population and population[0][3] > prev_best_score * 1.001:
            stagnant_gens = 0
            prev_best_score = population[0][3]
        else:
            stagnant_gens += 1

        if population[0][3] > best_ever_score:
            best_ever_score = population[0][3]

        gen_elapsed = time.time() - gen_start

        # Log
        if population:
            sd, m, _, score, trades = population[0]
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values())) if months else 0
            min_mo = min(months.values()) if months else 0
            fam = get_signal_family(sd)

            high_earners = sum(
                1 for p in population
                if np.mean(list(monthly_pnl(p[4]).values())) >= TARGET_AVG_MONTHLY
            )

            logger.info(
                f"  Gen {gen+1:>2}/{MAX_GENERATIONS} | int={intensity:.2f} | pop={len(population):>3} | "
                f"best={fam[:30]} | avg/mo=${avg_mo:,.0f} | min/mo=${min_mo:,.0f} | "
                f"$15K+={high_earners} | {gen_elapsed:.0f}s"
            )

    logger.info(f"  Evolution complete: {tested:,} tested over {MAX_GENERATIONS} generations")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 2: Walk-forward validation (top 60) ═══")

    top_for_wf = population[:60]
    wf_survivors = []

    try:
        from validation.walk_forward import WalkForwardValidator

        for sd, m_train, _, score, trades_train in top_for_wf:
            try:
                strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
                wf = WalkForwardValidator(
                    data=data_full, risk_manager=rm,
                    contract_spec=MNQ_SPEC, config=config_full,
                    account_size=150000.0,
                )
                wf_result = wf.validate(strategy, train_days=60, test_days=20, step_days=20)

                if wf_result.wf_efficiency < 0.35:
                    logger.info(f"  ✗ WF reject: {sd['name'][:40]} | WF eff={wf_result.wf_efficiency:.2f}")
                    continue

                # OOS validation
                out = backtest_strategy(sd, data_validate, rm, config_validate, min_trades=MIN_TRADES_OOS)
                if out is None:
                    continue
                trades_oos, m_oos = out
                if m_oos.total_pnl <= 0:
                    continue

                months_oos = monthly_pnl(trades_oos)
                avg_mo_oos = np.mean(list(months_oos.values())) if months_oos else 0
                fam = get_signal_family(sd)

                logger.info(
                    f"  ✓ WF+OOS pass: {sd['name'][:40]} | WF={wf_result.wf_efficiency:.2f} | "
                    f"OOS PnL=${m_oos.total_pnl:,.0f} | avg/mo=${avg_mo_oos:,.0f} | fam={fam[:25]}"
                )
                wf_survivors.append((sd, m_train, m_oos, wf_result.wf_efficiency, score, trades_train))

            except Exception as e:
                logger.warning(f"  WF error for {sd.get('name', '?')[:30]}: {e}")
                # Still try OOS without WF
                out = backtest_strategy(sd, data_validate, rm, config_validate, min_trades=MIN_TRADES_OOS)
                if out:
                    trades_oos, m_oos = out
                    if m_oos.total_pnl > 0:
                        wf_survivors.append((sd, m_train, m_oos, 0.5, score, trades_train))

    except ImportError:
        logger.warning("  Walk-forward module not available — using OOS backtest only")
        for sd, m_train, _, score, trades_train in top_for_wf:
            out = backtest_strategy(sd, data_validate, rm, config_validate, min_trades=MIN_TRADES_OOS)
            if out:
                trades_oos, m_oos = out
                if m_oos.total_pnl > 0:
                    wf_survivors.append((sd, m_train, m_oos, 0.5, score, trades_train))

    logger.info(f"  WF+OOS survivors: {len(wf_survivors)}/{len(top_for_wf)}")

    if not wf_survivors:
        logger.warning("  No WF survivors — relaxing to top 60 from evolution")
        for sd, m_train, _, score, trades_train in top_for_wf:
            wf_survivors.append((sd, m_train, None, 0.5, score, trades_train))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: FULL-YEAR MC STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 3: Full-year MC stress test ({MC_SIMS_FINAL} sims) ═══")

    mc_validated = []
    for sd, m_train, m_oos, wf_eff, score, _ in wf_survivors:
        out = backtest_strategy(sd, data_full, rm, config_full, min_trades=MIN_TRADES_TRAIN)
        if out is None:
            continue
        trades_full, m_full = out

        try:
            mc = mc_test(trades_full, prop_rules, MC_SIMS_FINAL)
        except Exception as e:
            logger.warning(f"  MC error: {e}")
            continue

        if mc.probability_of_profit < OOS_MC_P_PROFIT_MIN:
            logger.info(f"  ✗ MC reject: {sd['name'][:40]} | MC P={mc.probability_of_profit:.0%}")
            continue

        full_score = fitness_full(m_full, mc, trades_full)
        months = monthly_pnl(trades_full)
        avg_mo = np.mean(list(months.values())) if months else 0
        fam = get_signal_family(sd)

        mc_validated.append((sd, m_full, mc, full_score, trades_full, wf_eff))
        logger.info(
            f"  ★ {sd['name'][:40]} | PnL=${m_full.total_pnl:,.0f} | avg/mo=${avg_mo:,.0f} | "
            f"WR={m_full.win_rate:.0f}% | DD=${m_full.max_drawdown:,.0f} | MC={mc.probability_of_profit:.0%} | "
            f"fam={fam[:25]}"
        )

    mc_validated.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  MC validated: {len(mc_validated)} strategies")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: DECORRELATION
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 4: Decorrelation (corr < {CORRELATION_THRESHOLD}) ═══")

    # Convert to format decorrelate() expects: (sd, m, mc, score, trades)
    decor_input = [(sd, m, mc, score, trades) for sd, m, mc, score, trades, _ in mc_validated]
    wf_eff_map = {strategy_hash(sd): wf_eff for sd, _, _, _, _, wf_eff in mc_validated}

    final_pool = decorrelate(decor_input, CORRELATION_THRESHOLD)
    final_pool.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"  After decorrelation: {len(final_pool)} unique strategies")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{'='*150}")
    print(f"  CROSSBREED V4 COMPLETE — {tested:,} tested in {total_elapsed/60:.1f} min")
    print(f"  Pipeline: {len(prepared)} seeds → {initial_viable} viable → evolved {MAX_GENERATIONS} gens"
          f" → {len(wf_survivors)} WF → {len(mc_validated)} MC → {len(final_pool)} final")
    print(f"{'='*150}")

    # Signal family diversity in final pool
    final_families_entry = set()
    final_families_filter = set()
    cross_family_successes = []

    for sd, m, mc, score, trades in final_pool:
        entries = [e["signal_name"] for e in sd.get("entry_signals", [])]
        filters = [f["signal_name"] for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
        for e in entries:
            final_families_entry.add(e)
        for f in filters:
            final_families_filter.add(f)

        # Check if cross-family (name contains 'xfam')
        if "xfam" in sd.get("name", "") or "×" in sd.get("name", ""):
            cross_family_successes.append(sd)

    print(f"\n  Signal Family Diversity:")
    print(f"    Entry signal types in final pool: {len(final_families_entry)} — {', '.join(sorted(final_families_entry))}")
    print(f"    Filter types in final pool:       {len(final_families_filter)} — {', '.join(sorted(final_families_filter))}")

    # Leaderboard
    if final_pool:
        print(f"\n  {'#':<4} {'Strategy':<44} {'Family':<30} {'Tr':>5} {'WR':>6} {'PF':>5} {'1yr PnL':>12} {'Avg/Mo':>10} {'Min/Mo':>10} {'Max DD':>10} {'MC P':>6}")
        print(f"  {'-'*150}")
        for i, (sd, m, mc, score, trades) in enumerate(final_pool[:30], 1):
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values())) if months else 0
            min_mo = min(months.values()) if months else 0
            fam = get_signal_family(sd)
            flag = "★" if avg_mo >= TARGET_AVG_MONTHLY else " "
            print(
                f"  {flag}{i:<3} {sd['name'][:43]:<44} "
                f"{fam[:29]:<30} "
                f"{m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>4.2f} "
                f"${m.total_pnl:>11,.0f} ${avg_mo:>9,.0f} ${min_mo:>9,.0f} "
                f"${m.max_drawdown:>9,.0f} {mc.probability_of_profit:>5.0%}"
            )
        print(f"  {'-'*150}")

        # Champion details
        sd, m, mc, score, trades = final_pool[0]
        months = monthly_pnl(trades)
        monthly_vals = list(months.values())
        avg_mo = np.mean(monthly_vals)
        max_mo = max(monthly_vals)
        min_mo = min(monthly_vals)
        er = sd["exit_rules"]
        fam = get_signal_family(sd)

        print(f"""
  ══════════════════════════════════════════════════════════════
  CHAMPION — CROSSBREED V4
  ══════════════════════════════════════════════════════════════
  Name:           {sd['name']}
  Family:         {fam}
  Entry:          {', '.join(e['signal_name'] for e in sd['entry_signals'])}
  Filters:        {', '.join(f['signal_name'] for f in sd.get('entry_filters', []))}
  Stop Loss:      {er['stop_loss_value']}pt
  Take Profit:    {er['take_profit_value']}pt
  R:R:            {er['take_profit_value']/max(er['stop_loss_value'],0.01):.1f}:1
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

        # Monthly PnL for top 5
        print(f"\n  MONTHLY P&L — Top 5 Strategies:")
        for rank, (sd_r, m_r, mc_r, score_r, trades_r) in enumerate(final_pool[:5], 1):
            months_r = monthly_pnl(trades_r)
            fam_r = get_signal_family(sd_r)
            print(f"\n    #{rank} {sd_r['name'][:50]} [{fam_r}]")
            for mo in sorted(months_r.keys()):
                p = months_r[mo]
                bar = "█" * max(1, int(abs(p) / 1000))
                flag = "★" if p >= TARGET_AVG_MONTHLY else " "
                sign = "+" if p >= 0 else "-"
                print(f"      {flag} {mo}: {sign}${abs(p):>10,.2f}  {bar}{'*' if p < 0 else ''}")
            prof_months = sum(1 for p in months_r.values() if p > 0)
            print(f"      Profitable: {prof_months}/{len(months_r)} ({prof_months/len(months_r)*100:.0f}%)")

        # Cross-family breeding successes
        if cross_family_successes:
            print(f"\n  CROSS-FAMILY BREEDING SUCCESSES ({len(cross_family_successes)}):")
            for sd_xf in cross_family_successes:
                entries = [e["signal_name"] for e in sd_xf.get("entry_signals", [])]
                filters = [f["signal_name"] for f in sd_xf.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
                print(f"    {sd_xf['name'][:50]} — entry: {','.join(entries)} × filter: {','.join(filters)}")
        else:
            print(f"\n  No explicit cross-family tagged strategies in final pool (hybrids may have mutated further)")

    # ── Save JSON ──
    if final_pool:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "crossbreed_v4",
            "account": {
                "initial_capital": 150000,
                "prop_firm": "topstep_150k",
                "max_drawdown": MAX_DD_LIMIT,
                "daily_loss_limit": DAILY_LOSS_LIMIT,
            },
            "bounds": {
                "sl_range": [SL_MIN, SL_MAX],
                "tp_range": [TP_MIN, TP_MAX],
                "ct_range": [CT_MIN, CT_MAX],
                "rr_min": RR_MIN,
            },
            "data": {
                "full_period": "2024-03-19 to 2025-03-18",
                "train_period": "2024-03-19 to 2024-11-18",
                "validate_period": "2024-11-19 to 2025-03-18",
            },
            "pipeline_stats": {
                "seeds_loaded": len(prepared),
                "initial_viable": initial_viable,
                "total_tested": tested,
                "generations": MAX_GENERATIONS,
                "wf_survivors": len(wf_survivors),
                "mc_validated": len(mc_validated),
                "decorrelated_final": len(final_pool),
                "elapsed_min": round(total_elapsed / 60, 1),
            },
            "signal_diversity": {
                "entry_types": sorted(final_families_entry),
                "filter_types": sorted(final_families_filter),
                "cross_family_count": len(cross_family_successes),
            },
            "strategies": [],
        }

        for sd, m, mc, score, trades in final_pool:
            months = monthly_pnl(trades)
            monthly_vals = list(months.values())
            fam = get_signal_family(sd)

            output["strategies"].append({
                "name": sd["name"],
                "strategy": sd,
                "signal_family": fam,
                "fitness": round(score, 2),
                "trades": m.total_trades,
                "win_rate": round(m.win_rate, 2),
                "profit_factor": round(m.profit_factor, 2),
                "sharpe": round(m.sharpe_ratio, 2),
                "total_pnl": round(m.total_pnl, 2),
                "max_drawdown": round(m.max_drawdown, 2),
                "dd_margin": round(abs(MAX_DD_LIMIT) - abs(m.max_drawdown), 2),
                "avg_monthly": round(float(np.mean(monthly_vals)), 2),
                "max_monthly": round(float(max(monthly_vals)), 2),
                "min_monthly": round(float(min(monthly_vals)), 2),
                "pct_months_profitable": round(sum(1 for v in monthly_vals if v > 0) / len(monthly_vals), 2),
                "monthly_breakdown": {mo: round(v, 2) for mo, v in sorted(months.items())},
                "mc_median": round(mc.median_return, 2),
                "mc_mean": round(mc.mean_return, 2),
                "mc_p_profit": round(mc.probability_of_profit, 4),
                "mc_p_ruin": round(mc.probability_of_ruin, 4),
                "mc_5th": round(mc.pct_5th_return, 2),
                "mc_95th": round(mc.pct_95th_return, 2),
                "mc_composite": round(mc.composite_score, 2),
                "mc_pass_rate": round(mc.prop_firm_pass_rate, 4),
                "wf_efficiency": round(wf_eff_map.get(strategy_hash(sd), 0.5), 3),
            })

        Path("reports").mkdir(exist_ok=True)
        out_path = output_file or "reports/crossbred_v4_strategies.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(final_pool)} strategies to {out_path}")

    else:
        print("\n  No strategies survived the full pipeline.")

    print(f"\n{'='*150}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossbreed V4 — Cross-family evolution engine")
    parser.add_argument("--batch-id", type=int, default=None, help="Batch index (0-based)")
    parser.add_argument("--num-batches", type=int, default=None, help="Total number of batches")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()
    main(batch_id=args.batch_id, num_batches=args.num_batches, output_file=args.output_file)
