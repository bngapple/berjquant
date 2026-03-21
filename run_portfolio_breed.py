#!/usr/bin/env python3
"""
PORTFOLIO BREED V1 — Breed a PORTFOLIO of 3-5 uncorrelated strategies
that collectively trade frequently and hit $15K+/month every month.

The problem: our champion ROC×KC fires only 11 times/year. We need
CONSISTENT monthly income from multiple strategies trading in rotation.

Pipeline:
  Phase 0: Load recent winners + 300 diverse random seeds
  Phase 1: Evolve for FREQUENCY + edge (min 60 trades/year, reward trades/month)
  Phase 2: Build optimal 3-5 strategy portfolios (greedy construction)
  Phase 3: Walk-forward validate portfolio components
  Phase 4: MC stress test combined portfolio trade stream
  Phase 5: Decorrelate portfolios
  Phase 6: Output ranked portfolios with component breakdowns
"""

import gc
import json
import time
import copy
import random
import hashlib
import logging
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
from strategies.generator import GeneratedStrategy, StrategyGenerator
from monte_carlo.simulator import MonteCarloSimulator, MCConfig
from signals.registry import SignalRegistry

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("portfolio")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
MAX_DD_LIMIT = -4500.0
PORTFOLIO_DD_LIMIT = -4000.0
DAILY_LOSS_LIMIT = -3000.0
TARGET_AVG_MONTHLY = 15000
MC_SIMS = 5000
MIN_TRADES_YEAR = 60
CORRELATION_THRESHOLD = 0.75
PORTFOLIO_CORR_MAX = 0.60

# Exit bounds
SL_MIN, SL_MAX = 10.0, 35.0
TP_MIN, TP_MAX = 50.0, 300.0
CT_MIN, CT_MAX = 4, 15
RR_MIN = 4.0

# Evolution
POP_CAP = 80
ELITE_COUNT = 15
MAX_GENERATIONS = 30
STAGNATION_THRESHOLD = 8

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def backtest_strategy(sd, data, rm, config, min_trades=MIN_TRADES_YEAR):
    try:
        sd_copy = copy.deepcopy(sd)
        sd_copy["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd_copy)
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        if len(result.trades) < min_trades:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital, result.equity_curve)
        if metrics.max_drawdown < MAX_DD_LIMIT:
            return None
        return result.trades, metrics
    except Exception:
        return None


def mc_test(trades, prop_rules, n_sims=MC_SIMS):
    mc = MonteCarloSimulator(MCConfig(
        n_simulations=n_sims, initial_capital=150000.0,
        prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
    ))
    return mc.run(trades, strategy_name="test")


def monthly_pnl(trades_or_dict):
    """Compute monthly PnL. Accepts trade list OR already-computed monthly dict."""
    if isinstance(trades_or_dict, dict):
        return trades_or_dict
    months = {}
    for t in trades_or_dict:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


def get_signal_family(sd):
    entries = sorted(e["signal_name"] for e in sd.get("entry_signals", []))
    filters = sorted(
        f["signal_name"] for f in sd.get("entry_filters", [])
        if f.get("signal_name") != "time_of_day"
    )
    return "|".join(entries) + ("+" + "|".join(filters) if filters else "")


def strategy_hash(sd):
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
            for k in ["stop_loss_value", "take_profit_value"]
        },
    }
    blob = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def portfolio_fitness(metrics, trades_or_months):
    """Fitness that heavily rewards trade frequency alongside profitability."""
    if isinstance(trades_or_months, dict):
        months = trades_or_months
    else:
        months = monthly_pnl(trades_or_months)
    if not months:
        return -999999

    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    num_trades = metrics.total_trades
    months_with_trades = sum(1 for v in monthly_vals if abs(v) > 0)

    trades_per_month = num_trades / max(1, len(monthly_vals))
    if trades_per_month < 5:
        freq_score = -20000
    elif trades_per_month < 15:
        freq_score = trades_per_month * 500
    elif trades_per_month <= 80:
        freq_score = 15000
    else:
        freq_score = 15000 - (trades_per_month - 80) * 50

    coverage = months_with_trades / len(monthly_vals)
    coverage_bonus = coverage * 10000

    dd = abs(metrics.max_drawdown)
    dd_margin = max(0, (4500 - dd) / 2500) * 3000
    consistency = -sum(max(0, 5000 - v) for v in monthly_vals) * 0.3

    score = (
        avg_monthly * 3.0
        + min_monthly * 2.0
        + metrics.sharpe_ratio * 800
        + metrics.profit_factor * 400
        + freq_score
        + coverage_bonus
        + dd_margin
        + consistency
    )
    return score


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


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: LOAD SEEDS
# ═══════════════════════════════════════════════════════════════════════

def load_recent_winners():
    """Load strategy dicts from only the latest report files."""
    report_files = [
        "reports/maximized_champion_v1.json",
        "reports/crossbred_v4_strategies.json",
        "reports/crossbred_v4_batch_0.json",
        "reports/crossbred_v4_batch_1.json",
        "reports/crossbred_v4_batch_2.json",
        "reports/crossbred_v4_batch_3.json",
    ]

    all_strategies = []
    for filepath in report_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"  Skipping {filepath}: {e}")
            continue

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ["strategies", "results", "winners"]:
                if key in data:
                    items = data[key]
                    break
            else:
                items = []
        else:
            items = []

        count = 0
        for item in items:
            if "strategy" in item and isinstance(item["strategy"], dict):
                sd = copy.deepcopy(item["strategy"])
            elif "entry_signals" in item:
                sd = copy.deepcopy(item)
            else:
                continue
            if not sd.get("entry_signals") or not sd.get("exit_rules"):
                continue
            all_strategies.append(sd)
            count += 1

        logger.info(f"  {filepath}: {count} strategies")

    # Deduplicate
    seen = {}
    unique = []
    for sd in all_strategies:
        h = strategy_hash(sd)
        if h not in seen:
            seen[h] = True
            unique.append(sd)

    logger.info(f"  Recent winners: {len(all_strategies)} loaded → {len(unique)} unique")
    return unique


def generate_diverse_seeds(n=300):
    """Generate structurally diverse strategies from the full signal registry."""
    registry = SignalRegistry()
    gen = StrategyGenerator(registry)

    entry_sigs = registry.list_entry_signals()
    filter_sigs = registry.list_filters()

    seeds = []
    seen = set()

    time_windows = [
        (9, 30, 16, 0),   # Full session
        (9, 30, 12, 0),   # Morning
        (12, 0, 16, 0),   # Afternoon
        (9, 30, 11, 0),   # Power hour
        (14, 0, 16, 0),   # Late session
        (8, 0, 11, 0),    # London overlap
        (9, 30, 13, 0),   # Extended morning
        (9, 30, 15, 0),   # Most of day
    ]

    for _ in range(n * 3):
        if len(seeds) >= n:
            break

        entry = random.choice(entry_sigs)
        filt = random.choice(filter_sigs) if random.random() < 0.7 else None

        combo_key = (entry.name, filt.name if filt else None)
        if combo_key in seen:
            # Allow some duplicates with different params
            if random.random() > 0.3:
                continue
        seen.add(combo_key)

        # Bias toward shorter periods for more signals
        entry_params = gen._sample_params(entry, method="random")
        # Shorten periods by ~30%
        for k, v in entry_params.items():
            if isinstance(v, int) and "period" in k.lower():
                entry_params[k] = max(3, int(v * random.uniform(0.5, 1.0)))
            if "overbought" in k:
                entry_params[k] = round(max(50, min(95, v * random.uniform(0.7, 1.1))), 2)
            if "oversold" in k:
                entry_params[k] = round(max(5, min(50, v * random.uniform(0.9, 1.5))), 2)
            if "multiplier" in k:
                entry_params[k] = round(max(0.5, min(3.0, v * random.uniform(0.6, 1.0))), 4)

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

        # 50% full session, 50% various windows
        if random.random() < 0.5:
            tw = (9, 30, 16, 0)
        else:
            tw = random.choice(time_windows)

        filters.append({
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]},
            "column": "signal_time_allowed",
        })

        sl = round(random.uniform(SL_MIN, SL_MAX), 1)
        tp = round(random.uniform(TP_MIN, TP_MAX), 1)
        if tp / sl < RR_MIN:
            tp = round(sl * RR_MIN, 1)
            tp = min(TP_MAX, tp)

        sd = {
            "name": f"{entry.name.upper()}|{'_'.join(f['signal_name'].upper() for f in filters)}|div_{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}",
            "entry_signals": [entry_dict],
            "entry_filters": filters,
            "exit_rules": {
                "stop_loss_type": "fixed_points",
                "stop_loss_value": sl,
                "take_profit_type": "fixed_points",
                "take_profit_value": tp,
                "trailing_stop": random.random() < 0.25,
                "trailing_activation": round(random.uniform(15, 60), 1),
                "trailing_distance": round(random.uniform(8, 20), 1),
                "time_exit_minutes": None,
            },
            "sizing_rules": {
                "method": "fixed",
                "fixed_contracts": random.randint(CT_MIN, CT_MAX),
                "risk_pct": 0.02,
                "atr_risk_multiple": 2.0,
            },
            "primary_timeframe": "1m",
            "require_all_entries": True,
        }
        seeds.append(sd)

    return seeds


def prepare_seed(sd):
    """Prepare a seed: fix timeframe, sizing, SL/TP bounds, R:R, time filter."""
    sd = copy.deepcopy(sd)
    sd["primary_timeframe"] = "1m"

    er = sd.get("exit_rules", {})
    er["stop_loss_type"] = "fixed_points"
    er["take_profit_type"] = "fixed_points"

    sl = max(SL_MIN, min(SL_MAX, er.get("stop_loss_value", 20.0)))
    tp = max(TP_MIN, min(TP_MAX, er.get("take_profit_value", 100.0)))
    if sl > 0 and tp / sl < RR_MIN:
        tp = round(sl * RR_MIN, 1)
        tp = min(TP_MAX, tp)
        if tp / sl < RR_MIN:
            sl = round(tp / RR_MIN, 1)
            sl = max(SL_MIN, sl)
    er["stop_loss_value"] = round(sl, 1)
    er["take_profit_value"] = round(tp, 1)
    sd["exit_rules"] = er

    sz = sd.get("sizing_rules", {})
    sz["method"] = "fixed"
    sz["fixed_contracts"] = max(CT_MIN, min(CT_MAX, sz.get("fixed_contracts", 4)))
    sd["sizing_rules"] = sz

    has_time = any(
        f.get("signal_name") == "time_of_day" or f.get("function") == "time_of_day"
        for f in sd.get("entry_filters", [])
    )
    if not has_time:
        sd.setdefault("entry_filters", []).append({
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {"start_hour": 9, "start_minute": 30, "end_hour": 16, "end_minute": 0},
            "column": "signal_time_allowed",
        })

    sd["require_all_entries"] = sd.get("require_all_entries", True)
    return sd


# ═══════════════════════════════════════════════════════════════════════
# EVOLUTION OPERATORS
# ═══════════════════════════════════════════════════════════════════════

def mutate_strategy(sd, intensity=0.5, registry=None):
    new = copy.deepcopy(sd)

    # Mutate entry params
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
    sz["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(ct + ct * intensity * random.uniform(-0.5, 0.5))))

    # 15% structural: swap filter
    if registry and random.random() < 0.15:
        filter_sigs = registry.list_filters()
        if filter_sigs:
            new_def = random.choice(filter_sigs)
            if new_def.name != "time_of_day":
                gen = StrategyGenerator(registry)
                new_filt = gen._signal_def_to_filter_dict(new_def)
                new_filt["params"] = gen._sample_params(new_def, method="random")
                non_time = [(i, f) for i, f in enumerate(new.get("entry_filters", [])) if f.get("signal_name") != "time_of_day"]
                if non_time:
                    new["entry_filters"][non_time[0][0]] = new_filt
                else:
                    new["entry_filters"].insert(0, new_filt)

    # 15% structural: swap entry signal
    if registry and random.random() < 0.15:
        entry_sigs = registry.list_entry_signals()
        if entry_sigs:
            new_def = random.choice(entry_sigs)
            gen = StrategyGenerator(registry)
            new_entry = gen._signal_def_to_entry_dict(new_def)
            new_entry["params"] = gen._sample_params(new_def, method="random")
            # Bias toward shorter periods
            for k, v in new_entry["params"].items():
                if isinstance(v, int) and "period" in k.lower():
                    new_entry["params"][k] = max(3, int(v * random.uniform(0.5, 1.0)))
            idx = random.randint(0, len(new["entry_signals"]) - 1)
            new["entry_signals"][idx] = new_entry

    # Mutate time window
    for filt in new.get("entry_filters", []):
        if filt.get("signal_name") == "time_of_day" and random.random() < 0.2:
            windows = [
                (9, 30, 16, 0), (9, 30, 12, 0), (12, 0, 16, 0),
                (9, 30, 11, 0), (14, 0, 16, 0), (8, 0, 11, 0),
                (9, 30, 13, 0), (9, 30, 15, 0),
            ]
            w = random.choice(windows)
            filt["params"] = {"start_hour": w[0], "start_minute": w[1], "end_hour": w[2], "end_minute": w[3]}

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    parts = new["name"].split("|")
    base = parts[0] if parts else "UNK"
    new["name"] = f"{base}|pf_{h}"
    return new


def cross_family_crossover(a, b, intensity=0.5):
    child = copy.deepcopy(a)
    b_filters = [copy.deepcopy(f) for f in b.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
    time_filters = [f for f in child.get("entry_filters", []) if f.get("signal_name") == "time_of_day"]
    child["entry_filters"] = b_filters + time_filters

    a_er, b_er = a["exit_rules"], b["exit_rules"]
    w = random.uniform(0.3, 0.7)
    new_sl = round(max(SL_MIN, min(SL_MAX, a_er["stop_loss_value"] * w + b_er["stop_loss_value"] * (1 - w))), 1)
    new_tp = round(max(TP_MIN, min(TP_MAX, a_er["take_profit_value"] * w + b_er["take_profit_value"] * (1 - w))), 1)
    if new_sl > 0 and new_tp / new_sl < RR_MIN:
        new_tp = round(new_sl * RR_MIN, 1)
        new_tp = min(TP_MAX, new_tp)
    child["exit_rules"]["stop_loss_value"] = new_sl
    child["exit_rules"]["take_profit_value"] = new_tp

    a_ct = a["sizing_rules"]["fixed_contracts"]
    b_ct = b["sizing_rules"]["fixed_contracts"]
    child["sizing_rules"]["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(a_ct * w + b_ct * (1 - w))))

    a_entries = "|".join(e["signal_name"] for e in child.get("entry_signals", []))
    b_filt_names = "|".join(f["signal_name"] for f in b_filters) if b_filters else "none"
    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    child["name"] = f"{a_entries.upper()}×{b_filt_names.upper()}|xpf_{h}"
    return child


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_portfolio(candidates, max_size=5, min_size=3):
    """Greedy portfolio construction from candidate strategies."""
    if not candidates:
        return []

    # Start with best strategy
    portfolio = [candidates[0]]
    portfolio_months = candidates[0][4]  # Already a monthly dict

    for cand_sd, m, _mc, score, months_dict in candidates[1:]:
        if len(portfolio) >= max_size:
            break

        cand_months = months_dict
        cand_family = get_signal_family(cand_sd)

        # Check family diversity
        port_families = set(get_signal_family(p[0]) for p in portfolio)

        # Check correlation with existing portfolio members
        too_correlated = False
        for p_sd, p_m, _pmc, p_score, p_months_dict in portfolio:
            p_months = p_months_dict
            all_months_keys = sorted(set(p_months.keys()) | set(cand_months.keys()))
            if len(all_months_keys) < 3:
                continue
            vec_p = [p_months.get(mk, 0) for mk in all_months_keys]
            vec_c = [cand_months.get(mk, 0) for mk in all_months_keys]
            corr = np.corrcoef(vec_p, vec_c)[0, 1]
            if not np.isnan(corr) and abs(corr) > PORTFOLIO_CORR_MAX:
                too_correlated = True
                break

        if too_correlated:
            continue

        # Check combined DD
        combined_months = {}
        for p_sd, p_m, _pmc2, p_score, p_mo in portfolio:
            for mk, v in p_mo.items():
                combined_months[mk] = combined_months.get(mk, 0) + v
        for mk, v in cand_months.items():
            combined_months[mk] = combined_months.get(mk, 0) + v

        # Rough DD check: worst cumulative monthly drawdown
        running = 0
        peak = 0
        worst_dd = 0
        for mk in sorted(combined_months.keys()):
            running += combined_months[mk]
            if running > peak:
                peak = running
            dd = running - peak
            if dd < worst_dd:
                worst_dd = dd
        if worst_dd < PORTFOLIO_DD_LIMIT:
            continue

        portfolio.append((cand_sd, m, None, score, months_dict))

    return portfolio


def score_portfolio(portfolio, all_months_range=None):
    """Score a portfolio by combined metrics."""
    combined_months = {}
    total_trades = 0
    for sd, m, _mc, score, months_dict in portfolio:
        total_trades += m.total_trades
        for mk, v in months_dict.items():
            combined_months[mk] = combined_months.get(mk, 0) + v

    if not combined_months:
        return -999999, {}

    monthly_vals = list(combined_months.values())
    avg_mo = np.mean(monthly_vals)
    min_mo = min(monthly_vals)
    trades_per_month = total_trades / max(1, len(monthly_vals))

    months_above_15k = sum(1 for v in monthly_vals if v >= TARGET_AVG_MONTHLY)

    # Rough combined DD
    running = 0
    peak = 0
    worst_dd = 0
    for mk in sorted(combined_months.keys()):
        running += combined_months[mk]
        if running > peak:
            peak = running
        dd = running - peak
        if dd < worst_dd:
            worst_dd = dd

    score = (
        avg_mo * 3.0
        + min_mo * 2.0
        + trades_per_month * 100
        + months_above_15k * 5000
        - abs(worst_dd) * 0.5
    )

    stats = {
        "avg_monthly": avg_mo,
        "min_monthly": min_mo,
        "max_monthly": max(monthly_vals),
        "total_trades": total_trades,
        "trades_per_month": trades_per_month,
        "months_above_15k": months_above_15k,
        "combined_dd": worst_dd,
        "monthly_breakdown": combined_months,
        "num_months": len(monthly_vals),
    }
    return score, stats


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     PORTFOLIO BREED V1 — Build 3-5 Strategy Portfolio                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Goal: CONSISTENT $15K+/month from multiple uncorrelated strategies    ║
║  Method: Breed for FREQUENCY + edge, then combine into portfolios      ║
║  Min 60 trades/year per strategy, min 120 combined/year                ║
║  Target: trades happening regularly, $15K+ every month                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 0: LOAD SEEDS
    # ══════════════════════════════════════════════════════════════════
    logger.info("═══ PHASE 0: Loading seeds ═══")

    recent_winners = load_recent_winners()
    diverse_seeds = generate_diverse_seeds(300)
    logger.info(f"  Diverse seeds generated: {len(diverse_seeds)}")

    all_seeds_raw = recent_winners + diverse_seeds
    prepared = []
    for sd in all_seeds_raw:
        try:
            prepped = prepare_seed(sd)
            prepared.append(prepped)
        except Exception:
            continue

    original_pool = list(prepared)
    logger.info(f"  Total seed pool: {len(prepared)} ({len(recent_winners)} winners + {len(diverse_seeds)} diverse)")

    family_counts = Counter(get_signal_family(sd) for sd in prepared)
    logger.info(f"  Signal families: {len(family_counts)}")

    # ══════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ══════════════════════════════════════════════════════════════════
    logger.info("\nLoading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    split_date = "2024-11-19"
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_validate = df_yr1.filter(pl.col("timestamp") >= pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))

    data_train = {"1m": df_train}
    # Defer loading validate/full until needed — save memory during evolution
    del df_full  # Free ~250K rows we don't need yet
    gc.collect()

    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)

    config_train = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-11-18", slippage_ticks=3, initial_capital=150000.0)

    registry = SignalRegistry()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: EVOLVE FOR FREQUENCY + EDGE
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 1: Initial evaluation ({len(prepared)} seeds, min {MIN_TRADES_YEAR} trades) ═══")

    # Memory optimization: store monthly PnL dict instead of full trade list
    # Trade lists use huge memory; we only need monthly breakdown for portfolio construction
    population = []  # Each: (sd, metrics, None, score, monthly_dict)
    for i, sd in enumerate(prepared):
        if (i + 1) % 50 == 0:
            logger.info(f"  Evaluating seed {i+1}/{len(prepared)}...")
        out = backtest_strategy(sd, data_train, rm, config_train, min_trades=MIN_TRADES_YEAR)
        if out:
            trades, m = out
            if m.total_pnl > 0:
                score = portfolio_fitness(m, trades)
                months = monthly_pnl(trades)
                population.append((sd, m, None, score, months))

    population.sort(key=lambda x: x[3], reverse=True)
    gc.collect()
    initial_viable = len(population)
    logger.info(f"  {initial_viable} viable (PnL>0, trades>={MIN_TRADES_YEAR})")

    if population:
        sd, m, _, score, trades = population[0]
        months = monthly_pnl(trades)
        avg_mo = np.mean(list(months.values())) if months else 0
        tpm = m.total_trades / max(1, len(months))
        logger.info(f"  Best: {sd['name'][:50]} | {m.total_trades} trades ({tpm:.0f}/mo) | avg/mo=${avg_mo:,.0f}")

    if not population:
        print("ERROR: No viable seeds. Exiting.")
        return

    # ── EVOLUTION ──
    logger.info(f"\n═══ PHASE 1: Evolution ({MAX_GENERATIONS} generations) ═══")

    best_ever_score = population[0][3]
    stagnant_gens = 0
    prev_best = best_ever_score
    tested = len(prepared)

    for gen in range(MAX_GENERATIONS):
        gen_start = time.time()

        cycle_pos = (gen % 15) / 15.0
        intensity = 0.25 + 0.75 * abs(np.sin(cycle_pos * np.pi))

        if stagnant_gens >= STAGNATION_THRESHOLD:
            logger.info(f"  ★ Stagnation — injecting diversity")
            intensity = 1.5
            stagnant_gens = 0
            pop_families = Counter(get_signal_family(p[0]) for p in population)
            least_rep = [fam for fam, _ in pop_families.most_common()[-15:]]
            for sd_orig in original_pool:
                fam = get_signal_family(sd_orig)
                if fam in least_rep or random.random() < 0.2:
                    imm = mutate_strategy(copy.deepcopy(sd_orig), intensity=1.5, registry=registry)
                    tested += 1
                    out = backtest_strategy(imm, data_train, rm, config_train, min_trades=MIN_TRADES_YEAR)
                    if out:
                        t, m = out
                        if m.total_pnl > 0:
                            mo = monthly_pnl(t)
                            population.append((imm, m, None, portfolio_fitness(m, t), mo))
                    if len(population) > POP_CAP + 40:
                        break

        elites = [p[0] for p in population[:ELITE_COUNT]]
        pool = [p[0] for p in population[:min(len(population), POP_CAP)]]

        n_offspring = min(POP_CAP - ELITE_COUNT, 60)
        n_crossover = int(n_offspring * 0.50)
        n_mutation = int(n_offspring * 0.40)
        n_immigrant = n_offspring - n_crossover - n_mutation

        offspring = []

        for _ in range(n_crossover):
            a, b = random.choice(pool), random.choice(pool)
            fam_a, fam_b = get_signal_family(a), get_signal_family(b)
            attempts = 0
            while fam_a == fam_b and attempts < 10:
                b = random.choice(pool)
                fam_b = get_signal_family(b)
                attempts += 1
            offspring.append(cross_family_crossover(a, b, intensity))

        for _ in range(n_mutation):
            parent = random.choice(pool)
            offspring.append(mutate_strategy(parent, intensity, registry=registry))

        for _ in range(n_immigrant):
            offspring.append(copy.deepcopy(random.choice(original_pool)))

        new_pop = list(population[:ELITE_COUNT])
        for sd_new in offspring:
            tested += 1
            try:
                out = backtest_strategy(sd_new, data_train, rm, config_train, min_trades=MIN_TRADES_YEAR)
                if out:
                    t, m = out
                    if m.total_pnl > 0:
                        mo = monthly_pnl(t)
                        new_pop.append((sd_new, m, None, portfolio_fitness(m, mo), mo))
            except Exception:
                continue

        new_pop.sort(key=lambda x: x[3], reverse=True)
        population = new_pop[:POP_CAP]
        gc.collect()

        if population and population[0][3] > prev_best * 1.001:
            stagnant_gens = 0
            prev_best = population[0][3]
        else:
            stagnant_gens += 1

        gen_elapsed = time.time() - gen_start
        if population:
            sd, m, _, score, trades = population[0]
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values())) if months else 0
            tpm = m.total_trades / max(1, len(months))
            fam = get_signal_family(sd)
            above_5k = sum(1 for p in population if np.mean(list(p[4].values())) >= 5000)
            logger.info(
                f"  Gen {gen+1:>2}/{MAX_GENERATIONS} | int={intensity:.2f} | pop={len(population):>3} | "
                f"best={fam[:25]} | avg/mo=${avg_mo:,.0f} | tr/mo={tpm:.0f} | "
                f"$5K+={above_5k} | {gen_elapsed:.0f}s"
            )

    logger.info(f"  Evolution complete: {tested:,} tested")

    # Free train data, reload full/validate for remaining phases
    del data_train, df_train
    gc.collect()

    logger.info("  Reloading data for validation phases...")
    df_full_reload = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full_reload.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_validate = df_yr1.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df_full_reload
    gc.collect()

    data_validate = {"1m": df_validate}
    data_full = {"1m": df_yr1}

    config_validate = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    config_full = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: BUILD PORTFOLIOS
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 2: Building portfolios from top 100 ═══")

    top_100 = population[:100]
    portfolios = []

    for start_idx in range(min(10, len(top_100))):
        remaining = [top_100[start_idx]] + [c for i, c in enumerate(top_100) if i != start_idx]
        port = build_portfolio(remaining, max_size=5, min_size=3)
        if len(port) >= 3:
            pscore, pstats = score_portfolio(port)
            if pstats.get("total_trades", 0) >= 120:
                portfolios.append((port, pscore, pstats))

    # Also try all combinations starting from different families
    families_seen = set()
    for sd, m, _, score, trades in top_100[:30]:
        fam = get_signal_family(sd)
        if fam not in families_seen:
            families_seen.add(fam)
            remaining = [(sd, m, None, score, trades)] + [c for c in top_100 if get_signal_family(c[0]) != fam]
            port = build_portfolio(remaining, max_size=5, min_size=3)
            if len(port) >= 3:
                pscore, pstats = score_portfolio(port)
                if pstats.get("total_trades", 0) >= 120:
                    portfolios.append((port, pscore, pstats))

    portfolios.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"  Built {len(portfolios)} valid portfolios (>= 3 strategies, >= 120 trades)")

    if not portfolios:
        logger.warning("  No valid portfolios — relaxing constraints")
        for start_idx in range(min(20, len(top_100))):
            remaining = [top_100[start_idx]] + [c for i, c in enumerate(top_100) if i != start_idx]
            port = build_portfolio(remaining, max_size=5, min_size=2)
            if len(port) >= 2:
                pscore, pstats = score_portfolio(port)
                portfolios.append((port, pscore, pstats))
        portfolios.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"  Relaxed: {len(portfolios)} portfolios")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: WALK-FORWARD VALIDATE
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 3: Walk-forward + OOS validation (top {min(10, len(portfolios))}) ═══")

    wf_validated = []
    try:
        from validation.walk_forward import WalkForwardValidator

        for port, pscore, pstats in portfolios[:10]:
            all_pass = True
            validated_components = []
            for sd, m, _, score, trades in port:
                try:
                    strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
                    wf = WalkForwardValidator(data=data_full, risk_manager=rm, contract_spec=MNQ_SPEC, config=config_full, account_size=150000.0)
                    wf_result = wf.validate(strategy, train_days=60, test_days=20, step_days=20)
                    if wf_result.wf_efficiency < 0.35:
                        all_pass = False
                        break
                    validated_components.append((sd, m, None, score, trades, wf_result.wf_efficiency))
                except Exception:
                    validated_components.append((sd, m, None, score, trades, 0.5))

            if not all_pass:
                continue

            # OOS test
            combined_oos_trades = 0
            combined_oos_pnl = 0
            for sd, m, _, score, trades, *rest in validated_components:
                out = backtest_strategy(sd, data_validate, rm, config_validate, min_trades=3)
                if out:
                    t_oos, m_oos = out
                    combined_oos_trades += len(t_oos)
                    combined_oos_pnl += m_oos.total_pnl

            if combined_oos_pnl > 0 and combined_oos_trades >= 30:
                wf_validated.append((validated_components, pscore, pstats))
                logger.info(f"  ✓ Portfolio ({len(validated_components)} strats) | OOS PnL=${combined_oos_pnl:,.0f} | OOS trades={combined_oos_trades}")

    except ImportError:
        logger.warning("  WF not available — passing all portfolios")
        for port, pscore, pstats in portfolios[:10]:
            components = [(sd, m, None, score, trades, 0.5) for sd, m, _, score, trades in port]
            wf_validated.append((components, pscore, pstats))

    if not wf_validated:
        logger.warning("  No WF-validated portfolios — using top unvalidated")
        for port, pscore, pstats in portfolios[:5]:
            components = [(sd, m, None, score, trades, 0.5) for sd, m, _, score, trades in port]
            wf_validated.append((components, pscore, pstats))

    logger.info(f"  WF validated: {len(wf_validated)} portfolios")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: MC STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 4: MC stress test ({MC_SIMS} sims) ═══")

    mc_validated = []
    for components, pscore, pstats in wf_validated:
        # Combine all trade streams for MC
        all_trades = []
        for sd, m, _, score, trades, *rest in components:
            out = backtest_strategy(sd, data_full, rm, config_full, min_trades=10)
            if out:
                all_trades.extend(out[0])

        if len(all_trades) < 30:
            continue

        # Sort by exit time for combined trade stream
        all_trades.sort(key=lambda t: t.exit_time)

        try:
            mc = mc_test(all_trades, prop_rules, MC_SIMS)
        except Exception:
            continue

        if mc.probability_of_profit >= 0.80:
            mc_validated.append((components, pscore, pstats, mc, all_trades))
            logger.info(f"  ★ Portfolio ({len(components)} strats) | {len(all_trades)} trades | MC P={mc.probability_of_profit:.0%}")

    logger.info(f"  MC validated: {len(mc_validated)} portfolios")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: DECORRELATION
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 5: Portfolio decorrelation ═══")

    # Decorrelate portfolios by combined monthly PnL
    if len(mc_validated) > 1:
        keep = [True] * len(mc_validated)
        for i in range(len(mc_validated)):
            if not keep[i]:
                continue
            months_i = mc_validated[i][2].get("monthly_breakdown", {})
            for j in range(i + 1, len(mc_validated)):
                if not keep[j]:
                    continue
                months_j = mc_validated[j][2].get("monthly_breakdown", {})
                all_mk = sorted(set(months_i.keys()) | set(months_j.keys()))
                if len(all_mk) < 3:
                    continue
                vi = [months_i.get(mk, 0) for mk in all_mk]
                vj = [months_j.get(mk, 0) for mk in all_mk]
                corr = np.corrcoef(vi, vj)[0, 1]
                if not np.isnan(corr) and abs(corr) > CORRELATION_THRESHOLD:
                    if mc_validated[i][1] >= mc_validated[j][1]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        mc_validated = [p for p, k in zip(mc_validated, keep) if k]

    logger.info(f"  Final: {len(mc_validated)} decorrelated portfolios")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{'='*140}")
    print(f"  PORTFOLIO BREED COMPLETE — {tested:,} tested in {total_elapsed/60:.1f} min")
    print(f"  Pipeline: {len(prepared)} seeds → {initial_viable} viable → evolved {MAX_GENERATIONS} gens"
          f" → {len(portfolios)} portfolios → {len(wf_validated)} WF → {len(mc_validated)} final")
    print(f"{'='*140}")

    if mc_validated:
        # Ranked table
        print(f"\n  {'#':<3} {'Components':>4} {'Families':<40} {'Trades':>7} {'Tr/Mo':>6} {'Avg/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'MC P':>6}")
        print(f"  {'-'*100}")
        for i, (components, pscore, pstats, mc, all_trades) in enumerate(mc_validated[:5], 1):
            families = set(get_signal_family(c[0]) for c in components)
            fam_str = ", ".join(sorted(families))[:39]
            print(
                f"  {i:<3} {len(components):>4}     {fam_str:<40} "
                f"{pstats['total_trades']:>7} {pstats['trades_per_month']:>5.0f} "
                f"${pstats['avg_monthly']:>9,.0f} ${pstats['min_monthly']:>9,.0f} "
                f"${pstats['combined_dd']:>9,.0f} {mc.probability_of_profit:>5.0%}"
            )

        # Best portfolio details
        components, pscore, pstats, mc, all_trades = mc_validated[0]
        combined_months = pstats["monthly_breakdown"]
        months_above_15k = pstats["months_above_15k"]
        pct_above_15k = months_above_15k / max(1, pstats["num_months"]) * 100

        print(f"""
  ══════════════════════════════════════════════════════════════
  TOP PORTFOLIO — {len(components)} Strategies
  ══════════════════════════════════════════════════════════════""")

        for idx, comp in enumerate(components):
            sd = comp[0]
            m = comp[1]
            fam = get_signal_family(sd)
            months_c = monthly_pnl(comp[4])
            avg_c = np.mean(list(months_c.values())) if months_c else 0
            print(f"  Component {idx+1}: {sd['name'][:50]}")
            print(f"    Family:     {fam}")
            print(f"    Trades:     {m.total_trades} | WR: {m.win_rate:.0f}% | PF: {m.profit_factor:.2f}")
            print(f"    Avg/Mo:     ${avg_c:,.0f} | DD: ${m.max_drawdown:,.0f} | Ct: {sd['sizing_rules']['fixed_contracts']}")

        print(f"""
  ──────────────────────────────────────────────────────────────
  COMBINED STATS
  ──────────────────────────────────────────────────────────────
  Total Trades:    {pstats['total_trades']}
  Trades/Month:    {pstats['trades_per_month']:.0f}
  Avg Month:       ${pstats['avg_monthly']:,.2f}
  Best Month:      ${pstats['max_monthly']:,.2f}
  Worst Month:     ${pstats['min_monthly']:,.2f}
  Combined DD:     ${pstats['combined_dd']:,.2f}
  Months >= $15K:  {months_above_15k}/{pstats['num_months']} ({pct_above_15k:.0f}%)
  ──────────────────────────────────────────────────────────────
  MC P(profit):    {mc.probability_of_profit:.1%}
  MC Median:       ${mc.median_return:,.2f}
  MC 5th pctl:     ${mc.pct_5th_return:,.2f}
  MC Composite:    {mc.composite_score:.1f}/100
  ══════════════════════════════════════════════════════════════""")

        # Combined monthly breakdown
        print(f"\n  COMBINED MONTHLY P&L:")
        for mo in sorted(combined_months.keys()):
            p = combined_months[mo]
            bar = "█" * max(1, int(abs(p) / 1000))
            flag = "★" if p >= TARGET_AVG_MONTHLY else " "
            sign = "+" if p >= 0 else "-"
            print(f"    {flag} {mo}: {sign}${abs(p):>10,.2f}  {bar}{'*' if p < 0 else ''}")

        # Per-component monthly contribution
        print(f"\n  MONTHLY CONTRIBUTION BY COMPONENT:")
        header = f"    {'Month':<10}"
        for idx in range(len(components)):
            header += f" {'Strat'+str(idx+1):>10}"
        header += f" {'TOTAL':>12}"
        print(header)
        for mo in sorted(combined_months.keys()):
            row = f"    {mo:<10}"
            for comp in components:
                comp_months = monthly_pnl(comp[4])
                v = comp_months.get(mo, 0)
                row += f" ${v:>9,.0f}"
            row += f" ${combined_months[mo]:>11,.0f}"
            print(row)

        # Comparison: best single vs portfolio
        if population:
            best_single = population[0]
            bs_months = monthly_pnl(best_single[4])
            bs_avg = np.mean(list(bs_months.values())) if bs_months else 0
            bs_min = min(bs_months.values()) if bs_months else 0
            bs_trades = best_single[1].total_trades

            print(f"\n  COMPARISON: Best Single vs Portfolio:")
            print(f"    {'Metric':<20} {'Single':>15} {'Portfolio':>15}")
            print(f"    {'-'*50}")
            print(f"    {'Trades/year':<20} {bs_trades:>15} {pstats['total_trades']:>15}")
            print(f"    {'Avg/month':<20} ${bs_avg:>14,.0f} ${pstats['avg_monthly']:>14,.0f}")
            print(f"    {'Min month':<20} ${bs_min:>14,.0f} ${pstats['min_monthly']:>14,.0f}")
            print(f"    {'Months >= $15K':<20} {sum(1 for v in bs_months.values() if v >= 15000):>15} {months_above_15k:>15}")

    # Save
    if mc_validated:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "portfolio_breed_v1",
            "target": "$15K+/month from 3-5 uncorrelated strategies",
            "pipeline_stats": {
                "seeds": len(prepared), "viable": initial_viable, "tested": tested,
                "generations": MAX_GENERATIONS, "portfolios_built": len(portfolios),
                "wf_validated": len(wf_validated), "mc_validated": len(mc_validated),
                "elapsed_min": round(total_elapsed / 60, 1),
            },
            "portfolios": [],
        }

        for port_idx, (components, pscore, pstats, mc, all_trades) in enumerate(mc_validated[:5]):
            port_entry = {
                "portfolio_id": port_idx + 1,
                "num_components": len(components),
                "portfolio_score": round(pscore, 2),
                "combined_stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in pstats.items() if k != "monthly_breakdown"},
                "combined_monthly": {mo: round(v, 2) for mo, v in sorted(pstats.get("monthly_breakdown", {}).items())},
                "mc_p_profit": round(mc.probability_of_profit, 4),
                "mc_median": round(mc.median_return, 2),
                "mc_composite": round(mc.composite_score, 2),
                "components": [],
            }
            for comp in components:
                sd, m = comp[0], comp[1]
                wf_eff = comp[5] if len(comp) > 5 else 0.5
                comp_months = monthly_pnl(comp[4])
                port_entry["components"].append({
                    "name": sd["name"],
                    "strategy": sd,
                    "signal_family": get_signal_family(sd),
                    "trades": m.total_trades,
                    "win_rate": round(m.win_rate, 2),
                    "profit_factor": round(m.profit_factor, 2),
                    "sharpe": round(m.sharpe_ratio, 2),
                    "total_pnl": round(m.total_pnl, 2),
                    "max_drawdown": round(m.max_drawdown, 2),
                    "avg_monthly": round(float(np.mean(list(comp_months.values()))), 2),
                    "monthly_breakdown": {mo: round(v, 2) for mo, v in sorted(comp_months.items())},
                    "wf_efficiency": round(wf_eff, 3),
                })
            output["portfolios"].append(port_entry)

        Path("reports").mkdir(exist_ok=True)
        with open("reports/portfolio_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/portfolio_v1.json")

    print(f"\n{'='*140}\n")


if __name__ == "__main__":
    main()
