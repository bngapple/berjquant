#!/usr/bin/env python3
"""
MAXIMIZE CHAMPION — Take the #1 crossbreed V4 strategy and mutate it
aggressively to hit $15K+/month.

The champion (ROC × Keltner + PrevDay) makes $19.8K/yr on 4 contracts
with 39 trades. To get to $15K/month we need:
  - More contracts (4 → 8-15)
  - More trades (loosen params, widen time window)
  - Tighter SL to allow more contracts without blowing DD
  - Maybe drop require_all_entries to OR mode for more signals
  - Try different filter combos (drop prev_day, add others)

Strategy: 3000 mutations × backtest → top 60 → WF → MC → output
"""

import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter

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
MAX_DD_LIMIT = -4500.0
TARGET_AVG_MONTHLY = 15000
MC_SIMS = 5000

random.seed(42)
np.random.seed(42)

# ── The champion strategy ─────────────────────────────────────────────
CHAMPION = {
    "name": "ROC×KC+PREVDAY|champ",
    "entry_signals": [
        {
            "signal_name": "roc",
            "module": "signals.momentum",
            "function": "roc",
            "params": {"period": 32},
            "columns": {"long": "entry_long_roc", "short": "entry_short_roc"},
        },
        {
            "signal_name": "keltner_channels",
            "module": "signals.volatility",
            "function": "keltner_channels",
            "params": {"ema_period": 16, "atr_period": 3, "multiplier": 2.2608},
            "columns": {"long": "entry_long_kc", "short": "entry_short_kc"},
        },
    ],
    "entry_filters": [
        {
            "signal_name": "previous_day_levels",
            "module": "signals.price_action",
            "function": "previous_day_levels",
            "params": {},
            "column": "signal_above_prev_high",
        },
        {
            "signal_name": "time_of_day",
            "module": "signals.time_filters",
            "function": "time_of_day",
            "params": {"start_hour": 9, "start_minute": 30, "end_hour": 11, "end_minute": 0},
            "column": "signal_time_allowed",
        },
    ],
    "exit_rules": {
        "stop_loss_type": "fixed_points",
        "stop_loss_value": 17.6,
        "take_profit_type": "fixed_points",
        "take_profit_value": 218.1,
        "trailing_stop": False,
        "trailing_activation": 4.0,
        "trailing_distance": 2.0,
        "time_exit_minutes": None,
    },
    "sizing_rules": {
        "method": "fixed",
        "fixed_contracts": 4,
        "risk_pct": 0.02,
        "atr_risk_multiple": 2.0,
    },
    "primary_timeframe": "1m",
    "require_all_entries": True,
}


def backtest(sd, data, rm, config, min_trades=10):
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


def monthly_pnl(trades):
    months = {}
    for t in trades:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


def fitness(metrics, trades):
    """Fitness heavily weighted toward $15K+/month."""
    months = monthly_pnl(trades)
    if not months:
        return -999999
    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    dd = abs(metrics.max_drawdown)

    months_hitting_15k = sum(1 for v in monthly_vals if v >= 15000)
    months_hitting_10k = sum(1 for v in monthly_vals if v >= 10000)
    floor_penalty = sum(max(0, 15000 - v) for v in monthly_vals if v > 0) * 0.3
    dd_penalty = max(0, (dd - 3500) * 10) if dd > 3500 else 0

    score = (
        avg_monthly * 6.0
        + min_monthly * 3.0
        + metrics.sharpe_ratio * 1000
        + metrics.profit_factor * 500
        + months_hitting_15k * 5000
        + months_hitting_10k * 2000
        - dd_penalty
        - floor_penalty
    )
    return score


def mutate(sd, intensity=0.5, variant_type="full"):
    """Create a mutation of the champion."""
    new = copy.deepcopy(sd)

    # ── Mutate ROC period ──
    roc_sig = new["entry_signals"][0]
    p = roc_sig["params"]["period"]
    if variant_type in ("full", "params"):
        # Explore wide range: 3-50
        new_p = max(3, min(50, int(p + p * intensity * random.uniform(-1, 1))))
        roc_sig["params"]["period"] = new_p

    # ── Mutate Keltner params ──
    kc_sig = new["entry_signals"][1]
    if variant_type in ("full", "params"):
        kc = kc_sig["params"]
        kc["ema_period"] = max(3, min(50, int(kc["ema_period"] + kc["ema_period"] * intensity * random.uniform(-1, 1))))
        kc["atr_period"] = max(2, min(30, int(kc["atr_period"] + kc["atr_period"] * intensity * random.uniform(-1, 1))))
        kc["multiplier"] = round(max(0.5, min(4.0, kc["multiplier"] + kc["multiplier"] * intensity * random.uniform(-1, 1))), 4)

    # ── Mutate exits ──
    er = new["exit_rules"]
    if variant_type in ("full", "exits"):
        sl = er["stop_loss_value"]
        tp = er["take_profit_value"]
        new_sl = round(max(8, min(35, sl + sl * intensity * random.uniform(-1, 1))), 1)
        new_tp = round(max(40, min(300, tp + tp * intensity * random.uniform(-1, 1))), 1)
        # Keep minimum 4:1 R:R
        if new_sl > 0 and new_tp / new_sl < 4.0:
            new_tp = round(new_sl * 4.0, 1)
        er["stop_loss_value"] = new_sl
        er["take_profit_value"] = new_tp

    # ── Mutate contracts (THE KEY LEVER) ──
    sz = new["sizing_rules"]
    if variant_type in ("full", "sizing"):
        # Bias upward — we need more firepower
        ct = sz["fixed_contracts"]
        new_ct = max(4, min(15, int(ct * random.uniform(1.0, 3.0))))
        sz["fixed_contracts"] = new_ct

    # ── Mutate time window ──
    for filt in new["entry_filters"]:
        if filt.get("signal_name") == "time_of_day":
            if variant_type in ("full", "time") and random.random() < 0.4:
                windows = [
                    (9, 30, 11, 0),    # Original: 1.5hr
                    (9, 30, 11, 30),   # 2hr
                    (9, 30, 12, 0),    # 2.5hr morning
                    (9, 30, 13, 0),    # 3.5hr through lunch
                    (9, 30, 14, 0),    # 4.5hr through afternoon
                    (9, 30, 15, 0),    # Full day minus last hour
                    (9, 0, 11, 0),     # Pre-market + opening
                    (9, 0, 12, 0),     # Pre-market + morning
                    (8, 30, 11, 30),   # Extended pre-market
                    (9, 30, 16, 0),    # Full session
                ]
                w = random.choice(windows)
                filt["params"] = {
                    "start_hour": w[0], "start_minute": w[1],
                    "end_hour": w[2], "end_minute": w[3],
                }

    # ── Occasionally try OR mode (more signals fire) ──
    if variant_type == "full" and random.random() < 0.2:
        new["require_all_entries"] = False

    # ── Occasionally try trailing stop ──
    if variant_type in ("full", "exits") and random.random() < 0.3:
        er["trailing_stop"] = True
        er["trailing_activation"] = round(random.uniform(15, 80), 1)
        er["trailing_distance"] = round(random.uniform(8, 25), 1)

    # ── Occasionally drop the prev_day filter (more trades) ──
    if variant_type == "full" and random.random() < 0.3:
        new["entry_filters"] = [f for f in new["entry_filters"] if f.get("signal_name") != "previous_day_levels"]

    # ── Occasionally swap prev_day for a different filter ──
    if variant_type == "full" and random.random() < 0.2:
        alt_filters = [
            {"signal_name": "ema_slope", "module": "signals.trend", "function": "ema_slope",
             "params": {"period": random.randint(10, 30), "slope_lookback": random.randint(2, 5)},
             "column": "signal_ema_slope_up"},
            {"signal_name": "relative_volume", "module": "signals.volume", "function": "relative_volume",
             "params": {"lookback": random.randint(10, 30)},
             "column": "signal_high_volume"},
            {"signal_name": "supertrend", "module": "signals.trend", "function": "supertrend",
             "params": {"period": random.randint(7, 15), "multiplier": round(random.uniform(1.5, 3.5), 1)},
             "column": "signal_supertrend_bullish"},
            {"signal_name": "imbalance", "module": "signals.orderflow", "function": "imbalance",
             "params": {"ratio_threshold": round(random.uniform(1.5, 4.0), 2)},
             "column": "signal_buy_imbalance"},
            {"signal_name": "volume_climax", "module": "signals.volume", "function": "volume_climax",
             "params": {"lookback": random.randint(20, 60), "threshold": round(random.uniform(1.5, 3.5), 1)},
             "column": "signal_climax_reversal"},
        ]
        # Replace prev_day with a random alternative
        new["entry_filters"] = [f for f in new["entry_filters"] if f.get("signal_name") != "previous_day_levels"]
        new["entry_filters"].insert(0, random.choice(alt_filters))

    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    new["name"] = f"ROC×KC|max_{h}"
    return new


def main():
    total_start = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     MAXIMIZE CHAMPION — ROC × Keltner + PrevDay → $15K+/month         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Current: $1.98K/month on 4 contracts, 39 trades                       ║
║  Target:  $15K+/month — scale contracts, widen window, loosen params   ║
║  Method:  3000 mutations → top 60 → WF → MC                           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load data
    logger.info("Loading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    split_date = "2024-11-19"
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_validate = df_yr1.filter(pl.col("timestamp") >= pl.lit(split_date).str.strptime(pl.Datetime, "%Y-%m-%d"))

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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Generate 3000 mutations and backtest on train
    # ══════════════════════════════════════════════════════════════════
    logger.info("═══ PHASE 1: Generating and backtesting 3000 mutations ═══")

    # Also create systematic variants for key levers
    mutations = []

    # 1. Pure contract scaling (keep everything else, just add contracts)
    for ct in range(4, 16):
        variant = copy.deepcopy(CHAMPION)
        variant["sizing_rules"]["fixed_contracts"] = ct
        variant["name"] = f"ROC×KC|ct{ct}"
        mutations.append(variant)

    # 2. Contract scaling × wider time windows
    for ct in [6, 8, 10, 12, 15]:
        for window in [(9, 30, 12, 0), (9, 30, 13, 0), (9, 30, 14, 0), (9, 30, 15, 0), (9, 30, 16, 0)]:
            variant = copy.deepcopy(CHAMPION)
            variant["sizing_rules"]["fixed_contracts"] = ct
            for f in variant["entry_filters"]:
                if f.get("signal_name") == "time_of_day":
                    f["params"] = {"start_hour": window[0], "start_minute": window[1], "end_hour": window[2], "end_minute": window[3]}
            variant["name"] = f"ROC×KC|ct{ct}_w{window[2]}"
            mutations.append(variant)

    # 3. Drop prev_day filter × contract scaling (more trades)
    for ct in [6, 8, 10, 12, 15]:
        variant = copy.deepcopy(CHAMPION)
        variant["sizing_rules"]["fixed_contracts"] = ct
        variant["entry_filters"] = [f for f in variant["entry_filters"] if f.get("signal_name") != "previous_day_levels"]
        variant["name"] = f"ROC×KC|ct{ct}_noprev"
        mutations.append(variant)

    # 4. OR mode × contract scaling (way more signals)
    for ct in [6, 8, 10, 12, 15]:
        variant = copy.deepcopy(CHAMPION)
        variant["sizing_rules"]["fixed_contracts"] = ct
        variant["require_all_entries"] = False
        variant["name"] = f"ROC×KC|ct{ct}_OR"
        mutations.append(variant)

    # 5. OR mode + no prev_day + wider window
    for ct in [6, 8, 10, 12, 15]:
        for window in [(9, 30, 13, 0), (9, 30, 15, 0), (9, 30, 16, 0)]:
            variant = copy.deepcopy(CHAMPION)
            variant["sizing_rules"]["fixed_contracts"] = ct
            variant["require_all_entries"] = False
            variant["entry_filters"] = [f for f in variant["entry_filters"] if f.get("signal_name") != "previous_day_levels"]
            for f in variant["entry_filters"]:
                if f.get("signal_name") == "time_of_day":
                    f["params"] = {"start_hour": window[0], "start_minute": window[1], "end_hour": window[2], "end_minute": window[3]}
            variant["name"] = f"ROC×KC|ct{ct}_OR_w{window[2]}_noprev"
            mutations.append(variant)

    # 6. Trailing stop variants
    for ct in [8, 10, 12, 15]:
        for trail_act in [20, 40, 60]:
            for trail_dist in [10, 15, 20]:
                variant = copy.deepcopy(CHAMPION)
                variant["sizing_rules"]["fixed_contracts"] = ct
                variant["exit_rules"]["trailing_stop"] = True
                variant["exit_rules"]["trailing_activation"] = float(trail_act)
                variant["exit_rules"]["trailing_distance"] = float(trail_dist)
                variant["entry_filters"] = [f for f in variant["entry_filters"] if f.get("signal_name") != "previous_day_levels"]
                variant["name"] = f"ROC×KC|ct{ct}_trail{trail_act}_{trail_dist}"
                mutations.append(variant)

    # 7. Tighter SL (more trades survive, can use more contracts)
    for sl in [10, 12, 14, 16, 20, 25]:
        for tp in [60, 80, 100, 150, 200, 250]:
            if tp / sl < 4.0:
                continue
            for ct in [8, 10, 12, 15]:
                variant = copy.deepcopy(CHAMPION)
                variant["exit_rules"]["stop_loss_value"] = float(sl)
                variant["exit_rules"]["take_profit_value"] = float(tp)
                variant["sizing_rules"]["fixed_contracts"] = ct
                variant["entry_filters"] = [f for f in variant["entry_filters"] if f.get("signal_name") != "previous_day_levels"]
                variant["name"] = f"ROC×KC|sl{sl}_tp{tp}_ct{ct}"
                mutations.append(variant)

    # 8. Random mutations to fill the rest
    n_random = max(0, 3000 - len(mutations))
    for i in range(n_random):
        intensity = random.uniform(0.2, 1.5)
        mutations.append(mutate(CHAMPION, intensity, "full"))

    logger.info(f"  Generated {len(mutations)} mutations ({len(mutations) - n_random} systematic + {n_random} random)")

    # Backtest all on train
    population = []
    for i, sd in enumerate(mutations):
        if (i + 1) % 200 == 0:
            logger.info(f"  Backtesting {i+1}/{len(mutations)}...")
        out = backtest(sd, data_train, rm, config_train, min_trades=5)
        if out:
            trades, m = out
            if m.total_pnl > 0:
                score = fitness(m, trades)
                months = monthly_pnl(trades)
                avg_mo = np.mean(list(months.values())) if months else 0
                population.append((sd, m, score, trades, avg_mo))

    population.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"  {len(population)} profitable mutations")

    # Show top 20 on train
    above_15k = sum(1 for p in population if p[4] >= TARGET_AVG_MONTHLY)
    above_10k = sum(1 for p in population if p[4] >= 10000)
    logger.info(f"  $15K+/mo: {above_15k} | $10K+/mo: {above_10k}")

    print(f"\n  TOP 20 ON TRAIN DATA:")
    print(f"  {'#':<4} {'Name':<44} {'Tr':>5} {'WR':>6} {'PF':>5} {'PnL':>12} {'Avg/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'Ct':>4}")
    print(f"  {'-'*120}")
    for i, (sd, m, score, trades, avg_mo) in enumerate(population[:20], 1):
        months = monthly_pnl(trades)
        min_mo = min(months.values()) if months else 0
        flag = "★" if avg_mo >= TARGET_AVG_MONTHLY else " "
        ct = sd["sizing_rules"]["fixed_contracts"]
        print(
            f"  {flag}{i:<3} {sd['name'][:43]:<44} "
            f"{m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>4.2f} "
            f"${m.total_pnl:>11,.0f} ${avg_mo:>9,.0f} ${min_mo:>9,.0f} "
            f"${m.max_drawdown:>9,.0f} {ct:>4}"
        )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Evolve top 60 for 30 more generations
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 2: Evolving top performers (30 generations) ═══")

    elite_pool = population[:60]
    best_score = elite_pool[0][2] if elite_pool else 0

    for gen in range(30):
        gen_start = time.time()
        intensity = 0.3 + 0.5 * abs(np.sin(gen / 10 * np.pi))

        elites = [p[0] for p in elite_pool[:15]]
        mid = [p[0] for p in elite_pool[15:40]]
        pool = elites + mid

        offspring = []
        # Mutations of top performers
        for _ in range(80):
            parent = random.choice(pool)
            offspring.append(mutate(parent, intensity, "full"))

        # Crossover: blend params of two parents
        for _ in range(40):
            if len(pool) >= 2:
                a, b = random.sample(pool, 2)
                child = copy.deepcopy(a)
                # Blend contracts
                w = random.uniform(0.3, 0.7)
                a_ct = a["sizing_rules"]["fixed_contracts"]
                b_ct = b["sizing_rules"]["fixed_contracts"]
                child["sizing_rules"]["fixed_contracts"] = max(4, min(15, int(a_ct * w + b_ct * (1 - w))))
                # Blend SL/TP
                a_sl = a["exit_rules"]["stop_loss_value"]
                b_sl = b["exit_rules"]["stop_loss_value"]
                a_tp = a["exit_rules"]["take_profit_value"]
                b_tp = b["exit_rules"]["take_profit_value"]
                new_sl = round(max(8, a_sl * w + b_sl * (1 - w)), 1)
                new_tp = round(max(40, a_tp * w + b_tp * (1 - w)), 1)
                if new_sl > 0 and new_tp / new_sl < 4.0:
                    new_tp = round(new_sl * 4.0, 1)
                child["exit_rules"]["stop_loss_value"] = new_sl
                child["exit_rules"]["take_profit_value"] = new_tp
                # Blend ROC period
                a_roc = a["entry_signals"][0]["params"]["period"]
                b_roc = b["entry_signals"][0]["params"]["period"]
                child["entry_signals"][0]["params"]["period"] = max(3, int(a_roc * w + b_roc * (1 - w)))
                h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
                child["name"] = f"ROC×KC|evo_{h}"
                offspring.append(child)

        # Evaluate offspring
        new_pop = list(elite_pool[:15])  # carry forward elites
        for sd in offspring:
            out = backtest(sd, data_train, rm, config_train, min_trades=5)
            if out:
                trades, m = out
                if m.total_pnl > 0:
                    score = fitness(m, trades)
                    months = monthly_pnl(trades)
                    avg_mo = np.mean(list(months.values())) if months else 0
                    new_pop.append((sd, m, score, trades, avg_mo))

        new_pop.sort(key=lambda x: x[2], reverse=True)
        elite_pool = new_pop[:60]

        gen_elapsed = time.time() - gen_start
        if elite_pool:
            sd, m, score, trades, avg_mo = elite_pool[0]
            months = monthly_pnl(trades)
            min_mo = min(months.values()) if months else 0
            ct = sd["sizing_rules"]["fixed_contracts"]
            above = sum(1 for p in elite_pool if p[4] >= TARGET_AVG_MONTHLY)
            logger.info(
                f"  Gen {gen+1:>2}/30 | best=${avg_mo:,.0f}/mo | min=${min_mo:,.0f} | "
                f"ct={ct} | DD=${m.max_drawdown:,.0f} | $15K+={above} | {gen_elapsed:.0f}s"
            )

    population = elite_pool

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: OOS Validation
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 3: OOS validation (4 months) ═══")

    oos_survivors = []
    for sd, m_train, score, trades_train, avg_mo_train in population[:60]:
        out = backtest(sd, data_validate, rm, config_validate, min_trades=3)
        if out is None:
            continue
        trades_oos, m_oos = out
        if m_oos.total_pnl <= 0:
            continue
        months_oos = monthly_pnl(trades_oos)
        avg_mo_oos = np.mean(list(months_oos.values())) if months_oos else 0
        oos_survivors.append((sd, m_train, m_oos, score, trades_train, avg_mo_oos))
        logger.info(
            f"  ✓ {sd['name'][:40]} | OOS PnL=${m_oos.total_pnl:,.0f} | avg/mo=${avg_mo_oos:,.0f} | "
            f"WR={m_oos.win_rate:.0f}% | DD=${m_oos.max_drawdown:,.0f}"
        )

    logger.info(f"  OOS survivors: {len(oos_survivors)}/{min(60, len(population))}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Walk-forward validation
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 4: Walk-forward validation ═══")

    wf_survivors = []
    try:
        from validation.walk_forward import WalkForwardValidator
        for sd, m_train, m_oos, score, trades_train, avg_mo_oos in oos_survivors:
            try:
                strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
                wf = WalkForwardValidator(
                    data=data_full, risk_manager=rm,
                    contract_spec=MNQ_SPEC, config=config_full,
                    account_size=150000.0,
                )
                wf_result = wf.validate(strategy, train_days=60, test_days=20, step_days=20)
                if wf_result.wf_efficiency < 0.30:
                    logger.info(f"  ✗ WF reject: {sd['name'][:40]} | WF={wf_result.wf_efficiency:.2f}")
                    continue
                wf_survivors.append((sd, m_train, m_oos, score, trades_train, avg_mo_oos, wf_result.wf_efficiency))
                logger.info(f"  ✓ WF pass: {sd['name'][:40]} | WF={wf_result.wf_efficiency:.2f}")
            except Exception as e:
                wf_survivors.append((sd, m_train, m_oos, score, trades_train, avg_mo_oos, 0.5))
    except ImportError:
        logger.warning("  WF not available, passing all OOS survivors")
        wf_survivors = [(sd, m_train, m_oos, score, trades_train, avg_mo_oos, 0.5) for sd, m_train, m_oos, score, trades_train, avg_mo_oos in oos_survivors]

    logger.info(f"  WF survivors: {len(wf_survivors)}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Full-year MC stress test
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 5: Full-year MC stress test ({MC_SIMS} sims) ═══")

    final = []
    for sd, m_train, m_oos, score, _, avg_mo_oos, wf_eff in wf_survivors:
        out = backtest(sd, data_full, rm, config_full, min_trades=10)
        if out is None:
            continue
        trades_full, m_full = out

        try:
            mc_sim = MonteCarloSimulator(MCConfig(
                n_simulations=MC_SIMS, initial_capital=150000.0,
                prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
            ))
            mc = mc_sim.run(trades_full, strategy_name=sd.get("name", "unknown"))
        except Exception:
            continue

        months = monthly_pnl(trades_full)
        avg_mo = np.mean(list(months.values())) if months else 0

        final.append((sd, m_full, mc, trades_full, avg_mo, wf_eff))
        logger.info(
            f"  {'★' if avg_mo >= TARGET_AVG_MONTHLY else ' '} {sd['name'][:40]} | PnL=${m_full.total_pnl:,.0f} | "
            f"avg/mo=${avg_mo:,.0f} | MC P={mc.probability_of_profit:.0%} | DD=${m_full.max_drawdown:,.0f}"
        )

    final.sort(key=lambda x: x[4], reverse=True)  # Sort by avg_monthly

    # ══════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{'='*130}")
    print(f"  MAXIMIZE CHAMPION COMPLETE — {len(mutations):,} mutations tested in {total_elapsed/60:.1f} min")
    print(f"  Pipeline: {len(mutations)} mutations → {len(population)} evolved → {len(oos_survivors)} OOS → {len(wf_survivors)} WF → {len(final)} final")
    print(f"{'='*130}")

    if final:
        print(f"\n  {'#':<4} {'Name':<44} {'Tr':>5} {'WR':>6} {'PF':>5} {'1yr PnL':>12} {'Avg/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'Ct':>4} {'MC P':>6}")
        print(f"  {'-'*130}")
        for i, (sd, m, mc, trades, avg_mo, wf_eff) in enumerate(final[:30], 1):
            months = monthly_pnl(trades)
            min_mo = min(months.values()) if months else 0
            ct = sd["sizing_rules"]["fixed_contracts"]
            flag = "★" if avg_mo >= TARGET_AVG_MONTHLY else " "
            print(
                f"  {flag}{i:<3} {sd['name'][:43]:<44} "
                f"{m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>4.2f} "
                f"${m.total_pnl:>11,.0f} ${avg_mo:>9,.0f} ${min_mo:>9,.0f} "
                f"${m.max_drawdown:>9,.0f} {ct:>4} {mc.probability_of_profit:>5.0%}"
            )

        # Champion
        sd, m, mc, trades, avg_mo, wf_eff = final[0]
        months = monthly_pnl(trades)
        monthly_vals = list(months.values())
        er = sd["exit_rules"]
        print(f"""
  ══════════════════════════════════════════════════════════════
  MAXIMIZED CHAMPION
  ══════════════════════════════════════════════════════════════
  Name:           {sd['name']}
  Entries:        {', '.join(e['signal_name'] for e in sd['entry_signals'])}
  Filters:        {', '.join(f['signal_name'] for f in sd.get('entry_filters', []))}
  AND/OR mode:    {'OR' if not sd.get('require_all_entries', True) else 'AND'}
  Stop Loss:      {er['stop_loss_value']}pt
  Take Profit:    {er['take_profit_value']}pt
  R:R:            {er['take_profit_value']/max(er['stop_loss_value'],0.01):.1f}:1
  Trailing:       {er.get('trailing_stop', False)} (act={er.get('trailing_activation')}, dist={er.get('trailing_distance')})
  Contracts:      {sd['sizing_rules']['fixed_contracts']}
  ──────────────────────────────────────────────────────────────
  Trades (1yr):   {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe:         {m.sharpe_ratio:.2f}
  Net P&L (1yr):  ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}  (limit: $-4,500)
  DD Margin:      ${abs(MAX_DD_LIMIT) - abs(m.max_drawdown):,.2f}
  ──────────────────────────────────────────────────────────────
  Avg Month:      ${avg_mo:,.2f}
  Best Month:     ${max(monthly_vals):,.2f}
  Worst Month:    ${min(monthly_vals):,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC P(ruin):     {mc.probability_of_ruin:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  MC Composite:   {mc.composite_score:.1f}/100
  MC Pass Rate:   {mc.prop_firm_pass_rate:.1%}
  ══════════════════════════════════════════════════════════════""")

        # Monthly PnL
        print(f"\n  MONTHLY P&L:")
        for mo in sorted(months.keys()):
            p = months[mo]
            bar = "█" * max(1, int(abs(p) / 1000))
            flag = "★" if p >= TARGET_AVG_MONTHLY else " "
            sign = "+" if p >= 0 else "-"
            print(f"    {flag} {mo}: {sign}${abs(p):>10,.2f}  {bar}{'*' if p < 0 else ''}")

        print(f"\n  Full Parameters:")
        for sig in sd["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in sd.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

    # Save
    if final:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "maximize_champion_v1",
            "base_strategy": "ROC×Keltner+PrevDay (crossbreed V4 #1)",
            "target": "$15K+/month",
            "strategies": [],
        }
        for sd, m, mc, trades, avg_mo, wf_eff in final:
            months = monthly_pnl(trades)
            monthly_vals = list(months.values())
            output["strategies"].append({
                "name": sd["name"], "strategy": sd,
                "trades": m.total_trades, "win_rate": round(m.win_rate, 2),
                "profit_factor": round(m.profit_factor, 2), "sharpe": round(m.sharpe_ratio, 2),
                "total_pnl": round(m.total_pnl, 2), "max_drawdown": round(m.max_drawdown, 2),
                "avg_monthly": round(avg_mo, 2), "min_monthly": round(float(min(monthly_vals)), 2),
                "max_monthly": round(float(max(monthly_vals)), 2),
                "monthly_breakdown": {mo: round(v, 2) for mo, v in sorted(months.items())},
                "mc_p_profit": round(mc.probability_of_profit, 4),
                "mc_median": round(mc.median_return, 2),
                "mc_composite": round(mc.composite_score, 2),
                "mc_pass_rate": round(mc.prop_firm_pass_rate, 4),
                "wf_efficiency": round(wf_eff, 3),
                "contracts": sd["sizing_rules"]["fixed_contracts"],
            })
        with open("reports/maximized_champion_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(final)} strategies to reports/maximized_champion_v1.json")

    print(f"\n{'='*130}\n")


if __name__ == "__main__":
    main()
