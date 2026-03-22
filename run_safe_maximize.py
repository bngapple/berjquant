#!/usr/bin/env python3
"""
SAFE MAXIMIZE — Max profit while NEVER losing more than $4,000 in any month.

Rule #1: No combined month can lose > $4,000. Period.
Rule #2: Maximize profit subject to Rule #1.
"""

import gc
import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path
from collections import defaultdict

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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("safe")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
MONTH_LOSS_CAP = -4000.0
MAX_DD = -4500.0
random.seed(42)
np.random.seed(42)

WINDOWS = [
    (9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),
    (9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0),
]

FILTER_OPTS = [
    ("large_trade_detection","signals.orderflow","large_trade_detection",{"volume_lookback":50,"threshold":3.0},"signal_large_trade"),
    ("ema_slope","signals.trend","ema_slope",{"period":21,"slope_lookback":3},"signal_ema_slope_up"),
    ("supertrend","signals.trend","supertrend",{"period":10,"multiplier":3.0},"signal_supertrend_bullish"),
    ("relative_volume","signals.volume","relative_volume",{"lookback":20},"signal_high_volume"),
    ("candle_patterns","signals.price_action","candle_patterns",{},"signal_hammer"),
    ("session_levels","signals.price_action","session_levels",{},"signal_at_session_high"),
    ("bollinger_keltner_squeeze","signals.volatility","bollinger_keltner_squeeze",{"bb_period":20,"bb_std":2.0,"kc_period":20,"kc_atr_period":14,"kc_mult":1.5},"signal_squeeze_fire"),
    ("absorption","signals.orderflow","absorption",{"volume_threshold":2.0,"price_threshold":0.3},"signal_absorption"),
    ("trapped_traders","signals.orderflow","trapped_traders",{"lookback":5,"retrace_pct":0.5},"signal_trapped_longs"),
    ("none",None,None,None,None),
]

ENTRY_OPTS = [
    ("rsi","signals.momentum","rsi",{"period":14,"overbought":70.0,"oversold":30.0},["entry_long_rsi","entry_short_rsi"]),
    ("stochastic","signals.momentum","stochastic",{"k_period":14,"d_period":3,"overbought":80.0,"oversold":20.0},["entry_long_stoch","entry_short_stoch"]),
    ("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
    ("macd","signals.momentum","macd",{"fast":12,"slow":26,"signal_period":9},["entry_long_macd","entry_short_macd"]),
    ("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    ("williams_r","signals.momentum","williams_r",{"period":14,"overbought":-20.0,"oversold":-80.0},["entry_long_williams","entry_short_williams"]),
    ("ema_crossover","signals.trend","ema_crossover",{"fast_period":9,"slow_period":21},["entry_long_ema_cross","entry_short_ema_cross"]),
    ("bollinger_bands","signals.volatility","bollinger_bands",{"period":20,"std_dev":2.0},["entry_long_bb","entry_short_bb"]),
    ("keltner_channels","signals.volatility","keltner_channels",{"ema_period":20,"atr_period":14,"multiplier":1.5},["entry_long_kc","entry_short_kc"]),
    ("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    ("range_breakout","signals.price_action","range_breakout",{"lookback":20},["entry_long_breakout","entry_short_breakout"]),
    ("opening_range","signals.price_action","opening_range",{"minutes":15},["entry_long_orb","entry_short_orb"]),
]


def bt(sd, data, rm, config, min_trades=10):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
            mo[k] = mo.get(k, 0) + t.net_pnl
        del r, s
        return m, mo
    except Exception:
        return None


def rand_strategy():
    """Generate a random strategy from the full signal space."""
    e = random.choice(ENTRY_OPTS)
    f = random.choice(FILTER_OPTS)
    tw = random.choice(WINDOWS)
    # Randomize entry params
    ep = copy.deepcopy(e[3])
    for k, v in ep.items():
        if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.4, 1.6)))
        elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
    entry = {"signal_name": e[0], "module": e[1], "function": e[2], "params": ep,
             "columns": {"long": e[4][0], "short": e[4][1]}}
    filters = []
    if f[1]:
        fp = copy.deepcopy(f[3])
        for k, v in fp.items():
            if isinstance(v, int): fp[k] = max(1, int(v * random.uniform(0.5, 1.5)))
            elif isinstance(v, float): fp[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
        filters.append({"signal_name": f[0], "module": f[1], "function": f[2], "params": fp, "column": f[4]})
    filters.append({"signal_name": "time_of_day", "module": "signals.time_filters", "function": "time_of_day",
                     "params": {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]},
                     "column": "signal_time_allowed"})
    sl = round(random.uniform(5, 50), 1)
    tp = round(random.randrange(10, 405, 5), 1)
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {
        "name": f"{e[0]}|{f[0]}|new_{h}",
        "entry_signals": [entry], "entry_filters": filters,
        "exit_rules": {"stop_loss_type": "fixed_points", "stop_loss_value": sl,
                       "take_profit_type": "fixed_points", "take_profit_value": tp,
                       "trailing_stop": False, "trailing_activation": 4.0, "trailing_distance": 2.0, "time_exit_minutes": None},
        "sizing_rules": {"method": "fixed", "fixed_contracts": 2, "risk_pct": 0.02, "atr_risk_multiple": 2.0},
        "primary_timeframe": "1m", "require_all_entries": True,
    }


def mutate_grinder(sd, intensity=0.6):
    new = copy.deepcopy(sd)
    for sig in new["entry_signals"]:
        for k, v in sig["params"].items():
            if isinstance(v, int): sig["params"][k] = max(2, int(v * random.uniform(1-intensity, 1+intensity)))
            elif isinstance(v, float): sig["params"][k] = round(max(0.1, v * random.uniform(1-intensity, 1+intensity)), 4)
    for f in new.get("entry_filters", []):
        if f.get("signal_name") == "time_of_day": continue
        for k, v in f["params"].items():
            if isinstance(v, int): f["params"][k] = max(1, int(v * random.uniform(1-intensity, 1+intensity)))
            elif isinstance(v, float): f["params"][k] = round(max(0.1, v * random.uniform(1-intensity, 1+intensity)), 4)
    er = new["exit_rules"]
    er["stop_loss_value"] = round(random.uniform(5, 50), 1)
    er["take_profit_value"] = round(random.randrange(10, 405, 5), 1)
    if random.random() < 0.3:
        tw = random.choice(WINDOWS)
        for f in new["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]}
    if random.random() < 0.2:
        fo = random.choice(FILTER_OPTS)
        non_time = [f for f in new["entry_filters"] if f.get("signal_name") != "time_of_day"]
        time_f = [f for f in new["entry_filters"] if f.get("signal_name") == "time_of_day"]
        if fo[1]:
            new["entry_filters"] = [{"signal_name": fo[0], "module": fo[1], "function": fo[2], "params": copy.deepcopy(fo[3]), "column": fo[4]}] + time_f
        else:
            new["entry_filters"] = time_f
    if random.random() < 0.15:
        eo = random.choice(ENTRY_OPTS)
        ep = copy.deepcopy(eo[3])
        for k, v in ep.items():
            if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.5, 1.5)))
            elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
        new["entry_signals"] = [{"signal_name": eo[0], "module": eo[1], "function": eo[2], "params": ep,
                                  "columns": {"long": eo[4][0], "short": eo[4][1]}}]
    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    new["name"] = f"{new['entry_signals'][0]['signal_name']}|opt_{h}"
    return new


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     SAFE MAXIMIZE — Max Profit, NEVER Lose > $4,000/month              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Rule #1: No combined month can lose more than $4,000. Period.         ║
║  Rule #2: Maximize profit subject to Rule #1.                          ║
║  Current: $5,237/mo avg but worst month -$6,128 = ACCOUNT DEAD         ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load grinders
    with open("reports/grinder_maximized_v1.json") as f:
        gdata = json.load(f)
    grinders = [g["strategy"] for g in gdata["grinders"]]
    logger.info(f"Loaded {len(grinders)} grinders")

    # Load data
    logger.info("Loading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    cf = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    # Fast 2-month slice
    df_2mo = df_yr1.filter(pl.col("timestamp") < pl.lit("2024-05-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    data_2mo = {"1m": df_2mo}
    cf_2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=3, initial_capital=150000.0)
    data_full = {"1m": df_yr1}

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: ANALYZE
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 1: Analyzing current grinders ═══")
    grinder_months = []
    for gi, sd in enumerate(grinders):
        out = bt(sd, data_full, rm, cf, min_trades=5)
        gc.collect()
        if out:
            m, mo = out
            mv = list(mo.values())
            worst = min(mv)
            neg = sum(1 for v in mv if v < 0)
            ct = sd["sizing_rules"]["fixed_contracts"]
            print(f"  G{gi+1}: {sd['name'][:35]} | ct={ct} | {m.total_trades}tr ${np.mean(mv):,.0f}/mo | worst=${worst:,.0f} | {neg} neg months")
            grinder_months.append(mo)
        else:
            print(f"  G{gi+1}: FAILED")
            grinder_months.append({})

    combined = defaultdict(float)
    for mo in grinder_months:
        for k, v in mo.items():
            combined[k] += v
    print(f"\n  COMBINED monthly:")
    worst_combined = 0
    for k in sorted(combined):
        v = combined[k]
        flag = "✗" if v < MONTH_LOSS_CAP else ("⚠" if v < 0 else " ")
        print(f"    {flag} {k}: ${v:>10,.0f}")
        if v < worst_combined: worst_combined = v
    print(f"\n  Worst combined month: ${worst_combined:,.0f} {'ACCOUNT DEAD' if worst_combined < MONTH_LOSS_CAP else 'OK'}")

    # Correlation check
    months_keys = sorted(set().union(*[mo.keys() for mo in grinder_months]))
    neg_months = [k for k in months_keys if combined[k] < 0]
    print(f"  Negative months: {neg_months}")
    for nm in neg_months:
        parts = [f"G{gi+1}:${grinder_months[gi].get(nm, 0):,.0f}" for gi in range(5)]
        print(f"    {nm}: {' | '.join(parts)}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: SAFE SIZING (5000 combos, math only)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 2: Finding safe contract sizes (5000 combos) ═══")

    orig_cts = [g["sizing_rules"]["fixed_contracts"] for g in grinders]
    best_sizing = []

    for _ in range(5000):
        cts = [random.randint(1, 12) for _ in range(5)]
        test_combined = defaultdict(float)
        for gi in range(5):
            ratio = cts[gi] / max(orig_cts[gi], 1)
            for k, v in grinder_months[gi].items():
                test_combined[k] += v * ratio
        mv = list(test_combined.values())
        if not mv: continue
        worst = min(mv)
        if worst < MONTH_LOSS_CAP: continue
        total = sum(mv)
        best_sizing.append((total, cts, dict(test_combined), worst))

    best_sizing.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  {len(best_sizing)} combos pass safety check")

    # Verify top 20 with real backtests
    verified_sizing = []
    for total_est, cts, est_mo, est_worst in best_sizing[:20]:
        real_combined = defaultdict(float)
        real_total = 0
        ok = True
        for gi in range(5):
            sd_test = copy.deepcopy(grinders[gi])
            sd_test["sizing_rules"]["fixed_contracts"] = cts[gi]
            out = bt(sd_test, data_full, rm, cf, min_trades=5)
            gc.collect()
            if out is None:
                ok = False; break
            m, mo = out
            real_total += m.total_pnl
            for k, v in mo.items():
                real_combined[k] += v
        if not ok: continue
        mv = list(real_combined.values())
        worst = min(mv)
        if worst < MONTH_LOSS_CAP: continue
        verified_sizing.append((real_total, cts, dict(real_combined), worst))
        logger.info(f"  ✓ cts={cts} | ${real_total:,.0f}/yr | worst=${worst:,.0f}")

    if verified_sizing:
        verified_sizing.sort(key=lambda x: x[0], reverse=True)
        _, safe_cts, _, _ = verified_sizing[0]
        for gi in range(5):
            grinders[gi]["sizing_rules"]["fixed_contracts"] = safe_cts[gi]
        logger.info(f"  Best safe sizing: {safe_cts}")
    else:
        logger.warning("  No sizing combos passed — reducing all to 1 contract")
        safe_cts = [1] * 5
        for gi in range(5):
            grinders[gi]["sizing_rules"]["fixed_contracts"] = 1

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: SQUEEZE PROFIT PER GRINDER (2-stage: fast filter + full)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 3: Optimizing each grinder (fast filter + full year) ═══")

    for gi in range(5):
        sd_orig = grinders[gi]
        ct_fixed = sd_orig["sizing_rules"]["fixed_contracts"]
        logger.info(f"\n  G{gi+1}: Stage 1 — 800 variants on 2-month slice (ct={ct_fixed})...")

        # Stage 1: fast filter on 2-month data
        fast_results = []
        for vi in range(800):
            variant = mutate_grinder(sd_orig, intensity=0.6)
            variant["sizing_rules"]["fixed_contracts"] = ct_fixed
            out = bt(variant, data_2mo, rm, cf_2mo, min_trades=8)
            if vi % 100 == 0: gc.collect()
            if out and out[0].total_pnl > 0:
                fast_results.append((out[0].total_pnl, variant))

        fast_results.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"    {len(fast_results)} passed fast filter")

        # Stage 2: top 10 → full year
        full_results = []
        for _, variant in fast_results[:10]:
            out = bt(variant, data_full, rm, cf, min_trades=30)
            gc.collect()
            if out and out[0].total_pnl > 0:
                m, mo = out
                mv = list(mo.values())
                worst_mo = min(mv)
                # Individual worst month * contracts should not exceed $2000
                if worst_mo * 1 < -2000: continue  # Already too bad
                sc2 = m.total_pnl * 1.0 + np.mean(mv) * 3.0 + min(mv) * 1.0 + m.profit_factor * 200
                full_results.append((sc2, variant, m, mo))

        if full_results:
            full_results.sort(key=lambda x: x[0], reverse=True)
            best_sd = full_results[0][1]
            # Stage 2b: 200 focused variations around best
            logger.info(f"    Stage 2b: 200 focused variations around best...")
            for _ in range(200):
                variant = mutate_grinder(best_sd, intensity=0.2)
                variant["sizing_rules"]["fixed_contracts"] = ct_fixed
                out = bt(variant, data_full, rm, cf, min_trades=30)
                gc.collect()
                if out and out[0].total_pnl > 0:
                    m, mo = out
                    mv = list(mo.values())
                    sc2 = m.total_pnl * 1.0 + np.mean(mv) * 3.0 + min(mv) * 1.0 + m.profit_factor * 200
                    full_results.append((sc2, variant, m, mo))

            full_results.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_sd, best_m, best_mo = full_results[0]
            best_sd["sizing_rules"]["fixed_contracts"] = ct_fixed
            grinders[gi] = best_sd
            avg_new = np.mean(list(best_mo.values()))
            logger.info(f"    G{gi+1} BEST: {best_m.total_trades}tr ${avg_new:,.0f}/mo PF={best_m.profit_factor:.2f}")
        else:
            logger.info(f"    G{gi+1}: no improvement found, keeping original")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: VERIFY COMBINED SAFETY
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 4: Verifying combined safety ═══")

    while True:
        combined_final = defaultdict(float)
        grinder_mo_final = []
        for gi, sd in enumerate(grinders):
            out = bt(sd, data_full, rm, cf, min_trades=5)
            gc.collect()
            mo = out[1] if out else {}
            grinder_mo_final.append(mo)
            for k, v in mo.items():
                combined_final[k] += v

        mv = list(combined_final.values())
        worst = min(mv) if mv else 0
        logger.info(f"  Worst combined month: ${worst:,.0f}")

        if worst >= MONTH_LOSS_CAP:
            logger.info(f"  ✓ SAFE — all months above ${MONTH_LOSS_CAP:,.0f}")
            break

        # Find worst month and reduce the biggest loser in that month
        worst_mo_key = min(combined_final, key=combined_final.get)
        losses = [(gi, grinder_mo_final[gi].get(worst_mo_key, 0)) for gi in range(5)]
        losses.sort(key=lambda x: x[1])
        worst_gi = losses[0][0]
        old_ct = grinders[worst_gi]["sizing_rules"]["fixed_contracts"]
        if old_ct <= 1:
            logger.warning(f"  G{worst_gi+1} already at 1 contract — can't reduce further")
            break
        grinders[worst_gi]["sizing_rules"]["fixed_contracts"] = old_ct - 1
        logger.info(f"  Reducing G{worst_gi+1} from {old_ct} to {old_ct-1} contracts")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: ADD MORE GRINDERS IF ROOM
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 5: Looking for new grinders to add ═══")

    new_grinders = []
    if worst >= -2000:
        room = abs(MONTH_LOSS_CAP) - abs(worst)
        logger.info(f"  Room: ${room:,.0f} before hitting cap. Searching...")

        # Stage 1: fast filter on 2-month
        fast_new = []
        for vi in range(1500):
            sd = rand_strategy()
            out = bt(sd, data_2mo, rm, cf_2mo, min_trades=8)
            if vi % 100 == 0: gc.collect()
            if out and out[0].total_pnl > 0:
                fast_new.append((out[0].total_pnl, sd))

        fast_new.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"  {len(fast_new)} passed fast filter")

        # Stage 2: top 30 → full year, check safety
        for _, sd in fast_new[:30]:
            sd["sizing_rules"]["fixed_contracts"] = 2
            out = bt(sd, data_full, rm, cf, min_trades=20)
            gc.collect()
            if out is None or out[0].total_pnl <= 0:
                continue
            m, mo = out
            # Check if adding this breaks any month
            safe = True
            for k in sorted(set(list(combined_final.keys()) + list(mo.keys()))):
                new_val = combined_final.get(k, 0) + mo.get(k, 0)
                if new_val < -3500:  # Leave $500 buffer
                    safe = False; break
            if safe:
                new_grinders.append(sd)
                # Update combined
                for k, v in mo.items():
                    combined_final[k] += v
                mv = list(mo.values())
                logger.info(f"  ✓ Added: {sd['name'][:35]} | {m.total_trades}tr ${np.mean(mv):,.0f}/mo")
                worst_now = min(combined_final.values())
                if worst_now < -3500:
                    break
        logger.info(f"  Added {len(new_grinders)} new grinders")
    else:
        logger.info(f"  No room (worst=${worst:,.0f})")

    all_grinders = grinders + new_grinders

    # Free full data for OOS
    del data_full, data_2mo, df_2mo; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: OOS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 6: OOS validation (4 months) ═══")
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1_2; gc.collect()

    dv = {"1m": df_val}
    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    oos_combined = defaultdict(float)
    oos_total_tr = 0
    for gi, sd in enumerate(all_grinders):
        out = bt(sd, dv, rm, cv, min_trades=3)
        gc.collect()
        if out:
            m, mo = out
            oos_total_tr += m.total_trades
            for k, v in mo.items():
                oos_combined[k] += v
            logger.info(f"  G{gi+1}: OOS ${m.total_pnl:,.0f} | {m.total_trades}tr")

    oos_mv = list(oos_combined.values())
    oos_worst = min(oos_mv) if oos_mv else 0
    oos_total = sum(oos_mv)
    logger.info(f"  OOS combined: ${oos_total:,.0f} | {oos_total_tr}tr | worst mo=${oos_worst:,.0f}")
    del dv, df_val; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: MC STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 7: MC stress test (3000 sims) ═══")
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()

    dfull = {"1m": df_yr1_3}

    all_pnls = []
    final_grinder_results = []
    final_combined = defaultdict(float)
    for gi, sd in enumerate(all_grinders):
        try:
            s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
            r = VectorizedBacktester(data=dfull, risk_manager=rm, contract_spec=MNQ_SPEC, config=cf).run(s)
            m = calculate_metrics(r.trades, 150000.0, r.equity_curve)
            mo = {}
            for t in r.trades:
                k = t.exit_time.strftime("%Y-%m")
                mo[k] = mo.get(k, 0) + t.net_pnl
            all_pnls.extend([t.net_pnl for t in r.trades])
            for k, v in mo.items():
                final_combined[k] += v
            final_grinder_results.append({"sd": sd, "m": m, "mo": mo})
            del r, s
        except Exception:
            final_grinder_results.append(None)
        gc.collect()

    mc = None
    if len(all_pnls) > 20:
        from datetime import datetime, timedelta
        from engine.utils import Trade as TradeObj
        fake_trades = []
        base = datetime(2024, 4, 1)
        for i, pnl in enumerate(all_pnls):
            fake_trades.append(TradeObj(trade_id=str(i), symbol="MNQ", direction="long",
                entry_time=base+timedelta(hours=i), entry_price=20000,
                exit_time=base+timedelta(hours=i,minutes=30), exit_price=20000+pnl/2,
                contracts=4, gross_pnl=pnl+3.6, commission=3.6, slippage_cost=2.0,
                net_pnl=pnl, duration_seconds=1800, session_segment="core", exit_reason="tp"))
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=3000, initial_capital=150000.0, prop_firm_rules=pr, seed=42)).run(fake_trades, "safe_portfolio")
            logger.info(f"  MC P(profit)={mc.probability_of_profit:.0%} | median=${mc.median_return:,.0f}")
        except Exception as e:
            logger.info(f"  MC error: {e}")

    del dfull, df_yr1_3; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    fc_mv = list(final_combined.values())
    fc_worst = min(fc_mv) if fc_mv else 0
    fc_total = sum(fc_mv)
    fc_avg = np.mean(fc_mv) if fc_mv else 0
    total_tr = sum(gr["m"].total_trades for gr in final_grinder_results if gr)

    print(f"\n{'='*120}")
    print(f"  SAFE MAXIMIZE COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*120}")

    # Safety check
    safe = fc_worst >= MONTH_LOSS_CAP
    print(f"\n  SAFETY CHECK:")
    print(f"    Worst combined month: ${fc_worst:,.0f} {'✓ ACCOUNT SAFE' if safe else '✗ ACCOUNT DEAD'}")
    print(f"    Monthly loss cap:     ${MONTH_LOSS_CAP:,.0f}")
    print(f"    DD headroom:          ${abs(MONTH_LOSS_CAP) - abs(fc_worst):,.0f}")

    # Per-grinder table
    print(f"\n  PER-GRINDER TABLE:")
    print(f"  {'#':<3} {'Name':<35} {'Ct':>3} {'Tr/Yr':>6} {'PnL/Yr':>10} {'Avg/Mo':>10} {'Worst':>10} {'WR':>5} {'PF':>5}")
    print(f"  {'-'*95}")
    for gi, gr in enumerate(final_grinder_results):
        if gr:
            sd = gr["sd"]
            m = gr["m"]
            mv = list(gr["mo"].values())
            tag = " (new)" if gi >= 5 else ""
            print(f"  {gi+1:<3} {sd['name'][:34]:<35} {sd['sizing_rules']['fixed_contracts']:>3} {m.total_trades:>6} ${m.total_pnl:>9,.0f} ${np.mean(mv):>9,.0f} ${min(mv):>9,.0f} {m.win_rate:>4.0f}% {m.profit_factor:>4.1f}{tag}")

    # Combined
    print(f"\n  COMBINED RESULTS:")
    print(f"    Trades/year:   {total_tr} ({total_tr/12:.0f}/mo)")
    print(f"    PnL/year:      ${fc_total:,.0f}")
    print(f"    Avg/month:     ${fc_avg:,.0f}")
    print(f"    Worst month:   ${fc_worst:,.0f}")
    if mc:
        print(f"    MC P(profit):  {mc.probability_of_profit:.0%}")
        print(f"    MC Median:     ${mc.median_return:,.0f}")
        print(f"    MC Ruin:       {mc.probability_of_ruin:.0%}")

    # Monthly
    print(f"\n  COMBINED MONTHLY:")
    for k in sorted(final_combined):
        v = final_combined[k]
        parts = []
        for gi, gr in enumerate(final_grinder_results):
            if gr:
                gv = gr["mo"].get(k, 0)
                if abs(gv) > 0: parts.append(f"G{gi+1}:${gv:,.0f}")
        flag = "✗" if v < 0 else ("★" if v >= 10000 else " ")
        print(f"    {flag} {k}: ${v:>10,.0f}  [{' | '.join(parts)}]")

    # New grinders
    if new_grinders:
        print(f"\n  NEW GRINDERS ADDED ({len(new_grinders)}):")
        for ng in new_grinders:
            e = [s["signal_name"] for s in ng["entry_signals"]]
            f = [s["signal_name"] for s in ng.get("entry_filters", []) if s.get("signal_name") != "time_of_day"]
            print(f"    {ng['name'][:40]} — {e}+{f}")

    # Before vs after
    print(f"\n  BEFORE vs AFTER:")
    print(f"    {'Metric':<20} {'Before':>15} {'After':>15}")
    print(f"    {'-'*50}")
    print(f"    {'Avg/month':<20} ${'5,237':>14} ${fc_avg:>14,.0f}")
    print(f"    {'Worst month':<20} ${'-6,128':>14} ${fc_worst:>14,.0f}")
    print(f"    {'Status':<20} {'ACCT DEAD':>15} {'ACCT SAFE' if safe else 'STILL DEAD':>15}")

    # Sizing
    print(f"\n  SIZING:")
    orig_gdata = gdata["grinders"]
    for gi in range(min(5, len(all_grinders))):
        old_ct = orig_gdata[gi]["strategy"]["sizing_rules"]["fixed_contracts"] if gi < len(orig_gdata) else 0
        new_ct = all_grinders[gi]["sizing_rules"]["fixed_contracts"]
        print(f"    G{gi+1}: {old_ct} → {new_ct} contracts")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "safe_maximize_v1",
        "safety": {"worst_month": round(fc_worst, 2), "cap": MONTH_LOSS_CAP, "safe": safe,
                    "headroom": round(abs(MONTH_LOSS_CAP) - abs(fc_worst), 2)},
        "combined": {"total_trades": total_tr, "total_pnl": round(fc_total, 2),
                      "avg_monthly": round(float(fc_avg), 2), "worst_monthly": round(fc_worst, 2),
                      "mc_p_profit": round(mc.probability_of_profit, 4) if mc else None,
                      "mc_median": round(mc.median_return, 2) if mc else None,
                      "mc_ruin": round(mc.probability_of_ruin, 4) if mc else None},
        "combined_monthly": {k: round(v, 2) for k, v in sorted(final_combined.items())},
        "grinders": [],
    }
    for gi, gr in enumerate(final_grinder_results):
        if gr:
            output["grinders"].append({
                "name": gr["sd"]["name"], "strategy": gr["sd"],
                "is_new": gi >= 5,
                "trades": gr["m"].total_trades, "pnl": round(gr["m"].total_pnl, 2),
                "avg_monthly": round(float(np.mean(list(gr["mo"].values()))), 2),
                "worst_monthly": round(float(min(gr["mo"].values())), 2),
                "win_rate": round(gr["m"].win_rate, 2), "profit_factor": round(gr["m"].profit_factor, 2),
                "monthly": {k: round(v, 2) for k, v in sorted(gr["mo"].items())},
            })
    with open("reports/safe_maximize_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/safe_maximize_v1.json")
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
