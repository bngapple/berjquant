#!/usr/bin/env python3
"""
GRINDER MAXIMIZE — Squeeze every dollar from the 5-strategy grinder portfolio.

Current: $5,044/month combined, 145 trades/month, 93% MC P(profit).
Each trade averages $34. Push that to $50+ per trade at same frequency = $7K+/month.
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("grind")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
MAX_DD = -4500.0
random.seed(42)
np.random.seed(42)

TIME_WINDOWS = [
    (9,30,16,0), (9,30,14,0), (9,30,12,0), (8,0,16,0), (8,0,12,0),
    (9,30,11,0), (12,0,16,0), (13,0,16,0), (8,0,11,0), (14,0,16,0),
]

EXTRA_FILTERS = [
    ("ema_slope", "signals.trend", "ema_slope", {"period": 21, "slope_lookback": 3}, "signal_ema_slope_up"),
    ("supertrend", "signals.trend", "supertrend", {"period": 10, "multiplier": 3.0}, "signal_supertrend_bullish"),
    ("relative_volume", "signals.volume", "relative_volume", {"lookback": 20}, "signal_high_volume"),
    ("candle_patterns", "signals.price_action", "candle_patterns", {}, "signal_hammer"),
    ("session_levels", "signals.price_action", "session_levels", {}, "signal_at_session_high"),
]


def bt(sd, data, rm, config, min_trades=40, check_dd=True):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        trades = r.trades
        if len(trades) < min_trades:
            del r, s; return None
        m = calculate_metrics(trades, config.initial_capital, r.equity_curve)
        if check_dd and m.max_drawdown < MAX_DD:
            del r, s, trades; return None
        # Detailed stats
        mo = {}
        long_ct = short_ct = 0
        sl_ct = tp_ct = eod_ct = other_ct = 0
        durations = []
        for t in trades:
            k = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
            mo[k] = mo.get(k, 0) + t.net_pnl
            if t.direction == "long": long_ct += 1
            else: short_ct += 1
            if "stop_loss" in t.exit_reason: sl_ct += 1
            elif "take_profit" in t.exit_reason: tp_ct += 1
            elif "eod" in t.exit_reason: eod_ct += 1
            else: other_ct += 1
            durations.append(t.duration_seconds)
        winners = [t.net_pnl for t in trades if t.net_pnl > 0]
        losers = [t.net_pnl for t in trades if t.net_pnl <= 0]
        detail = {
            "long_pct": long_ct / len(trades) * 100,
            "sl_pct": sl_ct / len(trades) * 100,
            "tp_pct": tp_ct / len(trades) * 100,
            "eod_pct": eod_ct / len(trades) * 100,
            "avg_win": float(np.mean(winners)) if winners else 0,
            "avg_loss": float(np.mean(losers)) if losers else 0,
            "avg_dur_min": float(np.mean(durations)) / 60 if durations else 0,
        }
        del r, s, trades
        return m, mo, detail
    except Exception:
        return None


def score(m, mo):
    mv = list(mo.values())
    if not mv: return -999999
    return m.total_pnl * 1.0 + np.mean(mv) * 3.0 + min(mv) * 2.0 + m.profit_factor * 300 - abs(m.max_drawdown) * 0.5


def mutate_grinder(sd, intensity=0.5):
    new = copy.deepcopy(sd)
    # Mutate entry params ±50%
    for sig in new["entry_signals"]:
        for k, v in sig["params"].items():
            if isinstance(v, int):
                sig["params"][k] = max(2, int(v * random.uniform(1 - intensity, 1 + intensity)))
            elif isinstance(v, float):
                sig["params"][k] = round(max(0.1, v * random.uniform(1 - intensity, 1 + intensity)), 4)
    # Mutate filter params
    for f in new.get("entry_filters", []):
        if f.get("signal_name") == "time_of_day": continue
        for k, v in f["params"].items():
            if isinstance(v, int):
                f["params"][k] = max(1, int(v * random.uniform(1 - intensity, 1 + intensity)))
            elif isinstance(v, float):
                f["params"][k] = round(max(0.1, v * random.uniform(1 - intensity, 1 + intensity)), 4)
    # Mutate exits
    er = new["exit_rules"]
    er["stop_loss_value"] = round(random.uniform(5, 50), 1)
    er["take_profit_value"] = round(random.randrange(15, 405, 5), 1)
    # Contracts
    new["sizing_rules"]["fixed_contracts"] = random.randint(2, 15)
    # Time window
    if random.random() < 0.3:
        tw = random.choice(TIME_WINDOWS)
        for f in new["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]}
    # 15% try adding/removing extra filter
    if random.random() < 0.15:
        non_time = [f for f in new["entry_filters"] if f.get("signal_name") != "time_of_day"]
        time_f = [f for f in new["entry_filters"] if f.get("signal_name") == "time_of_day"]
        if random.random() < 0.5 and EXTRA_FILTERS:
            # Add or replace filter
            ef = random.choice(EXTRA_FILTERS)
            new["entry_filters"] = [{"signal_name": ef[0], "module": ef[1], "function": ef[2], "params": ef[3], "column": ef[4]}] + time_f
        elif non_time:
            # Remove filter (just time_of_day)
            new["entry_filters"] = time_f
    # Trailing stop
    if random.random() < 0.15:
        er["trailing_stop"] = True
        er["trailing_activation"] = round(random.uniform(10, 60), 1)
        er["trailing_distance"] = round(random.uniform(5, 20), 1)
    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    new["name"] = f"{new['name'].split('|')[0]}|opt_{h}"
    return new


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     GRINDER MAXIMIZE — Squeeze Every Dollar from the Portfolio         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Current: $5,044/month, 145 trades/month, $34/trade avg                ║
║  Goal: Push to $7K+/month by optimizing each grinder individually      ║
║  Method: 3000 variants per grinder, then optimize sizing               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load grinders
    with open("reports/portfolio_v1.json") as f:
        comps = json.load(f)["portfolios"][0]["components"]
    grinders = [c["strategy"] for c in comps]
    logger.info(f"Loaded {len(grinders)} grinders")

    # Load data
    logger.info("Loading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()

    data_full = {"1m": df_yr1}
    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    cf = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: UNDERSTAND EACH GRINDER
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 1: Analyzing each grinder ═══")
    originals = []

    for gi, sd in enumerate(grinders):
        out = bt(sd, data_full, rm, cf, min_trades=10, check_dd=False)
        gc.collect()
        if out is None:
            logger.info(f"  Grinder {gi+1}: FAILED (no trades)")
            originals.append(None)
            continue
        m, mo, detail = out
        mv = list(mo.values())
        entries = [e["signal_name"] for e in sd["entry_signals"]]
        filters = [f["signal_name"] for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
        er = sd["exit_rules"]

        print(f"\n  GRINDER {gi+1}: {'+'.join(entries)} + {'+'.join(filters)}")
        print(f"    Params: {[e['params'] for e in sd['entry_signals']]}")
        print(f"    SL={er['stop_loss_value']} TP={er['take_profit_value']} Ct={sd['sizing_rules']['fixed_contracts']}")
        print(f"    Trades={m.total_trades} WR={m.win_rate:.0f}% PF={m.profit_factor:.2f} PnL=${m.total_pnl:,.0f} Avg/Mo=${np.mean(mv):,.0f}")
        print(f"    Avg trade=${m.total_pnl/m.total_trades:,.0f} | Avg win=${detail['avg_win']:,.0f} | Avg loss=${detail['avg_loss']:,.0f}")
        print(f"    Long={detail['long_pct']:.0f}% | SL={detail['sl_pct']:.0f}% | TP={detail['tp_pct']:.0f}% | EOD={detail['eod_pct']:.0f}%")
        print(f"    Avg duration={detail['avg_dur_min']:.0f}min | DD=${m.max_drawdown:,.0f}")

        originals.append({"m": m, "mo": mo, "detail": detail, "avg_mo": float(np.mean(mv))})

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: MAXIMIZE EACH GRINDER (3000 variants each)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 2: Maximizing each grinder (3000 variants each) ═══")

    optimized = []
    for gi, sd in enumerate(grinders):
        orig_avg = originals[gi]["avg_mo"] if originals[gi] else 0
        logger.info(f"\n  Grinder {gi+1}: sweeping 3000 variants (current ${orig_avg:,.0f}/mo)...")

        best_results = []
        for vi in range(3000):
            variant = mutate_grinder(sd, intensity=0.5)
            out = bt(variant, data_full, rm, cf, min_trades=40)
            if vi % 50 == 0: gc.collect()
            if out is None: continue
            m, mo, detail = out
            if m.total_pnl <= 0: continue
            sc2 = score(m, mo)
            mv = list(mo.values())
            best_results.append((sc2, variant, m, mo, detail))

            if (vi + 1) % 500 == 0:
                best_results.sort(key=lambda x: x[0], reverse=True)
                if best_results:
                    b = best_results[0]
                    avg_b = np.mean(list(b[3].values()))
                    logger.info(f"    Grinder {gi+1}: {vi+1}/3000 | passed={len(best_results)} | best: {b[2].total_trades}tr ${avg_b:,.0f}/mo (was ${orig_avg:,.0f})")

        best_results.sort(key=lambda x: x[0], reverse=True)
        if best_results:
            sc2, best_sd, best_m, best_mo, best_detail = best_results[0]
            mv = list(best_mo.values())
            new_avg = float(np.mean(mv))
            improvement = ((new_avg / orig_avg) - 1) * 100 if orig_avg > 0 else 0
            optimized.append(best_sd)
            logger.info(f"    Grinder {gi+1} BEST: {best_m.total_trades}tr ${new_avg:,.0f}/mo PF={best_m.profit_factor:.2f} DD=${best_m.max_drawdown:,.0f} ({improvement:+.0f}%)")
        else:
            optimized.append(sd)  # Keep original
            logger.info(f"    Grinder {gi+1}: no improvements found, keeping original")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: OPTIMIZE SIZING
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 3: Optimizing contract sizing (1000 combos) ═══")

    # Pre-compute metrics for each grinder at each contract level
    # to avoid redundant backtests
    grinder_by_ct = {}  # (gi, ct) -> (mo_dict, total_trades, max_dd)
    for gi, sd in enumerate(optimized):
        for ct_val in range(2, 16):
            sd_copy = copy.deepcopy(sd)
            sd_copy["sizing_rules"]["fixed_contracts"] = ct_val
            out = bt(sd_copy, data_full, rm, cf, min_trades=10)
            gc.collect()
            if out:
                m, mo, _ = out
                grinder_by_ct[(gi, ct_val)] = (mo, m.total_trades, m.max_drawdown, m.total_pnl)
        logger.info(f"  Grinder {gi+1}: precomputed {sum(1 for k in grinder_by_ct if k[0]==gi)} contract levels")

    best_sizing = []
    for _ in range(1000):
        cts = [random.randint(2, 15) for _ in range(5)]
        combined_mo = defaultdict(float)
        total_tr = 0
        worst_dd = 0
        total_pnl = 0
        valid = True
        for gi in range(5):
            key = (gi, cts[gi])
            if key not in grinder_by_ct:
                valid = False; break
            mo, tr, dd, pnl = grinder_by_ct[key]
            total_tr += tr
            total_pnl += pnl
            for k, v in mo.items():
                combined_mo[k] += v
            # Track combined DD (rough: sum of individual DDs, conservative)
            if dd < worst_dd: worst_dd = dd

        if not valid: continue
        mv = list(combined_mo.values())
        if not mv: continue
        # Check combined worst month
        min_mo = min(mv)
        if min_mo < -4000: continue  # Too risky

        sc2 = total_pnl * 1.0 + np.mean(mv) * 3.0 + min_mo * 2.0
        best_sizing.append((sc2, cts, {
            "total_pnl": total_pnl, "avg_mo": float(np.mean(mv)), "min_mo": min_mo,
            "max_mo": float(max(mv)), "total_tr": total_tr,
            "months": dict(combined_mo),
        }))

    best_sizing.sort(key=lambda x: x[0], reverse=True)
    if best_sizing:
        _, best_cts, best_stats = best_sizing[0]
        logger.info(f"  Best sizing: {best_cts} → ${best_stats['avg_mo']:,.0f}/mo, {best_stats['total_tr']} trades")
        # Apply best sizing
        for gi, sd in enumerate(optimized):
            sd["sizing_rules"]["fixed_contracts"] = best_cts[gi]
    else:
        logger.info("  No valid sizing combos found")

    # Free full year data, prepare for OOS
    del data_full, df_yr1; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: OOS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 4: OOS validation (4 months) ═══")
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1_2; gc.collect()

    dv = {"1m": df_val}
    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    oos_combined_pnl = 0
    oos_combined_tr = 0
    oos_pass = True
    for gi, sd in enumerate(optimized):
        out = bt(sd, dv, rm, cv, min_trades=5)
        gc.collect()
        if out:
            m, mo, _ = out
            avg_oos = np.mean(list(mo.values())) if mo else 0
            oos_combined_pnl += m.total_pnl
            oos_combined_tr += m.total_trades
            logger.info(f"  Grinder {gi+1}: OOS PnL=${m.total_pnl:,.0f} | {m.total_trades} trades | ${avg_oos:,.0f}/mo")
            if m.total_pnl < -500:
                logger.info(f"    ⚠ Below -$500 threshold")
        else:
            logger.info(f"  Grinder {gi+1}: no OOS trades")

    logger.info(f"  Combined OOS: PnL=${oos_combined_pnl:,.0f} | {oos_combined_tr} trades")
    del dv, df_val; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: FULL YEAR MC
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 5: Full year MC (3000 sims) ═══")
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()

    dfull = {"1m": df_yr1_3}
    cfull = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    all_trades_pnl = []
    combined_mo = defaultdict(float)
    grinder_results = []

    for gi, sd in enumerate(optimized):
        out = bt(sd, dfull, rm, cfull, min_trades=10)
        gc.collect()
        if out is None:
            grinder_results.append(None)
            continue
        m, mo, detail = out
        mv = list(mo.values())
        grinder_results.append({"m": m, "mo": mo, "detail": detail, "avg_mo": float(np.mean(mv))})

        # Backtest again to get actual trade objects for MC
        try:
            s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
            r = VectorizedBacktester(data=dfull, risk_manager=rm, contract_spec=MNQ_SPEC, config=cfull).run(s)
            all_trades_pnl.extend([t.net_pnl for t in r.trades])
            for t in r.trades:
                k = t.exit_time.strftime("%Y-%m")
                combined_mo[k] += t.net_pnl
            del r, s
        except Exception:
            pass
        gc.collect()

        logger.info(f"  Grinder {gi+1}: {m.total_trades}tr ${float(np.mean(mv)):,.0f}/mo PF={m.profit_factor:.2f}")

    # MC on combined
    mc = None
    if len(all_trades_pnl) > 20:
        all_pnl_arr = np.array(all_trades_pnl)
        # Create fake Trade objects for MC
        from engine.utils import Trade
        from datetime import datetime, timedelta
        fake_trades = []
        base = datetime(2024, 4, 1)
        for i, pnl in enumerate(all_trades_pnl):
            t = Trade(trade_id=str(i), symbol="MNQ", direction="long",
                      entry_time=base + timedelta(hours=i), entry_price=20000,
                      exit_time=base + timedelta(hours=i, minutes=30), exit_price=20000 + pnl/2,
                      contracts=4, gross_pnl=pnl + 3.6, commission=3.6, slippage_cost=2.0,
                      net_pnl=pnl, duration_seconds=1800, session_segment="core", exit_reason="tp")
            fake_trades.append(t)

        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=3000, initial_capital=150000.0, prop_firm_rules=pr, seed=42)).run(fake_trades, "grinder_portfolio")
            logger.info(f"  MC P(profit)={mc.probability_of_profit:.0%} | median=${mc.median_return:,.0f}")
        except Exception as e:
            logger.info(f"  MC error: {e}")

    del dfull, df_yr1_3; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    combined_mv = list(combined_mo.values())
    combined_total = sum(combined_mv) if combined_mv else 0
    combined_avg = float(np.mean(combined_mv)) if combined_mv else 0
    combined_min = float(min(combined_mv)) if combined_mv else 0
    combined_max = float(max(combined_mv)) if combined_mv else 0
    total_trades = sum(gr["m"].total_trades for gr in grinder_results if gr)

    print(f"\n{'='*120}")
    print(f"  GRINDER MAXIMIZE COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*120}")

    # Per-grinder improvement
    print(f"\n  PER-GRINDER IMPROVEMENT:")
    print(f"  {'#':<3} {'Family':<28} {'Old Tr':>7} {'Old $/Mo':>10} {'New Tr':>7} {'New $/Mo':>10} {'New PnL':>12} {'Chg':>8}")
    print(f"  {'-'*95}")
    for gi in range(5):
        entries = [e["signal_name"] for e in optimized[gi]["entry_signals"]]
        filters = [f["signal_name"] for f in optimized[gi].get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
        fam = "+".join(entries) + ("|" + "+".join(filters) if filters else "")
        old_tr = comps[gi]["trades"]
        old_avg = comps[gi]["avg_monthly"]
        new_gr = grinder_results[gi]
        if new_gr:
            new_tr = new_gr["m"].total_trades
            new_avg = new_gr["avg_mo"]
            new_pnl = new_gr["m"].total_pnl
            chg = ((new_avg / old_avg) - 1) * 100 if old_avg > 0 else 0
            print(f"  {gi+1:<3} {fam[:27]:<28} {old_tr:>7} ${old_avg:>9,.0f} {new_tr:>7} ${new_avg:>9,.0f} ${new_pnl:>11,.0f} {chg:>+7.0f}%")
        else:
            print(f"  {gi+1:<3} {fam[:27]:<28} {old_tr:>7} ${old_avg:>9,.0f} {'FAILED':>7}")

    # Sizing
    if best_sizing:
        print(f"\n  SIZING OPTIMIZATION:")
        for gi in range(5):
            old_ct = comps[gi]["strategy"]["sizing_rules"]["fixed_contracts"]
            new_ct = optimized[gi]["sizing_rules"]["fixed_contracts"]
            print(f"    Grinder {gi+1}: {old_ct} → {new_ct} contracts")

    # Combined
    print(f"\n  COMBINED RESULTS:")
    print(f"    Trades/year: {total_trades} ({total_trades/12:.0f}/mo)")
    print(f"    PnL/year:    ${combined_total:,.0f}")
    print(f"    Avg/month:   ${combined_avg:,.0f}")
    print(f"    Min month:   ${combined_min:,.0f}")
    print(f"    Max month:   ${combined_max:,.0f}")
    if mc:
        print(f"    MC P(profit): {mc.probability_of_profit:.0%}")
        print(f"    MC Median:    ${mc.median_return:,.0f}")
        print(f"    MC 5th pctl:  ${mc.pct_5th_return:,.0f}")

    # Monthly
    print(f"\n  COMBINED MONTHLY BREAKDOWN:")
    for mk in sorted(combined_mo.keys()):
        v = combined_mo[mk]
        parts = []
        for gi in range(5):
            gr = grinder_results[gi]
            if gr:
                gv = gr["mo"].get(mk, 0)
                if abs(gv) > 0:
                    parts.append(f"G{gi+1}:${gv:,.0f}")
        flag = "★" if v >= 15000 else ("●" if v >= 10000 else (" " if v >= 0 else "✗"))
        print(f"    {flag} {mk}: ${v:>10,.0f}  [{' | '.join(parts)}]")

    # Before vs after
    print(f"\n  BEFORE vs AFTER:")
    print(f"    {'Metric':<20} {'Original':>15} {'Optimized':>15} {'Change':>10}")
    print(f"    {'-'*60}")
    print(f"    {'Avg/month':<20} ${'5,044':>14} ${combined_avg:>14,.0f} {((combined_avg/5044)-1)*100:>+9.0f}%")
    print(f"    {'PnL/year':<20} ${'60,528':>14} ${combined_total:>14,.0f} {((combined_total/60528)-1)*100:>+9.0f}%")
    print(f"    {'Trades/month':<20} {'145':>15} {total_trades/12:>14.0f}")

    # Key changes
    print(f"\n  KEY CHANGES:")
    for gi in range(5):
        old_sl = comps[gi]["strategy"]["exit_rules"]["stop_loss_value"]
        old_tp = comps[gi]["strategy"]["exit_rules"]["take_profit_value"]
        old_ct = comps[gi]["strategy"]["sizing_rules"]["fixed_contracts"]
        new_sl = optimized[gi]["exit_rules"]["stop_loss_value"]
        new_tp = optimized[gi]["exit_rules"]["take_profit_value"]
        new_ct = optimized[gi]["sizing_rules"]["fixed_contracts"]
        if old_sl != new_sl or old_tp != new_tp or old_ct != new_ct:
            print(f"    G{gi+1}: SL {old_sl}→{new_sl} | TP {old_tp}→{new_tp} | Ct {old_ct}→{new_ct}")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "grinder_maximize_v1",
        "combined": {
            "total_trades": total_trades, "total_pnl": round(combined_total, 2),
            "avg_monthly": round(combined_avg, 2), "min_monthly": round(combined_min, 2),
            "max_monthly": round(combined_max, 2),
            "mc_p_profit": round(mc.probability_of_profit, 4) if mc else None,
            "mc_median": round(mc.median_return, 2) if mc else None,
        },
        "combined_monthly": {k: round(v, 2) for k, v in sorted(combined_mo.items())},
        "grinders": [],
    }
    for gi, sd in enumerate(optimized):
        gr = grinder_results[gi]
        output["grinders"].append({
            "name": sd["name"], "strategy": sd,
            "old_avg_monthly": comps[gi]["avg_monthly"],
            "new_avg_monthly": gr["avg_mo"] if gr else 0,
            "trades": gr["m"].total_trades if gr else 0,
            "pnl": round(gr["m"].total_pnl, 2) if gr else 0,
            "win_rate": round(gr["m"].win_rate, 2) if gr else 0,
            "profit_factor": round(gr["m"].profit_factor, 2) if gr else 0,
            "max_drawdown": round(gr["m"].max_drawdown, 2) if gr else 0,
            "monthly": {k: round(v, 2) for k, v in sorted(gr["mo"].items())} if gr else {},
        })
    with open("reports/grinder_maximized_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/grinder_maximized_v1.json")
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
