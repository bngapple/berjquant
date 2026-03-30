#!/usr/bin/env python3
"""
HTF Swing v3 Hybrid v2 — RSI/IB optimized + MOM original (ATR=1.0).

3-way comparison: Current v3 | Hybrid (ATR=1.4) | Hybrid v2 (ATR=1.0)

Usage:
    python3 run_htf_swing_v3_hybrid_v2.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

import run_htf_swing as _base
from run_htf_swing import (
    load_and_resample, extract_arrays, backtest, rt_cost,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0
FLATTEN_TIME = 1645
CONTRACTS = 3

# ── Three param sets ────────────────────────────────────────────────

CURRENT = {
    "RSI": {"period": 7,  "ob": 70, "os": 30, "sl_pts": 15,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 20,  "tp_pts": 120, "hold": 10},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}

HYBRID_V1 = {
    "RSI": {"period": 5,  "ob": 65, "os": 35, "sl_pts": 10,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 10,  "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.4, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}

HYBRID_V2 = {
    "RSI": {"period": 5,  "ob": 65, "os": 35, "sl_pts": 10,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 10,  "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}

CONFIGS = [
    ("Current v3",       CURRENT),
    ("Hybrid (ATR=1.4)", HYBRID_V1),
    ("Hybrid v2 (1.0)",  HYBRID_V2),
]


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def daily_pnl(trades):
    d = defaultdict(float)
    for t in trades:
        d[str(t.entry_time)[:10]] += t.net_pnl
    return dict(d)


def monthly_pnl(trades):
    m = defaultdict(float)
    for t in trades:
        m[str(t.entry_time)[:7]] += t.net_pnl
    return dict(m)


def full_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "best_month": 0, "worst_day": 0, "best_day": 0, "max_dd": 0,
                "bars_mean": 0, "monthly": {}, "n_months": 0, "months_pos": 0,
                "sharpe": 0, "trades_per_day": 0}
    pnls = [t.net_pnl for t in trades]
    bars = [t.bars_held for t in trades]
    w = sum(1 for p in pnls if p > 0)
    monthly = defaultdict(float)
    daily = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
        daily[str(t.entry_time)[:10]] += t.net_pnl
    cum = 0; peak = 0; mdd = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    nm = max(len(monthly), 1)
    nd = max(len(daily), 1)
    total = sum(pnls)
    dv = list(daily.values())
    sharpe = float(np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0
    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": total / nm,
        "worst_month": min(monthly.values()) if monthly else 0,
        "best_month": max(monthly.values()) if monthly else 0,
        "worst_day": min(daily.values()) if daily else 0,
        "best_day": max(daily.values()) if daily else 0,
        "max_dd": mdd, "bars_mean": np.mean(bars),
        "monthly": dict(monthly), "n_months": nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "sharpe": sharpe,
        "trades_per_day": len(trades) / nd,
    }


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass_rate": 0, "blowup": 1, "med_days": 0, "p95_days": 0}
    passed = 0; blown = 0; dtp = []
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; p = False; ok = True; dc = 0
        for idx in order:
            dp = sum(days[idx])
            if dp < DAILY_LIMIT:
                dp = DAILY_LIMIT
            cum += dp; dc += 1; peak = max(peak, cum)
            if cum - peak <= MLL:
                ok = False; break
            if not p and cum >= EVAL_TARGET:
                p = True; dtp.append(dc)
        if not ok:
            blown += 1
        if p and ok:
            passed += 1
    da = np.array(dtp) if dtp else np.array([0])
    return {
        "pass_rate": passed / n_sims,
        "blowup": blown / n_sims,
        "med_days": int(np.median(da)),
        "p95_days": int(np.percentile(da, 95)) if len(da) > 0 else 0,
    }


def run_system(df, params, contracts=CONTRACTS, flatten=FLATTEN_TIME):
    o, h, l, c, ts, hm = extract_arrays(df)
    per_strat = {}
    p = params["RSI"]
    sigs = sig_rsi_extreme(df, p["period"], p["ob"], p["os"])
    per_strat["RSI"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], contracts, "RSI", flatten)
    p = params["IB"]
    sigs = sig_ib_breakout(df, p["ib_filter"])[0]
    per_strat["IB"] = backtest(o, h, l, c, ts, hm, sigs,
                                pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                p["hold"], contracts, "IB", flatten)
    p = params["MOM"]
    sigs = sig_momentum_bar(df, p["atr_mult"], p["vol_mult"])[0]
    per_strat["MOM"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], contracts, "MOM", flatten)
    all_trades = []
    for t in per_strat.values():
        all_trades.extend(t)
    return all_trades, per_strat


def run_with_slippage(df, params, slip_ticks, contracts=CONTRACTS):
    orig_slip_pts = _base.SLIP_PTS
    orig_slip_ticks = _base.SLIP_TICKS
    _base.SLIP_PTS = slip_ticks * TICK_SIZE
    _base.SLIP_TICKS = slip_ticks
    try:
        trades, _ = run_system(df, params, contracts)
    finally:
        _base.SLIP_PTS = orig_slip_pts
        _base.SLIP_TICKS = orig_slip_ticks
    return trades


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 75)
    print("  HTF SWING v3 HYBRID v2 — RSI/IB Optimized + MOM Original")
    print("═" * 75)
    print(f"  Only change from Hybrid v1: MOM ATR_mult 1.4 → 1.0")
    print(f"  Session: LucidFlex (flatten {FLATTEN_TIME}), {CONTRACTS} MNQ/strategy")
    print(f"  Cost per RT: ${rt_cost(CONTRACTS):.2f}")

    # ── Load data ───────────────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  LOADING DATA")
    print("━" * 75)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main (2024-2026)")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind (2022-2024)")

    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
    Y1_END = datetime(2025, 3, 1, tzinfo=_ET)

    df_main = main_data["15m"]
    yr1 = df_main.filter(pl.col("timestamp") < Y1_END)
    yr2 = df_main.filter(pl.col("timestamp") >= Y1_END)
    bl = blind_data["15m"]

    print(f"  Y1: {len(yr1):,} bars | Y2: {len(yr2):,} bars | Blind: {len(bl):,} bars")

    has_8yr = True
    try:
        df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
        df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
        df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
        combined_1m = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        combined_1m = combined_1m.filter(pl.col("close") > 0)
        df_8yr = resample_15m_rth(combined_1m)
        print(f"  8yr: {len(df_8yr):,} bars")
        del combined_1m, df1, df2, df3
        gc.collect()
    except Exception as e:
        print(f"  8yr data not available: {e}")
        has_8yr = False

    # ═════════════════════════════════════════════════════════════════
    # RUN ALL 3 CONFIGS ON ALL PERIODS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  RUNNING BACKTESTS")
    print("━" * 75)

    periods = [("Y1", yr1), ("Y2", yr2), ("Blind", bl)]
    if has_8yr:
        periods.append(("8yr", df_8yr))

    # results[config_label][period] = {metrics, trades, per_strat}
    results = {label: {} for label, _ in CONFIGS}

    for period_label, df in periods:
        print(f"\n  {period_label}:")
        for cfg_label, params in CONFIGS:
            trades, strats = run_system(df, params)
            m = full_metrics(trades)
            results[cfg_label][period_label] = {
                "metrics": m, "trades": trades,
                "per_strat": {k: full_metrics(v) for k, v in strats.items()},
            }
            print(f"    {cfg_label:<20} {m['n']:>5} trades  ${m['pnl']:>+10,.0f}  "
                  f"{m['wr']:.1f}% WR  Sharpe {m['sharpe']:.1f}")
        gc.collect()

    # ═════════════════════════════════════════════════════════════════
    # 3-WAY HEAD-TO-HEAD
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  HEAD-TO-HEAD: 3-WAY COMPARISON")
    print("═" * 75)

    w = 16  # column width
    header = f"  {'Period':<10}"
    for cfg_label, _ in CONFIGS:
        header += f" {cfg_label:>{w}}"
    print(f"\n{header}")
    print(f"  {'─'*10}" + f" {'─'*w}" * 3)

    period_labels = ["Y1", "Y2", "Blind"] + (["8yr"] if has_8yr else [])
    for period_label in period_labels:
        row = f"  {period_label:<10}"
        for cfg_label, _ in CONFIGS:
            avg = results[cfg_label][period_label]["metrics"]["monthly_avg"]
            row += f" ${avg:>{w-1},.0f}"
        print(row)

    # Months profitable
    print(f"\n  {'Months +':<10}", end="")
    for cfg_label, _ in CONFIGS:
        m = results[cfg_label]["Y2"]["metrics"]
        print(f" {m['months_pos']:>{w-1}}/{m['n_months']}", end="  ")
    print()

    # Detailed metrics for Y2
    print(f"\n  Y2 OOS Detail:")
    print(f"  {'Metric':<14}", end="")
    for cfg_label, _ in CONFIGS:
        print(f" {cfg_label:>{w}}", end="")
    print()
    print(f"  {'─'*14}" + f" {'─'*w}" * 3)

    metric_fns = [
        ("Trades/mo",  lambda m: f"{m['n']/m['n_months']:.0f}"),
        ("Win rate",   lambda m: f"{m['wr']:.1f}%"),
        ("$/month",    lambda m: f"${m['monthly_avg']:>+,.0f}"),
        ("Worst day",  lambda m: f"${m['worst_day']:>+,.0f}"),
        ("Max DD",     lambda m: f"${m['max_dd']:>+,.0f}"),
        ("Sharpe",     lambda m: f"{m['sharpe']:.1f}"),
    ]
    for label, fn in metric_fns:
        row = f"  {label:<14}"
        for cfg_label, _ in CONFIGS:
            val = fn(results[cfg_label]["Y2"]["metrics"])
            row += f" {val:>{w}}"
        print(row)

    # ═════════════════════════════════════════════════════════════════
    # PER-STRATEGY BREAKDOWN
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  PER-STRATEGY BREAKDOWN — Y2 OOS")
    print("━" * 75)

    for strat_name in ["RSI", "IB", "MOM"]:
        print(f"\n  {strat_name}:")
        print(f"  {'Config':<20} {'$/month':>10} {'Trades/mo':>10} {'Win rate':>9} {'Sharpe':>7}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*9} {'─'*7}")
        for cfg_label, _ in CONFIGS:
            s = results[cfg_label]["Y2"]["per_strat"][strat_name]
            tpm = s["n"] / s["n_months"] if s["n_months"] > 0 else 0
            print(f"  {cfg_label:<20} ${s['monthly_avg']:>+9,.0f} {tpm:>10.0f} "
                  f"{s['wr']:>8.1f}% {s['sharpe']:>6.1f}")

    print(f"\n{'━' * 75}")
    print("  PER-STRATEGY BREAKDOWN — BLIND")
    print("━" * 75)

    for strat_name in ["RSI", "IB", "MOM"]:
        print(f"\n  {strat_name}:")
        print(f"  {'Config':<20} {'$/month':>10} {'Trades/mo':>10} {'Win rate':>9} {'Sharpe':>7}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*9} {'─'*7}")
        for cfg_label, _ in CONFIGS:
            s = results[cfg_label]["Blind"]["per_strat"][strat_name]
            tpm = s["n"] / s["n_months"] if s["n_months"] > 0 else 0
            print(f"  {cfg_label:<20} ${s['monthly_avg']:>+9,.0f} {tpm:>10.0f} "
                  f"{s['wr']:>8.1f}% {s['sharpe']:>6.1f}")

    # ═════════════════════════════════════════════════════════════════
    # MOM ATR RECOVERY CHECK
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  MOM ATR RECOVERY CHECK")
    print("━" * 75)

    for period in ["Y2", "Blind"]:
        m_v1 = results["Hybrid (ATR=1.4)"][period]["per_strat"]["MOM"]
        m_v2 = results["Hybrid v2 (1.0)"][period]["per_strat"]["MOM"]
        m_cur = results["Current v3"][period]["per_strat"]["MOM"]
        delta = m_v2["monthly_avg"] - m_v1["monthly_avg"]
        print(f"\n  {period} MOM:")
        print(f"    ATR=1.4 (Hybrid v1): ${m_v1['monthly_avg']:>+,.0f}/mo, "
              f"{m_v1['n']/m_v1['n_months']:.0f} trades/mo, {m_v1['wr']:.1f}% WR")
        print(f"    ATR=1.0 (Hybrid v2): ${m_v2['monthly_avg']:>+,.0f}/mo, "
              f"{m_v2['n']/m_v2['n_months']:.0f} trades/mo, {m_v2['wr']:.1f}% WR")
        print(f"    ATR=1.0 (Current):   ${m_cur['monthly_avg']:>+,.0f}/mo, "
              f"{m_cur['n']/m_cur['n_months']:.0f} trades/mo, {m_cur['wr']:.1f}% WR")
        print(f"    Recovery from 1.4→1.0: ${delta:>+,.0f}/mo")

    # ═════════════════════════════════════════════════════════════════
    # MONTH-BY-MONTH (Y2 OOS only, 3-way)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  MONTH-BY-MONTH — Y2 OOS")
    print("━" * 75)

    all_y2_months = set()
    for cfg_label, _ in CONFIGS:
        all_y2_months.update(results[cfg_label]["Y2"]["metrics"]["monthly"].keys())
    all_y2_months = sorted(all_y2_months)

    print(f"\n  {'Month':<10} {'Current':>10} {'Hyb v1':>10} {'Hyb v2':>10} {'v2-Cur':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for m in all_y2_months:
        cv = results["Current v3"]["Y2"]["metrics"]["monthly"].get(m, 0)
        v1 = results["Hybrid (ATR=1.4)"]["Y2"]["metrics"]["monthly"].get(m, 0)
        v2 = results["Hybrid v2 (1.0)"]["Y2"]["metrics"]["monthly"].get(m, 0)
        delta = v2 - cv
        mark = "+" if v2 > 0 else "-"
        print(f"  {m:<10} ${cv:>+9,.0f} ${v1:>+9,.0f} ${v2:>+9,.0f} ${delta:>+9,.0f}  {mark}")

    # ═════════════════════════════════════════════════════════════════
    # MONTE CARLO (Hybrid v2 on Y2)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  MONTE CARLO (5,000 sims, Y2 trades)")
    print("━" * 75)

    mc = {}
    for cfg_label, _ in CONFIGS:
        y2t = results[cfg_label]["Y2"]["trades"]
        mc[cfg_label] = {
            "baseline": run_mc(y2t, 5000, 1.0),
            "70pct": run_mc(y2t, 5000, 0.70),
        }

    print(f"\n  {'System':<20} {'Pass':>7} {'Blowup':>7} {'Med d':>6} {'P95 d':>6} {'70% Pass':>9}")
    print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*9}")
    for cfg_label, _ in CONFIGS:
        b = mc[cfg_label]["baseline"]
        c70 = mc[cfg_label]["70pct"]
        print(f"  {cfg_label:<20} {b['pass_rate']:>6.0%} {b['blowup']:>6.0%} "
              f"{b['med_days']:>6} {b['p95_days']:>6} {c70['pass_rate']:>8.0%}")

    # ═════════════════════════════════════════════════════════════════
    # SLIPPAGE SENSITIVITY (Hybrid v2)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  SLIPPAGE SENSITIVITY — Hybrid v2")
    print("━" * 75)

    print(f"\n  {'Slippage':<10} {'Y2 $/mo':>10} {'Blind/mo':>10} {'WR':>7} {'Sharpe':>7}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*7}")
    slip_results = {}
    for slip in [2, 3, 4]:
        y2t = run_with_slippage(yr2, HYBRID_V2, slip)
        blt = run_with_slippage(bl, HYBRID_V2, slip)
        y2m = full_metrics(y2t)
        blm = full_metrics(blt)
        slip_results[slip] = {"y2": y2m, "blind": blm}
        print(f"  {slip} ticks    ${y2m['monthly_avg']:>+9,.0f} ${blm['monthly_avg']:>+9,.0f} "
              f"{y2m['wr']:>6.1f}% {y2m['sharpe']:>6.1f}")

    s2 = slip_results[2]["y2"]["pnl"]
    s4 = slip_results[4]["y2"]["pnl"]
    deg = (s4 - s2) / abs(s2) * 100 if s2 != 0 else 0
    print(f"\n  Degradation 2→4 ticks: {deg:+.1f}% of Y2 P&L")

    # ═════════════════════════════════════════════════════════════════
    # PROP FIRM CHECK
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  PROP FIRM CHECK — LucidFlex 150K (Hybrid v2)")
    print("━" * 75)

    v2_y2 = results["Hybrid v2 (1.0)"]["Y2"]["metrics"]
    v2_bl = results["Hybrid v2 (1.0)"]["Blind"]["metrics"]
    worst_day = min(v2_y2["worst_day"], v2_bl["worst_day"])
    max_dd = min(v2_y2["max_dd"], v2_bl["max_dd"])

    print(f"\n  Worst day:  ${worst_day:>+10,.0f}  vs  -$3,000  {'PASS' if worst_day > -3000 else 'FAIL'}")
    print(f"  Max DD:     ${max_dd:>+10,.0f}  vs  -$4,500  {'PASS' if max_dd > -4500 else 'FAIL'}")

    # ═════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  VERDICT")
    print("═" * 75)

    v2_y2_avg = results["Hybrid v2 (1.0)"]["Y2"]["metrics"]["monthly_avg"]
    v1_y2_avg = results["Hybrid (ATR=1.4)"]["Y2"]["metrics"]["monthly_avg"]
    cur_y2_avg = results["Current v3"]["Y2"]["metrics"]["monthly_avg"]

    v2_bl_avg = results["Hybrid v2 (1.0)"]["Blind"]["metrics"]["monthly_avg"]
    v1_bl_avg = results["Hybrid (ATR=1.4)"]["Blind"]["metrics"]["monthly_avg"]
    cur_bl_avg = results["Current v3"]["Blind"]["metrics"]["monthly_avg"]

    mom_v1_y2 = results["Hybrid (ATR=1.4)"]["Y2"]["per_strat"]["MOM"]["monthly_avg"]
    mom_v2_y2 = results["Hybrid v2 (1.0)"]["Y2"]["per_strat"]["MOM"]["monthly_avg"]
    mom_recovery = mom_v2_y2 - mom_v1_y2

    print(f"\n  Y2 OOS monthly avg:")
    print(f"    Current v3:       ${cur_y2_avg:>+10,.0f}")
    print(f"    Hybrid v1 (1.4):  ${v1_y2_avg:>+10,.0f}")
    print(f"    Hybrid v2 (1.0):  ${v2_y2_avg:>+10,.0f}")

    print(f"\n  Blind monthly avg:")
    print(f"    Current v3:       ${cur_bl_avg:>+10,.0f}")
    print(f"    Hybrid v1 (1.4):  ${v1_bl_avg:>+10,.0f}")
    print(f"    Hybrid v2 (1.0):  ${v2_bl_avg:>+10,.0f}")

    print(f"\n  MOM recovery (Y2): ${mom_recovery:>+,.0f}/mo from ATR 1.4→1.0")

    mc_v2 = mc["Hybrid v2 (1.0)"]["baseline"]
    mc_v1 = mc["Hybrid (ATR=1.4)"]["baseline"]
    mc_cur = mc["Current v3"]["baseline"]
    print(f"\n  MC pass rates: Current {mc_cur['pass_rate']:.0%} | "
          f"Hybrid v1 {mc_v1['pass_rate']:.0%} | Hybrid v2 {mc_v2['pass_rate']:.0%}")

    # Pick winner
    configs_ranked = sorted(
        [("Current v3", cur_y2_avg, cur_bl_avg, mc_cur["pass_rate"]),
         ("Hybrid v1", v1_y2_avg, v1_bl_avg, mc_v1["pass_rate"]),
         ("Hybrid v2", v2_y2_avg, v2_bl_avg, mc_v2["pass_rate"])],
        key=lambda x: x[2], reverse=True  # rank by blind
    )
    winner = configs_ranked[0][0]
    print(f"\n  BEST ON BLIND DATA: {winner}")
    print(f"    ${configs_ranked[0][2]:>+,.0f}/mo blind, ${configs_ranked[0][1]:>+,.0f}/mo Y2, "
          f"{configs_ranked[0][3]:.0%} MC pass")

    if v2_y2_avg > v1_y2_avg and v2_bl_avg > v1_bl_avg:
        print(f"\n  Hybrid v2 beats Hybrid v1 on BOTH Y2 and Blind.")
        print(f"  MOM ATR=1.0 is strictly better than ATR=1.4. Deploy Hybrid v2.")
    elif v2_y2_avg > v1_y2_avg or v2_bl_avg > v1_bl_avg:
        print(f"\n  Mixed result. Hybrid v2 wins on {'Y2' if v2_y2_avg > v1_y2_avg else 'Blind'} "
              f"but loses on {'Blind' if v2_y2_avg > v1_y2_avg else 'Y2'}.")
    else:
        print(f"\n  Hybrid v1 (ATR=1.4) is better on both periods. Keep ATR=1.4.")

    # ═════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════
    report = {
        "timestamp": str(datetime.now()),
        "params": {
            "current": {k: dict(v) for k, v in CURRENT.items()},
            "hybrid_v1": {k: dict(v) for k, v in HYBRID_V1.items()},
            "hybrid_v2": {k: dict(v) for k, v in HYBRID_V2.items()},
        },
        "head_to_head": {},
        "per_strategy": {},
        "mc": {},
        "slippage": {},
        "winner": winner,
    }

    for cfg_label, _ in CONFIGS:
        report["head_to_head"][cfg_label] = {}
        for period in period_labels:
            m = results[cfg_label][period]["metrics"]
            report["head_to_head"][cfg_label][period] = {
                k: v for k, v in m.items() if k != "monthly"
            }

    for strat_name in ["RSI", "IB", "MOM"]:
        report["per_strategy"][strat_name] = {}
        for cfg_label, _ in CONFIGS:
            report["per_strategy"][strat_name][cfg_label] = {
                "y2": {k: v for k, v in results[cfg_label]["Y2"]["per_strat"][strat_name].items() if k != "monthly"},
                "blind": {k: v for k, v in results[cfg_label]["Blind"]["per_strat"][strat_name].items() if k != "monthly"},
            }

    for cfg_label, _ in CONFIGS:
        report["mc"][cfg_label] = mc[cfg_label]

    for slip, sr in slip_results.items():
        report["slippage"][f"{slip}_ticks"] = {
            "y2_monthly_avg": sr["y2"]["monthly_avg"],
            "blind_monthly_avg": sr["blind"]["monthly_avg"],
        }

    out = REPORTS_DIR / "htf_swing_hybrid_v2.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 75)


if __name__ == "__main__":
    main()
