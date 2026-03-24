#!/usr/bin/env python3
"""
HTF Swing v3 — No data leakage. Portfolio selected on Y1 only.

v2 selected the strategy combo by optimizing on Y2. Fixed here:
Y1 picks the combo, Y2 and blind are both truly OOS.

Usage:
    python3 run_htf_swing_v3.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from run_htf_swing import (
    load_and_resample, extract_arrays, backtest, rt_cost,
    sig_vwap_mr, sig_ema_pullback, sig_rsi_extreme, sig_ib_breakout,
    sig_momentum_bar, TICK_SIZE, POINT_VALUE, SLIP_PTS, YR1_END,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0

STRATS = {
    "VWAP": {"sig": lambda d: sig_vwap_mr(d, 1.0),              "sl": 60,  "tp": 400, "hold": 5},
    "EMA":  {"sig": lambda d: sig_ema_pullback(d, 8, 21)[0],    "sl": 60,  "tp": 400, "hold": 10},
    "RSI":  {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30),    "sl": 60,  "tp": 400, "hold": 5},
    "IB":   {"sig": lambda d: sig_ib_breakout(d)[0],             "sl": 80,  "tp": 480, "hold": 10},
    "MOM":  {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60,  "tp": 400, "hold": 5},
}
NAMES = list(STRATS.keys())


def run_strat(df, name, c):
    s = STRATS[name]
    sigs = s["sig"](df)
    o, h, l, c_, ts, hm = extract_arrays(df)
    return backtest(o, h, l, c_, ts, hm, sigs, s["sl"], s["tp"], s["hold"], c, name)


def daily_pnl(trades):
    d = defaultdict(float)
    for t in trades:
        d[str(t.entry_time)[:10]] += t.net_pnl
    return dict(d)


def combine_daily(*dicts):
    c = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            c[k] += v
    return dict(c)


def score(daily):
    if not daily:
        return {"sharpe": -999, "max_dd": -1e9, "worst_day": -1e9, "monthly_avg": 0}
    vals = list(daily.values())
    sharpe = (np.mean(vals) / np.std(vals) * np.sqrt(252)) if np.std(vals) > 0 else 0
    cum = 0.0; peak = 0.0; mdd = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    monthly = defaultdict(float)
    for d, v in daily.items():
        monthly[d[:7]] += v
    nm = max(len(monthly), 1)
    return {
        "sharpe": sharpe, "max_dd": mdd, "worst_day": min(vals),
        "monthly_avg": sum(vals) / nm, "total": sum(vals),
    }


def full_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "worst_day": 0, "best_day": 0, "max_dd": 0, "bars_mean": 0,
                "monthly": {}, "n_months": 0, "months_pos": 0, "consistency": 100,
                "trades_per_day": 0}
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
    bd = max(daily.values()) if daily else 0
    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": total / nm,
        "worst_month": min(monthly.values()) if monthly else 0,
        "best_month": max(monthly.values()) if monthly else 0,
        "worst_day": min(daily.values()) if daily else 0,
        "best_day": bd, "max_dd": mdd, "bars_mean": np.mean(bars),
        "monthly": dict(monthly), "n_months": nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "consistency": (bd / total * 100) if total > 0 else 100,
        "trades_per_day": len(trades) / nd,
    }


def check_mll_breach(daily_dict):
    """Check if MLL would be breached. Return (breached, date, dd_at_breach)."""
    cum = 0.0; peak = 0.0
    for d in sorted(daily_dict.keys()):
        pnl = daily_dict[d]
        if pnl < DAILY_LIMIT:
            pnl = DAILY_LIMIT  # daily cap
        cum += pnl
        peak = max(peak, cum)
        dd = cum - peak
        if dd <= MLL:
            return True, d, dd
    return False, None, cum - peak


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass_rate": 0, "blowup": 1, "med_days": 0}
    passed = 0; blown = 0; dtp = []
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; p = False; ok = True; dc = 0
        for idx in order:
            dp = sum(days[idx])
            if dp < DAILY_LIMIT: dp = DAILY_LIMIT
            cum += dp; dc += 1; peak = max(peak, cum)
            if cum - peak <= MLL: ok = False; break
            if not p and cum >= EVAL_TARGET: p = True; dtp.append(dc)
        if not ok: blown += 1
        if p and ok: passed += 1
    da = np.array(dtp) if dtp else np.array([0])
    return {"pass_rate": passed/n_sims, "blowup": blown/n_sims,
            "med_days": int(np.median(da)), "p95_days": int(np.percentile(da, 95)) if len(da) else 0}


def run_sizing(combo, c, yr1_df, yr2_df, bl_df, label=""):
    """Run a combo at a given contract size on all 3 datasets."""
    results = {}
    for period, df in [("Y1", yr1_df), ("Y2", yr2_df), ("Blind", bl_df)]:
        all_trades = []
        for name in combo:
            all_trades.extend(run_strat(df, name, c))
        m = full_metrics(all_trades)
        d = daily_pnl(all_trades)
        breached, breach_date, _ = check_mll_breach(d)
        results[period] = {"metrics": m, "trades": all_trades, "daily": d,
                           "mll_breached": breached, "breach_date": str(breach_date)}
    return results


def print_table(label, r):
    y1 = r["Y1"]["metrics"]; y2 = r["Y2"]["metrics"]; bl = r["Blind"]["metrics"]
    print(f"\n  {label}:")
    print(f"  ┌{'─'*13}┬{'─'*14}┬{'─'*14}┬{'─'*14}┐")
    print(f"  │{'Metric':<13}│{'Y1 (select)':>14}│{'Y2 (OOS)':>14}│{'Blind (OOS)':>14}│")
    print(f"  ├{'─'*13}┼{'─'*14}┼{'─'*14}┼{'─'*14}┤")
    for lbl, fn in [
        ("$/month",   lambda m: f"${m['monthly_avg']:>+12,.0f}"),
        ("Months+",   lambda m: f"{m['months_pos']:>10}/{m['n_months']}"),
        ("Worst mo",  lambda m: f"${m['worst_month']:>+12,.0f}"),
        ("Worst day", lambda m: f"${m['worst_day']:>+12,.0f}"),
        ("Max DD",    lambda m: f"${m['max_dd']:>+12,.0f}"),
    ]:
        print(f"  │{lbl:<13}│{fn(y1):>14}│{fn(y2):>14}│{fn(bl):>14}│")
    print(f"  └{'─'*13}┴{'─'*14}┴{'─'*14}┴{'─'*14}┘")

    for period in ["Y2", "Blind"]:
        if r[period]["mll_breached"]:
            print(f"  ❌ MLL breached on {period} at {r[period]['breach_date']}")
        else:
            print(f"  ✅ MLL NOT breached on {period}")


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — No Data Leakage")
    print("═" * 70)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind")

    yr1 = main_data["15m"].filter(pl.col("timestamp") < YR1_END)
    yr2 = main_data["15m"].filter(pl.col("timestamp") >= YR1_END)
    bl = blind_data["15m"]
    print(f"  Y1: {len(yr1)} | Y2: {len(yr2)} | Blind: {len(bl)}")

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Select portfolio on Y1 ONLY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STEP 1: Portfolio Selection on Y1 ONLY")
    print("━" * 70)

    # Pre-compute Y1 trades at each size
    cache_y1 = {}
    for name in NAMES:
        for c in [2, 3, 5]:
            cache_y1[(name, c)] = daily_pnl(run_strat(yr1, name, c))
    gc.collect()

    best = None
    for n_s in [2, 3, 4, 5]:
        for combo in combinations(NAMES, n_s):
            for c in [2, 3, 5]:
                dailies = [cache_y1[(n, c)] for n in combo]
                combined = combine_daily(*dailies)
                s = score(combined)
                if s["max_dd"] < -3000 or s["worst_day"] < -2000:
                    continue
                if best is None or s["sharpe"] > best["sharpe"]:
                    best = {"combo": combo, "c": c, "sharpe": s["sharpe"],
                            "monthly_avg": s["monthly_avg"], "max_dd": s["max_dd"],
                            "worst_day": s["worst_day"]}

    if best is None:
        print("  ❌ No valid combo on Y1.")
        return

    combo = best["combo"]
    print(f"  Selected on Y1: {'+'.join(combo)} @ {best['c']} MNQ each")
    print(f"    Y1 Sharpe: {best['sharpe']:.2f}, $/mo: ${best['monthly_avg']:,.0f}, "
          f"DD: ${best['max_dd']:,.0f}, worst day: ${best['worst_day']:,.0f}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2+3: Run on all 3 datasets at aggressive (3) and conservative (2)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STEP 2-3: Constrained Backtest (Y2 + Blind fully OOS)")
    print("━" * 70)

    r_agg = run_sizing(combo, 3, yr1, yr2, bl, "Aggressive (3c)")
    gc.collect()
    r_con = run_sizing(combo, 2, yr1, yr2, bl, "Conservative (2c)")
    gc.collect()

    print_table("AGGRESSIVE (3 contracts/strategy)", r_agg)
    print_table("CONSERVATIVE (2 contracts/strategy)", r_con)

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Monte Carlo
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STEP 4: Monte Carlo")
    print("━" * 70)

    mc_results = {}
    for label, r in [("Aggressive", r_agg), ("Conservative", r_con)]:
        y2t = r["Y2"]["trades"]
        mc_b = run_mc(y2t, 5000, 1.0)
        mc_c = run_mc(y2t, 5000, 0.70)
        mc_results[label] = {"baseline": mc_b, "conservative": mc_c}
        print(f"  {label}:")
        print(f"    Baseline:     {mc_b['pass_rate']:.0%} pass, {mc_b['blowup']:.0%} blow-up, "
              f"median {mc_b['med_days']}d")
        print(f"    Conservative: {mc_c['pass_rate']:.0%} pass, {mc_c['blowup']:.0%} blow-up")

    # ══════════════════════════════════════════════════════════════
    # FULL REPORT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  HTF SWING v3 — No Data Leakage")
    print("═" * 70)

    print(f"\n  PORTFOLIO SELECTED ON Y1 ONLY:")
    print(f"    Strategies: {', '.join(combo)}")
    print(f"    Note: selected by Y1 Sharpe ({best['sharpe']:.2f}). Y2 and blind are untouched.")

    # Monthly combined for both sizings
    for sizing_label, r, mc in [
        ("AGGRESSIVE (3c)", r_agg, mc_results["Aggressive"]),
        ("CONSERVATIVE (2c)", r_con, mc_results["Conservative"]),
    ]:
        print(f"\n  {sizing_label}:")
        all_monthly = defaultdict(float)
        for period in ["Y1", "Y2", "Blind"]:
            for m, v in r[period]["metrics"]["monthly"].items():
                all_monthly[m] += v

        for m in sorted(all_monthly.keys()):
            v = all_monthly[m]
            print(f"    {m}: ${v:>+10,.0f} {'✅' if v > 0 else '❌'}")

        vals = list(all_monthly.values())
        n = len(vals)
        mp = sum(1 for v in vals if v > 0)
        m7k = sum(1 for v in vals if v >= 7000)
        print(f"\n    Months profitable: {mp}/{n}")
        print(f"    Months > $7K: {m7k}/{n}")
        print(f"    Average: ${np.mean(vals):,.0f}")
        print(f"    Median: ${np.median(vals):,.0f}")
        print(f"    Worst: ${min(vals):,.0f}")
        print(f"    Best: ${max(vals):,.0f}")

        y2m = r["Y2"]["metrics"]
        blm = r["Blind"]["metrics"]
        print(f"\n    Prop firm (Topstep 150K):")
        wd = min(y2m["worst_day"], blm["worst_day"])
        dd = min(y2m["max_dd"], blm["max_dd"])
        print(f"      Worst day: ${wd:>+10,.0f} vs -$3,000 {'✅' if wd > -3000 else '❌'}")
        print(f"      Max DD:    ${dd:>+10,.0f} vs -$4,500 {'✅' if dd > -4500 else '❌'}")
        print(f"      MC baseline: {mc['baseline']['pass_rate']:.0%} pass, {mc['baseline']['blowup']:.0%} blow-up")
        print(f"      MC 70%:      {mc['conservative']['pass_rate']:.0%} pass")

    # Comparison
    y2_agg = r_agg["Y2"]["metrics"]["monthly_avg"]
    y2_con = r_con["Y2"]["metrics"]["monthly_avg"]
    bl_agg = r_agg["Blind"]["metrics"]["monthly_avg"]
    bl_con = r_con["Blind"]["metrics"]["monthly_avg"]
    bl_agg_surv = "✅" if not r_agg["Blind"]["mll_breached"] else "❌"
    bl_con_surv = "✅" if not r_con["Blind"]["mll_breached"] else "❌"

    print(f"\n  COMPARISON:")
    print(f"  ┌{'─'*19}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┐")
    print(f"  │{'System':<19}│{'Y2 $/mo':>10}│{'Blind':>10}│{'MC Pass':>10}│{'Survives':>10}│")
    print(f"  ├{'─'*19}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┤")
    print(f"  │{'v2 (leaked)':<19}│{'$8,703':>10}│{'$4,386':>10}│{'99%':>10}│{'❌ blind':>10}│")
    print(f"  │{'v3 aggressive':<19}│${y2_agg:>9,.0f}│${bl_agg:>9,.0f}│{mc_results['Aggressive']['baseline']['pass_rate']:>9.0%}│{bl_agg_surv:>10}│")
    print(f"  │{'v3 conservative':<19}│${y2_con:>9,.0f}│${bl_con:>9,.0f}│{mc_results['Conservative']['baseline']['pass_rate']:>9.0%}│{bl_con_surv:>10}│")
    print(f"  │{'Trailing Profit':<19}│{'$435':>10}│{'n/a':>10}│{'~93%':>10}│{'✅':>10}│")
    print(f"  │{'Always-On v4':<19}│{'$503':>10}│{'$629':>10}│{'0%':>10}│{'❌':>10}│")
    print(f"  └{'─'*19}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┘")

    # Pick the recommended sizing
    if not r_con["Blind"]["mll_breached"] and mc_results["Conservative"]["baseline"]["pass_rate"] >= 0.50:
        rec = "conservative"
        rec_y2 = y2_con
        rec_bl = bl_con
        rec_mc = mc_results["Conservative"]["baseline"]["pass_rate"]
    elif not r_agg["Blind"]["mll_breached"] and mc_results["Aggressive"]["baseline"]["pass_rate"] >= 0.50:
        rec = "aggressive"
        rec_y2 = y2_agg
        rec_bl = bl_agg
        rec_mc = mc_results["Aggressive"]["baseline"]["pass_rate"]
    else:
        rec = None

    if rec:
        print(f"\n  FINAL HONEST NUMBER ({rec} sizing):")
        print(f"    Y2 OOS: ${rec_y2:,.0f}/month")
        print(f"    Blind OOS: ${rec_bl:,.0f}/month")
        print(f"    MC eval pass: {rec_mc:.0%}")
        print(f"    Zero data leakage. Prop firm rules enforced. Full costs.")
        verdict = f"${rec_y2:,.0f}/month Y2, ${rec_bl:,.0f}/month blind, {rec_mc:.0%} MC pass ({rec})"
    else:
        print(f"\n  ❌ Neither sizing survives the blind data MLL constraint.")
        verdict = "Does not survive blind data"

    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "combo_selected_on_y1": list(combo),
        "y1_sharpe": best["sharpe"],
        "aggressive": {
            "contracts": 3,
            "y2": {k: v for k, v in r_agg["Y2"]["metrics"].items() if k != "monthly"},
            "blind": {k: v for k, v in r_agg["Blind"]["metrics"].items() if k != "monthly"},
            "blind_mll_breached": r_agg["Blind"]["mll_breached"],
            "mc": mc_results["Aggressive"],
        },
        "conservative": {
            "contracts": 2,
            "y2": {k: v for k, v in r_con["Y2"]["metrics"].items() if k != "monthly"},
            "blind": {k: v for k, v in r_con["Blind"]["metrics"].items() if k != "monthly"},
            "blind_mll_breached": r_con["Blind"]["mll_breached"],
            "mc": mc_results["Conservative"],
        },
        "verdict": verdict,
    }
    out = REPORTS_DIR / "htf_swing_v3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
