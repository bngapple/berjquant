#!/usr/bin/env python3
"""
HTF Swing v3 on 5-minute bars. Same params, different timeframe.

Tests whether the 15m edge transfers to 5m with higher trade frequency.
Two options: same max_hold in bars (faster exits) or 3x max_hold (same clock time).

Usage:
    python3 run_htf_swing_5m.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from run_htf_swing import (
    extract_arrays, backtest, rt_cost,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
_ET = ZoneInfo("US/Eastern")

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0

# v3 exact params
STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}
CONTRACTS = 3
YR1_END = datetime(2025, 3, 19, tzinfo=_ET)


def load_and_resample_5m(path, label):
    """Load 1m, filter RTH, resample to 5m."""
    df = pl.read_parquet(path)
    if hasattr(df["timestamp"].dtype, 'time_zone') and df["timestamp"].dtype.time_zone:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    df = df.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
    )
    df = df.with_columns(
        pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
    )
    df = df.drop("_utc")
    df = df.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    df = df.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])

    rth = df.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))

    r = (
        rth.group_by_dynamic("ts_et", every="5m")
        .agg([
            pl.col("open").first(), pl.col("high").max(),
            pl.col("low").min(), pl.col("close").last(),
            pl.col("volume").sum(), pl.col("date_et").last(),
            pl.col("hhmm").last(),
        ])
        .filter(pl.col("open").is_not_null())
        .sort("ts_et")
        .rename({"ts_et": "timestamp"})
    )
    print(f"  {label}: {len(r):,} 5m RTH bars")
    return r


def run_strat(df, name, hold_override=None):
    s = STRATS[name]
    sigs = s["sig"](df)
    o, h, l, c, ts, hm = extract_arrays(df)
    hold = hold_override if hold_override else s["hold"]
    return backtest(o, h, l, c, ts, hm, sigs, s["sl"], s["tp"], hold, CONTRACTS, name)


def metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "worst_day": 0, "max_dd": 0, "bars_mean": 0, "monthly": {},
                "n_months": 0, "months_pos": 0, "sharpe": 0, "pf": 0,
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
    dv = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    nm = max(len(monthly), 1)
    nd = max(len(daily), 1)
    return {
        "pnl": sum(pnls), "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": sum(pnls) / nm, "worst_month": min(monthly.values()) if monthly else 0,
        "worst_day": min(dv) if dv else 0, "max_dd": mdd,
        "bars_mean": np.mean(bars), "monthly": dict(monthly),
        "n_months": nm, "months_pos": sum(1 for v in monthly.values() if v > 0),
        "sharpe": sharpe, "pf": pf, "trades_per_day": len(trades) / nd,
        "best_day": max(dv) if dv else 0,
    }


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass_rate": 0, "blowup": 1}
    p = 0; b = 0
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; ok = True; passed = False
        for idx in order:
            dp = sum(days[idx])
            if dp < DAILY_LIMIT: dp = DAILY_LIMIT
            cum += dp; peak = max(peak, cum)
            if cum - peak <= MLL: ok = False; break
            if cum >= EVAL_TARGET: passed = True
        if not ok: b += 1
        if passed and ok: p += 1
    return {"pass_rate": p / n_sims, "blowup": b / n_sims}


def print_table(label, m):
    print(f"  {label}:")
    print(f"    $/month:     ${m['monthly_avg']:>+10,.0f}")
    print(f"    Trades/mo:   {m['n'] / max(m['n_months'],1):>10.0f}")
    print(f"    Win rate:    {m['wr']:>10.1f}%")
    print(f"    Profit fac:  {m['pf']:>10.2f}")
    print(f"    Avg hold:    {m['bars_mean']:>10.1f} bars")
    print(f"    Worst month: ${m['worst_month']:>+10,.0f}")
    print(f"    Worst day:   ${m['worst_day']:>+10,.0f}")
    print(f"    Max DD:      ${m['max_dd']:>+10,.0f}")


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — 5-Minute Bars")
    print("═" * 70)

    # Load data
    print("\n  Loading and resampling to 5m RTH ...")
    main_5m = load_and_resample_5m("data/processed/MNQ/1m/full_2yr.parquet", "Main (2024-2026)")
    blind_5m = load_and_resample_5m("data/processed/MNQ/1m/databento_extended.parquet", "Blind (2022-2024)")

    yr1 = main_5m.filter(pl.col("timestamp") < YR1_END)
    yr2 = main_5m.filter(pl.col("timestamp") >= YR1_END)
    print(f"  Y1: {len(yr1):,} | Y2: {len(yr2):,} | Blind: {len(blind_5m):,}")

    # ── Option A: Same max_hold in bars ──
    print(f"\n{'━' * 70}")
    print("  OPTION A — Same max_hold in bars (faster exits on 5m)")
    print("━" * 70)

    results_a = {}
    for period, df, label in [("Y1", yr1, "Y1"), ("Y2", yr2, "Y2"), ("Blind", blind_5m, "Blind")]:
        all_t = []
        for name in STRATS:
            all_t.extend(run_strat(df, name))
            gc.collect()
        m = metrics(all_t)
        results_a[period] = {"metrics": m, "trades": all_t}
        print_table(f"  {label}", m)

    # ── Option B: 3x max_hold (same clock time as 15m) ──
    print(f"\n{'━' * 70}")
    print("  OPTION B — 3x max_hold (same clock time as 15m)")
    print("━" * 70)

    results_b = {}
    for period, df, label in [("Y1", yr1, "Y1"), ("Y2", yr2, "Y2"), ("Blind", blind_5m, "Blind")]:
        all_t = []
        for name in STRATS:
            hold_3x = STRATS[name]["hold"] * 3
            all_t.extend(run_strat(df, name, hold_override=hold_3x))
            gc.collect()
        m = metrics(all_t)
        results_b[period] = {"metrics": m, "trades": all_t}
        print_table(f"  {label}", m)

    # Pick best option
    a_y2 = results_a["Y2"]["metrics"]["monthly_avg"]
    b_y2 = results_b["Y2"]["metrics"]["monthly_avg"]
    best_label = "A" if a_y2 >= b_y2 else "B"
    best_results = results_a if best_label == "A" else results_b
    print(f"\n  Best option: {best_label} (Y2: ${max(a_y2, b_y2):,.0f}/mo)")

    # Monthly breakdown (best option)
    print(f"\n{'━' * 70}")
    print(f"  MONTHLY BREAKDOWN (Option {best_label})")
    print("━" * 70)

    all_monthly = defaultdict(float)
    all_monthly_tc = defaultdict(int)
    for period in ["Y1", "Y2", "Blind"]:
        m = best_results[period]["metrics"]
        for mo, v in m["monthly"].items():
            all_monthly[mo] += v
            all_monthly_tc[mo] += 1  # approximate

    # Count trades per month properly
    for period in ["Y1", "Y2", "Blind"]:
        for t in best_results[period]["trades"]:
            mo = str(t.entry_time)[:7]
            all_monthly_tc[mo] = all_monthly_tc.get(mo, 0)  # already counted above

    # Recount properly
    tc_map = defaultdict(int)
    for period in ["Y1", "Y2", "Blind"]:
        for t in best_results[period]["trades"]:
            tc_map[str(t.entry_time)[:7]] += 1

    for mo in sorted(all_monthly.keys()):
        v = all_monthly[mo]
        tc = tc_map[mo]
        flag = "✅" if v > 0 else "❌"
        print(f"  {mo}: ${v:>+10,.0f} ({tc:>4} trades) {flag}")

    vals = list(all_monthly.values())
    n = len(vals)
    mp = sum(1 for v in vals if v > 0)
    print(f"\n  Months profitable: {mp}/{n}")
    print(f"  Average: ${np.mean(vals):,.0f}")
    print(f"  Median: ${np.median(vals):,.0f}")

    # TradingView comparison — pull Feb-Mar 2026
    print(f"\n{'━' * 70}")
    print("  TRADINGVIEW COMPARISON (Feb 22 - Mar 24, 2026)")
    print("━" * 70)

    # Get Y2 monthly to find Feb-Mar 2026
    y2_m = best_results["Y2"]["metrics"]["monthly"]
    feb_mar = sum(y2_m.get(m, 0) for m in ["2026-02", "2026-03"])
    feb_trades = sum(1 for t in best_results["Y2"]["trades"] if str(t.entry_time)[:7] in ["2026-02", "2026-03"])
    print(f"  Python 5m (Feb-Mar 2026): ${feb_mar:,.0f} ({feb_trades} trades)")
    print(f"  TradingView 5m (1 month): $5,555 (256 trades, 28.5% WR)")
    if feb_mar > 0:
        ratio = feb_mar / 5555 if 5555 > 0 else 0
        print(f"  Ratio (Python/TV): {ratio:.2f}x")

    # Comparison table
    y2a = results_a["Y2"]["metrics"]
    y2b = results_b["Y2"]["metrics"]
    bla = results_a["Blind"]["metrics"]
    blb = results_b["Blind"]["metrics"]

    print(f"\n{'━' * 70}")
    print("  COMPARISON")
    print("━" * 70)
    print(f"  ┌{'─'*23}┬{'─'*10}┬{'─'*8}┬{'─'*7}┬{'─'*10}┬{'─'*10}┐")
    print(f"  │{'System':<23}│{'Y2 $/mo':>10}│{'Trades':>8}│{'WR':>7}│{'Max DD':>10}│{'Blind':>10}│")
    print(f"  ├{'─'*23}┼{'─'*10}┼{'─'*8}┼{'─'*7}┼{'─'*10}┼{'─'*10}┤")
    print(f"  │{'v3 on 15m':<23}│{'$7,140':>10}│{'~160':>8}│{'  25%':>7}│{'$-4,362':>10}│{'$3,242':>10}│")
    print(f"  │{'v3 on 5m (Opt A)':<23}│${y2a['monthly_avg']:>9,.0f}│{y2a['n']:>8}│{y2a['wr']:>6.0f}%│${y2a['max_dd']:>9,.0f}│${bla['monthly_avg']:>9,.0f}│")
    print(f"  │{'v3 on 5m (Opt B)':<23}│${y2b['monthly_avg']:>9,.0f}│{y2b['n']:>8}│{y2b['wr']:>6.0f}%│${y2b['max_dd']:>9,.0f}│${blb['monthly_avg']:>9,.0f}│")
    print(f"  │{'TradingView 5m':<23}│{'$5,555*':>10}│{'256*':>8}│{' 29%':>7}│{'$-3,484':>10}│{'n/a':>10}│")
    print(f"  └{'─'*23}┴{'─'*10}┴{'─'*8}┴{'─'*7}┴{'─'*10}┴{'─'*10}┘")
    print(f"  * TradingView = 1 month only")

    # Prop firm
    best_y2 = best_results["Y2"]["metrics"]
    best_bl = best_results["Blind"]["metrics"]
    print(f"\n  PROP FIRM (Topstep 150K):")
    wd = min(best_y2["worst_day"], best_bl["worst_day"])
    dd = min(best_y2["max_dd"], best_bl["max_dd"])
    con = max(best_y2.get("best_day", 0) / max(best_y2["pnl"], 1) * 100,
              best_bl.get("best_day", 0) / max(best_bl["pnl"], 1) * 100) if best_y2["pnl"] > 0 else 100
    print(f"    Worst day: ${wd:>+10,.0f} vs -$3,000 {'✅' if wd > -3000 else '❌'}")
    print(f"    Max DD:    ${dd:>+10,.0f} vs -$4,500 {'✅' if dd > -4500 else '❌'}")

    # MC
    print(f"\n  MONTE CARLO:")
    mc_b = run_mc(best_results["Y2"]["trades"], 5000, 1.0)
    mc_c = run_mc(best_results["Y2"]["trades"], 5000, 0.70)
    print(f"    Baseline:     {mc_b['pass_rate']:.0%} pass, {mc_b['blowup']:.0%} blow-up")
    print(f"    Conservative: {mc_c['pass_rate']:.0%} pass, {mc_c['blowup']:.0%} blow-up")

    # Verdict
    best_5m_y2 = best_results["Y2"]["metrics"]["monthly_avg"]
    ref_15m = 7140
    if best_5m_y2 > ref_15m * 1.1:
        verdict = f"5m is BETTER — ${best_5m_y2:,.0f}/mo vs ${ref_15m:,.0f}/mo on 15m. New deployment candidate."
    elif best_5m_y2 > ref_15m * 0.8:
        verdict = f"5m is SIMILAR — ${best_5m_y2:,.0f}/mo vs ${ref_15m:,.0f}/mo. More trades, similar profit."
    elif best_5m_y2 > 0:
        verdict = f"5m is WORSE — ${best_5m_y2:,.0f}/mo vs ${ref_15m:,.0f}/mo. Stick with 15m."
    else:
        verdict = f"5m DOESN'T WORK — ${best_5m_y2:,.0f}/mo. The edge is timeframe-dependent. Stay on 15m."

    print(f"\n  VERDICT: {verdict}")
    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "option_a_y2": {k: v for k, v in results_a["Y2"]["metrics"].items() if k != "monthly"},
        "option_a_blind": {k: v for k, v in results_a["Blind"]["metrics"].items() if k != "monthly"},
        "option_b_y2": {k: v for k, v in results_b["Y2"]["metrics"].items() if k != "monthly"},
        "option_b_blind": {k: v for k, v in results_b["Blind"]["metrics"].items() if k != "monthly"},
        "best_option": best_label,
        "mc_baseline": mc_b, "mc_conservative": mc_c,
        "verdict": verdict,
    }
    out = REPORTS_DIR / "htf_swing_5m.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
