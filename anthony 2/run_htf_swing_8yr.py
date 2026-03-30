#!/usr/bin/env python3
"""
HTF Swing v3 — 8-Year Backtest. Final validation.

Merges 3 datasets into 8 years of 15m RTH bars, runs RSI+IB+MOM
with exact v3 params. No re-optimization.

Usage:
    python3 run_htf_swing_8yr.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from run_htf_swing import (
    load_and_resample, extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}
CONTRACTS = 3


def load_1m(path, label):
    """Load 1m parquet and add ET columns."""
    df = pl.read_parquet(path)
    if "tick_count" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("tick_count"))
    if hasattr(df["timestamp"].dtype, 'time_zone') and df["timestamp"].dtype.time_zone:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    print(f"  {label}: {len(df):,} bars, {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def resample_15m_rth(df):
    """Add ET columns, filter RTH, resample to 15m."""
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
        rth.group_by_dynamic("ts_et", every="15m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("date_et").last(),
            pl.col("hhmm").last(),
        ])
        .filter(pl.col("open").is_not_null())
        .sort("ts_et")
        .rename({"ts_et": "timestamp"})
    )
    return r


def run_strat(df, name):
    s = STRATS[name]
    sigs = s["sig"](df)
    o, h, l, c, ts, hm = extract_arrays(df)
    return backtest(o, h, l, c, ts, hm, sigs, s["sl"], s["tp"], s["hold"], CONTRACTS, name)


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — 8-YEAR BACKTEST")
    print("═" * 70)

    # ── Phase 1-2: Load and merge ─────────────────────────────
    print("\n  Loading datasets ...")
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")

    # Merge
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    print(f"\n  Combined 1m: {len(combined):,} bars")
    print(f"    Range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")

    # Resample
    print("  Resampling to 15m RTH ...")
    df_15m = resample_15m_rth(combined)
    print(f"  15m bars: {len(df_15m):,}")
    print(f"    Range: {df_15m['timestamp'].min()} → {df_15m['timestamp'].max()}")

    dates = df_15m.select(pl.col("date_et").cast(pl.Utf8).str.slice(0, 7).alias("m")).unique()
    total_months = dates.height
    print(f"    Months: {total_months}")

    del df1, df2, df3, combined
    gc.collect()

    # ── Phase 3: Run each strategy ────────────────────────────
    print(f"\n{'━' * 70}")
    print("  Running 8-year backtest ...")
    print("━" * 70)

    layer_monthly = {}
    layer_trades_monthly = {}
    all_trades = []

    for name in STRATS:
        print(f"  {name} ...", end=" ", flush=True)
        trades = run_strat(df_15m, name)
        all_trades.extend(trades)

        m = defaultdict(float)
        mt = defaultdict(int)
        for t in trades:
            mo = str(t.entry_time)[:7]
            m[mo] += t.net_pnl
            mt[mo] += 1
        layer_monthly[name] = dict(m)
        layer_trades_monthly[name] = dict(mt)

        total = sum(m.values())
        n_m = max(len(m), 1)
        print(f"{len(trades)} trades, ${total:,.0f} total, ${total/n_m:,.0f}/mo")
        gc.collect()

    # Combined monthly
    all_months_set = set()
    for lm in layer_monthly.values():
        all_months_set.update(lm.keys())
    all_months = sorted(all_months_set)

    combined_monthly = {}
    combined_tc = {}
    for mo in all_months:
        combined_monthly[mo] = sum(layer_monthly[n].get(mo, 0) for n in STRATS)
        combined_tc[mo] = sum(layer_trades_monthly[n].get(mo, 0) for n in STRATS)

    # Daily for DD calc
    daily = defaultdict(float)
    for t in all_trades:
        daily[str(t.entry_time)[:10]] += t.net_pnl

    cum = 0; peak = 0; mdd = 0; worst_day = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
        worst_day = min(worst_day, daily[d])

    # ── Phase 4: Output ───────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  HTF SWING v3 — 8-YEAR BACKTEST RESULTS")
    print("═" * 70)
    print(f"\n  Data: {df_15m['timestamp'].min()} → {df_15m['timestamp'].max()}")
    print(f"  15m bars: {len(df_15m):,} | Months: {len(all_months)}")
    print(f"  NQ data used for Jan 2018 - May 2019 (pre-MNQ, same price action)")

    # Month-by-month table
    print(f"\n  ┌{'─'*9}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*12}┬{'─'*5}┬{'─'*3}┐")
    print(f"  │{'Month':<9}│{'RSI':>10}│{'IB':>10}│{'MOM':>10}│{'COMBINED':>12}│{'#':>5}│   │")
    print(f"  ├{'─'*9}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*12}┼{'─'*5}┼{'─'*3}┤")

    for mo in all_months:
        rsi = layer_monthly["RSI"].get(mo, 0)
        ib = layer_monthly["IB"].get(mo, 0)
        mom = layer_monthly["MOM"].get(mo, 0)
        tot = combined_monthly[mo]
        tc = combined_tc[mo]
        flag = "✅" if tot > 0 else "❌"
        print(f"  │{mo:<9}│${rsi:>+9,.0f}│${ib:>+9,.0f}│${mom:>+9,.0f}│${tot:>+11,.0f}│{tc:>5}│ {flag} │")

    print(f"  └{'─'*9}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*12}┴{'─'*5}┴{'─'*3}┘")

    # Yearly summary
    yearly = defaultdict(lambda: {"total": 0, "months": 0, "months_pos": 0,
                                   "worst": float("inf"), "best": float("-inf")})
    for mo, pnl in combined_monthly.items():
        yr = mo[:4]
        yearly[yr]["total"] += pnl
        yearly[yr]["months"] += 1
        if pnl > 0: yearly[yr]["months_pos"] += 1
        yearly[yr]["worst"] = min(yearly[yr]["worst"], pnl)
        yearly[yr]["best"] = max(yearly[yr]["best"], pnl)

    print(f"\n  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┐")
    print(f"  │{'Year':<6}│{'Total':>12}│{'Avg/Mo':>10}│{'Mo+':>10}│{'Worst Mo':>10}│{'Best Mo':>10}│")
    print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┤")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        avg = y["total"] / max(y["months"], 1)
        print(f"  │{yr:<6}│${y['total']:>+11,.0f}│${avg:>+9,.0f}│{y['months_pos']:>6}/{y['months']:<3}│${y['worst']:>+9,.0f}│${y['best']:>+9,.0f}│")
    print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┘")

    # Grand summary
    vals = list(combined_monthly.values())
    n = len(vals)
    mp = sum(1 for v in vals if v > 0)
    m3k = sum(1 for v in vals if v >= 3000)
    m5k = sum(1 for v in vals if v >= 5000)
    m7k = sum(1 for v in vals if v >= 7000)
    avg = np.mean(vals); med = np.median(vals)
    wm = min(vals); wm_k = all_months[vals.index(wm)]
    bm = max(vals); bm_k = all_months[vals.index(bm)]

    worst_yr = min(yearly.keys(), key=lambda y: yearly[y]["total"])
    best_yr = max(yearly.keys(), key=lambda y: yearly[y]["total"])

    # Streaks
    win_streak = 0; lose_streak = 0; max_win = 0; max_lose = 0; cur_w = 0; cur_l = 0
    for v in vals:
        if v > 0:
            cur_w += 1; cur_l = 0; max_win = max(max_win, cur_w)
        else:
            cur_l += 1; cur_w = 0; max_lose = max(max_lose, cur_l)

    print(f"\n  GRAND SUMMARY:")
    print(f"    Total months: {n}")
    print(f"    Months profitable: {mp}/{n} ({mp/n*100:.0f}%)")
    print(f"    Months above $3K: {m3k}/{n}")
    print(f"    Months above $5K: {m5k}/{n}")
    print(f"    Months above $7K: {m7k}/{n}")
    print(f"    Average monthly: ${avg:,.0f}")
    print(f"    Median monthly: ${med:,.0f}")
    print(f"    Worst month: ${wm:,.0f} ({wm_k})")
    print(f"    Best month: ${bm:,.0f} ({bm_k})")
    print(f"    Worst year: {worst_yr} at ${yearly[worst_yr]['total']:,.0f}")
    print(f"    Best year: {best_yr} at ${yearly[best_yr]['total']:,.0f}")
    print(f"    Max drawdown (8yr): ${mdd:,.0f}")
    print(f"    Worst day: ${worst_day:,.0f}")
    print(f"    Longest winning streak: {max_win} months")
    print(f"    Longest losing streak: {max_lose} months")

    # Market conditions
    def period_avg(start, end):
        mos = [mo for mo in all_months if start <= mo <= end]
        if not mos: return 0, 0
        pnls = [combined_monthly[mo] for mo in mos]
        return np.mean(pnls), len(mos)

    print(f"\n  MARKET CONDITIONS:")
    for label, s, e in [
        ("2018 Q4 selloff", "2018-10", "2018-12"),
        ("2020 COVID crash", "2020-02", "2020-04"),
        ("2022 bear market", "2022-01", "2022-09"),
        ("2023 chop", "2023-01", "2023-12"),
        ("2024-25 bull run", "2024-01", "2025-12"),
    ]:
        avg_p, n_p = period_avg(s, e)
        print(f"    {label}: ${avg_p:,.0f}/month over {n_p} months")

    # Prop firm
    print(f"\n  PROP FIRM (Topstep 150K):")
    surv = mdd > -4500
    print(f"    Worst day: ${worst_day:,.0f} vs -$3,000 {'✅' if worst_day > -3000 else '❌'}")
    print(f"    Max DD: ${mdd:,.0f} vs -$4,500 {'✅' if surv else '❌'}")
    if not surv:
        # Find when breached
        cum2 = 0; peak2 = 0
        for d in sorted(daily.keys()):
            cum2 += daily[d]; peak2 = max(peak2, cum2)
            if cum2 - peak2 <= -4500:
                print(f"    MLL breached on: {d}")
                break

    # Verdict
    if mp / n >= 0.75 and avg > 2000 and surv:
        verdict = f"SURVIVED — {mp}/{n} months profitable (${avg:,.0f}/mo avg) across 8 years including 2018 selloff, COVID crash, 2022 bear, and 2024-25 bull."
    elif mp / n >= 0.60 and avg > 0:
        verdict = f"MARGINAL — {mp}/{n} months profitable (${avg:,.0f}/mo avg). Edge exists but inconsistent across all regimes."
    else:
        verdict = f"FAILED — only {mp}/{n} months profitable. Edge does not hold across 8 years."

    print(f"\n  VERDICT: {verdict}")
    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "total_months": n, "months_profitable": mp,
        "months_3k": m3k, "months_5k": m5k, "months_7k": m7k,
        "avg_monthly": avg, "median_monthly": med,
        "worst_month": wm, "worst_month_date": wm_k,
        "best_month": bm, "best_month_date": bm_k,
        "max_dd": mdd, "worst_day": worst_day,
        "survives_mll": surv,
        "yearly": {y: {"total": d["total"], "months": d["months"], "months_pos": d["months_pos"]}
                   for y, d in yearly.items()},
        "monthly": combined_monthly,
        "verdict": verdict,
    }
    out = REPORTS_DIR / "htf_swing_8yr.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
