#!/usr/bin/env python3
"""
HTF Swing v5 — 5 independent optimizations on 8-year data.

1. Drop IB, reallocate to MOM
2. Volatility regime filter
3. Re-optimize on broader training data (2018-2022 train / 2023-2026 test)
4. Multi-timeframe MOM (5m, 15m, 1h)
5. Asymmetric sizing based on trailing performance

Usage:
    python3 run_htf_swing_v5.py
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
    extract_arrays, backtest, rt_cost, calc_atr, calc_ema,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE, SLIP_PTS,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
_ET = ZoneInfo("US/Eastern")

# Original params from v3
ORIG = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}


def run_strat(df, name, params, contracts):
    sigs = params["sig"](df)
    o, h, l, c, ts, hm = extract_arrays(df)
    return backtest(o, h, l, c, ts, hm, sigs, params["sl"], params["tp"],
                    params["hold"], contracts, name)


def summarize(trades, label=""):
    if not trades:
        return {"pnl": 0, "n": 0, "monthly_avg": 0, "months_pos": 0, "n_months": 0,
                "worst_month": 0, "max_dd": 0, "sharpe": 0, "monthly": {}}
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
    nm = max(len(monthly), 1)
    total = sum(t.net_pnl for t in trades)
    return {
        "pnl": total, "n": len(trades), "monthly_avg": total / nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "n_months": nm, "worst_month": min(monthly.values()) if monthly else 0,
        "max_dd": mdd, "sharpe": sharpe, "monthly": dict(monthly),
    }


def yearly_from_monthly(monthly):
    yearly = defaultdict(lambda: {"total": 0, "months": 0, "months_pos": 0})
    for mo, pnl in monthly.items():
        yr = mo[:4]
        yearly[yr]["total"] += pnl
        yearly[yr]["months"] += 1
        if pnl > 0: yearly[yr]["months_pos"] += 1
    return dict(yearly)


def print_row(name, s, orig_s=None):
    pct = s["months_pos"] / max(s["n_months"], 1) * 100
    yearly = yearly_from_monthly(s["monthly"])
    worst_yr_val = min((y["total"] for y in yearly.values()), default=0)
    diff = f"" if orig_s is None else f" ({'+' if s['monthly_avg'] > orig_s['monthly_avg'] else ''}{s['monthly_avg'] - orig_s['monthly_avg']:,.0f})"
    print(f"  │{name:<23}│${s['monthly_avg']:>8,.0f}│{pct:>7.0f}%│${worst_yr_val:>9,.0f}│${s['max_dd']:>9,.0f}│{s['sharpe']:>7.2f}│")


def load_8yr_15m():
    """Load and merge all data into 15m RTH bars."""
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    del df1, df2, df3; gc.collect()
    df_15m = resample_15m_rth(combined)
    del combined; gc.collect()
    return df_15m


def resample_nm_rth(df_1m, minutes):
    """Resample 1m to Nm RTH bars (need 1m with ET columns)."""
    # Add ET if needed
    if "hhmm" not in df_1m.columns:
        df_1m = df_1m.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
        )
        df_1m = df_1m.with_columns(
            pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
        )
        df_1m = df_1m.drop("_utc")
        df_1m = df_1m.with_columns([
            pl.col("ts_et").dt.date().alias("date_et"),
            pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
            pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
        ])
        df_1m = df_1m.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])

    rth = df_1m.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))
    r = (
        rth.group_by_dynamic("ts_et", every=f"{minutes}m")
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
    return r


# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v5 — 5 Optimizations on 8 Years")
    print("═" * 70)

    # Load
    print("\n  Loading 8-year 15m data ...")
    df_15m = load_8yr_15m()
    print(f"  15m bars: {len(df_15m):,}")

    # Baseline
    print(f"\n{'━' * 70}")
    print("  BASELINE (original v3: RSI+IB+MOM @ 3c each)")
    print("━" * 70)
    baseline_trades = []
    for name in ["RSI", "IB", "MOM"]:
        baseline_trades.extend(run_strat(df_15m, name, ORIG[name], 3))
    orig_s = summarize(baseline_trades)
    print(f"  ${orig_s['monthly_avg']:,.0f}/mo, {orig_s['months_pos']}/{orig_s['n_months']} mo+, "
          f"DD=${orig_s['max_dd']:,.0f}, Sharpe={orig_s['sharpe']:.2f}")
    gc.collect()

    results = {"Baseline": orig_s}

    # ── OPT 1: Drop IB, double MOM ──────────────────────────────
    print(f"\n{'━' * 70}")
    print("  OPT 1: Drop IB, RSI@3c + MOM@6c")
    print("━" * 70)
    opt1_trades = []
    opt1_trades.extend(run_strat(df_15m, "RSI", ORIG["RSI"], 3))
    opt1_trades.extend(run_strat(df_15m, "MOM", ORIG["MOM"], 6))
    s1 = summarize(opt1_trades)
    results["Opt1: Drop IB"] = s1
    print(f"  ${s1['monthly_avg']:,.0f}/mo, {s1['months_pos']}/{s1['n_months']} mo+, "
          f"DD=${s1['max_dd']:,.0f}, Sharpe={s1['sharpe']:.2f}")
    gc.collect()

    # ── OPT 2: Vol filter ────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  OPT 2: Volatility Regime Filter")
    print("━" * 70)

    # Compute daily returns and rolling vol from 15m close
    closes = df_15m["close"].to_numpy()
    dates_arr = [str(t)[:10] for t in df_15m["timestamp"].to_list()]

    # Daily close
    day_close = {}
    for i in range(len(closes)):
        day_close[dates_arr[i]] = closes[i]
    sorted_days = sorted(day_close.keys())
    daily_returns = {}
    for i in range(1, len(sorted_days)):
        d = sorted_days[i]
        prev = sorted_days[i-1]
        if day_close[prev] > 0:
            daily_returns[d] = (day_close[d] - day_close[prev]) / day_close[prev]

    # Rolling 20-day std of returns, rank vs trailing 252 days
    ret_list = [(d, daily_returns[d]) for d in sorted_days if d in daily_returns]
    vol_rank = {}  # date → percentile (0-1)
    for i in range(252, len(ret_list)):
        d, _ = ret_list[i]
        recent_20 = [abs(ret_list[j][1]) for j in range(i-20, i)]
        trailing_252 = [abs(ret_list[j][1]) for j in range(i-252, i)]
        current_vol = np.std(recent_20)
        pctile = np.mean([1 if np.std(trailing_252[j:j+20]) <= current_vol else 0
                          for j in range(0, 232, 20)])
        vol_rank[d] = pctile

    # Build skip sets
    skip_sets = {}
    for pct_label, threshold in [("15%", 0.15), ("25%", 0.25), ("33%", 0.33)]:
        skip_sets[pct_label] = {d for d, r in vol_rank.items() if r < threshold}

    # For each filter level, filter signals
    for pct_label in ["15%", "25%", "33%"]:
        skip = skip_sets[pct_label]
        filtered_trades = []
        for name in ["RSI", "IB", "MOM"]:
            all_t = run_strat(df_15m, name, ORIG[name], 3)
            kept = [t for t in all_t if str(t.entry_time)[:10] not in skip]
            filtered_trades.extend(kept)

        s = summarize(filtered_trades)
        results[f"Opt2: Vol {pct_label}"] = s
        n_skip = len(skip)
        print(f"  Filter {pct_label} (skip {n_skip} days): ${s['monthly_avg']:,.0f}/mo, "
              f"{s['months_pos']}/{s['n_months']} mo+, DD=${s['max_dd']:,.0f}, Sharpe={s['sharpe']:.2f}")
    gc.collect()

    # ── OPT 3: Re-optimize on 2018-2022 ─────────────────────────
    print(f"\n{'━' * 70}")
    print("  OPT 3: Re-optimize (train 2018-2022, test 2023-2026)")
    print("━" * 70)

    train_end = datetime(2023, 1, 1, tzinfo=_ET)
    train = df_15m.filter(pl.col("timestamp") < train_end)
    test = df_15m.filter(pl.col("timestamp") >= train_end)
    print(f"  Train: {len(train):,} bars | Test: {len(test):,} bars")

    best_params = {}
    for name in ["RSI", "MOM"]:
        print(f"  Sweeping {name} ...")
        best = None
        sl_list = [40, 60, 80, 100, 120]
        tp_list = [200, 300, 400, 500, 600]
        hold_list = [5, 10, 15, 20, 30]

        if name == "RSI":
            extra_iter = [(7, 30), (10, 25), (14, 20)]
        else:
            extra_iter = [(1.0,)]  # atr_mult placeholder

        for sl in sl_list:
            for tp in tp_list:
                if tp <= sl: continue
                for hold in hold_list:
                    for extra in extra_iter:
                        if name == "RSI":
                            period, os_ = extra
                            ob = 100 - os_
                            sig_fn = lambda d, p=period, o=ob, s=os_: sig_rsi_extreme(d, p, o, s)
                        else:
                            sig_fn = lambda d: sig_momentum_bar(d, 1.0, 1.0)[0]

                        sigs = sig_fn(train)
                        o, h, l, c, ts, hm = extract_arrays(train)
                        trades = backtest(o, h, l, c, ts, hm, sigs, sl, tp, hold, 3, name)
                        s = summarize(trades)
                        if s["n"] < 50 or s["sharpe"] <= 0: continue
                        if best is None or s["sharpe"] > best["sharpe"]:
                            best = {"sl": sl, "tp": tp, "hold": hold, "extra": extra,
                                    "sharpe": s["sharpe"], "monthly": s["monthly_avg"]}

        if best:
            best_params[name] = best
            print(f"    Best: SL={best['sl']} TP={best['tp']} hold={best['hold']} "
                  f"extra={best['extra']} Sharpe={best['sharpe']:.2f} ${best['monthly']:,.0f}/mo")

    # Run test with best params
    opt3_trades = []
    for name in ["RSI", "MOM"]:
        if name not in best_params: continue
        bp = best_params[name]
        if name == "RSI":
            p, os_ = bp["extra"]
            ob = 100 - os_
            sigs = sig_rsi_extreme(test, p, ob, os_)
        else:
            sigs = sig_momentum_bar(test, 1.0, 1.0)[0]
        o, h, l, c, ts, hm = extract_arrays(test)
        trades = backtest(o, h, l, c, ts, hm, sigs, bp["sl"], bp["tp"], bp["hold"], 3, name)
        opt3_trades.extend(trades)

    s3_test = summarize(opt3_trades)
    # Also run original on test for comparison
    orig_test_trades = []
    for name in ["RSI", "MOM"]:
        orig_test_trades.extend(run_strat(test, name, ORIG[name], 3))
    s3_orig = summarize(orig_test_trades)

    print(f"  Test (2023-2026): Re-opt ${s3_test['monthly_avg']:,.0f}/mo vs Orig ${s3_orig['monthly_avg']:,.0f}/mo")

    # Run re-optimized on full 8yr for comparison table
    opt3_full = []
    for name in ["RSI", "MOM"]:
        if name not in best_params: continue
        bp = best_params[name]
        if name == "RSI":
            p, os_ = bp["extra"]
            ob = 100 - os_
            sigs = sig_rsi_extreme(df_15m, p, ob, os_)
        else:
            sigs = sig_momentum_bar(df_15m, 1.0, 1.0)[0]
        o, h, l, c, ts, hm = extract_arrays(df_15m)
        trades = backtest(o, h, l, c, ts, hm, sigs, bp["sl"], bp["tp"], bp["hold"], 3, name)
        opt3_full.extend(trades)
    s3 = summarize(opt3_full)
    results["Opt3: Re-optimized"] = s3
    gc.collect()

    # ── OPT 4: Multi-TF MOM ─────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  OPT 4: Multi-Timeframe MOM (5m + 15m + 1h)")
    print("━" * 70)

    # Load raw 1m for resampling
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "raw1")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "raw2")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "raw3")
    raw_1m = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    raw_1m = raw_1m.filter(pl.col("close") > 0)

    # Add ET columns for resampling
    raw_1m = raw_1m.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
    )
    raw_1m = raw_1m.with_columns(
        pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
    )
    raw_1m = raw_1m.drop("_utc")
    raw_1m = raw_1m.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    raw_1m = raw_1m.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])
    del df1, df2, df3; gc.collect()

    df_5m = resample_nm_rth(raw_1m, 5)
    df_1h = resample_nm_rth(raw_1m, 60)
    del raw_1m; gc.collect()

    print(f"  5m: {len(df_5m):,} bars | 15m: {len(df_15m):,} | 1h: {len(df_1h):,}")

    opt4_trades = []
    for tf_label, tf_df, hold_adj in [("5m", df_5m, 15), ("15m", df_15m, 5), ("1h", df_1h, 3)]:
        sigs = sig_momentum_bar(tf_df, 1.0, 1.0)[0]
        o, h, l, c, ts, hm = extract_arrays(tf_df)
        # Adjust hold for timeframe
        trades = backtest(o, h, l, c, ts, hm, sigs, 60, 400, hold_adj, 2, f"MOM_{tf_label}")
        opt4_trades.extend(trades)
        s_tf = summarize(trades)
        print(f"    MOM@{tf_label}: {s_tf['n']} trades, ${s_tf['monthly_avg']:,.0f}/mo")

    # Add RSI@15m at 3c
    opt4_trades.extend(run_strat(df_15m, "RSI", ORIG["RSI"], 3))
    s4 = summarize(opt4_trades)
    results["Opt4: Multi-TF MOM"] = s4
    print(f"  Combined: ${s4['monthly_avg']:,.0f}/mo, Sharpe={s4['sharpe']:.2f}")
    del df_5m, df_1h; gc.collect()

    # ── OPT 5: Asymmetric sizing ─────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  OPT 5: Asymmetric Sizing (trailing 3-month performance)")
    print("━" * 70)

    # Run each strategy at 1c, then scale P&L based on trailing perf
    from run_htf_swing import Trade
    opt5_trades = []
    for name in ["RSI", "MOM"]:
        trades_1c = run_strat(df_15m, name, ORIG[name], 1)
        # Group by month
        monthly_pnl = defaultdict(float)
        for t in trades_1c:
            monthly_pnl[str(t.entry_time)[:7]] += t.net_pnl

        sorted_months = sorted(monthly_pnl.keys())
        trailing_3m = {}
        for i, mo in enumerate(sorted_months):
            if i >= 3:
                trailing_3m[mo] = sum(monthly_pnl[sorted_months[j]] for j in range(i-3, i))
            else:
                trailing_3m[mo] = 0

        # Determine multiplier per month
        month_mult = {}
        for mo in sorted_months:
            t3 = trailing_3m.get(mo, 0)
            if t3 > 10000 / 3:  # scale: 1c base, so $10K/3 ≈ $3,333 threshold for 1c
                month_mult[mo] = 4
            elif t3 > 0:
                month_mult[mo] = 3
            elif t3 > -3000 / 3:
                month_mult[mo] = 2
            else:
                month_mult[mo] = 1

        for t in trades_1c:
            mo = str(t.entry_time)[:7]
            mult = month_mult.get(mo, 3)
            scaled_pnl = t.net_pnl * mult  # Linear scaling from 1c base
            # Adjust cost: cost scales with contracts
            cost_diff = rt_cost(mult) - rt_cost(1)
            # Approximate: the 1c trade already has 1c cost, scale the raw part
            raw_1c = t.net_pnl + rt_cost(1)  # add back 1c cost to get raw
            new_pnl = raw_1c * mult - rt_cost(mult)
            opt5_trades.append(Trade(t.direction, t.entry_px, t.exit_px, mult,
                                     new_pnl, t.entry_time, t.exit_time,
                                     t.bars_held, t.reason, t.strategy))

    s5 = summarize(opt5_trades)
    results["Opt5: Asym sizing"] = s5
    print(f"  ${s5['monthly_avg']:,.0f}/mo, {s5['months_pos']}/{s5['n_months']} mo+, "
          f"DD=${s5['max_dd']:,.0f}, Sharpe={s5['sharpe']:.2f}")
    gc.collect()

    # ── COMBINE BEST ─────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  COMBINING BEST OPTIMIZATIONS")
    print("━" * 70)

    # Identify what helped
    # Compare all to baseline
    for label, s in sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True):
        imp = s["monthly_avg"] - orig_s["monthly_avg"]
        print(f"  {label:<25}: ${s['monthly_avg']:>6,.0f}/mo (Sharpe {s['sharpe']:.2f}) "
              f"{'↑' if imp > 0 else '↓'}${abs(imp):,.0f}")

    # Best combo: take the highest-Sharpe option
    best_key = max(results.keys(), key=lambda k: results[k]["sharpe"])
    best_s = results[best_key]

    # Also try: best vol filter + drop IB
    print("\n  Testing: Vol filter 25% + Drop IB (RSI@3c + MOM@6c) ...")
    skip_25 = skip_sets["25%"]
    combo_trades = []
    for name, c in [("RSI", 3), ("MOM", 6)]:
        all_t = run_strat(df_15m, name, ORIG[name], c)
        kept = [t for t in all_t if str(t.entry_time)[:10] not in skip_25]
        combo_trades.extend(kept)
    s_combo = summarize(combo_trades)
    results["Combined: DropIB+Vol25%"] = s_combo
    print(f"  ${s_combo['monthly_avg']:,.0f}/mo, Sharpe={s_combo['sharpe']:.2f}, DD=${s_combo['max_dd']:,.0f}")

    # Re-evaluate best
    best_key = max(results.keys(), key=lambda k: results[k]["sharpe"])
    best_s = results[best_key]

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  HTF SWING v5 — RESULTS")
    print("═" * 70)

    print(f"\n  ┌{'─'*25}┬{'─'*10}┬{'─'*9}┬{'─'*11}┬{'─'*11}┬{'─'*9}┐")
    print(f"  │{'Version':<25}│{'8yr$/mo':>10}│{'Mo prof':>9}│{'Worst Yr':>11}│{'Max DD':>11}│{'Sharpe':>9}│")
    print(f"  ├{'─'*25}┼{'─'*10}┼{'─'*9}┼{'─'*11}┼{'─'*11}┼{'─'*9}┤")
    for label, s in sorted(results.items(), key=lambda x: x[1]["monthly_avg"], reverse=True):
        pct = s["months_pos"] / max(s["n_months"], 1) * 100
        yr = yearly_from_monthly(s["monthly"])
        wy = min((y["total"] for y in yr.values()), default=0) if yr else 0
        print(f"  │{label:<25}│${s['monthly_avg']:>9,.0f}│{pct:>8.0f}%│${wy:>10,.0f}│${s['max_dd']:>10,.0f}│{s['sharpe']:>9.2f}│")
    print(f"  └{'─'*25}┴{'─'*10}┴{'─'*9}┴{'─'*11}┴{'─'*11}┴{'─'*9}┘")

    # Yearly breakdown for best
    print(f"\n  BEST SYSTEM: {best_key}")
    yr_best = yearly_from_monthly(best_s["monthly"])
    yr_orig = yearly_from_monthly(orig_s["monthly"])

    print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*10}┬{'─'*12}┐")
    print(f"  │{'Year':<6}│{'Total':>12}│{'Avg/Mo':>10}│{'Mo+':>10}│{'vs Orig':>12}│")
    print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*10}┼{'─'*12}┤")
    for yr in sorted(set(list(yr_best.keys()) + list(yr_orig.keys()))):
        b = yr_best.get(yr, {"total": 0, "months": 0, "months_pos": 0})
        o = yr_orig.get(yr, {"total": 0})
        diff = b["total"] - o["total"]
        avg = b["total"] / max(b["months"], 1)
        print(f"  │{yr:<6}│${b['total']:>+11,.0f}│${avg:>+9,.0f}│{b['months_pos']:>6}/{b['months']:<3}│${diff:>+11,.0f}│")
    print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*10}┴{'─'*12}┘")

    # Key improvements
    print(f"\n  KEY IMPROVEMENTS vs ORIGINAL:")
    for yr in ["2018", "2019"]:
        o_val = yr_orig.get(yr, {"total": 0})["total"]
        b_val = yr_best.get(yr, {"total": 0})["total"]
        print(f"    {yr}: ${o_val:,.0f} → ${b_val:,.0f} (Δ${b_val-o_val:+,.0f})")

    # Prop firm
    print(f"\n  PROP FIRM (Topstep 150K):")
    surv = best_s["max_dd"] > -4500
    print(f"    Max DD: ${best_s['max_dd']:,.0f} vs -$4,500 {'✅' if surv else '❌'}")

    # Assessment
    improved = best_s["monthly_avg"] > orig_s["monthly_avg"]
    print(f"\n  HONEST ASSESSMENT:")
    if improved:
        print(f"    Optimization improved from ${orig_s['monthly_avg']:,.0f} to ${best_s['monthly_avg']:,.0f}/month.")
        print(f"    Best approach: {best_key}")
        print(f"    Sharpe improved from {orig_s['sharpe']:.2f} to {best_s['sharpe']:.2f}")
    else:
        print(f"    No optimization meaningfully improved on the ${orig_s['monthly_avg']:,.0f}/month baseline.")
        print(f"    The original system IS the honest edge.")
    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "monthly"}
                    for k, v in results.items()},
        "best": best_key,
        "best_monthly_avg": best_s["monthly_avg"],
        "best_sharpe": best_s["sharpe"],
    }
    out = REPORTS_DIR / "htf_swing_v5.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
