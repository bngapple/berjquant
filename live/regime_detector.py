#!/usr/bin/env python3
"""
Regime Detector — Analyzes NQ volatility and system performance by regime.

Usage:
    python3 live/regime_detector.py
"""

import gc
import json
import sys
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import run_htf_swing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from run_htf_swing import (
    load_and_resample, extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    calc_atr, TICK_SIZE, POINT_VALUE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
FLATTEN_TIME = 1645
CONTRACTS = 3

HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True, "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def run_system(df, params):
    o, h, l, c, ts, hm = extract_arrays(df)
    trades = []
    p = params["RSI"]
    sigs = sig_rsi_extreme(df, p["period"], p["ob"], p["os"])
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "RSI", FLATTEN_TIME))
    p = params["IB"]
    sigs = sig_ib_breakout(df, p["ib_filter"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "IB", FLATTEN_TIME))
    p = params["MOM"]
    sigs = sig_momentum_bar(df, p["atr_mult"], p["vol_mult"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "MOM", FLATTEN_TIME))
    return trades


def main():
    t0 = _time.time()
    print("═" * 75)
    print("  REGIME DETECTOR — Hybrid v2")
    print("═" * 75)

    # Load 8yr 15m data
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    df_8yr = resample_15m_rth(combined)
    del combined, df1, df2, df3; gc.collect()

    # ── Build daily bars from 15m ───────────────────────────────────
    dates = df_8yr["date_et"].to_list()
    highs = df_8yr["high"].to_numpy()
    lows = df_8yr["low"].to_numpy()
    opens = df_8yr["open"].to_numpy()
    closes = df_8yr["close"].to_numpy()

    daily_bars = {}
    for i in range(len(df_8yr)):
        d = dates[i]
        if d not in daily_bars:
            daily_bars[d] = {"high": highs[i], "low": lows[i], "open": opens[i], "close": closes[i]}
        else:
            daily_bars[d]["high"] = max(daily_bars[d]["high"], highs[i])
            daily_bars[d]["low"] = min(daily_bars[d]["low"], lows[i])
            daily_bars[d]["close"] = closes[i]

    sorted_dates = sorted(daily_bars.keys())
    daily_ranges = np.array([daily_bars[d]["high"] - daily_bars[d]["low"] for d in sorted_dates])
    daily_highs = np.array([daily_bars[d]["high"] for d in sorted_dates])
    daily_lows = np.array([daily_bars[d]["low"] for d in sorted_dates])
    daily_closes = np.array([daily_bars[d]["close"] for d in sorted_dates])

    print(f"  {len(sorted_dates)} trading days ({sorted_dates[0]} to {sorted_dates[-1]})")
    print(f"  Daily range: mean={np.mean(daily_ranges):.0f}pts, "
          f"median={np.median(daily_ranges):.0f}pts, std={np.std(daily_ranges):.0f}pts")

    # ── Compute percentiles for regime boundaries ──────────────────
    p25 = float(np.percentile(daily_ranges, 25))
    p75 = float(np.percentile(daily_ranges, 75))
    p95 = float(np.percentile(daily_ranges, 95))

    print(f"\n  Regime boundaries (from 8yr daily range percentiles):")
    print(f"    Low vol:    < {p25:.0f} pts  (P25)")
    print(f"    Normal vol: {p25:.0f} – {p75:.0f} pts  (P25-P75)")
    print(f"    High vol:   {p75:.0f} – {p95:.0f} pts  (P75-P95)")
    print(f"    Extreme:    > {p95:.0f} pts  (P95)")

    # ── Classify each day ──────────────────────────────────────────
    def classify(rng):
        if rng < p25:
            return "Low vol"
        elif rng < p75:
            return "Normal vol"
        elif rng < p95:
            return "High vol"
        else:
            return "Extreme"

    day_regime = {d: classify(daily_ranges[i]) for i, d in enumerate(sorted_dates)}

    # ── Run trades ─────────────────────────────────────────────────
    trades = run_system(df_8yr, HYBRID_V2)
    print(f"  {len(trades):,} trades")

    # ── Bucket trades by regime of their trading day ───────────────
    regime_trades = defaultdict(list)
    regime_daily_pnl = defaultdict(lambda: defaultdict(float))

    for t in trades:
        d = str(t.entry_time)[:10]
        # Convert to date object to match day_regime keys
        from datetime import date as date_cls
        try:
            d_obj = date_cls.fromisoformat(d)
        except:
            continue
        regime = day_regime.get(d_obj, "Unknown")
        if regime == "Unknown":
            continue
        regime_trades[regime].append(t)
        regime_daily_pnl[regime][d] += t.net_pnl

    # Count days per regime
    regime_days = defaultdict(int)
    for d, r in day_regime.items():
        regime_days[r] += 1

    # ── Print regime analysis ──────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  REGIME ANALYSIS — Hybrid v2")
    print("━" * 75)

    print(f"\n  {'Regime':<12} {'Days':>5} {'% Days':>7} {'Avg Day $':>10} {'Mo Equiv':>10} {'Trades/Day':>11}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*10} {'─'*10} {'─'*11}")

    regime_order = ["Low vol", "Normal vol", "High vol", "Extreme"]
    regime_report = {}
    for regime in regime_order:
        days = regime_days[regime]
        pct = days / len(sorted_dates) * 100
        daily_pnls = list(regime_daily_pnl[regime].values())
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        mo_equiv = avg_daily * 21  # ~21 trading days/month
        n_trades = len(regime_trades[regime])
        trades_per_day = n_trades / days if days > 0 else 0

        regime_report[regime] = {
            "days": days, "pct": pct, "avg_daily": avg_daily,
            "mo_equiv": mo_equiv, "trades_per_day": trades_per_day,
            "total_trades": n_trades,
        }

        print(f"  {regime:<12} {days:>5} {pct:>6.0f}% ${avg_daily:>+9,.0f} ${mo_equiv:>+9,.0f} {trades_per_day:>11.1f}")

    # ── Current regime ─────────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  CURRENT REGIME (last 20 trading days)")
    print("━" * 75)

    last_20 = sorted_dates[-20:]
    last_20_ranges = [daily_bars[d]["high"] - daily_bars[d]["low"] for d in last_20]
    avg_range_20 = np.mean(last_20_ranges)
    current_regime = classify(avg_range_20)

    print(f"\n  Avg daily range (last 20): {avg_range_20:.0f} pts")
    print(f"  Regime: {current_regime}")
    expected = regime_report[current_regime]["mo_equiv"]
    print(f"  Expected monthly P&L in this regime: ${expected:>+,.0f}")

    # ── Regime transitions ─────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  REGIME TRANSITION ANALYSIS")
    print("━" * 75)

    # Compute 20-day rolling avg range and track regime transitions
    roll_window = 20
    rolling_regimes = []
    for i in range(roll_window, len(sorted_dates)):
        avg_r = np.mean(daily_ranges[i-roll_window:i])
        rolling_regimes.append((sorted_dates[i], classify(avg_r)))

    # Find transitions
    transitions = defaultdict(list)
    for i in range(1, len(rolling_regimes)):
        prev_r = rolling_regimes[i-1][1]
        curr_r = rolling_regimes[i][1]
        if prev_r != curr_r:
            d = rolling_regimes[i][0]
            transitions[(prev_r, curr_r)].append(d)

    # P&L around transitions
    interesting = [("High vol", "Normal vol"), ("Normal vol", "Low vol"),
                   ("Low vol", "Normal vol"), ("Normal vol", "High vol")]

    for from_r, to_r in interesting:
        dates_tr = transitions.get((from_r, to_r), [])
        if not dates_tr:
            continue
        # Look at 20-day P&L before and after each transition
        before_pnls = []
        after_pnls = []
        for d in dates_tr:
            d_idx = sorted_dates.index(d) if d in sorted_dates else -1
            if d_idx < 20 or d_idx > len(sorted_dates) - 21:
                continue
            before_days = sorted_dates[d_idx-20:d_idx]
            after_days = sorted_dates[d_idx:d_idx+20]
            b_pnl = sum(regime_daily_pnl.get(day_regime.get(dd, ""), {}).get(str(dd), 0)
                        for dd in before_days
                        for _ in [None])  # flatten
            # Simpler: use all trades in those days
            b_total = 0
            a_total = 0
            for dd in before_days:
                dd_str = str(dd)
                for regime, dpnl in regime_daily_pnl.items():
                    b_total += dpnl.get(dd_str, 0)
            for dd in after_days:
                dd_str = str(dd)
                for regime, dpnl in regime_daily_pnl.items():
                    a_total += dpnl.get(dd_str, 0)
            before_pnls.append(b_total)
            after_pnls.append(a_total)

        if before_pnls and after_pnls:
            avg_before = np.mean(before_pnls) / 20 * 21  # monthly equiv
            avg_after = np.mean(after_pnls) / 20 * 21
            change = avg_after - avg_before
            print(f"\n  {from_r} → {to_r} ({len(dates_tr)} occurrences):")
            print(f"    Avg P&L before (monthly): ${avg_before:>+,.0f}")
            print(f"    Avg P&L after (monthly):  ${avg_after:>+,.0f}")
            print(f"    Change: ${change:>+,.0f}/month")

    # ── Breakeven analysis ─────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  BREAKEVEN ANALYSIS")
    print("━" * 75)

    # Group daily P&L by daily range buckets
    range_pnl = defaultdict(list)
    for t in trades:
        d = str(t.entry_time)[:10]
        from datetime import date as date_cls
        try:
            d_obj = date_cls.fromisoformat(d)
        except:
            continue
        if d_obj not in daily_bars:
            continue
        r = daily_bars[d_obj]["high"] - daily_bars[d_obj]["low"]
        bucket = int(r / 25) * 25  # 25-point buckets
        range_pnl[bucket].append(t.net_pnl)

    # Find breakeven range
    print(f"\n  {'Range (pts)':<14} {'Days':>5} {'Avg P&L':>10} {'Win Rate':>9}")
    print(f"  {'─'*14} {'─'*5} {'─'*10} {'─'*9}")

    breakeven_range = None
    buckets_sorted = sorted(range_pnl.keys())
    for bucket in buckets_sorted:
        pnls = range_pnl[bucket]
        if len(pnls) < 5:
            continue
        avg = np.mean(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        # Count unique days
        print(f"  {bucket:>4}-{bucket+25:<4} pts  {len(pnls):>5} ${avg:>+9,.0f} {wr:>8.1f}%")
        if avg < 0 and breakeven_range is None:
            breakeven_range = bucket

    # More precise breakeven via daily aggregation
    daily_range_pnl = defaultdict(lambda: {"pnl": 0, "range": 0})
    for t in trades:
        d = str(t.entry_time)[:10]
        from datetime import date as date_cls
        try:
            d_obj = date_cls.fromisoformat(d)
        except:
            continue
        if d_obj in daily_bars:
            daily_range_pnl[d]["pnl"] += t.net_pnl
            daily_range_pnl[d]["range"] = daily_bars[d_obj]["high"] - daily_bars[d_obj]["low"]

    # Find breakeven by sorting
    day_data = [(v["range"], v["pnl"]) for v in daily_range_pnl.values() if v["range"] > 0]
    day_data.sort()
    # Rolling average P&L from lowest range upward
    cum_pnl = 0
    cum_days = 0
    be_range = 0
    for r, p in day_data:
        cum_pnl += p
        cum_days += 1
        if cum_pnl > 0 and be_range == 0:
            be_range = r

    # Count days below breakeven
    below_be = sum(1 for r, _ in day_data if r < be_range)
    pct_below = below_be / len(day_data) * 100 if day_data else 0

    # Longest streak below breakeven
    max_streak = 0
    cur_streak = 0
    for i, d in enumerate(sorted_dates):
        if d in daily_bars:
            r = daily_bars[d]["high"] - daily_bars[d]["low"]
            if r < be_range:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0

    print(f"\n  Daily range where cumulative system P&L turns positive: ~{be_range:.0f} pts")
    print(f"  Days below breakeven range: {below_be} ({pct_below:.0f}%)")
    print(f"  Longest streak below breakeven: {max_streak} days")

    if max_streak > 0:
        print(f"\n  MONITORING THRESHOLD: If NQ daily ranges stay below {be_range:.0f} pts")
        print(f"  for 10+ trading days, the edge is temporarily gone. Consider reducing size.")

    # ── Save ───────────────────────────────────────────────────────
    report = {
        "timestamp": str(datetime.now()),
        "regime_boundaries": {"p25": p25, "p75": p75, "p95": p95},
        "regime_analysis": {r: {k: v for k, v in d.items()} for r, d in regime_report.items()},
        "current_regime": {
            "avg_range_20d": float(avg_range_20),
            "regime": current_regime,
            "expected_monthly": expected,
        },
        "breakeven": {
            "range_pts": be_range,
            "days_below": below_be,
            "pct_below": pct_below,
            "max_streak": max_streak,
        },
    }
    out = REPORTS_DIR / "regime_analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 75)


if __name__ == "__main__":
    main()
