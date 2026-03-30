#!/usr/bin/env python3
"""
8-Year Annualized Sharpe Ratio — Hybrid v2.

Usage:
    python3 run_htf_swing_v3_sharpe.py
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
    print("═" * 70)
    print("  8-YEAR SHARPE RATIO — Hybrid v2")
    print("═" * 70)

    # Load 8yr
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    df_8yr = resample_15m_rth(combined)
    del combined, df1, df2, df3; gc.collect()

    trades = run_system(df_8yr, HYBRID_V2)
    print(f"  {len(trades):,} trades")

    # Daily P&L
    daily = defaultdict(float)
    for t in trades:
        daily[str(t.entry_time)[:10]] += t.net_pnl
    daily_sorted = sorted(daily.items())
    daily_vals = np.array([v for _, v in daily_sorted])
    daily_dates = [d for d, _ in daily_sorted]

    # Monthly P&L
    monthly = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
    monthly_sorted = sorted(monthly.items())
    monthly_vals = np.array([v for _, v in monthly_sorted])
    monthly_keys = [m for m, _ in monthly_sorted]

    # Sharpe calculations
    daily_sharpe = float(np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)) if np.std(daily_vals) > 0 else 0
    monthly_sharpe = float(np.mean(monthly_vals) / np.std(monthly_vals) * np.sqrt(12)) if np.std(monthly_vals) > 0 else 0

    # Per-year Sharpe
    daily_by_year = defaultdict(list)
    for d, v in daily_sorted:
        daily_by_year[d[:4]].append(v)

    # Output
    print(f"\n  Period: {daily_dates[0]} – {daily_dates[-1]}")
    print(f"  Trading days: {len(daily_vals):,}")
    print(f"  Trading months: {len(monthly_vals)}")

    print(f"\n  Daily Sharpe (annualized):   {daily_sharpe:.2f}")
    print(f"  Monthly Sharpe (annualized): {monthly_sharpe:.2f}")

    print(f"\n  For comparison:")
    print(f"    Y2 OOS Sharpe:      10.9")
    print(f"    S&P 500 8yr Sharpe: ~0.8-1.0")
    print(f"    Elite hedge fund:   2.0-3.0")

    print(f"\n  Breakdown by year:")
    print(f"  {'Year':<6} {'Sharpe':>8} {'Days':>6} {'Mean $/day':>11} {'Std $/day':>11}")
    print(f"  {'─'*6} {'─'*8} {'─'*6} {'─'*11} {'─'*11}")
    year_sharpes = {}
    for y in sorted(daily_by_year.keys()):
        vals = np.array(daily_by_year[y])
        s = float(np.mean(vals) / np.std(vals) * np.sqrt(252)) if np.std(vals) > 0 else 0
        year_sharpes[y] = s
        print(f"  {y:<6} {s:>8.2f} {len(vals):>6} ${np.mean(vals):>+10,.0f} ${np.std(vals):>10,.0f}")

    print(f"\n  Monthly P&L statistics:")
    print(f"    Mean:         ${np.mean(monthly_vals):>+10,.0f}")
    print(f"    Median:       ${np.median(monthly_vals):>+10,.0f}")
    print(f"    Std:          ${np.std(monthly_vals):>+10,.0f}")
    print(f"    Min:          ${np.min(monthly_vals):>+10,.0f}")
    print(f"    Max:          ${np.max(monthly_vals):>+10,.0f}")
    pct_pos = sum(1 for v in monthly_vals if v > 0) / len(monthly_vals) * 100
    print(f"    %% positive:   {pct_pos:.0f}%")

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "period": f"{daily_dates[0]} to {daily_dates[-1]}",
        "trading_days": len(daily_vals),
        "trading_months": len(monthly_vals),
        "daily_sharpe": daily_sharpe,
        "monthly_sharpe": monthly_sharpe,
        "year_sharpes": year_sharpes,
        "monthly_stats": {
            "mean": float(np.mean(monthly_vals)),
            "median": float(np.median(monthly_vals)),
            "std": float(np.std(monthly_vals)),
            "min": float(np.min(monthly_vals)),
            "max": float(np.max(monthly_vals)),
            "pct_positive": pct_pos,
        },
    }
    out = REPORTS_DIR / "hybrid_v2_sharpe.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 70)


if __name__ == "__main__":
    main()
