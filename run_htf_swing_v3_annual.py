#!/usr/bin/env python3
"""
Hybrid v2 — Year-by-Year Breakdown (8 years).

Usage:
    python3 run_htf_swing_v3_annual.py
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
    load_and_resample, extract_arrays, backtest, rt_cost,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
FLATTEN_TIME = 1645
CONTRACTS = 3

CURRENT = {
    "RSI": {"period": 7,  "ob": 70, "os": 30, "sl_pts": 15,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 20,  "tp_pts": 120, "hold": 10},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}

HYBRID_V2 = {
    "RSI": {"period": 5,  "ob": 65, "os": 35, "sl_pts": 10,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 10,  "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


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


def year_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0,
                "best_month": 0, "worst_month": 0, "monthly": {}}
    pnls = [t.net_pnl for t in trades]
    w = sum(1 for p in pnls if p > 0)
    monthly = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
    nm = max(len(monthly), 1)
    total = sum(pnls)
    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": total / nm,
        "best_month": max(monthly.values()) if monthly else 0,
        "worst_month": min(monthly.values()) if monthly else 0,
        "monthly": dict(monthly),
    }


def main():
    t0 = _time.time()
    print("═" * 85)
    print("  HYBRID V2 — ANNUAL BREAKDOWN (8 years)")
    print("═" * 85)

    # Load 8yr data
    print("\n  Loading 8-year dataset...")
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    df_8yr = resample_15m_rth(combined)
    print(f"  {len(df_8yr):,} bars ({df_8yr['timestamp'].min()} to {df_8yr['timestamp'].max()})")
    del combined, df1, df2, df3
    gc.collect()

    # Run both configs on full 8yr
    print("\n  Running Hybrid v2...")
    all_hyb, strats_hyb = run_system(df_8yr, HYBRID_V2)
    print(f"    {len(all_hyb)} trades")

    print("  Running Current v3...")
    all_cur, strats_cur = run_system(df_8yr, CURRENT)
    print(f"    {len(all_cur)} trades")

    # Group by year
    def group_by_year(trades):
        by_year = defaultdict(list)
        for t in trades:
            y = str(t.entry_time)[:4]
            by_year[y].append(t)
        return dict(by_year)

    def group_strats_by_year(per_strat):
        result = {}
        for name, trades in per_strat.items():
            by_year = defaultdict(list)
            for t in trades:
                y = str(t.entry_time)[:4]
                by_year[y].append(t)
            result[name] = dict(by_year)
        return result

    hyb_by_year = group_by_year(all_hyb)
    cur_by_year = group_by_year(all_cur)
    hyb_strats_by_year = group_strats_by_year(strats_hyb)
    cur_strats_by_year = group_strats_by_year(strats_cur)

    years = sorted(set(list(hyb_by_year.keys()) + list(cur_by_year.keys())))

    # ── Annual breakdown (Hybrid v2) ───────────────────────────────
    print(f"\n{'━' * 85}")
    print("  HYBRID V2 — ANNUAL BREAKDOWN")
    print("━" * 85)

    print(f"\n  {'Year':<6} {'Net P&L':>10} {'Trades':>7} {'WR':>7} {'Avg/Mo':>10} "
          f"{'Best Mo':>10} {'Worst Mo':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")

    report_hyb = {}
    total_pnl = 0
    total_trades = 0
    for y in years:
        trades = hyb_by_year.get(y, [])
        m = year_metrics(trades)
        total_pnl += m["pnl"]
        total_trades += m["n"]
        report_hyb[y] = m
        mark = "+" if m["pnl"] > 0 else "-"
        print(f"  {y:<6} ${m['pnl']:>+9,.0f} {m['n']:>7} {m['wr']:>6.1f}% "
              f"${m['monthly_avg']:>+9,.0f} ${m['best_month']:>+9,.0f} ${m['worst_month']:>+9,.0f}  {mark}")

    print(f"  {'─'*6} {'─'*10} {'─'*7} {'─'*7} {'─'*10}")
    n_months = sum(len(year_metrics(hyb_by_year.get(y, [])).get("monthly", {})) for y in years)
    print(f"  {'TOTAL':<6} ${total_pnl:>+9,.0f} {total_trades:>7} "
          f"{'':>7} ${total_pnl/max(n_months,1):>+9,.0f}")

    # ── Comparison table ───────────────────────────────────────────
    print(f"\n{'━' * 85}")
    print("  COMPARISON — Current v3 vs Hybrid v2 (annual P&L)")
    print("━" * 85)

    print(f"\n  {'Year':<6} {'Current v3':>12} {'Hybrid v2':>12} {'Delta':>12} {'Delta%':>8} ")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

    report_cmp = {}
    cum_cur = 0
    cum_hyb = 0
    for y in years:
        c_trades = cur_by_year.get(y, [])
        h_trades = hyb_by_year.get(y, [])
        c_pnl = sum(t.net_pnl for t in c_trades)
        h_pnl = sum(t.net_pnl for t in h_trades)
        delta = h_pnl - c_pnl
        pct = (delta / abs(c_pnl) * 100) if c_pnl != 0 else 0
        cum_cur += c_pnl
        cum_hyb += h_pnl
        report_cmp[y] = {"current": c_pnl, "hybrid_v2": h_pnl, "delta": delta}

        winner = "HYB" if h_pnl > c_pnl else "CUR"
        print(f"  {y:<6} ${c_pnl:>+11,.0f} ${h_pnl:>+11,.0f} ${delta:>+11,.0f} {pct:>+7.0f}%  {winner}")

    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'TOTAL':<6} ${cum_cur:>+11,.0f} ${cum_hyb:>+11,.0f} ${cum_hyb - cum_cur:>+11,.0f}")

    # ── Per-strategy breakdown for 2018 and 2019 ──────────────────
    print(f"\n{'━' * 85}")
    print("  PER-STRATEGY BREAKDOWN — 2018 & 2019 (did tighter RSI SL reduce losses?)")
    print("━" * 85)

    for y in ["2018", "2019"]:
        print(f"\n  {y}:")
        print(f"  {'Strategy':<6} {'Config':<12} {'P&L':>10} {'Trades':>7} {'WR':>7} {'Avg Loss':>10}")
        print(f"  {'─'*6} {'─'*12} {'─'*10} {'─'*7} {'─'*7} {'─'*10}")

        for name in ["RSI", "IB", "MOM"]:
            # Current
            c_trades = cur_strats_by_year.get(name, {}).get(y, [])
            c_pnl = sum(t.net_pnl for t in c_trades)
            c_wr = sum(1 for t in c_trades if t.net_pnl > 0) / len(c_trades) * 100 if c_trades else 0
            c_losses = [t.net_pnl for t in c_trades if t.net_pnl < 0]
            c_avg_loss = np.mean(c_losses) if c_losses else 0

            # Hybrid
            h_trades = hyb_strats_by_year.get(name, {}).get(y, [])
            h_pnl = sum(t.net_pnl for t in h_trades)
            h_wr = sum(1 for t in h_trades if t.net_pnl > 0) / len(h_trades) * 100 if h_trades else 0
            h_losses = [t.net_pnl for t in h_trades if t.net_pnl < 0]
            h_avg_loss = np.mean(h_losses) if h_losses else 0

            print(f"  {name:<6} {'Current':<12} ${c_pnl:>+9,.0f} {len(c_trades):>7} {c_wr:>6.1f}% ${c_avg_loss:>+9,.2f}")
            print(f"  {'':<6} {'Hybrid v2':<12} ${h_pnl:>+9,.0f} {len(h_trades):>7} {h_wr:>6.1f}% ${h_avg_loss:>+9,.2f}")
            delta = h_pnl - c_pnl
            print(f"  {'':<6} {'Delta':<12} ${delta:>+9,.0f} {len(h_trades) - len(c_trades):>+7} "
                  f"{h_wr - c_wr:>+6.1f}% ${h_avg_loss - c_avg_loss:>+9,.2f}")

        # Year total
        c_total = sum(t.net_pnl for t in cur_by_year.get(y, []))
        h_total = sum(t.net_pnl for t in hyb_by_year.get(y, []))
        print(f"\n  {y} Total: Current ${c_total:>+,.0f} → Hybrid ${h_total:>+,.0f} "
              f"(${h_total - c_total:>+,.0f})")

    # ── Save ───────────────────────────────────────────────────────
    report = {
        "timestamp": str(datetime.now()),
        "hybrid_v2_annual": {y: {k: v for k, v in m.items() if k != "monthly"}
                             for y, m in report_hyb.items()},
        "comparison": report_cmp,
        "total_hybrid_v2": cum_hyb,
        "total_current": cum_cur,
    }
    out = REPORTS_DIR / "hybrid_v2_annual.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 85)


if __name__ == "__main__":
    main()
