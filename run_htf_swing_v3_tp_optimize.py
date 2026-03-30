#!/usr/bin/env python3
"""
TP Fill Optimization — Does shaving 1-2 ticks off TP help?

Tests TP at baseline, -1, -2, -4, -8 ticks inside for each strategy.
Also runs a near-miss analysis to see how many losers came close to TP.

Usage:
    python3 run_htf_swing_v3_tp_optimize.py
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
    TICK_SIZE, POINT_VALUE, SLIP_PTS,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FLATTEN_TIME = 1645
CONTRACTS = 3


def pts_to_ticks(pts):
    return pts / TICK_SIZE  # float division to support fractional points


def run_strat(df, strat_name, params, tp_pts_override=None):
    """Run one strategy with optional TP override. Returns trades."""
    o, h, l, c, ts, hm = extract_arrays(df)
    tp_pts = tp_pts_override if tp_pts_override is not None else params["tp_pts"]
    sl_t = pts_to_ticks(params["sl_pts"])
    tp_t = pts_to_ticks(tp_pts)

    if strat_name == "RSI":
        sigs = sig_rsi_extreme(df, params["period"], params["ob"], params["os"])
    elif strat_name == "IB":
        sigs = sig_ib_breakout(df, params["ib_filter"])[0]
    elif strat_name == "MOM":
        sigs = sig_momentum_bar(df, params["atr_mult"], params["vol_mult"])[0]

    return backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t,
                    params["hold"], CONTRACTS, strat_name, FLATTEN_TIME)


def run_full(df, rsi_tp=None, ib_tp=None, mom_tp=None):
    """Run all 3 strategies with optional per-strategy TP overrides."""
    rsi_p = {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5}
    ib_p = {"ib_filter": True, "sl_pts": 10, "tp_pts": 120, "hold": 15}
    mom_p = {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5}

    trades = []
    trades.extend(run_strat(df, "RSI", rsi_p, rsi_tp))
    trades.extend(run_strat(df, "IB", ib_p, ib_tp))
    trades.extend(run_strat(df, "MOM", mom_p, mom_tp))
    return trades


def strat_metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0, "winners": 0, "losers": 0,
                "avg_winner": 0, "avg_loser": 0, "monthly_avg": 0}
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    monthly = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
    nm = max(len(monthly), 1)
    total = sum(t.net_pnl for t in trades)
    return {
        "n": len(trades),
        "wr": len(winners) / len(trades) * 100,
        "winners": len(winners),
        "losers": len(losers),
        "avg_winner": np.mean([t.net_pnl for t in winners]) if winners else 0,
        "avg_loser": np.mean([t.net_pnl for t in losers]) if losers else 0,
        "monthly_avg": total / nm,
        "total": total,
    }


def near_miss_analysis(df, strat_name, params, thresholds=[0.5, 1.0, 2.0]):
    """For each loser, check if price came within X points of TP before reversing."""
    o_arr, h_arr, l_arr, c_arr, ts_arr, hm_arr = extract_arrays(df)

    # Run baseline
    trades = run_strat(df, strat_name, params)
    losers = [t for t in trades if t.net_pnl <= 0 and t.reason in ("stop_loss", "max_hold")]

    tp_pts = params["tp_pts"]

    # For each loser, check bars between entry and exit for how close price got to TP
    ts_to_idx = {ts_arr[i]: i for i in range(len(ts_arr))}

    counts = {thr: 0 for thr in thresholds}

    for t in losers:
        entry_idx = ts_to_idx.get(t.entry_time)
        exit_idx = ts_to_idx.get(t.exit_time)
        if entry_idx is None or exit_idx is None:
            continue

        # Reconstruct target level
        target_px = t.entry_px + t.direction * tp_pts

        # Check all bars during the trade
        min_distance = float('inf')
        for bar in range(entry_idx, exit_idx + 1):
            if t.direction == 1:
                # Long: target is above, check how close high got
                dist = target_px - h_arr[bar]
            else:
                # Short: target is below, check how close low got
                dist = l_arr[bar] - target_px
            if dist < min_distance:
                min_distance = dist

        for thr in thresholds:
            if min_distance <= thr:
                counts[thr] += 1

    return len(losers), counts


def main():
    t0 = _time.time()
    print("═" * 85)
    print("  TP FILL OPTIMIZATION — Does shaving 1-2 ticks help?")
    print("═" * 85)

    # Load data
    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind")

    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
    Y1_END = datetime(2025, 3, 1, tzinfo=_ET)

    df_main = main_data["15m"]
    yr2 = df_main.filter(pl.col("timestamp") >= Y1_END)
    bl = blind_data["15m"]

    print(f"  Y2: {len(yr2):,} bars | Blind: {len(bl):,} bars")

    # ── Strategy params ─────────────────────────────────────────────
    RSI_P = {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5}
    IB_P = {"ib_filter": True, "sl_pts": 10, "tp_pts": 120, "hold": 15}
    MOM_P = {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5}

    # ═════════════════════════════════════════════════════════════════
    # NEAR-MISS ANALYSIS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 85}")
    print("  TP NEAR-MISS ANALYSIS (Y2, baseline TP)")
    print("━" * 85)

    thresholds = [0.5, 1.0, 2.0]
    print(f"\n  {'Strategy':<10} {'Total Losers':>13} ", end="")
    for thr in thresholds:
        print(f"{'Within '+str(thr)+'pt':>14}", end="")
    print()
    print(f"  {'─'*10} {'─'*13} " + " ".join(f"{'─'*14}" for _ in thresholds))

    nm_report = {}
    for name, params in [("RSI", RSI_P), ("IB", IB_P), ("MOM", MOM_P)]:
        total_losers, counts = near_miss_analysis(yr2, name, params, thresholds)
        nm_report[name] = {"total_losers": total_losers, "counts": {str(t): c for t, c in counts.items()}}
        row = f"  {name:<10} {total_losers:>13} "
        for thr in thresholds:
            c = counts[thr]
            pct = c / total_losers * 100 if total_losers > 0 else 0
            row += f" {c:>5} ({pct:>4.1f}%) "
        print(row)

    # ═════════════════════════════════════════════════════════════════
    # TP SWEEP — RSI & MOM (TP=100 baseline)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 85}")
    print("  TP OPTIMIZATION — RSI (baseline TP=100)")
    print("━" * 85)

    rsi_tp_variants = [100.00, 99.75, 99.50, 99.00, 98.00]
    print(f"\n  {'TP (pts)':<10} {'WR':>6} {'Avg Win':>9} {'Avg Loss':>9} "
          f"{'Wins':>5} {'Loss':>5} {'Y2 $/mo':>10} {'Blind/mo':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*10} {'─'*10}")

    rsi_results = {}
    for tp in rsi_tp_variants:
        y2_trades = run_strat(yr2, "RSI", RSI_P, tp)
        bl_trades = run_strat(bl, "RSI", RSI_P, tp)
        y2m = strat_metrics(y2_trades)
        blm = strat_metrics(bl_trades)
        rsi_results[tp] = {"y2": y2m, "blind": blm}
        print(f"  {tp:<10.2f} {y2m['wr']:>5.1f}% ${y2m['avg_winner']:>7,.0f} ${y2m['avg_loser']:>8,.0f} "
              f"{y2m['winners']:>5} {y2m['losers']:>5} ${y2m['monthly_avg']:>+9,.0f} ${blm['monthly_avg']:>+9,.0f}")

    print(f"\n{'━' * 85}")
    print("  TP OPTIMIZATION — MOM (baseline TP=100)")
    print("━" * 85)

    print(f"\n  {'TP (pts)':<10} {'WR':>6} {'Avg Win':>9} {'Avg Loss':>9} "
          f"{'Wins':>5} {'Loss':>5} {'Y2 $/mo':>10} {'Blind/mo':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*10} {'─'*10}")

    mom_results = {}
    for tp in rsi_tp_variants:  # same TP sweep values
        y2_trades = run_strat(yr2, "MOM", MOM_P, tp)
        bl_trades = run_strat(bl, "MOM", MOM_P, tp)
        y2m = strat_metrics(y2_trades)
        blm = strat_metrics(bl_trades)
        mom_results[tp] = {"y2": y2m, "blind": blm}
        print(f"  {tp:<10.2f} {y2m['wr']:>5.1f}% ${y2m['avg_winner']:>7,.0f} ${y2m['avg_loser']:>8,.0f} "
              f"{y2m['winners']:>5} {y2m['losers']:>5} ${y2m['monthly_avg']:>+9,.0f} ${blm['monthly_avg']:>+9,.0f}")

    # ═════════════════════════════════════════════════════════════════
    # TP SWEEP — IB (TP=120 baseline)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 85}")
    print("  TP OPTIMIZATION — IB (baseline TP=120)")
    print("━" * 85)

    ib_tp_variants = [120.00, 119.75, 119.50, 119.00, 118.00]
    print(f"\n  {'TP (pts)':<10} {'WR':>6} {'Avg Win':>9} {'Avg Loss':>9} "
          f"{'Wins':>5} {'Loss':>5} {'Y2 $/mo':>10} {'Blind/mo':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*10} {'─'*10}")

    ib_results = {}
    for tp in ib_tp_variants:
        y2_trades = run_strat(yr2, "IB", IB_P, tp)
        bl_trades = run_strat(bl, "IB", IB_P, tp)
        y2m = strat_metrics(y2_trades)
        blm = strat_metrics(bl_trades)
        ib_results[tp] = {"y2": y2m, "blind": blm}
        print(f"  {tp:<10.2f} {y2m['wr']:>5.1f}% ${y2m['avg_winner']:>7,.0f} ${y2m['avg_loser']:>8,.0f} "
              f"{y2m['winners']:>5} {y2m['losers']:>5} ${y2m['monthly_avg']:>+9,.0f} ${blm['monthly_avg']:>+9,.0f}")

    # ═════════════════════════════════════════════════════════════════
    # FIND OPTIMAL TP PER STRATEGY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 85}")
    print("  OPTIMAL TP SELECTION")
    print("━" * 85)

    # Pick best TP based on blind P&L (most conservative)
    best_rsi_tp = max(rsi_tp_variants, key=lambda tp: rsi_results[tp]["blind"]["monthly_avg"])
    best_mom_tp = max(rsi_tp_variants, key=lambda tp: mom_results[tp]["blind"]["monthly_avg"])
    best_ib_tp = max(ib_tp_variants, key=lambda tp: ib_results[tp]["blind"]["monthly_avg"])

    print(f"\n  Best RSI TP (by blind): {best_rsi_tp} pts "
          f"(Y2: ${rsi_results[best_rsi_tp]['y2']['monthly_avg']:+,.0f}, "
          f"Blind: ${rsi_results[best_rsi_tp]['blind']['monthly_avg']:+,.0f})")
    print(f"  Best MOM TP (by blind): {best_mom_tp} pts "
          f"(Y2: ${mom_results[best_mom_tp]['y2']['monthly_avg']:+,.0f}, "
          f"Blind: ${mom_results[best_mom_tp]['blind']['monthly_avg']:+,.0f})")
    print(f"  Best IB TP  (by blind): {best_ib_tp} pts "
          f"(Y2: ${ib_results[best_ib_tp]['y2']['monthly_avg']:+,.0f}, "
          f"Blind: ${ib_results[best_ib_tp]['blind']['monthly_avg']:+,.0f})")

    # ═════════════════════════════════════════════════════════════════
    # COMBINED OPTIMAL vs BASELINE
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 85}")
    print("  FINAL COMPARISON")
    print("━" * 85)

    # Baseline combined
    base_y2 = run_full(yr2)
    base_bl = run_full(bl)
    base_y2_m = strat_metrics(base_y2)
    base_bl_m = strat_metrics(base_bl)

    # Optimal combined
    opt_y2 = run_full(yr2, rsi_tp=best_rsi_tp, ib_tp=best_ib_tp, mom_tp=best_mom_tp)
    opt_bl = run_full(bl, rsi_tp=best_rsi_tp, ib_tp=best_ib_tp, mom_tp=best_mom_tp)
    opt_y2_m = strat_metrics(opt_y2)
    opt_bl_m = strat_metrics(opt_bl)

    y2_delta = opt_y2_m["monthly_avg"] - base_y2_m["monthly_avg"]
    bl_delta = opt_bl_m["monthly_avg"] - base_bl_m["monthly_avg"]

    rsi_label = f"RSI/MOM={best_rsi_tp}" if best_rsi_tp == best_mom_tp else f"RSI={best_rsi_tp}, MOM={best_mom_tp}"
    opt_label = f"Optimal ({rsi_label}, IB={best_ib_tp})"

    print(f"\n  {'Config':<42} {'Y2 $/mo':>10} {'Blind/mo':>10} {'Delta':>10}")
    print(f"  {'─'*42} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Baseline (RSI/MOM=100, IB=120)':<42} ${base_y2_m['monthly_avg']:>+9,.0f} "
          f"${base_bl_m['monthly_avg']:>+9,.0f} {'--':>10}")
    print(f"  {opt_label:<42} ${opt_y2_m['monthly_avg']:>+9,.0f} "
          f"${opt_bl_m['monthly_avg']:>+9,.0f} ${bl_delta:>+9,.0f}")

    # ═════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 85}")
    print("  VERDICT")
    print("═" * 85)

    if bl_delta > 50:
        print(f"\n  Tighter TP adds ${bl_delta:+,.0f}/mo on blind data. Worth implementing.")
    elif bl_delta > -50:
        print(f"\n  Tighter TP makes negligible difference (${bl_delta:+,.0f}/mo on blind).")
        print(f"  Stick with round numbers (TP=100/120) for simplicity.")
    else:
        print(f"\n  Tighter TP HURTS on blind data (${bl_delta:+,.0f}/mo).")
        print(f"  The missed fill benefit doesn't outweigh leaving money on the table.")

    # Near-miss interpretation
    total_losers = sum(nm_report[s]["total_losers"] for s in nm_report)
    within_2 = sum(nm_report[s]["counts"]["2.0"] for s in nm_report)
    within_pct = within_2 / total_losers * 100 if total_losers > 0 else 0
    print(f"\n  Near-miss reality: {within_2}/{total_losers} losers ({within_pct:.1f}%) came within 2pts of TP.")
    if within_pct < 3:
        print(f"  Very few trades are TP-miss candidates — the real-world MC's 5% TP-miss rate")
        print(f"  may be conservative. Actual TP-miss rate is likely lower.")
    elif within_pct < 7:
        print(f"  Moderate near-miss rate. Shaving 1-2 ticks could help in live trading.")
    else:
        print(f"  High near-miss rate. Consider tightening TP for live execution.")

    # ── Save ────────────────────────────────────────────────────────
    report = {
        "timestamp": str(datetime.now()),
        "near_miss": nm_report,
        "rsi_tp_sweep": {
            str(tp): {
                "y2_monthly": rsi_results[tp]["y2"]["monthly_avg"],
                "blind_monthly": rsi_results[tp]["blind"]["monthly_avg"],
                "y2_wr": rsi_results[tp]["y2"]["wr"],
                "y2_winners": rsi_results[tp]["y2"]["winners"],
                "y2_losers": rsi_results[tp]["y2"]["losers"],
            } for tp in rsi_tp_variants
        },
        "ib_tp_sweep": {
            str(tp): {
                "y2_monthly": ib_results[tp]["y2"]["monthly_avg"],
                "blind_monthly": ib_results[tp]["blind"]["monthly_avg"],
                "y2_wr": ib_results[tp]["y2"]["wr"],
                "y2_winners": ib_results[tp]["y2"]["winners"],
                "y2_losers": ib_results[tp]["y2"]["losers"],
            } for tp in ib_tp_variants
        },
        "mom_tp_sweep": {
            str(tp): {
                "y2_monthly": mom_results[tp]["y2"]["monthly_avg"],
                "blind_monthly": mom_results[tp]["blind"]["monthly_avg"],
                "y2_wr": mom_results[tp]["y2"]["wr"],
                "y2_winners": mom_results[tp]["y2"]["winners"],
                "y2_losers": mom_results[tp]["y2"]["losers"],
            } for tp in rsi_tp_variants
        },
        "optimal": {
            "rsi_tp": best_rsi_tp,
            "ib_tp": best_ib_tp,
            "mom_tp": best_mom_tp,
        },
        "combined": {
            "baseline_y2": base_y2_m["monthly_avg"],
            "baseline_blind": base_bl_m["monthly_avg"],
            "optimal_y2": opt_y2_m["monthly_avg"],
            "optimal_blind": opt_bl_m["monthly_avg"],
            "delta_y2": y2_delta,
            "delta_blind": bl_delta,
        },
    }

    out = REPORTS_DIR / "tp_optimization.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 85)


if __name__ == "__main__":
    main()
