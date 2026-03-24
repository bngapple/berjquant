#!/usr/bin/env python3
"""
Monte Carlo simulation for Always-On v2 × Topstep 150K.

Re-generates Y2 trades from the v2 system, then shuffles trade order 5,000 times
to estimate eval pass rate, blow-up risk, and drawdown distribution.

Two scenarios: baseline (100% backtest P&L) and conservative (70%).

Usage:
    python3 run_always_on_mc.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from run_always_on import (
    load_data, split_years, add_initial_balance,
    layer1_signals, layer2c_signals, layer3_signals,
    run_backtest, TradeRecord,
)
from run_always_on_v2 import momentum_continuation_signals, run_layer_at_size

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

# ── Topstep 150K rules ──────────────────────────────────────────────
DAILY_LOSS_LIMIT = -3000.0
MLL = -4500.0               # EOD trailing drawdown
PROFIT_TARGET = 9000.0

# ── v2 winning params ───────────────────────────────────────────────
V2_PARAMS = {
    "L1": {"z_threshold": 1.35, "lookback": 50, "stop_ticks": 20, "max_hold": 35},
    "L2C": {"ema_fast": 8, "ema_slow": 13, "trail_ticks": 6, "max_hold": 25},
    "L3": {"range_mult": 0.8, "ema_pullback": 5, "trail_ticks": 9, "max_hold": 25},
    "MomCont": {"z": 1.5, "atr_m": 1.0, "trail": 8, "hold": 15},
}
V2_SIZING = {"L1": 6, "L2C": 10, "L3": 10}


def regenerate_y2_trades(yr2: pl.DataFrame) -> list[TradeRecord]:
    """Re-run all v2 layers on Year 2 to get trade-level data."""
    all_trades = []

    # L1 refined
    p = V2_PARAMS["L1"]
    sigs, exits = layer1_signals(yr2, z_threshold=p["z_threshold"], lookback=p["lookback"])
    trades = run_backtest(yr2, sigs, exits, stop_ticks=p["stop_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L1"],
                          layer_name="L1_VWAP_MR")
    all_trades.extend(trades)
    print(f"  L1: {len(trades)} trades, ${sum(t.net_pnl for t in trades):,.0f}")

    # L2C refined
    p = V2_PARAMS["L2C"]
    sigs, exits = layer2c_signals(yr2, ema_fast=p["ema_fast"], ema_slow=p["ema_slow"])
    trades = run_backtest(yr2, sigs, exits, stop_ticks=p["trail_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L2C"],
                          trailing=True, trailing_ticks=p["trail_ticks"],
                          layer_name="L2C_Afternoon")
    all_trades.extend(trades)
    print(f"  L2C: {len(trades)} trades, ${sum(t.net_pnl for t in trades):,.0f}")

    # L3 refined
    p = V2_PARAMS["L3"]
    sigs, exits = layer3_signals(yr2, range_mult=p["range_mult"], ema_pullback=p["ema_pullback"])
    trades = run_backtest(yr2, sigs, exits, stop_ticks=p["trail_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L3"],
                          trailing=True, trailing_ticks=p["trail_ticks"],
                          layer_name="L3_VolMom")
    all_trades.extend(trades)
    print(f"  L3: {len(trades)} trades, ${sum(t.net_pnl for t in trades):,.0f}")

    # Momentum continuation
    p = V2_PARAMS["MomCont"]
    sigs, exits = momentum_continuation_signals(yr2, z_threshold=p["z"], atr_mult=p["atr_m"])
    trades = run_backtest(yr2, sigs, exits, stop_ticks=p["trail"], target_ticks=None,
                          max_hold_bars=p["hold"], contracts=V2_SIZING["L1"],
                          trailing=True, trailing_ticks=p["trail"],
                          layer_name="L1_MomCont")
    all_trades.extend(trades)
    print(f"  MomCont: {len(trades)} trades, ${sum(t.net_pnl for t in trades):,.0f}")

    print(f"  Total Y2 trades: {len(all_trades)}, P&L: ${sum(t.net_pnl for t in all_trades):,.0f}")
    return all_trades


def run_mc(trades: list[TradeRecord], n_sims: int = 5000,
           pnl_mult: float = 1.0, label: str = "Baseline") -> dict:
    """Run Monte Carlo with trade-order shuffling.

    Groups trades by original date. For each sim, shuffles the daily buckets
    (preserving intra-day trade structure), then replays sequentially.
    """
    # Group trades by day
    daily_buckets: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        daily_buckets[d].append(t.net_pnl * pnl_mult)

    days = list(daily_buckets.keys())
    day_pnls = [daily_buckets[d] for d in days]
    n_days = len(days)

    if n_days == 0:
        return _empty_mc(label)

    eval_passed = 0
    blown_up = 0
    max_drawdowns = []
    final_pnls = []
    days_to_pass = []
    monthly_pnls_all = []

    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(n_days)

        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        day_count = 0
        passed = False
        blown = False
        monthly_bucket = 0.0
        days_in_month = 0

        for idx in order:
            day_trades = day_pnls[idx]
            day_pnl = 0.0

            for trade_pnl in day_trades:
                day_pnl += trade_pnl
                # Check daily limit
                if day_pnl <= DAILY_LOSS_LIMIT:
                    break  # Stop trading this day

            cum_pnl += day_pnl
            day_count += 1

            # EOD trailing drawdown
            peak = max(peak, cum_pnl)
            dd = cum_pnl - peak
            max_dd = min(max_dd, dd)

            # MLL check
            if dd <= MLL:
                blown = True
                break

            # Eval pass check
            if not passed and cum_pnl >= PROFIT_TARGET:
                passed = True
                days_to_pass.append(day_count)

            # Monthly tracking (~21 trading days)
            monthly_bucket += day_pnl
            days_in_month += 1
            if days_in_month >= 21:
                monthly_pnls_all.append(monthly_bucket)
                monthly_bucket = 0.0
                days_in_month = 0

        if days_in_month > 0:
            monthly_pnls_all.append(monthly_bucket)

        if blown:
            blown_up += 1
        if passed and not blown:
            eval_passed += 1

        max_drawdowns.append(max_dd)
        final_pnls.append(cum_pnl)

    max_dd_arr = np.array(max_drawdowns)
    final_arr = np.array(final_pnls)
    monthly_arr = np.array(monthly_pnls_all) if monthly_pnls_all else np.array([0.0])
    days_arr = np.array(days_to_pass) if days_to_pass else np.array([0])

    # All months positive in any single sim
    months_per_sim = max(n_days, 1) // 21
    all_months_positive = 0
    if months_per_sim > 0 and len(monthly_arr) >= months_per_sim:
        for s in range(min(n_sims, len(monthly_arr) // months_per_sim)):
            start = s * months_per_sim
            end = start + months_per_sim
            if end <= len(monthly_arr) and all(monthly_arr[start:end] > 0):
                all_months_positive += 1
        prob_all_positive = all_months_positive / n_sims
    else:
        prob_all_positive = 0.0

    return {
        "label": label,
        "n_sims": n_sims,
        "pnl_mult": pnl_mult,
        "eval_pass_rate": eval_passed / n_sims,
        "blowup_rate": blown_up / n_sims,
        "median_days_to_pass": int(np.median(days_arr)) if len(days_arr) > 0 else None,
        "p95_days_to_pass": int(np.percentile(days_arr, 95)) if len(days_arr) > 0 else None,
        "median_max_dd": float(np.median(max_dd_arr)),
        "p95_max_dd": float(np.percentile(max_dd_arr, 95)),
        "median_final_pnl": float(np.median(final_arr)),
        "p5_final_pnl": float(np.percentile(final_arr, 5)),
        "p5_worst_month": float(np.percentile(monthly_arr, 5)) if len(monthly_arr) > 0 else 0,
        "prob_monthly_7k": float(np.mean(monthly_arr >= 7000)) if len(monthly_arr) > 0 else 0,
        "prob_all_months_positive": prob_all_positive,
    }


def _empty_mc(label):
    return {
        "label": label, "n_sims": 0, "pnl_mult": 1.0,
        "eval_pass_rate": 0, "blowup_rate": 1.0,
        "median_days_to_pass": None, "p95_days_to_pass": None,
        "median_max_dd": 0, "p95_max_dd": 0,
        "median_final_pnl": 0, "p5_final_pnl": 0,
        "p5_worst_month": 0, "prob_monthly_7k": 0,
        "prob_all_months_positive": 0,
    }


def verdict(mc: dict) -> str:
    if mc["eval_pass_rate"] > 0.80 and mc["blowup_rate"] < 0.10:
        return "PASS"
    elif mc["eval_pass_rate"] >= 0.50 and mc["blowup_rate"] <= 0.25:
        return "MARGINAL"
    else:
        return "FAIL"


def print_mc(mc: dict):
    v = verdict(mc)
    print(f"\n  {mc['label'].upper()} ({mc['pnl_mult']:.0%} of backtest)")
    print(f"    Simulations:            {mc['n_sims']:,}")
    print(f"    Eval pass rate:         {mc['eval_pass_rate']:.1%} (hit $9K before -$4.5K MLL)")
    print(f"    Account blow-up rate:   {mc['blowup_rate']:.1%}")
    print(f"    Median days to pass:    {mc['median_days_to_pass'] or 'N/A'}")
    print(f"    P95 days to pass:       {mc['p95_days_to_pass'] or 'N/A'}")
    print(f"    Median max drawdown:    ${mc['median_max_dd']:,.0f}")
    print(f"    P95 max drawdown:       ${mc['p95_max_dd']:,.0f}")
    print(f"    Median final P&L:       ${mc['median_final_pnl']:,.0f}")
    print(f"    P5 final P&L:           ${mc['p5_final_pnl']:,.0f}")
    print(f"    P5 worst month:         ${mc['p5_worst_month']:,.0f}")
    print(f"    P(monthly > $7K):       {mc['prob_monthly_7k']:.1%}")
    print(f"    P(all months positive): {mc['prob_all_months_positive']:.1%}")
    print(f"    VERDICT:                {v}")


def main():
    t0 = _time.time()
    print("═" * 60)
    print("  MONTE CARLO — Always-On v2 × Topstep 150K")
    print("═" * 60)

    # Load data and regenerate Y2 trades
    df = load_data()
    df = add_initial_balance(df)
    _, yr2 = split_years(df)

    print("\n  Regenerating Year 2 trades ...")
    y2_trades = regenerate_y2_trades(yr2)
    gc.collect()

    # Baseline MC
    print(f"\n  Running baseline Monte Carlo (5,000 sims) ...")
    mc_baseline = run_mc(y2_trades, n_sims=5000, pnl_mult=1.0, label="Baseline")
    print_mc(mc_baseline)
    gc.collect()

    # Conservative MC (70% of backtest)
    print(f"\n  Running conservative Monte Carlo (5,000 sims, 70% P&L) ...")
    mc_conservative = run_mc(y2_trades, n_sims=5000, pnl_mult=0.70, label="Conservative (70%)")
    print_mc(mc_conservative)
    gc.collect()

    # Final verdict
    print(f"\n  {'─' * 56}")
    print(f"  VERDICT:")
    print(f"    Baseline:     {verdict(mc_baseline)} for Topstep 150K")
    print(f"    Conservative: {verdict(mc_conservative)} for Topstep 150K")
    print(f"  {'─' * 56}")

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "y2_total_trades": len(y2_trades),
        "y2_total_pnl": sum(t.net_pnl for t in y2_trades),
        "baseline": mc_baseline,
        "conservative": mc_conservative,
        "baseline_verdict": verdict(mc_baseline),
        "conservative_verdict": verdict(mc_conservative),
    }
    out = REPORTS_DIR / "always_on_mc_v1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
