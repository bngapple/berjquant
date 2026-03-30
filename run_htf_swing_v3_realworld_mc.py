#!/usr/bin/env python3
"""
REAL-WORLD MONTE CARLO — Stress testing live execution.

6 simultaneous degradation factors applied to hybrid v2 Y2 trades:
  1. Variable slippage (1-10 ticks, realistic distribution)
  2. Missed trades (5-15% random drop)
  3. TP degradation (15% worse fill, 5% TP-miss → SL loss)
  4. Regime simulation (vol scaling 0.4x-1.0x)
  5. Adverse loss clustering (bad weeks)
  6. Monthly fixed costs ($50/mo)

Usage:
    python3 run_htf_swing_v3_realworld_mc.py
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

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0
FLATTEN_TIME = 1645
CONTRACTS = 3
COST_PER_TRADE = rt_cost(CONTRACTS)  # $11.34
MONTHLY_FIXED_COST = 50.0  # data feed

TICK_VALUE = TICK_SIZE * POINT_VALUE  # $0.50 per tick for MNQ

HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}

# Slippage distribution: ticks → probability
SLIP_DIST = [(1, 0.10), (2, 0.40), (3, 0.30), (4, 0.15), (6, 0.04), (10, 0.01)]
SLIP_TICKS = np.array([s[0] for s in SLIP_DIST])
SLIP_PROBS = np.array([s[1] for s in SLIP_DIST])

N_SIMS = 10000


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


# ═════════════════════════════════════════════════════════════════════
# TRADE REPRESENTATION FOR MC
# ═════════════════════════════════════════════════════════════════════

def trades_to_records(trades):
    """Convert Trade objects to dicts with fields needed for MC."""
    records = []
    for t in trades:
        sl_pts = 10 if t.strategy in ("RSI", "IB") else 15
        tp_pts = 100 if t.strategy in ("RSI", "MOM") else 120
        records.append({
            "strategy": t.strategy,
            "direction": t.direction,
            "entry_px": t.entry_px,
            "exit_px": t.exit_px,
            "contracts": t.contracts,
            "net_pnl": t.net_pnl,
            "reason": t.reason,
            "bars_held": t.bars_held,
            "entry_date": str(t.entry_time)[:10],
            "entry_month": str(t.entry_time)[:7],
            "sl_pts": sl_pts,
            "tp_pts": tp_pts,
            # Gross P&L before costs
            "gross_pnl": (t.exit_px - t.entry_px) * t.direction * POINT_VALUE * t.contracts,
        })
    return records


# ═════════════════════════════════════════════════════════════════════
# DEGRADATION FACTORS
# ═════════════════════════════════════════════════════════════════════

def apply_variable_slippage(records, rng):
    """Factor 1: Replace fixed 2-tick slippage with random distribution."""
    result = []
    for r in records:
        # Draw random slippage for entry and exit separately
        entry_slip = rng.choice(SLIP_TICKS, p=SLIP_PROBS)
        exit_slip = rng.choice(SLIP_TICKS, p=SLIP_PROBS)
        total_slip_ticks = entry_slip + exit_slip

        # Original backtest used 2 ticks per side = 4 total
        # Additional adverse slippage beyond baseline
        baseline_total = 4  # 2 entry + 2 exit
        extra_slip_ticks = total_slip_ticks - baseline_total

        # Each extra adverse tick costs TICK_VALUE * contracts
        extra_cost = extra_slip_ticks * TICK_VALUE * r["contracts"]

        # For SL trades: if total slippage exceeds SL distance, stop blowthrough
        new_pnl = r["net_pnl"] - extra_cost

        # SL blowthrough: if exit slippage pushes beyond SL
        if r["reason"] == "stop_loss" and exit_slip > 2:
            # Extra exit slippage beyond the 2 ticks already in backtest
            blowthrough_ticks = exit_slip - 2
            blowthrough_cost = blowthrough_ticks * TICK_VALUE * r["contracts"]
            # Already accounted in extra_cost, just flag awareness
            pass

        result.append({**r, "net_pnl": new_pnl})
    return result


def apply_missed_trades(records, rng):
    """Factor 2: Randomly drop 5-15% of trades."""
    drop_rate = rng.uniform(0.05, 0.15)
    mask = rng.random(len(records)) > drop_rate
    return [r for r, keep in zip(records, mask) if keep]


def apply_tp_degradation(records, rng):
    """Factor 3: Degrade winner execution quality."""
    result = []
    for r in records:
        if r["reason"] == "take_profit":
            roll = rng.random()
            if roll < 0.80:
                # Normal fill at TP — no change
                result.append(r)
            elif roll < 0.95:
                # Fill 1-3 ticks worse
                worse_ticks = rng.randint(1, 4)
                penalty = worse_ticks * TICK_VALUE * r["contracts"]
                result.append({**r, "net_pnl": r["net_pnl"] - penalty})
            else:
                # TP kiss — price touches TP but doesn't fill, reverses to SL
                sl_pts = r["sl_pts"]
                sl_ticks = pts_to_ticks(sl_pts)
                # Loss = SL distance + exit slippage, minus entry slippage already in entry_px
                # Reconstruct: SL loss = -(sl_pts * POINT_VALUE * contracts) - cost
                sl_loss = -(sl_pts * POINT_VALUE * r["contracts"]) - COST_PER_TRADE
                result.append({**r, "net_pnl": sl_loss, "reason": "tp_miss_to_sl"})
        else:
            result.append(r)
    return result


def apply_regime(records, rng):
    """Factor 4: Scale P&L and trade count based on volatility regime."""
    regime = rng.random()
    if regime < 0.30:
        # High vol (current) — no change
        return records
    elif regime < 0.70:
        # Normal vol — scale P&L by 0.75x
        return [{**r, "net_pnl": r["net_pnl"] * 0.75} for r in records]
    elif regime < 0.90:
        # Low vol — scale by 0.40x AND drop 30% of trades (fewer signals)
        scaled = [{**r, "net_pnl": r["net_pnl"] * 0.40} for r in records]
        mask = rng.random(len(scaled)) > 0.30
        return [r for r, keep in zip(scaled, mask) if keep]
    else:
        # Mixed — first half as-is, second half at 0.50x
        mid = len(records) // 2
        first_half = records[:mid]
        second_half = [{**r, "net_pnl": r["net_pnl"] * 0.50} for r in records[mid:]]
        return first_half + second_half


def apply_loss_clustering(records, rng):
    """Factor 5: 20% chance of injecting a bad week."""
    if rng.random() > 0.20:
        return records

    result = list(records)
    n = len(result)
    if n < 30:
        return result

    # Pick random start, apply to 15-25 consecutive trades
    cluster_len = rng.randint(15, 26)
    start = rng.randint(0, max(1, n - cluster_len))

    for i in range(start, min(start + cluster_len, n)):
        if result[i]["net_pnl"] < 0:
            result[i] = {**result[i], "net_pnl": result[i]["net_pnl"] * 1.3}

    return result


def apply_monthly_costs(monthly_pnl_dict):
    """Factor 6: Subtract $50/month fixed costs."""
    return {m: v - MONTHLY_FIXED_COST for m, v in monthly_pnl_dict.items()}


# ═════════════════════════════════════════════════════════════════════
# SIMULATION
# ═════════════════════════════════════════════════════════════════════

def simulate_one(base_records, rng, factors="all"):
    """Run one simulation with all (or specified) degradation factors.

    factors: "all", or a set of factor names to apply.
    """
    apply_all = factors == "all"
    fset = factors if isinstance(factors, set) else set()

    records = list(base_records)

    # Shuffle trade order (bootstrap-like)
    rng.shuffle(records)

    # Factor 1: Variable slippage
    if apply_all or "slippage" in fset:
        records = apply_variable_slippage(records, rng)

    # Factor 2: Missed trades
    if apply_all or "missed" in fset:
        records = apply_missed_trades(records, rng)

    # Factor 3: TP degradation
    if apply_all or "tp_degrade" in fset:
        records = apply_tp_degradation(records, rng)

    # Factor 4: Regime
    if apply_all or "regime" in fset:
        records = apply_regime(records, rng)

    # Factor 5: Loss clustering
    if apply_all or "clustering" in fset:
        records = apply_loss_clustering(records, rng)

    # Compute monthly P&L from the surviving trades
    # Distribute trades evenly across 13 months
    n_months = 13
    trades_per_month = max(1, len(records) // n_months)
    monthly = {}
    for i in range(n_months):
        month_label = f"M{i+1:02d}"
        start = i * trades_per_month
        end = start + trades_per_month if i < n_months - 1 else len(records)
        month_trades = records[start:end]
        monthly[month_label] = sum(r["net_pnl"] for r in month_trades)

    # Factor 6: Monthly costs
    if apply_all or "costs" in fset:
        monthly = apply_monthly_costs(monthly)

    # Compute stats
    monthly_vals = list(monthly.values())
    total = sum(monthly_vals)
    monthly_avg = total / n_months

    # Max drawdown
    cum = 0; peak = 0; mdd = 0
    for v in monthly_vals:
        cum += v; peak = max(peak, cum); mdd = min(mdd, cum - peak)

    # Worst month
    worst_month = min(monthly_vals) if monthly_vals else 0

    # Consecutive losing months
    max_losing = 0; cur_losing = 0
    for v in monthly_vals:
        if v < 0:
            cur_losing += 1
            max_losing = max(max_losing, cur_losing)
        else:
            cur_losing = 0

    # Months negative
    months_neg = sum(1 for v in monthly_vals if v < 0)

    # LucidFlex eval simulation (daily P&L from trades)
    # Distribute trades across ~260 trading days
    n_days = 260
    trades_per_day = max(1, len(records) // n_days)
    daily_pnls = []
    for i in range(n_days):
        start = i * trades_per_day
        end = start + trades_per_day if i < n_days - 1 else len(records)
        day_trades = records[start:end]
        dp = sum(r["net_pnl"] for r in day_trades)
        daily_pnls.append(dp)

    # Eval pass check
    cum_eval = 0; peak_eval = 0; passed = False; blown = False; days_to_pass = 0
    for i, dp in enumerate(daily_pnls):
        if dp < DAILY_LIMIT:
            dp = DAILY_LIMIT
        cum_eval += dp
        peak_eval = max(peak_eval, cum_eval)
        if cum_eval - peak_eval <= MLL:
            blown = True
            break
        if not passed and cum_eval >= EVAL_TARGET:
            passed = True
            days_to_pass = i + 1

    return {
        "total": total,
        "monthly_avg": monthly_avg,
        "max_dd": mdd,
        "worst_month": worst_month,
        "max_losing_streak": max_losing,
        "months_neg": months_neg,
        "eval_pass": passed and not blown,
        "eval_blown": blown,
        "days_to_pass": days_to_pass if passed and not blown else 0,
        "n_trades": len(records),
    }


def run_mc_batch(base_records, n_sims, factors="all", label="", return_raw=False):
    """Run n_sims simulations and return aggregated results."""
    results = []
    for sim in range(n_sims):
        rng = np.random.RandomState(sim + 1000)
        results.append(simulate_one(base_records, rng, factors))

    totals = np.array([r["total"] for r in results])
    avgs = np.array([r["monthly_avg"] for r in results])
    dds = np.array([r["max_dd"] for r in results])
    worst_months = np.array([r["worst_month"] for r in results])
    passes = sum(1 for r in results if r["eval_pass"])
    blowups = sum(1 for r in results if r["eval_blown"])
    days = np.array([r["days_to_pass"] for r in results if r["days_to_pass"] > 0])
    neg3mo = sum(1 for r in results if r["max_losing_streak"] >= 3)
    dd_4500 = sum(1 for r in results if r["max_dd"] <= -4500)
    under_3k = sum(1 for r in results if r["monthly_avg"] < 3000)
    under_0 = sum(1 for r in results if r["total"] < 0)

    out = {
        "n_sims": n_sims,
        "label": label,
        "monthly_p5": float(np.percentile(avgs, 5)),
        "monthly_p25": float(np.percentile(avgs, 25)),
        "monthly_p50": float(np.median(avgs)),
        "monthly_p75": float(np.percentile(avgs, 75)),
        "monthly_p95": float(np.percentile(avgs, 95)),
        "monthly_mean": float(np.mean(avgs)),
        "total_p50": float(np.median(totals)),
        "dd_p50": float(np.median(dds)),
        "dd_p95": float(np.percentile(dds, 5)),  # 5th percentile of DD = worst
        "worst_month_p50": float(np.median(worst_months)),
        "pass_rate": passes / n_sims,
        "blowup_rate": blowups / n_sims,
        "med_days": int(np.median(days)) if len(days) > 0 else 0,
        "p95_days": int(np.percentile(days, 95)) if len(days) > 0 else 0,
        "prob_3_losing_months": neg3mo / n_sims,
        "prob_dd_4500": dd_4500 / n_sims,
        "prob_under_3k": under_3k / n_sims,
        "prob_under_0": under_0 / n_sims,
    }
    if return_raw:
        out["_raw_monthly_avgs"] = avgs.tolist()
    return out


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 75)
    print("  REAL-WORLD MONTE CARLO — 10,000 Simulations")
    print("═" * 75)

    # ── Load data and run hybrid v2 ─────────────────────────────────
    print("\n  Loading data and running hybrid v2 backtest...")
    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
    Y1_END = datetime(2025, 3, 1, tzinfo=_ET)
    df_main = main_data["15m"]
    yr2 = df_main.filter(pl.col("timestamp") >= Y1_END)

    all_y2, strats_y2 = run_system(yr2, HYBRID_V2)
    base_records = trades_to_records(all_y2)

    # Backtest baseline stats
    bt_monthly = defaultdict(float)
    for t in all_y2:
        bt_monthly[str(t.entry_time)[:7]] += t.net_pnl
    bt_avg = sum(bt_monthly.values()) / len(bt_monthly)
    bt_total = sum(bt_monthly.values())

    print(f"  Y2 trades: {len(all_y2)}")
    print(f"  Backtest monthly avg: ${bt_avg:,.0f}")
    print(f"  Backtest total: ${bt_total:,.0f}")

    # ═════════════════════════════════════════════════════════════════
    # FULL SIMULATION (all factors)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print(f"  RUNNING {N_SIMS:,} SIMULATIONS (all 6 factors)...")
    print("━" * 75)

    full = run_mc_batch(base_records, N_SIMS, "all", "Full real-world", return_raw=True)
    gc.collect()

    # ═════════════════════════════════════════════════════════════════
    # FACTOR ATTRIBUTION (each factor in isolation)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 75}")
    print("  FACTOR ATTRIBUTION (each factor in isolation, 5000 sims)")
    print("━" * 75)

    factor_names = [
        ("slippage", "Variable slippage"),
        ("missed", "Missed trades"),
        ("tp_degrade", "TP degradation"),
        ("regime", "Regime simulation"),
        ("clustering", "Loss clustering"),
        ("costs", "Monthly costs"),
    ]

    factor_results = {}
    # Baseline (no factors, just shuffle)
    baseline = run_mc_batch(base_records, 5000, set(), "Baseline (shuffle only)")
    factor_results["baseline"] = baseline

    for factor_key, factor_label in factor_names:
        print(f"    {factor_label}...", end=" ", flush=True)
        r = run_mc_batch(base_records, 5000, {factor_key}, factor_label)
        factor_results[factor_key] = r
        print(f"${r['monthly_p50']:,.0f}/mo")
        gc.collect()

    # ═════════════════════════════════════════════════════════════════
    # OUTPUT
    # ═════════════════════════════════════════════════════════════════

    print(f"\n{'═' * 75}")
    print(f"  REAL-WORLD MONTE CARLO — {N_SIMS:,} simulations")
    print("═" * 75)

    print(f"\n  Monthly P&L distribution:")
    print(f"    P5  (worst 5%):      ${full['monthly_p5']:>+10,.0f}")
    print(f"    P25 (conservative):  ${full['monthly_p25']:>+10,.0f}")
    print(f"    P50 (median):        ${full['monthly_p50']:>+10,.0f}  <-- realistic target")
    print(f"    P75 (optimistic):    ${full['monthly_p75']:>+10,.0f}")
    print(f"    P95 (best case):     ${full['monthly_p95']:>+10,.0f}")

    print(f"\n    Backtest Y2 monthly: ${bt_avg:>+10,.0f}  (reference)")
    deg = (full['monthly_p50'] - bt_avg) / bt_avg * 100
    print(f"    Median degradation:  {deg:>+9.0f}%")

    print(f"\n  Max drawdown distribution:")
    print(f"    P50 (median):  ${full['dd_p50']:>+10,.0f}")
    print(f"    P95 (worst):   ${full['dd_p95']:>+10,.0f}")

    print(f"\n  LucidFlex eval:")
    print(f"    Pass rate:     {full['pass_rate']:>8.1%}")
    print(f"    Median days:   {full['med_days']:>8}")
    print(f"    P95 days:      {full['p95_days']:>8}")
    print(f"    Blowup rate:   {full['blowup_rate']:>8.1%}")

    print(f"\n  Risk of ruin:")
    print(f"    Prob 3+ consecutive losing months: {full['prob_3_losing_months']:>6.1%}")
    print(f"    Prob -$4,500+ drawdown:            {full['prob_dd_4500']:>6.1%}")
    print(f"    Prob making < $3,000/mo average:    {full['prob_under_3k']:>6.1%}")
    print(f"    Prob making < $0 over 13 months:    {full['prob_under_0']:>6.1%}")

    # ── Factor attribution ──────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  FACTOR ATTRIBUTION (median monthly P&L)")
    print("━" * 75)

    bl_median = factor_results["baseline"]["monthly_p50"]
    print(f"\n    {'Factor':<28} {'Median $/mo':>12} {'Impact':>10}")
    print(f"    {'─'*28} {'─'*12} {'─'*10}")
    print(f"    {'Backtest baseline':<28} ${bt_avg:>+11,.0f}")
    print(f"    {'Shuffle only (no factors)':<28} ${bl_median:>+11,.0f} ${bl_median - bt_avg:>+9,.0f}")

    prev = bl_median
    for factor_key, factor_label in factor_names:
        med = factor_results[factor_key]["monthly_p50"]
        impact = med - bl_median
        print(f"    {'+ ' + factor_label:<28} ${med:>+11,.0f} ${impact:>+9,.0f}")

    print(f"    {'─'*28} {'─'*12} {'─'*10}")
    print(f"    {'ALL FACTORS COMBINED':<28} ${full['monthly_p50']:>+11,.0f} ${full['monthly_p50'] - bl_median:>+9,.0f}")

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  COMPARISON TABLE")
    print("━" * 75)

    print(f"\n  {'':>20} {'Backtest':>10} {'Real P50':>10} {'Real P25':>10} {'Worst P5':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Monthly P&L':<20} ${bt_avg:>+9,.0f} ${full['monthly_p50']:>+9,.0f} "
          f"${full['monthly_p25']:>+9,.0f} ${full['monthly_p5']:>+9,.0f}")
    print(f"  {'Annual P&L':<20} ${bt_total:>+9,.0f} ${full['total_p50']:>+9,.0f} "
          f"${full['monthly_p25']*13:>+9,.0f} ${full['monthly_p5']*13:>+9,.0f}")
    print(f"  {'LucidFlex pass':<20} {'100%':>10} {full['pass_rate']:>9.0%} {'--':>10} {'--':>10}")

    # Backtest max DD
    bt_daily = defaultdict(float)
    for t in all_y2:
        bt_daily[str(t.entry_time)[:10]] += t.net_pnl
    cum = 0; peak = 0; bt_mdd = 0
    for d in sorted(bt_daily.keys()):
        cum += bt_daily[d]; peak = max(peak, cum); bt_mdd = min(bt_mdd, cum - peak)

    print(f"  {'Max drawdown':<20} ${bt_mdd:>+9,.0f} ${full['dd_p50']:>+9,.0f} "
          f"${'--':>8} ${full['dd_p95']:>+9,.0f}")

    # ── Survival analysis ───────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  SURVIVAL ANALYSIS")
    print("━" * 75)

    # Reuse the full simulation results (already ran N_SIMS)
    targets = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    monthly_avgs_all = np.array(full["_raw_monthly_avgs"])

    print(f"\n  Probability of achieving monthly average:")
    for target in targets:
        pct = np.mean(monthly_avgs_all >= target) * 100
        bar = "█" * int(pct / 2)
        print(f"    ≥ ${target:>6,}/mo:  {pct:>5.1f}%  {bar}")

    # ── Summary verdict ─────────────────────────────────────────────
    print(f"\n{'═' * 75}")
    print("  VERDICT")
    print("═" * 75)

    realistic = full["monthly_p50"]
    conservative = full["monthly_p25"]
    worst = full["monthly_p5"]

    print(f"\n  Backtest says:  ${bt_avg:>+,.0f}/month")
    print(f"  Reality (P50):  ${realistic:>+,.0f}/month")
    print(f"  Conservative:   ${conservative:>+,.0f}/month  (P25 — 75% chance of beating this)")
    print(f"  Worst case:     ${worst:>+,.0f}/month  (P5 — 95% chance of beating this)")

    haircut = (1 - realistic / bt_avg) * 100 if bt_avg > 0 else 0
    print(f"\n  Realistic haircut: {haircut:.0f}% off backtest")
    print(f"  Eval pass rate: {full['pass_rate']:.0%}")

    if full['pass_rate'] >= 0.90 and realistic > 3000:
        print(f"\n  STRONG GO — {full['pass_rate']:.0%} eval pass, ${realistic:,.0f}/mo realistic median")
    elif full['pass_rate'] >= 0.70 and realistic > 2000:
        print(f"\n  MODERATE GO — {full['pass_rate']:.0%} eval pass, but expect ${realistic:,.0f}/mo not ${bt_avg:,.0f}")
    elif full['pass_rate'] >= 0.50:
        print(f"\n  MARGINAL — {full['pass_rate']:.0%} eval pass. High variance, manage expectations.")
    else:
        print(f"\n  DO NOT DEPLOY — {full['pass_rate']:.0%} eval pass rate too low for live capital")

    # ── Save ────────────────────────────────────────────────────────
    report = {
        "timestamp": str(datetime.now()),
        "params": {k: dict(v) for k, v in HYBRID_V2.items()},
        "n_sims": N_SIMS,
        "backtest_monthly_avg": bt_avg,
        "backtest_total": bt_total,
        "backtest_max_dd": bt_mdd,
        "full_results": {k: v for k, v in full.items() if not k.startswith("_")},
        "factor_attribution": {k: v for k, v in factor_results.items()},
        "survival_analysis": {
            f"gte_{t}": float(np.mean(monthly_avgs_all >= t))
            for t in targets
        },
        "degradation_factors": {
            "slippage_dist": dict(zip([str(s) for s in SLIP_TICKS.tolist()], SLIP_PROBS.tolist())),
            "missed_rate": "5-15%",
            "tp_degrade": "80% clean, 15% worse fill, 5% TP miss",
            "regime_dist": "30% high vol, 40% normal (0.75x), 20% low (0.40x), 10% mixed",
            "clustering": "20% chance of bad week (losses 1.3x)",
            "monthly_cost": MONTHLY_FIXED_COST,
        },
    }

    out = REPORTS_DIR / "realworld_mc.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 75)


if __name__ == "__main__":
    main()
