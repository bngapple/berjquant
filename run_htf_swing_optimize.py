#!/usr/bin/env python3
"""
HTF Swing v3 — Parameter Optimization (Train / Test / Blind).

Golden rule: optimize on TRAIN only, validate on TEST, blind is untouched.
Train: 2022-01 to 2024-02 (26 months)
Test:  2024-03 to 2025-02 (12 months)
Blind: 2025-03 to 2026-03 (13 months)

Usage:
    python3 run_htf_swing_optimize.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from itertools import product
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
FLATTEN_TIME = 1645    # LucidFlex session
CONTRACTS = 3          # aggressive

# ── Current v3 params (baseline) ────────────────────────────────────
CURRENT = {
    "RSI": {"period": 7, "ob": 70, "os": 30, "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sl": 80, "tp": 480, "hold": 10, "ib_filter": True},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl": 60, "tp": 400, "hold": 5},
}

# ── Date boundaries ─────────────────────────────────────────────────
from zoneinfo import ZoneInfo
_ET = ZoneInfo("US/Eastern")
TRAIN_END = datetime(2024, 3, 1, tzinfo=_ET)
TEST_END  = datetime(2025, 3, 1, tzinfo=_ET)


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def pts_to_ticks(pts):
    """Convert points to ticks (1 point = 4 ticks for MNQ)."""
    return int(pts / TICK_SIZE)


def daily_pnl(trades):
    d = defaultdict(float)
    for t in trades:
        d[str(t.entry_time)[:10]] += t.net_pnl
    return dict(d)


def monthly_pnl(trades):
    m = defaultdict(float)
    for t in trades:
        m[str(t.entry_time)[:7]] += t.net_pnl
    return dict(m)


def calc_sharpe(trades):
    if len(trades) < 10:
        return -999.0
    dp = daily_pnl(trades)
    vals = list(dp.values())
    if len(vals) < 5 or np.std(vals) == 0:
        return -999.0
    return float(np.mean(vals) / np.std(vals) * np.sqrt(252))


def calc_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "sharpe": -999, "monthly_avg": 0,
                "worst_day": 0, "max_dd": 0, "months_pos": 0, "n_months": 0}
    pnls = [t.net_pnl for t in trades]
    w = sum(1 for p in pnls if p > 0)
    dp = daily_pnl(trades)
    mp = monthly_pnl(trades)
    vals = list(dp.values())
    sharpe = float(np.mean(vals) / np.std(vals) * np.sqrt(252)) if len(vals) > 1 and np.std(vals) > 0 else 0
    cum = 0.0; peak = 0.0; mdd = 0.0
    for d in sorted(dp.keys()):
        cum += dp[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    nm = max(len(mp), 1)
    total = sum(pnls)
    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "sharpe": sharpe, "monthly_avg": total / nm,
        "worst_day": min(vals) if vals else 0,
        "worst_month": min(mp.values()) if mp else 0,
        "best_month": max(mp.values()) if mp else 0,
        "max_dd": mdd,
        "months_pos": sum(1 for v in mp.values() if v > 0),
        "n_months": nm,
        "monthly": dict(mp),
    }


def run_mc(trades, n_sims=5000):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass_rate": 0, "blowup": 1, "med_days": 0, "p95_days": 0}
    passed = 0; blown = 0; dtp = []
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; p = False; ok = True; dc = 0
        for idx in order:
            dp = sum(days[idx])
            if dp < DAILY_LIMIT:
                dp = DAILY_LIMIT
            cum += dp; dc += 1; peak = max(peak, cum)
            if cum - peak <= MLL:
                ok = False; break
            if not p and cum >= EVAL_TARGET:
                p = True; dtp.append(dc)
        if not ok:
            blown += 1
        if p and ok:
            passed += 1
    da = np.array(dtp) if dtp else np.array([0])
    return {
        "pass_rate": passed / n_sims,
        "blowup": blown / n_sims,
        "med_days": int(np.median(da)),
        "p95_days": int(np.percentile(da, 95)) if len(da) > 0 else 0,
    }


# ═════════════════════════════════════════════════════════════════════
# GRID SWEEP FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def sweep_rsi(df, arrays, grid):
    """Sweep RSI params. Returns list of (params, trades)."""
    results = []
    # Cache signals per (period, ob, os) combo to avoid recomputing
    sig_cache = {}
    for period, ob, os_, sl_pts, tp_pts, hold in grid:
        key = (period, ob, os_)
        if key not in sig_cache:
            sig_cache[key] = sig_rsi_extreme(df, period, ob, os_)
        sigs = sig_cache[key]
        sl_t = pts_to_ticks(sl_pts)
        tp_t = pts_to_ticks(tp_pts)
        o, h, l, c, ts, hm = arrays
        trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, hold,
                          CONTRACTS, "RSI", FLATTEN_TIME)
        params = {"period": period, "ob": ob, "os": os_,
                  "sl": sl_pts, "tp": tp_pts, "hold": hold}
        results.append((params, trades))
    return results


def sweep_ib(df, arrays, grid):
    """Sweep IB params. Returns list of (params, trades)."""
    results = []
    sig_cache = {}
    for sl_pts, tp_pts, hold, ib_filter in grid:
        if ib_filter not in sig_cache:
            sig_cache[ib_filter] = sig_ib_breakout(df, ib_filter)[0]
        sigs = sig_cache[ib_filter]
        sl_t = pts_to_ticks(sl_pts)
        tp_t = pts_to_ticks(tp_pts)
        o, h, l, c, ts, hm = arrays
        trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, hold,
                          CONTRACTS, "IB", FLATTEN_TIME)
        params = {"sl": sl_pts, "tp": tp_pts, "hold": hold, "ib_filter": ib_filter}
        results.append((params, trades))
    return results


def sweep_mom(df, arrays, grid):
    """Sweep MOM params. Returns list of (params, trades)."""
    results = []
    sig_cache = {}
    for atr_m, vol_m, sl_pts, tp_pts, hold in grid:
        key = (atr_m, vol_m)
        if key not in sig_cache:
            sig_cache[key] = sig_momentum_bar(df, atr_m, vol_m)[0]
        sigs = sig_cache[key]
        sl_t = pts_to_ticks(sl_pts)
        tp_t = pts_to_ticks(tp_pts)
        o, h, l, c, ts, hm = arrays
        trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, hold,
                          CONTRACTS, "MOM", FLATTEN_TIME)
        params = {"atr_mult": atr_m, "vol_mult": vol_m,
                  "sl": sl_pts, "tp": tp_pts, "hold": hold}
        results.append((params, trades))
    return results


# ═════════════════════════════════════════════════════════════════════
# OPTIMIZATION PIPELINE
# ═════════════════════════════════════════════════════════════════════

def optimize_strategy(name, sweep_fn, grid, train_df, test_df, train_arrays, test_arrays,
                      min_trades=100, top_n=20):
    """
    Step 1: Sweep on train. Step 2: Validate top_n on test.
    Returns sorted list of surviving param sets.
    """
    print(f"\n  Sweeping {name}: {len(grid):,} combos on train data ...", flush=True)
    t0 = _time.time()

    # Step 1: Run all combos on train
    train_results = sweep_fn(train_df, train_arrays, grid)
    print(f"    Done in {_time.time() - t0:.0f}s")

    # Rank by Sharpe, filter by min trades + positive P&L
    scored = []
    for params, trades in train_results:
        n = len(trades)
        if n < min_trades:
            continue
        pnl = sum(t.net_pnl for t in trades)
        if pnl <= 0:
            continue
        sharpe = calc_sharpe(trades)
        if sharpe <= 0:
            continue
        scored.append({"params": params, "train_sharpe": sharpe, "train_pnl": pnl,
                       "train_n": n, "train_trades": trades})

    scored.sort(key=lambda x: x["train_sharpe"], reverse=True)
    top = scored[:top_n]
    print(f"    {len(scored)} profitable combos, top {len(top)} by Sharpe")

    if not top:
        return []

    # Step 2: Validate on test
    print(f"    Validating top {len(top)} on test data ...", flush=True)
    validated = []
    for entry in top:
        p = entry["params"]
        # Regenerate signals for test data and run backtest
        if name == "RSI":
            sigs = sig_rsi_extreme(test_df, p["period"], p["ob"], p["os"])
            sl_t, tp_t = pts_to_ticks(p["sl"]), pts_to_ticks(p["tp"])
            o, h, l, c, ts, hm = test_arrays
            trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, p["hold"],
                              CONTRACTS, "RSI", FLATTEN_TIME)
        elif name == "IB":
            sigs = sig_ib_breakout(test_df, p["ib_filter"])[0]
            sl_t, tp_t = pts_to_ticks(p["sl"]), pts_to_ticks(p["tp"])
            o, h, l, c, ts, hm = test_arrays
            trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, p["hold"],
                              CONTRACTS, "IB", FLATTEN_TIME)
        elif name == "MOM":
            sigs = sig_momentum_bar(test_df, p["atr_mult"], p["vol_mult"])[0]
            sl_t, tp_t = pts_to_ticks(p["sl"]), pts_to_ticks(p["tp"])
            o, h, l, c, ts, hm = test_arrays
            trades = backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, p["hold"],
                              CONTRACTS, "MOM", FLATTEN_TIME)

        test_pnl = sum(t.net_pnl for t in trades)
        test_sharpe = calc_sharpe(trades)
        test_n = len(trades)

        # Filters: profitable on test, Sharpe stability > 50%
        if test_pnl <= 0:
            continue
        if test_sharpe <= 0:
            continue
        if entry["train_sharpe"] > 0 and test_sharpe / entry["train_sharpe"] < 0.5:
            continue

        entry["test_sharpe"] = test_sharpe
        entry["test_pnl"] = test_pnl
        entry["test_n"] = test_n
        entry["test_trades"] = trades
        entry["stability"] = test_sharpe / entry["train_sharpe"] if entry["train_sharpe"] > 0 else 0
        validated.append(entry)

    # Rank by TEST Sharpe
    validated.sort(key=lambda x: x["test_sharpe"], reverse=True)
    print(f"    {len(validated)} survived validation (stability > 50%)")
    return validated


def run_on_blind(name, params, blind_df, blind_arrays):
    """Run a single param set on blind data."""
    if name == "RSI":
        sigs = sig_rsi_extreme(blind_df, params["period"], params["ob"], params["os"])
        sl_t, tp_t = pts_to_ticks(params["sl"]), pts_to_ticks(params["tp"])
        o, h, l, c, ts, hm = blind_arrays
        return backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, params["hold"],
                        CONTRACTS, "RSI", FLATTEN_TIME)
    elif name == "IB":
        sigs = sig_ib_breakout(blind_df, params["ib_filter"])[0]
        sl_t, tp_t = pts_to_ticks(params["sl"]), pts_to_ticks(params["tp"])
        o, h, l, c, ts, hm = blind_arrays
        return backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, params["hold"],
                        CONTRACTS, "IB", FLATTEN_TIME)
    elif name == "MOM":
        sigs = sig_momentum_bar(blind_df, params["atr_mult"], params["vol_mult"])[0]
        sl_t, tp_t = pts_to_ticks(params["sl"]), pts_to_ticks(params["tp"])
        o, h, l, c, ts, hm = blind_arrays
        return backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, params["hold"],
                        CONTRACTS, "MOM", FLATTEN_TIME)


def run_current_on_period(df, arrays):
    """Run current v3 params on a dataset. Returns combined trades."""
    o, h, l, c, ts, hm = arrays
    trades = []
    # RSI
    sigs = sig_rsi_extreme(df, CURRENT["RSI"]["period"], CURRENT["RSI"]["ob"], CURRENT["RSI"]["os"])
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           CURRENT["RSI"]["sl"], CURRENT["RSI"]["tp"], CURRENT["RSI"]["hold"],
                           CONTRACTS, "RSI", FLATTEN_TIME))
    # IB
    sigs = sig_ib_breakout(df, CURRENT["IB"]["ib_filter"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           CURRENT["IB"]["sl"], CURRENT["IB"]["tp"], CURRENT["IB"]["hold"],
                           CONTRACTS, "IB", FLATTEN_TIME))
    # MOM
    sigs = sig_momentum_bar(df, CURRENT["MOM"]["atr_mult"], CURRENT["MOM"]["vol_mult"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           CURRENT["MOM"]["sl"], CURRENT["MOM"]["tp"], CURRENT["MOM"]["hold"],
                           CONTRACTS, "MOM", FLATTEN_TIME))
    return trades


# ═════════════════════════════════════════════════════════════════════
# SL/TP PROFILE TEST
# ═════════════════════════════════════════════════════════════════════

def run_sltp_profiles(train_df, test_df, blind_df, train_arr, test_arr, blind_arr):
    """Test 4 extreme SL/TP profiles + current on all 3 strategies."""
    profiles = {
        "Tight/Tight (10/60)":  {"sl": 10, "tp": 60},
        "Wide/Wide (30/150)":   {"sl": 30, "tp": 150},
        "Tight SL/Wide TP":    {"sl": 10, "tp": 150},
        "Wide SL/Tight TP":    {"sl": 30, "tp": 60},
        "Current (15/100)":    {"sl": 15, "tp": 100},
    }
    results = {}
    for label, sltp in profiles.items():
        sl_t = pts_to_ticks(sltp["sl"])
        tp_t = pts_to_ticks(sltp["tp"])
        for period_label, df, arr in [("train", train_df, train_arr),
                                       ("test", test_df, test_arr),
                                       ("blind", blind_df, blind_arr)]:
            o, h, l, c, ts, hm = arr
            trades = []
            # RSI with current signal params, only vary SL/TP
            sigs = sig_rsi_extreme(df, 7, 70, 30)
            trades.extend(backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, 5,
                                   CONTRACTS, "RSI", FLATTEN_TIME))
            # IB
            sigs = sig_ib_breakout(df, True)[0]
            trades.extend(backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, 10,
                                   CONTRACTS, "IB", FLATTEN_TIME))
            # MOM
            sigs = sig_momentum_bar(df, 1.0, 1.0)[0]
            trades.extend(backtest(o, h, l, c, ts, hm, sigs, sl_t, tp_t, 5,
                                   CONTRACTS, "MOM", FLATTEN_TIME))
            m = calc_metrics(trades)
            if label not in results:
                results[label] = {}
            results[label][period_label] = m
    return results


# ═════════════════════════════════════════════════════════════════════
# PRINTING
# ═════════════════════════════════════════════════════════════════════

def print_top5(name, validated, current_entry):
    """Print top 5 per strategy + current params row."""
    print(f"\n  {name}:")
    if name == "RSI":
        hdr = f"  {'#':<3} {'Per':>3} {'OB':>3} {'OS':>3} {'SL':>4} {'TP':>4} {'Hold':>4} " \
              f"{'Train $':>10} {'Test $':>10} {'Blind $':>10} {'TrS':>5} {'TeS':>5} {'Stab':>5}"
        sep = "  " + "─" * len(hdr.strip())
        print(hdr)
        print(sep)
        for i, v in enumerate(validated[:5]):
            p = v["params"]
            tr = v.get("train_pnl", 0)
            te = v.get("test_pnl", 0)
            bl = v.get("blind_pnl", 0)
            ts_ = v.get("train_sharpe", 0)
            tes = v.get("test_sharpe", 0)
            stab = v.get("stability", 0)
            print(f"  {i+1:<3} {p['period']:>3} {p['ob']:>3} {p['os']:>3} "
                  f"{p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f} {stab:>4.0%}")
        # Current row
        if current_entry:
            p = current_entry["params"]
            tr = current_entry.get("train_pnl", 0)
            te = current_entry.get("test_pnl", 0)
            bl = current_entry.get("blind_pnl", 0)
            ts_ = current_entry.get("train_sharpe", 0)
            tes = current_entry.get("test_sharpe", 0)
            print(f"  {'C':<3} {p['period']:>3} {p['ob']:>3} {p['os']:>3} "
                  f"{p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f}  CUR")
    elif name == "IB":
        hdr = f"  {'#':<3} {'Filt':>4} {'SL':>4} {'TP':>4} {'Hold':>4} " \
              f"{'Train $':>10} {'Test $':>10} {'Blind $':>10} {'TrS':>5} {'TeS':>5} {'Stab':>5}"
        sep = "  " + "─" * len(hdr.strip())
        print(hdr)
        print(sep)
        for i, v in enumerate(validated[:5]):
            p = v["params"]
            tr = v.get("train_pnl", 0)
            te = v.get("test_pnl", 0)
            bl = v.get("blind_pnl", 0)
            ts_ = v.get("train_sharpe", 0)
            tes = v.get("test_sharpe", 0)
            stab = v.get("stability", 0)
            filt = "Y" if p["ib_filter"] else "N"
            print(f"  {i+1:<3} {filt:>4} {p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f} {stab:>4.0%}")
        if current_entry:
            p = current_entry["params"]
            tr = current_entry.get("train_pnl", 0)
            te = current_entry.get("test_pnl", 0)
            bl = current_entry.get("blind_pnl", 0)
            ts_ = current_entry.get("train_sharpe", 0)
            tes = current_entry.get("test_sharpe", 0)
            filt = "Y" if p["ib_filter"] else "N"
            print(f"  {'C':<3} {filt:>4} {p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f}  CUR")
    elif name == "MOM":
        hdr = f"  {'#':<3} {'ATR':>4} {'Vol':>4} {'SL':>4} {'TP':>4} {'Hold':>4} " \
              f"{'Train $':>10} {'Test $':>10} {'Blind $':>10} {'TrS':>5} {'TeS':>5} {'Stab':>5}"
        sep = "  " + "─" * len(hdr.strip())
        print(hdr)
        print(sep)
        for i, v in enumerate(validated[:5]):
            p = v["params"]
            tr = v.get("train_pnl", 0)
            te = v.get("test_pnl", 0)
            bl = v.get("blind_pnl", 0)
            ts_ = v.get("train_sharpe", 0)
            tes = v.get("test_sharpe", 0)
            stab = v.get("stability", 0)
            print(f"  {i+1:<3} {p['atr_mult']:>4.1f} {p['vol_mult']:>4.1f} "
                  f"{p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f} {stab:>4.0%}")
        if current_entry:
            p = current_entry["params"]
            tr = current_entry.get("train_pnl", 0)
            te = current_entry.get("test_pnl", 0)
            bl = current_entry.get("blind_pnl", 0)
            ts_ = current_entry.get("train_sharpe", 0)
            tes = current_entry.get("test_sharpe", 0)
            print(f"  {'C':<3} {p['atr_mult']:>4.1f} {p['vol_mult']:>4.1f} "
                  f"{p['sl']:>4} {p['tp']:>4} {p['hold']:>4} "
                  f"${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} "
                  f"{ts_:>5.1f} {tes:>5.1f}  CUR")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — PARAMETER OPTIMIZATION")
    print("═" * 70)
    print(f"  Train: 2022-01 to 2024-02 (26 months)")
    print(f"  Test:  2024-03 to 2025-02 (12 months)")
    print(f"  Blind: 2025-03 to 2026-03 (13 months, UNTOUCHED)")
    print(f"  Session: LucidFlex (flatten {FLATTEN_TIME})")
    print(f"  Contracts: {CONTRACTS} MNQ per strategy")
    print(f"  Cost per RT: ${rt_cost(CONTRACTS):.2f}")

    # ── Load data ───────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  LOADING DATA")
    print("━" * 70)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main (2024-2026)")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind (2022-2024)")

    # The "blind_data" file is actually 2022-2024.
    # We need: Train=2022-01 to 2024-02, Test=2024-03 to 2025-02, Blind=2025-03 to 2026-03
    # blind_data has 2022-2024 data, main_data has 2024-2026 data.
    # Split: Train = all of blind_data + main_data before 2024-03
    #        Test  = main_data from 2024-03 to 2025-03
    #        Blind = main_data from 2025-03 onward

    df_2022_2024 = blind_data["15m"]
    df_2024_2026 = main_data["15m"]

    # Train: blind_data (2022-2024) + main_data before TRAIN_END (2024-03-01)
    main_before_train_end = df_2024_2026.filter(pl.col("timestamp") < TRAIN_END)
    train_df = pl.concat([df_2022_2024, main_before_train_end]).sort("timestamp")

    # Test: main_data from TRAIN_END to TEST_END
    test_df = df_2024_2026.filter(
        (pl.col("timestamp") >= TRAIN_END) & (pl.col("timestamp") < TEST_END)
    )

    # Blind: main_data from TEST_END onward
    blind_df = df_2024_2026.filter(pl.col("timestamp") >= TEST_END)

    print(f"  Train: {len(train_df):,} bars ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Test:  {len(test_df):,} bars ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    print(f"  Blind: {len(blind_df):,} bars ({blind_df['timestamp'].min()} to {blind_df['timestamp'].max()})")

    train_arr = extract_arrays(train_df)
    test_arr = extract_arrays(test_df)
    blind_arr = extract_arrays(blind_df)

    # ═════════════════════════════════════════════════════════════════
    # STEP 1-2: OPTIMIZE EACH STRATEGY INDEPENDENTLY
    # ═════════════════════════════════════════════════════════════════

    # ── RSI ──────────────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  RSI STRATEGY OPTIMIZATION")
    print("━" * 70)
    rsi_grid = list(product(
        [5, 7, 10, 14],           # period
        [65, 70, 75, 80],         # overbought
        [20, 25, 30, 35],         # oversold
        [10, 15, 20, 25, 30],     # SL points
        [60, 80, 100, 120, 150],  # TP points
        [3, 5, 8, 10, 15],        # max hold
    ))
    print(f"  Grid: {len(rsi_grid):,} combos")
    rsi_validated = optimize_strategy("RSI", sweep_rsi, rsi_grid,
                                      train_df, test_df, train_arr, test_arr)
    gc.collect()

    # ── IB ───────────────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  IB STRATEGY OPTIMIZATION")
    print("━" * 70)
    ib_grid = list(product(
        [10, 15, 20, 25, 30],     # SL points
        [60, 80, 100, 120, 150],  # TP points
        [5, 8, 10, 15, 20],       # max hold
        [True, False],             # IB filter
    ))
    print(f"  Grid: {len(ib_grid):,} combos")
    ib_validated = optimize_strategy("IB", sweep_ib, ib_grid,
                                     train_df, test_df, train_arr, test_arr)
    gc.collect()

    # ── MOM ──────────────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  MOM STRATEGY OPTIMIZATION")
    print("━" * 70)
    mom_grid = list(product(
        [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],  # ATR mult
        [0.6, 0.8, 1.0, 1.2, 1.5],        # Vol mult
        [10, 15, 20, 25, 30],              # SL points
        [60, 80, 100, 120, 150],           # TP points
        [3, 5, 8, 10, 15],                 # max hold
    ))
    print(f"  Grid: {len(mom_grid):,} combos")
    mom_validated = optimize_strategy("MOM", sweep_mom, mom_grid,
                                      train_df, test_df, train_arr, test_arr)
    gc.collect()

    # ═════════════════════════════════════════════════════════════════
    # STEP 3: Run best + current on BLIND
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STEP 3-4: BLIND DATA EVALUATION")
    print("━" * 70)

    # Run current params on all 3 periods for comparison
    current_results = {}
    for label, df, arr in [("train", train_df, train_arr),
                           ("test", test_df, test_arr),
                           ("blind", blind_df, blind_arr)]:
        trades = run_current_on_period(df, arr)
        current_results[label] = calc_metrics(trades)
        current_results[label]["trades"] = trades

    # Run current params per-strategy on all periods (for per-strategy table)
    current_per_strat = {}
    for name, cfg in CURRENT.items():
        current_per_strat[name] = {"params": dict(cfg)}
        for label, df, arr in [("train", train_df, train_arr),
                               ("test", test_df, test_arr),
                               ("blind", blind_df, blind_arr)]:
            trades = run_on_blind(name, cfg, df, arr)
            m = calc_metrics(trades)
            current_per_strat[name][f"{label}_pnl"] = m["pnl"]
            current_per_strat[name][f"{label}_sharpe"] = m["sharpe"]
            current_per_strat[name][f"{label}_n"] = m["n"]

    # Run blind on validated top sets
    for name, validated in [("RSI", rsi_validated), ("IB", ib_validated), ("MOM", mom_validated)]:
        for v in validated:
            blind_trades = run_on_blind(name, v["params"], blind_df, blind_arr)
            m = calc_metrics(blind_trades)
            v["blind_pnl"] = m["pnl"]
            v["blind_sharpe"] = m["sharpe"]
            v["blind_n"] = m["n"]
            v["blind_trades"] = blind_trades
            v["blind_monthly"] = m.get("monthly", {})

    # ═════════════════════════════════════════════════════════════════
    # PRINT TOP 5 PER STRATEGY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  TOP 5 PER STRATEGY (ranked by test Sharpe)")
    print("═" * 70)

    print_top5("RSI", rsi_validated, current_per_strat.get("RSI"))
    print_top5("IB", ib_validated, current_per_strat.get("IB"))
    print_top5("MOM", mom_validated, current_per_strat.get("MOM"))

    # ═════════════════════════════════════════════════════════════════
    # STEP 4: COMBINE BEST AND COMPARE
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  COMBINED COMPARISON")
    print("═" * 70)

    # Build optimized combined trades on each period
    opt_best = {}
    opt_trades_combined = {"train": [], "test": [], "blind": []}
    for name, validated in [("RSI", rsi_validated), ("IB", ib_validated), ("MOM", mom_validated)]:
        if validated:
            best = validated[0]
            opt_best[name] = best["params"]
            # Run best on all periods for combined metrics
            for label, df, arr in [("train", train_df, train_arr),
                                   ("test", test_df, test_arr),
                                   ("blind", blind_df, blind_arr)]:
                trades = run_on_blind(name, best["params"], df, arr)
                opt_trades_combined[label].extend(trades)
        else:
            # Fallback to current if no validated params
            opt_best[name] = CURRENT[name]
            for label, df, arr in [("train", train_df, train_arr),
                                   ("test", test_df, test_arr),
                                   ("blind", blind_df, blind_arr)]:
                trades = run_on_blind(name, CURRENT[name], df, arr)
                opt_trades_combined[label].extend(trades)

    opt_combined_metrics = {label: calc_metrics(trades) for label, trades in opt_trades_combined.items()}
    cur_combined_metrics = {label: current_results[label] for label in ["train", "test", "blind"]}

    # Print combined table
    print(f"\n  {'System':<21} {'Train':>10} {'Test':>10} {'Blind':>10} {'Sharpe':>7}")
    print(f"  {'─' * 21} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 7}")
    for label, m_dict, s_label in [
        ("Current v3 params", cur_combined_metrics, "blind"),
        ("Optimized params", opt_combined_metrics, "blind"),
    ]:
        tr = m_dict["train"]["pnl"]
        te = m_dict["test"]["pnl"]
        bl = m_dict["blind"]["pnl"]
        sh = m_dict[s_label]["sharpe"]
        print(f"  {label:<21} ${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} {sh:>6.1f}")

    # Improvement
    cur_bl = cur_combined_metrics["blind"]["pnl"]
    opt_bl = opt_combined_metrics["blind"]["pnl"]
    if cur_bl != 0:
        imp = (opt_bl - cur_bl) / abs(cur_bl) * 100
        print(f"  {'Improvement':<21} {'':>10} {'':>10} {imp:>+9.1f}%")

    # ═════════════════════════════════════════════════════════════════
    # STEP 5: MONTE CARLO
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STEP 5: MONTE CARLO (5,000 sims on blind trades)")
    print("━" * 70)

    mc_current = run_mc(current_results["blind"]["trades"], 5000)
    mc_optimized = run_mc(opt_trades_combined["blind"], 5000)

    print(f"\n  {'System':<21} {'Pass':>7} {'Blowup':>7} {'Med Days':>9} {'P95 Days':>9}")
    print(f"  {'─' * 21} {'─' * 7} {'─' * 7} {'─' * 9} {'─' * 9}")
    print(f"  {'Current v3':<21} {mc_current['pass_rate']:>6.0%} {mc_current['blowup']:>6.0%} "
          f"{mc_current['med_days']:>9} {mc_current['p95_days']:>9}")
    print(f"  {'Optimized':<21} {mc_optimized['pass_rate']:>6.0%} {mc_optimized['blowup']:>6.0%} "
          f"{mc_optimized['med_days']:>9} {mc_optimized['p95_days']:>9}")

    # ═════════════════════════════════════════════════════════════════
    # SL/TP PROFILE TEST
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  SL/TP PROFILE TEST")
    print("━" * 70)

    profiles = run_sltp_profiles(train_df, test_df, blind_df, train_arr, test_arr, blind_arr)

    print(f"\n  {'Profile':<22} {'Train':>10} {'Test':>10} {'Blind':>10} {'WR':>6}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 6}")
    for label in ["Tight/Tight (10/60)", "Wide/Wide (30/150)",
                   "Tight SL/Wide TP", "Wide SL/Tight TP", "Current (15/100)"]:
        p = profiles[label]
        tr = p["train"]["pnl"]
        te = p["test"]["pnl"]
        bl = p["blind"]["pnl"]
        wr = p["blind"]["wr"]
        print(f"  {label:<22} ${tr:>+9,.0f} ${te:>+9,.0f} ${bl:>+9,.0f} {wr:>5.1f}%")

    # ═════════════════════════════════════════════════════════════════
    # MONTHLY BREAKDOWN (optimized, blind period)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  MONTHLY BREAKDOWN — BLIND PERIOD")
    print("━" * 70)

    opt_blind_monthly = opt_combined_metrics["blind"].get("monthly", {})
    cur_blind_monthly = cur_combined_metrics["blind"].get("monthly", {})
    all_months = sorted(set(list(opt_blind_monthly.keys()) + list(cur_blind_monthly.keys())))

    print(f"\n  {'Month':<10} {'Optimized':>12} {'Current':>12} {'Delta':>10}")
    print(f"  {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 10}")
    for m in all_months:
        ov = opt_blind_monthly.get(m, 0)
        cv = cur_blind_monthly.get(m, 0)
        delta = ov - cv
        o_mark = "+" if ov > 0 else "-"
        c_mark = "+" if cv > 0 else "-"
        print(f"  {m:<10} ${ov:>+11,.0f} ${cv:>+11,.0f} ${delta:>+9,.0f}")

    # ═════════════════════════════════════════════════════════════════
    # PARAMETER SENSITIVITY ANALYSIS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print("━" * 70)

    for name, validated in [("RSI", rsi_validated), ("IB", ib_validated), ("MOM", mom_validated)]:
        if len(validated) < 3:
            print(f"\n  {name}: Too few validated param sets to assess sensitivity")
            continue
        top5 = validated[:min(5, len(validated))]
        # Check how different the top params are
        for param_key in top5[0]["params"].keys():
            vals = [v["params"][param_key] for v in top5]
            unique = set(str(v) for v in vals)
            if len(unique) == 1:
                print(f"  {name}.{param_key}: STABLE — all top 5 use {vals[0]}")
            else:
                print(f"  {name}.{param_key}: VARIES — top 5 use {sorted(set(vals))}")

    # ═════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  VERDICT")
    print("═" * 70)

    opt_monthly_avg = opt_combined_metrics["blind"]["monthly_avg"]
    cur_monthly_avg = cur_combined_metrics["blind"]["monthly_avg"]
    opt_n_months = opt_combined_metrics["blind"]["n_months"]
    cur_n_months = cur_combined_metrics["blind"]["n_months"]

    # Check param distance
    param_diffs = []
    for name in ["RSI", "IB", "MOM"]:
        if name in opt_best:
            cur = CURRENT[name]
            opt = opt_best[name]
            for k in cur:
                if k in opt and cur[k] != opt[k]:
                    param_diffs.append(f"{name}.{k}: {cur[k]} -> {opt[k]}")

    if opt_bl > cur_bl * 1.05:
        # Optimized is >5% better
        print(f"\n  New params produce ${opt_monthly_avg:,.0f}/month vs ${cur_monthly_avg:,.0f} (current).")
        if cur_monthly_avg != 0:
            print(f"  Improvement of {(opt_bl - cur_bl) / abs(cur_bl) * 100:+.1f}% on blind data.")
        print(f"  MC pass rate: {mc_optimized['pass_rate']:.0%} (optimized) vs {mc_current['pass_rate']:.0%} (current)")
        if param_diffs:
            print(f"\n  Parameter changes:")
            for pd in param_diffs:
                print(f"    {pd}")
        print(f"\n  RECOMMEND: Switch to new params if MC pass rate >= current.")
    elif opt_bl >= cur_bl * 0.95:
        # Within 5%
        print(f"\n  The current params are near-optimal.")
        print(f"  Optimized: ${opt_monthly_avg:,.0f}/month vs Current: ${cur_monthly_avg:,.0f}/month")
        print(f"  Difference of {(opt_bl - cur_bl) / abs(cur_bl) * 100 if cur_bl != 0 else 0:+.1f}% — not worth changing.")
        print(f"\n  RECOMMEND: Stick with v3. The current params are robust.")
    else:
        # Current is better
        print(f"\n  The current params are BETTER than the 'optimized' ones on blind data.")
        print(f"  Current: ${cur_monthly_avg:,.0f}/month vs Optimized: ${opt_monthly_avg:,.0f}/month")
        print(f"  The grid search found params that overfit to 2022-2024 and don't generalize.")
        print(f"\n  RECOMMEND: Stick with v3. The original params are robust.")

    # Fragility warning
    if len(param_diffs) > 3:
        print(f"\n  WARNING: {len(param_diffs)} params differ between current and optimal.")
        print(f"  This suggests the edge may be sensitive to parameter choices.")
    elif rsi_validated and len(rsi_validated) >= 5:
        # Check if top 5 are similar
        rsi_pnls = [v["blind_pnl"] for v in rsi_validated[:5]]
        spread = max(rsi_pnls) - min(rsi_pnls) if rsi_pnls else 0
        if spread < abs(rsi_pnls[0]) * 0.3:
            print(f"\n  GOOD SIGN: Top 5 RSI param sets produce similar blind results "
                  f"(spread ${spread:,.0f}). Edge is robust, not param-dependent.")

    # ═════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════
    report = {
        "timestamp": str(datetime.now()),
        "data_splits": {
            "train": "2022-01 to 2024-02 (26 months)",
            "test": "2024-03 to 2025-02 (12 months)",
            "blind": "2025-03 to 2026-03 (13 months)",
        },
        "current_params": {k: {kk: vv for kk, vv in v.items()} for k, v in CURRENT.items()},
        "optimized_params": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list,))}
                             for k, v in opt_best.items()},
        "rsi_top5": [{"params": v["params"], "train_pnl": v["train_pnl"],
                       "test_pnl": v["test_pnl"], "blind_pnl": v["blind_pnl"],
                       "train_sharpe": v["train_sharpe"], "test_sharpe": v["test_sharpe"],
                       "stability": v["stability"]}
                      for v in rsi_validated[:5]],
        "ib_top5": [{"params": v["params"], "train_pnl": v["train_pnl"],
                      "test_pnl": v["test_pnl"], "blind_pnl": v["blind_pnl"],
                      "train_sharpe": v["train_sharpe"], "test_sharpe": v["test_sharpe"],
                      "stability": v["stability"]}
                     for v in ib_validated[:5]],
        "mom_top5": [{"params": v["params"], "train_pnl": v["train_pnl"],
                       "test_pnl": v["test_pnl"], "blind_pnl": v["blind_pnl"],
                       "train_sharpe": v["train_sharpe"], "test_sharpe": v["test_sharpe"],
                       "stability": v["stability"]}
                      for v in mom_validated[:5]],
        "combined_current": {
            "train_pnl": cur_combined_metrics["train"]["pnl"],
            "test_pnl": cur_combined_metrics["test"]["pnl"],
            "blind_pnl": cur_combined_metrics["blind"]["pnl"],
            "blind_sharpe": cur_combined_metrics["blind"]["sharpe"],
            "blind_monthly_avg": cur_combined_metrics["blind"]["monthly_avg"],
        },
        "combined_optimized": {
            "train_pnl": opt_combined_metrics["train"]["pnl"],
            "test_pnl": opt_combined_metrics["test"]["pnl"],
            "blind_pnl": opt_combined_metrics["blind"]["pnl"],
            "blind_sharpe": opt_combined_metrics["blind"]["sharpe"],
            "blind_monthly_avg": opt_combined_metrics["blind"]["monthly_avg"],
        },
        "mc_current": mc_current,
        "mc_optimized": mc_optimized,
        "sltp_profiles": {
            label: {period: {"pnl": m["pnl"], "n": m["n"], "wr": m["wr"], "sharpe": m["sharpe"]}
                    for period, m in periods.items()}
            for label, periods in profiles.items()
        },
        "param_changes": param_diffs,
    }

    out = REPORTS_DIR / "htf_swing_optimize.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 70)


if __name__ == "__main__":
    main()
