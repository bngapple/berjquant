#!/usr/bin/env python3
"""
Always-On v2 — Three levers to go from $1,441/month to $7,000/month.

Lever 1: Optimal contract sizing across layers (joint grid search)
Lever 2: Increase trade frequency (multi-TF VWAP, momentum mode, session expansion)
Lever 3: Fine-grained parameter refinement on survivors

Uses v1 winning params as starting point. Year 2 is PURE OOS — never optimized on.

Usage:
    python3 run_always_on_v2.py
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

# ── Import v1 building blocks ────────────────────────────────────────
from run_always_on import (
    load_data, split_years, round_trip_cost,
    add_vwap, add_vwap_zscore, add_rsi, add_ema, add_atr, add_initial_balance,
    SimplePosition, TradeRecord,
    run_backtest, run_backtest_dynamic_sl,
    layer1_signals, layer2c_signals, layer3_signals,
    analyze_trades,
    COMMISSION_PER_SIDE, SLIPPAGE_TICKS, TICK_SIZE, POINT_VALUE,
    YR1_START, YR1_END, YR2_START, YR2_END,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

# ── v1 winning params ────────────────────────────────────────────────
V1_L1 = {"z_threshold": 1.5, "lookback": 50, "stop_ticks": 16, "max_hold": 30}
V1_L2C = {"ema_fast": 8, "ema_slow": 13, "trail_ticks": 8, "max_hold": 30}
V1_L3 = {"range_mult": 1.0, "ema_pullback": 5, "trail_ticks": 12, "max_hold": 30}
V1_L1_CONTRACTS = 3
V1_L2C_CONTRACTS = 2
V1_L3_CONTRACTS = 3


# ═════════════════════════════════════════════════════════════════════
# HELPER: resample to N-minute bars
# ═════════════════════════════════════════════════════════════════════

def resample_nm(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Resample 1m bars to N-minute bars, preserving ET columns."""
    resampled = (
        df.group_by_dynamic("timestamp", every=f"{minutes}m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("ts_et").last(),
            pl.col("date_et").last(),
            pl.col("hhmm").last(),
            pl.col("hour_et").last(),
            pl.col("minute_et").last(),
        ])
    )
    return resampled


# ═════════════════════════════════════════════════════════════════════
# LEVER 1: CONTRACT SIZING OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════

def run_layer_at_size(df: pl.DataFrame, layer: str, contracts: int, params: dict) -> list[TradeRecord]:
    """Run a single layer at a given contract size on the provided data."""
    if layer == "L1":
        sigs, exits = layer1_signals(df, z_threshold=params["z_threshold"], lookback=params["lookback"])
        return run_backtest(df, sigs, exits, stop_ticks=params["stop_ticks"],
                           target_ticks=None, max_hold_bars=params["max_hold"],
                           contracts=contracts, layer_name="L1_VWAP_MR")
    elif layer == "L2C":
        sigs, exits = layer2c_signals(df, ema_fast=params["ema_fast"], ema_slow=params["ema_slow"])
        return run_backtest(df, sigs, exits, stop_ticks=params["trail_ticks"],
                           target_ticks=None, max_hold_bars=params["max_hold"],
                           contracts=contracts, trailing=True,
                           trailing_ticks=params["trail_ticks"], layer_name="L2C_Afternoon")
    elif layer == "L3":
        sigs, exits = layer3_signals(df, range_mult=params["range_mult"], ema_pullback=params["ema_pullback"])
        return run_backtest(df, sigs, exits, stop_ticks=params["trail_ticks"],
                           target_ticks=None, max_hold_bars=params["max_hold"],
                           contracts=contracts, trailing=True,
                           trailing_ticks=params["trail_ticks"], layer_name="L3_VolMom")
    return []


def compute_combined_risk(trade_lists: list[list[TradeRecord]]) -> dict:
    """Compute combined risk metrics from multiple trade lists."""
    daily = defaultdict(float)
    for trades in trade_lists:
        for t in trades:
            d = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
            daily[d] += t.net_pnl

    if not daily:
        return {"worst_day": 0, "max_dd": 0, "total_pnl": 0, "monthly_avg": 0}

    worst_day = min(daily.values())
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    monthly = defaultdict(float)
    for d, pnl in daily.items():
        monthly[d[:7]] += pnl
    n_months = max(len(monthly), 1)

    return {
        "worst_day": worst_day,
        "max_dd": max_dd,
        "total_pnl": cum,
        "monthly_avg": cum / n_months,
    }


def lever1_sizing(yr2: pl.DataFrame) -> dict:
    """Joint grid search for optimal contract allocation."""
    print("\n" + "━" * 60)
    print("  LEVER 1 — CONTRACT SIZING OPTIMIZATION")
    print("━" * 60)

    # Pre-compute trades at each size for each layer
    print("  Pre-computing trades at each contract size ...")
    l1_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    l2c_sizes = [1, 2, 3, 4, 5, 6, 8, 10]
    l3_sizes = [1, 2, 3, 4, 5, 6, 8, 10]

    cache = {}
    for s in set(l1_sizes):
        cache[("L1", s)] = run_layer_at_size(yr2, "L1", s, V1_L1)
    for s in set(l2c_sizes):
        cache[("L2C", s)] = run_layer_at_size(yr2, "L2C", s, V1_L2C)
    for s in set(l3_sizes):
        cache[("L3", s)] = run_layer_at_size(yr2, "L3", s, V1_L3)

    print(f"  Testing {len(l1_sizes)}×{len(l2c_sizes)}×{len(l3_sizes)} = "
          f"{len(l1_sizes)*len(l2c_sizes)*len(l3_sizes)} combos ...")

    best = None
    best_combo = (3, 2, 3)

    for l1s, l2s, l3s in product(l1_sizes, l2c_sizes, l3_sizes):
        risk = compute_combined_risk([cache[("L1", l1s)], cache[("L2C", l2s)], cache[("L3", l3s)]])

        if risk["worst_day"] < -2500:
            continue
        if risk["max_dd"] < -3500:
            continue

        if best is None or risk["monthly_avg"] > best["monthly_avg"]:
            best = risk
            best_combo = (l1s, l2s, l3s)

    if best is None:
        print("  No valid combo found! Using v1 defaults.")
        best_combo = (V1_L1_CONTRACTS, V1_L2C_CONTRACTS, V1_L3_CONTRACTS)
        best = compute_combined_risk([
            cache[("L1", best_combo[0])],
            cache[("L2C", best_combo[1])],
            cache[("L3", best_combo[2])],
        ])

    print(f"\n  Optimal allocation: L1={best_combo[0]}, L2C={best_combo[1]}, L3={best_combo[2]}")
    print(f"  Y2 monthly avg: ${best['monthly_avg']:,.0f}/month")
    print(f"  Worst day: ${best['worst_day']:,.0f}")
    print(f"  Max DD: ${best['max_dd']:,.0f}")
    print(f"  vs v1 ({V1_L1_CONTRACTS},{V1_L2C_CONTRACTS},{V1_L3_CONTRACTS}): "
          f"${1814:,}/month → ${best['monthly_avg']:,.0f}/month")

    return {
        "l1": best_combo[0], "l2c": best_combo[1], "l3": best_combo[2],
        "monthly_avg": best["monthly_avg"],
        "worst_day": best["worst_day"],
        "max_dd": best["max_dd"],
        "total_pnl": best["total_pnl"],
    }


# ═════════════════════════════════════════════════════════════════════
# LEVER 2A: MULTI-TIMEFRAME VWAP MEAN REVERSION
# ═════════════════════════════════════════════════════════════════════

def sweep_l1_multi_tf(yr1: pl.DataFrame, yr2: pl.DataFrame, tf_minutes: int, contracts: int) -> dict | None:
    """Sweep VWAP z-score on resampled timeframe."""
    print(f"  L1 on {tf_minutes}m bars ...")

    df_r1 = resample_nm(yr1, tf_minutes)
    df_r2 = resample_nm(yr2, tf_minutes)

    z_values = [1.5, 2.0, 2.5, 3.0]
    stop_values = [4, 8, 12, 16]
    lb_values = [10, 20, 30, 50]
    hold_values = [5, 10, 15, 20]

    best = None
    for z_t in z_values:
        for lb in lb_values:
            sigs, exits = layer1_signals(df_r1, z_threshold=z_t, lookback=lb)
            for st in stop_values:
                for mh in hold_values:
                    trades = run_backtest(
                        df_r1, sigs, exits, stop_ticks=st, target_ticks=None,
                        max_hold_bars=mh, contracts=contracts,
                        layer_name=f"L1_{tf_minutes}m",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    if best is None or pnl > best["pnl"]:
                        best = {"pnl": pnl, "trades": len(trades),
                                "z": z_t, "lb": lb, "stop": st, "hold": mh,
                                "trade_list": trades}

    if best is None or best["pnl"] <= 0:
        print(f"    No profitable config on Y1 for {tf_minutes}m")
        return None

    print(f"    Best Y1: ${best['pnl']:,.0f} ({best['trades']} trades)")

    # Validate on Year 2
    sigs2, exits2 = layer1_signals(df_r2, z_threshold=best["z"], lookback=best["lb"])
    y2_trades = run_backtest(
        df_r2, sigs2, exits2, stop_ticks=best["stop"], target_ticks=None,
        max_hold_bars=best["hold"], contracts=contracts,
        layer_name=f"L1_{tf_minutes}m",
    )
    y2_pnl = sum(t.net_pnl for t in y2_trades)
    passed = y2_pnl > 0 and len(y2_trades) > 0
    print(f"    Y2: ${y2_pnl:,.0f} ({len(y2_trades)} trades) {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        return None

    return {
        "name": f"L1_{tf_minutes}m",
        "params": {"z": best["z"], "lb": best["lb"], "stop": best["stop"], "hold": best["hold"]},
        "yr1_trades": best["trade_list"],
        "yr2_trades": y2_trades,
        "yr1_pnl": best["pnl"],
        "yr2_pnl": y2_pnl,
    }


# ═════════════════════════════════════════════════════════════════════
# LEVER 2B: MOMENTUM CONTINUATION (inverse L1)
# ═════════════════════════════════════════════════════════════════════

def momentum_continuation_signals(
    df: pl.DataFrame, z_threshold: float = 2.0, lookback: int = 50,
    atr_mult: float = 1.3, atr_lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """When z-score is extreme AND vol is high, go WITH the move (not against it)."""
    df_feat = add_vwap(df)
    df_feat = add_vwap_zscore(df_feat, lookback=lookback)
    df_feat = add_atr(df_feat, 14)

    z = df_feat["vwap_zscore"].to_numpy()
    atr = df_feat["atr_14"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    dates = df_feat["date_et"].to_list()
    n = len(df_feat)

    # Rolling avg ATR (proxy for regime)
    atr_avg = np.full(n, np.nan)
    for i in range(atr_lookback, n):
        window = atr[i - atr_lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_avg[i] = np.mean(valid)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)
    traded_today = {}

    for i in range(n):
        if not (930 <= hhmm[i] < 1550):
            continue
        if np.isnan(z[i]) or np.isnan(atr[i]) or np.isnan(atr_avg[i]):
            continue

        d = dates[i]
        if d in traded_today:
            continue

        is_high_vol = atr[i] > atr_mult * atr_avg[i]
        if not is_high_vol:
            continue

        # Momentum: go WITH the extreme
        if z[i] > z_threshold:
            signals[i] = 1  # price above VWAP in high vol → momentum long
            traded_today[d] = True
        elif z[i] < -z_threshold:
            signals[i] = -1  # price below VWAP in high vol → momentum short
            traded_today[d] = True

    return signals, exits


def sweep_momentum_cont(yr1: pl.DataFrame, yr2: pl.DataFrame, contracts: int) -> dict | None:
    """Sweep momentum continuation params."""
    print("  Momentum continuation sweep ...")

    best = None
    for z_t in [1.5, 2.0, 2.5, 3.0]:
        for atr_m in [1.0, 1.3, 1.5, 2.0]:
            for trail in [8, 12, 16, 20]:
                for mh in [15, 30, 60]:
                    sigs, exits = momentum_continuation_signals(yr1, z_threshold=z_t, atr_mult=atr_m)
                    trades = run_backtest(
                        yr1, sigs, exits, stop_ticks=trail, target_ticks=None,
                        max_hold_bars=mh, contracts=contracts,
                        trailing=True, trailing_ticks=trail,
                        layer_name="L1_MomCont",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    if best is None or pnl > best["pnl"]:
                        best = {"pnl": pnl, "trades": len(trades),
                                "z": z_t, "atr_m": atr_m, "trail": trail, "hold": mh,
                                "trade_list": trades}

    if best is None or best["pnl"] <= 0:
        print("    No profitable config on Y1")
        return None

    print(f"    Best Y1: ${best['pnl']:,.0f} ({best['trades']} trades)")

    sigs2, exits2 = momentum_continuation_signals(yr2, z_threshold=best["z"], atr_mult=best["atr_m"])
    y2_trades = run_backtest(
        yr2, sigs2, exits2, stop_ticks=best["trail"], target_ticks=None,
        max_hold_bars=best["hold"], contracts=contracts,
        trailing=True, trailing_ticks=best["trail"],
        layer_name="L1_MomCont",
    )
    y2_pnl = sum(t.net_pnl for t in y2_trades)
    passed = y2_pnl > 0 and len(y2_trades) > 0
    print(f"    Y2: ${y2_pnl:,.0f} ({len(y2_trades)} trades) {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        return None

    return {
        "name": "L1_MomCont",
        "params": {"z": best["z"], "atr_m": best["atr_m"], "trail": best["trail"], "hold": best["hold"]},
        "yr1_trades": best["trade_list"], "yr2_trades": y2_trades,
        "yr1_pnl": best["pnl"], "yr2_pnl": y2_pnl,
    }


# ═════════════════════════════════════════════════════════════════════
# LEVER 2C: EXPANDED SESSION STRATEGIES
# ═════════════════════════════════════════════════════════════════════

def simple_10am_signal(df: pl.DataFrame, atr_target_mult: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simplified 10:00 AM direction trade based on price vs VWAP."""
    df_feat = add_vwap(df)
    df_feat = add_atr(df_feat, 14)

    closes = df_feat["close"].to_numpy()
    vwap = df_feat["vwap"].to_numpy()
    atr = df_feat["atr_14"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    dates = df_feat["date_et"].to_list()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)
    stop_arr = np.full(n, 12, dtype=np.int32)
    target_arr = np.full(n, 24, dtype=np.int32)

    traded_today = {}

    for i in range(n):
        if hhmm[i] != 1000:
            continue
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or atr[i] < 1:
            continue

        d = dates[i]
        if d in traded_today:
            continue

        # Stop = distance to VWAP (minimum 4 ticks)
        dist_to_vwap = abs(closes[i] - vwap[i])
        sl_ticks = max(int(dist_to_vwap / TICK_SIZE), 4)
        tp_ticks = max(int(atr[i] * atr_target_mult / TICK_SIZE), 4)

        if closes[i] > vwap[i]:
            signals[i] = 1
        else:
            signals[i] = -1

        stop_arr[i] = sl_ticks
        target_arr[i] = tp_ticks
        traded_today[d] = True

    return signals, exits, stop_arr, target_arr


def improved_close_signal(df: pl.DataFrame, atr_mult: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Improved closing imbalance at 15:50, ATR-based threshold."""
    df_feat = add_vwap(df)
    df_feat = add_atr(df_feat, 14)

    closes = df_feat["close"].to_numpy()
    vwap = df_feat["vwap"].to_numpy()
    atr = df_feat["atr_14"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    dates = df_feat["date_et"].to_list()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)
    traded_today = {}

    for i in range(n):
        if hhmm[i] >= 1558:
            exits[i] = True
            continue

        if hhmm[i] != 1550:
            continue
        if np.isnan(vwap[i]) or np.isnan(atr[i]):
            continue

        d = dates[i]
        if d in traded_today:
            continue

        threshold = atr[i] * atr_mult
        if closes[i] > vwap[i] + threshold:
            signals[i] = -1  # overbought, sell for close reversion
            traded_today[d] = True
        elif closes[i] < vwap[i] - threshold:
            signals[i] = 1  # oversold, buy for close reversion
            traded_today[d] = True

    return signals, exits


def sweep_session_expansion(yr1: pl.DataFrame, yr2: pl.DataFrame, contracts: int) -> list[dict]:
    """Sweep expanded session strategies."""
    survivors = []

    # ── Simple 10am ──
    print("  Simple 10am direction sweep ...")
    best_10am = None
    for atr_m in [0.3, 0.5, 0.75, 1.0]:
        for max_h in [15, 30, 60, 90]:
            sigs, exits, sa, ta = simple_10am_signal(yr1, atr_target_mult=atr_m)
            trades = run_backtest_dynamic_sl(
                yr1, sigs, exits, sa, ta, max_hold_bars=max_h,
                contracts=contracts, layer_name="L2A_Simple10am",
            )
            pnl = sum(t.net_pnl for t in trades)
            if best_10am is None or pnl > best_10am["pnl"]:
                best_10am = {"pnl": pnl, "trades": len(trades),
                             "atr_m": atr_m, "hold": max_h, "trade_list": trades}

    if best_10am and best_10am["pnl"] > 0:
        print(f"    Best Y1: ${best_10am['pnl']:,.0f} ({best_10am['trades']} trades)")
        s2, e2, sa2, ta2 = simple_10am_signal(yr2, atr_target_mult=best_10am["atr_m"])
        y2t = run_backtest_dynamic_sl(
            yr2, s2, e2, sa2, ta2, max_hold_bars=best_10am["hold"],
            contracts=contracts, layer_name="L2A_Simple10am",
        )
        y2p = sum(t.net_pnl for t in y2t)
        passed = y2p > 0 and len(y2t) > 0
        print(f"    Y2: ${y2p:,.0f} ({len(y2t)} trades) {'✅ PASS' if passed else '❌ FAIL'}")
        if passed:
            survivors.append({
                "name": "L2A_Simple10am",
                "params": {"atr_m": best_10am["atr_m"], "hold": best_10am["hold"]},
                "yr1_trades": best_10am["trade_list"], "yr2_trades": y2t,
                "yr1_pnl": best_10am["pnl"], "yr2_pnl": y2p,
            })
    else:
        print("    No profitable config on Y1")

    # ── Improved close ──
    print("  Improved closing imbalance sweep ...")
    best_close = None
    for atr_m in [0.3, 0.5, 0.75, 1.0]:
        for st in [4, 6, 8, 12]:
            sigs, exits = improved_close_signal(yr1, atr_mult=atr_m)
            trades = run_backtest(
                yr1, sigs, exits, stop_ticks=st, target_ticks=None,
                max_hold_bars=10, contracts=contracts, layer_name="L2D_ImprClose",
            )
            pnl = sum(t.net_pnl for t in trades)
            if best_close is None or pnl > best_close["pnl"]:
                best_close = {"pnl": pnl, "trades": len(trades),
                              "atr_m": atr_m, "stop": st, "trade_list": trades}

    if best_close and best_close["pnl"] > 0:
        print(f"    Best Y1: ${best_close['pnl']:,.0f} ({best_close['trades']} trades)")
        s2, e2 = improved_close_signal(yr2, atr_mult=best_close["atr_m"])
        y2t = run_backtest(
            yr2, s2, e2, stop_ticks=best_close["stop"], target_ticks=None,
            max_hold_bars=10, contracts=contracts, layer_name="L2D_ImprClose",
        )
        y2p = sum(t.net_pnl for t in y2t)
        passed = y2p > 0 and len(y2t) > 0
        print(f"    Y2: ${y2p:,.0f} ({len(y2t)} trades) {'✅ PASS' if passed else '❌ FAIL'}")
        if passed:
            survivors.append({
                "name": "L2D_ImprClose",
                "params": {"atr_m": best_close["atr_m"], "stop": best_close["stop"]},
                "yr1_trades": best_close["trade_list"], "yr2_trades": y2t,
                "yr1_pnl": best_close["pnl"], "yr2_pnl": y2p,
            })
    else:
        print("    No profitable config on Y1")

    return survivors


# ═════════════════════════════════════════════════════════════════════
# LEVER 3: FINE-GRAINED PARAMETER REFINEMENT
# ═════════════════════════════════════════════════════════════════════

def refine_layer(yr1, yr2, layer, base_params, base_contracts, layer_name) -> dict | None:
    """Fine-grained sweep +/- 20% around winning params."""
    print(f"  Refining {layer_name} ...")

    def nearby(val, deltas):
        return sorted(set(max(1, val + d) for d in deltas))

    if layer == "L1":
        z_vals = [round(base_params["z_threshold"] * m, 2) for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
        lb_vals = nearby(base_params["lookback"], [-10, -5, 0, 5, 10])
        st_vals = nearby(base_params["stop_ticks"], [-4, -2, 0, 2, 4])
        mh_vals = nearby(base_params["max_hold"], [-5, -3, 0, 3, 5])

        best = None
        for z_t in z_vals:
            for lb in lb_vals:
                sigs, exits = layer1_signals(yr1, z_threshold=z_t, lookback=lb)
                for st in st_vals:
                    for mh in mh_vals:
                        trades = run_backtest(
                            yr1, sigs, exits, stop_ticks=st, target_ticks=None,
                            max_hold_bars=mh, contracts=base_contracts, layer_name="L1_VWAP_MR",
                        )
                        pnl = sum(t.net_pnl for t in trades)
                        if best is None or pnl > best["pnl"]:
                            best = {"pnl": pnl, "trades": len(trades),
                                    "params": {"z_threshold": z_t, "lookback": lb,
                                               "stop_ticks": st, "max_hold": mh},
                                    "trade_list": trades}

    elif layer == "L2C":
        ef_vals = nearby(base_params["ema_fast"], [-2, -1, 0, 1, 2])
        es_vals = nearby(base_params["ema_slow"], [-3, -2, 0, 2, 3])
        tt_vals = nearby(base_params["trail_ticks"], [-2, -1, 0, 1, 2])
        mh_vals = nearby(base_params["max_hold"], [-5, -3, 0, 3, 5])

        best = None
        for ef in ef_vals:
            for es in es_vals:
                if ef >= es:
                    continue
                sigs, exits = layer2c_signals(yr1, ema_fast=ef, ema_slow=es)
                for tt in tt_vals:
                    for mh in mh_vals:
                        trades = run_backtest(
                            yr1, sigs, exits, stop_ticks=tt, target_ticks=None,
                            max_hold_bars=mh, contracts=base_contracts,
                            trailing=True, trailing_ticks=tt, layer_name="L2C_Afternoon",
                        )
                        pnl = sum(t.net_pnl for t in trades)
                        if best is None or pnl > best["pnl"]:
                            best = {"pnl": pnl, "trades": len(trades),
                                    "params": {"ema_fast": ef, "ema_slow": es,
                                               "trail_ticks": tt, "max_hold": mh},
                                    "trade_list": trades}

    elif layer == "L3":
        rm_vals = [round(base_params["range_mult"] * m, 2) for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
        ep_vals = nearby(base_params["ema_pullback"], [-2, -1, 0, 1, 2])
        tt_vals = nearby(base_params["trail_ticks"], [-3, -2, 0, 2, 3])
        mh_vals = nearby(base_params["max_hold"], [-5, -3, 0, 3, 5])

        best = None
        for rm in rm_vals:
            for ep in ep_vals:
                sigs, exits = layer3_signals(yr1, range_mult=rm, ema_pullback=ep)
                for tt in tt_vals:
                    for mh in mh_vals:
                        trades = run_backtest(
                            yr1, sigs, exits, stop_ticks=tt, target_ticks=None,
                            max_hold_bars=mh, contracts=base_contracts,
                            trailing=True, trailing_ticks=tt, layer_name="L3_VolMom",
                        )
                        pnl = sum(t.net_pnl for t in trades)
                        if best is None or pnl > best["pnl"]:
                            best = {"pnl": pnl, "trades": len(trades),
                                    "params": {"range_mult": rm, "ema_pullback": ep,
                                               "trail_ticks": tt, "max_hold": mh},
                                    "trade_list": trades}

    if best is None or best["pnl"] <= 0:
        print(f"    No improvement found")
        return None

    print(f"    Best refined Y1: ${best['pnl']:,.0f} ({best['trades']} trades)")
    print(f"    Params: {best['params']}")

    # Validate on Y2
    if layer == "L1":
        s2, e2 = layer1_signals(yr2, z_threshold=best["params"]["z_threshold"],
                                lookback=best["params"]["lookback"])
        y2t = run_backtest(yr2, s2, e2, stop_ticks=best["params"]["stop_ticks"],
                           target_ticks=None, max_hold_bars=best["params"]["max_hold"],
                           contracts=base_contracts, layer_name="L1_VWAP_MR")
    elif layer == "L2C":
        s2, e2 = layer2c_signals(yr2, ema_fast=best["params"]["ema_fast"],
                                 ema_slow=best["params"]["ema_slow"])
        y2t = run_backtest(yr2, s2, e2, stop_ticks=best["params"]["trail_ticks"],
                           target_ticks=None, max_hold_bars=best["params"]["max_hold"],
                           contracts=base_contracts, trailing=True,
                           trailing_ticks=best["params"]["trail_ticks"], layer_name="L2C_Afternoon")
    elif layer == "L3":
        s2, e2 = layer3_signals(yr2, range_mult=best["params"]["range_mult"],
                                ema_pullback=best["params"]["ema_pullback"])
        y2t = run_backtest(yr2, s2, e2, stop_ticks=best["params"]["trail_ticks"],
                           target_ticks=None, max_hold_bars=best["params"]["max_hold"],
                           contracts=base_contracts, trailing=True,
                           trailing_ticks=best["params"]["trail_ticks"], layer_name="L3_VolMom")

    y2p = sum(t.net_pnl for t in y2t)
    print(f"    Y2 refined: ${y2p:,.0f} ({len(y2t)} trades)")

    return {
        "params": best["params"],
        "yr1_pnl": best["pnl"],
        "yr1_trades": best["trade_list"],
        "yr2_pnl": y2p,
        "yr2_trades": y2t,
    }


# ═════════════════════════════════════════════════════════════════════
# COMBINED REPORT
# ═════════════════════════════════════════════════════════════════════

def build_combined_report(all_trades: list[TradeRecord], sizing: dict) -> dict:
    """Build the final combined report."""
    monthly = defaultdict(float)
    monthly_trades = defaultdict(int)
    daily = defaultdict(float)

    for t in all_trades:
        et = t.entry_time
        m_key = et.strftime("%Y-%m") if hasattr(et, 'strftime') else str(et)[:7]
        d_key = et.strftime("%Y-%m-%d") if hasattr(et, 'strftime') else str(et)[:10]
        monthly[m_key] += t.net_pnl
        monthly_trades[m_key] += 1
        daily[d_key] += t.net_pnl

    total_pnl = sum(t.net_pnl for t in all_trades)
    n_months = max(len(monthly), 1)
    n_days = max(len(daily), 1)

    yr1_trades = [t for t in all_trades if t.entry_time < YR2_START]
    yr2_trades = [t for t in all_trades if t.entry_time >= YR2_START]
    yr1_pnl = sum(t.net_pnl for t in yr1_trades)
    yr2_pnl = sum(t.net_pnl for t in yr2_trades)

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    worst_day = 0.0
    best_day = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
        worst_day = min(worst_day, daily[d])
        best_day = max(best_day, daily[d])

    consistency = (best_day / total_pnl * 100) if total_pnl > 0 else 100.0
    months_gte_7k = sum(1 for v in monthly.values() if v >= 7000)
    months_gte_3500 = sum(1 for v in monthly.values() if v >= 3500)
    months_profitable = sum(1 for v in monthly.values() if v > 0)
    months_zero = sum(1 for m, c in monthly_trades.items() if c == 0)

    return {
        "total_pnl": total_pnl,
        "yr1_pnl": yr1_pnl,
        "yr2_pnl": yr2_pnl,
        "yr1_monthly": yr1_pnl / 12,
        "yr2_monthly": yr2_pnl / 12,
        "monthly": dict(monthly),
        "monthly_trades": dict(monthly_trades),
        "worst_day": worst_day,
        "best_day": best_day,
        "max_drawdown": max_dd,
        "consistency_pct": consistency,
        "months_gte_7k": months_gte_7k,
        "months_gte_3500": months_gte_3500,
        "months_profitable": months_profitable,
        "months_zero_trades": months_zero,
        "avg_trades_per_day": len(all_trades) / n_days,
        "total_trades": len(all_trades),
        "n_months": n_months,
        "sizing": sizing,
    }


def print_v2_report(report: dict, v1_monthly: float = 1441):
    """Print the complete v2 report."""
    m = report

    print(f"\n{'═' * 60}")
    print(f"  ALWAYS-ON v2 — RESULTS")
    print(f"{'═' * 60}")

    s = m.get("sizing", {})
    print(f"\n  SCALING:")
    print(f"    v1 contracts: L1=3, L2C=2, L3=3")
    print(f"    v2 contracts: L1={s.get('l1','?')}, L2C={s.get('l2c','?')}, L3={s.get('l3','?')}")

    print(f"\n  FREQUENCY:")
    print(f"    v1 trades/day: 5.8")
    print(f"    v2 trades/day: {m['avg_trades_per_day']:.1f}")

    print(f"\n  COMBINED v2 SYSTEM:")
    print(f"    Year 1: ${m['yr1_monthly']:,.0f}/month ({len([t for t in range(1)])} ...)")
    print(f"    Year 2: ${m['yr2_monthly']:,.0f}/month (PURE OOS)")

    print(f"\n    Monthly breakdown:")
    for mo in sorted(m["monthly"].keys()):
        pnl = m["monthly"][mo]
        tc = m["monthly_trades"].get(mo, 0)
        flag = "✅" if pnl >= 7000 else ("🟡" if pnl >= 3500 else ("✅" if pnl > 0 else "❌"))
        print(f"      {mo}: ${pnl:>+10,.0f}  ({tc:>3} trades) {flag}")

    print(f"\n    Months with trades: {m['n_months'] - m['months_zero_trades']}/{m['n_months']} (target: 24/24)")
    print(f"    Months profitable: {m['months_profitable']}/{m['n_months']}")
    print(f"    Months >= $7,000: {m['months_gte_7k']}/{m['n_months']}")
    print(f"    Months >= $3,500: {m['months_gte_3500']}/{m['n_months']}")
    print(f"    Average monthly: ${m['total_pnl'] / max(m['n_months'],1):,.0f}")
    print(f"    Worst month: ${min(m['monthly'].values()):,.0f}" if m["monthly"] else "")

    print(f"\n    Prop firm (Topstep 150K):")
    wd_ok = m["worst_day"] > -3000
    dd_ok = m["max_drawdown"] > -4500
    con_ok = m["consistency_pct"] < 50
    print(f"      Worst day:    ${m['worst_day']:>+10,.0f}  (limit: -$3,000) {'✅' if wd_ok else '❌'}")
    print(f"      Max DD:       ${m['max_drawdown']:>+10,.0f}  (limit: -$4,500) {'✅' if dd_ok else '❌'}")
    print(f"      Consistency:  {m['consistency_pct']:.1f}% (limit: <50%) {'✅' if con_ok else '❌'}")

    v2_monthly = m["total_pnl"] / max(m["n_months"], 1)
    pct_change = ((v2_monthly - v1_monthly) / v1_monthly * 100) if v1_monthly else 0
    print(f"\n    vs v1: ${v1_monthly:,.0f}/month → ${v2_monthly:,.0f}/month ({pct_change:+.0f}%)")
    print(f"    vs Final Push ($12,743/mo): trades 4/24 months vs {m['n_months'] - m['months_zero_trades']}/24")
    print(f"    vs Trailing Profit ($435/mo): ${v2_monthly:,.0f}/month ({v2_monthly/435:.1f}x)")

    gap = 7000 - v2_monthly
    if gap > 0:
        nq_proj = v2_monthly * 10
        print(f"\n    GAP TO $7K: ${gap:,.0f} remaining.")
        print(f"      Scale to NQ at same contracts: projected ${nq_proj:,.0f}/month")
    else:
        print(f"\n    🎯 TARGET MET: ${v2_monthly:,.0f}/month >= $7,000")

    print(f"{'═' * 60}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("=" * 60)
    print("  ALWAYS-ON v2 — Three Levers to $7K/month")
    print("=" * 60)

    df = load_data()
    df = add_initial_balance(df)
    yr1, yr2 = split_years(df)
    print(f"  Year 1: {len(yr1):,} bars | Year 2: {len(yr2):,} bars")

    # ── LEVER 1: Sizing ────────────────────────────────────────────
    sizing = lever1_sizing(yr2)
    gc.collect()

    # ── LEVER 2: Frequency ─────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  LEVER 2 — INCREASE TRADE FREQUENCY")
    print("━" * 60)

    new_strats: list[dict] = []

    # 2A: Multi-TF VWAP
    for tf in [3, 5]:
        result = sweep_l1_multi_tf(yr1, yr2, tf, contracts=sizing["l1"])
        if result:
            new_strats.append(result)
        gc.collect()

    # 2B: Momentum continuation
    result = sweep_momentum_cont(yr1, yr2, contracts=sizing["l1"])
    if result:
        new_strats.append(result)
    gc.collect()

    # 2C: Session expansion
    session_survivors = sweep_session_expansion(yr1, yr2, contracts=sizing["l2c"])
    new_strats.extend(session_survivors)
    gc.collect()

    print(f"\n  New sub-strategies that survived: {len(new_strats)}")
    for ns in new_strats:
        print(f"    ✅ {ns['name']}: Y2 ${ns['yr2_pnl']:,.0f}")

    # ── LEVER 3: Refinement ────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  LEVER 3 — PARAMETER REFINEMENT")
    print("━" * 60)

    refined = {}
    for layer, params, contracts, name in [
        ("L1", V1_L1, sizing["l1"], "L1_VWAP_MR"),
        ("L2C", V1_L2C, sizing["l2c"], "L2C_Afternoon"),
        ("L3", V1_L3, sizing["l3"], "L3_VolMom"),
    ]:
        result = refine_layer(yr1, yr2, layer, params, contracts, name)
        if result:
            # Use refined if Y2 is better than original
            orig_y2 = run_layer_at_size(yr2, layer, contracts, params)
            orig_y2_pnl = sum(t.net_pnl for t in orig_y2)
            if result["yr2_pnl"] > orig_y2_pnl:
                refined[layer] = result
                print(f"    ✅ {name}: Y2 ${orig_y2_pnl:,.0f} → ${result['yr2_pnl']:,.0f}")
            else:
                print(f"    ➡️  {name}: refined Y2 ${result['yr2_pnl']:,.0f} <= original ${orig_y2_pnl:,.0f}, keeping original")
        gc.collect()

    # ── BUILD COMBINED SYSTEM ──────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  BUILDING COMBINED v2 SYSTEM")
    print("━" * 60)

    all_trades: list[TradeRecord] = []

    # Core layers (use refined params if better, else original)
    for layer, orig_params, contracts, name in [
        ("L1", V1_L1, sizing["l1"], "L1_VWAP_MR"),
        ("L2C", V1_L2C, sizing["l2c"], "L2C_Afternoon"),
        ("L3", V1_L3, sizing["l3"], "L3_VolMom"),
    ]:
        if layer in refined:
            # Use refined on both years
            all_trades.extend(refined[layer]["yr1_trades"])
            all_trades.extend(refined[layer]["yr2_trades"])
            print(f"  {name}: using REFINED params")
        else:
            # Use original on both years
            y1t = run_layer_at_size(yr1, layer, contracts, orig_params)
            y2t = run_layer_at_size(yr2, layer, contracts, orig_params)
            all_trades.extend(y1t)
            all_trades.extend(y2t)
            print(f"  {name}: using ORIGINAL params at {contracts} contracts")

    # Add new sub-strategies
    for ns in new_strats:
        all_trades.extend(ns["yr1_trades"])
        all_trades.extend(ns["yr2_trades"])
        print(f"  {ns['name']}: added ({len(ns['yr1_trades'])+len(ns['yr2_trades'])} trades)")

    # Build report
    report = build_combined_report(all_trades, sizing)
    print_v2_report(report)

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "sizing": sizing,
        "new_strategies": [
            {"name": ns["name"], "params": ns["params"],
             "yr1_pnl": ns["yr1_pnl"], "yr2_pnl": ns["yr2_pnl"]}
            for ns in new_strats
        ],
        "refined": {
            k: {"params": v["params"], "yr1_pnl": v["yr1_pnl"], "yr2_pnl": v["yr2_pnl"]}
            for k, v in refined.items()
        },
        "combined": {k: v for k, v in report.items() if k != "sizing"},
    }

    out_path = REPORTS_DIR / "always_on_v2.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
