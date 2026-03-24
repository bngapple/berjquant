#!/usr/bin/env python3
"""
Always-On v4 — Honest exits. Fixed stops, no same-bar exits, conservative ordering.

The v3 system was an artifact of favorable intrabar trailing-stop ordering.
v4 keeps the same entry signals but replaces the exit mechanics entirely.

Rules:
  - NO exit on the entry bar
  - FIXED stop and take-profit (set at entry, never move)
  - Conservative ordering: if both SL and TP hit on the same bar, assume SL
  - 2-tick slippage per side, exchange fees included
  - Optional breakeven trail only after trade is profitable by 2×SL

Usage:
    python3 run_always_on_v4.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from run_always_on import (
    load_data, split_years, add_initial_balance,
    layer2c_signals, layer3_signals, TradeRecord,
    TICK_SIZE, POINT_VALUE, COMMISSION_PER_SIDE,
    YR2_START,
)
from run_always_on_v2 import momentum_continuation_signals
from run_always_on_v3 import load_blind_data

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

# ── Honest cost model ────────────────────────────────────────────────
SLIP_TICKS = 2                      # 2 ticks per side (conservative for 15 contracts)
EXCHANGE_FEE_PER_SIDE = 0.27        # CME MNQ

def honest_cost(contracts: int) -> float:
    """Round-trip cost: commission + exchange fees + slippage."""
    comm = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
    slip = SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * contracts
    return comm + slip

# Topstep 150K
DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0

# v3 entry params (unchanged)
P_L2C = {"ema_fast": 8, "ema_slow": 13}
P_L3  = {"range_mult": 0.8, "ema_pullback": 5}
P_L4  = {"z": 1.5, "atr_m": 1.0}


# ═════════════════════════════════════════════════════════════════════
# HONEST BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════

def run_honest(
    df: pl.DataFrame,
    signals: np.ndarray,
    stop_ticks: int,
    target_ticks: int,
    max_hold: int,
    contracts: int,
    layer_name: str = "",
    use_be_trail: bool = False,
) -> list[TradeRecord]:
    """Bar-by-bar backtest with honest execution.

    - Next-bar fills with 2-tick slippage
    - NO exit on entry bar
    - Fixed SL/TP (set at entry, never move)
    - Conservative: if same bar hits SL+TP, assume SL
    - Optional breakeven trail after trade profits by 2×SL
    """
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()
    hhmm = df["hhmm"].to_numpy()
    n = len(df)

    cost = honest_cost(contracts)
    slip = SLIP_TICKS * TICK_SIZE  # 0.50 points per side

    sl_offset = stop_ticks * TICK_SIZE
    tp_offset = target_ticks * TICK_SIZE
    be_activation = 2 * sl_offset  # activate BE trail after 2×SL profit

    trades: list[TradeRecord] = []

    # Position state
    in_pos = False
    direction = 0
    entry_px = 0.0
    entry_bar = 0
    stop_px = 0.0
    target_px = 0.0
    be_trail_active = False
    be_stop_px = 0.0

    pending = 0

    for i in range(n):
        h = hhmm[i]

        # ── EOD flatten at 15:55 (not on entry bar) ──
        if in_pos and h >= 1555 and i > entry_bar:
            exit_px = closes[i] - direction * slip
            raw = (exit_px - entry_px) * direction * POINT_VALUE * contracts
            trades.append(TradeRecord(
                direction, entry_px, exit_px, contracts, raw - cost,
                timestamps[entry_bar], timestamps[i], i - entry_bar,
                "eod_flatten", layer_name))
            in_pos = False
            pending = 0
            continue

        # ── Execute pending entry ──
        if pending != 0 and not in_pos:
            if h >= 1550 or not (930 <= h < 1600):
                pending = 0
            else:
                entry_px = opens[i] + pending * slip
                direction = pending
                entry_bar = i
                stop_px = entry_px - direction * sl_offset
                target_px = entry_px + direction * tp_offset
                in_pos = True
                be_trail_active = False
                be_stop_px = entry_px + direction * (5 * TICK_SIZE)  # BE + 5 ticks buffer
                pending = 0

        # ── Manage open position (SKIP entry bar) ──
        if in_pos and i > entry_bar:
            bars_held = i - entry_bar
            exit_px = None
            reason = ""

            # CHECK STOP FIRST (conservative ordering)
            if direction == 1 and lows[i] <= stop_px:
                exit_px = stop_px
                reason = "stop_loss"
            elif direction == -1 and highs[i] >= stop_px:
                exit_px = stop_px
                reason = "stop_loss"

            # Target (only if stop NOT hit — conservative)
            if exit_px is None:
                if direction == 1 and highs[i] >= target_px:
                    exit_px = target_px
                    reason = "take_profit"
                elif direction == -1 and lows[i] <= target_px:
                    exit_px = target_px
                    reason = "take_profit"

            # Breakeven trail check (only if active and no SL/TP hit)
            if exit_px is None and use_be_trail and be_trail_active:
                if direction == 1 and lows[i] <= be_stop_px:
                    exit_px = be_stop_px
                    reason = "be_trail"
                elif direction == -1 and highs[i] >= be_stop_px:
                    exit_px = be_stop_px
                    reason = "be_trail"

            # Max hold
            if exit_px is None and bars_held >= max_hold:
                exit_px = closes[i]
                reason = "max_hold"

            # Time exit (15:55 handled above, but belt-and-suspenders)
            if exit_px is None and h >= 1555:
                exit_px = closes[i]
                reason = "time_exit"

            if exit_px is not None:
                exit_px -= direction * slip  # adverse slippage
                raw = (exit_px - entry_px) * direction * POINT_VALUE * contracts
                trades.append(TradeRecord(
                    direction, entry_px, exit_px, contracts, raw - cost,
                    timestamps[entry_bar], timestamps[i], bars_held,
                    reason, layer_name))
                in_pos = False
            else:
                # Update breakeven trail activation
                if use_be_trail and not be_trail_active:
                    unrealized = (closes[i] - entry_px) * direction
                    if unrealized >= be_activation:
                        be_trail_active = True
                        # Set BE stop at entry + 5 ticks (lock in small profit)
                        be_stop_px = entry_px + direction * (5 * TICK_SIZE)

                # Ratchet BE trail if active (using close, not high — conservative)
                if use_be_trail and be_trail_active:
                    if direction == 1:
                        new_be = closes[i] - sl_offset * 0.5  # trail at 50% of SL
                        if new_be > be_stop_px:
                            be_stop_px = new_be
                    else:
                        new_be = closes[i] + sl_offset * 0.5
                        if new_be < be_stop_px:
                            be_stop_px = new_be

        # ── New signal (only when flat) ──
        if not in_pos and i < len(signals) and signals[i] != 0:
            if 930 <= h < 1550:
                pending = signals[i]

    # Close open position at end of data
    if in_pos:
        exit_px = closes[-1] - direction * slip
        raw = (exit_px - entry_px) * direction * POINT_VALUE * contracts
        trades.append(TradeRecord(
            direction, entry_px, exit_px, contracts, raw - cost,
            timestamps[entry_bar], timestamps[-1], n - 1 - entry_bar,
            "end_of_data", layer_name))

    return trades


# ═════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════

def metrics(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "avg": 0, "monthly_avg": 0,
                "worst_month": 0, "worst_day": 0, "max_dd": 0,
                "monthly": {}, "daily": {}, "bars_mean": 0, "bars_min": 0,
                "sharpe": 0, "n_months": 0, "months_pos": 0}

    pnls = [t.net_pnl for t in trades]
    bars = [t.bars_held for t in trades]
    total = sum(pnls)
    w = sum(1 for p in pnls if p > 0)

    monthly = defaultdict(float)
    monthly_tc = defaultdict(int)
    daily = defaultdict(float)
    for t in trades:
        m = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:7]
        d = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        monthly[m] += t.net_pnl
        monthly_tc[m] += 1
        daily[d] += t.net_pnl

    n_m = max(len(monthly), 1)
    n_d = max(len(daily), 1)
    worst_m = min(monthly.values()) if monthly else 0
    worst_d = min(daily.values()) if daily else 0
    best_d = max(daily.values()) if daily else 0

    cum = 0.0; peak = 0.0; mdd = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)

    daily_vals = list(daily.values())
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0

    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "avg": np.mean(pnls), "monthly_avg": total / n_m,
        "worst_month": worst_m, "worst_day": worst_d, "best_day": best_d,
        "max_dd": mdd, "monthly": dict(monthly), "monthly_tc": dict(monthly_tc),
        "daily": dict(daily), "bars_mean": np.mean(bars), "bars_min": min(bars),
        "bars_max": max(bars), "sharpe": sharpe, "n_months": n_m,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "same_bar_exits": sum(1 for t in trades if t.bars_held == 0),
    }


# ═════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP
# ═════════════════════════════════════════════════════════════════════

def sweep_layer(yr1, layer_name, sig_fn, sl_list, tp_list, hold_list, contract_list, use_be_list=[False, True]):
    """Grid search on Year 1, return best by Sharpe."""
    print(f"  Sweeping {layer_name} ...")
    signals, _ = sig_fn(yr1)

    total = len(sl_list) * len(tp_list) * len(hold_list) * len(contract_list) * len(use_be_list)
    print(f"    {total} combinations ...")

    best = None
    count = 0
    for sl in sl_list:
        for tp in tp_list:
            if tp <= sl:  # TP must be bigger than SL for R:R > 1
                continue
            for hold in hold_list:
                for c in contract_list:
                    for be in use_be_list:
                        trades = run_honest(yr1, signals, sl, tp, hold, c, layer_name, be)
                        m = metrics(trades)
                        if m["n"] < 20:  # too few trades
                            continue
                        # Score by Sharpe (rewards consistency, not just total P&L)
                        score = m["sharpe"]
                        if best is None or score > best["score"]:
                            best = {
                                "score": score, "sl": sl, "tp": tp, "hold": hold,
                                "contracts": c, "use_be": be, "metrics": m,
                                "trades": trades,
                            }
                        count += 1
                        if count % 200 == 0:
                            print(f"      {count}/{total} ...", end="\r")

    if best:
        print(f"    Done. Best: Sharpe={best['score']:.2f}, ${best['metrics']['pnl']:,.0f}")
    else:
        print(f"    Done. No profitable config.")
    return best


# ═════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═════════════════════════════════════════════════════════════════════

def run_mc(trades, n_sims=5000, pnl_mult=1.0, label="Baseline"):
    daily_buckets = defaultdict(list)
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        daily_buckets[d].append(t.net_pnl * pnl_mult)

    days = list(daily_buckets.keys())
    day_pnls = [daily_buckets[d] for d in days]
    nd = len(days)
    if nd == 0:
        return {"label": label, "pass_rate": 0, "blowup": 1}

    passed = 0; blown = 0
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0.0; peak = 0.0; ok = True; p = False
        for idx in order:
            dp = sum(day_pnls[idx])
            if dp < DAILY_LIMIT:
                dp = DAILY_LIMIT
            cum += dp
            peak = max(peak, cum)
            if cum - peak <= MLL:
                ok = False; break
            if cum >= EVAL_TARGET:
                p = True
        if not ok: blown += 1
        if p and ok: passed += 1

    return {"label": label, "pass_rate": passed/n_sims, "blowup": blown/n_sims}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("=" * 70)
    print("  ALWAYS-ON v4 — Honest Exits")
    print(f"  Cost per trade at 15 MNQ: ${honest_cost(15):.2f}")
    print("=" * 70)

    df = load_data(); df = add_initial_balance(df)
    yr1, yr2 = split_years(df)
    blind = load_blind_data()
    print(f"  Y1: {len(yr1):,} bars | Y2: {len(yr2):,} bars | Blind: {len(blind):,} bars")

    survivors = []

    # ── L2C: Afternoon Trend ──────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  L2C — Afternoon Trend")
    print("━" * 70)
    best = sweep_layer(
        yr1, "L2C",
        lambda d: layer2c_signals(d, P_L2C["ema_fast"], P_L2C["ema_slow"]),
        sl_list=[20, 30, 40, 50, 60, 80],
        tp_list=[30, 50, 80, 100, 120, 160],
        hold_list=[30, 60, 90, 120],
        contract_list=[5, 10, 15],
    )
    if best and best["metrics"]["pnl"] > 0:
        params = {k: best[k] for k in ["sl", "tp", "hold", "contracts", "use_be"]}
        # Validate on Y2
        s2, _ = layer2c_signals(yr2, P_L2C["ema_fast"], P_L2C["ema_slow"])
        y2t = run_honest(yr2, s2, stop_ticks=params["sl"], target_ticks=params["tp"],
                         max_hold=params["hold"], contracts=params["contracts"],
                         layer_name="L2C", use_be_trail=params["use_be"])
        y2m = metrics(y2t)
        # Blind
        sb, _ = layer2c_signals(blind, P_L2C["ema_fast"], P_L2C["ema_slow"])
        bt = run_honest(blind, sb, stop_ticks=params["sl"], target_ticks=params["tp"],
                        max_hold=params["hold"], contracts=params["contracts"],
                        layer_name="L2C", use_be_trail=params["use_be"])
        bm = metrics(bt)

        passed = y2m["pnl"] > 0
        print(f"    Params: SL={params['sl']}t TP={params['tp']}t hold={params['hold']} "
              f"contracts={params['contracts']} BE={params['use_be']}")
        print(f"    Y1: ${best['metrics']['monthly_avg']:,.0f}/mo, {best['metrics']['n']} trades, "
              f"{best['metrics']['wr']:.1f}% WR, {best['metrics']['bars_mean']:.1f} avg bars")
        print(f"    Y2: ${y2m['monthly_avg']:,.0f}/mo, {y2m['n']} trades, "
              f"{y2m['wr']:.1f}% WR, {y2m['bars_mean']:.1f} avg bars {'✅' if passed else '❌'}")
        print(f"    Blind: ${bm['monthly_avg']:,.0f}/mo, {bm['n']} trades, {bm['wr']:.1f}% WR")
        print(f"    Same-bar exits: {y2m['same_bar_exits']} (must be 0) {'✅' if y2m['same_bar_exits']==0 else '❌'}")
        if passed:
            survivors.append({"name": "L2C", "params": params,
                              "y1": best["metrics"], "y2": y2m, "blind": bm,
                              "y1_trades": best["trades"], "y2_trades": y2t, "blind_trades": bt})
    else:
        print("    No profitable config on Y1.")
    gc.collect()

    # ── L3: Volatility Momentum ───────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  L3 — Volatility Momentum")
    print("━" * 70)
    best = sweep_layer(
        yr1, "L3",
        lambda d: layer3_signals(d, P_L3["range_mult"], P_L3["ema_pullback"]),
        sl_list=[30, 50, 80, 100, 120],
        tp_list=[50, 100, 150, 200, 300],
        hold_list=[60, 120, 180, 240],
        contract_list=[5, 10, 15],
    )
    if best and best["metrics"]["pnl"] > 0:
        params = {k: best[k] for k in ["sl", "tp", "hold", "contracts", "use_be"]}
        s2, _ = layer3_signals(yr2, P_L3["range_mult"], P_L3["ema_pullback"])
        y2t = run_honest(yr2, s2, stop_ticks=params["sl"], target_ticks=params["tp"],
                         max_hold=params["hold"], contracts=params["contracts"],
                         layer_name="L3", use_be_trail=params["use_be"])
        y2m = metrics(y2t)
        sb, _ = layer3_signals(blind, P_L3["range_mult"], P_L3["ema_pullback"])
        bt = run_honest(blind, sb, stop_ticks=params["sl"], target_ticks=params["tp"],
                        max_hold=params["hold"], contracts=params["contracts"],
                        layer_name="L3", use_be_trail=params["use_be"])
        bm = metrics(bt)

        passed = y2m["pnl"] > 0
        print(f"    Params: SL={params['sl']}t TP={params['tp']}t hold={params['hold']} "
              f"contracts={params['contracts']} BE={params['use_be']}")
        print(f"    Y1: ${best['metrics']['monthly_avg']:,.0f}/mo, {best['metrics']['n']} trades, "
              f"{best['metrics']['wr']:.1f}% WR, {best['metrics']['bars_mean']:.1f} avg bars")
        print(f"    Y2: ${y2m['monthly_avg']:,.0f}/mo, {y2m['n']} trades, "
              f"{y2m['wr']:.1f}% WR, {y2m['bars_mean']:.1f} avg bars {'✅' if passed else '❌'}")
        print(f"    Blind: ${bm['monthly_avg']:,.0f}/mo, {bm['n']} trades, {bm['wr']:.1f}% WR")
        print(f"    Same-bar exits: {y2m['same_bar_exits']} {'✅' if y2m['same_bar_exits']==0 else '❌'}")
        if passed:
            survivors.append({"name": "L3", "params": params,
                              "y1": best["metrics"], "y2": y2m, "blind": bm,
                              "y1_trades": best["trades"], "y2_trades": y2t, "blind_trades": bt})
    else:
        print("    No profitable config on Y1.")
    gc.collect()

    # ── L4: Momentum Continuation ─────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  L4 — Momentum Continuation")
    print("━" * 70)
    best = sweep_layer(
        yr1, "L4",
        lambda d: momentum_continuation_signals(d, z_threshold=P_L4["z"], atr_mult=P_L4["atr_m"]),
        sl_list=[30, 50, 80, 100, 120],
        tp_list=[50, 100, 150, 200, 300],
        hold_list=[60, 120, 180, 240],
        contract_list=[5, 10, 15],
    )
    if best and best["metrics"]["pnl"] > 0:
        params = {k: best[k] for k in ["sl", "tp", "hold", "contracts", "use_be"]}
        s2, _ = momentum_continuation_signals(yr2, z_threshold=P_L4["z"], atr_mult=P_L4["atr_m"])
        y2t = run_honest(yr2, s2, stop_ticks=params["sl"], target_ticks=params["tp"],
                         max_hold=params["hold"], contracts=params["contracts"],
                         layer_name="L4", use_be_trail=params["use_be"])
        y2m = metrics(y2t)
        sb, _ = momentum_continuation_signals(blind, z_threshold=P_L4["z"], atr_mult=P_L4["atr_m"])
        bt = run_honest(blind, sb, stop_ticks=params["sl"], target_ticks=params["tp"],
                        max_hold=params["hold"], contracts=params["contracts"],
                        layer_name="L4", use_be_trail=params["use_be"])
        bm = metrics(bt)

        passed = y2m["pnl"] > 0
        print(f"    Params: SL={params['sl']}t TP={params['tp']}t hold={params['hold']} "
              f"contracts={params['contracts']} BE={params['use_be']}")
        print(f"    Y1: ${best['metrics']['monthly_avg']:,.0f}/mo, {best['metrics']['n']} trades, "
              f"{best['metrics']['wr']:.1f}% WR, {best['metrics']['bars_mean']:.1f} avg bars")
        print(f"    Y2: ${y2m['monthly_avg']:,.0f}/mo, {y2m['n']} trades, "
              f"{y2m['wr']:.1f}% WR, {y2m['bars_mean']:.1f} avg bars {'✅' if passed else '❌'}")
        print(f"    Blind: ${bm['monthly_avg']:,.0f}/mo, {bm['n']} trades, {bm['wr']:.1f}% WR")
        print(f"    Same-bar exits: {y2m['same_bar_exits']} {'✅' if y2m['same_bar_exits']==0 else '❌'}")
        if passed:
            survivors.append({"name": "L4", "params": params,
                              "y1": best["metrics"], "y2": y2m, "blind": bm,
                              "y1_trades": best["trades"], "y2_trades": y2t, "blind_trades": bt})
    else:
        print("    No profitable config on Y1.")
    gc.collect()

    # ═════════════════════════════════════════════════════════════
    # COMBINED SYSTEM
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  COMBINED v4 SYSTEM")
    print("═" * 70)

    if not survivors:
        print("\n  ❌ No layers survived. The entry signals do not produce a tradeable")
        print("  edge with realistic execution assumptions.")
        report = {"timestamp": datetime.now().isoformat(), "survivors": [],
                  "verdict": "NO TRADEABLE EDGE"}
        json.dump(report, open(REPORTS_DIR / "always_on_v4.json", "w"), indent=2, default=str)
        print(f"\n  Saved to reports/always_on_v4.json")
        print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")
        return

    # Combine all surviving trades
    all_y1 = []; all_y2 = []; all_bl = []
    for s in survivors:
        all_y1.extend(s["y1_trades"])
        all_y2.extend(s["y2_trades"])
        all_bl.extend(s["blind_trades"])

    m_y1 = metrics(all_y1)
    m_y2 = metrics(all_y2)
    m_bl = metrics(all_bl)

    print(f"\n  Surviving layers: {len(survivors)}")
    for s in survivors:
        print(f"    ✅ {s['name']}: SL={s['params']['sl']}t TP={s['params']['tp']}t "
              f"hold={s['params']['hold']} c={s['params']['contracts']} BE={s['params']['use_be']}")

    # Verify no same-bar exits
    print(f"\n  EXECUTION QUALITY:")
    print(f"    Y2 same-bar exits: {m_y2['same_bar_exits']} {'✅' if m_y2['same_bar_exits']==0 else '❌'}")
    print(f"    Y2 avg bars held: {m_y2['bars_mean']:.1f} (min={m_y2.get('bars_min', '?')})")
    print(f"    Blind avg bars held: {m_bl['bars_mean']:.1f}")

    # Monthly breakdown
    all_monthly = defaultdict(float)
    all_monthly_tc = defaultdict(int)
    for src in [m_y1, m_y2, m_bl]:
        for m, v in src["monthly"].items():
            all_monthly[m] += v
        for m, v in src.get("monthly_tc", {}).items():
            all_monthly_tc[m] += v

    print(f"\n  MONTHLY BREAKDOWN (all {len(all_monthly)} months):")
    for m in sorted(all_monthly.keys()):
        v = all_monthly[m]
        tc = all_monthly_tc.get(m, 0)
        print(f"    {m}: ${v:>+10,.0f} ({tc:>3} trades) {'✅' if v > 0 else '❌'}")

    total_months = len(all_monthly)
    months_pos = sum(1 for v in all_monthly.values() if v > 0)
    worst_m = min(all_monthly.values()) if all_monthly else 0
    best_m = max(all_monthly.values()) if all_monthly else 0
    avg_m = sum(all_monthly.values()) / max(total_months, 1)

    print(f"\n  Months profitable: {months_pos}/{total_months}")
    print(f"  Worst month: ${worst_m:,.0f}")
    print(f"  Best month: ${best_m:,.0f}")
    print(f"  Average: ${avg_m:,.0f}")

    # Prop firm
    all_daily = defaultdict(float)
    for src in [m_y2, m_bl]:
        for d, v in src["daily"].items():
            all_daily[d] += v
    worst_d = min(all_daily.values()) if all_daily else 0
    best_d = max(all_daily.values()) if all_daily else 0
    cum = 0; peak = 0; mdd = 0
    for d in sorted(all_daily.keys()):
        cum += all_daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    consistency = (best_d / sum(all_daily.values()) * 100) if sum(all_daily.values()) > 0 else 100

    print(f"\n  PROP FIRM (Topstep 150K):")
    print(f"    Worst day:   ${worst_d:>+10,.0f} (limit: -$3,000) {'✅' if worst_d > -3000 else '❌'}")
    print(f"    Max DD:      ${mdd:>+10,.0f} (limit: -$4,500) {'✅' if mdd > -4500 else '❌'}")
    print(f"    Consistency: {consistency:.1f}% (limit: <50%) {'✅' if consistency < 50 else '❌'}")

    # Monte Carlo on Y2 trades
    print(f"\n  MONTE CARLO:")
    mc_base = run_mc(all_y2, 5000, 1.0, "Baseline")
    mc_cons = run_mc(all_y2, 5000, 0.70, "Conservative 70%")
    print(f"    Baseline:     {mc_base['pass_rate']:.0%} pass, {mc_base['blowup']:.0%} blow-up")
    print(f"    Conservative: {mc_cons['pass_rate']:.0%} pass, {mc_cons['blowup']:.0%} blow-up")

    # Comparison table
    print(f"\n  COMPARISON:")
    print(f"  ┌{'─'*18}┬{'─'*14}┬{'─'*14}┬{'─'*14}┐")
    print(f"  │{'System':<18}│{'v3 (fake)':>14}│{'v3 conserv.':>14}│{'v4 (honest)':>14}│")
    print(f"  ├{'─'*18}┼{'─'*14}┼{'─'*14}┼{'─'*14}┤")
    print(f"  │{'Y2 $/month':<18}│{'$+11,209':>14}│{'$-3,196':>14}│${m_y2['monthly_avg']:>+13,.0f}│")
    print(f"  │{'Blind $/month':<18}│{'$+10,195':>14}│{'$-3,524':>14}│${m_bl['monthly_avg']:>+13,.0f}│")
    print(f"  │{'Avg bars held':<18}│{'0':>14}│{'0':>14}│{m_y2['bars_mean']:>14.1f}│")
    print(f"  │{'Months profit':<18}│{'41/41':>14}│{'0/41':>14}│{months_pos:>7}/{total_months:<6}│")
    print(f"  │{'MC pass rate':<18}│{'100%':>14}│{'0%':>14}│{mc_base['pass_rate']:>13.0%}│")
    print(f"  └{'─'*18}┴{'─'*14}┴{'─'*14}┴{'─'*14}┘")

    if m_y2["monthly_avg"] > 0 and m_bl["monthly_avg"] > 0:
        print(f"\n  ✅ v4 is profitable with honest execution on BOTH Y2 and blind data.")
        verdict = "REAL EDGE — READY FOR DEMO"
    elif m_y2["monthly_avg"] > 0:
        print(f"\n  ⚠️  v4 profitable on Y2 but not blind data. Edge may be regime-specific.")
        verdict = "MARGINAL — NEEDS MORE VALIDATION"
    else:
        print(f"\n  ❌ The Always-On entry signals do not produce a tradeable edge")
        print(f"     with realistic execution assumptions.")
        verdict = "NO TRADEABLE EDGE"

    print(f"\n  VERDICT: {verdict}")
    print("═" * 70)

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "cost_per_trade_15c": honest_cost(15),
        "survivors": [
            {"name": s["name"], "params": s["params"],
             "y1_monthly": s["y1"]["monthly_avg"], "y1_trades": s["y1"]["n"],
             "y2_monthly": s["y2"]["monthly_avg"], "y2_trades": s["y2"]["n"],
             "blind_monthly": s["blind"]["monthly_avg"], "blind_trades": s["blind"]["n"]}
            for s in survivors
        ],
        "combined_y2": {k: v for k, v in m_y2.items() if k not in ("daily", "monthly_tc")},
        "combined_blind": {k: v for k, v in m_bl.items() if k not in ("daily", "monthly_tc")},
        "all_monthly": dict(all_monthly),
        "months_profitable": months_pos,
        "total_months": total_months,
        "mc_baseline": mc_base,
        "mc_conservative": mc_cons,
        "verdict": verdict,
    }
    out = REPORTS_DIR / "always_on_v4.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
