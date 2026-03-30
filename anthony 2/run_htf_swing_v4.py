#!/usr/bin/env python3
"""
HTF Swing v4 — Adaptive sizing. Same RSI+IB+MOM strategies from v3.
Conditions score determines contract size per trade.

Usage:
    python3 run_htf_swing_v4.py
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
    load_and_resample, extract_arrays, rt_cost,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    calc_atr, calc_ema,
    TICK_SIZE, POINT_VALUE, SLIP_PTS, YR1_END,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0

STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}


# ═════════════════════════════════════════════════════════════════════
# CONDITIONS SCORE
# ═════════════════════════════════════════════════════════════════════

def compute_conditions(df_15m, df_1h):
    """Pre-compute conditions arrays for every 15m bar."""
    closes = df_15m["close"].to_numpy()
    highs = df_15m["high"].to_numpy()
    lows = df_15m["low"].to_numpy()
    hhmm = df_15m["hhmm"].to_numpy()
    n = len(df_15m)

    # ATR for vol regime
    atr14 = calc_atr(highs, lows, closes, 14)
    atr_avg20 = np.full(n, np.nan)
    for i in range(20, n):
        window = atr14[i-20:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_avg20[i] = np.mean(valid)

    # EMAs on 15m
    ema8_15 = calc_ema(closes, 8)
    ema21_15 = calc_ema(closes, 21)

    # EMAs on 1h — map back to 15m bars by date+hour
    c_1h = df_1h["close"].to_numpy()
    ema8_1h = calc_ema(c_1h, 8)
    ema21_1h = calc_ema(c_1h, 21)

    # Build 1h trend lookup: (date, hour) → trend direction
    ts_1h = df_1h["timestamp"].to_list()
    trend_1h = {}
    for j in range(len(df_1h)):
        if np.isnan(ema8_1h[j]) or np.isnan(ema21_1h[j]):
            continue
        t = ts_1h[j]
        key = (str(t)[:10], int(str(t)[11:13]))
        trend_1h[key] = 1 if ema8_1h[j] > ema21_1h[j] else -1

    ts_15m = df_15m["timestamp"].to_list()

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = 0.0

        # 1. Volatility regime (40%)
        if not np.isnan(atr14[i]) and not np.isnan(atr_avg20[i]) and atr_avg20[i] > 0:
            ratio = atr14[i] / atr_avg20[i]
            if ratio < 0.7:
                s += 40
            elif ratio <= 1.3:
                s += 25
            elif ratio <= 2.0:
                s += 10
            # else: 0 (extreme)
        else:
            s += 20  # unknown, default mid

        # 2. Trend clarity (30%)
        if not np.isnan(ema8_15[i]) and not np.isnan(ema21_15[i]):
            trend_15m = 1 if ema8_15[i] > ema21_15[i] else -1
            t = ts_15m[i]
            key = (str(t)[:10], int(str(t)[11:13]))
            t_1h = trend_1h.get(key, 0)
            if t_1h != 0 and trend_15m == t_1h:
                s += 30
            elif t_1h != 0:
                s += 0  # conflicting
            else:
                s += 15
        else:
            s += 10

        # 3. Time of day (15%)
        h = hhmm[i]
        if 945 <= h <= 1130:
            s += 15
        elif 1330 <= h <= 1530:
            s += 12
        elif 1130 < h < 1330:
            s += 5
        elif 930 <= h < 945:
            s += 0
        elif h > 1530:
            s += 3

        # 4. Recent performance (15%) — filled in during backtest, default 10
        s += 10

        scores[i] = min(s, 100)

    return scores, atr14


def map_contracts_a(score):
    """Scheme A: 1-5 contracts."""
    if score <= 20: return 1
    if score <= 40: return 2
    if score <= 60: return 3
    if score <= 80: return 4
    return 5

def map_contracts_b(score):
    """Scheme B: 2-4 contracts."""
    if score <= 30: return 2
    if score <= 60: return 3
    return 4


# ═════════════════════════════════════════════════════════════════════
# ADAPTIVE BACKTEST
# ═════════════════════════════════════════════════════════════════════

from run_htf_swing import Trade

def adaptive_backtest(
    df_15m, signals, sl_ticks, tp_ticks, max_hold,
    scores, sizing_fn, strategy_name,
    atr_for_dynamic_sl=None,  # if provided, use ATR-based SL/TP
):
    """Backtest with adaptive position sizing based on conditions score."""
    opens = df_15m["open"].to_numpy()
    highs = df_15m["high"].to_numpy()
    lows = df_15m["low"].to_numpy()
    closes = df_15m["close"].to_numpy()
    timestamps = df_15m["timestamp"].to_list()
    hhmm = df_15m["hhmm"].to_numpy()
    dates = [str(t)[:10] for t in timestamps]
    n = len(opens)

    trades = []
    in_pos = False
    direction = 0
    entry_px = 0.0
    entry_bar = 0
    stop_px = 0.0
    target_px = 0.0
    pos_contracts = 0
    pending = 0
    pending_score = 0

    # Daily P&L tracking for performance component
    daily_pnl = 0.0
    current_date = ""
    # Equity tracking for DD override
    equity = 0.0
    peak_equity = 0.0

    for i in range(n):
        h = hhmm[i]
        d = dates[i]

        # Daily reset
        if d != current_date:
            daily_pnl = 0.0
            current_date = d

        # Flatten at 15:45
        if in_pos and h >= 1545 and i > entry_bar:
            ex = closes[i] - direction * SLIP_PTS
            cost = rt_cost(pos_contracts)
            raw = (ex - entry_px) * direction * POINT_VALUE * pos_contracts
            pnl = raw - cost
            trades.append(Trade(direction, entry_px, ex, pos_contracts, pnl,
                                timestamps[entry_bar], timestamps[i], i - entry_bar,
                                "time_exit", strategy_name))
            daily_pnl += pnl; equity += pnl; peak_equity = max(peak_equity, equity)
            in_pos = False; pending = 0
            continue

        # Execute pending
        if pending != 0 and not in_pos:
            if h >= 1530 or not (930 <= h < 1600):
                pending = 0
            else:
                # Determine contracts
                score = pending_score

                # Adjust score for recent performance (replace the default 10)
                perf_adj = 0
                if daily_pnl > 50: perf_adj = 15
                elif daily_pnl >= -50: perf_adj = 10
                elif daily_pnl >= -500: perf_adj = 5
                else: perf_adj = 0
                score = score - 10 + perf_adj  # replace default 10 with actual
                score = max(0, min(100, score))

                c = sizing_fn(score)

                # Safety overrides
                if daily_pnl < -1500:
                    c = 1
                dd = equity - peak_equity
                if dd < -3000:
                    c = 1
                c = max(1, min(c, 5))  # hard bounds

                # Dynamic SL/TP if ATR provided
                if atr_for_dynamic_sl is not None and not np.isnan(atr_for_dynamic_sl[i]):
                    atr_val = atr_for_dynamic_sl[i]
                    dyn_sl = max(int(1.5 * atr_val / TICK_SIZE), 8)
                    dyn_tp = max(int(10 * atr_val / TICK_SIZE), dyn_sl + 4)
                    actual_sl = dyn_sl
                    actual_tp = dyn_tp
                else:
                    actual_sl = sl_ticks
                    actual_tp = tp_ticks

                entry_px = opens[i] + int(pending) * SLIP_PTS
                direction = int(pending)
                entry_bar = i
                pos_contracts = c
                stop_px = entry_px - direction * actual_sl * TICK_SIZE
                target_px = entry_px + direction * actual_tp * TICK_SIZE
                in_pos = True
                pending = 0

        # Manage (skip entry bar)
        if in_pos and i > entry_bar:
            bh = i - entry_bar
            ex = None; reason = ""

            if direction == 1 and lows[i] <= stop_px:
                ex = stop_px; reason = "stop_loss"
            elif direction == -1 and highs[i] >= stop_px:
                ex = stop_px; reason = "stop_loss"

            if ex is None:
                if direction == 1 and highs[i] >= target_px:
                    ex = target_px; reason = "take_profit"
                elif direction == -1 and lows[i] <= target_px:
                    ex = target_px; reason = "take_profit"

            if ex is None and bh >= max_hold:
                ex = closes[i]; reason = "max_hold"

            if ex is not None:
                ex -= direction * SLIP_PTS
                cost = rt_cost(pos_contracts)
                raw = (ex - entry_px) * direction * POINT_VALUE * pos_contracts
                pnl = raw - cost
                trades.append(Trade(direction, entry_px, ex, pos_contracts, pnl,
                                    timestamps[entry_bar], timestamps[i], bh,
                                    reason, strategy_name))
                daily_pnl += pnl; equity += pnl; peak_equity = max(peak_equity, equity)
                in_pos = False

        # New signal
        if not in_pos and i < len(signals) and signals[i] != 0:
            if 930 <= h < 1530:
                pending = signals[i]
                pending_score = scores[i]

    return trades


# ═════════════════════════════════════════════════════════════════════
# METRICS & MC (reuse from v3 pattern)
# ═════════════════════════════════════════════════════════════════════

def metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "worst_day": 0, "max_dd": 0, "monthly": {}, "n_months": 0,
                "months_pos": 0, "avg_c": 0, "trades_per_day": 0, "sharpe": 0}
    pnls = [t.net_pnl for t in trades]
    cts = [t.contracts for t in trades]
    monthly = defaultdict(float)
    daily = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
        daily[str(t.entry_time)[:10]] += t.net_pnl
    cum = 0; peak = 0; mdd = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    nm = max(len(monthly), 1); nd = max(len(daily), 1)
    dv = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0
    return {
        "pnl": sum(pnls), "n": len(trades), "wr": sum(1 for p in pnls if p > 0) / len(trades) * 100,
        "monthly_avg": sum(pnls) / nm, "worst_month": min(monthly.values()) if monthly else 0,
        "worst_day": min(dv) if dv else 0, "max_dd": mdd,
        "monthly": dict(monthly), "n_months": nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "avg_c": np.mean(cts), "trades_per_day": len(trades) / nd, "sharpe": sharpe,
    }


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0: return {"pass_rate": 0, "blowup": 1}
    p = 0; b = 0
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; ok = True; passed = False
        for idx in order:
            dp = sum(days[idx])
            if dp < DAILY_LIMIT: dp = DAILY_LIMIT
            cum += dp; peak = max(peak, cum)
            if cum - peak <= MLL: ok = False; break
            if cum >= EVAL_TARGET: passed = True
        if not ok: b += 1
        if passed and ok: p += 1
    return {"pass_rate": p / n_sims, "blowup": b / n_sims}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v4 — Adaptive Sizing")
    print("═" * 70)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind")

    yr1_15 = main_data["15m"].filter(pl.col("timestamp") < YR1_END)
    yr2_15 = main_data["15m"].filter(pl.col("timestamp") >= YR1_END)
    bl_15 = blind_data["15m"]

    yr1_1h = main_data["1h"].filter(pl.col("timestamp") < YR1_END)
    yr2_1h = main_data["1h"].filter(pl.col("timestamp") >= YR1_END)
    bl_1h = blind_data["1h"]

    # Pre-compute conditions scores
    print("\n  Computing conditions scores ...")
    scores_y1, atr_y1 = compute_conditions(yr1_15, yr1_1h)
    scores_y2, atr_y2 = compute_conditions(yr2_15, yr2_1h)
    scores_bl, atr_bl = compute_conditions(bl_15, bl_1h)

    # Score distribution
    for label, sc in [("Y2", scores_y2), ("Blind", scores_bl)]:
        print(f"  {label} score distribution:")
        for lo, hi, lbl in [(0, 20, "0-20 danger"), (21, 40, "21-40 caution"),
                             (41, 60, "41-60 normal"), (61, 80, "61-80 good"), (81, 100, "81-100 ideal")]:
            pct = np.mean((sc >= lo) & (sc <= hi)) * 100
            print(f"    {lbl}: {pct:.1f}%")
        print(f"    Average: {np.mean(sc):.1f}")

    # ── Run all schemes ──────────────────────────────────────
    schemes = {
        "Scheme A (1-5c)": map_contracts_a,
        "Scheme B (2-4c)": map_contracts_b,
    }

    all_results = {}

    for scheme_name, sizing_fn in schemes.items():
        print(f"\n{'━' * 70}")
        print(f"  {scheme_name}")
        print("━" * 70)

        for period, df, sc, atr_arr in [
            ("Y1", yr1_15, scores_y1, atr_y1),
            ("Y2", yr2_15, scores_y2, atr_y2),
            ("Blind", bl_15, scores_bl, atr_bl),
        ]:
            all_trades = []
            for name, s in STRATS.items():
                sigs = s["sig"](df)
                trades = adaptive_backtest(df, sigs, s["sl"], s["tp"], s["hold"],
                                           sc, sizing_fn, name)
                all_trades.extend(trades)
            m = metrics(all_trades)
            key = (scheme_name, period)
            all_results[key] = {"metrics": m, "trades": all_trades}
            print(f"  {period}: ${m['monthly_avg']:,.0f}/mo, avg {m['avg_c']:.1f}c, "
                  f"DD=${m['max_dd']:,.0f}, {m['months_pos']}/{m['n_months']} mo+")
        gc.collect()

    # ATR-based stops
    print(f"\n{'━' * 70}")
    print("  ATR Dynamic Stops (Scheme B sizing)")
    print("━" * 70)
    for period, df, sc, atr_arr in [
        ("Y1", yr1_15, scores_y1, atr_y1),
        ("Y2", yr2_15, scores_y2, atr_y2),
        ("Blind", bl_15, scores_bl, atr_bl),
    ]:
        all_trades = []
        for name, s in STRATS.items():
            sigs = s["sig"](df)
            trades = adaptive_backtest(df, sigs, s["sl"], s["tp"], s["hold"],
                                       sc, map_contracts_b, name,
                                       atr_for_dynamic_sl=atr_arr)
            all_trades.extend(trades)
        m = metrics(all_trades)
        all_results[("ATR stops", period)] = {"metrics": m, "trades": all_trades}
        print(f"  {period}: ${m['monthly_avg']:,.0f}/mo, avg {m['avg_c']:.1f}c, DD=${m['max_dd']:,.0f}")
    gc.collect()

    # ── MC for all schemes ──
    print(f"\n{'━' * 70}")
    print("  Monte Carlo")
    print("━" * 70)
    mc_results = {}
    for scheme in ["Scheme A (1-5c)", "Scheme B (2-4c)", "ATR stops"]:
        y2t = all_results[(scheme, "Y2")]["trades"]
        mc_b = run_mc(y2t, 5000, 1.0)
        mc_c = run_mc(y2t, 5000, 0.70)
        mc_results[scheme] = {"baseline": mc_b, "conservative": mc_c}
        print(f"  {scheme}: {mc_b['pass_rate']:.0%} pass / {mc_c['pass_rate']:.0%} @70%")

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  HTF SWING v4 — RESULTS")
    print("═" * 70)

    # Comparison table
    rows = [
        ("v3 conserv. (2c)", 2.0, 4760, 2161, -2908, 1.0, "✅"),
        ("v3 aggress. (3c)", 3.0, 7140, 3242, -4362, 1.0, "✅*"),
    ]
    for scheme in ["Scheme A (1-5c)", "Scheme B (2-4c)", "ATR stops"]:
        y2m = all_results[(scheme, "Y2")]["metrics"]
        blm = all_results[(scheme, "Blind")]["metrics"]
        dd = min(y2m["max_dd"], blm["max_dd"])
        surv = "✅" if dd > -4500 else "❌"
        mc_p = mc_results[scheme]["baseline"]["pass_rate"]
        rows.append((scheme, y2m["avg_c"], y2m["monthly_avg"], blm["monthly_avg"], dd, mc_p, surv))

    print(f"\n  ┌{'─'*23}┬{'─'*7}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*8}┬{'─'*8}┐")
    print(f"  │{'System':<23}│{'AvgC':>7}│{'Y2 $/mo':>10}│{'Blind':>10}│{'Max DD':>10}│{'MC':>8}│{'Surv':>8}│")
    print(f"  ├{'─'*23}┼{'─'*7}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*8}┼{'─'*8}┤")
    for name, ac, y2, bl, dd, mc_p, surv in rows:
        mc_str = f"{mc_p:.0%}" if isinstance(mc_p, float) else str(mc_p)
        print(f"  │{name:<23}│{ac:>7.1f}│${y2:>9,.0f}│${bl:>9,.0f}│${dd:>9,.0f}│{mc_str:>8}│{surv:>8}│")
    print(f"  └{'─'*23}┴{'─'*7}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*8}┴{'─'*8}┘")

    # Best scheme analysis
    best_scheme = max(
        ["Scheme A (1-5c)", "Scheme B (2-4c)", "ATR stops"],
        key=lambda s: all_results[(s, "Y2")]["metrics"]["sharpe"]
    )
    best_y2 = all_results[(best_scheme, "Y2")]["metrics"]
    best_bl = all_results[(best_scheme, "Blind")]["metrics"]

    print(f"\n  BEST SCHEME: {best_scheme}")
    print(f"    Y2 Sharpe: {best_y2['sharpe']:.2f}")

    # Monthly breakdown for best
    all_mo = defaultdict(float)
    for period in ["Y1", "Y2", "Blind"]:
        for m, v in all_results[(best_scheme, period)]["metrics"]["monthly"].items():
            all_mo[m] += v

    print(f"\n  MONTHLY BREAKDOWN ({len(all_mo)} months):")
    for m in sorted(all_mo.keys()):
        v = all_mo[m]
        print(f"    {m}: ${v:>+10,.0f} {'✅' if v > 0 else '❌'}")

    vals = list(all_mo.values())
    mp = sum(1 for v in vals if v > 0)
    print(f"\n    Months+: {mp}/{len(vals)}")
    print(f"    Average: ${np.mean(vals):,.0f}")
    print(f"    Median: ${np.median(vals):,.0f}")

    # Key insight
    flat3_y2 = 7140; flat3_dd = -4362
    adapt_y2 = best_y2["monthly_avg"]; adapt_dd = min(best_y2["max_dd"], best_bl["max_dd"])

    print(f"\n  KEY INSIGHT:")
    if adapt_y2 > flat3_y2 and adapt_dd > flat3_dd:
        print(f"    ✅ Adaptive sizing WORKS — more return (${adapt_y2:,.0f} vs ${flat3_y2:,.0f})")
        print(f"       AND less drawdown (${adapt_dd:,.0f} vs ${flat3_dd:,.0f})")
        insight = "Adaptive improves both returns and risk"
    elif adapt_y2 > flat3_y2:
        print(f"    ⚠️  Adaptive increases returns (${adapt_y2:,.0f} vs ${flat3_y2:,.0f})")
        print(f"       BUT also increases DD (${adapt_dd:,.0f} vs ${flat3_dd:,.0f})")
        sharpe_flat = 7140 / 4362; sharpe_adapt = adapt_y2 / abs(adapt_dd) if adapt_dd != 0 else 0
        print(f"       Risk-adjusted: flat {sharpe_flat:.2f} vs adaptive {sharpe_adapt:.2f}")
        insight = "Adaptive adds returns but also risk"
    else:
        print(f"    ❌ Adaptive doesn't help. Flat 3c: ${flat3_y2:,.0f}/mo. Adaptive: ${adapt_y2:,.0f}/mo.")
        print(f"       The conditions score doesn't predict trade quality. Stick with flat sizing.")
        insight = "Adaptive doesn't help — stick with flat"

    # Final recommendation
    all_options = [
        ("v3 conservative (2c)", 4760, 2161, -2908, 1.0),
        ("v3 aggressive (3c)", 7140, 3242, -4362, 1.0),
    ]
    for scheme in ["Scheme A (1-5c)", "Scheme B (2-4c)", "ATR stops"]:
        y2m = all_results[(scheme, "Y2")]["metrics"]
        blm = all_results[(scheme, "Blind")]["metrics"]
        dd = min(y2m["max_dd"], blm["max_dd"])
        all_options.append((f"v4 {scheme}", y2m["monthly_avg"], blm["monthly_avg"], dd,
                            mc_results[scheme]["baseline"]["pass_rate"]))

    # Pick best that survives MLL
    viable = [o for o in all_options if o[3] > -4500 and o[4] >= 0.80]
    if viable:
        best = max(viable, key=lambda x: x[1])  # max Y2 monthly
        print(f"\n  FINAL RECOMMENDATION: {best[0]}")
        print(f"    ${best[1]:,.0f}/month Y2, ${best[2]:,.0f}/month blind")
        print(f"    Max DD: ${best[3]:,.0f}, MC pass: {best[4]:.0%}")
    else:
        best = max(all_options, key=lambda x: x[1])
        print(f"\n  FINAL RECOMMENDATION: {best[0]} (best available)")

    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "insight": insight,
        "schemes": {},
    }
    for scheme in ["Scheme A (1-5c)", "Scheme B (2-4c)", "ATR stops"]:
        y2m = all_results[(scheme, "Y2")]["metrics"]
        blm = all_results[(scheme, "Blind")]["metrics"]
        report["schemes"][scheme] = {
            "y2": {k: v for k, v in y2m.items() if k != "monthly"},
            "blind": {k: v for k, v in blm.items() if k != "monthly"},
            "mc": mc_results[scheme],
        }
    out = REPORTS_DIR / "htf_swing_v4.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
