#!/usr/bin/env python3
"""
HTF Swing v3 — LucidFlex 150K daily-limit optimization.

Tests 5 daily limits × 2 session lengths = 10 variants.
Same RSI+IB+MOM strategies, same params, only risk management changes.

Usage:
    python3 run_htf_swing_lucid.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from run_htf_swing import (
    load_and_resample, extract_arrays, rt_cost, Trade,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    calc_atr, TICK_SIZE, POINT_VALUE, SLIP_PTS,
)

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
_ET = ZoneInfo("US/Eastern")

# LucidFlex 150K
MLL = -4500.0
EVAL_TARGET = 9000.0

STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}
CONTRACTS = 3
YR1_END = datetime(2025, 3, 19, tzinfo=_ET)


def backtest_with_daily_limit(
    df, signals_dict, daily_limit, flatten_hhmm, contracts=3,
):
    """Run all 3 strategies with a daily loss limit and configurable flatten time.

    signals_dict: {name: (signals_array, sl, tp, hold)}
    daily_limit: negative number (e.g. -2000) or None for no limit
    flatten_hhmm: 1545 for 3:45 PM, 1645 for 4:45 PM
    """
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()
    hhmm = df["hhmm"].to_numpy()
    dates = [str(t)[:10] for t in timestamps]
    n = len(opens)

    cost = rt_cost(contracts)
    slip = SLIP_PTS

    all_trades = []
    days_stopped = 0

    # Per-strategy position state
    positions = {}  # name -> {in_pos, direction, entry_px, entry_bar, stop_px, target_px, pending}
    for name in signals_dict:
        positions[name] = {"in_pos": False, "direction": 0, "entry_px": 0, "entry_bar": 0,
                           "stop_px": 0, "target_px": 0, "pending": 0}

    daily_pnl = 0.0
    current_date = ""
    daily_stopped = False

    for i in range(n):
        h = hhmm[i]
        d = dates[i]

        # New day
        if d != current_date:
            if daily_stopped:
                days_stopped += 1
            daily_pnl = 0.0
            daily_stopped = False
            current_date = d

        for name, (sigs, sl, tp, hold) in signals_dict.items():
            pos = positions[name]

            # Flatten at session close
            if pos["in_pos"] and h >= flatten_hhmm and i > pos["entry_bar"]:
                ex = closes[i] - pos["direction"] * slip
                raw = (ex - pos["entry_px"]) * pos["direction"] * POINT_VALUE * contracts
                pnl = raw - cost

                all_trades.append(Trade(pos["direction"], pos["entry_px"], ex, contracts,
                                        pnl, timestamps[pos["entry_bar"]], timestamps[i],
                                        i - pos["entry_bar"], "time_exit", name))
                daily_pnl += pnl
                pos["in_pos"] = False
                pos["pending"] = 0
                continue

            # Daily limit hit — close all open positions
            if daily_limit is not None and daily_pnl <= daily_limit and not daily_stopped:
                daily_stopped = True
                if pos["in_pos"] and i > pos["entry_bar"]:
                    ex = closes[i] - pos["direction"] * slip
                    raw = (ex - pos["entry_px"]) * pos["direction"] * POINT_VALUE * contracts
                    pnl = raw - cost
                    all_trades.append(Trade(pos["direction"], pos["entry_px"], ex, contracts,
                                            pnl, timestamps[pos["entry_bar"]], timestamps[i],
                                            i - pos["entry_bar"], "daily_stop", name))
                    daily_pnl += pnl
                    pos["in_pos"] = False
                pos["pending"] = 0
                continue

            if daily_stopped:
                pos["pending"] = 0
                continue

            # Execute pending
            if pos["pending"] != 0 and not pos["in_pos"]:
                if h >= flatten_hhmm - 15 or not (930 <= h < 1600):
                    pos["pending"] = 0
                else:
                    pos["entry_px"] = opens[i] + int(pos["pending"]) * slip
                    pos["direction"] = int(pos["pending"])
                    pos["entry_bar"] = i
                    pos["stop_px"] = pos["entry_px"] - pos["direction"] * sl * TICK_SIZE
                    pos["target_px"] = pos["entry_px"] + pos["direction"] * tp * TICK_SIZE
                    pos["in_pos"] = True
                    pos["pending"] = 0

            # Manage (skip entry bar)
            if pos["in_pos"] and i > pos["entry_bar"]:
                bh = i - pos["entry_bar"]
                ex = None; reason = ""

                # Stop first (conservative)
                if pos["direction"] == 1 and lows[i] <= pos["stop_px"]:
                    ex = pos["stop_px"]; reason = "stop_loss"
                elif pos["direction"] == -1 and highs[i] >= pos["stop_px"]:
                    ex = pos["stop_px"]; reason = "stop_loss"

                if ex is None:
                    if pos["direction"] == 1 and highs[i] >= pos["target_px"]:
                        ex = pos["target_px"]; reason = "take_profit"
                    elif pos["direction"] == -1 and lows[i] <= pos["target_px"]:
                        ex = pos["target_px"]; reason = "take_profit"

                if ex is None and bh >= hold:
                    ex = closes[i]; reason = "max_hold"

                if ex is not None:
                    ex -= pos["direction"] * slip
                    raw = (ex - pos["entry_px"]) * pos["direction"] * POINT_VALUE * contracts
                    pnl = raw - cost
                    all_trades.append(Trade(pos["direction"], pos["entry_px"], ex, contracts,
                                            pnl, timestamps[pos["entry_bar"]], timestamps[i],
                                            bh, reason, name))
                    daily_pnl += pnl
                    pos["in_pos"] = False

                    # Check daily limit after trade close
                    if daily_limit is not None and daily_pnl <= daily_limit:
                        daily_stopped = True

            # New signal
            if not pos["in_pos"] and not daily_stopped and i < len(sigs) and sigs[i] != 0:
                if 930 <= h < (flatten_hhmm - 15):
                    pos["pending"] = sigs[i]

    # Count final day if stopped
    if daily_stopped:
        days_stopped += 1

    return all_trades, days_stopped


def metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "worst_day": 0, "max_dd": 0, "monthly": {}, "n_months": 0,
                "months_pos": 0, "sharpe": 0, "trades_per_mo": 0}
    pnls = [t.net_pnl for t in trades]
    w = sum(1 for p in pnls if p > 0)
    monthly = defaultdict(float)
    daily = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
        daily[str(t.entry_time)[:10]] += t.net_pnl
    cum = 0; peak = 0; mdd = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    dv = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0
    nm = max(len(monthly), 1)
    return {
        "pnl": sum(pnls), "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": sum(pnls) / nm,
        "worst_month": min(monthly.values()) if monthly else 0,
        "worst_day": min(dv) if dv else 0,
        "max_dd": mdd, "monthly": dict(monthly), "n_months": nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "sharpe": sharpe, "trades_per_mo": len(trades) / nm,
    }


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass_rate": 0, "blowup": 1, "med_days": 0}
    p = 0; b = 0; dtp = []
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; ok = True; passed = False; dc = 0
        for idx in order:
            dp = sum(days[idx])
            cum += dp; dc += 1; peak = max(peak, cum)
            if cum - peak <= MLL: ok = False; break
            if not passed and cum >= EVAL_TARGET: passed = True; dtp.append(dc)
        if not ok: b += 1
        if passed and ok: p += 1
    da = np.array(dtp) if dtp else np.array([0])
    return {"pass_rate": p / n_sims, "blowup": b / n_sims,
            "med_days": int(np.median(da)) if len(da) else 0}


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — LucidFlex 150K Daily Limit Comparison")
    print("═" * 70)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind")

    yr2 = main_data["15m"].filter(pl.col("timestamp") >= YR1_END)
    bl = blind_data["15m"]

    # Pre-compute signals once
    print("\n  Pre-computing signals ...")
    sig_cache = {}
    for label, df in [("Y2", yr2), ("Blind", bl)]:
        sig_cache[label] = {}
        for name, s in STRATS.items():
            sigs = s["sig"](df)
            sig_cache[label][name] = (sigs, s["sl"], s["tp"], s["hold"])
    gc.collect()

    # Test configurations
    limits = [None, -3000, -2500, -2000, -1500]
    sessions = [1545, 1645]
    limit_labels = {None: "No limit", -3000: "$3K", -2500: "$2.5K", -2000: "$2K", -1500: "$1.5K"}
    session_labels = {1545: "3:45", 1645: "4:45"}

    results = {}

    for label, df in [("Y2", yr2), ("Blind", bl)]:
        print(f"\n{'━' * 70}")
        print(f"  {label} Results")
        print("━" * 70)

        print(f"\n  ┌{'─'*21}┬{'─'*9}┬{'─'*8}┬{'─'*7}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*6}┐")
        print(f"  │{'Config':<21}│{'$/mo':>9}│{'Tr/mo':>8}│{'WR':>7}│{'Wrst Day':>10}│{'Wrst Mo':>10}│{'Max DD':>10}│{'Stop':>6}│")
        print(f"  ├{'─'*21}┼{'─'*9}┼{'─'*8}┼{'─'*7}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*6}┤")

        for lim in limits:
            for sess in sessions:
                trades, days_stopped = backtest_with_daily_limit(
                    df, sig_cache[label], lim, sess, CONTRACTS)
                m = metrics(trades)
                key = (label, lim, sess)
                results[key] = {"metrics": m, "trades": trades, "days_stopped": days_stopped}

                cfg = f"{limit_labels[lim]} ({session_labels[sess]})"
                print(f"  │{cfg:<21}│${m['monthly_avg']:>8,.0f}│{m['trades_per_mo']:>8.0f}│{m['wr']:>6.1f}%│"
                      f"${m['worst_day']:>9,.0f}│${m['worst_month']:>9,.0f}│${m['max_dd']:>9,.0f}│{days_stopped:>6}│")

        print(f"  └{'─'*21}┴{'─'*9}┴{'─'*8}┴{'─'*7}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*6}┘")
        gc.collect()

    # ── Analysis ──
    print(f"\n{'━' * 70}")
    print("  ANALYSIS")
    print("━" * 70)

    # 1. Cost of each limit
    no_lim_y2 = results[("Y2", None, 1545)]["metrics"]["monthly_avg"]
    print(f"\n  1. COST OF DAILY LIMITS (Y2, 3:45 close):")
    for lim in limits:
        m = results[("Y2", lim, 1545)]["metrics"]
        diff = m["monthly_avg"] - no_lim_y2
        print(f"     {limit_labels[lim]:>10}: ${m['monthly_avg']:>+8,.0f}/mo  "
              f"({'$0' if lim is None else f'costs ${abs(diff):,.0f}/mo'})")

    # 2. Risk savings
    no_lim_wd = results[("Y2", None, 1545)]["metrics"]["worst_day"]
    print(f"\n  2. RISK SAVINGS (Y2 worst day):")
    for lim in limits:
        m = results[("Y2", lim, 1545)]["metrics"]
        saved = m["worst_day"] - no_lim_wd
        print(f"     {limit_labels[lim]:>10}: worst day ${m['worst_day']:>+8,.0f}  "
              f"({'baseline' if lim is None else f'saves ${saved:+,.0f}'})")

    # 3. Extra hour value
    print(f"\n  3. VALUE OF EXTENDED SESSION (3:45 → 4:45):")
    for lim in limits:
        m_345 = results[("Y2", lim, 1545)]["metrics"]["monthly_avg"]
        m_445 = results[("Y2", lim, 1645)]["metrics"]["monthly_avg"]
        diff = m_445 - m_345
        print(f"     {limit_labels[lim]:>10}: ${m_345:>+7,.0f} → ${m_445:>+7,.0f}  (Δ${diff:>+,.0f}/mo)")

    # 4. Sweet spot by Sharpe
    print(f"\n  4. SHARPE RANKING (Y2):")
    ranked = sorted(
        [(lim, sess) for lim in limits for sess in sessions],
        key=lambda x: results[("Y2", x[0], x[1])]["metrics"]["sharpe"],
        reverse=True
    )
    for i, (lim, sess) in enumerate(ranked[:5]):
        m = results[("Y2", lim, sess)]["metrics"]
        cfg = f"{limit_labels[lim]} ({session_labels[sess]})"
        print(f"     #{i+1}: {cfg:<21} Sharpe={m['sharpe']:.2f}  ${m['monthly_avg']:>+8,.0f}/mo  DD=${m['max_dd']:>+8,.0f}")

    best_lim, best_sess = ranked[0]
    best_key = ("Y2", best_lim, best_sess)
    best_m = results[best_key]["metrics"]
    best_cfg = f"{limit_labels[best_lim]} ({session_labels[best_sess]})"

    # Monthly breakdown (best config)
    print(f"\n{'━' * 70}")
    print(f"  MONTHLY BREAKDOWN — {best_cfg}")
    print("━" * 70)

    # Combine Y2 + blind for best config
    best_bl_key = ("Blind", best_lim, best_sess)
    for period_label, key in [("Y2", best_key), ("Blind", best_bl_key)]:
        m = results[key]["metrics"]
        print(f"\n  {period_label}:")
        for mo in sorted(m["monthly"].keys()):
            v = m["monthly"][mo]
            print(f"    {mo}: ${v:>+10,.0f} {'✅' if v > 0 else '❌'}")
        print(f"    Months+: {m['months_pos']}/{m['n_months']}")

    # MC
    print(f"\n{'━' * 70}")
    print(f"  MONTE CARLO — {best_cfg}")
    print("━" * 70)
    mc_b = run_mc(results[best_key]["trades"], 5000, 1.0)
    mc_c = run_mc(results[best_key]["trades"], 5000, 0.70)
    print(f"  Baseline:     {mc_b['pass_rate']:.0%} pass, {mc_b['blowup']:.0%} blow-up, median {mc_b['med_days']}d")
    print(f"  Conservative: {mc_c['pass_rate']:.0%} pass, {mc_c['blowup']:.0%} blow-up")

    # Also MC for Topstep-equivalent ($3K limit at 3:45)
    ts_key = ("Y2", -3000, 1545)
    mc_ts = run_mc(results[ts_key]["trades"], 5000, 1.0)

    # Recommendation
    ts_m = results[ts_key]["metrics"]
    gain = best_m["monthly_avg"] - ts_m["monthly_avg"]

    print(f"\n{'═' * 70}")
    print(f"  RECOMMENDATION")
    print("═" * 70)
    print(f"\n  For LucidFlex 150K, the optimal self-imposed daily limit is "
          f"{limit_labels[best_lim]} with session close at {session_labels[best_sess]} PM.")
    print(f"  This produces ${best_m['monthly_avg']:,.0f}/month on Y2 OOS "
          f"with {mc_b['pass_rate']:.0%} MC eval pass rate.")
    print(f"\n  Compared to Topstep rules ($3K DLL, 3:45 close): ${ts_m['monthly_avg']:,.0f}/mo, {mc_ts['pass_rate']:.0%} MC")
    if gain > 0:
        print(f"  LucidFlex gains ${gain:,.0f}/month because the wider/no daily limit "
              f"allows recovery from intraday drawdowns.")
    else:
        print(f"  LucidFlex loses ${abs(gain):,.0f}/month — the tighter Topstep limit actually protects against cascading losses.")

    print(f"\n  Days where daily limit triggered (Y2):")
    for lim in limits:
        ds = results[("Y2", lim, 1545)]["days_stopped"]
        print(f"    {limit_labels[lim]:>10}: {ds} days stopped")

    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "best_config": best_cfg,
        "best_limit": best_lim,
        "best_session": best_sess,
        "results_summary": {},
    }
    for (label, lim, sess), r in results.items():
        k = f"{label}_{limit_labels[lim]}_{session_labels[sess]}"
        report["results_summary"][k] = {
            kk: vv for kk, vv in r["metrics"].items() if kk != "monthly"
        }
        report["results_summary"][k]["days_stopped"] = r["days_stopped"]
    report["mc_best"] = {"baseline": mc_b, "conservative": mc_c}
    report["mc_topstep"] = mc_ts

    out = REPORTS_DIR / "htf_swing_lucid.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
