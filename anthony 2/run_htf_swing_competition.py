#!/usr/bin/env python3
"""
Head-to-head: Topstep vs LucidFlex for HTF Swing v3.

Same strategies, same params — only session close and daily loss limit differ.

Usage:
    python3 run_htf_swing_competition.py
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
    TICK_SIZE, POINT_VALUE, SLIP_PTS,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
_ET = ZoneInfo("US/Eastern")

YR1_END = datetime(2025, 3, 19, tzinfo=_ET)

STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}
C = 3


def run_config(df, flatten_hhmm, daily_limit, label=""):
    """Run all 3 strategies with a specific flatten time and daily limit."""
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()
    hhmm = df["hhmm"].to_numpy()
    dates = [str(t)[:10] for t in timestamps]
    n = len(opens)
    cost = rt_cost(C)
    slip = SLIP_PTS

    # Pre-compute all signals
    sig_arrays = {}
    for name, s in STRATS.items():
        sig_arrays[name] = (s["sig"](df), s["sl"], s["tp"], s["hold"])

    all_trades = []
    days_stopped = 0

    positions = {name: {"in_pos": False, "dir": 0, "ep": 0, "eb": 0,
                        "sp": 0, "tp_": 0, "pend": 0} for name in STRATS}

    daily_pnl = 0.0
    cur_date = ""
    stopped = False

    for i in range(n):
        h = hhmm[i]
        d = dates[i]

        if d != cur_date:
            if stopped: days_stopped += 1
            daily_pnl = 0.0
            stopped = False
            cur_date = d

        for name in STRATS:
            sigs, sl, tp, hold = sig_arrays[name]
            p = positions[name]

            # Flatten
            if p["in_pos"] and h >= flatten_hhmm and i > p["eb"]:
                ex = closes[i] - p["dir"] * slip
                raw = (ex - p["ep"]) * p["dir"] * POINT_VALUE * C
                pnl = raw - cost
                all_trades.append(Trade(p["dir"], p["ep"], ex, C, pnl,
                                        timestamps[p["eb"]], timestamps[i],
                                        i - p["eb"], "time_exit", name))
                daily_pnl += pnl
                p["in_pos"] = False; p["pend"] = 0
                continue

            # Daily limit
            if daily_limit is not None and daily_pnl <= daily_limit and not stopped:
                stopped = True
                if p["in_pos"] and i > p["eb"]:
                    ex = closes[i] - p["dir"] * slip
                    raw = (ex - p["ep"]) * p["dir"] * POINT_VALUE * C
                    pnl = raw - cost
                    all_trades.append(Trade(p["dir"], p["ep"], ex, C, pnl,
                                            timestamps[p["eb"]], timestamps[i],
                                            i - p["eb"], "daily_stop", name))
                    daily_pnl += pnl
                    p["in_pos"] = False
                p["pend"] = 0
                continue

            if stopped:
                p["pend"] = 0; continue

            # Execute pending
            if p["pend"] != 0 and not p["in_pos"]:
                if h >= flatten_hhmm - 15 or not (930 <= h < 1600):
                    p["pend"] = 0
                else:
                    p["ep"] = opens[i] + int(p["pend"]) * slip
                    p["dir"] = int(p["pend"])
                    p["eb"] = i
                    p["sp"] = p["ep"] - p["dir"] * sl * TICK_SIZE
                    p["tp_"] = p["ep"] + p["dir"] * tp * TICK_SIZE
                    p["in_pos"] = True
                    p["pend"] = 0

            # Manage
            if p["in_pos"] and i > p["eb"]:
                bh = i - p["eb"]
                ex = None; reason = ""

                if p["dir"] == 1 and lows[i] <= p["sp"]:
                    ex = p["sp"]; reason = "stop_loss"
                elif p["dir"] == -1 and highs[i] >= p["sp"]:
                    ex = p["sp"]; reason = "stop_loss"

                if ex is None:
                    if p["dir"] == 1 and highs[i] >= p["tp_"]:
                        ex = p["tp_"]; reason = "take_profit"
                    elif p["dir"] == -1 and lows[i] <= p["tp_"]:
                        ex = p["tp_"]; reason = "take_profit"

                if ex is None and bh >= hold:
                    ex = closes[i]; reason = "max_hold"

                if ex is not None:
                    ex -= p["dir"] * slip
                    raw = (ex - p["ep"]) * p["dir"] * POINT_VALUE * C
                    pnl = raw - cost
                    all_trades.append(Trade(p["dir"], p["ep"], ex, C, pnl,
                                            timestamps[p["eb"]], timestamps[i],
                                            bh, reason, name))
                    daily_pnl += pnl
                    p["in_pos"] = False
                    if daily_limit is not None and daily_pnl <= daily_limit:
                        stopped = True

            # New signal
            if not p["in_pos"] and not stopped and i < len(sigs) and sigs[i] != 0:
                if 930 <= h < (flatten_hhmm - 15):
                    p["pend"] = sigs[i]

    if stopped: days_stopped += 1
    return all_trades, days_stopped


def calc_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "worst_day": 0, "max_dd": 0, "sharpe": 0, "monthly": {},
                "n_months": 0, "months_pos": 0}
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
        "max_dd": mdd, "sharpe": sharpe, "monthly": dict(monthly),
        "n_months": nm, "months_pos": sum(1 for v in monthly.values() if v > 0),
    }


def run_mc(trades, mll=-4500, n_sims=5000):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl)
    days = list(daily.values())
    nd = len(days)
    if nd == 0: return {"pass": 0, "blow": 1}
    p = 0; b = 0
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; ok = True; passed = False
        for idx in order:
            cum += sum(days[idx]); peak = max(peak, cum)
            if cum - peak <= mll: ok = False; break
            if cum >= 9000: passed = True
        if not ok: b += 1
        if passed and ok: p += 1
    return {"pass": p / n_sims, "blow": b / n_sims}


def print_round(label, ts_m, lf_m, ts_ds, lf_ds):
    """Print one round of competition."""
    print(f"\n  {label}:")
    print(f"  ┌{'─'*20}┬{'─'*12}┬{'─'*12}┬{'─'*10}┐")
    print(f"  │{'Metric':<20}│{'Topstep':>12}│{'LucidFlex':>12}│{'Winner':>10}│")
    print(f"  ├{'─'*20}┼{'─'*12}┼{'─'*12}┼{'─'*10}┤")
    for lbl, tv, lv, higher_wins in [
        ("Monthly P&L", f"${ts_m['monthly_avg']:>+10,.0f}", f"${lf_m['monthly_avg']:>+10,.0f}", True),
        ("Total trades", f"{ts_m['n']:>12}", f"{lf_m['n']:>12}", True),
        ("Win rate", f"{ts_m['wr']:>11.1f}%", f"{lf_m['wr']:>11.1f}%", True),
        ("Worst day", f"${ts_m['worst_day']:>+10,.0f}", f"${lf_m['worst_day']:>+10,.0f}", False),
        ("Worst month", f"${ts_m['worst_month']:>+10,.0f}", f"${lf_m['worst_month']:>+10,.0f}", False),
        ("Max DD", f"${ts_m['max_dd']:>+10,.0f}", f"${lf_m['max_dd']:>+10,.0f}", False),
        ("Sharpe", f"{ts_m['sharpe']:>12.2f}", f"{lf_m['sharpe']:>12.2f}", True),
        ("Days stopped", f"{ts_ds:>12}", f"{lf_ds:>12}", False),
    ]:
        # Determine winner
        try:
            t_val = ts_m.get(lbl.lower().replace(" ", "_"), 0)
            l_val = lf_m.get(lbl.lower().replace(" ", "_"), 0)
        except:
            t_val = l_val = 0
        winner = "—"
        if lbl == "Monthly P&L":
            winner = "LF" if lf_m["monthly_avg"] > ts_m["monthly_avg"] else "TS"
        elif lbl == "Sharpe":
            winner = "LF" if lf_m["sharpe"] > ts_m["sharpe"] else "TS"
        elif lbl == "Max DD":
            winner = "LF" if lf_m["max_dd"] > ts_m["max_dd"] else "TS"
        elif lbl == "Worst day":
            winner = "LF" if lf_m["worst_day"] > ts_m["worst_day"] else "TS"

        print(f"  │{lbl:<20}│{tv:>12}│{lv:>12}│{winner:>10}│")
    print(f"  └{'─'*20}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  COMPETITION: Topstep vs LucidFlex")
    print("═" * 70)

    # Load data
    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind")

    yr1 = main_data["15m"].filter(pl.col("timestamp") < YR1_END)
    yr2 = main_data["15m"].filter(pl.col("timestamp") >= YR1_END)
    bl = blind_data["15m"]

    # Load 8yr if available
    try:
        df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "8yr-1")
        df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "8yr-2")
        df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "8yr-3")
        combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        combined = combined.filter(pl.col("close") > 0)
        full_8yr = resample_15m_rth(combined)
        del df1, df2, df3, combined
        has_8yr = True
        print(f"  8yr 15m: {len(full_8yr):,} bars")
    except Exception as e:
        has_8yr = False
        print(f"  8yr data not available: {e}")
    gc.collect()

    datasets = [("Y1", yr1), ("Y2", yr2), ("Blind", bl)]
    if has_8yr:
        datasets.append(("8yr", full_8yr))

    # Run both configs on all datasets
    results = {}
    for period, df in datasets:
        print(f"\n  Running {period} ...")
        ts_trades, ts_ds = run_config(df, 1545, -3000, f"TS_{period}")
        lf_trades, lf_ds = run_config(df, 1645, None, f"LF_{period}")
        results[period] = {
            "ts": {"trades": ts_trades, "metrics": calc_metrics(ts_trades), "ds": ts_ds},
            "lf": {"trades": lf_trades, "metrics": calc_metrics(lf_trades), "ds": lf_ds},
        }
        gc.collect()

    # ── Rounds ──
    for i, (period, _) in enumerate(datasets, 1):
        ts_m = results[period]["ts"]["metrics"]
        lf_m = results[period]["lf"]["metrics"]
        print_round(f"ROUND {i} — {period}", ts_m, lf_m,
                    results[period]["ts"]["ds"], results[period]["lf"]["ds"])

    # MC
    print(f"\n  Monte Carlo (Y2):")
    mc_ts = run_mc(results["Y2"]["ts"]["trades"])
    mc_lf = run_mc(results["Y2"]["lf"]["trades"])
    print(f"    Topstep:  {mc_ts['pass']:.0%} pass, {mc_ts['blow']:.0%} blow-up")
    print(f"    LucidFlex: {mc_lf['pass']:.0%} pass, {mc_lf['blow']:.0%} blow-up")

    # ── Extra hour breakdown ──
    print(f"\n{'━' * 70}")
    print("  THE EXTRA HOUR (3:45-4:45 PM)")
    print("━" * 70)

    # LucidFlex trades that enter or exit after 15:45
    # Compare: LF trades minus TS trades = extra hour trades
    for period in ["Y2", "Blind"]:
        ts_set = {(str(t.entry_time), t.strategy, t.direction) for t in results[period]["ts"]["trades"]}
        extra_trades = [t for t in results[period]["lf"]["trades"]
                        if (str(t.entry_time), t.strategy, t.direction) not in ts_set]

        extra_monthly = defaultdict(float)
        extra_tc = defaultdict(int)
        for t in extra_trades:
            mo = str(t.entry_time)[:7]
            extra_monthly[mo] += t.net_pnl
            extra_tc[mo] += 1

        extra_total = sum(t.net_pnl for t in extra_trades)
        extra_w = sum(1 for t in extra_trades if t.net_pnl > 0)
        extra_wr = extra_w / len(extra_trades) * 100 if extra_trades else 0

        print(f"\n  {period} — Extra trades from 3:45-4:45 PM:")
        print(f"  ┌{'─'*9}┬{'─'*14}┬{'─'*12}┬{'─'*8}┬{'─'*7}┐")
        print(f"  │{'Month':<9}│{'Extra Trades':>14}│{'Extra P&L':>12}│{'WR':>8}│{'Help?':>7}│")
        print(f"  ├{'─'*9}┼{'─'*14}┼{'─'*12}┼{'─'*8}┼{'─'*7}┤")

        all_months = sorted(set(
            list(results[period]["lf"]["metrics"]["monthly"].keys()) +
            list(extra_monthly.keys())
        ))
        helps = 0; hurts = 0
        for mo in all_months:
            et = extra_tc.get(mo, 0)
            ep = extra_monthly.get(mo, 0)
            wr_mo = "—"
            if et > 0:
                w_mo = sum(1 for t in extra_trades if str(t.entry_time)[:7] == mo and t.net_pnl > 0)
                wr_mo = f"{w_mo/et*100:.0f}%"
            flag = "✅" if ep > 0 else ("—" if et == 0 else "❌")
            if ep > 0: helps += 1
            elif et > 0: hurts += 1
            print(f"  │{mo:<9}│{et:>14}│${ep:>+11,.0f}│{wr_mo:>8}│{flag:>7}│")

        print(f"  ├{'─'*9}┼{'─'*14}┼{'─'*12}┼{'─'*8}┼{'─'*7}┤")
        print(f"  │{'TOTAL':<9}│{len(extra_trades):>14}│${extra_total:>+11,.0f}│{extra_wr:>7.0f}%│       │")
        print(f"  └{'─'*9}┴{'─'*14}┴{'─'*12}┴{'─'*8}┴{'─'*7}┘")
        print(f"  Months helped: {helps} | Months hurt: {hurts}")

    # ── Daily loss limit analysis ──
    print(f"\n{'━' * 70}")
    print("  DAILY LOSS LIMIT ANALYSIS")
    print("━" * 70)
    for period in ["Y2", "Blind"]:
        ds = results[period]["ts"]["ds"]
        print(f"  {period}: Topstep DLL ($3K) triggered {ds} times")

    # ── Final scorecard ──
    print(f"\n{'═' * 70}")
    print("  FINAL SCORECARD")
    print("═" * 70)
    print(f"\n  ┌{'─'*20}┬{'─'*12}┬{'─'*12}┐")
    print(f"  │{'':>20}│{'Topstep':>12}│{'LucidFlex':>12}│")
    print(f"  ├{'─'*20}┼{'─'*12}┼{'─'*12}┤")

    ts_y2 = results["Y2"]["ts"]["metrics"]["monthly_avg"]
    lf_y2 = results["Y2"]["lf"]["metrics"]["monthly_avg"]
    ts_bl = results["Blind"]["ts"]["metrics"]["monthly_avg"]
    lf_bl = results["Blind"]["lf"]["metrics"]["monthly_avg"]

    for lbl, tv, lv in [
        ("Y2 $/month", f"${ts_y2:>+10,.0f}", f"${lf_y2:>+10,.0f}"),
        ("Blind $/month", f"${ts_bl:>+10,.0f}", f"${lf_bl:>+10,.0f}"),
        ("MC eval pass", f"{mc_ts['pass']:>11.0%}", f"{mc_lf['pass']:>11.0%}"),
        ("Cost to start", f"{'$149/mo':>12}", f"{'~$300 once':>12}"),
        ("Extra hour value", f"{'n/a':>12}", f"${lf_y2 - ts_y2:>+10,.0f}"),
        ("DLL triggers (Y2)", f"{results['Y2']['ts']['ds']:>12}", f"{'0':>12}"),
    ]:
        print(f"  │{lbl:<20}│{tv:>12}│{lv:>12}│")

    # 8yr if available
    if has_8yr:
        ts_8 = results["8yr"]["ts"]["metrics"]["monthly_avg"]
        lf_8 = results["8yr"]["lf"]["metrics"]["monthly_avg"]
        print(f"  │{'8yr $/month':<20}│${ts_8:>+10,.0f}│${lf_8:>+10,.0f}│")

    winner = "LucidFlex" if lf_y2 > ts_y2 else "Topstep"
    print(f"  ├{'─'*20}┼{'─'*12}┼{'─'*12}┤")
    print(f"  │{'OVERALL WINNER':<20}│{'':>12}│{winner:>12}│" if winner == "LucidFlex"
          else f"  │{'OVERALL WINNER':<20}│{winner:>12}│{'':>12}│")
    print(f"  └{'─'*20}┴{'─'*12}┴{'─'*12}┘")

    # Recommendation
    diff = lf_y2 - ts_y2
    print(f"\n  RECOMMENDATION:")
    if diff > 0:
        print(f"  LucidFlex is the better choice — gains ${diff:,.0f}/month from the extra hour.")
        print(f"  The daily loss limit never triggers (worst day is only -${abs(results['Y2']['ts']['metrics']['worst_day']):,.0f}).")
        print(f"  One-time cost of ~$300 vs Topstep's $149/month recurring is cheaper after 3 months.")
    else:
        print(f"  Topstep is the better choice — the daily limit saves ${abs(diff):,.0f}/month.")

    print("═" * 70)

    # Save
    report = {
        "timestamp": str(datetime.now()),
        "winner": winner,
        "y2": {"topstep": ts_y2, "lucidflex": lf_y2, "diff": diff},
        "blind": {"topstep": ts_bl, "lucidflex": lf_bl},
        "mc": {"topstep": mc_ts, "lucidflex": mc_lf},
    }
    if has_8yr:
        report["8yr"] = {"topstep": results["8yr"]["ts"]["metrics"]["monthly_avg"],
                         "lucidflex": results["8yr"]["lf"]["metrics"]["monthly_avg"]}
    out = REPORTS_DIR / "htf_swing_competition.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
