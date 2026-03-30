#!/usr/bin/env python3
"""
HTF Swing v3 HYBRID — Full backtest of TradingView-validated params.

RSI/IB: tighter SL=10pts from optimizer.
MOM: ATR mult 1.4 from TradingView forward testing.
All else: original v3 or shared improvements.

Usage:
    python3 run_htf_swing_v3_hybrid.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

import run_htf_swing as _base
from run_htf_swing import (
    load_and_resample, extract_arrays, backtest, rt_cost,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

DAILY_LIMIT = -3000.0
MLL = -4500.0
EVAL_TARGET = 9000.0
FLATTEN_TIME = 1645    # LucidFlex session
CONTRACTS = 3

# ── Parameter sets ──────────────────────────────────────────────────

CURRENT = {
    "RSI": {"period": 7,  "ob": 70, "os": 30, "sl_pts": 15,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 20,  "tp_pts": 120, "hold": 10},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}

HYBRID = {
    "RSI": {"period": 5,  "ob": 65, "os": 35, "sl_pts": 10,  "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                 "sl_pts": 10,  "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.4, "vol_mult": 1.0,  "sl_pts": 15,  "tp_pts": 100, "hold": 5},
}


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def pts_to_ticks(pts):
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


def full_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "worst_month": 0,
                "best_month": 0, "worst_day": 0, "best_day": 0, "max_dd": 0,
                "bars_mean": 0, "monthly": {}, "n_months": 0, "months_pos": 0,
                "sharpe": 0, "trades_per_day": 0}
    pnls = [t.net_pnl for t in trades]
    bars = [t.bars_held for t in trades]
    w = sum(1 for p in pnls if p > 0)
    monthly = defaultdict(float)
    daily = defaultdict(float)
    for t in trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl
        daily[str(t.entry_time)[:10]] += t.net_pnl
    cum = 0; peak = 0; mdd = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    nm = max(len(monthly), 1)
    nd = max(len(daily), 1)
    total = sum(pnls)
    dv = list(daily.values())
    sharpe = float(np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0
    return {
        "pnl": total, "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": total / nm,
        "worst_month": min(monthly.values()) if monthly else 0,
        "best_month": max(monthly.values()) if monthly else 0,
        "worst_day": min(daily.values()) if daily else 0,
        "best_day": max(daily.values()) if daily else 0,
        "max_dd": mdd, "bars_mean": np.mean(bars),
        "monthly": dict(monthly), "n_months": nm,
        "months_pos": sum(1 for v in monthly.values() if v > 0),
        "sharpe": sharpe,
        "trades_per_day": len(trades) / nd,
    }


def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        daily[str(t.entry_time)[:10]].append(t.net_pnl * pnl_mult)
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
# RUN A PARAM SET ON A DATASET
# ═════════════════════════════════════════════════════════════════════

def run_system(df, params, contracts=CONTRACTS, flatten=FLATTEN_TIME):
    """Run all 3 strategies with given params. Returns (all_trades, per_strat_trades)."""
    o, h, l, c, ts, hm = extract_arrays(df)
    per_strat = {}

    # RSI
    p = params["RSI"]
    sigs = sig_rsi_extreme(df, p["period"], p["ob"], p["os"])
    per_strat["RSI"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], contracts, "RSI", flatten)

    # IB
    p = params["IB"]
    sigs = sig_ib_breakout(df, p["ib_filter"])[0]
    per_strat["IB"] = backtest(o, h, l, c, ts, hm, sigs,
                                pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                p["hold"], contracts, "IB", flatten)

    # MOM
    p = params["MOM"]
    sigs = sig_momentum_bar(df, p["atr_mult"], p["vol_mult"])[0]
    per_strat["MOM"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], contracts, "MOM", flatten)

    all_trades = []
    for t in per_strat.values():
        all_trades.extend(t)
    return all_trades, per_strat


def run_with_slippage(df, params, slip_ticks, contracts=CONTRACTS):
    """Run system with modified slippage (monkey-patch module globals)."""
    orig_slip_pts = _base.SLIP_PTS
    orig_slip_ticks = _base.SLIP_TICKS
    _base.SLIP_PTS = slip_ticks * TICK_SIZE
    _base.SLIP_TICKS = slip_ticks
    try:
        trades, _ = run_system(df, params, contracts)
    finally:
        _base.SLIP_PTS = orig_slip_pts
        _base.SLIP_TICKS = orig_slip_ticks
    return trades


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 HYBRID — Full Backtest")
    print("═" * 70)

    # ── Parameter comparison ────────────────────────────────────────
    print(f"\n  PARAMETER COMPARISON:")
    rows = [
        ("RSI period",    CURRENT["RSI"]["period"],   HYBRID["RSI"]["period"]),
        ("RSI oversold",  CURRENT["RSI"]["os"],       HYBRID["RSI"]["os"]),
        ("RSI overbought",CURRENT["RSI"]["ob"],       HYBRID["RSI"]["ob"]),
        ("RSI SL (pts)",  CURRENT["RSI"]["sl_pts"],   HYBRID["RSI"]["sl_pts"]),
        ("RSI TP (pts)",  CURRENT["RSI"]["tp_pts"],   HYBRID["RSI"]["tp_pts"]),
        ("RSI hold",      CURRENT["RSI"]["hold"],     HYBRID["RSI"]["hold"]),
        ("IB SL (pts)",   CURRENT["IB"]["sl_pts"],    HYBRID["IB"]["sl_pts"]),
        ("IB TP (pts)",   CURRENT["IB"]["tp_pts"],    HYBRID["IB"]["tp_pts"]),
        ("IB hold",       CURRENT["IB"]["hold"],      HYBRID["IB"]["hold"]),
        ("MOM ATR mult",  CURRENT["MOM"]["atr_mult"], HYBRID["MOM"]["atr_mult"]),
        ("MOM vol mult",  CURRENT["MOM"]["vol_mult"], HYBRID["MOM"]["vol_mult"]),
        ("MOM SL (pts)",  CURRENT["MOM"]["sl_pts"],   HYBRID["MOM"]["sl_pts"]),
        ("MOM TP (pts)",  CURRENT["MOM"]["tp_pts"],   HYBRID["MOM"]["tp_pts"]),
        ("MOM hold",      CURRENT["MOM"]["hold"],     HYBRID["MOM"]["hold"]),
    ]
    print(f"  {'Parameter':<18} {'Current':>10} {'Hybrid':>10} {'Change':>10}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}")
    for label, cur, hyb in rows:
        changed = " *" if cur != hyb else ""
        print(f"  {label:<18} {str(cur):>10} {str(hyb):>10}{changed}")

    # ── Load data ───────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  LOADING DATA")
    print("━" * 70)

    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main (2024-2026)")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind (2022-2024)")

    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
    Y1_END = datetime(2025, 3, 1, tzinfo=_ET)

    df_main = main_data["15m"]
    yr1 = df_main.filter(pl.col("timestamp") < Y1_END)
    yr2 = df_main.filter(pl.col("timestamp") >= Y1_END)
    bl = blind_data["15m"]

    print(f"  Y1 (Mar 2024 - Feb 2025): {len(yr1):,} bars")
    print(f"  Y2 (Mar 2025 - Mar 2026): {len(yr2):,} bars")
    print(f"  Blind (Dec 2021 - Mar 2024): {len(bl):,} bars")

    # Load 8yr data
    has_8yr = True
    try:
        df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
        df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
        df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
        combined_1m = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        combined_1m = combined_1m.filter(pl.col("close") > 0)
        df_8yr = resample_15m_rth(combined_1m)
        print(f"  8yr: {len(df_8yr):,} bars ({df_8yr['timestamp'].min()} to {df_8yr['timestamp'].max()})")
        del combined_1m, df1, df2, df3
        gc.collect()
    except Exception as e:
        print(f"  8yr data not available: {e}")
        has_8yr = False

    # ═════════════════════════════════════════════════════════════════
    # RUN BOTH SYSTEMS ON ALL PERIODS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  RUNNING BACKTESTS")
    print("━" * 70)

    periods = [("Y1", yr1), ("Y2", yr2), ("Blind", bl)]
    if has_8yr:
        periods.append(("8yr", df_8yr))

    results = {"current": {}, "hybrid": {}}

    for label, df in periods:
        print(f"\n  {label}:")
        # Current
        trades_cur, strat_cur = run_system(df, CURRENT)
        m_cur = full_metrics(trades_cur)
        results["current"][label] = {
            "metrics": m_cur, "trades": trades_cur,
            "per_strat": {k: full_metrics(v) for k, v in strat_cur.items()},
        }
        print(f"    Current:  {m_cur['n']} trades, ${m_cur['pnl']:>+10,.0f}, "
              f"{m_cur['wr']:.1f}% WR, Sharpe {m_cur['sharpe']:.1f}")

        # Hybrid
        trades_hyb, strat_hyb = run_system(df, HYBRID)
        m_hyb = full_metrics(trades_hyb)
        results["hybrid"][label] = {
            "metrics": m_hyb, "trades": trades_hyb,
            "per_strat": {k: full_metrics(v) for k, v in strat_hyb.items()},
        }
        print(f"    Hybrid:   {m_hyb['n']} trades, ${m_hyb['pnl']:>+10,.0f}, "
              f"{m_hyb['wr']:.1f}% WR, Sharpe {m_hyb['sharpe']:.1f}")
        gc.collect()

    # ═════════════════════════════════════════════════════════════════
    # HEAD-TO-HEAD
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  HEAD-TO-HEAD COMPARISON")
    print("═" * 70)

    print(f"\n  {'Period':<16} {'Current $/mo':>13} {'Hybrid $/mo':>13} {'Delta':>10} {'Delta%':>8}")
    print(f"  {'─'*16} {'─'*13} {'─'*13} {'─'*10} {'─'*8}")
    for label in ["Y1", "Y2", "Blind"] + (["8yr"] if has_8yr else []):
        c_avg = results["current"][label]["metrics"]["monthly_avg"]
        h_avg = results["hybrid"][label]["metrics"]["monthly_avg"]
        delta = h_avg - c_avg
        pct = (delta / abs(c_avg) * 100) if c_avg != 0 else 0
        print(f"  {label:<16} ${c_avg:>+12,.0f} ${h_avg:>+12,.0f} ${delta:>+9,.0f} {pct:>+7.1f}%")

    # Additional metrics for Y2
    print(f"\n  {'Metric':<16} {'Current':>13} {'Hybrid':>13} {'Winner':>10}")
    print(f"  {'─'*16} {'─'*13} {'─'*13} {'─'*10}")
    c_y2 = results["current"]["Y2"]["metrics"]
    h_y2 = results["hybrid"]["Y2"]["metrics"]
    metric_rows = [
        ("Trades/month", f"{c_y2['n']/c_y2['n_months']:.0f}", f"{h_y2['n']/h_y2['n_months']:.0f}",
         "hybrid" if h_y2['n']/h_y2['n_months'] > c_y2['n']/c_y2['n_months'] else "current"),
        ("Win rate", f"{c_y2['wr']:.1f}%", f"{h_y2['wr']:.1f}%",
         "hybrid" if h_y2['wr'] > c_y2['wr'] else "current"),
        ("Worst day", f"${c_y2['worst_day']:>+,.0f}", f"${h_y2['worst_day']:>+,.0f}",
         "hybrid" if h_y2['worst_day'] > c_y2['worst_day'] else "current"),
        ("Max DD", f"${c_y2['max_dd']:>+,.0f}", f"${h_y2['max_dd']:>+,.0f}",
         "hybrid" if c_y2['max_dd'] < h_y2['max_dd'] else "current"),
        ("Sharpe", f"{c_y2['sharpe']:.1f}", f"{h_y2['sharpe']:.1f}",
         "hybrid" if h_y2['sharpe'] > c_y2['sharpe'] else "current"),
    ]
    for label, cv, hv, winner in metric_rows:
        print(f"  {label:<16} {cv:>13} {hv:>13} {winner:>10}")

    # ═════════════════════════════════════════════════════════════════
    # STRATEGY BREAKDOWN (hybrid)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  STRATEGY BREAKDOWN (hybrid)")
    print("━" * 70)

    print(f"\n  {'Strategy':<10} {'Y2 $/mo':>10} {'Blind/mo':>10} {'Trades/mo':>10} {'Win rate':>9}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*9}")
    y2_total_pnl = 0; bl_total_pnl = 0; y2_total_n = 0; bl_total_n = 0
    for name in ["RSI", "IB", "MOM"]:
        y2s = results["hybrid"]["Y2"]["per_strat"][name]
        bls = results["hybrid"]["Blind"]["per_strat"][name]
        y2_total_pnl += y2s["pnl"]; bl_total_pnl += bls["pnl"]
        y2_total_n += y2s["n"]; bl_total_n += bls["n"]
        print(f"  {name:<10} ${y2s['monthly_avg']:>+9,.0f} ${bls['monthly_avg']:>+9,.0f} "
              f"{y2s['n']/y2s['n_months']:>10.0f} {y2s['wr']:>8.1f}%")
    h_y2m = results["hybrid"]["Y2"]["metrics"]
    h_blm = results["hybrid"]["Blind"]["metrics"]
    print(f"  {'TOTAL':<10} ${h_y2m['monthly_avg']:>+9,.0f} ${h_blm['monthly_avg']:>+9,.0f} "
          f"{h_y2m['n']/h_y2m['n_months']:>10.0f} {h_y2m['wr']:>8.1f}%")

    # Current breakdown for comparison
    print(f"\n  STRATEGY BREAKDOWN (current):")
    print(f"  {'Strategy':<10} {'Y2 $/mo':>10} {'Blind/mo':>10} {'Trades/mo':>10} {'Win rate':>9}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*9}")
    for name in ["RSI", "IB", "MOM"]:
        y2s = results["current"]["Y2"]["per_strat"][name]
        bls = results["current"]["Blind"]["per_strat"][name]
        print(f"  {name:<10} ${y2s['monthly_avg']:>+9,.0f} ${bls['monthly_avg']:>+9,.0f} "
              f"{y2s['n']/y2s['n_months']:>10.0f} {y2s['wr']:>8.1f}%")

    # ═════════════════════════════════════════════════════════════════
    # MONTH-BY-MONTH
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  MONTH-BY-MONTH (all periods)")
    print("━" * 70)

    all_cur_monthly = defaultdict(float)
    all_hyb_monthly = defaultdict(float)
    for label in ["Blind", "Y1", "Y2"] + (["8yr"] if has_8yr else []):
        # Only use non-8yr for month-by-month since 8yr overlaps
        if label == "8yr":
            continue
        for m, v in results["current"][label]["metrics"]["monthly"].items():
            all_cur_monthly[m] = v  # use = not += since periods don't overlap
        for m, v in results["hybrid"][label]["metrics"]["monthly"].items():
            all_hyb_monthly[m] = v

    all_months = sorted(set(list(all_cur_monthly.keys()) + list(all_hyb_monthly.keys())))

    print(f"\n  {'Month':<10} {'Current':>10} {'Hybrid':>10} {'Delta':>10} {'':>4}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*4}")
    flags = {"hybrid_lost_cur_won": [], "hybrid_won_cur_lost": []}
    for m in all_months:
        cv = all_cur_monthly.get(m, 0)
        hv = all_hyb_monthly.get(m, 0)
        delta = hv - cv
        mark = "+" if hv > 0 else "-"
        flag = ""
        if hv < 0 and cv > 0:
            flag = " <<"
            flags["hybrid_lost_cur_won"].append(m)
        elif hv > 0 and cv < 0:
            flag = " !!"
            flags["hybrid_won_cur_lost"].append(m)
        print(f"  {m:<10} ${cv:>+9,.0f} ${hv:>+9,.0f} ${delta:>+9,.0f}  {mark}{flag}")

    if flags["hybrid_lost_cur_won"]:
        print(f"\n  << CAUTION: Hybrid lost but current was profitable in: {', '.join(flags['hybrid_lost_cur_won'])}")
    if flags["hybrid_won_cur_lost"]:
        print(f"  !! GOOD: Hybrid won where current lost in: {', '.join(flags['hybrid_won_cur_lost'])}")

    # ═════════════════════════════════════════════════════════════════
    # SLIPPAGE SENSITIVITY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  SLIPPAGE SENSITIVITY (hybrid)")
    print("━" * 70)

    print(f"\n  {'Slippage':<10} {'Y2 $/mo':>10} {'Blind/mo':>10} {'Win rate':>9} {'Sharpe':>7}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*9} {'─'*7}")
    for slip in [2, 3, 4]:
        y2_trades = run_with_slippage(yr2, HYBRID, slip)
        bl_trades = run_with_slippage(bl, HYBRID, slip)
        y2m = full_metrics(y2_trades)
        blm = full_metrics(bl_trades)
        print(f"  {slip} ticks    ${y2m['monthly_avg']:>+9,.0f} ${blm['monthly_avg']:>+9,.0f} "
              f"{y2m['wr']:>8.1f}% {y2m['sharpe']:>6.1f}")

    # Degradation per tick
    y2_2t = full_metrics(run_with_slippage(yr2, HYBRID, 2))
    y2_4t = full_metrics(run_with_slippage(yr2, HYBRID, 4))
    if y2_2t["pnl"] != 0:
        deg = (y2_4t["pnl"] - y2_2t["pnl"]) / abs(y2_2t["pnl"]) * 100
        print(f"\n  Degradation from 2→4 ticks: {deg:+.1f}% of Y2 P&L")

    # ═════════════════════════════════════════════════════════════════
    # SL=10 SAFETY CHECK
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  SL=10 SAFETY CHECK")
    print("━" * 70)

    for label, df in [("Y2", yr2), ("Blind", bl)]:
        _, strats = run_system(df, HYBRID)
        for name in ["RSI", "IB"]:
            trades = strats[name]
            losers = [t for t in trades if t.net_pnl < 0]
            bar1_stops = [t for t in losers if t.bars_held == 1 and t.reason == "stop_loss"]
            total_losers = len(losers)
            b1_pct = len(bar1_stops) / total_losers * 100 if total_losers > 0 else 0
            print(f"  {label} {name}: {len(bar1_stops)}/{total_losers} losers stopped on bar 1 ({b1_pct:.0f}%)")
            if b1_pct > 80:
                print(f"    WARNING: SL may be too tight for live execution")

    # ═════════════════════════════════════════════════════════════════
    # MOM ATR=1.4 EFFECT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  MOM ATR=1.4 EFFECT")
    print("━" * 70)

    for label, df in [("Y2", yr2), ("Blind", bl)]:
        # ATR=1.0
        _, s10 = run_system(df, CURRENT)
        m10 = full_metrics(s10["MOM"])
        # ATR=1.4
        _, s14 = run_system(df, HYBRID)
        m14 = full_metrics(s14["MOM"])
        trades_mo_10 = m10["n"] / m10["n_months"] if m10["n_months"] > 0 else 0
        trades_mo_14 = m14["n"] / m14["n_months"] if m14["n_months"] > 0 else 0
        reduction = (1 - trades_mo_14 / trades_mo_10) * 100 if trades_mo_10 > 0 else 0
        print(f"\n  {label}:")
        print(f"    ATR=1.0: {trades_mo_10:.0f} trades/mo, {m10['wr']:.1f}% WR, ${m10['monthly_avg']:>+,.0f}/mo")
        print(f"    ATR=1.4: {trades_mo_14:.0f} trades/mo, {m14['wr']:.1f}% WR, ${m14['monthly_avg']:>+,.0f}/mo")
        print(f"    Reduction: {reduction:.0f}% fewer trades")
        if m14['wr'] > m10['wr']:
            print(f"    Higher threshold IMPROVED signal quality (+{m14['wr'] - m10['wr']:.1f}% WR)")
        else:
            print(f"    Higher threshold DID NOT improve win rate ({m14['wr'] - m10['wr']:+.1f}% WR)")

    # ═════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  MONTE CARLO (5,000 sims, Y2 trades)")
    print("━" * 70)

    mc_cur_base = run_mc(results["current"]["Y2"]["trades"], 5000, 1.0)
    mc_cur_70   = run_mc(results["current"]["Y2"]["trades"], 5000, 0.70)
    mc_hyb_base = run_mc(results["hybrid"]["Y2"]["trades"], 5000, 1.0)
    mc_hyb_70   = run_mc(results["hybrid"]["Y2"]["trades"], 5000, 0.70)

    print(f"\n  {'System':<20} {'Pass':>7} {'Blowup':>7} {'Med Days':>9} {'P95 Days':>9}")
    print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*9} {'─'*9}")
    print(f"  {'Current baseline':<20} {mc_cur_base['pass_rate']:>6.0%} {mc_cur_base['blowup']:>6.0%} "
          f"{mc_cur_base['med_days']:>9} {mc_cur_base['p95_days']:>9}")
    print(f"  {'Current 70%':<20} {mc_cur_70['pass_rate']:>6.0%} {mc_cur_70['blowup']:>6.0%} "
          f"{mc_cur_70['med_days']:>9} {mc_cur_70['p95_days']:>9}")
    print(f"  {'Hybrid baseline':<20} {mc_hyb_base['pass_rate']:>6.0%} {mc_hyb_base['blowup']:>6.0%} "
          f"{mc_hyb_base['med_days']:>9} {mc_hyb_base['p95_days']:>9}")
    print(f"  {'Hybrid 70%':<20} {mc_hyb_70['pass_rate']:>6.0%} {mc_hyb_70['blowup']:>6.0%} "
          f"{mc_hyb_70['med_days']:>9} {mc_hyb_70['p95_days']:>9}")

    # ═════════════════════════════════════════════════════════════════
    # PROP FIRM CHECK (LucidFlex 150K)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  PROP FIRM CHECK — LucidFlex 150K (hybrid)")
    print("━" * 70)

    h_y2m = results["hybrid"]["Y2"]["metrics"]
    h_blm = results["hybrid"]["Blind"]["metrics"]
    worst_day = min(h_y2m["worst_day"], h_blm["worst_day"])
    max_dd = min(h_y2m["max_dd"], h_blm["max_dd"])
    consistency = max(h_y2m.get("best_day", 0) / h_y2m["pnl"] * 100 if h_y2m["pnl"] > 0 else 100,
                      h_blm.get("best_day", 0) / h_blm["pnl"] * 100 if h_blm["pnl"] > 0 else 100)

    print(f"\n  Worst day:   ${worst_day:>+10,.0f}  vs  -$3,000 limit  {'PASS' if worst_day > -3000 else 'FAIL'}")
    print(f"  Max DD:      ${max_dd:>+10,.0f}  vs  -$4,500 MLL    {'PASS' if max_dd > -4500 else 'FAIL'}")
    print(f"  Consistency: {consistency:.1f}%  (max single day as % of total)")

    # ═════════════════════════════════════════════════════════════════
    # TRADINGVIEW VALIDATION
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("  TRADINGVIEW VALIDATION")
    print("━" * 70)

    # Extract Jan-Mar 2026 from hybrid Y2
    jan_mar_monthly = {m: v for m, v in results["hybrid"]["Y2"]["metrics"]["monthly"].items()
                       if m >= "2026-01" and m <= "2026-03"}
    python_jan_mar = sum(jan_mar_monthly.values())
    tv_reported = 15000  # from TradingView (single position)

    print(f"\n  TradingView (single position, hybrid params): ~${tv_reported:,.0f} for Jan-Mar 2026")
    print(f"  Python (3 positions, hybrid params):           ${python_jan_mar:>+,.0f} for Jan-Mar 2026")
    for m in sorted(jan_mar_monthly.keys()):
        print(f"    {m}: ${jan_mar_monthly[m]:>+,.0f}")
    ratio = python_jan_mar / tv_reported if tv_reported > 0 else 0
    print(f"  Expected ratio ~3x. Actual ratio: {ratio:.1f}x")
    match = "YES" if 2.0 <= ratio <= 4.5 else "NO"
    print(f"  Match: {match}")

    # ═════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  VERDICT")
    print("═" * 70)

    c_y2_avg = results["current"]["Y2"]["metrics"]["monthly_avg"]
    h_y2_avg = results["hybrid"]["Y2"]["metrics"]["monthly_avg"]
    c_bl_avg = results["current"]["Blind"]["metrics"]["monthly_avg"]
    h_bl_avg = results["hybrid"]["Blind"]["metrics"]["monthly_avg"]

    # MOM analysis summary
    _, s10_y2 = run_system(yr2, CURRENT)
    _, s14_y2 = run_system(yr2, HYBRID)
    m10_y2 = full_metrics(s10_y2["MOM"])
    m14_y2 = full_metrics(s14_y2["MOM"])
    mom_trades_10 = m10_y2["n"] / m10_y2["n_months"] if m10_y2["n_months"] > 0 else 0
    mom_trades_14 = m14_y2["n"] / m14_y2["n_months"] if m14_y2["n_months"] > 0 else 0
    mom_reduction = (1 - mom_trades_14 / mom_trades_10) * 100 if mom_trades_10 > 0 else 0
    mom_wr_delta = m14_y2["wr"] - m10_y2["wr"]
    mom_quality = "improved" if mom_wr_delta > 0 else "hurt"

    # SL safety
    _, strats_y2 = run_system(yr2, HYBRID)
    rsi_losers = [t for t in strats_y2["RSI"] if t.net_pnl < 0]
    rsi_b1 = sum(1 for t in rsi_losers if t.bars_held == 1 and t.reason == "stop_loss")
    rsi_b1_pct = rsi_b1 / len(rsi_losers) * 100 if rsi_losers else 0
    ib_losers = [t for t in strats_y2["IB"] if t.net_pnl < 0]
    ib_b1 = sum(1 for t in ib_losers if t.bars_held == 1 and t.reason == "stop_loss")
    ib_b1_pct = ib_b1 / len(ib_losers) * 100 if ib_losers else 0
    sl_safe = "safe" if rsi_b1_pct < 80 and ib_b1_pct < 80 else "risky"

    # Slippage sensitivity
    y2_s2 = full_metrics(run_with_slippage(yr2, HYBRID, 2))
    y2_s4 = full_metrics(run_with_slippage(yr2, HYBRID, 4))
    slip_deg = abs((y2_s4["pnl"] - y2_s2["pnl"]) / y2_s2["pnl"] * 100) if y2_s2["pnl"] != 0 else 0
    slip_label = "low" if slip_deg < 20 else "high"

    # Recommendation
    if h_bl_avg > c_bl_avg and h_y2_avg > c_y2_avg and mc_hyb_base["pass_rate"] >= mc_cur_base["pass_rate"]:
        recommendation = "DEPLOY HYBRID"
    elif h_bl_avg > c_bl_avg * 0.95:
        recommendation = "HYBRID is comparable — deploy if slippage sensitivity is acceptable"
    else:
        recommendation = "STICK WITH CURRENT"

    print(f"\n  The hybrid params produce ${h_y2_avg:,.0f}/month on Y2 OOS vs "
          f"${c_y2_avg:,.0f} (current v3).")
    print(f"  On blind data: ${h_bl_avg:,.0f} vs ${c_bl_avg:,.0f}.")
    print(f"  The MOM ATR=1.4 {mom_quality} MOM signal quality — "
          f"{mom_reduction:.0f}% fewer trades but {mom_wr_delta:+.1f}% win rate.")
    print(f"  SL=10 on RSI/IB is {sl_safe} with {rsi_b1_pct:.0f}%/{ib_b1_pct:.0f}% bar-1 stopouts.")
    print(f"  Slippage sensitivity: {slip_label} — degrades {slip_deg:.0f}% from 2→4 ticks.")
    print(f"\n  Recommendation: {recommendation}")

    # ═════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════
    report = {
        "timestamp": str(datetime.now()),
        "current_params": {k: dict(v) for k, v in CURRENT.items()},
        "hybrid_params": {k: dict(v) for k, v in HYBRID.items()},
        "head_to_head": {},
        "strategy_breakdown_hybrid": {},
        "mc": {
            "current_baseline": mc_cur_base,
            "current_70": mc_cur_70,
            "hybrid_baseline": mc_hyb_base,
            "hybrid_70": mc_hyb_70,
        },
        "slippage_sensitivity": {},
        "sl10_safety": {
            "rsi_bar1_pct": rsi_b1_pct,
            "ib_bar1_pct": ib_b1_pct,
        },
        "mom_atr_effect": {
            "trades_reduction_pct": mom_reduction,
            "wr_change": mom_wr_delta,
            "quality": mom_quality,
        },
        "tv_validation": {
            "tv_reported": tv_reported,
            "python_jan_mar": python_jan_mar,
            "ratio": ratio,
            "match": match,
        },
        "recommendation": recommendation,
    }

    for label in ["Y1", "Y2", "Blind"] + (["8yr"] if has_8yr else []):
        cm = results["current"][label]["metrics"]
        hm = results["hybrid"][label]["metrics"]
        report["head_to_head"][label] = {
            "current": {k: v for k, v in cm.items() if k != "monthly"},
            "hybrid": {k: v for k, v in hm.items() if k != "monthly"},
        }

    for name in ["RSI", "IB", "MOM"]:
        report["strategy_breakdown_hybrid"][name] = {
            "y2": {k: v for k, v in results["hybrid"]["Y2"]["per_strat"][name].items() if k != "monthly"},
            "blind": {k: v for k, v in results["hybrid"]["Blind"]["per_strat"][name].items() if k != "monthly"},
        }

    for slip in [2, 3, 4]:
        y2t = run_with_slippage(yr2, HYBRID, slip)
        blt = run_with_slippage(bl, HYBRID, slip)
        y2m_ = full_metrics(y2t)
        blm_ = full_metrics(blt)
        report["slippage_sensitivity"][f"{slip}_ticks"] = {
            "y2_monthly_avg": y2m_["monthly_avg"],
            "blind_monthly_avg": blm_["monthly_avg"],
            "y2_wr": y2m_["wr"],
            "y2_sharpe": y2m_["sharpe"],
        }

    out = REPORTS_DIR / "htf_swing_v3_hybrid.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Total time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 70)


if __name__ == "__main__":
    main()
