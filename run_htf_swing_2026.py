#!/usr/bin/env python3
"""
HTF Swing v3 — Hybrid v2 — 2026 YTD Backtest
Period: Jan 1 – Mar 28, 2026
"""

import json
import time as _time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, date
from pathlib import Path

import numpy as np
import polars as pl

from run_htf_swing import (
    extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE, SLIP_TICKS, COMM_PER_SIDE, EXCH_PER_SIDE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

# ── Config ─────────────────────────────────────────────────────
FLATTEN_TIME = 1645       # LucidFlex 150K
CONTRACTS = 3             # per strategy
INITIAL_CAPITAL = 150_000

# Date range for 2026 YTD
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 28)

# Hybrid v2 parameters
HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}

SLIP_PTS = SLIP_TICKS * TICK_SIZE  # 0.50

OUT_DIR = Path("reports/2026_ytd")


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def rt_cost(contracts):
    return 2 * (COMM_PER_SIDE + EXCH_PER_SIDE) * contracts


def run_system(df):
    """Run all 3 strategies on df, return combined + per-strategy trades."""
    o, h, l, c, ts, hm = extract_arrays(df)
    per_strat = {}

    # RSI Extremes
    p = HYBRID_V2["RSI"]
    sigs = sig_rsi_extreme(df, p["period"], p["ob"], p["os"])
    per_strat["RSI"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], CONTRACTS, "RSI", FLATTEN_TIME)

    # IB Breakout
    p = HYBRID_V2["IB"]
    sigs = sig_ib_breakout(df, p["ib_filter"])[0]
    per_strat["IB"] = backtest(o, h, l, c, ts, hm, sigs,
                                pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                p["hold"], CONTRACTS, "IB", FLATTEN_TIME)

    # Momentum Bars
    p = HYBRID_V2["MOM"]
    sigs = sig_momentum_bar(df, p["atr_mult"], p["vol_mult"])[0]
    per_strat["MOM"] = backtest(o, h, l, c, ts, hm, sigs,
                                 pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                                 p["hold"], CONTRACTS, "MOM", FLATTEN_TIME)

    all_trades = []
    for t in per_strat.values():
        all_trades.extend(t)
    all_trades.sort(key=lambda t: str(t.entry_time))
    return all_trades, per_strat


def filter_2026(trades):
    """Keep only trades with entry in the 2026 date range."""
    filtered = []
    for t in trades:
        entry_date = str(t.entry_time)[:10]
        if entry_date >= str(START_DATE) and entry_date <= str(END_DATE):
            filtered.append(t)
    return filtered


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


def equity_curve(trades, capital):
    eq = [capital]
    for t in sorted(trades, key=lambda t: str(t.exit_time)):
        eq.append(eq[-1] + t.net_pnl)
    return eq


def max_drawdown(eq):
    peak = eq[0]
    max_dd = 0
    max_dd_idx = 0
    for i, v in enumerate(eq):
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_idx = i
    return max_dd, max_dd_idx


def sharpe(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    arr = np.array(list(daily_pnls.values()))
    if arr.std() == 0:
        return 0.0
    return (arr.mean() / arr.std()) * np.sqrt(252)


def profit_factor(trades):
    gross_win = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    return gross_win / gross_loss if gross_loss > 0 else float('inf')


def trade_cost(contracts):
    slip = SLIP_PTS * POINT_VALUE * contracts * 2  # entry + exit
    comm = COMM_PER_SIDE * contracts * 2
    exch = EXCH_PER_SIDE * contracts * 2
    return slip, comm, exch


def print_report(all_trades, per_strat, daily, actual_end):
    """Print full console report."""
    winners = [t for t in all_trades if t.net_pnl > 0]
    losers = [t for t in all_trades if t.net_pnl <= 0]
    total_pnl = sum(t.net_pnl for t in all_trades)
    slip_cost, comm_cost, exch_cost = trade_cost(CONTRACTS)
    total_cost_per = slip_cost + comm_cost + exch_cost
    total_costs = total_cost_per * len(all_trades)
    gross_pnl = total_pnl + total_costs

    avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
    avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    eq = equity_curve(all_trades, INITIAL_CAPITAL)
    dd, dd_idx = max_drawdown(eq)

    daily_vals = list(daily.values())
    trading_days = len(daily)

    print()
    print("═" * 65)
    print("  HTF SWING V3 — HYBRID V2 — 2026 YTD BACKTEST")
    print(f"  Period: Jan 1 – {actual_end.strftime('%b %d')}, 2026")
    print("═" * 65)

    print(f"\n  OVERALL SUMMARY")
    print(f"  {'─' * 55}")
    print(f"  Total trades:        {len(all_trades):>8,}")
    print(f"  Winners:             {len(winners):>8,}   ({len(winners)/len(all_trades)*100:.1f}%)" if all_trades else "")
    print(f"  Losers:              {len(losers):>8,}   ({len(losers)/len(all_trades)*100:.1f}%)" if all_trades else "")
    print(f"  Win rate:            {len(winners)/len(all_trades)*100:>7.1f}%" if all_trades else "")
    print(f"  {'─' * 55}")
    print(f"  Net P&L:             ${total_pnl:>+12,.2f}")
    print(f"  Gross P&L:           ${gross_pnl:>+12,.2f}")
    print(f"  Total costs:         ${total_costs:>12,.2f}")
    print(f"  {'─' * 55}")
    print(f"  Avg winner:          ${avg_win:>+12,.2f}")
    print(f"  Avg loser:           ${avg_loss:>+12,.2f}")
    print(f"  R:R ratio:           {rr:>12.2f}")
    print(f"  Profit factor:       {profit_factor(all_trades):>12.2f}")
    print(f"  Sharpe (annualized): {sharpe(daily):>12.2f}")
    print(f"  Max drawdown:        ${dd:>12,.2f}")
    print(f"  {'─' * 55}")
    print(f"  Trading days:        {trading_days:>8}")
    print(f"  Avg trades/day:      {len(all_trades)/trading_days:>8.1f}" if trading_days else "")

    # Per-strategy
    print(f"\n  PER-STRATEGY BREAKDOWN")
    print(f"  {'Strat':<6} {'Trades':>7} {'WR':>7} {'Net P&L':>12} {'Avg Win':>10} {'Avg Loss':>10} {'PF':>6}")
    print(f"  {'─'*6} {'─'*7} {'─'*7} {'─'*12} {'─'*10} {'─'*10} {'─'*6}")
    for name in ["RSI", "IB", "MOM"]:
        st = per_strat.get(name, [])
        if not st:
            print(f"  {name:<6} {'0':>7}")
            continue
        sw = [t for t in st if t.net_pnl > 0]
        sl = [t for t in st if t.net_pnl <= 0]
        wr = len(sw) / len(st) * 100 if st else 0
        pnl = sum(t.net_pnl for t in st)
        aw = np.mean([t.net_pnl for t in sw]) if sw else 0
        al = np.mean([t.net_pnl for t in sl]) if sl else 0
        pf = profit_factor(st)
        print(f"  {name:<6} {len(st):>7} {wr:>6.1f}% ${pnl:>+10,.0f} ${aw:>+8,.0f} ${al:>+8,.0f} {pf:>6.2f}")

    # Monthly
    print(f"\n  MONTHLY BREAKDOWN")
    months = sorted(monthly_pnl(all_trades).items())
    print(f"  {'Month':<8} {'Trades':>7} {'Win':>5} {'Loss':>5} {'WR':>7} {'Net P&L':>12} {'Best Day':>10} {'Worst Day':>10}")
    print(f"  {'─'*8} {'─'*7} {'─'*5} {'─'*5} {'─'*7} {'─'*12} {'─'*10} {'─'*10}")
    for mo, _ in months:
        mt = [t for t in all_trades if str(t.entry_time)[:7] == mo]
        mw = [t for t in mt if t.net_pnl > 0]
        ml = [t for t in mt if t.net_pnl <= 0]
        mpnl = sum(t.net_pnl for t in mt)
        wr = len(mw) / len(mt) * 100 if mt else 0
        md = {d: v for d, v in daily.items() if d[:7] == mo}
        best = max(md.values()) if md else 0
        worst = min(md.values()) if md else 0
        print(f"  {mo:<8} {len(mt):>7} {len(mw):>5} {len(ml):>5} {wr:>6.1f}% ${mpnl:>+10,.0f} ${best:>+8,.0f} ${worst:>+8,.0f}")

    # Daily P&L distribution
    print(f"\n  DAILY P&L DISTRIBUTION")
    print(f"  {'─' * 55}")
    if daily_vals:
        arr = np.array(daily_vals)
        pct_pos = sum(1 for v in daily_vals if v > 0) / len(daily_vals) * 100
        print(f"  Mean:                ${arr.mean():>+12,.2f}")
        print(f"  Median:              ${np.median(arr):>+12,.2f}")
        print(f"  Std dev:             ${arr.std():>12,.2f}")
        print(f"  Best day:            ${arr.max():>+12,.2f}")
        print(f"  Worst day:           ${arr.min():>+12,.2f}")
        print(f"  Profitable days:     {pct_pos:>11.1f}%")

    # Equity curve
    print(f"\n  EQUITY CURVE")
    print(f"  {'─' * 55}")
    print(f"  Starting equity:     ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Ending equity:       ${eq[-1]:>12,.2f}")
    print(f"  Peak equity:         ${max(eq):>12,.2f}")
    print(f"  Max drawdown:        ${dd:>12,.2f}")

    print(f"\n{'═' * 65}")


def export_trades_csv(trades, path):
    """Export trade-level CSV matching ANTHONY format."""
    rows = []
    cum_pnl = 0
    # Daily tracking
    daily_pnl_tracker = defaultdict(float)
    daily_trade_counter = defaultdict(int)

    for i, t in enumerate(sorted(trades, key=lambda x: str(x.entry_time))):
        entry_date = str(t.entry_time)[:10]
        daily_trade_counter[entry_date] += 1
        daily_before = daily_pnl_tracker[entry_date]

        slip_cost = SLIP_PTS * POINT_VALUE * t.contracts * 2
        comm_cost = COMM_PER_SIDE * t.contracts * 2
        exch_fee = EXCH_PER_SIDE * t.contracts * 2
        total_cost = slip_cost + comm_cost + exch_fee
        gross_pnl = t.net_pnl + total_cost
        cum_pnl += t.net_pnl
        points = (t.exit_px - t.entry_px) * t.direction

        daily_pnl_tracker[entry_date] += t.net_pnl

        # Determine SL/TP distances based on strategy
        p = HYBRID_V2.get(t.strategy, {})
        sl_pts = p.get("sl_pts", 0)
        tp_pts = p.get("tp_pts", 0)

        entry_h = 0
        entry_ts = str(t.entry_time)
        if hasattr(t.entry_time, 'hour'):
            entry_h = t.entry_time.hour

        entry_wd = ""
        if hasattr(t.entry_time, 'strftime'):
            entry_wd = t.entry_time.strftime("%A")

        month = str(t.entry_time)[5:7] if len(str(t.entry_time)) >= 7 else ""
        is_first_hour = 1 if entry_h == 9 or (entry_h == 10 and hasattr(t.entry_time, 'minute') and t.entry_time.minute < 30) else 0

        rows.append({
            "trade_number": i + 1,
            "strategy": t.strategy,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "contracts": t.contracts,
            "entry_bar_timestamp": str(t.entry_time),
            "exit_bar_timestamp": str(t.exit_time),
            "entry_price": round(t.entry_px, 2),
            "exit_price": round(t.exit_px, 2),
            "exit_reason": t.reason,
            "gross_pnl": round(gross_pnl, 2),
            "slippage_cost": round(slip_cost, 2),
            "commission_cost": round(comm_cost, 2),
            "exchange_fee": round(exch_fee, 2),
            "total_cost": round(total_cost, 2),
            "net_pnl": round(t.net_pnl, 2),
            "cumulative_pnl": round(cum_pnl, 2),
            "bars_held": t.bars_held,
            "points_moved": round(points, 2),
            "sl_distance_points": sl_pts,
            "tp_distance_points": tp_pts,
            "day_of_week": entry_wd,
            "hour_of_entry": entry_h,
            "month": month,
            "is_first_hour": is_first_hour,
        })

    df = pl.DataFrame(rows)
    df.write_csv(path)
    print(f"  Trades CSV: {path} ({len(rows)} rows)")


def export_monthly_csv(trades, daily_map, path):
    rows = []
    months = sorted(set(str(t.entry_time)[:7] for t in trades))
    cum_pnl = 0
    for mo in months:
        mt = [t for t in trades if str(t.entry_time)[:7] == mo]
        mw = [t for t in mt if t.net_pnl > 0]
        ml = [t for t in mt if t.net_pnl <= 0]
        rsi_t = [t for t in mt if t.strategy == "RSI"]
        ib_t = [t for t in mt if t.strategy == "IB"]
        mom_t = [t for t in mt if t.strategy == "MOM"]
        pnl = sum(t.net_pnl for t in mt)
        cum_pnl += pnl
        md = {d: v for d, v in daily_map.items() if d[:7] == mo}
        trading_days = len(md)
        rows.append({
            "month": mo,
            "total_trades": len(mt),
            "rsi_trades": len(rsi_t),
            "ib_trades": len(ib_t),
            "mom_trades": len(mom_t),
            "winners": len(mw),
            "losers": len(ml),
            "win_rate": round(len(mw) / len(mt) * 100, 1) if mt else 0,
            "net_pnl": round(pnl, 2),
            "cumulative_pnl": round(cum_pnl, 2),
            "best_trade": round(max(t.net_pnl for t in mt), 2) if mt else 0,
            "worst_trade": round(min(t.net_pnl for t in mt), 2) if mt else 0,
            "avg_trade": round(pnl / len(mt), 2) if mt else 0,
            "best_day": round(max(md.values()), 2) if md else 0,
            "worst_day": round(min(md.values()), 2) if md else 0,
            "trading_days": trading_days,
            "avg_trades_per_day": round(len(mt) / trading_days, 1) if trading_days else 0,
            "avg_bars_held": round(np.mean([t.bars_held for t in mt]), 1) if mt else 0,
        })
    df = pl.DataFrame(rows)
    df.write_csv(path)
    print(f"  Monthly CSV: {path} ({len(rows)} rows)")


def export_daily_csv(trades, daily_map, path):
    rows = []
    cum = 0
    for d in sorted(daily_map.keys()):
        dt = [t for t in trades if str(t.entry_time)[:10] == d]
        dw = [t for t in dt if t.net_pnl > 0]
        dl = [t for t in dt if t.net_pnl <= 0]
        rsi_t = [t for t in dt if t.strategy == "RSI"]
        ib_t = [t for t in dt if t.strategy == "IB"]
        mom_t = [t for t in dt if t.strategy == "MOM"]
        pnl = daily_map[d]
        cum += pnl
        wd = ""
        if dt and hasattr(dt[0].entry_time, 'strftime'):
            wd = dt[0].entry_time.strftime("%A")
        rows.append({
            "date": d,
            "day_of_week": wd,
            "total_trades": len(dt),
            "rsi_trades": len(rsi_t),
            "ib_trades": len(ib_t),
            "mom_trades": len(mom_t),
            "winners": len(dw),
            "losers": len(dl),
            "win_rate": round(len(dw) / len(dt) * 100, 1) if dt else 0,
            "net_pnl": round(pnl, 2),
            "cumulative_pnl": round(cum, 2),
        })
    df = pl.DataFrame(rows)
    df.write_csv(path)
    print(f"  Daily CSV: {path} ({len(rows)} rows)")


def export_json(all_trades, per_strat, daily_map, actual_end, path):
    winners = [t for t in all_trades if t.net_pnl > 0]
    losers = [t for t in all_trades if t.net_pnl <= 0]
    total_pnl = sum(t.net_pnl for t in all_trades)
    eq = equity_curve(all_trades, INITIAL_CAPITAL)
    dd, _ = max_drawdown(eq)

    strat_summary = {}
    for name in ["RSI", "IB", "MOM"]:
        st = per_strat.get(name, [])
        sw = [t for t in st if t.net_pnl > 0]
        strat_summary[name] = {
            "trades": len(st),
            "win_rate": round(len(sw) / len(st) * 100, 1) if st else 0,
            "net_pnl": round(sum(t.net_pnl for t in st), 2),
            "profit_factor": round(profit_factor(st), 2),
        }

    report = {
        "period": {"start": str(START_DATE), "end": str(actual_end)},
        "parameters": "hybrid_v2",
        "flatten_time": FLATTEN_TIME,
        "contracts_per_strategy": CONTRACTS,
        "initial_capital": INITIAL_CAPITAL,
        "summary": {
            "total_trades": len(all_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(all_trades) * 100, 1) if all_trades else 0,
            "net_pnl": round(total_pnl, 2),
            "profit_factor": round(profit_factor(all_trades), 2),
            "sharpe": round(sharpe(daily_map), 2),
            "max_drawdown": round(dd, 2),
            "trading_days": len(daily_map),
            "avg_trades_per_day": round(len(all_trades) / len(daily_map), 1) if daily_map else 0,
        },
        "per_strategy": strat_summary,
        "monthly": {mo: round(pnl, 2) for mo, pnl in sorted(monthly_pnl(all_trades).items())},
        "equity": {"start": INITIAL_CAPITAL, "end": round(eq[-1], 2), "peak": round(max(eq), 2)},
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {path}")


def main():
    t0 = _time.time()

    # ── Load data ──────────────────────────────────────────────
    print("\n  Loading data...")
    df_1m = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    df_15m = resample_15m_rth(df_1m)
    del df_1m

    # Check date range
    dates = df_15m["date_et"].to_list()
    all_dates = sorted(set(str(d) for d in dates))
    actual_start = all_dates[0]
    actual_end_str = all_dates[-1]
    print(f"  15m bars: {len(df_15m):,} ({actual_start} → {actual_end_str})")

    # ── Run system on FULL data (signals need warmup) ──────────
    print("\n  Running Hybrid v2 strategies on full data...")
    all_trades_full, per_strat_full = run_system(df_15m)
    print(f"  Full-period trades: {len(all_trades_full):,}")

    # ── Filter to 2026 ────────────────────────────────────────
    all_trades = filter_2026(all_trades_full)
    per_strat = {}
    for name in ["RSI", "IB", "MOM"]:
        per_strat[name] = filter_2026(per_strat_full[name])

    actual_end = START_DATE
    for t in all_trades:
        d = str(t.entry_time)[:10]
        dt = date.fromisoformat(d)
        if dt > actual_end:
            actual_end = dt

    print(f"  2026 YTD trades: {len(all_trades):,} ({START_DATE} → {actual_end})")

    if not all_trades:
        print("\n  ERROR: No trades found in 2026 date range.")
        print("  Check data coverage — full_2yr.parquet may not extend to 2026.")
        return

    # ── Compute metrics ───────────────────────────────────────
    daily_map = daily_pnl(all_trades)

    # ── Print report ──────────────────────────────────────────
    print_report(all_trades, per_strat, daily_map, actual_end)

    # ── Export ────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  EXPORTS")
    print(f"  {'─' * 55}")
    export_trades_csv(all_trades, OUT_DIR / "htf_swing_v3_hybrid_v2_2026_trades.csv")
    export_monthly_csv(all_trades, daily_map, OUT_DIR / "htf_swing_v3_hybrid_v2_2026_monthly.csv")
    export_daily_csv(all_trades, daily_map, OUT_DIR / "htf_swing_v3_hybrid_v2_2026_daily.csv")
    export_json(all_trades, per_strat, daily_map, actual_end,
                OUT_DIR / "htf_swing_v3_hybrid_v2_2026_report.json")

    elapsed = _time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print("═" * 65)


if __name__ == "__main__":
    main()
