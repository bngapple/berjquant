#!/usr/bin/env python3
"""
Export every trade from HTF Swing v3 LucidFlex to CSV.

Three CSVs: all_trades, monthly_summary, daily_summary.

Usage:
    python3 run_htf_swing_export.py
"""

import csv
import gc
import json
import os
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from run_htf_swing import (
    extract_arrays, rt_cost, Trade,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    calc_rsi, calc_atr, calc_ema,
    TICK_SIZE, POINT_VALUE, SLIP_PTS,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS = Path("reports")
_ET = ZoneInfo("US/Eastern")

C = 3  # contracts per strategy
STRATS = {
    "RSI": {"sig": lambda d: sig_rsi_extreme(d, 7, 70, 30), "sl": 60, "tp": 400, "hold": 5},
    "IB":  {"sig": lambda d: sig_ib_breakout(d)[0],          "sl": 80, "tp": 480, "hold": 10},
    "MOM": {"sig": lambda d: sig_momentum_bar(d, 1.0, 1.0)[0], "sl": 60, "tp": 400, "hold": 5},
}
COMM_PER_SIDE = 0.62
EXCH_PER_SIDE = 0.27
SLIP_TICKS = 2
FLATTEN_HHMM = 1645  # LucidFlex


def backtest_detailed(df_15m, signals, sl_ticks, tp_ticks, max_hold, contracts,
                      strategy_name, indicators):
    """Backtest returning detailed trade records with market context."""
    opens = df_15m["open"].to_numpy()
    highs = df_15m["high"].to_numpy()
    lows = df_15m["low"].to_numpy()
    closes = df_15m["close"].to_numpy()
    timestamps = df_15m["timestamp"].to_list()
    hhmm = df_15m["hhmm"].to_numpy()
    n = len(opens)

    cost_comm = COMM_PER_SIDE * 2 * contracts
    cost_exch = EXCH_PER_SIDE * 2 * contracts
    cost_slip_total = SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * contracts
    total_cost = cost_comm + cost_exch + cost_slip_total
    slip = SLIP_TICKS * TICK_SIZE

    sl_pts = sl_ticks * TICK_SIZE
    tp_pts = tp_ticks * TICK_SIZE

    trades = []
    in_pos = False
    direction = 0
    entry_px = 0.0
    entry_bar = 0
    stop_px = 0.0
    target_px = 0.0
    pending = 0
    signal_bar = -1

    for i in range(n):
        h = hhmm[i]

        # Flatten
        if in_pos and h >= FLATTEN_HHMM and i > entry_bar:
            ex_raw = closes[i]
            ex = ex_raw - direction * slip
            gross = (ex - entry_px) * direction * POINT_VALUE * contracts
            trades.append({
                "strategy": strategy_name, "direction": direction, "contracts": contracts,
                "signal_bar": signal_bar, "entry_bar": entry_bar, "exit_bar": i,
                "entry_price": entry_px, "entry_price_raw": opens[entry_bar],
                "exit_price": ex, "exit_price_raw": ex_raw,
                "exit_reason": "EOD",
                "gross_pnl": gross, "slippage_cost": cost_slip_total,
                "commission_cost": cost_comm, "exchange_fee": cost_exch,
                "total_cost": total_cost, "net_pnl": gross - total_cost,
                "bars_held": i - entry_bar,
                "sl_distance_points": sl_pts, "tp_distance_points": tp_pts,
            })
            in_pos = False; pending = 0; continue

        # Execute pending
        if pending != 0 and not in_pos:
            if h >= FLATTEN_HHMM - 15 or not (930 <= h < 1600):
                pending = 0
            else:
                entry_px = opens[i] + int(pending) * slip
                direction = int(pending)
                entry_bar = i
                stop_px = entry_px - direction * sl_pts
                target_px = entry_px + direction * tp_pts
                in_pos = True
                pending = 0

        # Manage (skip entry bar)
        if in_pos and i > entry_bar:
            bh = i - entry_bar
            ex = None; reason = ""; ex_raw = 0

            if direction == 1 and lows[i] <= stop_px:
                ex_raw = stop_px; ex = stop_px - slip; reason = "SL"
            elif direction == -1 and highs[i] >= stop_px:
                ex_raw = stop_px; ex = stop_px + slip; reason = "SL"

            if ex is None:
                if direction == 1 and highs[i] >= target_px:
                    ex_raw = target_px; ex = target_px - slip; reason = "TP"
                elif direction == -1 and lows[i] <= target_px:
                    ex_raw = target_px; ex = target_px + slip; reason = "TP"

            if ex is None and bh >= max_hold:
                ex_raw = closes[i]; ex = closes[i] - direction * slip; reason = "max_hold"

            if ex is not None:
                gross = (ex - entry_px) * direction * POINT_VALUE * contracts
                trades.append({
                    "strategy": strategy_name, "direction": direction, "contracts": contracts,
                    "signal_bar": signal_bar, "entry_bar": entry_bar, "exit_bar": i,
                    "entry_price": entry_px, "entry_price_raw": opens[entry_bar],
                    "exit_price": ex, "exit_price_raw": ex_raw,
                    "exit_reason": reason,
                    "gross_pnl": gross, "slippage_cost": cost_slip_total,
                    "commission_cost": cost_comm, "exchange_fee": cost_exch,
                    "total_cost": total_cost, "net_pnl": gross - total_cost,
                    "bars_held": i - entry_bar,
                    "sl_distance_points": sl_pts, "tp_distance_points": tp_pts,
                })
                in_pos = False

        # New signal
        if not in_pos and i < len(signals) and signals[i] != 0:
            if 930 <= h < (FLATTEN_HHMM - 15):
                pending = signals[i]
                signal_bar = i

    return trades


def get_data_period(ts_str):
    """Classify a timestamp into a data period."""
    y = ts_str[:4]
    ym = ts_str[:7]
    if y in ("2017", "2018", "2019") and ym < "2019-06":
        return "2018-2019 (NQ, pre-MNQ)"
    elif ym < "2022-01":
        return "2020-2021 (extended blind)"
    elif ym < "2024-04":
        return "2022-2024 (blind test)"
    elif ym < "2025-04":
        return "2024-2025 (Y1 optimization)"
    else:
        return "2025-2026 (Y2 OOS)"


def get_validation(ts_str):
    ym = ts_str[:7]
    if ym < "2024-04":
        return "blind"
    elif ym < "2025-04":
        return "in_sample"
    else:
        return "out_of_sample"


def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING v3 — COMPLETE TRADE EXPORT")
    print("═" * 70)

    # Load 8-year data
    print("\n  Loading all data ...")
    try:
        df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
        df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
        df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
        combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        combined = combined.filter(pl.col("close") > 0)
        del df1, df2, df3
    except Exception as e:
        print(f"  8yr load failed ({e}), using blind + main ...")
        df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
        df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
        combined = pl.concat([df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        combined = combined.filter(pl.col("close") > 0)
        del df2, df3

    gc.collect()
    print(f"  Combined 1m: {len(combined):,} bars")

    df_15m = resample_15m_rth(combined)
    del combined; gc.collect()
    print(f"  15m RTH: {len(df_15m):,} bars")
    print(f"  Range: {df_15m['timestamp'].min()} → {df_15m['timestamp'].max()}")

    # Pre-compute indicators on full dataset
    print("  Computing indicators ...")
    closes = df_15m["close"].to_numpy()
    highs = df_15m["high"].to_numpy()
    lows = df_15m["low"].to_numpy()
    opens = df_15m["open"].to_numpy()
    volumes = df_15m["volume"].to_numpy().astype(float)
    timestamps = df_15m["timestamp"].to_list()
    hhmm_arr = df_15m["hhmm"].to_numpy()

    rsi7 = calc_rsi(closes, 7)
    atr14 = calc_atr(highs, lows, closes, 14)
    ema21 = calc_ema(closes, 21)
    avg_vol20 = np.full(len(closes), np.nan)
    for i in range(20, len(closes)):
        avg_vol20[i] = np.mean(volumes[i-20:i])

    indicators = {"rsi7": rsi7, "atr14": atr14, "ema21": ema21, "avg_vol20": avg_vol20}

    # Run each strategy
    print("  Running backtests ...")
    all_raw_trades = []
    for name, s in STRATS.items():
        print(f"    {name} ...", end=" ", flush=True)
        sigs = s["sig"](df_15m)
        trades = backtest_detailed(df_15m, sigs, s["sl"], s["tp"], s["hold"], C, name, indicators)
        all_raw_trades.extend(trades)
        print(f"{len(trades)} trades")
        gc.collect()

    # Sort by entry time
    all_raw_trades.sort(key=lambda t: t["entry_bar"])

    # ── Build full CSV rows ──
    print(f"\n  Building CSV ({len(all_raw_trades)} trades) ...")

    cumulative = 0.0
    daily_pnl_tracker = {}
    daily_trade_counter = {}

    csv_rows = []
    for idx, t in enumerate(all_raw_trades):
        eb = t["entry_bar"]
        sb = t["signal_bar"]
        xb = t["exit_bar"]

        entry_ts = str(timestamps[eb])[:19]
        signal_ts = str(timestamps[sb])[:19] if sb >= 0 else entry_ts
        exit_ts = str(timestamps[xb])[:19]

        net = t["net_pnl"]
        cumulative += net

        # Daily tracking
        day = entry_ts[:10]
        if day not in daily_pnl_tracker:
            daily_pnl_tracker[day] = 0.0
            daily_trade_counter[day] = 0
        daily_before = daily_pnl_tracker[day]
        daily_pnl_tracker[day] += net
        daily_trade_counter[day] += 1
        daily_trade_num = daily_trade_counter[day]

        # Time context
        h_et = hhmm_arr[eb]
        hour_entry = h_et // 100
        dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        try:
            ts_obj = timestamps[eb]
            if hasattr(ts_obj, 'weekday'):
                dow = dow_map.get(ts_obj.weekday(), "Unknown")
            else:
                dow = "Unknown"
        except:
            dow = "Unknown"

        pts_moved = abs(t["exit_price"] - t["entry_price"])

        row = {
            "trade_number": idx + 1,
            "strategy": t["strategy"],
            "direction": "LONG" if t["direction"] == 1 else "SHORT",
            "contracts": t["contracts"],
            "signal_bar_timestamp": signal_ts,
            "entry_bar_timestamp": entry_ts,
            "entry_price": f"{t['entry_price']:.2f}",
            "entry_price_raw": f"{t['entry_price_raw']:.2f}",
            "exit_bar_timestamp": exit_ts,
            "exit_price": f"{t['exit_price']:.2f}",
            "exit_price_raw": f"{t['exit_price_raw']:.2f}",
            "exit_reason": t["exit_reason"],
            "gross_pnl": f"{t['gross_pnl']:.2f}",
            "slippage_cost": f"{t['slippage_cost']:.2f}",
            "commission_cost": f"{t['commission_cost']:.2f}",
            "exchange_fee": f"{t['exchange_fee']:.2f}",
            "total_cost": f"{t['total_cost']:.2f}",
            "net_pnl": f"{net:.2f}",
            "cumulative_pnl": f"{cumulative:.2f}",
            "bars_held": t["bars_held"],
            "hold_time_minutes": t["bars_held"] * 15,
            "points_moved": f"{pts_moved:.2f}",
            "sl_distance_points": f"{t['sl_distance_points']:.2f}",
            "tp_distance_points": f"{t['tp_distance_points']:.2f}",
            "risk_dollars": f"{t['sl_distance_points'] * POINT_VALUE * C:.2f}",
            "reward_dollars": f"{t['tp_distance_points'] * POINT_VALUE * C:.2f}",
            "rsi_at_signal": f"{rsi7[sb]:.2f}" if sb >= 0 and not np.isnan(rsi7[sb]) else "",
            "atr_at_signal": f"{atr14[sb]:.2f}" if sb >= 0 and not np.isnan(atr14[sb]) else "",
            "volume_at_signal": f"{volumes[sb]:.0f}" if sb >= 0 else "",
            "avg_volume_20": f"{avg_vol20[sb]:.0f}" if sb >= 0 and not np.isnan(avg_vol20[sb]) else "",
            "bar_range_at_signal": f"{highs[sb] - lows[sb]:.2f}" if sb >= 0 else "",
            "ema21_at_signal": f"{ema21[sb]:.2f}" if sb >= 0 and not np.isnan(ema21[sb]) else "",
            "vwap_at_signal": "NA",
            "day_of_week": dow,
            "hour_of_entry": hour_entry,
            "month": entry_ts[:7],
            "year": entry_ts[:4],
            "is_first_hour": 930 <= h_et <= 1030,
            "is_last_hour": 1545 <= h_et <= 1645,
            "data_period": get_data_period(entry_ts),
            "validation_status": get_validation(entry_ts),
            "daily_trade_number": daily_trade_num,
            "daily_pnl_before_this_trade": f"{daily_before:.2f}",
            "daily_pnl_after_this_trade": f"{daily_pnl_tracker[day]:.2f}",
            "trades_today_so_far": daily_trade_num - 1,
        }
        csv_rows.append(row)

    # ── Write trades CSV ──
    trades_path = REPORTS / "htf_swing_v3_all_trades.csv"
    with open(trades_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # ── Monthly summary ──
    monthly = defaultdict(lambda: {"trades": [], "days": set()})
    for r in csv_rows:
        mo = r["month"]
        monthly[mo]["trades"].append(r)
        monthly[mo]["days"].add(r["entry_bar_timestamp"][:10])

    monthly_path = REPORTS / "htf_swing_v3_monthly_summary.csv"
    with open(monthly_path, "w", newline="") as f:
        fields = ["month", "total_trades", "rsi_trades", "ib_trades", "mom_trades",
                  "winners", "losers", "win_rate", "gross_pnl", "total_costs", "net_pnl",
                  "cumulative_pnl", "best_trade", "worst_trade", "avg_trade", "median_trade",
                  "best_day", "worst_day", "trading_days", "avg_trades_per_day",
                  "max_consecutive_wins", "max_consecutive_losses", "avg_bars_held",
                  "data_period", "validation_status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        cum_monthly = 0.0
        for mo in sorted(monthly.keys()):
            trades_m = monthly[mo]["trades"]
            pnls = [float(t["net_pnl"]) for t in trades_m]
            gross = [float(t["gross_pnl"]) for t in trades_m]
            costs = [float(t["total_cost"]) for t in trades_m]
            bars = [int(t["bars_held"]) for t in trades_m]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p <= 0]

            # Daily P&L for best/worst day
            day_pnl = defaultdict(float)
            for t in trades_m:
                day_pnl[t["entry_bar_timestamp"][:10]] += float(t["net_pnl"])

            # Consecutive wins/losses
            max_cw = 0; max_cl = 0; cw = 0; cl = 0
            for p in pnls:
                if p > 0: cw += 1; cl = 0; max_cw = max(max_cw, cw)
                else: cl += 1; cw = 0; max_cl = max(max_cl, cl)

            net_mo = sum(pnls)
            cum_monthly += net_mo

            w.writerow({
                "month": mo,
                "total_trades": len(trades_m),
                "rsi_trades": sum(1 for t in trades_m if t["strategy"] == "RSI"),
                "ib_trades": sum(1 for t in trades_m if t["strategy"] == "IB"),
                "mom_trades": sum(1 for t in trades_m if t["strategy"] == "MOM"),
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": f"{len(winners)/len(trades_m)*100:.1f}" if trades_m else "0",
                "gross_pnl": f"{sum(gross):.2f}",
                "total_costs": f"{sum(costs):.2f}",
                "net_pnl": f"{net_mo:.2f}",
                "cumulative_pnl": f"{cum_monthly:.2f}",
                "best_trade": f"{max(pnls):.2f}" if pnls else "0",
                "worst_trade": f"{min(pnls):.2f}" if pnls else "0",
                "avg_trade": f"{np.mean(pnls):.2f}" if pnls else "0",
                "median_trade": f"{np.median(pnls):.2f}" if pnls else "0",
                "best_day": f"{max(day_pnl.values()):.2f}" if day_pnl else "0",
                "worst_day": f"{min(day_pnl.values()):.2f}" if day_pnl else "0",
                "trading_days": len(monthly[mo]["days"]),
                "avg_trades_per_day": f"{len(trades_m)/max(len(monthly[mo]['days']),1):.1f}",
                "max_consecutive_wins": max_cw,
                "max_consecutive_losses": max_cl,
                "avg_bars_held": f"{np.mean(bars):.1f}" if bars else "0",
                "data_period": get_data_period(mo + "-15"),
                "validation_status": get_validation(mo + "-15"),
            })

    # ── Daily summary ──
    daily_data = defaultdict(lambda: {"trades": [], "strats": defaultdict(int)})
    for r in csv_rows:
        d = r["entry_bar_timestamp"][:10]
        daily_data[d]["trades"].append(r)
        daily_data[d]["strats"][r["strategy"]] += 1

    daily_path = REPORTS / "htf_swing_v3_daily_summary.csv"
    with open(daily_path, "w", newline="") as f:
        fields = ["date", "day_of_week", "total_trades", "rsi_trades", "ib_trades",
                  "mom_trades", "winners", "losers", "win_rate", "net_pnl",
                  "cumulative_pnl", "max_open_positions", "first_trade_time",
                  "last_trade_time", "data_period"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        cum_daily = 0.0
        for d in sorted(daily_data.keys()):
            dd = daily_data[d]
            pnls = [float(t["net_pnl"]) for t in dd["trades"]]
            winners = sum(1 for p in pnls if p > 0)
            net_d = sum(pnls)
            cum_daily += net_d
            times = [t["entry_bar_timestamp"] for t in dd["trades"]]

            w.writerow({
                "date": d,
                "day_of_week": dd["trades"][0]["day_of_week"] if dd["trades"] else "",
                "total_trades": len(dd["trades"]),
                "rsi_trades": dd["strats"].get("RSI", 0),
                "ib_trades": dd["strats"].get("IB", 0),
                "mom_trades": dd["strats"].get("MOM", 0),
                "winners": winners,
                "losers": len(pnls) - winners,
                "win_rate": f"{winners/len(pnls)*100:.1f}" if pnls else "0",
                "net_pnl": f"{net_d:.2f}",
                "cumulative_pnl": f"{cum_daily:.2f}",
                "max_open_positions": 3,  # max possible
                "first_trade_time": min(times) if times else "",
                "last_trade_time": max(times) if times else "",
                "data_period": get_data_period(d),
            })

    # ── Summary ──
    total_net = sum(float(r["net_pnl"]) for r in csv_rows)
    trades_size = os.path.getsize(trades_path)
    monthly_size = os.path.getsize(monthly_path)
    daily_size = os.path.getsize(daily_path)

    print(f"\n{'═' * 70}")
    print(f"  EXPORT COMPLETE")
    print(f"{'═' * 70}")
    print(f"\n  Total trades exported: {len(csv_rows):,}")
    print(f"  Date range: {csv_rows[0]['entry_bar_timestamp'][:10]} → {csv_rows[-1]['entry_bar_timestamp'][:10]}")
    print(f"  Total months: {len(monthly)}")
    print(f"\n  Files:")
    print(f"    {trades_path}: {trades_size:,} bytes ({len(csv_rows):,} rows)")
    print(f"    {monthly_path}: {monthly_size:,} bytes ({len(monthly)} rows)")
    print(f"    {daily_path}: {daily_size:,} bytes ({len(daily_data)} rows)")

    # First 5 rows preview
    print(f"\n  TRADES CSV — first 5 rows:")
    for r in csv_rows[:5]:
        print(f"    #{r['trade_number']} {r['strategy']} {r['direction']} "
              f"{r['entry_bar_timestamp']} @ {r['entry_price']} → "
              f"{r['exit_bar_timestamp']} @ {r['exit_price']} "
              f"[{r['exit_reason']}] ${r['net_pnl']}")

    # Sanity checks
    print(f"\n  SANITY CHECKS:")
    print(f"    Total net P&L: ${total_net:,.2f}")

    # Duplicate timestamps
    entry_keys = [(r["strategy"], r["entry_bar_timestamp"]) for r in csv_rows]
    dupes = len(entry_keys) - len(set(entry_keys))
    print(f"    Duplicate entries: {dupes} {'✅' if dupes == 0 else '❌'}")

    # Overlapping trades per strategy
    by_strat = defaultdict(list)
    for r in csv_rows:
        by_strat[r["strategy"]].append(r)
    overlaps = 0
    for name, strades in by_strat.items():
        for i in range(len(strades) - 1):
            if strades[i+1]["entry_bar_timestamp"] < strades[i]["exit_bar_timestamp"]:
                overlaps += 1
    print(f"    Overlapping trades: {overlaps} {'✅' if overlaps == 0 else '❌'}")

    # Trade counts per strategy
    for name in ["RSI", "IB", "MOM"]:
        print(f"    {name}: {len(by_strat[name]):,} trades")

    print(f"\n  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
