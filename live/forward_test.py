#!/usr/bin/env python3
"""
HTF Swing v3 Hybrid v2 — Daily Forward Tester

Fetches the latest MNQ 1m bars from Databento, resamples to 15m,
runs all 3 strategies, and logs new trades since last run.

Usage:
    python live/forward_test.py              # Fetch latest, report today
    python live/forward_test.py --backfill   # Fetch all available, rebuild from scratch
    python live/forward_test.py --summary    # Just print current state, no fetch

State is stored in live/forward_state.json.
Trade log is appended to live/forward_trades.csv.
Daily P&L log is appended to live/forward_daily.csv.
"""

import argparse
import json
import os
import sys
import time as _time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add parent dir to path so we can import from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_htf_swing import (
    extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE, SLIP_TICKS, COMM_PER_SIDE, EXCH_PER_SIDE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

# ── Config ─────────────────────────────────────────────────────
CONTRACTS = 3
FLATTEN_TIME = 1645
HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}

LIVE_DIR = Path(__file__).resolve().parent
DATA_DIR = LIVE_DIR.parent / "data" / "processed" / "MNQ" / "1m"
STATE_FILE = LIVE_DIR / "forward_state.json"
TRADES_CSV = LIVE_DIR / "forward_trades.csv"
DAILY_CSV = LIVE_DIR / "forward_daily.csv"
BARS_FILE = DATA_DIR / "forward_test_bars.parquet"
BASE_FILE = DATA_DIR / "full_2yr.parquet"

# Forward test start date — only count trades from here
FT_START = "2026-02-05"


def load_api_key():
    env_file = LIVE_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.getenv("DATABENTO_API_KEY")
    if key:
        return key
    print("ERROR: No DATABENTO_API_KEY found in .env or environment")
    sys.exit(1)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_fetch_date": None, "total_trades": 0, "cumulative_pnl": 0.0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def fetch_bars(api_key, start_date, end_date):
    """Fetch MNQ 1m bars from Databento."""
    import databento as db
    client = db.Historical(api_key)

    print(f"  Fetching MNQ 1m: {start_date} → {end_date}")
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=["MNQ.c.0"],
            stype_in="continuous",
            schema="ohlcv-1m",
            start=start_date,
            end=end_date,
        )
    except Exception as e:
        err = str(e)
        if "data_end_after_available_end" in err or "dataset_unavailable_range" in err:
            # Extract the available end from the error message
            print(f"  Data not yet available through {end_date}")
            print(f"  Databento has ~24hr lag on CME data. Try again tomorrow.")
            return None
        raise

    rows = []
    for rec in data:
        rows.append({
            "timestamp": rec.ts_event,
            "open": rec.open / 1e9,
            "high": rec.high / 1e9,
            "low": rec.low / 1e9,
            "close": rec.close / 1e9,
            "volume": rec.volume,
        })

    if not rows:
        print("  No new bars returned")
        return None

    df = pl.DataFrame(rows)
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ns").alias("timestamp"))
    if hasattr(df["timestamp"].dtype, "time_zone") and df["timestamp"].dtype.time_zone:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("tick_count"))
    df = df.filter(pl.col("close") > 0)

    print(f"  Fetched {len(df):,} bars ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df


def merge_data(new_bars=None):
    """Merge base data + any fetched forward bars into 15m RTH."""
    # Load base
    df_base = pl.read_parquet(BASE_FILE)
    if hasattr(df_base["timestamp"].dtype, "time_zone") and df_base["timestamp"].dtype.time_zone:
        df_base = df_base.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    parts = [df_base]

    # Load previously saved forward bars
    if BARS_FILE.exists():
        df_saved = pl.read_parquet(BARS_FILE)
        parts.append(df_saved)

    # Add new bars
    if new_bars is not None:
        # Align schema
        for col in df_base.columns:
            if col not in new_bars.columns:
                new_bars = new_bars.with_columns(pl.lit(0).cast(df_base[col].dtype).alias(col))
        new_bars = new_bars.select(df_base.columns)
        new_bars = new_bars.with_columns(pl.col("timestamp").cast(df_base["timestamp"].dtype))
        parts.append(new_bars)

        # Save updated forward bars
        if BARS_FILE.exists():
            df_saved = pl.read_parquet(BARS_FILE)
            combined_fwd = pl.concat([df_saved, new_bars]).sort("timestamp").unique(subset=["timestamp"], keep="first")
        else:
            combined_fwd = new_bars
        combined_fwd.write_parquet(BARS_FILE)

    merged = pl.concat(parts).sort("timestamp").unique(subset=["timestamp"], keep="first")
    merged = merged.filter(pl.col("close") > 0)
    return merged


def run_strategies(df_15m):
    """Run all 3 strategies, return trades."""
    o, h, l, c, ts, hm = extract_arrays(df_15m)

    p = HYBRID_V2["RSI"]
    rsi = backtest(o, h, l, c, ts, hm, sig_rsi_extreme(df_15m, p["period"], p["ob"], p["os"]),
                   pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                   p["hold"], CONTRACTS, "RSI", FLATTEN_TIME)

    p = HYBRID_V2["IB"]
    ib = backtest(o, h, l, c, ts, hm, sig_ib_breakout(df_15m, p["ib_filter"])[0],
                  pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                  p["hold"], CONTRACTS, "IB", FLATTEN_TIME)

    p = HYBRID_V2["MOM"]
    mom = backtest(o, h, l, c, ts, hm, sig_momentum_bar(df_15m, p["atr_mult"], p["vol_mult"])[0],
                   pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                   p["hold"], CONTRACTS, "MOM", FLATTEN_TIME)

    all_trades = sorted(rsi + ib + mom, key=lambda t: str(t.entry_time))
    return all_trades


def filter_forward(trades):
    """Only trades from forward test start date onward."""
    return [t for t in trades if str(t.entry_time)[:10] >= FT_START]


def write_trades_csv(trades):
    """Write/overwrite the full trade log."""
    rows = []
    cum = 0
    slip_cost = SLIP_TICKS * TICK_SIZE * POINT_VALUE * CONTRACTS * 2
    comm_cost = COMM_PER_SIDE * CONTRACTS * 2
    exch_cost = EXCH_PER_SIDE * CONTRACTS * 2
    total_cost = slip_cost + comm_cost + exch_cost

    for i, t in enumerate(trades):
        cum += t.net_pnl
        rows.append({
            "trade_number": i + 1,
            "strategy": t.strategy,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "contracts": t.contracts,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "entry_price": round(t.entry_px, 2),
            "exit_price": round(t.exit_px, 2),
            "exit_reason": t.reason,
            "bars_held": t.bars_held,
            "net_pnl": round(t.net_pnl, 2),
            "cumulative_pnl": round(cum, 2),
            "cost": round(total_cost, 2),
        })

    df = pl.DataFrame(rows)
    df.write_csv(TRADES_CSV)


def write_daily_csv(trades):
    """Write/overwrite daily P&L log."""
    daily = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        d = str(t.entry_time)[:10]
        daily[d]["pnl"] += t.net_pnl
        daily[d]["trades"] += 1
        if t.net_pnl > 0:
            daily[d]["wins"] += 1

    rows = []
    cum = 0
    for d in sorted(daily):
        cum += daily[d]["pnl"]
        rows.append({
            "date": d,
            "trades": daily[d]["trades"],
            "wins": daily[d]["wins"],
            "net_pnl": round(daily[d]["pnl"], 2),
            "cumulative_pnl": round(cum, 2),
        })

    df = pl.DataFrame(rows)
    df.write_csv(DAILY_CSV)


def print_summary(trades):
    """Print the current forward test summary."""
    if not trades:
        print("  No trades yet.")
        return

    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    total = sum(t.net_pnl for t in trades)

    daily = defaultdict(float)
    monthly = defaultdict(float)
    for t in trades:
        daily[str(t.entry_time)[:10]] += t.net_pnl
        monthly[str(t.entry_time)[:7]] += t.net_pnl

    dv = np.array(list(daily.values()))
    eq = [0.0]
    for t in sorted(trades, key=lambda t: str(t.exit_time)):
        eq.append(eq[-1] + t.net_pnl)
    pk = 0; dd = 0
    for v in eq:
        pk = max(pk, v); dd = min(dd, v - pk)

    wr = len(winners) / len(trades) * 100
    aw = np.mean([t.net_pnl for t in winners]) if winners else 0
    al = np.mean([t.net_pnl for t in losers]) if losers else 0
    gw = sum(t.net_pnl for t in winners)
    gl = abs(sum(t.net_pnl for t in losers))
    pf = gw / gl if gl else 0
    sh = dv.mean() / dv.std() * np.sqrt(252) if len(dv) > 1 and dv.std() > 0 else 0

    start = min(str(t.entry_time)[:10] for t in trades)
    end = max(str(t.entry_time)[:10] for t in trades)

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  HTF SWING V3 HYBRID V2 — FORWARD TEST             ║")
    print(f"  ║  {start} → {end}  ({len(daily)} trading days)          ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"\n  Net P&L:    ${total:>+12,.2f}")
    print(f"  Trades:     {len(trades):>6}  ({wr:.1f}% WR)")
    print(f"  Avg Win:    ${aw:>+10,.2f}")
    print(f"  Avg Loss:   ${al:>+10,.2f}")
    print(f"  R:R:        {abs(aw/al) if al else 0:>10.2f}")
    print(f"  PF:         {pf:>10.2f}")
    print(f"  Sharpe:     {sh:>10.1f}")
    print(f"  Max DD:     ${dd:>10,.2f}")
    print(f"  Avg/day:    ${dv.mean():>+10,.2f}")
    print(f"  Win days:   {sum(1 for v in dv if v>0)}/{len(dv)}")

    # Per-strategy
    print(f"\n  Strategy    Trades    WR      P&L         PF")
    print(f"  ─────────── ──────  ─────  ──────────── ──────")
    for name in ["RSI", "IB", "MOM"]:
        st = [t for t in trades if t.strategy == name]
        if not st:
            continue
        sw = [t for t in st if t.net_pnl > 0]
        sp = sum(t.net_pnl for t in st)
        sgw = sum(t.net_pnl for t in sw)
        sgl = abs(sum(t.net_pnl for t in st if t.net_pnl <= 0))
        print(f"  {name:<11} {len(st):>5}  {len(sw)/len(st)*100:>5.1f}%  ${sp:>+10,.0f}  {sgw/sgl if sgl else 0:>5.2f}")

    # Monthly
    print(f"\n  Month       P&L          Trades  WR")
    print(f"  ─────────── ──────────── ──────  ─────")
    for mo in sorted(monthly):
        mt = [t for t in trades if str(t.entry_time)[:7] == mo]
        mw = sum(1 for t in mt if t.net_pnl > 0)
        print(f"  {mo}     ${monthly[mo]:>+10,.0f}  {len(mt):>5}  {mw/len(mt)*100:>5.1f}%")

    # Last 5 trades
    print(f"\n  LAST 5 TRADES:")
    for t in trades[-5:]:
        d = "L" if t.direction == 1 else "S"
        print(f"    {t.strategy:>3} {d} {str(t.entry_time)[:16]} → {str(t.exit_time)[:16]}  "
              f"${t.net_pnl:>+8,.2f}  {t.reason}")


def main():
    parser = argparse.ArgumentParser(description="HTF Swing v3 Forward Tester")
    parser.add_argument("--backfill", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    state = load_state()

    print("=" * 58)
    print("  HTF SWING V3 HYBRID V2 — FORWARD TESTER")
    print("=" * 58)

    if not args.summary:
        # Determine fetch range
        api_key = load_api_key()

        if args.backfill or state["last_fetch_date"] is None:
            # Fetch everything from the day after base data ends
            df_base = pl.read_parquet(BASE_FILE)
            base_end = str(df_base["timestamp"].max())[:10]
            fetch_start = base_end
            print(f"  Base data ends: {base_end}")
        else:
            # Incremental: fetch from last fetch date
            fetch_start = state["last_fetch_date"]
            print(f"  Last fetch: {fetch_start}")

        # Fetch up to yesterday (Databento lag)
        fetch_end = str(date.today())
        print(f"  Fetching: {fetch_start} → {fetch_end}")

        new_bars = fetch_bars(api_key, fetch_start, fetch_end)

        if new_bars is not None:
            state["last_fetch_date"] = str(date.today() - timedelta(days=1))

        # Merge all data
        print("\n  Merging data...")
        merged = merge_data(new_bars)
        print(f"  Total 1m bars: {len(merged):,} → {merged['timestamp'].max()}")
    else:
        merged = merge_data()
        print(f"  Total 1m bars: {len(merged):,}")

    # Resample
    print("  Resampling to 15m RTH...")
    df_15m = resample_15m_rth(merged)
    print(f"  15m bars: {len(df_15m):,}")

    # Run strategies
    print("  Running strategies...")
    all_trades = run_strategies(df_15m)
    ft_trades = filter_forward(all_trades)

    # Find new trades since last run
    old_count = state["total_trades"]
    new_trades = ft_trades[old_count:]

    if new_trades:
        print(f"\n  *** {len(new_trades)} NEW TRADES ***")
        for t in new_trades:
            d = "LONG" if t.direction == 1 else "SHORT"
            print(f"    {t.strategy:>3} {d:<5} {str(t.entry_time)[:16]} → {str(t.exit_time)[:16]}  "
                  f"${t.net_pnl:>+8,.2f}  {t.reason}")
        new_pnl = sum(t.net_pnl for t in new_trades)
        print(f"    New trades P&L: ${new_pnl:>+,.2f}")
    else:
        print("\n  No new trades since last run.")

    # Update state
    state["total_trades"] = len(ft_trades)
    state["cumulative_pnl"] = round(sum(t.net_pnl for t in ft_trades), 2)
    save_state(state)

    # Write CSVs
    write_trades_csv(ft_trades)
    write_daily_csv(ft_trades)

    # Print summary
    print_summary(ft_trades)

    print(f"\n  Files:")
    print(f"    {TRADES_CSV} ({len(ft_trades)} trades)")
    print(f"    {DAILY_CSV}")
    print(f"    {STATE_FILE}")
    print("=" * 58)


if __name__ == "__main__":
    main()
