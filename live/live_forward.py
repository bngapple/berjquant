#!/usr/bin/env python3
"""
HTF Swing v3 Hybrid v2 — Live Streaming Forward Tester

Streams MNQ 1m bars from Databento, resamples to 15m RTH in real-time,
runs all 3 strategies, and prints trade alerts as signals fire.

Usage:
    python live/live_forward.py              # Start streaming
    python live/live_forward.py --dry-run    # Show config and exit
"""

import argparse
import json
import os
import signal
import sys
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_htf_swing import (
    extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE, SLIP_TICKS, COMM_PER_SIDE, EXCH_PER_SIDE,
)
from run_htf_swing_8yr import resample_15m_rth

# ── Config ─────────────────────────────────────────────────────
CONTRACTS = 3
FLATTEN_TIME = 1645
HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True,                "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}
FT_START = "2026-02-05"
ET = ZoneInfo("US/Eastern")

LIVE_DIR = Path(__file__).resolve().parent
DATA_DIR = LIVE_DIR.parent / "data" / "processed" / "MNQ" / "1m"
BASE_FILE = DATA_DIR / "full_2yr.parquet"
FWD_BARS_FILE = DATA_DIR / "forward_test_bars.parquet"
STATE_FILE = LIVE_DIR / "live_state.json"


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


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"total_trades": 0, "cumulative_pnl": 0.0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


class LiveForwardTester:
    def __init__(self):
        self.api_key = load_api_key()
        self.state = load_state()
        self.live_bars = []
        self.base_1m = None
        self.last_15m_count = 0
        self.prev_trade_count = 0
        self.bar_count = 0
        self.running = True
        self.last_price = 0.0
        self.session_trades = []
        self._prev_period = None
        self._last_save = _time.time()

    def load_base(self):
        """Load base 2yr parquet + any previously fetched forward bars."""
        print("  Loading base data...")
        df = pl.read_parquet(BASE_FILE)
        if hasattr(df["timestamp"].dtype, "time_zone") and df["timestamp"].dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        if FWD_BARS_FILE.exists():
            fwd = pl.read_parquet(FWD_BARS_FILE)
            if hasattr(fwd["timestamp"].dtype, "time_zone") and fwd["timestamp"].dtype.time_zone:
                fwd = fwd.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
            df = pl.concat([df, fwd]).sort("timestamp").unique(
                subset=["timestamp"], keep="first"
            )

        self.base_1m = df
        last_ts = df["timestamp"].max()
        print(f"  Base: {len(df):,} 1m bars through {last_ts}")

        # Initial strategy pass to get current trade count
        print("  Computing initial state...")
        df_15m = resample_15m_rth(df)
        trades = self._run_strategies(df_15m)
        ft = [t for t in trades if str(t.entry_time)[:10] >= FT_START]
        self.prev_trade_count = len(ft)
        self.last_15m_count = len(df_15m)

        cum_pnl = sum(t.net_pnl for t in ft)
        print(f"  15m bars: {len(df_15m):,}")
        print(f"  Forward trades: {self.prev_trade_count}")
        print(f"  Cumulative P&L: ${cum_pnl:>+,.2f}")

        return last_ts

    def _run_strategies(self, df_15m):
        """Run all 3 strategies on 15m bars."""
        o, h, l, c, ts, hm = extract_arrays(df_15m)

        p = HYBRID_V2["RSI"]
        rsi = backtest(o, h, l, c, ts, hm,
                       sig_rsi_extreme(df_15m, p["period"], p["ob"], p["os"]),
                       pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                       p["hold"], CONTRACTS, "RSI", FLATTEN_TIME)

        p = HYBRID_V2["IB"]
        ib = backtest(o, h, l, c, ts, hm,
                      sig_ib_breakout(df_15m, p["ib_filter"])[0],
                      pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                      p["hold"], CONTRACTS, "IB", FLATTEN_TIME)

        p = HYBRID_V2["MOM"]
        mom = backtest(o, h, l, c, ts, hm,
                       sig_momentum_bar(df_15m, p["atr_mult"], p["vol_mult"])[0],
                       pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                       p["hold"], CONTRACTS, "MOM", FLATTEN_TIME)

        return sorted(rsi + ib + mom, key=lambda t: str(t.entry_time))

    def _to_polars_df(self, bars):
        """Convert list-of-dicts live bars to Polars DataFrame matching base schema."""
        df = pl.DataFrame(bars)
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ns").alias("timestamp")
            )
        if hasattr(df["timestamp"].dtype, "time_zone") and df["timestamp"].dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("tick_count"))
        df = df.filter(pl.col("close") > 0)

        for col in self.base_1m.columns:
            if col not in df.columns:
                df = df.with_columns(
                    pl.lit(0).cast(self.base_1m[col].dtype).alias(col)
                )
        df = df.select(self.base_1m.columns)
        df = df.with_columns(
            pl.col("timestamp").cast(self.base_1m["timestamp"].dtype)
        )
        return df

    def _merge_all(self):
        """Merge base 1m + live buffer → single DataFrame."""
        if not self.live_bars:
            return self.base_1m
        live_df = self._to_polars_df(self.live_bars)
        return (
            pl.concat([self.base_1m, live_df])
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
        )

    def _get_15m_period(self, ts_ns):
        """Return (period_key, hhmm_ET, datetime_ET) for a nanosecond timestamp."""
        dt_utc = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ET)
        m = dt_et.minute - (dt_et.minute % 15)
        hhmm = dt_et.hour * 100 + m
        return f"{dt_et.date()}_{hhmm}", hhmm, dt_et

    def evaluate(self):
        """Resample to 15m, run strategies, check for new trades."""
        t0 = _time.time()
        merged = self._merge_all()
        df_15m = resample_15m_rth(merged)

        # Drop last bar — it may be incomplete (in-progress)
        if len(df_15m) > 1:
            df_15m = df_15m.head(len(df_15m) - 1)

        n = len(df_15m)
        if n <= self.last_15m_count:
            return
        self.last_15m_count = n

        new_bar = df_15m.tail(1)
        elapsed = _time.time() - t0
        print(f"\n  >> 15m #{n}: {new_bar['timestamp'][0]}  "
              f"O={new_bar['open'][0]:.2f} H={new_bar['high'][0]:.2f} "
              f"L={new_bar['low'][0]:.2f} C={new_bar['close'][0]:.2f}  "
              f"[{elapsed:.1f}s]")

        all_trades = self._run_strategies(df_15m)
        ft = [t for t in all_trades if str(t.entry_time)[:10] >= FT_START]
        new_trades = ft[self.prev_trade_count:]

        if new_trades:
            print(f"\n  +{'=' * 56}+")
            print(f"  |  {len(new_trades)} NEW TRADE{'S' if len(new_trades) > 1 else ' '}"
                  f"{'':>44}|")
            print(f"  +{'─' * 56}+")
            for t in new_trades:
                d = "LONG " if t.direction == 1 else "SHORT"
                w = "W" if t.net_pnl > 0 else "L"
                print(f"  | {w} {t.strategy:>3} {d} "
                      f"{str(t.entry_time)[:16]} -> {str(t.exit_time)[:16]}  "
                      f"${t.net_pnl:>+8,.2f}  {t.reason}")
            pnl = sum(t.net_pnl for t in new_trades)
            cum = sum(t.net_pnl for t in ft)
            print(f"  +{'─' * 56}+")
            print(f"  |  Batch P&L: ${pnl:>+10,.2f}    "
                  f"Cumulative: ${cum:>+10,.2f}       |")
            print(f"  +{'=' * 56}+")
            self.session_trades.extend(new_trades)

        self.prev_trade_count = len(ft)
        self.state["total_trades"] = len(ft)
        self.state["cumulative_pnl"] = round(sum(t.net_pnl for t in ft), 2)
        save_state(self.state)

    def save_bars(self):
        """Persist live bars to the shared forward_test_bars.parquet."""
        if not self.live_bars:
            return
        df = self._to_polars_df(self.live_bars)
        if FWD_BARS_FILE.exists():
            existing = pl.read_parquet(FWD_BARS_FILE)
            if hasattr(existing["timestamp"].dtype, "time_zone") and existing["timestamp"].dtype.time_zone:
                existing = existing.with_columns(
                    pl.col("timestamp").dt.replace_time_zone(None)
                )
            df = pl.concat([existing, df]).sort("timestamp").unique(
                subset=["timestamp"], keep="last"
            )
        df.write_parquet(FWD_BARS_FILE)

    def run(self):
        """Main streaming loop."""
        from databento import Live

        def _stop(*_):
            self.running = False
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        last_ts = self.load_base()

        # Replay from end of base data to fill any gaps, then go live
        start_utc = last_ts.replace(tzinfo=timezone.utc)
        start_ns = int(start_utc.timestamp() * 1e9)

        print(f"\n  Connecting to Databento live feed...")
        print(f"  Replaying from: {last_ts}")

        live_client = Live(key=self.api_key, reconnect_policy="reconnect")
        live_client.subscribe(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols="MNQ.c.0",
            stype_in="continuous",
            start=start_ns,
        )

        print(f"  Connected. Streaming MNQ 1m bars.")
        print(f"  Strategies: RSI / IB / MOM  ({CONTRACTS} cts each, {CONTRACTS * 3} total)")
        print(f"  Press Ctrl+C to stop\n")
        print(f"  {'=' * 58}")

        try:
            for record in live_client:
                if not self.running:
                    break

                # Skip non-OHLCV records (heartbeats, system msgs, etc.)
                if not hasattr(record, "open"):
                    continue

                bar = {
                    "timestamp": record.ts_event,
                    "open": record.open / 1e9,
                    "high": record.high / 1e9,
                    "low": record.low / 1e9,
                    "close": record.close / 1e9,
                    "volume": record.volume,
                }
                if bar["close"] <= 0:
                    continue

                self.live_bars.append(bar)
                self.bar_count += 1
                self.last_price = bar["close"]

                # Status line
                dt_utc = datetime.fromtimestamp(record.ts_event / 1e9, tz=timezone.utc)
                dt_et = dt_utc.astimezone(ET)
                wall_lag = _time.time() - (record.ts_event / 1e9)
                mode = "REPLAY" if wall_lag > 120 else "LIVE"
                print(f"  {dt_et.strftime('%H:%M')} ET | "
                      f"MNQ {bar['close']:>10.2f} | "
                      f"vol {bar['volume']:>6,} | "
                      f"bars {self.bar_count:>5,} | "
                      f"{mode}     ", end="\r")

                # Detect 15m period change → evaluate
                period, hhmm, _ = self._get_15m_period(record.ts_event)
                if period != self._prev_period and self._prev_period is not None:
                    # Only evaluate during/just after RTH
                    if 945 <= hhmm <= 1600:
                        self.evaluate()
                self._prev_period = period

                # Save bars every 15 min
                if _time.time() - self._last_save > 900:
                    self.save_bars()
                    self._last_save = _time.time()

        except Exception as e:
            print(f"\n\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Session summary
            print(f"\n\n  {'=' * 58}")
            print(f"  SESSION ENDED  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"  {'─' * 58}")
            print(f"  Bars received:     {self.bar_count:,}")
            print(f"  Last price:        {self.last_price:.2f}")
            if self.session_trades:
                pnl = sum(t.net_pnl for t in self.session_trades)
                wins = sum(1 for t in self.session_trades if t.net_pnl > 0)
                losses = len(self.session_trades) - wins
                print(f"  Session trades:    {len(self.session_trades)}  "
                      f"({wins}W / {losses}L)")
                print(f"  Session P&L:       ${pnl:>+,.2f}")
            else:
                print(f"  Session trades:    0")
            print(f"  Total trades:      {self.state['total_trades']}")
            print(f"  Cumulative P&L:    ${self.state['cumulative_pnl']:>+,.2f}")
            print(f"  {'=' * 58}")

            self.save_bars()
            try:
                live_client.terminate()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="HTF Swing v3 Live Forward Tester")
    parser.add_argument("--dry-run", action="store_true", help="Show config and exit")
    args = parser.parse_args()

    print("=" * 58)
    print("  HTF SWING V3 HYBRID V2 — LIVE STREAMING")
    print("=" * 58)

    if args.dry_run:
        api_key = load_api_key()
        state = load_state()
        print(f"\n  API key:      ...{api_key[-6:]}")
        print(f"  Contracts:    {CONTRACTS}/strategy ({CONTRACTS * 3} total)")
        print(f"  Flatten:      {FLATTEN_TIME}")
        print(f"  Strategies:   {', '.join(HYBRID_V2.keys())}")
        print(f"  Base data:    {BASE_FILE}")
        print(f"  Fwd bars:     {FWD_BARS_FILE}")
        print(f"  FT start:     {FT_START}")
        print(f"  Trades:       {state['total_trades']}")
        print(f"  Cum P&L:      ${state['cumulative_pnl']:>+,.2f}")
        return

    tester = LiveForwardTester()
    tester.run()


if __name__ == "__main__":
    main()
