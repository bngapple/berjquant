#!/usr/bin/env python3
"""
Always-On Trading System — trades EVERY day, ALL 12 months.

Three layers:
  1. Microstructure Mean Reversion (VWAP z-score, 1m bars)
  2. Session Structure Trades (OR breakout, lunch reversion, afternoon trend, closing imbalance)
  3. Volatility-Adaptive Momentum (trend days only)

Dual-year validation:
  Year 1 (2024-03-19 → 2025-03-18): parameter optimization
  Year 2 (2025-03-19 → 2026-03-18): pure out-of-sample

Usage:
    python3 run_always_on.py
"""

import gc
import json
import time as _time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

# ── Project imports ──────────────────────────────────────────────────
from engine.utils import MNQ_SPEC, ContractSpec

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")
SEED = 42
np.random.seed(SEED)

# ── Cost model ───────────────────────────────────────────────────────
COMMISSION_PER_SIDE = 0.62          # $0.62 per MNQ per side
SLIPPAGE_TICKS = 1                  # 1 tick per entry and per exit
TICK_SIZE = MNQ_SPEC.tick_size      # 0.25
POINT_VALUE = MNQ_SPEC.point_value  # $2.00/point


def round_trip_cost(contracts: int) -> float:
    """Total cost per trade (commission + slippage) in dollars."""
    commission = COMMISSION_PER_SIDE * 2 * contracts
    slippage_dollars = SLIPPAGE_TICKS * TICK_SIZE * POINT_VALUE * 2 * contracts
    return commission + slippage_dollars  # $2.24 per contract


# ── Data loading ─────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/MNQ/1m/full_2yr.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

YR1_START = datetime(2024, 3, 19)
YR1_END   = datetime(2025, 3, 19)
YR2_START = datetime(2025, 3, 19)
YR2_END   = datetime(2026, 3, 19)


def load_data() -> pl.DataFrame:
    """Load 1-minute MNQ bars and add ET time columns."""
    print("Loading data ...")
    df = pl.read_parquet(DATA_PATH)
    # Add ET datetime and helpers
    df = df.with_columns([
        pl.col("timestamp").dt.convert_time_zone("US/Eastern").alias("ts_et"),
    ])
    df = df.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("hour_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("minute_et"),
    ])
    # HH:MM as integer for fast comparison: e.g. 9:30 = 930, 16:00 = 1600
    df = df.with_columns([
        (pl.col("hour_et") * 100 + pl.col("minute_et")).alias("hhmm"),
    ])
    print(f"  {len(df):,} bars loaded  ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df


def split_years(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into Year 1 and Year 2."""
    yr1 = df.filter(
        (pl.col("timestamp") >= YR1_START) & (pl.col("timestamp") < YR1_END)
    )
    yr2 = df.filter(
        (pl.col("timestamp") >= YR2_START) & (pl.col("timestamp") < YR2_END)
    )
    return yr1, yr2


def resample_5m(df: pl.DataFrame) -> pl.DataFrame:
    """Resample 1m bars to 5m bars."""
    return (
        df.group_by_dynamic("timestamp", every="5m")
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


# ═════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (vectorized)
# ═════════════════════════════════════════════════════════════════════

def add_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Add daily VWAP (resets each trading day)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"].cast(pl.Float64)
    vol_f = df["volume"].cast(pl.Float64)

    # Cumulative within each ET date
    df = df.with_columns([
        tp_vol.alias("_tp_vol"),
        vol_f.alias("_vol_f"),
    ])
    df = df.with_columns([
        pl.col("_tp_vol").cum_sum().over("date_et").alias("_cum_tp_vol"),
        pl.col("_vol_f").cum_sum().over("date_et").alias("_cum_vol"),
    ])
    df = df.with_columns([
        (pl.col("_cum_tp_vol") / pl.col("_cum_vol").clip(lower_bound=1)).alias("vwap"),
    ])
    df = df.drop(["_tp_vol", "_vol_f", "_cum_tp_vol", "_cum_vol"])
    return df


def add_vwap_zscore(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Add z-score of (close - VWAP) using rolling std."""
    if "vwap" not in df.columns:
        df = add_vwap(df)
    diff = df["close"] - df["vwap"]
    df = df.with_columns([diff.alias("_vwap_diff")])
    df = df.with_columns([
        pl.col("_vwap_diff").rolling_std(window_size=lookback, min_periods=lookback).alias("_vwap_std"),
    ])
    df = df.with_columns([
        (pl.col("_vwap_diff") / pl.col("_vwap_std").clip(lower_bound=0.01)).alias("vwap_zscore"),
    ])
    df = df.drop(["_vwap_diff", "_vwap_std"])
    return df


def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add RSI indicator."""
    delta = df["close"].diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(window_size=period, min_periods=period)
    avg_loss = loss.rolling_mean(window_size=period, min_periods=period)
    # Handle zero avg_loss
    df = df.with_columns([avg_gain.alias("_ag"), avg_loss.alias("_al")])
    df = df.with_columns([
        pl.when(pl.col("_al") < 0.0001)
        .then(100.0)
        .otherwise(100.0 - 100.0 / (1.0 + pl.col("_ag") / pl.col("_al").clip(lower_bound=0.0001)))
        .alias(f"rsi_{period}")
    ])
    df = df.drop(["_ag", "_al"])
    return df


def add_ema(df: pl.DataFrame, period: int) -> pl.DataFrame:
    """Add EMA."""
    col_name = f"ema_{period}"
    if col_name in df.columns:
        return df
    df = df.with_columns([
        pl.col("close").ewm_mean(span=period, min_periods=period).alias(col_name)
    ])
    return df


def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add ATR."""
    col_name = f"atr_{period}"
    if col_name in df.columns:
        return df
    tr = pl.max_horizontal(
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    )
    df = df.with_columns([tr.alias("_tr")])
    df = df.with_columns([
        pl.col("_tr").rolling_mean(window_size=period, min_periods=period).alias(col_name)
    ])
    df = df.drop(["_tr"])
    return df


def add_initial_balance(df: pl.DataFrame) -> pl.DataFrame:
    """Add IB high/low (first 30 min: 9:30-10:00 ET) per day."""
    # IB bars: hhmm >= 930 and hhmm < 1000
    ib = df.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1000))
    ib_stats = ib.group_by("date_et").agg([
        pl.col("high").max().alias("ib_high"),
        pl.col("low").min().alias("ib_low"),
    ])
    ib_stats = ib_stats.with_columns([
        ((pl.col("ib_high") + pl.col("ib_low")) / 2.0).alias("ib_mid"),
        (pl.col("ib_high") - pl.col("ib_low")).alias("ib_range"),
    ])
    df = df.join(ib_stats, on="date_et", how="left")
    return df


# ═════════════════════════════════════════════════════════════════════
# SIMPLE BACKTESTER
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SimplePosition:
    direction: int   # 1=long, -1=short
    entry_price: float
    entry_bar: int
    contracts: int
    stop_price: float
    target_price: float | None
    max_hold_bars: int
    trailing: bool = False
    trailing_distance: float = 0.0
    best_price: float = 0.0  # for trailing stop

    def update_trailing(self, high: float, low: float):
        if not self.trailing:
            return
        if self.direction == 1:
            if high > self.best_price:
                self.best_price = high
                self.stop_price = self.best_price - self.trailing_distance
        else:
            if low < self.best_price or self.best_price == 0.0:
                if self.best_price == 0.0:
                    self.best_price = low
                elif low < self.best_price:
                    self.best_price = low
                self.stop_price = self.best_price + self.trailing_distance


@dataclass
class TradeRecord:
    direction: int
    entry_price: float
    exit_price: float
    contracts: int
    net_pnl: float
    entry_time: datetime
    exit_time: datetime
    bars_held: int
    exit_reason: str
    layer: str


def run_backtest(
    df: pl.DataFrame,
    signals: np.ndarray,         # +1=long, -1=short, 0=no signal
    exit_signals: np.ndarray,    # True = exit now
    stop_ticks: int,
    target_ticks: int | None,
    max_hold_bars: int,
    contracts: int,
    trailing: bool = False,
    trailing_ticks: int = 0,
    layer_name: str = "",
    rth_only: bool = True,
) -> list[TradeRecord]:
    """Simple bar-by-bar backtest. Next-bar fills. 1 position at a time."""

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()
    hhmm = df["hhmm"].to_numpy()
    n = len(df)

    cost = round_trip_cost(contracts)
    slip_points = SLIPPAGE_TICKS * TICK_SIZE  # 0.25 per entry, 0.25 per exit

    trades: list[TradeRecord] = []
    pos: SimplePosition | None = None
    pending_signal: int = 0  # queued for next bar

    for i in range(n):
        h_et = hhmm[i]
        is_rth = 930 <= h_et < 1600

        # ── EOD flatten at 15:55 ──
        if pos is not None and h_et >= 1555:
            exit_px = closes[i]
            raw = (exit_px - pos.entry_price) * pos.direction * POINT_VALUE * pos.contracts
            net = raw - cost
            trades.append(TradeRecord(
                direction=pos.direction, entry_price=pos.entry_price,
                exit_price=exit_px, contracts=pos.contracts, net_pnl=net,
                entry_time=timestamps[pos.entry_bar], exit_time=timestamps[i],
                bars_held=i - pos.entry_bar, exit_reason="eod_flatten", layer=layer_name,
            ))
            pos = None
            pending_signal = 0
            continue

        # ── Execute pending entry (next-bar fill) ──
        if pending_signal != 0 and pos is None:
            if rth_only and not is_rth:
                pending_signal = 0
            elif h_et >= 1550:  # too close to close
                pending_signal = 0
            else:
                entry_px = opens[i] + pending_signal * slip_points
                stop_offset = stop_ticks * TICK_SIZE
                stop_px = entry_px - pending_signal * stop_offset

                tp_px = None
                if target_ticks is not None:
                    tp_px = entry_px + pending_signal * target_ticks * TICK_SIZE

                pos = SimplePosition(
                    direction=pending_signal,
                    entry_price=entry_px,
                    entry_bar=i,
                    contracts=contracts,
                    stop_price=stop_px,
                    target_price=tp_px,
                    max_hold_bars=max_hold_bars,
                    trailing=trailing,
                    trailing_distance=trailing_ticks * TICK_SIZE if trailing else 0,
                    best_price=entry_px,
                )
                pending_signal = 0

        # ── Manage open position ──
        if pos is not None:
            bars_held = i - pos.entry_bar
            exit_px = None
            exit_reason = ""

            # Update trailing stop
            pos.update_trailing(highs[i], lows[i])

            # Stop loss check
            if pos.direction == 1 and lows[i] <= pos.stop_price:
                exit_px = pos.stop_price
                exit_reason = "stop_loss"
            elif pos.direction == -1 and highs[i] >= pos.stop_price:
                exit_px = pos.stop_price
                exit_reason = "stop_loss"

            # Target check (only if stop not hit, or if stop hit use conservative)
            if exit_px is None and pos.target_price is not None:
                if pos.direction == 1 and highs[i] >= pos.target_price:
                    exit_px = pos.target_price
                    exit_reason = "take_profit"
                elif pos.direction == -1 and lows[i] <= pos.target_price:
                    exit_px = pos.target_price
                    exit_reason = "take_profit"

            # Exit signal
            if exit_px is None and i < len(exit_signals) and exit_signals[i]:
                exit_px = closes[i]
                exit_reason = "signal_exit"

            # Max hold
            if exit_px is None and bars_held >= pos.max_hold_bars:
                exit_px = closes[i]
                exit_reason = "max_hold"

            if exit_px is not None:
                # Apply exit slippage (adverse direction)
                exit_px -= pos.direction * slip_points
                raw = (exit_px - pos.entry_price) * pos.direction * POINT_VALUE * pos.contracts
                net = raw - cost
                trades.append(TradeRecord(
                    direction=pos.direction, entry_price=pos.entry_price,
                    exit_price=exit_px, contracts=pos.contracts, net_pnl=net,
                    entry_time=timestamps[pos.entry_bar], exit_time=timestamps[i],
                    bars_held=bars_held, exit_reason=exit_reason, layer=layer_name,
                ))
                pos = None

        # ── Generate new signal (only if flat) ──
        if pos is None and i < len(signals) and signals[i] != 0:
            if rth_only and not is_rth:
                continue
            if h_et >= 1550:
                continue
            pending_signal = signals[i]

    # Close any open position at end
    if pos is not None:
        exit_px = closes[-1]
        raw = (exit_px - pos.entry_price) * pos.direction * POINT_VALUE * pos.contracts
        net = raw - cost
        trades.append(TradeRecord(
            direction=pos.direction, entry_price=pos.entry_price,
            exit_price=exit_px, contracts=pos.contracts, net_pnl=net,
            entry_time=timestamps[pos.entry_bar], exit_time=timestamps[-1],
            bars_held=len(df) - 1 - pos.entry_bar, exit_reason="end_of_data", layer=layer_name,
        ))

    return trades


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — MICROSTRUCTURE MEAN REVERSION
# ═════════════════════════════════════════════════════════════════════

def layer1_signals(
    df: pl.DataFrame,
    z_threshold: float = 2.0,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """VWAP z-score mean reversion signals.

    Entry: z < -threshold → LONG, z > threshold → SHORT
    Exit: z crosses back through 0 (reverted to VWAP)
    """
    df_feat = add_vwap(df)
    df_feat = add_vwap_zscore(df_feat, lookback=lookback)

    z = df_feat["vwap_zscore"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)

    in_long = False
    in_short = False

    for i in range(n):
        if not (930 <= hhmm[i] < 1555):
            continue
        if np.isnan(z[i]):
            continue

        # Entry signals (only when flat)
        if not in_long and not in_short:
            if z[i] < -z_threshold:
                signals[i] = 1  # long
                in_long = True
            elif z[i] > z_threshold:
                signals[i] = -1  # short
                in_short = True
        else:
            # Exit: z crosses 0
            if in_long and z[i] >= 0:
                exits[i] = True
                in_long = False
            elif in_short and z[i] <= 0:
                exits[i] = True
                in_short = False

    return signals, exits


def sweep_layer1(df: pl.DataFrame, label: str = "Year") -> list[dict]:
    """Sweep Layer 1 parameters."""
    z_thresholds = [1.5, 2.0, 2.5, 3.0]
    lookbacks = [10, 20, 30, 50]
    stop_ticks_list = [4, 6, 8, 12, 16]
    max_hold_list = [5, 10, 15, 20, 30]
    contracts = 3

    results = []
    total = len(z_thresholds) * len(lookbacks) * len(stop_ticks_list) * len(max_hold_list)
    print(f"  Layer 1 sweep: {total} combinations on {label} ...")

    count = 0
    for z_thresh in z_thresholds:
        for lb in lookbacks:
            sigs, exits = layer1_signals(df, z_threshold=z_thresh, lookback=lb)
            for stop_t in stop_ticks_list:
                for max_h in max_hold_list:
                    trades = run_backtest(
                        df, sigs, exits,
                        stop_ticks=stop_t, target_ticks=None,
                        max_hold_bars=max_h, contracts=contracts,
                        layer_name="L1_VWAP_MR",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    n_trades = len(trades)
                    results.append({
                        "z_threshold": z_thresh, "lookback": lb,
                        "stop_ticks": stop_t, "max_hold": max_h,
                        "total_pnl": pnl, "trades": n_trades,
                        "trade_list": trades,
                    })
                    count += 1
                    if count % 100 == 0:
                        print(f"    {count}/{total} done ...")

    # Sort by P&L
    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — SESSION STRUCTURE TRADES
# ═════════════════════════════════════════════════════════════════════

# ── 2A: Opening Range Breakout / Failure ──

def layer2a_signals(
    df: pl.DataFrame,
    tp_mult: float = 1.0,
    sl_mult: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Opening range breakout signals.

    After 10:00 ET, if price breaks IB high → LONG, breaks IB low → SHORT.
    Returns: signals, exits, stop_ticks_arr, target_ticks_arr (per-signal dynamic S/T).
    """
    if "ib_high" not in df.columns:
        df = add_initial_balance(df)

    hhmm = df["hhmm"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ib_high = df["ib_high"].to_numpy()
    ib_low = df["ib_low"].to_numpy()
    ib_range = df["ib_range"].to_numpy()
    n = len(df)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)
    stop_arr = np.full(n, 8, dtype=np.int32)  # default
    target_arr = np.full(n, 16, dtype=np.int32)  # default

    traded_today = {}
    dates = df["date_et"].to_list()

    for i in range(n):
        h = hhmm[i]
        if not (1000 <= h < 1530):
            continue
        if np.isnan(ib_high[i]) or np.isnan(ib_low[i]):
            continue

        d = dates[i]
        if d in traded_today:
            continue

        ib_r = ib_range[i]
        if ib_r < 2.0:  # too narrow, skip
            continue

        # Breakout long
        if highs[i] > ib_high[i]:
            signals[i] = 1
            sl_pts = ib_r * sl_mult
            tp_pts = ib_r * tp_mult
            stop_arr[i] = max(int(sl_pts / TICK_SIZE), 4)
            target_arr[i] = max(int(tp_pts / TICK_SIZE), 4)
            traded_today[d] = True

        # Breakout short
        elif lows[i] < ib_low[i]:
            signals[i] = -1
            sl_pts = ib_r * sl_mult
            tp_pts = ib_r * tp_mult
            stop_arr[i] = max(int(sl_pts / TICK_SIZE), 4)
            target_arr[i] = max(int(tp_pts / TICK_SIZE), 4)
            traded_today[d] = True

    return signals, exits, stop_arr, target_arr


def run_backtest_dynamic_sl(
    df: pl.DataFrame,
    signals: np.ndarray,
    exit_signals: np.ndarray,
    stop_arr: np.ndarray,
    target_arr: np.ndarray,
    max_hold_bars: int,
    contracts: int,
    trailing: bool = False,
    trailing_ticks: int = 0,
    layer_name: str = "",
) -> list[TradeRecord]:
    """Backtest with per-signal dynamic stop/target (in ticks)."""
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()
    hhmm = df["hhmm"].to_numpy()
    n = len(df)

    cost = round_trip_cost(contracts)
    slip_points = SLIPPAGE_TICKS * TICK_SIZE

    trades: list[TradeRecord] = []
    pos: SimplePosition | None = None
    pending_signal: int = 0
    pending_stop: int = 8
    pending_target: int = 16

    for i in range(n):
        h_et = hhmm[i]

        # EOD flatten
        if pos is not None and h_et >= 1555:
            exit_px = closes[i]
            raw = (exit_px - pos.entry_price) * pos.direction * POINT_VALUE * pos.contracts
            net = raw - cost
            trades.append(TradeRecord(
                pos.direction, pos.entry_price, exit_px, pos.contracts, net,
                timestamps[pos.entry_bar], timestamps[i], i - pos.entry_bar, "eod_flatten", layer_name,
            ))
            pos = None
            pending_signal = 0
            continue

        # Execute pending
        if pending_signal != 0 and pos is None:
            if h_et >= 1550:
                pending_signal = 0
            else:
                entry_px = opens[i] + pending_signal * slip_points
                stop_off = pending_stop * TICK_SIZE
                target_off = pending_target * TICK_SIZE
                pos = SimplePosition(
                    direction=pending_signal,
                    entry_price=entry_px,
                    entry_bar=i,
                    contracts=contracts,
                    stop_price=entry_px - pending_signal * stop_off,
                    target_price=entry_px + pending_signal * target_off,
                    max_hold_bars=max_hold_bars,
                    trailing=trailing,
                    trailing_distance=trailing_ticks * TICK_SIZE if trailing else 0,
                    best_price=entry_px,
                )
                pending_signal = 0

        # Manage position
        if pos is not None:
            bars_held = i - pos.entry_bar
            exit_px = None
            exit_reason = ""

            pos.update_trailing(highs[i], lows[i])

            # Stop
            if pos.direction == 1 and lows[i] <= pos.stop_price:
                exit_px = pos.stop_price
                exit_reason = "stop_loss"
            elif pos.direction == -1 and highs[i] >= pos.stop_price:
                exit_px = pos.stop_price
                exit_reason = "stop_loss"

            # Target
            if exit_px is None and pos.target_price is not None:
                if pos.direction == 1 and highs[i] >= pos.target_price:
                    exit_px = pos.target_price
                    exit_reason = "take_profit"
                elif pos.direction == -1 and lows[i] <= pos.target_price:
                    exit_px = pos.target_price
                    exit_reason = "take_profit"

            # Exit signal
            if exit_px is None and i < len(exit_signals) and exit_signals[i]:
                exit_px = closes[i]
                exit_reason = "signal_exit"

            # Max hold
            if exit_px is None and bars_held >= max_hold_bars:
                exit_px = closes[i]
                exit_reason = "max_hold"

            if exit_px is not None:
                exit_px -= pos.direction * slip_points
                raw = (exit_px - pos.entry_price) * pos.direction * POINT_VALUE * pos.contracts
                net = raw - cost
                trades.append(TradeRecord(
                    pos.direction, pos.entry_price, exit_px, pos.contracts, net,
                    timestamps[pos.entry_bar], timestamps[i], bars_held, exit_reason, layer_name,
                ))
                pos = None

        # New signal
        if pos is None and signals[i] != 0:
            if 930 <= hhmm[i] < 1550:
                pending_signal = signals[i]
                pending_stop = stop_arr[i]
                pending_target = target_arr[i]

    return trades


# ── 2B: Lunch Reversion ──

def layer2b_signals(
    df: pl.DataFrame,
    rsi_period: int = 14,
    rsi_ob: float = 70.0,
    rsi_os: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lunch session mean reversion (12:00-13:30 ET).

    Entry: RSI crosses below oversold or above overbought during lunch.
    Exit: RSI returns to 50 or 30 bars pass.
    """
    # Use 1m RSI (equivalent to shorter-period on higher TF)
    df_feat = add_rsi(df, rsi_period)

    rsi = df_feat[f"rsi_{rsi_period}"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)

    in_long = False
    in_short = False

    for i in range(n):
        if not (1200 <= hhmm[i] < 1330):
            if in_long or in_short:
                exits[i] = True
                in_long = in_short = False
            continue
        if np.isnan(rsi[i]):
            continue

        if not in_long and not in_short:
            if rsi[i] < rsi_os:
                signals[i] = 1  # buy oversold
                in_long = True
            elif rsi[i] > rsi_ob:
                signals[i] = -1  # sell overbought
                in_short = True
        else:
            # Exit when RSI returns to 50
            if in_long and rsi[i] >= 50:
                exits[i] = True
                in_long = False
            elif in_short and rsi[i] <= 50:
                exits[i] = True
                in_short = False

    return signals, exits


# ── 2C: Afternoon Trend ──

def layer2c_signals(
    df: pl.DataFrame,
    ema_fast: int = 8,
    ema_slow: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Afternoon trend continuation (14:00-15:30 ET).

    At 14:00+: if price > VWAP AND EMA fast > EMA slow → LONG.
    """
    df_feat = add_vwap(df)
    df_feat = add_ema(df_feat, ema_fast)
    df_feat = add_ema(df_feat, ema_slow)

    closes = df_feat["close"].to_numpy()
    vwap = df_feat["vwap"].to_numpy()
    ema_f = df_feat[f"ema_{ema_fast}"].to_numpy()
    ema_s = df_feat[f"ema_{ema_slow}"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)

    traded_today = {}
    dates = df_feat["date_et"].to_list()

    for i in range(n):
        if not (1400 <= hhmm[i] < 1530):
            continue
        if np.isnan(vwap[i]) or np.isnan(ema_f[i]) or np.isnan(ema_s[i]):
            continue

        d = dates[i]
        if d in traded_today:
            continue

        # Check at 14:00 (first bar of the window)
        if closes[i] > vwap[i] and ema_f[i] > ema_s[i]:
            signals[i] = 1
            traded_today[d] = True
        elif closes[i] < vwap[i] and ema_f[i] < ema_s[i]:
            signals[i] = -1
            traded_today[d] = True

    return signals, exits


# ── 2D: Closing Imbalance ──

def layer2d_signals(
    df: pl.DataFrame,
    threshold_pct: float = 0.003,  # 0.3%
) -> tuple[np.ndarray, np.ndarray]:
    """Closing imbalance (15:45-15:58 ET). Reversion into close."""
    df_feat = add_vwap(df)

    closes = df_feat["close"].to_numpy()
    vwap = df_feat["vwap"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    n = len(df_feat)

    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)

    traded_today = {}
    dates = df_feat["date_et"].to_list()

    for i in range(n):
        # Exit at 15:58
        if hhmm[i] >= 1558:
            exits[i] = True
            continue

        if not (1545 <= hhmm[i] < 1555):
            continue
        if np.isnan(vwap[i]) or vwap[i] < 1:
            continue

        d = dates[i]
        if d in traded_today:
            continue

        pct_from_vwap = (closes[i] - vwap[i]) / vwap[i]

        if pct_from_vwap > threshold_pct:
            signals[i] = -1  # overbought, sell for close reversion
            traded_today[d] = True
        elif pct_from_vwap < -threshold_pct:
            signals[i] = 1  # oversold, buy for close reversion
            traded_today[d] = True

    return signals, exits


def sweep_layer2(df: pl.DataFrame, label: str = "Year") -> dict:
    """Sweep all Layer 2 sub-strategies."""
    contracts = 2
    results = {}

    # ── 2A: Opening Range ──
    print(f"  Layer 2A (Opening Range) sweep on {label} ...")
    best_2a = None
    for tp_mult in [0.75, 1.0, 1.5, 2.0]:
        for sl_mult in [0.3, 0.5, 0.75]:
            for max_hold in [15, 30, 60]:
                sigs, exits, stop_arr, target_arr = layer2a_signals(df, tp_mult=tp_mult, sl_mult=sl_mult)
                trades = run_backtest_dynamic_sl(
                    df, sigs, exits, stop_arr, target_arr,
                    max_hold_bars=max_hold, contracts=contracts, layer_name="L2A_OR",
                )
                pnl = sum(t.net_pnl for t in trades)
                if best_2a is None or pnl > best_2a["total_pnl"]:
                    best_2a = {
                        "tp_mult": tp_mult, "sl_mult": sl_mult, "max_hold": max_hold,
                        "total_pnl": pnl, "trades": len(trades), "trade_list": trades,
                    }
    results["2A"] = best_2a

    # ── 2B: Lunch Reversion ──
    print(f"  Layer 2B (Lunch Reversion) sweep on {label} ...")
    best_2b = None
    for rsi_p in [7, 14, 21]:
        for rsi_ob in [65, 70, 75, 80]:
            rsi_os = 100 - rsi_ob
            for stop_t in [6, 10, 14, 20]:
                for max_hold in [15, 30, 45]:
                    sigs, exits = layer2b_signals(df, rsi_period=rsi_p, rsi_ob=rsi_ob, rsi_os=rsi_os)
                    trades = run_backtest(
                        df, sigs, exits,
                        stop_ticks=stop_t, target_ticks=None,
                        max_hold_bars=max_hold, contracts=contracts, layer_name="L2B_Lunch",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    if best_2b is None or pnl > best_2b["total_pnl"]:
                        best_2b = {
                            "rsi_period": rsi_p, "rsi_ob": rsi_ob, "rsi_os": rsi_os,
                            "stop_ticks": stop_t, "max_hold": max_hold,
                            "total_pnl": pnl, "trades": len(trades), "trade_list": trades,
                        }
    results["2B"] = best_2b

    # ── 2C: Afternoon Trend ──
    print(f"  Layer 2C (Afternoon Trend) sweep on {label} ...")
    best_2c = None
    for ema_f in [5, 8, 13]:
        for ema_s in [13, 21, 34]:
            if ema_f >= ema_s:
                continue
            for trail_t in [8, 12, 16, 20, 24]:
                for max_hold in [30, 60, 90]:
                    sigs, exits = layer2c_signals(df, ema_fast=ema_f, ema_slow=ema_s)
                    trades = run_backtest(
                        df, sigs, exits,
                        stop_ticks=trail_t, target_ticks=None,
                        max_hold_bars=max_hold, contracts=contracts,
                        trailing=True, trailing_ticks=trail_t,
                        layer_name="L2C_Afternoon",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    if best_2c is None or pnl > best_2c["total_pnl"]:
                        best_2c = {
                            "ema_fast": ema_f, "ema_slow": ema_s,
                            "trail_ticks": trail_t, "max_hold": max_hold,
                            "total_pnl": pnl, "trades": len(trades), "trade_list": trades,
                        }
    results["2C"] = best_2c

    # ── 2D: Closing Imbalance ──
    print(f"  Layer 2D (Closing Imbalance) sweep on {label} ...")
    best_2d = None
    for thresh in [0.001, 0.002, 0.003, 0.004, 0.005]:
        for stop_t in [4, 6, 8, 12]:
            sigs, exits = layer2d_signals(df, threshold_pct=thresh)
            trades = run_backtest(
                df, sigs, exits,
                stop_ticks=stop_t, target_ticks=None,
                max_hold_bars=15, contracts=contracts, layer_name="L2D_Close",
            )
            pnl = sum(t.net_pnl for t in trades)
            if best_2d is None or pnl > best_2d["total_pnl"]:
                best_2d = {
                    "threshold_pct": thresh, "stop_ticks": stop_t,
                    "total_pnl": pnl, "trades": len(trades), "trade_list": trades,
                }
    results["2D"] = best_2d

    return results


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — VOLATILITY-ADAPTIVE MOMENTUM
# ═════════════════════════════════════════════════════════════════════

def layer3_signals(
    df: pl.DataFrame,
    range_mult: float = 1.5,
    ema_pullback: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Trend-day momentum: enter on pullback to EMA after confirming trend day.

    At 10:00 ET: if today's range > range_mult * avg range → trending day.
    Direction: with the trend (price > VWAP + first 30min green → LONG).
    Entry: first pullback to EMA on 1m after confirmation.
    """
    df_feat = add_vwap(df)
    df_feat = add_ema(df_feat, ema_pullback)
    df_feat = add_atr(df_feat, 14)

    closes = df_feat["close"].to_numpy()
    opens = df_feat["open"].to_numpy()
    highs = df_feat["high"].to_numpy()
    lows = df_feat["low"].to_numpy()
    vwap = df_feat["vwap"].to_numpy()
    ema_vals = df_feat[f"ema_{ema_pullback}"].to_numpy()
    hhmm = df_feat["hhmm"].to_numpy()
    dates = df_feat["date_et"].to_list()
    n = len(df_feat)

    # Pre-compute daily opening range (9:30-10:00)
    day_or_range = {}  # date -> (or_high, or_low, or_direction)
    for i in range(n):
        if 930 <= hhmm[i] < 1000:
            d = dates[i]
            if d not in day_or_range:
                day_or_range[d] = {"high": highs[i], "low": lows[i],
                                   "open": opens[i], "close": closes[i]}
            else:
                day_or_range[d]["high"] = max(day_or_range[d]["high"], highs[i])
                day_or_range[d]["low"] = min(day_or_range[d]["low"], lows[i])
                day_or_range[d]["close"] = closes[i]  # last bar of OR

    # Compute rolling avg OR range (14 day)
    sorted_dates = sorted(day_or_range.keys())
    or_ranges = [day_or_range[d]["high"] - day_or_range[d]["low"] for d in sorted_dates]
    avg_or = {}
    for j, d in enumerate(sorted_dates):
        if j >= 14:
            avg_or[d] = np.mean(or_ranges[j - 14: j])
        elif j >= 3:
            avg_or[d] = np.mean(or_ranges[:j])

    # Determine trend days
    trend_days = {}  # date -> direction (+1/-1)
    for d in sorted_dates:
        if d not in avg_or:
            continue
        info = day_or_range[d]
        today_range = info["high"] - info["low"]
        if today_range > range_mult * avg_or[d]:
            # Direction: was OR green or red?
            if info["close"] > info["open"]:
                trend_days[d] = 1
            else:
                trend_days[d] = -1

    # Generate signals: first pullback to EMA after 10:00 on trend days
    signals = np.zeros(n, dtype=np.int8)
    exits = np.zeros(n, dtype=bool)
    traded_today = {}

    for i in range(n):
        if not (1000 <= hhmm[i] < 1550):
            continue
        d = dates[i]
        if d not in trend_days or d in traded_today:
            continue
        if np.isnan(ema_vals[i]):
            continue

        direction = trend_days[d]

        # Pullback to EMA: price touches EMA from the trend side
        if direction == 1:
            # Long: price pulled back down to EMA
            if lows[i] <= ema_vals[i] and closes[i] > ema_vals[i]:
                signals[i] = 1
                traded_today[d] = True
        else:
            # Short: price pulled back up to EMA
            if highs[i] >= ema_vals[i] and closes[i] < ema_vals[i]:
                signals[i] = -1
                traded_today[d] = True

    return signals, exits


def sweep_layer3(df: pl.DataFrame, label: str = "Year") -> list[dict]:
    """Sweep Layer 3 parameters."""
    contracts = 3
    results = []

    range_mults = [1.0, 1.25, 1.5, 2.0]
    ema_pullbacks = [5, 8, 13]
    trail_ticks_list = [12, 16, 20, 24, 30]
    max_hold_list = [30, 60, 90, 120]

    total = len(range_mults) * len(ema_pullbacks) * len(trail_ticks_list) * len(max_hold_list)
    print(f"  Layer 3 sweep: {total} combinations on {label} ...")

    count = 0
    for rm in range_mults:
        for ep in ema_pullbacks:
            sigs, exits = layer3_signals(df, range_mult=rm, ema_pullback=ep)
            for trail_t in trail_ticks_list:
                for max_h in max_hold_list:
                    trades = run_backtest(
                        df, sigs, exits,
                        stop_ticks=trail_t, target_ticks=None,
                        max_hold_bars=max_h, contracts=contracts,
                        trailing=True, trailing_ticks=trail_t,
                        layer_name="L3_VolMom",
                    )
                    pnl = sum(t.net_pnl for t in trades)
                    results.append({
                        "range_mult": rm, "ema_pullback": ep,
                        "trail_ticks": trail_t, "max_hold": max_h,
                        "total_pnl": pnl, "trades": len(trades), "trade_list": trades,
                    })
                    count += 1
                    if count % 50 == 0:
                        print(f"    {count}/{total} done ...")

    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ═════════════════════════════════════════════════════════════════════

def analyze_trades(trades: list[TradeRecord], label: str, months: int = 12) -> dict:
    """Compute metrics from trade list."""
    if not trades:
        return {
            "label": label, "total_pnl": 0, "monthly_avg": 0, "trades": 0,
            "win_rate": 0, "trades_per_day": 0, "worst_month": 0,
            "monthly_breakdown": {}, "daily_pnl": {},
        }

    total_pnl = sum(t.net_pnl for t in trades)
    winners = [t for t in trades if t.net_pnl > 0]
    wr = len(winners) / len(trades) * 100 if trades else 0

    # Monthly breakdown
    monthly = defaultdict(float)
    monthly_trades = defaultdict(int)
    daily = defaultdict(float)
    daily_trades = defaultdict(int)

    for t in trades:
        et = t.entry_time
        if hasattr(et, 'strftime'):
            m_key = et.strftime("%Y-%m")
            d_key = et.strftime("%Y-%m-%d")
        else:
            m_key = str(et)[:7]
            d_key = str(et)[:10]
        monthly[m_key] += t.net_pnl
        monthly_trades[m_key] += 1
        daily[d_key] += t.net_pnl
        daily_trades[d_key] += 1

    n_trading_days = len(daily) if daily else 1
    trades_per_day = len(trades) / n_trading_days

    worst_month = min(monthly.values()) if monthly else 0
    worst_day = min(daily.values()) if daily else 0
    best_day = max(daily.values()) if daily else 0

    # Max drawdown from daily equity curve
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in sorted(daily.keys()):
        cumulative += daily[d]
        peak = max(peak, cumulative)
        dd = cumulative - peak
        max_dd = min(max_dd, dd)

    # Consistency: best day as % of total
    consistency = (best_day / total_pnl * 100) if total_pnl > 0 else 100.0

    return {
        "label": label,
        "total_pnl": total_pnl,
        "monthly_avg": total_pnl / months,
        "trades": len(trades),
        "win_rate": wr,
        "trades_per_day": trades_per_day,
        "worst_month": worst_month,
        "worst_day": worst_day,
        "best_day": best_day,
        "max_drawdown": max_dd,
        "consistency_pct": consistency,
        "monthly_breakdown": dict(monthly),
        "monthly_trades": dict(monthly_trades),
        "daily_pnl": dict(daily),
        "n_trading_days": n_trading_days,
        "months_with_0_trades": 0,  # filled in later
    }


def print_layer_result(name: str, yr1: dict, yr2: dict, params: dict):
    """Print a single layer's results."""
    passed = yr2["total_pnl"] > 0

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Params: {params}")
    print(f"  Year 1: ${yr1['total_pnl']:,.0f} total, ${yr1['monthly_avg']:,.0f}/month, "
          f"{yr1['trades']} trades, {yr1['win_rate']:.1f}% WR")
    print(f"  Year 2: ${yr2['total_pnl']:,.0f} total, ${yr2['monthly_avg']:,.0f}/month, "
          f"{yr2['trades']} trades, {yr2['win_rate']:.1f}% WR")
    print(f"  Trades per day: {yr1['trades_per_day']:.1f} (Y1), {yr2['trades_per_day']:.1f} (Y2)")
    print(f"  Worst month: ${yr1['worst_month']:,.0f} (Y1), ${yr2['worst_month']:,.0f} (Y2)")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def print_combined_report(all_trades: list[TradeRecord]):
    """Print the combined always-on system report."""
    if not all_trades:
        print("\n  No surviving strategies. Combined system: EMPTY.")
        return {}

    # Monthly breakdown across all 24 months
    monthly = defaultdict(float)
    monthly_trades = defaultdict(int)
    daily = defaultdict(float)

    for t in all_trades:
        et = t.entry_time
        if hasattr(et, 'strftime'):
            m_key = et.strftime("%Y-%m")
            d_key = et.strftime("%Y-%m-%d")
        else:
            m_key = str(et)[:7]
            d_key = str(et)[:10]
        monthly[m_key] += t.net_pnl
        monthly_trades[m_key] += 1
        daily[d_key] += t.net_pnl

    total_pnl = sum(t.net_pnl for t in all_trades)
    n_months = len(monthly)
    n_days = len(daily)

    # Per-year splits
    yr1_trades = [t for t in all_trades
                  if (hasattr(t.entry_time, 'year') and t.entry_time < YR2_START)
                  or str(t.entry_time)[:4] == '2024']
    yr2_trades = [t for t in all_trades if t not in yr1_trades]

    yr1_pnl = sum(t.net_pnl for t in yr1_trades)
    yr2_pnl = sum(t.net_pnl for t in yr2_trades)

    # Max DD
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    worst_day = 0.0
    best_day = 0.0
    for d in sorted(daily.keys()):
        cumulative += daily[d]
        peak = max(peak, cumulative)
        dd = cumulative - peak
        max_dd = min(max_dd, dd)
        worst_day = min(worst_day, daily[d])
        best_day = max(best_day, daily[d])

    consistency = (best_day / total_pnl * 100) if total_pnl > 0 else 100.0

    months_gte_7k = sum(1 for v in monthly.values() if v >= 7000)
    months_gte_3500 = sum(1 for v in monthly.values() if v >= 3500)
    months_profitable = sum(1 for v in monthly.values() if v > 0)
    months_zero_trades = sum(1 for m, c in monthly_trades.items() if c == 0)

    print(f"\n{'═' * 60}")
    print(f"  COMBINED ALWAYS-ON SYSTEM")
    print(f"{'═' * 60}")
    print(f"  Year 1: ${yr1_pnl:,.0f} ({len(yr1_trades)} trades, "
          f"${yr1_pnl / 12:,.0f}/month)")
    print(f"  Year 2: ${yr2_pnl:,.0f} ({len(yr2_trades)} trades, "
          f"${yr2_pnl / 12:,.0f}/month)")
    print(f"  Total:  ${total_pnl:,.0f} across {n_months} months")
    print(f"  Avg trades/day: {len(all_trades) / max(n_days, 1):.1f}")

    print(f"\n  Monthly breakdown:")
    for m in sorted(monthly.keys()):
        status = "✅" if monthly[m] >= 0 else "❌"
        print(f"    {m}: ${monthly[m]:>+10,.0f}  ({monthly_trades[m]:>3} trades) {status}")

    print(f"\n  Months with 0 trades: {months_zero_trades} (TARGET: 0)")
    print(f"  Months profitable: {months_profitable}/{n_months}")
    print(f"  Worst month: ${min(monthly.values()):,.0f}" if monthly else "  N/A")

    print(f"\n  PROP FIRM CHECK (Topstep 150K):")
    wd_ok = worst_day > -3000
    dd_ok = max_dd > -4500
    con_ok = consistency < 50
    print(f"    Worst day:    ${worst_day:>+10,.0f}  (limit: -$3,000) {'✅' if wd_ok else '❌'}")
    print(f"    Max drawdown: ${max_dd:>+10,.0f}  (limit: -$4,500) {'✅' if dd_ok else '❌'}")
    print(f"    Consistency:  best day = {consistency:.1f}% of total (limit: <50%) {'✅' if con_ok else '❌'}")

    print(f"\n  MONTHLY TARGET CHECK ($7,000/month):")
    print(f"    Months >= $7K:   {months_gte_7k}/{n_months}")
    print(f"    Months >= $3.5K: {months_gte_3500}/{n_months}")
    print(f"    Average monthly: ${total_pnl / max(n_months, 1):,.0f}")
    print(f"{'═' * 60}")

    return {
        "total_pnl": total_pnl,
        "yr1_pnl": yr1_pnl,
        "yr2_pnl": yr2_pnl,
        "monthly": dict(monthly),
        "monthly_trades": dict(monthly_trades),
        "worst_day": worst_day,
        "max_drawdown": max_dd,
        "consistency_pct": consistency,
        "months_gte_7k": months_gte_7k,
        "months_gte_3500": months_gte_3500,
        "months_profitable": months_profitable,
        "months_zero_trades": months_zero_trades,
        "avg_trades_per_day": len(all_trades) / max(n_days, 1),
        "total_trades": len(all_trades),
    }


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()

    print("=" * 60)
    print("  ALWAYS-ON TRADING SYSTEM — Parameter Sweep + Dual-Year OOS")
    print("=" * 60)

    df = load_data()
    df = add_initial_balance(df)  # pre-compute IB for all
    yr1, yr2 = split_years(df)
    print(f"  Year 1: {len(yr1):,} bars")
    print(f"  Year 2: {len(yr2):,} bars")

    surviving_layers: list[dict] = []
    all_yr1_trades: list[TradeRecord] = []
    all_yr2_trades: list[TradeRecord] = []

    # ── LAYER 1: Microstructure Mean Reversion ──────────────────────
    print(f"\n{'━' * 60}")
    print("  LAYER 1 — MICROSTRUCTURE MEAN REVERSION")
    print(f"{'━' * 60}")

    yr1_l1 = sweep_layer1(yr1, "Year 1")
    best_l1 = yr1_l1[0] if yr1_l1 else None

    if best_l1 and best_l1["total_pnl"] > 0:
        params = {k: v for k, v in best_l1.items() if k not in ("total_pnl", "trades", "trade_list")}
        print(f"  Best Year 1: ${best_l1['total_pnl']:,.0f} ({best_l1['trades']} trades)")
        print(f"  Params: {params}")

        # Run Year 2 with same params
        sigs2, exits2 = layer1_signals(yr2, z_threshold=params["z_threshold"], lookback=params["lookback"])
        yr2_trades = run_backtest(
            yr2, sigs2, exits2,
            stop_ticks=params["stop_ticks"], target_ticks=None,
            max_hold_bars=params["max_hold"], contracts=3, layer_name="L1_VWAP_MR",
        )
        yr1_m = analyze_trades(best_l1["trade_list"], "Year 1")
        yr2_m = analyze_trades(yr2_trades, "Year 2")
        passed = print_layer_result("LAYER 1 — VWAP Mean Reversion", yr1_m, yr2_m, params)

        if passed:
            surviving_layers.append({"name": "L1_VWAP_MR", "params": params, "yr1": yr1_m, "yr2": yr2_m})
            all_yr1_trades.extend(best_l1["trade_list"])
            all_yr2_trades.extend(yr2_trades)
    else:
        print("  Layer 1: no profitable configuration found on Year 1.")

    gc.collect()

    # ── LAYER 2: Session Structure ──────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  LAYER 2 — SESSION STRUCTURE TRADES")
    print(f"{'━' * 60}")

    yr1_l2 = sweep_layer2(yr1, "Year 1")

    sub_names = {
        "2A": "Opening Range", "2B": "Lunch Reversion",
        "2C": "Afternoon Trend", "2D": "Closing Imbalance",
    }

    for key in ["2A", "2B", "2C", "2D"]:
        best = yr1_l2[key]
        if best is None or best["total_pnl"] <= 0:
            print(f"  Layer {key} ({sub_names[key]}): no profitable config on Year 1.")
            continue

        params = {k: v for k, v in best.items() if k not in ("total_pnl", "trades", "trade_list")}
        print(f"\n  Best {key} Year 1: ${best['total_pnl']:,.0f} ({best['trades']} trades)")

        # Run Year 2
        if key == "2A":
            s2, e2, sa2, ta2 = layer2a_signals(yr2, tp_mult=params["tp_mult"], sl_mult=params["sl_mult"])
            yr2_trades = run_backtest_dynamic_sl(
                yr2, s2, e2, sa2, ta2,
                max_hold_bars=params["max_hold"], contracts=2, layer_name=f"L{key}_{sub_names[key].replace(' ', '')}",
            )
        elif key == "2B":
            s2, e2 = layer2b_signals(yr2, rsi_period=params["rsi_period"],
                                     rsi_ob=params["rsi_ob"], rsi_os=params["rsi_os"])
            yr2_trades = run_backtest(
                yr2, s2, e2,
                stop_ticks=params["stop_ticks"], target_ticks=None,
                max_hold_bars=params["max_hold"], contracts=2, layer_name="L2B_Lunch",
            )
        elif key == "2C":
            s2, e2 = layer2c_signals(yr2, ema_fast=params["ema_fast"], ema_slow=params["ema_slow"])
            yr2_trades = run_backtest(
                yr2, s2, e2,
                stop_ticks=params["trail_ticks"], target_ticks=None,
                max_hold_bars=params["max_hold"], contracts=2,
                trailing=True, trailing_ticks=params["trail_ticks"],
                layer_name="L2C_Afternoon",
            )
        elif key == "2D":
            s2, e2 = layer2d_signals(yr2, threshold_pct=params["threshold_pct"])
            yr2_trades = run_backtest(
                yr2, s2, e2,
                stop_ticks=params["stop_ticks"], target_ticks=None,
                max_hold_bars=15, contracts=2, layer_name="L2D_Close",
            )
        else:
            continue

        yr1_m = analyze_trades(best["trade_list"], "Year 1")
        yr2_m = analyze_trades(yr2_trades, "Year 2")
        passed = print_layer_result(f"LAYER {key} — {sub_names[key]}", yr1_m, yr2_m, params)

        if passed:
            surviving_layers.append({"name": f"L{key}_{sub_names[key]}", "params": params, "yr1": yr1_m, "yr2": yr2_m})
            all_yr1_trades.extend(best["trade_list"])
            all_yr2_trades.extend(yr2_trades)

    gc.collect()

    # ── LAYER 3: Volatility-Adaptive Momentum ──────────────────────
    print(f"\n{'━' * 60}")
    print("  LAYER 3 — VOLATILITY-ADAPTIVE MOMENTUM")
    print(f"{'━' * 60}")

    yr1_l3 = sweep_layer3(yr1, "Year 1")
    best_l3 = yr1_l3[0] if yr1_l3 else None

    if best_l3 and best_l3["total_pnl"] > 0:
        params = {k: v for k, v in best_l3.items() if k not in ("total_pnl", "trades", "trade_list")}
        print(f"  Best Year 1: ${best_l3['total_pnl']:,.0f} ({best_l3['trades']} trades)")

        sigs2, exits2 = layer3_signals(yr2, range_mult=params["range_mult"], ema_pullback=params["ema_pullback"])
        yr2_trades = run_backtest(
            yr2, sigs2, exits2,
            stop_ticks=params["trail_ticks"], target_ticks=None,
            max_hold_bars=params["max_hold"], contracts=3,
            trailing=True, trailing_ticks=params["trail_ticks"],
            layer_name="L3_VolMom",
        )
        yr1_m = analyze_trades(best_l3["trade_list"], "Year 1")
        yr2_m = analyze_trades(yr2_trades, "Year 2")
        passed = print_layer_result("LAYER 3 — Volatility Momentum", yr1_m, yr2_m, params)

        if passed:
            surviving_layers.append({"name": "L3_VolMom", "params": params, "yr1": yr1_m, "yr2": yr2_m})
            all_yr1_trades.extend(best_l3["trade_list"])
            all_yr2_trades.extend(yr2_trades)
    else:
        print("  Layer 3: no profitable configuration found on Year 1.")

    gc.collect()

    # ── COMBINED SYSTEM ─────────────────────────────────────────────
    all_trades = all_yr1_trades + all_yr2_trades
    combined_report = print_combined_report(all_trades)

    # ── Layer summary ───────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  SURVIVING LAYERS: {len(surviving_layers)}")
    for sl in surviving_layers:
        print(f"    ✅ {sl['name']}")
    if not surviving_layers:
        print("    ❌ No layers survived dual-year validation.")
    print(f"{'─' * 60}")

    # ── Save results ────────────────────────────────────────────────
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "surviving_layers": [
            {
                "name": s["name"],
                "params": s["params"],
                "yr1_pnl": s["yr1"]["total_pnl"],
                "yr1_trades": s["yr1"]["trades"],
                "yr1_monthly_avg": s["yr1"]["monthly_avg"],
                "yr2_pnl": s["yr2"]["total_pnl"],
                "yr2_trades": s["yr2"]["trades"],
                "yr2_monthly_avg": s["yr2"]["monthly_avg"],
            }
            for s in surviving_layers
        ],
        "combined": combined_report,
    }

    out_path = REPORTS_DIR / "always_on_v1.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    elapsed = _time.time() - t0
    print(f"  Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
