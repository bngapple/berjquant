#!/usr/bin/env python3
"""
HTF Swing System — 15m/1h/4h bars. Honest execution. No trailing stops.

6 strategy families × 3 timeframes = 18 combinations.
Full cost model, conservative ordering, no same-bar exits.

Usage:
    python3 run_htf_swing.py
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

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Cost model ───────────────────────────────────────────────────────
TICK_SIZE = 0.25
POINT_VALUE = 2.0      # MNQ
SLIP_TICKS = 2          # per side
COMM_PER_SIDE = 0.62
EXCH_PER_SIDE = 0.27

def rt_cost(c: int) -> float:
    return (COMM_PER_SIDE + EXCH_PER_SIDE) * 2 * c + SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * c

SLIP_PTS = SLIP_TICKS * TICK_SIZE  # 0.50

# ── Data ─────────────────────────────────────────────────────────────
from zoneinfo import ZoneInfo
_ET = ZoneInfo("US/Eastern")
YR1_END = datetime(2025, 3, 19, tzinfo=_ET)


def load_and_resample(path: str, label: str) -> dict[str, pl.DataFrame]:
    """Load 1m data, filter RTH, resample to 15m/1h/4h."""
    df = pl.read_parquet(path)

    # Add ET columns
    if "ts_et" not in df.columns:
        ts = df["timestamp"]
        if hasattr(ts.dtype, 'time_zone') and ts.dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
        )
        df = df.with_columns(
            pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
        )
        df = df.drop("_utc")
    df = df.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    df = df.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])

    # RTH only: 9:30-16:00 ET
    rth = df.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))
    print(f"  {label}: {len(rth):,} RTH 1m bars")

    result = {}
    for tf_min, tf_label in [(15, "15m"), (60, "1h"), (240, "4h")]:
        r = (
            rth.group_by_dynamic("ts_et", every=f"{tf_min}m")
            .agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("date_et").last(),
                pl.col("hhmm").last(),
            ])
            .filter(pl.col("open").is_not_null())
            .sort("ts_et")
        )
        # Rename ts_et to timestamp for consistency
        r = r.rename({"ts_et": "timestamp"})
        result[tf_label] = r
        print(f"    {tf_label}: {len(r):,} bars")

    return result


# ═════════════════════════════════════════════════════════════════════
# INDICATORS (vectorized, polars)
# ═════════════════════════════════════════════════════════════════════

def calc_vwap(df: pl.DataFrame) -> np.ndarray:
    """Daily VWAP (resets each date_et)."""
    tp = (df["high"].to_numpy() + df["low"].to_numpy() + df["close"].to_numpy()) / 3
    vol = df["volume"].to_numpy().astype(float)
    dates = df["date_et"].to_list()

    vwap = np.full(len(df), np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_date = None
    for i in range(len(df)):
        if dates[i] != prev_date:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_date = dates[i]
        cum_tpv += tp[i] * vol[i]
        cum_vol += vol[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else tp[i]
    return vwap


def calc_atr(highs, lows, closes, period=14):
    """ATR."""
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calc_ema(data, period):
    """EMA."""
    n = len(data)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    k = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


def calc_rsi(closes, period=14):
    """RSI."""
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    if avg_l == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100 - 100 / (1 + avg_g / avg_l)
    return rsi


# ═════════════════════════════════════════════════════════════════════
# HONEST BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    direction: int
    entry_px: float
    exit_px: float
    contracts: int
    net_pnl: float
    entry_time: object
    exit_time: object
    bars_held: int
    reason: str
    strategy: str


def backtest(
    opens, highs, lows, closes, timestamps, hhmm_arr,
    signals,          # +1/-1/0 per bar
    sl_ticks,         # fixed stop in ticks
    tp_ticks,         # fixed target in ticks
    max_hold,         # bars
    contracts,
    strategy_name="",
    flatten_time=1545, # flatten by 15:45
    allow_multiday=False,
) -> list[Trade]:
    """Honest bar-by-bar backtest. No same-bar exits. Conservative ordering."""
    n = len(opens)
    cost = rt_cost(contracts)

    trades = []
    in_pos = False
    direction = 0
    entry_px = 0.0
    entry_bar = 0
    stop_px = 0.0
    target_px = 0.0
    pending = 0

    for i in range(n):
        h = hhmm_arr[i] if i < len(hhmm_arr) else 0

        # Flatten near close (skip for multi-day)
        if in_pos and not allow_multiday and h >= flatten_time and i > entry_bar:
            ex = closes[i] - direction * SLIP_PTS
            raw = (ex - entry_px) * direction * POINT_VALUE * contracts
            trades.append(Trade(direction, entry_px, ex, contracts, raw - cost,
                                timestamps[entry_bar], timestamps[i], i - entry_bar,
                                "time_exit", strategy_name))
            in_pos = False
            pending = 0
            continue

        # Execute pending
        if pending != 0 and not in_pos:
            if not allow_multiday and h >= flatten_time - 15:  # don't enter near close
                pending = 0
            else:
                entry_px = opens[i] + int(pending) * SLIP_PTS
                direction = int(pending)
                entry_bar = i
                stop_px = entry_px - direction * sl_ticks * TICK_SIZE
                target_px = entry_px + direction * tp_ticks * TICK_SIZE
                in_pos = True
                pending = 0

        # Manage (skip entry bar)
        if in_pos and i > entry_bar:
            bh = i - entry_bar
            ex = None
            reason = ""

            # STOP FIRST (conservative)
            if direction == 1 and lows[i] <= stop_px:
                ex = stop_px; reason = "stop_loss"
            elif direction == -1 and highs[i] >= stop_px:
                ex = stop_px; reason = "stop_loss"

            # TARGET (only if stop not hit)
            if ex is None:
                if direction == 1 and highs[i] >= target_px:
                    ex = target_px; reason = "take_profit"
                elif direction == -1 and lows[i] <= target_px:
                    ex = target_px; reason = "take_profit"

            # Max hold
            if ex is None and bh >= max_hold:
                ex = closes[i]; reason = "max_hold"

            if ex is not None:
                ex -= direction * SLIP_PTS
                raw = (ex - entry_px) * direction * POINT_VALUE * contracts
                trades.append(Trade(direction, entry_px, ex, contracts, raw - cost,
                                    timestamps[entry_bar], timestamps[i], bh,
                                    reason, strategy_name))
                in_pos = False

        # New signal
        if not in_pos and i < len(signals) and signals[i] != 0:
            pending = signals[i]

    return trades


# ═════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS
# ═════════════════════════════════════════════════════════════════════

def sig_vwap_mr(df, threshold=1.5):
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    vwap = calc_vwap(df)
    atr = calc_atr(highs, lows, closes, 14)
    n = len(df)
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(atr[i]) or atr[i] < 0.5:
            continue
        dist = (closes[i] - vwap[i]) / atr[i]
        if dist < -threshold:
            sigs[i] = 1
        elif dist > threshold:
            sigs[i] = -1
    return sigs


def sig_ema_pullback(df, fast=8, slow=21, use_rsi_filter=False):
    closes = df["close"].to_numpy()
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    rsi = calc_rsi(closes, 14) if use_rsi_filter else None
    n = len(closes)
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(ema_f[i]) or np.isnan(ema_s[i]):
            continue
        if use_rsi_filter and (np.isnan(rsi[i]) or not (40 <= rsi[i] <= 60)):
            continue
        # Bullish trend + pullback to fast EMA
        if ema_f[i] > ema_s[i] and closes[i-1] > ema_f[i-1] and closes[i] <= ema_f[i]:
            sigs[i] = 1
        # Bearish trend + pullback
        elif ema_f[i] < ema_s[i] and closes[i-1] < ema_f[i-1] and closes[i] >= ema_f[i]:
            sigs[i] = -1
    return sigs, ema_f, ema_s


def sig_rsi_extreme(df, period=14, ob=70, os_=30):
    closes = df["close"].to_numpy()
    rsi = calc_rsi(closes, period)
    n = len(closes)
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < os_:
            sigs[i] = 1
        elif rsi[i] > ob:
            sigs[i] = -1
    return sigs


def sig_ib_breakout(df, ib_range_filter=True):
    """Opening range breakout on 15m bars. IB = 9:30-10:00 = first 2 bars."""
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    hhmm = df["hhmm"].to_numpy()
    dates = df["date_et"].to_list()
    n = len(df)

    # Compute IB per day
    ib_data = {}
    for i in range(n):
        if hhmm[i] < 1000:
            d = dates[i]
            if d not in ib_data:
                ib_data[d] = {"high": highs[i], "low": lows[i]}
            else:
                ib_data[d]["high"] = max(ib_data[d]["high"], highs[i])
                ib_data[d]["low"] = min(ib_data[d]["low"], lows[i])

    # Rolling IB range percentiles for filter
    ib_ranges = []
    ib_dates = sorted(ib_data.keys())
    for d in ib_dates:
        ib_ranges.append(ib_data[d]["high"] - ib_data[d]["low"])

    sigs = np.zeros(n, dtype=np.int8)
    sl_ticks_arr = np.full(n, 40, dtype=np.int32)
    tp_ticks_arr = np.full(n, 60, dtype=np.int32)
    traded = {}

    for i in range(n):
        if hhmm[i] < 1000 or hhmm[i] >= 1530:
            continue
        d = dates[i]
        if d not in ib_data or d in traded:
            continue

        ib_h = ib_data[d]["high"]
        ib_l = ib_data[d]["low"]
        ib_r = ib_h - ib_l

        if ib_r < 2.0:
            continue

        # Filter: IB range between 25th-75th percentile
        if ib_range_filter and len(ib_ranges) > 20:
            idx = ib_dates.index(d) if d in ib_dates else -1
            if idx > 20:
                recent = ib_ranges[max(0, idx-50):idx]
                p25, p75 = np.percentile(recent, 25), np.percentile(recent, 75)
                if ib_r < p25 or ib_r > p75:
                    continue

        if highs[i] > ib_h:
            sigs[i] = 1
            sl = max(int((ib_h - ib_l) / TICK_SIZE), 8)
            sl_ticks_arr[i] = sl
            traded[d] = True
        elif lows[i] < ib_l:
            sigs[i] = -1
            sl = max(int((ib_h - ib_l) / TICK_SIZE), 8)
            sl_ticks_arr[i] = sl
            traded[d] = True

    return sigs, sl_ticks_arr, tp_ticks_arr


def sig_momentum_bar(df, atr_mult=1.5, vol_mult=1.5):
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    volumes = df["volume"].to_numpy().astype(float)
    atr = calc_atr(highs, lows, closes, 14)
    ema21 = calc_ema(closes, 21)
    n = len(closes)

    # Rolling avg volume
    avg_vol = np.full(n, np.nan)
    for i in range(20, n):
        avg_vol[i] = np.mean(volumes[i-20:i])

    sigs = np.zeros(n, dtype=np.int8)
    sl_arr = np.full(n, 40, dtype=np.int32)

    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(ema21[i]) or np.isnan(avg_vol[i]):
            continue
        bar_range = highs[i] - lows[i]
        if bar_range < atr[i] * atr_mult:
            continue
        if volumes[i] < avg_vol[i] * vol_mult:
            continue

        bar_dir = 1 if closes[i] > opens[i] else -1
        trend_dir = 1 if closes[i] > ema21[i] else -1

        if bar_dir == trend_dir:
            sigs[i] = bar_dir
            sl_arr[i] = max(int(bar_range / TICK_SIZE), 12)

    return sigs, sl_arr


def sig_session_transition(df):
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    hhmm = df["hhmm"].to_numpy()
    dates = df["date_et"].to_list()
    n = len(df)
    vwap = calc_vwap(df)

    # Track morning performance
    morning_dir = {}
    for i in range(n):
        if 930 <= hhmm[i] < 1300:
            d = dates[i]
            if d not in morning_dir:
                morning_dir[d] = {"open": opens[i], "close": closes[i]}
            else:
                morning_dir[d]["close"] = closes[i]

    sigs = np.zeros(n, dtype=np.int8)
    traded = {}
    for i in range(n):
        if hhmm[i] != 1300 and not (1300 <= hhmm[i] < 1315):
            continue
        d = dates[i]
        if d in traded or d not in morning_dir:
            continue
        if np.isnan(vwap[i]):
            continue

        md = morning_dir[d]
        morning_bull = md["close"] > md["open"]
        above_vwap = closes[i] > vwap[i]

        if morning_bull and above_vwap:
            sigs[i] = 1  # continuation
        elif morning_bull and not above_vwap:
            sigs[i] = -1  # reversal
        elif not morning_bull and not above_vwap:
            sigs[i] = -1  # continuation
        elif not morning_bull and above_vwap:
            sigs[i] = 1  # reversal
        traded[d] = True

    return sigs


# ═════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════

def calc_metrics(trades):
    if not trades:
        return {"pnl": 0, "n": 0, "wr": 0, "monthly_avg": 0, "sharpe": 0,
                "bars_mean": 0, "bars_min": 0, "worst_month": 0, "worst_day": 0,
                "max_dd": 0, "monthly": {}, "daily": {}, "n_months": 0,
                "months_pos": 0, "same_bar": 0}

    pnls = [t.net_pnl for t in trades]
    bars = [t.bars_held for t in trades]
    w = sum(1 for p in pnls if p > 0)

    monthly = defaultdict(float)
    daily = defaultdict(float)
    for t in trades:
        m = str(t.entry_time)[:7]
        d = str(t.entry_time)[:10]
        monthly[m] += t.net_pnl
        daily[d] += t.net_pnl

    dv = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0

    cum = 0; peak = 0; mdd = 0
    for d in sorted(daily.keys()):
        cum += daily[d]; peak = max(peak, cum); mdd = min(mdd, cum - peak)

    nm = max(len(monthly), 1)
    return {
        "pnl": sum(pnls), "n": len(trades), "wr": w / len(trades) * 100,
        "monthly_avg": sum(pnls) / nm, "sharpe": sharpe,
        "bars_mean": np.mean(bars), "bars_min": min(bars), "bars_max": max(bars),
        "worst_month": min(monthly.values()) if monthly else 0,
        "worst_day": min(daily.values()) if daily else 0,
        "best_day": max(daily.values()) if daily else 0,
        "max_dd": mdd, "monthly": dict(monthly), "daily": dict(daily),
        "n_months": nm, "months_pos": sum(1 for v in monthly.values() if v > 0),
        "same_bar": sum(1 for b in bars if b == 0),
    }


# ═════════════════════════════════════════════════════════════════════
# SWEEP + VALIDATE
# ═════════════════════════════════════════════════════════════════════

def extract_arrays(df):
    return (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
            df["close"].to_numpy(), df["timestamp"].to_list(),
            df["hhmm"].to_numpy() if "hhmm" in df.columns else np.zeros(len(df), dtype=np.int32))


def sweep_and_validate(name, tf, yr1_df, yr2_df, blind_df, sig_fn, sl_list, tp_list,
                       hold_list, contract_list, min_trades=50):
    """Grid search on Y1, validate on Y2 and blind."""
    print(f"  {name} @ {tf} ...", end=" ", flush=True)

    sigs_y1 = sig_fn(yr1_df)
    # Handle signal generators that return tuples
    extra_y1 = None
    if isinstance(sigs_y1, tuple):
        if len(sigs_y1) == 3:
            sigs_y1, sl_arr_y1, tp_arr_y1 = sigs_y1
            extra_y1 = (sl_arr_y1, tp_arr_y1)
        else:
            sigs_y1 = sigs_y1[0]

    o, h, l, c, ts, hm = extract_arrays(yr1_df)
    total_sigs = np.sum(np.abs(sigs_y1))
    if total_sigs < 10:
        print(f"only {total_sigs} signals, skip")
        return None

    best = None
    count = 0
    for sl in sl_list:
        for tp in tp_list:
            if tp <= sl:
                continue
            for hold in hold_list:
                for contracts in contract_list:
                    trades = backtest(o, h, l, c, ts, hm, sigs_y1,
                                      sl, tp, hold, contracts, name)
                    m = calc_metrics(trades)
                    if m["n"] < min_trades or m["bars_mean"] < 2:
                        continue
                    if best is None or m["sharpe"] > best["sharpe"]:
                        best = {"sharpe": m["sharpe"], "sl": sl, "tp": tp,
                                "hold": hold, "contracts": contracts,
                                "y1": m, "y1_trades": trades}
                    count += 1

    if best is None or best["y1"]["pnl"] <= 0:
        print(f"no profitable config ({count} tested)")
        return None

    print(f"Y1: ${best['y1']['monthly_avg']:,.0f}/mo, Sharpe={best['sharpe']:.2f}")

    # Validate Y2
    sigs_y2 = sig_fn(yr2_df)
    if isinstance(sigs_y2, tuple):
        sigs_y2 = sigs_y2[0]
    o2, h2, l2, c2, ts2, hm2 = extract_arrays(yr2_df)
    y2_trades = backtest(o2, h2, l2, c2, ts2, hm2, sigs_y2,
                          best["sl"], best["tp"], best["hold"], best["contracts"], name)
    y2m = calc_metrics(y2_trades)

    # Validate blind
    sigs_bl = sig_fn(blind_df)
    if isinstance(sigs_bl, tuple):
        sigs_bl = sigs_bl[0]
    ob, hb, lb, cb, tsb, hmb = extract_arrays(blind_df)
    bl_trades = backtest(ob, hb, lb, cb, tsb, hmb, sigs_bl,
                          best["sl"], best["tp"], best["hold"], best["contracts"], name)
    blm = calc_metrics(bl_trades)

    y2_pass = y2m["pnl"] > 0 and y2m["n"] >= 20 and y2m["bars_mean"] >= 2
    bl_pass = blm["pnl"] > 0 and blm["n"] >= 20
    validated = y2_pass and bl_pass

    status = "✅ PASS" if validated else ("⚠️ Y2 only" if y2_pass else "❌ FAIL")
    print(f"    Y2: ${y2m['monthly_avg']:,.0f}/mo, {y2m['n']}t, {y2m['wr']:.0f}%WR, "
          f"{y2m['bars_mean']:.1f}bars | "
          f"Blind: ${blm['monthly_avg']:,.0f}/mo, {blm['n']}t | {status}")

    if y2m["same_bar"] > 0:
        print(f"    ⚠️  {y2m['same_bar']} same-bar exits on Y2!")

    return {
        "name": name, "tf": tf, "status": status, "validated": validated,
        "params": {"sl": best["sl"], "tp": best["tp"], "hold": best["hold"],
                   "contracts": best["contracts"]},
        "y1": {k: v for k, v in best["y1"].items() if k not in ("daily",)},
        "y2": {k: v for k, v in y2m.items() if k not in ("daily",)},
        "blind": {k: v for k, v in blm.items() if k not in ("daily",)},
        "y1_trades": best["y1_trades"], "y2_trades": y2_trades, "blind_trades": bl_trades,
    }


# ═════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═════════════════════════════════════════════════════════════════════

def run_mc(trades, n_sims=5000, pnl_mult=1.0):
    daily = defaultdict(list)
    for t in trades:
        d = str(t.entry_time)[:10]
        daily[d].append(t.net_pnl * pnl_mult)
    days = list(daily.values())
    nd = len(days)
    if nd == 0:
        return {"pass": 0, "blow": 1}
    passed = 0; blown = 0
    for sim in range(n_sims):
        rng = np.random.RandomState(sim)
        order = rng.permutation(nd)
        cum = 0; peak = 0; p = False; ok = True
        for idx in order:
            dp = min(sum(days[idx]), 0) if sum(days[idx]) < -3000 else sum(days[idx])
            cum += dp; peak = max(peak, cum)
            if cum - peak <= -4500: ok = False; break
            if cum >= 9000: p = True
        if not ok: blown += 1
        if p and ok: passed += 1
    return {"pass": passed/n_sims, "blow": blown/n_sims}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    print("═" * 70)
    print("  HTF SWING SYSTEM — 15m / 1h / 4h bars")
    print(f"  Cost per RT at 10 MNQ: ${rt_cost(10):.2f}")
    print("═" * 70)

    # Load and resample
    print("\n  Loading data ...")
    main_data = load_and_resample("data/processed/MNQ/1m/full_2yr.parquet", "Main (2024-2026)")
    blind_data = load_and_resample("data/processed/MNQ/1m/databento_extended.parquet", "Blind (2022-2024)")

    # Split main into Y1/Y2
    yr1_data = {}
    yr2_data = {}
    for tf, df in main_data.items():
        yr1_data[tf] = df.filter(pl.col("timestamp") < YR1_END)
        yr2_data[tf] = df.filter(pl.col("timestamp") >= YR1_END)
        print(f"  {tf}: Y1={len(yr1_data[tf])} Y2={len(yr2_data[tf])} Blind={len(blind_data[tf])}")

    all_results = []

    # ── Strategy 1: VWAP Mean Reversion ───────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 1: VWAP Mean Reversion")
    print("━" * 70)
    # Stop sizes calibrated to bar ranges: 15m~214 ticks, 1h~500 ticks, 4h~800 ticks
    tf_stops = {
        "15m": {"sl": [60, 100, 160, 240], "tp": [100, 160, 240, 400]},
        "1h":  {"sl": [120, 200, 320, 480], "tp": [200, 320, 480, 640]},
        "4h":  {"sl": [200, 400, 600, 800], "tp": [400, 600, 800, 1200]},
    }
    for tf in ["15m", "1h", "4h"]:
        for thresh in [1.0, 1.5, 2.0, 2.5]:
            r = sweep_and_validate(
                f"VWAP_MR_t{thresh}", tf,
                yr1_data[tf], yr2_data[tf], blind_data[tf],
                lambda d, t=thresh: sig_vwap_mr(d, t),
                sl_list=tf_stops[tf]["sl"],
                tp_list=tf_stops[tf]["tp"],
                hold_list=[5, 10, 20, 40],
                contract_list=[3, 5, 10, 15],
            )
            if r:
                all_results.append(r)
        gc.collect()

    # ── Strategy 2: EMA Pullback ──────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 2: EMA Trend + Pullback")
    print("━" * 70)
    for tf in ["15m", "1h", "4h"]:
        for rsi_f in [False, True]:
            r = sweep_and_validate(
                f"EMA_PB{'_rsi' if rsi_f else ''}", tf,
                yr1_data[tf], yr2_data[tf], blind_data[tf],
                lambda d, rf=rsi_f: sig_ema_pullback(d, 8, 21, rf)[0],
                sl_list=tf_stops[tf]["sl"],
                tp_list=tf_stops[tf]["tp"],
                hold_list=[10, 20, 40],
                contract_list=[3, 5, 10, 15],
            )
            if r:
                all_results.append(r)
        gc.collect()

    # ── Strategy 3: RSI Extremes ──────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 3: RSI Extremes")
    print("━" * 70)
    for tf in ["15m", "1h", "4h"]:
        for period in [7, 14]:
            for ob, os_ in [(70, 30), (75, 25), (80, 20)]:
                r = sweep_and_validate(
                    f"RSI_{period}_{os_}_{ob}", tf,
                    yr1_data[tf], yr2_data[tf], blind_data[tf],
                    lambda d, p=period, o=ob, s=os_: sig_rsi_extreme(d, p, o, s),
                    sl_list=tf_stops[tf]["sl"],
                    tp_list=tf_stops[tf]["tp"],
                    hold_list=[10, 20, 40],
                    contract_list=[3, 5, 10, 15],
                )
                if r:
                    all_results.append(r)
        gc.collect()

    # ── Strategy 4: IB Breakout (15m only) ────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 4: Opening Range Breakout (15m)")
    print("━" * 70)
    for filt in [True, False]:
        r = sweep_and_validate(
            f"IB_BO{'_filt' if filt else ''}", "15m",
            yr1_data["15m"], yr2_data["15m"], blind_data["15m"],
            lambda d, f=filt: sig_ib_breakout(d, f)[0],
            sl_list=[80, 120, 200, 320],
            tp_list=[120, 200, 320, 480],
            hold_list=[10, 20, 40, 60],
            contract_list=[3, 5, 10, 15],
        )
        if r:
            all_results.append(r)
    gc.collect()

    # ── Strategy 5: Momentum Bar ──────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 5: Momentum Continuation")
    print("━" * 70)
    for tf in ["15m", "1h"]:
        for am in [1.0, 1.5, 2.0]:
            for vm in [1.0, 1.5, 2.0]:
                r = sweep_and_validate(
                    f"MOM_a{am}_v{vm}", tf,
                    yr1_data[tf], yr2_data[tf], blind_data[tf],
                    lambda d, a=am, v=vm: sig_momentum_bar(d, a, v)[0],
                    sl_list=tf_stops[tf]["sl"],
                    tp_list=tf_stops[tf]["tp"],
                    hold_list=[5, 10, 20],
                    contract_list=[3, 5, 10, 15],
                )
                if r:
                    all_results.append(r)
        gc.collect()

    # ── Strategy 6: Session Transition ────────────────────────────
    print(f"\n{'━' * 70}")
    print("  STRATEGY 6: Session Transition")
    print("━" * 70)
    for tf in ["15m", "1h"]:
        r = sweep_and_validate(
            "SESSION", tf,
            yr1_data[tf], yr2_data[tf], blind_data[tf],
            lambda d: sig_session_transition(d),
            sl_list=tf_stops[tf]["sl"],
            tp_list=tf_stops[tf]["tp"],
            hold_list=[10, 20, 40],
            contract_list=[3, 5, 10, 15],
        )
        if r:
            all_results.append(r)
    gc.collect()

    # ═════════════════════════════════════════════════════════════
    # RESULTS
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  HTF SWING SYSTEM — RESULTS")
    print("═" * 70)

    # All results table
    print(f"\n  ALL TESTED (with Y1 profit):")
    print(f"  {'─' * 66}")
    print(f"  {'Strategy':<22} {'TF':>4} {'#':>5} {'WR':>5} {'$/mo':>8} {'Shrp':>6} {'Bars':>5} {'Status':<10}")
    print(f"  {'─' * 66}")
    for r in sorted(all_results, key=lambda x: x["y2"]["monthly_avg"], reverse=True):
        y2 = r["y2"]
        print(f"  {r['name']:<22} {r['tf']:>4} {y2['n']:>5} {y2['wr']:>4.0f}% ${y2['monthly_avg']:>7,.0f} "
              f"{y2.get('sharpe',0):>6.2f} {y2['bars_mean']:>5.1f} {r['status']:<10}")
    print(f"  {'─' * 66}")

    # Survivors
    survivors = [r for r in all_results if r["validated"]]
    print(f"\n  VALIDATED SURVIVORS: {len(survivors)}")
    for s in survivors:
        print(f"    ✅ {s['name']} @ {s['tf']}: Y2=${s['y2']['monthly_avg']:,.0f}/mo, "
              f"Blind=${s['blind']['monthly_avg']:,.0f}/mo, "
              f"SL={s['params']['sl']}t TP={s['params']['tp']}t c={s['params']['contracts']}")

    if not survivors:
        print("\n  ❌ None of the 6 strategy families produced a walk-forward valid edge")
        print("     on 15m/1h/4h NQ data across 4.3 years of history.")
        print("     The NQ intraday market may be too efficient for systematic edges")
        print("     at these timeframes with honest execution assumptions.")
        verdict = "NO TRADEABLE EDGE"
    else:
        # Combine survivors
        all_y2 = []; all_bl = []
        for s in survivors:
            all_y2.extend(s["y2_trades"])
            all_bl.extend(s["blind_trades"])
        m_y2 = calc_metrics(all_y2)
        m_bl = calc_metrics(all_bl)

        # Monthly across both
        all_monthly = defaultdict(float)
        for src in [m_y2, m_bl]:
            for m, v in src["monthly"].items():
                all_monthly[m] += v

        print(f"\n  COMBINED PORTFOLIO:")
        print(f"    Y2: ${m_y2['monthly_avg']:,.0f}/mo ({m_y2['n']} trades)")
        print(f"    Blind: ${m_bl['monthly_avg']:,.0f}/mo ({m_bl['n']} trades)")
        print(f"    Months profitable: {sum(1 for v in all_monthly.values() if v > 0)}/{len(all_monthly)}")
        print(f"    Worst month: ${min(all_monthly.values()) if all_monthly else 0:,.0f}")

        # Prop firm
        print(f"\n    Prop firm:")
        print(f"      Y2 worst day: ${m_y2['worst_day']:,.0f} (limit: -$3,000) "
              f"{'✅' if m_y2['worst_day'] > -3000 else '❌'}")
        print(f"      Y2 max DD: ${m_y2['max_dd']:,.0f} (limit: -$4,500) "
              f"{'✅' if m_y2['max_dd'] > -4500 else '❌'}")

        mc_b = run_mc(all_y2, 5000, 1.0)
        mc_c = run_mc(all_y2, 5000, 0.70)
        print(f"\n    MC baseline: {mc_b['pass']:.0%} pass, {mc_b['blow']:.0%} blow-up")
        print(f"    MC conservative: {mc_c['pass']:.0%} pass, {mc_c['blow']:.0%} blow-up")

        verdict = "REAL EDGE" if m_y2["monthly_avg"] > 0 and m_bl["monthly_avg"] > 0 else "MARGINAL"

    # Comparison
    print(f"\n  COMPARISON:")
    print(f"  ┌{'─'*22}┬{'─'*10}┬{'─'*10}┬{'─'*13}┐")
    print(f"  │{'System':<22}│{'Monthly':>10}│{'Real?':>10}│{'Problem':>13}│")
    print(f"  ├{'─'*22}┼{'─'*10}┼{'─'*10}┼{'─'*13}┤")
    print(f"  │{'Always-On v3':<22}│{'$10,913':>10}│{'No':>10}│{'Fake exits':>13}│")
    print(f"  │{'Always-On v4':<22}│{'$503':>10}│{'Yes':>10}│{'Too small':>13}│")
    print(f"  │{'Trailing Profit':<22}│{'$435':>10}│{'Yes':>10}│{'Too small':>13}│")
    if survivors:
        y2_avg = m_y2["monthly_avg"]
        print(f"  │{'HTF Swing':<22}│${y2_avg:>9,.0f}│{'???':>10}│{'':>13}│")
    else:
        print(f"  │{'HTF Swing':<22}│{'$0':>10}│{'No':>10}│{'No edge':>13}│")
    print(f"  └{'─'*22}┴{'─'*10}┴{'─'*10}┴{'─'*13}┘")

    print(f"\n  VERDICT: {verdict}")
    print("═" * 70)

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {k: v for k, v in r.items() if k not in ("y1_trades", "y2_trades", "blind_trades")}
            for r in all_results
        ],
        "survivors": [s["name"] + "@" + s["tf"] for s in survivors],
        "verdict": verdict,
    }
    out = REPORTS_DIR / "htf_swing_v1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
