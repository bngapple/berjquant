#!/usr/bin/env python3
"""
Databento Blind Test — Run Always-On v2 on data it has never seen (pre-2024).

Phase A: Download MNQ 1-minute data from Databento (2022-01 to 2024-03-18)
Phase B: Run each v2 layer with EXACT winning params — no optimization
Phase C: Combined results + comparison to Y1/Y2

Usage:
    python3 run_always_on_databento.py
"""

import gc
import json
import os
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from run_always_on import (
    load_data, add_initial_balance, add_vwap, add_vwap_zscore, add_atr,
    layer1_signals, layer2c_signals, layer3_signals,
    run_backtest, TradeRecord,
    TICK_SIZE, POINT_VALUE, SLIPPAGE_TICKS, COMMISSION_PER_SIDE,
)
from run_always_on_v2 import momentum_continuation_signals

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data/processed/MNQ/1m")
REPORTS_DIR = Path("reports")
EXTENDED_PATH = DATA_DIR / "databento_extended.parquet"

# ── v2 winning params (from reports/always_on_v2.json) ───────────────
V2_L1 = {"z_threshold": 1.35, "lookback": 50, "stop_ticks": 20, "max_hold": 35}
V2_L2C = {"ema_fast": 8, "ema_slow": 13, "trail_ticks": 6, "max_hold": 25}
V2_L3 = {"range_mult": 0.8, "ema_pullback": 5, "trail_ticks": 9, "max_hold": 25}
V2_MOMCONT = {"z": 1.5, "atr_m": 1.0, "trail": 8, "hold": 15}
V2_SIZING = {"L1": 6, "L2C": 10, "L3": 10}


# ═════════════════════════════════════════════════════════════════════
# PHASE A: DOWNLOAD DATA
# ═════════════════════════════════════════════════════════════════════

def check_existing_format():
    """Print the existing data format for reference."""
    existing = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    print(f"  Existing data format:")
    print(f"    Columns: {existing.columns}")
    print(f"    Dtypes: {existing.dtypes}")
    print(f"    Range: {existing['timestamp'].min()} → {existing['timestamp'].max()}")
    return existing.schema


def download_databento() -> pl.DataFrame | None:
    """Download MNQ 1-minute OHLCV from Databento."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  python-dotenv not installed. pip install python-dotenv")
        return None

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("  DATABENTO_API_KEY not found in .env")
        return None

    try:
        import databento as db
    except ImportError:
        print("  databento not installed. pip install databento")
        return None

    client = db.Historical(api_key)

    # Try different symbol formats
    symbols_to_try = [
        ("MNQ.c.0", None),
        ("MNQ.FUT", None),
        ("MNQ", "continuous"),
        ("NQ.c.0", None),
    ]

    data = None
    used_symbol = None
    is_nq = False

    for sym, stype in symbols_to_try:
        try:
            print(f"  Trying symbol: {sym}" + (f" (stype_in={stype})" if stype else "") + " ...")
            kwargs = {
                "dataset": "GLBX.MDP3",
                "symbols": [sym],
                "schema": "ohlcv-1m",
                "start": "2022-01-01",
                "end": "2024-03-18",
            }
            if stype:
                kwargs["stype_in"] = stype
            data = client.timeseries.get_range(**kwargs)
            used_symbol = sym
            is_nq = "NQ.c.0" == sym and "MNQ" not in sym
            print(f"  ✅ Success with {sym}")
            break
        except Exception as e:
            print(f"    ❌ {sym}: {e}")
            continue

    if data is None:
        print("  All symbol attempts failed.")
        # Try listing available symbols
        try:
            print("  Listing available symbols ...")
            syms = client.metadata.list_fields(dataset="GLBX.MDP3", schema="ohlcv-1m")
            print(f"    Fields: {syms}")
        except Exception:
            pass
        return None

    # Convert to DataFrame
    df = data.to_df()
    print(f"  Raw rows: {len(df)}")
    print(f"  Raw columns: {list(df.columns)}")

    # Convert to polars
    df_pl = pl.from_pandas(df.reset_index()) if hasattr(df, 'index') else pl.from_pandas(df)

    return df_pl, used_symbol, is_nq


def normalize_databento(df: pl.DataFrame, used_symbol: str, is_nq: bool) -> pl.DataFrame:
    """Convert Databento OHLCV to match existing data format."""
    # Map column names — Databento uses ts_event, open, high, low, close, volume
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ts_event" in cl or "timestamp" in cl:
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl == "volume":
            col_map[c] = "volume"

    if "timestamp" not in col_map.values():
        # Try first datetime column
        for c in df.columns:
            if "datetime" in str(df[c].dtype).lower() or "date" in c.lower():
                col_map[c] = "timestamp"
                break

    df = df.rename(col_map)

    # Keep only needed columns
    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    available = [c for c in needed if c in df.columns]
    df = df.select(available)

    # Add tick_count if missing
    if "tick_count" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("tick_count"))

    # Ensure correct types
    df = df.with_columns([
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    ])

    # Handle timestamp timezone — strip to naive UTC like existing data
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Datetime:
        if hasattr(ts_col.dtype, 'time_zone') and ts_col.dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    elif ts_col.dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())

    # Cast to same precision as existing data (μs)
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    # Sort and deduplicate
    df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")

    # Drop NaN rows
    df = df.drop_nulls(subset=["open", "high", "low", "close"])

    # Drop zero-price rows
    df = df.filter(pl.col("close") > 0)

    # If NQ instead of MNQ, note it but keep the data as-is
    # (price action is identical, just different multiplier)
    if is_nq:
        print("  ⚠️  Using NQ data (not MNQ). Prices are identical; P&L will be adjusted.")

    return df, is_nq


def add_et_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add the ET time columns needed by the strategies."""
    df = df.with_columns([
        pl.col("timestamp").dt.convert_time_zone("US/Eastern").alias("ts_et"),
    ])
    df = df.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("hour_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("minute_et"),
    ])
    df = df.with_columns([
        (pl.col("hour_et") * 100 + pl.col("minute_et")).alias("hhmm"),
    ])
    return df


# ═════════════════════════════════════════════════════════════════════
# PHASE B: BLIND TEST
# ═════════════════════════════════════════════════════════════════════

def analyze_layer(trades: list[TradeRecord], name: str, pnl_scale: float = 1.0) -> dict:
    """Compute metrics for a layer."""
    if not trades:
        return {"name": name, "trades": 0, "wr": 0, "total": 0, "monthly_avg": 0,
                "worst_month": 0, "monthly": {}, "monthly_trades": {}, "daily": {}}

    scaled_trades = []
    for t in trades:
        scaled_trades.append(t._replace(net_pnl=t.net_pnl * pnl_scale) if hasattr(t, '_replace')
                             else t)

    # Use original trades if _replace not available (dataclass)
    active = trades
    total = sum(t.net_pnl for t in active) * pnl_scale
    winners = sum(1 for t in active if t.net_pnl > 0)
    wr = winners / len(active) * 100 if active else 0

    monthly = defaultdict(float)
    monthly_tc = defaultdict(int)
    daily = defaultdict(float)
    for t in active:
        et = t.entry_time
        m = et.strftime("%Y-%m") if hasattr(et, 'strftime') else str(et)[:7]
        d = et.strftime("%Y-%m-%d") if hasattr(et, 'strftime') else str(et)[:10]
        monthly[m] += t.net_pnl * pnl_scale
        monthly_tc[m] += 1
        daily[d] += t.net_pnl * pnl_scale

    n_months = max(len(monthly), 1)
    worst_month = min(monthly.values()) if monthly else 0
    n_days = max(len(daily), 1)

    return {
        "name": name,
        "trades": len(active),
        "wr": wr,
        "total": total,
        "monthly_avg": total / n_months,
        "worst_month": worst_month,
        "monthly": dict(monthly),
        "monthly_trades": dict(monthly_tc),
        "daily": dict(daily),
        "n_months": n_months,
        "trades_per_day": len(active) / n_days,
    }


def run_blind_test(df: pl.DataFrame, is_nq: bool) -> list[dict]:
    """Run all v2 layers on the extended data with exact winning params."""
    # P&L scale: if data is NQ instead of MNQ, signals are same but
    # we want MNQ P&L for comparison, so no scaling needed — backtester
    # uses MNQ point_value ($2) already via run_backtest cost model.
    # If the data IS NQ prices, the price movements are identical to MNQ
    # (same index), so P&L is correct as computed.

    results = []

    # L1: VWAP Mean Reversion
    print("  Running L1 (VWAP Mean Reversion) ...")
    p = V2_L1
    sigs, exits = layer1_signals(df, z_threshold=p["z_threshold"], lookback=p["lookback"])
    trades = run_backtest(df, sigs, exits, stop_ticks=p["stop_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L1"],
                          layer_name="L1_VWAP_MR")
    r = analyze_layer(trades, "L1 VWAP MR")
    results.append(r)
    print(f"    {r['trades']} trades, {r['wr']:.1f}% WR, ${r['total']:,.0f} total, ${r['monthly_avg']:,.0f}/mo")
    gc.collect()

    # L2C: Afternoon Trend
    print("  Running L2C (Afternoon Trend) ...")
    p = V2_L2C
    sigs, exits = layer2c_signals(df, ema_fast=p["ema_fast"], ema_slow=p["ema_slow"])
    trades = run_backtest(df, sigs, exits, stop_ticks=p["trail_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L2C"],
                          trailing=True, trailing_ticks=p["trail_ticks"],
                          layer_name="L2C_Afternoon")
    r = analyze_layer(trades, "L2C Afternoon")
    results.append(r)
    print(f"    {r['trades']} trades, {r['wr']:.1f}% WR, ${r['total']:,.0f} total, ${r['monthly_avg']:,.0f}/mo")
    gc.collect()

    # L3: Volatility Momentum
    print("  Running L3 (Volatility Momentum) ...")
    p = V2_L3
    sigs, exits = layer3_signals(df, range_mult=p["range_mult"], ema_pullback=p["ema_pullback"])
    trades = run_backtest(df, sigs, exits, stop_ticks=p["trail_ticks"], target_ticks=None,
                          max_hold_bars=p["max_hold"], contracts=V2_SIZING["L3"],
                          trailing=True, trailing_ticks=p["trail_ticks"],
                          layer_name="L3_VolMom")
    r = analyze_layer(trades, "L3 VolMom")
    results.append(r)
    print(f"    {r['trades']} trades, {r['wr']:.1f}% WR, ${r['total']:,.0f} total, ${r['monthly_avg']:,.0f}/mo")
    gc.collect()

    # L4: Momentum Continuation
    print("  Running L4 (Momentum Continuation) ...")
    p = V2_MOMCONT
    sigs, exits = momentum_continuation_signals(df, z_threshold=p["z"], atr_mult=p["atr_m"])
    trades = run_backtest(df, sigs, exits, stop_ticks=p["trail"], target_ticks=None,
                          max_hold_bars=p["hold"], contracts=V2_SIZING["L1"],
                          trailing=True, trailing_ticks=p["trail"],
                          layer_name="L1_MomCont")
    r = analyze_layer(trades, "L4 MomCont")
    results.append(r)
    print(f"    {r['trades']} trades, {r['wr']:.1f}% WR, ${r['total']:,.0f} total, ${r['monthly_avg']:,.0f}/mo")
    gc.collect()

    return results


def combine_layer_results(layer_results: list[dict]) -> dict:
    """Combine all layer results into a single combined view."""
    monthly = defaultdict(float)
    monthly_tc = defaultdict(int)
    daily = defaultdict(float)
    total_trades = 0
    total_winners = 0

    for lr in layer_results:
        for m, pnl in lr["monthly"].items():
            monthly[m] += pnl
            monthly_tc[m] += lr["monthly_trades"].get(m, 0)
        for d, pnl in lr["daily"].items():
            daily[d] += pnl
        total_trades += lr["trades"]
        total_winners += int(lr["wr"] * lr["trades"] / 100)

    total_pnl = sum(monthly.values())
    n_months = max(len(monthly), 1)
    n_days = max(len(daily), 1)
    worst_month = min(monthly.values()) if monthly else 0
    worst_day = min(daily.values()) if daily else 0

    # Max DD
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "total_pnl": total_pnl,
        "monthly_avg": total_pnl / n_months,
        "monthly": dict(monthly),
        "monthly_trades": dict(monthly_tc),
        "total_trades": total_trades,
        "wr": total_winners / total_trades * 100 if total_trades else 0,
        "worst_month": worst_month,
        "worst_day": worst_day,
        "max_dd": max_dd,
        "n_months": n_months,
        "months_profitable": sum(1 for v in monthly.values() if v > 0),
        "months_gte_7k": sum(1 for v in monthly.values() if v >= 7000),
        "trades_per_day": total_trades / n_days,
    }


def print_results(layer_results: list[dict], combined: dict, data_info: dict):
    """Print Phase 2C results."""
    print(f"\n{'═' * 60}")
    print(f"  DATABENTO BLIND TEST — Always-On v2 on Unseen Data")
    print(f"{'═' * 60}")

    print(f"\n  DATA DOWNLOADED:")
    print(f"    Symbol: {data_info.get('symbol', 'N/A')}")
    print(f"    Date range: {data_info.get('start', '?')} to {data_info.get('end', '?')}")
    print(f"    Total bars: {data_info.get('bars', 0):,}")
    print(f"    Months of data: {data_info.get('months', 0)}")
    if data_info.get("is_nq"):
        print(f"    ⚠️  NQ data (same price action as MNQ, P&L at MNQ scale)")

    print(f"\n  PER-LAYER BLIND RESULTS:")
    print(f"  {'─' * 56}")
    print(f"  {'Layer':<12} {'Trades':>7} {'WR':>7} {'Total':>10} {'Avg/Mo':>10} {'Worst Mo':>10}")
    print(f"  {'─' * 56}")
    for lr in layer_results:
        print(f"  {lr['name']:<12} {lr['trades']:>7} {lr['wr']:>6.1f}% ${lr['total']:>9,.0f} ${lr['monthly_avg']:>9,.0f} ${lr['worst_month']:>9,.0f}")
    print(f"  {'─' * 56}")
    c = combined
    print(f"  {'COMBINED':<12} {c['total_trades']:>7} {c['wr']:>6.1f}% ${c['total_pnl']:>9,.0f} ${c['monthly_avg']:>9,.0f} ${c['worst_month']:>9,.0f}")
    print(f"  {'─' * 56}")

    print(f"\n  MONTHLY BREAKDOWN:")
    for m in sorted(c["monthly"].keys()):
        pnl = c["monthly"][m]
        tc = c["monthly_trades"].get(m, 0)
        flag = "✅" if pnl > 0 else "❌"
        print(f"    {m}: ${pnl:>+10,.0f} ({tc:>3} trades) {flag}")

    print(f"\n  Months profitable: {c['months_profitable']}/{c['n_months']}")
    print(f"  Months above $7K: {c['months_gte_7k']}/{c['n_months']}")

    # Load v2 combined for comparison
    v2 = json.loads(Path("reports/always_on_v2.json").read_text())["combined"]

    print(f"\n  COMPARISON:")
    print(f"  {'─' * 56}")
    print(f"  {'Metric':<18} {'Y1 (optimized)':>14} {'Y2 (OOS)':>14} {'Databento':>14}")
    print(f"  {'─' * 56}")
    print(f"  {'Avg monthly':<18} ${v2['yr1_monthly']:>13,.0f} ${v2['yr2_monthly']:>13,.0f} ${c['monthly_avg']:>13,.0f}")
    y1_worst = min(v for k, v in v2["monthly"].items() if k < "2025-03")
    y2_worst = min(v for k, v in v2["monthly"].items() if k >= "2025-03")
    print(f"  {'Worst month':<18} ${y1_worst:>13,.0f} ${y2_worst:>13,.0f} ${c['worst_month']:>13,.0f}")
    y1_prof = sum(1 for k, v in v2["monthly"].items() if k < "2025-03" and v > 0)
    y2_prof = sum(1 for k, v in v2["monthly"].items() if k >= "2025-03" and v > 0)
    print(f"  {'Months profit.':<18} {'%d/12' % y1_prof:>14} {'%d/13' % y2_prof:>14} {'%d/%d' % (c['months_profitable'], c['n_months']):>14}")
    print(f"  {'Trades/day':<18} {'~7.5':>14} {'~7.5':>14} {c['trades_per_day']:>14.1f}")
    print(f"  {'Win rate':<18} {'~13%':>14} {'~13%':>14} {c['wr']:>13.1f}%")
    print(f"  {'─' * 56}")

    # Verdict
    blind_profitable = c["monthly_avg"] > 0
    blind_worst_ok = c["worst_month"] > -4500
    blind_months_ratio = c["months_profitable"] / max(c["n_months"], 1)

    if blind_profitable and blind_months_ratio >= 0.5 and abs(c["monthly_avg"]) > 500:
        edge_type = "STRUCTURAL"
    elif blind_profitable and blind_months_ratio >= 0.3:
        edge_type = "REGIME-SPECIFIC"
    else:
        edge_type = "INCONCLUSIVE"

    all_months = 25 + c["n_months"]  # Y1+Y2 + Databento
    all_total = v2["total_pnl"] + c["total_pnl"]
    all_monthly = all_total / all_months

    print(f"\n  VERDICT:")
    status = "PASSED" if blind_profitable else "FAILED"
    print(f"    The Always-On v2 system {status} on {c['n_months']} months of unseen data.")
    print(f"    It made ${c['monthly_avg']:,.0f}/month on data from {data_info.get('start','?')} "
          f"to {data_info.get('end','?')}.")
    print(f"    Combined across all {all_months} months of history: ${all_monthly:,.0f}/month.")
    print(f"    Edge is {edge_type}.")

    return {
        "status": status,
        "edge_type": edge_type,
        "all_months": all_months,
        "all_monthly": all_monthly,
    }


def main():
    t0 = _time.time()
    print("═" * 60)
    print("  DATABENTO BLIND TEST — Always-On v2")
    print("═" * 60)

    # Phase A: Download
    print("\n  PHASE A: Download extended data ...")
    check_existing_format()

    data_info = {"symbol": None, "start": None, "end": None, "bars": 0, "months": 0, "is_nq": False}

    if EXTENDED_PATH.exists():
        print(f"\n  Extended data already exists at {EXTENDED_PATH}, loading ...")
        df_ext = pl.read_parquet(EXTENDED_PATH)
        data_info["bars"] = len(df_ext)
        data_info["start"] = str(df_ext["timestamp"].min())[:10]
        data_info["end"] = str(df_ext["timestamp"].max())[:10]
        data_info["symbol"] = "cached"
    else:
        result = download_databento()
        if result is None:
            print("\n  ❌ Failed to download Databento data. Skipping blind test.")
            # Save partial report
            out = REPORTS_DIR / "always_on_databento_v1.json"
            json.dump({"error": "download_failed", "timestamp": datetime.now().isoformat()}, open(out, "w"), indent=2)
            return

        df_raw, used_symbol, is_nq = result
        data_info["symbol"] = used_symbol
        data_info["is_nq"] = is_nq

        print(f"\n  Normalizing Databento data ...")
        df_ext, is_nq = normalize_databento(df_raw, used_symbol, is_nq)

        # Save
        df_ext.write_parquet(EXTENDED_PATH)
        print(f"  Saved to {EXTENDED_PATH}")

        data_info["bars"] = len(df_ext)
        data_info["start"] = str(df_ext["timestamp"].min())[:10]
        data_info["end"] = str(df_ext["timestamp"].max())[:10]

    print(f"  Extended data: {data_info['bars']:,} bars, {data_info['start']} → {data_info['end']}")

    # Add ET columns
    df_ext = pl.read_parquet(EXTENDED_PATH)

    # Ensure timestamp has no timezone for consistency
    if hasattr(df_ext["timestamp"].dtype, 'time_zone') and df_ext["timestamp"].dtype.time_zone:
        df_ext = df_ext.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Need to localize to UTC first for ET conversion
    df_ext = df_ext.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp_utc")
    )
    df_ext = df_ext.with_columns(
        pl.col("timestamp_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
    )
    df_ext = df_ext.drop("timestamp_utc")
    df_ext = df_ext.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("hour_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("minute_et"),
    ])
    df_ext = df_ext.with_columns([
        (pl.col("hour_et") * 100 + pl.col("minute_et")).alias("hhmm"),
    ])

    # Add IB
    df_ext = add_initial_balance(df_ext)

    # Count months
    dates = df_ext.select(pl.col("date_et").cast(pl.Utf8).str.slice(0, 7).alias("m")).unique()
    data_info["months"] = dates.height

    print(f"  {data_info['months']} months of data ready")

    # Phase B: Blind test
    print(f"\n  PHASE B: Running blind test ...")
    layer_results = run_blind_test(df_ext, data_info.get("is_nq", False))

    # Combine
    combined = combine_layer_results(layer_results)

    # Phase C: Results
    verdict = print_results(layer_results, combined, data_info)

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_info": {k: str(v) for k, v in data_info.items()},
        "layers": [
            {k: v for k, v in lr.items() if k not in ("daily",)}
            for lr in layer_results
        ],
        "combined": {k: v for k, v in combined.items() if k != "monthly_trades"},
        "verdict": verdict,
    }
    out = REPORTS_DIR / "always_on_databento_v1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
