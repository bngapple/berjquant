#!/usr/bin/env python3
"""
Pre-Market Volatility Filter — Should we trade today?

Uses London session and US pre-market (3:00-9:30 AM ET) to predict
whether RTH will have enough volatility for the system to be profitable.

Data: 8yr + extended datasets have 24h bars. 2yr only has 8AM+.
Coverage: Dec 2017 – Mar 2024 for pre-market (6 years).

Usage:
    python3 run_htf_swing_v3_premarket_filter.py
"""

import gc
import json
import time as _time
from collections import defaultdict
from datetime import datetime, date as date_cls
from pathlib import Path

import numpy as np
import polars as pl

from run_htf_swing import (
    extract_arrays, backtest,
    sig_rsi_extreme, sig_ib_breakout, sig_momentum_bar,
    TICK_SIZE, POINT_VALUE,
)
from run_htf_swing_8yr import load_1m, resample_15m_rth

SEED = 42
np.random.seed(SEED)
REPORTS_DIR = Path("reports")
FLATTEN_TIME = 1645
CONTRACTS = 3

HYBRID_V2 = {
    "RSI": {"period": 5, "ob": 65, "os": 35, "sl_pts": 10, "tp_pts": 100, "hold": 5},
    "IB":  {"ib_filter": True, "sl_pts": 10, "tp_pts": 120, "hold": 15},
    "MOM": {"atr_mult": 1.0, "vol_mult": 1.0, "sl_pts": 15, "tp_pts": 100, "hold": 5},
}


def pts_to_ticks(pts):
    return int(pts / TICK_SIZE)


def run_system(df, params):
    o, h, l, c, ts, hm = extract_arrays(df)
    trades = []
    p = params["RSI"]
    sigs = sig_rsi_extreme(df, p["period"], p["ob"], p["os"])
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "RSI", FLATTEN_TIME))
    p = params["IB"]
    sigs = sig_ib_breakout(df, p["ib_filter"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "IB", FLATTEN_TIME))
    p = params["MOM"]
    sigs = sig_momentum_bar(df, p["atr_mult"], p["vol_mult"])[0]
    trades.extend(backtest(o, h, l, c, ts, hm, sigs,
                           pts_to_ticks(p["sl_pts"]), pts_to_ticks(p["tp_pts"]),
                           p["hold"], CONTRACTS, "MOM", FLATTEN_TIME))
    return trades


def main():
    t0 = _time.time()
    print("═" * 80)
    print("  PRE-MARKET VOLATILITY FILTER — Should We Trade Today?")
    print("═" * 80)

    # ── Load 1m data with full 24h hours ────────────────────────────
    print("\n  Loading 24-hour 1m data...")
    df1 = load_1m("data/processed/MNQ/1m/databento_8yr_ext.parquet", "2018-2021")
    df2 = load_1m("data/processed/MNQ/1m/databento_extended.parquet", "2022-2024")
    df3 = load_1m("data/processed/MNQ/1m/full_2yr.parquet", "2024-2026")
    combined = pl.concat([df1, df2, df3]).sort("timestamp").unique(subset=["timestamp"], keep="first")
    combined = combined.filter(pl.col("close") > 0)
    del df1, df2, df3; gc.collect()

    # Add ET time columns
    if hasattr(combined["timestamp"].dtype, 'time_zone') and combined["timestamp"].dtype.time_zone:
        combined = combined.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    combined = combined.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
    )
    combined = combined.with_columns(
        pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
    )
    combined = combined.drop("_utc")
    combined = combined.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    combined = combined.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])

    print(f"  Total 1m bars: {len(combined):,}")

    # ── Resample RTH to 15m and run backtest ────────────────────────
    rth = combined.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))
    df_15m = (
        rth.group_by_dynamic("ts_et", every="15m")
        .agg([
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
            pl.col("date_et").last(), pl.col("hhmm").last(),
        ])
        .filter(pl.col("open").is_not_null())
        .sort("ts_et")
        .rename({"ts_et": "timestamp"})
    )
    print(f"  15m RTH bars: {len(df_15m):,}")

    trades = run_system(df_15m, HYBRID_V2)
    print(f"  Total trades: {len(trades):,}")

    # Daily P&L from trades
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[str(t.entry_time)[:10]] += t.net_pnl
    del rth; gc.collect()

    # ── Step 1: Compute pre-market indicators per day ───────────────
    print(f"\n{'━' * 80}")
    print("  STEP 1: Computing pre-market indicators...")
    print("━" * 80)

    # Get all 1m data grouped by date
    dates = combined["date_et"].to_list()
    hhmm_arr = combined["hhmm"].to_numpy()
    highs = combined["high"].to_numpy()
    lows = combined["low"].to_numpy()
    closes = combined["close"].to_numpy()
    volumes = combined["volume"].to_numpy().astype(float)

    # Build daily pre-market features
    premarket = {}
    prev_rth_close = {}

    # First pass: get RTH closes for gap calculation
    for i in range(len(combined)):
        d = dates[i]
        h = hhmm_arr[i]
        if 930 <= h < 1600:
            prev_rth_close[d] = closes[i]  # last RTH close of the day

    # Second pass: compute pre-market indicators
    london = defaultdict(lambda: {"high": -np.inf, "low": np.inf, "bar_ranges": [], "volume": 0})
    premarket_session = defaultdict(lambda: {"high": -np.inf, "low": np.inf, "volume": 0})
    full_pre = defaultdict(lambda: {"high": -np.inf, "low": np.inf, "volume": 0, "last_close": None})
    rth_daily = defaultdict(lambda: {"high": -np.inf, "low": np.inf})

    for i in range(len(combined)):
        d = dates[i]
        h = hhmm_arr[i]

        # London session: 3:00-8:00 AM ET (hhmm 300-800)
        if 300 <= h < 800:
            london[d]["high"] = max(london[d]["high"], highs[i])
            london[d]["low"] = min(london[d]["low"], lows[i])
            london[d]["bar_ranges"].append(highs[i] - lows[i])
            london[d]["volume"] += volumes[i]

        # Pre-market: 8:00-9:30 AM ET
        if 800 <= h < 930:
            premarket_session[d]["high"] = max(premarket_session[d]["high"], highs[i])
            premarket_session[d]["low"] = min(premarket_session[d]["low"], lows[i])
            premarket_session[d]["volume"] += volumes[i]

        # Full pre-RTH: 3:00-9:30 AM ET
        if 300 <= h < 930:
            full_pre[d]["high"] = max(full_pre[d]["high"], highs[i])
            full_pre[d]["low"] = min(full_pre[d]["low"], lows[i])
            full_pre[d]["volume"] += volumes[i]
            full_pre[d]["last_close"] = closes[i]

        # RTH daily range
        if 930 <= h < 1600:
            rth_daily[d]["high"] = max(rth_daily[d]["high"], highs[i])
            rth_daily[d]["low"] = min(rth_daily[d]["low"], lows[i])

    # Build feature matrix
    sorted_dates = sorted(set(d for d in daily_pnl.keys()))
    features = {}
    prev_date = None

    for d_str in sorted_dates:
        d = date_cls.fromisoformat(d_str)

        london_range = london[d]["high"] - london[d]["low"] if london[d]["high"] > -np.inf else np.nan
        premarket_range = premarket_session[d]["high"] - premarket_session[d]["low"] if premarket_session[d]["high"] > -np.inf else np.nan
        full_pre_range = full_pre[d]["high"] - full_pre[d]["low"] if full_pre[d]["high"] > -np.inf else np.nan
        pre_volume = full_pre[d]["volume"]
        london_atr_proxy = np.mean(london[d]["bar_ranges"]) if london[d]["bar_ranges"] else np.nan
        rth_range = rth_daily[d]["high"] - rth_daily[d]["low"] if rth_daily[d]["high"] > -np.inf else np.nan

        # Gap: previous day RTH close vs today's pre-market last price
        gap = np.nan
        if prev_date is not None and prev_date in prev_rth_close and full_pre[d]["last_close"] is not None:
            gap = abs(full_pre[d]["last_close"] - prev_rth_close[prev_date])

        features[d_str] = {
            "london_range": london_range,
            "premarket_range": premarket_range,
            "full_pre_range": full_pre_range,
            "pre_volume": pre_volume,
            "london_atr_proxy": london_atr_proxy,
            "gap_size": gap,
            "rth_range": rth_range,
            "daily_pnl": daily_pnl.get(d_str, 0),
        }
        prev_date = d

    # Filter to days with valid pre-market data
    valid_days = [d for d, f in features.items() if not np.isnan(f["london_range"]) and f["london_range"] > 0]
    print(f"  Days with pre-market data: {len(valid_days)} / {len(sorted_dates)}")

    # Days without pre-market (2024-2026 from full_2yr)
    no_premarket = [d for d in sorted_dates if d not in valid_days or np.isnan(features[d]["london_range"])]
    if no_premarket:
        print(f"  Days WITHOUT pre-market data: {len(no_premarket)} ({no_premarket[0]} to {no_premarket[-1]})")
        print(f"  Note: full_2yr.parquet only has 8AM+ data, no London session")

    del combined; gc.collect()

    # ── Step 2: Correlations ────────────────────────────────────────
    print(f"\n{'━' * 80}")
    print("  STEP 2: PRE-MARKET → RTH CORRELATION")
    print("━" * 80)

    indicators = ["london_range", "premarket_range", "full_pre_range",
                   "pre_volume", "london_atr_proxy", "gap_size"]
    indicator_labels = {
        "london_range": "London range",
        "premarket_range": "Pre-market range",
        "full_pre_range": "Full pre-RTH range",
        "pre_volume": "Pre-market volume",
        "london_atr_proxy": "London ATR proxy",
        "gap_size": "Gap size",
    }

    print(f"\n  {'Indicator':<22} {'Corr w/ P&L':>12} {'Corr w/ range':>14} {'N':>6}")
    print(f"  {'─'*22} {'─'*12} {'─'*14} {'─'*6}")

    correlations = {}
    for ind in indicators:
        vals = []
        pnls = []
        ranges = []
        for d in valid_days:
            f = features[d]
            v = f[ind]
            if np.isnan(v):
                continue
            vals.append(v)
            pnls.append(f["daily_pnl"])
            rr = f["rth_range"]
            ranges.append(rr if not np.isnan(rr) else 0)

        if len(vals) < 30:
            correlations[ind] = {"pnl_corr": 0, "range_corr": 0, "n": len(vals)}
            print(f"  {indicator_labels[ind]:<22} {'N/A':>12} {'N/A':>14} {len(vals):>6}")
            continue

        pnl_corr = float(np.corrcoef(vals, pnls)[0, 1])
        range_corr = float(np.corrcoef(vals, ranges)[0, 1])
        correlations[ind] = {"pnl_corr": pnl_corr, "range_corr": range_corr, "n": len(vals)}
        print(f"  {indicator_labels[ind]:<22} {pnl_corr:>+12.3f} {range_corr:>+14.3f} {len(vals):>6}")

    # Best indicator by P&L correlation
    best_ind = max(indicators, key=lambda x: abs(correlations[x]["pnl_corr"]))
    best_corr = correlations[best_ind]["pnl_corr"]
    print(f"\n  Best predictor: {indicator_labels[best_ind]} (r={best_corr:+.3f})")

    # Also find best for range prediction (more useful)
    best_range_ind = max(indicators, key=lambda x: abs(correlations[x]["range_corr"]))
    best_range_corr = correlations[best_range_ind]["range_corr"]
    print(f"  Best range predictor: {indicator_labels[best_range_ind]} (r={best_range_corr:+.3f})")

    # Use the one with highest range correlation (more stable than P&L correlation)
    filter_ind = best_range_ind
    print(f"\n  Using {indicator_labels[filter_ind]} for filter (range correlation is more stable)")

    # ── Step 3: Filter thresholds ───────────────────────────────────
    print(f"\n{'━' * 80}")
    print(f"  STEP 3: FILTER PERFORMANCE ({indicator_labels[filter_ind]})")
    print("━" * 80)

    # Get indicator values for valid days
    ind_vals = []
    for d in valid_days:
        v = features[d][filter_ind]
        if not np.isnan(v):
            ind_vals.append(v)
    ind_vals = np.array(ind_vals)

    percentiles = [10, 20, 25, 30, 40, 50]
    thresholds = {p: float(np.percentile(ind_vals, p)) for p in percentiles}

    print(f"\n  Threshold values:")
    for p, v in thresholds.items():
        print(f"    P{p}: {v:.1f}")

    # For each threshold, compute filtered performance
    # Baseline: all days
    baseline_total = sum(features[d]["daily_pnl"] for d in valid_days)
    baseline_months = defaultdict(float)
    for d in valid_days:
        baseline_months[d[:7]] += features[d]["daily_pnl"]
    baseline_monthly = baseline_total / max(len(baseline_months), 1)

    baseline_dv = [features[d]["daily_pnl"] for d in valid_days]
    baseline_sharpe = float(np.mean(baseline_dv) / np.std(baseline_dv) * np.sqrt(252)) if np.std(baseline_dv) > 0 else 0

    print(f"\n  {'Threshold':<12} {'Days Traded':>12} {'Skipped':>8} {'Monthly $':>10} {'vs Base':>10} {'Sharpe':>7}")
    print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*7}")
    print(f"  {'None':<12} {len(valid_days):>12} {0:>8} ${baseline_monthly:>+9,.0f} {'--':>10} {baseline_sharpe:>6.1f}")

    filter_results = {}
    for p in percentiles:
        thr = thresholds[p]
        traded_days = [d for d in valid_days if not np.isnan(features[d][filter_ind]) and features[d][filter_ind] >= thr]
        skipped_days = [d for d in valid_days if np.isnan(features[d][filter_ind]) or features[d][filter_ind] < thr]

        traded_pnl = sum(features[d]["daily_pnl"] for d in traded_days)
        traded_months = defaultdict(float)
        for d in traded_days:
            traded_months[d[:7]] += features[d]["daily_pnl"]
        traded_monthly = traded_pnl / max(len(traded_months), 1)
        delta = traded_monthly - baseline_monthly

        dv = [features[d]["daily_pnl"] for d in traded_days]
        sharpe = float(np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0

        filter_results[p] = {
            "threshold": thr, "days_traded": len(traded_days), "days_skipped": len(skipped_days),
            "monthly_pnl": traded_monthly, "delta": delta, "sharpe": sharpe,
            "traded_days": traded_days, "skipped_days": skipped_days,
        }
        print(f"  {'P'+str(p):<12} {len(traded_days):>12} {len(skipped_days):>8} "
              f"${traded_monthly:>+9,.0f} ${delta:>+9,.0f} {sharpe:>6.1f}")

    # ── Step 4: Skipped day analysis ────────────────────────────────
    print(f"\n{'━' * 80}")
    print("  STEP 4: SKIPPED DAY ANALYSIS")
    print("━" * 80)

    for p in [20, 25, 30]:
        fr = filter_results[p]
        skipped = fr["skipped_days"]
        if not skipped:
            continue

        skipped_pnls = [features[d]["daily_pnl"] for d in skipped]
        profitable_skipped = [p for p in skipped_pnls if p > 0]
        unprofitable_skipped = [p for p in skipped_pnls if p <= 0]

        n_prof = len(profitable_skipped)
        n_unprof = len(unprofitable_skipped)
        avg_prof = np.mean(profitable_skipped) if profitable_skipped else 0
        avg_unprof = np.mean(unprofitable_skipped) if unprofitable_skipped else 0

        # Monthly value of filter
        saved = sum(abs(p) for p in unprofitable_skipped)  # avoided losses
        lost = sum(p for p in profitable_skipped)  # missed gains
        n_months_skip = len(set(d[:7] for d in skipped))
        n_months_total = len(set(d[:7] for d in valid_days))
        saved_mo = saved / max(n_months_total, 1)
        lost_mo = lost / max(n_months_total, 1)
        net_mo = (saved_mo - lost_mo)
        # Net is actually: total traded monthly - baseline monthly
        net_mo_actual = fr["delta"]

        print(f"\n  Filter P{p} (threshold = {fr['threshold']:.0f}):")
        print(f"    Total days skipped:            {len(skipped)}")
        print(f"    Skipped profitable days:       {n_prof} ({n_prof/len(skipped)*100:.0f}%) ← missed opportunities")
        print(f"    Skipped unprofitable days:     {n_unprof} ({n_unprof/len(skipped)*100:.0f}%) ← correctly avoided")
        print(f"    Avg P&L of skipped good days:  ${avg_prof:>+,.0f}")
        print(f"    Avg P&L of skipped bad days:   ${avg_unprof:>+,.0f}")
        print(f"    Net monthly impact:            ${net_mo_actual:>+,.0f}/month")

    # ── Step 5: Composite filter ────────────────────────────────────
    print(f"\n{'━' * 80}")
    print("  STEP 5: COMPOSITE FILTER")
    print("━" * 80)

    # Use top 3 by range correlation
    top3 = sorted(indicators, key=lambda x: abs(correlations[x]["range_corr"]), reverse=True)[:3]
    print(f"  Top 3 indicators: {', '.join(indicator_labels[i] for i in top3)}")

    # Normalize each indicator to 0-1 range and combine
    norm_data = {}
    for ind in top3:
        vals = [features[d][ind] for d in valid_days if not np.isnan(features[d][ind])]
        if not vals:
            continue
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1
        norm_data[ind] = {"min": mn, "range": rng}

    # Compute composite score for each day
    for d in valid_days:
        score = 0
        n_valid = 0
        for ind in top3:
            v = features[d][ind]
            if np.isnan(v) or ind not in norm_data:
                continue
            normalized = (v - norm_data[ind]["min"]) / norm_data[ind]["range"]
            score += normalized
            n_valid += 1
        features[d]["composite"] = score / n_valid if n_valid > 0 else np.nan

    # Test composite thresholds
    comp_vals = [features[d]["composite"] for d in valid_days if not np.isnan(features[d]["composite"])]
    comp_vals = np.array(comp_vals)

    print(f"\n  {'Threshold':<12} {'Days Traded':>12} {'Skipped':>8} {'Monthly $':>10} {'vs Base':>10} {'Sharpe':>7}")
    print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*7}")
    print(f"  {'None':<12} {len(valid_days):>12} {0:>8} ${baseline_monthly:>+9,.0f} {'--':>10} {baseline_sharpe:>6.1f}")

    composite_results = {}
    for p in percentiles:
        thr = float(np.percentile(comp_vals, p))
        traded = [d for d in valid_days if not np.isnan(features[d]["composite"]) and features[d]["composite"] >= thr]
        skipped = [d for d in valid_days if np.isnan(features[d]["composite"]) or features[d]["composite"] < thr]

        traded_pnl = sum(features[d]["daily_pnl"] for d in traded)
        traded_months = defaultdict(float)
        for d in traded:
            traded_months[d[:7]] += features[d]["daily_pnl"]
        traded_monthly = traded_pnl / max(len(traded_months), 1)
        delta = traded_monthly - baseline_monthly

        dv = [features[d]["daily_pnl"] for d in traded]
        sharpe = float(np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0

        composite_results[p] = {"monthly_pnl": traded_monthly, "delta": delta, "sharpe": sharpe,
                                "days_traded": len(traded), "days_skipped": len(skipped)}
        print(f"  {'CP'+str(p):<12} {len(traded):>12} {len(skipped):>8} "
              f"${traded_monthly:>+9,.0f} ${delta:>+9,.0f} {sharpe:>6.1f}")

    # ── Step 6: Year-by-year impact ─────────────────────────────────
    print(f"\n{'━' * 80}")
    print("  STEP 6: ANNUAL IMPACT")
    print("━" * 80)

    # Pick the best single filter (highest delta on valid data)
    best_p = max(filter_results.keys(), key=lambda p: filter_results[p]["delta"])
    best_fr = filter_results[best_p]

    # Also check if composite is better
    best_comp_p = max(composite_results.keys(), key=lambda p: composite_results[p]["delta"])
    if composite_results[best_comp_p]["delta"] > best_fr["delta"]:
        print(f"\n  Best filter: Composite P{best_comp_p} (${composite_results[best_comp_p]['delta']:+,.0f}/mo)")
        use_composite = True
    else:
        print(f"\n  Best filter: {indicator_labels[filter_ind]} P{best_p} (${best_fr['delta']:+,.0f}/mo)")
        use_composite = False

    # Year-by-year with best single filter
    print(f"\n  Using {indicator_labels[filter_ind]} P{best_p}:")
    thr = thresholds[best_p]
    traded_set = set(best_fr["traded_days"])

    years = sorted(set(d[:4] for d in valid_days))
    print(f"\n  {'Year':<6} {'No Filter':>10} {'Filtered':>10} {'Delta':>10} {'Skipped':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    annual_report = {}
    for y in years:
        y_days = [d for d in valid_days if d[:4] == y]
        y_traded = [d for d in y_days if d in traded_set]
        y_skipped = len(y_days) - len(y_traded)

        no_filter_pnl = sum(features[d]["daily_pnl"] for d in y_days)
        filtered_pnl = sum(features[d]["daily_pnl"] for d in y_traded)
        delta = filtered_pnl - no_filter_pnl

        annual_report[y] = {"no_filter": no_filter_pnl, "filtered": filtered_pnl, "delta": delta, "skipped": y_skipped}
        print(f"  {y:<6} ${no_filter_pnl:>+9,.0f} ${filtered_pnl:>+9,.0f} ${delta:>+9,.0f} {y_skipped:>8}")

    # ═════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  VERDICT")
    print("═" * 80)

    best_delta = best_fr["delta"]
    best_sharpe_change = best_fr["sharpe"] - baseline_sharpe

    # Check consistency: does filter help in BOTH early (2018-2021) and late (2022-2024) periods?
    early_years = [y for y in years if int(y) <= 2021]
    late_years = [y for y in years if int(y) >= 2022]
    early_delta = sum(annual_report[y]["delta"] for y in early_years if y in annual_report)
    late_delta = sum(annual_report[y]["delta"] for y in late_years if y in annual_report)
    consistent = early_delta > 0 and late_delta >= 0

    print(f"\n  Best filter: {indicator_labels[filter_ind]} at P{best_p} threshold ({thresholds[best_p]:.0f} pts)")
    print(f"  Monthly impact: ${best_delta:>+,.0f}/month")
    print(f"  Sharpe change: {best_sharpe_change:>+.1f}")
    print(f"  Days skipped: {best_fr['days_skipped']} ({best_fr['days_skipped']/len(valid_days)*100:.0f}%)")
    print(f"\n  Consistency check:")
    print(f"    Early years (2018-2021) impact: ${early_delta:>+,.0f}")
    print(f"    Late years (2022-2024) impact:  ${late_delta:>+,.0f}")
    print(f"    Consistent: {'YES' if consistent else 'NO'}")

    if best_delta > 500 and consistent:
        print(f"\n  RECOMMENDED: Use {indicator_labels[filter_ind]} >= {thresholds[best_p]:.0f} as a daily go/no-go filter.")
        print(f"  Check at 9:29 AM ET. Skip the day if below threshold.")
    elif best_delta > 200:
        print(f"\n  MARGINAL: Filter adds ${best_delta:+,.0f}/mo but introduces optimization risk.")
        print(f"  Consider using as a position-sizing signal (half size on low days) instead of binary skip.")
    else:
        print(f"\n  NOT RECOMMENDED: Pre-market data doesn't reliably predict RTH profitability.")
        print(f"  The correlation is too weak to justify the added complexity.")
        print(f"  Trade every day and accept the low-vol drag as cost of doing business.")

    # Note about 2024-2026 gap
    if no_premarket:
        print(f"\n  NOTE: 2024-2026 data lacks pre-market bars (full_2yr.parquet starts at 8AM).")
        print(f"  This filter was tested on 2018-2024 data only. Need 24h data feed for live use.")

    # ── Save ────────────────────────────────────────────────────────
    report = {
        "timestamp": str(datetime.now()),
        "data_coverage": f"{valid_days[0]} to {valid_days[-1]}" if valid_days else "none",
        "days_with_premarket": len(valid_days),
        "days_without_premarket": len(no_premarket),
        "correlations": {indicator_labels[k]: v for k, v in correlations.items()},
        "best_indicator": indicator_labels[filter_ind],
        "filter_results": {
            f"P{p}": {k: v for k, v in r.items() if k not in ("traded_days", "skipped_days")}
            for p, r in filter_results.items()
        },
        "composite_results": composite_results,
        "annual_impact": annual_report,
        "best_filter": {
            "indicator": indicator_labels[filter_ind],
            "percentile": best_p,
            "threshold": thresholds[best_p],
            "monthly_delta": best_delta,
            "consistent": consistent,
        },
    }
    out = REPORTS_DIR / "premarket_filter.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {out}")
    print(f"  Time: {(_time.time() - t0) / 60:.1f} minutes")
    print("═" * 80)


if __name__ == "__main__":
    main()
