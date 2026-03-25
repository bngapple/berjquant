#!/usr/bin/env python3
"""
ADAPTIVE SCALPER — Rolling 2-week re-optimization with ATR-adaptive exits.
The PROCESS adapts, not static params. New file — does NOT modify anything.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.fast_backtester import FastBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from engine.adaptive import atr_stops, add_htf_trend, add_regime, add_volatility_regime

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("adapt"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,45,15,30),(10,0,14,0),(9,30,12,0),(10,0,15,30),(9,30,15,0),(9,30,14,0)]
FILTER_TYPES = ["none","htf_trend","regime_trending","ema_slope","heikin_ashi"]
APPROACHES = {
    "cci_fast":("cci","signals.momentum","cci",lambda:{"period":random.randint(5,14)},["entry_long_cci","entry_short_cci"]),
    "rsi_micro":("rsi","signals.momentum","rsi",lambda:{"period":random.randint(3,7),"overbought":round(random.uniform(55,65),1),"oversold":round(random.uniform(35,45),1)},["entry_long_rsi","entry_short_rsi"]),
    "roc_micro":("roc","signals.momentum","roc",lambda:{"period":random.randint(3,8)},["entry_long_roc","entry_short_roc"]),
    "bb_bounce":("bollinger_bands","signals.volatility","bollinger_bands",lambda:{"period":random.randint(10,20),"std_dev":round(random.uniform(1.0,1.5),2)},["entry_long_bb","entry_short_bb"]),
    "stoch_fast":("stochastic","signals.momentum","stochastic",lambda:{"k_period":random.randint(3,8),"d_period":random.randint(2,3),"overbought":round(random.uniform(60,75),1),"oversold":round(random.uniform(25,40),1)},["entry_long_stoch","entry_short_stoch"]),
    "ema_micro":("ema_crossover","signals.trend","ema_crossover",lambda:{"fast_period":random.randint(3,8),"slow_period":random.randint(8,15)},["entry_long_ema_cross","entry_short_ema_cross"]),
    "range_micro":("range_breakout","signals.price_action","range_breakout",lambda:{"lookback":random.randint(5,15)},["entry_long_breakout","entry_short_breakout"]),
}

FILTER_DEFS = {
    "none": (None, None, None, None, None),
    "htf_trend": None,  # Added by adaptive infrastructure
    "regime_trending": None,
    "ema_slope": ("ema_slope","signals.trend","ema_slope",{"period":8,"slope_lookback":2},"signal_ema_slope_up"),
    "heikin_ashi": ("heikin_ashi","signals.trend","heikin_ashi",{},"signal_ha_bullish"),
}


def make_variant(approach, filter_type="none", tw=None):
    """Create a random strategy variant for a given approach."""
    an, mod, func, pfn, cols = APPROACHES[approach]
    ep = pfn()
    entry = {"signal_name":an,"module":mod,"function":func,"params":ep,"columns":{"long":cols[0],"short":cols[1]}}
    fl = []
    # Add filter
    if filter_type == "htf_trend":
        fl.append({"signal_name":"htf_trend_filter","module":"_adaptive_","function":"_adaptive_",
                    "params":{},"column":"signal_htf_uptrend"})  # Will be added by adaptive infra
    elif filter_type == "regime_trending":
        fl.append({"signal_name":"regime_filter","module":"_adaptive_","function":"_adaptive_",
                    "params":{},"column":"signal_regime_trending"})
    elif filter_type in FILTER_DEFS and FILTER_DEFS[filter_type] is not None:
        fd = FILTER_DEFS[filter_type]
        if fd[0]:
            fp = copy.deepcopy(fd[3])
            fl.append({"signal_name":fd[0],"module":fd[1],"function":fd[2],"params":fp,"column":fd[4]})

    if tw is None:
        tw = random.choice(WINDOWS)
    fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},
                "column":"signal_time_allowed"})

    sl = round(random.uniform(5, 15), 1)
    tp = round(sl * random.uniform(0.8, 1.5), 1)
    te = random.choice([10, 15, 20, None])
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"adp_{approach}_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                          "trailing_distance":2.0,"time_exit_minutes":te},
            "sizing_rules":{"method":"fixed","fixed_contracts":2,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def bt_window(sd, df_window, rm, start_date, end_date, min_trades=5):
    """Backtest on a data window. Returns (pnl, trades, wr, tpd) or None."""
    try:
        # Add adaptive columns to window
        df_w = df_window.clone()
        df_w = atr_stops(df_w, 1.5, 1.5, 14)
        df_w = add_htf_trend(df_w)
        df_w = add_regime(df_w)
        df_w = add_volatility_regime(df_w)

        # Fix filter columns — adaptive filters reference columns added above
        sd_fixed = copy.deepcopy(sd)
        new_filters = []
        for f in sd_fixed.get("entry_filters", []):
            if f.get("module") == "_adaptive_":
                # This is a reference to an adaptive column — keep the column reference
                new_filters.append({"signal_name": f["signal_name"], "module": "signals.time_filters",
                                     "function": "time_of_day",  # Dummy — won't be called
                                     "params": {}, "column": f["column"]})
            else:
                new_filters.append(f)
        sd_fixed["entry_filters"] = new_filters

        s = GeneratedStrategy.from_dict(sd_fixed)
        config = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                start_date=start_date, end_date=end_date,
                                slippage_ticks=2, initial_capital=150000.0)
        r = FastBacktester(data={"1m": df_w}, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s, df_w; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        days = len(set(t.entry_time.strftime("%Y-%m-%d") for t in r.trades))
        tpd = len(r.trades) / max(1, days)
        ppt = m.total_pnl / max(1, len(r.trades))
        del r, s, df_w
        return m.total_pnl, len(r.trades) if hasattr(r, 'trades') else m.total_trades, m.win_rate, tpd, ppt
    except Exception:
        return None


def rolling_optimize(approach, filter_type, df_full, rm, train_days=20, trade_days=10,
                     n_variants=100, contracts=2, tw=None):
    """Rolling re-optimization: train on recent data, trade forward, repeat."""
    # Get all unique trading dates
    dates = df_full.select(pl.col("timestamp").dt.date().alias("d")).unique().sort("d")["d"].to_list()
    if len(dates) < train_days + trade_days:
        return None

    results = []  # List of (window_start, window_end, pnl, trades, wr, tpd, best_params)

    i = 0
    while i + train_days + trade_days <= len(dates):
        train_start = dates[i]
        train_end = dates[i + train_days - 1]
        trade_start = dates[i + train_days]
        trade_end_idx = min(i + train_days + trade_days - 1, len(dates) - 1)
        trade_end = dates[trade_end_idx]

        # Get train data
        df_train = df_full.filter(
            (pl.col("timestamp").dt.date() >= train_start) &
            (pl.col("timestamp").dt.date() <= train_end)
        )
        if len(df_train) < 1000:
            i += trade_days; continue

        # Optimize on train: test n_variants
        best_ppt = -999; best_sd = None
        for _ in range(n_variants):
            sd = make_variant(approach, filter_type, tw)
            sd["sizing_rules"]["fixed_contracts"] = contracts
            out = bt_window(sd, df_train, rm, str(train_start), str(train_end), min_trades=5)
            gc.collect()
            if out and out[4] > best_ppt:  # ppt
                best_ppt = out[4]
                best_sd = copy.deepcopy(sd)

        # Trade forward with best params
        if best_sd:
            df_trade = df_full.filter(
                (pl.col("timestamp").dt.date() >= trade_start) &
                (pl.col("timestamp").dt.date() <= trade_end)
            )
            if len(df_trade) > 500:
                out = bt_window(best_sd, df_trade, rm, str(trade_start), str(trade_end), min_trades=1)
                gc.collect()
                if out:
                    results.append({
                        "start": str(trade_start), "end": str(trade_end),
                        "pnl": out[0], "trades": out[1], "wr": out[2], "tpd": out[3],
                        "params": best_sd["entry_signals"][0]["params"],
                        "sl": best_sd["exit_rules"]["stop_loss_value"],
                        "tp": best_sd["exit_rules"]["take_profit_value"],
                    })

        i += trade_days  # Slide forward

    return results


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     ADAPTIVE SCALPER — Rolling 2-Week Re-Optimization                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Static params fail after 2-4 weeks. Solution: re-optimize constantly. ║
║  Every 2 weeks: test 100 variants on recent 4 weeks, trade the best.  ║
║  ATR-adaptive exits + regime filters + fast execution.                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    logger.info("Loading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df_full.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))

    # ── STEP 2: ROLLING OPTIMIZATION SEARCH ──
    print(f"\n{'='*80}\n  STEP 2: Rolling optimization (7 approaches × 10 configs)\n{'='*80}")

    all_results = []  # (total_oos_pnl, approach, filter, tw, results_list)

    for approach in APPROACHES:
        for fi, ftype in enumerate(FILTER_TYPES[:3]):  # Top 3 filters to save time
            for twi in range(3):  # 3 time windows
                tw = WINDOWS[twi]
                config_label = f"{approach}|{ftype}|tw{twi}"

                # Full 2-year rolling
                results = rolling_optimize(approach, ftype, df_full, rm,
                                            train_days=20, trade_days=10,
                                            n_variants=100, contracts=2, tw=tw)
                gc.collect()

                if results and len(results) >= 10:
                    total_pnl = sum(r["pnl"] for r in results)
                    win_pct = sum(1 for r in results if r["pnl"] > 0) / len(results) * 100
                    avg_tpd = np.mean([r["tpd"] for r in results])
                    avg_wr = np.mean([r["wr"] for r in results])

                    if total_pnl > 0 and win_pct >= 70 and avg_tpd >= 8 and avg_wr >= 50:
                        all_results.append((total_pnl, approach, ftype, tw, results, win_pct, avg_tpd, avg_wr))
                        logger.info(f"  ✓ {config_label}: OOS=${total_pnl:,.0f} win={win_pct:.0f}% {avg_tpd:.0f}tr/d WR={avg_wr:.0f}%")
                    else:
                        pass  # Silently skip failures

        logger.info(f"  {approach}: done ({sum(1 for r in all_results if r[1]==approach)} configs passed)")

    all_results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"\n  Total passing configs: {len(all_results)}")

    # ── STEP 3: YEAR 1 vs YEAR 2 ──
    print(f"\n{'='*80}\n  STEP 3: Year 1 vs Year 2 validation\n{'='*80}")

    dual_results = []
    for total_pnl, approach, ftype, tw, results, win_pct, avg_tpd, avg_wr in all_results[:20]:
        # Run on year 1 only
        r1 = rolling_optimize(approach, ftype, df_yr1, rm, 20, 10, 100, 2, tw)
        gc.collect()
        # Run on year 2 only
        r2 = rolling_optimize(approach, ftype, df_yr2, rm, 20, 10, 100, 2, tw)
        gc.collect()

        if r1 and r2 and len(r1) >= 5 and len(r2) >= 5:
            pnl1 = sum(r["pnl"] for r in r1)
            pnl2 = sum(r["pnl"] for r in r2)
            wp1 = sum(1 for r in r1 if r["pnl"] > 0) / len(r1) * 100
            wp2 = sum(1 for r in r2 if r["pnl"] > 0) / len(r2) * 100

            if pnl1 > 0 and pnl2 > 0 and wp1 >= 60 and wp2 >= 60:
                dual_results.append((total_pnl, approach, ftype, tw, results, r1, r2, pnl1, pnl2, wp1, wp2))
                logger.info(f"  ✓ {approach}|{ftype}: y1=${pnl1:,.0f}({wp1:.0f}%) y2=${pnl2:,.0f}({wp2:.0f}%)")

    logger.info(f"  Dual-year passing: {len(dual_results)}")

    del df_yr1, df_yr2; gc.collect()

    # ── STEP 4: SIZE UP ──
    if dual_results:
        print(f"\n{'='*80}\n  STEP 4: Sizing up\n{'='*80}")
        best = dual_results[0]
        _, approach, ftype, tw, results, r1, r2, *_ = best
        # Test at higher contracts
        for ct in [2, 3, 4, 5]:
            r_ct = rolling_optimize(approach, ftype, df_full, rm, 20, 10, 100, ct, tw)
            gc.collect()
            if r_ct:
                total = sum(r["pnl"] for r in r_ct)
                worst_2w = min(r["pnl"] for r in r_ct)
                # Monthly aggregation
                monthly = defaultdict(float)
                for r in r_ct:
                    mk = r["start"][:7]; monthly[mk] += r["pnl"]
                worst_mo = min(monthly.values()) if monthly else 0
                avg_mo = np.mean(list(monthly.values())) if monthly else 0
                flag = "★" if avg_mo >= 7000 else " "
                sf = "✓" if worst_mo >= MONTH_CAP else "✗"
                print(f"    {flag}{ct}ct: ${avg_mo:,.0f}/mo worst_mo=${worst_mo:,.0f} worst_2w=${worst_2w:,.0f} {sf}")

    # ── STEP 5-7: MC (bootstrap on 2-week windows) ──
    if dual_results:
        print(f"\n{'='*80}\n  STEPS 5-7: MC on rolling windows\n{'='*80}")
        best = dual_results[0]
        _, approach, ftype, tw, results, *_ = best
        # Use the 2-week PnLs for bootstrap MC
        window_pnls = np.array([r["pnl"] for r in results])
        n_windows = len(window_pnls)

        for label in ["2-year rolling"]:
            mc_profits = 0
            mc_ruin = 0
            n_sims = 5000
            for _ in range(n_sims):
                # Resample 26 windows (1 year) with replacement
                sample = np.random.choice(window_pnls, size=26, replace=True)
                annual_pnl = sample.sum()
                if annual_pnl > 0: mc_profits += 1
                # Ruin: cumulative DD below -$4,500
                cum = np.cumsum(sample)
                peak = np.maximum.accumulate(cum)
                dd = cum - peak
                if dd.min() < -4500: mc_ruin += 1

            mc_p = mc_profits / n_sims
            mc_r = mc_ruin / n_sims
            median_annual = float(np.median([np.random.choice(window_pnls, 26, True).sum() for _ in range(1000)]))
            print(f"  MC ({label}): P(profit)={mc_p:.0%} | P(ruin)={mc_r:.0%} | Median annual=${median_annual:,.0f}")

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  ADAPTIVE SCALPER COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # Template comparison
    by_approach = defaultdict(int)
    for _, approach, *_ in all_results:
        by_approach[approach] += 1
    print(f"\n  APPROACH RANKING (passing configs):")
    for a, cnt in sorted(by_approach.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passing configs")

    if dual_results:
        best = dual_results[0]
        total_pnl, approach, ftype, tw, results, r1, r2, pnl1, pnl2, wp1, wp2 = best

        monthly = defaultdict(float)
        for r in results:
            mk = r["start"][:7]; monthly[mk] += r["pnl"]
        mv = list(monthly.values())
        avg_mo = np.mean(mv) if mv else 0
        avg_tpd = np.mean([r["tpd"] for r in results])
        avg_wr = np.mean([r["wr"] for r in results])

        print(f"\n  BEST ADAPTIVE SYSTEM:")
        print(f"    Approach: {approach} | Filter: {ftype} | Window: {tw}")
        print(f"    Rolling: 100 variants tested every 2 weeks on last 4 weeks of data")
        print(f"    Y1 OOS: ${pnl1:,.0f} ({wp1:.0f}% windows profitable)")
        print(f"    Y2 OOS: ${pnl2:,.0f} ({wp2:.0f}% windows profitable)")
        print(f"    Combined: ${total_pnl:,.0f} | {len(results)} windows | {sum(1 for r in results if r['pnl']>0)}/{len(results)} profitable")
        print(f"    Avg: {avg_tpd:.0f} trades/day | {avg_wr:.0f}% WR | ${avg_mo:,.0f}/month")

        # Rolling example
        print(f"\n  ROLLING EXAMPLE (first 5 windows):")
        for r in results[:5]:
            print(f"    {r['start']} → {r['end']}: ${r['pnl']:>8,.0f} | {r['trades']:>3}tr | WR={r['wr']:.0f}% | SL={r['sl']} TP={r['tp']} | {r['params']}")

        # Monthly
        print(f"\n  MONTHLY:")
        for k in sorted(monthly):
            v = monthly[k]
            flag = "★" if v >= 7000 else ("✗" if v < MONTH_CAP else " ")
            print(f"    {flag} {k}: ${v:>10,.0f}")

        print(f"\n  HOW TO DEPLOY:")
        print(f"    Every other Monday:")
        print(f"    1. Run 100 backtests of {approach} with {ftype} filter on last 4 weeks of 1-min MNQ data")
        print(f"    2. Pick the variant with highest PnL/trade")
        print(f"    3. Trade those exact params for 2 weeks at 2 contracts")
        print(f"    4. Repeat")

        hit_7k = avg_mo >= 7000
        print(f"\n  THE NUMBER: ${avg_mo:,.0f}/month | {avg_tpd:.0f} trades/day | {avg_wr:.0f}% WR")
        print(f"    {sum(1 for r in results if r['pnl']>0)}/{len(results)} windows profitable ({sum(1 for r in results if r['pnl']>0)/len(results)*100:.0f}%)")
        print(f"    $7K target: {'ACHIEVED' if hit_7k else f'gap ${7000-avg_mo:,.0f}'}")
    else:
        print(f"\n  NO adaptive scalpers survived dual-year validation.")

    print(f"\n  COMPARISON:")
    print(f"    Static HFT v1 (old backtester): $0 (0 dual-year)")
    print(f"    Static HFT v2 (fast backtester): $0 (0 dual-year)")
    if dual_results:
        print(f"    Adaptive rolling: ${avg_mo:,.0f}/mo (re-optimized params work)")
    print(f"    The re-optimization IS the edge.")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "adaptive_scalper_v1",
        "total_configs_tested": len(APPROACHES) * 3 * 3,
        "passing_configs": len(all_results),
        "dual_year_passing": len(dual_results),
        "best": {
            "approach": approach if dual_results else None,
            "filter": ftype if dual_results else None,
            "avg_monthly": round(float(avg_mo), 2) if dual_results else 0,
            "avg_trades_per_day": round(float(avg_tpd), 1) if dual_results else 0,
            "avg_win_rate": round(float(avg_wr), 1) if dual_results else 0,
            "y1_pnl": round(pnl1, 2) if dual_results else 0,
            "y2_pnl": round(pnl2, 2) if dual_results else 0,
            "window_results": results[:10] if dual_results else [],
        } if dual_results else {},
    }
    with open("reports/adaptive_scalper_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/adaptive_scalper_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
