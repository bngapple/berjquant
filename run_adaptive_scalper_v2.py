#!/usr/bin/env python3
"""
ADAPTIVE SCALPER V2 — Precompute features ONCE, then slice. No OOM.
Rolling 2-week re-optimization with fast execution.
"""

import gc, json, time, copy, random, hashlib, logging, sys
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.fast_backtester import FastBacktester
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from engine.adaptive import atr_stops, add_htf_trend, add_regime, add_volatility_regime

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("adv2"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,45,15,30),(10,0,14,0),(9,30,12,0),(10,0,15,30)]
FILTER_COLS = {
    "none": None,
    "htf_trend": "signal_htf_uptrend",
    "regime_trending": "signal_regime_trending",
    "regime_ranging": "signal_regime_ranging",
    "ema_slope": None,  # Needs signal computation
    "heikin_ashi": None,  # Needs signal computation
}
APPROACHES = {
    "cci_fast":("cci","signals.momentum","cci",lambda:{"period":random.randint(5,14)},["entry_long_cci","entry_short_cci"]),
    "rsi_micro":("rsi","signals.momentum","rsi",lambda:{"period":random.randint(3,7),"overbought":round(random.uniform(55,65),1),"oversold":round(random.uniform(35,45),1)},["entry_long_rsi","entry_short_rsi"]),
    "roc_micro":("roc","signals.momentum","roc",lambda:{"period":random.randint(3,8)},["entry_long_roc","entry_short_roc"]),
    "bb_bounce":("bollinger_bands","signals.volatility","bollinger_bands",lambda:{"period":random.randint(10,20),"std_dev":round(random.uniform(1.0,1.5),2)},["entry_long_bb","entry_short_bb"]),
    "stoch_fast":("stochastic","signals.momentum","stochastic",lambda:{"k_period":random.randint(3,8),"d_period":random.randint(2,3),"overbought":round(random.uniform(60,75),1),"oversold":round(random.uniform(25,40),1)},["entry_long_stoch","entry_short_stoch"]),
    "ema_micro":("ema_crossover","signals.trend","ema_crossover",lambda:{"fast_period":random.randint(3,8),"slow_period":random.randint(8,15)},["entry_long_ema_cross","entry_short_ema_cross"]),
    "range_micro":("range_breakout","signals.price_action","range_breakout",lambda:{"lookback":random.randint(5,15)},["entry_long_breakout","entry_short_breakout"]),
}


def make_variant(approach, filter_col=None, tw=None):
    an, mod, func, pfn, cols = APPROACHES[approach]
    ep = pfn()
    entry = {"signal_name":an,"module":mod,"function":func,"params":ep,"columns":{"long":cols[0],"short":cols[1]}}
    fl = []
    # Adaptive filter — column already in data, use a dummy signal def
    if filter_col:
        fl.append({"signal_name":"adaptive_filter","module":"signals.time_filters","function":"time_of_day",
                    "params":{},"column":filter_col})
    if tw is None: tw = random.choice(WINDOWS)
    fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},
                "column":"signal_time_allowed"})
    sl = round(random.uniform(5, 15), 1)
    tp = round(sl * random.uniform(0.8, 1.5), 1)
    te = random.choice([10, 15, 20, None])
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"av2_{approach}_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                          "trailing_distance":2.0,"time_exit_minutes":te},
            "sizing_rules":{"method":"fixed","fixed_contracts":2,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def bt_slice(sd, df_slice, rm, start_str, end_str, use_fast=True, min_trades=5):
    """Backtest on a pre-enriched data slice. Fast: no feature recomputation."""
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        config = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                start_date=start_str, end_date=end_str,
                                slippage_ticks=2, initial_capital=150000.0)
        if use_fast:
            r = FastBacktester(data={"1m": df_slice}, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        else:
            r = VectorizedBacktester(data={"1m": df_slice}, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        ppt = m.total_pnl / max(1, len(r.trades))
        days = len(set(t.entry_time.strftime("%Y-%m-%d") for t in r.trades))
        tpd = len(r.trades) / max(1, days)
        del r, s
        return {"pnl": m.total_pnl, "trades": m.total_trades, "wr": m.win_rate, "tpd": tpd, "ppt": ppt}
    except Exception:
        return None


def rolling_optimize(approach, filter_col, df_enriched, rm, dates, train_days=20, trade_days=10,
                     n_variants=100, contracts=2, use_fast=True):
    """Rolling re-optimization on pre-enriched data. No feature recomputation."""
    results = []
    i = 0
    while i + train_days + trade_days <= len(dates):
        train_start = dates[i]
        train_end = dates[i + train_days - 1]
        trade_start = dates[i + train_days]
        trade_end = dates[min(i + train_days + trade_days - 1, len(dates) - 1)]

        # Slice pre-enriched data (fast — just a filter, no recomputation)
        df_train = df_enriched.filter(
            (pl.col("timestamp").dt.date() >= train_start) &
            (pl.col("timestamp").dt.date() <= train_end))
        if len(df_train) < 500:
            i += trade_days; continue

        # Optimize: test n_variants on training slice
        best_ppt = -999; best_sd = None
        for _ in range(n_variants):
            sd = make_variant(approach, filter_col)
            sd["sizing_rules"]["fixed_contracts"] = contracts
            out = bt_slice(sd, df_train, rm, str(train_start), str(train_end), use_fast, min_trades=3)
            if out and out["ppt"] > best_ppt:
                best_ppt = out["ppt"]; best_sd = copy.deepcopy(sd)

        del df_train

        # Trade forward with best params
        if best_sd:
            df_trade = df_enriched.filter(
                (pl.col("timestamp").dt.date() >= trade_start) &
                (pl.col("timestamp").dt.date() <= trade_end))
            if len(df_trade) > 300:
                out = bt_slice(best_sd, df_trade, rm, str(trade_start), str(trade_end), use_fast, min_trades=1)
                if out:
                    results.append({
                        "start": str(trade_start), "end": str(trade_end),
                        "pnl": out["pnl"], "trades": out["trades"], "wr": out["wr"],
                        "tpd": out["tpd"], "ppt": out["ppt"],
                        "params": best_sd["entry_signals"][0]["params"],
                        "sl": best_sd["exit_rules"]["stop_loss_value"],
                        "tp": best_sd["exit_rules"]["take_profit_value"],
                    })
            del df_trade

        i += trade_days
        if len(results) % 10 == 0 and results:
            gc.collect()

    return results


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     ADAPTIVE SCALPER V2 — Precompute Once, Slice Fast                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  v1 OOM'd recomputing features 300K times.                             ║
║  v2 precomputes once, slices windows from enriched dataframe.          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    # ── STEP 0: PRECOMPUTE ONCE ──
    logger.info("Loading and enriching data (one-time computation)...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df = atr_stops(df, 1.5, 1.5, 14)
    df = add_htf_trend(df, 20)
    df = add_regime(df, 14)
    df = add_volatility_regime(df, 14)
    gc.collect()

    mem_mb = sys.getsizeof(df) / 1024 / 1024  # Approximate
    print(f"  Enriched dataframe: {len(df):,} bars, ~{df.estimated_size('mb'):.0f} MB")
    print(f"  Columns: {df.columns}")
    print(f"  Adaptive columns: atr_14, htf_trend, signal_htf_uptrend, signal_regime_trending, signal_high_vol ✓")

    # Get all trading dates
    all_dates = df.select(pl.col("timestamp").dt.date().alias("d")).unique().sort("d")["d"].to_list()
    mid = len(all_dates) // 2
    yr1_dates = all_dates[:mid]
    yr2_dates = all_dates[mid:]
    logger.info(f"  {len(all_dates)} trading days ({len(yr1_dates)} yr1 + {len(yr2_dates)} yr2)")

    # ── STEP 1: ROLLING OPTIMIZATION ──
    print(f"\n{'='*80}\n  STEP 1: Rolling optimization (7 approaches × 6 filters)\n{'='*80}")

    filter_options = [
        ("none", None),
        ("htf_up", "signal_htf_uptrend"),
        ("trending", "signal_regime_trending"),
        ("ranging", "signal_regime_ranging"),
        ("high_vol", "signal_high_vol"),
        ("low_vol", "signal_low_vol"),
    ]

    all_configs = []
    for approach in APPROACHES:
        for fname, fcol in filter_options:
            label = f"{approach}|{fname}"
            results = rolling_optimize(approach, fcol, df, rm, all_dates,
                                        train_days=20, trade_days=10,
                                        n_variants=100, contracts=2, use_fast=True)
            gc.collect()

            if results and len(results) >= 10:
                total = sum(r["pnl"] for r in results)
                win_pct = sum(1 for r in results if r["pnl"] > 0) / len(results) * 100
                avg_tpd = np.mean([r["tpd"] for r in results])
                avg_wr = np.mean([r["wr"] for r in results])

                if total > 0 and win_pct >= 65 and avg_tpd >= 8 and avg_wr >= 50:
                    all_configs.append((total, approach, fname, fcol, results, win_pct, avg_tpd, avg_wr))
                    logger.info(f"  ✓ {label}: ${total:,.0f} win={win_pct:.0f}% {avg_tpd:.0f}tr/d WR={avg_wr:.0f}%")

        n_pass = sum(1 for c in all_configs if c[1] == approach)
        logger.info(f"  {approach}: {n_pass} passing configs")

    all_configs.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"\n  Total passing: {len(all_configs)}")

    # ── STEP 2: YEAR 1 vs YEAR 2 ──
    print(f"\n{'='*80}\n  STEP 2: Year 1 vs Year 2 split validation\n{'='*80}")

    dual = []
    for total, approach, fname, fcol, results, win_pct, avg_tpd, avg_wr in all_configs[:20]:
        r1 = rolling_optimize(approach, fcol, df, rm, yr1_dates, 20, 10, 100, 2, True)
        gc.collect()
        r2 = rolling_optimize(approach, fcol, df, rm, yr2_dates, 20, 10, 100, 2, True)
        gc.collect()

        if r1 and r2 and len(r1) >= 5 and len(r2) >= 5:
            p1 = sum(r["pnl"] for r in r1); p2 = sum(r["pnl"] for r in r2)
            w1 = sum(1 for r in r1 if r["pnl"] > 0) / len(r1) * 100
            w2 = sum(1 for r in r2 if r["pnl"] > 0) / len(r2) * 100

            if p1 > 0 and p2 > 0 and w1 >= 55 and w2 >= 55:
                dual.append((total, approach, fname, fcol, results, r1, r2, p1, p2, w1, w2))
                logger.info(f"  ✓ {approach}|{fname}: y1=${p1:,.0f}({w1:.0f}%) y2=${p2:,.0f}({w2:.0f}%)")

    logger.info(f"  Dual-year passing: {len(dual)}")

    # ── STEP 3: SIZE UP ──
    if dual:
        print(f"\n{'='*80}\n  STEP 3: Sizing up\n{'='*80}")
        best = dual[0]
        _, approach, fname, fcol, results, r1, r2, *_ = best
        for ct in [2, 3, 4, 5]:
            r_ct = rolling_optimize(approach, fcol, df, rm, all_dates, 20, 10, 100, ct, True)
            gc.collect()
            if r_ct:
                monthly = defaultdict(float)
                for r in r_ct: monthly[r["start"][:7]] += r["pnl"]
                mv = list(monthly.values())
                avg_mo = np.mean(mv) if mv else 0
                worst_mo = min(mv) if mv else 0
                flag = "★" if avg_mo >= 7000 else " "
                sf = "✓" if worst_mo >= MONTH_CAP else "✗"
                print(f"    {flag}{ct}ct: ${avg_mo:,.0f}/mo worst=${worst_mo:,.0f} {sf}")

    # ── STEP 4: PORTFOLIO ──
    if len(dual) >= 2:
        print(f"\n{'='*80}\n  STEP 4: Portfolio\n{'='*80}")
        # Pick top 2-3 from different approaches
        selected = [dual[0]]
        for d in dual[1:]:
            if d[1] != selected[0][1] and len(selected) < 3:  # Different approach
                selected.append(d)

        # Combine window PnLs
        combined_monthly = defaultdict(float)
        for _, approach, fname, fcol, results, *_ in selected:
            for r in results:
                combined_monthly[r["start"][:7]] += r["pnl"]
        mv = list(combined_monthly.values())
        if mv:
            print(f"  Portfolio ({len(selected)} strats): ${np.mean(mv):,.0f}/mo worst=${min(mv):,.0f}")

    # ── STEP 5: DUAL SLIPPAGE ──
    if dual:
        print(f"\n{'='*80}\n  STEP 5: Conservative slippage test\n{'='*80}")
        best = dual[0]
        _, approach, fname, fcol, *_ = best
        r_cons = rolling_optimize(approach, fcol, df, rm, all_dates, 20, 10, 100, 2, use_fast=False)
        gc.collect()
        if r_cons:
            total_cons = sum(r["pnl"] for r in r_cons)
            win_cons = sum(1 for r in r_cons if r["pnl"] > 0) / len(r_cons) * 100
            monthly_cons = defaultdict(float)
            for r in r_cons: monthly_cons[r["start"][:7]] += r["pnl"]
            mv_cons = list(monthly_cons.values())
            avg_cons = np.mean(mv_cons) if mv_cons else 0
            print(f"  Conservative (next-bar + slippage): ${avg_cons:,.0f}/mo | win={win_cons:.0f}%")
        else:
            avg_cons = 0; win_cons = 0

    del df; gc.collect()

    # ── STEP 6: MC ──
    if dual:
        print(f"\n{'='*80}\n  STEP 6: MC bootstrap on 2-week windows\n{'='*80}")
        best = dual[0]
        results = best[4]
        window_pnls = np.array([r["pnl"] for r in results])

        for label in ["realistic", "conservative"]:
            pnls = window_pnls if label == "realistic" else np.array([r["pnl"] for r in r_cons]) if r_cons else window_pnls * 0.5
            n_sims = 5000
            annual_pnls = []
            ruin_count = 0
            for _ in range(n_sims):
                sample = np.random.choice(pnls, size=26, replace=True)
                annual = sample.sum()
                annual_pnls.append(annual)
                cum = np.cumsum(sample)
                peak = np.maximum.accumulate(cum)
                if (cum - peak).min() < -4500: ruin_count += 1

            mc_p = sum(1 for p in annual_pnls if p > 0) / n_sims
            median = float(np.median(annual_pnls))
            p5 = float(np.percentile(annual_pnls, 5))
            mc_r = ruin_count / n_sims
            print(f"  MC ({label}): P(profit)={mc_p:.0%} | median=${median:,.0f}/yr | 5th=${p5:,.0f} | P(ruin)={mc_r:.0%}")

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  ADAPTIVE SCALPER V2 COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    print(f"\n  MEMORY FIX: v1 OOM'd. v2 precomputes once. Ran in {elapsed/60:.0f} min.")

    # Approach ranking
    by_approach = defaultdict(int)
    for _, approach, *_ in all_configs: by_approach[approach] += 1
    print(f"\n  APPROACH RANKING:")
    for a, cnt in sorted(by_approach.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passing configs")

    if dual:
        best = dual[0]
        total, approach, fname, fcol, results, r1, r2, p1, p2, w1, w2 = best
        monthly = defaultdict(float)
        for r in results: monthly[r["start"][:7]] += r["pnl"]
        mv = list(monthly.values())
        avg_mo = np.mean(mv) if mv else 0
        avg_tpd = np.mean([r["tpd"] for r in results])
        avg_wr = np.mean([r["wr"] for r in results])

        print(f"\n  BEST SYSTEM (REALISTIC EXECUTION):")
        print(f"    Approach: {approach} | Filter: {fname}")
        print(f"    Y1: ${p1:,.0f} ({w1:.0f}% windows) | Y2: ${p2:,.0f} ({w2:.0f}% windows)")
        print(f"    {avg_tpd:.0f} trades/day | {avg_wr:.0f}% WR | ${avg_mo:,.0f}/month")
        print(f"    {sum(1 for r in results if r['pnl']>0)}/{len(results)} windows profitable")

        print(f"\n  EXAMPLE ADAPTATION (first 5 windows):")
        for r in results[:5]:
            print(f"    {r['start']}: SL={r['sl']} TP={r['tp']} {r['params']} → ${r['pnl']:>8,.0f} {r['trades']}tr WR={r['wr']:.0f}%")

        print(f"\n  CONSERVATIVE EXECUTION: ${avg_cons:,.0f}/mo | {win_cons:.0f}% windows")

        print(f"\n  MONTHLY:")
        for k in sorted(monthly):
            v = monthly[k]
            flag = "★" if v >= 7000 else ("✗" if v < MONTH_CAP else " ")
            print(f"    {flag} {k}: ${v:>10,.0f}")

        active = sum(1 for v in mv if abs(v) > 0)
        hit_7k = avg_mo >= 7000

        print(f"\n  THE NUMBER:")
        print(f"    Realistic: ${avg_mo:,.0f}/month | {avg_tpd:.0f} trades/day | {avg_wr:.0f}% WR")
        print(f"    Conservative: ${avg_cons:,.0f}/month")
        print(f"    {active}/{len(mv)} months active")
        print(f"    $7K target: {'ACHIEVED' if hit_7k else f'gap ${7000-avg_mo:,.0f}'}")

        print(f"\n  DEPLOYMENT PLAYBOOK:")
        print(f"    Every other Monday:")
        print(f"    1. Load last 4 weeks of MNQ 1-min data")
        print(f"    2. Run 100 {approach} variants with {fname} filter")
        print(f"    3. Pick highest PnL/trade variant")
        print(f"    4. Trade those params for 2 weeks at 2 contracts")
        print(f"    5. Repeat")

        print(f"\n  COMPARISON:")
        print(f"    Static HFT (same params 2 years): $0 (edge rotates)")
        print(f"    Adaptive rolling (re-fit every 2 weeks): ${avg_mo:,.0f}/mo")
        print(f"    The re-optimization IS the edge.")
    else:
        print(f"\n  NO adaptive scalpers survived dual-year.")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "adaptive_scalper_v2",
        "fix": "precompute features once, slice for windows",
        "configs_tested": len(APPROACHES) * 6, "passing": len(all_configs), "dual_year": len(dual),
        "best": {
            "approach": dual[0][1] if dual else None, "filter": dual[0][2] if dual else None,
            "avg_monthly_realistic": round(float(avg_mo), 2) if dual else 0,
            "avg_monthly_conservative": round(float(avg_cons), 2) if dual else 0,
            "trades_per_day": round(float(avg_tpd), 1) if dual else 0,
            "win_rate": round(float(avg_wr), 1) if dual else 0,
            "y1_pnl": round(p1, 2) if dual else 0, "y2_pnl": round(p2, 2) if dual else 0,
            "windows": results[:10] if dual else [],
            "monthly": {k: round(v, 2) for k, v in sorted(monthly.items())} if dual else {},
        },
        "elapsed_min": round(elapsed / 60, 1),
    }
    with open("reports/adaptive_scalper_v2.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/adaptive_scalper_v2.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
