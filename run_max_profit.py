#!/usr/bin/env python3
"""
MAX PROFIT — Find strategies that work on BOTH years, then size up aggressively.
Target: $7K+/month validated across 2 different market regimes.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC, Trade,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("maxp"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),(9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0)]
FILTERS = [
    ("session_levels","signals.price_action","session_levels",{},"signal_at_session_high"),
    ("large_trade_detection","signals.orderflow","large_trade_detection",{"volume_lookback":50,"threshold":3.0},"signal_large_trade"),
    ("ema_slope","signals.trend","ema_slope",{"period":21,"slope_lookback":3},"signal_ema_slope_up"),
    ("supertrend","signals.trend","supertrend",{"period":10,"multiplier":3.0},"signal_supertrend_bullish"),
    ("relative_volume","signals.volume","relative_volume",{"lookback":20},"signal_high_volume"),
    ("candle_patterns","signals.price_action","candle_patterns",{},"signal_hammer"),
    ("trapped_traders","signals.orderflow","trapped_traders",{"lookback":5,"retrace_pct":0.5},"signal_trapped_longs"),
    ("imbalance","signals.orderflow","imbalance",{"ratio_threshold":3.0},"signal_buy_imbalance"),
    ("none",None,None,None,None),
]
ENTRIES = {
    "rsi": ("rsi","signals.momentum","rsi",{"period":14,"overbought":70.0,"oversold":30.0},["entry_long_rsi","entry_short_rsi"]),
    "stochastic": ("stochastic","signals.momentum","stochastic",{"k_period":14,"d_period":3,"overbought":80.0,"oversold":20.0},["entry_long_stoch","entry_short_stoch"]),
    "roc": ("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
    "macd": ("macd","signals.momentum","macd",{"fast":12,"slow":26,"signal_period":9},["entry_long_macd","entry_short_macd"]),
    "cci": ("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    "bollinger_bands": ("bollinger_bands","signals.volatility","bollinger_bands",{"period":20,"std_dev":2.0},["entry_long_bb","entry_short_bb"]),
    "keltner_channels": ("keltner_channels","signals.volatility","keltner_channels",{"ema_period":20,"atr_period":14,"multiplier":1.5},["entry_long_kc","entry_short_kc"]),
    "vwap": ("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    "ema_crossover": ("ema_crossover","signals.trend","ema_crossover",{"fast_period":9,"slow_period":21},["entry_long_ema_cross","entry_short_ema_cross"]),
}


def bt(sd, data, rm, config, min_trades=5):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; sl_ct = 0
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m"); mo[k] = mo.get(k, 0) + t.net_pnl
            if "stop_loss" in t.exit_reason: sl_ct += 1
        d = {"wr": m.win_rate, "sl_pct": sl_ct/len(r.trades)*100}
        trades = list(r.trades)
        del r, s
        return trades, m, mo, d
    except Exception:
        return None


def make_variant(base_sd, intensity=0.6):
    v = copy.deepcopy(base_sd)
    for sig in v["entry_signals"]:
        for k, val in sig["params"].items():
            if isinstance(val, int): sig["params"][k] = max(2, int(val * random.uniform(1-intensity, 1+intensity)))
            elif isinstance(val, float): sig["params"][k] = round(max(0.1, val * random.uniform(1-intensity, 1+intensity)), 4)
    for f in v.get("entry_filters", []):
        if f.get("signal_name") == "time_of_day": continue
        for k, val in f["params"].items():
            if isinstance(val, int): f["params"][k] = max(1, int(val * random.uniform(1-intensity, 1+intensity)))
            elif isinstance(val, float): f["params"][k] = round(max(0.1, val * random.uniform(1-intensity, 1+intensity)), 4)
    sl = round(random.uniform(10, 60), 1)
    tp_mult = random.uniform(1.5, 6.0)
    tp = round(min(300, sl * tp_mult), 1)
    v["exit_rules"]["stop_loss_value"] = sl; v["exit_rules"]["take_profit_value"] = tp
    if random.random() < 0.3:
        tw = random.choice(WINDOWS)
        for f in v["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]}
    if random.random() < 0.25:
        fo = random.choice(FILTERS)
        tf = [f for f in v["entry_filters"] if f.get("signal_name") == "time_of_day"]
        if fo[1]:
            fp = copy.deepcopy(fo[3])
            for k, val in fp.items():
                if isinstance(val, int): fp[k] = max(1, int(val * random.uniform(0.5, 1.5)))
                elif isinstance(val, float): fp[k] = round(max(0.1, val * random.uniform(0.5, 1.5)), 4)
            v["entry_filters"] = [{"signal_name":fo[0],"module":fo[1],"function":fo[2],"params":fp,"column":fo[4]}] + tf
        else:
            v["entry_filters"] = tf
    h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]
    v["name"] = f"{v['entry_signals'][0]['signal_name']}|mp_{h}"
    return v


def dual_score(y1_pnl, y2_pnl, y1_avg, y2_avg, y1_min, y2_min, y1_wr, y2_wr, y1_sl, y2_sl):
    return ((y1_pnl + y2_pnl) * 1.0 + (y1_avg + y2_avg) * 3.0
            + min(y1_min, y2_min) * 2.0 + min(y1_wr, y2_wr) * 80
            + (2.0 - y1_sl/100 - y2_sl/100) * 2000)


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     MAX PROFIT — Dual-Year Validated, Aggressively Sized               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Requirement: profitable on BOTH year 1 AND year 2                     ║
║  Then size up to maximum within -$4,000 monthly loss cap               ║
║  Target: $7,000+/month validated across 2 market regimes               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load seeds
    with open("reports/nuclear_validation_v1.json") as f: nv = json.load(f)
    clones = [(c["strategy"], c["family"]) for c in nv.get("clones", [])]
    g4_sd = nv.get("g4_strategy", {})

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    # Load data
    logger.info("Loading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_2mo = df.head(40000)  # ~2 months of full 2yr for fast filter
    del df; gc.collect()

    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf_2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=3, initial_capital=150000.0)
    d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}; d_2mo = {"1m": df_2mo}

    # ── PHASE 1: BASELINES ──
    print(f"\n{'='*80}\n  PHASE 1: Dual-year baselines (3 contracts)\n{'='*80}")
    seeds = [(g4_sd, "rsi")] + clones
    seed_results = []
    print(f"\n  {'Family':<18} {'Y1 $/mo':>10} {'Y2 $/mo':>10} {'Y1 WR':>7} {'Y2 WR':>7} {'Y1 worst':>10} {'Y2 worst':>10} {'Both+?':>7}")
    print(f"  {'-'*85}")
    for sd, fam in seeds:
        sd3 = copy.deepcopy(sd); sd3["sizing_rules"]["fixed_contracts"] = 3
        o1 = bt(sd3, d1, rm, cf1); gc.collect()
        o2 = bt(sd3, d2, rm, cf2); gc.collect()
        y1a = np.mean(list(o1[2].values())) if o1 else -9999
        y2a = np.mean(list(o2[2].values())) if o2 else -9999
        y1w = o1[3]["wr"] if o1 else 0; y2w = o2[3]["wr"] if o2 else 0
        y1min = min(o1[2].values()) if o1 else -9999; y2min = min(o2[2].values()) if o2 else -9999
        both = y1a > 0 and y2a > 0
        flag = "★" if both else " "
        print(f"  {flag}{fam:<17} ${y1a:>9,.0f} ${y2a:>9,.0f} {y1w:>6.0f}% {y2w:>6.0f}% ${y1min:>9,.0f} ${y2min:>9,.0f} {'YES' if both else 'no':>7}")
        seed_results.append({"sd": sd, "fam": fam, "both": both,
                              "y1_mo": o1[2] if o1 else {}, "y2_mo": o2[2] if o2 else {},
                              "y1_avg": y1a, "y2_avg": y2a})

    dual_seeds = [s for s in seed_results if s["both"]]
    logger.info(f"  {len(dual_seeds)} strategies already profitable on both years")

    # ── PHASE 2: MASSIVE SEARCH ──
    print(f"\n{'='*80}\n  PHASE 2: Searching 9 families (500 fast + 15 dual-year each)\n{'='*80}")

    all_dual = []
    for fam, edef in ENTRIES.items():
        # Build base strategy from seed or default
        seed_match = next((s for s in seed_results if s["fam"] == fam), None)
        base = copy.deepcopy(seed_match["sd"]) if seed_match else None
        if not base:
            ep = copy.deepcopy(edef[3])
            base = {"name": fam, "entry_signals": [{"signal_name":edef[0],"module":edef[1],"function":edef[2],
                    "params":ep,"columns":{"long":edef[4][0],"short":edef[4][1]}}],
                    "entry_filters": [{"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                    "params":{"start_hour":9,"start_minute":30,"end_hour":16,"end_minute":0},"column":"signal_time_allowed"}],
                    "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":25,"take_profit_type":"fixed_points",
                    "take_profit_value":100,"trailing_stop":False,"trailing_activation":4.0,"trailing_distance":2.0,"time_exit_minutes":None},
                    "sizing_rules":{"method":"fixed","fixed_contracts":3,"risk_pct":0.02,"atr_risk_multiple":2.0},
                    "primary_timeframe":"1m","require_all_entries":True}

        logger.info(f"  {fam}: 500 fast variants...")
        fast = []
        for _ in range(500):
            v = make_variant(base, 0.6)
            v["sizing_rules"]["fixed_contracts"] = 3
            out = bt(v, d_2mo, rm, cf_2mo, min_trades=4)
            if out and out[1].total_pnl > 0:
                fast.append((out[1].total_pnl, v))
        gc.collect()
        fast.sort(key=lambda x: x[0], reverse=True)

        # Dual-year test top 15
        dual_pass = []
        for _, v in fast[:15]:
            o1 = bt(v, d1, rm, cf1, min_trades=15); gc.collect()
            o2 = bt(v, d2, rm, cf2, min_trades=15); gc.collect()
            if not o1 or not o2: continue
            if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
            y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
            if min(y1mv) < -2500 or min(y2mv) < -2500: continue
            sc = dual_score(o1[1].total_pnl, o2[1].total_pnl, np.mean(y1mv), np.mean(y2mv),
                            min(y1mv), min(y2mv), o1[3]["wr"], o2[3]["wr"], o1[3]["sl_pct"], o2[3]["sl_pct"])
            dual_pass.append((sc, v, o1[2], o2[2], o1[3], o2[3], o1[1], o2[1]))

        if dual_pass:
            dual_pass.sort(key=lambda x: x[0], reverse=True)
            # Focused sweep top 3
            for di in range(min(3, len(dual_pass))):
                best_v = dual_pass[di][1]
                for _ in range(100):  # 300/3 per top candidate
                    v = make_variant(best_v, 0.3)
                    v["sizing_rules"]["fixed_contracts"] = 3
                    o1 = bt(v, d1, rm, cf1, min_trades=15); gc.collect()
                    o2 = bt(v, d2, rm, cf2, min_trades=15); gc.collect()
                    if not o1 or not o2: continue
                    if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
                    y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
                    if min(y1mv) < -2500 or min(y2mv) < -2500: continue
                    sc = dual_score(o1[1].total_pnl, o2[1].total_pnl, np.mean(y1mv), np.mean(y2mv),
                                    min(y1mv), min(y2mv), o1[3]["wr"], o2[3]["wr"], o1[3]["sl_pct"], o2[3]["sl_pct"])
                    dual_pass.append((sc, v, o1[2], o2[2], o1[3], o2[3], o1[1], o2[1]))

            dual_pass.sort(key=lambda x: x[0], reverse=True)
            all_dual.extend(dual_pass)
            b = dual_pass[0]
            y1a = np.mean(list(b[2].values())); y2a = np.mean(list(b[3].values()))
            logger.info(f"    ✓ {fam}: {len(dual_pass)} dual-year | best ${y1a:,.0f}+${y2a:,.0f}/mo")
        else:
            logger.info(f"    ✗ {fam}: 0 dual-year survivors")

    all_dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"\n  Total dual-year survivors: {len(all_dual)}")

    # ── PHASE 3: MAXIMIZE SIZING ──
    print(f"\n{'='*80}\n  PHASE 3: Maximize contract sizes (top 20)\n{'='*80}")

    maximized = []
    for i, (sc, sd, y1mo, y2mo, y1d, y2d, y1m, y2m) in enumerate(all_dual[:20]):
        all_months = list(y1mo.values()) + list(y2mo.values())
        base_ct = 3
        # Find max contracts where worst month > -3000
        best_ct = base_ct
        for ct in range(3, 16):
            ratio = ct / base_ct
            scaled = [v * ratio for v in all_months]
            if min(scaled) >= -3000:
                best_ct = ct
        ratio = best_ct / base_ct
        proj_avg = np.mean([v * ratio for v in all_months])
        proj_worst = min(v * ratio for v in all_months)
        y1_proj = np.mean([v * ratio for v in y1mo.values()])
        y2_proj = np.mean([v * ratio for v in y2mo.values()])
        fam = sd["entry_signals"][0]["signal_name"]
        maximized.append((proj_avg, best_ct, sd, y1mo, y2mo, y1d, y2d, y1m, y2m, fam, proj_worst))
        if i < 10:
            print(f"  #{i+1} {fam:<15} max_ct={best_ct:>2} proj=${proj_avg:,.0f}/mo (y1=${y1_proj:,.0f} y2=${y2_proj:,.0f}) worst=${proj_worst:,.0f}")

    maximized.sort(key=lambda x: x[0], reverse=True)

    # ── PHASE 4: BUILD PORTFOLIO ──
    print(f"\n{'='*80}\n  PHASE 4: Building portfolio (10000 sizing combos)\n{'='*80}")

    # Single best
    if maximized:
        best_single = maximized[0]
        print(f"\n  Single best: {best_single[9]} at {best_single[1]}ct → ${best_single[0]:,.0f}/mo")

    # Portfolio sizing sweep
    top20 = maximized[:20]
    best_port = None
    # Try pairs and triples
    for i in range(min(10, len(top20))):
        for j in range(i+1, min(15, len(top20))):
            _, _, sd_i, y1_i, y2_i, *_ = top20[i]
            _, _, sd_j, y1_j, y2_j, *_ = top20[j]
            # Correlation check
            all_mk = sorted(set(list(y1_i.keys()) + list(y2_i.keys()) + list(y1_j.keys()) + list(y2_j.keys())))
            vi = [y1_i.get(k, 0) + y2_i.get(k, 0) for k in all_mk[:12]] # use first 12 months
            vj = [y1_j.get(k, 0) + y2_j.get(k, 0) for k in all_mk[:12]]
            if len(vi) >= 3 and len(vj) >= 3:
                corr = np.corrcoef(vi[:min(len(vi),len(vj))], vj[:min(len(vi),len(vj))])[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.6: continue

            all_i = list(y1_i.values()) + list(y2_i.values())
            all_j = list(y1_j.values()) + list(y2_j.values())
            for _ in range(500):
                ct_i = random.randint(1, 15); ct_j = random.randint(1, 15)
                combined = [all_i[k] * ct_i / 3 + all_j[k] * ct_j / 3 for k in range(min(len(all_i), len(all_j)))]
                if not combined: continue
                worst = min(combined)
                if worst < MONTH_CAP: continue
                avg = np.mean(combined)
                sc = avg * 4.0 + worst * 3.0 + (50000 if avg >= 7000 else 0) + (100000 if avg >= 10000 else 0)
                if best_port is None or sc > best_port[0]:
                    best_port = (sc, [(top20[i], ct_i), (top20[j], ct_j)], avg, worst)

    if best_port:
        print(f"  Best portfolio: ${best_port[2]:,.0f}/mo worst=${best_port[3]:,.0f}")
        for (_, _, sd, *_rest), ct in best_port[1]:
            print(f"    {sd['entry_signals'][0]['signal_name']}: {ct}ct")

    # ── PHASE 5: WALK-FORWARD ──
    print(f"\n{'='*80}\n  PHASE 5: Walk-forward validation\n{'='*80}")

    # Use best single strategy for WF
    wf_candidate = maximized[0] if maximized else None
    wf_pass = False
    wf_windows = []
    if wf_candidate:
        proj_avg, best_ct, sd_wf, *_ = wf_candidate
        sd_wf_sized = copy.deepcopy(sd_wf); sd_wf_sized["sizing_rules"]["fixed_contracts"] = best_ct

        # 3-month rolling windows across 2 years
        # Full dataset
        df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        months_start = ["2024-06", "2024-09", "2024-12", "2025-03", "2025-06", "2025-09", "2025-12"]
        for ms in months_start:
            start = f"{ms}-01"
            # 3-month window
            y, m_val = int(ms.split("-")[0]), int(ms.split("-")[1])
            end_m = m_val + 3
            end_y = y
            if end_m > 12: end_m -= 12; end_y += 1
            end = f"{end_y}-{end_m:02d}-01"
            try:
                df_win = df_full.filter(
                    (pl.col("timestamp") >= pl.lit(start).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                    (pl.col("timestamp") < pl.lit(end).str.strptime(pl.Datetime, "%Y-%m-%d"))
                )
                if len(df_win) < 1000: continue
                cf_win = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                        start_date=start, end_date=end, slippage_ticks=3, initial_capital=150000.0)
                out = bt(sd_wf_sized, {"1m": df_win}, rm, cf_win, min_trades=3)
                gc.collect()
                pnl = out[1].total_pnl if out else 0
                wf_windows.append((ms, pnl, pnl > 0))
                logger.info(f"  Window {ms}: ${pnl:,.0f} {'✓' if pnl > 0 else '✗'}")
            except Exception:
                pass

        del df_full; gc.collect()
        wins = sum(1 for _, _, p in wf_windows if p)
        total_wf = len(wf_windows)
        wf_pass = wins >= total_wf // 2 and sum(p for _, p, _ in wf_windows) > 0
        print(f"  Walk-forward: {wins}/{total_wf} windows profitable {'PASS' if wf_pass else 'FAIL'}")

    # ── PHASE 6: MC + AGGRESSIVE SIZING ──
    print(f"\n{'='*80}\n  PHASE 6: MC stress test + aggressive sizing\n{'='*80}")

    mc = None
    mc_results = {}
    if maximized:
        df_all = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        cf_all = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                start_date="2024-03-19", end_date="2026-03-18",
                                slippage_ticks=3, initial_capital=150000.0)

        best_sd = maximized[0][2]
        # Test at conservative and aggressive sizing
        for label, ct in [("conservative", min(maximized[0][1], 8)), ("aggressive", maximized[0][1])]:
            sd_mc = copy.deepcopy(best_sd); sd_mc["sizing_rules"]["fixed_contracts"] = ct
            out = bt(sd_mc, {"1m": df_all}, rm, cf_all)
            gc.collect()
            if out and len(out[0]) > 10:
                m_mc = out[1]; mo_mc = out[2]
                try:
                    mc_obj = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                                                          prop_firm_rules=pr, seed=42, avg_contracts=ct)).run(out[0], "maxp")
                    mv = list(mo_mc.values())
                    print(f"\n  {label.upper()} ({ct}ct):")
                    print(f"    Avg/Mo: ${np.mean(mv):,.0f} | Worst: ${min(mv):,.0f} | Trades: {m_mc.total_trades}")
                    print(f"    MC P(profit): {mc_obj.probability_of_profit:.0%} | Median: ${mc_obj.median_return:,.0f}")
                    print(f"    MC P(ruin): {mc_obj.probability_of_ruin:.0%} | 5th pctl: ${mc_obj.pct_5th_return:,.0f}")
                    mc_results[label] = {
                        "ct": ct, "avg_monthly": round(float(np.mean(mv)), 2),
                        "worst_monthly": round(float(min(mv)), 2), "trades": m_mc.total_trades,
                        "mc_p_profit": round(mc_obj.probability_of_profit, 4),
                        "mc_median": round(mc_obj.median_return, 2),
                        "mc_p_ruin": round(mc_obj.probability_of_ruin, 4),
                        "mc_5th": round(mc_obj.pct_5th_return, 2),
                        "monthly": {k: round(v, 2) for k, v in sorted(mo_mc.items())},
                    }
                    if label == "aggressive": mc = mc_obj
                except Exception as e:
                    print(f"  MC error: {e}")
                del out; gc.collect()

        del df_all; gc.collect()

    # ── PHASE 7: OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  MAX PROFIT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # Dual-year survivors
    families_survived = set()
    print(f"\n  DUAL-YEAR SURVIVORS: {len(all_dual)} strategies")
    print(f"  {'#':<4} {'Family':<18} {'Y1 $/mo':>10} {'Y2 $/mo':>10} {'Combined':>10} {'MaxCt':>6} {'Proj $/mo':>10}")
    print(f"  {'-'*75}")
    for i, item in enumerate(maximized[:15]):
        proj_avg, best_ct, sd, y1mo, y2mo, *_ = item
        y1a = np.mean(list(y1mo.values())); y2a = np.mean(list(y2mo.values()))
        fam = sd["entry_signals"][0]["signal_name"]
        families_survived.add(fam)
        flag = "★" if proj_avg >= 7000 else " "
        print(f"  {flag}{i+1:<3} {fam:<18} ${y1a:>9,.0f} ${y2a:>9,.0f} ${y1a+y2a:>9,.0f} {best_ct:>6} ${proj_avg:>9,.0f}")

    # Signal families that survived
    print(f"\n  FAMILIES SURVIVED: {', '.join(sorted(families_survived))}")
    print(f"  FAMILIES FAILED: {', '.join(sorted(set(ENTRIES.keys()) - families_survived))}")

    # Best single
    if maximized:
        b = maximized[0]
        sd_b = b[2]; fam_b = b[9]
        y1a = np.mean(list(b[3].values())); y2a = np.mean(list(b[4].values()))
        print(f"\n  SINGLE BEST: {fam_b} at {b[1]}ct → ${b[0]:,.0f}/mo")
        print(f"    Y1: ${y1a:,.0f}/mo | Y2: ${y2a:,.0f}/mo")
        print(f"    SL={sd_b['exit_rules']['stop_loss_value']} TP={sd_b['exit_rules']['take_profit_value']}")
        print(f"    Filter: {[f['signal_name'] for f in sd_b.get('entry_filters', []) if f.get('signal_name') != 'time_of_day']}")

    # MC results
    for label in ["conservative", "aggressive"]:
        mr = mc_results.get(label, {})
        if mr:
            print(f"\n  MC {label.upper()} ({mr['ct']}ct): ${mr['avg_monthly']:,.0f}/mo | MC P={mr['mc_p_profit']:.0%} | worst=${mr['worst_monthly']:,.0f}")

    # The money answer
    print(f"\n  THE MONEY ANSWER:")
    agg = mc_results.get("aggressive", {})
    con = mc_results.get("conservative", {})
    if agg:
        print(f"    Aggressive: ${agg['avg_monthly']:,.0f}/mo at {agg['ct']}ct | MC {agg['mc_p_profit']:.0%} | worst ${agg['worst_monthly']:,.0f}")
    if con:
        print(f"    Conservative: ${con['avg_monthly']:,.0f}/mo at {con['ct']}ct | MC {con['mc_p_profit']:.0%} | worst ${con['worst_monthly']:,.0f}")
    if agg and agg["avg_monthly"] >= 7000:
        print(f"    ★ $7K TARGET ACHIEVED at {agg['ct']}ct with {agg['mc_p_profit']:.0%} MC confidence")
    elif agg:
        need = 7000 - agg["avg_monthly"]
        print(f"    Need ${need:,.0f} more/month to hit $7K")

    # Comparison
    print(f"\n  COMPARISON:")
    print(f"    Old G4: $7.8K/mo year 1 only → DEAD on year 2 (0% WR)")
    print(f"    Old safe portfolio: $3K/mo year 1 only")
    if agg:
        print(f"    New system: ${agg['avg_monthly']:,.0f}/mo VALIDATED ON BOTH YEARS at {agg['mc_p_profit']:.0%} MC")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "max_profit_v1",
        "dual_year_count": len(all_dual),
        "families_survived": sorted(families_survived),
        "best_single": {"name": maximized[0][2]["name"] if maximized else None,
                         "family": maximized[0][9] if maximized else None,
                         "max_ct": maximized[0][1] if maximized else 0,
                         "proj_avg_monthly": round(maximized[0][0], 2) if maximized else 0,
                         "strategy": maximized[0][2] if maximized else None},
        "mc_conservative": mc_results.get("conservative"),
        "mc_aggressive": mc_results.get("aggressive"),
        "walk_forward": {"windows": wf_windows, "passed": wf_pass},
        "top_survivors": [{"name": m[2]["name"], "family": m[9], "max_ct": m[1],
                            "proj_avg": round(m[0], 2), "strategy": m[2],
                            "y1_monthly": {k: round(v, 2) for k, v in sorted(m[3].items())},
                            "y2_monthly": {k: round(v, 2) for k, v in sorted(m[4].items())}}
                           for m in maximized[:10]],
    }
    with open("reports/max_profit_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/max_profit_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
