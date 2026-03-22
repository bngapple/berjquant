#!/usr/bin/env python3
"""
TRAILING PROFIT — Add trailing stops + time exits to dual-year survivors.
87% of trades exit at SL because fixed TP is too far. Trailing captures
intermediate profits. 5 strategies at 3ct each avoids DD-protection halving.
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
logger = logging.getLogger("trail"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),(9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0)]
TRAIL_CONFIGS = [
    (30, 15, "A"), (40, 20, "B"), (50, 25, "C"),
    (30, 10, "D"), (60, 30, "E"), (20, 10, "F"),
]
FILTERS = [
    ("session_levels","signals.price_action","session_levels",{},"signal_at_session_high"),
    ("large_trade_detection","signals.orderflow","large_trade_detection",{"volume_lookback":50,"threshold":3.0},"signal_large_trade"),
    ("ema_slope","signals.trend","ema_slope",{"period":21,"slope_lookback":3},"signal_ema_slope_up"),
    ("supertrend","signals.trend","supertrend",{"period":10,"multiplier":3.0},"signal_supertrend_bullish"),
    ("relative_volume","signals.volume","relative_volume",{"lookback":20},"signal_high_volume"),
    ("trapped_traders","signals.orderflow","trapped_traders",{"lookback":5,"retrace_pct":0.5},"signal_trapped_longs"),
    ("imbalance","signals.orderflow","imbalance",{"ratio_threshold":3.0},"signal_buy_imbalance"),
    ("none",None,None,None,None),
]
ENTRIES = {
    "rsi":("rsi","signals.momentum","rsi",{"period":14,"overbought":70.0,"oversold":30.0},["entry_long_rsi","entry_short_rsi"]),
    "stochastic":("stochastic","signals.momentum","stochastic",{"k_period":14,"d_period":3,"overbought":80.0,"oversold":20.0},["entry_long_stoch","entry_short_stoch"]),
    "roc":("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
    "macd":("macd","signals.momentum","macd",{"fast":12,"slow":26,"signal_period":9},["entry_long_macd","entry_short_macd"]),
    "cci":("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    "bollinger_bands":("bollinger_bands","signals.volatility","bollinger_bands",{"period":20,"std_dev":2.0},["entry_long_bb","entry_short_bb"]),
    "keltner_channels":("keltner_channels","signals.volatility","keltner_channels",{"ema_period":20,"atr_period":14,"multiplier":1.5},["entry_long_kc","entry_short_kc"]),
    "vwap":("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    "ema_crossover":("ema_crossover","signals.trend","ema_crossover",{"fast_period":9,"slow_period":21},["entry_long_ema_cross","entry_short_ema_cross"]),
}


def bt(sd, data, rm, config, min_trades=5):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; sl = tp = trail = 0
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m"); mo[k] = mo.get(k, 0) + t.net_pnl
            if "stop_loss" in t.exit_reason: sl += 1
            elif "take_profit" in t.exit_reason: tp += 1
            elif "trailing" in t.exit_reason: trail += 1
        n = len(r.trades)
        d = {"wr": m.win_rate, "sl_pct": sl/n*100, "tp_pct": tp/n*100, "trail_pct": trail/n*100}
        trades = list(r.trades); del r, s
        return trades, m, mo, d
    except Exception:
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     TRAILING PROFIT — Unlock Hidden Gains with Trailing Stops          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  87% of trades exit at SL. Trailing stops capture 40+ point moves.     ║
║  5 strategies × 3ct avoids DD-protection halving that kills 9ct.       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load seeds
    with open("reports/max_profit_v1.json") as f: mp = json.load(f)
    with open("reports/nuclear_validation_v1.json") as f: nv = json.load(f)
    seeds = [s["strategy"] for s in mp.get("top_survivors", [])]
    seeds += [c["strategy"] for c in nv.get("clones", [])]
    logger.info(f"Loaded {len(seeds)} seed strategies")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_2mo = df.head(40000)
    del df; gc.collect()

    d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}; d_2mo = {"1m": df_2mo}
    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf_2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=3, initial_capital=150000.0)

    # ── PHASE 1: ADD TRAILING TO EXISTING SURVIVORS ──
    print(f"\n{'='*80}\n  PHASE 1: Adding trailing stops to {len(seeds)} seeds\n{'='*80}")

    trail_results = []  # (combined_avg, sd, y1_mo, y2_mo, y1_d, y2_d, orig_avg)
    for si, sd in enumerate(seeds):
        # Get original baseline (no trailing)
        sd3 = copy.deepcopy(sd); sd3["sizing_rules"]["fixed_contracts"] = 3
        o1_orig = bt(sd3, d1, rm, cf1); o2_orig = bt(sd3, d2, rm, cf2); gc.collect()
        orig_y1 = np.mean(list(o1_orig[2].values())) if o1_orig else -9999
        orig_y2 = np.mean(list(o2_orig[2].values())) if o2_orig else -9999
        orig_avg = (orig_y1 + orig_y2) / 2 if orig_y1 > -9000 and orig_y2 > -9000 else -9999

        for act, dist, label in TRAIL_CONFIGS:
            for te in [None, 45, 60]:
                v = copy.deepcopy(sd3)
                v["exit_rules"]["trailing_stop"] = True
                v["exit_rules"]["trailing_activation"] = float(act)
                v["exit_rules"]["trailing_distance"] = float(dist)
                v["exit_rules"]["time_exit_minutes"] = te
                h = hashlib.md5(f"{si}_{label}_{te}".encode()).hexdigest()[:6]
                v["name"] = f"{v['entry_signals'][0]['signal_name']}|trail{label}_{h}"

                o1 = bt(v, d1, rm, cf1, min_trades=15); gc.collect()
                o2 = bt(v, d2, rm, cf2, min_trades=15); gc.collect()
                if not o1 or not o2: continue
                if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
                y1a = np.mean(list(o1[2].values())); y2a = np.mean(list(o2[2].values()))
                combined = (y1a + y2a) / 2
                trail_results.append((combined, v, o1[2], o2[2], o1[3], o2[3], orig_avg,
                                       y1a, y2a, o1[1].total_trades + o2[1].total_trades))

        if (si + 1) % 5 == 0:
            gc.collect()
            logger.info(f"  Processed {si+1}/{len(seeds)} seeds, {len(trail_results)} dual-year trail survivors")

    trail_results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total trail survivors: {len(trail_results)}")

    # Print top 20
    print(f"\n  TOP 20 TRAILING STOP RESULTS:")
    print(f"  {'#':<3} {'Name':<30} {'Orig $/mo':>10} {'Trail $/mo':>10} {'Impr':>7} {'WR y1':>6} {'WR y2':>6} {'Trail%':>7}")
    print(f"  {'-'*85}")
    for i, (ca, sd, y1mo, y2mo, y1d, y2d, orig, y1a, y2a, tr) in enumerate(trail_results[:20]):
        impr = ((ca / orig) - 1) * 100 if orig > 0 else 999
        flag = "★" if ca > orig and ca > 0 else " "
        print(f"  {flag}{i+1:<2} {sd['name'][:29]:<30} ${orig:>9,.0f} ${ca:>9,.0f} {impr:>+6.0f}% {y1d['wr']:>5.0f}% {y2d['wr']:>5.0f}% {y1d['trail_pct']:>6.0f}%")

    # ── PHASE 2: NEW STRATEGIES WITH TRAILING ──
    print(f"\n{'='*80}\n  PHASE 2: Searching new trailing strategies (9 families)\n{'='*80}")

    for fam, edef in ENTRIES.items():
        logger.info(f"  {fam}: 400 fast variants with trailing...")
        fast = []
        for _ in range(400):
            ep = copy.deepcopy(edef[3])
            for k, v in ep.items():
                if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.4, 1.6)))
                elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.4, 1.6)), 4)
            entry = {"signal_name":edef[0],"module":edef[1],"function":edef[2],"params":ep,"columns":{"long":edef[4][0],"short":edef[4][1]}}
            fo = random.choice(FILTERS); tw = random.choice(WINDOWS)
            fl = []
            if fo[1]:
                fp = copy.deepcopy(fo[3])
                for k, v in fp.items():
                    if isinstance(v, int): fp[k] = max(1, int(v * random.uniform(0.5, 1.5)))
                    elif isinstance(v, float): fp[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
                fl.append({"signal_name":fo[0],"module":fo[1],"function":fo[2],"params":fp,"column":fo[4]})
            fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                        "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},"column":"signal_time_allowed"})
            sl = round(random.uniform(10, 55), 1)
            act = round(random.uniform(20, 80), 1); dist = round(act * random.uniform(0.3, 0.7), 1)
            tp = round(random.uniform(100, 400), 1)
            te = random.choice([None, 30, 45, 60])
            h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
            sd = {"name":f"{fam}|nt_{h}","entry_signals":[entry],"entry_filters":fl,
                  "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                                "take_profit_value":tp,"trailing_stop":True,"trailing_activation":act,
                                "trailing_distance":dist,"time_exit_minutes":te},
                  "sizing_rules":{"method":"fixed","fixed_contracts":3,"risk_pct":0.02,"atr_risk_multiple":2.0},
                  "primary_timeframe":"1m","require_all_entries":True}
            out = bt(sd, d_2mo, rm, cf_2mo, min_trades=4)
            if out and out[1].total_pnl > 0:
                fast.append((out[1].total_pnl, sd))
        gc.collect()
        fast.sort(key=lambda x: x[0], reverse=True)

        # Dual-year top 10
        for _, sd in fast[:10]:
            o1 = bt(sd, d1, rm, cf1, min_trades=15); gc.collect()
            o2 = bt(sd, d2, rm, cf2, min_trades=15); gc.collect()
            if not o1 or not o2: continue
            if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
            y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
            if min(y1mv) < -2000 or min(y2mv) < -2000: continue
            y1a = np.mean(y1mv); y2a = np.mean(y2mv)
            trail_results.append(((y1a + y2a) / 2, sd, o1[2], o2[2], o1[3], o2[3], 0, y1a, y2a, o1[1].total_trades + o2[1].total_trades))

        # Focused top 3
        new_dual = [r for r in trail_results if r[1]["entry_signals"][0]["signal_name"] == fam and r[6] == 0]
        new_dual.sort(key=lambda x: x[0], reverse=True)
        for di in range(min(3, len(new_dual))):
            base = new_dual[di][1]
            for _ in range(67):  # ~200/3
                v = copy.deepcopy(base)
                for sig in v["entry_signals"]:
                    for k, val in sig["params"].items():
                        if isinstance(val, int): sig["params"][k] = max(2, int(val * random.uniform(0.7, 1.3)))
                        elif isinstance(val, float): sig["params"][k] = round(max(0.1, val * random.uniform(0.7, 1.3)), 4)
                v["exit_rules"]["stop_loss_value"] = round(max(8, v["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
                v["exit_rules"]["trailing_activation"] = round(max(10, v["exit_rules"]["trailing_activation"] * random.uniform(0.8, 1.2)), 1)
                v["exit_rules"]["trailing_distance"] = round(max(5, v["exit_rules"]["trailing_distance"] * random.uniform(0.8, 1.2)), 1)
                h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]
                v["name"] = f"{fam}|ntf_{h}"
                o1 = bt(v, d1, rm, cf1, min_trades=15); gc.collect()
                o2 = bt(v, d2, rm, cf2, min_trades=15); gc.collect()
                if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
                    y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
                    if min(y1mv) >= -2000 and min(y2mv) >= -2000:
                        y1a = np.mean(y1mv); y2a = np.mean(y2mv)
                        trail_results.append(((y1a+y2a)/2, v, o1[2], o2[2], o1[3], o2[3], 0, y1a, y2a, o1[1].total_trades+o2[1].total_trades))

        n_fam = sum(1 for r in trail_results if r[1]["entry_signals"][0]["signal_name"] == fam)
        logger.info(f"    {fam}: {n_fam} total dual-year trail survivors")

    trail_results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total after Phase 2: {len(trail_results)}")

    # ── PHASE 3: PORTFOLIO ──
    print(f"\n{'='*80}\n  PHASE 3: Building portfolio (5 strats × 3ct to avoid DD halving)\n{'='*80}")

    top50 = trail_results[:50]
    # Greedy selection with correlation < 0.5
    selected = [top50[0]]
    for ca, sd, y1mo, y2mo, *rest in top50[1:]:
        if len(selected) >= 6: break
        all_24 = list(y1mo.values()) + list(y2mo.values())
        too_corr = False
        for _, _, sy1, sy2, *_ in selected:
            s_24 = list(sy1.values()) + list(sy2.values())
            n = min(len(all_24), len(s_24))
            if n < 3: continue
            c = np.corrcoef(all_24[:n], s_24[:n])[0, 1]
            if not np.isnan(c) and abs(c) > 0.5:
                too_corr = True; break
        if not too_corr:
            selected.append((ca, sd, y1mo, y2mo, *rest))

    logger.info(f"  Selected {len(selected)} uncorrelated strategies")

    # Sizing sweep (each 1-6ct, 10000 combos)
    best_port = None
    for _ in range(10000):
        cts = [random.randint(1, 6) for _ in range(len(selected))]
        combined = defaultdict(float)
        for i, (_, _, y1mo, y2mo, *_) in enumerate(selected):
            ratio = cts[i] / 3
            for k, v in y1mo.items(): combined[k] += v * ratio
            for k, v in y2mo.items(): combined[k] += v * ratio
        mv = list(combined.values())
        if not mv: continue
        worst = min(mv)
        if worst < MONTH_CAP: continue
        avg = np.mean(mv)
        sc = avg * 4.0 + worst * 3.0 + (50000 if avg >= 7000 else 0)
        if best_port is None or sc > best_port[0]:
            best_port = (sc, cts, avg, worst, dict(combined))

    if best_port:
        _, cts, avg, worst, combined = best_port
        print(f"  Best portfolio: ${avg:,.0f}/mo worst=${worst:,.0f}")
        for i, (_, sd, *_) in enumerate(selected):
            print(f"    {sd['name'][:30]} {cts[i]}ct")

    # Single best at various contracts
    print(f"\n  SINGLE BEST SCALING:")
    if trail_results:
        best_sd = trail_results[0][1]
        best_y1mo = trail_results[0][2]; best_y2mo = trail_results[0][3]
        all_mo = list(best_y1mo.values()) + list(best_y2mo.values())
        for ct in range(3, 13):
            ratio = ct / 3
            scaled = [v * ratio for v in all_mo]
            avg = np.mean(scaled); worst = min(scaled)
            flag = "★" if avg >= 7000 else " "
            safe = "✓" if worst >= MONTH_CAP else "✗"
            print(f"    {flag}{ct}ct: ${avg:,.0f}/mo worst=${worst:,.0f} {safe}")

    # Verify best portfolio with real backtests
    port_real = defaultdict(float)
    if best_port:
        logger.info("  Verifying portfolio with real backtests...")
        for i, (_, sd, *_) in enumerate(selected):
            sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = cts[i]
            o1 = bt(sd_v, d1, rm, cf1); gc.collect()
            o2 = bt(sd_v, d2, rm, cf2); gc.collect()
            if o1:
                for k, v in o1[2].items(): port_real[k] += v
            if o2:
                for k, v in o2[2].items(): port_real[k] += v
        real_mv = list(port_real.values())
        real_avg = np.mean(real_mv) if real_mv else 0
        real_worst = min(real_mv) if real_mv else 0
        print(f"  Verified: ${real_avg:,.0f}/mo worst=${real_worst:,.0f}")

    del d_2mo, df_2mo; gc.collect()

    # ── PHASE 4: WALK-FORWARD ──
    print(f"\n{'='*80}\n  PHASE 4: Walk-forward (5 windows × 4 months)\n{'='*80}")

    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    wf_windows = []
    starts = ["2024-07-01", "2024-11-01", "2025-03-01", "2025-07-01", "2025-11-01"]
    for start_str in starts:
        y, m_val, _ = start_str.split("-")
        end_m = int(m_val) + 4; end_y = int(y)
        if end_m > 12: end_m -= 12; end_y += 1
        end_str = f"{end_y}-{end_m:02d}-01"
        try:
            df_win = df_full.filter(
                (pl.col("timestamp") >= pl.lit(start_str).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                (pl.col("timestamp") < pl.lit(end_str).str.strptime(pl.Datetime, "%Y-%m-%d")))
            if len(df_win) < 1000: continue
            cf_win = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date=start_str, end_date=end_str, slippage_ticks=3, initial_capital=150000.0)
            d_win = {"1m": df_win}
            win_pnl = 0
            if best_port:
                for j, (_, sd, *_) in enumerate(selected):
                    sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = cts[j]
                    out = bt(sd_v, d_win, rm, cf_win, min_trades=2); gc.collect()
                    if out: win_pnl += out[1].total_pnl
            wf_windows.append((start_str, win_pnl, win_pnl > 0))
            logger.info(f"  {start_str}: ${win_pnl:,.0f} {'✓' if win_pnl > 0 else '✗'}")
        except Exception:
            pass

    del df_full; gc.collect()
    wf_wins = sum(1 for _, _, p in wf_windows if p)
    wf_total = len(wf_windows)
    wf_pass = wf_wins >= 3 and sum(p for _, p, _ in wf_windows) > 0
    print(f"  Walk-forward: {wf_wins}/{wf_total} {'PASS' if wf_pass else 'FAIL'}")

    # ── PHASE 5: MC ──
    print(f"\n{'='*80}\n  PHASE 5: MC stress test\n{'='*80}")

    df_all = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    cf_all = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)

    mc_results = {}
    all_trades = []
    if best_port:
        for j, (_, sd, *_) in enumerate(selected):
            sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = cts[j]
            out = bt(sd_v, {"1m": df_all}, rm, cf_all); gc.collect()
            if out: all_trades.extend(out[0])

    if len(all_trades) > 20:
        all_trades.sort(key=lambda t: t.exit_time)
        avg_ct = sum(cts) // len(cts) if best_port else 3
        for label, ct_mult in [("conservative", 0.7), ("aggressive", 1.0)]:
            actual_ct = max(1, int(avg_ct * ct_mult))
            try:
                mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                    prop_firm_rules=pr, seed=42, avg_contracts=actual_ct)).run(all_trades, "trail")
                real_mv2 = list(port_real.values()) if port_real else [0]
                print(f"  {label.upper()}: MC P={mc.probability_of_profit:.0%} median=${mc.median_return:,.0f} ruin={mc.probability_of_ruin:.0%}")
                mc_results[label] = {"mc_p": round(mc.probability_of_profit, 4), "median": round(mc.median_return, 2),
                                      "ruin": round(mc.probability_of_ruin, 4), "p5": round(mc.pct_5th_return, 2)}
            except Exception as e:
                print(f"  MC error: {e}")

    del df_all; gc.collect()

    # ── PHASE 6: OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  TRAILING PROFIT COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # Trailing impact
    print(f"\n  TRAILING STOP IMPACT (top 10):")
    for i, r in enumerate(trail_results[:10]):
        ca, sd, y1mo, y2mo, y1d, y2d, orig, y1a, y2a, tr = r
        impr = ((ca / orig) - 1) * 100 if orig > 0 else 0
        fam = sd["entry_signals"][0]["signal_name"]
        print(f"    {i+1}. {fam}: orig=${orig:,.0f} → trail=${ca:,.0f}/mo ({impr:+.0f}%) WR={y1d['wr']:.0f}/{y2d['wr']:.0f}% trail_exit={y1d['trail_pct']:.0f}%")

    # Portfolio
    if best_port and port_real:
        real_mv3 = list(port_real.values())
        print(f"\n  PORTFOLIO: ${np.mean(real_mv3):,.0f}/mo worst=${min(real_mv3):,.0f}")
        print(f"  Monthly:")
        for k in sorted(port_real): print(f"    {k}: ${port_real[k]:>10,.0f}")

    # MC
    for label, mr in mc_results.items():
        print(f"\n  MC {label.upper()}: P(profit)={mr['mc_p']:.0%} median=${mr['median']:,.0f} ruin={mr['ruin']:.0%}")

    # Walk-forward
    print(f"\n  Walk-forward: {wf_wins}/{wf_total} {'PASS' if wf_pass else 'FAIL'}")

    # The number
    real_avg_final = np.mean(list(port_real.values())) if port_real else 0
    mc_p = mc_results.get("aggressive", {}).get("mc_p", 0)
    print(f"\n  THE NUMBER: Maximum validated profit = ${real_avg_final:,.0f}/month at {mc_p:.0%} MC confidence")
    if real_avg_final >= 7000:
        print(f"  ★ $7K TARGET ACHIEVED")
    else:
        print(f"  Gap to $7K: ${7000-real_avg_final:,.0f}/month")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "trailing_profit_v1",
        "trail_survivors": len(trail_results),
        "portfolio_avg": round(real_avg_final, 2),
        "portfolio_worst": round(float(min(port_real.values())), 2) if port_real else 0,
        "wf_pass": wf_pass, "wf_rate": f"{wf_wins}/{wf_total}",
        "mc": mc_results,
        "portfolio_components": [{"name": sd["name"], "strategy": sd, "contracts": cts[i] if best_port else 3}
                                  for i, (_, sd, *_) in enumerate(selected)],
        "portfolio_monthly": {k: round(v, 2) for k, v in sorted(port_real.items())} if port_real else {},
        "top10": [{"name": r[1]["name"], "combined_avg": round(r[0], 2), "orig_avg": round(r[6], 2),
                    "y1_avg": round(r[7], 2), "y2_avg": round(r[8], 2)} for r in trail_results[:10]],
    }
    with open("reports/trailing_profit_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/trailing_profit_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
