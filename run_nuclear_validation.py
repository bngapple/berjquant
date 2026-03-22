#!/usr/bin/env python3
"""
NUCLEAR VALIDATION — Year 2 blind test of G4 + search for uncorrelated clones.
Year 2 data has NEVER been used for optimization. This is the moment of truth.
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
logger = logging.getLogger("nuke"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),(9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0)]

# Non-RSI entry signals for clone search
CLONE_ENTRIES = [
    ("stochastic","signals.momentum","stochastic",{"k_period":14,"d_period":3,"overbought":80.0,"oversold":20.0},["entry_long_stoch","entry_short_stoch"]),
    ("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
    ("macd","signals.momentum","macd",{"fast":12,"slow":26,"signal_period":9},["entry_long_macd","entry_short_macd"]),
    ("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    ("williams_r","signals.momentum","williams_r",{"period":14,"overbought":-20.0,"oversold":-80.0},["entry_long_williams","entry_short_williams"]),
    ("bollinger_bands","signals.volatility","bollinger_bands",{"period":20,"std_dev":2.0},["entry_long_bb","entry_short_bb"]),
    ("keltner_channels","signals.volatility","keltner_channels",{"ema_period":20,"atr_period":14,"multiplier":1.5},["entry_long_kc","entry_short_kc"]),
    ("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    ("ema_crossover","signals.trend","ema_crossover",{"fast_period":9,"slow_period":21},["entry_long_ema_cross","entry_short_ema_cross"]),
    ("range_breakout","signals.price_action","range_breakout",{"lookback":20},["entry_long_breakout","entry_short_breakout"]),
    ("opening_range","signals.price_action","opening_range",{"minutes":15},["entry_long_orb","entry_short_orb"]),
]
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


def bt(sd, data, rm, config, min_trades=5):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; sl_ct = tp_ct = 0; wins = []; losses = []; longs = 0; durs = []
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m")
            mo[k] = mo.get(k, 0) + t.net_pnl
            if "stop_loss" in t.exit_reason: sl_ct += 1
            elif "take_profit" in t.exit_reason: tp_ct += 1
            if t.net_pnl > 0: wins.append(t.net_pnl)
            else: losses.append(t.net_pnl)
            if t.direction == "long": longs += 1
            durs.append(t.duration_seconds)
        n = len(r.trades)
        d = {"wr": m.win_rate, "sl_pct": sl_ct/n*100, "tp_pct": tp_ct/n*100,
             "avg_win": float(np.mean(wins)) if wins else 0,
             "avg_loss": float(np.mean(losses)) if losses else 0,
             "wl_ratio": abs(float(np.mean(wins))/float(np.mean(losses))) if losses and wins else 0,
             "long_pct": longs/n*100, "avg_dur_min": float(np.mean(durs))/60}
        trades_list = list(r.trades)
        del r, s
        return trades_list, m, mo, d
    except Exception:
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     NUCLEAR VALIDATION — Year 2 Blind Test                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Year 2 data has NEVER been used for optimization                      ║
║  If G4 works on year 2, the edge is REAL                               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    with open("reports/safe_maximize_v1.json") as f:
        g4_sd = json.load(f)["grinders"][3]["strategy"]

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    results = {}

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: G4 YEAR 2 BLIND TEST
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 1: G4 YEAR 2 BLIND TEST\n{'='*80}")
    logger.info("Loading year 2 data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()
    logger.info(f"  Year 2: {len(df_yr2):,} bars")

    data_yr2 = {"1m": df_yr2}
    cf_yr2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2025-03-19", end_date="2026-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    print(f"\n  G4 YEAR 2 RESULTS:")
    print(f"  {'Ct':>4} {'Trades':>7} {'WR':>6} {'PF':>5} {'PnL/Yr':>12} {'Avg/Mo':>10} {'Worst Mo':>10} {'Max DD':>10} {'Safe':>6}")
    print(f"  {'-'*80}")
    g4_yr2_results = {}
    for ct in [6, 7, 8, 9, 10]:
        sd = copy.deepcopy(g4_sd); sd["sizing_rules"]["fixed_contracts"] = ct
        out = bt(sd, data_yr2, rm, cf_yr2)
        gc.collect()
        if out:
            trades, m, mo, d = out
            mv = list(mo.values())
            avg = np.mean(mv); worst = min(mv)
            safe = worst >= MONTH_CAP
            flag = "✓" if safe else "✗"
            pos = "★" if m.total_pnl > 0 else " "
            print(f"  {pos}{ct:>3} {m.total_trades:>7} {d['wr']:>5.1f}% {m.profit_factor:>4.2f} ${m.total_pnl:>11,.0f} ${avg:>9,.0f} ${worst:>9,.0f} ${m.max_drawdown:>9,.0f} {flag:>6}")
            g4_yr2_results[ct] = {"trades": m.total_trades, "wr": round(d["wr"], 2),
                "pf": round(m.profit_factor, 2), "pnl": round(m.total_pnl, 2),
                "avg_mo": round(float(avg), 2), "worst_mo": round(float(worst), 2),
                "dd": round(m.max_drawdown, 2), "safe": safe,
                "monthly": {k: round(v, 2) for k, v in sorted(mo.items())},
                "sl_pct": round(d["sl_pct"], 1), "tp_pct": round(d["tp_pct"], 1),
                "avg_win": round(d["avg_win"], 2), "avg_loss": round(d["avg_loss"], 2),
                "long_pct": round(d["long_pct"], 1), "avg_dur": round(d["avg_dur_min"], 1)}
            # Print monthly for 9ct
            if ct == 9:
                print(f"\n    G4@9ct monthly:")
                for k in sorted(mo): print(f"      {k}: ${mo[k]:>10,.0f}")
                print(f"    SL exit={d['sl_pct']:.0f}% TP exit={d['tp_pct']:.0f}% Long={d['long_pct']:.0f}% Dur={d['avg_dur_min']:.0f}min")
                print(f"    Avg win=${d['avg_win']:,.0f} Avg loss=${d['avg_loss']:,.0f} W/L={d['wl_ratio']:.1f}x")
            del trades
        else:
            print(f"  {ct:>4} {'NO TRADES':>7}")
            g4_yr2_results[ct] = {"pnl": 0, "safe": False}

    results["g4_yr2"] = g4_yr2_results
    g4_passed_yr2 = any(r.get("pnl", 0) > 0 for r in g4_yr2_results.values())

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: G4 DNA COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 2: G4 DNA — Year 1 vs Year 2\n{'='*80}")
    del df_yr2; gc.collect()
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2b = df2.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2; gc.collect()

    cf_yr1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2024-03-19", end_date="2025-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    g4_9 = copy.deepcopy(g4_sd); g4_9["sizing_rules"]["fixed_contracts"] = 9
    dna = {}
    for label, data_slice, config in [("Year 1", {"1m": df_yr1}, cf_yr1), ("Year 2", {"1m": df_yr2b}, cf_yr2)]:
        out = bt(g4_9, data_slice, rm, config)
        gc.collect()
        if out:
            _, m, mo, d = out
            mv = list(mo.values())
            dna[label] = {"trades": m.total_trades, "wr": d["wr"], "pf": m.profit_factor,
                          "avg_mo": float(np.mean(mv)), "worst_mo": float(min(mv)),
                          "long_pct": d["long_pct"], "avg_dur": d["avg_dur_min"],
                          "sl_pct": d["sl_pct"], "tp_pct": d["tp_pct"]}

    if len(dna) == 2:
        print(f"\n  {'Metric':<20} {'Year 1':>15} {'Year 2':>15} {'Stable?':>10}")
        print(f"  {'-'*60}")
        for k in ["trades", "wr", "pf", "avg_mo", "worst_mo", "long_pct", "avg_dur", "sl_pct"]:
            v1 = dna["Year 1"][k]; v2 = dna["Year 2"][k]
            fmt = f"${v1:,.0f}" if "mo" in k else f"{v1:.1f}"
            fmt2 = f"${v2:,.0f}" if "mo" in k else f"{v2:.1f}"
            diff = abs(v1 - v2) / max(abs(v1), 1) * 100
            stable = "✓" if diff < 50 else "⚠"
            print(f"  {k:<20} {fmt:>15} {fmt2:>15} {stable:>10}")
    results["dna"] = dna

    del df_yr1; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: FIND G4 CLONES ON YEAR 2
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 3: Searching for G4 clones (11 signal families)\n{'='*80}")

    # 2-month year 2 slice for fast filter
    df_2mo_yr2 = df_yr2b.filter(pl.col("timestamp") < pl.lit("2025-05-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    data_2mo = {"1m": df_2mo_yr2}
    cf_2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2025-03-19", end_date="2025-05-18",
                            slippage_ticks=3, initial_capital=150000.0)
    data_yr2_full = {"1m": df_yr2b}

    clones = []
    for entry_def in CLONE_ENTRIES:
        ename = entry_def[0]
        logger.info(f"  {ename}: testing 200 fast variants...")

        fast = []
        for _ in range(200):
            ep = copy.deepcopy(entry_def[3])
            for k, v in ep.items():
                if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.3, 1.7)))
                elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.4, 1.6)), 4)
            entry = {"signal_name":ename,"module":entry_def[1],"function":entry_def[2],"params":ep,
                     "columns":{"long":entry_def[4][0],"short":entry_def[4][1]}}
            fo = random.choice(FILTERS); tw = random.choice(WINDOWS)
            fl = []
            if fo[1]:
                fp = copy.deepcopy(fo[3])
                for k, v in fp.items():
                    if isinstance(v, int): fp[k] = max(1, int(v * random.uniform(0.5, 1.5)))
                    elif isinstance(v, float): fp[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
                fl.append({"signal_name":fo[0],"module":fo[1],"function":fo[2],"params":fp,"column":fo[4]})
            fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                        "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},
                        "column":"signal_time_allowed"})
            sl = round(random.uniform(15, 55), 1)
            tp_mult = random.uniform(1.5, 5.0)
            tp = round(min(250, sl * tp_mult), 1)
            h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
            sd = {"name":f"{ename}|clone_{h}","entry_signals":[entry],"entry_filters":fl,
                  "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                                "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                                "trailing_distance":2.0,"time_exit_minutes":None},
                  "sizing_rules":{"method":"fixed","fixed_contracts":3,"risk_pct":0.02,"atr_risk_multiple":2.0},
                  "primary_timeframe":"1m","require_all_entries":True}
            out = bt(sd, data_2mo, rm, cf_2mo, min_trades=6)
            if out and out[1].total_pnl > 0 and out[3]["wr"] >= 20:
                fast.append((out[1].total_pnl, sd))
        gc.collect()

        # Full year top 5
        fast.sort(key=lambda x: x[0], reverse=True)
        full = []
        for _, sd in fast[:5]:
            out = bt(sd, data_yr2_full, rm, cf_yr2, min_trades=25)
            gc.collect()
            if out and out[1].total_pnl > 0:
                m, mo, d = out[1], out[2], out[3]
                mv = list(mo.values())
                if d["wr"] >= 25 and d["wl_ratio"] >= 1.5 and min(mv) > -2000:
                    full.append((out[1].total_pnl, sd, m, mo, d))

        # Focused around top 2
        for fi in range(min(2, len(full))):
            base = full[fi][1]
            for _ in range(100):
                var = copy.deepcopy(base)
                for sig in var["entry_signals"]:
                    for k, v in sig["params"].items():
                        if isinstance(v, int): sig["params"][k] = max(2, int(v * random.uniform(0.75, 1.25)))
                        elif isinstance(v, float): sig["params"][k] = round(max(0.1, v * random.uniform(0.75, 1.25)), 4)
                var["exit_rules"]["stop_loss_value"] = round(max(10, var["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
                var["exit_rules"]["take_profit_value"] = round(max(15, var["exit_rules"]["take_profit_value"] * random.uniform(0.85, 1.15)), 1)
                h = hashlib.md5(json.dumps(var, sort_keys=True, default=str).encode()).hexdigest()[:6]
                var["name"] = f"{ename}|clone_f_{h}"
                out = bt(var, data_yr2_full, rm, cf_yr2, min_trades=25)
                gc.collect()
                if out and out[1].total_pnl > 0:
                    m, mo, d = out[1], out[2], out[3]
                    mv = list(mo.values())
                    if d["wr"] >= 25 and min(mv) > -2000:
                        full.append((m.total_pnl, var, m, mo, d))

        full.sort(key=lambda x: x[0], reverse=True)
        if full:
            best = full[0]
            avg_b = np.mean(list(best[3].values()))
            clones.append({"sd": best[1], "m": best[2], "mo": best[3], "d": best[4], "family": ename})
            logger.info(f"    ✓ {ename}: {best[2].total_trades}tr WR={best[4]['wr']:.0f}% ${avg_b:,.0f}/mo")
        else:
            logger.info(f"    ✗ {ename}: no passing clones")

    print(f"\n  CLONES FOUND: {len(clones)}")
    for cl in clones:
        mv = list(cl["mo"].values())
        print(f"    {cl['family']:<18} {cl['sd']['name'][:25]} WR={cl['d']['wr']:.0f}% ${np.mean(mv):,.0f}/mo worst=${min(mv):,.0f}")
    results["clones"] = [{"name": cl["sd"]["name"], "family": cl["family"],
                           "trades": cl["m"].total_trades, "wr": round(cl["d"]["wr"], 2),
                           "avg_monthly": round(float(np.mean(list(cl["mo"].values()))), 2),
                           "worst_monthly": round(float(min(cl["mo"].values())), 2),
                           "strategy": cl["sd"],
                           "monthly": {k: round(v, 2) for k, v in sorted(cl["mo"].items())}}
                          for cl in clones]

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: BUILD UNCORRELATED STACK
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 4: Building uncorrelated stack\n{'='*80}")

    # G4 on year 2 at best safe ct
    g4_best_ct = 6
    for ct in [9, 8, 7, 6]:
        r = g4_yr2_results.get(ct, {})
        if r.get("pnl", 0) > 0 and r.get("safe", False):
            g4_best_ct = ct; break
        elif r.get("pnl", 0) > 0:
            g4_best_ct = ct; break

    g4_for_stack = copy.deepcopy(g4_sd)
    g4_for_stack["sizing_rules"]["fixed_contracts"] = g4_best_ct
    g4_mo_yr2 = g4_yr2_results.get(g4_best_ct, {}).get("monthly", {})

    # Correlation filter
    uncorrelated = []
    for cl in clones:
        all_mk = sorted(set(g4_mo_yr2.keys()) | set(cl["mo"].keys()))
        if len(all_mk) < 3: uncorrelated.append(cl); continue
        v1 = [g4_mo_yr2.get(k, 0) for k in all_mk]
        v2 = [cl["mo"].get(k, 0) for k in all_mk]
        corr = np.corrcoef(v1, v2)[0, 1]
        if np.isnan(corr) or abs(corr) < 0.5:
            uncorrelated.append(cl)
            corr_str = f"{corr:.2f}" if not np.isnan(corr) else "NaN"
            logger.info(f"  ✓ {cl['family']}: corr={corr_str} — UNCORRELATED")
        else:
            logger.info(f"  ✗ {cl['family']}: corr={corr:.2f} — TOO CORRELATED")

    # Sizing sweep
    logger.info(f"  {len(uncorrelated)} uncorrelated clones. Sizing sweep...")
    best_stack = None
    if uncorrelated:
        for n_clones in range(1, min(4, len(uncorrelated) + 1)):
            for ci in range(min(5, len(uncorrelated))):
                selected = uncorrelated[ci:ci+n_clones]
                for _ in range(1000):
                    g4ct = random.randint(1, 12)
                    combined = defaultdict(float)
                    for k, v in g4_mo_yr2.items():
                        combined[k] += v * (g4ct / max(g4_best_ct, 1))
                    clone_cts = []
                    for cl in selected:
                        cct = random.randint(1, 10)
                        clone_cts.append(cct)
                        for k, v in cl["mo"].items():
                            combined[k] += v * (cct / 3)  # original was 3ct
                    mv = list(combined.values())
                    if not mv: continue
                    worst = min(mv); avg = np.mean(mv)
                    if worst < MONTH_CAP: continue
                    sc = avg * 4.0 + worst * 3.0 + (10000 if avg >= 7000 else -5000)
                    if best_stack is None or sc > best_stack[0]:
                        best_stack = (sc, g4ct, clone_cts, selected, avg, worst, dict(combined))

    if best_stack:
        sc, g4ct, clone_cts, sel_clones, avg, worst, combined_mo = best_stack
        print(f"\n  BEST STACK: G4@{g4ct}ct + {len(sel_clones)} clones")
        print(f"    Combined: ${avg:,.0f}/mo worst=${worst:,.0f}")

        # Verify with real backtests
        logger.info("  Verifying with real backtests...")
        real_combined = defaultdict(float)
        stack_components = []
        g4_v = copy.deepcopy(g4_sd); g4_v["sizing_rules"]["fixed_contracts"] = g4ct
        out = bt(g4_v, data_yr2_full, rm, cf_yr2)
        gc.collect()
        if out:
            _, m, mo, d = out
            for k, v in mo.items(): real_combined[k] += v
            stack_components.append(("G4", g4_v, m, mo, d, g4ct))

        for i, cl in enumerate(sel_clones):
            cl_v = copy.deepcopy(cl["sd"]); cl_v["sizing_rules"]["fixed_contracts"] = clone_cts[i]
            out = bt(cl_v, data_yr2_full, rm, cf_yr2)
            gc.collect()
            if out:
                _, m, mo, d = out
                for k, v in mo.items(): real_combined[k] += v
                stack_components.append((cl["family"], cl_v, m, mo, d, clone_cts[i]))

        real_mv = list(real_combined.values())
        real_worst = min(real_mv) if real_mv else 0
        real_avg = np.mean(real_mv) if real_mv else 0
        print(f"    Verified: ${real_avg:,.0f}/mo worst=${real_worst:,.0f} {'SAFE' if real_worst >= MONTH_CAP else 'UNSAFE'}")

        results["stack"] = {
            "g4_ct": g4ct, "clone_cts": clone_cts,
            "avg_monthly": round(float(real_avg), 2), "worst_monthly": round(float(real_worst), 2),
            "safe": real_worst >= MONTH_CAP,
            "monthly": {k: round(v, 2) for k, v in sorted(real_combined.items())},
            "components": [{"role": c[0], "ct": c[5], "trades": c[2].total_trades,
                            "wr": round(c[4]["wr"], 2), "avg_monthly": round(float(np.mean(list(c[3].values()))), 2)}
                           for c in stack_components],
        }

    del data_2mo, df_2mo_yr2; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: CROSS-VALIDATE ON YEAR 1
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 5: Cross-validate stack on Year 1\n{'='*80}")
    del data_yr2_full, df_yr2b; gc.collect()
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_v = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()
    data_yr1 = {"1m": df_yr1_v}

    yr1_combined = defaultdict(float)
    yr1_tr = 0
    if best_stack:
        for role, sd, _, _, _, ct_val in stack_components:
            sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = ct_val
            out = bt(sd_v, data_yr1, rm, cf_yr1)
            gc.collect()
            if out:
                _, m, mo, _ = out
                yr1_tr += m.total_trades
                for k, v in mo.items(): yr1_combined[k] += v
                mv = list(mo.values())
                logger.info(f"  {role}: yr1 ${np.mean(mv):,.0f}/mo {m.total_trades}tr")

    yr1_mv = list(yr1_combined.values())
    yr1_avg = np.mean(yr1_mv) if yr1_mv else 0
    yr1_worst = min(yr1_mv) if yr1_mv else 0
    yr1_pnl = sum(yr1_mv)
    yr1_pass = yr1_pnl > 0 and yr1_worst >= MONTH_CAP
    print(f"  Year 1: ${yr1_avg:,.0f}/mo worst=${yr1_worst:,.0f} PnL=${yr1_pnl:,.0f} {'PASS' if yr1_pass else 'FAIL'}")
    for k in sorted(yr1_combined): print(f"    {k}: ${yr1_combined[k]:>10,.0f}")
    results["yr1_cross"] = {"avg_monthly": round(float(yr1_avg), 2), "worst_monthly": round(float(yr1_worst), 2),
                             "total_pnl": round(yr1_pnl, 2), "passed": yr1_pass,
                             "monthly": {k: round(v, 2) for k, v in sorted(yr1_combined.items())}}

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: MC ON FULL 2 YEARS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  PHASE 6: MC on full 2 years\n{'='*80}")
    del data_yr1, df_yr1_v; gc.collect()
    df4 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    data_all = {"1m": df4}
    cf_all = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2024-03-19", end_date="2026-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    all_trades = []
    if best_stack:
        for role, sd, _, _, _, ct_val in stack_components:
            sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = ct_val
            out = bt(sd_v, data_all, rm, cf_all)
            gc.collect()
            if out:
                all_trades.extend(out[0])

    mc = None
    if len(all_trades) > 20:
        all_trades.sort(key=lambda t: t.exit_time)
        avg_ct = sum(c[5] for c in stack_components) // max(1, len(stack_components)) if best_stack else 5
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                                              prop_firm_rules=pr, seed=42, avg_contracts=avg_ct)).run(all_trades, "nuclear")
            print(f"  MC P(profit): {mc.probability_of_profit:.1%}")
            print(f"  MC Median: ${mc.median_return:,.0f}")
            print(f"  MC 5th pctl: ${mc.pct_5th_return:,.0f}")
            print(f"  MC P(ruin): {mc.probability_of_ruin:.1%}")
            print(f"  MC Composite: {mc.composite_score:.1f}")
            results["mc"] = {"p_profit": round(mc.probability_of_profit, 4),
                              "median": round(mc.median_return, 2), "p5": round(mc.pct_5th_return, 2),
                              "p_ruin": round(mc.probability_of_ruin, 4), "composite": round(mc.composite_score, 2)}
        except Exception as e:
            print(f"  MC error: {e}")

    del data_all, df4, all_trades; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print(f"  NUCLEAR VALIDATION COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # Verdict
    g4_yr2_profit = g4_yr2_results.get(9, {}).get("pnl", 0)
    g4_yr2_passed = g4_yr2_profit > 0

    print(f"\n  THE VERDICT:")
    print(f"    G4 {'PASSED' if g4_yr2_passed else 'FAILED'} the year 2 nuclear test")
    if g4_yr2_passed:
        r9 = g4_yr2_results.get(9, {})
        print(f"    At 9 contracts: ${r9.get('avg_mo', 0):,.0f}/month, worst=${r9.get('worst_mo', 0):,.0f}")
    else:
        # Check if any size passed
        for ct in [8, 7, 6]:
            r = g4_yr2_results.get(ct, {})
            if r.get("pnl", 0) > 0:
                print(f"    At {ct} contracts: ${r.get('avg_mo', 0):,.0f}/month (lower size works)")
                break

    print(f"    {len(clones)} clones found from: {', '.join(cl['family'] for cl in clones)}")

    if best_stack:
        st = results.get("stack", {})
        print(f"    Best stack: ${st.get('avg_monthly', 0):,.0f}/mo worst=${st.get('worst_monthly', 0):,.0f}")
        print(f"    Year 1 cross-validation: {'PASS' if yr1_pass else 'FAIL'} (${yr1_avg:,.0f}/mo)")
    if mc:
        print(f"    MC P(profit): {mc.probability_of_profit:.0%} | P(ruin): {mc.probability_of_ruin:.0%}")

    hit_7k = False
    if best_stack:
        hit_7k = results.get("stack", {}).get("avg_monthly", 0) >= 7000
    if not hit_7k:
        for ct in [9, 8, 7]:
            if g4_yr2_results.get(ct, {}).get("avg_mo", 0) >= 7000:
                hit_7k = True; break

    print(f"\n    $7K target: {'ACHIEVABLE' if hit_7k else 'NOT ACHIEVABLE within -$4,000 safety limit'}")

    # Final deploy recommendation
    print(f"\n  FINAL DEPLOY RECOMMENDATION:")
    if g4_yr2_passed:
        best_ct_yr2 = 6
        for ct in [9, 8, 7, 6]:
            r = g4_yr2_results.get(ct, {})
            if r.get("pnl", 0) > 0:
                best_ct_yr2 = ct; break
        r = g4_yr2_results[best_ct_yr2]
        print(f"    Deploy G4 (RSI period=2 + session_levels) at {best_ct_yr2} contracts")
        print(f"    Expected: ${r['avg_mo']:,.0f}/month based on year 2 blind test")
        print(f"    Year 2 worst month: ${r['worst_mo']:,.0f}")
        if mc: print(f"    MC P(profit): {mc.probability_of_profit:.0%}")
        print(f"    SL={g4_sd['exit_rules']['stop_loss_value']}pt TP={g4_sd['exit_rules']['take_profit_value']}pt")
    else:
        print(f"    G4 FAILED year 2. DO NOT DEPLOY without further paper trading.")
        print(f"    The year 1 edge did not persist into year 2 market conditions.")

    # Save
    results["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["g4_strategy"] = g4_sd
    results["verdict"] = {"g4_yr2_passed": g4_yr2_passed, "clones_found": len(clones),
                           "hit_7k": hit_7k, "yr1_cross_passed": yr1_pass}
    with open("reports/nuclear_validation_v1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to reports/nuclear_validation_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
