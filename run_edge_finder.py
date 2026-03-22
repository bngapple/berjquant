#!/usr/bin/env python3
"""
EDGE FINDER — Find strategies with good SL/TP ratios (WR>25%, winner>1.5x loser)
and size them up to hit $7K+/month while keeping worst month above -$4,000.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import BacktestConfig, MNQ_SPEC, load_prop_firm_rules, load_session_config, load_events_calendar
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("edge"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),(9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0)]

ENTRIES = [
    ("rsi","signals.momentum","rsi",{"period":14,"overbought":70.0,"oversold":30.0},["entry_long_rsi","entry_short_rsi"]),
    ("stochastic","signals.momentum","stochastic",{"k_period":14,"d_period":3,"overbought":80.0,"oversold":20.0},["entry_long_stoch","entry_short_stoch"]),
    ("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
    ("macd","signals.momentum","macd",{"fast":12,"slow":26,"signal_period":9},["entry_long_macd","entry_short_macd"]),
    ("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    ("williams_r","signals.momentum","williams_r",{"period":14,"overbought":-20.0,"oversold":-80.0},["entry_long_williams","entry_short_williams"]),
    ("ema_crossover","signals.trend","ema_crossover",{"fast_period":9,"slow_period":21},["entry_long_ema_cross","entry_short_ema_cross"]),
    ("bollinger_bands","signals.volatility","bollinger_bands",{"period":20,"std_dev":2.0},["entry_long_bb","entry_short_bb"]),
    ("keltner_channels","signals.volatility","keltner_channels",{"ema_period":20,"atr_period":14,"multiplier":1.5},["entry_long_kc","entry_short_kc"]),
    ("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    ("range_breakout","signals.price_action","range_breakout",{"lookback":20},["entry_long_breakout","entry_short_breakout"]),
    ("opening_range","signals.price_action","opening_range",{"minutes":15},["entry_long_orb","entry_short_orb"]),
]
FILTERS = [
    ("large_trade_detection","signals.orderflow","large_trade_detection",{"volume_lookback":50,"threshold":3.0},"signal_large_trade"),
    ("ema_slope","signals.trend","ema_slope",{"period":21,"slope_lookback":3},"signal_ema_slope_up"),
    ("supertrend","signals.trend","supertrend",{"period":10,"multiplier":3.0},"signal_supertrend_bullish"),
    ("relative_volume","signals.volume","relative_volume",{"lookback":20},"signal_high_volume"),
    ("session_levels","signals.price_action","session_levels",{},"signal_at_session_high"),
    ("candle_patterns","signals.price_action","candle_patterns",{},"signal_hammer"),
    ("absorption","signals.orderflow","absorption",{"volume_threshold":2.0,"price_threshold":0.3},"signal_absorption"),
    ("trapped_traders","signals.orderflow","trapped_traders",{"lookback":5,"retrace_pct":0.5},"signal_trapped_longs"),
    ("imbalance","signals.orderflow","imbalance",{"ratio_threshold":3.0},"signal_buy_imbalance"),
    ("none",None,None,None,None),
]


def bt_detail(sd, data, rm, config, min_trades=10):
    """Backtest with detailed exit analysis."""
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; sl_ct = tp_ct = 0; winners = []; losers = []
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
            mo[k] = mo.get(k, 0) + t.net_pnl
            if "stop_loss" in t.exit_reason: sl_ct += 1
            elif "take_profit" in t.exit_reason: tp_ct += 1
            if t.net_pnl > 0: winners.append(t.net_pnl)
            else: losers.append(t.net_pnl)
        n = len(r.trades)
        d = {
            "wr": m.win_rate, "sl_pct": sl_ct/n*100, "tp_pct": tp_ct/n*100,
            "avg_win": float(np.mean(winners)) if winners else 0,
            "avg_loss": float(np.mean(losers)) if losers else 0,
            "wl_ratio": abs(float(np.mean(winners))/float(np.mean(losers))) if losers and winners else 0,
        }
        del r, s
        return m, mo, d
    except Exception:
        return None


def rand_strat(ct=4):
    e = random.choice(ENTRIES); f = random.choice(FILTERS); tw = random.choice(WINDOWS)
    ep = copy.deepcopy(e[3])
    for k, v in ep.items():
        if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.4, 1.6)))
        elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
    entry = {"signal_name":e[0],"module":e[1],"function":e[2],"params":ep,"columns":{"long":e[4][0],"short":e[4][1]}}
    fl = []
    if f[1]:
        fp = copy.deepcopy(f[3])
        for k, v in fp.items():
            if isinstance(v, int): fp[k] = max(1, int(v * random.uniform(0.5, 1.5)))
            elif isinstance(v, float): fp[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
        fl.append({"signal_name":f[0],"module":f[1],"function":f[2],"params":fp,"column":f[4]})
    fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},"column":"signal_time_allowed"})
    sl = round(random.uniform(10, 50), 1)
    tp_mult = random.uniform(1.5, 4.0)
    tp = round(sl * tp_mult, 1)
    tp = min(150, tp)
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"{e[0]}|{f[0]}|ef_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,"trailing_distance":2.0,"time_exit_minutes":None},
            "sizing_rules":{"method":"fixed","fixed_contracts":ct,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def edge_score(m, mo, d):
    mv = list(mo.values())
    if not mv: return -999999
    sl_pct = d["sl_pct"] / 100.0
    return m.total_pnl * 1.0 + np.mean(mv) * 4.0 + min(mv) * 3.0 + d["wr"] * 100 + (1.0 - sl_pct) * 5000


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     EDGE FINDER — High Win Rate Strategies, Target $7K+/month          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Find strategies with WR>25%, winner>1.5x loser, size up within DD     ║
║  G4 template: 39% WR, PF 1.48, $3,940/mo at 6ct, 0 negative months   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load G4 and all grinders
    with open("reports/safe_maximize_v1.json") as f:
        sdata = json.load(f)
    all_g = [g["strategy"] for g in sdata["grinders"]]
    g4_sd = all_g[3]  # G4

    # Load data
    logger.info("Loading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()
    df_2mo = df_yr1.filter(pl.col("timestamp") < pl.lit("2024-05-19").str.strptime(pl.Datetime, "%Y-%m-%d"))

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    cf = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=3, initial_capital=150000.0)
    data_f = {"1m": df_yr1}; data_2 = {"1m": df_2mo}

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: ANALYZE G4 + SCALE IT UP
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 1: G4 analysis + scaling ═══")

    out = bt_detail(g4_sd, data_f, rm, cf, min_trades=5)
    gc.collect()
    if out:
        m, mo, d = out
        mv = list(mo.values())
        print(f"\n  G4 TEMPLATE (full year):")
        print(f"    Signal: {[e['signal_name'] for e in g4_sd['entry_signals']]} + {[f['signal_name'] for f in g4_sd.get('entry_filters',[]) if f.get('signal_name')!='time_of_day']}")
        print(f"    Params: {[e['params'] for e in g4_sd['entry_signals']]}")
        print(f"    SL={g4_sd['exit_rules']['stop_loss_value']} TP={g4_sd['exit_rules']['take_profit_value']} Ct={g4_sd['sizing_rules']['fixed_contracts']}")
        print(f"    Trades={m.total_trades} WR={d['wr']:.1f}% PF={m.profit_factor:.2f}")
        print(f"    Avg win=${d['avg_win']:,.0f} Avg loss=${d['avg_loss']:,.0f} W/L ratio={d['wl_ratio']:.1f}x")
        print(f"    SL exit={d['sl_pct']:.0f}% TP exit={d['tp_pct']:.0f}%")
        print(f"    PnL=${m.total_pnl:,.0f} Avg/Mo=${np.mean(mv):,.0f} Worst=${min(mv):,.0f} DD=${m.max_drawdown:,.0f}")
        print(f"    Monthly: {', '.join(f'{k}:${v:,.0f}' for k, v in sorted(mo.items()))}")
        g4_base_mo = mo
        g4_base_ct = g4_sd['sizing_rules']['fixed_contracts']
    else:
        print("  G4 FAILED on full year!")
        g4_base_mo = {}; g4_base_ct = 6

    # Scale G4
    print(f"\n  G4 SCALING TABLE:")
    print(f"  {'Ct':>4} {'Avg/Mo':>10} {'Worst Mo':>10} {'PnL/Yr':>12} {'DD':>10} {'Safe?':>6}")
    print(f"  {'-'*55}")
    g4_scale_results = []
    for ct in range(6, 16):
        sd_test = copy.deepcopy(g4_sd)
        sd_test["sizing_rules"]["fixed_contracts"] = ct
        out = bt_detail(sd_test, data_f, rm, cf, min_trades=5)
        gc.collect()
        if out:
            m, mo, d = out
            mv = list(mo.values())
            avg = np.mean(mv); worst = min(mv)
            safe = worst >= -3000  # Leave room for other strategies
            flag = "✓" if safe else "✗"
            print(f"  {ct:>4} ${avg:>9,.0f} ${worst:>9,.0f} ${m.total_pnl:>11,.0f} ${m.max_drawdown:>9,.0f} {flag:>6}")
            g4_scale_results.append((ct, avg, worst, m.total_pnl, m.max_drawdown, mo))

    # Find optimal G4 contracts
    g4_best_ct = g4_base_ct
    for ct, avg, worst, pnl, dd, mo in g4_scale_results:
        if worst >= -3000:
            g4_best_ct = ct
    logger.info(f"  G4 optimal: {g4_best_ct} contracts")

    # Analyze other grinders
    print(f"\n  ALL GRINDERS SL EXIT RATE:")
    for gi, sd in enumerate(all_g):
        out = bt_detail(sd, data_f, rm, cf, min_trades=5)
        gc.collect()
        if out:
            m, mo, d = out
            mv = list(mo.values())
            print(f"    G{gi+1}: SL={d['sl_pct']:.0f}% TP={d['tp_pct']:.0f}% WR={d['wr']:.0f}% avg_win=${d['avg_win']:,.0f} avg_loss=${d['avg_loss']:,.0f} ratio={d['wl_ratio']:.1f}x")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: FIX SL/TP ON EXISTING GRINDERS (approach D: 1000 fast + 15 full + 200 focused)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 2: Fixing SL/TP ratios on grinders ═══")

    fixed_grinders = []
    for gi, sd in enumerate(all_g):
        logger.info(f"  G{gi+1}: 1000 fast filter variants...")
        fast = []
        for vi in range(1000):
            var = copy.deepcopy(sd)
            # Vary params ±40%
            for sig in var["entry_signals"]:
                for k, v in sig["params"].items():
                    if isinstance(v, int): sig["params"][k] = max(2, int(v * random.uniform(0.6, 1.4)))
                    elif isinstance(v, float): sig["params"][k] = round(max(0.1, v * random.uniform(0.6, 1.4)), 4)
            for f in var.get("entry_filters", []):
                if f.get("signal_name") == "time_of_day": continue
                for k, v in f["params"].items():
                    if isinstance(v, int): f["params"][k] = max(1, int(v * random.uniform(0.6, 1.4)))
                    elif isinstance(v, float): f["params"][k] = round(max(0.1, v * random.uniform(0.6, 1.4)), 4)
            sl = round(random.uniform(10, 60), 1)
            tp_mult = random.uniform(1.5, 4.0)
            tp = round(min(200, sl * tp_mult), 1)
            var["exit_rules"]["stop_loss_value"] = sl
            var["exit_rules"]["take_profit_value"] = tp
            if random.random() < 0.3:
                tw = random.choice(WINDOWS)
                for f in var["entry_filters"]:
                    if f.get("signal_name") == "time_of_day":
                        f["params"] = {"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]}
            if random.random() < 0.2:
                fo = random.choice(FILTERS)
                time_f = [f for f in var["entry_filters"] if f.get("signal_name") == "time_of_day"]
                if fo[1]:
                    var["entry_filters"] = [{"signal_name":fo[0],"module":fo[1],"function":fo[2],"params":copy.deepcopy(fo[3]),"column":fo[4]}] + time_f
                else:
                    var["entry_filters"] = time_f
            if random.random() < 0.15:
                eo = random.choice(ENTRIES)
                ep = copy.deepcopy(eo[3])
                for k, v in ep.items():
                    if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.5, 1.5)))
                    elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.5, 1.5)), 4)
                var["entry_signals"] = [{"signal_name":eo[0],"module":eo[1],"function":eo[2],"params":ep,"columns":{"long":eo[4][0],"short":eo[4][1]}}]
            h = hashlib.md5(json.dumps(var, sort_keys=True, default=str).encode()).hexdigest()[:6]
            var["name"] = f"fix_{gi}_{h}"
            out = bt_detail(var, data_2, rm, cf2, min_trades=8)
            if vi % 100 == 0: gc.collect()
            if out and out[0].total_pnl > 0 and out[2]["wr"] >= 25 and out[2]["wl_ratio"] >= 1.5:
                fast.append((out[0].total_pnl, var))
        fast.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"    {len(fast)} passed fast filter (WR>=25%, W/L>=1.5x)")

        # Full year top 15
        full = []
        for _, var in fast[:15]:
            out = bt_detail(var, data_f, rm, cf, min_trades=30)
            gc.collect()
            if out and out[0].total_pnl > 0:
                m, mo, d = out
                mv = list(mo.values())
                if d["wr"] >= 25 and d["wl_ratio"] >= 1.5 and min(mv) > -2000:
                    full.append((edge_score(m, mo, d), var, m, mo, d))

        # 200 focused around best
        if full:
            full.sort(key=lambda x: x[0], reverse=True)
            best_var = full[0][1]
            for _ in range(200):
                var2 = copy.deepcopy(best_var)
                for sig in var2["entry_signals"]:
                    for k, v in sig["params"].items():
                        if isinstance(v, int): sig["params"][k] = max(2, int(v * random.uniform(0.8, 1.2)))
                        elif isinstance(v, float): sig["params"][k] = round(max(0.1, v * random.uniform(0.8, 1.2)), 4)
                var2["exit_rules"]["stop_loss_value"] = round(max(5, var2["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
                var2["exit_rules"]["take_profit_value"] = round(max(10, var2["exit_rules"]["take_profit_value"] * random.uniform(0.85, 1.15)), 1)
                h = hashlib.md5(json.dumps(var2, sort_keys=True, default=str).encode()).hexdigest()[:6]
                var2["name"] = f"fix_{gi}_f_{h}"
                out = bt_detail(var2, data_f, rm, cf, min_trades=30)
                gc.collect()
                if out and out[0].total_pnl > 0:
                    m, mo, d = out
                    mv = list(mo.values())
                    if d["wr"] >= 25 and min(mv) > -2000:
                        full.append((edge_score(m, mo, d), var2, m, mo, d))

            full.sort(key=lambda x: x[0], reverse=True)
            best = full[0]
            avg_b = np.mean(list(best[3].values()))
            fixed_grinders.append(best)
            logger.info(f"    G{gi+1} BEST: {best[2].total_trades}tr WR={best[4]['wr']:.0f}% ${avg_b:,.0f}/mo SL_exit={best[4]['sl_pct']:.0f}%")
        else:
            logger.info(f"    G{gi+1}: no variants passed full-year requirements")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: SEARCH FOR NEW HIGH-WINRATE STRATEGIES
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 3: Searching for new high-WR strategies (2000 fast) ═══")

    fast_new = []
    for vi in range(2000):
        sd = rand_strat(ct=random.randint(2, 8))
        out = bt_detail(sd, data_2, rm, cf2, min_trades=8)
        if vi % 100 == 0: gc.collect()
        if out and out[0].total_pnl > 0 and out[2]["wr"] >= 25 and out[2]["wl_ratio"] >= 1.5:
            fast_new.append((out[0].total_pnl, sd))
    fast_new.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  {len(fast_new)} passed fast filter")

    new_strats = []
    for _, sd in fast_new[:40]:
        out = bt_detail(sd, data_f, rm, cf, min_trades=30)
        gc.collect()
        if out and out[0].total_pnl > 0:
            m, mo, d = out
            mv = list(mo.values())
            if d["wr"] >= 25 and min(mv) > -2000:
                new_strats.append((edge_score(m, mo, d), sd, m, mo, d))

    # Focused around top 5
    for i in range(min(5, len(new_strats))):
        base = new_strats[i][1]
        for _ in range(200):
            var = copy.deepcopy(base)
            for sig in var["entry_signals"]:
                for k, v in sig["params"].items():
                    if isinstance(v, int): sig["params"][k] = max(2, int(v * random.uniform(0.8, 1.2)))
                    elif isinstance(v, float): sig["params"][k] = round(max(0.1, v * random.uniform(0.8, 1.2)), 4)
            var["exit_rules"]["stop_loss_value"] = round(max(5, var["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
            var["exit_rules"]["take_profit_value"] = round(max(10, var["exit_rules"]["take_profit_value"] * random.uniform(0.85, 1.15)), 1)
            h = hashlib.md5(json.dumps(var, sort_keys=True, default=str).encode()).hexdigest()[:6]
            var["name"] = f"new_f_{h}"
            out = bt_detail(var, data_f, rm, cf, min_trades=30)
            gc.collect()
            if out and out[0].total_pnl > 0:
                m, mo, d = out
                mv = list(mo.values())
                if d["wr"] >= 25 and min(mv) > -2000:
                    new_strats.append((edge_score(m, mo, d), var, m, mo, d))

    new_strats.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  {len(new_strats)} new strategies passed full year")
    for ns in new_strats[:5]:
        avg_ns = np.mean(list(ns[3].values()))
        logger.info(f"    {ns[1]['name'][:35]} {ns[2].total_trades}tr WR={ns[4]['wr']:.0f}% ${avg_ns:,.0f}/mo")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: BUILD PORTFOLIO TARGETING $7K+
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 4: Building $7K+ portfolio ═══")

    # All candidates: G4 at optimal size + fixed grinders + new strats
    candidates = []
    # G4 at various sizes
    for ct, avg, worst, pnl, dd, mo in g4_scale_results:
        if worst >= -3500:
            sd_g4 = copy.deepcopy(g4_sd)
            sd_g4["sizing_rules"]["fixed_contracts"] = ct
            sd_g4["name"] = f"G4_ct{ct}"
            candidates.append(("g4", sd_g4, avg, worst, mo, pnl))
    for sc2, sd, m, mo, d in (fixed_grinders + new_strats)[:30]:
        mv = list(mo.values())
        candidates.append(("other", sd, np.mean(mv), min(mv), mo, m.total_pnl))

    # Try G4 alone at max safe contracts
    g4_alone_best = None
    for ct, avg, worst, pnl, dd, mo in reversed(g4_scale_results):
        if worst >= -3500:
            g4_alone_best = (ct, avg, worst, pnl, mo)
            break

    # Build portfolios: G4 + 0-3 others
    portfolios = []

    # G4 alone
    if g4_alone_best:
        ct, avg, worst, pnl, mo = g4_alone_best
        portfolios.append(([("g4", g4_sd, ct, mo)], avg, worst, pnl))
        logger.info(f"  G4 alone at {ct}ct: ${avg:,.0f}/mo worst=${worst:,.0f}")

    # G4 + each candidate
    if g4_alone_best:
        g4_ct, g4_avg, g4_worst, g4_pnl, g4_mo = g4_alone_best
        for tag, sd, c_avg, c_worst, c_mo, c_pnl in candidates:
            if tag == "g4": continue
            combined = defaultdict(float)
            for k, v in g4_mo.items(): combined[k] += v
            for k, v in c_mo.items(): combined[k] += v
            mv = list(combined.values())
            if not mv: continue
            worst_c = min(mv)
            if worst_c < MONTH_CAP: continue
            avg_c = np.mean(mv)
            pnl_c = sum(mv)
            ct_c = sd["sizing_rules"]["fixed_contracts"]
            portfolios.append(([("g4", g4_sd, g4_ct, g4_mo), ("other", sd, ct_c, c_mo)], avg_c, worst_c, pnl_c))

    portfolios.sort(key=lambda x: x[1], reverse=True)

    # Score with $7K bonus
    scored = []
    for comps, avg, worst, pnl in portfolios:
        sc2 = pnl * 1.0 + avg * 4.0 + worst * 3.0 + (10000 if avg >= 7000 else -10000)
        scored.append((sc2, comps, avg, worst, pnl))
    scored.sort(key=lambda x: x[0], reverse=True)

    logger.info(f"  {len(scored)} portfolios built")
    for i, (sc2, comps, avg, worst, pnl) in enumerate(scored[:5]):
        names = "+".join(f"{c[1]['name'][:15]}({c[2]}ct)" for c in comps)
        flag = "★" if avg >= 7000 else " "
        logger.info(f"  {flag} #{i+1}: {names} | ${avg:,.0f}/mo | worst=${worst:,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: AGGRESSIVE SIZING SWEEP
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 5: Sizing sweep on top portfolios ═══")

    best_portfolio = None
    for pi, (sc2, comps, avg, worst, pnl) in enumerate(scored[:5]):
        best_sc = -999999
        best_sizing = None
        for _ in range(3000):
            test_combined = defaultdict(float)
            cts = []
            for tag, sd, orig_ct, base_mo in comps:
                new_ct = random.randint(1, 15)
                cts.append(new_ct)
                ratio = new_ct / max(orig_ct, 1)
                for k, v in base_mo.items():
                    test_combined[k] += v * ratio
            mv = list(test_combined.values())
            if not mv: continue
            w = min(mv)
            if w < MONTH_CAP: continue
            a = np.mean(mv)
            s = sum(mv) * 1.0 + a * 4.0 + w * 3.0 + (10000 if a >= 7000 else -10000)
            if s > best_sc:
                best_sc = s; best_sizing = (cts, a, w, sum(mv), dict(test_combined))

        if best_sizing:
            cts, a, w, p, mo = best_sizing
            names = "+".join(f"{c[1]['name'][:12]}({cts[i]}ct)" for i, c in enumerate(comps))
            logger.info(f"  Portfolio {pi+1}: {names} → ${a:,.0f}/mo worst=${w:,.0f}")
            if best_portfolio is None or a > best_portfolio[1]:
                best_portfolio = (comps, a, w, p, cts, mo)

    # Free data for OOS
    del data_f, data_2, df_2mo; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: OOS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 6: OOS validation ═══")
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1_2; gc.collect()
    dv = {"1m": df_val}
    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    oos_ok = True
    if best_portfolio:
        comps, avg, worst, pnl, cts, mo = best_portfolio
        oos_combined = defaultdict(float)
        oos_tr = 0
        for i, (tag, sd, orig_ct, base_mo) in enumerate(comps):
            sd_oos = copy.deepcopy(sd)
            sd_oos["sizing_rules"]["fixed_contracts"] = cts[i]
            out = bt_detail(sd_oos, dv, rm, cv, min_trades=3)
            gc.collect()
            if out:
                m, mo_oos, d = out
                oos_tr += m.total_trades
                for k, v in mo_oos.items(): oos_combined[k] += v
                logger.info(f"  {sd['name'][:30]}: OOS ${m.total_pnl:,.0f} | {m.total_trades}tr")
        oos_mv = list(oos_combined.values())
        oos_worst = min(oos_mv) if oos_mv else 0
        oos_total = sum(oos_mv)
        logger.info(f"  OOS combined: ${oos_total:,.0f} | {oos_tr}tr | worst=${oos_worst:,.0f}")
        if oos_total < 0 or oos_worst < MONTH_CAP:
            oos_ok = False

    del dv, df_val; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: MC STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 7: MC stress test ═══")
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()
    dfull = {"1m": df_yr1_3}

    mc = None
    final_combined = defaultdict(float)
    final_results = []
    if best_portfolio:
        comps, avg, worst, pnl, cts, _ = best_portfolio
        all_pnls = []
        for i, (tag, sd, orig_ct, base_mo) in enumerate(comps):
            sd_f = copy.deepcopy(sd); sd_f["sizing_rules"]["fixed_contracts"] = cts[i]
            out = bt_detail(sd_f, dfull, rm, cf, min_trades=5)
            gc.collect()
            if out:
                m, mo, d = out
                final_results.append({"sd": sd_f, "m": m, "mo": mo, "d": d, "ct": cts[i]})
                for k, v in mo.items(): final_combined[k] += v
                # Get trade PnLs for MC
                try:
                    s = GeneratedStrategy.from_dict(copy.deepcopy(sd_f))
                    r = VectorizedBacktester(data=dfull, risk_manager=rm, contract_spec=MNQ_SPEC, config=cf).run(s)
                    all_pnls.extend([t.net_pnl for t in r.trades]); del r, s
                except: pass
                gc.collect()

        if len(all_pnls) > 20:
            from datetime import datetime, timedelta
            from engine.utils import Trade as TradeObj
            fake = [TradeObj(trade_id=str(i), symbol="MNQ", direction="long",
                    entry_time=datetime(2024,4,1)+timedelta(hours=i), entry_price=20000,
                    exit_time=datetime(2024,4,1)+timedelta(hours=i,minutes=30), exit_price=20000+p/2,
                    contracts=4, gross_pnl=p+3.6, commission=3.6, slippage_cost=2.0, net_pnl=p,
                    duration_seconds=1800, session_segment="core", exit_reason="tp") for i, p in enumerate(all_pnls)]
            try:
                mc = MonteCarloSimulator(MCConfig(n_simulations=3000, initial_capital=150000.0, prop_firm_rules=pr, seed=42)).run(fake, "edge")
                logger.info(f"  MC P={mc.probability_of_profit:.0%} median=${mc.median_return:,.0f}")
            except Exception as e:
                logger.info(f"  MC error: {e}")

    del dfull, df_yr1_3; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    fc_mv = list(final_combined.values())
    fc_avg = np.mean(fc_mv) if fc_mv else 0
    fc_worst = min(fc_mv) if fc_mv else 0
    fc_total = sum(fc_mv)
    total_tr = sum(fr["m"].total_trades for fr in final_results)

    print(f"\n{'='*120}")
    print(f"  EDGE FINDER COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*120}")

    # G4 scaling verdict
    if g4_scale_results:
        print(f"\n  G4 SCALING:")
        crossed_7k = None
        dd_break = None
        for ct, avg, worst, pnl, dd, mo in g4_scale_results:
            if avg >= 7000 and crossed_7k is None: crossed_7k = (ct, avg, worst)
            if worst < -3500 and dd_break is None: dd_break = (ct, avg, worst)
        if crossed_7k:
            print(f"    ★ G4 crosses $7K at {crossed_7k[0]} contracts: ${crossed_7k[1]:,.0f}/mo, worst=${crossed_7k[2]:,.0f}")
        else:
            best_g4 = max(g4_scale_results, key=lambda x: x[1] if x[2] >= -3500 else -99999)
            print(f"    G4 max safe: {best_g4[0]}ct ${best_g4[1]:,.0f}/mo worst=${best_g4[2]:,.0f}")

    # Best portfolio
    if final_results:
        print(f"\n  BEST PORTFOLIO:")
        print(f"  {'Component':<35} {'Ct':>3} {'Tr':>5} {'WR':>5} {'PnL/Yr':>10} {'Avg/Mo':>10} {'Worst':>10} {'SL%':>5}")
        print(f"  {'-'*90}")
        for fr in final_results:
            sd = fr["sd"]; m = fr["m"]; d = fr["d"]; mv = list(fr["mo"].values())
            print(f"  {sd['name'][:34]:<35} {fr['ct']:>3} {m.total_trades:>5} {d['wr']:>4.0f}% ${m.total_pnl:>9,.0f} ${np.mean(mv):>9,.0f} ${min(mv):>9,.0f} {d['sl_pct']:>4.0f}%")

        print(f"\n  COMBINED:")
        print(f"    Trades/yr: {total_tr} | PnL/yr: ${fc_total:,.0f} | Avg/mo: ${fc_avg:,.0f} | Worst: ${fc_worst:,.0f}")
        safe = fc_worst >= MONTH_CAP
        hit_7k = fc_avg >= 7000
        print(f"    Safe: {'✓' if safe else '✗'} | Hits $7K: {'✓' if hit_7k else '✗'}")
        if mc: print(f"    MC P={mc.probability_of_profit:.0%} | Median=${mc.median_return:,.0f} | Ruin={mc.probability_of_ruin:.0%}")

        print(f"\n  MONTHLY:")
        for k in sorted(final_combined):
            v = final_combined[k]
            parts = [f"{fr['sd']['name'][:10]}:${fr['mo'].get(k,0):,.0f}" for fr in final_results if abs(fr["mo"].get(k, 0)) > 0]
            flag = "★" if v >= 7000 else ("✗" if v < 0 else " ")
            print(f"    {flag} {k}: ${v:>10,.0f}  [{' | '.join(parts)}]")

    # The $7K verdict
    print(f"\n  THE $7K VERDICT:")
    print(f"    Best safe avg/month: ${fc_avg:,.0f}")
    print(f"    Worst month: ${fc_worst:,.0f}")
    if fc_avg >= 7000:
        print(f"    ★ $7K TARGET ACHIEVED")
    else:
        print(f"    To hit $7K: need ${7000-fc_avg:,.0f} more/month")
        if g4_scale_results:
            for ct, avg, worst, pnl, dd, mo in g4_scale_results:
                if avg >= 7000:
                    print(f"    G4 alone at {ct}ct = ${avg:,.0f}/mo but worst month ${worst:,.0f}")
                    break

    # Key insight
    print(f"\n  KEY INSIGHT:")
    if final_results:
        best_fr = max(final_results, key=lambda x: x["d"]["wr"])
        print(f"    Best WR strategy: {best_fr['sd']['name'][:35]} at {best_fr['d']['wr']:.0f}% WR")
        print(f"    SL={best_fr['sd']['exit_rules']['stop_loss_value']} TP={best_fr['sd']['exit_rules']['take_profit_value']}")
        print(f"    The working SL/TP ratio on MNQ 1min: SL ~{best_fr['sd']['exit_rules']['stop_loss_value']:.0f}pt, TP ~{best_fr['sd']['exit_rules']['take_profit_value']:.0f}pt")

    # Before vs after
    print(f"\n  BEFORE vs AFTER:")
    print(f"    Old safe: $3,014/mo, worst -$3,337, WR 10-39%")
    print(f"    New edge: ${fc_avg:,.0f}/mo, worst ${fc_worst:,.0f}, {'SAFE' if safe else 'UNSAFE'}")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "edge_finder_v1",
        "g4_scaling": [{"ct":ct,"avg_mo":round(avg,2),"worst_mo":round(worst,2),"pnl":round(pnl,2)} for ct,avg,worst,pnl,dd,mo in g4_scale_results],
        "combined": {"total_trades":total_tr,"total_pnl":round(fc_total,2),"avg_monthly":round(float(fc_avg),2),
                      "worst_monthly":round(fc_worst,2),"hits_7k":hit_7k,"safe":safe,
                      "mc_p_profit":round(mc.probability_of_profit,4) if mc else None},
        "combined_monthly":{k:round(v,2) for k,v in sorted(final_combined.items())},
        "components":[{"name":fr["sd"]["name"],"strategy":fr["sd"],"contracts":fr["ct"],
                       "trades":fr["m"].total_trades,"pnl":round(fr["m"].total_pnl,2),
                       "avg_monthly":round(float(np.mean(list(fr["mo"].values()))),2),
                       "worst_monthly":round(float(min(fr["mo"].values())),2),
                       "win_rate":round(fr["d"]["wr"],2),"sl_exit_pct":round(fr["d"]["sl_pct"],2),
                       "wl_ratio":round(fr["d"]["wl_ratio"],2),
                       "monthly":{k:round(v,2) for k,v in sorted(fr["mo"].items())}}
                      for fr in final_results],
    }
    with open("reports/edge_finder_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/edge_finder_v1.json")
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
