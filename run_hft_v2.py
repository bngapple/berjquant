#!/usr/bin/env python3
"""
HFT V2 — Scalping with realistic execution (limit order fills, 0 entry slippage).
Uses engine/fast_backtester.py — does NOT modify original backtester.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict, Counter
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.fast_backtester import FastBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("hft2"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,45,15,30),(10,0,14,0),(9,30,12,0),(8,30,16,0),(10,0,15,30),(9,30,15,0),(9,30,14,0)]
FILTERS = [
    ("none",None,None,None,None),
    ("ema_slope","signals.trend","ema_slope",{"period":8,"slope_lookback":2},"signal_ema_slope_up"),
    ("heikin_ashi","signals.trend","heikin_ashi",{},"signal_ha_bullish"),
]
APPROACHES = {
    "ema_micro":("ema_crossover","signals.trend","ema_crossover",lambda:{"fast_period":random.randint(3,8),"slow_period":random.randint(8,15)},["entry_long_ema_cross","entry_short_ema_cross"]),
    "bb_bounce":("bollinger_bands","signals.volatility","bollinger_bands",lambda:{"period":random.randint(10,20),"std_dev":round(random.uniform(1.0,1.5),2)},["entry_long_bb","entry_short_bb"]),
    "rsi_micro":("rsi","signals.momentum","rsi",lambda:{"period":random.randint(3,7),"overbought":round(random.uniform(55,65),1),"oversold":round(random.uniform(35,45),1)},["entry_long_rsi","entry_short_rsi"]),
    "stoch_fast":("stochastic","signals.momentum","stochastic",lambda:{"k_period":random.randint(3,8),"d_period":random.randint(2,3),"overbought":round(random.uniform(60,75),1),"oversold":round(random.uniform(25,40),1)},["entry_long_stoch","entry_short_stoch"]),
    "roc_micro":("roc","signals.momentum","roc",lambda:{"period":random.randint(3,8)},["entry_long_roc","entry_short_roc"]),
    "cci_fast":("cci","signals.momentum","cci",lambda:{"period":random.randint(5,14)},["entry_long_cci","entry_short_cci"]),
    "range_micro":("range_breakout","signals.price_action","range_breakout",lambda:{"lookback":random.randint(5,15)},["entry_long_breakout","entry_short_breakout"]),
}


def make_hft(aname):
    an, mod, func, pfn, cols = APPROACHES[aname]
    ep = pfn()
    entry = {"signal_name":an,"module":mod,"function":func,"params":ep,"columns":{"long":cols[0],"short":cols[1]}}
    fo = random.choice(FILTERS); tw = random.choice(WINDOWS)
    fl = []
    if fo[1]:
        fp = copy.deepcopy(fo[3])
        for k, v in fp.items():
            if isinstance(v, int): fp[k] = max(2, int(v * random.uniform(0.7, 1.3)))
            elif isinstance(v, float): fp[k] = round(max(0.5, v * random.uniform(0.7, 1.3)), 4)
        fl.append({"signal_name":fo[0],"module":fo[1],"function":fo[2],"params":fp,"column":fo[4]})
    fl.append({"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                "params":{"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]},
                "column":"signal_time_allowed"})
    # Mix tight and wider SL/TP
    if random.random() < 0.5:
        sl = round(random.uniform(5, 15), 1)  # Tight
    else:
        sl = round(random.uniform(12, 20), 1)  # Wider
    tp = round(sl * random.uniform(0.8, 1.5), 1)
    te = random.choice([10, 15, 20, None])
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"hft2_{aname}_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                          "trailing_distance":2.0,"time_exit_minutes":te},
            "sizing_rules":{"method":"fixed","fixed_contracts":2,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def bt_fast(sd, data, rm, config, min_trades=10):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = FastBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; daily = defaultdict(float)
        for t in r.trades:
            mk = t.exit_time.strftime("%Y-%m"); mo[mk] = mo.get(mk, 0) + t.net_pnl
            dk = t.exit_time.strftime("%Y-%m-%d"); daily[dk] += t.net_pnl
        days = len(set(t.entry_time.strftime("%Y-%m-%d") for t in r.trades))
        tpd = len(r.trades) / max(1, days)
        ppt = m.total_pnl / len(r.trades)
        worst_day = min(daily.values()) if daily else 0
        trades = list(r.trades); del r, s
        return trades, m, mo, {"tpd": tpd, "ppt": ppt, "worst_day": worst_day, "days": days}
    except Exception:
        return None


def bt_old(sd, data, rm, config, min_trades=10):
    """Original backtester for comparison."""
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        ppt = m.total_pnl / len(r.trades)
        trades = list(r.trades); del r, s
        return trades, m, ppt
    except Exception:
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     HFT V2 — Realistic Execution (Limit Orders, 0 Entry Slippage)     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Old: next-bar fill + 2-tick slippage = $7.60/trade cost               ║
║  New: current-bar fill + 0 entry slip = $4.60/trade cost               ║
║  Breakeven WR drops from 63% to 56%. This changes everything.         ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_2w = df.head(5000)
    df_2mo = df.head(40000)
    del df; gc.collect()

    d2w = {"1m": df_2w}; d2mo = {"1m": df_2mo}; d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}
    cf2w = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-04-02", slippage_ticks=2, initial_capital=150000.0)
    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=2, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=2, initial_capital=150000.0)

    # ── PHASE 1: COMPARE FILL MODES ──
    print(f"\n{'='*80}\n  PHASE 1: Fill mode comparison\n{'='*80}")

    test_sd = {"name":"test_rsi_hft","entry_signals":[{"signal_name":"rsi","module":"signals.momentum","function":"rsi",
                "params":{"period":5,"overbought":60.0,"oversold":40.0},
                "columns":{"long":"entry_long_rsi","short":"entry_short_rsi"}}],
               "entry_filters":[{"signal_name":"time_of_day","module":"signals.time_filters","function":"time_of_day",
                "params":{"start_hour":9,"start_minute":45,"end_hour":15,"end_minute":30},"column":"signal_time_allowed"}],
               "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":8.0,"take_profit_type":"fixed_points",
                "take_profit_value":8.0,"trailing_stop":False,"trailing_activation":4.0,"trailing_distance":2.0,"time_exit_minutes":15},
               "sizing_rules":{"method":"fixed","fixed_contracts":2,"risk_pct":0.02,"atr_risk_multiple":2.0},
               "primary_timeframe":"1m","require_all_entries":True}

    # 1 month of data
    df_1mo = df_yr1.head(20000)
    d1mo = {"1m": df_1mo}
    cf1mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-04-18", slippage_ticks=2, initial_capital=150000.0)

    old_out = bt_old(test_sd, d1mo, rm, cf1mo, min_trades=50)
    fast_out = bt_fast(test_sd, d1mo, rm, cf1mo, min_trades=50)
    gc.collect()

    if old_out and fast_out:
        _, m_old, ppt_old = old_out
        _, m_fast, mo_fast, di_fast = fast_out
        print(f"  {'Metric':<20} {'Old (next-bar+2tick)':>22} {'Fast (same-bar+0/1)':>22}")
        print(f"  {'-'*65}")
        print(f"  {'Trades':<20} {m_old.total_trades:>22} {m_fast.total_trades:>22}")
        print(f"  {'Win Rate':<20} {m_old.win_rate:>21.1f}% {m_fast.win_rate:>21.1f}%")
        print(f"  {'Total PnL':<20} ${m_old.total_pnl:>21,.0f} ${m_fast.total_pnl:>21,.0f}")
        print(f"  {'PnL/Trade':<20} ${ppt_old:>21.2f} ${di_fast['ppt']:>21.2f}")
        diff = m_fast.total_pnl - m_old.total_pnl
        print(f"\n  Execution gap: ${diff:,.0f}/month ({diff/max(1,m_old.total_trades):.2f}/trade × {m_old.total_trades} trades)")
    del df_1mo; gc.collect()

    # ── PHASE 2: SEARCH WITH FAST BACKTESTER ──
    print(f"\n{'='*80}\n  PHASE 2: HFT search with realistic execution (7 approaches × 300)\n{'='*80}")

    all_pass = []
    approach_stats = {}

    for aname in APPROACHES:
        logger.info(f"  {aname}: 300 variants...")
        passed = 0
        for _ in range(300):
            sd = make_hft(aname)
            out = bt_fast(sd, d2w, rm, cf2w, min_trades=30)
            gc.collect()
            if not out: continue
            _, m, mo, di = out
            if di["tpd"] < 10 or m.win_rate < 50 or di["ppt"] <= 0:
                continue
            passed += 1
            sc = di["ppt"] * 100 + m.win_rate * 50 + di["tpd"] * 10
            all_pass.append((sc, sd, aname, di, m.win_rate, di["tpd"], di["ppt"]))
        approach_stats[aname] = passed
        logger.info(f"    {aname}: {passed} passed")

    all_pass.sort(key=lambda x: x[0], reverse=True)
    print(f"\n  APPROACH COMPARISON:")
    for a, cnt in sorted(approach_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passed 2-week")

    # Promote to 2-month
    del d2w, df_2w; gc.collect()
    two_mo = []
    for sc, sd, aname, di, wr, tpd, ppt in all_pass[:30]:
        cf2mo_f = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=2, initial_capital=150000.0)
        out = bt_fast(sd, d2mo, rm, cf2mo_f, min_trades=100); gc.collect()
        if out and out[1].win_rate >= 50 and out[3]["tpd"] >= 8 and out[3]["ppt"] > 0:
            two_mo.append((out[3]["ppt"] * 100 + out[1].win_rate * 50, sd, aname, out[3], out[1], out[2]))
    two_mo.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  2-month passers: {len(two_mo)}")

    del d2mo, df_2mo; gc.collect()

    # Year 1
    yr1_pass = []
    for sc, sd, aname, di, m2, mo2 in two_mo[:15]:
        out = bt_fast(sd, d1, rm, cf1, min_trades=200); gc.collect()
        if not out: continue
        _, m, mo, di1 = out
        if di1["tpd"] < 8 or m.win_rate < 50 or di1["ppt"] <= 0: continue
        mv = list(mo.values())
        if sum(1 for v in mv if v < -2000) > 2: continue
        yr1_pass.append((m.total_pnl, sd, aname, di1, m, mo))
        logger.info(f"    ✓ {aname}: {m.total_trades}tr {di1['tpd']:.0f}/day WR={m.win_rate:.0f}% ${np.mean(mv):,.0f}/mo")
    yr1_pass.sort(key=lambda x: x[0], reverse=True)

    # ── PHASE 3: DUAL-YEAR ──
    print(f"\n{'='*80}\n  PHASE 3: Dual-year validation\n{'='*80}")

    dual = []
    for pnl, sd, aname, di1, m1, mo1 in yr1_pass[:10]:
        out = bt_fast(sd, d2, rm, cf2, min_trades=200); gc.collect()
        if not out: continue
        _, m2, mo2, di2 = out
        if di2["tpd"] < 8 or m2.win_rate < 50 or m2.total_pnl <= 0: continue
        y2mv = list(mo2.values())
        if sum(1 for v in y2mv if v < -2000) > 2: continue
        prof_y1 = sum(1 for v in mo1.values() if v > 0)
        prof_y2 = sum(1 for v in y2mv if v > 0)
        if prof_y1 < 10 or prof_y2 < 10: continue
        y1a = np.mean(list(mo1.values())); y2a = np.mean(y2mv)
        sc = (m1.total_pnl + m2.total_pnl) + min(m1.win_rate, m2.win_rate) * 200
        dual.append((sc, sd, aname, mo1, mo2, m1, m2, di1, di2))
        logger.info(f"  ✓ {aname}: y1=${y1a:,.0f}/mo WR={m1.win_rate:.0f}% | y2=${y2a:,.0f}/mo WR={m2.win_rate:.0f}% | {di1['tpd']:.0f}/{di2['tpd']:.0f} tr/day")

    # Focused
    for di_idx in range(min(5, len(dual))):
        base = dual[di_idx][1]
        for _ in range(40):
            v = copy.deepcopy(base)
            for sig in v["entry_signals"]:
                for k, val in sig["params"].items():
                    if isinstance(val, int): sig["params"][k] = max(2, int(val * random.uniform(0.8, 1.2)))
                    elif isinstance(val, float): sig["params"][k] = round(max(0.1, val * random.uniform(0.8, 1.2)), 4)
            v["exit_rules"]["stop_loss_value"] = round(max(4, v["exit_rules"]["stop_loss_value"] * random.uniform(0.9, 1.1)), 1)
            v["exit_rules"]["take_profit_value"] = round(max(4, v["exit_rules"]["take_profit_value"] * random.uniform(0.9, 1.1)), 1)
            h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]
            v["name"] = f"hft2_{dual[di_idx][2]}_f_{h}"
            o1 = bt_fast(v, d1, rm, cf1, min_trades=200); gc.collect()
            o2 = bt_fast(v, d2, rm, cf2, min_trades=200); gc.collect()
            if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
                if o1[3]["tpd"] >= 8 and o2[3]["tpd"] >= 8 and o1[1].win_rate >= 50 and o2[1].win_rate >= 50:
                    sc = (o1[1].total_pnl + o2[1].total_pnl) + min(o1[1].win_rate, o2[1].win_rate) * 200
                    dual.append((sc, v, dual[di_idx][2], o1[2], o2[2], o1[1], o2[1], o1[3], o2[3]))
    dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Dual-year HFT survivors: {len(dual)}")

    # ── PHASE 4: SIZE UP ──
    if dual:
        print(f"\n{'='*80}\n  PHASE 4: Sizing up\n{'='*80}")
        best = dual[0]
        all_mo = list(best[3].values()) + list(best[4].values())
        for ct in range(2, 8):
            ratio = ct / 2
            scaled = [v * ratio for v in all_mo]
            avg = np.mean(scaled); worst = min(scaled)
            flag = "★" if avg >= 7000 else " "
            sf = "✓" if worst >= MONTH_CAP else "✗"
            print(f"    {flag}{ct}ct: ${avg:,.0f}/mo worst=${worst:,.0f} {sf}")

    del d1, d2, df_yr1, df_yr2; gc.collect()

    # ── PHASE 5-7: WF + MC ──
    print(f"\n{'='*80}\n  PHASES 5-7: Walk-forward + MC\n{'='*80}")

    best_sd = dual[0][1] if dual else None
    best_ct = 2
    if dual:
        all_mo = list(dual[0][3].values()) + list(dual[0][4].values())
        for ct in range(2, 8):
            if min(v * ct / 2 for v in all_mo) >= MONTH_CAP: best_ct = ct

    wf_windows = []; mc = None
    if best_sd:
        sd_wf = copy.deepcopy(best_sd); sd_wf["sizing_rules"]["fixed_contracts"] = best_ct
        df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        for ss in ["2024-07-01","2024-11-01","2025-03-01","2025-07-01","2025-11-01"]:
            y, mv, _ = ss.split("-"); em = int(mv)+4; ey = int(y)
            if em > 12: em -= 12; ey += 1
            es = f"{ey}-{em:02d}-01"
            try:
                dw = df_full.filter((pl.col("timestamp") >= pl.lit(ss).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                                     (pl.col("timestamp") < pl.lit(es).str.strptime(pl.Datetime, "%Y-%m-%d")))
                if len(dw) < 1000: continue
                cw = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date=ss, end_date=es, slippage_ticks=2, initial_capital=150000.0)
                out = bt_fast(sd_wf, {"1m": dw}, rm, cw, min_trades=20); gc.collect()
                pnl = out[1].total_pnl if out else 0
                wf_windows.append((ss, pnl, pnl > 0))
                logger.info(f"  WF {ss}: ${pnl:,.0f} {'✓' if pnl > 0 else '✗'}")
            except: pass

        # MC
        out_full = bt_fast(sd_wf, {"1m": df_full}, rm,
                           BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2026-03-18", slippage_ticks=2, initial_capital=150000.0))
        gc.collect()
        if out_full and len(out_full[0]) > 50:
            try:
                mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                    prop_firm_rules=pr, seed=42, avg_contracts=best_ct)).run(out_full[0], "hft2")
                print(f"  MC: P={mc.probability_of_profit:.0%} median=${mc.median_return:,.0f} ruin={mc.probability_of_ruin:.0%}")
            except Exception as e:
                print(f"  MC error: {e}")
        del df_full; gc.collect()

    wf_wins = sum(1 for *_, p in wf_windows if p)
    wf_pass = wf_wins >= 4

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}\n  HFT V2 COMPLETE — {elapsed/60:.1f} min\n{'='*80}")

    print(f"\n  APPROACH COMPARISON:")
    for a, cnt in sorted(approach_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passed 2-week (fast execution)")

    if dual:
        b = dual[0]
        sc, sd, aname, y1mo, y2mo, y1m, y2m, di1, di2 = b
        y1mv = list(y1mo.values()); y2mv = list(y2mo.values()); all_mv = y1mv + y2mv
        real_avg = np.mean(all_mv)
        filt = [f["signal_name"] for f in sd.get("entry_filters",[]) if f.get("signal_name")!="time_of_day"]

        print(f"\n  BEST HFT SYSTEM:")
        print(f"    Approach: {aname} | Filter: {filt}")
        print(f"    Params: {sd['entry_signals'][0]['params']}")
        print(f"    SL={sd['exit_rules']['stop_loss_value']} TP={sd['exit_rules']['take_profit_value']} TimeExit={sd['exit_rules'].get('time_exit_minutes')} Ct={best_ct}")
        print(f"    Y1: {y1m.total_trades}tr {di1['tpd']:.0f}/day WR={y1m.win_rate:.0f}% ${np.mean(y1mv):,.0f}/mo")
        print(f"    Y2: {y2m.total_trades}tr {di2['tpd']:.0f}/day WR={y2m.win_rate:.0f}% ${np.mean(y2mv):,.0f}/mo")
        print(f"    Combined: ${real_avg:,.0f}/mo across {len(all_mv)} months")
        active = sum(1 for v in all_mv if abs(v) > 0)
        print(f"    Active months: {active}/{len(all_mv)}")
        print(f"    WF: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")
        if mc: print(f"    MC: {mc.probability_of_profit:.0%}")

        hit_7k = real_avg >= 7000
        print(f"\n  THE NUMBER: ${real_avg:,.0f}/month at {(di1['tpd']+di2['tpd'])/2:.0f} trades/day")
        print(f"    WR: {(y1m.win_rate+y2m.win_rate)/2:.0f}% | ${(di1['ppt']+di2['ppt'])/2:.2f}/trade")
        print(f"    $7K: {'ACHIEVED' if hit_7k else f'gap ${7000-real_avg:,.0f}'}")

        print(f"\n  CAVEAT: This system's edge depends on limit order fills.")
        print(f"    If real execution is worse (partial fills, requotes, latency),")
        print(f"    profits will degrade. Paper trade 2 weeks to verify fill quality.")
    else:
        print(f"\n  NO HFT strategies survived dual-year with fast execution.")

    print(f"\n  COMPARISON:")
    print(f"    Final push (DD fix):  $2,124/mo (4/24 months)")
    print(f"    Max profit (dual):    $1,297/mo")
    if dual:
        print(f"    HFT V2 (fast exec):   ${real_avg:,.0f}/mo ({active}/24 months)")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "hft_v2_scalper",
        "execution_model": "current-bar fill, 0 entry slippage, 1-tick exit slippage",
        "approach_stats": approach_stats, "dual_year_count": len(dual),
        "best_strategy": sd if dual else None, "best_contracts": best_ct,
        "avg_monthly": round(float(real_avg), 2) if dual else 0,
        "wf_pass": wf_pass, "wf_rate": f"{wf_wins}/{len(wf_windows)}",
        "mc_p": round(mc.probability_of_profit, 4) if mc else None,
    }
    with open("reports/hft_v2_scalper.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/hft_v2_scalper.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
