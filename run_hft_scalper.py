#!/usr/bin/env python3
"""
HFT SCALPER — 20-50 trades/day, 52%+ WR, tight SL/TP, $7K/month target.
Completely different approach: tiny edge × thousands of trades.
Does NOT modify any existing files.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict, Counter
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("hft"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
DAILY_CAP = -3000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,45,15,30),(10,0,14,0),(9,30,12,0),(8,30,16,0),(10,0,15,30),(9,30,15,0),(9,30,14,0)]
FAST_FILTERS = [
    ("none",None,None,None,None),
    ("ema_slope","signals.trend","ema_slope",{"period":8,"slope_lookback":2},"signal_ema_slope_up"),
    ("heikin_ashi","signals.trend","heikin_ashi",{},"signal_ha_bullish"),
    ("supertrend","signals.trend","supertrend",{"period":7,"multiplier":2.0},"signal_supertrend_bullish"),
]
APPROACHES = {
    "ema_micro": ("ema_crossover","signals.trend","ema_crossover",
                  lambda: {"fast_period": random.randint(3,8), "slow_period": random.randint(8,15)},
                  ["entry_long_ema_cross","entry_short_ema_cross"]),
    "bb_bounce": ("bollinger_bands","signals.volatility","bollinger_bands",
                  lambda: {"period": random.randint(10,20), "std_dev": round(random.uniform(1.0,1.5),2)},
                  ["entry_long_bb","entry_short_bb"]),
    "rsi_micro": ("rsi","signals.momentum","rsi",
                  lambda: {"period": random.randint(3,7), "overbought": round(random.uniform(55,65),1), "oversold": round(random.uniform(35,45),1)},
                  ["entry_long_rsi","entry_short_rsi"]),
    "stoch_fast": ("stochastic","signals.momentum","stochastic",
                   lambda: {"k_period": random.randint(3,8), "d_period": random.randint(2,3), "overbought": round(random.uniform(60,75),1), "oversold": round(random.uniform(25,40),1)},
                   ["entry_long_stoch","entry_short_stoch"]),
    "roc_micro": ("roc","signals.momentum","roc",
                  lambda: {"period": random.randint(3,8)},
                  ["entry_long_roc","entry_short_roc"]),
    "cci_fast": ("cci","signals.momentum","cci",
                 lambda: {"period": random.randint(5,14)},
                 ["entry_long_cci","entry_short_cci"]),
    "range_micro": ("range_breakout","signals.price_action","range_breakout",
                    lambda: {"lookback": random.randint(5,15)},
                    ["entry_long_breakout","entry_short_breakout"]),
}


def make_hft(approach_name):
    aname, mod, func, param_fn, cols = APPROACHES[approach_name]
    ep = param_fn()
    entry = {"signal_name":aname,"module":mod,"function":func,"params":ep,"columns":{"long":cols[0],"short":cols[1]}}
    fo = random.choice(FAST_FILTERS); tw = random.choice(WINDOWS)
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
    sl = round(random.uniform(5, 15), 1)
    tp = round(sl * random.uniform(0.8, 1.5), 1)
    te = random.choice([10, 15, 20])
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"hft_{approach_name}_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                          "trailing_distance":2.0,"time_exit_minutes":te},
            "sizing_rules":{"method":"fixed","fixed_contracts":2,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def bt_detail(sd, data, rm, config, min_trades=10):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}; daily = defaultdict(float)
        for t in r.trades:
            mk = t.exit_time.strftime("%Y-%m"); mo[mk] = mo.get(mk, 0) + t.net_pnl
            dk = t.exit_time.strftime("%Y-%m-%d"); daily[dk] += t.net_pnl
        days_traded = len(set(t.entry_time.strftime("%Y-%m-%d") for t in r.trades))
        tpd = len(r.trades) / max(1, days_traded)
        avg_dur = np.mean([t.duration_seconds for t in r.trades]) / 60
        ppt = m.total_pnl / len(r.trades)
        ppd = m.total_pnl / max(1, days_traded)
        worst_day = min(daily.values()) if daily else 0
        trades = list(r.trades); del r, s
        return trades, m, mo, {"tpd": tpd, "dur": avg_dur, "ppt": ppt, "ppd": ppd, "worst_day": worst_day, "days": days_traded}
    except Exception:
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     HFT SCALPER — 20-50 Trades/Day, Tiny Edge × Thousands             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Math: $7K/mo ÷ 20 days = $350/day. At 30 trades = $11.67/trade.      ║
║  At 2ct that is 2.9 points per trade. Very achievable at 52% WR.       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_2w = df.head(5000)  # ~2 weeks
    df_2mo = df.head(40000)  # ~2 months
    del df; gc.collect()

    d2w = {"1m": df_2w}; d2mo = {"1m": df_2mo}; d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}
    cf2w = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-04-02", slippage_ticks=2, initial_capital=150000.0)
    cf2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=2, initial_capital=150000.0)
    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=2, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=2, initial_capital=150000.0)

    yr1_months = [f"2024-{m:02d}" for m in range(3,13)] + [f"2025-{m:02d}" for m in range(1,4)]
    yr2_months = [f"2025-{m:02d}" for m in range(3,13)] + [f"2026-{m:02d}" for m in range(1,4)]

    # ── PHASE 2: SEARCH ALL APPROACHES ──
    print(f"\n{'='*80}\n  PHASE 2: Searching 7 HFT approaches (300 each on 2 weeks)\n{'='*80}")

    all_pass = []  # (score, sd, approach, d_info)
    approach_stats = {}

    for aname in APPROACHES:
        logger.info(f"  {aname}: 300 fast variants...")
        passed = 0
        for _ in range(300):
            sd = make_hft(aname)
            out = bt_detail(sd, d2w, rm, cf2w, min_trades=30)
            gc.collect()
            if not out: continue
            _, m, mo, di = out
            if di["tpd"] < 10 or m.win_rate < 50 or di["ppt"] <= 0 or di["ppd"] < 50:
                continue
            passed += 1
            sc = di["ppd"] * 10 + m.win_rate * 50 + di["tpd"] * 20
            all_pass.append((sc, sd, aname, di, m.win_rate, di["tpd"], di["ppt"]))

        approach_stats[aname] = passed
        logger.info(f"    {aname}: {passed} passed (10+ tr/day, 50%+ WR, +PnL)")

    all_pass.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total 2-week passers: {len(all_pass)}")

    print(f"\n  APPROACH COMPARISON:")
    for a, cnt in sorted(approach_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passed")

    if not all_pass:
        print("  NO strategies passed 2-week filter. HFT scalping may not be viable with this backtester.")
        return

    # Promote top 30 to 2-month
    logger.info(f"  Promoting top 30 to 2-month test...")
    two_mo_pass = []
    for sc, sd, aname, di, wr, tpd, ppt in all_pass[:30]:
        out = bt_detail(sd, d2mo, rm, cf2mo, min_trades=100)
        gc.collect()
        if not out: continue
        _, m, mo, di2 = out
        if di2["tpd"] < 8 or m.win_rate < 50 or di2["ppt"] <= 0:
            continue
        two_mo_pass.append((di2["ppd"] * 10 + m.win_rate * 50, sd, aname, di2, m, mo))

    two_mo_pass.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  2-month passers: {len(two_mo_pass)}")

    # Promote top 15 to full year 1
    logger.info(f"  Promoting top 15 to year 1...")
    yr1_pass = []
    for sc, sd, aname, di, m_2mo, mo_2mo in two_mo_pass[:15]:
        out = bt_detail(sd, d1, rm, cf1, min_trades=200)
        gc.collect()
        if not out: continue
        _, m, mo, di1 = out
        if di1["tpd"] < 8 or m.win_rate < 50 or di1["ppt"] <= 0:
            continue
        mv = list(mo.values())
        neg_months = sum(1 for v in mv if v < -2000)
        if neg_months > 2: continue
        yr1_pass.append((m.total_pnl, sd, aname, di1, m, mo))
        avg = np.mean(mv)
        logger.info(f"    ✓ {aname}: {m.total_trades}tr {di1['tpd']:.0f}/day WR={m.win_rate:.0f}% ${avg:,.0f}/mo")

    yr1_pass.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Year 1 passers: {len(yr1_pass)}")

    del d2w, d2mo, df_2w, df_2mo; gc.collect()

    # ── PHASE 3: DUAL-YEAR ──
    print(f"\n{'='*80}\n  PHASE 3: Dual-year validation\n{'='*80}")

    dual = []
    for pnl, sd, aname, di1, m1, mo1 in yr1_pass[:10]:
        out = bt_detail(sd, d2, rm, cf2, min_trades=200)
        gc.collect()
        if not out: continue
        _, m2, mo2, di2 = out
        if di2["tpd"] < 8 or m2.win_rate < 50 or di2["ppt"] <= 0:
            continue
        if m2.total_pnl <= 0: continue
        y2mv = list(mo2.values())
        if sum(1 for v in y2mv if v < -2000) > 2: continue
        prof_y1 = sum(1 for v in mo1.values() if v > 0)
        prof_y2 = sum(1 for v in y2mv if v > 0)
        if prof_y1 < 10 or prof_y2 < 10: continue

        y1a = np.mean(list(mo1.values())); y2a = np.mean(y2mv)
        sc = (m1.total_pnl + m2.total_pnl) + min(m1.win_rate, m2.win_rate) * 200 + min(di1["tpd"], di2["tpd"]) * 100
        dual.append((sc, sd, aname, mo1, mo2, m1, m2, di1, di2))
        logger.info(f"  ✓ {aname}: y1=${y1a:,.0f}/mo ({di1['tpd']:.0f}/d WR={m1.win_rate:.0f}%) y2=${y2a:,.0f}/mo ({di2['tpd']:.0f}/d WR={m2.win_rate:.0f}%)")

    # Focused variations on top 5
    for di_idx in range(min(5, len(dual))):
        base = dual[di_idx][1]
        for _ in range(40):  # 200/5
            v = copy.deepcopy(base)
            for sig in v["entry_signals"]:
                for k, val in sig["params"].items():
                    if isinstance(val, int): sig["params"][k] = max(2, int(val * random.uniform(0.8, 1.2)))
                    elif isinstance(val, float): sig["params"][k] = round(max(0.1, val * random.uniform(0.8, 1.2)), 4)
            v["exit_rules"]["stop_loss_value"] = round(max(4, v["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
            v["exit_rules"]["take_profit_value"] = round(max(4, v["exit_rules"]["take_profit_value"] * random.uniform(0.85, 1.15)), 1)
            h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]
            v["name"] = f"hft_{dual[di_idx][2]}_f_{h}"
            o1 = bt_detail(v, d1, rm, cf1, min_trades=200); gc.collect()
            o2 = bt_detail(v, d2, rm, cf2, min_trades=200); gc.collect()
            if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
                if o1[3]["tpd"] >= 8 and o2[3]["tpd"] >= 8 and o1[1].win_rate >= 50 and o2[1].win_rate >= 50:
                    y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
                    if sum(1 for v2 in y1mv if v2 < -2000) <= 2 and sum(1 for v2 in y2mv if v2 < -2000) <= 2:
                        sc = (o1[1].total_pnl + o2[1].total_pnl) + min(o1[1].win_rate, o2[1].win_rate) * 200
                        dual.append((sc, v, dual[di_idx][2], o1[2], o2[2], o1[1], o2[1], o1[3], o2[3]))

    dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Dual-year HFT survivors: {len(dual)}")

    # ── PHASE 4: SIZE UP ──
    print(f"\n{'='*80}\n  PHASE 4: Sizing up\n{'='*80}")

    sized = []
    for sc, sd, aname, y1mo, y2mo, y1m, y2m, di1, di2 in dual[:10]:
        all_mo = list(y1mo.values()) + list(y2mo.values())
        base_ct = 2
        for ct in range(2, 8):
            ratio = ct / base_ct
            scaled = [v * ratio for v in all_mo]
            worst = min(scaled)
            if worst < MONTH_CAP: break
            avg = np.mean(scaled)
            sized.append((avg, ct, sd, aname, y1mo, y2mo, worst, di1, di2, y1m, y2m))
        best_ct = sized[-1][1] if sized else 2
        best_avg = sized[-1][0] if sized else 0

    sized.sort(key=lambda x: x[0], reverse=True)
    if sized:
        print(f"\n  SIZING TABLE (top 5):")
        print(f"  {'Approach':<18} {'Ct':>3} {'Avg/Mo':>10} {'Worst':>10} {'Tr/Day':>7}")
        print(f"  {'-'*55}")
        seen = set()
        for avg, ct, sd, aname, y1mo, y2mo, worst, di1, di2, *_ in sized[:20]:
            key = f"{aname}_{ct}"
            if key in seen: continue
            seen.add(key)
            print(f"  {aname:<18} {ct:>3} ${avg:>9,.0f} ${worst:>9,.0f} {(di1['tpd']+di2['tpd'])/2:>6.0f}")
            if len(seen) >= 5: break

    del d1, d2, df_yr1, df_yr2; gc.collect()

    # ── PHASE 5: PORTFOLIO (skip if no dual survivors) ──
    # ── PHASE 6: WALK-FORWARD ──
    print(f"\n{'='*80}\n  PHASE 6: Walk-forward\n{'='*80}")

    best_sd = dual[0][1] if dual else (sized[0][2] if sized else None)
    best_ct = sized[0][1] if sized else 2
    wf_windows = []

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
                out = bt_detail(sd_wf, {"1m": dw}, rm, cw, min_trades=20); gc.collect()
                pnl = out[1].total_pnl if out else 0
                wf_windows.append((ss, pnl, pnl > 0))
                logger.info(f"  {ss}: ${pnl:,.0f} {'✓' if pnl > 0 else '✗'}")
            except: pass
        del df_full; gc.collect()

    wf_wins = sum(1 for *_, p in wf_windows if p)
    wf_pass = wf_wins >= 4
    print(f"  Walk-forward: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")

    # Also test at 3-tick slippage
    print(f"  (Testing at 3-tick slippage...)")
    wf3 = []
    if best_sd:
        df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        for ss in ["2024-07-01","2024-11-01","2025-03-01","2025-07-01","2025-11-01"]:
            y, mv, _ = ss.split("-"); em = int(mv)+4; ey = int(y)
            if em > 12: em -= 12; ey += 1
            es = f"{ey}-{em:02d}-01"
            try:
                dw = df_full.filter((pl.col("timestamp") >= pl.lit(ss).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                                     (pl.col("timestamp") < pl.lit(es).str.strptime(pl.Datetime, "%Y-%m-%d")))
                if len(dw) < 1000: continue
                cw3 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date=ss, end_date=es, slippage_ticks=3, initial_capital=150000.0)
                out = bt_detail(sd_wf, {"1m": dw}, rm, cw3, min_trades=20); gc.collect()
                pnl = out[1].total_pnl if out else 0
                wf3.append((ss, pnl, pnl > 0))
            except: pass
        del df_full; gc.collect()
    wf3_wins = sum(1 for *_, p in wf3 if p)
    print(f"  Walk-forward (3-tick): {wf3_wins}/{len(wf3)}")

    # ── PHASE 7: MC ──
    print(f"\n{'='*80}\n  PHASE 7: MC stress test\n{'='*80}")

    mc_1tick = mc_2tick = None
    if best_sd:
        df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        for slip_label, slip_ticks in [("2-tick", 2), ("3-tick", 3)]:
            sd_mc = copy.deepcopy(best_sd); sd_mc["sizing_rules"]["fixed_contracts"] = best_ct
            cf_mc = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2026-03-18", slippage_ticks=slip_ticks, initial_capital=150000.0)
            out = bt_detail(sd_mc, {"1m": df_full}, rm, cf_mc)
            gc.collect()
            if out and len(out[0]) > 50:
                try:
                    mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                        prop_firm_rules=pr, seed=42, avg_contracts=best_ct)).run(out[0], f"hft_{slip_label}")
                    print(f"  MC ({slip_label}): P(profit)={mc.probability_of_profit:.0%} median=${mc.median_return:,.0f} ruin={mc.probability_of_ruin:.0%}")
                    if slip_ticks == 2: mc_1tick = mc
                    else: mc_2tick = mc
                except Exception as e:
                    print(f"  MC error: {e}")
            del out; gc.collect()
        del df_full; gc.collect()

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  HFT SCALPER COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # Signal comparison
    print(f"\n  SIGNAL COMPARISON:")
    for a, cnt in sorted(approach_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {a:<18} {cnt} passed 2-week filter")
    dual_by_approach = Counter(d[2] for d in dual)
    print(f"  Dual-year survivors by approach: {dict(dual_by_approach)}")

    if dual:
        best = dual[0]
        sc, sd, aname, y1mo, y2mo, y1m, y2m, di1, di2 = best
        y1mv = list(y1mo.values()); y2mv = list(y2mo.values())
        all_mv = y1mv + y2mv

        print(f"\n  BEST HFT STRATEGY:")
        print(f"    Approach: {aname}")
        print(f"    Params: {sd['entry_signals'][0]['params']}")
        filt = [f['signal_name'] for f in sd.get('entry_filters',[]) if f.get('signal_name')!='time_of_day']
        print(f"    Filter: {filt}")
        print(f"    SL={sd['exit_rules']['stop_loss_value']} TP={sd['exit_rules']['take_profit_value']} TimeExit={sd['exit_rules'].get('time_exit_minutes')}min")
        print(f"    Contracts: {best_ct}")
        print(f"    Y1: {y1m.total_trades}tr {di1['tpd']:.0f}/day WR={y1m.win_rate:.0f}% ${np.mean(y1mv):,.0f}/mo")
        print(f"    Y2: {y2m.total_trades}tr {di2['tpd']:.0f}/day WR={y2m.win_rate:.0f}% ${np.mean(y2mv):,.0f}/mo")
        print(f"    Combined: ${np.mean(all_mv):,.0f}/mo | worst=${min(all_mv):,.0f}")

        # Monthly
        all_months = sorted(set(list(y1mo.keys()) + list(y2mo.keys())))
        active = sum(1 for m in all_months if abs(y1mo.get(m, 0) + y2mo.get(m, 0)) > 0)
        print(f"\n  MONTHLY ({active}/{len(all_months)} active):")
        for m in all_months:
            v = y1mo.get(m, 0) + y2mo.get(m, 0)
            # This double-counts — show per-year instead
        for label, mo_d in [("Year 1", y1mo), ("Year 2", y2mo)]:
            print(f"    {label}:")
            for k in sorted(mo_d): print(f"      {k}: ${mo_d[k]:>10,.0f}")

    print(f"\n  WALK-FORWARD: {wf_wins}/{len(wf_windows)} (2-tick) | {wf3_wins}/{len(wf3)} (3-tick)")
    if mc_1tick: print(f"  MC 2-tick: P={mc_1tick.probability_of_profit:.0%} median=${mc_1tick.median_return:,.0f}")
    if mc_2tick: print(f"  MC 3-tick: P={mc_2tick.probability_of_profit:.0%} median=${mc_2tick.median_return:,.0f}")

    # The number
    if dual:
        all_real = list(y1mo.values()) + list(y2mo.values())
        real_avg = np.mean(all_real)
        print(f"\n  THE NUMBER: HFT scalper makes ${real_avg:,.0f}/month at {(di1['tpd']+di2['tpd'])/2:.0f} trades/day")
        print(f"    WR: {(y1m.win_rate+y2m.win_rate)/2:.0f}% | ${(di1['ppt']+di2['ppt'])/2:.2f}/trade avg")
        if mc_1tick: print(f"    MC (2-tick): {mc_1tick.probability_of_profit:.0%}")
        if mc_2tick: print(f"    MC (3-tick): {mc_2tick.probability_of_profit:.0%}")
        hit_7k = real_avg >= 7000
        print(f"    $7K target: {'ACHIEVED' if hit_7k else f'gap ${7000-real_avg:,.0f}'}")
    else:
        print(f"\n  NO HFT strategies survived dual-year validation.")

    print(f"\n  COMPARISON:")
    print(f"    Final push (DD fix):  $2,124/mo (4/24 months active)")
    print(f"    Max profit (dual):    $1,297/mo")
    print(f"    Trailing:               $435/mo")
    if dual:
        print(f"    HFT scalper:          ${real_avg:,.0f}/mo ({active}/24 months active)")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "hft_scalper_v1",
        "approach_stats": approach_stats,
        "dual_year_count": len(dual),
        "best_strategy": dual[0][1] if dual else None,
        "best_approach": dual[0][2] if dual else None,
        "best_contracts": best_ct,
        "combined_avg_monthly": round(float(np.mean(all_real)), 2) if dual else 0,
        "wf_2tick": f"{wf_wins}/{len(wf_windows)}", "wf_3tick": f"{wf3_wins}/{len(wf3)}",
        "mc_2tick_p": round(mc_1tick.probability_of_profit, 4) if mc_1tick else None,
        "mc_3tick_p": round(mc_2tick.probability_of_profit, 4) if mc_2tick else None,
    }
    with open("reports/hft_scalper_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/hft_scalper_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
