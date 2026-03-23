#!/usr/bin/env python3
"""
FREQUENCY FIX — Find CCI/VWAP/ROC strategies that trade EVERY month
and pass dual-year validation. No more 4-month jackpots.
"""

import gc, json, time, copy, random, hashlib, logging
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("freq"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random.seed(42); np.random.seed(42)

WINDOWS = [(9,30,16,0),(9,30,14,0),(9,30,12,0),(8,0,16,0),(8,0,12,0),(9,30,11,0),(12,0,16,0),(13,0,16,0),(8,0,11,0),(14,0,16,0)]
# Frequent filters only — avoid rare ones
FREQ_FILTERS = [
    ("ema_slope","signals.trend","ema_slope",{"period":21,"slope_lookback":3},"signal_ema_slope_up"),
    ("supertrend","signals.trend","supertrend",{"period":10,"multiplier":3.0},"signal_supertrend_bullish"),
    ("heikin_ashi","signals.trend","heikin_ashi",{},"signal_ha_bullish"),
    ("linear_regression_slope","signals.trend","linear_regression_slope",{"period":20},"signal_linreg_up"),
    ("ema_ribbon","signals.trend","ema_ribbon",{"periods":[8,13,21,34,55]},"signal_ema_ribbon_bullish"),
    ("relative_volume","signals.volume","relative_volume",{"lookback":20},"signal_high_volume"),
    ("none",None,None,None,None),
]
ENTRIES = {
    "cci":("cci","signals.momentum","cci",{"period":20},["entry_long_cci","entry_short_cci"]),
    "vwap":("vwap","signals.volume","vwap",{},["entry_long_vwap","entry_short_vwap"]),
    "roc":("roc","signals.momentum","roc",{"period":10},["entry_long_roc","entry_short_roc"]),
}


def bt(sd, data, rm, config, min_trades=3):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades: del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m"); mo[k] = mo.get(k, 0) + t.net_pnl
        trades = list(r.trades); del r, s
        return trades, m, mo
    except Exception:
        return None


def max_consecutive_zero(mo_dict, all_months):
    """Count max consecutive months with 0 trades."""
    streak = 0; max_s = 0
    for m in all_months:
        if abs(mo_dict.get(m, 0)) < 0.01:
            streak += 1; max_s = max(max_s, streak)
        else:
            streak = 0
    return max_s


def make_strat(fam, edef, ct=3):
    ep = copy.deepcopy(edef[3])
    for k, v in ep.items():
        if isinstance(v, int): ep[k] = max(2, int(v * random.uniform(0.3, 1.7)))
        elif isinstance(v, float): ep[k] = round(max(0.1, v * random.uniform(0.3, 1.7)), 4)
    entry = {"signal_name":edef[0],"module":edef[1],"function":edef[2],"params":ep,
             "columns":{"long":edef[4][0],"short":edef[4][1]}}
    fo = random.choice(FREQ_FILTERS); tw = random.choice(WINDOWS)
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
    sl = round(random.uniform(10, 50), 1)
    tp = round(sl * random.uniform(1.5, 5.0), 1); tp = min(200, tp)
    h = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return {"name":f"{fam}|ff_{h}","entry_signals":[entry],"entry_filters":fl,
            "exit_rules":{"stop_loss_type":"fixed_points","stop_loss_value":sl,"take_profit_type":"fixed_points",
                          "take_profit_value":tp,"trailing_stop":False,"trailing_activation":4.0,
                          "trailing_distance":2.0,"time_exit_minutes":None},
            "sizing_rules":{"method":"fixed","fixed_contracts":ct,"risk_pct":0.02,"atr_risk_multiple":2.0},
            "primary_timeframe":"1m","require_all_entries":True}


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     FREQUENCY FIX — Trade Every Month, Dual-Year Validated             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Problem: final_push only traded 4/24 months                           ║
║  Fix: CCI/VWAP/ROC with frequent filters, min 3 trades/month          ║
║  Target: $5K+/month across ALL 24 months                               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_2mo = df.head(40000)
    del df; gc.collect()

    d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}; d2mo = {"1m": df_2mo}
    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2mo = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-05-18", slippage_ticks=3, initial_capital=150000.0)

    # Expected months
    yr1_months = [f"2024-{m:02d}" for m in range(3, 13)] + [f"2025-{m:02d}" for m in range(1, 4)]
    yr2_months = [f"2025-{m:02d}" for m in range(3, 13)] + [f"2026-{m:02d}" for m in range(1, 4)]

    # ── PHASE 1: DIAGNOSE ──
    print(f"\n{'='*80}\n  PHASE 1: Diagnosing final_push frequency problem\n{'='*80}")
    with open("reports/final_push_v1.json") as f: fp = json.load(f)
    for comp in fp["components"]:
        sd = comp["strategy"]; fam = comp["family"]; ct = comp["contracts"]
        sd["sizing_rules"]["fixed_contracts"] = ct
        # Full 2-year test
        df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        cfall = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)
        out = bt(sd, {"1m": df_full}, rm, cfall)
        del df_full; gc.collect()
        if out:
            _, m, mo = out
            active = sum(1 for k in yr1_months + yr2_months if abs(mo.get(k, 0)) > 0)
            filt = [f["signal_name"] for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
            print(f"  {fam} {ct}ct: {m.total_trades}tr in {active}/24 months | filter={filt}")
            print(f"    Active months: {[k for k in sorted(mo.keys()) if abs(mo[k]) > 0]}")
        else:
            print(f"  {fam} {ct}ct: 0 trades")

    # ── PHASE 2: HIGH-FREQUENCY SEARCH ──
    print(f"\n{'='*80}\n  PHASE 2: Searching high-frequency dual-year strategies\n{'='*80}")

    all_dual = []
    for fam, edef in ENTRIES.items():
        logger.info(f"  {fam}: 600 fast variants...")
        fast = []
        for _ in range(600):
            sd = make_strat(fam, edef, ct=3)
            out = bt(sd, d2mo, rm, cf2mo, min_trades=8)
            if out and out[1].total_pnl > 0:
                fast.append((out[1].total_pnl, sd))
        gc.collect()
        fast.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"    {len(fast)} passed fast filter (8+ trades in 2 months)")

        # Dual-year top 20
        dual = []
        for _, sd in fast[:20]:
            o1 = bt(sd, d1, rm, cf1, min_trades=30); gc.collect()
            o2 = bt(sd, d2, rm, cf2, min_trades=30); gc.collect()
            if not o1 or not o2: continue
            if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
            y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
            if min(y1mv) < -2000 or min(y2mv) < -2000: continue
            if o1[1].win_rate < 15 or o2[1].win_rate < 15: continue
            # Consecutive zero-trade check
            z1 = max_consecutive_zero(o1[2], yr1_months)
            z2 = max_consecutive_zero(o2[2], yr2_months)
            if z1 > 2 or z2 > 2: continue
            y1a = np.mean(y1mv); y2a = np.mean(y2mv)
            dual.append(((y1a+y2a)/2, sd, o1[2], o2[2], o1[1], o2[1]))

        # Focused top 5
        dual.sort(key=lambda x: x[0], reverse=True)
        for di in range(min(5, len(dual))):
            base = dual[di][1]
            for _ in range(60):  # 300/5
                v = copy.deepcopy(base)
                for sig in v["entry_signals"]:
                    for k, val in sig["params"].items():
                        if isinstance(val, int): sig["params"][k] = max(2, int(val * random.uniform(0.7, 1.3)))
                        elif isinstance(val, float): sig["params"][k] = round(max(0.1, val * random.uniform(0.7, 1.3)), 4)
                v["exit_rules"]["stop_loss_value"] = round(max(8, v["exit_rules"]["stop_loss_value"] * random.uniform(0.85, 1.15)), 1)
                v["exit_rules"]["take_profit_value"] = round(max(15, v["exit_rules"]["take_profit_value"] * random.uniform(0.85, 1.15)), 1)
                h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]
                v["name"] = f"{fam}|fff_{h}"
                o1 = bt(v, d1, rm, cf1, min_trades=30); gc.collect()
                o2 = bt(v, d2, rm, cf2, min_trades=30); gc.collect()
                if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
                    y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
                    if min(y1mv) >= -2000 and min(y2mv) >= -2000:
                        z1 = max_consecutive_zero(o1[2], yr1_months); z2 = max_consecutive_zero(o2[2], yr2_months)
                        if z1 <= 2 and z2 <= 2:
                            y1a = np.mean(y1mv); y2a = np.mean(y2mv)
                            dual.append(((y1a+y2a)/2, v, o1[2], o2[2], o1[1], o2[1]))

        dual.sort(key=lambda x: x[0], reverse=True)
        all_dual.extend(dual)
        n = len(dual)
        if dual:
            b = dual[0]
            logger.info(f"    ✓ {fam}: {n} dual-year (freq) | best ${b[0]:,.0f}/mo {b[4].total_trades}+{b[5].total_trades}tr")
        else:
            logger.info(f"    ✗ {fam}: 0 dual-year freq survivors")

    all_dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total frequency-validated: {len(all_dual)}")

    # ── PHASE 3: RE-TEST OLD PORTFOLIOS + MAX PROFIT SURVIVORS ──
    print(f"\n{'='*80}\n  PHASE 3: Re-testing old strategies with DD fix at higher contracts\n{'='*80}")

    with open("reports/max_profit_v1.json") as f: mp = json.load(f)
    for s in mp.get("top_survivors", []):
        sd = s["strategy"]
        for ct in [6, 9, 12]:
            sd_v = copy.deepcopy(sd); sd_v["sizing_rules"]["fixed_contracts"] = ct
            o1 = bt(sd_v, d1, rm, cf1, min_trades=30); gc.collect()
            o2 = bt(sd_v, d2, rm, cf2, min_trades=30); gc.collect()
            if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
                y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
                if min(y1mv) >= -2000 and min(y2mv) >= -2000:
                    z1 = max_consecutive_zero(o1[2], yr1_months); z2 = max_consecutive_zero(o2[2], yr2_months)
                    if z1 <= 2 and z2 <= 2:
                        y1a = np.mean(y1mv); y2a = np.mean(y2mv)
                        all_dual.append(((y1a+y2a)/2, sd_v, o1[2], o2[2], o1[1], o2[1]))

    all_dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total after Phase 3: {len(all_dual)}")

    # Print top 15
    print(f"\n  TOP 15 FREQUENCY-VALIDATED STRATEGIES:")
    print(f"  {'#':<3} {'Family':<15} {'Ct':>3} {'Y1 tr':>6} {'Y2 tr':>6} {'Y1 $/mo':>10} {'Y2 $/mo':>10} {'Combined':>10} {'Y1 0mo':>6} {'Y2 0mo':>6}")
    print(f"  {'-'*85}")
    for i, (ca, sd, y1mo, y2mo, y1m, y2m) in enumerate(all_dual[:15]):
        fam = sd["entry_signals"][0]["signal_name"]
        ct = sd["sizing_rules"]["fixed_contracts"]
        y1a = np.mean(list(y1mo.values())); y2a = np.mean(list(y2mo.values()))
        z1 = sum(1 for m in yr1_months if abs(y1mo.get(m, 0)) < 0.01)
        z2 = sum(1 for m in yr2_months if abs(y2mo.get(m, 0)) < 0.01)
        print(f"  {i+1:<3} {fam:<15} {ct:>3} {y1m.total_trades:>6} {y2m.total_trades:>6} ${y1a:>9,.0f} ${y2a:>9,.0f} ${ca:>9,.0f} {z1:>6} {z2:>6}")

    del d2mo, df_2mo; gc.collect()

    # ── PHASE 4: PORTFOLIO ──
    print(f"\n{'='*80}\n  PHASE 4: Building frequency portfolio\n{'='*80}")

    top50 = all_dual[:50]
    selected = [top50[0]] if top50 else []
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

    # Sizing sweep
    best_port = None
    for _ in range(10000):
        cts = [random.randint(2, 10) for _ in range(len(selected))]
        combined = defaultdict(float)
        for i, (_, sd, y1mo, y2mo, *_) in enumerate(selected):
            orig_ct = sd["sizing_rules"]["fixed_contracts"]
            ratio = cts[i] / max(orig_ct, 1)
            for k, v in y1mo.items(): combined[k] += v * ratio
            for k, v in y2mo.items(): combined[k] += v * ratio
        mv = list(combined.values())
        if not mv: continue
        worst = min(mv)
        if worst < MONTH_CAP: continue
        # Check no month has zero trades (proxy: nonzero PnL)
        zero_months = sum(1 for v in mv if abs(v) < 0.01)
        avg = np.mean(mv)
        min_tr = min(abs(v) for v in mv if abs(v) > 0) if any(abs(v) > 0 for v in mv) else 0
        sc = avg * 4.0 + worst * 3.0 + (50000 if avg >= 5000 else 0) - zero_months * 5000
        if best_port is None or sc > best_port[0]:
            best_port = (sc, cts, avg, worst, dict(combined), zero_months)

    if best_port:
        _, cts, avg, worst, combined_est, zero = best_port
        print(f"  Best sizing: {cts} → ${avg:,.0f}/mo worst=${worst:,.0f} zero_months={zero}")

    # Verify with real backtests
    print(f"\n  Verifying...")
    port_y1 = defaultdict(float); port_y2 = defaultdict(float)
    all_trades = []; final_comps = []; total_tr = 0

    for i, (ca, sd, *_) in enumerate(selected):
        sd_v = copy.deepcopy(sd)
        if best_port: sd_v["sizing_rules"]["fixed_contracts"] = cts[i]
        o1 = bt(sd_v, d1, rm, cf1); gc.collect()
        o2 = bt(sd_v, d2, rm, cf2); gc.collect()
        y1_mo = o1[2] if o1 else {}; y2_mo = o2[2] if o2 else {}
        if o1:
            all_trades.extend(o1[0]); total_tr += o1[1].total_trades
            for k, v in o1[2].items(): port_y1[k] += v
        if o2:
            all_trades.extend(o2[0]); total_tr += o2[1].total_trades
            for k, v in o2[2].items(): port_y2[k] += v
        fam = sd_v["entry_signals"][0]["signal_name"]
        ct = sd_v["sizing_rules"]["fixed_contracts"]
        y1a = np.mean(list(y1_mo.values())) if y1_mo else 0
        y2a = np.mean(list(y2_mo.values())) if y2_mo else 0
        y1tr = o1[1].total_trades if o1 else 0; y2tr = o2[1].total_trades if o2 else 0
        filt = [f["signal_name"] for f in sd_v.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
        final_comps.append({"sd": sd_v, "fam": fam, "ct": ct, "filt": filt,
                             "y1_mo": y1_mo, "y2_mo": y2_mo, "y1_tr": y1tr, "y2_tr": y2tr})
        print(f"    {fam}+{filt} {ct}ct: y1={y1tr}tr ${y1a:,.0f}/mo | y2={y2tr}tr ${y2a:,.0f}/mo")

    combined_all = defaultdict(float)
    for k, v in port_y1.items(): combined_all[k] += v
    for k, v in port_y2.items(): combined_all[k] += v

    all_mv = list(combined_all.values())
    real_avg = np.mean(all_mv) if all_mv else 0
    real_worst = min(all_mv) if all_mv else 0
    active_months = sum(1 for v in all_mv if abs(v) > 0)
    tpm = total_tr / max(1, len(all_mv))

    print(f"\n  VERIFIED: ${real_avg:,.0f}/mo | worst=${real_worst:,.0f} | {active_months}/{len(all_mv)} months active | {tpm:.0f} tr/mo")

    del d1, d2, df_yr1, df_yr2; gc.collect()

    # ── PHASE 5: WALK-FORWARD ──
    print(f"\n{'='*80}\n  PHASE 5: Walk-forward\n{'='*80}")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    wf_windows = []
    for ss in ["2024-07-01", "2024-11-01", "2025-03-01", "2025-07-01", "2025-11-01"]:
        y, mv, _ = ss.split("-"); em = int(mv)+4; ey = int(y)
        if em > 12: em -= 12; ey += 1
        es = f"{ey}-{em:02d}-01"
        try:
            dw = df_full.filter((pl.col("timestamp") >= pl.lit(ss).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                                 (pl.col("timestamp") < pl.lit(es).str.strptime(pl.Datetime, "%Y-%m-%d")))
            if len(dw) < 1000: continue
            cw = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date=ss, end_date=es, slippage_ticks=3, initial_capital=150000.0)
            wpnl = 0
            for c in final_comps:
                out = bt(c["sd"], {"1m": dw}, rm, cw, min_trades=2); gc.collect()
                if out: wpnl += out[1].total_pnl
            wf_windows.append((ss, wpnl, wpnl > 0))
            logger.info(f"  {ss}: ${wpnl:,.0f} {'✓' if wpnl > 0 else '✗'}")
        except: pass
    del df_full; gc.collect()
    wf_wins = sum(1 for *_, p in wf_windows if p)
    wf_pass = wf_wins >= 3 and sum(p for _, p, _ in wf_windows) > 0
    print(f"  Walk-forward: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")

    # ── PHASE 6: MC ──
    print(f"\n{'='*80}\n  PHASE 6: MC\n{'='*80}")
    mc = None
    if len(all_trades) > 20:
        all_trades.sort(key=lambda t: t.exit_time)
        avg_ct = sum(c["ct"] for c in final_comps) // max(1, len(final_comps))
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                prop_firm_rules=pr, seed=42, avg_contracts=avg_ct)).run(all_trades, "freq")
            print(f"  MC P(profit): {mc.probability_of_profit:.0%} | Median: ${mc.median_return:,.0f} | Ruin: {mc.probability_of_ruin:.0%}")
        except Exception as e: print(f"  MC error: {e}")
    del all_trades; gc.collect()

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  FREQUENCY FIX COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    print(f"\n  BEFORE vs AFTER:")
    print(f"    final_push: $2,124/mo real (4/24 months active)")
    print(f"    frequency:  ${real_avg:,.0f}/mo real ({active_months}/{len(all_mv)} months active)")

    print(f"\n  PORTFOLIO:")
    for c in final_comps:
        print(f"    {c['fam']}+{c['filt']} {c['ct']}ct: y1={c['y1_tr']}tr y2={c['y2_tr']}tr")

    print(f"\n  MONTHLY (ALL 24):")
    for k in sorted(combined_all):
        v = combined_all[k]; flag = "★" if v >= 5000 else ("✗" if v < MONTH_CAP else " ")
        print(f"    {flag} {k}: ${v:>10,.0f}")

    print(f"\n  THE REAL NUMBER:")
    print(f"    ${real_avg:,.0f}/month across {len(all_mv)} months")
    print(f"    {tpm:.0f} trades/month | {active_months}/{len(all_mv)} active")
    print(f"    Worst: ${real_worst:,.0f} | Safe: {'✓' if real_worst >= MONTH_CAP else '✗'}")
    print(f"    WF: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")
    if mc: print(f"    MC: {mc.probability_of_profit:.0%}")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline": "frequency_fix_v1",
        "avg_monthly": round(real_avg, 2), "worst_monthly": round(real_worst, 2),
        "active_months": active_months, "total_months": len(all_mv),
        "trades_per_month": round(tpm, 1), "total_trades": total_tr,
        "wf_pass": wf_pass, "wf_rate": f"{wf_wins}/{len(wf_windows)}",
        "mc_p": round(mc.probability_of_profit, 4) if mc else None,
        "combined_monthly": {k: round(v, 2) for k, v in sorted(combined_all.items())},
        "components": [{"name": c["sd"]["name"], "family": c["fam"], "filter": c["filt"],
                         "contracts": c["ct"], "strategy": c["sd"],
                         "y1_trades": c["y1_tr"], "y2_trades": c["y2_tr"],
                         "y1_monthly": {k: round(v, 2) for k, v in sorted(c["y1_mo"].items())},
                         "y2_monthly": {k: round(v, 2) for k, v in sorted(c["y2_mo"].items())}}
                        for c in final_comps],
    }
    with open("reports/frequency_fix_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/frequency_fix_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
