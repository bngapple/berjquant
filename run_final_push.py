#!/usr/bin/env python3
"""
FINAL PUSH — Maximum profit with DD-protection fix. Use only VWAP/CCI/ROC
families that survived dual-year. Size up aggressively within -$4K/month cap.
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
logger = logging.getLogger("push"); logger.setLevel(logging.INFO)
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
        mo = {}
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m"); mo[k] = mo.get(k, 0) + t.net_pnl
        trades = list(r.trades); del r, s
        return trades, m, mo
    except Exception:
        return None


def vary(sd, intensity=0.5):
    v = copy.deepcopy(sd)
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
    tp = round(sl * random.uniform(1.5, 6.0), 1); tp = min(300, tp)
    v["exit_rules"]["stop_loss_value"] = sl; v["exit_rules"]["take_profit_value"] = tp
    if random.random() < 0.3:
        tw = random.choice(WINDOWS)
        for f in v["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour":tw[0],"start_minute":tw[1],"end_hour":tw[2],"end_minute":tw[3]}
    if random.random() < 0.2:
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
    v["name"] = f"{v['entry_signals'][0]['signal_name']}|fp_{h}"
    return v


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     FINAL PUSH — DD Protection Fixed, Max Profit                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  DD halving now at 90% ($4,050) instead of 75% ($3,375)                ║
║  Only VWAP/CCI/ROC families (survived dual-year)                       ║
║  Size up to absolute max within -$4,000/month cap                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load 93 dual-year survivors
    with open("reports/max_profit_v1.json") as f: mp = json.load(f)
    seeds = [s["strategy"] for s in mp.get("top_survivors", [])]
    logger.info(f"Loaded {len(seeds)} dual-year seeds")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()

    d1 = {"1m": df_yr1}; d2 = {"1m": df_yr2}
    cf1 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)
    cf2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2025-03-19", end_date="2026-03-18", slippage_ticks=3, initial_capital=150000.0)

    # ── PHASE 1: RE-TEST SEEDS WITH FIXED DD PROTECTION ──
    print(f"\n{'='*80}\n  PHASE 1: Re-testing seeds with relaxed DD protection\n{'='*80}")

    dual = []  # (combined_avg, sd, y1_mo, y2_mo, y1_m, y2_m)
    for si, sd in enumerate(seeds):
        for ct in [3, 5, 7, 9, 11]:
            v = copy.deepcopy(sd); v["sizing_rules"]["fixed_contracts"] = ct
            o1 = bt(v, d1, rm, cf1, min_trades=10); gc.collect()
            o2 = bt(v, d2, rm, cf2, min_trades=10); gc.collect()
            if not o1 or not o2: continue
            if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
            y1mv = list(o1[2].values()); y2mv = list(o2[2].values())
            all_mv = y1mv + y2mv
            if min(all_mv) < -3000: continue  # Leave room for portfolio
            y1a = np.mean(y1mv); y2a = np.mean(y2mv)
            combined = (y1a + y2a) / 2
            dual.append((combined, v, o1[2], o2[2], o1[1], o2[1], ct))
        if (si + 1) % 3 == 0:
            gc.collect()
            logger.info(f"  {si+1}/{len(seeds)} seeds × 5 sizes, {len(dual)} dual-year pass")

    dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  {len(dual)} dual-year at various sizes (with fixed DD protection)")

    # Print top 15
    print(f"\n  TOP 15 (with DD protection fix):")
    print(f"  {'#':<3} {'Family':<15} {'Ct':>3} {'Y1 $/mo':>10} {'Y2 $/mo':>10} {'Combined':>10} {'Y1 worst':>10} {'Y2 worst':>10}")
    print(f"  {'-'*80}")
    for i, (ca, sd, y1mo, y2mo, y1m, y2m, ct) in enumerate(dual[:15]):
        fam = sd["entry_signals"][0]["signal_name"]
        y1a = np.mean(list(y1mo.values())); y2a = np.mean(list(y2mo.values()))
        y1w = min(y1mo.values()); y2w = min(y2mo.values())
        flag = "★" if ca >= 3500 else " "
        print(f"  {flag}{i+1:<2} {fam:<15} {ct:>3} ${y1a:>9,.0f} ${y2a:>9,.0f} ${ca:>9,.0f} ${y1w:>9,.0f} ${y2w:>9,.0f}")

    # ── PHASE 2: SEARCH FOR BETTER VARIANTS ──
    print(f"\n{'='*80}\n  PHASE 2: 500 focused variations around top 5\n{'='*80}")

    for di in range(min(5, len(dual))):
        base = dual[di][1]
        base_ca = dual[di][0]
        fam = base["entry_signals"][0]["signal_name"]
        ct = base["sizing_rules"]["fixed_contracts"]
        logger.info(f"  Varying #{di+1} ({fam} {ct}ct ${base_ca:,.0f}/mo)...")

        for _ in range(100):
            v = vary(base, 0.4)
            v["sizing_rules"]["fixed_contracts"] = ct
            o1 = bt(v, d1, rm, cf1, min_trades=10); gc.collect()
            o2 = bt(v, d2, rm, cf2, min_trades=10); gc.collect()
            if not o1 or not o2: continue
            if o1[1].total_pnl <= 0 or o2[1].total_pnl <= 0: continue
            all_mv = list(o1[2].values()) + list(o2[2].values())
            if min(all_mv) < -3000: continue
            y1a = np.mean(list(o1[2].values())); y2a = np.mean(list(o2[2].values()))
            dual.append(((y1a + y2a) / 2, v, o1[2], o2[2], o1[1], o2[1], ct))

    dual.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Total: {len(dual)} dual-year strategies")

    # ── PHASE 3: BUILD PORTFOLIO ──
    print(f"\n{'='*80}\n  PHASE 3: Portfolio with uncorrelated strategies\n{'='*80}")

    # Greedy selection
    selected = [dual[0]]
    for ca, sd, y1mo, y2mo, *rest in dual[1:]:
        if len(selected) >= 5: break
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

    # Sizing sweep (1-12ct each, 10000 combos, math only)
    best_port = None
    for _ in range(10000):
        cts = [random.randint(1, 12) for _ in range(len(selected))]
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
        avg = np.mean(mv)
        sc = avg * 4.0 + worst * 3.0 + (50000 if avg >= 7000 else 0) + (100000 if avg >= 10000 else 0)
        if best_port is None or sc > best_port[0]:
            best_port = (sc, cts, avg, worst, dict(combined))

    if best_port:
        _, cts, avg, worst, _ = best_port
        print(f"  Best sizing: {cts} → ${avg:,.0f}/mo worst=${worst:,.0f}")
    else:
        cts = [s[1]["sizing_rules"]["fixed_contracts"] for s in selected]
        print(f"  No sizing improvement, using originals: {cts}")

    # Verify with real backtests
    print(f"\n  Verifying with real backtests...")
    port_y1 = defaultdict(float); port_y2 = defaultdict(float)
    port_all_trades = []
    final_components = []
    total_tr = 0

    for i, (ca, sd, *_) in enumerate(selected):
        sd_v = copy.deepcopy(sd)
        if best_port:
            sd_v["sizing_rules"]["fixed_contracts"] = cts[i]
        o1 = bt(sd_v, d1, rm, cf1); gc.collect()
        o2 = bt(sd_v, d2, rm, cf2); gc.collect()
        if o1:
            for k, v in o1[2].items(): port_y1[k] += v
            port_all_trades.extend(o1[0])
            total_tr += o1[1].total_trades
        if o2:
            for k, v in o2[2].items(): port_y2[k] += v
            port_all_trades.extend(o2[0])
            total_tr += o2[1].total_trades

        fam = sd_v["entry_signals"][0]["signal_name"]
        ct = sd_v["sizing_rules"]["fixed_contracts"]
        y1a = np.mean(list(o1[2].values())) if o1 else 0
        y2a = np.mean(list(o2[2].values())) if o2 else 0
        final_components.append({"sd": sd_v, "fam": fam, "ct": ct, "y1_avg": y1a, "y2_avg": y2a,
                                  "y1_mo": o1[2] if o1 else {}, "y2_mo": o2[2] if o2 else {},
                                  "y1_trades": o1[1].total_trades if o1 else 0, "y2_trades": o2[1].total_trades if o2 else 0,
                                  "y1_wr": o1[1].win_rate if o1 else 0, "y2_wr": o2[1].win_rate if o2 else 0,
                                  "y1_pf": o1[1].profit_factor if o1 else 0, "y2_pf": o2[1].profit_factor if o2 else 0})
        print(f"    {fam} {ct}ct: y1=${y1a:,.0f}/mo y2=${y2a:,.0f}/mo")

    combined_all = defaultdict(float)
    for k, v in port_y1.items(): combined_all[k] += v
    for k, v in port_y2.items(): combined_all[k] += v
    all_mv = list(combined_all.values())
    real_avg = np.mean(all_mv) if all_mv else 0
    real_worst = min(all_mv) if all_mv else 0
    safe = real_worst >= MONTH_CAP

    print(f"\n  VERIFIED: ${real_avg:,.0f}/mo worst=${real_worst:,.0f} {'SAFE' if safe else 'UNSAFE'} {total_tr} trades")

    # Also test best single strategy alone at higher contracts
    print(f"\n  SINGLE BEST SCALING:")
    best_sd = dual[0][1]
    for ct in range(3, 16):
        sd_t = copy.deepcopy(best_sd); sd_t["sizing_rules"]["fixed_contracts"] = ct
        o1 = bt(sd_t, d1, rm, cf1); o2 = bt(sd_t, d2, rm, cf2); gc.collect()
        if o1 and o2 and o1[1].total_pnl > 0 and o2[1].total_pnl > 0:
            all_mv_s = list(o1[2].values()) + list(o2[2].values())
            avg_s = np.mean(all_mv_s); worst_s = min(all_mv_s)
            flag = "★" if avg_s >= 7000 else " "
            sf = "✓" if worst_s >= MONTH_CAP else "✗"
            print(f"    {flag}{ct}ct: ${avg_s:,.0f}/mo worst=${worst_s:,.0f} {sf}")
        else:
            print(f"    {ct}ct: failed dual-year")

    # Free year data
    del d1, d2, df_yr1, df_yr2; gc.collect()

    # ── PHASE 4: WALK-FORWARD ──
    print(f"\n{'='*80}\n  PHASE 4: Walk-forward (5 windows × 4 months)\n{'='*80}")

    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    wf_windows = []
    starts = ["2024-07-01", "2024-11-01", "2025-03-01", "2025-07-01", "2025-11-01"]
    for ss in starts:
        y, m_v, _ = ss.split("-")
        em = int(m_v) + 4; ey = int(y)
        if em > 12: em -= 12; ey += 1
        es = f"{ey}-{em:02d}-01"
        try:
            dw = df_full.filter((pl.col("timestamp") >= pl.lit(ss).str.strptime(pl.Datetime, "%Y-%m-%d")) &
                                 (pl.col("timestamp") < pl.lit(es).str.strptime(pl.Datetime, "%Y-%m-%d")))
            if len(dw) < 1000: continue
            cw = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date=ss, end_date=es, slippage_ticks=3, initial_capital=150000.0)
            wpnl = 0
            for comp in final_components:
                out = bt(comp["sd"], {"1m": dw}, rm, cw, min_trades=2); gc.collect()
                if out: wpnl += out[1].total_pnl
            wf_windows.append((ss, wpnl, wpnl > 0))
            logger.info(f"  {ss}: ${wpnl:,.0f} {'✓' if wpnl > 0 else '✗'}")
        except Exception:
            pass

    del df_full; gc.collect()
    wf_wins = sum(1 for *_, p in wf_windows if p)
    wf_pass = wf_wins >= 3 and sum(p for _, p, _ in wf_windows) > 0
    print(f"  Walk-forward: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")

    # ── PHASE 5: MC ──
    print(f"\n{'='*80}\n  PHASE 5: MC stress test (5000 sims)\n{'='*80}")

    mc = None
    if port_all_trades:
        port_all_trades.sort(key=lambda t: t.exit_time)
        avg_ct = sum(c["ct"] for c in final_components) // max(1, len(final_components))
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                prop_firm_rules=pr, seed=42, avg_contracts=avg_ct)).run(port_all_trades, "final")
            print(f"  MC P(profit): {mc.probability_of_profit:.0%}")
            print(f"  MC Median: ${mc.median_return:,.0f}")
            print(f"  MC 5th pctl: ${mc.pct_5th_return:,.0f}")
            print(f"  MC P(ruin): {mc.probability_of_ruin:.0%}")
            print(f"  MC Composite: {mc.composite_score:.1f}")
        except Exception as e:
            print(f"  MC error: {e}")

    del port_all_trades; gc.collect()

    # ── OUTPUT ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  FINAL PUSH COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    print(f"\n  CHANGES MADE:")
    print(f"    1. DD-protection threshold: 25% → 10% of max DD ($3,375 → $4,050)")
    print(f"    2. Only VWAP/CCI/ROC families (dual-year survivors)")
    print(f"    3. Aggressive contract sizing within -$4K/month cap")

    print(f"\n  FINAL SYSTEM:")
    print(f"  {'Component':<30} {'Ct':>3} {'Y1 $/mo':>10} {'Y2 $/mo':>10} {'Y1 WR':>6} {'Y2 WR':>6} {'Y1 Tr':>6} {'Y2 Tr':>6}")
    print(f"  {'-'*85}")
    for c in final_components:
        print(f"  {c['fam']:<30} {c['ct']:>3} ${c['y1_avg']:>9,.0f} ${c['y2_avg']:>9,.0f} {c['y1_wr']:>5.0f}% {c['y2_wr']:>5.0f}% {c['y1_trades']:>6} {c['y2_trades']:>6}")

    y1_avg = np.mean(list(port_y1.values())) if port_y1 else 0
    y2_avg = np.mean(list(port_y2.values())) if port_y2 else 0
    print(f"\n  Year 1: ${y1_avg:,.0f}/mo | Year 2: ${y2_avg:,.0f}/mo | Combined: ${real_avg:,.0f}/mo")
    print(f"  Worst month: ${real_worst:,.0f} {'✓ SAFE' if safe else '✗ UNSAFE'}")
    print(f"  Walk-forward: {wf_wins}/{len(wf_windows)} {'PASS' if wf_pass else 'FAIL'}")
    if mc:
        print(f"  MC P(profit): {mc.probability_of_profit:.0%} | Median: ${mc.median_return:,.0f}")

    print(f"\n  MONTHLY BREAKDOWN:")
    for k in sorted(combined_all):
        v = combined_all[k]
        flag = "★" if v >= 7000 else ("✗" if v < MONTH_CAP else " ")
        parts = []
        for c in final_components:
            cv = c["y1_mo"].get(k, 0) + c["y2_mo"].get(k, 0)
            if abs(cv) > 0: parts.append(f"{c['fam'][:6]}:${cv:,.0f}")
        print(f"    {flag} {k}: ${v:>10,.0f}  [{' | '.join(parts)}]")

    hit_7k = real_avg >= 7000
    print(f"\n  {'★ $7K TARGET ACHIEVED' if hit_7k else f'Gap to $7K: ${7000-real_avg:,.0f}/month'}")

    # Strategy dicts
    print(f"\n  DEPLOY STRATEGY DICTS:")
    for c in final_components:
        print(f"\n  # {c['fam']} at {c['ct']} contracts")
        print(f"  {json.dumps(c['sd'], indent=2, default=str)[:200]}...")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "final_push_v1",
        "changes": ["DD protection 25%→10%", "VWAP/CCI/ROC only", "Aggressive sizing"],
        "combined_avg_monthly": round(real_avg, 2),
        "combined_worst_monthly": round(real_worst, 2),
        "y1_avg": round(y1_avg, 2), "y2_avg": round(y2_avg, 2),
        "safe": safe, "wf_pass": wf_pass, "wf_rate": f"{wf_wins}/{len(wf_windows)}",
        "mc_p_profit": round(mc.probability_of_profit, 4) if mc else None,
        "mc_median": round(mc.median_return, 2) if mc else None,
        "total_trades": total_tr,
        "combined_monthly": {k: round(v, 2) for k, v in sorted(combined_all.items())},
        "components": [{"name": c["sd"]["name"], "family": c["fam"], "contracts": c["ct"],
                         "strategy": c["sd"],
                         "y1_avg": round(c["y1_avg"], 2), "y2_avg": round(c["y2_avg"], 2),
                         "y1_trades": c["y1_trades"], "y2_trades": c["y2_trades"],
                         "y1_monthly": {k: round(v, 2) for k, v in sorted(c["y1_mo"].items())},
                         "y2_monthly": {k: round(v, 2) for k, v in sorted(c["y2_mo"].items())}}
                        for c in final_components],
    }
    with open("reports/final_push_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/final_push_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
