#!/usr/bin/env python3
"""
PORTFOLIO BREED — Year 2 Out-of-Sample Validation

Runs the 5 Portfolio Breed components against Y2 data (2025-03-19 to 2026-03-18)
that was NEVER used during the original optimization. Reports month-by-month P&L,
Monte Carlo stress test, and Y1 vs Y2 comparison.
"""

import gc, json, time, copy, logging
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
logger = logging.getLogger("pb_y2"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
np.random.seed(42)


def bt(sd, data, rm, config, min_trades=0):
    """Run a single backtest. Returns (trades, metrics, monthly_pnl, details) or None."""
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s
            return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}
        sl_ct = tp_ct = 0
        wins = []; losses = []; longs = 0; durs = []
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
        d = {"wr": m.win_rate, "sl_pct": sl_ct/n*100 if n else 0, "tp_pct": tp_ct/n*100 if n else 0,
             "avg_win": float(np.mean(wins)) if wins else 0,
             "avg_loss": float(np.mean(losses)) if losses else 0,
             "wl_ratio": abs(float(np.mean(wins))/float(np.mean(losses))) if losses and wins else 0,
             "long_pct": longs/n*100 if n else 0, "avg_dur_min": float(np.mean(durs))/60 if durs else 0}
        trades_list = list(r.trades)
        del r, s
        return trades_list, m, mo, d
    except Exception as e:
        logger.error(f"  Backtest error: {e}")
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║   PORTFOLIO BREED — Year 2 Out-of-Sample Validation                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  5 components tested on data NEVER used for optimization               ║
║  Y2: 2025-03-19 to 2026-03-18                                         ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load portfolio configs
    with open("reports/portfolio_v1.json") as f:
        portfolio = json.load(f)
    components = portfolio["portfolios"][0]["components"]
    y1_combined = portfolio["portfolios"][0]["combined_monthly"]
    y1_stats = portfolio["portfolios"][0]["combined_stats"]

    print(f"  Loaded {len(components)} components from portfolio_v1.json")
    for c in components:
        print(f"    • {c['name']}: {c['trades']} trades, {c['win_rate']:.1f}% WR, ${c['avg_monthly']:,.0f}/mo (Y1)")

    # Setup infrastructure
    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    # Load data
    logger.info("Loading full 2yr data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()
    logger.info(f"  Year 2: {len(df_yr2):,} bars")

    data_yr2 = {"1m": df_yr2}
    cf_yr2 = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2025-03-19", end_date="2026-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: RUN EACH COMPONENT ON YEAR 2
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: Year 2 Blind Test — Individual Components")
    print(f"{'='*80}")

    labels = ["Stoch", "RSI #1", "ROC", "RSI #2", "RSI #3"]
    y2_results = []
    all_y2_trades = []

    for i, comp in enumerate(components):
        sd = comp["strategy"]
        out = bt(sd, data_yr2, rm, cf_yr2)
        gc.collect()

        if out:
            trades, m, mo, d = out
            all_y2_trades.extend(trades)
            mv = list(mo.values())
            avg_mo = np.mean(mv) if mv else 0
            worst_mo = min(mv) if mv else 0
            y2_results.append({
                "label": labels[i],
                "name": comp["name"],
                "trades": m.total_trades,
                "wr": d["wr"],
                "pf": m.profit_factor,
                "total_pnl": m.total_pnl,
                "avg_monthly": float(avg_mo),
                "worst_monthly": float(worst_mo),
                "max_dd": m.max_drawdown,
                "monthly": mo,
                "details": d,
            })
            print(f"\n  {labels[i]} ({comp['name'][:40]}):")
            print(f"    Trades: {m.total_trades}  WR: {d['wr']:.1f}%  PF: {m.profit_factor:.2f}")
            print(f"    Total PnL: ${m.total_pnl:,.0f}  Avg/Mo: ${avg_mo:,.0f}  Worst Mo: ${worst_mo:,.0f}")
            print(f"    Max DD: ${m.max_drawdown:,.0f}")
            print(f"    Monthly:")
            for k in sorted(mo):
                print(f"      {k}: ${mo[k]:>10,.1f}")
            del trades
        else:
            y2_results.append({
                "label": labels[i], "name": comp["name"],
                "trades": 0, "total_pnl": 0, "monthly": {},
            })
            print(f"\n  {labels[i]}: NO TRADES on Year 2")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: COMBINED PORTFOLIO MONTHLY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 2: Combined Portfolio — Month by Month")
    print(f"{'='*80}")

    # Gather all months
    all_months = sorted(set().union(*(r["monthly"].keys() for r in y2_results)))

    # Header
    hdr = f"  {'Month':<10}"
    for r in y2_results:
        hdr += f" {r['label']:>10}"
    hdr += f" {'COMBINED':>12}"
    print(f"\n{hdr}")
    print(f"  {'-' * (10 + 11 * len(y2_results) + 13)}")

    combined_monthly = {}
    for month in all_months:
        row = f"  {month:<10}"
        total = 0
        for r in y2_results:
            v = r["monthly"].get(month, 0)
            total += v
            row += f" ${v:>9,.0f}"
        combined_monthly[month] = total
        flag = " ✓" if total > 0 else " ✗"
        row += f" ${total:>10,.0f}{flag}"
        print(row)

    # Totals row
    row_total = f"  {'TOTAL':<10}"
    for r in y2_results:
        t = sum(r["monthly"].values())
        row_total += f" ${t:>9,.0f}"
    combined_total = sum(combined_monthly.values())
    row_total += f" ${combined_total:>10,.0f}"
    print(f"  {'-' * (10 + 11 * len(y2_results) + 13)}")
    print(row_total)

    # Avg row
    row_avg = f"  {'AVG/MO':<10}"
    for r in y2_results:
        vals = list(r["monthly"].values())
        a = np.mean(vals) if vals else 0
        row_avg += f" ${a:>9,.0f}"
    cmv = list(combined_monthly.values())
    combined_avg = np.mean(cmv) if cmv else 0
    combined_worst = min(cmv) if cmv else 0
    row_avg += f" ${combined_avg:>10,.0f}"
    print(row_avg)

    pos_months = sum(1 for v in cmv if v > 0)
    print(f"\n  Combined Y2: ${combined_avg:,.0f}/mo avg | Worst: ${combined_worst:,.0f} | {pos_months}/{len(cmv)} months positive")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: MONTE CARLO SIMULATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 3: Monte Carlo Stress Test (5,000 simulations)")
    print(f"{'='*80}")

    mc_result = None
    if len(all_y2_trades) > 10:
        all_y2_trades.sort(key=lambda t: t.exit_time)
        try:
            mc = MonteCarloSimulator(MCConfig(
                n_simulations=5000,
                initial_capital=150000.0,
                prop_firm_rules=pr,
                seed=42,
                avg_contracts=4,  # All components use 4 contracts
            ))
            mc_result = mc.run(all_y2_trades, "Portfolio_Breed_Y2")
            print(f"\n  P(Profit):      {mc_result.probability_of_profit:.1%}")
            print(f"  Median Return:  ${mc_result.median_return:,.0f}")
            print(f"  5th Percentile: ${mc_result.pct_5th_return:,.0f}")
            print(f"  95th Percentile:${mc_result.pct_95th_return:,.0f}")
            print(f"  P(Ruin):        {mc_result.probability_of_ruin:.1%}")
            print(f"  Composite:      {mc_result.composite_score:.1f}/100")
        except Exception as e:
            print(f"  MC Error: {e}")
    else:
        print(f"  Only {len(all_y2_trades)} trades — insufficient for MC simulation")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Y1 vs Y2 COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 4: Year 1 vs Year 2 Comparison")
    print(f"{'='*80}")

    y1_mv = list(y1_combined.values())
    y1_avg = np.mean(y1_mv)
    y1_worst = min(y1_mv)
    y1_pos = sum(1 for v in y1_mv if v > 0)

    print(f"\n  {'Metric':<30} {'Year 1 (IS)':>15} {'Year 2 (OOS)':>15} {'Change':>12}")
    print(f"  {'-'*72}")
    print(f"  {'Avg Monthly P&L':<30} ${y1_avg:>13,.0f} ${combined_avg:>13,.0f} {(combined_avg/y1_avg-1)*100 if y1_avg else 0:>+10.0f}%")
    print(f"  {'Worst Month':<30} ${y1_worst:>13,.0f} ${combined_worst:>13,.0f}")
    print(f"  {'Months Positive':<30} {y1_pos}/{len(y1_mv):>10} {pos_months}/{len(cmv):>12}")
    print(f"  {'Total P&L':<30} ${sum(y1_mv):>13,.0f} ${combined_total:>13,.0f}")
    print(f"  {'Total Trades (combined)':<30} {y1_stats['total_trades']:>15} {sum(r['trades'] for r in y2_results):>15}")

    # Per-component comparison
    print(f"\n  Per-Component Avg Monthly:")
    print(f"  {'Component':<12} {'Y1 Avg/Mo':>12} {'Y2 Avg/Mo':>12} {'Y2 Trades':>10} {'Verdict':>10}")
    print(f"  {'-'*56}")
    for i, comp in enumerate(components):
        y1_comp_avg = comp["avg_monthly"]
        if y2_results[i]["monthly"]:
            y2_comp_vals = list(y2_results[i]["monthly"].values())
            y2_comp_avg = np.mean(y2_comp_vals) if y2_comp_vals else 0
        else:
            y2_comp_avg = 0
        verdict = "PASS" if y2_comp_avg > 0 else "FAIL"
        color = "✓" if y2_comp_avg > 0 else "✗"
        print(f"  {labels[i]:<12} ${y1_comp_avg:>10,.0f} ${y2_comp_avg:>10,.0f} {y2_results[i]['trades']:>10} {color} {verdict}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: VERDICT & SAVE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")

    passed_components = sum(1 for r in y2_results if sum(r["monthly"].values()) > 0)
    portfolio_passed = combined_total > 0

    print(f"\n  Components passed Y2: {passed_components}/{len(y2_results)}")
    print(f"  Portfolio P&L Y2: ${combined_total:,.0f} ({'POSITIVE' if combined_total > 0 else 'NEGATIVE'})")
    print(f"  Avg Monthly Y2: ${combined_avg:,.0f} (Y1 was ${y1_avg:,.0f})")
    print(f"  All-months-positive: {'YES' if pos_months == len(cmv) else 'NO'} ({pos_months}/{len(cmv)})")
    if mc_result:
        print(f"  MC P(Profit): {mc_result.probability_of_profit:.1%} (Y1 was {portfolio['portfolios'][0]['mc_p_profit']:.1%})")
        print(f"  MC Composite: {mc_result.composite_score:.1f} (Y1 was {portfolio['portfolios'][0]['mc_composite']:.1f})")

    if portfolio_passed:
        print(f"\n  ✓ Portfolio Breed SURVIVED Year 2 OOS validation")
        if combined_avg >= y1_avg * 0.5:
            print(f"    Performance retained ≥50% of Y1 — edge appears REAL")
        else:
            print(f"    Performance degraded >50% from Y1 — edge is WEAK")
    else:
        print(f"\n  ✗ Portfolio Breed FAILED Year 2 OOS validation")
        print(f"    The Year 1 edge did NOT persist into Year 2")

    # Save results
    elapsed = time.time() - t0
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "portfolio_breed_y2_validation",
        "y2_period": "2025-03-19 to 2026-03-18",
        "y1_reference": {
            "avg_monthly": round(y1_avg, 2),
            "worst_monthly": round(y1_worst, 2),
            "months_positive": f"{y1_pos}/{len(y1_mv)}",
            "total_pnl": round(sum(y1_mv), 2),
            "mc_p_profit": portfolio["portfolios"][0]["mc_p_profit"],
        },
        "y2_results": {
            "combined_monthly": {k: round(v, 2) for k, v in sorted(combined_monthly.items())},
            "avg_monthly": round(combined_avg, 2),
            "worst_monthly": round(combined_worst, 2),
            "total_pnl": round(combined_total, 2),
            "months_positive": f"{pos_months}/{len(cmv)}",
            "total_trades": sum(r["trades"] for r in y2_results),
            "portfolio_passed": portfolio_passed,
        },
        "components": [
            {
                "label": r["label"],
                "name": r["name"],
                "trades": r["trades"],
                "wr": round(r.get("wr", 0), 2),
                "pf": round(r.get("pf", 0), 2),
                "total_pnl": round(r.get("total_pnl", 0), 2),
                "avg_monthly": round(r.get("avg_monthly", 0), 2),
                "worst_monthly": round(r.get("worst_monthly", 0), 2),
                "monthly": {k: round(v, 2) for k, v in sorted(r["monthly"].items())},
            }
            for r in y2_results
        ],
        "monte_carlo": {
            "p_profit": round(mc_result.probability_of_profit, 4) if mc_result else None,
            "median_return": round(mc_result.median_return, 2) if mc_result else None,
            "pct_5th": round(mc_result.pct_5th_return, 2) if mc_result else None,
            "pct_95th": round(mc_result.pct_95th_return, 2) if mc_result else None,
            "p_ruin": round(mc_result.probability_of_ruin, 4) if mc_result else None,
            "composite": round(mc_result.composite_score, 2) if mc_result else None,
        },
        "verdict": {
            "passed": portfolio_passed,
            "components_passed": passed_components,
            "all_months_positive": pos_months == len(cmv),
            "edge_retained": combined_avg >= y1_avg * 0.5 if portfolio_passed else False,
        },
        "elapsed_min": round(elapsed / 60, 1),
    }

    with open("reports/portfolio_breed_y2_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/portfolio_breed_y2_v1.json")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
