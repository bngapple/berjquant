#!/usr/bin/env python3
"""
EXTENDED VALIDATION — Test the $12.7K system on 2 years of UNSEEN data (2022-2024).
The system was built on 2024-2026. If it works on 2022-2024 the edge is structural.
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
logger = logging.getLogger("ext"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
MONTH_CAP = -4000.0
random_seed = 42
np.random.seed(42)


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
    except Exception as e:
        logger.warning(f"  bt error: {e}")
        return None


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     EXTENDED VALIDATION — 2022-2024 Blind Test                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  System built on 2024-2026. Testing on 2022-2024 (never seen).         ║
║  If it works: the edge is structural, not regime-specific.             ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load strategies
    with open("reports/final_push_v1.json") as f:
        fp = json.load(f)
    components = fp["components"]
    logger.info(f"Loaded {len(components)} strategies from final_push_v1.json")

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    # ── PHASE 1: DOWNLOAD INFO ──
    print(f"\n  NEW DATA:")
    df_new = pl.read_parquet("data/processed/MNQ/1m/extended_history.parquet")
    print(f"    Range: {df_new['timestamp'].min()} to {df_new['timestamp'].max()}")
    print(f"    Bars: {len(df_new):,}")
    months_in_data = df_new.select(pl.col("timestamp").dt.strftime("%Y-%m").alias("m")).unique().height
    print(f"    Months: {months_in_data}")

    data_new = {"1m": df_new}
    cf_new = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2022-03-20", end_date="2024-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    # ── PHASE 2: BLIND TEST ──
    print(f"\n{'='*80}\n  PHASE 2: Blind test on 2022-2024 data\n{'='*80}")

    combined_new = defaultdict(float)
    all_trades_new = []
    total_tr = 0

    for ci, comp in enumerate(components):
        sd = comp["strategy"]
        sd["sizing_rules"]["fixed_contracts"] = comp["contracts"]
        fam = comp["family"]
        ct = comp["contracts"]

        out = bt(sd, data_new, rm, cf_new)
        gc.collect()

        if out:
            trades, m, mo = out
            all_trades_new.extend(trades)
            total_tr += m.total_trades
            for k, v in mo.items(): combined_new[k] += v
            mv = list(mo.values())
            print(f"\n  {fam} {ct}ct:")
            print(f"    Trades: {m.total_trades} | WR: {m.win_rate:.1f}% | PF: {m.profit_factor:.2f}")
            print(f"    PnL: ${m.total_pnl:,.0f} | Avg/Mo: ${np.mean(mv):,.0f} | Worst: ${min(mv):,.0f}")
            for k in sorted(mo): print(f"      {k}: ${mo[k]:>10,.0f}")
            del trades
        else:
            print(f"\n  {fam} {ct}ct: NO TRADES on new data")

    new_mv = list(combined_new.values())
    new_avg = np.mean(new_mv) if new_mv else 0
    new_worst = min(new_mv) if new_mv else 0
    new_total = sum(new_mv)
    new_safe = new_worst >= MONTH_CAP

    print(f"\n  COMBINED on 2022-2024:")
    print(f"    Trades: {total_tr} | Avg/Mo: ${new_avg:,.0f} | Worst: ${new_worst:,.0f} | Total: ${new_total:,.0f}")
    print(f"    Safe: {'✓' if new_safe else '✗'} (worst month {'above' if new_safe else 'BELOW'} ${MONTH_CAP:,.0f})")
    for k in sorted(combined_new):
        v = combined_new[k]
        flag = "★" if v >= 7000 else ("✗" if v < MONTH_CAP else " ")
        print(f"      {flag} {k}: ${v:>10,.0f}")

    blind_pass = new_total > 0 and new_safe

    del df_new, data_new; gc.collect()

    # ── PHASE 3: COMBINE ALL DATA ──
    print(f"\n{'='*80}\n  PHASE 3: Full 4-year combined test\n{'='*80}")

    # Load both datasets
    df_old = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_hist = pl.read_parquet("data/processed/MNQ/1m/extended_history.parquet")
    df_all = pl.concat([df_hist, df_old]).sort("timestamp")
    del df_old, df_hist; gc.collect()

    print(f"  Combined dataset: {len(df_all):,} bars")
    print(f"  Range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")

    data_all = {"1m": df_all}
    cf_all = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                            start_date="2022-03-20", end_date="2026-03-18",
                            slippage_ticks=3, initial_capital=150000.0)

    combined_all = defaultdict(float)
    all_trades_full = []

    for comp in components:
        sd = comp["strategy"]
        sd["sizing_rules"]["fixed_contracts"] = comp["contracts"]

        out = bt(sd, data_all, rm, cf_all)
        gc.collect()

        if out:
            trades, m, mo = out
            all_trades_full.extend(trades)
            for k, v in mo.items(): combined_all[k] += v
            mv = list(mo.values())
            print(f"  {comp['family']} {comp['contracts']}ct: {m.total_trades}tr ${np.mean(mv):,.0f}/mo")
            del trades

    all_mv = list(combined_all.values())
    all_avg = np.mean(all_mv) if all_mv else 0
    all_worst = min(all_mv) if all_mv else 0
    all_total = sum(all_mv)
    total_months = len(all_mv)

    print(f"\n  COMBINED 4-YEAR:")
    print(f"    Trades: {len(all_trades_full)} | Months: {total_months}")
    print(f"    Avg/Mo: ${all_avg:,.0f} | Worst: ${all_worst:,.0f} | Total: ${all_total:,.0f}")

    # MC on full 4-year trade stream
    mc = None
    if len(all_trades_full) > 20:
        all_trades_full.sort(key=lambda t: t.exit_time)
        avg_ct = sum(c["contracts"] for c in components) // len(components)
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=5000, initial_capital=150000.0,
                prop_firm_rules=pr, seed=42, avg_contracts=avg_ct)).run(all_trades_full, "4yr")
            print(f"    MC P(profit): {mc.probability_of_profit:.0%}")
            print(f"    MC Median: ${mc.median_return:,.0f}")
            print(f"    MC 5th pctl: ${mc.pct_5th_return:,.0f}")
            print(f"    MC P(ruin): {mc.probability_of_ruin:.0%}")
            print(f"    MC Composite: {mc.composite_score:.1f}")
        except Exception as e:
            print(f"    MC error: {e}")

    del data_all, df_all, all_trades_full; gc.collect()

    # ── PHASE 4: OUTPUT ──
    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print(f"  EXTENDED VALIDATION COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    print(f"\n  VERDICT:")
    print(f"    The $12,743/month system {'PASSED' if blind_pass else 'FAILED'} on 2022-2024 unseen data")
    if blind_pass:
        print(f"    Making ${new_avg:,.0f}/month on the new period ({total_tr} trades)")
    else:
        if new_total <= 0:
            print(f"    Lost ${abs(new_total):,.0f} total on the new period")
        if not new_safe:
            print(f"    Worst month ${new_worst:,.0f} breached ${MONTH_CAP:,.0f} cap")

    print(f"\n    Combined across {total_months} months (4 years):")
    print(f"    ${all_avg:,.0f}/month at {mc.probability_of_profit:.0%} MC confidence" if mc else f"    ${all_avg:,.0f}/month")

    # Full monthly
    print(f"\n  FULL 4-YEAR MONTHLY:")
    for k in sorted(combined_all):
        v = combined_all[k]
        flag = "★" if v >= 7000 else ("✗" if v < MONTH_CAP else " ")
        print(f"    {flag} {k}: ${v:>10,.0f}")

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "extended_validation_v1",
        "new_data": {"start": "2022-03-20", "end": "2024-03-18", "bars": 703860, "months": months_in_data},
        "blind_test": {
            "passed": blind_pass, "total_pnl": round(new_total, 2),
            "avg_monthly": round(float(new_avg), 2), "worst_monthly": round(float(new_worst), 2),
            "trades": total_tr, "safe": new_safe,
            "monthly": {k: round(v, 2) for k, v in sorted(combined_new.items())},
        },
        "combined_4yr": {
            "avg_monthly": round(float(all_avg), 2), "worst_monthly": round(float(all_worst), 2),
            "total_pnl": round(all_total, 2), "total_months": total_months,
            "total_trades": len(all_trades_full) if 'all_trades_full' in dir() else 0,
            "mc_p_profit": round(mc.probability_of_profit, 4) if mc else None,
            "mc_median": round(mc.median_return, 2) if mc else None,
            "mc_p_ruin": round(mc.probability_of_ruin, 4) if mc else None,
            "monthly": {k: round(v, 2) for k, v in sorted(combined_all.items())},
        },
        "strategies": components,
    }
    with open("reports/extended_validation_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/extended_validation_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
