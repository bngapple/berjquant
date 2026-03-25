"""
Phase 2: Multi-Timeframe Validation of Final Push components.
Resample 1m data to 3m/5m/10m, backtest each component on each TF across Y1/Y2.
"""
import json, copy, gc, time, sys
import polars as pl, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.utils import (BacktestConfig, MNQ_SPEC, load_prop_firm_rules,
                           load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

# ── Config ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
CONFIG_DIR = BASE / "config"
DATA_PATH = BASE / "data" / "processed" / "MNQ" / "1m" / "full_2yr.parquet"
REPORT_PATH = BASE / "reports" / "final_push_v1.json"

PROP_PROFILE = "topstep_150k"
SLIPPAGE = 3
CAPITAL = 150_000.0

YEARS = {
    "Y1": ("2024-03-19", "2025-03-18"),
    "Y2": ("2025-03-19", "2026-03-18"),
}

TIMEFRAMES = [3, 5, 10]  # minutes


def resample(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=f"{minutes}m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )


def filter_dates(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Filter dataframe to [start, end] date range inclusive."""
    from datetime import datetime
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    return df.filter((pl.col("timestamp") >= s) & (pl.col("timestamp") <= e))


def run_one(resampled_df, name, strat, year_key, prop_rules, session_cfg, events_cal):
    start, end = YEARS[year_key]
    # Filter data to the year range
    df_filtered = filter_dates(resampled_df, start, end)

    cfg = BacktestConfig(
        symbol="MNQ",
        prop_firm_profile=PROP_PROFILE,
        start_date=start,
        end_date=end,
        slippage_ticks=SLIPPAGE,
        initial_capital=CAPITAL,
    )
    rm = RiskManager(prop_rules, session_cfg, events_cal, MNQ_SPEC)
    data = {"1m": df_filtered}
    bt = VectorizedBacktester(data, rm, MNQ_SPEC, cfg)
    strat_copy = copy.deepcopy(strat)
    res = bt.run(strat_copy)
    metrics = calculate_metrics(res.trades, CAPITAL, res.equity_curve)

    # Monthly PnL
    monthly = defaultdict(float)
    for t in res.trades:
        mo = t.exit_time.strftime("%Y-%m")
        monthly[mo] += t.net_pnl

    return {
        "trades": len(res.trades),
        "net_pnl": metrics.total_pnl,
        "win_rate": metrics.win_rate,
        "pf": metrics.profit_factor,
        "sharpe": metrics.sharpe_ratio,
        "max_dd": metrics.max_drawdown,
        "monthly": dict(monthly),
        "trade_list": res.trades,
        "equity_curve": res.equity_curve,
    }


def main():
    # ── Load data ─────────────────────────────────────────────────────
    print("Loading 1m data...")
    raw = pl.read_parquet(str(DATA_PATH))
    print(f"  Rows: {raw.shape[0]:,}  Range: {raw['timestamp'].min()} -> {raw['timestamp'].max()}")

    # ── Resample ──────────────────────────────────────────────────────
    resampled = {}
    for m in TIMEFRAMES:
        t0 = time.time()
        resampled[m] = resample(raw, m)
        print(f"  {m}m: {resampled[m].shape[0]:,} bars ({time.time()-t0:.1f}s)")

    # ── Load strategies ───────────────────────────────────────────────
    with open(REPORT_PATH) as f:
        report = json.load(f)

    components = []
    for c in report["components"]:
        strat = GeneratedStrategy.from_dict(c["strategy"])
        components.append((c["name"], strat))
    print(f"\nLoaded {len(components)} components: {[n for n,_ in components]}")

    # ── Shared resources ──────────────────────────────────────────────
    prop_rules = load_prop_firm_rules(CONFIG_DIR, PROP_PROFILE)
    session_cfg = load_session_config(CONFIG_DIR)
    events_cal = load_events_calendar(CONFIG_DIR)

    # ── Run backtests ─────────────────────────────────────────────────
    results = {}

    print("\n" + "=" * 80)
    print("RUNNING BACKTESTS")
    print("=" * 80)

    total = len(TIMEFRAMES) * len(components) * len(YEARS)
    done = 0

    for tf in TIMEFRAMES:
        for name, strat in components:
            for yr in YEARS:
                t0 = time.time()
                r = run_one(resampled[tf], name, strat, yr,
                            prop_rules, session_cfg, events_cal)
                dt = time.time() - t0
                done += 1
                results[(tf, name, yr)] = r
                print(f"  [{done:2d}/{total}] {tf:2d}m | {name[:20]:20s} | {yr} | "
                      f"{r['trades']:3d} trades | ${r['net_pnl']:>+10,.0f} | {dt:.1f}s")
        gc.collect()

    # ── Monte Carlo on Y2 results ────────────────────────────────────
    print("\n" + "=" * 80)
    print("MONTE CARLO (2000 sims) on Y2 results")
    print("=" * 80)

    mc_results = {}
    for tf in TIMEFRAMES:
        for name, strat in components:
            key = (tf, name, "Y2")
            r = results[key]
            if r["trades"] < 2:
                mc_results[(tf, name)] = {"p_profit": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}
                print(f"  {tf}m | {name[:20]:20s} | SKIP (<2 trades)")
                continue
            mc_cfg = MCConfig(
                n_simulations=2000,
                initial_capital=CAPITAL,
                avg_contracts=12,
                tick_value=MNQ_SPEC.tick_value,
                prop_firm_rules=prop_rules,
                seed=42,
            )
            sim = MonteCarloSimulator(mc_cfg)
            mc_res = sim.run(r["trade_list"], strategy_name=f"{tf}m_{name}")
            mc_results[(tf, name)] = {
                "p_profit": mc_res.probability_of_profit,
                "median": mc_res.median_return,
                "p5": mc_res.pct_5th_return,
                "p95": mc_res.pct_95th_return,
            }
            print(f"  {tf}m | {name[:20]:20s} | P(profit)={mc_res.probability_of_profit:.1%} "
                  f"| median=${mc_res.median_return:>+,.0f}")

    # ── Pass/Fail evaluation on Y2 ───────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = (f"{'TF':>4s} | {'Component':20s} | {'Yr':2s} | {'Trades':>6s} | "
              f"{'Net PnL':>10s} | {'WR%':>5s} | {'PF':>5s} | {'Sharpe':>6s} | "
              f"{'MaxDD':>8s} | {'MC P%':>5s} | {'Pass':4s}")
    print(header)
    print("-" * len(header))

    passing = []

    for tf in TIMEFRAMES:
        for name, strat in components:
            for yr in ["Y1", "Y2"]:
                r = results[(tf, name, yr)]
                mc = mc_results.get((tf, name), {})

                mc_pp = mc.get("p_profit", 0.0) if yr == "Y2" else None

                verdict = ""
                if yr == "Y2":
                    monthly = r["monthly"]
                    worst_month = min(monthly.values()) if monthly else -99999
                    n_months = len(monthly)
                    n_profitable = sum(1 for v in monthly.values() if v > 0)
                    pct_profitable = n_profitable / n_months if n_months > 0 else 0

                    pass_net = r["net_pnl"] > 0
                    pass_worst = worst_month > -5000
                    pass_pct = pct_profitable >= 0.50
                    pass_mc = mc_pp > 0.80 if mc_pp is not None else False

                    if pass_net and pass_worst and pass_pct and pass_mc:
                        verdict = "PASS"
                        passing.append((tf, name))
                    else:
                        fails = []
                        if not pass_net: fails.append("pnl")
                        if not pass_worst: fails.append(f"worst={worst_month:+,.0f}")
                        if not pass_pct: fails.append(f"mo%={pct_profitable:.0%}")
                        if not pass_mc: fails.append(f"mc={mc_pp:.0%}" if mc_pp else "mc=N/A")
                        verdict = "FAIL(" + ",".join(fails) + ")"

                mc_str = f"{mc_pp:.0%}" if mc_pp is not None else "  -  "
                print(f"{tf:>3d}m | {name[:20]:20s} | {yr:2s} | {r['trades']:6d} | "
                      f"${r['net_pnl']:>+9,.0f} | {r['win_rate']:4.1f}% | {r['pf']:5.2f} | "
                      f"{r['sharpe']:6.2f} | ${r['max_dd']:>7,.0f} | {mc_str:>5s} | {verdict}")
            # blank line between components
        print("-" * len(header))

    # ── Monthly detail for PASS combos ────────────────────────────────
    if passing:
        print(f"\n{'=' * 80}")
        print(f"MONTHLY DETAIL - PASSING COMBOS (Y2)")
        print(f"{'=' * 80}")
        for tf, name in passing:
            r = results[(tf, name, "Y2")]
            monthly = r["monthly"]
            mc = mc_results.get((tf, name), {})
            print(f"\n  {tf}m | {name}")
            print(f"  MC: P(profit)={mc.get('p_profit', 0):.1%}  median=${mc.get('median', 0):+,.0f}  "
                  f"5th=${mc.get('p5', 0):+,.0f}  95th=${mc.get('p95', 0):+,.0f}")
            for mo in sorted(monthly.keys()):
                bar = "+" * int(max(0, monthly[mo]) / 500) + "-" * int(max(0, -monthly[mo]) / 500)
                print(f"    {mo}: ${monthly[mo]:>+9,.1f}  {bar}")
            print(f"    TOTAL: ${sum(monthly.values()):>+9,.1f}")
    else:
        print("\n  ** No combos passed all Y2 criteria **")

    print(f"\nDone. {len(passing)} / {len(TIMEFRAMES) * len(components)} combos passed.")


if __name__ == "__main__":
    main()
