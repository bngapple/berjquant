#!/usr/bin/env python3
"""
Out-of-Sample Test — Run 39 evolved strategies on a DIFFERENT 30-day period.

Evolution was trained on: Feb 5 - Mar 18, 2026
OOS test period:          Oct 1 - Oct 31, 2025 (NQZ5 front-month)

This is the acid test: if the strategies are genuinely robust and not
curve-fit to Feb-Mar 2026, they should still produce positive MC results
on completely unseen October 2025 data.
"""

import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timezone

import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("oos")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")


def load_oct_2025_data():
    """Extract Oct 2025 NQZ5 1-minute bars from Databento raw data.

    IMPORTANT: Filter to RTH session (08:00-17:00 ET) to match training data.
    Training data (Feb 2026, EST=UTC-5) was 13:00-22:00 UTC.
    October 2025 is EDT (UTC-4), so 08:00-17:00 ET = 12:00-21:00 UTC.
    """
    logger.info("Loading Databento raw data...")
    df = pl.read_parquet("data/processed/MNQ/databento_nq_1m_raw.parquet")

    # Filter NQZ5 for October 2025
    oct = df.filter(
        (pl.col("symbol") == "NQZ5") &
        (pl.col("ts_event") >= datetime(2025, 10, 1, tzinfo=timezone.utc)) &
        (pl.col("ts_event") < datetime(2025, 11, 1, tzinfo=timezone.utc))
    )

    # Convert to Eastern time and filter to active session (08:00-17:00 ET)
    # Oct 2025 is EDT (UTC-4): 08:00 ET = 12:00 UTC, 17:00 ET = 21:00 UTC
    oct = oct.filter(
        (pl.col("ts_event").dt.hour() >= 12) &
        (pl.col("ts_event").dt.hour() < 21)
    )

    # Transform to standard format (matching all.parquet schema)
    # Strip timezone to match training data format (naive datetime)
    result = oct.select([
        pl.col("ts_event").dt.replace_time_zone(None).alias("timestamp"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume").cast(pl.Int64),
        (pl.col("volume") / 10).cast(pl.Int64).alias("tick_count"),
    ]).sort("timestamp")

    return result


def main():
    start_time = time.time()

    # ── Load OOS data ──
    df_oos = load_oct_2025_data()
    trading_days = df_oos["timestamp"].dt.date().n_unique()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          OUT-OF-SAMPLE TEST — 39 EVOLVED STRATEGIES                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Training Period:  Feb 5 – Mar 18, 2026  (what evolution saw)      ║
║  OOS Test Period:  Oct 1 – Oct 31, 2025  (completely unseen)       ║
║  Data Source:      Databento NQZ5 front-month                      ║
║  Bars:             {len(df_oos):,} 1-minute bars ({trading_days} trading days)       ║
║  Account:          Topstep $50K                                    ║
║  Slippage:         2 ticks/side ($1.00 RT adverse)                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    logger.info(f"OOS data: {len(df_oos):,} bars | {df_oos['timestamp'][0]} -> {df_oos['timestamp'][-1]}")

    # ── Load infrastructure ──
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2025-10-01", end_date="2025-10-31",
        slippage_ticks=2, initial_capital=50000.0,
    )

    data = {"1m": df_oos}

    # ── Load evolved strategies ──
    with open("reports/evolved_strategies.json") as f:
        evolved = json.load(f)
    logger.info(f"Loaded {len(evolved)} evolved strategies")

    # ── Run backtest + MC on each ──
    results = []
    mc_config = MCConfig(
        n_simulations=5000,
        initial_capital=50000.0,
        prop_firm_rules=prop_rules,
        seed=42,
    )
    simulator = MonteCarloSimulator(mc_config)

    for i, entry in enumerate(evolved, 1):
        name = entry["name"]
        strat_dict = entry["strategy"]
        strat_dict["primary_timeframe"] = "1m"

        try:
            strategy = GeneratedStrategy.from_dict(strat_dict)
            bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
            result = bt.run(strategy)

            if len(result.trades) < 3:
                logger.info(f"  [{i:2d}/39] {name[:50]} — {len(result.trades)} trades (skipped)")
                results.append({
                    "rank": i, "name": name, "status": "too_few_trades",
                    "trades": len(result.trades),
                })
                continue

            metrics = calculate_metrics(result.trades, config.initial_capital)

            # Run Monte Carlo
            mc = simulator.run(result.trades, strategy_name=name)

            results.append({
                "rank": i, "name": name, "status": "ok",
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "sharpe": metrics.sharpe_ratio,
                "total_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,
                "avg_trade": metrics.avg_trade_pnl,
                "mc_median": mc.median_return,
                "mc_mean": mc.mean_return,
                "mc_5th": mc.pct_5th_return,
                "mc_95th": mc.pct_95th_return,
                "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin,
                "mc_composite": mc.composite_score,
                "mc_pass_rate": mc.prop_firm_pass_rate,
                # Store original evolution fitness for comparison
                "evo_fitness": entry.get("fitness", 0),
                "evo_win_rate": entry.get("win_rate", 0),
                "evo_trades": entry.get("trades", 0),
            })

            status = "✓" if mc.median_return > 0 else "✗"
            logger.info(
                f"  [{i:2d}/39] {status} {name[:45]:45s} | "
                f"{metrics.total_trades:3d}t | WR={metrics.win_rate:5.1f}% | "
                f"PF={metrics.profit_factor:5.2f} | PnL=${metrics.total_pnl:>8,.0f} | "
                f"MC=${mc.median_return:>8,.0f} | P={mc.probability_of_profit:.0%}"
            )

        except Exception as e:
            logger.warning(f"  [{i:2d}/39] {name[:50]} — ERROR: {e}")
            results.append({"rank": i, "name": name, "status": "error", "error": str(e)})

    elapsed = time.time() - start_time

    # ── Results Summary ──
    ok = [r for r in results if r["status"] == "ok"]
    profitable = [r for r in ok if r["mc_median"] > 0]
    mc_strong = [r for r in ok if r["mc_p_profit"] >= 0.90]

    print(f"\n{'='*90}")
    print(f"  OUT-OF-SAMPLE RESULTS — OCTOBER 2025 ({elapsed:.0f}s)")
    print(f"{'='*90}")
    print(f"  Strategies tested:     {len(evolved)}")
    print(f"  Produced trades:       {len(ok)}")
    print(f"  MC median > $0:        {len(profitable)}")
    print(f"  MC P(profit) >= 90%:   {len(mc_strong)}")
    print()

    if ok:
        # Sort by MC median
        ok.sort(key=lambda r: r["mc_median"], reverse=True)

        print(f"  {'#':<4} {'Strategy':<45} {'Tr':>4} {'WR':>6} {'PF':>6} {'PnL':>10} {'MC Med':>10} {'MC 5th':>9} {'MC 95th':>10} {'P(prof)':>8}")
        print(f"  {'-'*112}")
        for i, r in enumerate(ok, 1):
            flag = "★" if r["mc_p_profit"] >= 0.90 else " "
            print(
                f"  {flag}{i:<3} {r['name'][:44]:<45s} "
                f"{r['trades']:>4} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
                f"${r['total_pnl']:>9,.0f} ${r['mc_median']:>9,.0f} "
                f"${r['mc_5th']:>8,.0f} ${r['mc_95th']:>9,.0f} "
                f"{r['mc_p_profit']:>7.1%}"
            )
        print(f"  {'-'*112}")

    # ── Comparison: Evolution vs OOS ──
    if profitable:
        print(f"\n  ── EVOLUTION vs OUT-OF-SAMPLE COMPARISON (top 10 by OOS MC median) ──")
        print(f"  {'Strategy':<40} {'Evo WR':>7} {'OOS WR':>7} {'OOS PnL':>10} {'OOS MC Med':>11} {'OOS P(prof)':>11}")
        print(f"  {'-'*90}")
        for r in ok[:10]:
            print(
                f"  {r['name'][:39]:<40s} "
                f"{r['evo_win_rate']:>6.1f}% {r['win_rate']:>6.1f}% "
                f"${r['total_pnl']:>9,.0f} ${r['mc_median']:>10,.0f} "
                f"{r['mc_p_profit']:>10.1%}"
            )

    # ── Verdict ──
    print(f"\n{'='*90}")
    if len(mc_strong) >= 10:
        print(f"  ✓ STRONG OOS PERFORMANCE — {len(mc_strong)}/39 strategies have P(profit) >= 90%")
        print(f"  The RSI + EMA Ribbon combo appears genuinely robust, not curve-fit.")
    elif len(profitable) >= 20:
        print(f"  ~ MODERATE OOS PERFORMANCE — {len(profitable)}/39 profitable, {len(mc_strong)} strong")
        print(f"  Strategies show some edge but less robust than in-sample.")
    elif len(profitable) >= 5:
        print(f"  ⚠ WEAK OOS PERFORMANCE — only {len(profitable)}/39 profitable")
        print(f"  Possible overfitting to Feb-Mar 2026 market conditions.")
    else:
        print(f"  ✗ POOR OOS PERFORMANCE — only {len(profitable)}/39 profitable")
        print(f"  Strong evidence of curve-fitting. Strategies do NOT generalize.")
    print(f"{'='*90}\n")

    # ── Save results ──
    output = {
        "test_type": "out_of_sample",
        "training_period": "2026-02-05 to 2026-03-18",
        "oos_period": "2025-10-01 to 2025-10-31",
        "oos_data_source": "Databento NQZ5 1m",
        "oos_bars": len(df_oos),
        "oos_trading_days": trading_days,
        "total_strategies": len(evolved),
        "strategies_with_trades": len(ok),
        "mc_profitable": len(profitable),
        "mc_strong_90pct": len(mc_strong),
        "elapsed_seconds": elapsed,
        "results": ok,
    }
    with open("reports/oos_test_oct2025.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to reports/oos_test_oct2025.json\n")


if __name__ == "__main__":
    main()
