#!/usr/bin/env python3
"""Run top 3 MC winners on 2 years of hourly data with full prop firm rules."""

import json
import time
import logging
from pathlib import Path

import numpy as np
import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics, print_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("top3")

CONFIG_DIR = Path("config")


def main():
    # Load 2-year hourly data
    logger.info("Loading 2-year hourly data...")
    df_1h = pl.read_parquet("data/processed/MNQ/1h/all.parquet")
    logger.info(f"  {len(df_1h)} bars | {df_1h['timestamp'][0]} -> {df_1h['timestamp'][-1]}")

    # Load prop firm config
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)

    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2023-10-01", end_date="2026-12-31",
        slippage_ticks=2, initial_capital=50000.0,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           TOP 3 STRATEGIES — 2-YEAR BACKTEST                ║
╠══════════════════════════════════════════════════════════════╣
║  Data:     {len(df_1h):,} hourly bars (Oct 2023 – Mar 2026)         ║
║  Account:  Topstep $50K prop firm evaluation                ║
║  Rules:    Daily limit $-1K | Max DD $-2K trailing          ║
║  Slippage: 2 ticks/side ($1.00 RT adverse)                  ║
║  Commission: $1.80 RT per contract                          ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Load top 3 winners
    with open("reports/mc_winners.json") as f:
        winners = json.load(f)

    # Deduplicate by base strategy name (remove mutation hashes)
    seen_bases = set()
    top3 = []
    for w in winners:
        base = w["name"].rsplit("_", 1)[0] if "_" in w["name"] and len(w["name"].rsplit("_", 1)[1]) == 6 else w["name"]
        if base not in seen_bases:
            seen_bases.add(base)
            top3.append(w)
        if len(top3) == 3:
            break

    for i, winner in enumerate(top3, 1):
        strat_dict = winner["strategy"]
        name = winner["name"]

        # Override timeframe to 1h
        strat_dict["primary_timeframe"] = "1h"
        strategy = GeneratedStrategy.from_dict(strat_dict)

        print(f"\n{'='*70}")
        print(f"  STRATEGY {i}: {name}")
        print(f"{'='*70}")
        print(f"  Entry: {', '.join(s['signal_name'] for s in strat_dict['entry_signals'])}")
        if strat_dict.get("entry_filters"):
            print(f"  Filter: {', '.join(f['signal_name'] for f in strat_dict['entry_filters'])}")
        exit_r = strat_dict["exit_rules"]
        print(f"  Exit: SL={exit_r['stop_loss_value']}pt ({exit_r['stop_loss_type']}) | TP={exit_r['take_profit_value']}pt ({exit_r['take_profit_type']})")
        sizing = strat_dict["sizing_rules"]
        print(f"  Sizing: {sizing['method']} (contracts={sizing.get('fixed_contracts', 'N/A')}, risk_pct={sizing.get('risk_pct', 'N/A')})")
        print()

        # Run backtest
        logger.info(f"Running backtest on 2 years of hourly data...")
        start = time.time()

        bt = VectorizedBacktester(
            data={"1h": df_1h},
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            config=config,
        )
        result = bt.run(strategy)
        elapsed = time.time() - start

        if len(result.trades) == 0:
            print("  NO TRADES GENERATED on hourly data.")
            print("  (Signal parameters may need re-tuning for hourly timeframe)")
            continue

        metrics = calculate_metrics(result.trades, config.initial_capital)
        result.metrics = metrics

        print(f"  ── BACKTEST RESULTS ({elapsed:.1f}s) ──")
        print(f"  Trades:        {metrics.total_trades}")
        print(f"  Win Rate:      {metrics.win_rate:.1f}%")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:.2f}")
        print(f"  Net P&L:       ${metrics.total_pnl:,.2f}")
        print(f"  Max Drawdown:  ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1f}%)")
        print(f"  Avg Trade:     ${metrics.avg_trade_pnl:,.2f}")
        print(f"  Avg Winner:    ${metrics.avg_winner:,.2f}")
        print(f"  Avg Loser:     ${metrics.avg_loser:,.2f}")
        print(f"  Largest Win:   ${metrics.largest_winner:,.2f}")
        print(f"  Largest Loss:  ${metrics.largest_loser:,.2f}")

        # Check prop firm compliance
        final_balance = config.initial_capital + metrics.total_pnl
        dd_ok = abs(metrics.max_drawdown) <= abs(prop_rules.max_drawdown)
        print(f"\n  ── PROP FIRM CHECK ──")
        print(f"  Final Balance: ${final_balance:,.2f}")
        print(f"  Max DD vs Limit: ${metrics.max_drawdown:,.2f} vs ${prop_rules.max_drawdown:,.2f} {'✓ PASS' if dd_ok else '✗ FAIL'}")

        # Monte Carlo on the 2-year trades
        logger.info(f"Running Monte Carlo (3000 sims) on 2-year trades...")
        mc_config = MCConfig(
            n_simulations=3000,
            initial_capital=50000.0,
            prop_firm_rules=prop_rules,
            seed=42,
        )
        sim = MonteCarloSimulator(mc_config)
        mc = sim.run(result.trades, strategy_name=name)

        print(f"\n  ── MONTE CARLO (3,000 simulations) ──")
        print(f"  Median Return:     ${mc.median_return:,.2f}")
        print(f"  Mean Return:       ${mc.mean_return:,.2f}")
        print(f"  5th Percentile:    ${mc.pct_5th_return:,.2f}")
        print(f"  95th Percentile:   ${mc.pct_95th_return:,.2f}")
        print(f"  P(profit):         {mc.probability_of_profit:.1%}")
        print(f"  P(ruin):           {mc.probability_of_ruin:.1%}")
        print(f"  Pass Rate:         {mc.prop_firm_pass_rate:.1%}")
        print(f"  Composite Score:   {mc.composite_score:.1f}/100")

        # Monthly P&L breakdown
        daily_pnl = {}
        for t in result.trades:
            month = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:7]
            daily_pnl[month] = daily_pnl.get(month, 0) + t.net_pnl

        if daily_pnl:
            print(f"\n  ── MONTHLY P&L ──")
            for month in sorted(daily_pnl.keys()):
                pnl = daily_pnl[month]
                bar = "█" * max(1, int(abs(pnl) / 200))
                sign = "+" if pnl >= 0 else "-"
                color = "" if pnl >= 0 else "*"
                print(f"    {month}:  {sign}${abs(pnl):>8,.2f}  {bar}{color}")

    print(f"\n{'='*70}")
    print(f"  ALL BACKTESTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
