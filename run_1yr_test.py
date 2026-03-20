#!/usr/bin/env python3
"""
1-Year MC Validation — Test top strategies from both Oct 2025 and Feb-Mar 2026
on a full year of data to find the single best strategy.

Data: Mar 2025 – Mar 2026 (1 year from full_2yr.parquet)
Candidates:
  - Top 25 from Feb-Mar 2026 evolution (RSI+EMA Ribbon variants)
  - Top Oct 2025 signal combos reconstructed with default+best exit params
"""

import json
import time
import random
import logging
from pathlib import Path

import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry
from strategies.generator import GeneratedStrategy, ExitRules, SizingRules
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("1yr")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")


def build_strategy(name, entries, filters, sl, tp, sl_type="fixed_points", tp_type="fixed_points",
                   sizing_method="risk_pct", risk_pct=0.015, fixed_contracts=2):
    """Build a strategy from signal names and exit params."""
    registry = SignalRegistry()
    entry_dicts = []
    for sig_name in entries:
        sig = next((s for s in registry.list_entry_signals() if s.name == sig_name), None)
        if not sig:
            return None
        entry_dicts.append({
            "signal_name": sig.name, "module": sig.module, "function": sig.function,
            "params": {k: v["default"] for k, v in sig.parameters.items()},
            "columns": {
                "long": next((c for c in sig.entry_columns if "long" in c), ""),
                "short": next((c for c in sig.entry_columns if "short" in c), ""),
            },
        })

    filt_dicts = []
    for f_name in filters:
        filt = next((f for f in registry.list_filters() if f.name == f_name), None)
        if not filt:
            return None
        filt_dicts.append({
            "signal_name": filt.name, "module": filt.module, "function": filt.function,
            "params": {k: v["default"] for k, v in filt.parameters.items()},
            "column": filt.filter_columns[0] if filt.filter_columns else "",
        })

    return GeneratedStrategy(
        name=name,
        entry_signals=entry_dicts,
        entry_filters=filt_dicts,
        exit_rules=ExitRules(
            stop_loss_type=sl_type, stop_loss_value=float(sl),
            take_profit_type=tp_type, take_profit_value=float(tp),
        ),
        sizing_rules=SizingRules(method=sizing_method, risk_pct=risk_pct, fixed_contracts=fixed_contracts),
    )


def generate_oct_winners():
    """Reconstruct the top Oct 2025 signal combos with their best exit params."""
    strats = []

    # Top performers from Oct search (signal combo + exits from names)
    combos = [
        # (name, entries, filters, sl, tp, sizing_method, risk_pct, fixed_contracts)
        ("CCI|FOOTPRINT_IMBALANCE|SL30_TP180", ["cci"], ["footprint_imbalance"], 30, 180, "risk_pct", 0.02, 2),
        ("CCI|FOOTPRINT_IMBALANCE|SL25_TP75", ["cci"], ["footprint_imbalance"], 25, 75, "risk_pct", 0.015, 2),
        ("CCI|FOOTPRINT_IMBALANCE|SL15_TP90", ["cci"], ["footprint_imbalance"], 15, 90, "risk_pct", 0.015, 2),
        ("ROC|LINEAR_REG_SLOPE|SL30_TP180", ["roc"], ["linear_regression_slope"], 30, 180, "risk_pct", 0.02, 2),
        ("ROC|LINEAR_REG_SLOPE|SL30_TP150", ["roc"], ["linear_regression_slope"], 30, 150, "risk_pct", 0.015, 2),
        ("ROC|VOLUME_PROFILE|SL25_TP150", ["roc"], ["volume_profile"], 25, 150, "risk_pct", 0.015, 2),
        ("ROC|VOLUME_PROFILE|SL25_TP75", ["roc"], ["volume_profile"], 25, 75, "risk_pct", 0.015, 2),
        ("ROC|SESSION_LEVELS|SL25_TP100", ["roc"], ["session_levels"], 25, 100, "risk_pct", 0.015, 2),
        ("ROC|EMA_SLOPE|SL25_TP125", ["roc"], ["ema_slope"], 25, 125, "risk_pct", 0.015, 2),
        ("ROC|SL20_TP160", ["roc"], [], 20, 160, "risk_pct", 0.02, 2),
        ("ROC|SL25_TP200", ["roc"], [], 25, 200, "risk_pct", 0.02, 2),
        ("ROC|SL25_TP150", ["roc"], [], 25, 150, "risk_pct", 0.015, 2),
        ("MACD|SL25_TP200", ["macd"], [], 25, 200, "risk_pct", 0.02, 2),
        ("MACD|SL30_TP180", ["macd"], [], 30, 180, "risk_pct", 0.02, 2),
        ("MACD|EMA_SLOPE|SL25_TP125", ["macd"], ["ema_slope"], 25, 125, "risk_pct", 0.015, 2),
        ("MACD|VOLUME_PROFILE|SL25_TP200", ["macd"], ["volume_profile"], 25, 200, "risk_pct", 0.015, 2),
        ("MACD|LARGE_TRADE|SL30_TP150", ["macd"], ["large_trade_detection"], 30, 150, "risk_pct", 0.015, 2),
        ("MACD|LARGE_TRADE|SL15_TP120", ["macd"], ["large_trade_detection"], 15, 120, "risk_pct", 0.015, 2),
        ("MACD|TIME_OF_DAY|SL25_TP200", ["macd"], ["time_of_day"], 25, 200, "risk_pct", 0.02, 2),
        ("MACD|CANDLE_PATTERNS|SL30_TP60", ["macd"], ["candle_patterns"], 30, 60, "risk_pct", 0.015, 2),
        ("STOCH|LARGE_TRADE|SL30_TP150", ["stochastic"], ["large_trade_detection"], 30, 150, "risk_pct", 0.015, 2),
        ("STOCH|LARGE_TRADE|SL30_TP180", ["stochastic"], ["large_trade_detection"], 30, 180, "risk_pct", 0.015, 2),
        ("STOCH|LARGE_TRADE|SL30_TP60", ["stochastic"], ["large_trade_detection"], 30, 60, "risk_pct", 0.015, 2),
        ("STOCH|BK_SQUEEZE|SL10_TP30", ["stochastic"], ["bollinger_keltner_squeeze"], 10, 30, "risk_pct", 0.015, 2),
        ("WILLIAMS_R|FOOTPRINT|SL30_TP180", ["williams_r"], ["footprint_imbalance"], 30, 180, "risk_pct", 0.015, 2),
        ("WILLIAMS_R|FOOTPRINT|SL20_TP80", ["williams_r"], ["footprint_imbalance"], 20, 80, "risk_pct", 0.015, 2),
        ("VWAP|FOOTPRINT|SL12_TP96", ["vwap"], ["footprint_imbalance"], 12, 96, "risk_pct", 0.015, 2),
        ("VWAP|FOOTPRINT|SL10_TP50", ["vwap"], ["footprint_imbalance"], 10, 50, "risk_pct", 0.015, 2),
        ("VWAP|LAST_N_MIN|SL15_TP120", ["vwap"], ["last_n_minutes"], 15, 120, "risk_pct", 0.015, 2),
        ("KELTNER|SESSION_LEVELS|SL30_TP240", ["keltner_channels"], ["session_levels"], 30, 240, "risk_pct", 0.015, 2),
    ]

    for name, entries, filters, sl, tp, sm, rp, fc in combos:
        s = build_strategy(name, entries, filters, sl, tp, sizing_method=sm, risk_pct=rp, fixed_contracts=fc)
        if s:
            strats.append(s)

    return strats


def main():
    total_start = time.time()

    # ── Load 1 year of data (Mar 2025 – Mar 2026) ──
    logger.info("Loading 1-year data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_1yr = df_full.filter(pl.col("timestamp") >= pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    trading_days = df_1yr["timestamp"].dt.date().n_unique()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║        1-YEAR MC VALIDATION — FIND THE CHAMPION                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data:       {len(df_1yr):,} 1-minute bars ({trading_days} trading days)         ║
║  Period:     Mar 2025 – Mar 2026 (1 full year)                     ║
║  Account:    Topstep $50K | Slippage: 2 ticks/side                 ║
║  MC Sims:    5,000 per strategy                                    ║
║  Candidates: Top 25 Feb-Mar evolved + Top 30 Oct search winners    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    logger.info(f"1-year data: {len(df_1yr):,} bars | {df_1yr['timestamp'][0]} -> {df_1yr['timestamp'][-1]}")

    # ── Infrastructure ──
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    config = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_50k",
        start_date="2025-03-19", end_date="2026-03-18",
        slippage_ticks=2, initial_capital=50000.0,
    )
    data = {"1m": df_1yr}

    mc_config = MCConfig(
        n_simulations=5000,
        initial_capital=50000.0,
        prop_firm_rules=prop_rules,
        seed=42,
    )
    simulator = MonteCarloSimulator(mc_config)

    # ── Load candidates ──
    # Group A: Feb-Mar evolved strategies (top 25)
    with open("reports/evolved_strategies.json") as f:
        feb_evolved = json.load(f)
    feb_strats = [(e["strategy"], f"[FEB] {e['name']}") for e in feb_evolved[:25]]

    # Group B: Oct 2025 best signal combos (reconstructed)
    oct_strats = [(s.to_dict(), f"[OCT] {s.name}") for s in generate_oct_winners()]

    all_candidates = feb_strats + oct_strats
    logger.info(f"Candidates: {len(feb_strats)} Feb-Mar + {len(oct_strats)} Oct = {len(all_candidates)} total")

    # ── Run backtest + MC on each ──
    results = []
    for i, (strat_dict, label) in enumerate(all_candidates, 1):
        strat_dict["primary_timeframe"] = "1m"
        try:
            strategy = GeneratedStrategy.from_dict(strat_dict)
            bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
            result = bt.run(strategy)

            if len(result.trades) < 5:
                logger.info(f"  [{i:2d}/{len(all_candidates)}] {label[:50]} — {len(result.trades)} trades (skip)")
                continue

            metrics = calculate_metrics(result.trades, config.initial_capital)
            mc = simulator.run(result.trades, strategy_name=label)

            results.append({
                "label": label,
                "strategy": strat_dict,
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "sharpe": metrics.sharpe_ratio,
                "total_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,
                "avg_trade": metrics.avg_trade_pnl,
                "avg_winner": metrics.avg_winner,
                "avg_loser": metrics.avg_loser,
                "mc_median": mc.median_return,
                "mc_mean": mc.mean_return,
                "mc_5th": mc.pct_5th_return,
                "mc_95th": mc.pct_95th_return,
                "mc_p_profit": mc.probability_of_profit,
                "mc_p_ruin": mc.probability_of_ruin,
                "mc_composite": mc.composite_score,
                "mc_pass_rate": mc.prop_firm_pass_rate,
            })

            flag = "★" if mc.probability_of_profit >= 0.90 else "✓" if mc.median_return > 0 else "✗"
            logger.info(
                f"  [{i:2d}/{len(all_candidates)}] {flag} {label[:45]:45s} | "
                f"{metrics.total_trades:>4}t | WR={metrics.win_rate:5.1f}% | "
                f"PF={metrics.profit_factor:5.2f} | Sharpe={metrics.sharpe_ratio:5.2f} | "
                f"PnL=${metrics.total_pnl:>10,.0f} | MC=${mc.median_return:>10,.0f} | P={mc.probability_of_profit:.0%}"
            )

        except Exception as e:
            logger.warning(f"  [{i:2d}/{len(all_candidates)}] {label[:50]} — ERROR: {e}")

    elapsed = time.time() - total_start

    # ── Sort by MC median ──
    results.sort(key=lambda r: r["mc_median"], reverse=True)

    # ── Results ──
    profitable = [r for r in results if r["mc_median"] > 0]
    strong = [r for r in results if r["mc_p_profit"] >= 0.90]

    print(f"\n{'='*120}")
    print(f"  1-YEAR RESULTS — Mar 2025 to Mar 2026 ({elapsed:.0f}s)")
    print(f"{'='*120}")
    print(f"  Candidates tested:     {len(all_candidates)}")
    print(f"  Produced results:      {len(results)}")
    print(f"  MC median > $0:        {len(profitable)}")
    print(f"  MC P(profit) >= 90%:   {len(strong)}")
    print()

    if results:
        print(f"  {'#':<4} {'Strategy':<50} {'Tr':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'1yr PnL':>12} {'MC Med':>11} {'MC 5th':>10} {'MC 95th':>11} {'P(prof)':>8}")
        print(f"  {'-'*130}")
        for i, r in enumerate(results, 1):
            flag = "★" if r["mc_p_profit"] >= 0.90 else " "
            print(
                f"  {flag}{i:<3} {r['label'][:49]:<50} "
                f"{r['trades']:>5} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
                f"{r['sharpe']:>6.2f} ${r['total_pnl']:>11,.0f} "
                f"${r['mc_median']:>10,.0f} ${r['mc_5th']:>9,.0f} "
                f"${r['mc_95th']:>10,.0f} {r['mc_p_profit']:>7.1%}"
            )
        print(f"  {'-'*130}")

    # ── Champion breakdown ──
    if results:
        champ = results[0]
        sd = champ["strategy"]
        print(f"""
  ══════════════════════════════════════════════════════════════
  🏆 CHAMPION — 1 YEAR OF NQ DATA
  ══════════════════════════════════════════════════════════════
  Strategy:       {champ['label']}
  Entry signals:  {', '.join(e['signal_name'] for e in sd['entry_signals'])}""")
        if sd.get('entry_filters'):
            print(f"  Filters:        {', '.join(f['signal_name'] for f in sd['entry_filters'])}")
        print(f"""  Stop Loss:      {sd['exit_rules']['stop_loss_value']}pt ({sd['exit_rules']['stop_loss_type']})
  Take Profit:    {sd['exit_rules']['take_profit_value']}pt ({sd['exit_rules']['take_profit_type']})
  Sizing:         {sd['sizing_rules']}
  ──────────────────────────────────────────────────────────────
  Trades (1yr):   {champ['trades']}
  Win Rate:       {champ['win_rate']:.1f}%
  Profit Factor:  {champ['profit_factor']:.2f}
  Sharpe Ratio:   {champ['sharpe']:.2f}
  Net P&L (1yr):  ${champ['total_pnl']:,.2f}
  Max Drawdown:   ${champ['max_drawdown']:,.2f}
  Avg Winner:     ${champ['avg_winner']:,.2f}
  Avg Loser:      ${champ['avg_loser']:,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${champ['mc_median']:,.2f}
  MC P(profit):   {champ['mc_p_profit']:.1%}
  MC P(ruin):     {champ['mc_p_ruin']:.1%}
  MC 5th pctl:    ${champ['mc_5th']:,.2f}
  MC 95th pctl:   ${champ['mc_95th']:,.2f}
  MC Composite:   {champ['mc_composite']:.1f}/100
  Pass Rate:      {champ['mc_pass_rate']:.1%}
  ══════════════════════════════════════════════════════════════""")

        # Signal parameters
        print(f"\n  Signal Parameters:")
        for sig in sd["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in sd.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

    # ── Save ──
    output = {
        "test_type": "1_year_validation",
        "period": "2025-03-19 to 2026-03-18",
        "bars": len(df_1yr),
        "trading_days": trading_days,
        "candidates": len(all_candidates),
        "profitable": len(profitable),
        "strong_90pct": len(strong),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with open("reports/1yr_validation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/1yr_validation.json")
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
