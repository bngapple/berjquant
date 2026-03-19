#!/usr/bin/env python3
"""
MCQ Engine — Full Pipeline Runner

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --stage search     # Strategy search only
    python run_pipeline.py --stage mc         # Monte Carlo on saved strategies
    python run_pipeline.py --stage validate   # Walk-forward + regime analysis
    python run_pipeline.py --stage paper      # Paper trading replay
    python run_pipeline.py --fetch            # Fetch fresh data first
    python run_pipeline.py --max-strategies 100  # Limit strategy count
    python run_pipeline.py --mc-sims 5000     # MC simulation count
    python run_pipeline.py --top 10           # Keep top N strategies
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import polars as pl

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcq_pipeline")

# ── Project paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
STRATEGIES_DIR = PROJECT_ROOT / "strategies" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure key directories exist
for _d in (RESULTS_DIR, STRATEGIES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MCQ Engine Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["search", "mc", "validate", "paper", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch fresh data from Yahoo Finance before running",
    )
    parser.add_argument("--symbol", default="MNQ", help="Symbol to trade (MNQ or NQ)")
    parser.add_argument("--firm", default="topstep_50k", help="Prop firm profile name")
    parser.add_argument(
        "--max-strategies", type=int, default=200,
        help="Max strategies to generate in search stage",
    )
    parser.add_argument(
        "--max-entry-signals", type=int, default=2,
        help="Max entry signals per strategy combination",
    )
    parser.add_argument(
        "--max-filters", type=int, default=1,
        help="Max filters per strategy combination",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=5000,
        help="Monte Carlo simulations per strategy",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Keep top N strategies after each filtering stage",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--output-dir", default=str(REPORTS_DIR),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--paper-speed", type=float, default=0.0,
        help="Paper trading replay speed in seconds (0 = as fast as possible)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run backtests sequentially instead of in parallel (useful for debugging)",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  Banner
# ═════════════════════════════════════════════════════════════════════

def banner():
    print(r"""
 _____ _____ _____    _____         _
|     |     |     |  |   __|___ ___|_|___ ___
| | | |   --|  |  |  |   __|   | . | |   | -_|
|_|_|_|_____|__  _|  |_____|_|_|_  |_|_|_|___|
               |__|            |___|

+==============================================================+
|                    MCQ ENGINE PIPELINE                        |
|         Monte Carlo Quantitative Trading Engine              |
+--------------------------------------------------------------+
|  Phase 1: Data Pipeline       [==========]  Complete         |
|  Phase 2: Strategy Generation [==========]  Complete         |
|  Phase 3: Monte Carlo         [==========]  Complete         |
|  Phase 4: Validation          [==========]  Complete         |
|  Phase 5: Paper Trading       [==========]  Complete         |
+==============================================================+
""")


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def stage_header(name: str):
    """Print a prominent stage header."""
    width = 62
    print()
    print("=" * width)
    print(f"  STAGE: {name}")
    print("=" * width)
    print()


def stage_footer(name: str, elapsed: float, summary: str = ""):
    """Print stage completion footer."""
    print()
    print(f"  [{name}] completed in {elapsed:.1f}s{' -- ' + summary if summary else ''}")
    print("-" * 62)


def _get_contract_spec(symbol: str):
    """Return the correct ContractSpec for a symbol."""
    from engine.utils import MNQ_SPEC, NQ_SPEC
    return MNQ_SPEC if symbol.upper() == "MNQ" else NQ_SPEC


def _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec):
    """Build a RiskManager from configs."""
    from engine.risk_manager import RiskManager
    return RiskManager(
        prop_rules=prop_rules,
        session_config=session_config,
        events_calendar=events_calendar,
        contract_spec=contract_spec,
    )


def _build_backtest_config(args, data: dict[str, pl.DataFrame]):
    """Build a BacktestConfig with date range from the data."""
    from engine.utils import BacktestConfig
    primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
    df = data[primary_tf]
    start = str(df["timestamp"][0])[:10]
    end = str(df["timestamp"][-1])[:10]
    return BacktestConfig(
        symbol=args.symbol,
        prop_firm_profile=args.firm,
        start_date=start,
        end_date=end,
        slippage_ticks=2,
        initial_capital=50000.0,
    )


def _save_json(data: dict, filepath: Path):
    """Save a dict as JSON with numpy-safe serializer."""
    import numpy as np

    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return str(obj)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2, default=_default))
    logger.info("Saved %s", filepath)


# ═════════════════════════════════════════════════════════════════════
#  Stage 0 — Data Loading
# ═════════════════════════════════════════════════════════════════════

def load_data(args) -> dict[str, pl.DataFrame]:
    """Load or fetch market data, returning {"1m": df, "5m": df}."""
    stage_header("DATA LOADING")
    t0 = time.time()

    if args.fetch:
        logger.info("Fetching fresh data from Yahoo Finance...")
        from engine.data_fetcher import YFinanceFetcher
        fetcher = YFinanceFetcher(output_dir=str(DATA_DIR))
        data = fetcher.fetch_multi_timeframe(symbol=args.symbol, days_back_1m=59)
        # Persist for reuse
        for tf, df in data.items():
            out_dir = DATA_DIR / args.symbol / tf
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / "all.parquet")
        logger.info("Fresh data saved to %s", DATA_DIR / args.symbol)
    else:
        logger.info("Loading data from %s ...", DATA_DIR)
        from engine.data_pipeline import load_parquet, BarBuilder

        try:
            df_1m = load_parquet(DATA_DIR, args.symbol, "1m")
        except FileNotFoundError:
            # Fallback: try fetching from Yahoo Finance
            logger.warning(
                "No local data found for %s/1m. Attempting Yahoo Finance fetch...",
                args.symbol,
            )
            from engine.data_fetcher import YFinanceFetcher
            fetcher = YFinanceFetcher(output_dir=str(DATA_DIR))
            data = fetcher.fetch_multi_timeframe(symbol=args.symbol, days_back_1m=59)
            for tf, df in data.items():
                out_dir = DATA_DIR / args.symbol / tf
                out_dir.mkdir(parents=True, exist_ok=True)
                df.write_parquet(out_dir / "all.parquet")
            elapsed = time.time() - t0
            _print_data_summary(data)
            stage_footer("DATA LOADING", elapsed, f"fetched {sum(len(d) for d in data.values())} bars")
            return data

        # Build 5m from 1m
        try:
            df_5m = load_parquet(DATA_DIR, args.symbol, "5m")
        except FileNotFoundError:
            logger.info("No 5m data found; resampling from 1m bars...")
            df_5m = BarBuilder.resample(df_1m, freq="5m")

        data = {"1m": df_1m, "5m": df_5m}

    # Clean the data
    from engine.data_pipeline import DataCleaner
    try:
        from engine.utils import load_session_config
        session_config = load_session_config(CONFIG_DIR)
    except Exception:
        session_config = None

    cleaner = DataCleaner(session_config=session_config)
    for tf in list(data.keys()):
        data[tf] = cleaner.clean(data[tf])

    elapsed = time.time() - t0
    _print_data_summary(data)
    stage_footer("DATA LOADING", elapsed, f"{sum(len(d) for d in data.values())} total bars")
    return data


def _print_data_summary(data: dict[str, pl.DataFrame]):
    """Print a summary of loaded data."""
    print()
    print("  Data Summary")
    print("  " + "-" * 50)
    for tf in sorted(data.keys()):
        df = data[tf]
        if df.is_empty():
            print(f"  {tf:>4s}:  (empty)")
            continue
        n = len(df)
        start = df["timestamp"][0]
        end = df["timestamp"][-1]
        print(f"  {tf:>4s}:  {n:>8,} bars  |  {start}  ->  {end}")
    print()


# ═════════════════════════════════════════════════════════════════════
#  Stage 1 — Strategy Search
# ═════════════════════════════════════════════════════════════════════

def stage_search(args, data, prop_rules, session_config, events_calendar) -> list:
    """
    Generate strategy combinations, run parallel backtests, filter and rank.

    Returns list of (strategy_dict, BacktestResult) tuples sorted by composite score.
    """
    stage_header("STRATEGY SEARCH")
    t0 = time.time()

    from signals.registry import SignalRegistry
    from strategies.generator import StrategyGenerator, ExitRules, SizingRules
    from strategies.serializer import StrategySerializer
    from engine.parallel_runner import ParallelRunner, RunConfig
    from engine.utils import BacktestConfig

    contract_spec = _get_contract_spec(args.symbol)
    bt_config = _build_backtest_config(args, data)

    # ── 1. Generate strategies ─────────────────────────────────────
    logger.info("Building strategy combinations...")

    registry = SignalRegistry()
    generator = StrategyGenerator(registry)

    # Define exit rule variations
    exit_variations = [
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="fixed_points", take_profit_value=8.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=6.0,
                  take_profit_type="fixed_points", take_profit_value=12.0),
        ExitRules(stop_loss_type="atr_multiple", stop_loss_value=1.5,
                  take_profit_type="atr_multiple", take_profit_value=3.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="rr_ratio", take_profit_value=2.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="rr_ratio", take_profit_value=3.0),
    ]

    sizing_variations = [
        SizingRules(method="fixed", fixed_contracts=1),
        SizingRules(method="risk_pct", risk_pct=0.01),
    ]

    # Count possible combos first
    total_possible = generator.count_combinations(
        max_entry_signals=args.max_entry_signals,
        max_filters=args.max_filters,
        exit_variations=len(exit_variations),
        sizing_variations=len(sizing_variations),
    )
    logger.info(
        "Combinatorial space: %d possible strategies (capped at %d)",
        total_possible,
        args.max_strategies,
    )

    strategies = generator.generate(
        max_entry_signals=args.max_entry_signals,
        max_filters=args.max_filters,
        exit_variations=exit_variations,
        sizing_variations=sizing_variations,
        max_strategies=args.max_strategies,
    )
    logger.info("Generated %d strategies", len(strategies))

    if not strategies:
        logger.warning("No strategies generated. Check signal registry and parameters.")
        stage_footer("STRATEGY SEARCH", time.time() - t0, "0 strategies")
        return []

    # ── 2. Parallel backtesting ────────────────────────────────────
    run_config = RunConfig(
        data=data,
        config=bt_config,
        prop_rules=prop_rules,
        session_config=session_config,
        events_calendar=events_calendar,
        contract_spec=contract_spec,
        min_trades=10,
        min_profit_factor=0.0,
        min_sharpe=-999.0,
    )

    runner = ParallelRunner(run_config, n_workers=args.workers)

    if args.sequential:
        logger.info("Running backtests sequentially (debug mode)...")
        batch_result = runner.run_sequential(strategies)
    else:
        batch_result = runner.run(strategies)

    logger.info(
        "Backtest batch: %d completed, %d failed, %d passed filters (%.1f strats/sec)",
        batch_result.completed,
        batch_result.failed,
        batch_result.filtered,
        batch_result.total_strategies / batch_result.elapsed_seconds
        if batch_result.elapsed_seconds > 0 else 0,
    )

    if batch_result.errors:
        logger.warning("%d strategies failed:", len(batch_result.errors))
        for name, err in batch_result.errors[:5]:
            logger.warning("  %s: %s", name, err.split("\n")[0])
        if len(batch_result.errors) > 5:
            logger.warning("  ... and %d more", len(batch_result.errors) - 5)

    # ── 3. Rank by composite score ─────────────────────────────────
    top_results = ParallelRunner.rank_results(
        batch_result, metric="composite", top_n=args.top,
    )

    # Print leaderboard
    ParallelRunner.print_leaderboard(top_results, top_n=args.top)

    # ── 4. Save results ────────────────────────────────────────────
    # Save top strategy definitions for downstream stages
    serializer = StrategySerializer(strategies_dir=str(STRATEGIES_DIR))
    top_strategy_objects = []
    for strategy_dict, bt_result in top_results:
        from strategies.generator import GeneratedStrategy
        try:
            strat = GeneratedStrategy.from_dict(strategy_dict)
            top_strategy_objects.append(strat)
        except Exception:
            logger.warning("Could not reconstruct strategy from dict: %s", strategy_dict.get("name"))

    if top_strategy_objects:
        batch_path = serializer.save_batch(top_strategy_objects, batch_name="search_top")
        logger.info("Saved top %d strategies to %s", len(top_strategy_objects), batch_path)

    # Save search results summary
    search_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_generated": len(strategies),
        "total_completed": batch_result.completed,
        "total_failed": batch_result.failed,
        "total_passed_filter": batch_result.filtered,
        "elapsed_seconds": round(batch_result.elapsed_seconds, 2),
        "top_strategies": [],
    }
    for rank, (sd, br) in enumerate(top_results, 1):
        m = br.metrics
        search_summary["top_strategies"].append({
            "rank": rank,
            "name": br.strategy_name,
            "trades": m.total_trades if m else 0,
            "win_rate": round(m.win_rate, 1) if m else 0,
            "profit_factor": round(m.profit_factor, 2) if m else 0,
            "sharpe_ratio": round(m.sharpe_ratio, 2) if m else 0,
            "total_pnl": round(m.total_pnl, 2) if m else 0,
            "max_drawdown": round(m.max_drawdown, 2) if m else 0,
        })
    _save_json(search_summary, REPORTS_DIR / "search_results.json")

    elapsed = time.time() - t0
    stage_footer(
        "STRATEGY SEARCH", elapsed,
        f"{batch_result.filtered} passed -> top {len(top_results)} selected",
    )
    return top_results


# ═════════════════════════════════════════════════════════════════════
#  Stage 2 — Monte Carlo Stress Testing
# ═════════════════════════════════════════════════════════════════════

def stage_monte_carlo(args, top_strategies, prop_rules) -> list:
    """
    Run Monte Carlo simulation on each top strategy, score and rank.

    Args:
        top_strategies: list of (strategy_dict, BacktestResult) from search stage.

    Returns list of (strategy_dict, MCResult, StrategyScore) tuples.
    """
    stage_header("MONTE CARLO STRESS TESTING")
    t0 = time.time()

    from monte_carlo.simulator import MonteCarloSimulator, MCConfig
    from monte_carlo.scoring import StrategyScorer
    from monte_carlo.visualization import MCVisualizer

    contract_spec = _get_contract_spec(args.symbol)

    mc_config = MCConfig(
        n_simulations=args.mc_sims,
        initial_capital=prop_rules.account_size,
        prop_firm_rules=prop_rules,
        tick_value=contract_spec.tick_value,
        n_workers=args.workers,
    )

    simulator = MonteCarloSimulator(mc_config)
    scorer = StrategyScorer(prop_rules=prop_rules)
    viz = MCVisualizer(output_dir=str(Path(args.output_dir) / "charts"))

    mc_results = []  # (strategy_dict, mc_result)
    total = len(top_strategies)

    for i, (strategy_dict, bt_result) in enumerate(top_strategies, 1):
        name = bt_result.strategy_name
        trades = bt_result.trades
        n_trades = len(trades)

        if n_trades < 5:
            logger.warning(
                "[%d/%d] Skipping '%s' — only %d trades (need >= 5)",
                i, total, name, n_trades,
            )
            continue

        logger.info(
            "[%d/%d] Running MC simulation for '%s' (%d trades, %d sims)...",
            i, total, name, n_trades, args.mc_sims,
        )

        try:
            mc_result = simulator.run(trades, strategy_name=name)
            mc_results.append((strategy_dict, mc_result))

            logger.info(
                "  -> median=$%.0f  P(profit)=%.1f%%  P(ruin)=%.1f%%  "
                "pass_rate=%.1f%%  composite=%.1f",
                mc_result.median_return,
                mc_result.probability_of_profit * 100,
                mc_result.probability_of_ruin * 100,
                mc_result.prop_firm_pass_rate * 100,
                mc_result.composite_score,
            )
        except Exception:
            logger.error("MC simulation failed for '%s':", name)
            logger.error(traceback.format_exc())
            continue

    if not mc_results:
        logger.warning("No strategies survived Monte Carlo simulation.")
        stage_footer("MONTE CARLO", time.time() - t0, "0 survivors")
        return []

    # ── Score all MC results ───────────────────────────────────────
    mc_result_objects = [mcr for _, mcr in mc_results]
    scores = scorer.score_batch(mc_result_objects)

    # Rank by composite score (viable strategies first)
    ranked_scores = scorer.rank(scores, top_n=args.top, viable_only=False)

    # Print the MC leaderboard
    scorer.print_leaderboard(ranked_scores, top_n=args.top)

    # ── Generate visualization for top strategies ──────────────────
    for score in ranked_scores[:min(5, len(ranked_scores))]:
        try:
            viz.full_report(score.mc_result, strategy_score=score)
        except Exception:
            logger.warning("Failed to generate charts for '%s'", score.strategy_name)

    # Generate leaderboard comparison chart
    if len(ranked_scores) >= 2:
        try:
            viz.leaderboard_comparison(ranked_scores, top_n=min(10, len(ranked_scores)))
        except Exception:
            logger.warning("Failed to generate leaderboard comparison chart")

    # ── Export MC report ───────────────────────────────────────────
    try:
        report_path = scorer.export_report(
            ranked_scores,
            filepath=Path(args.output_dir) / "mc_analysis_report.json",
        )
        logger.info("MC analysis report saved to %s", report_path)
    except Exception:
        logger.warning("Failed to export MC report")

    # ── Build combined result list ─────────────────────────────────
    # Map strategy_name -> strategy_dict for lookup
    name_to_dict = {}
    for sd, mcr in mc_results:
        name_to_dict[mcr.strategy_name] = sd

    scored_results = []
    for score in ranked_scores:
        sd = name_to_dict.get(score.strategy_name, {})
        scored_results.append((sd, score.mc_result, score))

    elapsed = time.time() - t0
    viable_count = sum(1 for s in ranked_scores if s.is_viable)
    stage_footer(
        "MONTE CARLO", elapsed,
        f"{len(ranked_scores)} scored, {viable_count} viable",
    )
    return scored_results


# ═════════════════════════════════════════════════════════════════════
#  Stage 3 — Walk-Forward Validation + Regime Analysis
# ═════════════════════════════════════════════════════════════════════

def stage_validate(args, mc_survivors, data, prop_rules, session_config, events_calendar) -> list:
    """
    Walk-forward validation, regime analysis, and overfitting detection.

    Args:
        mc_survivors: list of (strategy_dict, MCResult, StrategyScore) from MC stage.

    Returns list of validated (strategy_dict, results_dict) tuples.
    """
    stage_header("WALK-FORWARD VALIDATION + REGIME ANALYSIS")
    t0 = time.time()

    from strategies.generator import GeneratedStrategy
    from validation.walk_forward import WalkForwardValidator
    from validation.regime import RegimeDetector
    from validation.correlation import CorrelationAnalyzer, OverfitDetector
    from engine.utils import BacktestConfig

    contract_spec = _get_contract_spec(args.symbol)
    bt_config = _build_backtest_config(args, data)
    rm = _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec)

    wf_validator = WalkForwardValidator(
        data=data,
        risk_manager=rm,
        contract_spec=contract_spec,
        config=bt_config,
        account_size=prop_rules.account_size,
    )

    regime_detector = RegimeDetector()
    overfit_detector = OverfitDetector(account_size=prop_rules.account_size)

    validated = []
    total = len(mc_survivors)

    for i, (strategy_dict, mc_result, mc_score) in enumerate(mc_survivors, 1):
        name = mc_result.strategy_name
        logger.info("[%d/%d] Validating '%s' ...", i, total, name)

        results = {
            "strategy_name": name,
            "mc_score": mc_score,
            "mc_result": mc_result,
        }

        # Reconstruct strategy object
        try:
            strategy = GeneratedStrategy.from_dict(strategy_dict)
        except Exception:
            logger.warning("  Could not reconstruct strategy '%s'; skipping.", name)
            continue

        # ── Walk-Forward ───────────────────────────────────────────
        try:
            wf_result = wf_validator.validate(
                strategy,
                train_days=40,
                test_days=15,
                step_days=15,
            )
            wf_validator.print_report(wf_result)
            results["wf_result"] = wf_result
            results["wf_efficiency"] = wf_result.wf_efficiency
            results["is_overfit_wf"] = wf_result.is_overfit
            results["pct_profitable_windows"] = wf_result.pct_profitable_windows
        except Exception:
            logger.error("  Walk-forward failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["wf_result"] = None
            results["wf_efficiency"] = 0.0
            results["is_overfit_wf"] = True

        # ── Regime Analysis ────────────────────────────────────────
        primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
        df_primary = data[primary_tf]

        try:
            # We need trades from the full backtest for regime tagging.
            # Re-run a quick backtest to get trades (the WF result has
            # fragmented trades across windows).
            from engine.backtester import VectorizedBacktester
            bt = VectorizedBacktester(
                data=data,
                risk_manager=_build_risk_manager(
                    prop_rules, session_config, events_calendar, contract_spec
                ),
                contract_spec=contract_spec,
                config=bt_config,
            )
            full_bt = bt.run(strategy)
            full_trades = full_bt.trades

            regime_analysis = regime_detector.analyze_strategy(
                full_trades, df_primary, strategy_name=name,
            )
            regime_detector.print_analysis(regime_analysis)
            results["regime_analysis"] = regime_analysis
            results["regime_sensitivity"] = regime_analysis.regime_sensitivity
        except Exception:
            logger.error("  Regime analysis failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["regime_analysis"] = None
            results["regime_sensitivity"] = 1.0
            full_trades = []

        # ── Overfitting Detection ──────────────────────────────────
        try:
            wf_res = results.get("wf_result")
            overfit_report = overfit_detector.analyze(
                strategy_name=name,
                trades=full_trades,
                wf_result=wf_res,
            )
            overfit_detector.print_report(overfit_report)
            results["overfit_report"] = overfit_report
            results["overfit_verdict"] = overfit_report.verdict
        except Exception:
            logger.error("  Overfit detection failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["overfit_report"] = None
            results["overfit_verdict"] = "unknown"

        validated.append((strategy_dict, results))

    # ── Correlation Analysis (portfolio-level) ─────────────────────
    if len(validated) >= 2:
        print()
        print("  Running portfolio correlation analysis...")
        try:
            correlation_analyzer = CorrelationAnalyzer(account_size=prop_rules.account_size)
            strategy_trade_pairs = []
            for sd, res in validated:
                # Use WF OOS trades if available, else empty
                wf_res = res.get("wf_result")
                if wf_res and wf_res.oos_trades:
                    strategy_trade_pairs.append((res["strategy_name"], wf_res.oos_trades))
            if len(strategy_trade_pairs) >= 2:
                portfolio = correlation_analyzer.analyze_portfolio(strategy_trade_pairs)
                correlation_analyzer.print_portfolio_report(portfolio)
        except Exception:
            logger.warning("Portfolio correlation analysis failed")

    # ── Filter overfit strategies ──────────────────────────────────
    surviving = []
    rejected_count = 0
    for sd, res in validated:
        verdict = res.get("overfit_verdict", "unknown")
        is_overfit_wf = res.get("is_overfit_wf", False)

        # Reject strategies that are "likely_overfit" AND failed WF
        if verdict == "likely_overfit" and is_overfit_wf:
            logger.info(
                "  REJECTED '%s': likely overfit (verdict=%s, WF overfit=%s)",
                res["strategy_name"], verdict, is_overfit_wf,
            )
            rejected_count += 1
            continue

        surviving.append((sd, res))

    # ── Save validation summary ────────────────────────────────────
    val_summary = {
        "timestamp": datetime.now().isoformat(),
        "strategies_validated": len(validated),
        "strategies_rejected": rejected_count,
        "strategies_surviving": len(surviving),
        "strategies": [],
    }
    for sd, res in surviving:
        entry = {
            "name": res["strategy_name"],
            "wf_efficiency": round(res.get("wf_efficiency", 0), 3),
            "pct_profitable_windows": round(res.get("pct_profitable_windows", 0), 1),
            "regime_sensitivity": round(res.get("regime_sensitivity", 1.0), 3),
            "overfit_verdict": res.get("overfit_verdict", "unknown"),
            "mc_composite_score": round(res["mc_score"].composite_score, 2),
            "mc_grade": res["mc_score"].grade,
        }
        val_summary["strategies"].append(entry)
    _save_json(val_summary, REPORTS_DIR / "validation_results.json")

    elapsed = time.time() - t0
    stage_footer(
        "VALIDATION", elapsed,
        f"{len(validated)} validated, {rejected_count} rejected, {len(surviving)} surviving",
    )
    return surviving


# ═════════════════════════════════════════════════════════════════════
#  Stage 4 — Paper Trading Replay
# ═════════════════════════════════════════════════════════════════════

def stage_paper_trade(args, validated_strategies, data, prop_rules, session_config, events_calendar):
    """
    Replay the best strategy through the signal engine and paper trader.

    Args:
        validated_strategies: list of (strategy_dict, results_dict) from validation.
    """
    stage_header("PAPER TRADING REPLAY")
    t0 = time.time()

    if not validated_strategies:
        logger.warning("No validated strategies available for paper trading.")
        stage_footer("PAPER TRADING", time.time() - t0, "skipped (no strategies)")
        return

    from strategies.generator import GeneratedStrategy
    from live.signal_engine import SignalEngine
    from live.paper_trader import PaperTrader
    from live.dashboard import TradingDashboard
    from live.alerts import AlertManager, ConsoleChannel, FileChannel

    contract_spec = _get_contract_spec(args.symbol)
    rm = _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec)

    # Pick the best strategy (first in the validated list, already ranked)
    best_sd, best_res = validated_strategies[0]
    best_name = best_res["strategy_name"]
    logger.info("Paper trading with best strategy: '%s'", best_name)
    logger.info(
        "  MC Grade: %s  |  MC Score: %.1f  |  WF Efficiency: %.2f",
        best_res["mc_score"].grade,
        best_res["mc_score"].composite_score,
        best_res.get("wf_efficiency", 0),
    )

    # Reconstruct strategy
    try:
        strategy = GeneratedStrategy.from_dict(best_sd)
    except Exception:
        logger.error("Could not reconstruct best strategy '%s'", best_name)
        stage_footer("PAPER TRADING", time.time() - t0, "failed")
        return

    # ── Setup ──────────────────────────────────────────────────────
    account_state = rm.init_account(prop_rules.account_size)

    # Alert manager with console and file channels
    alert_manager = AlertManager()
    alert_manager.add_channel(ConsoleChannel())
    alert_manager.add_channel(
        FileChannel(filepath=str(PROJECT_ROOT / "logs" / "paper_alerts.log")),
    )

    # Signal engine
    engine = SignalEngine(
        strategies=[strategy],
        risk_manager=rm,
        contract_spec=contract_spec,
        account_state=account_state,
        prop_rules=prop_rules,
    )

    # Paper trader
    trader = PaperTrader(
        risk_manager=_build_risk_manager(
            prop_rules, session_config, events_calendar, contract_spec,
        ),
        contract_spec=contract_spec,
        prop_rules=prop_rules,
        initial_balance=prop_rules.account_size,
        slippage_ticks=2,
        log_dir=str(PROJECT_ROOT / "logs" / "paper_trading"),
    )

    # Wire signal engine to paper trader
    engine.on_signal(trader.on_signal)

    # Dashboard
    dashboard = TradingDashboard(
        prop_rules=prop_rules,
        output_dir=str(Path(args.output_dir)),
    )

    # ── Replay ─────────────────────────────────────────────────────
    primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
    df_replay = data[primary_tf]

    total_bars = len(df_replay)
    logger.info("Replaying %d bars (speed=%.2fs per bar)...", total_bars, args.paper_speed)

    progress_interval = max(1, total_bars // 20)

    def replay_callback(bar_idx, total, bar_signals):
        """Progress callback during replay."""
        row = df_replay.row(bar_idx, named=True)
        ts = row.get("timestamp")
        price = row.get("close", 0.0)

        if isinstance(ts, datetime):
            trader.on_price_update(ts, price)
        elif ts is not None:
            try:
                ts_dt = datetime.fromisoformat(str(ts))
                trader.on_price_update(ts_dt, price)
            except (ValueError, TypeError):
                pass

        if bar_idx % progress_interval == 0 or bar_idx == total - 1:
            pct = (bar_idx + 1) / total * 100
            n_trades = len(trader.account.trade_history)
            balance = trader.account.account_state.current_balance
            logger.info(
                "  Replay progress: %d/%d (%.0f%%) | %d trades | balance=$%.2f",
                bar_idx + 1, total, pct, n_trades, balance,
            )

    signals = engine.replay(
        df_replay,
        speed=args.paper_speed,
        callback=replay_callback,
    )

    # ── Results ────────────────────────────────────────────────────
    print()
    print("  Paper Trading Complete")
    print("  " + "-" * 50)
    print(f"  Signals generated:  {len(signals)}")
    print(f"  Signals executed:   {trader.account.signals_executed}")
    print(f"  Signals rejected:   {trader.account.signals_rejected}")
    print(f"  Trades completed:   {len(trader.account.trade_history)}")

    # Print status and daily summary
    trader.print_status()
    trader.print_daily_summary()
    trader.print_trade_log(last_n=20)

    # Dashboard update and report
    dashboard.update(trader.account)
    dashboard.print_status()

    try:
        report_path = dashboard.generate_report(output_dir=str(Path(args.output_dir)))
        logger.info("Performance report saved to %s", report_path)
    except Exception:
        logger.warning("Failed to generate HTML dashboard report")

    # Export paper trading results
    try:
        results_path = trader.export_results(
            filepath=Path(args.output_dir) / "paper_trading_results.json",
        )
        logger.info("Paper trading results exported to %s", results_path)
    except Exception:
        logger.warning("Failed to export paper trading results")

    elapsed = time.time() - t0
    n_trades = len(trader.account.trade_history)
    final_balance = trader.account.account_state.current_balance
    net_pnl = final_balance - prop_rules.account_size
    stage_footer(
        "PAPER TRADING", elapsed,
        f"{n_trades} trades, net P&L ${net_pnl:+,.2f}, balance ${final_balance:,.2f}",
    )


# ═════════════════════════════════════════════════════════════════════
#  Leaderboard Save
# ═════════════════════════════════════════════════════════════════════

def save_final_leaderboard(validated_strategies, output_dir: str):
    """Save the final leaderboard to reports/leaderboard.json."""
    if not validated_strategies:
        return

    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "type": "final_leaderboard",
        "count": len(validated_strategies),
        "entries": [],
    }

    for rank, (sd, res) in enumerate(validated_strategies, 1):
        mc_score = res.get("mc_score")
        entry = {
            "rank": rank,
            "strategy_name": res["strategy_name"],
            "mc_composite_score": round(mc_score.composite_score, 2) if mc_score else 0,
            "mc_grade": mc_score.grade if mc_score else "N/A",
            "mc_pass_rate": round(mc_score.mc_result.prop_firm_pass_rate * 100, 1)
            if mc_score else 0,
            "mc_median_return": round(mc_score.mc_result.median_return, 2) if mc_score else 0,
            "mc_probability_of_ruin": round(mc_score.mc_result.probability_of_ruin * 100, 1)
            if mc_score else 0,
            "wf_efficiency": round(res.get("wf_efficiency", 0), 3),
            "pct_profitable_windows": round(res.get("pct_profitable_windows", 0), 1),
            "regime_sensitivity": round(res.get("regime_sensitivity", 1.0), 3),
            "overfit_verdict": res.get("overfit_verdict", "unknown"),
        }

        # Include strategy definition if available
        if sd:
            entry["strategy"] = sd

        leaderboard["entries"].append(entry)

    filepath = Path(output_dir) / "leaderboard.json"
    _save_json(leaderboard, filepath)
    logger.info("Final leaderboard saved to %s", filepath)


# ═════════════════════════════════════════════════════════════════════
#  Final Summary
# ═════════════════════════════════════════════════════════════════════

def print_final_summary(stage_timings: dict[str, float], total_time: float):
    """Print a summary table with timing per stage."""
    print()
    print("=" * 62)
    print("  PIPELINE SUMMARY")
    print("=" * 62)
    print()
    print(f"  {'Stage':<30s}  {'Time':>10s}  {'% of Total':>10s}")
    print("  " + "-" * 52)

    for stage, elapsed in stage_timings.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        time_str = f"{elapsed:.1f}s"
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.0f}s"
        print(f"  {stage:<30s}  {time_str:>10s}  {pct:>9.1f}%")

    print("  " + "-" * 52)

    total_str = f"{total_time:.1f}s"
    if total_time >= 60:
        mins = int(total_time // 60)
        secs = total_time % 60
        total_str = f"{mins}m {secs:.0f}s"
    print(f"  {'TOTAL':<30s}  {total_str:>10s}  {'100.0':>9s}%")
    print()
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    banner()

    pipeline_start = time.time()
    stage_timings: dict[str, float] = {}

    # ── Load configuration ─────────────────────────────────────────
    from engine.utils import (
        load_prop_firm_rules,
        load_session_config,
        load_events_calendar,
    )

    try:
        prop_rules = load_prop_firm_rules(CONFIG_DIR, args.firm)
        session_config = load_session_config(CONFIG_DIR)
        events_calendar = load_events_calendar(CONFIG_DIR)
    except Exception as e:
        logger.error("Failed to load configuration from %s: %s", CONFIG_DIR, e)
        sys.exit(1)

    logger.info("Loaded config: firm=%s, symbol=%s", prop_rules.firm_name, args.symbol)
    logger.info(
        "  Account: $%,.0f | Daily limit: $%,.0f | Max DD: $%,.0f (%s)",
        prop_rules.account_size,
        prop_rules.daily_loss_limit,
        prop_rules.max_drawdown,
        prop_rules.drawdown_type,
    )

    # ── Load data ──────────────────────────────────────────────────
    t0 = time.time()
    data = load_data(args)
    stage_timings["Data Loading"] = time.time() - t0

    if not data or all(df.is_empty() for df in data.values()):
        logger.error("No data available. Cannot proceed.")
        sys.exit(1)

    # ── Determine which stages to run ──────────────────────────────
    run_search = args.stage in ("search", "all")
    run_mc = args.stage in ("mc", "all")
    run_validate = args.stage in ("validate", "all")
    run_paper = args.stage in ("paper", "all")

    # Intermediate results that carry between stages
    top_strategies = []
    mc_survivors = []
    validated_strategies = []

    # ── Stage 1: Strategy Search ───────────────────────────────────
    if run_search:
        t0 = time.time()
        top_strategies = stage_search(
            args, data, prop_rules, session_config, events_calendar,
        )
        stage_timings["Strategy Search"] = time.time() - t0

    # ── Stage 2: Monte Carlo ───────────────────────────────────────
    if run_mc:
        # If we skipped search, try to load saved strategies
        if not top_strategies and not run_search:
            top_strategies = _load_saved_search_results()

        if top_strategies:
            t0 = time.time()
            mc_survivors = stage_monte_carlo(args, top_strategies, prop_rules)
            stage_timings["Monte Carlo"] = time.time() - t0
        else:
            logger.warning("No strategies available for Monte Carlo. Run --stage search first.")

    # ── Stage 3: Validation ────────────────────────────────────────
    if run_validate:
        # If we skipped MC, try to use search results directly
        if not mc_survivors and not run_mc:
            # Try loading from saved results
            top_strategies = _load_saved_search_results()
            if top_strategies:
                t0 = time.time()
                mc_survivors = stage_monte_carlo(args, top_strategies, prop_rules)
                stage_timings["Monte Carlo"] = time.time() - t0

        if mc_survivors:
            t0 = time.time()
            validated_strategies = stage_validate(
                args, mc_survivors, data, prop_rules, session_config, events_calendar,
            )
            stage_timings["Validation"] = time.time() - t0
        else:
            logger.warning("No MC survivors for validation. Run earlier stages first.")

    # ── Stage 4: Paper Trading ─────────────────────────────────────
    if run_paper:
        # If we skipped earlier stages, use whatever we have
        if not validated_strategies and mc_survivors:
            # Use MC survivors directly without full validation
            validated_strategies = [
                (sd, {"strategy_name": mcr.strategy_name, "mc_score": score, "mc_result": mcr})
                for sd, mcr, score in mc_survivors
            ]

        if validated_strategies:
            t0 = time.time()
            stage_paper_trade(
                args, validated_strategies, data, prop_rules, session_config, events_calendar,
            )
            stage_timings["Paper Trading"] = time.time() - t0
        else:
            logger.warning("No validated strategies for paper trading. Run earlier stages first.")

    # ── Save final leaderboard ─────────────────────────────────────
    if validated_strategies:
        save_final_leaderboard(validated_strategies, args.output_dir)

    # ── Final summary ──────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print_final_summary(stage_timings, total_time)


def _load_saved_search_results() -> list:
    """
    Try to load search results from the last saved batch.

    Returns list of (strategy_dict, BacktestResult) tuples,
    or empty list if nothing is found.
    """
    from strategies.serializer import StrategySerializer
    from strategies.generator import GeneratedStrategy
    from engine.utils import BacktestResult, BacktestConfig

    serializer = StrategySerializer(strategies_dir=str(STRATEGIES_DIR))
    saved = serializer.list_saved()

    # Find the most recent batch file
    batch_files = [
        s for s in saved
        if s.get("type") == "batch" and "search_top" in s.get("name", "")
    ]

    if not batch_files:
        logger.info("No saved search results found in %s", STRATEGIES_DIR)
        return []

    # Sort by saved_at descending
    batch_files.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    latest = batch_files[0]

    logger.info("Loading saved strategies from %s", latest["path"])
    try:
        strategies = serializer.load_batch(latest["path"])

        # We need BacktestResult objects — create minimal ones
        # (the MC stage only uses .trades, so we need at least a dummy config)
        dummy_config = BacktestConfig(
            symbol="MNQ",
            prop_firm_profile="topstep_50k",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        results = []
        for strat in strategies:
            # Use the strategy dict as the strategy_dict, and create a
            # placeholder BacktestResult. The MC stage will re-run if needed.
            sd = strat.to_dict()
            # We don't have trades — the user needs to re-run search or
            # run MC separately after producing backtests.
            logger.warning(
                "Strategy '%s' loaded but has no backtest trades. "
                "Monte Carlo will require re-running search.",
                strat.name,
            )
            # Return empty results — caller should handle gracefully
            br = BacktestResult(
                strategy_name=strat.name,
                config=dummy_config,
                trades=[],
                equity_curve=[],
            )
            results.append((sd, br))

        return results
    except Exception:
        logger.error("Failed to load saved strategies: %s", traceback.format_exc())
        return []


if __name__ == "__main__":
    main()
