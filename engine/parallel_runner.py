"""Parallel backtest runner — test hundreds/thousands of strategies concurrently."""

import multiprocessing as mp
from multiprocessing import Pool
import time
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import polars as pl

from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from engine.risk_manager import RiskManager
from engine.utils import (
    BacktestConfig,
    BacktestResult,
    ContractSpec,
    PerformanceMetrics,
    PropFirmRules,
)

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a parallel backtest run."""

    data: dict[str, pl.DataFrame]  # {"1m": df, "5m": df, ...}
    config: BacktestConfig
    prop_rules: PropFirmRules
    session_config: dict
    events_calendar: dict
    contract_spec: ContractSpec

    # Filtering thresholds (skip poor strategies early)
    min_trades: int = 10  # skip if fewer trades
    min_profit_factor: float = 0.0  # skip if below (0 = no filter)
    min_sharpe: float = -999.0  # skip if below


@dataclass
class BatchResult:
    """Results from a parallel backtest batch."""

    total_strategies: int
    completed: int
    failed: int
    filtered: int  # met min thresholds
    results: list  # list of (strategy, BacktestResult) tuples
    elapsed_seconds: float
    errors: list[tuple[str, str]]  # (strategy_name, error_message)


# ── Worker Function (top-level for pickling) ─────────────────────────


def _run_single_backtest(args: tuple) -> tuple:
    """
    Worker function for multiprocessing. Must be top-level (picklable).

    Args: (strategy_dict, data_dict, config, prop_rules, session_config,
           events_calendar, contract_spec, min_trades)

    Returns: (strategy_dict, result_dict | None, error_str | None)

    Note: We pass dicts/serializable objects because multiprocessing
    pickles arguments. Strategy and config objects need to be
    reconstructed in each worker.
    """
    (
        strategy_dict,
        data,
        config,
        prop_rules,
        session_config,
        events_calendar,
        contract_spec,
        min_trades,
    ) = args

    strategy_name = strategy_dict.get("name", "unknown")

    try:
        # Reconstruct strategy from dict.
        # Import here to avoid circular imports and ensure each worker
        # has the module loaded.
        strategy = _reconstruct_strategy(strategy_dict)

        # Build risk manager and backtester in this process
        risk_manager = RiskManager(
            prop_rules=prop_rules,
            session_config=session_config,
            events_calendar=events_calendar,
            contract_spec=contract_spec,
        )
        backtester = VectorizedBacktester(
            data=data,
            risk_manager=risk_manager,
            contract_spec=contract_spec,
            config=config,
        )

        # Run the backtest
        result = backtester.run(strategy)

        # Calculate metrics
        metrics = calculate_metrics(result.trades, config.initial_capital)
        result.metrics = metrics

        # Early filter: skip if too few trades
        if metrics.total_trades < min_trades:
            return (strategy_dict, None, None)

        # Serialize result for transfer back to main process
        result_dict = _serialize_result(result)
        return (strategy_dict, result_dict, None)

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}\n{tb}"
        return (strategy_dict, None, error_msg)


def _reconstruct_strategy(strategy_dict: dict):
    """
    Reconstruct a strategy object from its serialized dict.

    Supports two patterns:
    1. Strategies with a 'class_path' key — dynamically import and call from_dict
    2. Strategies with a 'type' key — known strategy types reconstructed directly
    """
    class_path = strategy_dict.get("class_path")

    if class_path:
        # Dynamic import: "strategies.generator.GeneratedStrategy"
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        if hasattr(cls, "from_dict"):
            return cls.from_dict(strategy_dict)
        else:
            # Try passing params directly to __init__
            params = {k: v for k, v in strategy_dict.items() if k != "class_path"}
            return cls(**params)

    # Fallback: try known strategy types
    strategy_type = strategy_dict.get("type", "")

    if strategy_type == "ema_crossover":
        from strategies.ema_crossover import EMACrossoverStrategy

        params = strategy_dict.get("params", strategy_dict)
        return EMACrossoverStrategy(
            fast_period=params.get("fast_period", 9),
            slow_period=params.get("slow_period", 21),
            stop_loss_points=params.get("stop_loss_points", 4.0),
            take_profit_points=params.get("take_profit_points", 8.0),
            contracts=params.get("contracts", 1),
            primary_timeframe=params.get("primary_timeframe", "1m"),
        )

    # Fallback: try GeneratedStrategy if dict has entry_signals key
    if "entry_signals" in strategy_dict:
        from strategies.generator import GeneratedStrategy
        return GeneratedStrategy.from_dict(strategy_dict)

    raise ValueError(
        f"Cannot reconstruct strategy from dict: missing 'class_path' or "
        f"recognized 'type'. Got keys: {list(strategy_dict.keys())}"
    )


def _serialize_result(result: BacktestResult) -> dict:
    """Serialize a BacktestResult into a picklable dict."""
    metrics = result.metrics
    metrics_dict = None
    if metrics is not None:
        metrics_dict = {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "total_pnl": metrics.total_pnl,
            "gross_profit": metrics.gross_profit,
            "gross_loss": metrics.gross_loss,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "avg_trade_pnl": metrics.avg_trade_pnl,
            "avg_winner": metrics.avg_winner,
            "avg_loser": metrics.avg_loser,
            "largest_winner": metrics.largest_winner,
            "largest_loser": metrics.largest_loser,
            "avg_hold_time_seconds": metrics.avg_hold_time_seconds,
            "profit_by_session": metrics.profit_by_session,
            "consistency_score": metrics.consistency_score,
        }

    return {
        "strategy_name": result.strategy_name,
        "trades": result.trades,  # Trade is a frozen dataclass, picklable
        "equity_curve": result.equity_curve,
        "config": result.config,
        "metrics": metrics_dict,
    }


def _deserialize_result(result_dict: dict) -> BacktestResult:
    """Reconstruct a BacktestResult from a serialized dict."""
    metrics = None
    if result_dict["metrics"] is not None:
        metrics = PerformanceMetrics(**result_dict["metrics"])

    return BacktestResult(
        strategy_name=result_dict["strategy_name"],
        config=result_dict["config"],
        trades=result_dict["trades"],
        equity_curve=result_dict["equity_curve"],
        metrics=metrics,
    )


# ── Parallel Runner ──────────────────────────────────────────────────


class ParallelRunner:
    """
    Run many strategy backtests in parallel.

    Usage:
        runner = ParallelRunner(run_config, n_workers=8)
        batch_result = runner.run(strategies)

        # Get top strategies
        top = runner.rank_results(batch_result, metric="sharpe_ratio", top_n=20)
    """

    def __init__(self, run_config: RunConfig, n_workers: int | None = None):
        """
        Initialize parallel runner.
        n_workers: number of parallel processes (default: cpu_count - 1)
        """
        self.config = run_config
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

    def run(
        self,
        strategies: list,
        callback: Callable | None = None,
        chunk_size: int = 10,
    ) -> BatchResult:
        """
        Run all strategies in parallel.

        Uses Pool.imap_unordered for memory efficiency with large batches.
        callback(strategy_name, result_or_none, progress_pct) called after each.
        """
        total = len(strategies)
        if total == 0:
            return BatchResult(
                total_strategies=0,
                completed=0,
                failed=0,
                filtered=0,
                results=[],
                elapsed_seconds=0.0,
                errors=[],
            )

        logger.info(
            "Starting parallel backtest: %d strategies across %d workers",
            total,
            self.n_workers,
        )

        # Prepare serialized args for each strategy
        worker_args = [self._prepare_worker_args(s) for s in strategies]

        results: list[tuple] = []
        errors: list[tuple[str, str]] = []
        completed = 0
        failed = 0
        skipped_low_trades = 0

        start_time = time.monotonic()

        with Pool(processes=self.n_workers) as pool:
            for strategy_dict, result_dict, error in pool.imap_unordered(
                _run_single_backtest, worker_args, chunksize=chunk_size
            ):
                completed += 1
                strategy_name = strategy_dict.get("name", "unknown")

                if error is not None:
                    failed += 1
                    errors.append((strategy_name, error))
                    logger.warning(
                        "Strategy '%s' failed: %s",
                        strategy_name,
                        error.split("\n")[0],
                    )
                elif result_dict is None:
                    # Filtered out (too few trades)
                    skipped_low_trades += 1
                else:
                    # Reconstruct strategy and result
                    bt_result = _deserialize_result(result_dict)
                    metrics = bt_result.metrics

                    # Apply additional filters
                    passes_filter = True
                    if metrics is not None:
                        if (
                            self.config.min_profit_factor > 0
                            and metrics.profit_factor < self.config.min_profit_factor
                        ):
                            passes_filter = False
                        if metrics.sharpe_ratio < self.config.min_sharpe:
                            passes_filter = False

                    if passes_filter:
                        results.append((strategy_dict, bt_result))
                    else:
                        skipped_low_trades += 1

                # Progress logging
                elapsed = time.monotonic() - start_time
                pct = completed / total * 100
                rate = completed / elapsed if elapsed > 0 else 0

                if completed % max(1, total // 20) == 0 or completed == total:
                    logger.info(
                        "Completed %d/%d strategies (%.0f%%) - %.1f strategies/sec",
                        completed,
                        total,
                        pct,
                        rate,
                    )

                if callback is not None:
                    result_for_cb = (
                        _deserialize_result(result_dict) if result_dict else None
                    )
                    callback(strategy_name, result_for_cb, pct)

        elapsed_total = time.monotonic() - start_time

        filtered_count = len(results)

        logger.info(
            "Batch complete: %d/%d passed filters, %d failed, %.1fs elapsed (%.1f strats/sec)",
            filtered_count,
            total,
            failed,
            elapsed_total,
            total / elapsed_total if elapsed_total > 0 else 0,
        )

        return BatchResult(
            total_strategies=total,
            completed=completed - failed,
            failed=failed,
            filtered=filtered_count,
            results=results,
            elapsed_seconds=elapsed_total,
            errors=errors,
        )

    def run_sequential(self, strategies: list) -> BatchResult:
        """Run strategies sequentially (for debugging)."""
        total = len(strategies)
        if total == 0:
            return BatchResult(
                total_strategies=0,
                completed=0,
                failed=0,
                filtered=0,
                results=[],
                elapsed_seconds=0.0,
                errors=[],
            )

        logger.info("Starting sequential backtest: %d strategies", total)

        results: list[tuple] = []
        errors: list[tuple[str, str]] = []
        completed = 0
        failed = 0

        start_time = time.monotonic()

        for strategy in strategies:
            args = self._prepare_worker_args(strategy)
            strategy_dict, result_dict, error = _run_single_backtest(args)

            completed += 1
            strategy_name = strategy_dict.get("name", "unknown")

            if error is not None:
                failed += 1
                errors.append((strategy_name, error))
                logger.warning("Strategy '%s' failed: %s", strategy_name, error.split("\n")[0])
            elif result_dict is None:
                pass  # filtered out
            else:
                bt_result = _deserialize_result(result_dict)
                metrics = bt_result.metrics

                passes_filter = True
                if metrics is not None:
                    if (
                        self.config.min_profit_factor > 0
                        and metrics.profit_factor < self.config.min_profit_factor
                    ):
                        passes_filter = False
                    if metrics.sharpe_ratio < self.config.min_sharpe:
                        passes_filter = False

                if passes_filter:
                    results.append((strategy_dict, bt_result))

            # Progress
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Completed %d/%d strategies (%.0f%%) - %.1f strategies/sec",
                    completed,
                    total,
                    completed / total * 100,
                    rate,
                )

        elapsed_total = time.monotonic() - start_time

        return BatchResult(
            total_strategies=total,
            completed=completed - failed,
            failed=failed,
            filtered=len(results),
            results=results,
            elapsed_seconds=elapsed_total,
            errors=errors,
        )

    @staticmethod
    def rank_results(
        batch_result: BatchResult,
        metric: str = "sharpe_ratio",
        top_n: int = 20,
        ascending: bool = False,
    ) -> list:
        """
        Rank results by a metric. Returns top_n (strategy_dict, BacktestResult) tuples.

        Supported metrics: sharpe_ratio, profit_factor, net_pnl (total_pnl),
        win_rate, max_drawdown, consistency_score, composite.

        For "composite": weighted score of sharpe (40%) + profit_factor (30%) +
        consistency (20%) + win_rate (10%). Each metric is normalized to 0-1
        range across the batch before weighting.
        """
        results = batch_result.results
        if not results:
            return []

        if metric == "composite":
            return ParallelRunner._rank_composite(results, top_n, ascending)

        # Map friendly names to PerformanceMetrics attributes
        metric_map = {
            "sharpe_ratio": "sharpe_ratio",
            "sharpe": "sharpe_ratio",
            "profit_factor": "profit_factor",
            "pf": "profit_factor",
            "net_pnl": "total_pnl",
            "total_pnl": "total_pnl",
            "pnl": "total_pnl",
            "win_rate": "win_rate",
            "max_drawdown": "max_drawdown",
            "consistency": "consistency_score",
            "consistency_score": "consistency_score",
            "total_trades": "total_trades",
            "trades": "total_trades",
        }

        attr = metric_map.get(metric, metric)

        def sort_key(item):
            _, bt_result = item
            if bt_result.metrics is None:
                return float("-inf") if not ascending else float("inf")
            val = getattr(bt_result.metrics, attr, 0)
            # For max_drawdown, more negative = worse, so negate if descending
            if attr == "max_drawdown" and not ascending:
                return -abs(val)  # rank by smallest drawdown
            return val

        # For consistency_score, lower is better (less concentrated)
        reverse = not ascending
        if attr == "consistency_score":
            reverse = ascending  # lower consistency_score = more consistent = better

        sorted_results = sorted(results, key=sort_key, reverse=reverse)
        return sorted_results[:top_n]

    @staticmethod
    def _rank_composite(results: list, top_n: int, ascending: bool) -> list:
        """
        Rank by composite score: sharpe (40%) + profit_factor (30%) +
        consistency (20%) + win_rate (10%).

        Each metric is min-max normalized to [0, 1] across the batch.
        For consistency_score, lower is better so it is inverted.
        """
        if not results:
            return []

        # Extract raw metric values
        sharpes = []
        pfs = []
        consistencies = []
        win_rates = []

        for _, bt_result in results:
            m = bt_result.metrics
            if m is None:
                sharpes.append(0.0)
                pfs.append(0.0)
                consistencies.append(100.0)
                win_rates.append(0.0)
            else:
                sharpes.append(m.sharpe_ratio)
                pfs.append(m.profit_factor)
                consistencies.append(m.consistency_score)
                win_rates.append(m.win_rate)

        def normalize(values: list[float], invert: bool = False) -> list[float]:
            """Min-max normalize to [0, 1]. If invert, lower raw = higher score."""
            mn = min(values)
            mx = max(values)
            rng = mx - mn
            if rng == 0:
                return [0.5] * len(values)
            normed = [(v - mn) / rng for v in values]
            if invert:
                normed = [1.0 - n for n in normed]
            return normed

        norm_sharpe = normalize(sharpes)
        norm_pf = normalize(pfs)
        norm_consist = normalize(consistencies, invert=True)  # lower = better
        norm_wr = normalize(win_rates)

        # Weighted composite
        scores = []
        for i in range(len(results)):
            score = (
                0.40 * norm_sharpe[i]
                + 0.30 * norm_pf[i]
                + 0.20 * norm_consist[i]
                + 0.10 * norm_wr[i]
            )
            scores.append(score)

        # Pair with results and sort
        paired = list(zip(scores, results))
        paired.sort(key=lambda x: x[0], reverse=not ascending)

        return [item for _, item in paired[:top_n]]

    @staticmethod
    def print_leaderboard(ranked_results: list, top_n: int = 20):
        """Pretty-print a leaderboard table of top strategies."""
        if not ranked_results:
            print("No results to display.")
            return

        display = ranked_results[:top_n]

        # Header
        header = (
            f"{'Rank':>4}  "
            f"{'Strategy':<30}  "
            f"{'Trades':>6}  "
            f"{'Win%':>6}  "
            f"{'PF':>6}  "
            f"{'Sharpe':>7}  "
            f"{'Net P&L':>10}  "
            f"{'Max DD':>10}"
        )
        separator = "-" * len(header)

        print()
        print(separator)
        print("  STRATEGY LEADERBOARD")
        print(separator)
        print(header)
        print(separator)

        for rank, (strategy_dict, bt_result) in enumerate(display, 1):
            m = bt_result.metrics
            name = bt_result.strategy_name
            if len(name) > 30:
                name = name[:27] + "..."

            if m is None:
                print(f"{rank:>4}  {name:<30}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>7}  {'N/A':>10}  {'N/A':>10}")
                continue

            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "inf"

            print(
                f"{rank:>4}  "
                f"{name:<30}  "
                f"{m.total_trades:>6}  "
                f"{m.win_rate:>5.1f}%  "
                f"{pf_str:>6}  "
                f"{m.sharpe_ratio:>7.2f}  "
                f"${m.total_pnl:>9,.0f}  "
                f"${m.max_drawdown:>9,.0f}"
            )

        print(separator)
        print(f"  Showing top {len(display)} of {len(ranked_results)} strategies")
        print(separator)
        print()

    def _prepare_worker_args(self, strategy) -> tuple:
        """
        Serialize strategy and config for worker process.

        Strategies must implement one of:
        - to_dict() method returning a dict with at minimum 'name' and 'class_path'
        - __dict__ attribute (fallback, with manual class_path injection)
        """
        # Serialize strategy
        if hasattr(strategy, "to_dict"):
            strategy_dict = strategy.to_dict()
        else:
            # Fallback: build dict from object attributes
            strategy_dict = {}
            if hasattr(strategy, "__dict__"):
                # Copy public attributes
                strategy_dict = {
                    k: v
                    for k, v in strategy.__dict__.items()
                    if not k.startswith("_")
                }
            # Add class path for reconstruction
            cls = type(strategy)
            strategy_dict["class_path"] = f"{cls.__module__}.{cls.__qualname__}"
            strategy_dict.setdefault("name", getattr(strategy, "name", "unknown"))

        return (
            strategy_dict,
            self.config.data,
            self.config.config,
            self.config.prop_rules,
            self.config.session_config,
            self.config.events_calendar,
            self.config.contract_spec,
            self.config.min_trades,
        )
