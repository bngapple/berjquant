"""
Walk-forward optimization and validation for detecting overfitting.

The gold standard for validating that a trading strategy has real edge
rather than curve-fitted noise. Splits historical data into rolling
train/test windows, optionally optimizes on each training window, then
measures out-of-sample consistency across all test windows.

Key metric: Walk-Forward Efficiency = OOS Sharpe / IS Sharpe
    > 0.5 = good, > 0.7 = great, < 0.3 = likely overfit
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from engine.utils import BacktestConfig, BacktestResult, PerformanceMetrics

logger = logging.getLogger(__name__)


# ── Data Structures ─────────────────────────────────────────────────


@dataclass
class WFWindow:
    """A single walk-forward window (train + test period)."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    train_result: Any = None  # BacktestResult
    test_result: Any = None  # BacktestResult
    train_metrics: Any = None  # PerformanceMetrics
    test_metrics: Any = None  # PerformanceMetrics
    best_params: dict = field(default_factory=dict)


@dataclass
class WFResult:
    """Complete walk-forward validation result."""

    strategy_name: str
    windows: list[WFWindow]

    # Aggregate OOS metrics
    oos_trades: list  # All out-of-sample trades combined
    oos_metrics: Any  # PerformanceMetrics from combined OOS trades

    # Walk-forward efficiency
    wf_efficiency: float  # OOS Sharpe / IS Sharpe (>0.5 good, >0.7 great)
    wf_profit_ratio: float  # OOS PF / IS PF

    # Consistency across windows
    window_sharpes: list[float]  # Sharpe ratio per OOS window
    window_pnls: list[float]  # Net P&L per OOS window
    pct_profitable_windows: float  # % of OOS windows that are profitable
    sharpe_stability: float  # Std/Mean of window Sharpes (lower is better)

    # Overfitting flags
    is_overfit: bool
    overfit_reasons: list[str] = field(default_factory=list)


# ── Walk-Forward Validator ──────────────────────────────────────────


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.

    Splits data into rolling train/test windows, optimizes on train,
    validates on test, and measures out-of-sample consistency.

    Usage:
        validator = WalkForwardValidator(
            data={"1m": df},
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            config=backtest_config,
        )
        result = validator.validate(
            strategy=generated_strategy,
            train_days=60,
            test_days=20,
            step_days=20,
        )
        print(f"WF Efficiency: {result.wf_efficiency:.2f}")
        print(f"Overfit: {result.is_overfit}")
    """

    def __init__(
        self,
        data: dict[str, pl.DataFrame],
        risk_manager,
        contract_spec,
        config: BacktestConfig,
        account_size: float = 50000.0,
    ):
        self.data = data
        self.risk_manager = risk_manager
        self.contract_spec = contract_spec
        self.config = config
        self.account_size = account_size

    # ── Public API ──────────────────────────────────────────────────

    def validate(
        self,
        strategy,
        train_days: int = 60,
        test_days: int = 20,
        step_days: int = 20,
        optimize_params: bool = False,
        n_param_variations: int = 20,
    ) -> WFResult:
        """
        Run walk-forward validation.

        1. Generate rolling windows (train_days/test_days, stepping by step_days)
        2. For each window:
           a. Run backtest on train period
           b. (Optional) Optimize parameters on train period
           c. Run backtest on test period with best params
        3. Combine all OOS results
        4. Calculate WF efficiency and check for overfitting

        Args:
            strategy: Strategy object to validate.
            train_days: Training window size in trading days.
            test_days: Testing window size in trading days.
            step_days: How far to advance between windows.
            optimize_params: Whether to re-optimize params in each window.
            n_param_variations: Number of param variations to test during
                optimization.

        Returns:
            WFResult with aggregate metrics and overfitting assessment.
        """
        # Use primary timeframe to determine date boundaries
        primary_tf = self._get_primary_timeframe()
        df = self.data[primary_tf]

        windows = self._generate_windows(df, train_days, test_days, step_days)

        if not windows:
            logger.warning(
                "Not enough data for walk-forward windows "
                "(need %d + %d trading days, got fewer).",
                train_days,
                test_days,
            )
            return self._empty_result(strategy.name)

        logger.info(
            "Walk-forward: %d windows (train=%d, test=%d, step=%d days)",
            len(windows),
            train_days,
            test_days,
            step_days,
        )

        # Run each window
        for i, window in enumerate(windows):
            logger.info(
                "  Window %d/%d: train %s -> %s, test %s -> %s",
                i + 1,
                len(windows),
                window.train_start.strftime("%Y-%m-%d"),
                window.train_end.strftime("%Y-%m-%d"),
                window.test_start.strftime("%Y-%m-%d"),
                window.test_end.strftime("%Y-%m-%d"),
            )
            windows[i] = self._run_window(
                window, strategy, optimize_params, n_param_variations
            )

        # Aggregate OOS results
        oos_trades, oos_metrics, is_metrics_agg = self._aggregate_oos(windows)

        # Walk-forward efficiency
        wf_efficiency = self._safe_ratio(
            oos_metrics.sharpe_ratio if oos_metrics else 0.0,
            is_metrics_agg.sharpe_ratio if is_metrics_agg else 0.0,
        )

        wf_profit_ratio = self._safe_ratio(
            oos_metrics.profit_factor if oos_metrics else 0.0,
            is_metrics_agg.profit_factor if is_metrics_agg else 0.0,
        )

        # Per-window stats
        window_sharpes = []
        window_pnls = []
        for w in windows:
            if w.test_metrics and w.test_metrics.total_trades > 0:
                window_sharpes.append(w.test_metrics.sharpe_ratio)
                window_pnls.append(w.test_metrics.total_pnl)
            else:
                window_sharpes.append(0.0)
                window_pnls.append(0.0)

        profitable_windows = sum(1 for pnl in window_pnls if pnl > 0)
        pct_profitable = (
            profitable_windows / len(windows) * 100.0 if windows else 0.0
        )

        # Sharpe stability: std / mean (lower = more consistent)
        if len(window_sharpes) > 1 and np.mean(window_sharpes) != 0:
            sharpe_stability = float(
                np.std(window_sharpes, ddof=1) / abs(np.mean(window_sharpes))
            )
        else:
            sharpe_stability = float("inf")

        # Overfitting check
        is_overfit, overfit_reasons = self._check_overfitting(
            windows, oos_metrics, wf_efficiency, pct_profitable, sharpe_stability
        )

        return WFResult(
            strategy_name=strategy.name,
            windows=windows,
            oos_trades=oos_trades,
            oos_metrics=oos_metrics,
            wf_efficiency=wf_efficiency,
            wf_profit_ratio=wf_profit_ratio,
            window_sharpes=window_sharpes,
            window_pnls=window_pnls,
            pct_profitable_windows=pct_profitable,
            sharpe_stability=sharpe_stability,
            is_overfit=is_overfit,
            overfit_reasons=overfit_reasons,
        )

    def holdout_test(self, strategy, holdout_pct: float = 0.2) -> dict:
        """
        Simple holdout test: reserve last X% of data as OOS.

        Returns a dict with IS and OOS metrics for quick comparison.
        """
        primary_tf = self._get_primary_timeframe()
        df = self.data[primary_tf]
        trading_dates = self._get_trading_dates(df)

        if len(trading_dates) < 10:
            logger.warning("Too few trading days for holdout test (%d).", len(trading_dates))
            return {"error": "insufficient_data", "trading_days": len(trading_dates)}

        split_idx = int(len(trading_dates) * (1.0 - holdout_pct))
        is_end = trading_dates[split_idx - 1]
        oos_start = trading_dates[split_idx]

        # Slice and run IS
        is_data = self._slice_data(trading_dates[0], is_end)
        is_result = self._run_backtest(is_data, strategy)
        is_metrics = calculate_metrics(is_result.trades, self.account_size)

        # Slice and run OOS
        oos_data = self._slice_data(oos_start, trading_dates[-1])
        oos_result = self._run_backtest(oos_data, strategy)
        oos_metrics = calculate_metrics(oos_result.trades, self.account_size)

        # Efficiency
        wf_efficiency = self._safe_ratio(
            oos_metrics.sharpe_ratio, is_metrics.sharpe_ratio
        )

        return {
            "is_start": trading_dates[0].isoformat(),
            "is_end": is_end.isoformat(),
            "oos_start": oos_start.isoformat(),
            "oos_end": trading_dates[-1].isoformat(),
            "is_trades": is_metrics.total_trades,
            "is_sharpe": is_metrics.sharpe_ratio,
            "is_pf": is_metrics.profit_factor,
            "is_total_pnl": is_metrics.total_pnl,
            "oos_trades": oos_metrics.total_trades,
            "oos_sharpe": oos_metrics.sharpe_ratio,
            "oos_pf": oos_metrics.profit_factor,
            "oos_total_pnl": oos_metrics.total_pnl,
            "wf_efficiency": wf_efficiency,
        }

    def print_report(self, result: WFResult) -> None:
        """Pretty-print walk-forward validation report."""
        print()
        print("=" * 78)
        print("  WALK-FORWARD VALIDATION REPORT")
        print(f"  Strategy: {result.strategy_name}")
        print("=" * 78)

        # Window-by-window table
        print()
        print(
            f"  {'Win':>3}  {'Train Period':^23}  {'Test Period':^23}"
            f"  {'IS Sharpe':>9}  {'OOS Sharpe':>10}  {'OOS P&L':>10}  {'OOS Trades':>10}"
        )
        print("  " + "-" * 74)

        for w in result.windows:
            is_sharpe = w.train_metrics.sharpe_ratio if w.train_metrics else 0.0
            oos_sharpe = w.test_metrics.sharpe_ratio if w.test_metrics else 0.0
            oos_pnl = w.test_metrics.total_pnl if w.test_metrics else 0.0
            oos_trades = w.test_metrics.total_trades if w.test_metrics else 0

            train_range = (
                f"{w.train_start.strftime('%Y-%m-%d')}"
                f" - {w.train_end.strftime('%Y-%m-%d')}"
            )
            test_range = (
                f"{w.test_start.strftime('%Y-%m-%d')}"
                f" - {w.test_end.strftime('%Y-%m-%d')}"
            )

            print(
                f"  {w.window_id:>3}  {train_range:^23}  {test_range:^23}"
                f"  {is_sharpe:>9.2f}  {oos_sharpe:>10.2f}"
                f"  ${oos_pnl:>9,.2f}  {oos_trades:>10}"
            )

        # Aggregate
        print()
        print("  " + "-" * 74)
        print("  AGGREGATE OUT-OF-SAMPLE RESULTS")
        print("  " + "-" * 74)

        m = result.oos_metrics
        if m and m.total_trades > 0:
            print(f"  Total OOS Trades:        {m.total_trades}")
            print(f"  OOS Win Rate:            {m.win_rate:.1f}%")
            print(f"  OOS Total P&L:           ${m.total_pnl:,.2f}")
            print(f"  OOS Profit Factor:       {m.profit_factor:.2f}")
            print(f"  OOS Sharpe Ratio:        {m.sharpe_ratio:.2f}")
            print(f"  OOS Max Drawdown:        ${m.max_drawdown:,.2f}")
        else:
            print("  No OOS trades generated.")

        print()
        print("  WALK-FORWARD METRICS")
        print("  " + "-" * 74)
        print(f"  WF Efficiency (Sharpe):  {result.wf_efficiency:.2f}")
        print(f"  WF Profit Ratio:         {result.wf_profit_ratio:.2f}")
        print(f"  Profitable Windows:      {result.pct_profitable_windows:.0f}%")
        print(f"  Sharpe Stability (CV):   {result.sharpe_stability:.2f}")

        # Overfitting assessment
        print()
        if result.is_overfit:
            print("  ** OVERFITTING DETECTED **")
            for reason in result.overfit_reasons:
                print(f"     - {reason}")
        else:
            print("  No overfitting flags raised.")

        print("=" * 78)
        print()

    # ── Internal: Window Generation ─────────────────────────────────

    def _generate_windows(
        self,
        df: pl.DataFrame,
        train_days: int,
        test_days: int,
        step_days: int,
    ) -> list[WFWindow]:
        """Generate rolling train/test window specifications from actual trading dates."""
        trading_dates = self._get_trading_dates(df)
        total_dates = len(trading_dates)
        required = train_days + test_days

        if total_dates < required:
            return []

        windows: list[WFWindow] = []
        window_id = 1
        start_idx = 0

        while start_idx + required <= total_dates:
            train_start = trading_dates[start_idx]
            train_end = trading_dates[start_idx + train_days - 1]
            test_start = trading_dates[start_idx + train_days]
            test_end_idx = min(start_idx + train_days + test_days - 1, total_dates - 1)
            test_end = trading_dates[test_end_idx]

            windows.append(
                WFWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            window_id += 1
            start_idx += step_days

        return windows

    def _get_trading_dates(self, df: pl.DataFrame) -> list[datetime]:
        """Extract unique sorted trading dates from the data."""
        dates = (
            df.select(pl.col("timestamp").cast(pl.Date).alias("date"))
            .unique()
            .sort("date")
            .get_column("date")
            .to_list()
        )
        # Convert date objects to datetime at midnight for consistent comparison
        return [
            datetime(d.year, d.month, d.day) if not isinstance(d, datetime) else d
            for d in dates
        ]

    def _get_primary_timeframe(self) -> str:
        """Return the primary (smallest) timeframe key from data dict."""
        # Prefer "1m", then smallest available
        if "1m" in self.data:
            return "1m"
        keys = sorted(self.data.keys())
        return keys[0] if keys else "1m"

    # ── Internal: Data Slicing ──────────────────────────────────────

    def _slice_data(
        self, start: datetime, end: datetime
    ) -> dict[str, pl.DataFrame]:
        """Slice all timeframes to a date range (inclusive on both ends)."""
        sliced: dict[str, pl.DataFrame] = {}

        for tf, df in self.data.items():
            ts_col = pl.col("timestamp")
            # Filter: date portion of timestamp falls within [start, end]
            filtered = df.filter(
                (ts_col.cast(pl.Date) >= start.date())
                & (ts_col.cast(pl.Date) <= end.date())
            )
            sliced[tf] = filtered

        return sliced

    # ── Internal: Running a Window ──────────────────────────────────

    def _run_window(
        self,
        window: WFWindow,
        strategy,
        optimize: bool,
        n_variations: int,
    ) -> WFWindow:
        """Run train + test for a single window."""
        train_data = self._slice_data(window.train_start, window.train_end)
        test_data = self._slice_data(window.test_start, window.test_end)

        # Check that we actually have data
        primary_tf = self._get_primary_timeframe()
        if primary_tf not in train_data or train_data[primary_tf].height == 0:
            logger.warning("Window %d: no training data, skipping.", window.window_id)
            return window
        if primary_tf not in test_data or test_data[primary_tf].height == 0:
            logger.warning("Window %d: no test data, skipping.", window.window_id)
            return window

        # Determine which strategy to use for this window
        if optimize and n_variations > 1:
            best_strategy, best_params = self._optimize_on_train(
                strategy, train_data, n_variations
            )
            window.best_params = best_params
        else:
            best_strategy = strategy
            window.best_params = {}

        # Run on train data
        try:
            train_result = self._run_backtest(train_data, best_strategy)
            window.train_result = train_result
            window.train_metrics = calculate_metrics(
                train_result.trades, self.account_size
            )
        except Exception:
            logger.exception("Window %d: train backtest failed.", window.window_id)
            window.train_metrics = calculate_metrics([], self.account_size)

        # Run on test data
        try:
            test_result = self._run_backtest(test_data, best_strategy)
            window.test_result = test_result
            window.test_metrics = calculate_metrics(
                test_result.trades, self.account_size
            )
        except Exception:
            logger.exception("Window %d: test backtest failed.", window.window_id)
            window.test_metrics = calculate_metrics([], self.account_size)

        return window

    def _run_backtest(
        self, data: dict[str, pl.DataFrame], strategy
    ) -> BacktestResult:
        """Run a single backtest with the given data slice and strategy."""
        bt = VectorizedBacktester(
            data=data,
            risk_manager=self.risk_manager,
            contract_spec=self.contract_spec,
            config=self.config,
        )
        return bt.run(strategy)

    # ── Internal: Parameter Optimization ────────────────────────────

    def _optimize_on_train(
        self,
        strategy,
        train_data: dict[str, pl.DataFrame],
        n_variations: int,
    ) -> tuple:
        """
        Optimize strategy parameters on training data.

        Returns (best_strategy, best_params_dict).
        Uses StrategyGenerator.generate_parameter_variations() if the
        strategy supports it, otherwise falls back to running the
        strategy as-is.
        """
        # Try to generate parameter variations
        try:
            from strategies.generator import StrategyGenerator
            from signals.registry import SignalRegistry

            registry = SignalRegistry()
            generator = StrategyGenerator(registry)
            variations = generator.generate_parameter_variations(
                strategy, num_variations=n_variations, method="random"
            )
        except Exception:
            logger.debug(
                "Could not generate parameter variations; "
                "using strategy as-is for optimization."
            )
            variations = [strategy]

        # Ensure the original strategy is always included
        if strategy not in variations:
            variations.insert(0, strategy)

        best_strategy = strategy
        best_sharpe = -float("inf")
        best_params: dict = {}

        for variant in variations:
            try:
                result = self._run_backtest(train_data, variant)
                metrics = calculate_metrics(result.trades, self.account_size)

                # Require a minimum number of trades for statistical relevance
                if metrics.total_trades < 5:
                    continue

                if metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = metrics.sharpe_ratio
                    best_strategy = variant
                    best_params = self._extract_params(variant)
            except Exception:
                logger.debug(
                    "Variation %s failed during optimization, skipping.",
                    getattr(variant, "name", "unknown"),
                )
                continue

        logger.info(
            "  Optimization: tested %d variations, best Sharpe = %.2f",
            len(variations),
            best_sharpe if best_sharpe > -float("inf") else 0.0,
        )

        return best_strategy, best_params

    def _extract_params(self, strategy) -> dict:
        """Extract parameter values from a strategy for recording."""
        params: dict = {}
        if hasattr(strategy, "entry_signals"):
            for sig in strategy.entry_signals:
                for k, v in sig.get("params", {}).items():
                    params[f"{sig.get('signal_name', 'sig')}.{k}"] = v
        if hasattr(strategy, "entry_filters"):
            for filt in strategy.entry_filters:
                for k, v in filt.get("params", {}).items():
                    params[f"{filt.get('signal_name', 'filt')}.{k}"] = v
        return params

    # ── Internal: Aggregation ───────────────────────────────────────

    def _aggregate_oos(
        self, windows: list[WFWindow]
    ) -> tuple[list, PerformanceMetrics | None, PerformanceMetrics | None]:
        """
        Combine all OOS trades and compute aggregate metrics.

        Returns (oos_trades, oos_metrics, is_metrics_agg).
        """
        all_oos_trades = []
        all_is_trades = []

        for w in windows:
            if w.test_result and w.test_result.trades:
                all_oos_trades.extend(w.test_result.trades)
            if w.train_result and w.train_result.trades:
                all_is_trades.extend(w.train_result.trades)

        oos_metrics = calculate_metrics(all_oos_trades, self.account_size)
        is_metrics = calculate_metrics(all_is_trades, self.account_size)

        return all_oos_trades, oos_metrics, is_metrics

    # ── Internal: Overfitting Detection ─────────────────────────────

    def _check_overfitting(
        self,
        windows: list[WFWindow],
        oos_metrics: PerformanceMetrics | None,
        wf_efficiency: float,
        pct_profitable: float,
        sharpe_stability: float,
    ) -> tuple[bool, list[str]]:
        """
        Check for overfitting red flags.

        Flags are conservative -- they raise concerns for the user to
        investigate rather than hard-reject strategies.

        Checks:
            1. WF efficiency < 0.3 (OOS much worse than IS)
            2. Less than 40% of OOS windows profitable
            3. Large variance in OOS window performance (CV > 2.0)
            4. Sharpe drops > 60% from IS to OOS
            5. Too few trades for statistical significance
        """
        reasons: list[str] = []

        # 1. Low walk-forward efficiency
        if wf_efficiency < 0.3:
            reasons.append(
                f"WF efficiency is {wf_efficiency:.2f} (< 0.3 threshold). "
                "OOS performance is far below IS -- likely curve-fitted."
            )

        # 2. Too few profitable windows
        if len(windows) >= 3 and pct_profitable < 40.0:
            reasons.append(
                f"Only {pct_profitable:.0f}% of OOS windows are profitable "
                "(< 40% threshold). Strategy lacks consistency."
            )

        # 3. High variance across windows
        if len(windows) >= 3 and sharpe_stability > 2.0:
            reasons.append(
                f"Sharpe stability (CV) is {sharpe_stability:.2f} (> 2.0). "
                "Performance varies wildly across windows."
            )

        # 4. Large Sharpe drop from IS to OOS
        is_sharpes = []
        oos_sharpes = []
        for w in windows:
            if w.train_metrics and w.train_metrics.total_trades > 0:
                is_sharpes.append(w.train_metrics.sharpe_ratio)
            if w.test_metrics and w.test_metrics.total_trades > 0:
                oos_sharpes.append(w.test_metrics.sharpe_ratio)

        if is_sharpes and oos_sharpes:
            avg_is = float(np.mean(is_sharpes))
            avg_oos = float(np.mean(oos_sharpes))
            if avg_is > 0:
                drop_pct = (1.0 - avg_oos / avg_is) * 100.0
                if drop_pct > 60.0:
                    reasons.append(
                        f"Average Sharpe drops {drop_pct:.0f}% from IS ({avg_is:.2f}) "
                        f"to OOS ({avg_oos:.2f}). Significant degradation."
                    )

        # 5. Insufficient trades for statistical significance
        if oos_metrics and oos_metrics.total_trades < 30:
            reasons.append(
                f"Only {oos_metrics.total_trades} OOS trades total "
                "(< 30). Results may not be statistically significant."
            )

        is_overfit = len(reasons) >= 2  # Two or more flags = overfit
        return is_overfit, reasons

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Compute a ratio, handling zero/inf/negative denominators."""
        if denominator == 0 or not np.isfinite(denominator):
            return 0.0
        ratio = numerator / denominator
        # Clamp to a reasonable range
        return float(np.clip(ratio, -10.0, 10.0))

    def _empty_result(self, strategy_name: str) -> WFResult:
        """Return an empty WFResult when validation cannot proceed."""
        empty_metrics = calculate_metrics([], self.account_size)
        return WFResult(
            strategy_name=strategy_name,
            windows=[],
            oos_trades=[],
            oos_metrics=empty_metrics,
            wf_efficiency=0.0,
            wf_profit_ratio=0.0,
            window_sharpes=[],
            window_pnls=[],
            pct_profitable_windows=0.0,
            sharpe_stability=float("inf"),
            is_overfit=False,
            overfit_reasons=["Insufficient data for walk-forward analysis."],
        )
