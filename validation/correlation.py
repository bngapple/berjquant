"""
Correlation analysis and overfitting detection for strategy portfolios.

Provides two main capabilities:
1. CorrelationAnalyzer -- pairwise and portfolio-level correlation analysis
   to identify redundant strategies and build diversified portfolios.
2. OverfitDetector -- pragmatic overfitting detection using parameter
   sensitivity, time stability, IS/OOS degradation, and statistical red flags.

Usage:
    from validation.correlation import CorrelationAnalyzer, OverfitDetector

    # Portfolio correlation analysis
    analyzer = CorrelationAnalyzer()
    portfolio = analyzer.analyze_portfolio([
        ("EMA_RSI", trades_1),
        ("MACD_BB", trades_2),
        ("VWAP_DELTA", trades_3),
    ])
    analyzer.print_portfolio_report(portfolio)

    # Overfitting detection
    detector = OverfitDetector()
    report = detector.analyze("EMA_RSI", trades, param_variations=param_results)
    detector.print_report(report)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    """Pairwise correlation analysis between two strategies."""
    strategy_a: str
    strategy_b: str
    return_correlation: float       # Pearson correlation of daily returns
    trade_overlap_pct: float        # % of time both have open positions
    drawdown_correlation: float     # Do they draw down together?
    diversification_benefit: float  # Expected portfolio DD reduction (0-1)


@dataclass
class PortfolioAnalysis:
    """Analysis of a portfolio of strategies."""
    strategy_names: list[str]
    n_strategies: int

    # Correlation matrix
    correlation_matrix: np.ndarray  # NxN correlation matrix
    avg_correlation: float          # Mean pairwise correlation
    max_correlation: float          # Highest pairwise correlation

    # Pairwise details
    pairs: list[CorrelationResult]

    # Portfolio-level metrics
    portfolio_sharpe: float         # Combined portfolio Sharpe
    portfolio_max_dd: float         # Combined max drawdown
    diversification_ratio: float    # Sum of individual Sharpes / Portfolio Sharpe

    # Recommendations
    redundant_pairs: list[tuple[str, str]]  # Pairs with correlation > 0.7
    recommended_portfolio: list[str]        # Optimally diversified subset


@dataclass
class OverfitReport:
    """Comprehensive overfitting analysis for a strategy."""
    strategy_name: str

    # Parameter sensitivity
    param_sensitivity: float        # 0=stable, 1=fragile
    sensitive_params: list[str]     # Which parameters are most sensitive

    # Time stability
    is_time_stable: bool
    period_sharpes: list[float]
    period_labels: list[str]
    sharpe_decay: float             # Trend in Sharpe over time (negative = decaying)

    # IS vs OOS comparison
    is_sharpe: float
    oos_sharpe: float
    sharpe_degradation: float       # (IS - OOS) / IS

    # Red flags
    red_flags: list[str]
    overfit_probability: float      # 0-1 estimate
    verdict: str                    # "likely_overfit", "possibly_overfit", "likely_robust"


# ---------------------------------------------------------------------------
# CorrelationAnalyzer
# ---------------------------------------------------------------------------

class CorrelationAnalyzer:
    """
    Analyze correlations between strategies and build diversified portfolios.

    Usage:
        analyzer = CorrelationAnalyzer()
        portfolio = analyzer.analyze_portfolio(
            strategy_results=[
                ("EMA_RSI", trades_1),
                ("MACD_BB", trades_2),
                ("VWAP_DELTA", trades_3),
            ]
        )
        analyzer.print_correlation_matrix(portfolio)
    """

    def __init__(self, account_size: float = 50000.0):
        self.account_size = account_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_pair(
        self,
        name_a: str,
        trades_a: list,
        name_b: str,
        trades_b: list,
    ) -> CorrelationResult:
        """
        Analyze correlation between two strategies.

        Computes:
        - Daily return correlation (Pearson)
        - Trade time overlap percentage
        - Drawdown correlation (do they draw down together?)
        - Diversification benefit estimate
        """
        returns_a = self._trades_to_daily_returns(trades_a)
        returns_b = self._trades_to_daily_returns(trades_b)

        aligned_a, aligned_b = self._align_daily_returns(returns_a, returns_b)

        # Pearson correlation of daily returns
        if len(aligned_a) < 2:
            return_corr = 0.0
        else:
            std_a = np.std(aligned_a)
            std_b = np.std(aligned_b)
            if std_a > 0 and std_b > 0:
                return_corr = float(np.corrcoef(aligned_a, aligned_b)[0, 1])
            else:
                return_corr = 0.0

        # Trade overlap
        trade_overlap = self._compute_trade_overlap(trades_a, trades_b)

        # Drawdown correlation
        dd_a = self._compute_drawdown_series(returns_a)
        dd_b = self._compute_drawdown_series(returns_b)
        # Align drawdown series to common dates
        common_dates = sorted(set(returns_a.keys()) & set(returns_b.keys()))
        if len(common_dates) >= 2:
            all_dates_a = sorted(returns_a.keys())
            all_dates_b = sorted(returns_b.keys())
            dd_map_a = dict(zip(all_dates_a, dd_a))
            dd_map_b = dict(zip(all_dates_b, dd_b))
            dd_aligned_a = np.array([dd_map_a.get(d, 0.0) for d in common_dates])
            dd_aligned_b = np.array([dd_map_b.get(d, 0.0) for d in common_dates])
            std_dd_a = np.std(dd_aligned_a)
            std_dd_b = np.std(dd_aligned_b)
            if std_dd_a > 0 and std_dd_b > 0:
                dd_corr = float(np.corrcoef(dd_aligned_a, dd_aligned_b)[0, 1])
            else:
                dd_corr = 0.0
        else:
            dd_corr = 0.0

        # Diversification benefit: estimated portfolio DD reduction
        # Higher when correlation is lower: benefit = (1 - corr) / 2
        div_benefit = max(0.0, min(1.0, (1.0 - return_corr) / 2.0))

        return CorrelationResult(
            strategy_a=name_a,
            strategy_b=name_b,
            return_correlation=round(return_corr, 4),
            trade_overlap_pct=round(trade_overlap, 2),
            drawdown_correlation=round(dd_corr, 4),
            diversification_benefit=round(div_benefit, 4),
        )

    def analyze_portfolio(
        self,
        strategy_results: list[tuple[str, list]],
    ) -> PortfolioAnalysis:
        """
        Analyze a portfolio of strategies.

        Args:
            strategy_results: list of (strategy_name, trades) tuples

        Returns PortfolioAnalysis with correlation matrix, diversification
        metrics, and recommendations for redundant/overlapping strategies.
        """
        names = [name for name, _ in strategy_results]
        n = len(names)

        # Handle single-strategy portfolio
        if n == 0:
            return PortfolioAnalysis(
                strategy_names=[],
                n_strategies=0,
                correlation_matrix=np.array([]),
                avg_correlation=0.0,
                max_correlation=0.0,
                pairs=[],
                portfolio_sharpe=0.0,
                portfolio_max_dd=0.0,
                diversification_ratio=1.0,
                redundant_pairs=[],
                recommended_portfolio=[],
            )

        if n == 1:
            sharpe, max_dd = self._compute_portfolio_metrics(strategy_results)
            return PortfolioAnalysis(
                strategy_names=names,
                n_strategies=1,
                correlation_matrix=np.array([[1.0]]),
                avg_correlation=0.0,
                max_correlation=0.0,
                pairs=[],
                portfolio_sharpe=sharpe,
                portfolio_max_dd=max_dd,
                diversification_ratio=1.0,
                redundant_pairs=[],
                recommended_portfolio=names[:],
            )

        # Build correlation matrix and pairwise results
        corr_matrix = np.eye(n)
        pairs: list[CorrelationResult] = []

        for i in range(n):
            for j in range(i + 1, n):
                result = self.analyze_pair(
                    names[i], strategy_results[i][1],
                    names[j], strategy_results[j][1],
                )
                pairs.append(result)
                corr_matrix[i, j] = result.return_correlation
                corr_matrix[j, i] = result.return_correlation

        # Pairwise correlation stats (upper triangle only)
        upper_corrs = [corr_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_corr = float(np.mean(upper_corrs)) if upper_corrs else 0.0
        max_corr = float(np.max(upper_corrs)) if upper_corrs else 0.0

        # Portfolio-level metrics
        portfolio_sharpe, portfolio_max_dd = self._compute_portfolio_metrics(
            strategy_results,
        )

        # Diversification ratio: sum of individual Sharpes / portfolio Sharpe
        individual_sharpes = []
        for name, trades in strategy_results:
            s = self._compute_sharpe(trades)
            individual_sharpes.append(s)

        sum_sharpes = sum(individual_sharpes)
        if portfolio_sharpe > 0:
            div_ratio = sum_sharpes / portfolio_sharpe
        else:
            div_ratio = 1.0

        # Identify redundant pairs (correlation > 0.7)
        redundant_pairs = [
            (p.strategy_a, p.strategy_b)
            for p in pairs
            if p.return_correlation > 0.7
        ]

        # Select diversified portfolio
        recommended = self._select_diversified_portfolio(
            corr_matrix, names,
        )

        return PortfolioAnalysis(
            strategy_names=names,
            n_strategies=n,
            correlation_matrix=corr_matrix,
            avg_correlation=round(avg_corr, 4),
            max_correlation=round(max_corr, 4),
            pairs=pairs,
            portfolio_sharpe=round(portfolio_sharpe, 4),
            portfolio_max_dd=round(portfolio_max_dd, 2),
            diversification_ratio=round(div_ratio, 4),
            redundant_pairs=redundant_pairs,
            recommended_portfolio=recommended,
        )

    # ------------------------------------------------------------------
    # Internal: daily return helpers
    # ------------------------------------------------------------------

    def _trades_to_daily_returns(self, trades: list) -> dict[str, float]:
        """Convert trades to daily P&L dict: date_str -> daily_pnl."""
        daily: dict[str, float] = defaultdict(float)
        for t in trades:
            if isinstance(t.exit_time, datetime):
                day = t.exit_time.strftime("%Y-%m-%d")
            else:
                day = str(t.exit_time)[:10]
            daily[day] += t.net_pnl
        return dict(daily)

    def _align_daily_returns(
        self,
        returns_a: dict[str, float],
        returns_b: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two daily return series to common dates, filling missing with 0."""
        all_dates = sorted(set(returns_a.keys()) | set(returns_b.keys()))
        if not all_dates:
            return np.array([]), np.array([])
        arr_a = np.array([returns_a.get(d, 0.0) for d in all_dates])
        arr_b = np.array([returns_b.get(d, 0.0) for d in all_dates])
        return arr_a, arr_b

    def _compute_drawdown_series(self, daily_returns: dict[str, float]) -> np.ndarray:
        """Compute drawdown series from daily returns."""
        if not daily_returns:
            return np.array([0.0])
        sorted_dates = sorted(daily_returns.keys())
        pnls = np.array([daily_returns[d] for d in sorted_dates])
        equity = self.account_size + np.cumsum(pnls)
        peak = np.maximum.accumulate(np.concatenate([[self.account_size], equity]))[1:]
        drawdown = equity - peak
        return drawdown

    def _compute_trade_overlap(self, trades_a: list, trades_b: list) -> float:
        """Compute % of time both strategies have open positions simultaneously."""
        if not trades_a or not trades_b:
            return 0.0

        def _to_intervals(trades: list) -> list[tuple[datetime, datetime]]:
            intervals = []
            for t in trades:
                entry = t.entry_time if isinstance(t.entry_time, datetime) else t.entry_time
                exit_ = t.exit_time if isinstance(t.exit_time, datetime) else t.exit_time
                if entry and exit_:
                    intervals.append((entry, exit_))
            return intervals

        intervals_a = _to_intervals(trades_a)
        intervals_b = _to_intervals(trades_b)

        if not intervals_a or not intervals_b:
            return 0.0

        # Find overall time range
        all_starts = [i[0] for i in intervals_a] + [i[0] for i in intervals_b]
        all_ends = [i[1] for i in intervals_a] + [i[1] for i in intervals_b]
        global_start = min(all_starts)
        global_end = max(all_ends)
        total_seconds = (global_end - global_start).total_seconds()
        if total_seconds <= 0:
            return 0.0

        # Compute overlap using sorted event sweep
        # Build events: +1 when position opens, -1 when closes, per strategy
        events_a: list[tuple[datetime, int]] = []
        for entry, exit_ in intervals_a:
            events_a.append((entry, 1))
            events_a.append((exit_, -1))
        events_a.sort(key=lambda x: x[0])

        events_b: list[tuple[datetime, int]] = []
        for entry, exit_ in intervals_b:
            events_b.append((entry, 1))
            events_b.append((exit_, -1))
        events_b.sort(key=lambda x: x[0])

        # Merge all events and sweep
        all_events: list[tuple[datetime, str, int]] = []
        for t, delta in events_a:
            all_events.append((t, "a", delta))
        for t, delta in events_b:
            all_events.append((t, "b", delta))
        all_events.sort(key=lambda x: x[0])

        count_a = 0
        count_b = 0
        overlap_seconds = 0.0
        prev_time = global_start
        both_open = False

        for event_time, source, delta in all_events:
            if both_open and event_time > prev_time:
                overlap_seconds += (event_time - prev_time).total_seconds()

            if source == "a":
                count_a += delta
            else:
                count_b += delta

            both_open = count_a > 0 and count_b > 0
            prev_time = event_time

        return (overlap_seconds / total_seconds) * 100.0

    # ------------------------------------------------------------------
    # Internal: Sharpe helpers
    # ------------------------------------------------------------------

    def _compute_sharpe(self, trades: list) -> float:
        """Compute annualized Sharpe ratio for a list of trades."""
        daily = self._trades_to_daily_returns(trades)
        if len(daily) < 2:
            return 0.0
        values = np.array(list(daily.values()))
        mean_d = np.mean(values)
        std_d = np.std(values, ddof=1)
        if std_d <= 0:
            return 0.0
        return float(mean_d / std_d * np.sqrt(252))

    def _compute_portfolio_metrics(
        self,
        strategy_results: list[tuple[str, list]],
    ) -> tuple[float, float]:
        """Compute combined portfolio Sharpe and max drawdown."""
        # Combine daily returns across all strategies
        combined_daily: dict[str, float] = defaultdict(float)
        for _, trades in strategy_results:
            daily = self._trades_to_daily_returns(trades)
            for date, pnl in daily.items():
                combined_daily[date] += pnl

        if not combined_daily:
            return 0.0, 0.0

        sorted_dates = sorted(combined_daily.keys())
        daily_pnls = np.array([combined_daily[d] for d in sorted_dates])

        # Portfolio Sharpe
        if len(daily_pnls) < 2:
            portfolio_sharpe = 0.0
        else:
            mean_d = np.mean(daily_pnls)
            std_d = np.std(daily_pnls, ddof=1)
            portfolio_sharpe = float(mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0

        # Portfolio max drawdown
        equity = self.account_size + np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(
            np.concatenate([[self.account_size], equity]),
        )
        drawdown = np.concatenate([[self.account_size], equity]) - peak
        portfolio_max_dd = float(drawdown.min())

        return portfolio_sharpe, portfolio_max_dd

    # ------------------------------------------------------------------
    # Internal: diversified portfolio selection
    # ------------------------------------------------------------------

    def _select_diversified_portfolio(
        self,
        correlation_matrix: np.ndarray,
        names: list[str],
        max_strategies: int = 5,
        max_correlation: float = 0.5,
    ) -> list[str]:
        """
        Select optimally diversified subset of strategies.

        Greedy algorithm:
        1. Start with the first strategy (assumed best Sharpe by convention,
           or pick the one with lowest average correlation if no Sharpe info).
        2. Iteratively add the strategy with lowest average correlation
           to already-selected strategies.
        3. Stop if adding next strategy would exceed max_correlation threshold
           with any selected strategy, or we hit max_strategies.
        """
        n = len(names)
        if n <= 1:
            return names[:]

        # Start with strategy index 0 (caller can pre-sort by Sharpe)
        selected_indices = [0]
        remaining = set(range(1, n))

        while len(selected_indices) < min(max_strategies, n) and remaining:
            best_candidate = None
            best_avg_corr = float("inf")

            for candidate in remaining:
                # Check max correlation constraint
                max_corr_to_selected = max(
                    abs(correlation_matrix[candidate, s]) for s in selected_indices
                )
                if max_corr_to_selected > max_correlation:
                    continue

                # Average correlation to selected strategies
                avg_corr = float(np.mean([
                    abs(correlation_matrix[candidate, s]) for s in selected_indices
                ]))
                if avg_corr < best_avg_corr:
                    best_avg_corr = avg_corr
                    best_candidate = candidate

            if best_candidate is None:
                break  # No candidate meets the correlation threshold

            selected_indices.append(best_candidate)
            remaining.discard(best_candidate)

        return [names[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_correlation_matrix(self, portfolio: PortfolioAnalysis):
        """Pretty-print the correlation matrix."""
        names = portfolio.strategy_names
        n = portfolio.n_strategies
        matrix = portfolio.correlation_matrix

        if n == 0:
            print("  No strategies to display.")
            return

        # Determine column width
        max_name_len = max(len(name) for name in names)
        col_w = max(max_name_len + 1, 10)

        print("\n" + "=" * (col_w + n * (col_w + 1)))
        print("  CORRELATION MATRIX")
        print("=" * (col_w + n * (col_w + 1)))

        # Header row
        header = " " * col_w + "".join(f"{name:>{col_w}s}" for name in names)
        print(header)

        # Data rows
        for i in range(n):
            row = f"{names[i]:<{col_w}s}"
            for j in range(n):
                val = matrix[i, j]
                row += f"{val:>{col_w}.3f}"
            print(row)

        print()

    def print_portfolio_report(self, portfolio: PortfolioAnalysis):
        """Print full portfolio analysis report."""
        print("\n" + "=" * 60)
        print("  PORTFOLIO CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"  Strategies:           {portfolio.n_strategies}")
        print(f"  Avg Correlation:      {portfolio.avg_correlation:.3f}")
        print(f"  Max Correlation:      {portfolio.max_correlation:.3f}")
        print(f"  Portfolio Sharpe:     {portfolio.portfolio_sharpe:.2f}")
        print(f"  Portfolio Max DD:     ${portfolio.portfolio_max_dd:,.2f}")
        print(f"  Diversification:      {portfolio.diversification_ratio:.2f}x")
        print("-" * 60)

        if portfolio.redundant_pairs:
            print("  REDUNDANT PAIRS (correlation > 0.7):")
            for a, b in portfolio.redundant_pairs:
                pair = next(
                    (p for p in portfolio.pairs
                     if (p.strategy_a == a and p.strategy_b == b)
                     or (p.strategy_a == b and p.strategy_b == a)),
                    None,
                )
                corr = pair.return_correlation if pair else 0.0
                print(f"    {a} <-> {b}  (r={corr:.3f})")
        else:
            print("  No redundant pairs detected.")

        print("-" * 60)
        print("  RECOMMENDED PORTFOLIO:")
        for name in portfolio.recommended_portfolio:
            print(f"    + {name}")

        # Print correlation matrix
        self.print_correlation_matrix(portfolio)

        # Pairwise details
        if portfolio.pairs:
            print("  PAIRWISE DETAILS:")
            print(f"  {'Pair':<30s} {'RetCorr':>8s} {'Overlap':>8s} {'DDCorr':>8s} {'DivBen':>8s}")
            print("  " + "-" * 62)
            for p in portfolio.pairs:
                pair_name = f"{p.strategy_a} / {p.strategy_b}"
                print(
                    f"  {pair_name:<30s} "
                    f"{p.return_correlation:>8.3f} "
                    f"{p.trade_overlap_pct:>7.1f}% "
                    f"{p.drawdown_correlation:>8.3f} "
                    f"{p.diversification_benefit:>8.3f}"
                )

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# OverfitDetector
# ---------------------------------------------------------------------------

class OverfitDetector:
    """
    Detect overfitting in trading strategies.

    Usage:
        detector = OverfitDetector()
        report = detector.analyze(
            strategy_name="EMA_RSI",
            trades=trades,
            wf_result=walk_forward_result,       # optional
            param_variations=param_var_results,   # optional
        )
        detector.print_report(report)
    """

    def __init__(self, account_size: float = 50000.0):
        self.account_size = account_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        strategy_name: str,
        trades: list,
        data: Any = None,
        wf_result: Any = None,
        param_variations: list[tuple[dict, list]] | None = None,
    ) -> OverfitReport:
        """
        Comprehensive overfitting analysis.

        Checks:
        1. Parameter sensitivity (if param_variations provided)
        2. Time stability (split trades into N periods, compare)
        3. IS vs OOS degradation (if WF result provided)
        4. Statistical significance (enough trades?)
        5. Return distribution anomalies (too good to be true?)
        """
        red_flags: list[str] = []

        # 1. Parameter sensitivity
        if param_variations and len(param_variations) >= 2:
            param_sensitivity, sensitive_params = self._check_param_sensitivity(
                param_variations,
            )
            if param_sensitivity > 0.5:
                red_flags.append(
                    f"High parameter sensitivity ({param_sensitivity:.2f}): "
                    f"fragile to param changes"
                )
        else:
            param_sensitivity = 0.0
            sensitive_params = []

        # 2. Time stability
        is_stable, period_sharpes, period_labels, sharpe_decay = (
            self._check_time_stability(trades)
        )
        if not is_stable:
            red_flags.append("Performance inconsistent across time periods")
        if sharpe_decay < -0.5:
            red_flags.append(
                f"Sharpe ratio decaying over time (slope={sharpe_decay:.2f})"
            )

        # 3. IS vs OOS from walk-forward result
        is_sharpe = 0.0
        oos_sharpe = 0.0
        sharpe_degradation = 0.0

        if wf_result is not None:
            is_sharpe, oos_sharpe, sharpe_degradation = self._extract_wf_metrics(
                wf_result,
            )
            if sharpe_degradation > 0.7:
                red_flags.append(
                    f"Severe IS/OOS degradation: IS Sharpe={is_sharpe:.2f}, "
                    f"OOS Sharpe={oos_sharpe:.2f} ({sharpe_degradation:.0%} drop)"
                )
            elif sharpe_degradation > 0.5:
                red_flags.append(
                    f"Moderate IS/OOS degradation ({sharpe_degradation:.0%} drop)"
                )

        # 4. Statistical significance
        sig_flags = self._check_significance(trades)
        red_flags.extend(sig_flags)

        # 5. Compute overall probability
        overfit_prob = self._compute_overfit_probability(
            red_flags, param_sensitivity, sharpe_degradation,
        )
        verdict = self._determine_verdict(overfit_prob)

        return OverfitReport(
            strategy_name=strategy_name,
            param_sensitivity=round(param_sensitivity, 4),
            sensitive_params=sensitive_params,
            is_time_stable=is_stable,
            period_sharpes=period_sharpes,
            period_labels=period_labels,
            sharpe_decay=round(sharpe_decay, 4),
            is_sharpe=round(is_sharpe, 4),
            oos_sharpe=round(oos_sharpe, 4),
            sharpe_degradation=round(sharpe_degradation, 4),
            red_flags=red_flags,
            overfit_probability=round(overfit_prob, 4),
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Internal: parameter sensitivity
    # ------------------------------------------------------------------

    def _check_param_sensitivity(
        self,
        param_variations: list[tuple[dict, list]],
    ) -> tuple[float, list[str]]:
        """
        Measure how much performance changes with parameter variations.

        For each variation, compute Sharpe and compare to baseline (first entry).
        Sensitivity = std(sharpes) / abs(mean(sharpes)) if mean > 0.

        Returns (sensitivity_score, list_of_sensitive_parameter_names).
        """
        if len(param_variations) < 2:
            return 0.0, []

        # Compute Sharpe for each variation
        sharpes: list[float] = []
        for _, trades in param_variations:
            s = self._compute_sharpe(trades)
            sharpes.append(s)

        sharpes_arr = np.array(sharpes)
        mean_sharpe = np.mean(sharpes_arr)
        std_sharpe = np.std(sharpes_arr)

        if abs(mean_sharpe) > 0:
            sensitivity = float(std_sharpe / abs(mean_sharpe))
        else:
            sensitivity = 1.0 if std_sharpe > 0 else 0.0

        sensitivity = min(sensitivity, 1.0)

        # Identify which parameters are most sensitive
        # Compare each variation's params to the baseline
        baseline_params, baseline_trades = param_variations[0]
        baseline_sharpe = sharpes[0]
        sensitive_params: list[str] = []

        for i in range(1, len(param_variations)):
            params_i, _ = param_variations[i]
            sharpe_i = sharpes[i]
            sharpe_change = abs(sharpe_i - baseline_sharpe)

            if sharpe_change > 0.5:  # Meaningful change
                # Find which params differ
                for key in params_i:
                    if key in baseline_params:
                        if params_i[key] != baseline_params[key]:
                            if key not in sensitive_params:
                                sensitive_params.append(key)

        return sensitivity, sensitive_params

    # ------------------------------------------------------------------
    # Internal: time stability
    # ------------------------------------------------------------------

    def _check_time_stability(
        self,
        trades: list,
        n_periods: int = 4,
    ) -> tuple[bool, list[float], list[str], float]:
        """
        Split trades into N equal periods and check performance consistency.

        Returns (is_stable, period_sharpes, period_labels, sharpe_decay).
        Stability criterion: no period Sharpe is negative while others are
        positive, and the coefficient of variation of period Sharpes < 1.0.
        """
        if len(trades) < n_periods * 2:
            # Not enough trades to split meaningfully
            sharpe = self._compute_sharpe(trades)
            label = "all"
            return True, [sharpe], [label], 0.0

        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        chunk_size = len(sorted_trades) // n_periods

        period_sharpes: list[float] = []
        period_labels: list[str] = []

        for p in range(n_periods):
            start_idx = p * chunk_size
            if p == n_periods - 1:
                end_idx = len(sorted_trades)
            else:
                end_idx = (p + 1) * chunk_size

            period_trades = sorted_trades[start_idx:end_idx]
            if not period_trades:
                continue

            sharpe = self._compute_sharpe(period_trades)
            period_sharpes.append(sharpe)

            # Label with date range
            first_date = period_trades[0].exit_time
            last_date = period_trades[-1].exit_time
            if isinstance(first_date, datetime):
                first_str = first_date.strftime("%Y-%m-%d")
            else:
                first_str = str(first_date)[:10]
            if isinstance(last_date, datetime):
                last_str = last_date.strftime("%Y-%m-%d")
            else:
                last_str = str(last_date)[:10]
            period_labels.append(f"{first_str} to {last_str}")

        if len(period_sharpes) < 2:
            return True, period_sharpes, period_labels, 0.0

        sharpes_arr = np.array(period_sharpes)

        # Sharpe decay: linear regression slope
        x = np.arange(len(sharpes_arr), dtype=np.float64)
        if len(x) >= 2:
            coeffs = np.polyfit(x, sharpes_arr, 1)
            sharpe_decay = float(coeffs[0])
        else:
            sharpe_decay = 0.0

        # Stability check
        mean_s = np.mean(sharpes_arr)
        std_s = np.std(sharpes_arr)

        if abs(mean_s) > 0:
            cv = std_s / abs(mean_s)
        else:
            cv = float("inf")

        # Unstable if: high CV or sign flips (some positive, some negative)
        has_positive = bool(np.any(sharpes_arr > 0))
        has_negative = bool(np.any(sharpes_arr < 0))
        sign_flip = has_positive and has_negative

        is_stable = cv < 1.0 and not sign_flip

        return is_stable, [round(s, 4) for s in period_sharpes], period_labels, sharpe_decay

    # ------------------------------------------------------------------
    # Internal: walk-forward metrics extraction
    # ------------------------------------------------------------------

    def _extract_wf_metrics(self, wf_result: Any) -> tuple[float, float, float]:
        """
        Extract IS/OOS Sharpe from a walk-forward result.

        Supports WFResult objects with .folds attribute (list of fold objects
        with .is_sharpe / .oos_sharpe), or dicts with similar structure.

        Returns (is_sharpe, oos_sharpe, sharpe_degradation).
        """
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        folds = getattr(wf_result, "folds", None)
        if folds is None and isinstance(wf_result, dict):
            folds = wf_result.get("folds", [])

        if folds:
            for fold in folds:
                if hasattr(fold, "is_sharpe"):
                    is_sharpes.append(fold.is_sharpe)
                    oos_sharpes.append(fold.oos_sharpe)
                elif isinstance(fold, dict):
                    is_sharpes.append(fold.get("is_sharpe", 0.0))
                    oos_sharpes.append(fold.get("oos_sharpe", 0.0))
        else:
            # Try top-level attributes
            is_s = getattr(wf_result, "is_sharpe", None)
            oos_s = getattr(wf_result, "oos_sharpe", None)
            if is_s is not None:
                is_sharpes.append(is_s)
            if oos_s is not None:
                oos_sharpes.append(oos_s)

        if not is_sharpes:
            return 0.0, 0.0, 0.0

        avg_is = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))

        if avg_is > 0:
            degradation = (avg_is - avg_oos) / avg_is
        else:
            degradation = 0.0

        return avg_is, avg_oos, max(0.0, degradation)

    # ------------------------------------------------------------------
    # Internal: statistical significance
    # ------------------------------------------------------------------

    def _check_significance(self, trades: list) -> list[str]:
        """Check statistical significance. Returns list of red flags."""
        flags: list[str] = []

        if not trades:
            flags.append("No trades to analyze")
            return flags

        n_trades = len(trades)

        # Too few trades
        if n_trades < 30:
            flags.append(
                f"Too few trades ({n_trades}): need >= 30 for statistical significance"
            )

        pnls = np.array([t.net_pnl for t in trades])
        winners = pnls[pnls > 0]
        n_winners = len(winners)

        # Win rate suspiciously high
        win_rate = n_winners / n_trades * 100 if n_trades > 0 else 0.0
        if win_rate > 80:
            flags.append(f"Suspiciously high win rate ({win_rate:.1f}%)")

        # Sharpe suspiciously high
        sharpe = self._compute_sharpe(trades)
        if sharpe > 4.0:
            flags.append(f"Suspiciously high Sharpe ratio ({sharpe:.2f})")

        # Returns concentrated in few trades
        total_pnl = float(pnls.sum())
        if total_pnl > 0 and n_trades >= 5:
            sorted_pnls = np.sort(pnls)[::-1]  # Descending
            top_n = max(1, n_trades // 10)  # Top 10% of trades
            top_pnl = float(sorted_pnls[:top_n].sum())
            concentration = top_pnl / total_pnl
            if concentration > 0.8:
                flags.append(
                    f"Returns concentrated in top {top_n} trades "
                    f"({concentration:.0%} of total P&L)"
                )

        # Profit factor too good
        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        losers = pnls[pnls <= 0]
        gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
        if gross_loss > 0:
            pf = gross_profit / gross_loss
            if pf > 5.0:
                flags.append(f"Suspiciously high profit factor ({pf:.2f})")

        return flags

    # ------------------------------------------------------------------
    # Internal: Sharpe helper
    # ------------------------------------------------------------------

    def _compute_sharpe(self, trades: list) -> float:
        """Compute annualized Sharpe ratio from trades."""
        if not trades:
            return 0.0
        daily: dict[str, float] = defaultdict(float)
        for t in trades:
            if isinstance(t.exit_time, datetime):
                day = t.exit_time.strftime("%Y-%m-%d")
            else:
                day = str(t.exit_time)[:10]
            daily[day] += t.net_pnl

        if len(daily) < 2:
            return 0.0
        values = np.array(list(daily.values()))
        mean_d = np.mean(values)
        std_d = np.std(values, ddof=1)
        if std_d <= 0:
            return 0.0
        return float(mean_d / std_d * np.sqrt(252))

    # ------------------------------------------------------------------
    # Internal: overfit probability
    # ------------------------------------------------------------------

    def _compute_overfit_probability(
        self,
        red_flags: list[str],
        param_sensitivity: float,
        sharpe_degradation: float,
    ) -> float:
        """
        Estimate probability of overfitting (0-1).

        Pragmatic weighted approach:
        - Each red flag contributes ~0.12
        - High param sensitivity adds up to 0.2
        - High IS/OOS degradation adds up to 0.3
        - Clamped to [0, 1]
        """
        prob = 0.0

        # Red flag contributions
        n_flags = len(red_flags)
        prob += n_flags * 0.12

        # Parameter sensitivity (0 to 0.2)
        prob += param_sensitivity * 0.2

        # IS/OOS degradation (0 to 0.3)
        prob += min(sharpe_degradation, 1.0) * 0.3

        return min(1.0, max(0.0, prob))

    def _determine_verdict(self, probability: float) -> str:
        """Map overfitting probability to verdict string."""
        if probability >= 0.7:
            return "likely_overfit"
        elif probability >= 0.4:
            return "possibly_overfit"
        return "likely_robust"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: OverfitReport):
        """Pretty-print overfitting analysis report."""
        print("\n" + "=" * 60)
        print("  OVERFITTING ANALYSIS")
        print("=" * 60)
        print(f"  Strategy:             {report.strategy_name}")
        print(f"  Verdict:              {report.verdict.upper()}")
        print(f"  Overfit Probability:  {report.overfit_probability:.0%}")
        print("-" * 60)

        # Parameter sensitivity
        print(f"  Param Sensitivity:    {report.param_sensitivity:.2f}")
        if report.sensitive_params:
            print(f"  Sensitive Params:     {', '.join(report.sensitive_params)}")

        # Time stability
        print(f"  Time Stable:          {'Yes' if report.is_time_stable else 'No'}")
        if report.period_sharpes:
            print(f"  Period Sharpes:       {', '.join(f'{s:.2f}' for s in report.period_sharpes)}")
        print(f"  Sharpe Decay:         {report.sharpe_decay:.3f}")

        # IS/OOS
        if report.is_sharpe > 0 or report.oos_sharpe > 0:
            print("-" * 60)
            print(f"  IS Sharpe:            {report.is_sharpe:.2f}")
            print(f"  OOS Sharpe:           {report.oos_sharpe:.2f}")
            print(f"  Degradation:          {report.sharpe_degradation:.0%}")

        # Red flags
        if report.red_flags:
            print("-" * 60)
            print("  RED FLAGS:")
            for i, flag in enumerate(report.red_flags, 1):
                print(f"    {i}. {flag}")
        else:
            print("-" * 60)
            print("  No red flags detected.")

        print("=" * 60 + "\n")
