"""
Scoring and ranking of strategies based on Monte Carlo simulation results.

Takes MCResult objects from monte_carlo.simulator and produces ranked
leaderboards with detailed distribution analysis and prop firm compliance.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class StrategyScore:
    """Detailed scoring breakdown for a strategy."""

    strategy_name: str

    # Raw MC statistics
    mc_result: Any  # MCResult

    # Component scores (0-100 each)
    sharpe_score: float  # Based on median Sharpe from MC
    profit_factor_score: float  # Based on median PF
    drawdown_score: float  # Based on max DD compliance with prop firm
    consistency_score: float  # Low variance in returns
    pass_rate_score: float  # Prop firm pass probability
    ruin_score: float  # Inverse of probability of ruin
    robustness_score: float  # Sensitivity to parameter jitter

    # Final composite (0-100)
    composite_score: float

    # Grade (A+, A, B+, B, C, D, F)
    grade: str

    # Flags
    is_viable: bool  # Meets minimum thresholds
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict (excluding the raw MCResult)."""
        return {
            "strategy_name": self.strategy_name,
            "sharpe_score": round(self.sharpe_score, 2),
            "profit_factor_score": round(self.profit_factor_score, 2),
            "drawdown_score": round(self.drawdown_score, 2),
            "consistency_score": round(self.consistency_score, 2),
            "pass_rate_score": round(self.pass_rate_score, 2),
            "ruin_score": round(self.ruin_score, 2),
            "robustness_score": round(self.robustness_score, 2),
            "composite_score": round(self.composite_score, 2),
            "grade": self.grade,
            "is_viable": self.is_viable,
            "warnings": self.warnings,
            "mc_stats": {
                "n_simulations": self.mc_result.n_simulations,
                "median_return": round(self.mc_result.median_return, 2),
                "mean_return": round(self.mc_result.mean_return, 2),
                "pct_5th_return": round(self.mc_result.pct_5th_return, 2),
                "pct_95th_return": round(self.mc_result.pct_95th_return, 2),
                "probability_of_profit": round(self.mc_result.probability_of_profit, 4),
                "probability_of_ruin": round(self.mc_result.probability_of_ruin, 4),
                "prop_firm_pass_rate": round(self.mc_result.prop_firm_pass_rate, 4),
            },
        }


# ── Scorer ───────────────────────────────────────────────────────────


class StrategyScorer:
    """
    Score and rank strategies based on Monte Carlo results.

    Usage:
        scorer = StrategyScorer(prop_rules=topstep_rules)
        scores = scorer.score_batch(mc_results)
        leaderboard = scorer.rank(scores, top_n=20)
        scorer.print_leaderboard(leaderboard)
    """

    # Minimum thresholds for viability
    MIN_SHARPE: float = 0.5
    MIN_PROFIT_FACTOR: float = 1.1
    MIN_PASS_RATE: float = 0.30  # At least 30% of MC sims pass eval
    MAX_RUIN_RATE: float = 0.40  # Max 40% probability of ruin
    MIN_WIN_RATE: float = 0.30  # At least 30% win rate
    MIN_TRADES: int = 20  # Need enough trades for significance

    # Composite weights — sum to 1.0
    WEIGHTS = {
        "sharpe": 0.20,
        "profit_factor": 0.10,
        "drawdown": 0.15,
        "consistency": 0.10,
        "pass_rate": 0.25,
        "ruin": 0.10,
        "robustness": 0.10,
    }

    def __init__(self, prop_rules=None):
        """
        Args:
            prop_rules: PropFirmRules instance for drawdown/compliance scoring.
                        If None, drawdown scoring uses absolute thresholds.
        """
        self.prop_rules = prop_rules

    # ── Public API ───────────────────────────────────────────────────

    def score(self, mc_result) -> StrategyScore:
        """
        Score a single strategy based on its MC simulation results.

        Each component is scored 0-100 using calibrated nonlinear mappings
        that reward meaningfully good values without saturating too early.
        """
        sharpe_score = self._score_sharpe(mc_result)
        profit_factor_score = self._score_profit_factor(mc_result)
        drawdown_score = self._score_drawdown(mc_result)
        consistency_score = self._score_consistency(mc_result)
        pass_rate_score = self._score_pass_rate(mc_result)
        ruin_score = self._score_ruin(mc_result)
        robustness_score = self._score_robustness(mc_result)

        composite = (
            self.WEIGHTS["sharpe"] * sharpe_score
            + self.WEIGHTS["profit_factor"] * profit_factor_score
            + self.WEIGHTS["drawdown"] * drawdown_score
            + self.WEIGHTS["consistency"] * consistency_score
            + self.WEIGHTS["pass_rate"] * pass_rate_score
            + self.WEIGHTS["ruin"] * ruin_score
            + self.WEIGHTS["robustness"] * robustness_score
        )

        warnings = self._check_warnings(mc_result)
        is_viable = self._check_viability(mc_result)
        grade = self._compute_grade(composite)

        return StrategyScore(
            strategy_name=mc_result.strategy_name,
            mc_result=mc_result,
            sharpe_score=sharpe_score,
            profit_factor_score=profit_factor_score,
            drawdown_score=drawdown_score,
            consistency_score=consistency_score,
            pass_rate_score=pass_rate_score,
            ruin_score=ruin_score,
            robustness_score=robustness_score,
            composite_score=round(composite, 2),
            grade=grade,
            is_viable=is_viable,
            warnings=warnings,
        )

    def score_batch(self, mc_results: list) -> list[StrategyScore]:
        """Score multiple strategies."""
        scores = []
        for r in mc_results:
            try:
                scores.append(self.score(r))
            except Exception:
                logger.exception("Failed to score strategy %s", r.strategy_name)
        return scores

    def rank(
        self,
        scores: list[StrategyScore],
        top_n: int = 20,
        viable_only: bool = True,
    ) -> list[StrategyScore]:
        """
        Rank strategies by composite score.

        If viable_only is True, non-viable strategies are filtered out.
        Returns sorted list (descending score), limited to top_n.
        """
        pool = scores
        if viable_only:
            pool = [s for s in pool if s.is_viable]

        ranked = sorted(pool, key=lambda s: s.composite_score, reverse=True)
        return ranked[:top_n]

    def print_leaderboard(
        self, ranked_scores: list[StrategyScore], top_n: int = 20
    ):
        """Pretty-print leaderboard table with box-drawing characters."""
        display = ranked_scores[:top_n]
        if not display:
            print("No strategies to display.")
            return

        # Column widths
        w_rank = 6
        w_name = 30
        w_grade = 7
        w_score = 8
        w_sharpe = 8
        w_pf = 6
        w_pass = 7
        w_ruin = 7
        w_med = 10

        def _hline(left, mid, right, fill="═"):
            parts = [
                fill * w_rank,
                fill * w_name,
                fill * w_grade,
                fill * w_score,
                fill * w_sharpe,
                fill * w_pf,
                fill * w_pass,
                fill * w_ruin,
                fill * w_med,
            ]
            return left + mid.join(parts) + right

        header_top = _hline("╔", "╦", "╗")
        header_sep = _hline("╠", "╬", "╣")
        footer = _hline("╚", "╩", "╝")

        header = (
            f"║{'Rank':^{w_rank}}"
            f"║{'Strategy':^{w_name}}"
            f"║{'Grade':^{w_grade}}"
            f"║{'Score':^{w_score}}"
            f"║{'Sharpe':^{w_sharpe}}"
            f"║{'PF':^{w_pf}}"
            f"║{'Pass%':^{w_pass}}"
            f"║{'Ruin%':^{w_ruin}}"
            f"║{'Median$':^{w_med}}║"
        )

        print(header_top)
        print(header)
        print(header_sep)

        for i, s in enumerate(display, 1):
            mc = s.mc_result
            median_sharpe = float(np.median(mc.sharpe_ratios))
            median_pf = float(np.median(mc.profit_factors))
            pass_pct = mc.prop_firm_pass_rate * 100
            ruin_pct = mc.probability_of_ruin * 100
            median_ret = mc.median_return

            # Truncate long strategy names
            name = s.strategy_name
            if len(name) > w_name - 2:
                name = name[: w_name - 4] + ".."

            row = (
                f"║{i:^{w_rank}}"
                f"║ {name:<{w_name - 1}}"
                f"║{s.grade:^{w_grade}}"
                f"║{s.composite_score:^{w_score}.1f}"
                f"║{median_sharpe:^{w_sharpe}.2f}"
                f"║{median_pf:^{w_pf}.2f}"
                f"║{pass_pct:^{w_pass}.0f}%"
                f"║{ruin_pct:^{w_ruin}.0f}%"
                f"║{_format_money(median_ret):>{w_med}}║"
            )
            print(row)

        print(footer)

        # Summary line
        viable_count = sum(1 for s in ranked_scores if s.is_viable)
        total = len(ranked_scores)
        print(
            f"\n  Showing top {len(display)} of {total} strategies "
            f"({viable_count} viable)"
        )

    # ── Component Scoring Functions ──────────────────────────────────

    def _score_sharpe(self, mc_result) -> float:
        """
        Score based on median Sharpe ratio across MC simulations.

        Calibration (using sigmoid-like curve):
          Sharpe <= 0   ->  0
          Sharpe  0.5   -> 17  (bare minimum)
          Sharpe  1.0   -> 33  (decent)
          Sharpe  1.5   -> 50  (good)
          Sharpe  2.0   -> 67  (very good)
          Sharpe  2.5   -> 83  (excellent)
          Sharpe >= 3.0 -> 100 (elite)
        """
        median_sharpe = float(np.median(mc_result.sharpe_ratios))
        return self._linear_clamp(median_sharpe, low=0.0, high=3.0)

    def _score_profit_factor(self, mc_result) -> float:
        """
        Score based on median profit factor across MC simulations.

        PF = 1.0 means breakeven, so scoring starts there.
        Calibration:
          PF <= 1.0 ->  0
          PF  1.5   -> 25
          PF  2.0   -> 50
          PF  2.5   -> 75
          PF >= 3.0 -> 100
        """
        median_pf = float(np.median(mc_result.profit_factors))
        return self._linear_clamp(median_pf, low=1.0, high=3.0)

    def _score_drawdown(self, mc_result) -> float:
        """
        Score based on how well max drawdown stays within limits.

        If prop_rules are set, scores relative to the max_drawdown limit.
        A strategy using less than 50% of the limit scores 100;
        a strategy at or above the limit scores 0.

        Without prop_rules, uses absolute thresholds based on typical
        $50k account sizing (DD > $5000 = 0, DD < $1000 = 100).
        """
        median_dd = float(np.median(mc_result.max_drawdowns))
        # max_drawdowns are typically negative or positive magnitude — normalize
        dd_magnitude = abs(median_dd)

        if self.prop_rules is not None:
            limit = self.prop_rules.max_drawdown
            # Score: 100 when DD <= 40% of limit, 0 when DD >= limit
            safe_threshold = limit * 0.40
            if dd_magnitude <= safe_threshold:
                return 100.0
            if dd_magnitude >= limit:
                return 0.0
            # Linear interpolation between safe zone and limit
            return 100.0 * (1.0 - (dd_magnitude - safe_threshold) / (limit - safe_threshold))
        else:
            # Absolute thresholds for generic scoring
            return self._linear_clamp_inverse(dd_magnitude, low=500.0, high=5000.0)

    def _score_consistency(self, mc_result) -> float:
        """
        Score based on coefficient of variation (CV) of total PnLs.

        Low CV = consistent outcomes across MC paths = high score.
        Also penalizes heavy reliance on outlier wins by checking skew.

        Calibration:
          CV <= 0.2  -> 100 (very tight distribution)
          CV  0.5    ->  67
          CV  1.0    ->  33
          CV >= 1.5  ->   0 (extremely noisy)
        """
        pnls = mc_result.total_pnls
        mean_pnl = float(np.mean(pnls))

        if abs(mean_pnl) < 1e-6:
            # Near-zero mean — can't compute meaningful CV
            return 0.0

        std_pnl = float(np.std(pnls))
        cv = std_pnl / abs(mean_pnl)

        base_score = self._linear_clamp_inverse(cv, low=0.2, high=1.5)

        # Skewness penalty: heavy left skew (negative) is bad
        n = len(pnls)
        if n >= 3:
            skew = float(_compute_skewness(pnls))
            if skew < -1.0:
                # Penalize up to 20 points for severe negative skew
                penalty = min(20.0, abs(skew + 1.0) * 10.0)
                base_score = max(0.0, base_score - penalty)

        return base_score

    def _score_pass_rate(self, mc_result) -> float:
        """
        Score based on prop firm pass rate from MC simulations.

        Uses a slightly convex mapping so that high pass rates
        are rewarded more than proportionally:
          0%   ->  0
          30%  -> 20  (bare minimum viable)
          50%  -> 40
          70%  -> 65
          85%  -> 85
          95%+ -> 98-100
        """
        rate = mc_result.prop_firm_pass_rate  # 0.0 to 1.0
        # Convex mapping: score = 100 * rate^0.8
        # This gives: 0.3->36, 0.5->57, 0.7->75, 0.85->87, 0.95->95
        # Slightly more generous than linear in the middle range
        return float(np.clip(100.0 * (rate ** 0.85), 0.0, 100.0))

    def _score_ruin(self, mc_result) -> float:
        """
        Score based on probability of ruin (inverse).

        Calibration:
          ruin = 0%   -> 100
          ruin = 10%  ->  85
          ruin = 20%  ->  65
          ruin = 40%  ->  30
          ruin = 60%  ->  10
          ruin >= 80% ->   0
        """
        ruin_rate = mc_result.probability_of_ruin  # 0.0 to 1.0
        # Slightly concave mapping to penalize high ruin more aggressively
        if ruin_rate >= 0.8:
            return 0.0
        raw = 1.0 - (ruin_rate / 0.8)
        # Apply concave curve: score = 100 * raw^1.3
        return float(np.clip(100.0 * (raw ** 1.3), 0.0, 100.0))

    def _score_robustness(self, mc_result) -> float:
        """
        Score based on how tight the MC distribution is (IQR of final equity).

        A robust strategy produces similar outcomes regardless of trade
        ordering. Measured by the interquartile range of final equities
        relative to the median.

        Low relative IQR = robust, high = fragile.
        """
        equities = mc_result.final_equities
        q25 = float(np.percentile(equities, 25))
        q75 = float(np.percentile(equities, 75))
        median_eq = float(np.median(equities))

        if abs(median_eq) < 1e-6:
            return 0.0

        iqr = q75 - q25
        relative_iqr = iqr / abs(median_eq)

        # Calibration:
        #   rel_IQR <= 0.05 -> 100 (extremely tight)
        #   rel_IQR  0.15   ->  75
        #   rel_IQR  0.30   ->  50
        #   rel_IQR  0.50   ->  25
        #   rel_IQR >= 0.80 ->   0 (all over the place)
        return self._linear_clamp_inverse(relative_iqr, low=0.05, high=0.80)

    # ── Viability and Warnings ───────────────────────────────────────

    def _check_viability(self, mc_result) -> bool:
        """Check whether strategy meets all minimum thresholds."""
        median_sharpe = float(np.median(mc_result.sharpe_ratios))
        median_pf = float(np.median(mc_result.profit_factors))
        median_wr = float(np.median(mc_result.win_rates))

        if median_sharpe < self.MIN_SHARPE:
            return False
        if median_pf < self.MIN_PROFIT_FACTOR:
            return False
        if mc_result.prop_firm_pass_rate < self.MIN_PASS_RATE:
            return False
        if mc_result.probability_of_ruin > self.MAX_RUIN_RATE:
            return False
        if median_wr < self.MIN_WIN_RATE:
            return False

        return True

    def _check_warnings(self, mc_result) -> list[str]:
        """Generate warning flags for concerning patterns."""
        warnings: list[str] = []

        # High variance in outcomes
        pnls = mc_result.total_pnls
        mean_pnl = float(np.mean(pnls))
        if abs(mean_pnl) > 1e-6:
            cv = float(np.std(pnls)) / abs(mean_pnl)
            if cv > 1.0:
                warnings.append(
                    f"HIGH_VARIANCE: CV of total PnL is {cv:.2f} "
                    f"(outcomes highly path-dependent)"
                )

        # Negative skew — fat left tail
        if len(pnls) >= 3:
            skew = float(_compute_skewness(pnls))
            if skew < -1.0:
                warnings.append(
                    f"NEGATIVE_SKEW: Return distribution skew is {skew:.2f} "
                    f"(heavy left tail / blow-up risk)"
                )

        # High kurtosis — fat tails in general
        if len(pnls) >= 4:
            kurt = float(_compute_kurtosis(pnls))
            if kurt > 5.0:
                warnings.append(
                    f"FAT_TAILS: Excess kurtosis is {kurt:.2f} "
                    f"(extreme outcomes more likely than normal)"
                )

        # Too few trades for statistical significance
        n_trades = len(mc_result.simulations[0].equity_curve) if mc_result.simulations else 0
        # Fallback: use equity_percentiles shape
        if hasattr(mc_result, "equity_percentiles") and mc_result.equity_percentiles is not None:
            n_trades = max(n_trades, mc_result.equity_percentiles.shape[0])
        if 0 < n_trades < self.MIN_TRADES:
            warnings.append(
                f"LOW_SAMPLE: Only {n_trades} trades — "
                f"results may not be statistically significant"
            )

        # Ruin probability elevated
        if mc_result.probability_of_ruin > 0.20:
            warnings.append(
                f"ELEVATED_RUIN: {mc_result.probability_of_ruin:.1%} probability "
                f"of ruin across MC simulations"
            )

        # Returns heavily dependent on a few big winners
        if len(pnls) >= 10:
            sorted_pnls = np.sort(mc_result.total_pnls)[::-1]
            top_10_pct_count = max(1, len(sorted_pnls) // 10)
            top_slice = sorted_pnls[:top_10_pct_count]
            total_positive = float(np.sum(sorted_pnls[sorted_pnls > 0]))
            if total_positive > 0:
                concentration = float(np.sum(top_slice)) / total_positive
                if concentration > 0.50:
                    warnings.append(
                        f"CONCENTRATED_WINNERS: Top 10% of simulations account "
                        f"for {concentration:.0%} of total profit"
                    )

        # Low win rate warning
        median_wr = float(np.median(mc_result.win_rates))
        if median_wr < 0.35:
            warnings.append(
                f"LOW_WIN_RATE: Median win rate is {median_wr:.1%} — "
                f"requires large avg winner:loser ratio to be profitable"
            )

        # Drawdown close to prop firm limit
        if self.prop_rules is not None:
            median_dd = abs(float(np.median(mc_result.max_drawdowns)))
            dd_ratio = median_dd / self.prop_rules.max_drawdown if self.prop_rules.max_drawdown > 0 else 0
            if dd_ratio > 0.70:
                warnings.append(
                    f"DRAWDOWN_RISK: Median max DD is {dd_ratio:.0%} of "
                    f"prop firm limit (${self.prop_rules.max_drawdown:,.0f})"
                )

        return warnings

    # ── Distribution Analysis ────────────────────────────────────────

    def distribution_analysis(self, mc_result) -> dict:
        """
        Detailed distribution analysis of MC results.

        Returns a comprehensive statistical breakdown of the simulation
        outcome distributions.
        """
        pnls = mc_result.total_pnls
        dds = np.abs(mc_result.max_drawdowns)
        sharpes = mc_result.sharpe_ratios
        equities = mc_result.final_equities

        pctl_keys = [5, 10, 25, 50, 75, 90, 95]

        # Return distribution
        return_dist = {
            "mean": float(np.mean(pnls)),
            "median": float(np.median(pnls)),
            "std": float(np.std(pnls)),
            "skew": float(_compute_skewness(pnls)) if len(pnls) >= 3 else None,
            "kurtosis": float(_compute_kurtosis(pnls)) if len(pnls) >= 4 else None,
            "percentiles": {
                str(p): float(np.percentile(pnls, p)) for p in pctl_keys
            },
            "min": float(np.min(pnls)),
            "max": float(np.max(pnls)),
        }

        # Drawdown distribution
        dd_dist = {
            "mean": float(np.mean(dds)),
            "median": float(np.median(dds)),
            "std": float(np.std(dds)),
            "percentiles": {
                str(p): float(np.percentile(dds, p)) for p in pctl_keys
            },
            "max": float(np.max(dds)),
        }

        # Sharpe distribution
        sharpe_dist = {
            "mean": float(np.mean(sharpes)),
            "median": float(np.median(sharpes)),
            "std": float(np.std(sharpes)),
            "percentiles": {
                str(p): float(np.percentile(sharpes, p)) for p in pctl_keys
            },
        }

        # Tail risk metrics
        var_95 = float(np.percentile(pnls, 5))  # 5th percentile = worst 5%
        losses = pnls[pnls < var_95]
        cvar_95 = float(np.mean(losses)) if len(losses) > 0 else var_95

        tail_risk = {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_loss": float(np.min(pnls)),
            "probability_of_loss_gt_2x_median_dd": float(
                np.mean(pnls < -2.0 * float(np.median(dds)))
            ),
        }

        # Equity curve shape (from percentile bands)
        equity_shape = {}
        if (
            hasattr(mc_result, "equity_percentiles")
            and mc_result.equity_percentiles is not None
            and mc_result.equity_percentiles.shape[0] > 1
        ):
            ep = mc_result.equity_percentiles
            n_steps = ep.shape[0]
            # Check if equity tends to grow (median end > median start)
            equity_shape["median_start"] = float(ep[0, 2])
            equity_shape["median_end"] = float(ep[-1, 2])
            equity_shape["median_growth"] = float(ep[-1, 2] - ep[0, 2])
            equity_shape["n_trade_steps"] = n_steps
            # Band width at end vs start (expansion = increasing uncertainty)
            start_band = float(ep[0, 3] - ep[0, 1])  # 75th - 25th at start
            end_band = float(ep[-1, 3] - ep[-1, 1])
            equity_shape["band_expansion_ratio"] = (
                end_band / start_band if start_band > 1e-6 else None
            )

        return {
            "return_distribution": return_dist,
            "drawdown_distribution": dd_dist,
            "sharpe_distribution": sharpe_dist,
            "tail_risk": tail_risk,
            "equity_shape": equity_shape,
        }

    def prop_firm_analysis(self, mc_result, prop_rules=None) -> dict:
        """
        Detailed prop firm pass analysis.

        Examines failure reasons by classifying each simulation outcome
        against prop firm rules.
        """
        rules = prop_rules or self.prop_rules
        if rules is None:
            return {
                "error": "No prop firm rules provided",
                "pass_rate": mc_result.prop_firm_pass_rate,
            }

        n_sims = mc_result.n_simulations
        pass_rate = mc_result.prop_firm_pass_rate

        # Classify failure reasons from MC statistics
        dd_limit = rules.max_drawdown
        dd_magnitudes = np.abs(mc_result.max_drawdowns)

        # Failure: max drawdown exceeded
        dd_failures = int(np.sum(dd_magnitudes >= dd_limit))

        # Failure: not profitable (total PnL <= 0)
        unprofitable = int(np.sum(mc_result.total_pnls <= 0))

        # Failure: consistency rule violated
        # (estimated from win_rates and total_pnls distribution)
        consistency_failures = 0
        if rules.consistency_rule_enabled and len(mc_result.simulations) > 0:
            # For each simulation, check if any single day exceeds
            # consistency_max_single_day_pct of total profit.
            # Since we may not have daily granularity in MCResult,
            # approximate from available data.
            for sim in mc_result.simulations:
                if hasattr(sim, "daily_pnls") and sim.daily_pnls is not None:
                    total_profit = sum(p for p in sim.daily_pnls if p > 0)
                    if total_profit > 0:
                        max_day = max(sim.daily_pnls)
                        if max_day / total_profit > rules.consistency_max_single_day_pct:
                            consistency_failures += 1

        # Approximate failure breakdown (some sims may fail multiple criteria)
        result = {
            "pass_rate": round(pass_rate, 4),
            "total_simulations": n_sims,
            "passed": int(round(pass_rate * n_sims)),
            "failed": int(round((1 - pass_rate) * n_sims)),
            "failure_reasons": {
                "max_drawdown_exceeded": dd_failures,
                "max_drawdown_exceeded_pct": round(dd_failures / n_sims, 4) if n_sims > 0 else 0,
                "not_profitable": unprofitable,
                "not_profitable_pct": round(unprofitable / n_sims, 4) if n_sims > 0 else 0,
                "consistency_violated": consistency_failures,
                "consistency_violated_pct": (
                    round(consistency_failures / n_sims, 4)
                    if n_sims > 0
                    else 0
                ),
            },
            "prop_firm": rules.firm_name,
            "account_size": rules.account_size,
            "max_drawdown_limit": dd_limit,
            "median_max_drawdown": round(float(np.median(dd_magnitudes)), 2),
            "drawdown_utilization": round(
                float(np.median(dd_magnitudes)) / dd_limit, 4
            )
            if dd_limit > 0
            else None,
            "expected_value": round(float(np.mean(mc_result.total_pnls)), 2),
            "median_outcome": round(float(np.median(mc_result.total_pnls)), 2),
        }

        return result

    # ── Export ────────────────────────────────────────────────────────

    def export_report(
        self, ranked_scores: list[StrategyScore], filepath: str | Path
    ) -> Path:
        """
        Export full MC analysis report as JSON.

        Includes leaderboard, distribution analysis, prop firm analysis,
        warnings, and flags for every ranked strategy.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "n_strategies": len(ranked_scores),
            "prop_firm": self.prop_rules.firm_name if self.prop_rules else None,
            "scoring_weights": self.WEIGHTS,
            "viability_thresholds": {
                "min_sharpe": self.MIN_SHARPE,
                "min_profit_factor": self.MIN_PROFIT_FACTOR,
                "min_pass_rate": self.MIN_PASS_RATE,
                "max_ruin_rate": self.MAX_RUIN_RATE,
                "min_win_rate": self.MIN_WIN_RATE,
                "min_trades": self.MIN_TRADES,
            },
            "leaderboard": [],
        }

        for rank, score in enumerate(ranked_scores, 1):
            entry = {
                "rank": rank,
                **score.to_dict(),
                "distribution_analysis": self.distribution_analysis(
                    score.mc_result
                ),
                "prop_firm_analysis": self.prop_firm_analysis(score.mc_result),
            }
            report["leaderboard"].append(entry)

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=_json_serializer)

        logger.info("Exported MC report to %s (%d strategies)", filepath, len(ranked_scores))
        return filepath

    # ── Grade Mapping ────────────────────────────────────────────────

    def _compute_grade(self, score: float) -> str:
        """Map composite score to letter grade."""
        if score >= 90:
            return "A+"
        if score >= 80:
            return "A"
        if score >= 70:
            return "B+"
        if score >= 60:
            return "B"
        if score >= 50:
            return "C"
        if score >= 40:
            return "D"
        return "F"

    # ── Utility Functions ────────────────────────────────────────────

    @staticmethod
    def _linear_clamp(value: float, low: float, high: float) -> float:
        """
        Linear mapping from [low, high] -> [0, 100], clamped.
        Values at or below `low` score 0; at or above `high` score 100.
        """
        if value <= low:
            return 0.0
        if value >= high:
            return 100.0
        return 100.0 * (value - low) / (high - low)

    @staticmethod
    def _linear_clamp_inverse(value: float, low: float, high: float) -> float:
        """
        Inverse linear mapping: lower values score higher.
        Values at or below `low` score 100; at or above `high` score 0.
        """
        if value <= low:
            return 100.0
        if value >= high:
            return 0.0
        return 100.0 * (1.0 - (value - low) / (high - low))


# ── Module-Level Helpers ─────────────────────────────────────────────


def _compute_skewness(arr) -> float:
    """Compute sample skewness (Fisher's definition)."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-10:
        return 0.0
    m3 = np.mean((arr - mean) ** 3)
    return float(m3 / (std ** 3))


def _compute_kurtosis(arr) -> float:
    """Compute excess kurtosis (Fisher's definition, normal = 0)."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((arr - mean) ** 4)
    return float(m4 / (std ** 4) - 3.0)


def _format_money(value: float) -> str:
    """Format dollar value with sign and commas."""
    if value >= 0:
        return f"${value:,.0f}"
    return f"-${abs(value):,.0f}"


def _json_serializer(obj):
    """Fallback JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
