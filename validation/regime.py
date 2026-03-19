"""Market regime detection and strategy-regime performance analysis.

Classifies market conditions (trending, ranging, high/low volatility, breakout)
and measures how a strategy performs across each regime.  A strategy that only
works in one regime is fragile — regime sensitivity is the key metric.

Usage:
    from validation.regime import RegimeDetector

    detector = RegimeDetector()
    df = detector.classify(data_1m)          # adds 'regime' column
    analysis = detector.analyze_strategy(trades, data_1m, "EMA_CROSS")
    detector.print_analysis(analysis)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

class RegimeType:
    """Market regime classifications."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

    ALL = [
        TRENDING_UP,
        TRENDING_DOWN,
        RANGING,
        HIGH_VOL,
        LOW_VOL,
        BREAKOUT,
        UNKNOWN,
    ]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Performance statistics within a specific regime."""

    regime: str
    n_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    sharpe_ratio: float
    profit_factor: float
    avg_hold_time: float        # seconds
    pct_of_all_trades: float    # what % of total trades occurred in this regime


@dataclass
class RegimeAnalysis:
    """Complete regime analysis for a strategy."""

    strategy_name: str
    regime_stats: dict[str, RegimeStats]   # regime_type -> stats
    best_regime: str                        # regime with highest Sharpe
    worst_regime: str                       # regime with lowest Sharpe
    regime_sensitivity: float               # 0-1, lower is better
    regime_distribution: dict[str, float]   # regime -> % of time
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detect and classify market regimes from OHLCV price data.

    Combines:
    - **Trend**: EMA slope normalised by ATR
    - **Volatility**: ATR percentile relative to a rolling window
    - **Breakout**: sudden ATR expansion after a contraction phase

    The detector works on 1-minute bars but internally resamples to a
    smoother timeframe (default 15 min) so regime labels are not whipsawed
    by intra-bar noise.
    """

    def __init__(
        self,
        trend_ema_period: int = 50,
        trend_slope_lookback: int = 10,
        vol_atr_period: int = 14,
        vol_lookback: int = 100,
        vol_high_threshold: float = 70.0,
        vol_low_threshold: float = 30.0,
        trend_threshold: float = 0.3,
        range_atr_ratio: float = 0.5,
    ):
        self.trend_ema_period = trend_ema_period
        self.trend_slope_lookback = trend_slope_lookback
        self.vol_atr_period = vol_atr_period
        self.vol_lookback = vol_lookback
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold
        self.trend_threshold = trend_threshold
        self.range_atr_ratio = range_atr_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        df: pl.DataFrame,
        resample_period: str = "15m",
    ) -> pl.DataFrame:
        """Add a ``regime`` column to *df* (1-min or 5-min bars).

        1. Resample to *resample_period* for smoother regime detection.
        2. Compute trend slope (EMA slope / ATR).
        3. Compute volatility percentile (rolling ATR percentile).
        4. Classify each resampled bar.
        5. Map regime labels back to the original timeframe.

        Returns a **copy** of *df* with the ``regime`` column appended.
        """
        if df.is_empty():
            return df.with_columns(pl.lit(RegimeType.UNKNOWN).alias("regime"))

        # Drop existing regime column if present so we recalculate cleanly
        if "regime" in df.columns:
            df = df.drop("regime")

        # Ensure sorted by timestamp
        df = df.sort("timestamp")

        # ── 1. Resample to higher timeframe ──────────────────────────
        resampled = self._resample(df, resample_period)

        if resampled.height < self.trend_ema_period + self.trend_slope_lookback:
            logger.warning(
                "Not enough resampled bars (%d) for regime detection; "
                "labelling everything as UNKNOWN.",
                resampled.height,
            )
            return df.with_columns(pl.lit(RegimeType.UNKNOWN).alias("regime"))

        # ── 2-4. Indicators & classification on resampled frame ──────
        resampled = self._add_indicators(resampled)
        resampled = self._label_regimes(resampled)

        # ── 5. Map back to original timeframe via as-of join ─────────
        regime_map = resampled.select("timestamp", "regime")
        df = df.sort("timestamp")

        df = df.join_asof(
            regime_map.sort("timestamp"),
            on="timestamp",
            strategy="backward",
        )

        # Fill any leading nulls (before first resampled bar) with UNKNOWN
        df = df.with_columns(
            pl.col("regime").fill_null(RegimeType.UNKNOWN),
        )

        return df

    def tag_trades(
        self,
        trades: list,
        data: pl.DataFrame,
    ) -> list[tuple]:
        """Tag each trade with the regime at its entry time.

        *data* is auto-classified if it lacks a ``regime`` column.

        Returns ``[(trade, regime_str), ...]``.
        """
        if "regime" not in data.columns:
            data = self.classify(data)

        data = data.sort("timestamp")

        timestamps = data["timestamp"].to_list()
        regimes = data["regime"].to_list()

        result: list[tuple] = []
        for trade in trades:
            entry_ts = trade.entry_time
            regime = self._find_regime_at(entry_ts, timestamps, regimes)
            result.append((trade, regime))

        return result

    def analyze_strategy(
        self,
        trades: list,
        data: pl.DataFrame,
        strategy_name: str = "unknown",
    ) -> RegimeAnalysis:
        """Full regime-aware performance analysis.

        1. Classify data into regimes.
        2. Tag every trade with its entry regime.
        3. Compute per-regime stats.
        4. Identify best / worst regimes by Sharpe.
        5. Measure regime sensitivity.
        6. Generate recommendations.
        """
        if "regime" not in data.columns:
            data = self.classify(data)

        tagged = self.tag_trades(trades, data)

        # Group by regime
        by_regime: dict[str, list[tuple]] = defaultdict(list)
        for trade, regime in tagged:
            by_regime[regime].append((trade, regime))

        total_trades = len(trades)
        regime_stats: dict[str, RegimeStats] = {}
        for regime, group in by_regime.items():
            regime_stats[regime] = self._compute_regime_stats(
                group, regime, total_trades,
            )

        # Best / worst by Sharpe (skip regimes with < 5 trades)
        eligible = {
            k: v for k, v in regime_stats.items() if v.n_trades >= 5
        }
        if eligible:
            best = max(eligible, key=lambda k: eligible[k].sharpe_ratio)
            worst = min(eligible, key=lambda k: eligible[k].sharpe_ratio)
        else:
            best = worst = RegimeType.UNKNOWN

        sensitivity = self._compute_sensitivity(regime_stats)
        distribution = self.regime_summary(data)
        recommendations = self._generate_recommendations(
            regime_stats, best, worst,
        )

        return RegimeAnalysis(
            strategy_name=strategy_name,
            regime_stats=regime_stats,
            best_regime=best,
            worst_regime=worst,
            regime_sensitivity=sensitivity,
            regime_distribution=distribution,
            recommendations=recommendations,
        )

    def regime_summary(self, df: pl.DataFrame) -> dict[str, float]:
        """Return ``{regime: pct}`` where *pct* is 0-100."""
        if "regime" not in df.columns:
            df = self.classify(df)

        total = df.height
        if total == 0:
            return {}

        counts = (
            df.group_by("regime")
            .agg(pl.len().alias("cnt"))
        )
        return {
            row["regime"]: round(row["cnt"] / total * 100, 2)
            for row in counts.iter_rows(named=True)
        }

    def print_analysis(self, analysis: RegimeAnalysis) -> None:
        """Pretty-print a regime analysis report to stdout."""
        sep = "-" * 72
        print(f"\n{'=' * 72}")
        print(f"  REGIME ANALYSIS — {analysis.strategy_name}")
        print(f"{'=' * 72}\n")

        # Distribution
        print("  Market Time Distribution:")
        for regime, pct in sorted(
            analysis.regime_distribution.items(),
            key=lambda x: -x[1],
        ):
            bar = "#" * int(pct / 2)
            print(f"    {regime:<20s} {pct:5.1f}%  {bar}")
        print()

        # Per-regime table
        print(
            f"  {'Regime':<20s} {'Trades':>6s} {'Win%':>6s} "
            f"{'AvgPnL':>9s} {'TotalPnL':>10s} {'Sharpe':>7s} "
            f"{'PF':>6s} {'HoldSec':>8s} {'%Trades':>8s}"
        )
        print(f"  {sep}")
        for regime in sorted(analysis.regime_stats):
            s = analysis.regime_stats[regime]
            print(
                f"  {s.regime:<20s} {s.n_trades:>6d} {s.win_rate:>5.1f}% "
                f"${s.avg_pnl:>8.2f} ${s.total_pnl:>9.2f} {s.sharpe_ratio:>7.2f} "
                f"{s.profit_factor:>6.2f} {s.avg_hold_time:>8.0f} "
                f"{s.pct_of_all_trades:>7.1f}%"
            )

        print(f"\n  {sep}")
        print(f"  Best regime:  {analysis.best_regime}")
        print(f"  Worst regime: {analysis.worst_regime}")
        print(
            f"  Regime sensitivity: {analysis.regime_sensitivity:.3f}  "
            f"(0=regime-agnostic, 1=regime-dependent)"
        )

        if analysis.recommendations:
            print(f"\n  Recommendations:")
            for rec in analysis.recommendations:
                print(f"    - {rec}")

        print(f"\n{'=' * 72}\n")

    # ------------------------------------------------------------------
    # Internal — indicators
    # ------------------------------------------------------------------

    def _resample(self, df: pl.DataFrame, period: str) -> pl.DataFrame:
        """Resample 1m/5m bars to a higher timeframe OHLCV."""
        return (
            df.sort("timestamp")
            .group_by_dynamic("timestamp", every=period)
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
            .sort("timestamp")
        )

    def _add_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add trend slope, ATR, and ATR percentile columns."""
        # ── EMA ──────────────────────────────────────────────────────
        df = df.with_columns(
            pl.col("close")
            .ewm_mean(span=self.trend_ema_period, adjust=False)
            .alias("_ema"),
        )

        # ── EMA slope (change over lookback bars, normalised later) ──
        df = df.with_columns(
            (pl.col("_ema") - pl.col("_ema").shift(self.trend_slope_lookback))
            .alias("_ema_slope_raw"),
        )

        # ── ATR ──────────────────────────────────────────────────────
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        )
        df = df.with_columns(tr.alias("_tr"))
        df = df.with_columns(
            pl.col("_tr")
            .rolling_mean(window_size=self.vol_atr_period)
            .alias("_atr"),
        )

        # ── Normalised slope: slope / ATR ────────────────────────────
        df = df.with_columns(
            pl.when(pl.col("_atr") > 0)
            .then(pl.col("_ema_slope_raw") / pl.col("_atr"))
            .otherwise(0.0)
            .alias("_norm_slope"),
        )

        # ── ATR percentile over rolling window ───────────────────────
        df = df.with_columns(
            pl.col("_atr")
            .rolling_quantile(
                quantile=0.5,
                window_size=self.vol_lookback,
            )
            .alias("_atr_median"),
        )

        # Percentile approximation: rank current ATR in rolling window
        # using rolling_min and rolling_max to normalise.
        df = df.with_columns([
            pl.col("_atr")
            .rolling_min(window_size=self.vol_lookback)
            .alias("_atr_min"),
            pl.col("_atr")
            .rolling_max(window_size=self.vol_lookback)
            .alias("_atr_max"),
        ])
        df = df.with_columns(
            pl.when((pl.col("_atr_max") - pl.col("_atr_min")) > 0)
            .then(
                (pl.col("_atr") - pl.col("_atr_min"))
                / (pl.col("_atr_max") - pl.col("_atr_min"))
                * 100.0
            )
            .otherwise(50.0)
            .alias("_atr_pctl"),
        )

        # ── Breakout flag: ATR > 2x ATR from N bars ago & previous
        #    bar was in the low-vol zone ──────────────────────────────
        lookback_for_breakout = max(self.trend_slope_lookback, 5)
        df = df.with_columns(
            pl.col("_atr").shift(lookback_for_breakout).alias("_atr_lag"),
        )
        df = df.with_columns(
            pl.col("_atr_pctl").shift(1).alias("_prev_atr_pctl"),
        )
        df = df.with_columns(
            (
                (pl.col("_atr") > 2.0 * pl.col("_atr_lag"))
                & (pl.col("_prev_atr_pctl") < self.vol_low_threshold)
            )
            .fill_null(False)
            .alias("_breakout"),
        )

        # ── Price-range / ATR ratio for ranging detection ────────────
        df = df.with_columns(
            pl.when(pl.col("_atr") > 0)
            .then((pl.col("high") - pl.col("low")) / pl.col("_atr"))
            .otherwise(1.0)
            .alias("_range_ratio"),
        )

        return df

    def _label_regimes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assign a regime label to each bar based on indicators."""
        df = df.with_columns(
            pl.when(pl.col("_breakout"))
            .then(pl.lit(RegimeType.BREAKOUT))
            .when(pl.col("_atr_pctl") > self.vol_high_threshold)
            .then(pl.lit(RegimeType.HIGH_VOL))
            .when(
                (pl.col("_norm_slope") > self.trend_threshold)
                & (pl.col("_atr_pctl") <= self.vol_high_threshold)
            )
            .then(pl.lit(RegimeType.TRENDING_UP))
            .when(
                (pl.col("_norm_slope") < -self.trend_threshold)
                & (pl.col("_atr_pctl") <= self.vol_high_threshold)
            )
            .then(pl.lit(RegimeType.TRENDING_DOWN))
            .when(pl.col("_atr_pctl") < self.vol_low_threshold)
            .then(pl.lit(RegimeType.LOW_VOL))
            .when(
                (pl.col("_norm_slope").abs() <= self.trend_threshold)
                & (pl.col("_range_ratio") < self.range_atr_ratio)
            )
            .then(pl.lit(RegimeType.RANGING))
            .when(pl.col("_norm_slope").abs() <= self.trend_threshold)
            .then(pl.lit(RegimeType.RANGING))
            .otherwise(pl.lit(RegimeType.UNKNOWN))
            .alias("regime"),
        )

        # Drop temporary columns
        tmp_cols = [c for c in df.columns if c.startswith("_")]
        df = df.drop(tmp_cols)

        return df

    # ------------------------------------------------------------------
    # Internal — trade analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_regime_at(
        ts: datetime,
        timestamps: list,
        regimes: list[str],
    ) -> str:
        """Binary-search for the regime at *ts* (backward match)."""
        lo, hi = 0, len(timestamps) - 1
        best_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_ts = timestamps[mid]
            if mid_ts <= ts:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_idx < 0:
            return RegimeType.UNKNOWN
        return regimes[best_idx]

    def _compute_regime_stats(
        self,
        trades_with_regime: list[tuple],
        regime: str,
        total_trades: int,
    ) -> RegimeStats:
        """Compute performance stats for trades in a specific regime."""
        pnls = [t.net_pnl for t, _ in trades_with_regime]
        n = len(pnls)
        if n == 0:
            return RegimeStats(
                regime=regime,
                n_trades=0,
                win_rate=0.0,
                avg_pnl=0.0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_hold_time=0.0,
                pct_of_all_trades=0.0,
            )

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0

        # Sharpe (annualised from per-trade returns)
        pnl_arr = np.array(pnls, dtype=np.float64)
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr, ddof=1)) if n > 1 else 0.0
        # Approximate annualisation: assume ~4 trades/day, 252 days
        trades_per_year = 252 * 4
        sharpe = (
            (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
            if std_pnl > 0
            else 0.0
        )

        hold_times = [t.duration_seconds for t, _ in trades_with_regime]

        return RegimeStats(
            regime=regime,
            n_trades=n,
            win_rate=len(winners) / n * 100.0,
            avg_pnl=mean_pnl,
            total_pnl=sum(pnls),
            sharpe_ratio=round(float(sharpe), 3),
            profit_factor=(
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            ),
            avg_hold_time=float(np.mean(hold_times)) if hold_times else 0.0,
            pct_of_all_trades=n / total_trades * 100.0 if total_trades else 0.0,
        )

    @staticmethod
    def _compute_sensitivity(regime_stats: dict[str, RegimeStats]) -> float:
        """Regime sensitivity via coefficient of variation of Sharpe ratios.

        0 = identical performance in every regime (ideal).
        1 = wildly varying performance (fragile).
        Capped at [0, 1].
        """
        sharpes = [
            s.sharpe_ratio
            for s in regime_stats.values()
            if s.n_trades >= 5 and np.isfinite(s.sharpe_ratio)
        ]
        if len(sharpes) < 2:
            return 0.0

        arr = np.array(sharpes, dtype=np.float64)
        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr, ddof=1))

        if abs(mean_s) < 1e-9:
            # Mean near zero — if std is also tiny, not sensitive; else very
            return float(np.clip(std_s, 0.0, 1.0))

        cv = abs(std_s / mean_s)
        # Normalise: CV of 2+ is mapped to 1.0
        return float(np.clip(cv / 2.0, 0.0, 1.0))

    @staticmethod
    def _generate_recommendations(
        regime_stats: dict[str, RegimeStats],
        best: str,
        worst: str,
    ) -> list[str]:
        """Generate actionable, plain-English recommendations."""
        recs: list[str] = []

        best_stats = regime_stats.get(best)
        worst_stats = regime_stats.get(worst)

        if best_stats and best_stats.n_trades >= 5:
            recs.append(
                f"Best in {best} "
                f"(Sharpe {best_stats.sharpe_ratio:.2f}, "
                f"win rate {best_stats.win_rate:.1f}%, "
                f"{best_stats.n_trades} trades)"
            )

        if worst_stats and worst_stats.n_trades >= 5:
            if worst_stats.sharpe_ratio < 0:
                recs.append(
                    f"Consider disabling in {worst} "
                    f"(negative Sharpe {worst_stats.sharpe_ratio:.2f}, "
                    f"{worst_stats.n_trades} trades)"
                )
            else:
                recs.append(
                    f"Weakest in {worst} "
                    f"(Sharpe {worst_stats.sharpe_ratio:.2f}) — "
                    f"consider tighter risk limits"
                )

        # Flag any regime with negative expectancy and meaningful sample
        for regime, stats in regime_stats.items():
            if regime in (best, worst):
                continue
            if stats.n_trades >= 10 and stats.avg_pnl < 0:
                recs.append(
                    f"Negative avg PnL in {regime} "
                    f"(${stats.avg_pnl:.2f}/trade, {stats.n_trades} trades) "
                    f"— consider regime filter"
                )

        # High-vol specific advice
        hv = regime_stats.get(RegimeType.HIGH_VOL)
        if hv and hv.n_trades >= 5 and hv.profit_factor < 1.0:
            recs.append(
                "Unprofitable during high volatility — "
                "consider reducing size or widening stops in volatile regimes"
            )

        # Breakout specific
        bo = regime_stats.get(RegimeType.BREAKOUT)
        if bo and bo.n_trades >= 5 and bo.sharpe_ratio > 1.0:
            recs.append(
                f"Strong breakout performance (Sharpe {bo.sharpe_ratio:.2f}) "
                f"— consider adding a breakout-specific entry signal"
            )

        if not recs:
            recs.append("Insufficient data for regime-specific recommendations.")

        return recs
