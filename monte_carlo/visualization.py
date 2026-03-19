"""
Monte Carlo Visualization Suite.

Generates interactive HTML charts (plotly) or static PNG fallback (matplotlib)
for Monte Carlo simulation results: equity fan charts, drawdown distributions,
return heatmaps, score radar charts, and strategy comparison leaderboards.
"""

import numpy as np
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotting backend selection
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Color palette — consistent across all charts
# ---------------------------------------------------------------------------
_GREEN = "#22c55e"
_GREEN_LIGHT = "rgba(34,197,94,0.15)"
_GREEN_MID = "rgba(34,197,94,0.30)"
_RED = "#ef4444"
_RED_LIGHT = "rgba(239,68,68,0.15)"
_BLUE = "#3b82f6"
_BLUE_LIGHT = "rgba(59,130,246,0.15)"
_GOLD = "#f59e0b"
_GRAY = "#64748b"
_BG = "#0f172a"
_GRID = "#1e293b"
_TEXT = "#e2e8f0"

# Matplotlib equivalents
_MPL_GREEN = "#22c55e"
_MPL_RED = "#ef4444"
_MPL_BLUE = "#3b82f6"
_MPL_GOLD = "#f59e0b"
_MPL_GRAY = "#64748b"
_MPL_BG = "#0f172a"
_MPL_FACE = "#1e293b"
_MPL_TEXT = "#e2e8f0"


def _safe_name(strategy_name: str) -> str:
    """Sanitize strategy name for filenames."""
    return strategy_name.replace(" ", "_").replace("/", "_")[:30]


def _plotly_layout(title: str) -> dict:
    """Shared dark-theme layout for plotly charts."""
    return dict(
        title=dict(text=title, font=dict(size=18, color=_TEXT)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="Consolas, monospace"),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT)),
        margin=dict(l=60, r=30, t=60, b=50),
    )


def _mpl_style(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent dark style to a matplotlib axes."""
    ax.set_facecolor(_MPL_FACE)
    ax.set_title(title, color=_MPL_TEXT, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=_MPL_TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=_MPL_TEXT, fontsize=11)
    ax.tick_params(colors=_MPL_TEXT, which="both")
    for spine in ax.spines.values():
        spine.set_color(_MPL_GRAY)
    ax.grid(True, color=_MPL_GRAY, alpha=0.3, linewidth=0.5)


class MCVisualizer:
    """
    Visualization suite for Monte Carlo simulation results.

    Generates interactive HTML charts (plotly) or static PNG (matplotlib).

    Usage:
        viz = MCVisualizer(output_dir="reports/charts")
        viz.equity_fan_chart(mc_result)
        viz.drawdown_distribution(mc_result)
        viz.full_report(mc_result)  # generates all charts
    """

    def __init__(self, output_dir: str | Path = "reports/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Equity Fan Chart
    # ------------------------------------------------------------------
    def equity_fan_chart(
        self, mc_result, title: str | None = None, save: bool = True
    ) -> Path | None:
        """
        Equity curve fan chart showing percentile bands.

        Shows median (50th) as solid line, 25th-75th darker band, 5th-95th
        lighter band, horizontal reference at initial capital, and max-DD
        limit line when prop-firm config is available.
        """
        pctls = np.asarray(mc_result.equity_percentiles)
        if pctls.ndim != 2 or pctls.shape[0] < 2:
            logger.warning("equity_percentiles has unexpected shape; skipping fan chart.")
            return None

        n_steps = pctls.shape[0]
        x = np.arange(n_steps)
        p5, p25, p50, p75, p95 = pctls[:, 0], pctls[:, 1], pctls[:, 2], pctls[:, 3], pctls[:, 4]

        name = _safe_name(mc_result.strategy_name)
        chart_title = title or f"{mc_result.strategy_name} — Equity Fan Chart ({mc_result.n_simulations:,} sims)"

        initial_capital = float(p50[0]) if p50[0] != 0 else 50_000.0
        # Try to get max DD limit from config
        max_dd_limit = getattr(mc_result.config, "max_drawdown_limit", None)
        dd_line = initial_capital - max_dd_limit if max_dd_limit and max_dd_limit > 0 else None

        if HAS_PLOTLY:
            fig = go.Figure()

            # 5th-95th band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([p95, p5[::-1]]),
                fill="toself",
                fillcolor=_BLUE_LIGHT,
                line=dict(width=0),
                name="5th–95th pctile",
                hoverinfo="skip",
            ))
            # 25th-75th band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([p75, p25[::-1]]),
                fill="toself",
                fillcolor=_GREEN_MID,
                line=dict(width=0),
                name="25th–75th pctile",
                hoverinfo="skip",
            ))
            # Median line
            fig.add_trace(go.Scatter(
                x=x, y=p50,
                mode="lines",
                line=dict(color=_GREEN, width=2.5),
                name="Median (50th)",
            ))
            # Initial capital
            fig.add_hline(
                y=initial_capital, line_dash="dash", line_color=_GRAY,
                annotation_text=f"Start ${initial_capital:,.0f}",
                annotation_font_color=_GRAY,
            )
            # Max DD limit
            if dd_line is not None:
                fig.add_hline(
                    y=dd_line, line_dash="dot", line_color=_RED,
                    annotation_text=f"Max DD Limit ${dd_line:,.0f}",
                    annotation_font_color=_RED,
                )

            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis_title="Trade #",
                yaxis_title="Equity ($)",
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                hovermode="x unified",
            )
            return self._save_plotly(fig, f"{name}_equity_fan") if save else None

        else:
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=_MPL_BG)
            ax.fill_between(x, p5, p95, color=_MPL_BLUE, alpha=0.15, label="5th–95th pctile")
            ax.fill_between(x, p25, p75, color=_MPL_GREEN, alpha=0.25, label="25th–75th pctile")
            ax.plot(x, p50, color=_MPL_GREEN, linewidth=2, label="Median (50th)")
            ax.axhline(initial_capital, linestyle="--", color=_MPL_GRAY, linewidth=1, label=f"Start ${initial_capital:,.0f}")
            if dd_line is not None:
                ax.axhline(dd_line, linestyle=":", color=_MPL_RED, linewidth=1, label=f"Max DD Limit ${dd_line:,.0f}")
            _mpl_style(ax, chart_title, "Trade #", "Equity ($)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.legend(fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, f"{name}_equity_fan") if save else None

    # ------------------------------------------------------------------
    # 2. Drawdown Distribution
    # ------------------------------------------------------------------
    def drawdown_distribution(self, mc_result, save: bool = True) -> Path | None:
        """
        Histogram of maximum drawdowns across all simulations.

        Shows distribution, median DD, and prop-firm DD limit with
        annotation of percentage of sims exceeding the limit.
        """
        dd = np.asarray(mc_result.max_drawdowns)
        if dd.size == 0:
            logger.warning("No drawdown data; skipping chart.")
            return None

        name = _safe_name(mc_result.strategy_name)
        chart_title = f"{mc_result.strategy_name} — Max Drawdown Distribution"
        median_dd = float(np.median(dd))
        max_dd_limit = getattr(mc_result.config, "max_drawdown_limit", None)
        pct_exceed = float(np.mean(dd >= max_dd_limit) * 100) if max_dd_limit else None

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=dd, nbinsx=60,
                marker_color=_RED,
                opacity=0.75,
                name="Max Drawdown",
            ))
            fig.add_vline(
                x=median_dd, line_dash="dash", line_color=_GOLD,
                annotation_text=f"Median ${median_dd:,.0f}",
                annotation_font_color=_GOLD,
            )
            if max_dd_limit is not None:
                label = f"Limit ${max_dd_limit:,.0f}"
                if pct_exceed is not None:
                    label += f" ({pct_exceed:.1f}% exceed)"
                fig.add_vline(
                    x=max_dd_limit, line_dash="dot", line_color=_TEXT,
                    annotation_text=label,
                    annotation_font_color=_TEXT,
                )
            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis_title="Max Drawdown ($)",
                yaxis_title="Count",
                xaxis_tickprefix="$",
                xaxis_tickformat=",",
                bargap=0.05,
                showlegend=False,
            )
            return self._save_plotly(fig, f"{name}_drawdown_dist") if save else None

        else:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=_MPL_BG)
            ax.hist(dd, bins=60, color=_MPL_RED, alpha=0.75, edgecolor=_MPL_FACE)
            ax.axvline(median_dd, linestyle="--", color=_MPL_GOLD, linewidth=1.5, label=f"Median ${median_dd:,.0f}")
            if max_dd_limit is not None:
                lbl = f"Limit ${max_dd_limit:,.0f}"
                if pct_exceed is not None:
                    lbl += f" ({pct_exceed:.1f}% exceed)"
                ax.axvline(max_dd_limit, linestyle=":", color=_MPL_TEXT, linewidth=1.5, label=lbl)
            _mpl_style(ax, chart_title, "Max Drawdown ($)", "Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.legend(fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, f"{name}_drawdown_dist") if save else None

    # ------------------------------------------------------------------
    # 3. Return Distribution
    # ------------------------------------------------------------------
    def return_distribution(self, mc_result, save: bool = True) -> Path | None:
        """
        Histogram of final P&L across all simulations with breakeven,
        5th/95th percentile lines, and probability-of-profit annotation.
        """
        pnl = np.asarray(mc_result.total_pnls)
        if pnl.size == 0:
            logger.warning("No P&L data; skipping chart.")
            return None

        name = _safe_name(mc_result.strategy_name)
        chart_title = f"{mc_result.strategy_name} — Return Distribution"
        p5 = float(np.percentile(pnl, 5))
        p95 = float(np.percentile(pnl, 95))
        median_pnl = float(np.median(pnl))
        prob_profit = mc_result.probability_of_profit

        # Color bars by sign
        if HAS_PLOTLY:
            fig = go.Figure()
            # Separate positive / negative for color
            fig.add_trace(go.Histogram(
                x=pnl[pnl >= 0], nbinsx=60,
                marker_color=_GREEN, opacity=0.75, name="Profit",
            ))
            fig.add_trace(go.Histogram(
                x=pnl[pnl < 0], nbinsx=60,
                marker_color=_RED, opacity=0.75, name="Loss",
            ))
            fig.add_vline(x=0, line_dash="solid", line_color=_TEXT, line_width=1.5)
            fig.add_vline(
                x=median_pnl, line_dash="dash", line_color=_GOLD,
                annotation_text=f"Median ${median_pnl:,.0f}",
                annotation_font_color=_GOLD,
            )
            fig.add_vline(
                x=p5, line_dash="dot", line_color=_GRAY,
                annotation_text=f"5th ${p5:,.0f}",
                annotation_font_color=_GRAY,
                annotation_position="bottom left",
            )
            fig.add_vline(
                x=p95, line_dash="dot", line_color=_GRAY,
                annotation_text=f"95th ${p95:,.0f}",
                annotation_font_color=_GRAY,
                annotation_position="bottom right",
            )
            # Probability of profit annotation
            fig.add_annotation(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"P(profit) = {prob_profit:.1%}",
                showarrow=False,
                font=dict(size=14, color=_GREEN if prob_profit >= 0.5 else _RED),
                bgcolor="rgba(0,0,0,0.5)",
            )
            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis_title="Total P&L ($)",
                yaxis_title="Count",
                xaxis_tickprefix="$",
                xaxis_tickformat=",",
                barmode="overlay",
                bargap=0.05,
                showlegend=True,
            )
            return self._save_plotly(fig, f"{name}_return_dist") if save else None

        else:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=_MPL_BG)
            colors = [_MPL_GREEN if v >= 0 else _MPL_RED for v in pnl]
            # Use two hist calls for coloring
            if np.any(pnl >= 0):
                ax.hist(pnl[pnl >= 0], bins=60, color=_MPL_GREEN, alpha=0.75, label="Profit")
            if np.any(pnl < 0):
                ax.hist(pnl[pnl < 0], bins=60, color=_MPL_RED, alpha=0.75, label="Loss")
            ax.axvline(0, color=_MPL_TEXT, linewidth=1.5)
            ax.axvline(median_pnl, linestyle="--", color=_MPL_GOLD, linewidth=1.5, label=f"Median ${median_pnl:,.0f}")
            ax.axvline(p5, linestyle=":", color=_MPL_GRAY, linewidth=1, label=f"5th ${p5:,.0f}")
            ax.axvline(p95, linestyle=":", color=_MPL_GRAY, linewidth=1, label=f"95th ${p95:,.0f}")
            ax.text(
                0.98, 0.95, f"P(profit) = {prob_profit:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color=_MPL_GREEN if prob_profit >= 0.5 else _MPL_RED,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            )
            _mpl_style(ax, chart_title, "Total P&L ($)", "Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.legend(fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, f"{name}_return_dist") if save else None

    # ------------------------------------------------------------------
    # 4. Sharpe Distribution
    # ------------------------------------------------------------------
    def sharpe_distribution(self, mc_result, save: bool = True) -> Path | None:
        """
        Histogram of Sharpe ratios with threshold annotations.
        """
        sr = np.asarray(mc_result.sharpe_ratios)
        # Filter out infs/nans
        sr = sr[np.isfinite(sr)]
        if sr.size == 0:
            logger.warning("No valid Sharpe data; skipping chart.")
            return None

        name = _safe_name(mc_result.strategy_name)
        chart_title = f"{mc_result.strategy_name} — Sharpe Ratio Distribution"
        median_sr = float(np.median(sr))
        pct_above_2 = float(np.mean(sr > 2.0) * 100)

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sr, nbinsx=60,
                marker_color=_BLUE, opacity=0.75,
                name="Sharpe Ratio",
            ))
            for thresh, col, label in [
                (1.0, _GOLD, "Good (1.0)"),
                (2.0, _GREEN, "Excellent (2.0)"),
                (3.0, "#a855f7", "Elite (3.0)"),
            ]:
                fig.add_vline(
                    x=thresh, line_dash="dash", line_color=col,
                    annotation_text=label, annotation_font_color=col,
                )
            fig.add_vline(
                x=median_sr, line_dash="solid", line_color=_GOLD, line_width=2,
                annotation_text=f"Median {median_sr:.2f}",
                annotation_font_color=_GOLD,
                annotation_position="top left",
            )
            fig.add_annotation(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"{pct_above_2:.1f}% of sims Sharpe > 2.0",
                showarrow=False,
                font=dict(size=13, color=_GREEN),
                bgcolor="rgba(0,0,0,0.5)",
            )
            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis_title="Sharpe Ratio",
                yaxis_title="Count",
                bargap=0.05,
                showlegend=False,
            )
            return self._save_plotly(fig, f"{name}_sharpe_dist") if save else None

        else:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=_MPL_BG)
            ax.hist(sr, bins=60, color=_MPL_BLUE, alpha=0.75)
            for thresh, col, label in [
                (1.0, _MPL_GOLD, "Good (1.0)"),
                (2.0, _MPL_GREEN, "Excellent (2.0)"),
                (3.0, "#a855f7", "Elite (3.0)"),
            ]:
                ax.axvline(thresh, linestyle="--", color=col, linewidth=1, label=label)
            ax.axvline(median_sr, color=_MPL_GOLD, linewidth=2, label=f"Median {median_sr:.2f}")
            ax.text(
                0.98, 0.95, f"{pct_above_2:.1f}% of sims Sharpe > 2.0",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=_MPL_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            )
            _mpl_style(ax, chart_title, "Sharpe Ratio", "Count")
            ax.legend(fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, f"{name}_sharpe_dist") if save else None

    # ------------------------------------------------------------------
    # 5. Score Radar Chart
    # ------------------------------------------------------------------
    def score_radar(self, strategy_score, save: bool = True) -> Path | None:
        """
        Radar/spider chart of strategy score components (0-100 each).
        """
        categories = [
            "Sharpe", "Profit Factor", "Drawdown",
            "Consistency", "Pass Rate", "Ruin", "Robustness",
        ]
        values = [
            strategy_score.sharpe_score,
            strategy_score.profit_factor_score,
            strategy_score.drawdown_score,
            strategy_score.consistency_score,
            strategy_score.pass_rate_score,
            strategy_score.ruin_score,
            strategy_score.robustness_score,
        ]
        # Close the polygon
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        name = _safe_name(strategy_score.strategy_name)
        chart_title = f"{strategy_score.strategy_name} — Score Radar (Grade: {strategy_score.grade})"

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor=_GREEN_LIGHT,
                line=dict(color=_GREEN, width=2),
                name=f"Composite: {strategy_score.composite_score:.1f}",
            ))
            # Reference circle at 50
            ref_vals = [50] * (len(categories) + 1)
            fig.add_trace(go.Scatterpolar(
                r=ref_vals,
                theta=categories_closed,
                line=dict(color=_GRAY, width=1, dash="dot"),
                name="Threshold (50)",
                fill=None,
            ))
            fig.update_layout(
                **_plotly_layout(chart_title),
                polar=dict(
                    bgcolor=_BG,
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor=_GRID, tickfont=dict(color=_TEXT),
                    ),
                    angularaxis=dict(
                        gridcolor=_GRID, tickfont=dict(color=_TEXT, size=12),
                    ),
                ),
                showlegend=True,
            )
            return self._save_plotly(fig, f"{name}_score_radar") if save else None

        else:
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            vals = values + [values[0]]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=_MPL_BG)
            ax.set_facecolor(_MPL_FACE)
            ax.fill(angles, vals, color=_MPL_GREEN, alpha=0.2)
            ax.plot(angles, vals, color=_MPL_GREEN, linewidth=2, label=f"Composite: {strategy_score.composite_score:.1f}")
            # Reference at 50
            ref = [50] * (len(categories) + 1)
            ax.plot(angles, ref, color=_MPL_GRAY, linewidth=1, linestyle=":", label="Threshold (50)")
            ax.set_thetagrids(
                np.degrees(angles[:-1]), categories,
                color=_MPL_TEXT, fontsize=10,
            )
            ax.set_ylim(0, 100)
            ax.set_title(chart_title, color=_MPL_TEXT, fontsize=13, fontweight="bold", pad=20)
            ax.tick_params(colors=_MPL_TEXT)
            ax.yaxis.grid(color=_MPL_GRAY, alpha=0.3)
            ax.xaxis.grid(color=_MPL_GRAY, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color(_MPL_GRAY)
            ax.legend(loc="lower right", fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, f"{name}_score_radar") if save else None

    # ------------------------------------------------------------------
    # 6. Leaderboard Comparison
    # ------------------------------------------------------------------
    def leaderboard_comparison(
        self, ranked_scores: list, top_n: int = 10, save: bool = True
    ) -> Path | None:
        """
        Grouped bar chart comparing top N strategies across key metrics:
        composite score, pass rate, median Sharpe, and median profit factor.
        """
        if not ranked_scores:
            logger.warning("No scores to compare; skipping leaderboard.")
            return None

        scores = ranked_scores[:top_n]
        names = [s.strategy_name[:20] for s in scores]

        composites = [s.composite_score for s in scores]
        pass_rates = [s.pass_rate_score for s in scores]
        sharpes = [s.sharpe_score for s in scores]
        pfs = [s.profit_factor_score for s in scores]

        chart_title = f"Strategy Leaderboard — Top {len(scores)}"

        if HAS_PLOTLY:
            fig = go.Figure()
            for vals, label, color in [
                (composites, "Composite", _GREEN),
                (pass_rates, "Pass Rate", _BLUE),
                (sharpes, "Sharpe", _GOLD),
                (pfs, "Profit Factor", "#a855f7"),
            ]:
                fig.add_trace(go.Bar(
                    x=names, y=vals, name=label,
                    marker_color=color, opacity=0.85,
                ))
            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis_title="Strategy",
                yaxis_title="Score (0–100)",
                barmode="group",
                bargap=0.2,
                bargroupgap=0.05,
            )
            return self._save_plotly(fig, "leaderboard_comparison") if save else None

        else:
            fig, ax = plt.subplots(figsize=(14, 6), facecolor=_MPL_BG)
            x = np.arange(len(names))
            width = 0.2
            for i, (vals, label, color) in enumerate([
                (composites, "Composite", _MPL_GREEN),
                (pass_rates, "Pass Rate", _MPL_BLUE),
                (sharpes, "Sharpe", _MPL_GOLD),
                (pfs, "Profit Factor", "#a855f7"),
            ]):
                ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
            _mpl_style(ax, chart_title, "Strategy", "Score (0–100)")
            ax.legend(fontsize=9, facecolor=_MPL_FACE, edgecolor=_MPL_GRAY, labelcolor=_MPL_TEXT)
            return self._save_matplotlib(fig, "leaderboard_comparison") if save else None

    # ------------------------------------------------------------------
    # 7. Profit by Session Heatmap
    # ------------------------------------------------------------------
    def profit_by_session_heatmap(self, mc_result, save: bool = True) -> Path | None:
        """
        Heatmap of P&L by session segment and day-of-week.

        Uses original trade session data if available in simulations.
        Falls back gracefully if data is not present.
        """
        # Attempt to aggregate daily P&L across simulations
        sims = getattr(mc_result, "simulations", None)
        if not sims:
            logger.warning("No simulation details available; skipping session heatmap.")
            return None

        # Aggregate daily_pnl across all sims (keyed by day index 0-4 Mon-Fri)
        # daily_pnl is dict[int, float] — we treat keys as day-of-week if in 0-4
        day_pnls: dict[int, list[float]] = {d: [] for d in range(5)}
        has_data = False
        for sim in sims:
            dp = getattr(sim, "daily_pnl", None)
            if not dp:
                continue
            for day_idx, pnl_val in dp.items():
                if isinstance(day_idx, int) and 0 <= day_idx <= 4:
                    day_pnls[day_idx].append(float(pnl_val))
                    has_data = True

        if not has_data:
            logger.warning("No daily_pnl data by day-of-week available; skipping session heatmap.")
            return None

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        avg_pnl = [np.mean(day_pnls[d]) if day_pnls[d] else 0.0 for d in range(5)]
        # Create a simple 1-row heatmap (daily average P&L)
        data_matrix = np.array([avg_pnl])
        row_labels = ["Avg P&L"]

        name = _safe_name(mc_result.strategy_name)
        chart_title = f"{mc_result.strategy_name} — P&L by Day of Week"

        if HAS_PLOTLY:
            # Diverging red-green colorscale
            colorscale = [
                [0.0, _RED],
                [0.5, _BG],
                [1.0, _GREEN],
            ]
            zmax = max(abs(np.min(data_matrix)), abs(np.max(data_matrix))) or 1.0
            fig = go.Figure(data=go.Heatmap(
                z=data_matrix,
                x=day_labels,
                y=row_labels,
                colorscale=colorscale,
                zmin=-zmax,
                zmax=zmax,
                text=[[f"${v:,.0f}" for v in row] for row in data_matrix],
                texttemplate="%{text}",
                textfont=dict(size=14, color=_TEXT),
                colorbar=dict(
                    title="P&L ($)",
                    tickprefix="$",
                    tickformat=",",
                    tickfont=dict(color=_TEXT),
                    titlefont=dict(color=_TEXT),
                ),
            ))
            fig.update_layout(
                **_plotly_layout(chart_title),
                xaxis=dict(side="bottom", tickfont=dict(size=13, color=_TEXT)),
                yaxis=dict(tickfont=dict(size=13, color=_TEXT)),
            )
            return self._save_plotly(fig, f"{name}_session_heatmap") if save else None

        else:
            fig, ax = plt.subplots(figsize=(10, 3), facecolor=_MPL_BG)
            zmax = max(abs(np.min(data_matrix)), abs(np.max(data_matrix))) or 1.0
            im = ax.imshow(
                data_matrix, cmap="RdYlGn", aspect="auto",
                vmin=-zmax, vmax=zmax,
            )
            ax.set_xticks(range(len(day_labels)))
            ax.set_xticklabels(day_labels, fontsize=11)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=11)
            # Annotate cells
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    ax.text(j, i, f"${data_matrix[i, j]:,.0f}",
                            ha="center", va="center", fontsize=12, color=_MPL_TEXT,
                            fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.tick_params(colors=_MPL_TEXT)
            cbar.set_label("P&L ($)", color=_MPL_TEXT)
            _mpl_style(ax, chart_title, "", "")
            return self._save_matplotlib(fig, f"{name}_session_heatmap") if save else None

    # ------------------------------------------------------------------
    # Full Report
    # ------------------------------------------------------------------
    def full_report(self, mc_result, strategy_score=None) -> list[Path]:
        """
        Generate all available charts for a strategy.
        Returns list of paths to saved chart files.
        """
        paths: list[Path] = []
        name = _safe_name(mc_result.strategy_name)

        charts = [
            ("equity_fan", self.equity_fan_chart),
            ("drawdown_dist", self.drawdown_distribution),
            ("return_dist", self.return_distribution),
            ("sharpe_dist", self.sharpe_distribution),
            ("session_heatmap", self.profit_by_session_heatmap),
        ]

        for chart_name, func in charts:
            try:
                path = func(mc_result, save=True)
                if path:
                    paths.append(path)
                    logger.info(f"Generated {chart_name}: {path}")
            except Exception as e:
                logger.warning(f"Failed to generate {chart_name}: {e}")

        if strategy_score:
            try:
                path = self.score_radar(strategy_score, save=True)
                if path:
                    paths.append(path)
                    logger.info(f"Generated score_radar: {path}")
            except Exception as e:
                logger.warning(f"Failed to generate radar chart: {e}")

        logger.info(f"Full report generated {len(paths)} charts for '{mc_result.strategy_name}'")
        return paths

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    def _save_plotly(self, fig, name: str) -> Path:
        """Save a plotly figure as interactive HTML."""
        path = self.output_dir / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        logger.debug(f"Saved plotly chart: {path}")
        return path

    def _save_matplotlib(self, fig, name: str) -> Path:
        """Save a matplotlib figure as PNG."""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.debug(f"Saved matplotlib chart: {path}")
        return path
