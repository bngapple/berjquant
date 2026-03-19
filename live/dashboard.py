"""Performance monitoring dashboard for paper/live trading.

Tracks real-time equity curves, daily P&L, drawdown, and prop firm
compliance status.  Two output modes:

  1. **Terminal** -- box-drawn tables refreshable via ``print_status()``.
  2. **HTML report** -- interactive plotly charts with the project's
     dark theme (same palette as the Monte Carlo visualization suite).

The dashboard reads from a ``PaperAccount``-like object that exposes:
  - ``account_state``  (``AccountState``)
  - ``trade_history``  (``list[Trade]``)
  - ``daily_summaries`` (``list[dict]`` with date/pnl/trades/balance)
  - ``equity_curve``   (``list[tuple[datetime, float]]``)

and from ``PropFirmRules`` for compliance limits.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Color palette (matches monte_carlo/visualization.py)
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


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class TradingDashboard:
    """
    Performance monitoring dashboard for paper/live trading.

    Generates:
    1. Terminal status display (refreshable)
    2. HTML performance report
    3. Individual chart components

    Usage::

        dashboard = TradingDashboard(prop_rules=topstep_rules)

        # Update with new data
        dashboard.update(paper_account)

        # Terminal display
        dashboard.print_status()

        # Full HTML report
        dashboard.generate_report(output_dir="reports/")
    """

    def __init__(self, prop_rules=None, output_dir: str | Path = "reports"):
        self.prop_rules = prop_rules
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._account = None

    # ------------------------------------------------------------------
    # Data access helpers (duck-typed against PaperAccount)
    # ------------------------------------------------------------------

    def update(self, paper_account) -> None:
        """Update dashboard with latest account data."""
        self._account = paper_account

    def _get_account_state(self):
        """Return the ``AccountState`` from the attached account object."""
        if self._account is None:
            return None
        # Support both attribute and dict-like access.
        return getattr(self._account, "account_state", self._account)

    def _get_trade_history(self) -> list:
        if self._account is None:
            return []
        return getattr(self._account, "trade_history", [])

    def _get_daily_summaries(self) -> list[dict]:
        if self._account is None:
            return []
        return getattr(self._account, "daily_summaries", [])

    def _get_equity_curve(self) -> list[tuple]:
        if self._account is None:
            return []
        return getattr(self._account, "equity_curve", [])

    # ------------------------------------------------------------------
    # Terminal: main status box
    # ------------------------------------------------------------------

    def print_status(self) -> str:
        """
        Print current trading status to terminal and return the string.

        Example::

            +====================================================+
            |           MCQ ENGINE -- PAPER TRADING               |
            +====================================================+
            | Account Balance:     $50,234.50                     |
            | Daily P&L:           +$234.50                       |
            | Open Position:       LONG 1x MNQ @ 18,250.50       |
            | Unrealized P&L:      +$45.00                        |
            |----------------------------------------------------+
            | Max Drawdown:        -$415.00 (of -$2,000 max)     |
            | DD Utilization:      20.8%  ====........            |
            | Daily Loss:          -$123.00 (of -$1,000 max)     |
            | Daily Utilization:   12.3%  ==..........            |
            | Kill Switch:         OK                             |
            |----------------------------------------------------+
            | Today's Trades:      5 (3W / 2L)                   |
            | Total Trades:        47                             |
            | Win Rate:            61.7%                          |
            | Sharpe Ratio:        1.85                           |
            +====================================================+

        Shows utilization bars for visual DD tracking.
        """
        state = self._get_account_state()
        trades = self._get_trade_history()
        W = 54  # inner width

        lines: list[str] = []

        def row(text: str) -> str:
            return f"\u2551 {text:<{W}} \u2551"

        def sep() -> str:
            return f"\u2551 {'\u2500' * W} \u2551"

        # Top border
        lines.append(f"\u2554{'=' * (W + 2)}\u2557")
        lines.append(row("MCQ ENGINE -- PAPER TRADING".center(W)))
        lines.append(f"\u2560{'=' * (W + 2)}\u2563")

        if state is None:
            lines.append(row("No account data loaded."))
            lines.append(f"\u255a{'=' * (W + 2)}\u255d")
            output = "\n".join(lines)
            print(output)
            return output

        # -- Account section --
        balance = getattr(state, "current_balance", 0.0)
        daily_pnl = getattr(state, "daily_pnl", 0.0)
        lines.append(row(f"Account Balance:     {self._format_money_unsigned(balance)}"))
        lines.append(row(f"Daily P&L:           {self._format_money(daily_pnl)}"))

        # Open position
        pos = getattr(state, "open_position", None)
        if pos is not None:
            direction = getattr(pos, "direction", "?").upper()
            contracts = getattr(pos, "contracts", 0)
            symbol = getattr(pos, "symbol", "?")
            entry_price = getattr(pos, "entry_price", 0.0)
            lines.append(
                row(f"Open Position:       {direction} {contracts}x {symbol} @ {entry_price:,.2f}")
            )
            # Unrealized P&L -- requires contract spec & current price; best-effort
            unrealized = self._estimate_unrealized_pnl()
            if unrealized is not None:
                lines.append(row(f"Unrealized P&L:      {self._format_money(unrealized)}"))
        else:
            lines.append(row("Open Position:       FLAT"))

        lines.append(sep())

        # -- Prop firm compliance section --
        if self.prop_rules is not None:
            dd = getattr(state, "current_drawdown", 0.0)
            max_dd = self.prop_rules.max_drawdown
            dd_type = self.prop_rules.drawdown_type.capitalize()
            dd_pct = (abs(dd) / abs(max_dd) * 100) if max_dd != 0 else 0.0
            dd_bar = self._progress_bar(dd, max_dd, width=20)
            lines.append(
                row(f"Max Drawdown:        {self._format_money(dd)} (of {self._format_money(max_dd)} max)")
            )
            lines.append(
                row(f"DD Utilization:      {dd_pct:5.1f}%  {dd_bar}  [{dd_type}]")
            )

            daily_limit = self.prop_rules.daily_loss_limit
            daily_pct = (abs(daily_pnl) / abs(daily_limit) * 100) if daily_limit != 0 else 0.0
            if daily_pnl >= 0:
                daily_pct = 0.0
            daily_bar = self._progress_bar(daily_pnl, daily_limit, width=20) if daily_pnl < 0 else "\u2591" * 20
            lines.append(
                row(f"Daily Loss:          {self._format_money(daily_pnl)} (of {self._format_money(daily_limit)} max)")
            )
            lines.append(
                row(f"Daily Utilization:   {daily_pct:5.1f}%  {daily_bar}")
            )

            is_killed = getattr(state, "is_killed", False)
            kill_status = "TRIGGERED" if is_killed else "OK"
            lines.append(row(f"Kill Switch:         {kill_status}"))
        else:
            dd = getattr(state, "current_drawdown", 0.0)
            lines.append(row(f"Drawdown:            {self._format_money(dd)}"))

        lines.append(sep())

        # -- Trade statistics --
        trades_today = getattr(state, "trades_today", [])
        wins_today = sum(1 for t in trades_today if getattr(t, "net_pnl", 0) > 0)
        losses_today = len(trades_today) - wins_today
        lines.append(
            row(f"Today's Trades:      {len(trades_today)} ({wins_today}W / {losses_today}L)")
        )
        lines.append(row(f"Total Trades:        {len(trades)}"))

        if trades:
            pnls = [getattr(t, "net_pnl", 0.0) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100 if pnls else 0.0
            lines.append(row(f"Win Rate:            {wr:.1f}%"))

            sharpe = self._compute_sharpe(trades)
            lines.append(row(f"Sharpe Ratio:        {sharpe:.2f}"))
        else:
            lines.append(row("Win Rate:            --"))
            lines.append(row("Sharpe Ratio:        --"))

        # Bottom border
        lines.append(f"\u255a{'=' * (W + 2)}\u255d")

        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    # Terminal: daily summary table
    # ------------------------------------------------------------------

    def print_daily_summary(self) -> str:
        """
        Print daily P&L summary table.

        Returns the formatted string.
        """
        summaries = self._get_daily_summaries()

        # If no pre-built summaries, build from trade history.
        if not summaries:
            summaries = self._build_daily_summaries()

        if not summaries:
            msg = "  No daily data available."
            print(msg)
            return msg

        # Column widths
        cw = {"date": 12, "pnl": 12, "trades": 8, "wr": 10, "bal": 12}

        def hline(left: str, mid: str, right: str) -> str:
            return (
                f"{left}"
                f"{'\u2500' * cw['date']}{mid}"
                f"{'\u2500' * cw['pnl']}{mid}"
                f"{'\u2500' * cw['trades']}{mid}"
                f"{'\u2500' * cw['wr']}{mid}"
                f"{'\u2500' * cw['bal']}"
                f"{right}"
            )

        lines: list[str] = []
        lines.append(hline("\u250c", "\u252c", "\u2510"))
        lines.append(
            f"\u2502{'Date':^{cw['date']}}\u2502"
            f"{'P&L':^{cw['pnl']}}\u2502"
            f"{'Trades':^{cw['trades']}}\u2502"
            f"{'Win Rate':^{cw['wr']}}\u2502"
            f"{'Balance':^{cw['bal']}}\u2502"
        )
        lines.append(hline("\u251c", "\u253c", "\u2524"))

        for s in summaries:
            date_str = str(s.get("date", ""))[:10]
            pnl = s.get("pnl", 0.0)
            n_trades = s.get("trades", 0)
            wr = s.get("win_rate", 0.0)
            bal = s.get("balance", 0.0)

            pnl_str = self._format_money(pnl)
            wr_str = f"{wr:.1f}%" if n_trades > 0 else "--"
            bal_str = f"${bal:,.0f}"

            lines.append(
                f"\u2502{date_str:^{cw['date']}}\u2502"
                f"{pnl_str:^{cw['pnl']}}\u2502"
                f"{n_trades:^{cw['trades']}}\u2502"
                f"{wr_str:^{cw['wr']}}\u2502"
                f"{bal_str:^{cw['bal']}}\u2502"
            )

        lines.append(hline("\u2514", "\u2534", "\u2518"))

        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    # Terminal: recent trade log
    # ------------------------------------------------------------------

    def print_trade_log(self, last_n: int = 20) -> str:
        """Print recent trades in a table. Returns the formatted string."""
        trades = self._get_trade_history()
        if not trades:
            msg = "  No trades recorded."
            print(msg)
            return msg

        trades = trades[-last_n:]

        cw = {"ts": 19, "dir": 7, "entry": 12, "exit": 12, "pnl": 12, "reason": 14}

        def hline(left: str, mid: str, right: str) -> str:
            return (
                f"{left}"
                f"{'\u2500' * cw['ts']}{mid}"
                f"{'\u2500' * cw['dir']}{mid}"
                f"{'\u2500' * cw['entry']}{mid}"
                f"{'\u2500' * cw['exit']}{mid}"
                f"{'\u2500' * cw['pnl']}{mid}"
                f"{'\u2500' * cw['reason']}"
                f"{right}"
            )

        lines: list[str] = []
        lines.append(hline("\u250c", "\u252c", "\u2510"))
        lines.append(
            f"\u2502{'Time':^{cw['ts']}}\u2502"
            f"{'Dir':^{cw['dir']}}\u2502"
            f"{'Entry':^{cw['entry']}}\u2502"
            f"{'Exit':^{cw['exit']}}\u2502"
            f"{'P&L':^{cw['pnl']}}\u2502"
            f"{'Reason':^{cw['reason']}}\u2502"
        )
        lines.append(hline("\u251c", "\u253c", "\u2524"))

        for t in trades:
            exit_time = getattr(t, "exit_time", None)
            if isinstance(exit_time, datetime):
                ts_str = exit_time.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = str(exit_time)[:19] if exit_time else "--"

            direction = getattr(t, "direction", "?").upper()
            entry_px = getattr(t, "entry_price", 0.0)
            exit_px = getattr(t, "exit_price", 0.0)
            net_pnl = getattr(t, "net_pnl", 0.0)
            reason = getattr(t, "exit_reason", "")

            lines.append(
                f"\u2502{ts_str:^{cw['ts']}}\u2502"
                f"{direction:^{cw['dir']}}\u2502"
                f"{entry_px:^{cw['entry']},.2f}\u2502"
                f"{exit_px:^{cw['exit']},.2f}\u2502"
                f"{self._format_money(net_pnl):^{cw['pnl']}}\u2502"
                f"{reason:^{cw['reason']}}\u2502"
            )

        lines.append(hline("\u2514", "\u2534", "\u2518"))

        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    # HTML: full report
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path | None = None) -> Path:
        """
        Generate a complete HTML performance report with:
        1. Equity curve
        2. Daily P&L bar chart
        3. Drawdown chart with prop firm limit line
        4. Win rate over time (rolling)
        5. P&L distribution histogram
        6. Trade duration distribution
        7. Summary statistics table

        Returns path to the HTML file.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        if not HAS_PLOTLY:
            logger.warning(
                "plotly is not installed -- HTML report requires plotly. "
                "Install with: pip install plotly"
            )
            # Fall back to a plain-text summary file.
            return self._generate_text_report(out)

        trades = self._get_trade_history()
        equity = self._get_equity_curve()
        summaries = self._get_daily_summaries() or self._build_daily_summaries()
        state = self._get_account_state()

        # -- Build individual figures --
        figs: list[tuple[str, go.Figure]] = []

        eq_fig = self._build_equity_curve_fig(equity)
        if eq_fig is not None:
            figs.append(("Equity Curve", eq_fig))

        pnl_fig = self._build_daily_pnl_fig(summaries)
        if pnl_fig is not None:
            figs.append(("Daily P&L", pnl_fig))

        dd_fig = self._build_drawdown_fig(equity)
        if dd_fig is not None:
            figs.append(("Drawdown", dd_fig))

        wr_fig = self._build_rolling_winrate_fig(trades)
        if wr_fig is not None:
            figs.append(("Rolling Win Rate", wr_fig))

        pnl_dist_fig = self._build_pnl_distribution_fig(trades)
        if pnl_dist_fig is not None:
            figs.append(("P&L Distribution", pnl_dist_fig))

        dur_fig = self._build_duration_distribution_fig(trades)
        if dur_fig is not None:
            figs.append(("Trade Duration", dur_fig))

        # -- Assemble HTML --
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        firm_name = self.prop_rules.firm_name if self.prop_rules else "N/A"

        stats_html = self._build_stats_html(trades, state)

        chart_divs = []
        for i, (label, fig) in enumerate(figs):
            div_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
            chart_divs.append(f'<div class="chart-section"><h2>{label}</h2>{div_html}</div>')

        charts_joined = "\n".join(chart_divs)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCQ Engine -- Performance Report</title>
<style>
  body {{
    background: {_BG};
    color: {_TEXT};
    font-family: Consolas, 'Courier New', monospace;
    margin: 0; padding: 20px;
  }}
  h1 {{
    text-align: center;
    color: {_GREEN};
    border-bottom: 2px solid {_GRID};
    padding-bottom: 12px;
  }}
  .meta {{
    text-align: center;
    color: {_GRAY};
    margin-bottom: 24px;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    max-width: 1100px;
    margin: 0 auto 30px auto;
  }}
  .stat-card {{
    background: {_GRID};
    border-radius: 8px;
    padding: 14px 18px;
  }}
  .stat-card .label {{
    color: {_GRAY};
    font-size: 0.85em;
    margin-bottom: 4px;
  }}
  .stat-card .value {{
    font-size: 1.25em;
    font-weight: bold;
  }}
  .positive {{ color: {_GREEN}; }}
  .negative {{ color: {_RED}; }}
  .neutral  {{ color: {_TEXT}; }}
  .chart-section {{
    max-width: 1100px;
    margin: 0 auto 30px auto;
  }}
  .chart-section h2 {{
    color: {_BLUE};
    border-bottom: 1px solid {_GRID};
    padding-bottom: 6px;
  }}
</style>
</head>
<body>
<h1>MCQ Engine -- Performance Report</h1>
<div class="meta">Generated {timestamp_str} | Firm: {firm_name}</div>
{stats_html}
{charts_joined}
</body>
</html>"""

        report_path = out / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path.write_text(html, encoding="utf-8")
        logger.info("Performance report saved to %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Individual chart builders (return Path to saved HTML)
    # ------------------------------------------------------------------

    def equity_curve_chart(self) -> Path | None:
        """Equity curve with drawdown overlay. Saved as standalone HTML."""
        if not HAS_PLOTLY:
            return None
        equity = self._get_equity_curve()
        fig = self._build_equity_curve_fig(equity)
        if fig is None:
            return None
        path = self.output_dir / "equity_curve.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        return path

    def daily_pnl_chart(self) -> Path | None:
        """Bar chart of daily P&L with cumulative line. Saved as standalone HTML."""
        if not HAS_PLOTLY:
            return None
        summaries = self._get_daily_summaries() or self._build_daily_summaries()
        fig = self._build_daily_pnl_fig(summaries)
        if fig is None:
            return None
        path = self.output_dir / "daily_pnl.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        return path

    def drawdown_chart(self) -> Path | None:
        """Drawdown over time with prop firm limit horizontal line."""
        if not HAS_PLOTLY:
            return None
        equity = self._get_equity_curve()
        fig = self._build_drawdown_fig(equity)
        if fig is None:
            return None
        path = self.output_dir / "drawdown.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        return path

    # ------------------------------------------------------------------
    # Plotly figure builders (internal)
    # ------------------------------------------------------------------

    def _build_equity_curve_fig(self, equity: list[tuple]) -> go.Figure | None:
        """Build plotly equity curve with drawdown shading."""
        if not equity or len(equity) < 2:
            return None

        timestamps = [e[0] for e in equity]
        values = [e[1] for e in equity]

        # Compute drawdown series for overlay
        peak = values[0]
        dd_series = []
        for v in values:
            if v > peak:
                peak = v
            dd_series.append(v - peak)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.06,
            subplot_titles=["Equity", "Drawdown"],
        )

        # Equity line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                line=dict(color=_GREEN, width=2),
                name="Equity",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Starting balance reference
        start_val = values[0]
        fig.add_hline(
            y=start_val,
            line_dash="dash",
            line_color=_GRAY,
            row=1,
            col=1,
            annotation_text=f"Start ${start_val:,.0f}",
            annotation_font_color=_GRAY,
        )

        # Drawdown fill
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=dd_series,
                mode="lines",
                fill="tozeroy",
                fillcolor=_RED_LIGHT,
                line=dict(color=_RED, width=1.5),
                name="Drawdown",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Prop firm DD limit
        if self.prop_rules is not None:
            fig.add_hline(
                y=self.prop_rules.max_drawdown,
                line_dash="dot",
                line_color=_RED,
                row=2,
                col=1,
                annotation_text=f"Max DD ${self.prop_rules.max_drawdown:,.0f}",
                annotation_font_color=_RED,
            )

        layout = _plotly_layout("Equity Curve & Drawdown")
        layout["hovermode"] = "x unified"
        layout["yaxis_tickprefix"] = "$"
        layout["yaxis_tickformat"] = ","
        layout["yaxis2"] = dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickprefix="$",
            tickformat=",",
        )
        fig.update_layout(**layout)
        return fig

    def _build_daily_pnl_fig(self, summaries: list[dict]) -> go.Figure | None:
        """Bar chart of daily P&L + cumulative P&L line."""
        if not summaries:
            return None

        dates = [str(s.get("date", ""))[:10] for s in summaries]
        pnls = [s.get("pnl", 0.0) for s in summaries]
        cum_pnl = list(np.cumsum(pnls))

        colors = [_GREEN if p >= 0 else _RED for p in pnls]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=dates,
                y=pnls,
                marker_color=colors,
                name="Daily P&L",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum_pnl,
                mode="lines+markers",
                line=dict(color=_BLUE, width=2),
                marker=dict(size=4),
                name="Cumulative P&L",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

        layout = _plotly_layout("Daily P&L")
        layout["hovermode"] = "x unified"
        layout["yaxis_tickprefix"] = "$"
        layout["yaxis_tickformat"] = ","
        fig.update_layout(**layout)
        fig.update_yaxes(
            tickprefix="$",
            tickformat=",",
            gridcolor=_GRID,
            secondary_y=True,
        )
        return fig

    def _build_drawdown_fig(self, equity: list[tuple]) -> go.Figure | None:
        """Standalone drawdown chart with prop firm limit line."""
        if not equity or len(equity) < 2:
            return None

        timestamps = [e[0] for e in equity]
        values = [e[1] for e in equity]

        peak = values[0]
        dd_series = []
        dd_pct_series = []
        for v in values:
            if v > peak:
                peak = v
            dd = v - peak
            dd_series.append(dd)
            dd_pct_series.append((dd / peak * 100) if peak > 0 else 0.0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=dd_series,
                mode="lines",
                fill="tozeroy",
                fillcolor=_RED_LIGHT,
                line=dict(color=_RED, width=2),
                name="Drawdown ($)",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            )
        )

        if self.prop_rules is not None:
            fig.add_hline(
                y=self.prop_rules.max_drawdown,
                line_dash="dot",
                line_color=_TEXT,
                annotation_text=f"Limit ${self.prop_rules.max_drawdown:,.0f} ({self.prop_rules.drawdown_type})",
                annotation_font_color=_TEXT,
            )

        layout = _plotly_layout("Drawdown")
        layout["hovermode"] = "x unified"
        layout["yaxis_tickprefix"] = "$"
        layout["yaxis_tickformat"] = ","
        fig.update_layout(**layout)
        return fig

    def _build_rolling_winrate_fig(
        self, trades: list, window: int = 20
    ) -> go.Figure | None:
        """Rolling win rate line chart."""
        if len(trades) < window:
            return None

        pnls = [getattr(t, "net_pnl", 0.0) for t in trades]
        wins = [1.0 if p > 0 else 0.0 for p in pnls]

        rolling_wr: list[float] = []
        for i in range(window, len(wins) + 1):
            rolling_wr.append(sum(wins[i - window : i]) / window * 100)

        x = list(range(window, len(wins) + 1))

        # Try to use exit timestamps for x-axis
        timestamps = []
        for t in trades[window - 1 :]:
            et = getattr(t, "exit_time", None)
            if isinstance(et, datetime):
                timestamps.append(et)
            else:
                timestamps = []
                break
        x_axis = timestamps if len(timestamps) == len(rolling_wr) else x

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=rolling_wr,
                mode="lines",
                line=dict(color=_BLUE, width=2),
                name=f"Win Rate ({window}-trade rolling)",
                hovertemplate="%{y:.1f}%<extra></extra>",
            )
        )

        # 50% reference
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color=_GRAY,
            annotation_text="50%",
            annotation_font_color=_GRAY,
        )

        layout = _plotly_layout(f"Rolling Win Rate ({window}-trade window)")
        layout["hovermode"] = "x unified"
        layout["yaxis_title"] = "Win Rate (%)"
        layout["yaxis_range"] = [0, 100]
        fig.update_layout(**layout)
        return fig

    def _build_pnl_distribution_fig(self, trades: list) -> go.Figure | None:
        """Histogram of individual trade P&L values."""
        if not trades:
            return None

        pnls = np.array([getattr(t, "net_pnl", 0.0) for t in trades])

        fig = go.Figure()

        pos = pnls[pnls >= 0]
        neg = pnls[pnls < 0]

        if len(pos) > 0:
            fig.add_trace(
                go.Histogram(
                    x=pos,
                    marker_color=_GREEN,
                    opacity=0.75,
                    name="Winners",
                )
            )
        if len(neg) > 0:
            fig.add_trace(
                go.Histogram(
                    x=neg,
                    marker_color=_RED,
                    opacity=0.75,
                    name="Losers",
                )
            )

        # Median line
        median = float(np.median(pnls))
        fig.add_vline(
            x=median,
            line_dash="dash",
            line_color=_GOLD,
            annotation_text=f"Median ${median:,.2f}",
            annotation_font_color=_GOLD,
        )

        layout = _plotly_layout("Trade P&L Distribution")
        layout["xaxis_title"] = "P&L ($)"
        layout["yaxis_title"] = "Count"
        layout["xaxis_tickprefix"] = "$"
        layout["barmode"] = "overlay"
        layout["bargap"] = 0.05
        fig.update_layout(**layout)
        return fig

    def _build_duration_distribution_fig(self, trades: list) -> go.Figure | None:
        """Histogram of trade durations in minutes."""
        if not trades:
            return None

        durations_min = []
        for t in trades:
            secs = getattr(t, "duration_seconds", None)
            if secs is not None and secs >= 0:
                durations_min.append(secs / 60.0)

        if not durations_min:
            return None

        durations = np.array(durations_min)

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=durations,
                marker_color=_BLUE,
                opacity=0.75,
                name="Duration",
            )
        )

        med = float(np.median(durations))
        fig.add_vline(
            x=med,
            line_dash="dash",
            line_color=_GOLD,
            annotation_text=f"Median {med:.1f} min",
            annotation_font_color=_GOLD,
        )

        layout = _plotly_layout("Trade Duration Distribution")
        layout["xaxis_title"] = "Duration (minutes)"
        layout["yaxis_title"] = "Count"
        layout["bargap"] = 0.05
        fig.update_layout(**layout)
        return fig

    # ------------------------------------------------------------------
    # HTML stats summary card
    # ------------------------------------------------------------------

    def _build_stats_html(self, trades: list, state) -> str:
        """Build the HTML stats grid for the report header."""
        if not trades and state is None:
            return '<div class="stats-grid"><div class="stat-card"><div class="label">Status</div><div class="value neutral">No data</div></div></div>'

        cards: list[str] = []

        def card(label: str, value: str, css_class: str = "neutral") -> str:
            return (
                f'<div class="stat-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value {css_class}">{value}</div>'
                f"</div>"
            )

        # Account info
        if state is not None:
            balance = getattr(state, "current_balance", 0.0)
            starting = getattr(state, "starting_balance", balance)
            total_pnl = balance - starting
            pnl_class = "positive" if total_pnl >= 0 else "negative"
            cards.append(card("Account Balance", f"${balance:,.2f}"))
            cards.append(card("Total P&L", self._format_money(total_pnl), pnl_class))

            dd = getattr(state, "current_drawdown", 0.0)
            cards.append(card("Current Drawdown", self._format_money(dd), "negative" if dd < 0 else "neutral"))

        # Trade stats
        if trades:
            pnls = [getattr(t, "net_pnl", 0.0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            wr = len(wins) / len(pnls) * 100

            cards.append(card("Total Trades", str(len(trades))))
            cards.append(card("Win Rate", f"{wr:.1f}%", "positive" if wr >= 50 else "negative"))
            cards.append(card("Avg Winner", f"${np.mean(wins):,.2f}" if wins else "--", "positive"))
            cards.append(card("Avg Loser", f"${np.mean(losses):,.2f}" if losses else "--", "negative"))

            sharpe = self._compute_sharpe(trades)
            s_class = "positive" if sharpe >= 1.0 else ("negative" if sharpe < 0 else "neutral")
            cards.append(card("Sharpe Ratio", f"{sharpe:.2f}", s_class))

            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            cards.append(card("Profit Factor", pf_str, "positive" if pf >= 1.0 else "negative"))

            if pnls:
                cards.append(card("Largest Winner", self._format_money(max(pnls)), "positive"))
                cards.append(card("Largest Loser", self._format_money(min(pnls)), "negative"))

        # Prop firm compliance
        if self.prop_rules is not None and state is not None:
            dd = getattr(state, "current_drawdown", 0.0)
            dd_util = abs(dd) / abs(self.prop_rules.max_drawdown) * 100 if self.prop_rules.max_drawdown != 0 else 0.0
            dd_cls = "negative" if dd_util > 75 else ("neutral" if dd_util > 50 else "positive")
            cards.append(card(
                f"DD Utilization ({self.prop_rules.drawdown_type})",
                f"{dd_util:.1f}%",
                dd_cls,
            ))

            is_killed = getattr(state, "is_killed", False)
            cards.append(card("Kill Switch", "TRIGGERED" if is_killed else "OK", "negative" if is_killed else "positive"))

        return f'<div class="stats-grid">{"".join(cards)}</div>'

    # ------------------------------------------------------------------
    # Fallback: plain-text report
    # ------------------------------------------------------------------

    def _generate_text_report(self, output_dir: Path) -> Path:
        """Fallback report when plotly is not available."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("MCQ ENGINE -- PERFORMANCE REPORT (text fallback)")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # Reuse terminal output
        lines.append(self.print_status())
        lines.append("")
        lines.append("DAILY SUMMARY")
        lines.append(self.print_daily_summary())
        lines.append("")
        lines.append("RECENT TRADES")
        lines.append(self.print_trade_log(last_n=50))

        path = output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def _compute_rolling_stats(self, trades: list, window: int = 20) -> dict:
        """
        Compute rolling win rate, Sharpe, etc. from trade list.

        Returns dict with keys: rolling_win_rate, rolling_sharpe,
        rolling_avg_pnl -- each a list of floats aligned to
        trades[window-1:].
        """
        if len(trades) < window:
            return {"rolling_win_rate": [], "rolling_sharpe": [], "rolling_avg_pnl": []}

        pnls = [getattr(t, "net_pnl", 0.0) for t in trades]
        wins = [1.0 if p > 0 else 0.0 for p in pnls]

        rolling_wr: list[float] = []
        rolling_sharpe: list[float] = []
        rolling_avg: list[float] = []

        for i in range(window, len(pnls) + 1):
            window_pnls = pnls[i - window : i]
            window_wins = wins[i - window : i]

            rolling_wr.append(sum(window_wins) / window * 100)
            rolling_avg.append(float(np.mean(window_pnls)))

            std = float(np.std(window_pnls, ddof=1)) if window > 1 else 0.0
            mean = float(np.mean(window_pnls))
            sr = (mean / std * np.sqrt(252)) if std > 0 else 0.0
            rolling_sharpe.append(sr)

        return {
            "rolling_win_rate": rolling_wr,
            "rolling_sharpe": rolling_sharpe,
            "rolling_avg_pnl": rolling_avg,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_sharpe(self, trades: list) -> float:
        """Compute annualized Sharpe from daily P&L of the trade list."""
        if not trades:
            return 0.0

        daily_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            exit_time = getattr(t, "exit_time", None)
            if isinstance(exit_time, datetime):
                day = exit_time.strftime("%Y-%m-%d")
            else:
                day = str(exit_time)[:10] if exit_time else "unknown"
            daily_pnl[day] += getattr(t, "net_pnl", 0.0)

        daily_vals = list(daily_pnl.values())
        if len(daily_vals) < 2:
            return 0.0

        mean_d = float(np.mean(daily_vals))
        std_d = float(np.std(daily_vals, ddof=1))
        if std_d == 0:
            return 0.0
        return mean_d / std_d * np.sqrt(252)

    def _build_daily_summaries(self) -> list[dict]:
        """Build daily summaries from trade history if none pre-computed."""
        trades = self._get_trade_history()
        if not trades:
            return []

        state = self._get_account_state()
        starting = getattr(state, "starting_balance", 50_000.0) if state else 50_000.0

        daily: dict[str, list] = defaultdict(list)
        for t in trades:
            exit_time = getattr(t, "exit_time", None)
            if isinstance(exit_time, datetime):
                day = exit_time.strftime("%Y-%m-%d")
            else:
                day = str(exit_time)[:10] if exit_time else "unknown"
            daily[day].append(t)

        summaries = []
        running_balance = starting
        for day in sorted(daily.keys()):
            day_trades = daily[day]
            day_pnls = [getattr(t, "net_pnl", 0.0) for t in day_trades]
            day_pnl = sum(day_pnls)
            n_trades = len(day_trades)
            wins = sum(1 for p in day_pnls if p > 0)
            wr = wins / n_trades * 100 if n_trades > 0 else 0.0
            running_balance += day_pnl

            summaries.append(
                {
                    "date": day,
                    "pnl": day_pnl,
                    "trades": n_trades,
                    "win_rate": wr,
                    "balance": running_balance,
                }
            )

        return summaries

    def _estimate_unrealized_pnl(self) -> float | None:
        """
        Best-effort unrealized P&L from the open position.

        Returns None if we cannot compute it (no contract spec, no price).
        """
        state = self._get_account_state()
        if state is None:
            return None
        pos = getattr(state, "open_position", None)
        if pos is None:
            return None
        # We need a current price -- check if account exposes one.
        current_price = getattr(self._account, "current_price", None)
        if current_price is None:
            return None
        contract_spec = getattr(self._account, "contract_spec", None)
        if contract_spec is None:
            return None
        try:
            return pos.unrealized_pnl(current_price, contract_spec)
        except Exception:
            return None

    def _progress_bar(self, current: float, maximum: float, width: int = 20) -> str:
        """Generate a text progress bar: ████████░░░░"""
        if maximum == 0:
            return "\u2591" * width
        pct = min(abs(current / maximum), 1.0)
        filled = int(pct * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    def _format_money(self, amount: float) -> str:
        """Format as money: +$1,234.50 or -$567.00"""
        sign = "+" if amount >= 0 else "-"
        return f"{sign}${abs(amount):,.2f}"

    def _format_money_unsigned(self, amount: float) -> str:
        """Format as money without sign: $1,234.50"""
        return f"${amount:,.2f}"
