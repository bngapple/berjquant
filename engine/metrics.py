"""Performance metrics calculation and trade logging."""

import sqlite3
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from engine.utils import BacktestResult, PerformanceMetrics, Trade


# ── Metrics Calculator ───────────────────────────────────────────────

def calculate_metrics(
    trades: list[Trade], account_size: float
) -> PerformanceMetrics:
    """Compute all performance metrics from a list of trades."""
    if not trades:
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, gross_profit=0.0,
            gross_loss=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0, avg_trade_pnl=0.0,
            avg_winner=0.0, avg_loser=0.0, largest_winner=0.0,
            largest_loser=0.0, avg_hold_time_seconds=0.0,
            profit_by_session={}, consistency_score=0.0,
        )

    pnls = [t.net_pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily P&L for Sharpe calculation
    daily_pnl = defaultdict(float)
    for t in trades:
        day = t.exit_time.strftime("%Y-%m-%d") if isinstance(t.exit_time, datetime) else str(t.exit_time)[:10]
        daily_pnl[day] += t.net_pnl

    daily_returns = list(daily_pnl.values())

    # Sharpe ratio (annualized, risk-free rate = 0)
    if len(daily_returns) > 1:
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns, ddof=1)
        sharpe_ratio = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    equity = account_size
    peak = account_size
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / account_size) * 100 if account_size > 0 else 0.0

    # Session breakdown
    profit_by_session = defaultdict(float)
    for t in trades:
        profit_by_session[t.session_segment] += t.net_pnl

    # Consistency: max single day PnL as % of total profit
    if total_pnl > 0 and daily_returns:
        max_single_day = max(daily_returns)
        consistency_score = (max_single_day / total_pnl) * 100
    else:
        consistency_score = 100.0

    hold_times = [t.duration_seconds for t in trades]

    return PerformanceMetrics(
        total_trades=len(trades),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=len(winners) / len(trades) * 100,
        total_pnl=total_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_trade_pnl=np.mean(pnls),
        avg_winner=np.mean(winners) if winners else 0.0,
        avg_loser=np.mean(losers) if losers else 0.0,
        largest_winner=max(pnls),
        largest_loser=min(pnls),
        avg_hold_time_seconds=np.mean(hold_times),
        profit_by_session=dict(profit_by_session),
        consistency_score=consistency_score,
    )


def print_metrics(metrics: PerformanceMetrics):
    """Print metrics in a readable format."""
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Trades:       {metrics.total_trades}")
    print(f"  Win Rate:           {metrics.win_rate:.1f}%")
    print(f"  Total P&L:          ${metrics.total_pnl:,.2f}")
    print(f"  Profit Factor:      {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:       ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1f}%)")
    print(f"  Avg Trade P&L:      ${metrics.avg_trade_pnl:,.2f}")
    print(f"  Avg Winner:         ${metrics.avg_winner:,.2f}")
    print(f"  Avg Loser:          ${metrics.avg_loser:,.2f}")
    print(f"  Largest Winner:     ${metrics.largest_winner:,.2f}")
    print(f"  Largest Loser:      ${metrics.largest_loser:,.2f}")
    print(f"  Avg Hold Time:      {metrics.avg_hold_time_seconds / 60:.1f} min")
    print(f"  Consistency:        {metrics.consistency_score:.1f}% (max day / total)")
    if metrics.profit_by_session:
        print("  P&L by Session:")
        for session, pnl in sorted(metrics.profit_by_session.items()):
            print(f"    {session:15s} ${pnl:,.2f}")
    print("=" * 50 + "\n")


# ── Trade Logger (SQLite) ────────────────────────────────────────────

class TradeLogger:
    """Log trades and backtest runs to SQLite."""

    def __init__(self, db_path: Path | str = "results/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                prop_firm TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                slippage_ticks INTEGER,
                total_trades INTEGER,
                total_pnl REAL,
                sharpe_ratio REAL,
                profit_factor REAL,
                max_drawdown REAL,
                win_rate REAL,
                config_json TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                exit_time TEXT,
                exit_price REAL,
                contracts INTEGER,
                gross_pnl REAL,
                commission REAL,
                slippage_cost REAL,
                net_pnl REAL,
                duration_seconds INTEGER,
                session_segment TEXT,
                exit_reason TEXT,
                signals_used TEXT,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                date TEXT,
                pnl REAL,
                num_trades INTEGER,
                cumulative_pnl REAL,
                drawdown REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );
        """)
        self.conn.commit()

    def log_result(self, result: BacktestResult, run_id: str | None = None) -> str:
        """Log a complete backtest result. Returns run_id."""
        import uuid as _uuid
        if run_id is None:
            run_id = str(_uuid.uuid4())[:12]

        metrics = result.metrics

        self.conn.execute(
            """INSERT INTO backtest_runs
               (run_id, strategy_name, timestamp, symbol, prop_firm,
                start_date, end_date, initial_capital, slippage_ticks,
                total_trades, total_pnl, sharpe_ratio, profit_factor,
                max_drawdown, win_rate, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result.strategy_name,
                datetime.now().isoformat(),
                result.config.symbol,
                result.config.prop_firm_profile,
                result.config.start_date,
                result.config.end_date,
                result.config.initial_capital,
                result.config.slippage_ticks,
                metrics.total_trades if metrics else 0,
                metrics.total_pnl if metrics else 0,
                metrics.sharpe_ratio if metrics else 0,
                metrics.profit_factor if metrics else 0,
                metrics.max_drawdown if metrics else 0,
                metrics.win_rate if metrics else 0,
                json.dumps({
                    "symbol": result.config.symbol,
                    "prop_firm": result.config.prop_firm_profile,
                    "slippage_ticks": result.config.slippage_ticks,
                }),
            ),
        )

        for trade in result.trades:
            self.conn.execute(
                """INSERT INTO trades
                   (trade_id, run_id, symbol, direction, entry_time,
                    entry_price, exit_time, exit_price, contracts,
                    gross_pnl, commission, slippage_cost, net_pnl,
                    duration_seconds, session_segment, exit_reason,
                    signals_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.trade_id,
                    run_id,
                    trade.symbol,
                    trade.direction,
                    trade.entry_time.isoformat() if isinstance(trade.entry_time, datetime) else str(trade.entry_time),
                    trade.entry_price,
                    trade.exit_time.isoformat() if isinstance(trade.exit_time, datetime) else str(trade.exit_time),
                    trade.exit_price,
                    trade.contracts,
                    trade.gross_pnl,
                    trade.commission,
                    trade.slippage_cost,
                    trade.net_pnl,
                    trade.duration_seconds,
                    trade.session_segment,
                    trade.exit_reason,
                    json.dumps(trade.signals_used),
                ),
            )

        self.conn.commit()
        return run_id

    def close(self):
        self.conn.close()
