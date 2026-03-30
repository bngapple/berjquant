"""
Trade Logger — CSV logging of all fills with slippage tracking.
Output format matches the backtest CSV for easy comparison.
"""

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from config import POINT_VALUE

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

CSV_HEADERS = [
    "timestamp",
    "strategy",
    "account",
    "action",        # Entry/Exit
    "side",          # Buy/Sell
    "contracts",
    "signal_price",  # Price at signal bar close
    "fill_price",    # Actual fill price
    "slippage_pts",  # fill - signal (abs)
    "sl_price",
    "tp_price",
    "exit_reason",   # SL/TP/MaxHold/EOD
    "exit_price",
    "pnl_per_contract",
    "pnl_total",
    "bars_held",
    "daily_pnl",
    "monthly_pnl",
]


@dataclass
class TradeEntry:
    """One complete trade (entry → exit)."""
    strategy: str
    account: str
    side: str               # "Buy" or "Sell"
    contracts: int
    signal_price: float
    fill_price: float
    sl_price: float
    tp_price: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_reason: str = ""   # "SL", "TP", "MaxHold", "EOD", "DailyLimit"
    exit_price: float = 0.0
    bars_held: int = 0

    @property
    def slippage_pts(self) -> float:
        return abs(self.fill_price - self.signal_price)

    @property
    def pnl_per_contract(self) -> float:
        if self.exit_price == 0:
            return 0.0
        if self.side == "Buy":
            return (self.exit_price - self.fill_price) * POINT_VALUE
        else:
            return (self.fill_price - self.exit_price) * POINT_VALUE

    @property
    def pnl_total(self) -> float:
        return self.pnl_per_contract * self.contracts


class TradeLogger:
    """
    Logs all trades to CSV.
    One file per day: trades_YYYY-MM-DD.csv
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._open_trades: dict[str, TradeEntry] = {}  # key: f"{strategy}_{account}"
        self._daily_pnl: float = 0.0
        self._monthly_pnl: float = 0.0

    def _csv_path(self, dt: datetime) -> str:
        return os.path.join(self.log_dir, f"trades_{dt.strftime('%Y-%m-%d')}.csv")

    def _ensure_csv(self, path: str):
        """Create CSV with headers if it doesn't exist."""
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def log_entry(self, trade: TradeEntry):
        """Log an entry fill."""
        key = f"{trade.strategy}_{trade.account}"
        self._open_trades[key] = trade

        now = datetime.now(ET)
        path = self._csv_path(now)
        self._ensure_csv(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                trade.strategy,
                trade.account,
                "Entry",
                trade.side,
                trade.contracts,
                f"{trade.signal_price:.2f}",
                f"{trade.fill_price:.2f}",
                f"{trade.slippage_pts:.2f}",
                f"{trade.sl_price:.2f}",
                f"{trade.tp_price:.2f}",
                "",   # exit_reason
                "",   # exit_price
                "",   # pnl_per_contract
                "",   # pnl_total
                "",   # bars_held
                "",   # daily_pnl
                "",   # monthly_pnl
            ])

        logger.info(
            f"[LOG] Entry: {trade.strategy} {trade.account} "
            f"{trade.side} {trade.contracts} @ {trade.fill_price:.2f} "
            f"(signal: {trade.signal_price:.2f}, slip: {trade.slippage_pts:.2f})"
        )

    def log_exit(
        self,
        strategy: str,
        account: str,
        exit_price: float,
        exit_reason: str,
        bars_held: int = 0,
        daily_pnl: float = 0.0,
        monthly_pnl: float = 0.0,
    ):
        """Log an exit fill and compute P&L."""
        key = f"{strategy}_{account}"
        trade = self._open_trades.pop(key, None)

        if trade is None:
            logger.warning(f"[LOG] Exit for unknown trade: {key}")
            return

        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.exit_time = datetime.now(ET)
        trade.bars_held = bars_held

        now = datetime.now(ET)
        path = self._csv_path(now)
        self._ensure_csv(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                trade.strategy,
                trade.account,
                "Exit",
                trade.side,
                trade.contracts,
                f"{trade.signal_price:.2f}",
                f"{trade.fill_price:.2f}",
                f"{trade.slippage_pts:.2f}",
                f"{trade.sl_price:.2f}",
                f"{trade.tp_price:.2f}",
                exit_reason,
                f"{exit_price:.2f}",
                f"{trade.pnl_per_contract:.2f}",
                f"{trade.pnl_total:.2f}",
                bars_held,
                f"{daily_pnl:.2f}",
                f"{monthly_pnl:.2f}",
            ])

        logger.info(
            f"[LOG] Exit: {trade.strategy} {trade.account} "
            f"@ {exit_price:.2f} ({exit_reason}) "
            f"P&L: ${trade.pnl_total:+,.2f} ({trade.pnl_per_contract:+,.2f}/ct)"
        )

        return trade.pnl_total

    def get_slippage_summary(self) -> dict:
        """Return aggregate slippage stats for the day."""
        # Would iterate over today's CSV — simplified here
        return {"avg_slippage_pts": 0.0, "max_slippage_pts": 0.0, "total_trades": 0}
