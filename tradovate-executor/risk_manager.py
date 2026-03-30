"""
Risk Manager — enforces all risk limits and session rules.
- Daily P&L tracking with auto-shutoff at -$3,000
- Monthly P&L tracking with auto-shutoff at -$4,500
- No new entries after 4:30 PM ET
- Flatten all positions at 4:45 PM ET
- Graceful shutdown flatten
"""

import asyncio
import logging
from datetime import datetime, time as dt_time, date
from typing import Optional, Callable
from zoneinfo import ZoneInfo

from config import SessionConfig, POINT_VALUE

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")


class RiskManager:
    """
    Tracks P&L and enforces risk limits across all accounts.
    The flatten callback should handle flattening both master + copy accounts.
    """

    def __init__(
        self,
        session_config: SessionConfig,
        on_flatten_all: Optional[Callable] = None,  # async callback
    ):
        self.cfg = session_config
        self.on_flatten_all = on_flatten_all

        # P&L tracking (all in USD)
        self.daily_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.current_date: Optional[date] = None
        self.current_month: Optional[int] = None

        # State flags
        self.daily_limit_hit: bool = False
        self.monthly_limit_hit: bool = False
        self.trading_halted: bool = False
        self.eod_flattened: bool = False

        # EOD flatten task
        self._eod_task: Optional[asyncio.Task] = None

    def start_eod_timer(self):
        """Launch background task that flattens at 4:45 PM ET."""
        self._eod_task = asyncio.create_task(self._eod_loop())

    async def _eod_loop(self):
        """Check every 5 seconds if it's time to flatten."""
        while True:
            try:
                await asyncio.sleep(5)
                now = datetime.now(ET)

                # Reset daily state at session start
                if now.time() >= dt_time(9, 30) and now.time() < dt_time(9, 31):
                    if self.current_date != now.date():
                        self._reset_daily(now.date())

                # Reset monthly state on 1st of month
                if self.current_month != now.month:
                    self._reset_monthly(now.month)

                # Flatten at 4:45 PM ET
                flatten_time = dt_time(16, 45)
                if now.time() >= flatten_time and not self.eod_flattened:
                    logger.warning("EOD FLATTEN — 4:45 PM ET reached")
                    self.eod_flattened = True
                    await self._execute_flatten("EOD 4:45 PM ET")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EOD loop error: {e}", exc_info=True)

    def _reset_daily(self, today: date):
        """Reset daily state for a new trading day."""
        if self.daily_pnl != 0:
            logger.info(f"Previous day P&L: ${self.daily_pnl:,.2f}")
        self.current_date = today
        self.daily_pnl = 0.0
        self.daily_limit_hit = False
        self.eod_flattened = False
        # Don't reset trading_halted if monthly limit is hit
        if not self.monthly_limit_hit:
            self.trading_halted = False
        logger.info(f"Daily reset — {today}")

    def _reset_monthly(self, month: int):
        """Reset monthly state."""
        if self.monthly_pnl != 0:
            logger.info(f"Previous month P&L: ${self.monthly_pnl:,.2f}")
        self.current_month = month
        self.monthly_pnl = 0.0
        self.monthly_limit_hit = False
        self.trading_halted = False
        logger.info(f"Monthly reset — month {month}")

    # ------------------------------------------------------------------
    # P&L Updates
    # ------------------------------------------------------------------
    def record_trade_pnl(self, pnl: float, strategy: str = ""):
        """
        Called when a trade closes (SL, TP, max hold, or EOD flatten).
        `pnl` is realized P&L in USD for the master account only.
        """
        self.daily_pnl += pnl
        self.monthly_pnl += pnl

        logger.info(
            f"Trade P&L: ${pnl:+,.2f} ({strategy}) | "
            f"Daily: ${self.daily_pnl:+,.2f} | "
            f"Monthly: ${self.monthly_pnl:+,.2f}"
        )

        # Check limits
        if self.daily_pnl <= self.cfg.daily_loss_limit and not self.daily_limit_hit:
            self.daily_limit_hit = True
            self.trading_halted = True
            logger.critical(
                f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:,.2f} "
                f"(limit: ${self.cfg.daily_loss_limit:,.2f})"
            )
            asyncio.create_task(self._execute_flatten("Daily loss limit"))

        if self.monthly_pnl <= self.cfg.monthly_loss_limit and not self.monthly_limit_hit:
            self.monthly_limit_hit = True
            self.trading_halted = True
            logger.critical(
                f"MONTHLY LOSS LIMIT HIT: ${self.monthly_pnl:,.2f} "
                f"(limit: ${self.cfg.monthly_loss_limit:,.2f})"
            )
            asyncio.create_task(self._execute_flatten("Monthly loss limit"))

    def can_trade(self) -> bool:
        """Check if new entries are allowed right now."""
        if self.trading_halted:
            return False

        now = datetime.now(ET)

        # No new entries after 4:30 PM ET
        if now.time() >= dt_time(16, 30):
            return False

        # Before session start
        if now.time() < dt_time(9, 30):
            return False

        return True

    def get_status(self) -> dict:
        """Return current risk state for logging/display."""
        return {
            "daily_pnl": self.daily_pnl,
            "monthly_pnl": self.monthly_pnl,
            "daily_limit_hit": self.daily_limit_hit,
            "monthly_limit_hit": self.monthly_limit_hit,
            "trading_halted": self.trading_halted,
            "eod_flattened": self.eod_flattened,
            "can_trade": self.can_trade(),
        }

    # ------------------------------------------------------------------
    # Flatten Execution
    # ------------------------------------------------------------------
    async def _execute_flatten(self, reason: str):
        """Flatten all positions across all accounts."""
        logger.warning(f"FLATTEN ALL — reason: {reason}")
        if self.on_flatten_all:
            try:
                await self.on_flatten_all()
            except Exception as e:
                logger.critical(f"Flatten callback FAILED: {e}")

    async def emergency_flatten(self):
        """Manual / graceful shutdown flatten."""
        await self._execute_flatten("Emergency / shutdown")

    async def shutdown(self):
        """Cancel background tasks."""
        if self._eod_task:
            self._eod_task.cancel()
