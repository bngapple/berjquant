"""
Market Data Engine — aggregates tick/quote data from Tradovate WebSocket
into 15-minute OHLCV bars, calculates all indicators needed by the strategies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Callable
from zoneinfo import ZoneInfo

from indicators import rsi, atr, ema, sma, bar_range, percentile

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")


@dataclass
class Bar:
    """One 15-minute OHLCV bar."""
    timestamp: datetime      # Bar open time (ET)
    open: float = 0.0
    high: float = -float("inf")
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    is_complete: bool = False

    def update(self, price: float, vol: int = 0):
        if self.open == 0.0:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += vol


@dataclass
class IBRange:
    """Initial Balance range for one day."""
    date: datetime
    high: float = -float("inf")
    low: float = float("inf")

    @property
    def range(self) -> float:
        if self.high == -float("inf") or self.low == float("inf"):
            return 0.0
        return self.high - self.low

    @property
    def is_valid(self) -> bool:
        return self.range > 0


@dataclass
class MarketState:
    """Current calculated market state — consumed by signal engine."""
    # Latest completed bar data
    last_bar: Optional[Bar] = None
    current_bar: Optional[Bar] = None

    # Indicator values (updated each completed bar)
    rsi_5: Optional[float] = None
    atr_14: Optional[float] = None
    ema_21: Optional[float] = None
    vol_sma_20: Optional[float] = None

    # IB data for today
    today_ib: Optional[IBRange] = None
    ib_complete: bool = False        # True after 10:00 ET
    ib_percentile_low: float = 0.0   # P25 of trailing 50-day IB ranges
    ib_percentile_high: float = 0.0  # P75

    # Bar history for indicator calculation
    closes: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    volumes: list[int] = field(default_factory=list)

    # IB history
    ib_history: list[float] = field(default_factory=list)  # trailing IB ranges


class MarketDataEngine:
    """
    Receives ticks from Tradovate market data WebSocket,
    builds 15-minute bars, and maintains all indicator state.
    """

    BAR_INTERVAL = timedelta(minutes=15)
    MAX_HISTORY = 500  # Keep last N bars for indicator lookback

    def __init__(self, on_bar_complete: Optional[Callable] = None):
        """
        Args:
            on_bar_complete: async callback(state: MarketState) called when a 15m bar closes
        """
        self.state = MarketState()
        self.on_bar_complete = on_bar_complete
        self._today: Optional[datetime] = None

    def _bar_start_time(self, ts: datetime) -> datetime:
        """Round down to the nearest 15-minute boundary."""
        minute = (ts.minute // 15) * 15
        return ts.replace(minute=minute, second=0, microsecond=0)

    def _is_ib_time(self, ts: datetime) -> bool:
        """Is this timestamp within IB window (9:30-10:00 ET)?"""
        t = ts.time()
        return dt_time(9, 30) <= t < dt_time(10, 0)

    def _is_past_ib(self, ts: datetime) -> bool:
        """Is this timestamp after IB window?"""
        return ts.time() >= dt_time(10, 0)

    async def on_tick(self, price: float, volume: int, timestamp: datetime):
        """
        Process a single tick. Called by WebSocket handler.
        `timestamp` should be in ET.
        """
        # Convert to ET if needed
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ET)
        else:
            timestamp = timestamp.astimezone(ET)

        # Check for new day → reset IB
        today = timestamp.date()
        if self._today != today:
            self._finalize_ib_day()
            self._today = today
            self.state.today_ib = IBRange(date=timestamp)
            self.state.ib_complete = False
            logger.info(f"New trading day: {today}")

        # Update IB if within window
        if self._is_ib_time(timestamp) and self.state.today_ib:
            self.state.today_ib.high = max(self.state.today_ib.high, price)
            self.state.today_ib.low = min(self.state.today_ib.low, price)

        # Mark IB complete once past window
        if self._is_past_ib(timestamp) and not self.state.ib_complete:
            self.state.ib_complete = True
            if self.state.today_ib and self.state.today_ib.is_valid:
                logger.info(
                    f"IB complete — H: {self.state.today_ib.high:.2f} "
                    f"L: {self.state.today_ib.low:.2f} "
                    f"Range: {self.state.today_ib.range:.2f}"
                )

        # Bar aggregation
        bar_start = self._bar_start_time(timestamp)

        if self.state.current_bar is None:
            self.state.current_bar = Bar(timestamp=bar_start)

        # Did we roll into a new bar?
        if bar_start > self.state.current_bar.timestamp:
            await self._complete_bar(self.state.current_bar)
            self.state.current_bar = Bar(timestamp=bar_start)

        self.state.current_bar.update(price, volume)

    async def _complete_bar(self, bar: Bar):
        """Finalize a bar, update indicators, fire callback."""
        bar.is_complete = True
        self.state.last_bar = bar

        # Append to history
        self.state.closes.append(bar.close)
        self.state.highs.append(bar.high)
        self.state.lows.append(bar.low)
        self.state.volumes.append(bar.volume)

        # Trim history
        if len(self.state.closes) > self.MAX_HISTORY:
            trim = len(self.state.closes) - self.MAX_HISTORY
            self.state.closes = self.state.closes[trim:]
            self.state.highs = self.state.highs[trim:]
            self.state.lows = self.state.lows[trim:]
            self.state.volumes = self.state.volumes[trim:]

        # Recalculate indicators
        self._update_indicators()

        logger.debug(
            f"Bar complete: {bar.timestamp.strftime('%H:%M')} "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} "
            f"V={bar.volume} | RSI={self.state.rsi_5:.1f}" if self.state.rsi_5 else ""
        )

        # Notify signal engine
        if self.on_bar_complete:
            try:
                await self.on_bar_complete(self.state)
            except Exception as e:
                logger.error(f"Bar complete callback error: {e}", exc_info=True)

    def _update_indicators(self):
        """Recalculate all indicators from current history."""
        self.state.rsi_5 = rsi(self.state.closes, period=5)
        self.state.atr_14 = atr(
            self.state.highs, self.state.lows, self.state.closes, period=14
        )
        self.state.ema_21 = ema(self.state.closes, period=21)
        self.state.vol_sma_20 = sma(
            [float(v) for v in self.state.volumes], period=20
        )

        # IB percentiles from trailing history
        if len(self.state.ib_history) >= 5:
            self.state.ib_percentile_low = percentile(self.state.ib_history, 25)
            self.state.ib_percentile_high = percentile(self.state.ib_history, 75)

    def _finalize_ib_day(self):
        """Store today's IB range into trailing history."""
        if self.state.today_ib and self.state.today_ib.is_valid:
            self.state.ib_history.append(self.state.today_ib.range)
            # Keep trailing 50 days
            if len(self.state.ib_history) > 50:
                self.state.ib_history = self.state.ib_history[-50:]

    def get_last_bar_range(self) -> Optional[float]:
        """Range of the most recently completed bar."""
        if self.state.last_bar:
            return bar_range(self.state.last_bar.high, self.state.last_bar.low)
        return None

    def get_last_bar_direction(self) -> Optional[str]:
        """'bullish' if close > open, 'bearish' if close < open."""
        b = self.state.last_bar
        if b is None:
            return None
        if b.close > b.open:
            return "bullish"
        elif b.close < b.open:
            return "bearish"
        return "neutral"
