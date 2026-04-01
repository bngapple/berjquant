"""
Signal Engine — implements RSI Extremes, IB Breakout, and Momentum Bars.
Each strategy evaluates independently on every completed 15m bar.
Signals are queued for execution on the NEXT bar's open.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from enum import Enum
from typing import Optional

from market_data import MarketState
from config import RSIParams, IBParams, MOMParams, SessionConfig

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = "Buy"
    SELL = "Sell"


@dataclass
class Signal:
    """A trade signal to execute on the next bar open."""
    strategy: str          # "RSI", "IB", "MOM"
    side: Side
    contracts: int
    stop_loss_pts: float
    take_profit_pts: float
    max_hold_bars: int
    reason: str            # Human-readable trigger description
    bar_timestamp: datetime  # The bar that generated this signal
    signal_price: float      # Close of the signal bar (for slippage tracking)


class PositionState:
    """Tracks whether a strategy currently has an open position."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.is_flat = True
        self.side: Optional[Side] = None
        self.entry_bar_index: int = 0
        self.bars_held: int = 0

    def enter(self, side: Side, bar_index: int):
        self.is_flat = False
        self.side = side
        self.entry_bar_index = bar_index
        self.bars_held = 0

    def tick_bar(self):
        """Called each bar while position is open."""
        if not self.is_flat:
            self.bars_held += 1

    def flatten(self):
        self.is_flat = True
        self.side = None
        self.bars_held = 0


class SignalEngine:
    """
    Evaluates all three strategies on each completed bar.
    Returns a list of signals (0 to 3 per bar).
    """

    def __init__(
        self,
        rsi_params: RSIParams,
        ib_params: IBParams,
        mom_params: MOMParams,
        session: SessionConfig,
    ):
        self.rsi_p = rsi_params
        self.ib_p = ib_params
        self.mom_p = mom_params
        self.session = session

        # Position tracking per strategy
        self.positions = {
            "RSI": PositionState("RSI"),
            "IB": PositionState("IB"),
            "MOM": PositionState("MOM"),
        }

        self._bar_count = 0
        self._ib_traded_today = False
        self._current_date = None

    def evaluate(self, state: MarketState) -> list[Signal]:
        """
        Run all strategies against current market state.
        Called when a 15m bar completes.
        Returns list of signals to execute on next bar open.
        """
        self._bar_count += 1
        signals: list[Signal] = []

        if state.last_bar is None:
            return signals

        bar = state.last_bar
        now = bar.timestamp

        # Reset daily state
        if self._current_date != now.date():
            self._current_date = now.date()
            self._ib_traded_today = False

        # Tick held positions
        for pos in self.positions.values():
            pos.tick_bar()

        # No new entries after cutoff
        cutoff = dt_time(16, 30)
        allow_new_entries = now.time() < cutoff

        # Check for max-hold flattens (these generate exit signals)
        flatten_signals = self._check_max_hold()
        signals.extend(flatten_signals)

        if allow_new_entries:
            # Strategy 1: RSI Extremes
            rsi_sig = self._eval_rsi(state)
            if rsi_sig:
                signals.append(rsi_sig)

            # Strategy 2: IB Breakout
            ib_sig = self._eval_ib(state)
            if ib_sig:
                signals.append(ib_sig)

            # Strategy 3: Momentum Bars
            mom_sig = self._eval_mom(state)
            if mom_sig:
                signals.append(mom_sig)

        return signals

    # ------------------------------------------------------------------
    # Strategy 1: RSI Extremes
    # ------------------------------------------------------------------
    def _eval_rsi(self, state: MarketState) -> Optional[Signal]:
        if not self.positions["RSI"].is_flat:
            return None  # Already in a position

        if state.rsi_5 is None:
            return None

        bar = state.last_bar
        side = None
        reason = ""

        if state.rsi_5 < self.rsi_p.oversold:
            side = Side.BUY
            reason = f"RSI({state.rsi_5:.1f}) < {self.rsi_p.oversold}"
        elif state.rsi_5 > self.rsi_p.overbought:
            side = Side.SELL
            reason = f"RSI({state.rsi_5:.1f}) > {self.rsi_p.overbought}"

        if side is None:
            return None

        logger.info(f"[RSI] Signal: {side.value} — {reason}")
        return Signal(
            strategy="RSI",
            side=side,
            contracts=self.rsi_p.contracts,
            stop_loss_pts=self.rsi_p.stop_loss_pts,
            take_profit_pts=self.rsi_p.take_profit_pts,
            max_hold_bars=self.rsi_p.max_hold_bars,
            reason=reason,
            bar_timestamp=bar.timestamp,
            signal_price=bar.close,
        )

    # ------------------------------------------------------------------
    # Strategy 2: IB Breakout
    # ------------------------------------------------------------------
    def _eval_ib(self, state: MarketState) -> Optional[Signal]:
        if not self.positions["IB"].is_flat:
            return None

        if self._ib_traded_today:
            return None  # Max 1 IB trade per day

        if not state.ib_complete or state.today_ib is None:
            return None  # IB not formed yet

        ib = state.today_ib
        if not ib.is_valid:
            return None

        # IB range filter: must be P25-P75 of trailing 50 days
        if state.ib_percentile_low > 0 and state.ib_percentile_high > 0:
            if not (state.ib_percentile_low <= ib.range <= state.ib_percentile_high):
                return None  # IB range outside filter

        bar = state.last_bar
        side = None
        reason = ""

        if bar.close > ib.high:
            side = Side.BUY
            reason = f"IB Breakout UP — close {bar.close:.2f} > IB high {ib.high:.2f}"
        elif bar.close < ib.low:
            side = Side.SELL
            reason = f"IB Breakout DOWN — close {bar.close:.2f} < IB low {ib.low:.2f}"

        if side is None:
            return None

        logger.info(f"[IB] Signal: {side.value} — {reason}")
        return Signal(
            strategy="IB",
            side=side,
            contracts=self.ib_p.contracts,
            stop_loss_pts=self.ib_p.stop_loss_pts,
            take_profit_pts=self.ib_p.take_profit_pts,
            max_hold_bars=self.ib_p.max_hold_bars,
            reason=reason,
            bar_timestamp=bar.timestamp,
            signal_price=bar.close,
        )

    # ------------------------------------------------------------------
    # Strategy 3: Momentum Bars
    # ------------------------------------------------------------------
    def _eval_mom(self, state: MarketState) -> Optional[Signal]:
        if not self.positions["MOM"].is_flat:
            return None

        if state.atr_14 is None or state.ema_21 is None or state.vol_sma_20 is None:
            return None

        bar = state.last_bar
        bar_rng = bar.high - bar.low

        # Condition 1: bar range > ATR(14)
        if bar_rng <= state.atr_14:
            return None

        # Condition 2: volume > SMA(volume, 20)
        if bar.volume <= state.vol_sma_20:
            return None

        # Direction
        side = None
        reason = ""
        direction = "bullish" if bar.close > bar.open else "bearish"

        if direction == "bullish" and bar.close > state.ema_21:
            side = Side.BUY
            reason = (
                f"MOM bullish — range {bar_rng:.2f} > ATR {state.atr_14:.2f}, "
                f"vol {bar.volume} > SMA {state.vol_sma_20:.0f}, "
                f"close {bar.close:.2f} > EMA {state.ema_21:.2f}"
            )
        elif direction == "bearish" and bar.close < state.ema_21:
            side = Side.SELL
            reason = (
                f"MOM bearish — range {bar_rng:.2f} > ATR {state.atr_14:.2f}, "
                f"vol {bar.volume} > SMA {state.vol_sma_20:.0f}, "
                f"close {bar.close:.2f} < EMA {state.ema_21:.2f}"
            )

        if side is None:
            return None

        logger.info(f"[MOM] Signal: {side.value} — {reason}")
        return Signal(
            strategy="MOM",
            side=side,
            contracts=self.mom_p.contracts,
            stop_loss_pts=self.mom_p.stop_loss_pts,
            take_profit_pts=self.mom_p.take_profit_pts,
            max_hold_bars=self.mom_p.max_hold_bars,
            reason=reason,
            bar_timestamp=bar.timestamp,
            signal_price=bar.close,
        )

    # ------------------------------------------------------------------
    # Max Hold Flattening
    # ------------------------------------------------------------------
    def _check_max_hold(self) -> list[Signal]:
        """Generate flatten signals for positions that exceeded max hold."""
        flatten = []
        limits = {
            "RSI": self.rsi_p.max_hold_bars,
            "IB": self.ib_p.max_hold_bars,
            "MOM": self.mom_p.max_hold_bars,
        }
        for name, pos in self.positions.items():
            if not pos.is_flat and pos.bars_held >= limits[name]:
                logger.info(f"[{name}] Max hold reached ({pos.bars_held} bars) — flattening")
                flatten.append(Signal(
                    strategy=name,
                    side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                    contracts=0,  # 0 = flatten existing position
                    stop_loss_pts=0,
                    take_profit_pts=0,
                    max_hold_bars=0,
                    reason=f"Max hold: {pos.bars_held} bars",
                    bar_timestamp=datetime.now(),
                    signal_price=0,
                ))
        return flatten

    def mark_filled(self, strategy: str, side: Side):
        """Called by executor when an entry is confirmed filled."""
        self.positions[strategy].enter(side, self._bar_count)
        if strategy == "IB":
            self._ib_traded_today = True

    def mark_flat(self, strategy: str):
        """Called when a position is closed (SL, TP, max hold, EOD)."""
        if strategy in self.positions:
            self.positions[strategy].flatten()
        else:
            logger.warning(f"mark_flat called with unknown strategy: {strategy}")
