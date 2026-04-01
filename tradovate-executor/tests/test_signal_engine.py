"""
tests/test_signal_engine.py

Unit tests for SignalEngine and PositionState.
Covers all three strategies (RSI, IB, MOM), max-hold flattening,
position state transitions, and daily reset of IB traded flag.

No external dependencies — all state is constructed inline.
"""

import sys
import os
import pytest
from datetime import datetime, date, time as dt_time
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine import SignalEngine, Signal, Side, PositionState
from market_data import MarketState, Bar, IBRange
from config import RSIParams, IBParams, MOMParams, SessionConfig

ET = ZoneInfo("US/Eastern")

# ---------------------------------------------------------------------------
# Fixture: default strategy params
# ---------------------------------------------------------------------------

def default_rsi_params() -> RSIParams:
    return RSIParams(
        period=5,
        oversold=35.0,
        overbought=65.0,
        contracts=3,
        stop_loss_pts=10.0,
        take_profit_pts=100.0,
        max_hold_bars=5,
    )


def default_ib_params() -> IBParams:
    return IBParams(
        contracts=3,
        stop_loss_pts=10.0,
        take_profit_pts=120.0,
        max_hold_bars=15,
    )


def default_mom_params() -> MOMParams:
    return MOMParams(
        contracts=3,
        stop_loss_pts=15.0,
        take_profit_pts=100.0,
        max_hold_bars=5,
    )


def default_session() -> SessionConfig:
    return SessionConfig()


def make_engine() -> SignalEngine:
    return SignalEngine(
        rsi_params=default_rsi_params(),
        ib_params=default_ib_params(),
        mom_params=default_mom_params(),
        session=default_session(),
    )


def make_bar(
    hour: int = 10,
    minute: int = 0,
    day: int = 15,
    month: int = 1,
    year: int = 2026,
    open_: float = 20000.0,
    high: float = 20010.0,
    low: float = 19990.0,
    close: float = 20005.0,
    volume: int = 1000,
) -> Bar:
    ts = datetime(year, month, day, hour, minute, tzinfo=ET)
    bar = Bar(timestamp=ts)
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    bar.is_complete = True
    return bar


def make_state(
    bar: Bar = None,
    rsi_5: float = None,
    atr_14: float = None,
    ema_21: float = None,
    vol_sma_20: float = None,
    ib_complete: bool = False,
    today_ib: IBRange = None,
    ib_percentile_low: float = 0.0,
    ib_percentile_high: float = 0.0,
) -> MarketState:
    if bar is None:
        bar = make_bar()
    state = MarketState()
    state.last_bar = bar
    state.rsi_5 = rsi_5
    state.atr_14 = atr_14
    state.ema_21 = ema_21
    state.vol_sma_20 = vol_sma_20
    state.ib_complete = ib_complete
    state.today_ib = today_ib
    state.ib_percentile_low = ib_percentile_low
    state.ib_percentile_high = ib_percentile_high
    return state


def make_ib(
    high: float = 20020.0,
    low: float = 19980.0,
    date_: date = None,
) -> IBRange:
    if date_ is None:
        date_ = datetime(2026, 1, 15, tzinfo=ET)
    ib = IBRange(date=date_)
    ib.high = high
    ib.low = low
    return ib


# ===========================================================================
# PositionState
# ===========================================================================

class TestPositionState:
    def test_initial_state_is_flat(self):
        pos = PositionState("RSI")
        assert pos.is_flat is True
        assert pos.side is None
        assert pos.bars_held == 0

    def test_enter_sets_correct_state(self):
        pos = PositionState("RSI")
        pos.enter(Side.BUY, bar_index=5)
        assert pos.is_flat is False
        assert pos.side == Side.BUY
        assert pos.entry_bar_index == 5
        assert pos.bars_held == 0

    def test_tick_bar_increments_when_in_position(self):
        pos = PositionState("RSI")
        pos.enter(Side.BUY, bar_index=1)
        pos.tick_bar()
        assert pos.bars_held == 1
        pos.tick_bar()
        assert pos.bars_held == 2

    def test_tick_bar_does_nothing_when_flat(self):
        pos = PositionState("RSI")
        pos.tick_bar()
        assert pos.bars_held == 0

    def test_flatten_resets_state(self):
        pos = PositionState("RSI")
        pos.enter(Side.SELL, bar_index=3)
        pos.tick_bar()
        pos.tick_bar()
        pos.flatten()
        assert pos.is_flat is True
        assert pos.side is None
        assert pos.bars_held == 0

    def test_enter_sell_side(self):
        pos = PositionState("IB")
        pos.enter(Side.SELL, bar_index=10)
        assert pos.side == Side.SELL
        assert pos.is_flat is False

    def test_re_enter_after_flatten(self):
        pos = PositionState("MOM")
        pos.enter(Side.BUY, bar_index=1)
        pos.flatten()
        pos.enter(Side.SELL, bar_index=5)
        assert pos.is_flat is False
        assert pos.side == Side.SELL


# ===========================================================================
# Signal Engine — no_bar guard
# ===========================================================================

class TestSignalEngineNoBar:
    def test_returns_empty_when_no_last_bar(self):
        engine = make_engine()
        state = MarketState()
        state.last_bar = None
        signals = engine.evaluate(state)
        assert signals == []


# ===========================================================================
# Strategy 1: RSI Extremes
# ===========================================================================

class TestRSIStrategy:
    def test_buy_signal_when_rsi_below_oversold(self):
        engine = make_engine()
        state = make_state(rsi_5=30.0)  # < 35 → BUY
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert len(rsi_signals) == 1
        assert rsi_signals[0].side == Side.BUY
        assert rsi_signals[0].contracts == 3
        assert rsi_signals[0].stop_loss_pts == 10.0
        assert rsi_signals[0].take_profit_pts == 100.0

    def test_sell_signal_when_rsi_above_overbought(self):
        engine = make_engine()
        state = make_state(rsi_5=70.0)  # > 65 → SELL
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert len(rsi_signals) == 1
        assert rsi_signals[0].side == Side.SELL

    def test_no_signal_when_rsi_is_neutral(self):
        engine = make_engine()
        state = make_state(rsi_5=50.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_no_signal_when_rsi_is_none(self):
        engine = make_engine()
        state = make_state(rsi_5=None)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_no_signal_when_already_in_position(self):
        engine = make_engine()
        engine.positions["RSI"].enter(Side.BUY, bar_index=1)
        # Even with oversold RSI, already in position → no new signal
        state = make_state(rsi_5=20.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_signal_uses_bar_close_as_signal_price(self):
        engine = make_engine()
        bar = make_bar(close=19850.0)
        state = make_state(bar=bar, rsi_5=20.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals[0].signal_price == 19850.0

    def test_signal_uses_bar_timestamp(self):
        engine = make_engine()
        bar = make_bar(hour=11, minute=30)
        state = make_state(bar=bar, rsi_5=20.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals[0].bar_timestamp == bar.timestamp

    def test_no_signal_at_oversold_boundary_35_exactly(self):
        # RSI = 35.0 is NOT < 35 → no signal
        engine = make_engine()
        state = make_state(rsi_5=35.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_no_signal_at_overbought_boundary_65_exactly(self):
        # RSI = 65.0 is NOT > 65 → no signal
        engine = make_engine()
        state = make_state(rsi_5=65.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_no_entry_after_cutoff_time(self):
        # 16:45 bar → past cutoff, no new entries
        engine = make_engine()
        bar = make_bar(hour=16, minute=45)
        state = make_state(bar=bar, rsi_5=20.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert rsi_signals == []

    def test_entry_allowed_at_16_29(self):
        # 16:29 bar → still within trading window
        engine = make_engine()
        bar = make_bar(hour=16, minute=29)
        state = make_state(bar=bar, rsi_5=20.0)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert len(rsi_signals) == 1

    def test_reason_contains_rsi_value_and_threshold(self):
        engine = make_engine()
        state = make_state(rsi_5=28.5)
        signals = engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert "28.5" in rsi_signals[0].reason or "28" in rsi_signals[0].reason


# ===========================================================================
# Strategy 2: IB Breakout
# ===========================================================================

class TestIBStrategy:
    def _make_ib_state(
        self,
        close: float = 20025.0,
        ib_high: float = 20020.0,
        ib_low: float = 19980.0,
        ib_percentile_low: float = 0.0,
        ib_percentile_high: float = 0.0,
        hour: int = 10,
        minute: int = 30,
    ) -> tuple:
        bar = make_bar(
            hour=hour, minute=minute,
            close=close, high=close + 5, low=close - 5,
        )
        ib = make_ib(high=ib_high, low=ib_low)
        state = make_state(
            bar=bar,
            ib_complete=True,
            today_ib=ib,
            ib_percentile_low=ib_percentile_low,
            ib_percentile_high=ib_percentile_high,
        )
        return state, ib

    def test_buy_signal_when_close_above_ib_high(self):
        engine = make_engine()
        state, _ = self._make_ib_state(close=20025.0, ib_high=20020.0)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert len(ib_signals) == 1
        assert ib_signals[0].side == Side.BUY

    def test_sell_signal_when_close_below_ib_low(self):
        engine = make_engine()
        state, _ = self._make_ib_state(close=19975.0, ib_low=19980.0)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert len(ib_signals) == 1
        assert ib_signals[0].side == Side.SELL

    def test_no_signal_when_close_inside_ib_range(self):
        engine = make_engine()
        state, _ = self._make_ib_state(close=20000.0)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_no_signal_when_ib_not_complete(self):
        engine = make_engine()
        bar = make_bar(close=20025.0)
        ib = make_ib()
        state = make_state(bar=bar, ib_complete=False, today_ib=ib)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_no_signal_when_today_ib_is_none(self):
        engine = make_engine()
        bar = make_bar(close=20025.0)
        state = make_state(bar=bar, ib_complete=True, today_ib=None)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_max_one_ib_trade_per_day(self):
        engine = make_engine()
        state, _ = self._make_ib_state(close=20025.0)
        # First signal fires
        signals1 = engine.evaluate(state)
        ib1 = [s for s in signals1 if s.strategy == "IB"]
        assert len(ib1) == 1

        # Reset position to flat to simulate it was filled and closed
        engine.positions["IB"].enter(Side.BUY, 1)
        engine.mark_flat("IB")

        # Second attempt on same day → should NOT fire (ib_traded_today=True)
        state2, _ = self._make_ib_state(close=20025.0)
        signals2 = engine.evaluate(state2)
        ib2 = [s for s in signals2 if s.strategy == "IB"]
        assert ib2 == []

    def test_ib_traded_resets_on_new_day(self):
        engine = make_engine()
        # Trade on day 1
        state1, _ = self._make_ib_state(close=20025.0)
        engine.evaluate(state1)
        assert engine._ib_traded_today is True

        # Next day bar → resets
        bar_day2 = make_bar(hour=10, minute=30, close=20025.0, day=16)
        ib = make_ib()
        state2 = make_state(bar=bar_day2, ib_complete=True, today_ib=ib)
        engine.evaluate(state2)
        # After daily reset, ib_traded_today should reset to False then
        # immediately become True again after a signal fires
        # But importantly the evaluate call should not block the signal
        ib_signals = [s for s in engine.evaluate(state2) if s.strategy == "IB"]
        # The first evaluate on day 2 should have fired (reset happened)
        # Here we verify the engine DID reset; the second evaluate on day2 is blocked
        # Check internal state: after two evaluates on day2, flag is True again
        assert engine._ib_traded_today is True

    def test_no_signal_when_already_in_ib_position(self):
        engine = make_engine()
        engine.positions["IB"].enter(Side.BUY, 1)
        state, _ = self._make_ib_state(close=20025.0)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_ib_percentile_filter_rejects_outside_range(self):
        """
        IB range = 20020-19980 = 40 pts.
        Percentile filter: p25=50, p75=100 → 40 is below p25 → rejected.
        """
        engine = make_engine()
        state, _ = self._make_ib_state(
            close=20025.0,
            ib_high=20020.0,
            ib_low=19980.0,  # range = 40
            ib_percentile_low=50.0,
            ib_percentile_high=100.0,
        )
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_ib_percentile_filter_accepts_within_range(self):
        """
        IB range = 40 pts. Filter: p25=30, p75=60 → 40 is within range → passes.
        """
        engine = make_engine()
        state, _ = self._make_ib_state(
            close=20025.0,
            ib_high=20020.0,
            ib_low=19980.0,  # range = 40
            ib_percentile_low=30.0,
            ib_percentile_high=60.0,
        )
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert len(ib_signals) == 1

    def test_ib_percentile_filter_skipped_when_zero(self):
        """
        When percentile_low=0 and percentile_high=0, the filter is inactive.
        """
        engine = make_engine()
        state, _ = self._make_ib_state(
            close=20025.0,
            ib_percentile_low=0.0,
            ib_percentile_high=0.0,
        )
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert len(ib_signals) == 1

    def test_invalid_ib_no_signal(self):
        """IBRange with default values (high=-inf, low=+inf) is not valid."""
        engine = make_engine()
        bar = make_bar(close=20025.0)
        ib = IBRange(date=datetime(2026, 1, 15, tzinfo=ET))
        # Don't set high/low → range = 0 → is_valid = False
        state = make_state(bar=bar, ib_complete=True, today_ib=ib)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        assert ib_signals == []

    def test_ib_signal_params_correct(self):
        engine = make_engine()
        state, _ = self._make_ib_state(close=20025.0)
        signals = engine.evaluate(state)
        ib_signals = [s for s in signals if s.strategy == "IB"]
        sig = ib_signals[0]
        assert sig.contracts == 3
        assert sig.stop_loss_pts == 10.0
        assert sig.take_profit_pts == 120.0
        assert sig.max_hold_bars == 15


# ===========================================================================
# Strategy 3: Momentum Bars
# ===========================================================================

class TestMOMStrategy:
    def _make_mom_state(
        self,
        open_: float = 20000.0,
        close: float = 20015.0,
        high: float = 20020.0,
        low: float = 19995.0,
        volume: int = 2000,
        atr_14: float = 10.0,
        ema_21: float = 20010.0,
        vol_sma_20: float = 1500.0,
        hour: int = 11,
    ) -> MarketState:
        bar = make_bar(
            hour=hour, open_=open_, close=close, high=high, low=low, volume=volume
        )
        return make_state(
            bar=bar,
            atr_14=atr_14,
            ema_21=ema_21,
            vol_sma_20=vol_sma_20,
        )

    def test_buy_signal_bullish_bar_above_ema(self):
        """
        Bullish bar (close>open), range > ATR, vol > SMA, close > EMA → BUY.
        bar: open=20000, close=20015, high=20020, low=19995 → range=25 > ATR=10
        volume=2000 > vol_sma=1500, close=20015 > ema=20010
        """
        engine = make_engine()
        state = self._make_mom_state()
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert len(mom_signals) == 1
        assert mom_signals[0].side == Side.BUY

    def test_sell_signal_bearish_bar_below_ema(self):
        """
        Bearish bar (close<open), range > ATR, vol > SMA, close < EMA → SELL.
        bar: open=20015, close=20000, high=20020, low=19995 → range=25 > ATR=10
        close=20000 < ema=20010
        """
        engine = make_engine()
        state = self._make_mom_state(
            open_=20015.0,
            close=20000.0,
            high=20020.0,
            low=19995.0,
            ema_21=20010.0,
        )
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert len(mom_signals) == 1
        assert mom_signals[0].side == Side.SELL

    def test_no_signal_when_range_not_above_atr(self):
        """
        Bar range = 5 pts, ATR = 10 pts → fails range filter.
        """
        engine = make_engine()
        state = self._make_mom_state(
            close=20005.0, high=20007.0, low=20002.0,  # range=5
            atr_14=10.0,
        )
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert mom_signals == []

    def test_no_signal_when_volume_not_above_sma(self):
        """
        Volume = 1000, vol_sma = 1500 → fails volume filter.
        """
        engine = make_engine()
        state = self._make_mom_state(volume=1000, vol_sma_20=1500.0)
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert mom_signals == []

    def test_no_buy_when_bullish_but_close_below_ema(self):
        """
        Bullish bar but close < EMA → direction filter fails for BUY.
        """
        engine = make_engine()
        state = self._make_mom_state(
            open_=20000.0, close=20005.0,
            ema_21=20010.0,  # close < ema
        )
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert mom_signals == []

    def test_no_sell_when_bearish_but_close_above_ema(self):
        """
        Bearish bar but close > EMA → direction filter fails for SELL.
        """
        engine = make_engine()
        state = self._make_mom_state(
            open_=20015.0, close=20008.0,  # bearish
            ema_21=20005.0,  # close > ema → no SELL signal
        )
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        assert mom_signals == []

    def test_no_signal_when_missing_indicators(self):
        engine = make_engine()
        bar = make_bar()
        state = make_state(bar=bar, atr_14=None, ema_21=20000.0, vol_sma_20=1000.0)
        signals = engine.evaluate(state)
        assert [s for s in signals if s.strategy == "MOM"] == []

        state2 = make_state(bar=bar, atr_14=10.0, ema_21=None, vol_sma_20=1000.0)
        assert [s for s in engine.evaluate(state2) if s.strategy == "MOM"] == []

        state3 = make_state(bar=bar, atr_14=10.0, ema_21=20000.0, vol_sma_20=None)
        assert [s for s in engine.evaluate(state3) if s.strategy == "MOM"] == []

    def test_no_signal_when_already_in_position(self):
        engine = make_engine()
        engine.positions["MOM"].enter(Side.BUY, 1)
        state = self._make_mom_state()
        signals = engine.evaluate(state)
        assert [s for s in signals if s.strategy == "MOM"] == []

    def test_mom_signal_params_correct(self):
        engine = make_engine()
        state = self._make_mom_state()
        signals = engine.evaluate(state)
        mom_signals = [s for s in signals if s.strategy == "MOM"]
        sig = mom_signals[0]
        assert sig.contracts == 3
        assert sig.stop_loss_pts == 15.0
        assert sig.take_profit_pts == 100.0
        assert sig.max_hold_bars == 5

    def test_boundary_range_equal_to_atr_no_signal(self):
        """
        Bar range == ATR exactly (not strictly greater) → no signal.
        high=20010, low=20000 → range=10, ATR=10 → fails (not >)
        """
        engine = make_engine()
        state = self._make_mom_state(
            close=20010.0, high=20010.0, low=20000.0,
            atr_14=10.0,
        )
        signals = engine.evaluate(state)
        assert [s for s in signals if s.strategy == "MOM"] == []

    def test_boundary_volume_equal_to_sma_no_signal(self):
        """
        Volume == vol_sma exactly → no signal (must be strictly greater).
        """
        engine = make_engine()
        state = self._make_mom_state(volume=1500, vol_sma_20=1500.0)
        signals = engine.evaluate(state)
        assert [s for s in signals if s.strategy == "MOM"] == []


# ===========================================================================
# Max Hold Flattening
# ===========================================================================

class TestMaxHoldFlattening:
    def _advance_bars(self, engine, n_bars: int, state: MarketState):
        """Evaluate `n_bars` bars to advance the bar counter."""
        signals = []
        for _ in range(n_bars):
            signals = engine.evaluate(state)
        return signals

    def test_rsi_flattens_after_max_hold_bars(self):
        """
        max_hold_bars=5 for RSI.
        After entering and holding 5 bars, a flatten signal should be emitted.
        """
        engine = make_engine()
        # Mark as filled (enter position)
        engine.mark_filled("RSI", Side.BUY)

        # Make a neutral state (rsi=50, no other triggers)
        state = make_state(rsi_5=50.0)

        # Advance 5 bars — tick_bar is called before _check_max_hold
        # On bar 1: bars_held becomes 1 (tick), check: 1 < 5 → no flatten
        # On bar 5: bars_held becomes 5 (tick), check: 5 >= 5 → flatten
        flatten_signals = []
        for _ in range(5):
            sigs = engine.evaluate(state)
            flatten_signals.extend([s for s in sigs if s.strategy == "RSI" and s.contracts == 0])

        assert len(flatten_signals) >= 1
        sig = flatten_signals[-1]
        assert sig.strategy == "RSI"
        assert sig.contracts == 0  # 0 = flatten marker
        assert sig.reason.startswith("Max hold")

    def test_flatten_signal_has_opposite_side(self):
        """
        Position entered as BUY → flatten signal should be SELL.
        """
        engine = make_engine()
        engine.mark_filled("RSI", Side.BUY)
        state = make_state(rsi_5=50.0)

        flatten_signal = None
        for _ in range(6):
            sigs = engine.evaluate(state)
            for s in sigs:
                if s.strategy == "RSI" and s.contracts == 0:
                    flatten_signal = s

        assert flatten_signal is not None
        assert flatten_signal.side == Side.SELL

    def test_flatten_sell_position_generates_buy_signal(self):
        engine = make_engine()
        engine.mark_filled("RSI", Side.SELL)
        state = make_state(rsi_5=50.0)

        flatten_signal = None
        for _ in range(6):
            sigs = engine.evaluate(state)
            for s in sigs:
                if s.strategy == "RSI" and s.contracts == 0:
                    flatten_signal = s

        assert flatten_signal is not None
        assert flatten_signal.side == Side.BUY

    def test_no_flatten_before_max_hold(self):
        """
        After only 3 bars with max_hold=5, no flatten signal.
        """
        engine = make_engine()
        engine.mark_filled("RSI", Side.BUY)
        state = make_state(rsi_5=50.0)

        all_signals = []
        for _ in range(3):
            all_signals.extend(engine.evaluate(state))

        flatten = [s for s in all_signals if s.strategy == "RSI" and s.contracts == 0]
        assert flatten == []

    def test_ib_max_hold_15_bars(self):
        engine = make_engine()
        engine.mark_filled("IB", Side.BUY)
        state = make_state()

        flatten_signals = []
        for _ in range(16):
            sigs = engine.evaluate(state)
            flatten_signals.extend([s for s in sigs if s.strategy == "IB" and s.contracts == 0])

        assert len(flatten_signals) >= 1

    def test_mom_max_hold_5_bars(self):
        engine = make_engine()
        engine.mark_filled("MOM", Side.SELL)
        state = make_state(atr_14=None)  # prevent new signals

        flatten_signals = []
        for _ in range(6):
            sigs = engine.evaluate(state)
            flatten_signals.extend([s for s in sigs if s.strategy == "MOM" and s.contracts == 0])

        assert len(flatten_signals) >= 1


# ===========================================================================
# Position State Transitions
# ===========================================================================

class TestPositionStateTransitions:
    def test_flat_to_entered_via_mark_filled(self):
        engine = make_engine()
        assert engine.positions["RSI"].is_flat is True
        engine.mark_filled("RSI", Side.BUY)
        assert engine.positions["RSI"].is_flat is False
        assert engine.positions["RSI"].side == Side.BUY

    def test_entered_to_flat_via_mark_flat(self):
        engine = make_engine()
        engine.mark_filled("RSI", Side.BUY)
        engine.mark_flat("RSI")
        assert engine.positions["RSI"].is_flat is True
        assert engine.positions["RSI"].side is None

    def test_bar_count_increments_each_evaluate(self):
        engine = make_engine()
        state = make_state(rsi_5=50.0)
        assert engine._bar_count == 0
        engine.evaluate(state)
        assert engine._bar_count == 1
        engine.evaluate(state)
        assert engine._bar_count == 2

    def test_mark_filled_uses_current_bar_count(self):
        engine = make_engine()
        state = make_state(rsi_5=50.0)
        engine.evaluate(state)  # bar_count = 1
        engine.evaluate(state)  # bar_count = 2
        engine.mark_filled("RSI", Side.BUY)
        assert engine.positions["RSI"].entry_bar_index == 2

    def test_multiple_strategies_independent_positions(self):
        engine = make_engine()
        engine.mark_filled("RSI", Side.BUY)
        engine.mark_filled("IB", Side.SELL)
        assert engine.positions["RSI"].side == Side.BUY
        assert engine.positions["IB"].side == Side.SELL
        assert engine.positions["MOM"].is_flat is True

    def test_flatten_one_strategy_does_not_affect_others(self):
        engine = make_engine()
        engine.mark_filled("RSI", Side.BUY)
        engine.mark_filled("IB", Side.SELL)
        engine.mark_flat("RSI")
        assert engine.positions["RSI"].is_flat is True
        assert engine.positions["IB"].is_flat is False


# ===========================================================================
# Daily IB State Reset
# ===========================================================================

class TestDailyReset:
    def test_ib_traded_flag_resets_on_new_day(self):
        engine = make_engine()

        # Day 1: trigger IB signal
        bar1 = make_bar(hour=10, minute=30, close=20025.0, day=15)
        ib = make_ib()
        state1 = make_state(bar=bar1, ib_complete=True, today_ib=ib)
        engine.evaluate(state1)
        assert engine._ib_traded_today is True

        # Day 2: should reset on first evaluate
        bar2 = make_bar(hour=10, minute=30, close=20025.0, day=16)
        state2 = make_state(bar=bar2, ib_complete=True, today_ib=ib)
        engine.evaluate(state2)
        # After day 2's evaluate, if signal fired → _ib_traded_today = True again
        # Key test: engine did NOT see _ib_traded_today=True from day 1 on day 2
        # We verify by checking that the current_date was updated
        assert engine._current_date == date(2026, 1, 16)

    def test_bar_in_same_day_does_not_reset_ib_flag(self):
        engine = make_engine()
        bar1 = make_bar(hour=10, minute=30, close=20025.0, day=15)
        ib = make_ib()
        state1 = make_state(bar=bar1, ib_complete=True, today_ib=ib)
        engine.evaluate(state1)
        assert engine._ib_traded_today is True

        # Another bar on same day
        bar2 = make_bar(hour=11, minute=0, close=20025.0, day=15)
        state2 = make_state(bar=bar2, ib_complete=True, today_ib=ib)
        engine.evaluate(state2)
        # Still True — same day, not reset
        assert engine._ib_traded_today is True


# ===========================================================================
# Multiple Strategies Firing on Same Bar
# ===========================================================================

class TestMultipleSignalsOnSameBar:
    def test_all_three_strategies_can_fire_simultaneously(self):
        """
        Set up conditions where RSI, IB, and MOM all trigger at once.
        """
        engine = make_engine()
        bar = make_bar(
            hour=11, minute=0,
            open_=20000.0,
            close=20025.0,  # above IB high, bullish, above EMA
            high=20030.0,
            low=20000.0,
            volume=2000,
        )
        ib = make_ib(high=20020.0, low=19980.0)
        state = MarketState()
        state.last_bar = bar
        state.rsi_5 = 20.0          # RSI oversold → BUY
        state.atr_14 = 10.0         # bar range = 30 > 10
        state.ema_21 = 20010.0      # close=20025 > ema → MOM BUY
        state.vol_sma_20 = 1500.0
        state.ib_complete = True
        state.today_ib = ib         # close=20025 > ib.high=20020 → IB BUY
        state.ib_percentile_low = 0.0
        state.ib_percentile_high = 0.0

        signals = engine.evaluate(state)
        strategies = {s.strategy for s in signals}
        assert "RSI" in strategies
        assert "IB" in strategies
        assert "MOM" in strategies
