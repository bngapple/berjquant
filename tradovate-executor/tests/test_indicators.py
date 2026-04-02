"""
tests/test_indicators.py

Unit tests for all indicator functions in indicators.py.
Verifies exact math using hand-computed sequences, including
Wilder's smoothing for RSI and ATR.

All expected values are derived independently from the formulas,
not from the implementation.
"""

import sys
import os
import math
import pytest

# Add project root to sys.path so imports resolve without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import sma, ema, rsi, atr, bar_range, percentile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(actual, expected, rel_tol=1e-6, msg=""):
    """Assert two floats are within rel_tol of each other."""
    assert actual is not None, f"Expected {expected}, got None. {msg}"
    assert math.isclose(actual, expected, rel_tol=rel_tol), (
        f"Expected {expected}, got {actual:.8f}. {msg}"
    )


# ===========================================================================
# SMA
# ===========================================================================

class TestSMA:
    def test_exact_value_period_3(self):
        # mean([10, 20, 30]) = 20.0
        result = sma([10.0, 20.0, 30.0], period=3)
        assert_close(result, 20.0)

    def test_uses_only_last_period_elements(self):
        # sma([1,2,3,4,5], period=3) = mean([3,4,5]) = 4.0
        result = sma([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
        assert_close(result, 4.0)

    def test_exact_period_length(self):
        # Exactly period elements
        result = sma([10.0, 20.0, 30.0], period=3)
        assert_close(result, 20.0)

    def test_period_1_returns_last_value(self):
        result = sma([5.0, 10.0, 99.0], period=1)
        assert_close(result, 99.0)

    def test_insufficient_data_returns_none(self):
        assert sma([10.0, 20.0], period=3) is None

    def test_empty_list_returns_none(self):
        assert sma([], period=3) is None

    def test_single_element_period_1(self):
        result = sma([42.0], period=1)
        assert_close(result, 42.0)

    def test_all_same_values(self):
        result = sma([7.0, 7.0, 7.0, 7.0, 7.0], period=5)
        assert_close(result, 7.0)

    def test_float_precision(self):
        # Verify floating point doesn't introduce large errors
        result = sma([0.1, 0.2, 0.3], period=3)
        assert_close(result, 0.2, rel_tol=1e-9)

    def test_large_dataset_period(self):
        data = [float(i) for i in range(1, 101)]  # 1..100
        # sma of last 10 = mean(91..100) = 95.5
        result = sma(data, period=10)
        assert_close(result, 95.5)

    def test_negative_values(self):
        result = sma([-10.0, -20.0, -30.0], period=3)
        assert_close(result, -20.0)

    def test_mixed_positive_negative(self):
        result = sma([-5.0, 5.0, -5.0, 5.0], period=4)
        assert_close(result, 0.0)

    def test_returns_float_not_numpy_scalar(self):
        result = sma([1.0, 2.0, 3.0], period=3)
        assert isinstance(result, float)


# ===========================================================================
# EMA
# ===========================================================================

class TestEMA:
    def test_insufficient_data_returns_none(self):
        assert ema([10.0, 20.0], period=3) is None

    def test_exactly_period_values_returns_sma(self):
        # When len(data) == period, EMA seeds with SMA and applies no recursive step
        # => result equals the SMA = mean([10,20,30]) = 20.0
        result = ema([10.0, 20.0, 30.0], period=3)
        assert_close(result, 20.0)

    def test_one_extra_bar_applies_one_recursive_step(self):
        # period=3, multiplier = 2/(3+1) = 0.5
        # seed = mean([10,20,30]) = 20.0
        # step: ema = (40 - 20) * 0.5 + 20 = 30.0
        result = ema([10.0, 20.0, 30.0, 40.0], period=3)
        assert_close(result, 30.0)

    def test_ascending_series_ema_period_5(self):
        # Manually trace:
        # data = [1,2,3,4,5,6,7,8,9,10], period=5
        # multiplier = 2/6 = 0.33333...
        # seed = mean([1,2,3,4,5]) = 3.0
        # bar6=6: ema = (6-3)*0.3333 + 3 = 4.0
        # bar7=7: ema = (7-4)*0.3333 + 4 = 5.0
        # bar8=8: ema = (8-5)*0.3333 + 5 = 6.0
        # bar9=9: ema = (9-6)*0.3333 + 6 = 7.0
        # bar10=10: ema = (10-7)*0.3333 + 7 = 8.0
        result = ema([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], period=5)
        assert_close(result, 8.0, rel_tol=1e-5)

    def test_ema_lags_sma_on_uptrend(self):
        # EMA gives more weight to recent prices, so it's higher than SMA when
        # prices are *accelerating* upward (not just linearly trending — for a
        # perfect linear trend EMA == SMA in steady state).
        data = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]  # exponential
        ema_val = ema(data, period=5)
        sma_val = sma(data, period=5)
        assert ema_val is not None
        assert sma_val is not None
        # EMA > SMA for accelerating (exponential) uptrend
        assert ema_val > sma_val

    def test_all_same_values_equals_that_value(self):
        result = ema([5.0] * 10, period=5)
        assert_close(result, 5.0)

    def test_returns_float_not_numpy_scalar(self):
        result = ema([1.0, 2.0, 3.0, 4.0, 5.0], period=5)
        assert isinstance(result, float)

    def test_period_1_returns_last_value(self):
        # multiplier = 2/(1+1) = 1.0; seed = data[0]; ema = (data[i]-prev)*1 + prev = data[i]
        result = ema([10.0, 20.0, 30.0], period=1)
        assert_close(result, 30.0)

    def test_empty_list_returns_none(self):
        assert ema([], period=3) is None


# ===========================================================================
# RSI — Wilder's Smoothing
# ===========================================================================

class TestRSI:
    """
    All expected RSI values computed by hand using Wilder's method:
      1. Compute changes[i] = closes[i] - closes[i-1]
      2. Initial avg_gain/avg_loss = mean of first `period` gains/losses
      3. For each subsequent change: avg = (avg*(period-1) + val) / period
      4. RS = avg_gain / avg_loss
      5. RSI = 100 - 100/(1+RS)
    """

    # --- Insufficient Data ---

    def test_needs_period_plus_one_values(self):
        # period=5 needs len >= 6
        assert rsi([100.0] * 5, period=5) is None

    def test_exactly_period_plus_one_values(self):
        # Exactly 6 prices for period=5: should return a value
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        result = rsi(closes, period=5)
        assert result is not None

    def test_empty_returns_none(self):
        assert rsi([], period=5) is None

    def test_single_value_returns_none(self):
        assert rsi([100.0], period=5) is None

    # --- All Gains (RSI = 100) ---

    def test_all_gains_returns_100(self):
        # No losses → avg_loss = 0 → RSI = 100
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        result = rsi(closes, period=5)
        assert_close(result, 100.0)

    # --- All Losses (RSI = 0) ---

    def test_all_losses_returns_0(self):
        # No gains → avg_gain = 0 → RS = 0 → RSI = 0
        closes = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        result = rsi(closes, period=5)
        assert_close(result, 0.0)

    # --- Mixed: verified against Wilder's formula ---

    def test_mixed_changes_period_5_wilder(self):
        """
        closes = [44, 47, 48, 44, 46, 43, 44, 48, 51, 49]
        changes = [3, 1, -4, 2, -3, 1, 4, 3, -2]

        Initial 5 changes: [3, 1, -4, 2, -3]
        gains:  [3, 1, 0, 2, 0]  → avg_gain = 6/5 = 1.2
        losses: [0, 0, 4, 0, 3]  → avg_loss = 7/5 = 1.4

        Wilder smoothing over remaining [1, 4, 3, -2]:
        c=1:  g=1, l=0  → avg_gain=(1.2*4+1)/5=1.16, avg_loss=(1.4*4+0)/5=1.12
        c=4:  g=4, l=0  → avg_gain=(1.16*4+4)/5=1.728, avg_loss=(1.12*4+0)/5=0.896
        c=3:  g=3, l=0  → avg_gain=(1.728*4+3)/5=1.9824, avg_loss=(0.896*4+0)/5=0.7168
        c=-2: g=0, l=2  → avg_gain=(1.9824*4+0)/5=1.58592, avg_loss=(0.7168*4+2)/5=0.97344

        RS = 1.58592 / 0.97344 = 1.62904...
        RSI = 100 - 100/(1+1.62904) = 100 - 100/2.62904 = 62.0...
        """
        closes = [44.0, 47.0, 48.0, 44.0, 46.0, 43.0, 44.0, 48.0, 51.0, 49.0]
        result = rsi(closes, period=5)
        assert result is not None

        # Re-derive independently
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(c, 0) for c in changes[:5]]
        losses = [abs(min(c, 0)) for c in changes[:5]]
        ag = sum(gains) / 5
        al = sum(losses) / 5
        for c in changes[5:]:
            g = max(c, 0)
            lo = abs(min(c, 0))
            ag = (ag * 4 + g) / 5
            al = (al * 4 + lo) / 5
        rs = ag / al
        expected = 100.0 - 100.0 / (1.0 + rs)

        assert_close(result, expected, rel_tol=1e-9)

    def test_rsi_below_35_for_strong_downtrend(self):
        # Consistent price drops → oversold RSI
        closes = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0]
        result = rsi(closes, period=5)
        assert result is not None
        assert result < 35, f"Expected RSI < 35 for strong downtrend, got {result:.2f}"

    def test_rsi_above_65_for_strong_uptrend(self):
        closes = [80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0]
        result = rsi(closes, period=5)
        assert result is not None
        assert result > 65, f"Expected RSI > 65 for strong uptrend, got {result:.2f}"

    def test_rsi_range_0_to_100(self):
        # RSI must always be within [0, 100]
        closes = [100.0, 105.0, 102.0, 108.0, 103.0, 110.0, 107.0, 115.0]
        result = rsi(closes, period=5)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_rsi_returns_float(self):
        closes = [100.0, 101.0, 100.0, 102.0, 101.0, 103.0]
        result = rsi(closes, period=5)
        assert isinstance(result, float)

    def test_wilder_smoothing_differs_from_simple_average(self):
        """
        Wilder smoothing is NOT the same as re-averaging all changes each bar.
        Verify that with 10 bars the result matches the Wilder formula, not simple EMA.
        """
        closes = [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0]
        result = rsi(closes, period=5)

        # Recompute inline
        ch = [closes[i] - closes[i-1] for i in range(1, 10)]
        ag = sum(max(c, 0) for c in ch[:5]) / 5
        al = sum(abs(min(c, 0)) for c in ch[:5]) / 5
        for c in ch[5:]:
            ag = (ag * 4 + max(c, 0)) / 5
            al = (al * 4 + abs(min(c, 0))) / 5

        if al == 0:
            expected = 100.0
        else:
            expected = 100.0 - 100.0 / (1.0 + ag / al)

        assert_close(result, expected, rel_tol=1e-9)

    def test_flat_prices_after_initial_rsi(self):
        # When avg_loss = 0 (pure gains), RSI = 100.0 by Wilder's definition.
        # Flat bars produce 0 changes, so avg_loss stays 0 and avg_gain decays —
        # but RS = avg_gain / 0 is still infinity, so RSI stays at 100.0.
        closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]  # pure gains first
        result_up = rsi(closes, period=5)
        assert result_up == 100.0  # all gains → avg_loss = 0

        # Extend with flat bars — avg_loss stays 0, RS stays ∞, RSI stays 100
        flat = [110.0] * 10
        result_flat = rsi(closes + flat, period=5)
        assert result_flat is not None
        assert result_flat == 100.0  # avg_loss=0 → RSI=100 (Wilder's convention)

    def test_period_3_minimal(self):
        # period=3 needs at least 4 closes
        closes = [10.0, 12.0, 11.0, 13.0]
        result = rsi(closes, period=3)
        assert result is not None
        assert 0.0 <= result <= 100.0


# ===========================================================================
# ATR — Wilder's Smoothing
# ===========================================================================

class TestATR:
    """
    ATR computation:
      TR[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)
      Initial ATR = mean(TR[1..period])
      Wilder: ATR[n] = (ATR[n-1]*(period-1) + TR[n]) / period
    """

    def test_insufficient_data_returns_none(self):
        # period=14 needs 15 bars
        highs = [101.0] * 14
        lows = [99.0] * 14
        closes = [100.0] * 14
        assert atr(highs, lows, closes, period=14) is None

    def test_empty_returns_none(self):
        assert atr([], [], [], period=14) is None

    def test_mismatched_lengths_returns_none(self):
        highs = [101.0, 102.0, 103.0]
        lows = [99.0, 100.0]  # one shorter
        closes = [100.0, 101.0, 102.0]
        assert atr(highs, lows, closes, period=2) is None

    def test_uniform_bars_exact_atr(self):
        """
        Bars with H=101, L=99, C=100 and previous C=100 each bar.
        TR[i] = max(101-99, |101-100|, |99-100|) = max(2, 1, 1) = 2 for all.
        Initial ATR (period=14) = mean of 14 TRs = 2.0.
        No extra bars beyond seed → ATR = 2.0.
        """
        period = 14
        n = period + 1  # exactly enough for seeding only
        highs = [101.0] * n
        lows = [99.0] * n
        closes = [100.0] * n
        result = atr(highs, lows, closes, period=period)
        assert_close(result, 2.0)

    def test_wilder_smoothing_one_extra_bar(self):
        """
        15 seed bars with TR=2, then one extra bar with TR=4.
        seed ATR = 2.0
        Wilder: (2.0*13 + 4) / 14 = (26+4)/14 = 30/14 = 2.142857...
        """
        period = 14
        highs = [101.0] * 15 + [103.0]
        lows = [99.0] * 15 + [99.0]
        closes = [100.0] * 16

        result = atr(highs, lows, closes, period=period)
        expected = (2.0 * 13 + 4.0) / 14
        assert_close(result, expected)

    def test_gap_up_increases_atr(self):
        """
        Simulate a gap: previous close = 100, new bar H=102, L=101.
        TR = max(102-101, |102-100|, |101-100|) = max(1, 2, 1) = 2.
        Regular bars with TR=1 should produce ATR < 2 (the gapped bar's TR).
        """
        period = 3
        # Tight regular bars: H=100.5, L=99.5, C=100 → TR=1
        highs  = [100.5, 100.5, 100.5, 102.0]
        lows   = [99.5,  99.5,  99.5, 101.0]
        closes = [100.0, 100.0, 100.0, 101.5]

        result = atr(highs, lows, closes, period=period)
        assert result is not None
        # The gap bar adds TR=2 (|102-100|), previous bars had TR=1
        # Initial: (1+1+1)/3 = 1.0; after gap: (1*2 + 2)/3 = 1.333...
        assert result > 1.0

    def test_atr_positive_always(self):
        highs  = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
        lows   = [99.0,  100.0, 101.0, 102.0, 103.0, 104.0]
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        result = atr(highs, lows, closes, period=5)
        assert result is not None
        assert result > 0.0

    def test_atr_period_3_exact(self):
        """
        Hand-compute period=3 ATR:
        bars:   H    L    C
        bar0: 105  95  100  (only for prev close reference)
        bar1: 103  97  100   TR = max(6, |103-100|, |97-100|) = max(6,3,3) = 6
        bar2: 102  98  100   TR = max(4, |102-100|, |98-100|) = max(4,2,2) = 4
        bar3: 104  96  100   TR = max(8, |104-100|, |96-100|) = max(8,4,4) = 8
        bar4: 101  99  100   TR = max(2, |101-100|, |99-100|) = max(2,1,1) = 2

        Initial ATR = mean(TR[1..3]) = (6+4+8)/3 = 6.0
        Wilder: (6.0*2 + 2) / 3 = 14/3 = 4.6667
        """
        highs  = [105.0, 103.0, 102.0, 104.0, 101.0]
        lows   = [95.0,  97.0,  98.0,  96.0,  99.0]
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]
        result = atr(highs, lows, closes, period=3)
        expected = 14.0 / 3.0
        assert_close(result, expected)

    def test_atr_returns_float(self):
        period = 5
        n = period + 2
        highs  = [101.0] * n
        lows   = [99.0]  * n
        closes = [100.0] * n
        result = atr(highs, lows, closes, period=period)
        assert isinstance(result, float)


# ===========================================================================
# bar_range
# ===========================================================================

class TestBarRange:
    def test_basic(self):
        assert bar_range(105.0, 100.0) == 5.0

    def test_zero_range(self):
        assert bar_range(100.0, 100.0) == 0.0

    def test_very_small_range(self):
        assert_close(bar_range(100.25, 100.0), 0.25)

    def test_returns_float(self):
        result = bar_range(110.0, 100.0)
        assert isinstance(result, float)


# ===========================================================================
# percentile
# ===========================================================================

class TestPercentile:
    def test_median_of_odd_list(self):
        # Percentile 50 of [1,2,3,4,5] = 3.0
        result = percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        assert_close(result, 3.0)

    def test_p0_returns_minimum(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert_close(percentile(data, 0), 10.0)

    def test_p100_returns_maximum(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert_close(percentile(data, 100), 50.0)

    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_element(self):
        assert_close(percentile([42.0], 25), 42.0)

    def test_p25_p75_range_filter(self):
        # 50 IB ranges between 20 and 70
        data = [float(i) for i in range(20, 70)]
        p25 = percentile(data, 25)
        p75 = percentile(data, 75)
        # A range of 44 should be between p25 and p75 of 20..69
        assert p25 < 44.0 < p75

    def test_returns_float(self):
        result = percentile([1.0, 2.0, 3.0], 50)
        assert isinstance(result, float)


# ===========================================================================
# Integration: indicator pipeline reproduces correct RSI(5) on realistic data
# ===========================================================================

class TestIndicatorPipeline:
    """
    End-to-end validation: feed a sequence of 15-minute bar closes that
    represent a realistic intraday session and verify RSI direction.
    """

    def test_rsi5_tracks_price_direction(self):
        """
        Given a sequence of 20 bars:
        - first 10 bars: price rises from 18000 to 18050
        - last 10 bars: price falls from 18050 to 18000
        RSI(5) computed on first 15 bars (rising end) should be > 50.
        RSI(5) computed on all 20 bars (falling end) should be < 50.
        """
        rising  = [18000.0 + i * 5 for i in range(10)]
        falling = [18050.0 - i * 5 for i in range(1, 11)]

        rsi_rising = rsi(rising, period=5)
        rsi_falling = rsi(rising + falling, period=5)

        assert rsi_rising is not None
        assert rsi_falling is not None
        assert rsi_rising > 50, f"RSI on rising bars should be > 50, got {rsi_rising:.2f}"
        assert rsi_falling < 50, f"RSI on falling bars should be < 50, got {rsi_falling:.2f}"

    def test_atr14_period_boundary(self):
        """
        Exactly 15 bars of data (period=14): ATR is computed using only
        the first seed average (no Wilder smoothing steps).
        All TRs = 2.0 → ATR = 2.0.
        """
        highs  = [101.0] * 15
        lows   = [99.0]  * 15
        closes = [100.0] * 15
        result = atr(highs, lows, closes, period=14)
        assert_close(result, 2.0)

    def test_atr14_still_none_at_14_bars(self):
        """
        14 bars: not enough for period=14 (need 15).
        """
        highs  = [101.0] * 14
        lows   = [99.0]  * 14
        closes = [100.0] * 14
        result = atr(highs, lows, closes, period=14)
        assert result is None

    def test_ema21_needs_21_values(self):
        assert ema([100.0] * 20, period=21) is None
        assert ema([100.0] * 21, period=21) is not None
