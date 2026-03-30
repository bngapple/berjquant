"""
Technical Indicators — RSI, ATR, EMA, SMA
All calculations match standard definitions used in backtesting.
Operates on lists/arrays of floats (close prices, volumes, etc).
"""

import numpy as np
from typing import Optional


def sma(data: list[float], period: int) -> Optional[float]:
    """Simple Moving Average. Returns None if insufficient data."""
    if len(data) < period:
        return None
    return float(np.mean(data[-period:]))


def ema(data: list[float], period: int) -> Optional[float]:
    """
    Exponential Moving Average.
    Uses the full history for proper EMA seeding (SMA for first `period` values,
    then recursive EMA from there).
    """
    if len(data) < period:
        return None
    multiplier = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    ema_val = float(np.mean(data[:period]))
    for price in data[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def rsi(closes: list[float], period: int = 5) -> Optional[float]:
    """
    RSI (Relative Strength Index) using Wilder's smoothing method.
    Matches standard RSI(5) calculation.
    Returns 0-100 scale, or None if insufficient data.
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Initial average gain/loss over first `period` changes
    gains = [max(c, 0) for c in changes[:period]]
    losses = [abs(min(c, 0)) for c in changes[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for remaining changes
    for c in changes[period:]:
        gain = max(c, 0)
        loss = abs(min(c, 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> Optional[float]:
    """
    Average True Range using Wilder's smoothing.
    Needs at least `period + 1` bars.
    """
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None

    # True ranges
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    # Initial ATR = simple average of first `period` TRs
    atr_val = sum(trs[:period]) / period

    # Wilder smoothing
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period

    return atr_val


def bar_range(high: float, low: float) -> float:
    """Single bar's range (high - low)."""
    return high - low


def percentile(data: list[float], pct: float) -> float:
    """Calculate percentile (0-100) of a dataset."""
    if not data:
        return 0.0
    return float(np.percentile(data, pct))
