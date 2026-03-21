"""
Trend signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (with at minimum: open, high, low, close columns)
and returns the DataFrame with new signal columns appended.
"""

import polars as pl
import numpy as np


# ---------------------------------------------------------------------------
# 1. EMA Crossover
# ---------------------------------------------------------------------------

def ema_crossover(df: pl.DataFrame, fast_period: int, slow_period: int) -> pl.DataFrame:
    """Classic EMA crossover. Detects when fast EMA crosses above/below slow EMA."""
    df = df.with_columns([
        pl.col("close").ewm_mean(span=fast_period).alias("ema_fast"),
        pl.col("close").ewm_mean(span=slow_period).alias("ema_slow"),
    ])

    df = df.with_columns([
        pl.col("ema_fast").shift(1).alias("_ema_fast_prev"),
        pl.col("ema_slow").shift(1).alias("_ema_slow_prev"),
    ])

    df = df.with_columns([
        (
            (pl.col("ema_fast") > pl.col("ema_slow"))
            & (pl.col("_ema_fast_prev") <= pl.col("_ema_slow_prev"))
        ).alias("entry_long_ema_cross"),
        (
            (pl.col("ema_fast") < pl.col("ema_slow"))
            & (pl.col("_ema_fast_prev") >= pl.col("_ema_slow_prev"))
        ).alias("entry_short_ema_cross"),
    ])

    df = df.with_columns([
        pl.col("entry_long_ema_cross").fill_null(False),
        pl.col("entry_short_ema_cross").fill_null(False),
    ])

    return df.drop(["_ema_fast_prev", "_ema_slow_prev"])


# ---------------------------------------------------------------------------
# 2. EMA Slope
# ---------------------------------------------------------------------------

def ema_slope(df: pl.DataFrame, period: int, slope_lookback: int = 3) -> pl.DataFrame:
    """EMA direction/slope. Positive slope = uptrend, negative = downtrend."""
    ema_col = f"ema_{period}"
    slope_col = f"ema_slope_{period}"

    df = df.with_columns(
        pl.col("close").ewm_mean(span=period).alias(ema_col),
    )

    df = df.with_columns(
        (pl.col(ema_col) - pl.col(ema_col).shift(slope_lookback)).alias(slope_col),
    )

    df = df.with_columns([
        (pl.col(slope_col) > 0).fill_null(False).alias("signal_ema_slope_up"),
        (pl.col(slope_col) < 0).fill_null(False).alias("signal_ema_slope_down"),
    ])

    return df


# ---------------------------------------------------------------------------
# 3. EMA Ribbon
# ---------------------------------------------------------------------------

def ema_ribbon(
    df: pl.DataFrame,
    periods: list[int] | None = None,
) -> pl.DataFrame:
    """Multiple EMAs for trend strength. Bullish when all EMAs are in ascending order
    (fastest on top), bearish when all are in descending order."""
    if periods is None:
        periods = [8, 13, 21, 34, 55]

    sorted_periods = sorted(periods)
    ema_cols = []

    for p in sorted_periods:
        col_name = f"ema_{p}"
        ema_cols.append(col_name)
        df = df.with_columns(
            pl.col("close").ewm_mean(span=p).alias(col_name),
        )

    # Bullish ribbon: fastest EMA (shortest period) > next > ... > slowest
    # i.e. ema_8 > ema_13 > ema_21 > ema_34 > ema_55
    bullish_expr = pl.lit(True)
    bearish_expr = pl.lit(True)
    for i in range(len(ema_cols) - 1):
        bullish_expr = bullish_expr & (pl.col(ema_cols[i]) > pl.col(ema_cols[i + 1]))
        bearish_expr = bearish_expr & (pl.col(ema_cols[i]) < pl.col(ema_cols[i + 1]))

    df = df.with_columns([
        bullish_expr.fill_null(False).alias("signal_ema_ribbon_bullish"),
        bearish_expr.fill_null(False).alias("signal_ema_ribbon_bearish"),
    ])

    return df


# ---------------------------------------------------------------------------
# 4. Linear Regression Slope
# ---------------------------------------------------------------------------

def linear_regression_slope(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Rolling linear regression slope of close price. Positive slope = uptrend.

    Fix #21: Vectorized using numpy sliding_window_view instead of Python loop.
    """
    slope_col = f"linreg_slope_{period}"

    close = df.get_column("close").to_numpy().astype(np.float64)
    n = len(close)
    slopes = np.full(n, np.nan)

    if n >= period:
        from numpy.lib.stride_tricks import sliding_window_view
        x = np.arange(period, dtype=np.float64)
        x_mean = x.mean()
        x_dev = x - x_mean
        x_var = (x_dev ** 2).sum()

        windows = sliding_window_view(close, period)  # (n-period+1, period)
        y_means = windows.mean(axis=1)
        # Vectorized: sum((x - x_mean) * (y - y_mean)) / x_var for each window
        slopes[period - 1:] = (windows - y_means[:, np.newaxis]) @ x_dev / x_var

    df = df.with_columns(
        pl.Series(name=slope_col, values=slopes),
    )

    df = df.with_columns([
        (pl.col(slope_col) > 0).fill_null(False).alias("signal_linreg_up"),
        (pl.col(slope_col) < 0).fill_null(False).alias("signal_linreg_down"),
    ])

    return df


# ---------------------------------------------------------------------------
# 5. Heikin-Ashi
# ---------------------------------------------------------------------------

def heikin_ashi(df: pl.DataFrame) -> pl.DataFrame:
    """Heikin-Ashi candle transformation. Smooths price action for clearer trend reading."""
    # HA close = (open + high + low + close) / 4
    df = df.with_columns(
        ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
        .alias("ha_close"),
    )

    # HA open: first bar = (open + close) / 2, then recursive: (prev_ha_open + prev_ha_close) / 2
    # Must compute iteratively due to recursive dependency
    ha_close = df.get_column("ha_close").to_numpy()
    open_arr = df.get_column("open").to_numpy()
    close_arr = df.get_column("close").to_numpy()
    n = len(df)

    ha_open = np.empty(n)
    ha_open[0] = (open_arr[0] + close_arr[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    df = df.with_columns(
        pl.Series(name="ha_open", values=ha_open),
    )

    # HA high = max(high, ha_open, ha_close), HA low = min(low, ha_open, ha_close)
    df = df.with_columns([
        pl.max_horizontal("high", "ha_open", "ha_close").alias("ha_high"),
        pl.min_horizontal("low", "ha_open", "ha_close").alias("ha_low"),
    ])

    df = df.with_columns([
        (pl.col("ha_close") > pl.col("ha_open")).alias("signal_ha_bullish"),
        (pl.col("ha_close") < pl.col("ha_open")).alias("signal_ha_bearish"),
    ])

    return df


# ---------------------------------------------------------------------------
# 6. Supertrend
# ---------------------------------------------------------------------------

def supertrend(df: pl.DataFrame, period: int = 10, multiplier: float = 3.0) -> pl.DataFrame:
    """ATR-based Supertrend indicator. Flips between support (bullish) and resistance (bearish)."""
    # True Range
    df = df.with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("_tr"),
    ])

    # ATR via rolling mean
    df = df.with_columns(
        pl.col("_tr").rolling_mean(window_size=period).alias("_atr"),
    )

    # Basic upper/lower bands
    hl2 = (df.get_column("high").to_numpy() + df.get_column("low").to_numpy()) / 2.0
    atr = df.get_column("_atr").to_numpy()
    close = df.get_column("close").to_numpy()
    n = len(df)

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Final bands (with clamping) and direction, computed iteratively
    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)
    st = np.full(n, np.nan)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish

    for i in range(1, n):
        # Clamp upper band: only decrease
        if np.isnan(final_upper[i]):
            pass
        elif not np.isnan(final_upper[i - 1]) and close[i - 1] <= final_upper[i - 1]:
            final_upper[i] = min(basic_upper[i], final_upper[i - 1])

        # Clamp lower band: only increase
        if np.isnan(final_lower[i]):
            pass
        elif not np.isnan(final_lower[i - 1]) and close[i - 1] >= final_lower[i - 1]:
            final_lower[i] = max(basic_lower[i], final_lower[i - 1])

        # Direction logic
        if np.isnan(atr[i]):
            direction[i] = direction[i - 1]
            continue

        if direction[i - 1] == 1:  # was bullish
            if close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:  # was bearish
            if close[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1

    # Supertrend value: lower band when bullish, upper band when bearish
    for i in range(n):
        if direction[i] == 1:
            st[i] = final_lower[i]
        else:
            st[i] = final_upper[i]

    df = df.with_columns([
        pl.Series(name="supertrend", values=st),
        pl.Series(name="signal_supertrend_bullish", values=direction == 1),
        pl.Series(name="signal_supertrend_bearish", values=direction == -1),
    ])

    df = df.drop(["_tr", "_atr"])

    return df
