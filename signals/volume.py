"""Volume-based signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (expected columns: datetime, open,
high, low, close, volume) and returns the same DataFrame with new signal
columns appended.  All computations use Polars-native operations.
"""

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_col(df: pl.DataFrame) -> pl.Expr:
    """Return an expression that extracts the date from the datetime column."""
    return pl.col("timestamp").cast(pl.Date).alias("_date")


# ---------------------------------------------------------------------------
# 1. Session VWAP (resets each day)
# ---------------------------------------------------------------------------

def vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Session VWAP with standard-deviation bands and cross signals.

    Adds columns:
        vwap            — volume-weighted average price, resetting daily
        vwap_std        — rolling std of (close - vwap) within the session
        vwap_upper_1    — vwap + 1 * vwap_std
        vwap_lower_1    — vwap - 1 * vwap_std
        entry_long_vwap — price crosses above vwap from below
        entry_short_vwap — price crosses below vwap from above
    """
    df = df.with_columns(_date_col(df))

    # Cumulative price*volume and cumulative volume per session day
    df = df.with_columns([
        (pl.col("close") * pl.col("volume")).alias("_pv"),
    ])

    df = df.with_columns([
        pl.col("_pv").cum_sum().over("_date").alias("_cum_pv"),
        pl.col("volume").cum_sum().over("_date").alias("_cum_vol"),
    ])

    df = df.with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_vol")).alias("vwap"),
    )

    # Deviation and expanding std within each session
    df = df.with_columns(
        (pl.col("close") - pl.col("vwap")).alias("_dev"),
    )

    # Compute expanding std per session: sqrt(cum_sum(dev^2) / count)
    df = df.with_columns([
        (pl.col("_dev").pow(2)).alias("_dev2"),
        pl.lit(1).cum_sum().over("_date").alias("_bar_num"),
    ])
    df = df.with_columns(
        pl.col("_dev2").cum_sum().over("_date").alias("_cum_dev2"),
    )
    df = df.with_columns(
        (pl.col("_cum_dev2") / pl.col("_bar_num")).sqrt().alias("vwap_std"),
    )

    # Bands
    df = df.with_columns([
        (pl.col("vwap") + pl.col("vwap_std")).alias("vwap_upper_1"),
        (pl.col("vwap") - pl.col("vwap_std")).alias("vwap_lower_1"),
    ])

    # Crossover signals
    df = df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
        pl.col("vwap").shift(1).alias("_prev_vwap"),
    ])

    df = df.with_columns([
        (
            (pl.col("close") > pl.col("vwap"))
            & (pl.col("_prev_close") <= pl.col("_prev_vwap"))
        ).fill_null(False).alias("entry_long_vwap"),
        (
            (pl.col("close") < pl.col("vwap"))
            & (pl.col("_prev_close") >= pl.col("_prev_vwap"))
        ).fill_null(False).alias("entry_short_vwap"),
    ])

    # Cleanup temp columns
    df = df.drop([
        "_date", "_pv", "_cum_pv", "_cum_vol", "_dev", "_dev2",
        "_bar_num", "_cum_dev2", "_prev_close", "_prev_vwap",
    ])

    return df


# ---------------------------------------------------------------------------
# 2. Volume delta (buy/sell estimation)
# ---------------------------------------------------------------------------

def volume_delta(df: pl.DataFrame) -> pl.DataFrame:
    """Estimate buy/sell volume using close position within the bar.

    Uses the close-open ratio relative to the high-low range to split
    each bar's volume into buy and sell components.

    Adds columns:
        buy_volume       — estimated buy volume
        sell_volume      — estimated sell volume
        volume_delta     — buy_volume - sell_volume
        cumulative_delta — running sum of volume_delta, resets daily
    """
    df = df.with_columns(_date_col(df))

    # Buy ratio: how close the close is to the high relative to the range
    df = df.with_columns(
        pl.when(pl.col("high") != pl.col("low"))
        .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .otherwise(0.5)
        .alias("_buy_ratio"),
    )

    df = df.with_columns([
        (pl.col("volume") * pl.col("_buy_ratio")).alias("buy_volume"),
        (pl.col("volume") * (1.0 - pl.col("_buy_ratio"))).alias("sell_volume"),
    ])

    df = df.with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("volume_delta"),
    )

    df = df.with_columns(
        pl.col("volume_delta").cum_sum().over("_date").alias("cumulative_delta"),
    )

    df = df.drop(["_date", "_buy_ratio"])
    return df


# ---------------------------------------------------------------------------
# 3. Rolling volume profile
# ---------------------------------------------------------------------------

def volume_profile(
    df: pl.DataFrame,
    lookback_bars: int = 100,
    num_bins: int = 50,
) -> pl.DataFrame:
    """Rolling volume profile over a lookback window.

    Fix #19: Vectorized with numpy — eliminated O(n^2 * bins) Python loops.
    Computes at ~10-bar stride for large datasets, interpolating between.

    Adds columns:
        vpoc             — volume point of control (price with most volume)
        vah              — value area high (70th percentile price by volume)
        val              — value area low (30th percentile price by volume)
        signal_above_vah — close is above value area high
        signal_below_val — close is below value area low
    """
    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)
    volumes = df["volume"].to_numpy().astype(np.float64)
    n = len(df)

    vpoc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    # Compute every `stride` bars, then forward-fill
    stride = max(1, min(10, n // 100))

    for i in range(0, n, stride):
        start = max(0, i - lookback_bars + 1)
        wh = highs[start:i + 1]
        wl = lows[start:i + 1]
        wv = volumes[start:i + 1]

        range_high = wh.max()
        range_low = wl.min()
        total_vol = wv.sum()

        if range_high == range_low or total_vol == 0:
            vpoc_arr[i] = closes[i]
            vah_arr[i] = closes[i]
            val_arr[i] = closes[i]
            continue

        bin_size = (range_high - range_low) / num_bins
        bin_volumes = np.zeros(num_bins)

        # Vectorized bin assignment
        lo_bins = np.clip(((wl - range_low) / bin_size).astype(int), 0, num_bins - 1)
        hi_bins = np.clip(((wh - range_low) / bin_size).astype(int), 0, num_bins - 1)

        for j in range(len(wv)):
            lb, hb = lo_bins[j], hi_bins[j]
            num_touched = hb - lb + 1
            if num_touched > 0:
                bin_volumes[lb:hb + 1] += wv[j] / num_touched

        vpoc_arr[i] = range_low + (np.argmax(bin_volumes) + 0.5) * bin_size

        cum = np.cumsum(bin_volumes)
        cum_pct = cum / total_vol
        val_idx = np.searchsorted(cum_pct, 0.30)
        vah_idx = np.searchsorted(cum_pct, 0.70)
        val_idx = min(val_idx, num_bins - 1)
        vah_idx = min(vah_idx, num_bins - 1)

        val_arr[i] = range_low + (val_idx + 0.5) * bin_size
        vah_arr[i] = range_low + (vah_idx + 0.5) * bin_size

    # Forward-fill the gaps from striding
    for arr in (vpoc_arr, vah_arr, val_arr):
        mask = np.isnan(arr)
        if mask.any():
            idx = np.where(~mask, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            arr[mask] = arr[idx[mask]]

    df = df.with_columns([
        pl.Series("vpoc", vpoc_arr, dtype=pl.Float64),
        pl.Series("vah", vah_arr, dtype=pl.Float64),
        pl.Series("val", val_arr, dtype=pl.Float64),
    ])

    df = df.with_columns([
        (pl.col("close") > pl.col("vah")).alias("signal_above_vah"),
        (pl.col("close") < pl.col("val")).alias("signal_below_val"),
    ])

    return df


# ---------------------------------------------------------------------------
# 4. Relative volume
# ---------------------------------------------------------------------------

def relative_volume(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Current bar volume relative to its rolling average.

    Adds columns:
        rvol               — ratio of current volume to rolling mean (1.0 = average)
        signal_high_volume — True when rvol > 1.5
    """
    df = df.with_columns(
        pl.col("volume").rolling_mean(window_size=lookback).alias("_vol_avg"),
    )

    df = df.with_columns(
        (pl.col("volume") / pl.col("_vol_avg")).alias("rvol"),
    )

    df = df.with_columns(
        (pl.col("rvol") > 1.5).fill_null(False).alias("signal_high_volume"),
    )

    df = df.drop("_vol_avg")
    return df


# ---------------------------------------------------------------------------
# 5. Volume climax
# ---------------------------------------------------------------------------

def volume_climax(
    df: pl.DataFrame,
    lookback: int = 50,
    threshold: float = 2.5,
) -> pl.DataFrame:
    """Detect volume spikes and potential reversals.

    A climax bar has volume exceeding threshold * rolling mean.
    A climax reversal is a climax bar where the close reverses against
    the bar's direction (upper-wick exhaustion for up-bars, lower-wick
    for down-bars).

    Adds columns:
        signal_volume_climax   — volume exceeds threshold * rolling mean
        signal_climax_reversal — climax bar with price reversal pattern
    """
    df = df.with_columns(
        pl.col("volume").rolling_mean(window_size=lookback).alias("_vol_avg"),
    )

    df = df.with_columns(
        (pl.col("volume") > (pl.lit(threshold) * pl.col("_vol_avg")))
        .fill_null(False)
        .alias("signal_volume_climax"),
    )

    # Reversal detection: climax bar closes in opposite half of its range
    # Up-bar that closes in lower half -> bearish exhaustion
    # Down-bar that closes in upper half -> bullish exhaustion
    df = df.with_columns(
        pl.when(pl.col("high") != pl.col("low"))
        .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .otherwise(0.5)
        .alias("_close_pos"),
    )

    is_up_bar = pl.col("close") > pl.col("open")
    is_down_bar = pl.col("close") < pl.col("open")

    df = df.with_columns(
        (
            pl.col("signal_volume_climax")
            & (
                (is_up_bar & (pl.col("_close_pos") < 0.5))
                | (is_down_bar & (pl.col("_close_pos") > 0.5))
            )
        ).fill_null(False).alias("signal_climax_reversal"),
    )

    df = df.drop(["_vol_avg", "_close_pos"])
    return df
