"""Volatility signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (with OHLC columns: open, high, low, close)
and returns the DataFrame with new signal columns appended.
"""

import polars as pl
import numpy as np


# ---------------------------------------------------------------------------
# 1. Average True Range
# ---------------------------------------------------------------------------

def atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Average True Range — volatility measure used for sizing and filters.

    Adds column: atr_{period}
    """
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )

    df = df.with_columns(tr.alias("_tr"))
    df = df.with_columns(
        pl.col("_tr")
        .rolling_mean(window_size=period)
        .alias(f"atr_{period}")
    )
    return df.drop("_tr")


# ---------------------------------------------------------------------------
# 2. Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.DataFrame:
    """Bollinger Bands — mean-reversion signals from price vs. volatility envelope.

    Adds columns: bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b,
                  entry_long_bb, entry_short_bb
    """
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=period).alias("bb_middle"),
        pl.col("close").rolling_std(window_size=period).alias("_bb_std"),
    ])

    df = df.with_columns([
        (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
    ])

    df = df.with_columns([
        (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width"),
        (
            (pl.col("close") - pl.col("bb_lower"))
            / (pl.col("bb_upper") - pl.col("bb_lower"))
        ).alias("bb_pct_b"),
    ])

    # Mean-reversion entry signals (cross-based)
    df = df.with_columns([
        pl.col("close").shift(1).alias("_prev_close"),
    ])

    df = df.with_columns([
        # Long: close crosses above lower band (was below, now above)
        (
            (pl.col("_prev_close") <= pl.col("bb_lower").shift(1))
            & (pl.col("close") > pl.col("bb_lower"))
        ).fill_null(False).alias("entry_long_bb"),
        # Short: close crosses below upper band (was above, now below)
        (
            (pl.col("_prev_close") >= pl.col("bb_upper").shift(1))
            & (pl.col("close") < pl.col("bb_upper"))
        ).fill_null(False).alias("entry_short_bb"),
    ])

    return df.drop(["_bb_std", "_prev_close"])


# ---------------------------------------------------------------------------
# 3. Keltner Channels
# ---------------------------------------------------------------------------

def keltner_channels(
    df: pl.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 1.5,
) -> pl.DataFrame:
    """Keltner Channels — EMA +/- ATR*multiplier envelope.

    Adds columns: kc_upper, kc_middle, kc_lower,
                  entry_long_kc, entry_short_kc
    """
    # EMA middle line
    df = df.with_columns(
        pl.col("close").ewm_mean(span=ema_period).alias("kc_middle"),
    )

    # ATR for channel width
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(tr.alias("_kc_tr"))
    df = df.with_columns(
        pl.col("_kc_tr").rolling_mean(window_size=atr_period).alias("_kc_atr"),
    )

    df = df.with_columns([
        (pl.col("kc_middle") + multiplier * pl.col("_kc_atr")).alias("kc_upper"),
        (pl.col("kc_middle") - multiplier * pl.col("_kc_atr")).alias("kc_lower"),
    ])

    # Entry signals (cross-based)
    df = df.with_columns(
        pl.col("close").shift(1).alias("_prev_close"),
    )

    df = df.with_columns([
        # Long: price crosses above lower channel
        (
            (pl.col("_prev_close") <= pl.col("kc_lower").shift(1))
            & (pl.col("close") > pl.col("kc_lower"))
        ).fill_null(False).alias("entry_long_kc"),
        # Short: price crosses below upper channel
        (
            (pl.col("_prev_close") >= pl.col("kc_upper").shift(1))
            & (pl.col("close") < pl.col("kc_upper"))
        ).fill_null(False).alias("entry_short_kc"),
    ])

    return df.drop(["_kc_tr", "_kc_atr", "_prev_close"])


# ---------------------------------------------------------------------------
# 4. Bollinger-Keltner Squeeze (TTM Squeeze)
# ---------------------------------------------------------------------------

def bollinger_keltner_squeeze(
    df: pl.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_period: int = 14,
    kc_mult: float = 1.5,
) -> pl.DataFrame:
    """TTM Squeeze — detects low-volatility compression and expansion.

    BB inside KC = squeeze on (low vol). When squeeze releases, momentum fires.

    Adds columns: signal_squeeze_on (bool), signal_squeeze_fire (bool),
                  squeeze_momentum (float)
    """
    # --- Bollinger Bands ---
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=bb_period).alias("_sq_bb_mid"),
        pl.col("close").rolling_std(window_size=bb_period).alias("_sq_bb_std"),
    ])
    df = df.with_columns([
        (pl.col("_sq_bb_mid") + bb_std * pl.col("_sq_bb_std")).alias("_sq_bb_upper"),
        (pl.col("_sq_bb_mid") - bb_std * pl.col("_sq_bb_std")).alias("_sq_bb_lower"),
    ])

    # --- Keltner Channels ---
    df = df.with_columns(
        pl.col("close").ewm_mean(span=kc_period).alias("_sq_kc_mid"),
    )
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(tr.alias("_sq_tr"))
    df = df.with_columns(
        pl.col("_sq_tr").rolling_mean(window_size=kc_atr_period).alias("_sq_kc_atr"),
    )
    df = df.with_columns([
        (pl.col("_sq_kc_mid") + kc_mult * pl.col("_sq_kc_atr")).alias("_sq_kc_upper"),
        (pl.col("_sq_kc_mid") - kc_mult * pl.col("_sq_kc_atr")).alias("_sq_kc_lower"),
    ])

    # --- Squeeze detection ---
    df = df.with_columns(
        (
            (pl.col("_sq_bb_lower") > pl.col("_sq_kc_lower"))
            & (pl.col("_sq_bb_upper") < pl.col("_sq_kc_upper"))
        ).fill_null(False).alias("signal_squeeze_on"),
    )

    # Squeeze fire: was on, now off
    df = df.with_columns(
        (
            pl.col("signal_squeeze_on").shift(1).fill_null(False)
            & ~pl.col("signal_squeeze_on")
        ).alias("signal_squeeze_fire"),
    )

    # --- Momentum (linear regression-style via deviation from midline) ---
    # Simplified momentum: close minus average of (BB mid + KC mid) / 2
    # Then smoothed — approximates the TTM momentum histogram
    df = df.with_columns(
        (
            pl.col("close")
            - (pl.col("_sq_bb_mid") + pl.col("_sq_kc_mid")) / 2.0
        ).alias("squeeze_momentum"),
    )

    # Clean up temp columns
    temp_cols = [
        "_sq_bb_mid", "_sq_bb_std", "_sq_bb_upper", "_sq_bb_lower",
        "_sq_kc_mid", "_sq_tr", "_sq_kc_atr", "_sq_kc_upper", "_sq_kc_lower",
    ]
    return df.drop(temp_cols)


# ---------------------------------------------------------------------------
# 5. ATR Percentile
# ---------------------------------------------------------------------------

def atr_percentile(
    df: pl.DataFrame,
    atr_period: int = 14,
    lookback: int = 100,
) -> pl.DataFrame:
    """ATR percentile — current ATR ranked against its own rolling history.

    Adds columns: atr_percentile (0-100). Useful as a volatility filter.
    """
    atr_col = f"atr_{atr_period}"

    # Compute ATR if not already present
    if atr_col not in df.columns:
        df = atr(df, period=atr_period)

    # Rolling percentile rank: fraction of values in the lookback <= current value
    # Polars doesn't have a built-in rolling_rank, so we use a map approach
    atr_vals = df[atr_col].to_numpy()
    pct = np.full(len(atr_vals), np.nan)
    for i in range(lookback - 1, len(atr_vals)):
        window = atr_vals[max(0, i - lookback + 1) : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pct[i] = (np.sum(valid <= atr_vals[i]) / len(valid)) * 100.0

    df = df.with_columns(
        pl.Series("atr_percentile", pct, dtype=pl.Float64),
    )
    return df


# ---------------------------------------------------------------------------
# 6. Historical Volatility
# ---------------------------------------------------------------------------

def historical_volatility(
    df: pl.DataFrame,
    period: int = 20,
) -> pl.DataFrame:
    """Annualized historical volatility from log returns.

    Adds column: hv_{period}
    Assumes ~252 trading days for annualization.
    """
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_log_ret"),
    )

    df = df.with_columns(
        (
            pl.col("_log_ret").rolling_std(window_size=period)
            * (252.0 ** 0.5)
        ).alias(f"hv_{period}"),
    )

    return df.drop("_log_ret")
