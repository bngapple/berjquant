"""
Momentum signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (with at minimum 'close', and for
stochastic also 'high'/'low') and returns the same DataFrame with new
signal columns appended.
"""

import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crosses_above(series: pl.Expr, level: float) -> pl.Expr:
    """True on the bar where *series* crosses from <= level to > level."""
    return (series > level) & (series.shift(1) <= level)


def _crosses_below(series: pl.Expr, level: float) -> pl.Expr:
    """True on the bar where *series* crosses from >= level to < level."""
    return (series < level) & (series.shift(1) >= level)


def _crosses_above_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """True on the bar where expr *a* crosses above expr *b*."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crosses_below_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """True on the bar where expr *a* crosses below expr *b*."""
    return (a < b) & (a.shift(1) >= b.shift(1))


# ---------------------------------------------------------------------------
# 1. RSI — Relative Strength Index (Wilder smoothing)
# ---------------------------------------------------------------------------

def rsi(
    df: pl.DataFrame,
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> pl.DataFrame:
    """Relative Strength Index with Wilder exponential smoothing.

    Adds columns:
        rsi_{period}        — RSI value (0-100)
        entry_long_rsi      — crosses above oversold level
        entry_short_rsi     — crosses below overbought level
    """
    col = f"rsi_{period}"
    delta = df["close"].diff()

    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    # Wilder smoothing: EWM with alpha = 1/period (equivalent to com=period-1)
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)

    rs = avg_gain / avg_loss
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))

    df = df.with_columns(pl.Series(name=col, values=rsi_vals))

    df = df.with_columns([
        _crosses_above(pl.col(col), oversold).alias("entry_long_rsi"),
        _crosses_below(pl.col(col), overbought).alias("entry_short_rsi"),
    ])

    return df


# ---------------------------------------------------------------------------
# 2. MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------

def macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """MACD with signal line and histogram.

    Adds columns:
        macd_line           — fast EMA minus slow EMA
        macd_signal         — EMA of the MACD line
        macd_histogram      — macd_line minus macd_signal
        entry_long_macd     — histogram crosses above 0
        entry_short_macd    — histogram crosses below 0
    """
    close = df["close"]

    ema_fast = close.ewm_mean(span=fast, adjust=False)
    ema_slow = close.ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow

    macd_signal = macd_line.ewm_mean(span=signal_period, adjust=False)
    macd_hist = macd_line - macd_signal

    df = df.with_columns([
        pl.Series(name="macd_line", values=macd_line),
        pl.Series(name="macd_signal", values=macd_signal),
        pl.Series(name="macd_histogram", values=macd_hist),
    ])

    df = df.with_columns([
        _crosses_above(pl.col("macd_histogram"), 0.0).alias("entry_long_macd"),
        _crosses_below(pl.col("macd_histogram"), 0.0).alias("entry_short_macd"),
    ])

    return df


# ---------------------------------------------------------------------------
# 3. Stochastic Oscillator
# ---------------------------------------------------------------------------

def stochastic(
    df: pl.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    overbought: float = 80.0,
    oversold: float = 20.0,
) -> pl.DataFrame:
    """Stochastic Oscillator (%K and %D).

    Adds columns:
        stoch_k             — raw %K
        stoch_d             — SMA of %K
        entry_long_stoch    — K crosses above D while in oversold zone
        entry_short_stoch   — K crosses below D while in overbought zone
    """
    lowest_low = df["low"].rolling_min(window_size=k_period)
    highest_high = df["high"].rolling_max(window_size=k_period)

    k = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100.0
    d = k.rolling_mean(window_size=d_period)

    df = df.with_columns([
        pl.Series(name="stoch_k", values=k),
        pl.Series(name="stoch_d", values=d),
    ])

    df = df.with_columns([
        (
            _crosses_above_expr(pl.col("stoch_k"), pl.col("stoch_d"))
            & (pl.col("stoch_k") < oversold)
        ).alias("entry_long_stoch"),
        (
            _crosses_below_expr(pl.col("stoch_k"), pl.col("stoch_d"))
            & (pl.col("stoch_k") > overbought)
        ).alias("entry_short_stoch"),
    ])

    return df


# ---------------------------------------------------------------------------
# 4. ROC — Rate of Change
# ---------------------------------------------------------------------------

def roc(
    df: pl.DataFrame,
    period: int = 10,
) -> pl.DataFrame:
    """Rate of Change (percentage).

    Adds columns:
        roc_{period}        — percentage change over *period* bars
        entry_long_roc      — crosses above 0
        entry_short_roc     — crosses below 0
    """
    col = f"roc_{period}"

    roc_vals = ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100.0

    df = df.with_columns(pl.Series(name=col, values=roc_vals))

    df = df.with_columns([
        _crosses_above(pl.col(col), 0.0).alias("entry_long_roc"),
        _crosses_below(pl.col(col), 0.0).alias("entry_short_roc"),
    ])

    return df


# ---------------------------------------------------------------------------
# 5. CCI — Commodity Channel Index
# ---------------------------------------------------------------------------

def cci(
    df: pl.DataFrame,
    period: int = 20,
) -> pl.DataFrame:
    """Commodity Channel Index.

    Adds columns:
        cci_{period}        — CCI value
        entry_long_cci      — crosses above -100
        entry_short_cci     — crosses below +100
    """
    col = f"cci_{period}"

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_sma = tp.rolling_mean(window_size=period)

    # Mean absolute deviation (rolling)
    # Polars doesn't have a built-in rolling MAD, so we compute it via map.
    tp_series = tp.to_list()
    mad_vals: list[float | None] = [None] * len(tp_series)
    for i in range(period - 1, len(tp_series)):
        window = tp_series[i - period + 1 : i + 1]
        if any(v is None for v in window):
            continue
        mean = sum(window) / period  # type: ignore[arg-type]
        mad_vals[i] = sum(abs(v - mean) for v in window) / period  # type: ignore[arg-type, operator]

    mad = pl.Series("mad", mad_vals, dtype=pl.Float64)

    cci_vals = (tp - tp_sma) / (0.015 * mad)

    df = df.with_columns(pl.Series(name=col, values=cci_vals))

    df = df.with_columns([
        _crosses_above(pl.col(col), -100.0).alias("entry_long_cci"),
        _crosses_below(pl.col(col), 100.0).alias("entry_short_cci"),
    ])

    return df


# ---------------------------------------------------------------------------
# 6. Williams %R
# ---------------------------------------------------------------------------

def williams_r(
    df: pl.DataFrame,
    period: int = 14,
    overbought: float = -20.0,
    oversold: float = -80.0,
) -> pl.DataFrame:
    """Williams %R oscillator.

    Adds columns:
        williams_r_{period}     — Williams %R value (-100 to 0)
        entry_long_williams     — crosses above oversold level
        entry_short_williams    — crosses below overbought level
    """
    col = f"williams_r_{period}"

    highest_high = df["high"].rolling_max(window_size=period)
    lowest_low = df["low"].rolling_min(window_size=period)

    wr = ((highest_high - df["close"]) / (highest_high - lowest_low)) * -100.0

    df = df.with_columns(pl.Series(name=col, values=wr))

    df = df.with_columns([
        _crosses_above(pl.col(col), oversold).alias("entry_long_williams"),
        _crosses_below(pl.col(col), overbought).alias("entry_short_williams"),
    ])

    return df
