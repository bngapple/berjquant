"""
Price action signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (with columns: timestamp, open, high,
low, close at minimum) and returns the same DataFrame with new signal columns
appended.
"""

import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_date(ts_col: str = "timestamp") -> pl.Expr:
    """Extract the trading date from a timestamp.

    For CME equity futures the electronic session rolls at 18:00 ET, but for
    simplicity we assign bars before midnight to the *next* calendar date only
    when they fall in the evening session.  A pragmatic approach: use the
    calendar date of each bar directly (works correctly for RTH-only data and
    is a reasonable approximation for 24-hour data).
    """
    return pl.col(ts_col).dt.date().alias("_trading_date")


# ---------------------------------------------------------------------------
# 1. Session Levels — open / high / low for the current trading day
# ---------------------------------------------------------------------------

def session_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Session open, high, low — resets each trading day.

    Adds columns:
        session_open              — first open of the day
        session_high              — running high of the day
        session_low               — running low of the day
        signal_above_session_open — close > session open (bool)
        signal_below_session_open — close < session open (bool)
    """
    df = df.with_columns(_trading_date())

    # Session open: first open per trading date
    session_opens = (
        df.group_by("_trading_date", maintain_order=True)
        .agg(pl.col("open").first().alias("session_open"))
    )
    df = df.join(session_opens, on="_trading_date", how="left")

    # Running high / low within each day using cumulative max/min
    df = df.with_columns([
        pl.col("high")
        .cum_max()
        .over("_trading_date")
        .alias("session_high"),
        pl.col("low")
        .cum_min()
        .over("_trading_date")
        .alias("session_low"),
    ])

    df = df.with_columns([
        (pl.col("close") > pl.col("session_open")).alias("signal_above_session_open"),
        (pl.col("close") < pl.col("session_open")).alias("signal_below_session_open"),
    ])

    return df.drop("_trading_date")


# ---------------------------------------------------------------------------
# 2. Previous Day Levels — prior day high / low / close
# ---------------------------------------------------------------------------

def previous_day_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Previous trading day's high, low, and close.

    Adds columns:
        prev_day_high              — previous day's high
        prev_day_low               — previous day's low
        prev_day_close             — previous day's close
        entry_long_prev_low_bounce — price touches prev low and bounces up
        entry_short_prev_high_reject — price touches prev high and reverses down
    """
    df = df.with_columns(_trading_date())

    # Aggregate daily OHLC
    daily = (
        df.group_by("_trading_date", maintain_order=True)
        .agg([
            pl.col("high").max().alias("_day_high"),
            pl.col("low").min().alias("_day_low"),
            pl.col("close").last().alias("_day_close"),
        ])
    )

    # Shift to get *previous* day values
    daily = daily.with_columns([
        pl.col("_day_high").shift(1).alias("prev_day_high"),
        pl.col("_day_low").shift(1).alias("prev_day_low"),
        pl.col("_day_close").shift(1).alias("prev_day_close"),
    ]).drop(["_day_high", "_day_low", "_day_close"])

    df = df.join(daily, on="_trading_date", how="left")

    # Bounce off prev low: bar's low touches (<=) prev_day_low and close > prev_day_low
    df = df.with_columns([
        (
            (pl.col("low") <= pl.col("prev_day_low"))
            & (pl.col("close") > pl.col("prev_day_low"))
        ).fill_null(False).alias("entry_long_prev_low_bounce"),
        (
            (pl.col("high") >= pl.col("prev_day_high"))
            & (pl.col("close") < pl.col("prev_day_high"))
        ).fill_null(False).alias("entry_short_prev_high_reject"),
    ])

    return df.drop("_trading_date")


# ---------------------------------------------------------------------------
# 3. Range Breakout — consolidation range breakout
# ---------------------------------------------------------------------------

def range_breakout(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Consolidation range breakout over a rolling lookback window.

    Adds columns:
        range_high          — rolling max of high over lookback bars
        range_low           — rolling min of low over lookback bars
        range_width         — range_high - range_low
        entry_long_breakout — close breaks above range_high (prior bar's range)
        entry_short_breakout — close breaks below range_low (prior bar's range)
    """
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=lookback).alias("range_high"),
        pl.col("low").rolling_min(window_size=lookback).alias("range_low"),
    ])

    df = df.with_columns(
        (pl.col("range_high") - pl.col("range_low")).alias("range_width"),
    )

    # Compare current close against *previous* bar's range to detect the
    # breakout bar (avoids look-ahead: the range is fully formed before
    # the current bar).
    df = df.with_columns([
        (pl.col("close") > pl.col("range_high").shift(1))
        .fill_null(False)
        .alias("entry_long_breakout"),
        (pl.col("close") < pl.col("range_low").shift(1))
        .fill_null(False)
        .alias("entry_short_breakout"),
    ])

    return df


# ---------------------------------------------------------------------------
# 4. Pivot Points — standard floor pivots from previous day
# ---------------------------------------------------------------------------

def pivot_points(df: pl.DataFrame) -> pl.DataFrame:
    """Standard floor pivot points derived from the previous trading day.

    Adds columns:
        pivot, r1, r2, r3, s1, s2, s3
    No entry signals — these are reference levels.
    """
    df = df.with_columns(_trading_date())

    daily = (
        df.group_by("_trading_date", maintain_order=True)
        .agg([
            pl.col("high").max().alias("_day_high"),
            pl.col("low").min().alias("_day_low"),
            pl.col("close").last().alias("_day_close"),
        ])
    )

    # Compute pivots from previous day
    daily = daily.with_columns([
        pl.col("_day_high").shift(1).alias("_ph"),
        pl.col("_day_low").shift(1).alias("_pl"),
        pl.col("_day_close").shift(1).alias("_pc"),
    ])

    daily = daily.with_columns(
        ((pl.col("_ph") + pl.col("_pl") + pl.col("_pc")) / 3.0).alias("pivot"),
    )

    daily = daily.with_columns([
        (2.0 * pl.col("pivot") - pl.col("_pl")).alias("r1"),
        (pl.col("pivot") + pl.col("_ph") - pl.col("_pl")).alias("r2"),
        (pl.col("_ph") + 2.0 * (pl.col("pivot") - pl.col("_pl"))).alias("r3"),
        (2.0 * pl.col("pivot") - pl.col("_ph")).alias("s1"),
        (pl.col("pivot") - pl.col("_ph") + pl.col("_pl")).alias("s2"),
        (pl.col("_pl") - 2.0 * (pl.col("_ph") - pl.col("pivot"))).alias("s3"),
    ])

    pivot_cols = daily.select(["_trading_date", "pivot", "r1", "r2", "r3", "s1", "s2", "s3"])

    df = df.join(pivot_cols, on="_trading_date", how="left")

    return df.drop(["_trading_date", "_day_high", "_day_low", "_day_close"],
                   strict=False)


# ---------------------------------------------------------------------------
# 5. Opening Range Breakout — first N minutes of RTH (9:30 ET)
# ---------------------------------------------------------------------------

def opening_range(df: pl.DataFrame, or_minutes: int = 30) -> pl.DataFrame:
    """Opening range breakout: first *or_minutes* of the core session (9:30 ET).

    Adds columns:
        or_high                — high of the opening range
        or_low                 — low of the opening range
        entry_long_or_breakout — close breaks above OR high after the OR period
        entry_short_or_breakout — close breaks below OR low after the OR period

    Assumes the ``timestamp`` column is in US/Eastern (or a timezone-aware
    equivalent).  If timestamps are UTC, convert before calling.
    """
    df = df.with_columns([
        _trading_date(),
        pl.col("timestamp").dt.hour().alias("_hour"),
        pl.col("timestamp").dt.minute().alias("_minute"),
    ])

    # Compute minutes since 9:30
    df = df.with_columns(
        ((pl.col("_hour") - 9) * 60 + pl.col("_minute") - 30).alias("_mins_since_open"),
    )

    # Flag bars within the opening range
    df = df.with_columns(
        (
            (pl.col("_mins_since_open") >= 0)
            & (pl.col("_mins_since_open") < or_minutes)
        ).alias("_in_or"),
    )

    # OR high / low per day (only from bars in the opening range)
    or_levels = (
        df.filter(pl.col("_in_or"))
        .group_by("_trading_date", maintain_order=True)
        .agg([
            pl.col("high").max().alias("or_high"),
            pl.col("low").min().alias("or_low"),
        ])
    )

    df = df.join(or_levels, on="_trading_date", how="left")

    # Breakout signals: only *after* the OR period has ended
    df = df.with_columns([
        (
            (~pl.col("_in_or"))
            & (pl.col("_mins_since_open") >= or_minutes)
            & (pl.col("close") > pl.col("or_high"))
        ).fill_null(False).alias("entry_long_or_breakout"),
        (
            (~pl.col("_in_or"))
            & (pl.col("_mins_since_open") >= or_minutes)
            & (pl.col("close") < pl.col("or_low"))
        ).fill_null(False).alias("entry_short_or_breakout"),
    ])

    return df.drop(["_trading_date", "_hour", "_minute", "_mins_since_open", "_in_or"])


# ---------------------------------------------------------------------------
# 6. Candle Patterns — engulfing, pin bar, inside bar
# ---------------------------------------------------------------------------

def candle_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """Basic candlestick pattern detection.

    Adds columns:
        signal_engulfing_bullish  — bullish engulfing pattern
        signal_engulfing_bearish  — bearish engulfing pattern
        signal_pin_bar_bullish    — long lower wick, small body at top
        signal_pin_bar_bearish    — long upper wick, small body at bottom
        signal_inside_bar         — current bar fully inside previous bar's range
    """
    df = df.with_columns([
        (pl.col("close") - pl.col("open")).alias("_body"),
        (pl.col("high") - pl.col("low")).alias("_range"),
    ])

    df = df.with_columns([
        pl.col("_body").shift(1).alias("_prev_body"),
        pl.col("open").shift(1).alias("_prev_open"),
        pl.col("close").shift(1).alias("_prev_close"),
        pl.col("high").shift(1).alias("_prev_high"),
        pl.col("low").shift(1).alias("_prev_low"),
    ])

    # Bullish engulfing: previous bar bearish, current bar bullish and body
    # engulfs previous body
    df = df.with_columns([
        (
            (pl.col("_prev_body") < 0)  # prev bearish
            & (pl.col("_body") > 0)      # current bullish
            & (pl.col("open") <= pl.col("_prev_close"))
            & (pl.col("close") >= pl.col("_prev_open"))
        ).fill_null(False).alias("signal_engulfing_bullish"),

        (
            (pl.col("_prev_body") > 0)  # prev bullish
            & (pl.col("_body") < 0)      # current bearish
            & (pl.col("open") >= pl.col("_prev_close"))
            & (pl.col("close") <= pl.col("_prev_open"))
        ).fill_null(False).alias("signal_engulfing_bearish"),
    ])

    # Pin bars: body is small relative to range, wick is dominant
    # Bullish pin bar: long lower wick, body at top third
    body_abs = pl.col("_body").abs()
    upper_wick = pl.col("high") - pl.max_horizontal("open", "close")
    lower_wick = pl.min_horizontal("open", "close") - pl.col("low")

    df = df.with_columns([
        (
            (body_abs < pl.col("_range") * 0.33)   # small body
            & (lower_wick > pl.col("_range") * 0.60)  # long lower wick
            & (pl.col("_range") > 0)
        ).fill_null(False).alias("signal_pin_bar_bullish"),

        (
            (body_abs < pl.col("_range") * 0.33)   # small body
            & (upper_wick > pl.col("_range") * 0.60)  # long upper wick
            & (pl.col("_range") > 0)
        ).fill_null(False).alias("signal_pin_bar_bearish"),
    ])

    # Inside bar: current high/low within previous bar's range
    df = df.with_columns(
        (
            (pl.col("high") <= pl.col("_prev_high"))
            & (pl.col("low") >= pl.col("_prev_low"))
        ).fill_null(False).alias("signal_inside_bar"),
    )

    return df.drop([
        "_body", "_range", "_prev_body",
        "_prev_open", "_prev_close", "_prev_high", "_prev_low",
    ])
