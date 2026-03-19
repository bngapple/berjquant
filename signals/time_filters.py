"""
Time-based filter signals for NQ/MNQ futures trading.

These are context/filter signals (not primary entry signals).  Each function
takes a Polars DataFrame with a ``datetime`` column (US Eastern Time) and
returns the same DataFrame with new boolean or numeric columns appended.

All times assume US Eastern Time.
"""

import polars as pl


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _minute_of_day() -> pl.Expr:
    """Return an i32 expression for the minute-of-day (0–1439).

    Polars ``dt.hour()`` returns i8, which overflows when multiplied by 60,
    so we cast to i32 first.
    """
    return (
        pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
        + pl.col("timestamp").dt.minute().cast(pl.Int32)
    )


# ---------------------------------------------------------------------------
# 1. Time-of-Day Window
# ---------------------------------------------------------------------------

def time_of_day(
    df: pl.DataFrame,
    start_hour: int,
    start_minute: int,
    end_hour: int,
    end_minute: int,
) -> pl.DataFrame:
    """Filter bars to a specific time-of-day window.

    Adds ``signal_in_time_window`` (bool) — True when the bar falls within
    [start_hour:start_minute, end_hour:end_minute).

    Example: ``time_of_day(df, 9, 30, 11, 0)`` keeps the 9:30–11:00 ET window.
    """
    start_total = start_hour * 60 + start_minute
    end_total = end_hour * 60 + end_minute
    mod = _minute_of_day()

    return df.with_columns(
        (mod.ge(start_total) & mod.lt(end_total))
        .alias("signal_in_time_window")
    )


# ---------------------------------------------------------------------------
# 2. Day-of-Week
# ---------------------------------------------------------------------------

def day_of_week(
    df: pl.DataFrame,
    allowed_days: list[int] | None = None,
) -> pl.DataFrame:
    """Tag each bar with its weekday and whether it falls on an allowed day.

    Adds:
      - ``day_of_week``        (u32)  0 = Monday … 4 = Friday
      - ``signal_day_allowed`` (bool) True if the weekday is in *allowed_days*

    *allowed_days* defaults to ``[0, 1, 2, 3, 4]`` (Mon–Fri).
    """
    if allowed_days is None:
        allowed_days = [0, 1, 2, 3, 4]

    # Polars weekday(): 1=Mon..7=Sun  ->  subtract 1 to get 0=Mon..6=Sun
    dow = pl.col("timestamp").dt.weekday().cast(pl.Int32) - 1

    return df.with_columns(
        dow.cast(pl.UInt32).alias("day_of_week"),
        dow.is_in(allowed_days).alias("signal_day_allowed"),
    )


# ---------------------------------------------------------------------------
# 3. Session Segment
# ---------------------------------------------------------------------------

def session_segment(df: pl.DataFrame) -> pl.DataFrame:
    """Tag each bar by market session.

    Adds:
      - ``session_segment``      (str)  one of "pre_market", "core",
        "post_close", "outside"
      - ``signal_is_premarket``  (bool)
      - ``signal_is_core``       (bool)
      - ``signal_is_postclose``  (bool)

    Session boundaries (ET):
      - pre_market : 08:00 – 09:30
      - core       : 09:30 – 16:00
      - post_close : 16:00 – 17:00
      - outside    : everything else
    """
    mod = _minute_of_day()

    pre_start = 8 * 60          # 08:00
    core_start = 9 * 60 + 30    # 09:30
    core_end = 16 * 60          # 16:00
    post_end = 17 * 60          # 17:00

    segment = (
        pl.when(mod.ge(pre_start) & mod.lt(core_start))
        .then(pl.lit("pre_market"))
        .when(mod.ge(core_start) & mod.lt(core_end))
        .then(pl.lit("core"))
        .when(mod.ge(core_end) & mod.lt(post_end))
        .then(pl.lit("post_close"))
        .otherwise(pl.lit("outside"))
    )

    cols: list[pl.Expr] = []
    if "session_segment" not in df.columns:
        cols.append(segment.alias("session_segment"))

    cols.extend([
        (segment.eq(pl.lit("core"))).alias("signal_is_core"),
        (segment.eq(pl.lit("pre_market"))).alias("signal_is_premarket"),
        (segment.eq(pl.lit("post_close"))).alias("signal_is_postclose"),
    ])

    return df.with_columns(cols)


# ---------------------------------------------------------------------------
# 4. Minutes Since Open
# ---------------------------------------------------------------------------

def minutes_since_open(
    df: pl.DataFrame,
    open_hour: int = 9,
    open_minute: int = 30,
) -> pl.DataFrame:
    """Compute elapsed minutes since market open for each bar.

    Adds ``minutes_since_open`` (i64).  Negative values indicate bars before
    the open.
    """
    open_total = open_hour * 60 + open_minute
    mod = _minute_of_day()

    return df.with_columns(
        (mod - open_total).cast(pl.Int64).alias("minutes_since_open")
    )


# ---------------------------------------------------------------------------
# 5. First N Minutes
# ---------------------------------------------------------------------------

def first_n_minutes(
    df: pl.DataFrame,
    n: int = 30,
    open_hour: int = 9,
    open_minute: int = 30,
) -> pl.DataFrame:
    """Flag bars that fall within the first *n* minutes after the open.

    Adds ``signal_first_n_minutes`` (bool).
    """
    open_total = open_hour * 60 + open_minute
    mod = _minute_of_day()

    return df.with_columns(
        (mod.ge(open_total) & mod.lt(open_total + n))
        .alias("signal_first_n_minutes")
    )


# ---------------------------------------------------------------------------
# 6. Last N Minutes
# ---------------------------------------------------------------------------

def last_n_minutes(
    df: pl.DataFrame,
    n: int = 30,
    close_hour: int = 16,
    close_minute: int = 0,
) -> pl.DataFrame:
    """Flag bars that fall within the last *n* minutes before the close.

    Adds ``signal_last_n_minutes`` (bool).
    """
    close_total = close_hour * 60 + close_minute
    mod = _minute_of_day()

    return df.with_columns(
        (mod.ge(close_total - n) & mod.lt(close_total))
        .alias("signal_last_n_minutes")
    )


# ---------------------------------------------------------------------------
# 7. London / New York Overlap
# ---------------------------------------------------------------------------

def london_overlap(df: pl.DataFrame) -> pl.DataFrame:
    """Flag bars during the London–New York session overlap (08:00–11:00 ET).

    This window often exhibits the highest intraday volatility for NQ/MNQ.

    Adds ``signal_london_overlap`` (bool).
    """
    mod = _minute_of_day()
    start = 8 * 60    # 08:00 ET
    end = 11 * 60     # 11:00 ET

    return df.with_columns(
        (mod.ge(start) & mod.lt(end)).alias("signal_london_overlap")
    )
