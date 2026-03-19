"""
Order flow signal functions for NQ/MNQ futures trading.

Each function takes a Polars DataFrame (with OHLCV columns at minimum) and
returns the same DataFrame with new signal columns appended.  Where tick-level
data (e.g. cumulative_delta) is unavailable, functions estimate order flow
characteristics from bar data.

This is the core edge module — order flow analysis is what separates
profitable futures strategies from noise.
"""

import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crosses_above_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """True on the bar where expr *a* crosses above expr *b*."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crosses_below_expr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """True on the bar where expr *a* crosses below expr *b*."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def _has_column(df: pl.DataFrame, name: str) -> bool:
    return name in df.columns


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Division that returns 0 when denominator is 0 or null."""
    return pl.when(denominator == 0).then(0.0).otherwise(numerator / denominator)


# ---------------------------------------------------------------------------
# 1. Delta Divergence
# ---------------------------------------------------------------------------

def delta_divergence(
    df: pl.DataFrame,
    lookback: int = 10,
) -> pl.DataFrame:
    """Detect divergence between price and cumulative delta.

    Price makes a new high/low but cumulative delta does not confirm,
    signaling a potential reversal.

    Requires: ``cumulative_delta`` column (from volume.volume_delta).

    Adds:
        signal_delta_div_bearish — price new high, delta not (bearish)
        signal_delta_div_bullish — price new low, delta not (bullish)
    """
    if not _has_column(df, "cumulative_delta"):
        return df.with_columns(
            pl.lit(False).alias("signal_delta_div_bearish"),
            pl.lit(False).alias("signal_delta_div_bullish"),
        )

    df = df.with_columns([
        # Price makes new rolling high / low
        (pl.col("high") >= pl.col("high").rolling_max(window_size=lookback))
            .alias("_price_new_high"),
        (pl.col("low") <= pl.col("low").rolling_min(window_size=lookback))
            .alias("_price_new_low"),
        # Delta does NOT make new high / low
        (pl.col("cumulative_delta")
            < pl.col("cumulative_delta").rolling_max(window_size=lookback))
            .alias("_delta_no_new_high"),
        (pl.col("cumulative_delta")
            > pl.col("cumulative_delta").rolling_min(window_size=lookback))
            .alias("_delta_no_new_low"),
    ])

    df = df.with_columns([
        (pl.col("_price_new_high") & pl.col("_delta_no_new_high"))
            .alias("signal_delta_div_bearish"),
        (pl.col("_price_new_low") & pl.col("_delta_no_new_low"))
            .alias("signal_delta_div_bullish"),
    ])

    df = df.drop(
        ["_price_new_high", "_price_new_low",
         "_delta_no_new_high", "_delta_no_new_low"]
    )
    return df


# ---------------------------------------------------------------------------
# 2. Absorption
# ---------------------------------------------------------------------------

def absorption(
    df: pl.DataFrame,
    volume_threshold: float = 2.0,
    price_threshold: float = 0.3,
) -> pl.DataFrame:
    """Detect absorption — high volume with small price movement.

    Indicates that aggressive buyers/sellers are being absorbed by passive
    limit orders on the other side.

    Adds:
        signal_absorption          — absorption detected (bool)
        signal_absorption_bullish  — absorption near bar lows (support)
        signal_absorption_bearish  — absorption near bar highs (resistance)
    """
    df = df.with_columns([
        # Relative volume: volume / rolling mean volume
        _safe_divide(
            pl.col("volume").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64).rolling_mean(window_size=20),
        ).alias("_rel_volume"),
        # Relative price range: bar range / rolling mean range
        _safe_divide(
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("low")).rolling_mean(window_size=20),
        ).alias("_rel_range"),
    ])

    df = df.with_columns(
        (
            (pl.col("_rel_volume") >= volume_threshold)
            & (pl.col("_rel_range") <= price_threshold)
        ).alias("signal_absorption"),
    )

    # Direction: where does the close sit within the bar?
    # Close near low = bullish absorption (buyers absorbing selling)
    # Close near high = bearish absorption (sellers absorbing buying)
    bar_range = pl.col("high") - pl.col("low")
    close_position = _safe_divide(pl.col("close") - pl.col("low"), bar_range)

    df = df.with_columns([
        (pl.col("signal_absorption") & (close_position > 0.6))
            .alias("signal_absorption_bullish"),
        (pl.col("signal_absorption") & (close_position < 0.4))
            .alias("signal_absorption_bearish"),
    ])

    df = df.drop(["_rel_volume", "_rel_range"])
    return df


# ---------------------------------------------------------------------------
# 3. Exhaustion
# ---------------------------------------------------------------------------

def exhaustion(
    df: pl.DataFrame,
    delta_lookback: int = 20,
    threshold: float = 2.0,
) -> pl.DataFrame:
    """Detect exhaustion moves — large delta spike followed by stalling price.

    A large positive delta spike with price failing to advance means buying
    exhaustion (short signal).  A large negative delta spike with price
    failing to drop means selling exhaustion (long signal).

    Requires: ``cumulative_delta`` column.

    Adds:
        signal_exhaustion_long  — selling exhaustion, potential long entry
        signal_exhaustion_short — buying exhaustion, potential short entry
    """
    if not _has_column(df, "cumulative_delta"):
        return df.with_columns(
            pl.lit(False).alias("signal_exhaustion_long"),
            pl.lit(False).alias("signal_exhaustion_short"),
        )

    df = df.with_columns(
        (pl.col("cumulative_delta") - pl.col("cumulative_delta").shift(1))
            .alias("_delta_change"),
    )

    df = df.with_columns([
        pl.col("_delta_change")
            .rolling_mean(window_size=delta_lookback)
            .alias("_delta_mean"),
        pl.col("_delta_change")
            .rolling_std(window_size=delta_lookback)
            .alias("_delta_std"),
        (pl.col("close") - pl.col("close").shift(1)).alias("_price_change"),
    ])

    # Z-score of delta change
    delta_z = _safe_divide(
        pl.col("_delta_change") - pl.col("_delta_mean"),
        pl.col("_delta_std"),
    )

    # Price stalling: small absolute price change relative to recent volatility
    price_stall = (
        pl.col("_price_change").abs()
        < pl.col("_price_change").abs().rolling_mean(window_size=delta_lookback) * 0.5
    )

    df = df.with_columns([
        # Large negative delta spike + price not dropping = selling exhaustion
        ((delta_z < -threshold) & price_stall).alias("signal_exhaustion_long"),
        # Large positive delta spike + price not advancing = buying exhaustion
        ((delta_z > threshold) & price_stall).alias("signal_exhaustion_short"),
    ])

    df = df.drop(["_delta_change", "_delta_mean", "_delta_std", "_price_change"])
    return df


# ---------------------------------------------------------------------------
# 4. Buy/Sell Imbalance
# ---------------------------------------------------------------------------

def imbalance(
    df: pl.DataFrame,
    ratio_threshold: float = 3.0,
) -> pl.DataFrame:
    """Estimate buy/sell imbalance from bar data.

    Uses body-to-range ratio and close vs open to estimate buying vs selling
    pressure.  A large body relative to the full range in the direction of
    the close indicates one-sided flow.

    Adds:
        buy_sell_ratio         — estimated buy/sell ratio (>1 = more buying)
        signal_buy_imbalance   — ratio exceeds threshold (strong buying)
        signal_sell_imbalance  — ratio below 1/threshold (strong selling)
    """
    bar_range = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
    body = pl.col("close") - pl.col("open")
    body_ratio = body / bar_range  # -1 to +1

    # Convert to buy/sell ratio: map [-1,1] -> ratio
    # body_ratio > 0 means bullish; we scale so 1.0 maps to a large ratio
    # Using: ratio = (1 + body_ratio) / (1 - body_ratio), clamped
    buy_pct = (1.0 + body_ratio) / 2.0   # 0 to 1
    sell_pct = (1.0 - body_ratio) / 2.0   # 1 to 0

    df = df.with_columns(
        _safe_divide(
            buy_pct.clip(lower_bound=0.01),
            sell_pct.clip(lower_bound=0.01),
        ).alias("buy_sell_ratio"),
    )

    df = df.with_columns([
        (pl.col("buy_sell_ratio") > ratio_threshold)
            .alias("signal_buy_imbalance"),
        (pl.col("buy_sell_ratio") < (1.0 / ratio_threshold))
            .alias("signal_sell_imbalance"),
    ])

    return df


# ---------------------------------------------------------------------------
# 5. Large Trade Detection
# ---------------------------------------------------------------------------

def large_trade_detection(
    df: pl.DataFrame,
    volume_lookback: int = 50,
    threshold: float = 3.0,
) -> pl.DataFrame:
    """Flag bars with unusually large volume — likely institutional activity.

    Uses a z-score approach: volume must exceed the rolling mean by
    *threshold* standard deviations.

    Adds:
        signal_large_trade      — institutional-size volume detected (bool)
        large_trade_direction   — "long" if close > open, "short" otherwise
    """
    vol = pl.col("volume").cast(pl.Float64)

    df = df.with_columns([
        vol.rolling_mean(window_size=volume_lookback).alias("_vol_mean"),
        vol.rolling_std(window_size=volume_lookback).alias("_vol_std"),
    ])

    df = df.with_columns(
        (
            vol > (pl.col("_vol_mean") + threshold * pl.col("_vol_std"))
        ).alias("signal_large_trade"),
    )

    df = df.with_columns(
        pl.when(pl.col("signal_large_trade") & (pl.col("close") > pl.col("open")))
            .then(pl.lit("long"))
            .when(pl.col("signal_large_trade") & (pl.col("close") <= pl.col("open")))
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("large_trade_direction"),
    )

    df = df.drop(["_vol_mean", "_vol_std"])
    return df


# ---------------------------------------------------------------------------
# 6. Delta Momentum (EMA crossover on cumulative delta)
# ---------------------------------------------------------------------------

def delta_momentum(
    df: pl.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20,
) -> pl.DataFrame:
    """EMA crossover applied to cumulative delta — trend in order flow.

    Works like a MACD but on delta rather than price, capturing shifts in
    aggressive buying/selling momentum.

    Requires: ``cumulative_delta`` column.

    Adds:
        delta_ema_fast              — fast EMA of cumulative delta
        delta_ema_slow              — slow EMA of cumulative delta
        entry_long_delta_momentum   — fast crosses above slow
        entry_short_delta_momentum  — fast crosses below slow
    """
    if not _has_column(df, "cumulative_delta"):
        return df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("delta_ema_fast"),
            pl.lit(None).cast(pl.Float64).alias("delta_ema_slow"),
            pl.lit(False).alias("entry_long_delta_momentum"),
            pl.lit(False).alias("entry_short_delta_momentum"),
        )

    cd = df["cumulative_delta"].cast(pl.Float64)

    df = df.with_columns([
        pl.Series(name="delta_ema_fast",
                  values=cd.ewm_mean(span=fast_period, adjust=False)),
        pl.Series(name="delta_ema_slow",
                  values=cd.ewm_mean(span=slow_period, adjust=False)),
    ])

    df = df.with_columns([
        _crosses_above_expr(
            pl.col("delta_ema_fast"), pl.col("delta_ema_slow"),
        ).alias("entry_long_delta_momentum"),
        _crosses_below_expr(
            pl.col("delta_ema_fast"), pl.col("delta_ema_slow"),
        ).alias("entry_short_delta_momentum"),
    ])

    return df


# ---------------------------------------------------------------------------
# 7. Footprint / Stacked Imbalance
# ---------------------------------------------------------------------------

def footprint_imbalance(
    df: pl.DataFrame,
    stacked_threshold: int = 3,
) -> pl.DataFrame:
    """Simulated footprint chart analysis from bar data.

    Estimates per-bar buy/sell dominance from bar structure and looks for
    consecutive bars with the same-direction imbalance (stacked imbalances).
    Stacked imbalances indicate strong directional conviction.

    Adds:
        stacked_buy_imbalance   — consecutive buy-dominant bar count
        stacked_sell_imbalance  — consecutive sell-dominant bar count
        signal_stacked_buy      — count >= threshold (strong buy zone)
        signal_stacked_sell     — count >= threshold (strong sell zone)
    """
    # Determine per-bar dominance: buy-dominant if close > open
    df = df.with_columns(
        (pl.col("close") > pl.col("open")).alias("_buy_dominant"),
    )

    # Build consecutive counts via cumsum trick:
    # Increment counter when same direction continues, reset on flip.
    buy_dom = df["_buy_dominant"].to_list()
    n = len(buy_dom)
    stacked_buy: list[int] = [0] * n
    stacked_sell: list[int] = [0] * n

    for i in range(n):
        if buy_dom[i] is None:
            continue
        if buy_dom[i]:
            stacked_buy[i] = (stacked_buy[i - 1] + 1) if i > 0 else 1
            stacked_sell[i] = 0
        else:
            stacked_sell[i] = (stacked_sell[i - 1] + 1) if i > 0 else 1
            stacked_buy[i] = 0

    df = df.with_columns([
        pl.Series(name="stacked_buy_imbalance", values=stacked_buy),
        pl.Series(name="stacked_sell_imbalance", values=stacked_sell),
    ])

    df = df.with_columns([
        (pl.col("stacked_buy_imbalance") >= stacked_threshold)
            .alias("signal_stacked_buy"),
        (pl.col("stacked_sell_imbalance") >= stacked_threshold)
            .alias("signal_stacked_sell"),
    ])

    df = df.drop("_buy_dominant")
    return df


# ---------------------------------------------------------------------------
# 8. Trapped Traders
# ---------------------------------------------------------------------------

def trapped_traders(
    df: pl.DataFrame,
    lookback: int = 5,
    retrace_pct: float = 0.5,
) -> pl.DataFrame:
    """Detect trapped trader scenarios — false breakouts that reverse.

    Identifies breakouts above rolling resistance (or below rolling support)
    that quickly retrace back inside the range, trapping breakout traders
    on the wrong side.  These are high-probability mean-reversion entries.

    Adds:
        signal_trapped_longs  — false breakout above resistance, reversal down
        signal_trapped_shorts — false breakout below support, reversal up
    """
    df = df.with_columns([
        pl.col("high").shift(1).rolling_max(window_size=lookback)
            .alias("_resistance"),
        pl.col("low").shift(1).rolling_min(window_size=lookback)
            .alias("_support"),
    ])

    # Breakout above resistance on the previous bar, then close back below
    # Current bar's high exceeded resistance but close retraced
    resistance_range = pl.col("high") - pl.col("_resistance")
    bar_range = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)

    # Trapped longs: bar pierced above resistance but closed back down
    # (retrace_pct of the breakout move given back)
    broke_above = pl.col("high") > pl.col("_resistance")
    closed_back_below = pl.col("close") < (
        pl.col("_resistance") + (1.0 - retrace_pct) * resistance_range
    )

    # Trapped shorts: bar pierced below support but closed back up
    support_range = pl.col("_support") - pl.col("low")
    broke_below = pl.col("low") < pl.col("_support")
    closed_back_above = pl.col("close") > (
        pl.col("_support") - (1.0 - retrace_pct) * support_range
    )

    df = df.with_columns([
        (broke_above & closed_back_below).alias("signal_trapped_longs"),
        (broke_below & closed_back_above).alias("signal_trapped_shorts"),
    ])

    df = df.drop(["_resistance", "_support"])
    return df
