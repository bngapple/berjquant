"""Adaptive infrastructure for HFT scalping — ATR stops, HTF trend, regime detection.
New file — does NOT modify any existing code."""

import polars as pl
import numpy as np


def atr_stops(df: pl.DataFrame, sl_multiple: float = 1.5, tp_multiple: float = 1.5,
              atr_period: int = 14) -> pl.DataFrame:
    """Add adaptive_sl and adaptive_tp columns based on ATR."""
    # Compute ATR if not present
    atr_col = f"atr_{atr_period}"
    if atr_col not in df.columns:
        high = pl.col("high"); low = pl.col("low"); close = pl.col("close")
        tr = pl.max_horizontal(
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        )
        df = df.with_columns(tr.alias("_tr"))
        df = df.with_columns(pl.col("_tr").rolling_mean(window_size=atr_period).alias(atr_col))
        df = df.drop("_tr")

    df = df.with_columns([
        (pl.col(atr_col) * sl_multiple).clip(4.0, 25.0).alias("adaptive_sl"),
        (pl.col(atr_col) * tp_multiple).clip(4.0, 25.0).alias("adaptive_tp"),
    ])
    return df


def add_htf_trend(df: pl.DataFrame, ema_period: int = 20) -> pl.DataFrame:
    """Add htf_trend column: +1 uptrend, -1 downtrend, 0 neutral on 15-min timeframe."""
    # Resample to 15-min by taking every 15th bar's close (approximate)
    close = df["close"].to_numpy()
    n = len(close)
    htf_ema = np.full(n, np.nan)

    # Compute rolling EMA on 15-bar sampled closes
    window = ema_period * 15  # 15-min bars × period
    if n > window:
        # Simple: use rolling mean of close over ema_period*15 bars as HTF EMA proxy
        kernel = np.ones(window) / window
        padded = np.pad(close, (window - 1, 0), mode='edge')
        htf_ema = np.convolve(padded, kernel, mode='valid')[:n]

    # Trend: +1 if price > ema + 0.15%, -1 if price < ema - 0.15%, else 0
    trend = np.zeros(n, dtype=np.int8)
    valid = ~np.isnan(htf_ema)
    threshold = htf_ema * 0.0015
    trend[valid & (close > htf_ema + threshold)] = 1
    trend[valid & (close < htf_ema - threshold)] = -1

    df = df.with_columns([
        pl.Series("htf_trend", trend),
        pl.Series("signal_htf_uptrend", trend > 0),
        pl.Series("signal_htf_downtrend", trend < 0),
    ])
    return df


def add_regime(df: pl.DataFrame, adx_period: int = 14,
               trend_threshold: int = 25, range_threshold: int = 20) -> pl.DataFrame:
    """Add regime column based on ADX: trending, ranging, or neutral."""
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    close = df["close"].to_numpy().astype(np.float64)
    n = len(close)

    # Compute +DM, -DM, TR
    dm_plus = np.zeros(n); dm_minus = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        dm_plus[i] = up if (up > down and up > 0) else 0
        dm_minus[i] = down if (down > up and down > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # Smoothed with EMA
    alpha = 1.0 / adx_period
    sm_tr = np.zeros(n); sm_dp = np.zeros(n); sm_dm = np.zeros(n)
    sm_tr[adx_period] = np.sum(tr[1:adx_period+1])
    sm_dp[adx_period] = np.sum(dm_plus[1:adx_period+1])
    sm_dm[adx_period] = np.sum(dm_minus[1:adx_period+1])
    for i in range(adx_period + 1, n):
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1] / adx_period + tr[i]
        sm_dp[i] = sm_dp[i-1] - sm_dp[i-1] / adx_period + dm_plus[i]
        sm_dm[i] = sm_dm[i-1] - sm_dm[i-1] / adx_period + dm_minus[i]

    di_plus = np.where(sm_tr > 0, 100 * sm_dp / sm_tr, 0)
    di_minus = np.where(sm_tr > 0, 100 * sm_dm / sm_tr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)

    adx = np.zeros(n)
    start = 2 * adx_period
    if start < n:
        adx[start] = np.mean(dx[adx_period:start+1])
        for i in range(start + 1, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period

    trending = adx > trend_threshold
    ranging = adx < range_threshold

    df = df.with_columns([
        pl.Series("adx", adx),
        pl.Series("signal_regime_trending", trending),
        pl.Series("signal_regime_ranging", ranging),
    ])
    return df


def add_volatility_regime(df: pl.DataFrame, atr_period: int = 14,
                          lookback: int = 100) -> pl.DataFrame:
    """Add vol_regime: high_vol, low_vol, normal based on ATR vs rolling mean."""
    atr_col = f"atr_{atr_period}"
    if atr_col not in df.columns:
        df = atr_stops(df, 1.0, 1.0, atr_period)  # Just to get ATR computed

    atr_vals = df[atr_col].to_numpy()
    n = len(atr_vals)
    mean_atr = np.full(n, np.nan)
    for i in range(lookback, n):
        mean_atr[i] = np.nanmean(atr_vals[max(0, i - lookback):i])

    high_vol = np.zeros(n, dtype=bool)
    low_vol = np.zeros(n, dtype=bool)
    valid = ~np.isnan(mean_atr) & ~np.isnan(atr_vals)
    high_vol[valid] = atr_vals[valid] > 1.3 * mean_atr[valid]
    low_vol[valid] = atr_vals[valid] < 0.7 * mean_atr[valid]

    df = df.with_columns([
        pl.Series("signal_high_vol", high_vol),
        pl.Series("signal_low_vol", low_vol),
    ])
    return df
