"""Data fetcher — download NQ/MNQ futures data from free sources."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)
ET = ZoneInfo("US/Eastern")


class YFinanceFetcher:
    """
    Fetch NQ futures data from Yahoo Finance.

    Limitations:
    - 1m bars: max ~60 days of history (7 days per request)
    - 5m bars: max ~60 days
    - 1h bars: max ~730 days
    - 1d bars: max ~20 years
    - MNQ not available — use NQ=F and scale for MNQ tick values
    """

    SYMBOL_MAP = {
        "NQ": "NQ=F",
        "MNQ": "NQ=F",  # MNQ tracks NQ; same price, different tick value
    }

    def __init__(self, output_dir: str | Path = "data/processed"):
        self.output_dir = Path(output_dir)

    def fetch(
        self,
        symbol: str = "MNQ",
        interval: str = "1m",
        days_back: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: "NQ" or "MNQ" (both use NQ=F under the hood)
            interval: "1m", "5m", "15m", "1h", "1d"
            days_back: number of days to look back (default depends on interval)
            start_date: explicit start date "YYYY-MM-DD" (overrides days_back)
            end_date: explicit end date "YYYY-MM-DD"
        """
        yf_symbol = self.SYMBOL_MAP.get(symbol, f"{symbol}=F")
        ticker = yf.Ticker(yf_symbol)

        if interval == "1m":
            # Yahoo limits 1m to 7 days per request, max ~30 days back
            df = self._fetch_1m_chunked(ticker, days_back or 59)
        else:
            if start_date and end_date:
                pdf = ticker.history(start=start_date, end=end_date, interval=interval)
            elif days_back:
                period_map = {
                    "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max",
                }
                pdf = ticker.history(period=f"{days_back}d", interval=interval)
            else:
                period_map = {
                    "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max",
                }
                pdf = ticker.history(period=period_map.get(interval, "60d"), interval=interval)

            df = self._pandas_to_polars(pdf)

        if df.is_empty():
            raise ValueError(f"No data returned for {yf_symbol} interval={interval}")

        logger.info(
            f"Fetched {len(df)} bars for {symbol} ({interval}) "
            f"from {df['timestamp'][0]} to {df['timestamp'][-1]}"
        )
        return df

    def _fetch_1m_chunked(self, ticker, total_days: int) -> pl.DataFrame:
        """Fetch 1-minute data in 7-day chunks (Yahoo API limit)."""
        chunks = []
        end = datetime.now()
        days_fetched = 0

        while days_fetched < total_days:
            chunk_days = min(7, total_days - days_fetched)
            start = end - timedelta(days=chunk_days)

            try:
                pdf = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1m",
                )
                if not pdf.empty:
                    chunk_df = self._pandas_to_polars(pdf)
                    chunks.append(chunk_df)
                    logger.info(f"  Chunk: {len(chunk_df)} bars ({start.date()} to {end.date()})")
            except Exception as e:
                logger.warning(f"  Failed chunk {start.date()}-{end.date()}: {e}")

            end = start
            days_fetched += chunk_days

        if not chunks:
            return pl.DataFrame()

        combined = pl.concat(chunks)
        return combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

    def _pandas_to_polars(self, pdf) -> pl.DataFrame:
        """Convert yfinance pandas DataFrame to our standard Polars format."""
        if pdf.empty:
            return pl.DataFrame()

        pdf = pdf.reset_index()

        # yfinance returns "Datetime" for intraday, "Date" for daily
        ts_col = "Datetime" if "Datetime" in pdf.columns else "Date"

        df = pl.DataFrame({
            "timestamp": pdf[ts_col].values,
            "open": pdf["Open"].values,
            "high": pdf["High"].values,
            "low": pdf["Low"].values,
            "close": pdf["Close"].values,
            "volume": pdf["Volume"].values,
        })

        # Ensure timestamp is proper datetime, strip timezone for consistency
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )

        # Add tick_count estimate (not available from Yahoo, estimate from volume)
        df = df.with_columns(
            (pl.col("volume") / 10).cast(pl.Int64).alias("tick_count")
        )

        # Filter to trading hours only (8:00-17:00 ET)
        df = df.with_columns(
            pl.col("timestamp").dt.hour().alias("_hour")
        )
        df = df.filter(
            (pl.col("_hour") >= 8) & (pl.col("_hour") < 17)
        ).drop("_hour")

        return df

    def fetch_and_save(
        self,
        symbol: str = "MNQ",
        interval: str = "1m",
        days_back: int | None = None,
    ) -> Path:
        """Fetch data and save to parquet."""
        df = self.fetch(symbol=symbol, interval=interval, days_back=days_back)

        # Map interval to timeframe dir name
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
        tf = tf_map.get(interval, interval)

        out_dir = self.output_dir / symbol / tf
        out_dir.mkdir(parents=True, exist_ok=True)

        # Partition by year-month
        df = df.with_columns(
            pl.col("timestamp").dt.strftime("%Y-%m").alias("_partition")
        )
        for partition, group in df.group_by("_partition"):
            month_str = partition[0]
            path = out_dir / f"{month_str}.parquet"
            group.drop("_partition").write_parquet(path)

        total_path = out_dir / "all.parquet"
        df.drop("_partition").write_parquet(total_path)

        logger.info(f"Saved {len(df)} bars to {out_dir}")
        return total_path

    def fetch_multi_timeframe(
        self,
        symbol: str = "MNQ",
        days_back_1m: int = 59,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch both 1m and 5m data. 5m is resampled from 1m for consistency.
        Returns dict: {"1m": df, "5m": df}
        """
        from engine.data_pipeline import BarBuilder

        df_1m = self.fetch(symbol=symbol, interval="1m", days_back=days_back_1m)
        df_5m = BarBuilder.resample(df_1m, freq="5m")

        logger.info(f"Multi-timeframe: {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
        return {"1m": df_1m, "5m": df_5m}
