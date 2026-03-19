"""Data pipeline: ingestion, cleaning, contract stitching, bar construction."""

import re
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")


# ── MotiveWave Ingestion ─────────────────────────────────────────────

class MotiveWaveIngestor:
    """Parse MotiveWave CSV exports into normalized Polars DataFrames."""

    # Common date/time formats MotiveWave may use
    DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y%m%d %H:%M:%S",
    ]

    def __init__(self, raw_dir: Path, output_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_format(self, filepath: Path) -> dict:
        """Auto-detect CSV format by inspecting first lines."""
        with open(filepath) as f:
            lines = [f.readline() for _ in range(5)]

        # Detect delimiter
        header = lines[0]
        if "\t" in header:
            delimiter = "\t"
        elif ";" in header:
            delimiter = ";"
        else:
            delimiter = ","

        # Detect columns
        cols = [c.strip().strip('"').lower() for c in header.split(delimiter)]

        # Detect date format from first data line
        date_format = None
        if len(lines) > 1:
            first_val = lines[1].split(delimiter)[0].strip().strip('"')
            for fmt in self.DATE_FORMATS:
                try:
                    datetime.strptime(first_val, fmt)
                    date_format = fmt
                    break
                except ValueError:
                    continue

        return {
            "delimiter": delimiter,
            "columns": cols,
            "date_format": date_format,
            "has_header": True,
        }

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize column names to standard: timestamp, open, high, low, close, volume, tick_count."""
        col_map = {}
        lower_cols = {c.lower().strip(): c for c in df.columns}

        # Map common variants
        mappings = {
            "timestamp": ["timestamp", "date", "datetime", "date/time", "time"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c", "last"],
            "volume": ["volume", "vol", "v"],
            "tick_count": ["tick_count", "ticks", "tick count", "tickcount", "count"],
        }

        for target, variants in mappings.items():
            for v in variants:
                if v in lower_cols:
                    col_map[lower_cols[v]] = target
                    break

        df = df.rename(col_map)

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {df.columns}")

        # Add tick_count if missing
        if "tick_count" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("tick_count"))

        return df.select(["timestamp", "open", "high", "low", "close", "volume", "tick_count"])

    def ingest_file(
        self,
        filepath: Path,
        symbol: str,
        timezone: str = "US/Eastern",
    ) -> pl.DataFrame:
        """Parse a single MotiveWave CSV export."""
        fmt = self.detect_format(filepath)

        df = pl.read_csv(
            filepath,
            separator=fmt["delimiter"],
            has_header=fmt["has_header"],
            try_parse_dates=True,
            ignore_errors=True,
        )

        df = self._normalize_columns(df)

        # Parse timestamp if it's a string
        if df["timestamp"].dtype == pl.Utf8:
            if fmt["date_format"]:
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, fmt["date_format"]).alias("timestamp")
                )
            else:
                df = df.with_columns(
                    pl.col("timestamp").str.to_datetime().alias("timestamp")
                )

        # Ensure numeric types
        for col in ["open", "high", "low", "close"]:
            if df[col].dtype != pl.Float64:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        for col in ["volume", "tick_count"]:
            if df[col].dtype != pl.Int64:
                df = df.with_columns(pl.col(col).cast(pl.Int64))

        # Add symbol column
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # Sort by timestamp
        df = df.sort("timestamp")

        return df

    def ingest_directory(self, symbol: str, pattern: str = "*.csv") -> pl.DataFrame:
        """Batch ingest all CSV files matching pattern."""
        files = sorted(self.raw_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.raw_dir}")

        dfs = [self.ingest_file(f, symbol) for f in files]
        combined = pl.concat(dfs)
        return combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

    def save_parquet(self, df: pl.DataFrame, symbol: str, timeframe: str) -> Path:
        """Save DataFrame to partitioned Parquet file."""
        out_dir = self.output_dir / symbol / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)

        # Partition by year-month
        df = df.with_columns(
            pl.col("timestamp").dt.strftime("%Y-%m").alias("_partition")
        )

        for partition, group in df.group_by("_partition"):
            month_str = partition[0]
            path = out_dir / f"{month_str}.parquet"
            group.drop("_partition").write_parquet(path)

        return out_dir


# ── Data Cleaning ────────────────────────────────────────────────────

class DataCleaner:
    """Clean and validate OHLCV data."""

    def __init__(self, session_config: dict | None = None):
        self.session_config = session_config

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Full cleaning pipeline."""
        df = self.remove_duplicates(df)
        df = self.validate_ohlcv(df)
        df = self.remove_bad_ticks(df)
        if self.session_config:
            df = self.tag_sessions(df)
        return df

    def remove_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.unique(subset=["timestamp"], keep="first").sort("timestamp")

    def validate_ohlcv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fix OHLCV consistency: H >= max(O,C), L <= min(O,C), Volume >= 0."""
        df = df.filter(
            pl.col("open").is_not_null()
            & pl.col("high").is_not_null()
            & pl.col("low").is_not_null()
            & pl.col("close").is_not_null()
            & pl.col("volume").is_not_null()
        )

        # Fix high/low if they violate OHLC relationship
        df = df.with_columns([
            pl.max_horizontal("open", "high", "close").alias("high"),
            pl.min_horizontal("open", "low", "close").alias("low"),
            pl.max_horizontal("volume", pl.lit(0)).alias("volume"),
        ])

        return df

    def remove_bad_ticks(
        self, df: pl.DataFrame, max_pct_change: float = 2.0
    ) -> pl.DataFrame:
        """Remove ticks where price changes more than max_pct_change% from previous."""
        df = df.with_columns(
            (pl.col("close").pct_change().abs() * 100).alias("_pct_change")
        )
        # Keep first row (null pct_change) and rows within threshold
        df = df.filter(
            pl.col("_pct_change").is_null() | (pl.col("_pct_change") <= max_pct_change)
        )
        return df.drop("_pct_change")

    def detect_gaps(
        self, df: pl.DataFrame, max_gap_seconds: int = 120
    ) -> pl.DataFrame:
        """Return DataFrame of detected gaps in the data."""
        gaps = df.with_columns(
            (pl.col("timestamp").diff().dt.total_seconds()).alias("gap_seconds")
        ).filter(
            pl.col("gap_seconds") > max_gap_seconds
        ).select([
            "timestamp",
            "gap_seconds",
            pl.col("close").alias("price_at_gap"),
        ])
        return gaps

    def tag_sessions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session_segment column based on time of day (ET)."""
        sessions = self.session_config["sessions"]

        def _parse_time(t: str) -> time:
            parts = t.split(":")
            return time(int(parts[0]), int(parts[1]))

        pre_start = _parse_time(sessions["pre_market"]["start"])
        pre_end = _parse_time(sessions["pre_market"]["end"])
        core_start = _parse_time(sessions["core"]["start"])
        core_end = _parse_time(sessions["core"]["end"])
        post_start = _parse_time(sessions["post_close"]["start"])
        post_end = _parse_time(sessions["post_close"]["end"])

        # Extract hour and minute for comparison (cast to i32 to avoid i8 overflow)
        df = df.with_columns([
            pl.col("timestamp").dt.hour().cast(pl.Int32).alias("_hour"),
            pl.col("timestamp").dt.minute().cast(pl.Int32).alias("_minute"),
        ])

        df = df.with_columns(
            (pl.col("_hour") * 60 + pl.col("_minute")).alias("_minutes")
        )

        pre_start_min = pre_start.hour * 60 + pre_start.minute
        pre_end_min = pre_end.hour * 60 + pre_end.minute
        core_start_min = core_start.hour * 60 + core_start.minute
        core_end_min = core_end.hour * 60 + core_end.minute
        post_start_min = post_start.hour * 60 + post_start.minute
        post_end_min = post_end.hour * 60 + post_end.minute

        df = df.with_columns(
            pl.when(
                (pl.col("_minutes") >= pre_start_min) & (pl.col("_minutes") < pre_end_min)
            ).then(pl.lit("pre_market"))
            .when(
                (pl.col("_minutes") >= core_start_min) & (pl.col("_minutes") < core_end_min)
            ).then(pl.lit("core"))
            .when(
                (pl.col("_minutes") >= post_start_min) & (pl.col("_minutes") < post_end_min)
            ).then(pl.lit("post_close"))
            .otherwise(pl.lit("outside"))
            .alias("session_segment")
        )

        return df.drop(["_hour", "_minute", "_minutes"])


# ── Continuous Contract Stitching ────────────────────────────────────

class ContractStitcher:
    """Stitch multiple contract months into a continuous series."""

    def __init__(self, method: str = "back_adjusted"):
        if method not in ("back_adjusted", "ratio_adjusted"):
            raise ValueError(f"Unknown stitching method: {method}")
        self.method = method

    def detect_rollover_dates(
        self,
        contracts: list[tuple[pl.DataFrame, str]],
        volume_lookback_days: int = 5,
    ) -> list[tuple[str, str, datetime]]:
        """
        Detect rollovers by volume crossover between consecutive contracts.
        contracts: list of (dataframe, contract_code) e.g. ("MNQ_H26", "MNQ_M26")
        Returns: list of (from_contract, to_contract, rollover_date)
        """
        rollovers = []

        for i in range(len(contracts) - 1):
            current_df, current_code = contracts[i]
            next_df, next_code = contracts[i + 1]

            # Find overlapping dates
            current_daily = current_df.group_by(
                pl.col("timestamp").dt.date().alias("date")
            ).agg(pl.col("volume").sum().alias("current_vol"))

            next_daily = next_df.group_by(
                pl.col("timestamp").dt.date().alias("date")
            ).agg(pl.col("volume").sum().alias("next_vol"))

            overlap = current_daily.join(next_daily, on="date", how="inner")

            if overlap.is_empty():
                continue

            # Find first date where next contract volume exceeds current
            crossover = overlap.filter(
                pl.col("next_vol") > pl.col("current_vol")
            ).sort("date")

            if not crossover.is_empty():
                roll_date = crossover["date"][0]
                rollovers.append((current_code, next_code, datetime.combine(roll_date, time())))

        return rollovers

    def stitch(
        self,
        contracts: list[tuple[pl.DataFrame, str, datetime, datetime]],
    ) -> pl.DataFrame:
        """
        Stitch contracts into continuous series.
        contracts: list of (dataframe, contract_code, start_date, end_date)
                   ordered chronologically. Each df is used from start_date to end_date.
        """
        if not contracts:
            raise ValueError("No contracts to stitch")

        if len(contracts) == 1:
            df, _, start, end = contracts[0]
            return df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )

        # Work backwards from most recent contract (no adjustment needed)
        segments = []
        adjustment = 0.0

        for i in range(len(contracts) - 1, -1, -1):
            df, code, start, end = contracts[i]
            segment = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )

            if adjustment != 0.0:
                price_cols = ["open", "high", "low", "close"]
                if self.method == "back_adjusted":
                    segment = segment.with_columns([
                        (pl.col(c) + adjustment).alias(c) for c in price_cols
                    ])
                else:  # ratio_adjusted
                    segment = segment.with_columns([
                        (pl.col(c) * (1 + adjustment)).alias(c) for c in price_cols
                    ])

            segments.insert(0, segment)

            # Calculate adjustment for next older contract
            if i > 0:
                prev_df, _, _, _ = contracts[i - 1]
                # Find the gap at the rollover point
                roll_point = start
                current_at_roll = df.filter(
                    pl.col("timestamp").dt.date() == roll_point.date()
                )
                prev_at_roll = prev_df.filter(
                    pl.col("timestamp").dt.date() == roll_point.date()
                )

                if not current_at_roll.is_empty() and not prev_at_roll.is_empty():
                    current_close = current_at_roll["close"][0]
                    prev_close = prev_at_roll["close"][-1]
                    if self.method == "back_adjusted":
                        adjustment += current_close - prev_close
                    else:
                        if prev_close != 0:
                            adjustment += (current_close - prev_close) / prev_close

        return pl.concat(segments).sort("timestamp")


# ── Bar Builder ──────────────────────────────────────────────────────

class BarBuilder:
    """Construct time bars from tick or lower-timeframe data."""

    @staticmethod
    def resample(df: pl.DataFrame, freq: str = "5m") -> pl.DataFrame:
        """
        Resample bars to a higher timeframe.
        freq: "1m", "5m", "15m", "30m", "1h", etc.
        """
        return df.group_by_dynamic(
            "timestamp", every=freq
        ).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("tick_count").sum(),
        ]).sort("timestamp")

    @staticmethod
    def tick_to_bars(ticks: pl.DataFrame, freq: str = "1m") -> pl.DataFrame:
        """Aggregate raw tick data into time bars."""
        bars = ticks.group_by_dynamic(
            "timestamp", every=freq
        ).agg([
            pl.col("close").first().alias("open"),
            pl.col("close").max().alias("high"),
            pl.col("close").min().alias("low"),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.lit(1).count().alias("tick_count"),
        ]).sort("timestamp")

        return bars


# ── Parquet Loader ───────────────────────────────────────────────────

def load_parquet(data_dir: Path, symbol: str, timeframe: str) -> pl.DataFrame:
    """Load all Parquet files for a symbol/timeframe into a single DataFrame."""
    path = Path(data_dir) / symbol / timeframe
    if not path.exists():
        raise FileNotFoundError(f"No data at {path}")

    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")

    dfs = [pl.read_parquet(f) for f in files]
    return pl.concat(dfs).sort("timestamp")
