"""Simple EMA crossover strategy for engine validation."""

import polars as pl

from engine.utils import AccountState, ContractSpec, PropFirmRules


class EMACrossoverStrategy:
    """
    Long when EMA(fast) crosses above EMA(slow).
    Short when EMA(fast) crosses below EMA(slow).
    Fixed stop-loss and take-profit in points.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        stop_loss_points: float = 4.0,
        take_profit_points: float = 8.0,
        contracts: int = 1,
        primary_timeframe: str = "1m",
    ):
        self.name = f"EMA_{fast_period}_{slow_period}"
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_points = stop_loss_points
        self.take_profit_points = take_profit_points
        self._contracts = contracts
        self.primary_timeframe = primary_timeframe

    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Compute EMA crossover signals on primary timeframe."""
        df = data[self.primary_timeframe].clone()

        # Calculate EMAs
        df = df.with_columns([
            pl.col("close").ewm_mean(span=self.fast_period).alias("ema_fast"),
            pl.col("close").ewm_mean(span=self.slow_period).alias("ema_slow"),
        ])

        # Crossover detection
        df = df.with_columns([
            pl.col("ema_fast").shift(1).alias("ema_fast_prev"),
            pl.col("ema_slow").shift(1).alias("ema_slow_prev"),
        ])

        df = df.with_columns([
            # Long entry: fast crosses above slow
            (
                (pl.col("ema_fast") > pl.col("ema_slow"))
                & (pl.col("ema_fast_prev") <= pl.col("ema_slow_prev"))
            ).alias("entry_long"),
            # Short entry: fast crosses below slow
            (
                (pl.col("ema_fast") < pl.col("ema_slow"))
                & (pl.col("ema_fast_prev") >= pl.col("ema_slow_prev"))
            ).alias("entry_short"),
            # Exit signals: opposite crossover
            (
                (pl.col("ema_fast") < pl.col("ema_slow"))
                & (pl.col("ema_fast_prev") >= pl.col("ema_slow_prev"))
            ).alias("exit_long"),
            (
                (pl.col("ema_fast") > pl.col("ema_slow"))
                & (pl.col("ema_fast_prev") <= pl.col("ema_slow_prev"))
            ).alias("exit_short"),
        ])

        # Fill nulls from shift/ewm
        for col in ["entry_long", "entry_short", "exit_long", "exit_short"]:
            df = df.with_columns(pl.col(col).fill_null(False))

        return df.drop(["ema_fast_prev", "ema_slow_prev"])

    def get_stop_loss(self, entry_price: float, direction: str) -> float | None:
        if self.stop_loss_points is None:
            return None
        if direction == "long":
            return entry_price - self.stop_loss_points
        else:
            return entry_price + self.stop_loss_points

    def get_take_profit(self, entry_price: float, direction: str) -> float | None:
        if self.take_profit_points is None:
            return None
        if direction == "long":
            return entry_price + self.take_profit_points
        else:
            return entry_price - self.take_profit_points

    def get_position_size(
        self,
        account_state: AccountState,
        contract_spec: ContractSpec,
        prop_rules: PropFirmRules,
    ) -> int:
        return self._contracts
