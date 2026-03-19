"""Vectorized backtesting engine with prop firm rule enforcement."""

import uuid
from datetime import datetime
from typing import Protocol

import numpy as np
import polars as pl

from engine.risk_manager import RiskManager
from engine.utils import (
    AccountState,
    BacktestConfig,
    BacktestResult,
    ContractSpec,
    Position,
    PropFirmRules,
    Trade,
)


# ── Strategy Interface ───────────────────────────────────────────────

class Strategy(Protocol):
    """Interface that all strategies must implement."""

    name: str

    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Add entry/exit signal columns to the primary timeframe DataFrame.
        Must add columns: entry_long, entry_short, exit_long, exit_short (boolean).
        Returns the primary timeframe DataFrame with signal columns added.
        """
        ...

    def get_stop_loss(self, entry_price: float, direction: str) -> float | None:
        """Return stop-loss price or None."""
        ...

    def get_take_profit(self, entry_price: float, direction: str) -> float | None:
        """Return take-profit price or None."""
        ...

    def get_position_size(
        self,
        account_state: AccountState,
        contract_spec: ContractSpec,
        prop_rules: PropFirmRules,
    ) -> int:
        """Return number of contracts to trade."""
        ...


# ── Slippage Model ───────────────────────────────────────────────────

class SlippageModel:
    """Model execution slippage."""

    def __init__(self, fixed_ticks: int = 2):
        self.fixed_ticks = fixed_ticks

    def apply(
        self, price: float, direction: str, contract_spec: ContractSpec
    ) -> float:
        """Apply slippage to fill price. Slippage always works against us."""
        slip = self.fixed_ticks * contract_spec.tick_size
        if direction == "long":
            return price + slip  # buy higher
        else:
            return price - slip  # sell lower


# ── Vectorized Backtester ────────────────────────────────────────────

class VectorizedBacktester:
    """
    Core backtesting engine.

    Workflow:
    1. Strategy computes all signals vectorized across full dataset
    2. Engine walks through bars chronologically
    3. At each entry signal, risk manager checks are applied
    4. P&L calculated with slippage and commissions
    """

    def __init__(
        self,
        data: dict[str, pl.DataFrame],  # {"1m": df, "5m": df}
        risk_manager: RiskManager,
        contract_spec: ContractSpec,
        config: BacktestConfig,
    ):
        self.data = data
        self.risk_manager = risk_manager
        self.contract_spec = contract_spec
        self.config = config
        self.slippage = SlippageModel(fixed_ticks=config.slippage_ticks)

    def run(self, strategy: Strategy) -> BacktestResult:
        """Execute a full backtest."""
        # Step 1: Compute signals across the full dataset
        df = strategy.compute_signals(self.data)

        # Validate signal columns exist
        required_cols = ["entry_long", "entry_short", "exit_long", "exit_short"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Strategy must produce '{col}' column")

        # Step 2: Initialize account
        prop_rules = self.risk_manager.prop_rules
        account = self.risk_manager.init_account(self.config.initial_capital)

        # Step 3: Walk through bars and execute trades
        trades: list[Trade] = []
        equity_curve: list[tuple[datetime, float]] = []

        # Convert to rows for sequential processing (signal points only)
        rows = df.to_dicts()
        current_date = ""

        for row in rows:
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            # New trading day reset
            date_str = ts.strftime("%Y-%m-%d")
            if date_str != current_date:
                current_date = date_str
                self.risk_manager.reset_daily(account, date_str)

            # Check if we need to flatten at EOD
            if account.open_position and self.risk_manager.should_flatten(ts):
                trade = self._close_position(
                    account, ts, row["close"], "eod_flatten"
                )
                trades.append(trade)
                self.risk_manager.post_trade_update(trade, account)

            # Check stop loss / take profit on open position
            if account.open_position:
                trade = self._check_stops(account, row, ts)
                if trade:
                    trades.append(trade)
                    self.risk_manager.post_trade_update(trade, account)

            # Process exit signals
            if account.open_position:
                pos = account.open_position
                should_exit = (
                    (pos.direction == "long" and row.get("exit_long", False))
                    or (pos.direction == "short" and row.get("exit_short", False))
                )
                if should_exit:
                    trade = self._close_position(account, ts, row["close"], "signal")
                    trades.append(trade)
                    self.risk_manager.post_trade_update(trade, account)

            # Process entry signals (only if no open position)
            if not account.open_position:
                direction = None
                if row.get("entry_long", False):
                    direction = "long"
                elif row.get("entry_short", False):
                    direction = "short"

                if direction:
                    contracts = strategy.get_position_size(
                        account, self.contract_spec, prop_rules
                    )

                    allowed, reason = self.risk_manager.pre_trade_check(
                        ts, direction, contracts, account
                    )

                    if allowed:
                        fill_price = self.slippage.apply(
                            row["close"], direction, self.contract_spec
                        )
                        stop = strategy.get_stop_loss(fill_price, direction)
                        tp = strategy.get_take_profit(fill_price, direction)

                        account.open_position = Position(
                            symbol=self.contract_spec.symbol,
                            direction=direction,
                            entry_time=ts,
                            entry_price=fill_price,
                            contracts=contracts,
                            stop_loss=stop,
                            take_profit=tp,
                        )

            # Record equity
            equity = account.current_balance
            if account.open_position:
                equity += account.open_position.unrealized_pnl(
                    row["close"], self.contract_spec
                )
            equity_curve.append((ts, equity))

        # Force close any remaining position
        if account.open_position and rows:
            last = rows[-1]
            ts = last["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            trade = self._close_position(account, ts, last["close"], "backtest_end")
            trades.append(trade)
            self.risk_manager.post_trade_update(trade, account)

        return BacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _close_position(
        self,
        account: AccountState,
        exit_time: datetime,
        exit_price_raw: float,
        exit_reason: str,
    ) -> Trade:
        """Close the current position and return the Trade."""
        pos = account.open_position
        if pos is None:
            raise RuntimeError("No open position to close")

        # Apply slippage on exit (opposite direction)
        exit_direction = "short" if pos.direction == "long" else "long"
        exit_price = self.slippage.apply(
            exit_price_raw, exit_direction, self.contract_spec
        )

        # Calculate P&L
        if pos.direction == "long":
            points = exit_price - pos.entry_price
        else:
            points = pos.entry_price - exit_price

        gross_pnl = points * self.contract_spec.point_value * pos.contracts
        commission = (
            self.risk_manager.prop_rules.total_cost_per_contract_rt * pos.contracts
        )
        # Slippage cost is already baked into entry/exit prices,
        # but we track it separately for reporting
        slippage_ticks = self.config.slippage_ticks * 2  # entry + exit
        slippage_cost = (
            slippage_ticks
            * self.contract_spec.tick_value
            * pos.contracts
        )
        net_pnl = gross_pnl - commission

        duration = int((exit_time - pos.entry_time).total_seconds())

        # Determine session segment
        session = "core"
        if hasattr(pos.entry_time, "hour"):
            h = pos.entry_time.hour
            m = pos.entry_time.minute
            mins = h * 60 + m
            if mins < 570:  # before 9:30
                session = "pre_market"
            elif mins >= 960:  # after 16:00
                session = "post_close"

        return Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=pos.symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            contracts=pos.contracts,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            duration_seconds=duration,
            session_segment=session,
            exit_reason=exit_reason,
        )

    def _check_stops(
        self, account: AccountState, row: dict, ts: datetime
    ) -> Trade | None:
        """Check if stop loss or take profit was hit on current bar."""
        pos = account.open_position
        if pos is None:
            return None

        high = row["high"]
        low = row["low"]

        # Check stop loss
        if pos.stop_loss is not None:
            hit = (
                (pos.direction == "long" and low <= pos.stop_loss)
                or (pos.direction == "short" and high >= pos.stop_loss)
            )
            if hit:
                return self._close_position(account, ts, pos.stop_loss, "stop_loss")

        # Check take profit
        if pos.take_profit is not None:
            hit = (
                (pos.direction == "long" and high >= pos.take_profit)
                or (pos.direction == "short" and low <= pos.take_profit)
            )
            if hit:
                return self._close_position(
                    account, ts, pos.take_profit, "take_profit"
                )

        return None
