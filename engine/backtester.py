"""Vectorized backtesting engine with prop firm rule enforcement."""

import uuid
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

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

ET = ZoneInfo("US/Eastern")


def _get_trailing_config(strategy) -> tuple[float, float] | None:
    """Extract trailing stop config from a strategy if it has one."""
    if hasattr(strategy, 'exit_rules'):
        rules = strategy.exit_rules
        if hasattr(rules, 'trailing_stop') and rules.trailing_stop:
            return (rules.trailing_activation, rules.trailing_distance)
    return None


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
    """Model execution slippage.

    Fix #35: Supports variable slippage based on volume and time of day.
    """

    def __init__(self, fixed_ticks: int = 2):
        self.fixed_ticks = fixed_ticks

    def apply(
        self, price: float, direction: str, contract_spec: ContractSpec,
        volume: float | None = None, timestamp: datetime | None = None,
    ) -> float:
        """Apply slippage to fill price. Slippage always works against us."""
        ticks = self.fixed_ticks

        # Variable slippage adjustments
        if volume is not None and volume < 100:
            ticks += 2  # Low liquidity penalty
        if timestamp is not None:
            try:
                et = timestamp
                if et.tzinfo is None:
                    et = et.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
                else:
                    et = et.astimezone(ET)
                h = et.hour
                if h < 9 or h >= 16:
                    ticks += 1  # Off-hours penalty
            except Exception:
                pass

        slip = ticks * contract_spec.tick_size
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

        # Convert to rows for sequential processing
        rows = df.to_dicts()
        current_date = ""

        # Pending entry: signals fire on bar N, fill on bar N+1 open (fix #3)
        pending_entry: tuple[str, int, Strategy] | None = None

        # Extract trailing stop config from strategy if available
        trailing_cfg = _get_trailing_config(strategy)

        prev_close: float | None = None

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

            # Execute pending entry from previous bar's signal (fix #3)
            if pending_entry and not account.open_position:
                direction, contracts, strat_ref = pending_entry
                allowed, reason = self.risk_manager.pre_trade_check(
                    ts, direction, contracts, account
                )
                if allowed:
                    fill_price = self.slippage.apply(
                        row["open"], direction, self.contract_spec
                    )
                    stop = strat_ref.get_stop_loss(fill_price, direction)
                    tp = strat_ref.get_take_profit(fill_price, direction)

                    pos = Position(
                        symbol=self.contract_spec.symbol,
                        direction=direction,
                        entry_time=ts,
                        entry_price=fill_price,
                        contracts=contracts,
                        stop_loss=stop,
                        take_profit=tp,
                    )
                    # Store trailing stop config on position (fix #1)
                    if trailing_cfg:
                        pos._trailing_active = False
                        pos._trailing_activation = trailing_cfg[0]
                        pos._trailing_distance = trailing_cfg[1]
                    account.open_position = pos
                pending_entry = None

            # Check for overnight gap (fix #32) — if open gaps beyond SL, fill at open
            if account.open_position and prev_close is not None:
                trade = self._check_gap(account, row, ts, prev_close)
                if trade:
                    trades.append(trade)
                    self.risk_manager.post_trade_update(trade, account)

            # Update trailing stop before checking stops (fix #1)
            if account.open_position and trailing_cfg:
                self._update_trailing_stop(account.open_position, row)

            # Check stop loss / take profit on open position (fix #2: order)
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

            # Process entry signals — queue for next bar (fix #3)
            if not account.open_position and pending_entry is None:
                direction = None
                if row.get("entry_long", False):
                    direction = "long"
                elif row.get("entry_short", False):
                    direction = "short"

                if direction:
                    contracts = strategy.get_position_size(
                        account, self.contract_spec, prop_rules
                    )
                    pending_entry = (direction, contracts, strategy)

            # Record equity (with unrealized P&L for intrabar DD)
            equity = account.current_balance
            if account.open_position:
                equity += account.open_position.unrealized_pnl(
                    row["close"], self.contract_spec
                )
            equity_curve.append((ts, equity))

            prev_close = row["close"]

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

        # Determine session segment (fix #8: convert UTC to ET)
        session = "core"
        if hasattr(pos.entry_time, "hour"):
            et_time = pos.entry_time
            # Convert to ET if naive (assumed UTC) or already tz-aware
            if et_time.tzinfo is None:
                et_time = et_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
            else:
                et_time = et_time.astimezone(ET)
            mins = et_time.hour * 60 + et_time.minute
            if mins < 570:  # before 9:30 ET
                session = "pre_market"
            elif mins >= 960:  # after 16:00 ET
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
        """Check if stop loss or take profit was hit on current bar.

        Fix #2: When both SL and TP are hit on the same bar, use
        open-distance heuristic to determine which was hit first.
        """
        pos = account.open_position
        if pos is None:
            return None

        high = row["high"]
        low = row["low"]
        bar_open = row["open"]

        # Determine if SL/TP were hit
        sl_hit = False
        tp_hit = False

        if pos.stop_loss is not None:
            sl_hit = (
                (pos.direction == "long" and low <= pos.stop_loss)
                or (pos.direction == "short" and high >= pos.stop_loss)
            )

        if pos.take_profit is not None:
            tp_hit = (
                (pos.direction == "long" and high >= pos.take_profit)
                or (pos.direction == "short" and low <= pos.take_profit)
            )

        if sl_hit and tp_hit:
            # Both hit on same bar — use open-distance heuristic
            dist_to_sl = abs(bar_open - pos.stop_loss)
            dist_to_tp = abs(bar_open - pos.take_profit)
            if dist_to_sl <= dist_to_tp:
                exit_reason = "trailing_stop" if getattr(pos, '_trailing_active', False) else "stop_loss"
                return self._close_position(account, ts, pos.stop_loss, exit_reason)
            else:
                return self._close_position(account, ts, pos.take_profit, "take_profit")

        if sl_hit:
            exit_reason = "trailing_stop" if getattr(pos, '_trailing_active', False) else "stop_loss"
            return self._close_position(account, ts, pos.stop_loss, exit_reason)

        if tp_hit:
            return self._close_position(account, ts, pos.take_profit, "take_profit")

        return None

    def _check_gap(
        self, account: AccountState, row: dict, ts: datetime, prev_close: float,
    ) -> Trade | None:
        """Fix #32: Handle overnight gaps that blow past stop loss."""
        pos = account.open_position
        if pos is None or pos.stop_loss is None:
            return None

        bar_open = row["open"]

        # Check if the open gapped beyond the stop loss
        if pos.direction == "long" and bar_open < pos.stop_loss:
            # Gapped below stop — fill at open (worse than SL)
            return self._close_position(account, ts, bar_open, "stop_loss_gap")
        elif pos.direction == "short" and bar_open > pos.stop_loss:
            # Gapped above stop — fill at open (worse than SL)
            return self._close_position(account, ts, bar_open, "stop_loss_gap")

        return None

    @staticmethod
    def _update_trailing_stop(pos: Position, row: dict) -> None:
        """Fix #1: Update trailing stop level if activated."""
        if not hasattr(pos, '_trailing_activation'):
            return

        high = row["high"]
        low = row["low"]

        # Check if trailing should activate
        if not pos._trailing_active:
            if pos.direction == "long":
                unrealized = high - pos.entry_price
            else:
                unrealized = pos.entry_price - low
            if unrealized >= pos._trailing_activation:
                pos._trailing_active = True
                # Set initial trailing stop
                if pos.direction == "long":
                    new_stop = high - pos._trailing_distance
                else:
                    new_stop = low + pos._trailing_distance
                if pos.stop_loss is not None:
                    if pos.direction == "long":
                        pos.stop_loss = max(pos.stop_loss, new_stop)
                    else:
                        pos.stop_loss = min(pos.stop_loss, new_stop)
                else:
                    pos.stop_loss = new_stop
            return

        # Trailing is active — ratchet the stop
        if pos.direction == "long":
            new_stop = high - pos._trailing_distance
            if pos.stop_loss is None or new_stop > pos.stop_loss:
                pos.stop_loss = new_stop
        else:
            new_stop = low + pos._trailing_distance
            if pos.stop_loss is None or new_stop < pos.stop_loss:
                pos.stop_loss = new_stop
