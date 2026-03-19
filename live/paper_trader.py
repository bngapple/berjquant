"""
Paper Trading Simulator — validates strategies before real capital deployment.

Takes signals from the signal engine, simulates fills with realistic slippage,
manages positions, tracks P&L, and enforces prop firm rules in real-time.
This is the final gate before live trading.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from engine.utils import (
    AccountState,
    ContractSpec,
    Position,
    PropFirmRules,
    Trade,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal dataclass — the contract between the signal engine and paper trader
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A trade signal produced by the signal engine."""
    timestamp: datetime
    strategy_name: str
    direction: str          # "long" | "short"
    signal_type: str        # "entry" | "exit" | "reversal"
    price: float
    stop_loss: float | None
    take_profit: float | None
    contracts: int
    confidence: float       # 0.0 - 1.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paper account — extended tracking wrapper around AccountState
# ---------------------------------------------------------------------------

@dataclass
class PaperAccount:
    """Extended account tracking for paper trading."""
    account_state: AccountState
    trade_history: list = field(default_factory=list)
    daily_summaries: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    signals_received: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paper Trader
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Paper trading simulator with full prop firm compliance.

    Receives signals, simulates fills with slippage, manages positions,
    tracks P&L, and enforces prop firm rules in real-time.

    Usage:
        trader = PaperTrader(
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            prop_rules=topstep_rules,
            initial_balance=50000.0,
            slippage_ticks=2,
        )

        # Connect to signal engine
        signal_engine.on_signal(trader.on_signal)

        # Or manually feed signals
        trader.on_signal(signal)

        # Update with each new price bar
        trader.on_price_update(timestamp, price)

        # Check status
        trader.print_status()
        trader.print_daily_summary()
    """

    def __init__(
        self,
        risk_manager,
        contract_spec: ContractSpec,
        prop_rules: PropFirmRules,
        initial_balance: float = 50000.0,
        slippage_ticks: int = 2,
        log_dir: str | Path = "logs/paper_trading",
    ):
        self.risk_manager = risk_manager
        self.contract_spec = contract_spec
        self.prop_rules = prop_rules
        self.slippage_ticks = slippage_ticks
        self.initial_balance = initial_balance
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize account
        self.account = PaperAccount(
            account_state=risk_manager.init_account(initial_balance)
        )

        # Track the last price we saw (for equity curve updates between signals)
        self._last_price: float | None = None
        self._last_timestamp: datetime | None = None

        # Dedup: reject identical signals within this window
        self._recent_signal_keys: dict[str, datetime] = {}
        self._dedup_window = timedelta(seconds=5)

        # Callbacks
        self._trade_handlers: list[Callable] = []
        self._alert_handlers: list[Callable] = []

    # ------------------------------------------------------------------
    # Public: Signal ingestion
    # ------------------------------------------------------------------

    def on_signal(self, signal: Signal) -> None:
        """
        Process a trade signal.

        Flow:
        1. Validate and dedup the signal.
        2. If we have an open position:
           a. Exit signal or reversal -> close first.
           b. Same-direction entry -> reject (already positioned).
        3. For entry: run risk manager pre_trade_check.
        4. Simulate fill with slippage.
        5. Open position and update account state.
        6. Log everything.
        """
        self.account.signals_received += 1
        state = self.account.account_state

        # --- Guard: account killed ---
        if state.is_killed:
            self._reject_signal(signal, "kill_switch_active")
            return

        # --- Guard: dedup identical signals ---
        sig_key = f"{signal.strategy_name}:{signal.direction}:{signal.signal_type}"
        now = signal.timestamp
        if sig_key in self._recent_signal_keys:
            last_seen = self._recent_signal_keys[sig_key]
            if (now - last_seen) < self._dedup_window:
                self._reject_signal(signal, "duplicate_signal")
                return
        self._recent_signal_keys[sig_key] = now

        # --- Prune old dedup keys ---
        cutoff = now - self._dedup_window * 10
        self._recent_signal_keys = {
            k: v for k, v in self._recent_signal_keys.items() if v > cutoff
        }

        # --- Handle based on signal type ---
        if signal.signal_type == "exit":
            self._handle_exit_signal(signal)
        elif signal.signal_type == "reversal":
            self._handle_reversal_signal(signal)
        elif signal.signal_type == "entry":
            self._handle_entry_signal(signal)
        else:
            self._reject_signal(signal, f"unknown_signal_type:{signal.signal_type}")

    def on_price_update(self, timestamp: datetime, current_price: float) -> None:
        """
        Called on each new price bar. Checks:
        1. New trading day reset.
        2. Kill switch.
        3. Stop loss / take profit on open position.
        4. EOD flatten requirement.
        5. Equity curve update.
        """
        self._last_price = current_price
        self._last_timestamp = timestamp
        state = self.account.account_state

        # --- New day check (must come first to reset daily state) ---
        self._check_new_day(timestamp)

        # --- Kill switch re-check against current unrealized ---
        if state.open_position and not state.is_killed:
            unrealized = state.open_position.unrealized_pnl(
                current_price, self.contract_spec
            )
            effective_daily = state.daily_pnl + unrealized
            if effective_daily <= self.prop_rules.kill_switch_threshold:
                state.is_killed = True
                self._emit_alert(
                    "kill_switch",
                    f"Kill switch triggered: daily P&L with unrealized "
                    f"= ${effective_daily:,.2f} (threshold: "
                    f"${self.prop_rules.kill_switch_threshold:,.2f})",
                )
                # Force close the position
                exit_price = self._apply_slippage(
                    current_price, state.open_position.direction, is_entry=False,
                )
                self._close_position(timestamp, exit_price, "kill_switch")
                return

        # --- Stop loss / take profit ---
        if state.open_position:
            self._check_stops(timestamp, current_price)

        # --- EOD flatten ---
        if state.open_position:
            self._check_eod_flatten(timestamp, current_price)

        # --- Drawdown warning alerts ---
        if not state.is_killed:
            drawdown = state.current_drawdown
            max_dd = self.prop_rules.max_drawdown
            # Warn at 60% and 80% of max drawdown
            if max_dd < 0:
                pct_used = drawdown / max_dd if max_dd != 0 else 0
                if pct_used >= 0.8:
                    self._emit_alert(
                        "drawdown_critical",
                        f"CRITICAL: Drawdown ${drawdown:,.2f} is {pct_used:.0%} of "
                        f"max allowed ${max_dd:,.2f}",
                    )
                elif pct_used >= 0.6:
                    self._emit_alert(
                        "drawdown_warning",
                        f"WARNING: Drawdown ${drawdown:,.2f} is {pct_used:.0%} of "
                        f"max allowed ${max_dd:,.2f}",
                    )

        # --- Update equity curve ---
        equity = state.current_balance
        if state.open_position:
            equity += state.open_position.unrealized_pnl(
                current_price, self.contract_spec,
            )
        self.account.equity_curve.append((timestamp, equity))

    # ------------------------------------------------------------------
    # Public: Callback registration
    # ------------------------------------------------------------------

    def on_trade(self, handler: Callable) -> None:
        """Register callback for completed trades."""
        self._trade_handlers.append(handler)

    def on_alert(self, handler: Callable) -> None:
        """Register callback for alerts (drawdown warnings, kill switch, etc.)."""
        self._alert_handlers.append(handler)

    # ------------------------------------------------------------------
    # Internal: Signal handlers
    # ------------------------------------------------------------------

    def _handle_entry_signal(self, signal: Signal) -> None:
        """Process an entry signal."""
        state = self.account.account_state

        # If we already have a position in the same direction, reject
        if state.open_position:
            if state.open_position.direction == signal.direction:
                self._reject_signal(signal, "already_positioned_same_direction")
                return
            # Opposite direction entry while positioned -> treat as close + new entry
            logger.info(
                "Entry signal %s while positioned %s — closing existing position first",
                signal.direction,
                state.open_position.direction,
            )
            close_price = self._apply_slippage(
                signal.price, state.open_position.direction, is_entry=False,
            )
            self._close_position(signal.timestamp, close_price, "signal_reversal")

        # Run risk manager pre-trade checks
        allowed, reason = self.risk_manager.pre_trade_check(
            signal.timestamp,
            signal.direction,
            signal.contracts,
            state,
        )
        if not allowed:
            self._reject_signal(signal, reason)
            return

        # Simulate fill
        fill_price = self._apply_slippage(signal.price, signal.direction, is_entry=True)
        self._open_position(signal, fill_price)

    def _handle_exit_signal(self, signal: Signal) -> None:
        """Process an exit signal."""
        state = self.account.account_state

        if not state.open_position:
            self._reject_signal(signal, "no_position_to_exit")
            return

        # Exit direction should match (exit a long = exit signal for "long")
        # Accept exit signals that target our current position direction
        if signal.direction != state.open_position.direction:
            self._reject_signal(signal, "exit_direction_mismatch")
            return

        exit_price = self._apply_slippage(
            signal.price, state.open_position.direction, is_entry=False,
        )
        self._close_position(signal.timestamp, exit_price, "signal")

    def _handle_reversal_signal(self, signal: Signal) -> None:
        """Process a reversal signal (close current + open opposite)."""
        state = self.account.account_state

        # Close existing position if any
        if state.open_position:
            close_price = self._apply_slippage(
                signal.price, state.open_position.direction, is_entry=False,
            )
            self._close_position(signal.timestamp, close_price, "signal_reversal")

        # Now open the new direction
        allowed, reason = self.risk_manager.pre_trade_check(
            signal.timestamp,
            signal.direction,
            signal.contracts,
            state,
        )
        if not allowed:
            self._reject_signal(signal, f"reversal_entry_blocked:{reason}")
            return

        fill_price = self._apply_slippage(signal.price, signal.direction, is_entry=True)
        self._open_position(signal, fill_price)

    # ------------------------------------------------------------------
    # Internal: Position management
    # ------------------------------------------------------------------

    def _open_position(self, signal: Signal, fill_price: float) -> Position:
        """Open a new position from a signal."""
        state = self.account.account_state

        position = Position(
            symbol=self.contract_spec.symbol,
            direction=signal.direction,
            entry_time=signal.timestamp,
            entry_price=fill_price,
            contracts=signal.contracts,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        state.open_position = position

        self.account.signals_executed += 1

        logger.info(
            "OPEN %s %d %s @ %.2f (signal: %.2f, slippage: %.2f) | SL=%.2f TP=%.2f",
            signal.direction.upper(),
            signal.contracts,
            self.contract_spec.symbol,
            fill_price,
            signal.price,
            abs(fill_price - signal.price),
            signal.stop_loss or 0.0,
            signal.take_profit or 0.0,
        )

        return position

    def _close_position(
        self, exit_time: datetime, exit_price: float, exit_reason: str,
    ) -> Trade:
        """Close the current position and record the trade."""
        state = self.account.account_state
        pos = state.open_position

        if pos is None:
            raise RuntimeError("No open position to close")

        # Calculate P&L
        if pos.direction == "long":
            points = exit_price - pos.entry_price
        else:
            points = pos.entry_price - exit_price

        gross_pnl = points * self.contract_spec.point_value * pos.contracts
        commission = self.prop_rules.total_cost_per_contract_rt * pos.contracts

        # Slippage cost tracked separately for reporting (already baked into prices)
        slippage_ticks_total = self.slippage_ticks * 2  # entry + exit
        slippage_cost = (
            slippage_ticks_total * self.contract_spec.tick_value * pos.contracts
        )
        net_pnl = gross_pnl - commission

        duration = int((exit_time - pos.entry_time).total_seconds())

        # Determine session segment from entry time
        session = self._classify_session(pos.entry_time)

        trade = Trade(
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

        # Update account via risk manager
        self.risk_manager.post_trade_update(trade, state)

        # Store in history
        self.account.trade_history.append(trade)

        # Log and notify
        self._log_trade(trade)
        self._emit_trade(trade)

        logger.info(
            "CLOSE %s %d %s @ %.2f | reason=%s | gross=%.2f net=%.2f | "
            "duration=%ds | balance=%.2f",
            trade.direction.upper(),
            trade.contracts,
            trade.symbol,
            exit_price,
            exit_reason,
            gross_pnl,
            net_pnl,
            duration,
            state.current_balance,
        )

        return trade

    # ------------------------------------------------------------------
    # Internal: Slippage model
    # ------------------------------------------------------------------

    def _apply_slippage(
        self, price: float, direction: str, is_entry: bool,
    ) -> float:
        """
        Apply slippage to fill price.

        Slippage is always adverse:
        - Buying (long entry or short exit):  price moves UP
        - Selling (short entry or long exit): price moves DOWN
        """
        slip = self.slippage_ticks * self.contract_spec.tick_size
        is_buying = (direction == "long" and is_entry) or (
            direction == "short" and not is_entry
        )
        if is_buying:
            return price + slip
        else:
            return price - slip

    # ------------------------------------------------------------------
    # Internal: Stop / TP / EOD checks
    # ------------------------------------------------------------------

    def _check_stops(self, timestamp: datetime, current_price: float) -> None:
        """Check if stop loss or take profit has been hit."""
        state = self.account.account_state
        pos = state.open_position
        if pos is None:
            return

        # Check stop loss
        if pos.stop_loss is not None:
            stop_hit = (
                (pos.direction == "long" and current_price <= pos.stop_loss)
                or (pos.direction == "short" and current_price >= pos.stop_loss)
            )
            if stop_hit:
                # Fill at stop price (with slippage away from us)
                exit_price = self._apply_slippage(
                    pos.stop_loss, pos.direction, is_entry=False,
                )
                self._close_position(timestamp, exit_price, "stop_loss")
                return

        # Check take profit
        if pos.take_profit is not None:
            tp_hit = (
                (pos.direction == "long" and current_price >= pos.take_profit)
                or (pos.direction == "short" and current_price <= pos.take_profit)
            )
            if tp_hit:
                # Fill at TP price (with slippage — even exits have slippage)
                exit_price = self._apply_slippage(
                    pos.take_profit, pos.direction, is_entry=False,
                )
                self._close_position(timestamp, exit_price, "take_profit")
                return

    def _check_eod_flatten(
        self, timestamp: datetime, current_price: float,
    ) -> None:
        """Flatten position if end of day."""
        if self.risk_manager.should_flatten(timestamp):
            state = self.account.account_state
            if state.open_position:
                exit_price = self._apply_slippage(
                    current_price, state.open_position.direction, is_entry=False,
                )
                self._emit_alert(
                    "eod_flatten",
                    f"Flattening {state.open_position.direction} position at EOD "
                    f"({timestamp.strftime('%H:%M:%S')})",
                )
                self._close_position(timestamp, exit_price, "eod_flatten")

    def _check_new_day(self, timestamp: datetime) -> None:
        """Reset daily tracking if new trading day."""
        state = self.account.account_state
        date_str = timestamp.strftime("%Y-%m-%d")

        if date_str != state.current_date and state.current_date != "":
            # Save daily summary for the completed day before resetting
            self._save_daily_summary(state)

        if date_str != state.current_date:
            self.risk_manager.reset_daily(state, date_str)
            logger.info(
                "New trading day: %s | balance: $%,.2f | drawdown: $%,.2f",
                date_str,
                state.current_balance,
                state.current_drawdown,
            )

    # ------------------------------------------------------------------
    # Internal: Session classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_session(entry_time: datetime) -> str:
        """Classify entry time into a session segment."""
        if not hasattr(entry_time, "hour"):
            return "unknown"
        mins = entry_time.hour * 60 + entry_time.minute
        if mins < 570:       # before 9:30
            return "pre_market"
        elif mins >= 960:     # after 16:00
            return "post_close"
        else:
            return "core"

    # ------------------------------------------------------------------
    # Internal: Event emission
    # ------------------------------------------------------------------

    def _emit_trade(self, trade: Trade) -> None:
        """Notify trade handlers."""
        for handler in self._trade_handlers:
            try:
                handler(trade)
            except Exception:
                logger.exception("Trade handler raised an exception")

    def _emit_alert(self, alert_type: str, message: str) -> None:
        """Notify alert handlers."""
        logger.warning("[%s] %s", alert_type, message)
        for handler in self._alert_handlers:
            try:
                handler(alert_type, message)
            except Exception:
                logger.exception("Alert handler raised an exception")

    # ------------------------------------------------------------------
    # Internal: Rejection tracking
    # ------------------------------------------------------------------

    def _reject_signal(self, signal: Signal, reason: str) -> None:
        """Record a rejected signal."""
        self.account.signals_rejected += 1
        self.account.rejection_reasons[reason] = (
            self.account.rejection_reasons.get(reason, 0) + 1
        )
        logger.debug(
            "REJECTED signal %s %s from %s — reason: %s",
            signal.direction,
            signal.signal_type,
            signal.strategy_name,
            reason,
        )

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _log_trade(self, trade: Trade) -> None:
        """Append trade to the daily log file."""
        date_str = (
            trade.exit_time.strftime("%Y-%m-%d")
            if isinstance(trade.exit_time, datetime)
            else str(trade.exit_time)[:10]
        )
        log_file = self.log_dir / f"trades_{date_str}.jsonl"

        record = {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_time": (
                trade.entry_time.isoformat()
                if isinstance(trade.entry_time, datetime)
                else str(trade.entry_time)
            ),
            "entry_price": trade.entry_price,
            "exit_time": (
                trade.exit_time.isoformat()
                if isinstance(trade.exit_time, datetime)
                else str(trade.exit_time)
            ),
            "exit_price": trade.exit_price,
            "contracts": trade.contracts,
            "gross_pnl": round(trade.gross_pnl, 2),
            "commission": round(trade.commission, 2),
            "slippage_cost": round(trade.slippage_cost, 2),
            "net_pnl": round(trade.net_pnl, 2),
            "duration_seconds": trade.duration_seconds,
            "session_segment": trade.session_segment,
            "exit_reason": trade.exit_reason,
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write trade log to %s", log_file)

    def _save_daily_summary(self, state: AccountState) -> None:
        """Capture end-of-day summary before resetting."""
        if not state.current_date:
            return

        day_trades = list(state.trades_today)
        day_pnl = state.daily_pnl
        summary = {
            "date": state.current_date,
            "pnl": round(day_pnl, 2),
            "num_trades": len(day_trades),
            "balance": round(state.current_balance, 2),
            "drawdown": round(state.current_drawdown, 2),
            "high_water_mark": round(state.high_water_mark, 2),
        }
        self.account.daily_summaries.append(summary)

    # ------------------------------------------------------------------
    # Public: Status & reporting
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current paper trading status as a dict."""
        state = self.account.account_state
        acct = self.account

        position_info = None
        if state.open_position:
            pos = state.open_position
            unrealized = 0.0
            if self._last_price is not None:
                unrealized = pos.unrealized_pnl(
                    self._last_price, self.contract_spec,
                )
            position_info = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_time": (
                    pos.entry_time.isoformat()
                    if isinstance(pos.entry_time, datetime)
                    else str(pos.entry_time)
                ),
                "entry_price": pos.entry_price,
                "contracts": pos.contracts,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "unrealized_pnl": round(unrealized, 2),
            }

        return {
            "balance": round(state.current_balance, 2),
            "starting_balance": state.starting_balance,
            "daily_pnl": round(state.daily_pnl, 2),
            "drawdown": round(state.current_drawdown, 2),
            "high_water_mark": round(state.high_water_mark, 2),
            "is_killed": state.is_killed,
            "current_date": state.current_date,
            "open_position": position_info,
            "trades_today": len(state.trades_today),
            "total_trades": len(acct.trade_history),
            "signals_received": acct.signals_received,
            "signals_executed": acct.signals_executed,
            "signals_rejected": acct.signals_rejected,
            "rejection_reasons": dict(acct.rejection_reasons),
            "prop_firm": self.prop_rules.firm_name,
            "daily_loss_limit": self.prop_rules.daily_loss_limit,
            "max_drawdown": self.prop_rules.max_drawdown,
        }

    def print_status(self) -> None:
        """Print current account status to stdout."""
        s = self.get_status()
        state = self.account.account_state

        print()
        print("=" * 60)
        print("  PAPER TRADING STATUS")
        print("=" * 60)
        print(f"  Firm:              {s['prop_firm']}")
        print(f"  Date:              {s['current_date']}")
        print(f"  Balance:           ${s['balance']:>12,.2f}")
        print(f"  Starting Balance:  ${s['starting_balance']:>12,.2f}")
        print(
            f"  Net P&L:           ${s['balance'] - s['starting_balance']:>12,.2f}"
        )
        print(f"  Daily P&L:         ${s['daily_pnl']:>12,.2f}")
        print(f"  Drawdown:          ${s['drawdown']:>12,.2f}")
        print(f"  High Water Mark:   ${s['high_water_mark']:>12,.2f}")
        print(f"  Kill Switch:       {'ACTIVE' if s['is_killed'] else 'OK'}")
        print()

        if s["open_position"]:
            p = s["open_position"]
            print(f"  Open Position:     {p['direction'].upper()} "
                  f"{p['contracts']} {p['symbol']} @ {p['entry_price']:.2f}")
            print(f"  Unrealized P&L:    ${p['unrealized_pnl']:>12,.2f}")
            if p["stop_loss"]:
                print(f"  Stop Loss:         {p['stop_loss']:.2f}")
            if p["take_profit"]:
                print(f"  Take Profit:       {p['take_profit']:.2f}")
        else:
            print("  Open Position:     FLAT")
        print()

        print(f"  Trades Today:      {s['trades_today']}")
        print(f"  Total Trades:      {s['total_trades']}")
        print(f"  Signals Received:  {s['signals_received']}")
        print(f"  Signals Executed:  {s['signals_executed']}")
        print(f"  Signals Rejected:  {s['signals_rejected']}")

        if s["rejection_reasons"]:
            print("  Rejection Breakdown:")
            for reason, count in sorted(
                s["rejection_reasons"].items(), key=lambda x: -x[1],
            ):
                print(f"    {reason:35s} {count:>5d}")
        print("=" * 60)
        print()

    def print_daily_summary(self) -> None:
        """Print daily P&L summary table."""
        summaries = list(self.account.daily_summaries)

        # Include the current day if it has activity
        state = self.account.account_state
        if state.current_date and state.trades_today:
            summaries.append({
                "date": state.current_date,
                "pnl": round(state.daily_pnl, 2),
                "num_trades": len(state.trades_today),
                "balance": round(state.current_balance, 2),
                "drawdown": round(state.current_drawdown, 2),
                "high_water_mark": round(state.high_water_mark, 2),
            })

        if not summaries:
            print("\n  No daily summaries available.\n")
            return

        print()
        print("=" * 78)
        print("  DAILY P&L SUMMARY")
        print("=" * 78)
        print(
            f"  {'Date':>12s}  {'P&L':>10s}  {'Trades':>6s}  "
            f"{'Balance':>12s}  {'Drawdown':>10s}  {'HWM':>12s}"
        )
        print("-" * 78)

        cumulative_pnl = 0.0
        for day in summaries:
            cumulative_pnl += day["pnl"]
            pnl_str = f"${day['pnl']:>9,.2f}"
            # Mark losing days
            marker = " " if day["pnl"] >= 0 else "*"
            print(
                f"  {day['date']:>12s}  {pnl_str}{marker} {day['num_trades']:>6d}  "
                f"${day['balance']:>11,.2f}  ${day['drawdown']:>9,.2f}  "
                f"${day['high_water_mark']:>11,.2f}"
            )

        print("-" * 78)
        print(f"  {'TOTAL':>12s}  ${cumulative_pnl:>9,.2f}  "
              f"{sum(d['num_trades'] for d in summaries):>6d}")
        print("=" * 78)
        print()

    def print_trade_log(self, last_n: int = 20) -> None:
        """Print recent trades."""
        trades = self.account.trade_history[-last_n:]
        if not trades:
            print("\n  No trades recorded.\n")
            return

        print()
        print("=" * 110)
        print(f"  TRADE LOG (last {last_n})")
        print("=" * 110)
        print(
            f"  {'ID':>8s}  {'Dir':>5s}  {'Entry':>10s}  {'Exit':>10s}  "
            f"{'Qty':>3s}  {'Gross':>10s}  {'Net':>10s}  {'Dur':>7s}  {'Reason':>15s}"
        )
        print("-" * 110)

        for t in trades:
            dur_str = self._format_duration(t.duration_seconds)
            net_marker = " " if t.net_pnl >= 0 else "*"
            print(
                f"  {t.trade_id:>8s}  {t.direction:>5s}  "
                f"{t.entry_price:>10.2f}  {t.exit_price:>10.2f}  "
                f"{t.contracts:>3d}  ${t.gross_pnl:>9,.2f}  "
                f"${t.net_pnl:>9,.2f}{net_marker} {dur_str:>7s}  "
                f"{t.exit_reason:>15s}"
            )

        print("-" * 110)

        total_net = sum(t.net_pnl for t in trades)
        total_gross = sum(t.gross_pnl for t in trades)
        winners = sum(1 for t in trades if t.net_pnl > 0)
        print(
            f"  {'':>8s}  {'':>5s}  {'':>10s}  {'':>10s}  "
            f"{'':>3s}  ${total_gross:>9,.2f}  ${total_net:>9,.2f}  "
            f"{'':>7s}  WR: {winners}/{len(trades)}"
        )
        print("=" * 110)
        print()

    # ------------------------------------------------------------------
    # Public: Export
    # ------------------------------------------------------------------

    def export_results(self, filepath: str | Path | None = None) -> Path:
        """Export complete paper trading results to JSON."""
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"paper_results_{ts}.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Finalize: save current day summary if not already saved
        state = self.account.account_state
        if state.current_date and state.trades_today:
            # Check we haven't already saved this day
            saved_dates = {s["date"] for s in self.account.daily_summaries}
            if state.current_date not in saved_dates:
                self._save_daily_summary(state)

        def _serialize_trade(t: Trade) -> dict:
            return {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": (
                    t.entry_time.isoformat()
                    if isinstance(t.entry_time, datetime)
                    else str(t.entry_time)
                ),
                "entry_price": t.entry_price,
                "exit_time": (
                    t.exit_time.isoformat()
                    if isinstance(t.exit_time, datetime)
                    else str(t.exit_time)
                ),
                "exit_price": t.exit_price,
                "contracts": t.contracts,
                "gross_pnl": round(t.gross_pnl, 2),
                "commission": round(t.commission, 2),
                "slippage_cost": round(t.slippage_cost, 2),
                "net_pnl": round(t.net_pnl, 2),
                "duration_seconds": t.duration_seconds,
                "session_segment": t.session_segment,
                "exit_reason": t.exit_reason,
                "signals_used": t.signals_used,
            }

        result = {
            "meta": {
                "exported_at": datetime.now().isoformat(),
                "prop_firm": self.prop_rules.firm_name,
                "account_size": self.prop_rules.account_size,
                "symbol": self.contract_spec.symbol,
                "slippage_ticks": self.slippage_ticks,
                "initial_balance": self.initial_balance,
            },
            "status": self.get_status(),
            "trades": [_serialize_trade(t) for t in self.account.trade_history],
            "daily_summaries": self.account.daily_summaries,
            "equity_curve": [
                {
                    "timestamp": (
                        ts.isoformat() if isinstance(ts, datetime) else str(ts)
                    ),
                    "equity": round(eq, 2),
                }
                for ts, eq in self.account.equity_curve
            ],
            "stats": self._compute_summary_stats(),
        }

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("Paper trading results exported to %s", filepath)
        return filepath

    # ------------------------------------------------------------------
    # Public: Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the paper trader to initial state."""
        self.account = PaperAccount(
            account_state=self.risk_manager.init_account(self.initial_balance)
        )
        self._last_price = None
        self._last_timestamp = None
        self._recent_signal_keys.clear()
        logger.info("Paper trader reset to initial state")

    # ------------------------------------------------------------------
    # Internal: Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Format duration in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        else:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h}h{m:02d}m"

    def _compute_summary_stats(self) -> dict:
        """Compute aggregate statistics for export."""
        trades = self.account.trade_history
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_net_pnl": 0.0,
                "total_gross_pnl": 0.0,
                "total_commissions": 0.0,
                "total_slippage_cost": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "largest_winner": 0.0,
                "largest_loser": 0.0,
                "avg_duration_seconds": 0.0,
                "max_consecutive_winners": 0,
                "max_consecutive_losers": 0,
                "exit_reasons": {},
            }

        pnls = [t.net_pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0

        # Max consecutive
        max_consec_w, max_consec_l = 0, 0
        cur_w, cur_l = 0, 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
            else:
                cur_l += 1
                cur_w = 0
            max_consec_w = max(max_consec_w, cur_w)
            max_consec_l = max(max_consec_l, cur_l)

        # Exit reason breakdown
        exit_reasons: dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": round(len(winners) / len(trades) * 100, 1),
            "total_net_pnl": round(sum(pnls), 2),
            "total_gross_pnl": round(sum(t.gross_pnl for t in trades), 2),
            "total_commissions": round(sum(t.commission for t in trades), 2),
            "total_slippage_cost": round(sum(t.slippage_cost for t in trades), 2),
            "profit_factor": (
                round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
            ),
            "avg_trade_pnl": round(sum(pnls) / len(pnls), 2),
            "avg_winner": round(sum(winners) / len(winners), 2) if winners else 0.0,
            "avg_loser": round(sum(losers) / len(losers), 2) if losers else 0.0,
            "largest_winner": round(max(pnls), 2),
            "largest_loser": round(min(pnls), 2),
            "avg_duration_seconds": round(
                sum(t.duration_seconds for t in trades) / len(trades), 1,
            ),
            "max_consecutive_winners": max_consec_w,
            "max_consecutive_losers": max_consec_l,
            "exit_reasons": exit_reasons,
        }
