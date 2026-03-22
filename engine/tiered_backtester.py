"""Tiered backtester — runs multiple strategy tiers with priority-based entries.

Champion tier gets priority: if a grinder position is open and the champion
signal fires, the grinder is closed immediately and the champion trade opens.
"""

import uuid
import importlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from engine.backtester import SlippageModel, ET
from engine.risk_manager import RiskManager
from engine.utils import (
    AccountState, BacktestConfig, BacktestResult,
    ContractSpec, Position, Trade, MNQ_SPEC,
)


@dataclass
class TierDef:
    """Definition of a single tier within a tiered strategy."""
    tier_name: str
    priority: int  # 1 = highest (champion), 2+ = grinders
    entry_signals: list[dict[str, Any]]
    entry_filters: list[dict[str, Any]]
    exit_rules: dict[str, Any]
    sizing_rules: dict[str, Any]
    require_all_entries: bool = True

    # Column names for this tier's signals (set during compute)
    long_col: str = ""
    short_col: str = ""


@dataclass
class TieredStrategy:
    """Strategy with multiple tiers, each with its own entry/exit/sizing rules."""
    name: str
    tiers: list[TierDef]

    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Run all tiers' signals in one pass. Each tier gets its own entry columns."""
        df = data["1m"].clone()
        applied: set[str] = set()

        for tier in self.tiers:
            # Apply entry signal functions
            for sig in tier.entry_signals:
                key = f"{sig['module']}.{sig['function']}"
                if key not in applied:
                    applied.add(key)
                    mod = importlib.import_module(sig["module"])
                    func = getattr(mod, sig["function"])
                    df = func(df, **sig.get("params", {}))

            # Apply filter signal functions
            for filt in tier.entry_filters:
                key = f"{filt['module']}.{filt['function']}"
                if key not in applied:
                    applied.add(key)
                    mod = importlib.import_module(filt["module"])
                    func = getattr(mod, filt["function"])
                    df = func(df, **filt.get("params", {}))

            # Combine entry signals for this tier
            long_exprs = []
            short_exprs = []
            for sig in tier.entry_signals:
                cols = sig["columns"]
                if cols["long"] in df.columns:
                    long_exprs.append(pl.col(cols["long"]).fill_null(False))
                if cols["short"] in df.columns:
                    short_exprs.append(pl.col(cols["short"]).fill_null(False))

            if long_exprs:
                combined_long = long_exprs[0]
                for e in long_exprs[1:]:
                    combined_long = combined_long & e if tier.require_all_entries else combined_long | e
            else:
                combined_long = pl.lit(False)

            if short_exprs:
                combined_short = short_exprs[0]
                for e in short_exprs[1:]:
                    combined_short = combined_short & e if tier.require_all_entries else combined_short | e
            else:
                combined_short = pl.lit(False)

            # Apply filters
            for filt in tier.entry_filters:
                col = filt.get("column", "")
                if col in df.columns:
                    fexpr = pl.col(col).fill_null(False)
                    combined_long = combined_long & fexpr
                    combined_short = combined_short & fexpr

            tier.long_col = f"entry_long_{tier.tier_name}"
            tier.short_col = f"entry_short_{tier.tier_name}"

            df = df.with_columns([
                combined_long.alias(tier.long_col),
                combined_short.alias(tier.short_col),
            ])

        return df


@dataclass
class TieredTrade:
    """A trade with tier info attached."""
    trade: Trade
    tier_name: str
    tier_priority: int


class TieredBacktester:
    """Backtester that handles priority-based tiered entries."""

    def __init__(self, data, risk_manager, contract_spec, config):
        self.data = data
        self.risk_manager = risk_manager
        self.contract_spec = contract_spec
        self.config = config
        self.slippage = SlippageModel(fixed_ticks=config.slippage_ticks)

    def run(self, strategy: TieredStrategy):
        """Run tiered backtest. Returns (trades_list, tiered_trades_list, equity_curve)."""
        df = strategy.compute_signals(self.data)
        rows = df.to_dicts()
        del df

        prop_rules = self.risk_manager.prop_rules
        account = self.risk_manager.init_account(self.config.initial_capital)

        trades: list[Trade] = []
        tiered_trades: list[TieredTrade] = []
        equity_curve = []
        current_date = ""
        prev_close = None

        # Track which tier owns the current position
        active_tier: TierDef | None = None
        # Pending entry: (direction, contracts, tier, exit_rules)
        pending_entry = None

        # Sort tiers by priority
        tiers_sorted = sorted(strategy.tiers, key=lambda t: t.priority)

        for row in rows:
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            date_str = ts.strftime("%Y-%m-%d")
            if date_str != current_date:
                current_date = date_str
                self.risk_manager.reset_daily(account, date_str)

            # EOD flatten
            if account.open_position and self.risk_manager.should_flatten(ts):
                trade = self._close(account, ts, row["close"], "eod_flatten")
                trades.append(trade)
                tiered_trades.append(TieredTrade(trade, active_tier.tier_name if active_tier else "?", active_tier.priority if active_tier else 99))
                self.risk_manager.post_trade_update(trade, account)
                active_tier = None

            # Execute pending entry
            if pending_entry and not account.open_position:
                direction, contracts, tier, exit_r = pending_entry
                allowed, _ = self.risk_manager.pre_trade_check(ts, direction, contracts, account)
                if allowed:
                    fill = self.slippage.apply(row["open"], direction, self.contract_spec)
                    sl_val = exit_r.get("stop_loss_value", 20)
                    tp_val = exit_r.get("take_profit_value", 100)
                    sl = fill - sl_val if direction == "long" else fill + sl_val
                    tp = fill + tp_val if direction == "long" else fill - tp_val

                    pos = Position(symbol=self.contract_spec.symbol, direction=direction,
                                   entry_time=ts, entry_price=fill, contracts=contracts,
                                   stop_loss=sl, take_profit=tp)
                    # Trailing stop
                    if exit_r.get("trailing_stop"):
                        pos._trailing_active = False
                        pos._trailing_activation = exit_r.get("trailing_activation", 4.0)
                        pos._trailing_distance = exit_r.get("trailing_distance", 2.0)
                    account.open_position = pos
                    active_tier = tier
                pending_entry = None

            # Gap check
            if account.open_position and prev_close is not None:
                trade = self._check_gap(account, row, ts, prev_close)
                if trade:
                    trades.append(trade)
                    tiered_trades.append(TieredTrade(trade, active_tier.tier_name if active_tier else "?", active_tier.priority if active_tier else 99))
                    self.risk_manager.post_trade_update(trade, account)
                    active_tier = None

            # Trailing stop update
            if account.open_position and hasattr(account.open_position, '_trailing_activation'):
                self._update_trailing(account.open_position, row)

            # SL/TP check
            if account.open_position:
                trade = self._check_stops(account, row, ts)
                if trade:
                    trades.append(trade)
                    tiered_trades.append(TieredTrade(trade, active_tier.tier_name if active_tier else "?", active_tier.priority if active_tier else 99))
                    self.risk_manager.post_trade_update(trade, account)
                    active_tier = None

            # ENTRY LOGIC — check tiers in priority order
            if pending_entry is None:
                for tier in tiers_sorted:
                    direction = None
                    if row.get(tier.long_col, False):
                        direction = "long"
                    elif row.get(tier.short_col, False):
                        direction = "short"

                    if direction is None:
                        continue

                    ct = tier.sizing_rules.get("fixed_contracts", 4)

                    if not account.open_position:
                        # No position — queue this tier's entry
                        pending_entry = (direction, ct, tier, tier.exit_rules)
                        break

                    elif active_tier and tier.priority < active_tier.priority:
                        # PREEMPTION: higher-priority tier signal fired while lower-priority position open
                        # Close the current position immediately
                        trade = self._close(account, ts, row["close"], "champion_preempt")
                        trades.append(trade)
                        tiered_trades.append(TieredTrade(trade, active_tier.tier_name, active_tier.priority))
                        self.risk_manager.post_trade_update(trade, account)
                        active_tier = None
                        # Queue the higher-priority entry
                        pending_entry = (direction, ct, tier, tier.exit_rules)
                        break

                    # If current position is same or higher priority, skip

            # Equity curve
            equity = account.current_balance
            if account.open_position:
                equity += account.open_position.unrealized_pnl(row["close"], self.contract_spec)
            equity_curve.append((ts, equity))
            prev_close = row["close"]

        # Force close remaining
        if account.open_position and rows:
            last = rows[-1]
            ts = last["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            trade = self._close(account, ts, last["close"], "backtest_end")
            trades.append(trade)
            tiered_trades.append(TieredTrade(trade, active_tier.tier_name if active_tier else "?", active_tier.priority if active_tier else 99))
            self.risk_manager.post_trade_update(trade, account)

        return trades, tiered_trades, equity_curve

    def _close(self, account, exit_time, exit_price_raw, exit_reason):
        pos = account.open_position
        exit_dir = "short" if pos.direction == "long" else "long"
        exit_price = self.slippage.apply(exit_price_raw, exit_dir, self.contract_spec)
        points = (exit_price - pos.entry_price) if pos.direction == "long" else (pos.entry_price - exit_price)
        gross = points * self.contract_spec.point_value * pos.contracts
        comm = self.risk_manager.prop_rules.total_cost_per_contract_rt * pos.contracts
        slip_cost = self.config.slippage_ticks * 2 * self.contract_spec.tick_value * pos.contracts
        net = gross - comm
        dur = int((exit_time - pos.entry_time).total_seconds())
        account.open_position = None
        account.current_balance += net
        return Trade(
            trade_id=str(uuid.uuid4())[:8], symbol=pos.symbol, direction=pos.direction,
            entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=exit_time, exit_price=exit_price, contracts=pos.contracts,
            gross_pnl=gross, commission=comm, slippage_cost=slip_cost, net_pnl=net,
            duration_seconds=dur, session_segment="core", exit_reason=exit_reason,
        )

    def _check_stops(self, account, row, ts):
        pos = account.open_position
        if not pos: return None
        h, l, o = row["high"], row["low"], row["open"]
        sl_hit = pos.stop_loss and ((pos.direction == "long" and l <= pos.stop_loss) or (pos.direction == "short" and h >= pos.stop_loss))
        tp_hit = pos.take_profit and ((pos.direction == "long" and h >= pos.take_profit) or (pos.direction == "short" and l <= pos.take_profit))
        if sl_hit and tp_hit:
            if abs(o - pos.stop_loss) <= abs(o - pos.take_profit):
                return self._close(account, ts, pos.stop_loss, "stop_loss")
            return self._close(account, ts, pos.take_profit, "take_profit")
        if sl_hit: return self._close(account, ts, pos.stop_loss, "stop_loss")
        if tp_hit: return self._close(account, ts, pos.take_profit, "take_profit")
        return None

    def _check_gap(self, account, row, ts, prev_close):
        pos = account.open_position
        if not pos or not pos.stop_loss: return None
        o = row["open"]
        if pos.direction == "long" and o < pos.stop_loss:
            return self._close(account, ts, o, "stop_loss_gap")
        if pos.direction == "short" and o > pos.stop_loss:
            return self._close(account, ts, o, "stop_loss_gap")
        return None

    @staticmethod
    def _update_trailing(pos, row):
        if not hasattr(pos, '_trailing_activation'): return
        h, l = row["high"], row["low"]
        if not pos._trailing_active:
            unr = (h - pos.entry_price) if pos.direction == "long" else (pos.entry_price - l)
            if unr >= pos._trailing_activation:
                pos._trailing_active = True
                ns = (h - pos._trailing_distance) if pos.direction == "long" else (l + pos._trailing_distance)
                if pos.stop_loss:
                    pos.stop_loss = max(pos.stop_loss, ns) if pos.direction == "long" else min(pos.stop_loss, ns)
                else:
                    pos.stop_loss = ns
            return
        ns = (h - pos._trailing_distance) if pos.direction == "long" else (l + pos._trailing_distance)
        if pos.direction == "long" and (not pos.stop_loss or ns > pos.stop_loss):
            pos.stop_loss = ns
        elif pos.direction == "short" and (not pos.stop_loss or ns < pos.stop_loss):
            pos.stop_loss = ns
