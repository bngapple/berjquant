"""Fast backtester for HFT scalping — current-bar fills, zero entry slippage.

Simulates limit order entries (fill at signal bar's close, 0 slippage)
with market order exits (1 tick slippage). Does NOT modify the original
VectorizedBacktester. This is a separate class for HFT testing only.
"""

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from engine.backtester import SlippageModel, ET, _get_trailing_config, Strategy
from engine.risk_manager import RiskManager
from engine.utils import (
    AccountState, BacktestConfig, BacktestResult,
    ContractSpec, Position, Trade,
)


class FastBacktester:
    """HFT backtester: current-bar fills, zero entry slippage, 1-tick exit slippage."""

    def __init__(self, data, risk_manager, contract_spec, config):
        self.data = data
        self.risk_manager = risk_manager
        self.contract_spec = contract_spec
        self.config = config
        # Entry: 0 slippage (limit order). Exit: 1 tick (market order).
        self.exit_slippage = SlippageModel(fixed_ticks=1)

    def run(self, strategy: Strategy) -> BacktestResult:
        df = strategy.compute_signals(self.data)
        for col in ["entry_long", "entry_short", "exit_long", "exit_short"]:
            if col not in df.columns:
                raise ValueError(f"Missing '{col}'")

        prop_rules = self.risk_manager.prop_rules
        account = self.risk_manager.init_account(self.config.initial_capital)
        trades = []
        equity_curve = []
        rows = df.to_dicts()
        current_date = ""
        trailing_cfg = _get_trailing_config(strategy)

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
                self.risk_manager.post_trade_update(trade, account)

            # Check time exit
            if account.open_position:
                pos = account.open_position
                er = getattr(strategy, 'exit_rules', None)
                if er and hasattr(er, 'time_exit_minutes') and er.time_exit_minutes:
                    hold_secs = (ts - pos.entry_time).total_seconds()
                    if hold_secs >= er.time_exit_minutes * 60:
                        trade = self._close(account, ts, row["close"], "time_exit")
                        trades.append(trade)
                        self.risk_manager.post_trade_update(trade, account)

            # Update trailing stop
            if account.open_position and trailing_cfg:
                self._update_trailing(account.open_position, row)

            # Check SL/TP
            if account.open_position:
                trade = self._check_stops(account, row, ts)
                if trade:
                    trades.append(trade)
                    self.risk_manager.post_trade_update(trade, account)

            # CURRENT-BAR ENTRY: signal fires on bar N, fill at bar N's close (not N+1 open)
            if not account.open_position:
                direction = None
                if row.get("entry_long", False):
                    direction = "long"
                elif row.get("entry_short", False):
                    direction = "short"

                if direction:
                    contracts = strategy.get_position_size(account, self.contract_spec, prop_rules)
                    allowed, _ = self.risk_manager.pre_trade_check(ts, direction, contracts, account)
                    if allowed:
                        # ZERO ENTRY SLIPPAGE — limit order fill at close
                        fill_price = row["close"]
                        stop = strategy.get_stop_loss(fill_price, direction)
                        tp = strategy.get_take_profit(fill_price, direction)
                        pos = Position(symbol=self.contract_spec.symbol, direction=direction,
                                       entry_time=ts, entry_price=fill_price, contracts=contracts,
                                       stop_loss=stop, take_profit=tp)
                        if trailing_cfg:
                            pos._trailing_active = False
                            pos._trailing_activation = trailing_cfg[0]
                            pos._trailing_distance = trailing_cfg[1]
                        account.open_position = pos

            equity = account.current_balance
            if account.open_position:
                equity += account.open_position.unrealized_pnl(row["close"], self.contract_spec)
            equity_curve.append((ts, equity))

        # Force close remaining
        if account.open_position and rows:
            last = rows[-1]
            ts = last["timestamp"]
            if isinstance(ts, str): ts = datetime.fromisoformat(ts)
            trade = self._close(account, ts, last["close"], "backtest_end")
            trades.append(trade)
            self.risk_manager.post_trade_update(trade, account)

        return BacktestResult(strategy_name=strategy.name, config=self.config,
                              trades=trades, equity_curve=equity_curve)

    def _close(self, account, exit_time, exit_price_raw, exit_reason):
        pos = account.open_position
        # 1-tick exit slippage (market order)
        exit_dir = "short" if pos.direction == "long" else "long"
        exit_price = self.exit_slippage.apply(exit_price_raw, exit_dir, self.contract_spec)
        points = (exit_price - pos.entry_price) if pos.direction == "long" else (pos.entry_price - exit_price)
        gross = points * self.contract_spec.point_value * pos.contracts
        comm = self.risk_manager.prop_rules.total_cost_per_contract_rt * pos.contracts
        # Slippage cost tracked: 1 tick on exit only
        slip_cost = 1 * self.contract_spec.tick_value * pos.contracts
        net = gross - comm
        dur = int((exit_time - pos.entry_time).total_seconds())
        account.open_position = None
        account.current_balance += net
        return Trade(trade_id=str(uuid.uuid4())[:8], symbol=pos.symbol, direction=pos.direction,
                     entry_time=pos.entry_time, entry_price=pos.entry_price,
                     exit_time=exit_time, exit_price=exit_price, contracts=pos.contracts,
                     gross_pnl=gross, commission=comm, slippage_cost=slip_cost, net_pnl=net,
                     duration_seconds=dur, session_segment="core", exit_reason=exit_reason)

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
        if pos.direction == "long" and (not pos.stop_loss or ns > pos.stop_loss): pos.stop_loss = ns
        elif pos.direction == "short" and (not pos.stop_loss or ns < pos.stop_loss): pos.stop_loss = ns
