"""Core data structures and utilities for the MCQ Engine."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ── Contract Specifications ──────────────────────────────────────────

@dataclass(frozen=True)
class ContractSpec:
    symbol: str           # "MNQ" or "NQ"
    tick_size: float      # price increment per tick (0.25 points)
    tick_value: float     # dollar value per tick
    point_value: float    # dollar value per point

    def ticks_to_dollars(self, ticks: int) -> float:
        return ticks * self.tick_value

    def points_to_dollars(self, points: float) -> float:
        return points * self.point_value


MNQ_SPEC = ContractSpec(symbol="MNQ", tick_size=0.25, tick_value=0.50, point_value=2.0)
NQ_SPEC = ContractSpec(symbol="NQ", tick_size=0.25, tick_value=5.00, point_value=20.0)

CONTRACT_SPECS = {"MNQ": MNQ_SPEC, "NQ": NQ_SPEC}


# ── Prop Firm Rules ──────────────────────────────────────────────────

@dataclass(frozen=True)
class PropFirmRules:
    firm_name: str
    account_size: float
    daily_loss_limit: float
    max_drawdown: float
    drawdown_type: str              # "trailing" | "eod" | "static"
    max_contracts: dict[str, int]
    consistency_rule_enabled: bool
    consistency_max_single_day_pct: float
    commission_per_contract_rt: float
    exchange_fees_per_contract_rt: float

    @property
    def total_cost_per_contract_rt(self) -> float:
        return self.commission_per_contract_rt + self.exchange_fees_per_contract_rt

    @property
    def kill_switch_threshold(self) -> float:
        """Kill switch at 80% of daily loss limit."""
        return self.daily_loss_limit * 0.8


# ── Account State ────────────────────────────────────────────────────

@dataclass
class AccountState:
    starting_balance: float
    current_balance: float
    daily_pnl: float = 0.0
    high_water_mark: float = 0.0
    open_position: "Position | None" = None
    trades_today: list["Trade"] = field(default_factory=list)
    is_killed: bool = False
    current_date: str = ""

    def __post_init__(self):
        if self.high_water_mark == 0.0:
            self.high_water_mark = self.starting_balance

    @property
    def current_drawdown(self) -> float:
        return self.current_balance - self.high_water_mark


# ── Position ─────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    direction: str          # "long" | "short"
    entry_time: datetime
    entry_price: float
    contracts: int
    stop_loss: float | None = None
    take_profit: float | None = None

    def unrealized_pnl(self, current_price: float, contract_spec: ContractSpec) -> float:
        if self.direction == "long":
            points = current_price - self.entry_price
        else:
            points = self.entry_price - current_price
        return points * contract_spec.point_value * self.contracts


# ── Trade ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trade:
    trade_id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    duration_seconds: int
    session_segment: str
    exit_reason: str        # "signal" | "stop_loss" | "take_profit" | "trailing_stop" | "time_exit" | "eod_flatten"
    signals_used: list[str] = field(default_factory=list)


# ── Backtest Config ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    prop_firm_profile: str
    start_date: str
    end_date: str
    slippage_ticks: int = 2
    initial_capital: float = 50000.0


# ── Backtest Result ──────────────────────────────────────────────────

@dataclass
class BacktestResult:
    strategy_name: str
    config: BacktestConfig
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]  # (timestamp, equity)
    metrics: "PerformanceMetrics | None" = None


# ── Performance Metrics ──────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_hold_time_seconds: float
    profit_by_session: dict[str, float]
    consistency_score: float   # max single day pnl as pct of total


# ── Config Loaders ───────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prop_firm_rules(config_dir: Path, profile: str) -> PropFirmRules:
    data = load_yaml(config_dir / "prop_firms.yaml")
    firm = data[profile]
    return PropFirmRules(
        firm_name=firm["firm_name"],
        account_size=firm["account_size"],
        daily_loss_limit=firm["daily_loss_limit"],
        max_drawdown=firm["max_drawdown"],
        drawdown_type=firm["drawdown_type"],
        max_contracts=firm["max_contracts"],
        consistency_rule_enabled=firm["consistency_rule"]["enabled"],
        consistency_max_single_day_pct=firm["consistency_rule"]["max_single_day_pct"],
        commission_per_contract_rt=firm["commission_per_contract_rt"],
        exchange_fees_per_contract_rt=firm["exchange_fees_per_contract_rt"],
    )


def load_session_config(config_dir: Path) -> dict:
    return load_yaml(config_dir / "sessions.yaml")


def load_events_calendar(config_dir: Path) -> dict:
    return load_yaml(config_dir / "events_calendar.yaml")
