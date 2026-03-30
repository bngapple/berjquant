"""
Configuration for HTF Swing v3 Hybrid v2 Executor
All strategy parameters, API endpoints, risk limits, and defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import os


class Environment(Enum):
    DEMO = "demo"
    LIVE = "live"


class SizingMode(Enum):
    MIRROR = "mirror"      # Same as master account
    FIXED = "fixed"        # Manually set per strategy
    SCALED = "scaled"      # Proportional to account size


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

API_URLS = {
    Environment.DEMO: {
        "rest": "https://demo.tradovateapi.com/v1",
        "ws_orders": "wss://demo.tradovateapi.com/v1/websocket",
        "ws_market": "wss://md-demo.tradovateapi.com/v1/websocket",
    },
    Environment.LIVE: {
        "rest": "https://live.tradovateapi.com/v1",
        "ws_orders": "wss://live.tradovateapi.com/v1/websocket",
        "ws_market": "wss://md.tradovateapi.com/v1/websocket",
    },
}


# ---------------------------------------------------------------------------
# Contract Spec
# ---------------------------------------------------------------------------

SYMBOL = "MNQ"
FRONT_MONTH = "MNQM6"          # June 2026 — update quarterly
TICK_SIZE = 0.25                # Minimum price increment
TICK_VALUE = 0.50               # Dollar value per tick per contract
POINT_VALUE = 2.00              # Dollar value per point per contract


# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RSIParams:
    """Strategy 1: RSI Extremes"""
    period: int = 5
    oversold: float = 35.0
    overbought: float = 65.0
    contracts: int = 3
    stop_loss_pts: float = 10.0
    take_profit_pts: float = 100.0
    max_hold_bars: int = 5       # 75 min at 15m bars


@dataclass(frozen=True)
class IBParams:
    """Strategy 2: IB Breakout"""
    ib_start: str = "09:30"      # ET
    ib_end: str = "10:00"        # ET
    contracts: int = 3
    stop_loss_pts: float = 10.0
    take_profit_pts: float = 120.0
    max_hold_bars: int = 15
    ib_range_lookback: int = 50  # Days for percentile calc
    ib_range_pct_low: float = 25.0
    ib_range_pct_high: float = 75.0


@dataclass(frozen=True)
class MOMParams:
    """Strategy 3: Momentum Bars"""
    atr_period: int = 14
    ema_period: int = 21
    vol_sma_period: int = 20
    contracts: int = 3
    stop_loss_pts: float = 15.0
    take_profit_pts: float = 100.0
    max_hold_bars: int = 5


# ---------------------------------------------------------------------------
# Session / Risk Limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    """LucidFlex 150K session rules"""
    session_start: str = "09:30"      # ET
    no_new_entries_after: str = "16:30"  # 4:30 PM ET
    flatten_time: str = "16:45"          # 4:45 PM ET
    daily_loss_limit: float = -3000.0
    monthly_loss_limit: float = -4500.0
    timezone: str = "US/Eastern"


# ---------------------------------------------------------------------------
# Account Config
# ---------------------------------------------------------------------------

@dataclass
class AccountConfig:
    """Configuration for a single Tradovate account."""
    name: str                          # Friendly name ("150k-1", "25k-eval")
    username: str
    password: str
    device_id: str                     # Unique per device/account
    app_id: str = "HTFSwing"
    app_version: str = "1.0.0"
    cid: int = 0                       # Client ID (from Tradovate dev portal)
    sec: str = ""                      # API secret (from Tradovate dev portal)
    is_master: bool = False
    sizing_mode: SizingMode = SizingMode.MIRROR
    account_size: float = 150_000.0    # For scaled sizing
    fixed_sizes: dict = field(default_factory=lambda: {
        "RSI": 3, "IB": 3, "MOM": 3
    })

    def get_contracts(self, strategy: str, master_contracts: int) -> int:
        """Return contract count for this account + strategy."""
        if self.sizing_mode == SizingMode.MIRROR:
            return master_contracts

        if self.sizing_mode == SizingMode.FIXED:
            return self.fixed_sizes.get(strategy, master_contracts)

        if self.sizing_mode == SizingMode.SCALED:
            # Scale relative to 150k baseline
            ratio = self.account_size / 150_000.0
            scaled = int(master_contracts * ratio)  # floor
            return max(scaled, 0)  # Never negative

        return master_contracts


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level application configuration."""
    environment: Environment = Environment.DEMO
    symbol: str = FRONT_MONTH
    rsi: RSIParams = field(default_factory=RSIParams)
    ib: IBParams = field(default_factory=IBParams)
    mom: MOMParams = field(default_factory=MOMParams)
    session: SessionConfig = field(default_factory=SessionConfig)
    accounts: list[AccountConfig] = field(default_factory=list)
    log_dir: str = "logs"

    @property
    def rest_url(self) -> str:
        return API_URLS[self.environment]["rest"]

    @property
    def ws_orders_url(self) -> str:
        return API_URLS[self.environment]["ws_orders"]

    @property
    def ws_market_url(self) -> str:
        return API_URLS[self.environment]["ws_market"]

    @property
    def master_account(self) -> Optional[AccountConfig]:
        for acct in self.accounts:
            if acct.is_master:
                return acct
        return None

    @property
    def copy_accounts(self) -> list[AccountConfig]:
        return [a for a in self.accounts if not a.is_master]

    def save(self, path: str = "config.json"):
        """Save config to JSON (WARNING: contains credentials)."""
        data = {
            "environment": self.environment.value,
            "symbol": self.symbol,
            "accounts": []
        }
        for acct in self.accounts:
            data["accounts"].append({
                "name": acct.name,
                "username": acct.username,
                "password": acct.password,
                "device_id": acct.device_id,
                "app_id": acct.app_id,
                "app_version": acct.app_version,
                "cid": acct.cid,
                "sec": acct.sec,
                "is_master": acct.is_master,
                "sizing_mode": acct.sizing_mode.value,
                "account_size": acct.account_size,
                "fixed_sizes": acct.fixed_sizes,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls(
            environment=Environment(data.get("environment", "demo")),
            symbol=data.get("symbol", FRONT_MONTH),
        )
        for acct_data in data.get("accounts", []):
            cfg.accounts.append(AccountConfig(
                name=acct_data["name"],
                username=acct_data["username"],
                password=acct_data["password"],
                device_id=acct_data["device_id"],
                app_id=acct_data.get("app_id", "HTFSwing"),
                app_version=acct_data.get("app_version", "1.0.0"),
                cid=acct_data.get("cid", 0),
                sec=acct_data.get("sec", ""),
                is_master=acct_data.get("is_master", False),
                sizing_mode=SizingMode(acct_data.get("sizing_mode", "mirror")),
                account_size=acct_data.get("account_size", 150_000.0),
                fixed_sizes=acct_data.get("fixed_sizes", {"RSI": 3, "IB": 3, "MOM": 3}),
            ))
        return cfg
