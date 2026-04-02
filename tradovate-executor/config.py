"""
Configuration for HTF Swing v3 Hybrid v2 Executor
All strategy parameters, API endpoints, risk limits, and defaults.
"""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Optional
import json
import os


# ---------------------------------------------------------------------------
# NinjaTrader Bridge Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NTAccountConfig:
    """Connection parameters for one NinjaTrader account."""
    host: str = "127.0.0.1"   # NinjaTrader VM IP (host-only network adapter IPv4)
    port: int = 6000            # Must match TcpPort in PythonBridge strategy


@dataclass
class NTConfig:
    """NinjaTrader TCP bridge configuration — multi-account setup.

    Each Tradovate account name maps to the NinjaTrader account connection details.
    Market data still comes from Tradovate WebSocket even in NT mode.

    Account names must match the NinjaTrader Accounts tab exactly (case-sensitive).
    """
    accounts: dict = field(default_factory=dict)  # account_name → NTAccountConfig
    default_atm_template: str = "MNQ_2R"
    order_timeout_seconds: int = 10
    status_timeout_seconds: int = 5
    reconnect_max_backoff_seconds: int = 30
    symbol: str = "MNQU6"


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
    """Session and risk rules. LucidFlex uses max drawdown only (no daily loss limit)."""
    session_start: str = "09:30"            # ET
    no_new_entries_after: str = "16:30"     # 4:30 PM ET
    flatten_time: str = "16:45"             # 4:45 PM ET
    monthly_loss_limit: float = -4500.0     # Max drawdown / monthly limit
    daily_loss_limit: Optional[float] = None  # Per-day loss limit (None = disabled, LucidFlex default)
    timezone: str = "US/Eastern"


# LucidFlex exact tier values
_LUCID_TIERS = {
    25_000:  {"monthly_loss_limit": -1000.0, "profit_target": 1250.0,  "max_drawdown": -1000.0},
    50_000:  {"monthly_loss_limit": -2000.0, "profit_target": 3000.0,  "max_drawdown": -2000.0},
    100_000: {"monthly_loss_limit": -3000.0, "profit_target": 6000.0,  "max_drawdown": -3000.0},
    150_000: {"monthly_loss_limit": -4500.0, "profit_target": 9000.0,  "max_drawdown": -4500.0},
}


def lucid_defaults(account_size: float) -> dict:
    """
    Return exact LucidFlex risk parameters for a given account size.
    Snaps to nearest tier (25k/50k/100k/150k).
    No daily loss limit — LucidFlex uses max drawdown only.
    """
    sizes = sorted(_LUCID_TIERS.keys())
    best = min(sizes, key=lambda s: abs(s - account_size))
    return dict(_LUCID_TIERS[best])


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
    cid: int = 8                       # Tradovate public API client ID
    sec: str = "9c4e7db2-0e37-4169-915c-2a8fc0571dc2"  # Tradovate public API secret
    is_master: bool = False
    sizing_mode: SizingMode = SizingMode.MIRROR
    account_size: float = 150_000.0    # For scaled sizing
    fixed_sizes: dict = field(default_factory=lambda: {
        "RSI": 3, "IB": 3, "MOM": 3
    })
    min_contracts: int = 1
    monthly_loss_limit: float = -4500.0  # Max drawdown limit (LucidFlex)
    max_drawdown: float = -4500.0        # LucidFlex max drawdown (same as monthly_loss_limit)

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
            return max(scaled, self.min_contracts)  # at least min_contracts

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
    nt: Optional[NTConfig] = None   # When set, order execution routes through NinjaTrader TCP bridge

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
        """
        Save non-sensitive config to JSON.
        Credentials (password, sec) are intentionally omitted — manage
        accounts via the dashboard server's config_store (which handles
        Fernet encryption). This method is only used to write a sample
        config on first run.
        """
        data = {
            "environment": self.environment.value,
            "symbol": self.symbol,
            "session": asdict(self.session),
            "rsi": asdict(self.rsi),
            "ib": asdict(self.ib),
            "mom": asdict(self.mom),
            "accounts": [],
        }
        if self.nt:
            data["ninjatrader"] = {
                "_setup_instructions": (
                    "Account names must match the NinjaTrader Accounts tab exactly. "
                    "Set each host to the Windows VM's host-only network IPv4 address "
                    "(run 'ipconfig' in the VM to find it)."
                ),
                "accounts": {
                    name: {"host": acct.host, "port": acct.port}
                    for name, acct in self.nt.accounts.items()
                },
                "default_atm_template": self.nt.default_atm_template,
                "order_timeout_seconds": self.nt.order_timeout_seconds,
                "status_timeout_seconds": self.nt.status_timeout_seconds,
                "reconnect_max_backoff_seconds": self.nt.reconnect_max_backoff_seconds,
                "symbol": self.nt.symbol,
            }
        for acct in self.accounts:
            data["accounts"].append({
                "name": acct.name,
                "username": acct.username,
                "password": "",   # Never written in plaintext — use config_store
                "device_id": acct.device_id,
                "app_id": acct.app_id,
                "app_version": acct.app_version,
                "cid": acct.cid,
                "sec": "",        # Never written in plaintext — use config_store
                "is_master": acct.is_master,
                "sizing_mode": acct.sizing_mode.value,
                "account_size": acct.account_size,
                "fixed_sizes": acct.fixed_sizes,
                "min_contracts": acct.min_contracts,
                "monthly_loss_limit": acct.monthly_loss_limit,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        """Load config from JSON, decrypting Fernet-encrypted fields if present."""
        # Decrypt any "enc:..." values written by the dashboard config_store
        try:
            from server.config_store import _decrypt
        except ImportError:
            _decrypt = lambda x: x  # noqa: E731

        with open(path) as f:
            data = json.load(f)
        cfg = cls(
            environment=Environment(data.get("environment", "demo")),
            symbol=data.get("symbol", FRONT_MONTH),
            session=_load_dataclass(SessionConfig, data.get("session")),
            rsi=_load_dataclass(RSIParams, data.get("rsi")),
            ib=_load_dataclass(IBParams, data.get("ib")),
            mom=_load_dataclass(MOMParams, data.get("mom")),
        )
        for acct_data in data.get("accounts", []):
            cfg.accounts.append(AccountConfig(
                name=acct_data["name"],
                username=acct_data["username"],
                password=_decrypt(acct_data.get("password", "")),
                device_id=acct_data.get("device_id", f"device-{acct_data['name']}"),
                app_id=acct_data.get("app_id", "HTFSwing"),
                app_version=acct_data.get("app_version", "1.0.0"),
                cid=acct_data.get("cid", 8),
                sec=_decrypt(acct_data.get("sec", "")),
                is_master=acct_data.get("is_master", False),
                sizing_mode=SizingMode(acct_data.get("sizing_mode", "mirror")),
                account_size=acct_data.get("account_size", 150_000.0),
                fixed_sizes=acct_data.get("fixed_sizes", {"RSI": 3, "IB": 3, "MOM": 3}),
                min_contracts=acct_data.get("min_contracts", 1),
                monthly_loss_limit=acct_data.get("monthly_loss_limit", -4500.0),
            ))
        # Load ninjatrader config — "ninjatrader" key (new) with "nt" fallback (old)
        nt_raw = data.get("ninjatrader") or data.get("nt")
        if nt_raw:
            if "accounts" in nt_raw:
                # New multi-account format
                accounts_dict = {}
                for name, acct_data in nt_raw["accounts"].items():
                    if name.startswith("_"):  # skip comment/instruction keys
                        continue
                    accounts_dict[name] = NTAccountConfig(
                        host=acct_data.get("host", "127.0.0.1"),
                        port=acct_data.get("port", 6000),
                    )
                cfg.nt = NTConfig(
                    accounts=accounts_dict,
                    default_atm_template=nt_raw.get("default_atm_template", "MNQ_2R"),
                    order_timeout_seconds=nt_raw.get("order_timeout_seconds", 10),
                    status_timeout_seconds=nt_raw.get("status_timeout_seconds", 5),
                    reconnect_max_backoff_seconds=nt_raw.get("reconnect_max_backoff_seconds", 30),
                    symbol=nt_raw.get("symbol", "MNQU6"),
                )
            else:
                # Old single-host "nt" format — convert to new structure
                cfg.nt = NTConfig(
                    accounts={"default": NTAccountConfig(
                        host=nt_raw.get("host", "127.0.0.1"),
                        port=nt_raw.get("port", 6000),
                    )},
                )
        return cfg


def _load_dataclass(cls_, raw: Optional[dict]):
    """Load a dataclass from a partial JSON object, ignoring unknown keys."""
    if not raw:
        return cls_()

    allowed = {f.name for f in fields(cls_)}
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return cls_(**filtered)
