from pydantic import BaseModel
from typing import Optional


class AccountCreate(BaseModel):
    name: str
    username: str
    password: str
    cid: int = 8
    sec: str = "9c4e7db2-0e37-4169-915c-2a8fc0571dc2"
    device_id: str = ""
    is_master: bool = False
    sizing_mode: str = "mirror"
    account_size: float = 150000.0
    fixed_sizes: dict[str, int] = {"RSI": 3, "IB": 3, "MOM": 3}
    starting_balance: float = 150000.0
    profit_target: float = 9000.0
    max_drawdown: float = -4500.0
    account_type: str = "eval"
    monthly_loss_limit: float = -4500.0
    min_contracts: int = 1


class AccountUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    cid: Optional[int] = None
    sec: Optional[str] = None
    device_id: Optional[str] = None
    is_master: Optional[bool] = None
    sizing_mode: Optional[str] = None
    account_size: Optional[float] = None
    fixed_sizes: Optional[dict[str, int]] = None
    starting_balance: Optional[float] = None
    profit_target: Optional[float] = None
    max_drawdown: Optional[float] = None
    account_type: Optional[str] = None
    monthly_loss_limit: Optional[float] = None
    min_contracts: Optional[int] = None


class AccountResponse(BaseModel):
    name: str
    username: str
    password: str
    cid: int
    sec: str
    device_id: str
    is_master: bool
    sizing_mode: str
    account_size: float
    fixed_sizes: dict[str, int]
    starting_balance: float = 150000.0
    profit_target: float = 9000.0
    max_drawdown: float = -4500.0
    account_type: str = "eval"
    monthly_loss_limit: float = -4500.0
    min_contracts: int = 1


class AuthTestRequest(BaseModel):
    name: str


class AuthTestResponse(BaseModel):
    success: bool
    account_id: Optional[int] = None
    user_id: Optional[int] = None
    error: Optional[str] = None


class EngineStatus(BaseModel):
    running: bool
    can_trade: bool
    daily_pnl: float
    monthly_pnl: float
    monthly_limit: float
    monthly_limit_hit: bool
    positions: dict
    pending_signals: int
    connected_accounts: list[dict]


class EnvironmentUpdate(BaseModel):
    environment: str


class StrategyConfigResponse(BaseModel):
    contracts: int
    stop_loss_pts: float
    take_profit_pts: float
    max_hold_bars: int
    period: Optional[int] = None
    oversold: Optional[float] = None
    overbought: Optional[float] = None
    ib_start: Optional[str] = None
    ib_end: Optional[str] = None
    ib_range_lookback: Optional[int] = None
    ib_range_pct_low: Optional[float] = None
    ib_range_pct_high: Optional[float] = None
    atr_period: Optional[int] = None
    ema_period: Optional[int] = None
    vol_sma_period: Optional[int] = None


class SessionConfigResponse(BaseModel):
    session_start: str
    no_new_entries_after: str
    flatten_time: str
    monthly_loss_limit: float
    daily_loss_limit: Optional[float] = None
    timezone: str


class NTConnectionResponse(BaseModel):
    name: str
    host: str
    port: int


class RuntimeConfigResponse(BaseModel):
    environment: str
    symbol: str
    nt_enabled: bool
    nt_only: bool
    nt_accounts: list[NTConnectionResponse]
    session: SessionConfigResponse
    rsi: StrategyConfigResponse
    ib: StrategyConfigResponse
    mom: StrategyConfigResponse


class NTOnlySetupUpdate(BaseModel):
    account_name: str
    host: str
    port: int = 6000
    symbol: str = "MNQU6"
    contracts: int = 1
    monthly_loss_limit: float = -1000.0


class AccountStatusResponse(BaseModel):
    name: str
    balance: float
    starting_balance: float
    pnl_total: float
    drawdown_current: float
    drawdown_max_allowed: float
    drawdown_remaining: float
    drawdown_pct_used: float
    profit_target: float
    profit_target_progress: float
    daily_pnl: float
    trades_today: int
    status: str
    is_master: bool
    account_type: str
    environment: str
