from pydantic import BaseModel
from typing import Optional


class AccountCreate(BaseModel):
    name: str
    username: str
    password: str
    cid: int = 0
    sec: str = ""
    device_id: str = ""
    is_master: bool = False
    sizing_mode: str = "mirror"
    account_size: float = 150000.0
    fixed_sizes: dict[str, int] = {"RSI": 3, "IB": 3, "MOM": 3}
    starting_balance: float = 150000.0
    profit_target: float = 9000.0
    max_drawdown: float = -4500.0
    account_type: str = "eval"


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
    daily_limit: float
    monthly_limit: float
    daily_limit_hit: bool
    monthly_limit_hit: bool
    positions: dict
    pending_signals: int
    connected_accounts: list[dict]


class EnvironmentUpdate(BaseModel):
    environment: str


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
