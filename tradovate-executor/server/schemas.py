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
