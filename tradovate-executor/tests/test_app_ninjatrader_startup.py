"""
Focused startup safety tests for NinjaTrader bridge mode.
"""

import os
import sys
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from config import AccountConfig, AppConfig, Environment, NTAccountConfig, NTConfig


class FakeAuth:
    def __init__(self, session):
        self._session = session
        self.authenticate_all = AsyncMock()
        self.on_token_renewed = None

    def get_master_session(self):
        return self._session

    def get_copy_sessions(self):
        return []


def make_session(name: str = "tv-master"):
    return SimpleNamespace(
        name=name,
        is_authenticated=True,
        access_token="token",
        md_access_token="md-token",
        user_id=123,
    )


def make_config(log_dir: str, nt_accounts: dict[str, NTAccountConfig]) -> AppConfig:
    return AppConfig(
        environment=Environment.DEMO,
        log_dir=log_dir,
        accounts=[
            AccountConfig(
                name="tv-master",
                username="user",
                password="pass",
                device_id="device-1",
                is_master=True,
            )
        ],
        nt=NTConfig(accounts=nt_accounts),
    )


def make_nt_only_config(log_dir: str, nt_accounts: dict[str, NTAccountConfig]) -> AppConfig:
    return AppConfig(
        environment=Environment.DEMO,
        log_dir=log_dir,
        accounts=[],
        nt=NTConfig(accounts=nt_accounts),
    )


@pytest.mark.asyncio
async def test_start_nt_mode_requires_exact_account_match(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app_module.RiskManager, "_load_state", lambda self: None)
    cfg = make_config(str(tmp_path), {
        "different-account": NTAccountConfig(host="127.0.0.1", port=6000),
    })
    fake_auth = FakeAuth(make_session())
    monkeypatch.setattr(app_module, "AuthManager", lambda config: fake_auth)

    executor = app_module.TradovateExecutor(cfg)

    await executor.start()

    assert executor.master_executor is None
    assert "Exact case-sensitive match required" in caplog.text
    fake_auth.authenticate_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_nt_mode_rejects_placeholder_host(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app_module.RiskManager, "_load_state", lambda self: None)
    cfg = make_config(str(tmp_path), {
        "tv-master": NTAccountConfig(host="REPLACE_WITH_VM_IPCONFIG_IPV4", port=6000),
    })
    fake_auth = FakeAuth(make_session())
    monkeypatch.setattr(app_module, "AuthManager", lambda config: fake_auth)

    executor = app_module.TradovateExecutor(cfg)

    await executor.start()

    assert executor.master_executor is None
    assert "still contains placeholder values" in caplog.text
    fake_auth.authenticate_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_nt_mode_aborts_if_bridge_never_connects(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app_module.RiskManager, "_load_state", lambda self: None)
    cfg = make_config(str(tmp_path), {
        "tv-master": NTAccountConfig(host="127.0.0.1", port=6000),
    })
    fake_auth = FakeAuth(make_session())
    monkeypatch.setattr(app_module, "AuthManager", lambda config: fake_auth)

    created = []

    class FakeBridge:
        def __init__(self, *args, **kwargs):
            self.connected = False
            self.shutdown_called = False
            self.exit_callback = None
            created.append(self)

        def set_exit_callback(self, callback):
            self.exit_callback = callback

        def set_market_callback(self, callback):
            self.market_callback = callback

        async def connect(self):
            return None

        async def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(app_module, "NinjaTraderBridge", FakeBridge)

    executor = app_module.TradovateExecutor(cfg)
    executor._wait_nt_connected = AsyncMock(return_value=False)

    await executor.start()

    assert len(created) == 1
    assert created[0].shutdown_called is True
    assert executor.master_executor is None
    executor._wait_nt_connected.assert_awaited_once()
    fake_auth.authenticate_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_nt_only_mode_skips_tradovate_auth(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app_module.RiskManager, "_load_state", lambda self: None)
    caplog.set_level(logging.INFO)
    cfg = make_nt_only_config(str(tmp_path), {
        "LTT07T22GBH": NTAccountConfig(host="127.0.0.1", port=6000),
    })
    fake_auth = FakeAuth(None)
    monkeypatch.setattr(app_module, "AuthManager", lambda config: fake_auth)

    created = []

    class FakeBridge:
        def __init__(self, config, session, nt_host, nt_port):
            self.session = session
            self.connected = True
            self.exit_callback = None
            self.market_callback = None
            self.strategy_orders = {}
            created.append(self)

        def set_exit_callback(self, callback):
            self.exit_callback = callback

        def set_market_callback(self, callback):
            self.market_callback = callback

        async def connect(self):
            return None

        async def shutdown(self):
            return None

    monkeypatch.setattr(app_module, "NinjaTraderBridge", FakeBridge)

    executor = app_module.TradovateExecutor(cfg)
    executor._wait_nt_connected = AsyncMock(return_value=True)
    executor._sync_positions = AsyncMock()
    executor.risk_manager.start_eod_timer = MagicMock()
    executor._shutdown_event.wait = AsyncMock(return_value=None)

    await executor.start()

    fake_auth.authenticate_all.assert_not_awaited()
    assert executor._master_session is not None
    assert executor._master_session.name == "LTT07T22GBH"
    assert len(created) == 1
    assert created[0].market_callback == executor._on_nt_market_message
    assert "NT-only mode detected" in caplog.text


@pytest.mark.asyncio
async def test_nt_market_messages_feed_market_data(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.RiskManager, "_load_state", lambda self: None)
    cfg = make_nt_only_config(str(tmp_path), {
        "LTT07T22GBH": NTAccountConfig(host="127.0.0.1", port=6000),
    })
    fake_auth = FakeAuth(None)
    monkeypatch.setattr(app_module, "AuthManager", lambda config: fake_auth)

    executor = app_module.TradovateExecutor(cfg)
    executor.market_data.on_tick = AsyncMock()
    executor.market_data.ingest_historical_bar = AsyncMock()

    await executor._on_nt_market_message({
        "type": "market",
        "timestamp": "2026-04-02T14:45:00-04:00",
        "price": 19510.25,
        "volume": 7,
    })
    await executor._on_nt_market_message({
        "type": "bar",
        "timestamp": "2026-04-02T14:30:00-04:00",
        "open": 19500.0,
        "high": 19525.0,
        "low": 19495.0,
        "close": 19510.25,
        "volume": 1234,
    })

    executor.market_data.on_tick.assert_awaited_once()
    executor.market_data.ingest_historical_bar.assert_awaited_once()
