"""
Tests for NinjaTraderBridge — uses a mock asyncio TCP server to simulate NinjaTrader.

Run: python -m pytest tests/test_ninjatrader_bridge.py -v
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
import pytest

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock
from config import AppConfig, Environment, AccountConfig, NTConfig
from signal_engine import Signal, Side
from order_executor import OrderStatus, OrderRecord
from ninjatrader_bridge import NinjaTraderBridge


# ---------------------------------------------------------------------------
# pytest-asyncio mode configuration
# ---------------------------------------------------------------------------
# Tell pytest-asyncio to auto-detect async tests in this module
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config() -> AppConfig:
    cfg = AppConfig(
        environment=Environment.DEMO,
        accounts=[AccountConfig(
            name="test-acct", username="u", password="p", device_id="d",
        )],
    )
    return cfg


def make_session() -> MagicMock:
    s = MagicMock()
    s.name = "test-acct"
    return s


def make_signal(strategy: str = "RSI", side: Side = Side.BUY) -> Signal:
    return Signal(
        strategy=strategy,
        side=side,
        contracts=3,
        stop_loss_pts=10.0,
        take_profit_pts=100.0,
        max_hold_bars=5,
        signal_price=19500.0,
        reason="test",
        bar_timestamp=datetime.now(timezone.utc),
    )


async def wait_for_condition(fn, timeout: float = 2.0):
    """Poll fn() every 50 ms until True or timeout."""
    for _ in range(int(timeout / 0.05)):
        if fn():
            return True
        await asyncio.sleep(0.05)
    return False


class MockServer:
    """Minimal TCP server that simulates NinjaTrader for tests."""

    def __init__(self):
        self._server = None
        self.port: int = 0
        self._writer: asyncio.StreamWriter | None = None
        self._client_task: asyncio.Task | None = None
        # Optional callable(msg_dict, writer) invoked for each message received
        self.on_message = None
        self.received: list[dict] = []

    async def start(self):
        self._server = await asyncio.start_server(
            self._accept, "127.0.0.1", 0,
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._writer = writer
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode().strip()
                if not line:
                    continue
                msg = json.loads(line)
                self.received.append(msg)
                if self.on_message:
                    await self.on_message(msg, writer)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def send(self, msg: dict):
        if self._writer:
            self._writer.write((json.dumps(msg) + "\n").encode())
            await self._writer.drain()

    async def stop(self):
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def make_bridge(port: int) -> NinjaTraderBridge:
    """Create and connect a bridge to a mock server."""
    bridge = NinjaTraderBridge(make_config(), make_session(), "127.0.0.1", port)
    await bridge.connect()
    ok = await wait_for_condition(lambda: bridge.connected)
    assert ok, "Bridge did not connect within 2s"
    return bridge


async def teardown(bridge: NinjaTraderBridge, server: MockServer):
    """Cleanly tear down bridge and server."""
    try:
        await asyncio.wait_for(bridge.shutdown(), timeout=2.0)
    except asyncio.TimeoutError:
        pass
    await server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_connects_to_server():
    """Bridge connects to the mock NT server."""
    server = MockServer()
    await server.start()
    bridge = await make_bridge(server.port)
    assert bridge.connected
    await teardown(bridge, server)


async def test_place_entry_success():
    """place_entry_with_bracket waits for fill and returns a filled OrderRecord."""
    server = MockServer()
    await server.start()

    async def auto_fill(msg, writer):
        if msg.get("cmd") == "ENTRY":
            resp = {
                "type": "fill", "id": msg["id"],
                "strategy": msg["strategy"], "side": msg["side"],
                "qty": msg["qty"], "fill_price": 19502.5,
                "sl_price": 19492.5, "tp_price": 19602.5,
            }
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()

    server.on_message = auto_fill
    bridge = await make_bridge(server.port)

    record = await bridge.place_entry_with_bracket(make_signal(), contracts=3)

    assert record.status == OrderStatus.FILLED
    assert record.fill_price == pytest.approx(19502.5)
    assert record.bracket is not None
    assert record.bracket.stop_price  == pytest.approx(19492.5)
    assert record.bracket.limit_price == pytest.approx(19602.5)
    assert record.qty == 3

    await teardown(bridge, server)


async def test_place_entry_timeout():
    """place_entry_with_bracket returns WORKING when NT never responds."""
    server = MockServer()
    await server.start()
    # No handler → NT never sends fill
    bridge = await make_bridge(server.port)
    bridge.FILL_TIMEOUT = 0.2   # Short timeout for speed

    record = await bridge.place_entry_with_bracket(make_signal(), contracts=3)

    assert record.status == OrderStatus.WORKING
    assert record.fill_price is None

    await teardown(bridge, server)


async def test_place_entry_nt_error():
    """place_entry_with_bracket marks record REJECTED on NT error."""
    server = MockServer()
    await server.start()

    async def send_error(msg, writer):
        if msg.get("cmd") == "ENTRY":
            err = {"type": "error", "id": msg["id"], "message": "Strategy not found"}
            writer.write((json.dumps(err) + "\n").encode())
            await writer.drain()

    server.on_message = send_error
    bridge = await make_bridge(server.port)

    record = await bridge.place_entry_with_bracket(make_signal(), contracts=3)

    assert record.status == OrderStatus.REJECTED

    await teardown(bridge, server)


async def test_exit_callback_sl():
    """SL message from NT triggers exit_callback with exit_type='SL'."""
    server = MockServer()
    await server.start()

    async def fill_then_sl(msg, writer):
        if msg.get("cmd") == "ENTRY":
            writer.write((json.dumps({
                "type": "fill", "id": msg["id"], "strategy": msg["strategy"],
                "side": "Buy", "qty": 3, "fill_price": 19502.0,
                "sl_price": 19492.0, "tp_price": 19602.0,
            }) + "\n").encode())
            await writer.drain()
            await asyncio.sleep(0.05)
            writer.write((json.dumps({
                "type": "exit", "strategy": msg["strategy"],
                "exit_type": "SL", "fill_price": 19492.0, "qty": 3,
            }) + "\n").encode())
            await writer.drain()

    server.on_message = fill_then_sl
    bridge = await make_bridge(server.port)

    exits = []
    async def on_exit(record, exit_type, fill_price):
        exits.append((exit_type, fill_price))
    bridge.set_exit_callback(on_exit)

    await bridge.place_entry_with_bracket(make_signal(), contracts=3)
    got_exit = await wait_for_condition(lambda: len(exits) > 0)
    assert got_exit
    assert exits[0] == ("SL", pytest.approx(19492.0))

    await teardown(bridge, server)


async def test_exit_callback_tp():
    """TP message from NT triggers exit_callback with exit_type='TP'."""
    server = MockServer()
    await server.start()

    async def fill_then_tp(msg, writer):
        if msg.get("cmd") == "ENTRY":
            writer.write((json.dumps({
                "type": "fill", "id": msg["id"], "strategy": msg["strategy"],
                "side": "Buy", "qty": 3, "fill_price": 19502.0,
                "sl_price": 19492.0, "tp_price": 19602.0,
            }) + "\n").encode())
            await writer.drain()
            await asyncio.sleep(0.05)
            writer.write((json.dumps({
                "type": "exit", "strategy": msg["strategy"],
                "exit_type": "TP", "fill_price": 19602.0, "qty": 3,
            }) + "\n").encode())
            await writer.drain()

    server.on_message = fill_then_tp
    bridge = await make_bridge(server.port)

    exits = []
    async def on_exit(record, exit_type, fill_price):
        exits.append((exit_type, fill_price))
    bridge.set_exit_callback(on_exit)

    await bridge.place_entry_with_bracket(make_signal(), contracts=3)
    await wait_for_condition(lambda: len(exits) > 0)

    assert exits[0][0] == "TP"
    assert exits[0][1] == pytest.approx(19602.0)

    await teardown(bridge, server)


async def test_flatten_position_sends_flatten():
    """flatten_position sends FLATTEN command with correct strategy."""
    server = MockServer()
    await server.start()
    bridge = await make_bridge(server.port)

    result = await bridge.flatten_position("RSI")
    await wait_for_condition(lambda: any(m.get("cmd") == "FLATTEN" for m in server.received))

    assert result is True
    flat_msg = next(m for m in server.received if m.get("cmd") == "FLATTEN")
    assert flat_msg["strategy"] == "RSI"

    await teardown(bridge, server)


async def test_liquidate_all_sends_flatten_all():
    """liquidate_all sends FLATTEN_ALL command."""
    server = MockServer()
    await server.start()
    bridge = await make_bridge(server.port)

    await bridge.liquidate_all()
    ok = await wait_for_condition(lambda: any(m.get("cmd") == "FLATTEN_ALL" for m in server.received))

    assert ok
    await teardown(bridge, server)


async def test_reconnects_after_disconnect():
    """Bridge reconnects automatically when the server drops the connection."""
    server = MockServer()
    await server.start()
    bridge = NinjaTraderBridge(make_config(), make_session(), "127.0.0.1", server.port)
    bridge.RECONNECT_DELAY = 0.2
    await bridge.connect()
    await wait_for_condition(lambda: bridge.connected)

    # Force-close the client from the server side
    if server._writer:
        try:
            server._writer.close()
        except Exception:
            pass

    dropped = await wait_for_condition(lambda: not bridge.connected, timeout=2.0)
    assert dropped, "Bridge should detect disconnect"

    # Bridge will reconnect automatically since server is still listening
    reconnected = await wait_for_condition(lambda: bridge.connected, timeout=3.0)
    assert reconnected, "Bridge should reconnect"

    await teardown(bridge, server)


async def test_on_fill_event_is_noop():
    """on_fill_event returns None — fills arrive via TCP, not external callback."""
    bridge = NinjaTraderBridge(make_config(), make_session())
    result = await bridge.on_fill_event(12345, 19500.0, 3)
    assert result is None


async def test_clear_strategy():
    """clear_strategy removes the entry from strategy_orders."""
    bridge = NinjaTraderBridge(make_config(), make_session())
    bridge.strategy_orders["RSI"] = OrderRecord(strategy="RSI", status=OrderStatus.FILLED)
    bridge.clear_strategy("RSI")
    assert "RSI" not in bridge.strategy_orders


async def test_market_callback_receives_market_and_bar_messages():
    """Market/history TCP messages are forwarded to the registered callback."""
    bridge = NinjaTraderBridge(make_config(), make_session())
    callback = AsyncMock()
    bridge.set_market_callback(callback)

    market_msg = {
        "type": "market",
        "timestamp": "2026-04-02T14:45:00-04:00",
        "price": 19510.25,
        "volume": 7,
    }
    bar_msg = {
        "type": "bar",
        "timestamp": "2026-04-02T14:30:00-04:00",
        "open": 19500.0,
        "high": 19525.0,
        "low": 19495.0,
        "close": 19510.25,
        "volume": 1234,
    }

    await bridge._dispatch(market_msg)
    await bridge._dispatch(bar_msg)

    assert callback.await_count == 2
    callback.assert_any_await(market_msg)
    callback.assert_any_await(bar_msg)
