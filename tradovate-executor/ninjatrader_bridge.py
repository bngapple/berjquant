"""
NinjaTrader Bridge — drop-in replacement for OrderExecutor.
Routes order execution through NinjaTrader 8.1 via TCP socket instead of Tradovate REST API.

Used when LucidFlex prop-firm accounts don't have Tradovate API credentials.
NinjaTrader runs on a Windows VM (e.g. VMware Fusion); Python connects to it via TCP.

Protocol: newline-delimited JSON on port 6000 (configurable).
See NinjaTrader/PythonBridge.cs for the NinjaScript counterpart.

Interface mirrors OrderExecutor exactly so DashboardExecutor requires no changes.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

from auth_manager import AuthSession
from config import AppConfig
from order_executor import BracketOrders, OrderRecord, OrderStatus
from signal_engine import Signal

logger = logging.getLogger(__name__)


class NinjaTraderBridge:
    """
    Async TCP bridge to NinjaTrader 8.1 PythonBridge strategy.
    Implements the same interface as OrderExecutor (strategy_orders, place_entry_with_bracket,
    flatten_position, cancel_all_orders, liquidate_all, clear_strategy, shutdown).

    Fills and exits arrive via TCP instead of through the Tradovate WebSocket.
    Register on_exit_callback to receive (record, exit_type, fill_price) when NT reports an exit.
    """

    FILL_TIMEOUT    = 30.0   # Seconds to wait for fill confirmation
    RECONNECT_DELAY = 5.0    # Seconds between reconnect attempts
    PING_INTERVAL   = 10.0   # Seconds between keepalive pings

    def __init__(
        self,
        config: AppConfig,
        session: AuthSession,
        nt_host: str = "127.0.0.1",
        nt_port: int = 6000,
    ):
        self.config  = config
        self.session = session
        self._nt_host = nt_host
        self._nt_port = nt_port

        # Public: mirrors OrderExecutor.strategy_orders
        self.strategy_orders: dict[str, OrderRecord] = {}

        # Pending fill futures: req_id → Future[OrderRecord]
        self._pending: dict[str, asyncio.Future] = {}

        # Callback invoked on every SL/TP/EOD exit from NT
        # Signature: async on_exit(record: OrderRecord, exit_type: str, fill_price: float)
        self._exit_callback: Optional[Callable] = None
        self._market_callback: Optional[Callable] = None

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._running   = False
        self._req_counter = 0

        self._conn_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API (OrderExecutor interface)
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    def set_exit_callback(self, callback: Callable):
        """Register async callback invoked when NT reports a bracket exit.
        Signature: async callback(record: OrderRecord, exit_type: str, fill_price: float)
        """
        self._exit_callback = callback

    def set_market_callback(self, callback: Callable):
        """Register async callback for NT market data/history messages."""
        self._market_callback = callback

    async def connect(self):
        """Start background TCP connection loop (non-blocking)."""
        self._running = True
        self._conn_task = asyncio.create_task(self._connection_loop())

    async def place_entry_with_bracket(
        self,
        signal: Signal,
        contracts: int,
    ) -> Optional[OrderRecord]:
        """
        Send ENTRY command to NinjaTrader and wait for fill confirmation.
        Returns an OrderRecord with fill_price and bracket set on success,
        or a WORKING record on timeout (assume fill happened at signal price).
        """
        req_id = self._next_req_id()

        record = OrderRecord(
            order_id=abs(hash(req_id)) % (10 ** 9),  # Synthetic ID for compatibility
            strategy=signal.strategy,
            side=signal.side.value,
            qty=contracts,
            signal_price=signal.signal_price,
            signal=signal,
            account_name=self.session.name,
            status=OrderStatus.WORKING,
        )
        self.strategy_orders[signal.strategy] = record

        # Create future before sending (race-free)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        cmd = {
            "cmd": "ENTRY",
            "id": req_id,
            "strategy": signal.strategy,
            "side": signal.side.value,
            "qty": contracts,
            "sl_pts": signal.stop_loss_pts,
            "tp_pts": signal.take_profit_pts,
        }

        try:
            await self._send(cmd)
        except Exception as e:
            logger.error(f"[NT] Failed to send ENTRY for {signal.strategy}: {e}")
            self._pending.pop(req_id, None)
            record.status = OrderStatus.REJECTED
            return record

        # Wait for fill with timeout
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=self.FILL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                f"[NT] Fill timeout for {signal.strategy} (req={req_id}) — "
                f"assuming fill at signal price {signal.signal_price:.2f}"
            )
            self._pending.pop(req_id, None)
            # Leave record as WORKING — parent will use signal_price as fallback fill
        except Exception as e:
            logger.error(f"[NT] Fill error for {signal.strategy}: {e}")
            self._pending.pop(req_id, None)
            record.status = OrderStatus.REJECTED

        return record

    async def on_fill_event(
        self,
        order_id: int,
        fill_price: float,
        fill_qty: int,
    ):
        """No-op: fills arrive via the TCP connection, not through an external callback.
        Exists only to match the OrderExecutor interface used by PositionSync.
        """
        return None

    async def flatten_position(self, strategy: str = "") -> bool:
        """Close one strategy's position (or all if strategy is empty)."""
        try:
            if strategy:
                await self._send({
                    "cmd": "FLATTEN",
                    "id": self._next_req_id(),
                    "strategy": strategy,
                })
            else:
                await self._send({"cmd": "FLATTEN_ALL", "id": self._next_req_id()})
            return True
        except Exception as e:
            logger.error(f"[NT] flatten_position({strategy!r}) failed: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Flatten all (NT manages bracket orders internally)."""
        return await self.flatten_position()

    async def liquidate_all(self) -> bool:
        """Flatten all (NT manages bracket orders internally)."""
        return await self.flatten_position()

    def clear_strategy(self, strategy: str):
        """Remove strategy record from local tracking dict."""
        self.strategy_orders.pop(strategy, None)

    async def get_current_position(self):
        """Position query is not supported via TCP bridge (NT manages internally)."""
        return None

    async def get_working_orders(self) -> list:
        """Order query is not supported via TCP bridge (NT manages internally)."""
        return []

    async def shutdown(self):
        """Cancel connection tasks and close TCP socket."""
        self._running = False
        self._connected = False

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._ping_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        if self._conn_task:
            self._conn_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._conn_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        if self._writer:
            try:
                self._writer.close()
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass

        logger.info("[NT] Bridge shutdown")

    # ------------------------------------------------------------------
    # Connection Loop
    # ------------------------------------------------------------------

    async def _connection_loop(self):
        """Persistent reconnect loop — tries to connect to NinjaTrader TCP server."""
        while self._running:
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._nt_host, self._nt_port
                )
                self._connected = True
                logger.info(
                    f"[NT] Connected to NinjaTrader at {self._nt_host}:{self._nt_port} "
                    f"(account: {self.session.name}) — "
                    f"verify this name matches the NinjaTrader Accounts tab exactly."
                )

                self._ping_task = asyncio.create_task(self._ping_loop())
                try:
                    await self._read_loop()
                finally:
                    self._ping_task.cancel()
                    self._ping_task = None
                    self._connected = False
                    # Fail all pending fill futures so place_entry_with_bracket returns promptly
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.set_exception(ConnectionError("NinjaTrader disconnected"))
                    self._pending.clear()

            except (ConnectionRefusedError, OSError) as e:
                self._connected = False
                logger.warning(
                    f"[NT] Cannot reach NinjaTrader at {self._nt_host}:{self._nt_port} ({e}). "
                    f"Check that: "
                    f"(1) NinjaTrader is running on the same Windows PC or VM, "
                    f"(2) PythonBridge strategy is active on a chart in NinjaTrader, "
                    f"(3) Windows Firewall allows inbound TCP on port {self._nt_port}, "
                    f"(4) host in config matches 127.0.0.1 for same-PC deployment "
                    f"or the VM's ipconfig IPv4 address."
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                logger.error(f"[NT] Connection error: {e}", exc_info=True)

            if self._running:
                logger.info(f"[NT] Reconnecting in {self.RECONNECT_DELAY}s…")
                await asyncio.sleep(self.RECONNECT_DELAY)

    async def _read_loop(self):
        """Read newline-delimited JSON messages from NinjaTrader."""
        while self._connected:
            try:
                raw = await self._reader.readline()
                if not raw:
                    logger.warning("[NT] Connection closed by NinjaTrader")
                    break
                line = raw.decode().strip()
                if not line:
                    continue
                msg = json.loads(line)
                await self._dispatch(msg)
            except json.JSONDecodeError as e:
                logger.warning(f"[NT] Invalid JSON from NT: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[NT] Read error: {e}", exc_info=True)
                break

    async def _ping_loop(self):
        """Send periodic PING to keep the connection alive."""
        while self._connected:
            await asyncio.sleep(self.PING_INTERVAL)
            if self._connected:
                try:
                    await self._send({"cmd": "PING"})
                except Exception:
                    break

    # ------------------------------------------------------------------
    # Message Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict):
        """Route incoming NT message to the appropriate handler."""
        msg_type = msg.get("type")

        if msg_type == "fill":
            await self._on_fill(msg)
        elif msg_type == "exit":
            await self._on_exit(msg)
        elif msg_type in ("market", "bar"):
            await self._on_market_data(msg)
        elif msg_type == "ack":
            pass  # Flatten acknowledged — nothing to do
        elif msg_type == "pong":
            pass  # Heartbeat response
        elif msg_type == "error":
            req_id = msg.get("id", "")
            error  = msg.get("message", "unknown error")
            logger.error(f"[NT] Error from NinjaTrader (req={req_id}): {error}")
            fut = self._pending.pop(req_id, None)
            if fut and not fut.done():
                fut.set_exception(RuntimeError(f"NT error: {error}"))
        else:
            logger.debug(f"[NT] Unknown message type: {msg_type}")

    async def _on_market_data(self, msg: dict):
        """Forward market data/history messages to the app."""
        if not self._market_callback:
            return
        try:
            await self._market_callback(msg)
        except Exception as e:
            logger.error(f"[NT] Market callback error: {e}", exc_info=True)

    async def _on_fill(self, msg: dict):
        """NT confirmed an entry fill — update OrderRecord and resolve the pending Future."""
        req_id     = msg.get("id", "")
        strategy   = msg.get("strategy", "")
        fill_price = float(msg.get("fill_price", 0))
        sl_price   = float(msg.get("sl_price", 0))
        tp_price   = float(msg.get("tp_price", 0))
        qty        = int(msg.get("qty", 0))

        record = self.strategy_orders.get(strategy)
        if record:
            record.fill_price = fill_price
            record.fill_time  = time.time()
            record.status     = OrderStatus.FILLED
            record.qty        = qty
            record.bracket    = BracketOrders(
                stop_price=sl_price,
                limit_price=tp_price,
                is_active=True,
            )
            logger.info(
                f"[NT] Fill: {strategy} @ {fill_price:.2f} "
                f"| SL={sl_price:.2f} TP={tp_price:.2f}"
            )

        fut = self._pending.pop(req_id, None)
        if fut and not fut.done():
            fut.set_result(record)

    async def _on_exit(self, msg: dict):
        """NT reports a bracket exit (SL, TP, EOD, or forced flatten)."""
        strategy   = msg.get("strategy", "")
        exit_type  = msg.get("exit_type", "Unknown")  # "SL", "TP", "EOD", "Command"
        fill_price = float(msg.get("fill_price", 0))

        record = self.strategy_orders.get(strategy)
        if record and record.bracket:
            record.bracket.is_active = False

        logger.info(f"[NT] Exit: {strategy} {exit_type} @ {fill_price:.2f}")

        if record and self._exit_callback:
            try:
                await self._exit_callback(record, exit_type, fill_price)
            except Exception as e:
                logger.error(f"[NT] Exit callback error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _next_req_id(self) -> str:
        self._req_counter += 1
        return f"req-{self._req_counter:06d}"

    async def _send(self, msg: dict):
        if not self._writer or not self._connected:
            raise RuntimeError("[NT] Not connected to NinjaTrader")
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        self._writer.write(data)
        await self._writer.drain()
