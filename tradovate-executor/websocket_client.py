"""
WebSocket Client — handles Tradovate WebSocket protocol.
Supports both market data and order/fill WebSocket connections.
Tradovate WS protocol: frames are `\n`-delimited, first char indicates type:
  'o' = open, 'h' = heartbeat, 'a' = data array (JSON), 'c' = close
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Any

import websockets

logger = logging.getLogger(__name__)

# Tradovate WS frame types
FRAME_OPEN = "o"
FRAME_HEARTBEAT = "h"
FRAME_DATA = "a"
FRAME_CLOSE = "c"


class TradovateWebSocket:
    """
    Persistent WebSocket connection to Tradovate with:
    - Auto-reconnection with exponential backoff
    - Heartbeat monitoring
    - Tradovate frame protocol parsing
    - Auth on connect
    """

    HEARTBEAT_INTERVAL = 2.5      # Send heartbeat every 2.5s
    HEARTBEAT_TIMEOUT = 10.0      # Reconnect if no response in 10s
    MAX_RECONNECT_DELAY = 30.0
    BASE_RECONNECT_DELAY = 1.0

    def __init__(
        self,
        url: str,
        access_token: str,
        name: str = "ws",
        on_message: Optional[Callable] = None,
    ):
        self.url = url
        self.access_token = access_token
        self.name = name
        self.on_message = on_message  # async callback(data: dict)

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._last_heartbeat = 0.0
        self._tasks: list[asyncio.Task] = []

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self):
        """Start the connection loop."""
        self._running = True
        self._tasks.append(asyncio.create_task(self._connection_loop()))

    async def _connection_loop(self):
        """Main loop — connect, listen, reconnect on failure."""
        while self._running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=None,   # We handle heartbeats ourselves
                    close_timeout=5,
                    max_size=2**22,       # 4MB max message
                ) as ws:
                    self._ws = ws
                    logger.info(f"[{self.name}] WebSocket connected to {self.url}")

                    # Wait for open frame
                    first = await asyncio.wait_for(ws.recv(), timeout=10)
                    if not first.startswith(FRAME_OPEN):
                        logger.warning(f"[{self.name}] Expected 'o' frame, got: {first[:50]}")

                    # Authenticate
                    await self._authorize()

                    self._connected = True
                    self._reconnect_count = 0
                    self._last_heartbeat = time.time()

                    # Start heartbeat sender
                    hb_task = asyncio.create_task(self._heartbeat_loop())

                    try:
                        async for raw in ws:
                            await self._handle_frame(raw)
                    finally:
                        hb_task.cancel()
                        self._connected = False

            except (websockets.ConnectionClosed, ConnectionError, asyncio.TimeoutError) as e:
                self._connected = False
                logger.warning(f"[{self.name}] Connection lost: {e}")

            except Exception as e:
                self._connected = False
                logger.error(f"[{self.name}] Unexpected error: {e}", exc_info=True)

            if self._running:
                delay = min(
                    self.BASE_RECONNECT_DELAY * (2 ** self._reconnect_count),
                    self.MAX_RECONNECT_DELAY,
                )
                self._reconnect_count += 1
                logger.info(f"[{self.name}] Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})")
                await asyncio.sleep(delay)

    async def _authorize(self):
        """Send auth message after WS open."""
        auth_msg = f"authorize\n0\n\n{self.access_token}"
        await self._ws.send(auth_msg)
        logger.debug(f"[{self.name}] Auth sent")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        while self._running and self._connected:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if self._ws and self._connected:
                    await self._ws.send("[]")
                    self._last_heartbeat = time.time()
            except Exception:
                break

    async def _handle_frame(self, raw: str):
        """Parse a Tradovate WebSocket frame."""
        if not raw:
            return

        frame_type = raw[0]

        if frame_type == FRAME_HEARTBEAT:
            self._last_heartbeat = time.time()
            return

        if frame_type == FRAME_DATA:
            try:
                payload = json.loads(raw[1:])
                if isinstance(payload, list):
                    for item in payload:
                        await self._dispatch(item)
                else:
                    await self._dispatch(payload)
            except json.JSONDecodeError:
                logger.warning(f"[{self.name}] Invalid JSON: {raw[:200]}")
            return

        if frame_type == FRAME_CLOSE:
            logger.info(f"[{self.name}] Server sent close frame")
            return

        # Unknown frame type — just log
        logger.debug(f"[{self.name}] Unknown frame: {raw[:100]}")

    async def _dispatch(self, data: Any):
        """Forward parsed message to callback.
        Tradovate sends two kinds of messages with a "d" key:
        1. Subscription response confirmations: {"s": status, "i": requestId, "d": payload}
           → Unwrap to just data["d"] so handlers get the actual data.
        2. Event push frames: {"e": "fill", "d": {...fill_data...}}
           → Pass the FULL dict so handlers can read data["e"] for the event type.
        """
        if self.on_message:
            try:
                if isinstance(data, dict) and "d" in data:
                    if "e" in data:
                        # Event push frame — pass full dict so callback can read "e"
                        await self.on_message(data)
                    else:
                        # Subscription response — unwrap the "d" payload
                        await self.on_message(data["d"])
                else:
                    await self.on_message(data)
            except Exception as e:
                logger.error(f"[{self.name}] Message handler error: {e}", exc_info=True)

    async def send(self, endpoint: str, body: dict = None, query_id: int = 0):
        """
        Send a request over WebSocket.
        Tradovate WS request format: endpoint\nid\n\nbody_json
        """
        if not self._ws or not self._connected:
            raise RuntimeError(f"[{self.name}] Not connected")

        body_str = json.dumps(body) if body else ""
        msg = f"{endpoint}\n{query_id}\n\n{body_str}"
        await self._ws.send(msg)

    async def subscribe(self, endpoint: str, body: dict = None):
        """Subscribe to a Tradovate data stream."""
        await self.send(endpoint, body)

    def update_token(self, new_token: str):
        """Update token for next reconnection."""
        self.access_token = new_token

    async def disconnect(self):
        """Gracefully close the connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info(f"[{self.name}] Disconnected")
