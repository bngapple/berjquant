"""
Order Executor — places orders on a single Tradovate account.
Handles: market entries, bracket (SL + TP) via separate OCO orders,
fill-based bracket correction, position flattening.
All orders carry isAutomated: true per CME requirement.

Tradovate bracket approach:
  1. Place market entry order → get orderId
  2. Poll for fill confirmation with actual fill price
  3. Place SL (Stop) + TP (Limit) at correct prices based on fill
  4. Link SL + TP as OCO so one cancels the other
  5. Track bracket order IDs for cancel on max-hold / EOD
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

from auth_manager import AuthSession
from signal_engine import Signal, Side
from config import AppConfig, TICK_SIZE, POINT_VALUE

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    WORKING = "working"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BracketOrders:
    """Tracks the SL + TP bracket pair for one position."""
    stop_order_id: Optional[int] = None
    limit_order_id: Optional[int] = None
    oco_id: Optional[int] = None
    stop_price: float = 0.0
    limit_price: float = 0.0
    is_active: bool = False


@dataclass
class OrderRecord:
    """Tracks one entry order through its lifecycle."""
    order_id: Optional[int] = None
    strategy: str = ""
    side: str = ""                          # "Buy" or "Sell"
    qty: int = 0
    order_type: str = "Market"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[float] = None
    signal_price: float = 0.0
    signal: Optional[Signal] = None
    bracket: Optional[BracketOrders] = None
    account_name: str = ""

    @property
    def slippage_pts(self) -> float:
        if self.fill_price is not None:
            return abs(self.fill_price - self.signal_price)
        return 0.0


class OrderExecutor:
    """
    Executes orders on one Tradovate account via REST API.
    Each account (master + copies) gets its own executor instance.
    """

    FILL_WAIT_TIMEOUT = 30.0
    FILL_POLL_INTERVAL = 0.5

    def __init__(self, config: AppConfig, session: AuthSession):
        self.config = config
        self.session = session
        self._http = httpx.AsyncClient(timeout=10.0)
        self.orders: dict[int, OrderRecord] = {}
        self.strategy_orders: dict[str, OrderRecord] = {}

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.session.access_token}",
            "Content-Type": "application/json",
        }

    @property
    def _base(self) -> str:
        return self.config.rest_url

    # ------------------------------------------------------------------
    # Entry + Bracket Flow
    # ------------------------------------------------------------------
    async def place_entry_with_bracket(
        self,
        signal: Signal,
        contracts: int,
    ) -> Optional[OrderRecord]:
        """
        Full entry flow:
          1. Place market order
          2. Poll for fill
          3. Place bracket at actual fill price
        """
        action = signal.side.value

        record = OrderRecord(
            strategy=signal.strategy,
            side=action,
            qty=contracts,
            signal_price=signal.signal_price,
            signal=signal,
            account_name=self.session.name,
        )

        # Step 1: Market entry
        entry_payload = {
            "accountSpec": self.session.account_config.username,
            "accountId": self.session.tradovate_account_id,
            "action": action,
            "symbol": self.config.symbol,
            "orderQty": contracts,
            "orderType": "Market",
            "isAutomated": True,
        }

        try:
            resp = await self._http.post(
                f"{self._base}/order/placeorder",
                json=entry_payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()

            if "orderId" not in data:
                logger.error(f"[{self.session.name}] Entry rejected: {data}")
                record.status = OrderStatus.REJECTED
                return record

            record.order_id = data["orderId"]
            record.status = OrderStatus.WORKING
            self.orders[record.order_id] = record
            self.strategy_orders[signal.strategy] = record

            logger.info(
                f"[{self.session.name}][{signal.strategy}] "
                f"Entry placed: {action} {contracts} @ Market — orderId={record.order_id}"
            )

        except Exception as e:
            logger.error(f"[{self.session.name}] Entry order failed: {e}")
            record.status = OrderStatus.REJECTED
            return record

        # Step 2: Wait for fill
        fill_price = await self._wait_for_fill(record)

        if fill_price is not None:
            record.fill_price = fill_price
            record.fill_time = time.time()
            record.status = OrderStatus.FILLED
            logger.info(
                f"[{self.session.name}][{signal.strategy}] "
                f"FILLED @ {fill_price:.2f} | slippage: {record.slippage_pts:.2f} pts"
            )
            # Step 3: Bracket at real fill price
            await self._place_bracket_at_fill(signal, contracts, fill_price, record)
        else:
            logger.error(
                f"[{self.session.name}][{signal.strategy}] "
                f"Fill not confirmed within {self.FILL_WAIT_TIMEOUT}s — "
                f"placing bracket at signal price as fallback"
            )
            await self._place_bracket_at_fill(
                signal, contracts, signal.signal_price, record
            )

        return record

    async def _wait_for_fill(self, record: OrderRecord) -> Optional[float]:
        """Poll order status until filled or timeout."""
        deadline = time.time() + self.FILL_WAIT_TIMEOUT

        while time.time() < deadline:
            try:
                resp = await self._http.get(
                    f"{self._base}/order/item?id={record.order_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                order_data = resp.json()
                status = order_data.get("ordStatus", "")

                if status == "Filled":
                    avg_price = order_data.get("avgPx")
                    if avg_price:
                        return float(avg_price)
                    return await self._get_fill_price(record.order_id)

                if status in ("Rejected", "Cancelled"):
                    logger.error(
                        f"[{self.session.name}] Order {record.order_id} "
                        f"status: {status} — {order_data.get('text', '')}"
                    )
                    record.status = OrderStatus.REJECTED
                    return None

            except Exception as e:
                logger.debug(f"Fill poll error: {e}")

            await asyncio.sleep(self.FILL_POLL_INTERVAL)

        return None

    async def _get_fill_price(self, order_id: int) -> Optional[float]:
        """Query fill details for an order."""
        try:
            resp = await self._http.get(
                f"{self._base}/fill/list",
                headers=self._headers,
            )
            resp.raise_for_status()
            fills = resp.json()
            for fill in fills:
                if fill.get("orderId") == order_id:
                    return float(fill.get("price", 0))
        except Exception as e:
            logger.debug(f"Fill query error: {e}")
        return None

    # ------------------------------------------------------------------
    # Bracket Placement
    # ------------------------------------------------------------------
    async def _place_bracket_at_fill(
        self,
        signal: Signal,
        contracts: int,
        fill_price: float,
        record: OrderRecord,
    ):
        """Place SL + TP at actual fill price, then link as OCO."""
        exit_action = "Sell" if signal.side == Side.BUY else "Buy"

        if signal.side == Side.BUY:
            sl_price = self._round_price(fill_price - signal.stop_loss_pts)
            tp_price = self._round_price(fill_price + signal.take_profit_pts)
        else:
            sl_price = self._round_price(fill_price + signal.stop_loss_pts)
            tp_price = self._round_price(fill_price - signal.take_profit_pts)

        bracket = BracketOrders(stop_price=sl_price, limit_price=tp_price)

        try:
            # Place Stop Loss
            sl_resp = await self._http.post(
                f"{self._base}/order/placeorder",
                json={
                    "accountSpec": self.session.account_config.username,
                    "accountId": self.session.tradovate_account_id,
                    "action": exit_action,
                    "symbol": self.config.symbol,
                    "orderQty": contracts,
                    "orderType": "Stop",
                    "stopPrice": sl_price,
                    "isAutomated": True,
                },
                headers=self._headers,
            )
            sl_resp.raise_for_status()
            bracket.stop_order_id = sl_resp.json().get("orderId")

            # Place Take Profit
            tp_resp = await self._http.post(
                f"{self._base}/order/placeorder",
                json={
                    "accountSpec": self.session.account_config.username,
                    "accountId": self.session.tradovate_account_id,
                    "action": exit_action,
                    "symbol": self.config.symbol,
                    "orderQty": contracts,
                    "orderType": "Limit",
                    "price": tp_price,
                    "isAutomated": True,
                },
                headers=self._headers,
            )
            tp_resp.raise_for_status()
            bracket.limit_order_id = tp_resp.json().get("orderId")

            bracket.is_active = True
            record.bracket = bracket

            logger.info(
                f"[{self.session.name}][{signal.strategy}] "
                f"Bracket @ fill {fill_price:.2f}: "
                f"SL={sl_price:.2f} (id={bracket.stop_order_id}) "
                f"TP={tp_price:.2f} (id={bracket.limit_order_id})"
            )

            # Link as OCO
            await self._link_oco(bracket)

        except Exception as e:
            logger.error(
                f"[{self.session.name}][{signal.strategy}] "
                f"Bracket FAILED: {e} — MANUAL INTERVENTION NEEDED"
            )

    async def _link_oco(self, bracket: BracketOrders):
        """Link SL + TP as OCO so one cancels the other."""
        if not bracket.stop_order_id or not bracket.limit_order_id:
            return
        try:
            resp = await self._http.post(
                f"{self._base}/orderStrategy/startorderstrategy",
                json={
                    "accountId": self.session.tradovate_account_id,
                    "orderStrategyTypeId": 2,
                    "params": json.dumps({
                        "orderIds": [bracket.stop_order_id, bracket.limit_order_id]
                    }),
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            bracket.oco_id = resp.json().get("id")
            logger.debug(f"[{self.session.name}] OCO linked: {bracket.stop_order_id} ↔ {bracket.limit_order_id}")
        except Exception as e:
            logger.warning(
                f"[{self.session.name}] OCO link failed: {e} — "
                f"brackets are independent, manual cancel needed on fill"
            )

    # ------------------------------------------------------------------
    # Fill Callback (from WebSocket)
    # ------------------------------------------------------------------
    async def on_fill_event(
        self,
        order_id: int,
        fill_price: float,
        fill_qty: int,
    ) -> Optional[tuple[OrderRecord, str]]:
        """
        Called when a fill arrives via WebSocket.
        Returns (record, exit_type) where exit_type is "SL", "TP", or "entry".
        """
        # Check if it's an entry fill
        record = self.orders.get(order_id)
        if record:
            record.fill_price = fill_price
            record.fill_time = time.time()
            record.status = OrderStatus.FILLED
            return (record, "entry")

        # Check bracket fills
        for strat, rec in self.strategy_orders.items():
            if rec.bracket:
                if order_id == rec.bracket.stop_order_id:
                    logger.info(f"[{self.session.name}][{strat}] STOP LOSS HIT @ {fill_price:.2f}")
                    rec.bracket.is_active = False
                    if rec.bracket.limit_order_id:
                        await self._cancel_order(rec.bracket.limit_order_id)
                    return (rec, "SL")

                if order_id == rec.bracket.limit_order_id:
                    logger.info(f"[{self.session.name}][{strat}] TAKE PROFIT HIT @ {fill_price:.2f}")
                    rec.bracket.is_active = False
                    if rec.bracket.stop_order_id:
                        await self._cancel_order(rec.bracket.stop_order_id)
                    return (rec, "TP")

        return None

    async def _cancel_order(self, order_id: int) -> bool:
        """Cancel a single order by ID."""
        try:
            resp = await self._http.post(
                f"{self._base}/order/cancelorder",
                json={"orderId": order_id, "isAutomated": True},
                headers=self._headers,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[{self.session.name}] Cancel {order_id} failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Cancel Bracket
    # ------------------------------------------------------------------
    async def cancel_bracket(self, strategy: str) -> bool:
        """Cancel SL + TP for a strategy (for max-hold / EOD flatten)."""
        record = self.strategy_orders.get(strategy)
        if not record or not record.bracket or not record.bracket.is_active:
            return True
        bracket = record.bracket
        success = True
        if bracket.stop_order_id:
            if not await self._cancel_order(bracket.stop_order_id):
                success = False
        if bracket.limit_order_id:
            if not await self._cancel_order(bracket.limit_order_id):
                success = False
        bracket.is_active = False
        return success

    # ------------------------------------------------------------------
    # Flatten / Liquidate
    # ------------------------------------------------------------------
    async def flatten_position(self, strategy: str = "") -> bool:
        """Cancel brackets, then liquidate position."""
        if strategy:
            await self.cancel_bracket(strategy)
        else:
            for strat in list(self.strategy_orders.keys()):
                await self.cancel_bracket(strat)
        try:
            resp = await self._http.post(
                f"{self._base}/order/liquidateposition",
                json={
                    "accountId": self.session.tradovate_account_id,
                    "symbol": self.config.symbol,
                    "isAutomated": True,
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            logger.info(f"[{self.session.name}] Flattened" + (f" ({strategy})" if strategy else ""))
            return True
        except Exception as e:
            logger.error(f"[{self.session.name}] Flatten FAILED: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all working orders."""
        try:
            resp = await self._http.get(f"{self._base}/order/list", headers=self._headers)
            resp.raise_for_status()
            orders = resp.json()
            tasks = [
                self._cancel_order(o["id"])
                for o in orders
                if o.get("ordStatus") in ("Working", "Accepted")
            ]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                ok = sum(1 for r in results if r is True)
                logger.info(f"[{self.session.name}] Cancelled {ok}/{len(tasks)} orders")
            return True
        except Exception as e:
            logger.error(f"[{self.session.name}] Cancel all failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Position Sync (reconnect recovery)
    # ------------------------------------------------------------------
    async def get_current_position(self) -> Optional[dict]:
        """Query current position for our symbol."""
        try:
            resp = await self._http.get(f"{self._base}/position/list", headers=self._headers)
            resp.raise_for_status()
            for pos in resp.json():
                if pos.get("accountId") == self.session.tradovate_account_id:
                    if pos.get("netPos", 0) != 0:
                        return pos
        except Exception as e:
            logger.error(f"[{self.session.name}] Position query failed: {e}")
        return None

    async def get_working_orders(self) -> list[dict]:
        """Query all working orders."""
        try:
            resp = await self._http.get(f"{self._base}/order/list", headers=self._headers)
            resp.raise_for_status()
            return [
                o for o in resp.json()
                if o.get("ordStatus") in ("Working", "Accepted")
                and o.get("accountId") == self.session.tradovate_account_id
            ]
        except Exception as e:
            logger.error(f"[{self.session.name}] Order list failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _round_price(price: float) -> float:
        return round(price / TICK_SIZE) * TICK_SIZE

    def clear_strategy(self, strategy: str):
        self.strategy_orders.pop(strategy, None)

    async def shutdown(self):
        await self._http.aclose()
