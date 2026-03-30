"""
Position Sync — recovers application state after reconnection or restart.
Queries Tradovate for existing positions and working orders, then rebuilds
the signal engine's position tracking to match reality.

This prevents:
  - Double entries (app thinks it's flat, but position exists)
  - Orphaned brackets (app doesn't know about working SL/TP orders)
  - Missed flattening (position exists but app has no record of it)
"""

import logging
from typing import Optional

from order_executor import OrderExecutor, OrderRecord, BracketOrders, OrderStatus
from signal_engine import SignalEngine, Side
from config import AppConfig

logger = logging.getLogger(__name__)


class PositionSync:
    """
    Reconciles app state with actual Tradovate account state.
    Called on startup and after reconnection.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    async def sync_account(
        self,
        executor: OrderExecutor,
        signal_engine: Optional[SignalEngine] = None,
    ) -> dict:
        """
        Query positions and orders for one account, rebuild internal state.
        Returns summary dict for logging.
        """
        account_name = executor.session.name
        summary = {
            "account": account_name,
            "position": None,
            "working_orders": 0,
            "actions_taken": [],
        }

        # 1. Query current position
        position = await executor.get_current_position()

        if position:
            net_pos = position.get("netPos", 0)
            net_price = position.get("netPrice", 0)
            summary["position"] = {
                "net": net_pos,
                "price": net_price,
                "side": "long" if net_pos > 0 else "short",
                "contracts": abs(net_pos),
            }
            logger.info(
                f"[Sync][{account_name}] Position found: "
                f"{'LONG' if net_pos > 0 else 'SHORT'} {abs(net_pos)} @ {net_price:.2f}"
            )
        else:
            logger.info(f"[Sync][{account_name}] No open position")

        # 2. Query working orders
        working = await executor.get_working_orders()
        summary["working_orders"] = len(working)

        if working:
            logger.info(f"[Sync][{account_name}] {len(working)} working orders found")
            for order in working:
                logger.debug(
                    f"  Order {order.get('id')}: {order.get('action')} "
                    f"{order.get('orderQty')} {order.get('orderType')} "
                    f"@ {order.get('price', order.get('stopPrice', '?'))}"
                )

        # 3. Reconcile state
        if position and net_pos != 0:
            await self._reconcile_with_position(
                executor, signal_engine, position, working, summary
            )
        elif not position and working:
            # No position but working orders → orphaned orders, cancel them
            logger.warning(
                f"[Sync][{account_name}] Orphaned orders found with no position — cancelling"
            )
            await executor.cancel_all_orders()
            summary["actions_taken"].append("cancelled_orphaned_orders")

        return summary

    async def _reconcile_with_position(
        self,
        executor: OrderExecutor,
        signal_engine: Optional[SignalEngine],
        position: dict,
        working_orders: list[dict],
        summary: dict,
    ):
        """
        We have an open position. Figure out which strategy it belongs to
        and ensure brackets are in place.
        """
        account_name = executor.session.name
        net_pos = position.get("netPos", 0)
        net_price = position.get("netPrice", 0)
        abs_qty = abs(net_pos)

        # Try to determine which strategy this position belongs to.
        # Heuristic: check contract count against strategy defaults.
        # With 3 contracts per strategy, a 3-lot is one strategy,
        # 6 is two, 9 is all three.
        # This is imperfect — in production you'd persist strategy state to disk.

        strategies_per_qty = {
            3: ["unknown_single"],
            6: ["unknown_double"],
            9: ["RSI", "IB", "MOM"],  # All three active
        }

        # Check if we have bracket orders (Stop + Limit)
        stops = [o for o in working_orders if o.get("orderType") == "Stop"]
        limits = [o for o in working_orders if o.get("orderType") == "Limit"]

        has_bracket = len(stops) > 0 and len(limits) > 0

        if has_bracket:
            logger.info(
                f"[Sync][{account_name}] Position has brackets: "
                f"{len(stops)} stops, {len(limits)} limits"
            )
            summary["actions_taken"].append("brackets_confirmed")

            # Rebuild bracket tracking in executor
            for stop, limit in zip(stops, limits):
                bracket = BracketOrders(
                    stop_order_id=stop.get("id"),
                    limit_order_id=limit.get("id"),
                    stop_price=stop.get("stopPrice", 0),
                    limit_price=limit.get("price", 0),
                    is_active=True,
                )
                # We don't know the strategy, but track the bracket
                record = OrderRecord(
                    strategy="SYNCED",
                    side="Buy" if net_pos > 0 else "Sell",
                    qty=abs_qty,
                    status=OrderStatus.FILLED,
                    fill_price=net_price,
                    bracket=bracket,
                    account_name=account_name,
                )
                executor.strategy_orders["SYNCED"] = record
        else:
            # Position exists but no brackets → dangerous state
            logger.warning(
                f"[Sync][{account_name}] Position WITHOUT brackets! "
                f"{'LONG' if net_pos > 0 else 'SHORT'} {abs_qty} @ {net_price:.2f}"
            )
            summary["actions_taken"].append("position_without_brackets")
            # Options: flatten immediately or add brackets
            # For safety, flatten the unprotected position
            logger.warning(f"[Sync][{account_name}] Flattening unprotected position for safety")
            await executor.flatten_position()
            summary["actions_taken"].append("flattened_unprotected")

        # Update signal engine position state if available
        if signal_engine and position and net_pos != 0:
            # Mark that we have an active position
            # We can't know the exact strategy, so mark a generic one
            side = Side.BUY if net_pos > 0 else Side.SELL
            if has_bracket:
                signal_engine.mark_filled("SYNCED", side)
                summary["actions_taken"].append("signal_engine_synced")

    async def sync_all(
        self,
        executors: dict[str, OrderExecutor],
        signal_engine: Optional[SignalEngine] = None,
    ) -> list[dict]:
        """Sync all accounts and return summaries."""
        summaries = []
        for name, executor in executors.items():
            summary = await self.sync_account(executor, signal_engine)
            summaries.append(summary)
        return summaries


class StateFile:
    """
    Persists strategy-to-position mapping to disk for crash recovery.
    Simple JSON file updated on every entry/exit.
    """

    def __init__(self, path: str = "state.json"):
        self.path = path

    def save(self, positions: dict):
        """
        Save current position state.
        positions = {
            "RSI": {"side": "Buy", "fill_price": 20100.25, "qty": 3, "entry_bar": 42},
            "IB": None,
            "MOM": None,
        }
        """
        import json
        with open(self.path, "w") as f:
            json.dump(positions, f, indent=2)

    def load(self) -> dict:
        """Load saved state, or return empty dict."""
        import json
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def clear(self):
        """Clear state file."""
        import os
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
