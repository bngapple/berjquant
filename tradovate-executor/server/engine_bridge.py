"""
Engine Bridge — wraps TradovateExecutor for the FastAPI dashboard.
Runs the real trading engine in-process, pushes events to frontend via asyncio.Queue.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

# Add project root to path so we can import engine modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, Environment, AccountConfig, SizingMode, POINT_VALUE
from app import TradovateExecutor, setup_logging
from market_data import MarketState
from signal_engine import Signal, Side
from server import config_store

logger = logging.getLogger(__name__)
ET = ZoneInfo("US/Eastern")


class DashboardExecutor(TradovateExecutor):
    """
    Extends TradovateExecutor to push events to the dashboard event bus.
    Overrides key methods to intercept signals, fills, exits, and P&L changes.
    """

    def __init__(self, config: AppConfig, event_queue: asyncio.Queue):
        super().__init__(config)
        self._event_queue = event_queue

    def _push(self, event: dict):
        """Non-blocking push to event queue."""
        event.setdefault("timestamp", datetime.now(ET).isoformat())
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest if queue is full

    async def _on_bar_complete(self, state: MarketState):
        """Override: push bar event, then run parent logic (which generates signals)."""
        bar = state.last_bar
        if bar:
            self._push({
                "type": "bar",
                "data": {
                    "timestamp": bar.timestamp.isoformat() if bar.timestamp else "",
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "rsi": round(state.rsi_5, 2) if state.rsi_5 is not None else None,
                    "atr": round(state.atr_14, 2) if state.atr_14 is not None else None,
                    "ema": round(state.ema_21, 2) if state.ema_21 is not None else None,
                },
            })

        # Run parent — this evaluates signals and executes pending ones
        await super()._on_bar_complete(state)

        # Push P&L after any trades
        self._push_pnl()
        self._push_positions()

    async def _execute_signal(self, signal: Signal, open_price: float):
        """Override: push signal event, run parent, push fill event."""
        # Push signal
        self._push({
            "type": "signal",
            "data": {
                "strategy": signal.strategy,
                "side": signal.side.value,
                "contracts": signal.contracts,
                "price": signal.signal_price,
                "reason": signal.reason,
            },
        })

        # Run parent execution
        await super()._execute_signal(signal, open_price)

        # Check if fill happened
        record = self.master_executor.strategy_orders.get(signal.strategy) if self.master_executor else None
        if record and record.fill_price:
            self._push({
                "type": "fill",
                "data": {
                    "strategy": signal.strategy,
                    "side": signal.side.value,
                    "price": record.fill_price,
                    "contracts": record.qty,
                    "slippage": round(record.slippage_pts, 2),
                    "account": record.account_name,
                },
            })

        # Push updated P&L and positions
        self._push_pnl()
        self._push_positions()

    async def _handle_fill(self, fill_data: dict):
        """Override: run parent, push exit event if SL/TP."""
        result = await self.master_executor.on_fill_event(
            fill_data.get("orderId", 0),
            float(fill_data.get("price", 0)),
            int(fill_data.get("qty", 0)),
        ) if self.master_executor else None

        if result is None:
            return

        record, exit_type = result

        if exit_type in ("SL", "TP"):
            fill_price = float(fill_data.get("price", 0))
            entry_price = record.fill_price or 0
            if record.side == "Buy":
                pnl = (fill_price - entry_price) * POINT_VALUE * record.qty
            else:
                pnl = (entry_price - fill_price) * POINT_VALUE * record.qty

            self._push({
                "type": "exit",
                "data": {
                    "strategy": record.strategy,
                    "side": record.side,
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_type,
                    "bars_held": self.signal_engine.positions.get(
                        record.strategy, type("", (), {"bars_held": 0})()
                    ).bars_held,
                    "contracts": record.qty,
                },
            })

            # Update risk manager
            self.risk_manager.record_trade_pnl(pnl, record.strategy)

            # Log exit
            self.trade_log.log_exit(
                strategy=record.strategy,
                account=record.account_name,
                exit_price=fill_price,
                exit_reason=exit_type,
                bars_held=self.signal_engine.positions.get(
                    record.strategy, type("", (), {"bars_held": 0})()
                ).bars_held,
                daily_pnl=self.risk_manager.daily_pnl,
                monthly_pnl=self.risk_manager.monthly_pnl,
            )

            self.signal_engine.mark_flat(record.strategy)
            self.master_executor.clear_strategy(record.strategy)

            if self.copy_engine:
                await self.copy_engine.copy_flatten(record.strategy)

            self._save_state()

        self._push_pnl()
        self._push_positions()

    async def _handle_flatten_signal(self, signal: Signal):
        """Override: push exit event for max-hold flatten."""
        self._push({
            "type": "exit",
            "data": {
                "strategy": signal.strategy,
                "side": signal.side.value,
                "exit_reason": "MaxHold",
                "pnl": 0,
                "bars_held": 0,
                "contracts": 0,
            },
        })
        await super()._handle_flatten_signal(signal)
        self._push_pnl()
        self._push_positions()

    def _push_pnl(self):
        """Push current P&L state."""
        self._push({
            "type": "pnl",
            "data": {
                "daily": round(self.risk_manager.daily_pnl, 2),
                "monthly": round(self.risk_manager.monthly_pnl, 2),
                "daily_limit": self.risk_manager.cfg.daily_loss_limit,
                "monthly_limit": self.risk_manager.cfg.monthly_loss_limit,
            },
        })

    def _push_positions(self):
        """Push current position state for all strategies."""
        if not self.master_executor:
            return
        for strategy in ["RSI", "IB", "MOM"]:
            pos = self.signal_engine.positions.get(strategy)
            rec = self.master_executor.strategy_orders.get(strategy) if self.master_executor else None
            if pos and not pos.is_flat and rec and rec.fill_price:
                self._push({
                    "type": "position",
                    "data": {
                        "strategy": strategy,
                        "side": pos.side.value if pos.side else "Buy",
                        "entry_price": rec.fill_price,
                        "current_price": rec.fill_price,  # Updated by market data
                        "contracts": rec.qty,
                        "pnl": 0.0,  # Calculated from current price
                        "bars_held": pos.bars_held,
                        "sl": rec.bracket.stop_price if rec.bracket else 0,
                        "tp": rec.bracket.limit_price if rec.bracket else 0,
                    },
                })


class EngineBridge:
    """
    Manages the DashboardExecutor lifecycle for the FastAPI server.
    Provides start/stop/flatten/status and an event queue for WebSocket broadcast.
    """

    def __init__(self):
        self.executor: Optional[DashboardExecutor] = None
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.executor is not None and self._task is not None and not self._task.done()

    async def start(self) -> dict:
        """Start the real trading engine."""
        if self.running:
            return {"error": "Engine already running"}

        self._error = None
        config = self._build_config()

        if not config.master_account:
            return {"error": "No master account configured"}

        self.executor = DashboardExecutor(config, self.event_queue)
        self._start_time = time.time()

        # Start engine in background task
        self._task = asyncio.create_task(self._run_engine())
        # Start periodic status pusher
        self._status_task = asyncio.create_task(self._status_loop())

        return {"status": "started"}

    async def _run_engine(self):
        """Run the executor, catch errors."""
        try:
            await self.executor.start()
        except Exception as e:
            self._error = str(e)
            logger.error(f"Engine error: {e}", exc_info=True)
            self.executor = None

    async def _status_loop(self):
        """Push status every 5 seconds while engine is running."""
        while self.running:
            try:
                self.event_queue.put_nowait({
                    "type": "status",
                    "data": self.get_status(),
                    "timestamp": datetime.now(ET).isoformat(),
                })
            except asyncio.QueueFull:
                pass
            await asyncio.sleep(5)

    async def stop(self) -> dict:
        """Stop the trading engine gracefully."""
        if not self.running:
            return {"error": "Engine not running"}

        if self._status_task:
            self._status_task.cancel()
            self._status_task = None

        if self.executor:
            await self.executor.shutdown()
            self.executor = None

        if self._task:
            self._task.cancel()
            self._task = None

        self._start_time = None
        return {"status": "stopped"}

    async def flatten(self) -> dict:
        """Flatten all positions."""
        if self.executor:
            await self.executor._flatten_all()
            return {"status": "flattened"}
        return {"status": "no_engine"}

    def get_status(self) -> dict:
        """Return current engine status."""
        if not self.executor or not self.running:
            accounts = config_store.get_accounts()
            return {
                "running": False,
                "can_trade": False,
                "daily_pnl": 0.0,
                "monthly_pnl": 0.0,
                "daily_limit": -3000.0,
                "monthly_limit": -4500.0,
                "daily_limit_hit": False,
                "monthly_limit_hit": False,
                "positions": {"RSI": None, "IB": None, "MOM": None},
                "pending_signals": 0,
                "connected_accounts": [
                    {"name": a["name"], "connected": False} for a in accounts
                ] if accounts else [],
                "error": self._error,
            }

        ex = self.executor
        risk = ex.risk_manager.get_status()

        # Build positions dict
        positions = {}
        for strategy in ["RSI", "IB", "MOM"]:
            pos = ex.signal_engine.positions.get(strategy)
            rec = ex.master_executor.strategy_orders.get(strategy) if ex.master_executor else None
            if pos and not pos.is_flat and rec and rec.fill_price:
                positions[strategy] = {
                    "strategy": strategy,
                    "side": pos.side.value if pos.side else "Buy",
                    "entry_price": rec.fill_price,
                    "current_price": rec.fill_price,
                    "contracts": rec.qty,
                    "pnl": 0.0,
                    "bars_held": pos.bars_held,
                    "sl": rec.bracket.stop_price if rec.bracket else 0,
                    "tp": rec.bracket.limit_price if rec.bracket else 0,
                }
            else:
                positions[strategy] = None

        # Account connections
        connected_accounts = []
        if ex.auth:
            for session in ex.auth.sessions.values():
                connected_accounts.append({
                    "name": session.name,
                    "connected": session.is_authenticated,
                })

        return {
            "running": True,
            "can_trade": risk["can_trade"],
            "daily_pnl": round(risk["daily_pnl"], 2),
            "monthly_pnl": round(risk["monthly_pnl"], 2),
            "daily_limit": ex.risk_manager.cfg.daily_loss_limit,
            "monthly_limit": ex.risk_manager.cfg.monthly_loss_limit,
            "daily_limit_hit": risk["daily_limit_hit"],
            "monthly_limit_hit": risk["monthly_limit_hit"],
            "positions": positions,
            "pending_signals": len(ex._pending_signals),
            "connected_accounts": connected_accounts,
        }

    def get_health(self) -> dict:
        """Health check endpoint data."""
        return {
            "uptime_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "engine_running": self.running,
            "error": self._error,
            "event_queue_size": self.event_queue.qsize(),
            "ws_clients": 0,  # Filled in by api.py
        }

    def _build_config(self) -> AppConfig:
        """Build AppConfig from the config store (dashboard config.json)."""
        raw = config_store.load_config()
        env = Environment(raw.get("environment", "demo"))

        accounts = []
        for a in raw.get("accounts", []):
            accounts.append(AccountConfig(
                name=a["name"],
                username=a["username"],
                password=a["password"],
                device_id=a.get("device_id", f"device-{a['name']}"),
                app_id=a.get("app_id", "HTFSwing"),
                app_version=a.get("app_version", "1.0.0"),
                cid=a.get("cid", 0),
                sec=a.get("sec", ""),
                is_master=a.get("is_master", False),
                sizing_mode=SizingMode(a.get("sizing_mode", "mirror")),
                account_size=a.get("account_size", 150000.0),
                fixed_sizes=a.get("fixed_sizes", {"RSI": 3, "IB": 3, "MOM": 3}),
            ))

        config = AppConfig(
            environment=env,
            symbol=raw.get("symbol", "MNQM6"),
            accounts=accounts,
        )
        return config


# Singleton bridge instance
bridge = EngineBridge()
