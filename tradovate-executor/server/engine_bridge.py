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

from config import (
    AppConfig,
    Environment,
    AccountConfig,
    SizingMode,
    SessionConfig,
    NTConfig,
    NTAccountConfig,
    POINT_VALUE,
    RSIParams,
    IBParams,
    MOMParams,
)
from app import TradovateExecutor, setup_logging
from market_data import MarketState
from order_executor import OrderRecord
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
        self._bars_buffer: list[dict] = []  # Ring buffer of last 200 15m bars for chart history
        self._last_price: float = 0.0       # Latest tick price for live unrealized P&L

    def _push(self, event: dict):
        """Non-blocking push to event queue."""
        event.setdefault("timestamp", datetime.now(ET).isoformat())
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest if queue is full

    async def _on_market_message(self, data: dict):
        """Override: track latest price for live P&L, buffer bars, then run parent."""
        # Track latest tick price for unrealized P&L calculation
        entries = data.get("entries", {})
        tick_price = entries.get("Trade", {}).get("price")
        if tick_price is not None:
            self._last_price = float(tick_price)
            # Push live position P&L on every tick if positions are open
            if self.master_executor and any(
                not p.is_flat for p in self.signal_engine.positions.values()
            ):
                self._push_positions()

        bars = data.get("bars", [])
        for bar_data in bars:
            o = bar_data.get("open", 0)
            h = bar_data.get("high", 0)
            l = bar_data.get("low", 0)
            c = bar_data.get("close", 0)
            if o > 0 and h > 0 and l > 0 and c > 0:
                ts = self._parse_timestamp(bar_data.get("timestamp"))
                v = bar_data.get("upVolume", 0) + bar_data.get("downVolume", 0)
                self._bars_buffer.append({
                    "timestamp": ts.isoformat(),
                    "open": float(o), "high": float(h),
                    "low": float(l), "close": float(c),
                    "volume": int(v),
                    "rsi": None, "atr": None, "ema": None,
                })
        if len(self._bars_buffer) > 200:
            self._bars_buffer = self._bars_buffer[-200:]
        await super()._on_market_message(data)

    async def _on_bar_complete(self, state: MarketState):
        """Override: push bar event, buffer it, then run parent logic (which generates signals)."""
        bar = state.last_bar
        if bar:
            bar_dict = {
                "timestamp": bar.timestamp.isoformat() if bar.timestamp else "",
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "rsi": round(state.rsi_5, 2) if state.rsi_5 is not None else None,
                "atr": round(state.atr_14, 2) if state.atr_14 is not None else None,
                "ema": round(state.ema_21, 2) if state.ema_21 is not None else None,
            }
            self._push({"type": "bar", "data": bar_dict})
            # Update buffer: replace same-timestamp bar or append
            idx = next((i for i, b in enumerate(self._bars_buffer)
                        if b["timestamp"] == bar_dict["timestamp"]), -1)
            if idx >= 0:
                self._bars_buffer[idx] = bar_dict
            else:
                self._bars_buffer.append(bar_dict)
                if len(self._bars_buffer) > 200:
                    self._bars_buffer = self._bars_buffer[-200:]

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
        order_id = fill_data.get("orderId")
        fill_price = fill_data.get("price", 0)
        if not order_id or not fill_price:
            return
        result = await self.master_executor.on_fill_event(
            int(order_id),
            float(fill_price),
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
        """Override: push exit event for max-hold flatten with actual P&L."""
        strategy = signal.strategy
        rec = self.master_executor.strategy_orders.get(strategy) if self.master_executor else None
        pos = self.signal_engine.positions.get(strategy)
        exit_price = self._last_price or (rec.fill_price if rec else 0)

        pnl = 0.0
        if rec and rec.fill_price and exit_price:
            if rec.side == "Buy":
                pnl = (exit_price - rec.fill_price) * POINT_VALUE * rec.qty
            else:
                pnl = (rec.fill_price - exit_price) * POINT_VALUE * rec.qty

        self._push({
            "type": "exit",
            "data": {
                "strategy": strategy,
                "side": signal.side.value,
                "entry_price": rec.fill_price if rec else 0,
                "exit_price": exit_price,
                "exit_reason": "MaxHold",
                "pnl": round(pnl, 2),
                "bars_held": pos.bars_held if pos else 0,
                "contracts": rec.qty if rec else 0,
            },
        })
        await super()._handle_flatten_signal(signal)
        self._push_pnl()
        self._push_positions()

    async def _on_nt_exit(self, record: OrderRecord, exit_type: str, fill_price: float):
        """Override: push exit event to dashboard before running parent handler."""
        entry_price = record.fill_price or 0.0
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

        await super()._on_nt_exit(record, exit_type, fill_price)
        self._push_pnl()
        self._push_positions()

    def _push_pnl(self):
        """Push current P&L state."""
        self._push({
            "type": "pnl",
            "data": {
                "daily": round(self.risk_manager.daily_pnl, 2),
                "monthly": round(self.risk_manager.monthly_pnl, 2),
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
                # Use latest tick price for unrealized P&L; fall back to entry price
                current_px = self._last_price if self._last_price > 0 else rec.fill_price
                side = pos.side.value if pos.side else "Buy"
                if side == "Buy":
                    unrealized = (current_px - rec.fill_price) * POINT_VALUE * rec.qty
                else:
                    unrealized = (rec.fill_price - current_px) * POINT_VALUE * rec.qty
                self._push({
                    "type": "position",
                    "data": {
                        "strategy": strategy,
                        "side": side,
                        "entry_price": rec.fill_price,
                        "current_price": current_px,
                        "contracts": rec.qty,
                        "pnl": round(unrealized, 2),
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

        nt_only = bool(config.nt and not config.accounts)
        if not config.accounts and not nt_only:
            return {"error": "No accounts configured"}

        if config.accounts and not config.master_account:
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
            cfg = self._build_config()
            accounts = config_store.get_accounts()
            connected_accounts = [
                {"name": a["name"], "connected": False} for a in accounts
            ] if accounts else []
            if cfg.nt:
                connected_accounts.extend(
                    {
                        "name": f"NinjaTrader ({acct.host}:{acct.port})",
                        "connected": False,
                    }
                    for acct in cfg.nt.accounts.values()
                )
            return {
                "running": False,
                "can_trade": False,
                "daily_pnl": 0.0,
                "monthly_pnl": 0.0,
                "monthly_limit": cfg.session.monthly_loss_limit,
                "monthly_limit_hit": False,
                "positions": {"RSI": None, "IB": None, "MOM": None},
                "pending_signals": 0,
                "connected_accounts": connected_accounts,
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
                current_px = ex._last_price if ex._last_price > 0 else rec.fill_price
                side = pos.side.value if pos.side else "Buy"
                if side == "Buy":
                    unrealized = (current_px - rec.fill_price) * POINT_VALUE * rec.qty
                else:
                    unrealized = (rec.fill_price - current_px) * POINT_VALUE * rec.qty
                positions[strategy] = {
                    "strategy": strategy,
                    "side": side,
                    "entry_price": rec.fill_price,
                    "current_price": current_px,
                    "contracts": rec.qty,
                    "pnl": round(unrealized, 2),
                    "bars_held": pos.bars_held,
                    "sl": rec.bracket.stop_price if rec.bracket else 0,
                    "tp": rec.bracket.limit_price if rec.bracket else 0,
                }
            else:
                positions[strategy] = None

        # Account connections — include NT bridge status when in NT mode
        connected_accounts = []
        if ex.config.nt and ex.master_executor:
            from ninjatrader_bridge import NinjaTraderBridge
            nt = ex.master_executor
            if isinstance(nt, NinjaTraderBridge):
                connected_accounts.append({
                    "name": f"NinjaTrader ({nt._nt_host}:{nt._nt_port})",
                    "connected": nt.connected,
                })
        elif ex.auth:
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
            "monthly_limit": ex.risk_manager.cfg.monthly_loss_limit,
            "monthly_limit_hit": risk["monthly_limit_hit"],
            "positions": positions,
            "pending_signals": len(ex._pending_signals),
            "connected_accounts": connected_accounts,
        }

    def get_bars(self) -> list[dict]:
        """Return buffered OHLCV bar history for the chart."""
        if self.executor:
            return list(self.executor._bars_buffer)
        return []

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
                cid=a.get("cid", 8),
                sec=a.get("sec", "9c4e7db2-0e37-4169-915c-2a8fc0571dc2"),
                is_master=a.get("is_master", False),
                sizing_mode=SizingMode(a.get("sizing_mode", "scaled")),
                account_size=a.get("account_size", 150000.0),
                fixed_sizes=a.get("fixed_sizes", {"RSI": 3, "IB": 3, "MOM": 3}),
                min_contracts=a.get("min_contracts", 1),
                monthly_loss_limit=a.get("monthly_loss_limit", -4500.0),
            ))

        config = AppConfig(
            environment=env,
            symbol=raw.get("symbol", "MNQM6"),
            session=SessionConfig(**raw.get("session", {})),
            rsi=RSIParams(**raw.get("rsi", {})),
            ib=IBParams(**raw.get("ib", {})),
            mom=MOMParams(**raw.get("mom", {})),
            accounts=accounts,
        )
        nt_raw = raw.get("ninjatrader") or raw.get("nt")
        if nt_raw:
            if "accounts" in nt_raw:
                accounts_dict = {}
                for name, acct_data in nt_raw["accounts"].items():
                    if name.startswith("_"):
                        continue
                    accounts_dict[name] = NTAccountConfig(
                        host=acct_data.get("host", "127.0.0.1"),
                        port=acct_data.get("port", 6000),
                    )
                config.nt = NTConfig(
                    accounts=accounts_dict,
                    default_atm_template=nt_raw.get("default_atm_template", "MNQ_2R"),
                    order_timeout_seconds=nt_raw.get("order_timeout_seconds", 10),
                    status_timeout_seconds=nt_raw.get("status_timeout_seconds", 5),
                    reconnect_max_backoff_seconds=nt_raw.get("reconnect_max_backoff_seconds", 30),
                    symbol=nt_raw.get("symbol", "MNQU6"),
                )
            else:
                # Old single-host "nt" key (backward compat)
                config.nt = NTConfig(
                    accounts={"default": NTAccountConfig(
                        host=nt_raw.get("host", "127.0.0.1"),
                        port=nt_raw.get("port", 6000),
                    )},
                )
        return config


# Singleton bridge instance
bridge = EngineBridge()
