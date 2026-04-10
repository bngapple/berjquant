"""
HTF Swing v3 Hybrid v2 — Tradovate Auto-Executor
Main application: wires all modules together and runs the trading loop.

Lifecycle:
  1. Load config → auth all accounts
  2. Sync positions (recover from restart)
  3. Connect market data WS (ticks → 15m bars → indicators)
  4. Connect order WS (fill notifications → bracket management → P&L)
  5. Signal engine evaluates on each bar close
  6. Signals queue → execute at next bar open
  7. Copy engine mirrors fills to all linked accounts
  8. Risk manager enforces P&L limits + EOD flatten
  9. Ctrl+C → flatten everything → exit

Usage:
    python app.py              # Uses config.json environment
    python app.py --demo       # Force demo mode
    python app.py --live       # Force live mode
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from config import AppConfig, AccountConfig, Environment, POINT_VALUE
from auth_manager import AuthManager, AuthSession
from websocket_client import TradovateWebSocket
from market_data import MarketDataEngine, MarketState
from signal_engine import SignalEngine, Signal, Side
from order_executor import OrderExecutor, OrderRecord
from ninjatrader_bridge import NinjaTraderBridge
from copy_engine import CopyEngine
from risk_manager import RiskManager
from trade_logger import TradeLogger, TradeEntry
from position_sync import PositionSync, StateFile

ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, f"executor_{datetime.now().strftime('%Y%m%d')}.log")
            ),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class TradovateExecutor:
    """Main application class — orchestrates all modules."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.auth = AuthManager(config)
        self.trade_log = TradeLogger(config.log_dir)
        self.state_file = StateFile()

        self.signal_engine = SignalEngine(
            rsi_params=config.rsi,
            ib_params=config.ib,
            mom_params=config.mom,
            session=config.session,
        )

        self.market_data = MarketDataEngine(on_bar_complete=self._on_bar_complete)

        self.risk_manager = RiskManager(
            session_config=config.session,
            on_flatten_all=self._flatten_all,
        )

        self.master_executor: Optional[OrderExecutor] = None
        self.copy_engine: Optional[CopyEngine] = None
        self.ws_market: Optional[TradovateWebSocket] = None
        self.ws_orders: Optional[TradovateWebSocket] = None

        # Signals generated on bar N, executed on bar N+1 open
        self._pending_signals: list[Signal] = []
        self._shutdown_event = asyncio.Event()
        self._master_session: Optional[AuthSession] = None

    # ==================================================================
    # STARTUP
    # ==================================================================
    async def start(self):
        logger.info("=" * 60)
        logger.info("HTF Swing v3 Hybrid v2 — Tradovate Executor")
        logger.info(f"Environment: {self.config.environment.value.upper()}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Accounts: {len(self.config.accounts)} configured")
        logger.info("=" * 60)

        # 1. Authenticate, or synthesize a local session for NT-only mode.
        if self._is_nt_only_mode():
            logger.info("NT-only mode detected — skipping Tradovate authentication")
            master_session = self._build_nt_only_master_session()
            if not master_session:
                logger.critical("NT-only startup FAILED — could not build master session")
                return
        else:
            logger.info("Authenticating all accounts...")
            await self.auth.authenticate_all()

            master_session = self.auth.get_master_session()
            if not master_session or not master_session.is_authenticated:
                logger.critical("Master account auth FAILED — cannot start")
                return

        self._master_session = master_session

        # 2. Initialize executors (NT bridge or Tradovate REST, depending on config)
        if self.config.nt:
            nt_acct = self._resolve_nt_account(master_session.name)
            if not nt_acct:
                return

            if not self._validate_nt_account(master_session.name, nt_acct):
                return

            logger.info(
                f"NT mode: routing execution through NinjaTrader at "
                f"{nt_acct.host}:{nt_acct.port} for account '{master_session.name}'"
            )
            self.master_executor = NinjaTraderBridge(
                self.config,
                master_session,
                nt_host=nt_acct.host,
                nt_port=nt_acct.port,
            )
            self.master_executor.set_exit_callback(self._on_nt_exit)
            self.master_executor.set_market_callback(self._on_nt_market_message)
            await self.master_executor.connect()

            nt_timeout = max(5.0, float(self.config.nt.status_timeout_seconds))
            if not await self._wait_nt_connected(timeout=nt_timeout):
                await self.master_executor.shutdown()
                self.master_executor = None
                return

            self.copy_engine = None
            copy_count = 0
        else:
            self.master_executor = OrderExecutor(self.config, master_session)
            self.copy_engine = CopyEngine(self.config, self.auth)
            await self.copy_engine.initialize()
            copy_count = len(self.auth.get_copy_sessions())

        logger.info(f"Master: {master_session.name} | Copies: {copy_count}")

        # 3. Position sync (recover from restart)
        await self._sync_positions()

        # 4. Connect market data (Tradovate WS or NT TCP, depending on mode)
        if not self._is_nt_only_mode():
            self.ws_market = TradovateWebSocket(
                url=self.config.ws_market_url,
                access_token=master_session.md_access_token or master_session.access_token,
                name="md",
                on_message=self._on_market_message,
                on_reconnect=self._on_md_reconnect,
            )
            await self.ws_market.connect()
            await self._wait_connected(self.ws_market, "Market data")

            # Subscribe to quotes
            await self.ws_market.subscribe(
                "md/subscribeQuote",
                {"symbol": self.config.symbol},
            )
            # Also subscribe to chart for historical bar seeding
            await self.ws_market.subscribe(
                "md/getChart",
                {
                    "symbol": self.config.symbol,
                    "chartDescription": {
                        "underlyingType": "MinuteBar",
                        "elementSize": 15,
                        "elementSizeUnit": "UnderlyingUnits",
                        "withHistogram": False,
                    },
                    "timeRange": {
                        "asFarAsTimestamp": datetime.now(ET).strftime("%Y-%m-%dT00:00:00"),
                    },
                },
            )
            logger.info(f"Subscribed to: {self.config.symbol}")
        else:
            logger.info(
                f"Waiting for market data from NinjaTrader on {self.config.symbol} "
                f"(attach PythonBridge to a 15-minute chart for historical warmup)"
            )

        # 5. Connect order/fill WebSocket (Tradovate mode only; NT mode uses TCP bridge)
        if not self.config.nt:
            self.ws_orders = TradovateWebSocket(
                url=self.config.ws_orders_url,
                access_token=master_session.access_token,
                name="orders",
                on_message=self._on_order_message,
                on_reconnect=self._on_orders_reconnect,
            )
            await self.ws_orders.connect()
            await self._wait_connected(self.ws_orders, "Orders")

            await self.ws_orders.subscribe("user/syncrequest", {
                "users": [master_session.user_id],
            })

        # Wire token renewal → propagate fresh tokens to market data WS
        self.auth.on_token_renewed = self._on_token_renewed

        # 6. Start risk manager
        self.risk_manager.start_eod_timer()

        logger.info("=" * 60)
        logger.info("EXECUTOR RUNNING — waiting for signals")
        logger.info("=" * 60)

        await self._shutdown_event.wait()

    async def _wait_connected(self, ws: TradovateWebSocket, name: str, timeout: float = 5.0):
        """Wait for a WebSocket to connect."""
        for _ in range(int(timeout / 0.1)):
            if ws.connected:
                logger.info(f"{name} WebSocket connected")
                return
            await asyncio.sleep(0.1)
        logger.warning(f"{name} WebSocket connection timeout — will retry in background")

    @staticmethod
    def _is_placeholder_nt_value(value: Optional[str]) -> bool:
        return not value or str(value).strip().startswith("REPLACE_")

    def _is_nt_only_mode(self) -> bool:
        """True when NinjaTrader is configured and no Tradovate accounts are present."""
        return bool(self.config.nt and not self.config.accounts)

    def _build_nt_only_master_session(self) -> Optional[AuthSession]:
        """Build a synthetic local session for true NT-only operation."""
        if not self.config.nt or not self.config.nt.accounts:
            return None

        account_name = next(iter(self.config.nt.accounts.keys()))
        account_config = AccountConfig(
            name=account_name,
            username="",
            password="",
            device_id="nt-only",
            is_master=True,
        )
        return AuthSession(
            account_config=account_config,
            access_token="nt-only",
            md_access_token="nt-only",
            is_authenticated=True,
        )

    def _get_master_session(self) -> Optional[AuthSession]:
        """Return the active master session for both Tradovate and NT-only modes."""
        return self._master_session or self.auth.get_master_session()

    def _resolve_nt_account(self, master_name: str):
        """Find the exact NinjaTrader account mapping for the authenticated master account."""
        if not self.config.nt or not self.config.nt.accounts:
            logger.error("No NinjaTrader accounts configured in ninjatrader config — cannot start")
            return None

        nt_acct = self.config.nt.accounts.get(master_name)
        if nt_acct:
            return nt_acct

        configured_names = list(self.config.nt.accounts.keys())
        if any(self._is_placeholder_nt_value(name) for name in configured_names):
            logger.error(
                "NinjaTrader config still contains placeholder account names. "
                "Replace the REPLACE_* entries in config.json with the exact case-sensitive "
                "names from the NinjaTrader Accounts tab before starting."
            )
        else:
            logger.error(
                f"NT account '{master_name}' not found in ninjatrader config. "
                f"Exact case-sensitive match required. Configured names: {configured_names}"
            )
        return None

    def _validate_nt_account(self, account_name: str, nt_acct) -> bool:
        """Reject placeholder or obviously invalid NinjaTrader bridge targets."""
        if self._is_placeholder_nt_value(account_name) or self._is_placeholder_nt_value(nt_acct.host):
            logger.error(
                "NinjaTrader config for '%s' still contains placeholder values. "
                "Update the exact account name and VM IPv4 host in config.json before starting.",
                account_name,
            )
            return False

        if nt_acct.port <= 0:
            logger.error(
                "NinjaTrader config for '%s' has invalid port %s. "
                "Set the port to the PythonBridge TcpPort value in config.json.",
                account_name,
                nt_acct.port,
            )
            return False

        return True

    async def _wait_nt_connected(self, timeout: float = 5.0) -> bool:
        """Wait for the NinjaTrader TCP bridge to come online before continuing startup."""
        for _ in range(int(timeout / 0.1)):
            if self.master_executor and getattr(self.master_executor, "connected", False):
                logger.info("NinjaTrader bridge connected")
                return True
            await asyncio.sleep(0.1)

        logger.error(
            "NinjaTrader bridge connection timeout after %.1fs — startup aborted. "
            "Verify NinjaTrader is running, PythonBridge is active on a chart, "
            "Windows Firewall allows the port, and config.json matches the local host/port "
            "(127.0.0.1 on the same PC, or the VM IPv4 if using a VM).",
            timeout,
        )
        return False

    async def _sync_positions(self):
        """Check for existing positions from a previous session."""
        syncer = PositionSync(self.config)
        saved_state = self.state_file.load()  # Load crash-recovery state for strategy identification
        if saved_state:
            active = [s for s, v in saved_state.items() if v is not None]
            logger.info(f"[Sync] StateFile found — active strategies: {active or 'none'}")

        all_executors = {"master": self.master_executor}
        if self.copy_engine:
            for name, executor in self.copy_engine.executors.items():
                all_executors[name] = executor

        summaries = await syncer.sync_all(all_executors, self.signal_engine, saved_state)
        for s in summaries:
            if s["actions_taken"]:
                logger.info(f"[Sync] {s['account']}: {s['actions_taken']}")

    # ==================================================================
    # MARKET DATA HANDLER
    # ==================================================================
    async def _on_market_message(self, data: dict):
        """Process incoming market data from WebSocket."""
        if not isinstance(data, dict):
            return

        # --- Quote updates ---
        entries = data.get("entries", {})
        trade_entry = entries.get("Trade", {})
        price = trade_entry.get("price")
        volume = trade_entry.get("size", 0)

        if price is not None:
            ts = self._parse_timestamp(data.get("timestamp"))
            await self.market_data.on_tick(float(price), int(volume), ts)
            return

        # --- Chart/bar data (historical seeding) ---
        bars = data.get("bars", [])
        for bar_data in bars:
            ts = self._parse_timestamp(bar_data.get("timestamp"))
            o = bar_data.get("open", 0)
            h = bar_data.get("high", 0)
            l = bar_data.get("low", 0)
            c = bar_data.get("close", 0)
            v = bar_data.get("upVolume", 0) + bar_data.get("downVolume", 0)

            if o > 0 and h > 0 and l > 0 and c > 0:
                await self.market_data.ingest_historical_bar(
                    timestamp=ts,
                    open_=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=int(v),
                )

    async def _on_nt_market_message(self, data: dict):
        """Process market data streamed from NinjaTrader over TCP."""
        if not isinstance(data, dict):
            return

        msg_type = data.get("type")
        if msg_type == "market":
            price = data.get("price")
            volume = data.get("volume", 0)
            if price is None:
                return
            ts = self._parse_timestamp(data.get("timestamp"))
            await self.market_data.on_tick(float(price), int(volume), ts)
            return

        if msg_type == "bar":
            o = data.get("open")
            h = data.get("high")
            l = data.get("low")
            c = data.get("close")
            if None in (o, h, l, c):
                return
            ts = self._parse_timestamp(data.get("timestamp"))
            await self.market_data.ingest_historical_bar(
                timestamp=ts,
                open_=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=int(data.get("volume", 0)),
            )

    def _parse_timestamp(self, ts_str) -> datetime:
        """Parse Tradovate timestamp or fall back to now."""
        if ts_str:
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return datetime.now(ET)

    # ==================================================================
    # SIGNAL HANDLER (on 15m bar close)
    # ==================================================================
    async def _on_bar_complete(self, state: MarketState):
        """
        Called when a 15-minute bar closes.
        1. Execute pending signals from previous bar at this bar's open
        2. Evaluate strategies for new signals (queued for next bar)
        """
        # Step 1: Execute pending signals at this bar's open
        if self._pending_signals and state.current_bar:
            open_price = state.current_bar.open
            for sig in self._pending_signals:
                await self._execute_signal(sig, open_price)
            self._pending_signals.clear()

        # Step 2: Risk check
        if not self.risk_manager.can_trade():
            return

        # Step 3: Generate new signals
        new_signals = self.signal_engine.evaluate(state)
        for sig in new_signals:
            if sig.contracts == 0:
                # Flatten signal (max hold)
                await self._handle_flatten_signal(sig)
            else:
                self._pending_signals.append(sig)
                logger.info(
                    f"QUEUED: {sig.strategy} {sig.side.value} {sig.contracts} "
                    f"— execute next bar open"
                )

    async def _execute_signal(self, signal: Signal, open_price: float):
        """Execute a queued signal."""
        if not self.risk_manager.can_trade():
            logger.info(f"SKIPPED (halted): {signal.strategy} {signal.side.value}")
            return

        logger.info(
            f"EXECUTING: {signal.strategy} {signal.side.value} "
            f"{signal.contracts} @ market (open ~{open_price:.2f})"
        )

        # Master execution
        record = await self.master_executor.place_entry_with_bracket(
            signal, signal.contracts
        )

        if record and record.order_id and record.status.value != "rejected":
            self.signal_engine.mark_filled(signal.strategy, signal.side)
            if record.status.value == "working":
                logger.warning(
                    f"[{signal.strategy}] Fill unconfirmed after timeout — "
                    f"marking position entered. Bracket at signal price. Verify manually."
                )

            # Determine actual fill price
            fill_px = record.fill_price if record.fill_price else open_price

            # Compute bracket prices for logging
            if signal.side == Side.BUY:
                sl_px = fill_px - signal.stop_loss_pts
                tp_px = fill_px + signal.take_profit_pts
            else:
                sl_px = fill_px + signal.stop_loss_pts
                tp_px = fill_px - signal.take_profit_pts

            # Log entry
            entry = TradeEntry(
                strategy=signal.strategy,
                account=self._get_master_session().name,
                side=signal.side.value,
                contracts=signal.contracts,
                signal_price=signal.signal_price,
                fill_price=fill_px,
                sl_price=sl_px,
                tp_price=tp_px,
                entry_time=datetime.now(ET),
            )
            self.trade_log.log_entry(entry)

            # Save state for crash recovery
            self._save_state()

            # Copy to linked accounts
            if self.copy_engine:
                await self.copy_engine.copy_entry(signal)
        else:
            logger.error(f"Entry FAILED: {signal.strategy} — {record}")

    async def _handle_flatten_signal(self, signal: Signal):
        """Flatten one strategy (max hold reached)."""
        strategy = signal.strategy
        logger.info(f"FLATTEN: {strategy} — {signal.reason}")

        if self.master_executor:
            await self.master_executor.flatten_position(strategy)
            self.master_executor.clear_strategy(strategy)

        if self.copy_engine:
            await self.copy_engine.copy_flatten(strategy)

        self.signal_engine.mark_flat(strategy)
        self._save_state()

    # ==================================================================
    # ORDER / FILL HANDLER (from WebSocket)
    # ==================================================================
    async def _on_order_message(self, data: dict):
        """Handle order status and fill events from the order WebSocket."""
        if not isinstance(data, dict):
            return

        # Tradovate sends various event types
        event_type = data.get("e", "")

        if event_type in ("order", "order/item"):
            await self._handle_order_update(data.get("d", {}))

        elif event_type in ("fill", "fill/item"):
            await self._handle_fill(data.get("d", {}))

        elif event_type == "position":
            await self._handle_position_update(data.get("d", {}))

    async def _handle_order_update(self, order_data: dict):
        """Track order status changes."""
        order_id = order_data.get("id")
        status = order_data.get("ordStatus", "")

        if status == "Rejected":
            text = order_data.get("text", "unknown reason")
            logger.error(f"ORDER REJECTED: id={order_id} — {text}")

    async def _handle_fill(self, fill_data: dict):
        """
        Process a fill — update brackets, P&L, signal engine state.
        This is where SL/TP hits are detected.
        """
        order_id = fill_data.get("orderId")
        fill_price = fill_data.get("price", 0)
        fill_qty = fill_data.get("qty", 0)

        if not order_id or not fill_price:
            return

        result = await self.master_executor.on_fill_event(
            order_id, float(fill_price), int(fill_qty)
        )

        if result is None:
            return

        record, exit_type = result

        if exit_type == "entry":
            logger.info(f"Fill confirmed: {record.strategy} @ {fill_price}")

        elif exit_type in ("SL", "TP"):
            # This is a bracket exit — compute P&L
            strategy = record.strategy
            entry_price = record.fill_price or 0

            if record.side == "Buy":
                pnl_per_contract = (float(fill_price) - entry_price) * POINT_VALUE
            else:
                pnl_per_contract = (entry_price - float(fill_price)) * POINT_VALUE

            total_pnl = pnl_per_contract * record.qty
            bars_held = self.signal_engine.positions.get(
                strategy, type("", (), {"bars_held": 0})()
            ).bars_held

            # Update risk manager
            self.risk_manager.record_trade_pnl(total_pnl, strategy)

            # Log exit
            self.trade_log.log_exit(
                strategy=strategy,
                account=record.account_name,
                exit_price=float(fill_price),
                exit_reason=exit_type,
                bars_held=bars_held,
                daily_pnl=self.risk_manager.daily_pnl,
                monthly_pnl=self.risk_manager.monthly_pnl,
            )

            # Mark strategy flat
            self.signal_engine.mark_flat(strategy)
            self.master_executor.clear_strategy(strategy)

            # Copy flatten (SL/TP on copies should fire independently,
            # but flatten as backup in case OCO didn't propagate)
            if self.copy_engine:
                await self.copy_engine.copy_flatten(strategy)

            self._save_state()

    async def _handle_position_update(self, pos_data: dict):
        """Track position changes from Tradovate."""
        net_pos = pos_data.get("netPos", 0)
        if net_pos == 0:
            logger.debug("Position update: FLAT")

    # ==================================================================
    # NT EXIT HANDLER (NinjaTrader mode — replaces WebSocket fill path)
    # ==================================================================
    async def _on_nt_exit(self, record: OrderRecord, exit_type: str, fill_price: float):
        """
        Called by NinjaTraderBridge when NT reports a bracket exit (SL, TP, EOD, etc.).
        Mirrors the SL/TP branch of _handle_fill for the Tradovate path.
        """
        strategy    = record.strategy
        entry_price = record.fill_price or 0.0

        if record.side == "Buy":
            pnl = (fill_price - entry_price) * POINT_VALUE * record.qty
        else:
            pnl = (entry_price - fill_price) * POINT_VALUE * record.qty

        bars_held = self.signal_engine.positions.get(
            strategy, type("", (), {"bars_held": 0})()
        ).bars_held

        self.risk_manager.record_trade_pnl(pnl, strategy)

        self.trade_log.log_exit(
            strategy=strategy,
            account=record.account_name,
            exit_price=fill_price,
            exit_reason=exit_type,
            bars_held=bars_held,
            daily_pnl=self.risk_manager.daily_pnl,
            monthly_pnl=self.risk_manager.monthly_pnl,
        )

        self.signal_engine.mark_flat(strategy)
        if self.master_executor:
            self.master_executor.clear_strategy(strategy)

        self._save_state()

        logger.info(
            f"[NT] Exit logged: {strategy} {exit_type} "
            f"entry={entry_price:.2f} exit={fill_price:.2f} pnl={pnl:+.2f}"
        )

    # ==================================================================
    # RECONNECT & TOKEN RENEWAL HANDLERS
    # ==================================================================
    def _on_token_renewed(self, session):
        """Push fresh tokens to live WS clients after auth_manager renews."""
        master = self._get_master_session()
        if master and session.name == master.name:
            if self.ws_market:
                self.ws_market.update_token(session.md_access_token or session.access_token)
            if self.ws_orders:  # None in NT mode
                self.ws_orders.update_token(session.access_token)
            logger.info(f"[{session.name}] WS tokens updated after renewal")

    async def _on_md_reconnect(self):
        """Re-subscribe to market data after a WS reconnect."""
        logger.info("Market data WS reconnected — re-subscribing")
        try:
            await self.ws_market.subscribe("md/subscribeQuote", {"symbol": self.config.symbol})
            await self.ws_market.subscribe(
                "md/getChart",
                {
                    "symbol": self.config.symbol,
                    "chartDescription": {
                        "underlyingType": "MinuteBar",
                        "elementSize": 15,
                        "elementSizeUnit": "UnderlyingUnits",
                        "withHistogram": False,
                    },
                    "timeRange": {
                        "asFarAsTimestamp": datetime.now(ET).strftime("%Y-%m-%dT00:00:00"),
                    },
                },
            )
            logger.info(f"Re-subscribed to {self.config.symbol} after reconnect")
        except Exception as e:
            logger.error(f"Re-subscription failed: {e}")

    async def _on_orders_reconnect(self):
        """Re-subscribe and re-sync positions after an order WS reconnect."""
        logger.warning("Order WS reconnected — re-syncing positions and re-subscribing")
        master = self._get_master_session()
        try:
            if master:
                await self.ws_orders.subscribe("user/syncrequest", {"users": [master.user_id]})
            await self._sync_positions()
            logger.info("Position re-sync complete after order WS reconnect")
        except Exception as e:
            logger.error(f"Re-sync after reconnect failed: {e}")

    # ==================================================================
    # FLATTEN ALL
    # ==================================================================
    async def _flatten_all(self):
        """Flatten every position on every account."""
        logger.critical("=== FLATTEN ALL POSITIONS ===")

        if self.master_executor:
            await self.master_executor.cancel_all_orders()
            await self.master_executor.liquidate_all()

        if self.copy_engine:
            await self.copy_engine.cancel_all_on_copies()
            await self.copy_engine.copy_flatten()

        for strategy in list(self.signal_engine.positions.keys()):
            self.signal_engine.mark_flat(strategy)

        self._pending_signals.clear()
        self._save_state()

    # ==================================================================
    # STATE PERSISTENCE
    # ==================================================================
    def _save_state(self):
        """Save current strategy positions to disk for crash recovery."""
        positions = {}
        for name, pos in self.signal_engine.positions.items():
            if not pos.is_flat:
                rec = self.master_executor.strategy_orders.get(name)
                positions[name] = {
                    "side": pos.side.value if pos.side else None,
                    "bars_held": pos.bars_held,
                    "fill_price": rec.fill_price if rec else None,
                    "qty": rec.qty if rec else 0,
                }
            else:
                positions[name] = None
        self.state_file.save(positions)

    # ==================================================================
    # SHUTDOWN
    # ==================================================================
    async def shutdown(self):
        """Graceful shutdown: flatten → disconnect → cleanup."""
        logger.warning("SHUTDOWN initiated — flattening all positions...")

        await self._flatten_all()

        if self.ws_market:
            await self.ws_market.disconnect()
        if self.ws_orders:  # None in NT mode
            await self.ws_orders.disconnect()
        if self.master_executor:
            await self.master_executor.shutdown()
        if self.copy_engine:
            await self.copy_engine.shutdown()
        await self.auth.shutdown()
        await self.risk_manager.shutdown()

        self.state_file.clear()
        logger.info("Shutdown complete.")
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    setup_logging()

    config_path = "config.json"
    try:
        config = AppConfig.load(config_path)
    except FileNotFoundError:
        logger.info(f"No {config_path} found — creating sample config")
        config = AppConfig(environment=Environment.DEMO)
        config.save(config_path)
        logger.info(f"Edit {config_path} with your credentials, then restart.")
        sys.exit(0)

    # CLI overrides
    if "--demo" in sys.argv:
        config.environment = Environment.DEMO
    elif "--live" in sys.argv:
        config.environment = Environment.LIVE

    app = TradovateExecutor(config)
    loop = asyncio.new_event_loop()

    def handle_sig(sig, frame):
        logger.warning(f"Signal {sig} received — shutting down...")
        loop.call_soon_threadsafe(loop.create_task, app.shutdown())

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        loop.run_until_complete(app.start())
    except KeyboardInterrupt:
        # SIGINT handler already scheduled shutdown via call_soon_threadsafe;
        # avoid calling shutdown a second time here.
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
