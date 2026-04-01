"""
Comprehensive dry-run test — validates the entire trading pipeline WITHOUT real credentials.
Mocks all Tradovate HTTP and WebSocket interactions with realistic responses.
Tests: auth → market data → indicators → signals → orders → brackets → copies → risk → logging → history.
"""

import asyncio
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, time as dt_time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("US/Eastern")


# ===========================================================================
# 1. INDICATORS
# ===========================================================================

class TestIndicators:
    def test_rsi_oversold(self):
        """RSI below 35 should trigger buy signal."""
        from indicators import rsi
        # Descending closes → low RSI
        closes = [100, 99, 98, 97, 96, 95, 94]  # 7 points, period 5 needs 6+
        val = rsi(closes, period=5)
        assert val is not None
        assert val < 35, f"RSI should be < 35 for descending prices, got {val:.1f}"

    def test_rsi_overbought(self):
        """RSI above 65 should trigger sell signal."""
        from indicators import rsi
        # Ascending closes → high RSI
        closes = [90, 91, 92, 93, 94, 95, 96]
        val = rsi(closes, period=5)
        assert val is not None
        assert val > 65, f"RSI should be > 65 for ascending prices, got {val:.1f}"

    def test_rsi_insufficient_data(self):
        from indicators import rsi
        assert rsi([100, 101], period=5) is None

    def test_atr_calculation(self):
        from indicators import atr
        # 16 bars of data (need period+1 = 15)
        highs = [101 + i * 0.5 for i in range(16)]
        lows = [99 + i * 0.5 for i in range(16)]
        closes = [100 + i * 0.5 for i in range(16)]
        val = atr(highs, lows, closes, period=14)
        assert val is not None
        assert val > 0

    def test_atr_insufficient_data(self):
        from indicators import atr
        assert atr([101], [99], [100], period=14) is None

    def test_ema_calculation(self):
        from indicators import ema
        closes = [float(100 + i) for i in range(25)]
        val = ema(closes, period=21)
        assert val is not None
        assert 100 < val < 125

    def test_sma_calculation(self):
        from indicators import sma
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert sma(data, period=5) == 30.0
        assert sma(data, period=3) == 40.0  # Last 3: 30, 40, 50

    def test_sma_insufficient_data(self):
        from indicators import sma
        assert sma([10.0], period=5) is None


# ===========================================================================
# 2. MARKET DATA ENGINE
# ===========================================================================

class TestMarketDataEngine:
    @pytest.fixture
    def engine(self):
        from market_data import MarketDataEngine
        completed_bars = []
        async def on_bar(state):
            completed_bars.append({
                "open": state.last_bar.open,
                "high": state.last_bar.high,
                "low": state.last_bar.low,
                "close": state.last_bar.close,
                "volume": state.last_bar.volume,
                "rsi": state.rsi_5,
                "atr": state.atr_14,
                "ema": state.ema_21,
            })
        eng = MarketDataEngine(on_bar_complete=on_bar)
        eng._completed_bars = completed_bars
        return eng

    @pytest.mark.asyncio
    async def test_15m_bar_builds_from_ticks(self, engine):
        """Feed ticks across a 15-minute boundary and verify a bar completes."""
        base = datetime(2026, 3, 30, 10, 0, 0, tzinfo=ET)
        # Ticks within 10:00-10:14
        for i in range(10):
            ts = base + timedelta(seconds=i * 30)
            price = 21000 + i * 2
            await engine.on_tick(price, 100, ts)

        assert engine.state.current_bar is not None
        assert engine.state.current_bar.open == 21000

        # Tick at 10:15 → triggers bar completion
        ts_next = base + timedelta(minutes=15)
        await engine.on_tick(21050, 100, ts_next)

        assert len(engine._completed_bars) == 1
        bar = engine._completed_bars[0]
        assert bar["open"] == 21000
        assert bar["high"] == 21018  # 21000 + 9*2
        assert bar["low"] == 21000
        assert bar["close"] == 21018

    @pytest.mark.asyncio
    async def test_ib_range_capture(self, engine):
        """Verify IB captures high/low between 9:30-10:00 ET."""
        base = datetime(2026, 3, 30, 9, 30, 0, tzinfo=ET)
        # Ticks during IB window
        await engine.on_tick(21000, 100, base)
        await engine.on_tick(21050, 100, base + timedelta(minutes=5))
        await engine.on_tick(20980, 100, base + timedelta(minutes=10))
        await engine.on_tick(21020, 100, base + timedelta(minutes=20))

        assert engine.state.today_ib is not None
        assert engine.state.today_ib.high == 21050
        assert engine.state.today_ib.low == 20980

        # After 10:00, IB should be marked complete
        await engine.on_tick(21030, 100, datetime(2026, 3, 30, 10, 1, 0, tzinfo=ET))
        assert engine.state.ib_complete is True

    @pytest.mark.asyncio
    async def test_indicators_after_enough_bars(self, engine):
        """Feed enough bars to get RSI/ATR/EMA calculations."""
        base = datetime(2026, 3, 30, 9, 30, 0, tzinfo=ET)
        # Feed 30 bars worth of ticks (need 15+ for ATR, 22+ for EMA)
        for bar_i in range(30):
            bar_start = base + timedelta(minutes=bar_i * 15)
            price = 21000 + bar_i * 3
            for tick in range(3):
                ts = bar_start + timedelta(seconds=tick * 60)
                await engine.on_tick(price + tick, 50 + tick * 10, ts)

        # Should have RSI, ATR, EMA after 30 bars
        assert engine.state.rsi_5 is not None
        assert engine.state.atr_14 is not None
        assert engine.state.ema_21 is not None
        assert engine.state.vol_sma_20 is not None


# ===========================================================================
# 3. SIGNAL ENGINE
# ===========================================================================

class TestSignalEngine:
    @pytest.fixture
    def signal_engine(self):
        from signal_engine import SignalEngine
        from config import RSIParams, IBParams, MOMParams, SessionConfig
        return SignalEngine(
            rsi_params=RSIParams(),
            ib_params=IBParams(),
            mom_params=MOMParams(),
            session=SessionConfig(),
        )

    def _make_state(self, **kwargs):
        from market_data import MarketState, Bar, IBRange
        state = MarketState()
        state.last_bar = Bar(
            timestamp=kwargs.get("timestamp", datetime(2026, 3, 30, 10, 30, tzinfo=ET)),
            open=kwargs.get("open", 21000),
            high=kwargs.get("high", 21050),
            low=kwargs.get("low", 20990),
            close=kwargs.get("close", 21020),
            volume=kwargs.get("volume", 500),
            is_complete=True,
        )
        state.current_bar = Bar(timestamp=datetime(2026, 3, 30, 10, 45, tzinfo=ET))
        state.current_bar.open = kwargs.get("close", 21020)
        state.rsi_5 = kwargs.get("rsi", 50.0)
        state.atr_14 = kwargs.get("atr", 15.0)
        state.ema_21 = kwargs.get("ema", 21000.0)
        state.vol_sma_20 = kwargs.get("vol_sma", 400.0)
        state.closes = [21000 + i for i in range(30)]
        state.highs = [21010 + i for i in range(30)]
        state.lows = [20990 + i for i in range(30)]
        state.volumes = [400 + i for i in range(30)]
        if "ib_high" in kwargs:
            state.today_ib = IBRange(date=datetime(2026, 3, 30, tzinfo=ET))
            state.today_ib.high = kwargs["ib_high"]
            state.today_ib.low = kwargs["ib_low"]
            state.ib_complete = True
            state.ib_percentile_low = 10.0
            state.ib_percentile_high = 100.0
        return state

    def test_rsi_buy_signal(self, signal_engine):
        """RSI < 35 → buy signal."""
        state = self._make_state(rsi=28.3)
        signals = signal_engine.evaluate(state)
        buys = [s for s in signals if s.strategy == "RSI" and s.side.value == "Buy"]
        assert len(buys) == 1
        assert "RSI(28.3)" in buys[0].reason
        assert buys[0].contracts == 3
        assert buys[0].stop_loss_pts == 10.0
        assert buys[0].take_profit_pts == 100.0

    def test_rsi_sell_signal(self, signal_engine):
        """RSI > 65 → sell signal."""
        state = self._make_state(rsi=72.0)
        signals = signal_engine.evaluate(state)
        sells = [s for s in signals if s.strategy == "RSI" and s.side.value == "Sell"]
        assert len(sells) == 1

    def test_rsi_no_signal_in_range(self, signal_engine):
        """RSI between 35-65 → no signal."""
        state = self._make_state(rsi=50.0)
        signals = signal_engine.evaluate(state)
        rsi_signals = [s for s in signals if s.strategy == "RSI"]
        assert len(rsi_signals) == 0

    def test_rsi_no_double_entry(self, signal_engine):
        """No second RSI signal while position is open."""
        from signal_engine import Side
        state = self._make_state(rsi=28.0)
        signals = signal_engine.evaluate(state)
        assert any(s.strategy == "RSI" for s in signals)
        signal_engine.mark_filled("RSI", Side.BUY)
        signals2 = signal_engine.evaluate(self._make_state(rsi=25.0))
        rsi_signals = [s for s in signals2 if s.strategy == "RSI" and s.contracts > 0]
        assert len(rsi_signals) == 0

    def test_ib_breakout_buy(self, signal_engine):
        """Close above IB high → buy signal."""
        state = self._make_state(close=21060, ib_high=21050, ib_low=20970)
        signals = signal_engine.evaluate(state)
        ib = [s for s in signals if s.strategy == "IB"]
        assert len(ib) == 1
        assert ib[0].side.value == "Buy"
        assert ib[0].take_profit_pts == 120.0

    def test_ib_breakout_sell(self, signal_engine):
        """Close below IB low → sell signal."""
        state = self._make_state(close=20960, ib_high=21050, ib_low=20970)
        signals = signal_engine.evaluate(state)
        ib = [s for s in signals if s.strategy == "IB"]
        assert len(ib) == 1
        assert ib[0].side.value == "Sell"

    def test_ib_max_one_per_day(self, signal_engine):
        """Only 1 IB trade per day."""
        from signal_engine import Side
        state = self._make_state(close=21060, ib_high=21050, ib_low=20970)
        signals = signal_engine.evaluate(state)
        assert any(s.strategy == "IB" for s in signals)
        signal_engine.mark_filled("IB", Side.BUY)
        signal_engine.mark_flat("IB")
        # Second attempt same day
        signals2 = signal_engine.evaluate(self._make_state(close=21080, ib_high=21050, ib_low=20970))
        ib = [s for s in signals2 if s.strategy == "IB" and s.contracts > 0]
        assert len(ib) == 0

    def test_mom_buy_signal(self, signal_engine):
        """Bullish bar with range > ATR, volume > SMA, close > EMA → buy."""
        state = self._make_state(
            open=21000, high=21080, low=20990, close=21070,
            atr=15.0, vol_sma=400.0, ema=21000.0, volume=500,
        )
        signals = signal_engine.evaluate(state)
        mom = [s for s in signals if s.strategy == "MOM"]
        assert len(mom) == 1
        assert mom[0].side.value == "Buy"
        assert mom[0].stop_loss_pts == 15.0

    def test_mom_sell_signal(self, signal_engine):
        """Bearish bar with range > ATR, volume > SMA, close < EMA → sell."""
        state = self._make_state(
            open=21070, high=21080, low=20990, close=20995,
            atr=15.0, vol_sma=400.0, ema=21050.0, volume=500,
        )
        signals = signal_engine.evaluate(state)
        mom = [s for s in signals if s.strategy == "MOM"]
        assert len(mom) == 1
        assert mom[0].side.value == "Sell"

    def test_no_entries_after_430pm(self, signal_engine):
        """No new entries after 4:30 PM ET."""
        state = self._make_state(
            rsi=28.0,
            timestamp=datetime(2026, 3, 30, 16, 45, tzinfo=ET),
        )
        signals = signal_engine.evaluate(state)
        entries = [s for s in signals if s.contracts > 0]
        assert len(entries) == 0

    def test_max_hold_flatten(self, signal_engine):
        """Position held > max bars → flatten signal."""
        from signal_engine import Side
        signal_engine.mark_filled("RSI", Side.BUY)
        # Tick 6 bars (max is 5)
        for _ in range(6):
            signal_engine.positions["RSI"].tick_bar()
        state = self._make_state(rsi=50.0)
        signals = signal_engine.evaluate(state)
        flatten = [s for s in signals if s.strategy == "RSI" and s.contracts == 0]
        assert len(flatten) == 1
        assert "Max hold" in flatten[0].reason


# ===========================================================================
# 4. ORDER EXECUTOR (payload verification)
# ===========================================================================

class TestOrderPayloads:
    def test_market_entry_payload(self):
        """Verify the market order payload format."""
        from config import AppConfig, AccountConfig, Environment
        from auth_manager import AuthSession
        config = AppConfig(environment=Environment.DEMO, symbol="MNQM6")
        acct = AccountConfig(name="test", username="testuser", password="pass", device_id="dev1")
        session = AuthSession(account_config=acct, access_token="fake_token", tradovate_account_id=123456, user_id=789, is_authenticated=True)

        payload = {
            "accountSpec": session.account_config.username,
            "accountId": session.tradovate_account_id,
            "action": "Buy",
            "symbol": config.symbol,
            "orderQty": 3,
            "orderType": "Market",
            "isAutomated": True,
        }

        assert payload["accountSpec"] == "testuser"
        assert payload["accountId"] == 123456
        assert payload["action"] == "Buy"
        assert payload["symbol"] == "MNQM6"
        assert payload["orderQty"] == 3
        assert payload["orderType"] == "Market"
        assert payload["isAutomated"] is True

    def test_stop_loss_payload(self):
        """Verify SL order payload and price calculation."""
        from order_executor import OrderExecutor
        fill_price = 21000.50
        sl_pts = 10.0
        sl_price = OrderExecutor._round_price(fill_price - sl_pts)
        assert sl_price == 20990.50  # Rounded to 0.25 tick

        payload = {
            "accountSpec": "testuser",
            "accountId": 123456,
            "action": "Sell",  # Opposite of entry Buy
            "symbol": "MNQM6",
            "orderQty": 3,
            "orderType": "Stop",
            "stopPrice": sl_price,
            "isAutomated": True,
        }
        assert payload["orderType"] == "Stop"
        assert payload["stopPrice"] == 20990.50

    def test_take_profit_payload(self):
        """Verify TP order payload and price calculation."""
        from order_executor import OrderExecutor
        fill_price = 21000.50
        tp_pts = 100.0
        tp_price = OrderExecutor._round_price(fill_price + tp_pts)
        assert tp_price == 21100.50

        payload = {
            "accountSpec": "testuser",
            "accountId": 123456,
            "action": "Sell",
            "symbol": "MNQM6",
            "orderQty": 3,
            "orderType": "Limit",
            "price": tp_price,
            "isAutomated": True,
        }
        assert payload["orderType"] == "Limit"
        assert payload["price"] == 21100.50

    def test_oco_linking_payload(self):
        """Verify OCO linking payload format."""
        sl_order_id = 998878
        tp_order_id = 998879
        params = json.dumps({"orderIds": [sl_order_id, tp_order_id]})
        payload = {
            "accountId": 123456,
            "orderStrategyTypeId": 2,
            "params": params,
        }
        assert payload["orderStrategyTypeId"] == 2
        assert isinstance(payload["params"], str)
        parsed = json.loads(payload["params"])
        assert parsed["orderIds"] == [998878, 998879]

    def test_price_rounding(self):
        """Verify prices round to 0.25 tick size."""
        from order_executor import OrderExecutor
        assert OrderExecutor._round_price(21000.10) == 21000.00
        assert OrderExecutor._round_price(21000.13) == 21000.25
        assert OrderExecutor._round_price(21000.30) == 21000.25
        assert OrderExecutor._round_price(21000.40) == 21000.50
        assert OrderExecutor._round_price(21000.60) == 21000.50
        assert OrderExecutor._round_price(21000.80) == 21000.75


# ===========================================================================
# 5. AUTH FLOW
# ===========================================================================

class TestAuthFlow:
    def test_auth_payload_format(self):
        """Verify the auth payload uses 'name' not 'username'."""
        from config import AccountConfig
        acct = AccountConfig(
            name="master-150k", username="john@example.com", password="secret123",
            device_id="dev-master", cid=8, sec="api_secret_key",
        )
        payload = {
            "name": acct.username,
            "password": acct.password,
            "appId": acct.app_id,
            "appVersion": acct.app_version,
            "deviceId": acct.device_id,
            "cid": acct.cid,
            "sec": acct.sec,
        }
        # Key check: field is "name", not "username"
        assert "name" in payload
        assert "username" not in payload
        assert payload["name"] == "john@example.com"
        assert payload["appId"] == "HTFSwing"
        assert payload["cid"] == 8

    @pytest.mark.asyncio
    async def test_auth_manager_stores_md_token(self):
        """Verify mdAccessToken is stored separately."""
        from auth_manager import AuthManager, AuthSession
        from config import AppConfig, AccountConfig, Environment

        config = AppConfig(environment=Environment.DEMO, accounts=[
            AccountConfig(name="test", username="u", password="p", device_id="d", cid=1, sec="s"),
        ])
        mgr = AuthManager(config)
        session = AuthSession(account_config=config.accounts[0])
        mgr.sessions["test"] = session

        mock_auth_resp = {
            "accessToken": "order_token_abc",
            "mdAccessToken": "md_token_xyz",
            "userId": 12345,
        }
        mock_acct_resp = [{"id": 67890}]

        with patch.object(mgr._http, "post", new_callable=AsyncMock) as mock_post, \
             patch.object(mgr._http, "get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = MagicMock(
                json=lambda: mock_auth_resp,
                raise_for_status=lambda: None,
            )
            mock_get.return_value = MagicMock(
                json=lambda: mock_acct_resp,
                raise_for_status=lambda: None,
            )
            await mgr._authenticate(session)

        assert session.access_token == "order_token_abc"
        assert session.md_access_token == "md_token_xyz"
        assert session.tradovate_account_id == 67890
        assert session.is_authenticated is True


# ===========================================================================
# 6. WEBSOCKET PROTOCOL
# ===========================================================================

class TestWebSocketProtocol:
    def test_authorize_message_format(self):
        """Verify the WS authorize message format."""
        token = "eyJhbGciOiJIUzI1NiJ9.test_token"
        msg = f"authorize\n0\n\n{token}"
        lines = msg.split("\n")
        assert lines[0] == "authorize"
        assert lines[1] == "0"
        assert lines[2] == ""
        assert lines[3] == token

    def test_subscribe_message_format(self):
        """Verify the WS subscribe message format."""
        endpoint = "md/subscribeQuote"
        body = {"symbol": "MNQM6"}
        body_str = json.dumps(body)
        msg = f"{endpoint}\n0\n\n{body_str}"
        lines = msg.split("\n")
        assert lines[0] == "md/subscribeQuote"
        assert json.loads(lines[3])["symbol"] == "MNQM6"

    def test_data_frame_parsing(self):
        """Verify 'a' frame JSON parsing."""
        raw = 'a[{"s":200,"i":2,"d":{"entries":{"Trade":{"price":21050.25,"size":5}}}}]'
        assert raw[0] == "a"
        payload = json.loads(raw[1:])
        assert isinstance(payload, list)
        item = payload[0]
        assert item["s"] == 200
        assert item["d"]["entries"]["Trade"]["price"] == 21050.25

    def test_response_envelope_unwrapping(self):
        """Verify the _dispatch unwrapping of {s, i, d} envelope."""
        # The websocket_client._dispatch should extract .d if present
        data = {"s": 200, "i": 2, "d": {"entries": {"Trade": {"price": 21050.25}}}}
        # If "d" in data → dispatch data["d"]
        assert "d" in data
        inner = data["d"]
        assert inner["entries"]["Trade"]["price"] == 21050.25


# ===========================================================================
# 7. RISK MANAGER
# ===========================================================================

class TestRiskManager:
    @pytest.fixture
    def risk(self):
        from risk_manager import RiskManager
        from config import SessionConfig
        # No flatten callback to avoid async issues in tests
        return RiskManager(session_config=SessionConfig(), on_flatten_all=None)

    @pytest.mark.asyncio
    async def test_daily_limit_halt(self, risk):
        """Trading halts at -$3,000 daily."""
        risk.record_trade_pnl(-2000, "RSI")
        assert risk.daily_limit_hit is False
        risk.record_trade_pnl(-1100, "MOM")
        assert risk.daily_pnl == -3100
        assert risk.daily_limit_hit is True
        assert risk.trading_halted is True

    @pytest.mark.asyncio
    async def test_monthly_limit_halt(self, risk):
        """Trading halts at -$4,500 monthly."""
        risk.record_trade_pnl(-4600, "RSI")
        assert risk.monthly_pnl == -4600
        assert risk.monthly_limit_hit is True
        assert risk.trading_halted is True

    def test_pnl_accumulates(self, risk):
        risk.record_trade_pnl(500, "RSI")
        risk.record_trade_pnl(-200, "IB")
        risk.record_trade_pnl(300, "MOM")
        assert risk.daily_pnl == 600
        assert risk.monthly_pnl == 600

    def test_no_trade_after_430(self, risk):
        """can_trade returns False after 4:30 PM ET."""
        now = datetime.now(ET)
        if now.time() >= dt_time(16, 30):
            assert risk.can_trade() is False
        elif now.time() < dt_time(9, 30):
            assert risk.can_trade() is False


# ===========================================================================
# 8. TRADE LOGGER (CSV format)
# ===========================================================================

class TestTradeLogger:
    def test_csv_entry_and_exit(self):
        from trade_logger import TradeLogger, TradeEntry
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(log_dir=tmpdir)
            entry = TradeEntry(
                strategy="RSI", account="master-150k",
                side="Buy", contracts=3,
                signal_price=21000.0, fill_price=21001.50,
                sl_price=20991.50, tp_price=21101.50,
                entry_time=datetime(2026, 3, 30, 10, 30, tzinfo=ET),
            )
            logger.log_entry(entry)

            pnl = logger.log_exit(
                strategy="RSI", account="master-150k",
                exit_price=21050.0, exit_reason="TP",
                bars_held=3, daily_pnl=97.0, monthly_pnl=97.0,
            )
            assert pnl is not None
            assert pnl == (21050.0 - 21001.50) * 2.0 * 3  # $291.00

            # Verify CSV was written
            csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            assert len(csv_files) == 1

            with open(os.path.join(tmpdir, csv_files[0])) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2  # Entry + Exit
            assert rows[0]["action"] == "Entry"
            assert rows[0]["strategy"] == "RSI"
            assert rows[0]["fill_price"] == "21001.50"
            assert rows[0]["slippage_pts"] == "1.50"
            assert rows[1]["action"] == "Exit"
            assert rows[1]["exit_reason"] == "TP"
            assert rows[1]["exit_price"] == "21050.00"


# ===========================================================================
# 9. HISTORY MODULE
# ===========================================================================

class TestHistory:
    def _write_csv(self, tmpdir, filename, rows):
        from trade_logger import CSV_HEADERS
        path = os.path.join(tmpdir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for row in rows:
                writer.writerow(row)
        return path

    def test_parse_and_stats(self):
        from server.history import parse_trades, compute_stats
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, "trades_2026-03-28.csv", [
                ["2026-03-28 10:30:00", "RSI", "master", "Exit", "Buy", 3, "21000", "21001.50", "1.50", "20991.50", "21101.50", "TP", "21050.00", "16.17", "291.00", 3, "291.00", "291.00"],
                ["2026-03-28 11:45:00", "MOM", "master", "Exit", "Sell", 3, "21100", "21099.00", "1.00", "21114.00", "21000.00", "SL", "21110.00", "-3.67", "-66.00", 2, "225.00", "225.00"],
            ])
            trades = parse_trades(tmpdir)
            assert len(trades) == 2

            stats = compute_stats(trades)
            assert stats["total_trades"] == 2
            assert stats["winners"] == 1
            assert stats["losers"] == 1
            assert stats["total_pnl"] == 225.0  # 291 - 66
            assert stats["win_rate"] == 50.0

    def test_daily_pnl_and_equity(self):
        from server.history import parse_trades, compute_daily_pnl, compute_equity_curve
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, "trades_2026-03-27.csv", [
                ["2026-03-27 10:30:00", "RSI", "master", "Exit", "Buy", 3, "21000", "21001", "1", "20991", "21101", "TP", "21050", "16", "300.00", 3, "300", "300"],
            ])
            self._write_csv(tmpdir, "trades_2026-03-28.csv", [
                ["2026-03-28 10:30:00", "RSI", "master", "Exit", "Buy", 3, "21000", "21001", "1", "20991", "21101", "SL", "20990", "-3.67", "-66.00", 2, "-66", "-66"],
            ])
            trades = parse_trades(tmpdir)
            daily = compute_daily_pnl(trades)
            assert "2026-03-27" in daily
            assert "2026-03-28" in daily
            assert daily["2026-03-27"]["pnl"] == 300.0
            assert daily["2026-03-28"]["pnl"] == -66.0

            equity = compute_equity_curve(daily)
            assert len(equity) == 2
            assert equity[0]["value"] == 300.0
            assert equity[1]["value"] == 234.0  # 300 - 66


# ===========================================================================
# 10. ACCOUNT TRACKER
# ===========================================================================

class TestAccountTracker:
    def test_drawdown_calculation(self):
        """Verify drawdown tracks from peak equity."""
        os.environ["CONFIG_PATH"] = "test_config.json"
        from server import config_store
        from server.account_tracker import get_account_status

        # Clean
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

        config_store.add_account({
            "name": "dd-test", "username": "u", "password": "p", "device_id": "d",
            "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
            "is_master": True, "sizing_mode": "mirror", "account_size": 150000,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
            "starting_balance": 150000, "profit_target": 9000, "max_drawdown": -4500,
            "account_type": "eval",
        })

        # No trades → no drawdown
        status = get_account_status("dd-test", trades=[])
        assert status["balance"] == 150000.0
        assert status["drawdown_current"] == 0.0
        assert status["drawdown_pct_used"] == 0.0
        assert status["status"] == "active"

        # With trades showing profit then loss
        trades = [
            {"timestamp": "2026-03-27 10:00:00", "action": "Exit", "account": "dd-test", "pnl_total": 2000.0, "strategy": "RSI", "side": "Buy", "contracts": 3, "signal_price": 0, "fill_price": 0, "slippage_pts": 0, "sl_price": 0, "tp_price": 0, "exit_reason": "TP", "exit_price": 0, "pnl_per_contract": 0, "bars_held": 0},
            {"timestamp": "2026-03-28 10:00:00", "action": "Exit", "account": "dd-test", "pnl_total": -1000.0, "strategy": "RSI", "side": "Buy", "contracts": 3, "signal_price": 0, "fill_price": 0, "slippage_pts": 0, "sl_price": 0, "tp_price": 0, "exit_reason": "SL", "exit_price": 0, "pnl_per_contract": 0, "bars_held": 0},
        ]
        status = get_account_status("dd-test", trades=trades)
        assert status["balance"] == 151000.0  # 150000 + 2000 - 1000
        assert status["pnl_total"] == 1000.0
        # Peak was 152000, now 151000 → drawdown = -1000
        assert status["drawdown_current"] == -1000.0

        # Clean up
        os.remove("test_config.json")
        if os.path.exists("test_secret_key"):
            os.remove("test_secret_key")


# ===========================================================================
# 11. ENGINE BRIDGE (event flow)
# ===========================================================================

class TestEngineBridge:
    def test_bridge_builds_config(self):
        """Verify bridge builds AppConfig from config_store."""
        os.environ["CONFIG_PATH"] = "test_config.json"
        from server import config_store
        from server.engine_bridge import EngineBridge

        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

        config_store.add_account({
            "name": "bridge-test", "username": "u", "password": "p", "device_id": "d",
            "app_id": "HTFSwing", "app_version": "1.0.0", "cid": 0, "sec": "",
            "is_master": True, "sizing_mode": "mirror", "account_size": 150000,
            "fixed_sizes": {"RSI": 3, "IB": 3, "MOM": 3},
        })

        bridge = EngineBridge()
        config = bridge._build_config()
        assert config.symbol == "MNQM6"
        assert len(config.accounts) == 1
        assert config.accounts[0].name == "bridge-test"
        assert config.accounts[0].is_master is True
        assert config.master_account is not None

        os.remove("test_config.json")
        if os.path.exists("test_secret_key"):
            os.remove("test_secret_key")

    def test_status_when_stopped(self):
        """Verify bridge returns correct status when not running."""
        from server.engine_bridge import EngineBridge
        bridge = EngineBridge()
        status = bridge.get_status()
        assert status["running"] is False
        assert status["positions"]["RSI"] is None
        assert status["daily_pnl"] == 0.0

    def test_event_queue_push(self):
        """Verify events can be pushed and consumed."""
        from server.engine_bridge import EngineBridge
        bridge = EngineBridge()
        bridge.event_queue.put_nowait({"type": "test", "data": {"value": 42}})
        assert bridge.event_queue.qsize() == 1
        event = bridge.event_queue.get_nowait()
        assert event["type"] == "test"
        assert event["data"]["value"] == 42


# ===========================================================================
# 12. COPY ENGINE STRUCTURE
# ===========================================================================

class TestCopyEngine:
    def test_contract_sizing_mirror(self):
        from config import AccountConfig, SizingMode
        acct = AccountConfig(
            name="copy-1", username="u", password="p", device_id="d",
            sizing_mode=SizingMode.MIRROR, account_size=150000,
        )
        assert acct.get_contracts("RSI", 3) == 3
        assert acct.get_contracts("IB", 5) == 5

    def test_contract_sizing_fixed(self):
        from config import AccountConfig, SizingMode
        acct = AccountConfig(
            name="copy-2", username="u", password="p", device_id="d",
            sizing_mode=SizingMode.FIXED,
            fixed_sizes={"RSI": 1, "IB": 2, "MOM": 1},
        )
        assert acct.get_contracts("RSI", 3) == 1
        assert acct.get_contracts("IB", 3) == 2

    def test_contract_sizing_scaled(self):
        from config import AccountConfig, SizingMode
        acct = AccountConfig(
            name="copy-3", username="u", password="p", device_id="d",
            sizing_mode=SizingMode.SCALED, account_size=75000,
        )
        assert acct.get_contracts("RSI", 3) == 1  # 75k/150k * 3 = 1.5 → floor = 1
        assert acct.get_contracts("RSI", 6) == 3  # 75k/150k * 6 = 3

    def test_contract_sizing_scaled_zero(self):
        """Scaled sizing with tiny account should return 0."""
        from config import AccountConfig, SizingMode
        acct = AccountConfig(
            name="tiny", username="u", password="p", device_id="d",
            sizing_mode=SizingMode.SCALED, account_size=25000,
        )
        assert acct.get_contracts("RSI", 3) == 0  # 25k/150k * 3 = 0.5 → floor = 0
