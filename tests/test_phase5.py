"""Phase 5 tests — signal engine, paper trader, alerts, dashboard."""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from strategies.ema_crossover import EMACrossoverStrategy

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Helpers ──────────────────────────────────────────────────────────

def generate_synthetic_data(num_bars=500, start_price=18000.0, volatility=2.0):
    np.random.seed(42)
    start_time = datetime(2025, 1, 6, 9, 30)
    prices = [start_price]
    for _ in range(num_bars - 1):
        prices.append(prices[-1] + np.random.normal(0, volatility))

    timestamps = []
    t = start_time
    for _ in range(num_bars):
        while t.weekday() >= 5:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        if t.hour >= 17:
            t += timedelta(days=1)
            t = t.replace(hour=9, minute=30)
        if t.hour < 8:
            t = t.replace(hour=9, minute=30)
        timestamps.append(t)
        t += timedelta(minutes=1)

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "low": [p - abs(np.random.normal(0, volatility * 0.5)) for p in prices],
        "close": prices,
        "volume": [int(abs(np.random.normal(500, 200))) for _ in range(num_bars)],
        "tick_count": [int(abs(np.random.normal(50, 20))) for _ in range(num_bars)],
    })
    df = df.with_columns([
        pl.max_horizontal("open", "high", "close").alias("high"),
        pl.min_horizontal("open", "low", "close").alias("low"),
    ])
    return df


def make_risk_manager():
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_50k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)
    return rm, prop_rules


# ── Signal Engine Tests ──────────────────────────────────────────────

class TestSignalEngine:
    def test_replay(self):
        from live.signal_engine import SignalEngine, Signal
        rm, prop_rules = make_risk_manager()
        account = rm.init_account(50000.0)
        strategy = EMACrossoverStrategy(
            fast_period=9, slow_period=21,
            stop_loss_points=4.0, take_profit_points=8.0, contracts=1,
        )
        engine = SignalEngine(
            strategies=[strategy],
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            account_state=account,
            prop_rules=prop_rules,
            lookback_bars=200,
        )

        signals_received = []
        engine.on_signal(lambda sig: signals_received.append(sig))

        df = generate_synthetic_data(300)
        engine.replay(df, speed=0.0)

        state = engine.get_state()
        assert state.bars_processed == 300
        assert state.strategies_loaded == 1

    def test_on_bar(self):
        from live.signal_engine import SignalEngine
        rm, prop_rules = make_risk_manager()
        account = rm.init_account(50000.0)
        strategy = EMACrossoverStrategy(
            fast_period=9, slow_period=21,
            stop_loss_points=4.0, take_profit_points=8.0, contracts=1,
        )
        engine = SignalEngine(
            strategies=[strategy],
            risk_manager=rm,
            contract_spec=MNQ_SPEC,
            account_state=account,
            prop_rules=prop_rules,
        )
        engine.start()

        # Feed bars manually
        for i in range(100):
            bar = {
                "timestamp": datetime(2025, 1, 6, 10, i % 60),
                "open": 18000.0 + i,
                "high": 18001.0 + i,
                "low": 17999.0 + i,
                "close": 18000.5 + i,
                "volume": 500,
                "tick_count": 50,
            }
            engine.on_bar(bar)

        assert engine.state.bars_processed == 100
        engine.stop()
        assert not engine.state.is_running


# ── Paper Trader Tests ───────────────────────────────────────────────

class TestPaperTrader:
    def test_signal_processing(self):
        from live.paper_trader import PaperTrader
        from live.signal_engine import Signal
        rm, prop_rules = make_risk_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            trader = PaperTrader(
                risk_manager=rm,
                contract_spec=MNQ_SPEC,
                prop_rules=prop_rules,
                initial_balance=50000.0,
                slippage_ticks=2,
                log_dir=tmpdir,
            )

            # Send an entry signal
            signal = Signal(
                timestamp=datetime(2025, 1, 6, 10, 0),
                strategy_name="test",
                direction="long",
                signal_type="entry",
                price=18000.0,
                stop_loss=17996.0,
                take_profit=18008.0,
                contracts=1,
            )
            trader.on_signal(signal)

            # Should have an open position
            assert trader.account.account_state.open_position is not None
            assert trader.account.signals_received >= 1

    def test_price_update_stops(self):
        from live.paper_trader import PaperTrader
        from live.signal_engine import Signal
        rm, prop_rules = make_risk_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            trader = PaperTrader(
                risk_manager=rm,
                contract_spec=MNQ_SPEC,
                prop_rules=prop_rules,
                initial_balance=50000.0,
                slippage_ticks=2,
                log_dir=tmpdir,
            )

            # Open a long position
            signal = Signal(
                timestamp=datetime(2025, 1, 6, 10, 0),
                strategy_name="test",
                direction="long",
                signal_type="entry",
                price=18000.0,
                stop_loss=17990.0,
                take_profit=18020.0,
                contracts=1,
            )
            trader.on_signal(signal)
            assert trader.account.account_state.open_position is not None

            # Price update that hits stop loss
            trader.on_price_update(datetime(2025, 1, 6, 10, 5), 17989.0)

            # Position should be closed
            assert trader.account.account_state.open_position is None
            assert len(trader.account.trade_history) >= 1

    def test_export_results(self):
        from live.paper_trader import PaperTrader
        rm, prop_rules = make_risk_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            trader = PaperTrader(
                risk_manager=rm,
                contract_spec=MNQ_SPEC,
                prop_rules=prop_rules,
                initial_balance=50000.0,
                log_dir=tmpdir,
            )
            path = trader.export_results()
            assert path.exists()


# ── Alert Tests ──────────────────────────────────────────────────────

class TestAlerts:
    def test_console_channel(self):
        from live.alerts import AlertManager, AlertLevel, Alert, ConsoleChannel
        mgr = AlertManager()
        mgr.add_channel(ConsoleChannel())

        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            title="Test Alert",
            message="This is a test",
        )
        mgr.send(alert)  # Should not raise

        history = mgr.get_history()
        assert len(history) == 1

    def test_file_channel(self):
        from live.alerts import AlertManager, AlertLevel, Alert, FileChannel
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AlertManager()
            mgr.add_channel(FileChannel(filepath=f"{tmpdir}/alerts.log"))

            mgr.send(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.TRADE,
                title="Trade",
                message="Entered long",
            ))

            log_path = Path(tmpdir) / "alerts.log"
            assert log_path.exists()

    def test_prebuilt_alerts(self):
        from live.alerts import AlertManager, ConsoleChannel
        mgr = AlertManager()
        mgr.add_channel(ConsoleChannel())

        mgr.drawdown_warning(-500.0, -2000.0, 25.0)
        mgr.daily_summary("2025-01-06", 234.50, 5, 50234.50, -415.0)

        assert len(mgr.get_history()) == 2

    def test_level_filtering(self):
        from live.alerts import AlertManager, AlertLevel, Alert, ConsoleChannel
        mgr = AlertManager()
        mgr.add_channel(ConsoleChannel(), levels=[AlertLevel.CRITICAL])

        # This should be filtered out (INFO, not CRITICAL)
        mgr.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            title="Filtered",
            message="Should not appear",
        ))
        # History still records it
        assert len(mgr.get_history()) == 1


# ── Dashboard Tests ──────────────────────────────────────────────────

class TestDashboard:
    def test_print_status_no_data(self):
        from live.dashboard import TradingDashboard
        _, prop_rules = make_risk_manager()
        dash = TradingDashboard(prop_rules=prop_rules)
        # Should handle no data gracefully
        dash.print_status()

    def test_with_paper_account(self):
        from live.dashboard import TradingDashboard
        from live.paper_trader import PaperTrader
        from live.signal_engine import Signal
        rm, prop_rules = make_risk_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            trader = PaperTrader(
                risk_manager=rm,
                contract_spec=MNQ_SPEC,
                prop_rules=prop_rules,
                initial_balance=50000.0,
                log_dir=tmpdir,
            )

            # Do a trade
            trader.on_signal(Signal(
                timestamp=datetime(2025, 1, 6, 10, 0),
                strategy_name="test", direction="long", signal_type="entry",
                price=18000.0, stop_loss=17996.0, take_profit=18008.0, contracts=1,
            ))
            trader.on_price_update(datetime(2025, 1, 6, 10, 5), 18009.0)

            dash = TradingDashboard(prop_rules=prop_rules)
            dash.update(trader.account)
            dash.print_status()
            dash.print_daily_summary()

    def test_generate_report(self):
        from live.dashboard import TradingDashboard
        from live.paper_trader import PaperTrader
        from live.signal_engine import Signal
        rm, prop_rules = make_risk_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            trader = PaperTrader(
                risk_manager=rm,
                contract_spec=MNQ_SPEC,
                prop_rules=prop_rules,
                initial_balance=50000.0,
                log_dir=tmpdir,
            )

            # Generate some trades
            trader.on_signal(Signal(
                timestamp=datetime(2025, 1, 6, 10, 0),
                strategy_name="test", direction="long", signal_type="entry",
                price=18000.0, stop_loss=17996.0, take_profit=18008.0, contracts=1,
            ))
            trader.on_price_update(datetime(2025, 1, 6, 10, 10), 18009.0)

            dash = TradingDashboard(prop_rules=prop_rules, output_dir=tmpdir)
            dash.update(trader.account)
            path = dash.generate_report()
            assert path is not None
            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
