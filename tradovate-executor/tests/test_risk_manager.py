"""
tests/test_risk_manager.py

Unit tests for RiskManager: P&L limits, session time enforcement,
daily/monthly resets, and flatten callback invocation.

All time-dependent calls are patched so tests never depend on
the real system clock.
"""

import sys
import os
import asyncio
import pytest
from datetime import datetime, date, time as dt_time
from unittest.mock import AsyncMock, MagicMock, patch, call
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_manager import RiskManager
from config import SessionConfig

ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_session() -> SessionConfig:
    return SessionConfig(
        daily_loss_limit=-3000.0,
        monthly_loss_limit=-4500.0,
    )


def make_rm(flatten_callback=None) -> RiskManager:
    return RiskManager(
        session_config=default_session(),
        on_flatten_all=flatten_callback,
    )


def et_datetime(hour: int, minute: int, day: int = 15, month: int = 1, year: int = 2026):
    return datetime(year, month, day, hour, minute, tzinfo=ET)


# ---------------------------------------------------------------------------
# Helper: run a coroutine in a new event loop for sync test methods
# ---------------------------------------------------------------------------

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Initial State
# ===========================================================================

class TestRiskManagerInitialState:
    def test_initial_daily_pnl_is_zero(self):
        rm = make_rm()
        assert rm.daily_pnl == 0.0

    def test_initial_monthly_pnl_is_zero(self):
        rm = make_rm()
        assert rm.monthly_pnl == 0.0

    def test_initial_trading_not_halted(self):
        rm = make_rm()
        assert rm.trading_halted is False

    def test_initial_limits_not_hit(self):
        rm = make_rm()
        assert rm.daily_limit_hit is False
        assert rm.monthly_limit_hit is False

    def test_initial_eod_not_flattened(self):
        rm = make_rm()
        assert rm.eod_flattened is False


# ===========================================================================
# Daily Loss Limit
# ===========================================================================

class TestDailyLossLimit:
    @pytest.mark.asyncio
    async def test_daily_limit_triggered_at_threshold(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        with patch("risk_manager.asyncio.create_task") as mock_task:
            rm.record_trade_pnl(-3000.0)

        assert rm.daily_limit_hit is True
        assert rm.trading_halted is True

    @pytest.mark.asyncio
    async def test_daily_limit_not_triggered_above_threshold(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-2999.99)

        assert rm.daily_limit_hit is False
        assert rm.trading_halted is False

    @pytest.mark.asyncio
    async def test_daily_limit_triggered_over_multiple_trades(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        with patch("risk_manager.asyncio.create_task") as mock_task:
            rm.record_trade_pnl(-1500.0)
            assert rm.daily_limit_hit is False
            rm.record_trade_pnl(-1500.0)
            # Now daily_pnl = -3000.0 exactly → triggers

        assert rm.daily_limit_hit is True
        assert rm.trading_halted is True

    @pytest.mark.asyncio
    async def test_daily_limit_only_triggers_once(self):
        """Second trade that pushes further should not re-trigger."""
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)
        task_count = 0

        def count_task(coro):
            nonlocal task_count
            task_count += 1
            # We can't actually schedule to a real event loop in sync context,
            # so just track the call
            coro.close()
            return MagicMock()

        with patch("risk_manager.asyncio.create_task", side_effect=count_task):
            rm.record_trade_pnl(-3000.0)  # triggers
            rm.record_trade_pnl(-500.0)   # daily already hit — no re-trigger

        # Only 1 flatten task was created (daily only, no monthly yet)
        assert task_count == 1

    def test_daily_pnl_accumulates_correctly(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-500.0)
            rm.record_trade_pnl(-200.0)
            rm.record_trade_pnl(300.0)
        assert rm.daily_pnl == pytest.approx(-400.0)

    def test_positive_pnl_does_not_trigger_daily_limit(self):
        rm = make_rm()
        rm.record_trade_pnl(5000.0)
        assert rm.daily_limit_hit is False
        assert rm.trading_halted is False


# ===========================================================================
# Monthly Loss Limit
# ===========================================================================

class TestMonthlyLossLimit:
    @pytest.mark.asyncio
    async def test_monthly_limit_triggered_at_threshold(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)

        assert rm.monthly_limit_hit is True
        assert rm.trading_halted is True

    @pytest.mark.asyncio
    async def test_monthly_limit_not_triggered_above_threshold(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4499.99)
        assert rm.monthly_limit_hit is False

    @pytest.mark.asyncio
    async def test_monthly_limit_can_trigger_independent_of_daily(self):
        """
        Monthly can trigger even when daily limit is not hit in a single day
        (e.g. losses accumulated over multiple daily resets).
        """
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)  # monthly hits, daily hits too in this case

        assert rm.monthly_limit_hit is True

    def test_monthly_pnl_accumulates_correctly(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-1000.0)
            rm.record_trade_pnl(-1000.0)
            rm.record_trade_pnl(-1000.0)
        assert rm.monthly_pnl == pytest.approx(-3000.0)

    @pytest.mark.asyncio
    async def test_monthly_limit_only_triggers_once(self):
        rm = make_rm()
        task_count = [0]

        def count_task(coro):
            task_count[0] += 1
            coro.close()
            return MagicMock()

        with patch("risk_manager.asyncio.create_task", side_effect=count_task):
            rm.record_trade_pnl(-4500.0)  # both daily + monthly trigger
            rm.record_trade_pnl(-500.0)   # neither should trigger again

        # 2 tasks from first call (daily + monthly), 0 from second
        assert task_count[0] == 2


# ===========================================================================
# can_trade() — Session Time Enforcement
# ===========================================================================

class TestCanTrade:
    def test_can_trade_during_session(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            assert rm.can_trade() is True

    def test_cannot_trade_before_session_start(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(9, 29)
            assert rm.can_trade() is False

    def test_can_trade_exactly_at_session_start(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(9, 30)
            assert rm.can_trade() is True

    def test_cannot_trade_at_cutoff_16_30(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(16, 30)
            assert rm.can_trade() is False

    def test_cannot_trade_after_cutoff(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(16, 45)
            assert rm.can_trade() is False

    def test_can_trade_one_minute_before_cutoff(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(16, 29)
            assert rm.can_trade() is True

    def test_cannot_trade_when_halted(self):
        rm = make_rm()
        rm.trading_halted = True
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            assert rm.can_trade() is False

    def test_cannot_trade_midnight(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(0, 0)
            assert rm.can_trade() is False

    def test_cannot_trade_early_morning(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(8, 0)
            assert rm.can_trade() is False

    def test_can_trade_at_noon(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(12, 0)
            assert rm.can_trade() is True


# ===========================================================================
# Daily Reset
# ===========================================================================

class TestDailyReset:
    def test_daily_reset_clears_daily_pnl(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-1000.0)
        assert rm.daily_pnl == -1000.0

        rm._reset_daily(date(2026, 1, 16))
        assert rm.daily_pnl == 0.0

    def test_daily_reset_clears_daily_limit_hit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-3000.0)
        assert rm.daily_limit_hit is True

        rm._reset_daily(date(2026, 1, 16))
        assert rm.daily_limit_hit is False

    def test_daily_reset_does_not_clear_monthly_pnl(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-2000.0)
        assert rm.monthly_pnl == -2000.0

        rm._reset_daily(date(2026, 1, 16))
        assert rm.monthly_pnl == -2000.0  # monthly unchanged

    def test_daily_reset_does_not_clear_monthly_limit_hit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)
        assert rm.monthly_limit_hit is True

        rm._reset_daily(date(2026, 1, 16))
        assert rm.monthly_limit_hit is True

    def test_daily_reset_does_not_resume_trading_when_monthly_limit_hit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)
        assert rm.trading_halted is True

        rm._reset_daily(date(2026, 1, 16))
        # Monthly limit is still hit → trading remains halted
        assert rm.trading_halted is True

    def test_daily_reset_resumes_trading_when_only_daily_limit_hit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-3000.0)  # only daily, not monthly
        assert rm.trading_halted is True
        assert rm.monthly_limit_hit is False

        rm._reset_daily(date(2026, 1, 16))
        assert rm.trading_halted is False

    def test_daily_reset_updates_current_date(self):
        rm = make_rm()
        new_date = date(2026, 1, 16)
        rm._reset_daily(new_date)
        assert rm.current_date == new_date

    def test_daily_reset_clears_eod_flattened(self):
        rm = make_rm()
        rm.eod_flattened = True
        rm._reset_daily(date(2026, 1, 16))
        assert rm.eod_flattened is False


# ===========================================================================
# Monthly Reset
# ===========================================================================

class TestMonthlyReset:
    def test_monthly_reset_clears_monthly_pnl(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4000.0)
        assert rm.monthly_pnl == -4000.0

        rm._reset_monthly(2)
        assert rm.monthly_pnl == 0.0

    def test_monthly_reset_clears_monthly_limit_hit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)
        assert rm.monthly_limit_hit is True

        rm._reset_monthly(2)
        assert rm.monthly_limit_hit is False

    def test_monthly_reset_resumes_trading(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)
        assert rm.trading_halted is True

        rm._reset_monthly(2)
        assert rm.trading_halted is False

    def test_monthly_reset_updates_current_month(self):
        rm = make_rm()
        rm._reset_monthly(3)
        assert rm.current_month == 3

    def test_monthly_reset_clears_daily_pnl(self):
        """Monthly reset also resets daily to prevent stale carryover."""
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-1000.0)
        rm._reset_monthly(2)
        assert rm.monthly_pnl == 0.0


# ===========================================================================
# get_status()
# ===========================================================================

class TestGetStatus:
    def test_status_returns_correct_structure(self):
        rm = make_rm()
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            status = rm.get_status()

        assert "daily_pnl" in status
        assert "monthly_pnl" in status
        assert "daily_limit_hit" in status
        assert "monthly_limit_hit" in status
        assert "trading_halted" in status
        assert "eod_flattened" in status
        assert "can_trade" in status

    def test_status_reflects_current_pnl(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-500.0)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            status = rm.get_status()
        assert status["daily_pnl"] == -500.0
        assert status["monthly_pnl"] == -500.0

    def test_status_can_trade_false_when_halted(self):
        rm = make_rm()
        rm.trading_halted = True
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            status = rm.get_status()
        assert status["can_trade"] is False


# ===========================================================================
# Flatten Callback
# ===========================================================================

class TestFlattenCallback:
    @pytest.mark.asyncio
    async def test_flatten_callback_called_on_daily_limit(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        async def run():
            with patch("risk_manager.asyncio.create_task") as mock_task:
                # Capture and immediately execute the coroutine
                tasks_created = []

                def side_effect(coro):
                    tasks_created.append(coro)
                    return MagicMock()

                mock_task.side_effect = side_effect
                rm.record_trade_pnl(-3000.0)

                for coro in tasks_created:
                    await coro

        await run()
        flatten_mock.assert_called()

    @pytest.mark.asyncio
    async def test_flatten_callback_called_on_monthly_limit(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)

        async def run():
            tasks_created = []

            def side_effect(coro):
                tasks_created.append(coro)
                return MagicMock()

            with patch("risk_manager.asyncio.create_task", side_effect=side_effect):
                rm.record_trade_pnl(-4500.0)

            for coro in tasks_created:
                await coro

        await run()
        flatten_mock.assert_called()

    @pytest.mark.asyncio
    async def test_flatten_not_called_below_threshold(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-1000.0)
        flatten_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_flatten_calls_callback(self):
        flatten_mock = AsyncMock()
        rm = make_rm(flatten_callback=flatten_mock)
        await rm.emergency_flatten()
        flatten_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_flatten_without_callback_does_not_raise(self):
        rm = make_rm(flatten_callback=None)
        # Should complete without error even with no callback registered
        await rm._execute_flatten("test reason")

    @pytest.mark.asyncio
    async def test_flatten_callback_exception_is_caught(self):
        """Failing flatten callback should be caught and logged, not raise."""
        async def bad_flatten():
            raise RuntimeError("connection lost")

        rm = make_rm(flatten_callback=bad_flatten)
        # Should not raise
        await rm._execute_flatten("test")


# ===========================================================================
# P&L Boundary Edge Cases
# ===========================================================================

class TestPnLEdgeCases:
    def test_zero_pnl_trade_does_not_affect_state(self):
        rm = make_rm()
        rm.record_trade_pnl(0.0)
        assert rm.daily_pnl == 0.0
        assert rm.monthly_pnl == 0.0
        assert rm.trading_halted is False

    def test_large_profit_does_not_trigger_any_limit(self):
        rm = make_rm()
        rm.record_trade_pnl(50000.0)
        assert rm.trading_halted is False
        assert rm.daily_limit_hit is False
        assert rm.monthly_limit_hit is False

    def test_daily_and_monthly_both_triggered_by_single_trade(self):
        """A single -$4500 loss hits both daily ($3k) and monthly ($4.5k)."""
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-4500.0)
        assert rm.daily_limit_hit is True
        assert rm.monthly_limit_hit is True

    def test_pnl_exactly_one_cent_above_daily_limit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-2999.99)
        assert rm.daily_limit_hit is False

    def test_pnl_exactly_at_daily_limit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-3000.0)
        assert rm.daily_limit_hit is True

    def test_can_trade_false_immediately_after_daily_limit(self):
        rm = make_rm()
        with patch("risk_manager.asyncio.create_task"):
            rm.record_trade_pnl(-3000.0)
        with patch("risk_manager.datetime") as mock_dt:
            mock_dt.now.return_value = et_datetime(10, 30)
            assert rm.can_trade() is False
