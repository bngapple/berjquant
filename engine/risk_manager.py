"""Risk management layer — enforced before every order."""

import calendar as cal
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

from engine.utils import (
    AccountState,
    ContractSpec,
    Position,
    PropFirmRules,
    Trade,
)

ET = ZoneInfo("US/Eastern")


class RiskManager:
    """Always-on risk layer. The backtester cannot bypass this."""

    def __init__(
        self,
        prop_rules: PropFirmRules,
        session_config: dict,
        events_calendar: dict,
        contract_spec: ContractSpec,
    ):
        self.prop_rules = prop_rules
        self.session_config = session_config
        self.events_calendar = events_calendar
        self.contract_spec = contract_spec
        self.blackout_buffer = timedelta(
            minutes=events_calendar.get("blackout_buffer_minutes", 5)
        )

        # Pre-parse event blackout windows
        self._blackout_windows: list[tuple[datetime, datetime]] = []
        self._parse_events()

        # Parse session bounds
        active = session_config["active_window"]
        self._active_start = self._parse_time(active["start"])
        self._active_end = self._parse_time(active["end"])
        self._flat_by = self._parse_time(session_config["flat_by"])

    def _parse_time(self, t: str) -> time:
        parts = t.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _parse_events(self):
        """Pre-compute blackout windows from events calendar.

        Fix #7: Also parse recurring_events and generate actual dates
        for the backtest period (2024-2026).
        """
        events = self.events_calendar.get("events", [])
        buffer = self.blackout_buffer

        for event in events:
            dt = datetime.strptime(
                f"{event['date']} {event['time']}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=ET)
            self._blackout_windows.append((dt - buffer, dt + buffer))

        # Generate dates from recurring event patterns
        recurring = self.events_calendar.get("recurring_events", [])
        for rec in recurring:
            schedule = rec.get("schedule", "")
            time_str = rec.get("time", "08:30")
            h, m = map(int, time_str.split(":"))

            for year in range(2024, 2027):
                dates = self._generate_recurring_dates(schedule, year)
                for d in dates:
                    dt = datetime(d.year, d.month, d.day, h, m, tzinfo=ET)
                    self._blackout_windows.append((dt - buffer, dt + buffer))

    @staticmethod
    def _generate_recurring_dates(schedule: str, year: int) -> list[date]:
        """Generate dates for a recurring event schedule."""
        dates: list[date] = []

        if schedule == "first_friday":
            # First Friday of each month
            for month in range(1, 13):
                for day in range(1, 8):
                    d = date(year, month, day)
                    if d.weekday() == 4:  # Friday
                        dates.append(d)
                        break

        elif schedule == "first_business_day":
            for month in range(1, 13):
                for day in range(1, 5):
                    d = date(year, month, day)
                    if d.weekday() < 5:  # Mon-Fri
                        dates.append(d)
                        break

        elif schedule == "monthly_mid":
            # ~13th-15th of each month (approximate for CPI/PPI/Retail)
            for month in range(1, 13):
                d = date(year, month, 13)
                while d.weekday() >= 5:
                    d += timedelta(days=1)
                dates.append(d)

        elif schedule == "quarterly":
            # End of Jan, Apr, Jul, Oct (GDP advance estimate)
            for month in [1, 4, 7, 10]:
                last_day = cal.monthrange(year, month)[1]
                d = date(year, month, last_day)
                while d.weekday() >= 5:
                    d -= timedelta(days=1)
                dates.append(d)

        elif schedule == "weekly_thursday":
            # Every Thursday
            d = date(year, 1, 1)
            while d.weekday() != 3:
                d += timedelta(days=1)
            while d.year == year:
                dates.append(d)
                d += timedelta(days=7)

        return dates

    def init_account(self, starting_balance: float) -> AccountState:
        """Create fresh account state."""
        return AccountState(
            starting_balance=starting_balance,
            current_balance=starting_balance,
            high_water_mark=starting_balance,
        )

    # ── Pre-Trade Checks ─────────────────────────────────────────

    def pre_trade_check(
        self,
        timestamp: datetime,
        direction: str,
        contracts: int,
        account_state: AccountState,
    ) -> tuple[bool, str]:
        """
        Run all risk checks before allowing a trade entry.
        Returns (allowed, reason_if_rejected).
        """
        checks = [
            self._check_kill_switch(account_state),
            self._check_session_bounds(timestamp),
            self._check_event_blackout(timestamp),
            self._check_daily_loss_limit(account_state),
            self._check_max_drawdown(account_state),
            self._check_max_contracts(contracts, account_state),
            self._check_eod_proximity(timestamp),
        ]

        for allowed, reason in checks:
            if not allowed:
                return False, reason

        return True, ""

    def _check_kill_switch(self, state: AccountState) -> tuple[bool, str]:
        if state.is_killed:
            return False, "kill_switch_active"
        if state.daily_pnl <= self.prop_rules.kill_switch_threshold:
            state.is_killed = True
            return False, "kill_switch_triggered"
        return True, ""

    def _check_session_bounds(self, timestamp: datetime) -> tuple[bool, str]:
        t = timestamp.time()
        if t < self._active_start or t >= self._active_end:
            return False, "outside_session"
        return True, ""

    def _check_event_blackout(self, timestamp: datetime) -> tuple[bool, str]:
        for start, end in self._blackout_windows:
            # Compare naive to naive or aware to aware
            ts = timestamp
            if ts.tzinfo is None and start.tzinfo is not None:
                ts = ts.replace(tzinfo=ET)
            elif ts.tzinfo is not None and start.tzinfo is None:
                start = start.replace(tzinfo=ts.tzinfo)
                end = end.replace(tzinfo=ts.tzinfo)

            if start <= ts <= end:
                return False, "event_blackout"
        return True, ""

    def _check_daily_loss_limit(self, state: AccountState) -> tuple[bool, str]:
        if state.daily_pnl <= self.prop_rules.daily_loss_limit:
            return False, "daily_loss_limit_breached"
        return True, ""

    def _check_max_drawdown(self, state: AccountState) -> tuple[bool, str]:
        drawdown = state.current_drawdown
        if drawdown <= self.prop_rules.max_drawdown:
            return False, "max_drawdown_breached"
        return True, ""

    def _check_max_contracts(
        self, requested: int, state: AccountState
    ) -> tuple[bool, str]:
        symbol = self.contract_spec.symbol
        max_allowed = self.prop_rules.max_contracts.get(symbol, 0)
        current_open = 0
        if state.open_position:
            current_open = state.open_position.contracts
        if current_open + requested > max_allowed:
            return False, f"max_contracts_exceeded ({current_open + requested} > {max_allowed})"
        return True, ""

    def _check_eod_proximity(
        self, timestamp: datetime, buffer_minutes: int = 5
    ) -> tuple[bool, str]:
        """Don't allow new entries too close to session end."""
        t = timestamp.time()
        flat_minutes = self._flat_by.hour * 60 + self._flat_by.minute
        current_minutes = t.hour * 60 + t.minute
        if flat_minutes - current_minutes <= buffer_minutes:
            return False, "too_close_to_session_end"
        return True, ""

    # ── Post-Trade Updates ───────────────────────────────────────

    def post_trade_update(self, trade: Trade, state: AccountState):
        """Update account state after a trade completes."""
        state.current_balance += trade.net_pnl
        state.daily_pnl += trade.net_pnl
        state.trades_today.append(trade)
        state.open_position = None

        # Update high water mark (trailing drawdown)
        if self.prop_rules.drawdown_type == "trailing":
            if state.current_balance > state.high_water_mark:
                state.high_water_mark = state.current_balance

    def reset_daily(self, state: AccountState, date_str: str):
        """Reset daily tracking at start of each trading day."""
        state.daily_pnl = 0.0
        state.trades_today = []
        state.is_killed = False
        state.current_date = date_str

        # For EOD drawdown type, update HWM at end of previous day
        if self.prop_rules.drawdown_type == "eod":
            if state.current_balance > state.high_water_mark:
                state.high_water_mark = state.current_balance

    def should_flatten(self, timestamp: datetime) -> bool:
        """Check if we need to force-flatten positions (end of session)."""
        t = timestamp.time()
        return t >= self._flat_by
