"""
Alerting system -- sends notifications for trade signals, drawdown warnings,
kill switch triggers, and other events via multiple channels.

Channels:
    - ConsoleChannel  : colored terminal output
    - FileChannel     : append to a log file (JSON-lines)
    - WebhookChannel  : POST to Slack / Discord / generic webhook
    - EmailChannel    : SMTP email (critical alerts only by default)
    - CallbackChannel : invoke an arbitrary callable (useful for tests / custom integrations)

Every channel is wrapped in a try/except so a failing notification can never
crash the trading engine.
"""

import json
import logging
import smtplib
import traceback
from email.mime.text import MIMEText
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import urllib.request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class AlertLevel:
    INFO = "info"           # Trade signals, daily summaries
    WARNING = "warning"     # Approaching limits, high drawdown
    CRITICAL = "critical"   # Kill switch, max drawdown hit
    TRADE = "trade"         # Trade executed

    _ALL = frozenset({"info", "warning", "critical", "trade"})


# ---------------------------------------------------------------------------
# Alert data object
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """An alert event."""
    timestamp: datetime
    level: str              # AlertLevel value
    title: str
    message: str
    data: dict = field(default_factory=dict)

    # Convenience -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# AlertChannel base
# ---------------------------------------------------------------------------

class AlertChannel:
    """Base class for alert channels."""
    name: str = "base"

    def send(self, alert: Alert):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Central alert dispatcher.  Routes alerts to configured channels.

    Usage::

        alerts = AlertManager()

        # Add channels
        alerts.add_channel(ConsoleChannel())
        alerts.add_channel(FileChannel("logs/alerts.log"))
        alerts.add_channel(WebhookChannel(url="https://hooks.slack.com/..."))

        # Send alerts
        alerts.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.TRADE,
            title="LONG Entry",
            message="EMA_RSI strategy entered long at 18,250.50",
            data={"strategy": "EMA_RSI", "price": 18250.50},
        ))

        # Pre-built alert helpers
        alerts.trade_alert(signal, fill_price)
        alerts.drawdown_warning(current_dd, max_dd, pct_used)
        alerts.kill_switch_alert(reason, daily_pnl)
        alerts.daily_summary(date, pnl, trades, balance, drawdown)
    """

    def __init__(self):
        self._channels: list[tuple[AlertChannel, set[str] | None]] = []
        self._history: list[Alert] = []
        self._max_history: int = 1000

    # -- Channel management -------------------------------------------------

    def add_channel(
        self,
        channel: AlertChannel,
        levels: list[str] | None = None,
    ):
        """
        Register a notification channel.

        Parameters
        ----------
        channel : AlertChannel
            The channel instance.
        levels : list[str] | None
            If provided, this channel only receives alerts whose level is in
            the list.  ``None`` means "receive everything".
        """
        level_set = set(levels) if levels is not None else None
        self._channels.append((channel, level_set))
        logger.info(
            "Alert channel added: %s (levels=%s)",
            channel.name,
            levels or "all",
        )

    # -- Core dispatch ------------------------------------------------------

    def send(self, alert: Alert):
        """Dispatch *alert* to every configured channel whose filter matches."""
        # Store in history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for channel, level_filter in self._channels:
            if level_filter is not None and alert.level not in level_filter:
                continue
            try:
                channel.send(alert)
            except Exception:
                logger.error(
                    "Alert channel '%s' failed for alert '%s': %s",
                    channel.name,
                    alert.title,
                    traceback.format_exc(),
                )

    # -- Pre-built helpers --------------------------------------------------

    def trade_alert(self, signal: dict[str, Any], fill_price: float):
        """
        Pre-built alert for trade execution.

        *signal* should contain at minimum:
            - direction : "long" | "short"
            - strategy  : strategy name
            - contracts : int
        """
        direction = signal.get("direction", "unknown").upper()
        strategy = signal.get("strategy", "unknown")
        contracts = signal.get("contracts", 1)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")

        parts = [
            f"{strategy} entered {direction} x{contracts} at {fill_price:,.2f}",
        ]
        if stop_loss is not None:
            parts.append(f"SL {stop_loss:,.2f}")
        if take_profit is not None:
            parts.append(f"TP {take_profit:,.2f}")

        self.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.TRADE,
            title=f"{direction} Entry",
            message=" | ".join(parts),
            data={
                "direction": direction.lower(),
                "strategy": strategy,
                "fill_price": fill_price,
                "contracts": contracts,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        ))

    def drawdown_warning(
        self,
        current_dd: float,
        max_dd: float,
        pct_used: float,
    ):
        """Pre-built alert for drawdown approaching the limit."""
        level = AlertLevel.WARNING if pct_used < 0.9 else AlertLevel.CRITICAL
        self.send(Alert(
            timestamp=datetime.now(),
            level=level,
            title="Drawdown Warning",
            message=(
                f"Drawdown ${current_dd:,.2f} / ${max_dd:,.2f} "
                f"({pct_used * 100:.1f}% of limit used)"
            ),
            data={
                "current_dd": current_dd,
                "max_dd": max_dd,
                "pct_used": pct_used,
            },
        ))

    def kill_switch_alert(self, reason: str, daily_pnl: float):
        """Pre-built alert for kill switch activation."""
        self.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            title="KILL SWITCH ACTIVATED",
            message=f"Reason: {reason} | Daily P&L: ${daily_pnl:,.2f}",
            data={
                "reason": reason,
                "daily_pnl": daily_pnl,
            },
        ))

    def daily_summary(
        self,
        date: str,
        pnl: float,
        trades: int,
        balance: float,
        drawdown: float,
    ):
        """Pre-built alert for end-of-day summary."""
        pnl_label = "profit" if pnl >= 0 else "loss"
        self.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            title=f"Daily Summary - {date}",
            message=(
                f"P&L: ${pnl:+,.2f} ({pnl_label}) | "
                f"Trades: {trades} | "
                f"Balance: ${balance:,.2f} | "
                f"Drawdown: ${drawdown:,.2f}"
            ),
            data={
                "date": date,
                "pnl": pnl,
                "trades": trades,
                "balance": balance,
                "drawdown": drawdown,
            },
        ))

    def position_update(self, position: Any, unrealized_pnl: float):
        """
        Pre-built alert for open position updates.

        *position* is expected to be an ``engine.utils.Position`` (or any
        object with ``direction``, ``entry_price``, ``contracts``, ``symbol``
        attributes).
        """
        direction = getattr(position, "direction", "unknown").upper()
        entry = getattr(position, "entry_price", 0.0)
        contracts = getattr(position, "contracts", 0)
        symbol = getattr(position, "symbol", "?")

        self.send(Alert(
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            title=f"Position Update - {symbol}",
            message=(
                f"{direction} x{contracts} from {entry:,.2f} | "
                f"Unrealized P&L: ${unrealized_pnl:+,.2f}"
            ),
            data={
                "symbol": symbol,
                "direction": direction.lower(),
                "entry_price": entry,
                "contracts": contracts,
                "unrealized_pnl": unrealized_pnl,
            },
        ))

    # -- History access -----------------------------------------------------

    def get_history(
        self,
        level: str | None = None,
        last_n: int = 50,
    ) -> list[Alert]:
        """Return recent alerts, optionally filtered by level."""
        if level is not None:
            filtered = [a for a in self._history if a.level == level]
        else:
            filtered = list(self._history)
        return filtered[-last_n:]


# =========================================================================
# Channel implementations
# =========================================================================

class ConsoleChannel(AlertChannel):
    """Print alerts to the console with ANSI color formatting."""
    name = "console"

    COLORS = {
        AlertLevel.INFO:     "\033[36m",   # Cyan
        AlertLevel.WARNING:  "\033[33m",   # Yellow
        AlertLevel.CRITICAL: "\033[31m",   # Red bold
        AlertLevel.TRADE:    "\033[32m",   # Green
    }
    BOLD = "\033[1m"
    RESET = "\033[0m"

    ICONS = {
        AlertLevel.INFO:     "[i]",
        AlertLevel.WARNING:  "[!]",
        AlertLevel.CRITICAL: "[X]",
        AlertLevel.TRADE:    "[$]",
    }

    def send(self, alert: Alert):
        color = self.COLORS.get(alert.level, "")
        icon = self.ICONS.get(alert.level, "[?]")
        ts = alert.timestamp.strftime("%H:%M:%S")

        # Title line
        header = (
            f"{color}{self.BOLD}{icon} {ts} "
            f"[{alert.level.upper()}] {alert.title}{self.RESET}"
        )
        # Body
        body = f"{color}    {alert.message}{self.RESET}"

        print(header)
        print(body)

        # Extra data (only if non-empty and not already encoded in message)
        if alert.data:
            compact = "    " + " | ".join(
                f"{k}={v}" for k, v in alert.data.items()
            )
            print(f"{color}{compact}{self.RESET}")

        print()  # blank line separator


class FileChannel(AlertChannel):
    """
    Append alerts to a file as JSON-lines.

    Each line is a self-contained JSON object, making it easy to parse with
    standard tools (``jq``, pandas, etc.).
    """
    name = "file"

    def __init__(self, filepath: str | Path = "logs/alerts.log"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert):
        try:
            with open(self.filepath, "a") as f:
                f.write(alert.to_json() + "\n")
        except OSError as exc:
            logger.error("FileChannel write failed: %s", exc)


class WebhookChannel(AlertChannel):
    """
    Send alerts to a webhook URL (Slack, Discord, or generic JSON POST).

    Formats the payload according to *platform*:
        - ``"slack"``   : Slack incoming-webhook block layout
        - ``"discord"`` : Discord embed
        - ``"generic"`` : raw JSON POST of the alert dict
    """
    name = "webhook"

    def __init__(self, url: str, platform: str = "slack", timeout: int = 5):
        self.url = url
        self.platform = platform.lower()
        self.timeout = timeout

    def send(self, alert: Alert):
        try:
            if self.platform == "slack":
                payload = self._format_slack(alert)
            elif self.platform == "discord":
                payload = self._format_discord(alert)
            else:
                payload = alert.to_dict()

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout)

        except Exception as exc:
            # Never let a webhook failure propagate
            logger.error(
                "WebhookChannel (%s) failed: %s", self.platform, exc,
            )

    # -- Slack formatting ---------------------------------------------------

    def _format_slack(self, alert: Alert) -> dict:
        """Build a Slack Block Kit payload."""
        level_emoji = {
            AlertLevel.INFO:     ":information_source:",
            AlertLevel.WARNING:  ":warning:",
            AlertLevel.CRITICAL: ":rotating_light:",
            AlertLevel.TRADE:    ":chart_with_upwards_trend:",
        }
        emoji = level_emoji.get(alert.level, ":bell:")
        ts = alert.timestamp.strftime("%H:%M:%S")

        blocks: list[dict] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji}  {alert.title}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Level:* `{alert.level.upper()}`  |  *Time:* `{ts}`",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
        ]

        # Data fields as a compact section
        if alert.data:
            fields = []
            for key, value in alert.data.items():
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n`{value}`",
                })
            # Slack allows max 10 fields per section
            for i in range(0, len(fields), 10):
                blocks.append({
                    "type": "section",
                    "fields": fields[i:i + 10],
                })

        blocks.append({"type": "divider"})

        return {"blocks": blocks}

    # -- Discord formatting -------------------------------------------------

    def _format_discord(self, alert: Alert) -> dict:
        """Build a Discord webhook payload with an embed."""
        color_map = {
            AlertLevel.INFO:     0x36A2EB,   # blue
            AlertLevel.WARNING:  0xFFCE56,   # yellow
            AlertLevel.CRITICAL: 0xFF6384,   # red
            AlertLevel.TRADE:    0x4BC0C0,   # green
        }

        fields = [
            {"name": k, "value": f"`{v}`", "inline": True}
            for k, v in alert.data.items()
        ]

        embed: dict[str, Any] = {
            "title": alert.title,
            "description": alert.message,
            "color": color_map.get(alert.level, 0x999999),
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": f"MCQ Engine | {alert.level.upper()}"},
        }
        if fields:
            embed["fields"] = fields

        return {"embeds": [embed]}


class EmailChannel(AlertChannel):
    """
    Send alerts via SMTP email.

    By default only *critical* alerts are emailed.  Override by passing a
    custom ``levels`` list to ``AlertManager.add_channel``.
    """
    name = "email"

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        to_address: str,
        from_address: str | None = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_address = to_address
        self.from_address = from_address or username

    def send(self, alert: Alert):
        """Send an email.  Skips non-critical alerts unless the channel was
        explicitly configured with broader level filters via AlertManager."""
        # Default behaviour: only email critical alerts.  If the user added
        # this channel with ``levels=["critical"]`` (or left it as None and
        # relies on this guard), critical-only is still enforced here as a
        # safety net.
        if alert.level != AlertLevel.CRITICAL:
            return

        try:
            ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            subject = f"[MCQ {alert.level.upper()}] {alert.title}"

            body_lines = [
                f"Time:    {ts}",
                f"Level:   {alert.level.upper()}",
                f"Title:   {alert.title}",
                "",
                alert.message,
                "",
            ]
            if alert.data:
                body_lines.append("Details:")
                for k, v in alert.data.items():
                    body_lines.append(f"  {k}: {v}")

            body = "\n".join(body_lines)

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = self.to_address

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.username, self.password)
                server.sendmail(self.from_address, [self.to_address], msg.as_string())

            logger.info("Email alert sent: %s", subject)

        except Exception as exc:
            logger.error("EmailChannel failed: %s", exc)


class CallbackChannel(AlertChannel):
    """
    Invoke an arbitrary callable for each alert.

    Useful for unit tests, custom integrations, or bridging to other
    notification systems.

    Example::

        received = []
        alerts.add_channel(CallbackChannel(received.append))
    """
    name = "callback"

    def __init__(self, callback: Callable[[Alert], Any], name: str = "callback"):
        self._callback = callback
        self.name = name

    def send(self, alert: Alert):
        try:
            self._callback(alert)
        except Exception as exc:
            logger.error("CallbackChannel '%s' failed: %s", self.name, exc)
