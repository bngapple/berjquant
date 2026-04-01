"""
Account Tracker — per-account status derived from CSV trade logs + config.
No Tradovate API calls. All metrics computed locally.
"""

import logging
from server import config_store
from server.history import parse_trades, get_exits

logger = logging.getLogger(__name__)


def get_account_status(name: str, trades: list[dict] | None = None) -> dict:
    """Compute status for one account from config + trade history."""
    acct = config_store.get_account(name)
    if not acct:
        return {"name": name, "error": "Account not found"}

    if trades is None:
        trades = parse_trades()

    exits = get_exits(trades, account=name)

    starting_balance = acct.get("starting_balance", 150000.0)
    profit_target = acct.get("profit_target", 9000.0)
    max_drawdown = acct.get("max_drawdown", -4500.0)

    # Total P&L from all exits
    total_pnl = sum(t["pnl_total"] for t in exits)

    # Current balance
    balance = starting_balance + total_pnl

    # Drawdown: track from peak equity
    peak = starting_balance
    equity = starting_balance
    max_dd = 0.0
    for t in exits:
        equity += t["pnl_total"]
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd

    current_dd = equity - peak
    dd_remaining = max_drawdown - current_dd  # Both negative, so remaining = limit - current
    dd_pct_used = (abs(current_dd) / abs(max_drawdown) * 100) if max_drawdown != 0 else 0.0

    # Profit target progress
    target_progress = (total_pnl / profit_target * 100) if profit_target > 0 else 0.0

    # Daily P&L (from today's exits)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
    today_exits = [t for t in exits if t["timestamp"].startswith(today)]
    daily_pnl = sum(t["pnl_total"] for t in today_exits)
    trades_today = len(today_exits)

    # Status
    if abs(max_drawdown) > 0 and current_dd <= max_drawdown:
        status = "breached"
    elif profit_target > 0 and total_pnl >= profit_target:
        status = "eval_passed"
    elif daily_pnl <= -3000:
        status = "daily_limit_hit"
    else:
        status = "active"

    env = config_store.get_environment()

    return {
        "name": name,
        "balance": round(balance, 2),
        "starting_balance": starting_balance,
        "pnl_total": round(total_pnl, 2),
        "drawdown_current": round(current_dd, 2),
        "drawdown_max_allowed": max_drawdown,
        "drawdown_remaining": round(dd_remaining, 2),
        "drawdown_pct_used": round(dd_pct_used, 1),
        "profit_target": profit_target,
        "profit_target_progress": round(target_progress, 1),
        "daily_pnl": round(daily_pnl, 2),
        "trades_today": trades_today,
        "status": status,
        "is_master": acct.get("is_master", False),
        "account_type": acct.get("account_type", "eval"),
        "environment": env,
    }


def get_all_statuses() -> list[dict]:
    """Compute status for ALL configured accounts."""
    accounts = config_store.get_accounts()
    if not accounts:
        return []

    trades = parse_trades()
    return [get_account_status(a["name"], trades) for a in accounts]


def get_fleet_alerts() -> list[dict]:
    """
    Return warnings/milestones for the dashboard fleet health strip.
    Only returns accounts with noteworthy status.
    """
    statuses = get_all_statuses()
    alerts = []

    for s in statuses:
        if s.get("error"):
            continue

        if s["drawdown_pct_used"] >= 75:
            alerts.append({
                "account": s["name"],
                "type": "danger",
                "message": f"{s['drawdown_pct_used']:.0f}% drawdown used",
            })
        elif s["drawdown_pct_used"] >= 50:
            alerts.append({
                "account": s["name"],
                "type": "warning",
                "message": f"{s['drawdown_pct_used']:.0f}% drawdown used",
            })

        if s["status"] == "eval_passed":
            alerts.append({
                "account": s["name"],
                "type": "success",
                "message": "Eval target reached",
            })

        if s["status"] == "breached":
            alerts.append({
                "account": s["name"],
                "type": "danger",
                "message": "Max drawdown breached",
            })

    return alerts
