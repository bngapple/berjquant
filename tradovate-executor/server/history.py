"""
History — reads CSV trade logs and computes stats, daily P&L, equity curve.
Source of truth for historical dashboard data when engine is stopped.
"""

import csv
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

LOG_DIR = os.environ.get("LOG_DIR", "logs")


def scan_trade_files(log_dir: str = LOG_DIR) -> list[str]:
    """Find all trades_YYYY-MM-DD.csv files, sorted oldest first."""
    if not os.path.isdir(log_dir):
        return []
    files = [
        os.path.join(log_dir, f)
        for f in sorted(os.listdir(log_dir))
        if f.startswith("trades_") and f.endswith(".csv")
    ]
    return files


def parse_trades(log_dir: str = LOG_DIR) -> list[dict]:
    """Read all CSV trade logs into structured dicts."""
    trades = []
    for path in scan_trade_files(log_dir):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(_parse_row(row))
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
    return trades


def _parse_row(row: dict) -> dict:
    """Parse one CSV row into a clean dict."""
    return {
        "timestamp": row.get("timestamp", ""),
        "strategy": row.get("strategy", ""),
        "account": row.get("account", ""),
        "action": row.get("action", ""),  # Entry or Exit
        "side": row.get("side", ""),
        "contracts": _int(row.get("contracts")),
        "signal_price": _float(row.get("signal_price")),
        "fill_price": _float(row.get("fill_price")),
        "slippage_pts": _float(row.get("slippage_pts")),
        "sl_price": _float(row.get("sl_price")),
        "tp_price": _float(row.get("tp_price")),
        "exit_reason": row.get("exit_reason", ""),
        "exit_price": _float(row.get("exit_price")),
        "pnl_per_contract": _float(row.get("pnl_per_contract")),
        "pnl_total": _float(row.get("pnl_total")),
        "bars_held": _int(row.get("bars_held")),
    }


def _float(v: Optional[str]) -> float:
    if not v or v.strip() == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _int(v: Optional[str]) -> int:
    if not v or v.strip() == "":
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def get_exits(trades: list[dict], account: str = "") -> list[dict]:
    """Filter to exit rows only, optionally for one account."""
    exits = [t for t in trades if t["action"] == "Exit"]
    if account:
        exits = [t for t in exits if t["account"] == account]
    return exits


def compute_stats(trades: list[dict], account: str = "") -> dict:
    """Compute aggregate trading statistics from exit trades."""
    exits = get_exits(trades, account)
    if not exits:
        return {
            "total_pnl": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
        }

    wins = [t for t in exits if t["pnl_total"] > 0]
    losses = [t for t in exits if t["pnl_total"] < 0]
    total_pnl = sum(t["pnl_total"] for t in exits)
    win_pnl = sum(t["pnl_total"] for t in wins)
    loss_pnl = abs(sum(t["pnl_total"] for t in losses))
    avg_win = (win_pnl / len(wins)) if wins else 0.0
    avg_loss = (loss_pnl / len(losses)) if losses else 0.0
    win_rate = (len(wins) / len(exits)) * 100 if exits else 0.0
    profit_factor = (win_pnl / loss_pnl) if loss_pnl > 0 else (999.0 if win_pnl > 0 else 0.0)

    return {
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_trades": len(exits),
        "winners": len(wins),
        "losers": len(losses),
    }


def compute_daily_pnl(trades: list[dict], account: str = "") -> dict[str, dict]:
    """
    Compute per-day P&L from exit trades.
    Returns: { "2026-03-15": { "pnl": 450.0, "trades": 4, "wins": 3, "losses": 1 }, ... }
    """
    exits = get_exits(trades, account)
    daily: dict[str, dict] = {}

    for t in exits:
        ts = t["timestamp"]
        if not ts:
            continue
        day = ts.split(" ")[0] if " " in ts else ts.split("T")[0]
        if day not in daily:
            daily[day] = {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        daily[day]["pnl"] += t["pnl_total"]
        daily[day]["trades"] += 1
        if t["pnl_total"] > 0:
            daily[day]["wins"] += 1
        elif t["pnl_total"] < 0:
            daily[day]["losses"] += 1

    # Round P&L
    for d in daily.values():
        d["pnl"] = round(d["pnl"], 2)

    return daily


def compute_equity_curve(daily_pnl: dict[str, dict]) -> list[dict]:
    """
    Build cumulative equity curve from daily P&L.
    Returns: [{"date": "2026-03-15", "value": 450.0}, ...]
    """
    if not daily_pnl:
        return []

    curve = []
    cumulative = 0.0
    for date in sorted(daily_pnl.keys()):
        cumulative += daily_pnl[date]["pnl"]
        curve.append({"date": date, "value": round(cumulative, 2)})

    return curve


def get_recent_trades(trades: list[dict], limit: int = 50) -> list[dict]:
    """Return the most recent N trades (entries + exits)."""
    return trades[-limit:] if len(trades) > limit else list(trades)
