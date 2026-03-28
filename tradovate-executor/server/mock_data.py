"""
Mock data generators for Phase 1 dashboard.
Produces realistic-looking trading data when engine is "running".
Replaced by real engine data in Phase 2.
"""

import random
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# Module-level simulated state
_positions: dict[str, dict | None] = {"RSI": None, "IB": None, "MOM": None}
_daily_pnl: float = 0.0
_monthly_pnl: float = 0.0
_signals: list[dict] = []
_trades: list[dict] = []

_SL_PTS = {"RSI": 10, "IB": 10, "MOM": 15}
_TP_PTS = {"RSI": 100, "IB": 120, "MOM": 100}
_REASONS = {
    "RSI": lambda s: f"RSI(5) = {random.uniform(20, 34):.1f} < 35"
    if s == "Buy"
    else f"RSI(5) = {random.uniform(66, 80):.1f} > 65",
    "IB": lambda s: f"Close {'>' if s == 'Buy' else '<'} IB {'high' if s == 'Buy' else 'low'}",
    "MOM": lambda _: "Bar range > ATR(14), volume > SMA(20)",
}


def get_engine_status(running: bool) -> dict:
    accounts = [{"name": a["name"], "connected": running} for a in _get_configured_accounts()]
    if not accounts:
        accounts = [{"name": "No accounts configured", "connected": False}]
    return {
        "running": running,
        "can_trade": running,
        "daily_pnl": round(_daily_pnl, 2),
        "monthly_pnl": round(_monthly_pnl, 2),
        "daily_limit": -3000.0,
        "monthly_limit": -4500.0,
        "daily_limit_hit": _daily_pnl <= -3000,
        "monthly_limit_hit": _monthly_pnl <= -4500,
        "positions": {k: v for k, v in _positions.items()},
        "pending_signals": 0,
        "connected_accounts": accounts,
    }


def _get_configured_accounts() -> list[dict]:
    try:
        from server.config_store import get_accounts
        return get_accounts()
    except Exception:
        return []


def generate_ws_batch(running: bool, tick: int) -> list[dict]:
    global _daily_pnl, _monthly_pnl
    now = datetime.now(ET).isoformat()
    messages = []

    # Always send status
    messages.append({"type": "status", "data": get_engine_status(running), "timestamp": now})

    if not running:
        return messages

    base_price = 21500 + random.uniform(-50, 50)

    # Position updates every tick
    for strategy, pos in _positions.items():
        if pos is not None:
            current = base_price + random.uniform(-5, 5)
            if pos["side"] == "Buy":
                pnl = (current - pos["entry_price"]) * 2.0 * pos["contracts"]
            else:
                pnl = (pos["entry_price"] - current) * 2.0 * pos["contracts"]
            pos["current_price"] = round(current, 2)
            pos["pnl"] = round(pnl, 2)
            if tick % 5 == 0:
                pos["bars_held"] += 1
            messages.append(
                {"type": "position", "data": {**pos, "strategy": strategy}, "timestamp": now}
            )

    # Every ~16s: maybe generate a signal
    if tick % 8 == 3 and random.random() > 0.5:
        strategy = random.choice(["RSI", "IB", "MOM"])
        if _positions[strategy] is None:
            side = random.choice(["Buy", "Sell"])
            signal = {
                "strategy": strategy,
                "side": side,
                "contracts": 3,
                "reason": _REASONS[strategy](side),
                "price": round(base_price, 2),
            }
            _signals.append({**signal, "timestamp": now})
            if len(_signals) > 50:
                _signals.pop(0)
            messages.append({"type": "signal", "data": signal, "timestamp": now})

    # Every ~20s: maybe fill a signal (open position)
    if tick % 10 == 7:
        for strategy in ["RSI", "IB", "MOM"]:
            if _positions[strategy] is None and random.random() > 0.6:
                side = random.choice(["Buy", "Sell"])
                fill_price = round(base_price + random.uniform(-2, 2), 2)
                sl_pts = _SL_PTS[strategy]
                tp_pts = _TP_PTS[strategy]
                sl = fill_price - sl_pts if side == "Buy" else fill_price + sl_pts
                tp = fill_price + tp_pts if side == "Buy" else fill_price - tp_pts

                _positions[strategy] = {
                    "side": side,
                    "entry_price": fill_price,
                    "current_price": fill_price,
                    "contracts": 3,
                    "pnl": 0.0,
                    "bars_held": 0,
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                }
                fill = {
                    "strategy": strategy,
                    "side": side,
                    "contracts": 3,
                    "fill_price": fill_price,
                    "slippage": round(random.uniform(0, 1.5), 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                }
                _trades.append({**fill, "timestamp": now, "action": "entry"})
                if len(_trades) > 50:
                    _trades.pop(0)
                messages.append({"type": "fill", "data": fill, "timestamp": now})
                break

    # Every ~30s: maybe close a position
    if tick % 15 == 12:
        for strategy in ["RSI", "IB", "MOM"]:
            pos = _positions[strategy]
            if pos is not None and random.random() > 0.5:
                exit_price = round(pos["current_price"], 2)
                if pos["side"] == "Buy":
                    pnl = (exit_price - pos["entry_price"]) * 2.0 * pos["contracts"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * 2.0 * pos["contracts"]

                _daily_pnl += pnl
                _monthly_pnl += pnl

                exit_data = {
                    "strategy": strategy,
                    "side": pos["side"],
                    "contracts": pos["contracts"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": random.choice(["SL", "TP", "MaxHold"]),
                    "bars_held": pos["bars_held"],
                }
                _trades.append({**exit_data, "timestamp": now, "action": "exit"})
                if len(_trades) > 50:
                    _trades.pop(0)
                _positions[strategy] = None
                messages.append({"type": "exit", "data": exit_data, "timestamp": now})
                break

    # P&L update every tick
    messages.append({
        "type": "pnl",
        "data": {
            "daily": round(_daily_pnl, 2),
            "monthly": round(_monthly_pnl, 2),
            "daily_limit": -3000.0,
            "monthly_limit": -4500.0,
        },
        "timestamp": now,
    })

    return messages
