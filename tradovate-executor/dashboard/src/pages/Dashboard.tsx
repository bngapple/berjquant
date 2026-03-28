import { useState } from "react";
import { api } from "../api/client";
import { useWebSocket } from "../hooks/useWebSocket";
import { StatusBadge } from "../components/StatusBadge";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { connected, status, positions, pnl, signals, trades } =
    useWebSocket();
  const [loading, setLoading] = useState("");

  const running = status?.running ?? false;

  const handleAction = async (
    action: "start" | "stop" | "flatten",
    fn: () => Promise<unknown>
  ) => {
    setLoading(action);
    try {
      await fn();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading("");
    }
  };

  const formatPnl = (value: number) => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}$${value.toFixed(2)}`;
  };

  const pnlColor = (value: number) =>
    value >= 0 ? "text-green-400" : "text-red-400";

  return (
    <div className="space-y-6">
      {/* Engine Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <StatusBadge running={running} connected={connected} />
          <span className="text-lg font-semibold">
            {running ? "Engine Running" : "Engine Stopped"}
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => handleAction("start", api.startEngine)}
            disabled={running || loading === "start"}
            className="px-4 py-2 text-sm font-medium rounded bg-green-700 hover:bg-green-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "start" ? "Starting..." : "Start"}
          </button>
          <button
            onClick={() => handleAction("stop", api.stopEngine)}
            disabled={!running || loading === "stop"}
            className="px-4 py-2 text-sm font-medium rounded bg-gray-700 hover:bg-gray-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "stop" ? "Stopping..." : "Stop"}
          </button>
          <button
            onClick={() => handleAction("flatten", api.flattenAll)}
            disabled={loading === "flatten"}
            className="px-4 py-2 text-sm font-medium rounded bg-red-700 hover:bg-red-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading === "flatten" ? "Flattening..." : "Flatten All"}
          </button>
        </div>
      </div>

      {/* Positions */}
      <div className="grid grid-cols-3 gap-4">
        {(["RSI", "IB", "MOM"] as const).map((strategy) => {
          const pos = positions[strategy] as Position | null | undefined;
          return (
            <div
              key={strategy}
              className="bg-gray-900 rounded-lg border border-gray-800 p-4"
            >
              <div className="flex justify-between items-center mb-3">
                <span className="font-semibold">{strategy}</span>
                {pos ? (
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-medium ${
                      pos.side === "Buy"
                        ? "bg-green-900/50 text-green-300"
                        : "bg-red-900/50 text-red-300"
                    }`}
                  >
                    {pos.side.toUpperCase()}
                  </span>
                ) : (
                  <span className="text-xs text-gray-600 uppercase">Flat</span>
                )}
              </div>
              {pos ? (
                <div className="space-y-1.5 text-sm">
                  <Row label="Entry" value={pos.entry_price.toFixed(2)} mono />
                  <Row
                    label="Current"
                    value={pos.current_price.toFixed(2)}
                    mono
                  />
                  <Row
                    label="P&L"
                    value={formatPnl(pos.pnl)}
                    mono
                    className={pnlColor(pos.pnl)}
                  />
                  <Row label="Bars" value={String(pos.bars_held)} mono />
                  <Row
                    label="SL / TP"
                    value={`${pos.sl.toFixed(2)} / ${pos.tp.toFixed(2)}`}
                    mono
                    className="text-gray-500"
                  />
                </div>
              ) : (
                <p className="text-gray-600 text-sm">No position</p>
              )}
            </div>
          );
        })}
      </div>

      {/* P&L + Copy Accounts */}
      <div className="grid grid-cols-2 gap-4">
        {/* P&L */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-4">Profit & Loss</h3>
          <div className="space-y-4">
            <PnLBar
              label="Daily"
              value={pnl?.daily ?? 0}
              limit={pnl?.daily_limit ?? -3000}
            />
            <PnLBar
              label="Monthly"
              value={pnl?.monthly ?? 0}
              limit={pnl?.monthly_limit ?? -4500}
            />
          </div>
        </div>

        {/* Copy Accounts */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-4">Accounts</h3>
          <div className="space-y-2">
            {(status?.connected_accounts ?? []).map((acct) => (
              <div
                key={acct.name}
                className="flex items-center justify-between text-sm"
              >
                <span className="text-gray-300">{acct.name}</span>
                <span
                  className={`flex items-center gap-1.5 text-xs ${acct.connected ? "text-green-400" : "text-gray-500"}`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${acct.connected ? "bg-green-500" : "bg-gray-600"}`}
                  />
                  {acct.connected ? "Connected" : "Disconnected"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Signal + Trade Logs */}
      <div className="grid grid-cols-2 gap-4">
        {/* Signal Log */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3">Signal Log</h3>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {signals.length === 0 && (
              <p className="text-gray-600 text-xs">No signals yet</p>
            )}
            {[...signals].reverse().map((sig: Signal, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs py-1 border-b border-gray-800/50"
              >
                <span className="text-gray-500 font-mono w-16 shrink-0">
                  {sig.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span
                  className={`w-12 font-medium ${sig.side === "Buy" ? "text-green-400" : "text-red-400"}`}
                >
                  {sig.strategy}
                </span>
                <span
                  className={
                    sig.side === "Buy" ? "text-green-300" : "text-red-300"
                  }
                >
                  {sig.side}
                </span>
                <span className="text-gray-500 truncate">{sig.reason}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Trade Log */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3">Trade Log</h3>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {trades.length === 0 && (
              <p className="text-gray-600 text-xs">No trades yet</p>
            )}
            {[...trades].reverse().map((trade: Trade, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs py-1 border-b border-gray-800/50"
              >
                <span className="text-gray-500 font-mono w-16 shrink-0">
                  {trade.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span className="w-12 font-medium text-gray-300">
                  {trade.strategy}
                </span>
                <span
                  className={`w-10 ${trade.action === "entry" ? "text-blue-400" : "text-orange-400"}`}
                >
                  {trade.action === "entry" ? "ENTRY" : "EXIT"}
                </span>
                {trade.action === "entry" ? (
                  <>
                    <span className="font-mono text-gray-300">
                      @{trade.fill_price?.toFixed(2)}
                    </span>
                    <span className="text-gray-500">
                      slip: {trade.slippage?.toFixed(2)}
                    </span>
                  </>
                ) : (
                  <>
                    <span className="text-gray-500">
                      {trade.exit_reason}
                    </span>
                    <span
                      className={`font-mono ${(trade.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}
                    >
                      {formatPnl(trade.pnl ?? 0)}
                    </span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* Helper components */

function Row({
  label,
  value,
  mono,
  className,
}: {
  label: string;
  value: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-400">{label}</span>
      <span className={`${mono ? "font-mono" : ""} ${className ?? ""}`}>
        {value}
      </span>
    </div>
  );
}

function PnLBar({
  label,
  value,
  limit,
}: {
  label: string;
  value: number;
  limit: number;
}) {
  const pct = Math.min(Math.abs(value / limit) * 100, 100);
  const barColor =
    value >= 0
      ? "bg-green-500"
      : pct > 80
        ? "bg-red-500"
        : pct > 50
          ? "bg-yellow-500"
          : "bg-blue-500";
  const textColor = value >= 0 ? "text-green-400" : "text-red-400";

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <div className="flex gap-3">
          <span className={`font-mono ${textColor}`}>
            {value >= 0 ? "+" : ""}${value.toFixed(2)}
          </span>
          <span className="text-gray-600 font-mono">
            / ${limit.toFixed(0)}
          </span>
        </div>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${value >= 0 ? 0 : pct}%` }}
        />
      </div>
    </div>
  );
}
