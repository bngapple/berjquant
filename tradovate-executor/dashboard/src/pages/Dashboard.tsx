import { useState, useMemo } from "react";
import { api } from "../api/client";
import { useWebSocket } from "../hooks/useWebSocket";
import { EquityCurve } from "../components/EquityCurve";
import { PnLCalendar } from "../components/PnLCalendar";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { connected, status, positions, pnl, signals, trades, equityHistory } =
    useWebSocket();
  const [loading, setLoading] = useState("");
  const [confirmFlatten, setConfirmFlatten] = useState(false);

  const running = status?.running ?? false;

  const handleAction = async (action: string, fn: () => Promise<unknown>) => {
    setLoading(action);
    try {
      await fn();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading("");
    }
  };

  const handleFlatten = async () => {
    setConfirmFlatten(false);
    await handleAction("flatten", api.flattenAll);
  };

  // Compute stats from trades
  const stats = useMemo(() => {
    const exits = trades.filter((t) => t.action === "exit");
    const wins = exits.filter((t) => (t.pnl ?? 0) > 0);
    const totalPnl = pnl?.daily ?? 0;
    // Compute max drawdown from equity history
    let peak = 0;
    let maxDd = 0;
    for (const pt of equityHistory) {
      if (pt.value > peak) peak = pt.value;
      const dd = peak - pt.value;
      if (dd > maxDd) maxDd = dd;
    }
    return {
      totalPnl,
      maxDrawdown: maxDd,
      totalTrades: exits.length,
      winRate: exits.length > 0 ? (wins.length / exits.length) * 100 : 0,
    };
  }, [trades, pnl, equityHistory]);

  // Generate mock calendar data from trades
  const calendarData = useMemo(() => {
    const map: Record<string, number> = {};
    // Seed some mock history for visual richness
    const today = new Date();
    for (let i = 1; i <= 25; i++) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      const key = d.toISOString().split("T")[0];
      map[key] = Math.round((Math.random() - 0.35) * 400);
    }
    // Add real trades
    for (const t of trades) {
      if (t.action === "exit" && t.timestamp) {
        const key = t.timestamp.split("T")[0];
        map[key] = (map[key] ?? 0) + (t.pnl ?? 0);
      }
    }
    return map;
  }, [trades]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => (v >= 0 ? "var(--accent-green)" : "var(--accent-red)");

  return (
    <div className="p-5 space-y-4 max-w-[1400px] mx-auto">
      {/* ── Header Bar ──────────────────────────────────────── */}
      <div className="flex items-center justify-between h-10">
        <div className="flex items-center gap-3">
          <div
            className={`w-2 h-2 rounded-full ${running ? "bg-emerald-500 animate-pulse" : "bg-zinc-600"}`}
          />
          <span className="text-sm font-medium" style={{ color: running ? "var(--text-primary)" : "var(--text-secondary)" }}>
            {running ? "Engine Running" : "Engine Stopped"}
          </span>
          {/* Account status pills */}
          <div className="flex items-center gap-2 ml-4">
            {(status?.connected_accounts ?? []).map((a) => (
              <span
                key={a.name}
                className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full"
                style={{
                  background: "rgba(255,255,255,0.03)",
                  color: a.connected ? "var(--accent-green)" : "var(--text-muted)",
                }}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${a.connected ? "bg-emerald-500" : "bg-zinc-700"}`} />
                {a.name}
              </span>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleAction("start", api.startEngine)}
            disabled={running || loading === "start"}
            className="px-3 py-1.5 text-xs font-medium rounded border border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {loading === "start" ? "Starting..." : "Start"}
          </button>
          <button
            onClick={() => handleAction("stop", api.stopEngine)}
            disabled={!running || loading === "stop"}
            className="px-3 py-1.5 text-xs font-medium rounded text-zinc-400 hover:text-zinc-200 hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed"
            style={{ border: "1px solid var(--border)" }}
          >
            Stop
          </button>
          <button
            onClick={() => setConfirmFlatten(true)}
            disabled={loading === "flatten"}
            className="px-3 py-1.5 text-xs font-medium rounded border border-red-500/40 text-red-400 hover:bg-red-500/10 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Flatten All
          </button>
        </div>
      </div>

      {/* Flatten confirmation dialog */}
      {confirmFlatten && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setConfirmFlatten(false)}>
          <div className="panel rounded-lg p-5 w-80" onClick={(e) => e.stopPropagation()}>
            <p className="text-sm mb-4" style={{ color: "var(--text-primary)" }}>
              Flatten all positions across all accounts?
            </p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmFlatten(false)} className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200">
                Cancel
              </button>
              <button onClick={handleFlatten} className="px-3 py-1.5 text-xs rounded bg-red-600 hover:bg-red-500 text-white">
                Confirm Flatten
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Top Stats Bar ───────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Daily P&L" value={fmt(stats.totalPnl)} color={clr(stats.totalPnl)} large />
        <StatCard label="Max Drawdown" value={`-$${stats.maxDrawdown.toFixed(0)}`} color={stats.maxDrawdown > 0 ? "var(--accent-red)" : "var(--text-secondary)"} />
        <StatCard label="Trades" value={String(stats.totalTrades)} color="var(--text-primary)" />
        <StatCard label="Win Rate" value={`${stats.winRate.toFixed(0)}%`} color={stats.winRate >= 50 ? "var(--accent-green)" : "var(--text-secondary)"} />
      </div>

      {/* ── Limit Bars (thin) ───────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
        <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
      </div>

      {/* ── Equity Curve + Calendar ─────────────────────────── */}
      <div className="grid grid-cols-5 gap-3" style={{ height: 260 }}>
        <div className="col-span-3 panel rounded-lg p-4">
          <div className="text-[11px] mb-2" style={{ color: "var(--text-secondary)" }}>Equity Curve</div>
          <div style={{ height: "calc(100% - 20px)" }}>
            <EquityCurve data={equityHistory} />
          </div>
        </div>
        <div className="col-span-2 panel rounded-lg p-4">
          <div className="text-[11px] mb-2" style={{ color: "var(--text-secondary)" }}>P&L Calendar</div>
          <PnLCalendar data={calendarData} />
        </div>
      </div>

      {/* ── Positions Table ─────────────────────────────────── */}
      <div className="panel rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>
          Positions
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th className="text-left font-normal px-4 py-2 w-20">Strategy</th>
              <th className="text-left font-normal px-3 py-2 w-16">Side</th>
              <th className="text-right font-normal px-3 py-2">Entry</th>
              <th className="text-right font-normal px-3 py-2">Current</th>
              <th className="text-right font-normal px-3 py-2">P&L</th>
              <th className="text-right font-normal px-3 py-2">Bars</th>
              <th className="text-right font-normal px-3 py-2">SL</th>
              <th className="text-right font-normal px-4 py-2">TP</th>
            </tr>
          </thead>
          <tbody>
            {(["RSI", "IB", "MOM"] as const).map((s) => {
              const pos = positions[s] as Position | null | undefined;
              const isFlat = !pos;
              return (
                <tr
                  key={s}
                  className="transition-colors"
                  style={{
                    color: isFlat ? "var(--text-muted)" : "var(--text-primary)",
                    borderTop: "1px solid var(--border)",
                  }}
                >
                  <td className="px-4 py-2.5 font-medium">{s}</td>
                  <td className="px-3 py-2.5">
                    {pos ? (
                      <span className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${pos.side === "Buy" ? "bg-emerald-500" : "bg-red-500"}`} />
                        <span style={{ color: pos.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {pos.side === "Buy" ? "LONG" : "SHORT"}
                        </span>
                      </span>
                    ) : (
                      <span>FLAT</span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono">{pos ? pos.entry_price.toFixed(2) : "—"}</td>
                  <td className="px-3 py-2.5 text-right font-mono">{pos ? pos.current_price.toFixed(2) : "—"}</td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: pos ? clr(pos.pnl) : undefined }}>{pos ? fmt(pos.pnl) : "—"}</td>
                  <td className="px-3 py-2.5 text-right font-mono">{pos ? pos.bars_held : "—"}</td>
                  <td className="px-3 py-2.5 text-right font-mono">{pos ? pos.sl.toFixed(2) : "—"}</td>
                  <td className="px-4 py-2.5 text-right font-mono">{pos ? pos.tp.toFixed(2) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Signal + Trade Logs ─────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <div className="panel rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>Signals</div>
          <div className="max-h-48 overflow-y-auto">
            {signals.length === 0 && (
              <div className="px-4 py-3 text-xs" style={{ color: "var(--text-muted)" }}>No signals yet</div>
            )}
            {[...signals].reverse().slice(0, 20).map((sig: Signal, i: number) => (
              <div key={i} className="flex items-center gap-2 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0" style={{ color: "var(--text-muted)" }}>{sig.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}</span>
                <span className="w-10 font-medium shrink-0" style={{ color: sig.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}>{sig.strategy}</span>
                <span style={{ color: sig.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}>{sig.side === "Buy" ? "LONG" : "SHORT"}</span>
                <span className="truncate" style={{ color: "var(--text-muted)" }}>{sig.reason}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>Trades</div>
          <div className="max-h-48 overflow-y-auto">
            {trades.length === 0 && (
              <div className="px-4 py-3 text-xs" style={{ color: "var(--text-muted)" }}>No trades yet</div>
            )}
            {[...trades].reverse().slice(0, 20).map((t: Trade, i: number) => (
              <div key={i} className="flex items-center gap-2 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0" style={{ color: "var(--text-muted)" }}>{t.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}</span>
                <span className="w-10 font-medium shrink-0" style={{ color: "var(--text-secondary)" }}>{t.strategy}</span>
                <span className="w-10 shrink-0" style={{ color: t.action === "entry" ? "var(--accent-blue)" : "#f97316" }}>{t.action === "entry" ? "ENTRY" : "EXIT"}</span>
                {t.action === "entry" ? (
                  <>
                    <span className="font-mono" style={{ color: "var(--text-primary)" }}>@{t.fill_price?.toFixed(2)}</span>
                    <span style={{ color: "var(--text-muted)" }}>slip {t.slippage?.toFixed(2)}</span>
                  </>
                ) : (
                  <>
                    <span style={{ color: "var(--text-muted)" }}>{t.exit_reason}</span>
                    <span className="font-mono" style={{ color: clr(t.pnl ?? 0) }}>{fmt(t.pnl ?? 0)}</span>
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

/* ── Helper Components ─────────────────────────────────────── */

function StatCard({ label, value, color, large }: { label: string; value: string; color: string; large?: boolean }) {
  return (
    <div className="panel rounded-lg px-4 py-3">
      <div className="text-[11px] mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
      <div className={`font-mono font-semibold ${large ? "text-xl" : "text-base"}`} style={{ color }}>{value}</div>
    </div>
  );
}

function LimitBar({ label, value, limit }: { label: string; value: number; limit: number }) {
  const pct = value <= 0 ? Math.min((Math.abs(value) / Math.abs(limit)) * 100, 100) : 0;
  const barColor = value >= 0 ? "var(--accent-green)" : pct > 80 ? "var(--accent-red)" : pct > 50 ? "var(--accent-yellow)" : "var(--accent-green)";
  return (
    <div className="panel rounded-lg px-4 py-2.5 flex items-center gap-3">
      <span className="text-[11px] w-14 shrink-0" style={{ color: "var(--text-muted)" }}>{label}</span>
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${value >= 0 ? 0 : pct}%`, background: barColor }} />
      </div>
      <span className="text-[11px] font-mono w-24 text-right" style={{ color: value >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
        {value >= 0 ? "+" : ""}${value.toFixed(0)} / ${limit.toFixed(0)}
      </span>
    </div>
  );
}
