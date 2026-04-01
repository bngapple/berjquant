import { useState, useMemo, useEffect } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { EquityCurve } from "../components/EquityCurve";
import { DailyBarChart } from "../components/DailyBarChart";
import { DonutRing } from "../components/DonutRing";
import { api } from "../api/client";
import type { Position, Signal, Trade, HistoryStats, FleetAlert, EquityPoint } from "../types";

export function Dashboard() {
  const { positions, pnl, signals, trades, equityHistory } = useLayoutData();
  const [logTab, setLogTab] = useState<"signals" | "trades">("signals");
  const [historyStats, setHistoryStats] = useState<HistoryStats | null>(null);
  const [dailyBars, setDailyBars] = useState<{ date: string; pnl: number }[]>([]);
  const [historyEquity, setHistoryEquity] = useState<EquityPoint[]>([]);
  const [alerts, setAlerts] = useState<FleetAlert[]>([]);

  // Load historical data on mount
  useEffect(() => {
    api.getHistoryStats().then(setHistoryStats).catch(() => {});
    api.getHistoryDaily().then(daily => {
      const bars = Object.entries(daily)
        .sort(([a], [b]) => a.localeCompare(b))
        .slice(-20)
        .map(([date, d]) => ({ date: date.slice(5), pnl: d.pnl }));
      setDailyBars(bars);
    }).catch(() => {});
    api.getHistoryEquity().then(eq => {
      setHistoryEquity(eq.map(p => ({ time: p.date, value: p.value })));
    }).catch(() => {});
    api.getFleetAlerts().then(setAlerts).catch(() => {});
  }, []);

  // Merge historical equity with live equity
  const mergedEquity = useMemo(() => {
    if (equityHistory.length > 0) return equityHistory;
    return historyEquity;
  }, [equityHistory, historyEquity]);

  const stats = useMemo(() => {
    // Use live data if trades are flowing, otherwise use historical
    const exits = trades.filter(t => t.action === "exit");
    if (exits.length > 0) {
      const wins = exits.filter(t => (t.pnl ?? 0) > 0);
      const losses = exits.filter(t => (t.pnl ?? 0) < 0);
      const totalPnl = pnl?.daily ?? 0;
      const winPnl = wins.reduce((s, t) => s + (t.pnl ?? 0), 0);
      const lossPnl = Math.abs(losses.reduce((s, t) => s + (t.pnl ?? 0), 0));
      const avgWin = wins.length > 0 ? winPnl / wins.length : 0;
      const avgLoss = losses.length > 0 ? lossPnl / losses.length : 0;
      return {
        totalPnl, winRate: exits.length > 0 ? (wins.length / exits.length) * 100 : 0,
        profitFactor: lossPnl > 0 ? winPnl / lossPnl : winPnl > 0 ? 999 : 0,
        avgWin, avgLoss, totalTrades: exits.length,
      };
    }
    // Fall back to historical stats
    if (historyStats && historyStats.total_trades > 0) {
      return {
        totalPnl: historyStats.total_pnl,
        winRate: historyStats.win_rate ?? 0,
        profitFactor: historyStats.profit_factor ?? 0,
        avgWin: historyStats.avg_win ?? 0,
        avgLoss: historyStats.avg_loss ?? 0,
        totalTrades: historyStats.total_trades,
      };
    }
    return { totalPnl: 0, winRate: 0, profitFactor: 0, avgWin: 0, avgLoss: 0, totalTrades: 0 };
  }, [trades, pnl, historyStats]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  const avgWinLossTotal = stats.avgWin + stats.avgLoss;
  const winProportion = avgWinLossTotal > 0 ? (stats.avgWin / avgWinLossTotal) * 100 : 50;
  const hasData = stats.totalTrades > 0;

  return (
    <div className="p-5 space-y-3 max-w-[1440px] mx-auto">
      {/* ── Fleet Health Alerts ─────────────────────────── */}
      {alerts.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {alerts.map((a, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px]"
              style={{
                background: a.type === "danger" ? "rgba(239,68,68,0.1)" : a.type === "warning" ? "rgba(245,158,11,0.1)" : "rgba(0,212,170,0.1)",
                color: a.type === "danger" ? "var(--red)" : a.type === "warning" ? "var(--amber)" : "var(--accent)",
                border: `1px solid ${a.type === "danger" ? "rgba(239,68,68,0.15)" : a.type === "warning" ? "rgba(245,158,11,0.15)" : "rgba(0,212,170,0.15)"}`,
              }}>
              <span>{a.type === "success" ? "\u2713" : "\u26A0"}</span>
              <span>{a.account}: {a.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Stat Strip ──────────────────────────────────── */}
      <div className="flex gap-3">
        <div className="panel rounded flex-[2.5] px-6 py-5">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Total P&L</div>
          <span className="text-[32px] font-bold font-mono tabular leading-none" style={{ color: clr(stats.totalPnl) }}>{fmt(stats.totalPnl)}</span>
        </div>
        <div className="panel rounded flex-1 px-4 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Win Rate</div>
          <div className="flex items-center gap-2">
            <DonutRing value={stats.winRate} size={28} />
            <span className="text-[15px] font-semibold tabular" style={{ color: "var(--text)" }}>{hasData ? `${stats.winRate.toFixed(0)}%` : "--"}</span>
          </div>
        </div>
        <div className="panel rounded flex-1 px-4 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Profit Factor</div>
          <div className="flex items-center gap-2">
            <DonutRing value={stats.profitFactor > 0 ? Math.min((stats.profitFactor / (stats.profitFactor + 1)) * 100, 100) : 0} size={28} />
            <span className="text-[15px] font-semibold tabular" style={{ color: "var(--text)" }}>{hasData ? stats.profitFactor.toFixed(2) : "--"}</span>
          </div>
        </div>
        <div className="panel rounded flex-[1.2] px-4 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Avg Win / Loss</div>
          <div className="space-y-1">
            <div className="text-[13px] font-mono tabular" style={{ color: "var(--accent)" }}>Avg Win: {hasData ? `$${stats.avgWin.toFixed(0)}` : "--"}</div>
            <div className="flex h-[3px] rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
              <div style={{ width: `${winProportion}%`, background: "var(--accent)" }} />
              <div style={{ width: `${100 - winProportion}%`, background: "var(--red)" }} />
            </div>
            <div className="text-[13px] font-mono tabular" style={{ color: "var(--red)" }}>Avg Loss: {hasData ? `$${stats.avgLoss.toFixed(0)}` : "--"}</div>
          </div>
        </div>
        <div className="panel rounded flex-[0.6] px-4 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Trades</div>
          <span className="text-[15px] font-semibold tabular" style={{ color: "var(--text)" }}>{stats.totalTrades}</span>
        </div>
      </div>

      {/* ── Charts ──────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-3 items-start">
        <div className="col-span-3 panel rounded px-4 pb-3 pt-2" style={{ height: 340 }}>
          <div className="text-[10px] font-normal tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Cumulative P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><EquityCurve data={mergedEquity} /></div>
        </div>
        <div className="col-span-2 panel rounded px-4 pb-3 pt-2" style={{ height: 220 }}>
          <div className="text-[10px] font-normal tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Net Daily P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><DailyBarChart data={dailyBars} /></div>
        </div>
      </div>

      {/* ── Positions + Limits ────────────────────────── */}
      <div className="space-y-3">
        <div className="panel rounded overflow-hidden">
          <div className="px-3 py-1.5 text-[10px] font-normal tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Positions</div>
          <table className="w-full text-[11px]">
            <thead><tr style={{ color: "var(--text-dim)" }}>
              <th className="text-left font-normal px-3 py-1">Strategy</th><th className="text-left font-normal px-2 py-1">Status</th><th className="text-left font-normal px-2 py-1">Side</th><th className="text-right font-normal px-2 py-1">Entry</th><th className="text-right font-normal px-2 py-1">P&L</th><th className="text-right font-normal px-2 py-1">Bars</th><th className="text-right font-normal px-2 py-1">SL</th><th className="text-right font-normal px-2 py-1">TP</th>
            </tr></thead>
            <tbody>{(["RSI", "IB", "MOM"] as const).map(s => {
              const p = positions[s] as Position | null | undefined;
              return (
                <tr key={s} style={{ color: p ? "var(--text)" : "var(--text-dim)", borderTop: "1px solid var(--border)", borderLeft: p ? "3px solid var(--accent)" : "3px solid transparent" }}>
                  <td className="px-3 py-1 font-medium">{s}</td>
                  <td className="px-2 py-1"><span className="flex items-center gap-1"><span className={`w-1.5 h-1.5 rounded-full ${p ? "bg-emerald-400 pulse-dot" : "bg-zinc-700"}`} />{p ? "Active" : "Flat"}</span></td>
                  <td className="px-2 py-1" style={{ color: p ? (p.side === "Buy" ? "var(--accent)" : "var(--red)") : undefined }}>{p ? (p.side === "Buy" ? "LONG" : "SHORT") : "\u2014"}</td>
                  <td className="px-2 py-1 text-right font-mono tabular">{p ? p.entry_price.toFixed(2) : "\u2014"}</td>
                  <td className="px-2 py-1 text-right font-mono tabular" style={{ color: p ? clr(p.pnl) : undefined }}>{p ? fmt(p.pnl) : "\u2014"}</td>
                  <td className="px-2 py-1 text-right font-mono tabular">{p ? p.bars_held : "\u2014"}</td>
                  <td className="px-2 py-1 text-right font-mono tabular">{p ? p.sl.toFixed(2) : "\u2014"}</td>
                  <td className="px-3 py-1 text-right font-mono tabular">{p ? p.tp.toFixed(2) : "\u2014"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <div className="panel rounded px-3 py-2.5 space-y-2">
          <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
          <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
        </div>
      </div>

      {/* ── Logs (tabbed) ───────────────────────────────── */}
      <div className="panel rounded overflow-hidden">
        <div className="flex px-3 pt-1.5 gap-4" style={{ borderBottom: "1px solid var(--border)" }}>
          {(["signals", "trades"] as const).map(t => (
            <button key={t} onClick={() => setLogTab(t)} className="pb-1.5 text-[11px] font-medium capitalize transition-colors"
              style={{ color: logTab === t ? "var(--text)" : "var(--text-muted)", borderBottom: logTab === t ? "2px solid white" : "2px solid transparent" }}>
              {t === "signals" ? "Signal Log" : "Trade Log"}
            </button>
          ))}
        </div>
        <div className="max-h-44 overflow-y-auto">
          {logTab === "signals" ? (
            signals.length === 0 ? <div className="p-3 text-[13px]" style={{ color: "var(--text-dim)" }}>No signals yet</div> :
            [...signals].reverse().slice(0, 15).map((sig: Signal, i: number) => (
              <div key={i} className="flex items-center gap-3 px-3 py-[5px] text-[13px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{sig.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: sig.strategy === "RSI" ? "rgba(0,212,170,0.15)" : sig.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: sig.strategy === "RSI" ? "#00b894" : sig.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{sig.strategy}</span>
                <span style={{ color: sig.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{sig.side === "Buy" ? "LONG" : "SHORT"}</span>
                <span className="font-mono tabular" style={{ color: "var(--text-muted)" }}>{sig.price?.toFixed(2)}</span>
                <span className="truncate" style={{ color: "var(--text-dim)" }}>{sig.reason}</span>
              </div>
            ))
          ) : (
            trades.length === 0 ? <div className="p-3 text-[13px]" style={{ color: "var(--text-dim)" }}>No trades yet</div> :
            [...trades].reverse().slice(0, 15).map((t: Trade, i: number) => (
              <div key={i} className="flex items-center gap-3 px-3 py-[5px] text-[13px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{t.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: t.strategy === "RSI" ? "rgba(0,212,170,0.15)" : t.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: t.strategy === "RSI" ? "#00b894" : t.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{t.strategy}</span>
                <span style={{ color: t.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{t.side === "Buy" ? "LONG" : "SHORT"}</span>
                {t.action === "entry" ? (
                  <><span className="font-mono tabular" style={{ color: "var(--text)" }}>@{t.fill_price?.toFixed(2)}</span><span style={{ color: "var(--text-dim)" }}>slip {t.slippage?.toFixed(2)}</span></>
                ) : (
                  <><span style={{ color: "var(--text-dim)" }}>{t.exit_reason}</span><span className="font-mono tabular" style={{ color: clr(t.pnl ?? 0) }}>{fmt(t.pnl ?? 0)}</span><span className="font-mono tabular" style={{ color: "var(--text-dim)" }}>{t.bars_held}b</span></>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function LimitBar({ label, value, limit }: { label: string; value: number; limit: number }) {
  const pct = value <= 0 ? Math.min((Math.abs(value) / Math.abs(limit)) * 100, 100) : 0;
  const color = value >= 0 ? "var(--accent)" : pct > 80 ? "var(--red)" : pct > 50 ? "var(--amber)" : "var(--accent)";
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] w-14 shrink-0 font-light" style={{ color: "var(--text-muted)" }}>{label}</span>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${value >= 0 ? 0 : pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-mono w-24 text-right tabular" style={{ color: value >= 0 ? "var(--accent)" : "var(--red)" }}>
        {value >= 0 ? "+" : ""}${value.toFixed(0)} / ${limit.toFixed(0)}
      </span>
    </div>
  );
}
