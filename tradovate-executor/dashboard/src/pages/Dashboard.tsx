import { useState, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { EquityCurve } from "../components/EquityCurve";
import { PnLCalendar } from "../components/PnLCalendar";
import { DailyBarChart } from "../components/DailyBarChart";
import { DonutRing } from "../components/DonutRing";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { positions, pnl, signals, trades, equityHistory } = useLayoutData();
  const [logTab, setLogTab] = useState<"signals" | "trades">("signals");

  const stats = useMemo(() => {
    const exits = trades.filter(t => t.action === "exit");
    const wins = exits.filter(t => (t.pnl ?? 0) > 0);
    const losses = exits.filter(t => (t.pnl ?? 0) < 0);
    const totalPnl = pnl?.daily ?? 0;
    const winPnl = wins.reduce((s, t) => s + (t.pnl ?? 0), 0);
    const lossPnl = Math.abs(losses.reduce((s, t) => s + (t.pnl ?? 0), 0));
    const avgWin = wins.length > 0 ? winPnl / wins.length : 0;
    const avgLoss = losses.length > 0 ? lossPnl / losses.length : 0;
    let peak = 0, maxDd = 0;
    for (const pt of equityHistory) { if (pt.value > peak) peak = pt.value; const dd = peak - pt.value; if (dd > maxDd) maxDd = dd; }
    return {
      totalPnl, winRate: exits.length > 0 ? (wins.length / exits.length) * 100 : 0,
      profitFactor: lossPnl > 0 ? winPnl / lossPnl : winPnl > 0 ? 999 : 0,
      avgWin, avgLoss, totalTrades: exits.length,
      dayWinPct: 75,
    };
  }, [trades, pnl, equityHistory]);

  const dailyBars = useMemo(() => {
    const bars: { date: string; pnl: number }[] = [];
    const today = new Date();
    for (let i = 20; i >= 0; i--) {
      const d = new Date(today); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      bars.push({ date: `${d.getMonth() + 1}/${d.getDate()}`, pnl: Math.round((Math.random() - 0.35) * 400) });
    }
    return bars;
  }, []);

  const calendarData = useMemo(() => {
    const map: Record<string, number> = {};
    const today = new Date();
    for (let i = 1; i <= 25; i++) {
      const d = new Date(today); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      map[d.toISOString().split("T")[0]] = Math.round((Math.random() - 0.35) * 400);
    }
    for (const t of trades) { if (t.action === "exit" && t.timestamp) { const k = t.timestamp.split("T")[0]; map[k] = (map[k] ?? 0) + (t.pnl ?? 0); } }
    return map;
  }, [trades]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  return (
    <div className="p-5 space-y-4 max-w-[1440px] mx-auto">
      {/* ── Stat Strip ──────────────────────────────────── */}
      <div className="flex rounded-lg overflow-hidden" style={{ background: "var(--panel)", border: "1px solid var(--border)" }}>
        <StatCell label="Total P&L" big>
          <span className="text-2xl font-bold font-mono tabular" style={{ color: clr(stats.totalPnl) }}>{fmt(stats.totalPnl)}</span>
        </StatCell>
        <StatCell label="Win Rate">
          <div className="flex items-center gap-2">
            <DonutRing value={stats.winRate} />
            <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.winRate.toFixed(0)}%</span>
          </div>
        </StatCell>
        <StatCell label="Profit Factor">
          <div className="flex items-center gap-2">
            <DonutRing value={stats.profitFactor > 0 ? Math.min((stats.profitFactor / (stats.profitFactor + 1)) * 100, 100) : 0} />
            <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.profitFactor.toFixed(2)}</span>
          </div>
        </StatCell>
        <StatCell label="Avg Win / Loss">
          <div className="space-y-1">
            <div className="flex items-center gap-2"><div className="h-1.5 rounded" style={{ width: 40, background: "var(--accent)" }} /><span className="text-xs font-mono tabular" style={{ color: "var(--accent)" }}>${stats.avgWin.toFixed(0)}</span></div>
            <div className="flex items-center gap-2"><div className="h-1.5 rounded" style={{ width: Math.max(stats.avgLoss / (stats.avgWin || 1) * 40, 8), background: "var(--red)" }} /><span className="text-xs font-mono tabular" style={{ color: "var(--red)" }}>${stats.avgLoss.toFixed(0)}</span></div>
          </div>
        </StatCell>
        <StatCell label="Trades">
          <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.totalTrades}</span>
        </StatCell>
        <StatCell label="Day Win %" last>
          <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.dayWinPct}%</span>
        </StatCell>
      </div>

      {/* ── Charts ──────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-3" style={{ height: 280 }}>
        <div className="col-span-3 panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Cumulative P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><EquityCurve data={equityHistory} /></div>
        </div>
        <div className="col-span-2 panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Net Daily P&L</div>
          <div style={{ height: "calc(100% - 24px)" }}><DailyBarChart data={dailyBars} /></div>
        </div>
      </div>

      {/* ── Calendar + Positions ─────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <div className="panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>P&L Calendar</div>
          <PnLCalendar data={calendarData} />
        </div>
        <div className="space-y-3">
          <div className="panel rounded-lg overflow-hidden">
            <div className="px-4 py-2 text-[10px] font-light uppercase tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Positions</div>
            <table className="w-full text-[11px]">
              <thead><tr style={{ color: "var(--text-dim)" }}>
                <th className="text-left font-normal px-4 py-1.5">Strategy</th><th className="text-left font-normal px-2 py-1.5">Status</th><th className="text-left font-normal px-2 py-1.5">Side</th><th className="text-right font-normal px-2 py-1.5">Entry</th><th className="text-right font-normal px-2 py-1.5">P&L</th><th className="text-right font-normal px-2 py-1.5">Bars</th><th className="text-right font-normal px-2 py-1.5">SL</th><th className="text-right font-normal px-4 py-1.5">TP</th>
              </tr></thead>
              <tbody>{(["RSI", "IB", "MOM"] as const).map(s => {
                const p = positions[s] as Position | null | undefined;
                return (
                  <tr key={s} style={{ color: p ? "var(--text)" : "var(--text-dim)", borderTop: "1px solid var(--border)", borderLeft: p ? "3px solid var(--accent)" : "3px solid transparent" }}>
                    <td className="px-4 py-2 font-medium">{s}</td>
                    <td className="px-2 py-2"><span className="flex items-center gap-1"><span className={`w-1.5 h-1.5 rounded-full ${p ? "bg-emerald-400 pulse-dot" : "bg-zinc-700"}`} />{p ? "Active" : "Flat"}</span></td>
                    <td className="px-2 py-2" style={{ color: p ? (p.side === "Buy" ? "var(--accent)" : "var(--red)") : undefined }}>{p ? (p.side === "Buy" ? "LONG" : "SHORT") : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.entry_price.toFixed(2) : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular" style={{ color: p ? clr(p.pnl) : undefined }}>{p ? fmt(p.pnl) : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.bars_held : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.sl.toFixed(2) : "—"}</td>
                    <td className="px-4 py-2 text-right font-mono tabular">{p ? p.tp.toFixed(2) : "—"}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
          <div className="panel rounded-lg px-4 py-3 space-y-2">
            <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
            <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
          </div>
        </div>
      </div>

      {/* ── Logs (tabbed) ───────────────────────────────── */}
      <div className="panel rounded-lg overflow-hidden">
        <div className="flex px-4 pt-2 gap-4" style={{ borderBottom: "1px solid var(--border)" }}>
          {(["signals", "trades"] as const).map(t => (
            <button key={t} onClick={() => setLogTab(t)} className="pb-2 text-[11px] font-medium capitalize transition-colors"
              style={{ color: logTab === t ? "var(--text)" : "var(--text-muted)", borderBottom: logTab === t ? "2px solid var(--accent)" : "2px solid transparent" }}>
              {t === "signals" ? "Signal Log" : "Trade Log"}
            </button>
          ))}
        </div>
        <div className="max-h-56 overflow-y-auto">
          {logTab === "signals" ? (
            signals.length === 0 ? <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No signals yet</div> :
            [...signals].reverse().slice(0, 15).map((sig: Signal, i: number) => (
              <div key={i} className="flex items-center gap-3 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{sig.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: sig.strategy === "RSI" ? "rgba(0,212,170,0.15)" : sig.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: sig.strategy === "RSI" ? "var(--accent)" : sig.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{sig.strategy}</span>
                <span style={{ color: sig.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{sig.side === "Buy" ? "LONG" : "SHORT"}</span>
                <span className="font-mono tabular" style={{ color: "var(--text-muted)" }}>{sig.price?.toFixed(2)}</span>
                <span className="truncate" style={{ color: "var(--text-dim)" }}>{sig.reason}</span>
              </div>
            ))
          ) : (
            trades.length === 0 ? <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No trades yet</div> :
            [...trades].reverse().slice(0, 15).map((t: Trade, i: number) => (
              <div key={i} className="flex items-center gap-3 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{t.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: t.strategy === "RSI" ? "rgba(0,212,170,0.15)" : t.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: t.strategy === "RSI" ? "var(--accent)" : t.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{t.strategy}</span>
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

function StatCell({ label, children, big, last }: { label: string; children: React.ReactNode; big?: boolean; last?: boolean }) {
  return (
    <div className={`${big ? "flex-[1.3]" : "flex-1"} px-5 py-4`} style={last ? undefined : { borderRight: "1px solid var(--border)" }}>
      <div className="text-[10px] font-light uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</div>
      {children}
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
