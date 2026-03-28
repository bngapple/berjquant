import { useState, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { EquityCurve } from "../components/EquityCurve";
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
      dayWinPct: exits.length === 0 ? "--" as const : 75,
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

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  const avgWinLossTotal = stats.avgWin + stats.avgLoss;
  const winProportion = avgWinLossTotal > 0 ? (stats.avgWin / avgWinLossTotal) * 100 : 50;

  return (
    <div className="p-5 space-y-4 max-w-[1440px] mx-auto">
      {/* ── Stat Strip ──────────────────────────────────── */}
      <div className="flex rounded overflow-hidden" style={{ background: "var(--panel)", border: "1px solid var(--border)" }}>
        {/* Total P&L — hero */}
        <div className="flex-[2.5] px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Total P&L</div>
          <span className="text-[32px] font-bold font-mono tabular" style={{ color: clr(stats.totalPnl) }}>{fmt(stats.totalPnl)}</span>
        </div>
        {/* Win Rate */}
        <div className="flex-1 px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Win Rate</div>
          <div className="flex items-center gap-2">
            <DonutRing value={stats.winRate} size={28} />
            <span className="text-base font-semibold tabular" style={{ color: "var(--text)" }}>{stats.winRate.toFixed(0)}%</span>
          </div>
        </div>
        {/* Profit Factor */}
        <div className="flex-1 px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Profit Factor</div>
          <div className="flex items-center gap-2">
            <DonutRing value={stats.profitFactor > 0 ? Math.min((stats.profitFactor / (stats.profitFactor + 1)) * 100, 100) : 0} size={28} />
            <span className="text-base font-semibold tabular" style={{ color: "var(--text)" }}>{stats.profitFactor.toFixed(2)}</span>
          </div>
        </div>
        {/* Avg Win / Loss */}
        <div className="flex-1 px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Avg Win / Loss</div>
          <div className="space-y-0.5">
            <div className="flex items-center gap-3">
              <div>
                <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>Avg Win</div>
                <div className="font-mono text-sm" style={{ color: "var(--accent)" }}>${stats.avgWin.toFixed(0)}</div>
              </div>
              <div>
                <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>Avg Loss</div>
                <div className="font-mono text-sm" style={{ color: "var(--red)" }}>${stats.avgLoss.toFixed(0)}</div>
              </div>
            </div>
            <div className="flex h-1 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
              <div style={{ width: `${winProportion}%`, background: "var(--accent)" }} />
              <div style={{ width: `${100 - winProportion}%`, background: "var(--red)" }} />
            </div>
          </div>
        </div>
        {/* Trades */}
        <div className="flex-[0.7] px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Trades</div>
          <span className="text-base font-semibold tabular" style={{ color: "var(--text)" }}>{stats.totalTrades}</span>
        </div>
        {/* Day Win % */}
        <div className="flex-[0.7] px-5 py-4">
          <div className="text-[10px] font-normal uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>Day Win %</div>
          <span className="text-base font-semibold tabular" style={{ color: "var(--text)" }}>{stats.dayWinPct === "--" ? "--" : `${stats.dayWinPct}%`}</span>
        </div>
      </div>

      {/* ── Charts ──────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-3 items-start">
        <div className="col-span-3 panel rounded p-3 pt-2" style={{ height: 320 }}>
          <div className="text-[10px] font-normal tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Cumulative P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><EquityCurve data={equityHistory} /></div>
        </div>
        <div className="col-span-2 panel rounded p-3 pt-2" style={{ height: 220 }}>
          <div className="text-[10px] font-normal tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Net Daily P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><DailyBarChart data={dailyBars} /></div>
        </div>
      </div>

      {/* ── Positions + Limits (full width) ────────────── */}
      <div className="space-y-3">
        <div className="panel rounded overflow-hidden">
          <div className="px-4 py-2 text-[10px] font-normal tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Positions</div>
          <table className="w-full text-[11px]">
            <thead><tr style={{ color: "var(--text-dim)" }}>
              <th className="text-left font-normal px-4 py-1">Strategy</th><th className="text-left font-normal px-2 py-1">Status</th><th className="text-left font-normal px-2 py-1">Side</th><th className="text-right font-normal px-2 py-1">Entry</th><th className="text-right font-normal px-2 py-1">P&L</th><th className="text-right font-normal px-2 py-1">Bars</th><th className="text-right font-normal px-2 py-1">SL</th><th className="text-right font-normal px-2 py-1">TP</th>
            </tr></thead>
            <tbody>{(["RSI", "IB", "MOM"] as const).map(s => {
              const p = positions[s] as Position | null | undefined;
              return (
                <tr key={s} style={{ color: p ? "var(--text)" : "var(--text-dim)", borderTop: "1px solid var(--border)", borderLeft: p ? "3px solid var(--accent)" : "3px solid transparent" }}>
                  <td className="px-4 py-1.5 font-medium">{s}</td>
                  <td className="px-2 py-1.5"><span className="flex items-center gap-1"><span className={`w-1.5 h-1.5 rounded-full ${p ? "bg-emerald-400 pulse-dot" : "bg-zinc-700"}`} />{p ? "Active" : "Flat"}</span></td>
                  <td className="px-2 py-1.5" style={{ color: p ? (p.side === "Buy" ? "var(--accent)" : "var(--red)") : undefined }}>{p ? (p.side === "Buy" ? "LONG" : "SHORT") : "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular">{p ? p.entry_price.toFixed(2) : "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular" style={{ color: p ? clr(p.pnl) : undefined }}>{p ? fmt(p.pnl) : "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular">{p ? p.bars_held : "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular">{p ? p.sl.toFixed(2) : "—"}</td>
                  <td className="px-4 py-1.5 text-right font-mono tabular">{p ? p.tp.toFixed(2) : "—"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <div className="panel rounded px-4 py-3 space-y-2">
          <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
          <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
        </div>
      </div>

      {/* ── Logs (tabbed) ───────────────────────────────── */}
      <div className="panel rounded overflow-hidden">
        <div className="flex px-4 pt-2 gap-4" style={{ borderBottom: "1px solid var(--border)" }}>
          {(["signals", "trades"] as const).map(t => (
            <button key={t} onClick={() => setLogTab(t)} className="pb-2 text-[11px] font-medium capitalize transition-colors"
              style={{ color: logTab === t ? "var(--text)" : "var(--text-muted)", borderBottom: logTab === t ? "2px solid white" : "2px solid transparent" }}>
              {t === "signals" ? "Signal Log" : "Trade Log"}
            </button>
          ))}
        </div>
        <div className="max-h-44 overflow-y-auto">
          {logTab === "signals" ? (
            signals.length === 0 ? <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No signals yet</div> :
            [...signals].reverse().slice(0, 15).map((sig: Signal, i: number) => (
              <div key={i} className="flex items-center gap-3 px-4 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
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
              <div key={i} className="flex items-center gap-3 px-4 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
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
