import { useState, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";

export function Calendar() {
  const { trades } = useLayoutData();
  const [offset, setOffset] = useState(0);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);

  // Seed mock + real trade data
  const dailyPnl = useMemo(() => {
    const map: Record<string, { pnl: number; trades: number; wins: number; losses: number }> = {};
    const today = new Date();
    for (let i = 1; i <= 45; i++) {
      const d = new Date(today); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      const key = d.toISOString().split("T")[0];
      const pnl = Math.round((Math.random() - 0.35) * 600);
      const tradeCount = Math.floor(Math.random() * 6) + 1;
      const wins = pnl > 0 ? Math.ceil(tradeCount * 0.6) : Math.floor(tradeCount * 0.3);
      map[key] = { pnl, trades: tradeCount, wins, losses: tradeCount - wins };
    }
    for (const t of trades) {
      if (t.action === "exit" && t.timestamp) {
        const key = t.timestamp.split("T")[0];
        if (!map[key]) map[key] = { pnl: 0, trades: 0, wins: 0, losses: 0 };
        map[key].pnl += t.pnl ?? 0;
        map[key].trades++;
        if ((t.pnl ?? 0) > 0) map[key].wins++; else map[key].losses++;
      }
    }
    return map;
  }, [trades]);

  const { label, weeks, monthKey: _monthKey } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear(), m = now.getMonth();
    const label = now.toLocaleString("en-US", { month: "long", year: "numeric" });
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const first = new Date(y, m, 1);
    const startDay = (first.getDay() + 6) % 7; // Mon=0

    type Cell = { day: number; key: string; data: { pnl: number; trades: number; wins: number; losses: number } | null };
    const cells: Cell[] = [];
    for (let i = 0; i < startDay; i++) cells.push({ day: 0, key: `p${i}`, data: null });
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      cells.push({ day: d, key, data: dailyPnl[key] ?? null });
    }
    // Group into weeks, compute weekly totals
    const weeks: { cells: Cell[]; weekPnl: number }[] = [];
    for (let i = 0; i < cells.length; i += 7) {
      const slice = cells.slice(i, i + 7);
      while (slice.length < 7) slice.push({ day: 0, key: `e${slice.length}`, data: null });
      const weekPnl = slice.reduce((s, c) => s + (c.data?.pnl ?? 0), 0);
      weeks.push({ cells: slice, weekPnl });
    }
    return { label, weeks, monthKey: `${y}-${m}` };
  }, [offset, dailyPnl]);

  // Summary
  const summary = useMemo(() => {
    let tradingDays = 0, winners = 0, losers = 0, netPnl = 0;
    for (const w of weeks) for (const c of w.cells) {
      if (c.data) { tradingDays++; netPnl += c.data.pnl; if (c.data.pnl > 0) winners++; else if (c.data.pnl < 0) losers++; }
    }
    return { tradingDays, winners, losers, netPnl };
  }, [weeks]);

  // Selected day trades (mock)
  const dayTrades = useMemo(() => {
    if (!selectedDay) return [];
    const d = dailyPnl[selectedDay];
    if (!d) return [];
    const strategies = ["RSI", "IB", "MOM"];
    const result = [];
    for (let i = 0; i < d.trades; i++) {
      const strat = strategies[i % 3];
      const side = Math.random() > 0.5 ? "Buy" : "Sell";
      const entry = 21000 + Math.round(Math.random() * 1000);
      const pnl = i < d.wins ? Math.abs(Math.random() * 200) : -Math.abs(Math.random() * 150);
      result.push({ time: `${9 + Math.floor(i * 1.5)}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`, strategy: strat, side, entry: entry.toFixed(2), exit: (entry + (side === "Buy" ? pnl / 2 : -pnl / 2)).toFixed(2), pnl: Math.round(pnl * 100) / 100, slippage: (Math.random() * 1.5).toFixed(2) });
    }
    return result;
  }, [selectedDay, dailyPnl]);

  const bg = (pnl: number | null) => {
    if (pnl === null) return "transparent";
    if (pnl === 0) return "rgba(255,255,255,0.02)";
    if (pnl > 0) return `rgba(0,212,170,${Math.min(0.08 + (pnl / 800) * 0.25, 0.35)})`;
    return `rgba(239,68,68,${Math.min(0.08 + (Math.abs(pnl) / 800) * 0.25, 0.35)})`;
  };

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(0)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";
  const cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <div className="p-5 max-w-[1100px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>Performance Calendar</span>
        <div className="flex items-center gap-4">
          <button onClick={() => setOffset(o => o - 1)} className="text-sm px-2 py-0.5 rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>&larr;</button>
          <span className="text-sm font-medium w-40 text-center" style={{ color: "var(--text)" }}>{label}</span>
          <button onClick={() => setOffset(o => Math.min(o + 1, 0))} disabled={offset >= 0} className="text-sm px-2 py-0.5 rounded disabled:opacity-20" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>&rarr;</button>
        </div>
      </div>

      {/* Calendar grid */}
      <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
        <thead>
          <tr>
            {cols.map(c => <th key={c} className="text-[11px] font-normal py-2 text-center" style={{ color: "var(--text-dim)" }}>{c}</th>)}
            <th className="text-[11px] font-normal py-2 text-center w-20" style={{ color: "var(--text-dim)" }}>Week</th>
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, wi) => (
            <tr key={wi}>
              {week.cells.map(cell => (
                <td key={cell.key}
                  className={`relative h-16 p-1.5 align-top cursor-pointer transition-colors ${selectedDay === cell.key ? "ring-1 ring-white/20" : ""}`}
                  style={{ background: bg(cell.data?.pnl ?? null), border: "1px solid var(--border)" }}
                  onClick={() => cell.data && setSelectedDay(selectedDay === cell.key ? null : cell.key)}>
                  {cell.day > 0 && (
                    <>
                      <span className="text-[9px] absolute top-1 left-1.5" style={{ color: "var(--text-dim)" }}>{cell.day}</span>
                      {cell.data && (
                        <div className="flex items-center justify-center h-full">
                          <span className="font-mono font-semibold text-sm tabular" style={{ color: clr(cell.data.pnl) }}>
                            {fmt(cell.data.pnl)}
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </td>
              ))}
              <td className="h-16 text-center font-mono text-[11px] tabular" style={{ color: week.weekPnl !== 0 ? clr(week.weekPnl) : "var(--text-dim)", border: "1px solid var(--border)" }}>
                {week.weekPnl !== 0 ? fmt(week.weekPnl) : "\u2014"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Summary row */}
      <div className="flex items-center gap-6 mt-3 text-[11px]" style={{ color: "var(--text-muted)" }}>
        <span>{summary.tradingDays} trading days</span>
        <span style={{ color: "var(--accent)" }}>{summary.winners} winners</span>
        <span style={{ color: "var(--red)" }}>{summary.losers} losers</span>
        <span className="ml-auto font-mono font-semibold text-sm tabular" style={{ color: clr(summary.netPnl) }}>
          Net: {fmt(summary.netPnl)}
        </span>
      </div>

      {/* Selected day detail */}
      {selectedDay && dayTrades.length > 0 && (
        <div className="mt-4 panel rounded overflow-hidden">
          <div className="px-4 py-2 text-[10px] uppercase tracking-wider flex items-center justify-between" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
            <span>Trades for {selectedDay}</span>
            <button onClick={() => setSelectedDay(null)} className="text-zinc-500 hover:text-zinc-300 text-xs">Close</button>
          </div>
          <table className="w-full text-[11px]">
            <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
              <th className="text-left font-normal px-4 py-1.5">Time</th>
              <th className="text-left font-normal px-3 py-1.5">Strategy</th>
              <th className="text-left font-normal px-3 py-1.5">Side</th>
              <th className="text-right font-normal px-3 py-1.5">Entry</th>
              <th className="text-right font-normal px-3 py-1.5">Exit</th>
              <th className="text-right font-normal px-3 py-1.5">P&L</th>
              <th className="text-right font-normal px-4 py-1.5">Slippage</th>
            </tr></thead>
            <tbody>
              {dayTrades.map((t, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="px-4 py-1.5 font-mono tabular" style={{ color: "var(--text-muted)" }}>{t.time}</td>
                  <td className="px-3 py-1.5 font-medium" style={{ color: "var(--text)" }}>{t.strategy}</td>
                  <td className="px-3 py-1.5" style={{ color: t.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{t.side === "Buy" ? "LONG" : "SHORT"}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: "var(--text)" }}>{t.entry}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: "var(--text)" }}>{t.exit}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: clr(t.pnl) }}>{t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}</td>
                  <td className="px-4 py-1.5 text-right font-mono tabular" style={{ color: "var(--text-dim)" }}>{t.slippage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
