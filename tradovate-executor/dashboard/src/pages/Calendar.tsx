import { useState, useMemo, useEffect } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { api } from "../api/client";
import type { DailyPnL } from "../types";

export function Calendar() {
  const { trades } = useLayoutData();
  const [offset, setOffset] = useState(0);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [historyDaily, setHistoryDaily] = useState<DailyPnL>({});

  // Load historical daily P&L on mount
  useEffect(() => {
    api.getHistoryDaily().then(setHistoryDaily).catch(() => {});
  }, []);

  // Merge historical + live trade data
  const dailyPnl = useMemo(() => {
    const map: Record<string, { pnl: number; trades: number; wins: number; losses: number }> = {};

    // Historical from CSV
    for (const [date, d] of Object.entries(historyDaily)) {
      map[date] = { pnl: d.pnl, trades: d.trades, wins: d.wins, losses: d.losses };
    }

    // Live trades on top
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
  }, [trades, historyDaily]);

  const today = new Date();
  const todayKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

  const { label, weeks, monthlyPnl } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear(), m = now.getMonth();
    const label = `${now.toLocaleString("en-US", { month: "long" })} ${y}`;
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const first = new Date(y, m, 1);
    const startDay = first.getDay();

    const prevMonth = new Date(y, m, 0);
    const prevDays = prevMonth.getDate();
    const prevM = prevMonth.getMonth();
    const prevY = prevMonth.getFullYear();

    type Cell = { day: number; key: string; data: { pnl: number; trades: number; wins: number; losses: number } | null; isCurrentMonth: boolean; isToday: boolean };
    const cells: Cell[] = [];

    for (let i = startDay - 1; i >= 0; i--) {
      const d = prevDays - i;
      const key = `${prevY}-${String(prevM + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      cells.push({ day: d, key, data: dailyPnl[key] ?? null, isCurrentMonth: false, isToday: false });
    }

    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      cells.push({ day: d, key, data: dailyPnl[key] ?? null, isCurrentMonth: true, isToday: key === todayKey });
    }

    const remaining = 7 - (cells.length % 7);
    if (remaining < 7) {
      const nextM = (m + 1) % 12;
      const nextY = m === 11 ? y + 1 : y;
      for (let d = 1; d <= remaining; d++) {
        const key = `${nextY}-${String(nextM + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
        cells.push({ day: d, key, data: dailyPnl[key] ?? null, isCurrentMonth: false, isToday: false });
      }
    }

    const weeks: { cells: Cell[]; weekNum: number; weekPnl: number; weekTrades: number }[] = [];
    for (let i = 0; i < cells.length; i += 7) {
      const slice = cells.slice(i, i + 7);
      const weekPnl = slice.reduce((s, c) => s + (c.data?.pnl ?? 0), 0);
      const weekTrades = slice.reduce((s, c) => s + (c.data?.trades ?? 0), 0);
      weeks.push({ cells: slice, weekNum: weeks.length + 1, weekPnl, weekTrades });
    }

    let monthlyPnl = 0;
    for (const c of cells) {
      if (c.isCurrentMonth && c.data) monthlyPnl += c.data.pnl;
    }

    return { label, weeks, monthlyPnl: Math.round(monthlyPnl * 100) / 100 };
  }, [offset, dailyPnl, todayKey]);

  // Selected day trades from history API
  const [dayTrades, setDayTrades] = useState<Record<string, unknown>[]>([]);
  useEffect(() => {
    if (!selectedDay) { setDayTrades([]); return; }
    api.getHistoryTrades(200).then(all => {
      const filtered = all.filter(t => {
        const ts = (t.timestamp as string) || "";
        return ts.startsWith(selectedDay) && t.action === "Exit";
      });
      setDayTrades(filtered);
    }).catch(() => setDayTrades([]));
  }, [selectedDay]);

  const bg = (cell: { data: { pnl: number } | null; isCurrentMonth: boolean }) => {
    if (!cell.data || !cell.isCurrentMonth) return "transparent";
    const pnl = cell.data.pnl;
    if (pnl === 0) return "rgba(255,255,255,0.02)";
    if (pnl > 0) return `rgba(0,212,170,${Math.min(0.06 + (Math.abs(pnl) / 1000) * 0.2, 0.25)})`;
    return `rgba(239,68,68,${Math.min(0.06 + (Math.abs(pnl) / 1000) * 0.2, 0.25)})`;
  };

  const fmtPnl = (v: number) => { const abs = Math.abs(v).toFixed(2); return v < 0 ? `-$${abs}` : `$${abs}`; };
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";
  const cols = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
  const goToday = () => setOffset(0);

  return (
    <div className="p-6 max-w-[1200px] mx-auto">
      <div className="text-center mb-4">
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>Monthly P/L: </span>
        <span className="text-sm font-bold font-mono tabular" style={{ color: clr(monthlyPnl) }}>{fmtPnl(monthlyPnl)}</span>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <button onClick={() => setOffset(o => o - 1)} className="text-sm px-1" style={{ color: "var(--text-muted)" }}>&lt;</button>
          <span className="text-sm font-medium" style={{ color: "var(--text)" }}>{label}</span>
          <button onClick={() => setOffset(o => o + 1)} className="text-sm px-1" style={{ color: "var(--text-muted)" }}>&gt;</button>
        </div>
        <button onClick={goToday} className="text-[11px] px-3 py-1 rounded" style={{ color: "var(--text)", background: "var(--elevated)", border: "1px solid var(--border)" }}>Today</button>
      </div>

      <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
        <thead>
          <tr>
            {cols.map(c => <th key={c} className="text-[11px] font-normal py-2.5 text-center" style={{ color: "var(--text-dim)" }}>{c}</th>)}
            <th className="text-[11px] font-normal py-2.5 text-center w-24" style={{ color: "var(--text-dim)" }}></th>
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, wi) => (
            <tr key={wi}>
              {week.cells.map(cell => (
                <td key={cell.key}
                  className={`relative p-2 align-top cursor-pointer transition-colors ${selectedDay === cell.key ? "ring-1 ring-white/20" : ""}`}
                  style={{ background: bg(cell), border: "1px solid var(--border)", height: 80, borderLeft: cell.isToday ? "2px solid var(--accent)" : undefined }}
                  onClick={() => cell.data && setSelectedDay(selectedDay === cell.key ? null : cell.key)}>
                  <div className="p-0.5 h-full flex flex-col">
                    <span className="text-[10px] leading-none" style={{ color: cell.isCurrentMonth ? "var(--text-dim)" : "rgba(255,255,255,0.15)" }}>{cell.day}</span>
                    {cell.data && cell.isCurrentMonth && (
                      <div className="flex-1 flex flex-col items-center justify-center -mt-1">
                        <span className="font-mono font-bold text-[15px] tabular leading-tight" style={{ color: clr(cell.data.pnl) }}>{fmtPnl(cell.data.pnl)}</span>
                        <span className="text-[10px] mt-0.5" style={{ color: "var(--text-dim)" }}>{cell.data.trades} trades</span>
                      </div>
                    )}
                    {cell.data && !cell.isCurrentMonth && (
                      <div className="flex-1 flex flex-col items-center justify-center -mt-1 opacity-40">
                        <span className="font-mono font-bold text-[14px] tabular leading-tight" style={{ color: clr(cell.data.pnl) }}>{fmtPnl(cell.data.pnl)}</span>
                        <span className="text-[10px] mt-0.5" style={{ color: "var(--text-dim)" }}>{cell.data.trades} trades</span>
                      </div>
                    )}
                  </div>
                </td>
              ))}
              <td className="align-top" style={{ border: "1px solid var(--border)", height: 80 }}>
                <div className="p-1.5 h-full flex flex-col items-center justify-center text-center">
                  <span className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>Week {week.weekNum}</span>
                  <span className="font-mono font-bold text-[14px] tabular leading-tight mt-0.5" style={{ color: week.weekPnl !== 0 ? clr(week.weekPnl) : "var(--text-dim)" }}>
                    {week.weekPnl !== 0 ? fmtPnl(week.weekPnl) : "\u2014"}
                  </span>
                  {week.weekTrades > 0 && <span className="text-[10px] mt-0.5" style={{ color: "var(--text-dim)" }}>{week.weekTrades} trades</span>}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Selected day trades from CSV history */}
      {selectedDay && dayTrades.length > 0 && (
        <div className="mt-5 panel rounded overflow-hidden">
          <div className="px-4 py-2 text-[10px] uppercase tracking-wider flex items-center justify-between" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
            <span>Trades for {selectedDay}</span>
            <button onClick={() => setSelectedDay(null)} className="text-zinc-500 hover:text-zinc-300 text-xs">Close</button>
          </div>
          <table className="w-full text-[12px]">
            <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
              <th className="text-left font-normal px-4 py-1.5">Time</th>
              <th className="text-left font-normal px-3 py-1.5">Strategy</th>
              <th className="text-left font-normal px-3 py-1.5">Side</th>
              <th className="text-right font-normal px-3 py-1.5">Entry</th>
              <th className="text-right font-normal px-3 py-1.5">Exit</th>
              <th className="text-right font-normal px-3 py-1.5">P&L</th>
              <th className="text-right font-normal px-4 py-1.5">Reason</th>
            </tr></thead>
            <tbody>
              {dayTrades.map((t, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="px-4 py-1.5 font-mono tabular" style={{ color: "var(--text-muted)" }}>{((t.timestamp as string) || "").split(" ")[1] || ""}</td>
                  <td className="px-3 py-1.5 font-medium" style={{ color: "var(--text)" }}>{t.strategy as string}</td>
                  <td className="px-3 py-1.5" style={{ color: t.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{t.side === "Buy" ? "LONG" : "SHORT"}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: "var(--text)" }}>{Number(t.fill_price || 0).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: "var(--text)" }}>{Number(t.exit_price || 0).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular" style={{ color: clr(Number(t.pnl_total || 0)) }}>{fmtPnl(Number(t.pnl_total || 0))}</td>
                  <td className="px-4 py-1.5 text-right" style={{ color: "var(--text-dim)" }}>{t.exit_reason as string}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selectedDay && dayTrades.length === 0 && (
        <div className="mt-5 text-center text-xs py-4" style={{ color: "var(--text-dim)" }}>No trade history for {selectedDay}</div>
      )}
    </div>
  );
}
