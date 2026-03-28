import { useState, useMemo } from "react";

interface Props { data: Record<string, number>; }

export function PnLCalendar({ data }: Props) {
  const [offset, setOffset] = useState(0);

  const { label, weeks } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear(), m = now.getMonth();
    const label = now.toLocaleString("en-US", { month: "long", year: "numeric" });

    const first = new Date(y, m, 1);
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const startDay = (first.getDay() + 6) % 7;

    const cells: { day: number; key: string; pnl: number | null }[] = [];
    for (let i = 0; i < startDay; i++) cells.push({ day: 0, key: `pad-${i}`, pnl: null });
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      cells.push({ day: d, key, pnl: data[key] ?? null });
    }

    const weeks: typeof cells[] = [];
    for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
    const last = weeks[weeks.length - 1];
    while (last && last.length < 7) last.push({ day: 0, key: `end-${last.length}`, pnl: null });

    return { label, weeks };
  }, [offset, data]);

  const bg = (pnl: number | null) => {
    if (pnl === null) return "transparent";
    if (pnl === 0) return "rgba(255,255,255,0.02)";
    if (pnl > 0) return `rgba(0,212,170,${Math.min(0.1 + (pnl / 500) * 0.3, 0.4)})`;
    return `rgba(239,68,68,${Math.min(0.1 + (Math.abs(pnl) / 500) * 0.3, 0.4)})`;
  };

  const cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <button onClick={() => setOffset(o => o - 1)} className="text-xs px-1.5" style={{ color: "var(--text-muted)" }}>&larr;</button>
        <span className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>{label}</span>
        <button onClick={() => setOffset(o => Math.min(o + 1, 0))} disabled={offset >= 0} className="text-xs px-1.5 disabled:opacity-20" style={{ color: "var(--text-muted)" }}>&rarr;</button>
      </div>

      <table className="w-full border-collapse text-[10px]">
        <thead>
          <tr>
            {cols.map(c => <th key={c} className="font-normal pb-1.5 text-center" style={{ color: "var(--text-dim)" }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, wi) => (
            <tr key={wi}>
              {week.map(cell => (
                <td key={cell.key} className="cal-cell relative p-0 text-center h-9" style={{ background: bg(cell.pnl), border: "1px solid var(--border)" }}>
                  {cell.day > 0 && (
                    <div className="flex flex-col items-center justify-center h-full">
                      <span style={{ color: "var(--text-dim)", fontSize: 8 }}>{cell.day}</span>
                      {cell.pnl !== null && (
                        <span className="font-mono font-medium" style={{ color: cell.pnl >= 0 ? "var(--accent)" : "var(--red)", fontSize: 9 }}>
                          {cell.pnl >= 0 ? "+" : ""}{cell.pnl.toFixed(0)}
                        </span>
                      )}
                    </div>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
