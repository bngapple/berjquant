import { useState, useMemo } from "react";

interface Props {
  /** Map of "YYYY-MM-DD" -> daily P&L in USD */
  data: Record<string, number>;
}

export function PnLCalendar({ data }: Props) {
  const [offset, setOffset] = useState(0); // 0 = current month, -1 = prev, etc.

  const { label, days } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear();
    const m = now.getMonth();
    const label = now.toLocaleString("en-US", { month: "long", year: "numeric" });

    const firstDay = new Date(y, m, 1).getDay(); // 0=Sun
    const daysInMonth = new Date(y, m + 1, 0).getDate();

    const days: { day: number; key: string; pnl: number | null }[] = [];
    // Padding
    for (let i = 0; i < firstDay; i++) {
      days.push({ day: 0, key: `pad-${i}`, pnl: null });
    }
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      days.push({ day: d, key, pnl: data[key] ?? null });
    }
    return { label, days };
  }, [offset, data]);

  const cellColor = (pnl: number | null): string => {
    if (pnl === null) return "transparent";
    if (pnl === 0) return "rgba(255,255,255,0.03)";
    if (pnl > 0) {
      const intensity = Math.min(pnl / 500, 1);
      return `rgba(16,185,129,${0.15 + intensity * 0.5})`;
    }
    const intensity = Math.min(Math.abs(pnl) / 500, 1);
    return `rgba(239,68,68,${0.15 + intensity * 0.5})`;
  };

  const weekdays = ["S", "M", "T", "W", "T", "F", "S"];

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setOffset((o) => o - 1)}
          className="text-zinc-500 hover:text-zinc-300 text-xs px-1.5 py-0.5"
        >
          &larr;
        </button>
        <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
          {label}
        </span>
        <button
          onClick={() => setOffset((o) => Math.min(o + 1, 0))}
          disabled={offset >= 0}
          className="text-zinc-500 hover:text-zinc-300 text-xs px-1.5 py-0.5 disabled:opacity-20"
        >
          &rarr;
        </button>
      </div>

      {/* Weekday headers */}
      <div className="grid grid-cols-7 gap-[3px] mb-[3px]">
        {weekdays.map((d, i) => (
          <div key={i} className="text-center text-[9px]" style={{ color: "var(--text-muted)" }}>
            {d}
          </div>
        ))}
      </div>

      {/* Calendar grid */}
      <div className="grid grid-cols-7 gap-[3px]">
        {days.map((d) => (
          <div
            key={d.key}
            className="cal-cell relative aspect-square rounded-[3px] flex items-center justify-center group/cell"
            style={{ background: cellColor(d.pnl) }}
            title={d.pnl !== null ? `$${d.pnl.toFixed(0)}` : undefined}
          >
            {d.day > 0 && (
              <span className="text-[8px]" style={{ color: d.pnl !== null ? "rgba(255,255,255,0.5)" : "var(--text-muted)" }}>
                {d.day}
              </span>
            )}
            {/* Tooltip on hover */}
            {d.pnl !== null && (
              <div className="absolute -top-7 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded text-[9px] font-mono whitespace-nowrap opacity-0 group-hover/cell:opacity-100 pointer-events-none z-20"
                   style={{
                     background: "var(--bg-surface)",
                     border: "1px solid var(--border-strong)",
                     color: d.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)",
                   }}>
                {d.pnl >= 0 ? "+" : ""}${d.pnl.toFixed(0)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
