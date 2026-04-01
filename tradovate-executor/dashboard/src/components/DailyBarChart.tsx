import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine, CartesianGrid } from "recharts";

interface DailyBar { date: string; pnl: number; trades?: number; wins?: number; }
interface Props { data: DailyBar[]; }

export function DailyBarChart({ data }: Props) {
  if (data.length === 0) return <div className="h-full flex items-center justify-center text-xs" style={{ color: "var(--text-dim)" }}>No data</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} interval="preserveStartEnd" minTickGap={40} />
        <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} tickFormatter={(v: number) => `$${v}`} />
        <Tooltip
          contentStyle={{ background: "var(--elevated)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 11, color: "var(--text)" }}
          formatter={(v) => [`$${Number(v).toFixed(2)}`, "P&L"]}
          labelStyle={{ color: "var(--text-muted)" }}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" />
        <Bar dataKey="pnl" radius={[2, 2, 0, 0]} maxBarSize={20}>
          {data.map((e, i) => <Cell key={i} fill={e.pnl >= 0 ? "var(--accent)" : "var(--red)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
