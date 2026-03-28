import { useState } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import type { EquityPoint } from "../types";

interface Props { data: EquityPoint[]; }

const TABS = ["1W", "1M", "3M", "ALL"] as const;

export function EquityCurve({ data }: Props) {
  const [tab, setTab] = useState<typeof TABS[number]>("ALL");
  const latest = data.length > 0 ? data[data.length - 1].value : 0;
  const isPos = latest >= 0;
  const color = isPos ? "#00d4aa" : "#ef4444";

  const cutoff = { "1W": 100, "1M": 150, "3M": 180, "ALL": 999 }[tab];
  const filtered = data.slice(-cutoff);

  if (filtered.length < 2) return <div className="h-full flex items-center justify-center text-xs" style={{ color: "var(--text-dim)" }}>Waiting for data...</div>;

  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 mb-2">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className="px-2.5 py-0.5 text-[10px] font-medium rounded-full transition-colors"
            style={{ background: tab === t ? "var(--accent)" : "transparent", color: tab === t ? "#0d0d0d" : "var(--text-muted)" }}>
            {t}
          </button>
        ))}
        <span className="ml-auto text-xs font-mono font-semibold tabular" style={{ color }}>
          {latest >= 0 ? "+" : ""}${latest.toFixed(2)}
        </span>
      </div>

      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filtered} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.2} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} interval="preserveStartEnd" minTickGap={60} />
            <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} orientation="right" />
            <Tooltip contentStyle={{ background: "var(--elevated)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 11, color: "var(--text)" }} formatter={(v: number) => [`$${v.toFixed(2)}`, "P&L"]} labelStyle={{ color: "var(--text-muted)" }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill="url(#eqGrad)" dot={false} activeDot={{ r: 3, fill: color }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
