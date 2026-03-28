import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { EquityPoint } from "../hooks/useWebSocket";

interface Props {
  data: EquityPoint[];
}

export function EquityCurve({ data }: Props) {
  const latest = data.length > 0 ? data[data.length - 1].value : 0;
  const isPositive = latest >= 0;
  const strokeColor = isPositive ? "#10b981" : "#ef4444";

  if (data.length < 2) {
    return (
      <div className="h-full flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
        <span className="text-sm">Waiting for data...</span>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={strokeColor} stopOpacity={0.15} />
            <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.03)"
          vertical={false}
        />
        <XAxis
          dataKey="time"
          axisLine={false}
          tickLine={false}
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          axisLine={false}
          tickLine={false}
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
        />
        <Tooltip
          contentStyle={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-strong)",
            borderRadius: 6,
            fontSize: 12,
            color: "var(--text-primary)",
          }}
          formatter={(value: number) => [`$${value.toFixed(2)}`, "P&L"]}
          labelStyle={{ color: "var(--text-secondary)" }}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
        <Area
          type="monotone"
          dataKey="value"
          stroke={strokeColor}
          strokeWidth={1.5}
          fill="url(#equityGrad)"
          dot={false}
          activeDot={{ r: 3, fill: strokeColor }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
