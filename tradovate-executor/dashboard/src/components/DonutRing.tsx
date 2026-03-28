import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

interface Props { value: number; size?: number; color?: string; }

export function DonutRing({ value, size = 36, color = "var(--accent)" }: Props) {
  const data = [{ v: Math.max(value, 0) }, { v: Math.max(100 - value, 0) }];
  return (
    <div style={{ width: size, height: size }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie data={data} innerRadius="60%" outerRadius="100%" dataKey="v" startAngle={90} endAngle={-270} stroke="none">
            <Cell fill={color} />
            <Cell fill="rgba(255,255,255,0.06)" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
