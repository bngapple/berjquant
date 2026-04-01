import { useState, useEffect } from "react";
import { api } from "../api/client";

export function Settings() {
  const [env, setEnv] = useState("demo");
  useEffect(() => { api.getEnvironment().then(d => setEnv(d.environment)); }, []);

  return (
    <div className="p-5 max-w-[800px] mx-auto space-y-5">
      <Section title="Session">
        <div className="grid grid-cols-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <BigStat label="Trading Session" value="9:30 – 4:45 ET" />
          <BigStat label="Daily Loss Limit" value="-$3,000" color="var(--red)" />
          <BigStat label="Monthly Loss Limit" value="-$4,500" color="var(--red)" />
        </div>
        <div className="px-4 py-2 flex gap-6 text-[11px]" style={{ color: "var(--text-muted)" }}>
          <span>No new entries after 4:30 PM ET</span>
          <span>Flatten at 4:45 PM ET</span>
          <span>US/Eastern</span>
          <span className="ml-auto px-2 py-0.5 rounded text-[9px] font-bold uppercase" style={{ background: env === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: env === "demo" ? "var(--amber)" : "var(--red)" }}>{env}</span>
        </div>
      </Section>

      <Section title="Contract">
        <div className="px-4 py-3 flex items-center gap-6 text-xs">
          <span className="font-mono font-semibold" style={{ color: "var(--text)" }}>MNQM6</span>
          <span style={{ color: "var(--text-muted)" }}>MNQ Micro Nasdaq 100</span>
          <span className="ml-auto flex gap-4" style={{ color: "var(--text-muted)" }}>
            <span>Tick: 0.25 = $0.50</span><span>Point: $2.00</span>
          </span>
        </div>
      </Section>

      <Section title="Strategy Parameters">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-4 py-2 w-[35%]">Parameter</th>
            <th className="text-center font-normal px-3 py-2">RSI</th>
            <th className="text-center font-normal px-3 py-2">IB</th>
            <th className="text-center font-normal px-3 py-2">MOM</th>
          </tr></thead>
          <tbody>
            {PARAMS.map(([label, rsi, ib, mom], i) => (
              <tr key={label} style={{ background: i % 2 === 1 ? "rgba(255,255,255,0.015)" : "transparent", borderTop: "1px solid var(--border)" }}>
                <td className="px-4 py-1.5" style={{ color: "var(--text-muted)" }}>{label}</td>
                <td className="px-3 py-1.5 text-center font-mono tabular" style={{ color: "var(--text)" }}>{rsi}</td>
                <td className="px-3 py-1.5 text-center font-mono tabular" style={{ color: "var(--text)" }}>{ib}</td>
                <td className="px-3 py-1.5 text-center font-mono tabular" style={{ color: "var(--text)" }}>{mom}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </div>
  );
}

const PARAMS: [string, string, string, string][] = [
  ["Contracts", "3", "3", "3"],
  ["Stop Loss", "10 pts", "10 pts", "15 pts"],
  ["Take Profit", "100 pts", "120 pts", "100 pts"],
  ["Max Hold", "5 bars", "15 bars", "5 bars"],
  ["Period / Window", "RSI(5)", "9:30–10:00 ET", "ATR(14)"],
  ["Thresholds", "35 / 65", "P25–P75 (50d)", "EMA(21)"],
  ["Extra", "—", "Max 1/day", "Vol > SMA(20)"],
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel rounded overflow-hidden" style={{ borderTop: "2px solid rgba(255,255,255,0.08)" }}>
      <div className="px-4 py-2 text-[10px] font-normal uppercase tracking-widest" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{title}</div>
      {children}
    </div>
  );
}

function BigStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="px-4 py-3 text-center">
      <div className="text-lg font-mono font-semibold tabular" style={{ color: color ?? "var(--text)" }}>{value}</div>
      <div className="text-[10px] mt-0.5 font-light" style={{ color: "var(--text-muted)" }}>{label}</div>
    </div>
  );
}
