import { useState, useEffect } from "react";
import { api } from "../api/client";

const STRATEGIES = [
  {
    name: "RSI Extremes",
    rows: [
      ["RSI Period", "5"],
      ["Oversold / Overbought", "35 / 65"],
      ["Contracts", "3"],
      ["Stop Loss", "10 pts"],
      ["Take Profit", "100 pts"],
      ["Max Hold", "5 bars (75 min)"],
    ],
  },
  {
    name: "IB Breakout",
    rows: [
      ["IB Window", "9:30 – 10:00 ET"],
      ["Range Filter", "P25 – P75 (50 day)"],
      ["Contracts", "3"],
      ["Stop Loss", "10 pts"],
      ["Take Profit", "120 pts"],
      ["Max Hold", "15 bars (225 min)"],
      ["Max / Day", "1"],
    ],
  },
  {
    name: "Momentum Bars",
    rows: [
      ["ATR Period", "14"],
      ["EMA Period", "21"],
      ["Volume SMA", "20"],
      ["Contracts", "3"],
      ["Stop Loss", "15 pts"],
      ["Take Profit", "100 pts"],
      ["Max Hold", "5 bars (75 min)"],
    ],
  },
];

export function Settings() {
  const [environment, setEnvironment] = useState("demo");

  useEffect(() => {
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  }, []);

  return (
    <div className="p-5 max-w-[900px] mx-auto space-y-6">
      <h2 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Settings
      </h2>

      {/* Session info — prominent */}
      <div className="panel rounded-lg overflow-hidden">
        <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>Session</span>
          <span
            className="text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider"
            style={{
              background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)",
              color: environment === "demo" ? "var(--accent-yellow)" : "var(--accent-red)",
            }}
          >
            {environment}
          </span>
        </div>
        <div className="grid grid-cols-3 divide-x" style={{ borderColor: "var(--border)" }}>
          <div className="px-5 py-4 text-center">
            <div className="text-xl font-mono font-semibold" style={{ color: "var(--text-primary)" }}>9:30 – 4:45</div>
            <div className="text-[10px] mt-1" style={{ color: "var(--text-muted)" }}>Trading Session (ET)</div>
          </div>
          <div className="px-5 py-4 text-center">
            <div className="text-xl font-mono font-semibold" style={{ color: "var(--accent-red)" }}>-$3,000</div>
            <div className="text-[10px] mt-1" style={{ color: "var(--text-muted)" }}>Daily Loss Limit</div>
          </div>
          <div className="px-5 py-4 text-center">
            <div className="text-xl font-mono font-semibold" style={{ color: "var(--accent-red)" }}>-$4,500</div>
            <div className="text-[10px] mt-1" style={{ color: "var(--text-muted)" }}>Monthly Loss Limit</div>
          </div>
        </div>
        <div className="px-5 py-2 text-[11px] flex gap-6" style={{ color: "var(--text-muted)", borderTop: "1px solid var(--border)" }}>
          <span>No new entries after 4:30 PM ET</span>
          <span>Flatten at 4:45 PM ET</span>
          <span>Timezone: US/Eastern</span>
        </div>
      </div>

      {/* Contract info */}
      <div className="panel rounded-lg px-5 py-3">
        <div className="flex items-center gap-6 text-xs">
          <span style={{ color: "var(--text-muted)" }}>Contract</span>
          <span className="font-mono" style={{ color: "var(--text-primary)" }}>MNQM6</span>
          <span style={{ color: "var(--text-muted)" }}>MNQ Micro Nasdaq 100</span>
          <span className="ml-auto flex gap-4" style={{ color: "var(--text-secondary)" }}>
            <span>Tick: 0.25 = $0.50</span>
            <span>Point: $2.00</span>
          </span>
        </div>
      </div>

      {/* Strategy params — table format */}
      {STRATEGIES.map((strat) => (
        <div key={strat.name} className="panel rounded-lg overflow-hidden">
          <div className="px-5 py-2.5 text-xs font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>
            {strat.name}
          </div>
          <table className="w-full text-xs">
            <tbody>
              {strat.rows.map(([label, value], i) => (
                <tr
                  key={label}
                  style={i < strat.rows.length - 1 ? { borderBottom: "1px solid var(--border)" } : undefined}
                >
                  <td className="px-5 py-2" style={{ color: "var(--text-muted)", width: "40%" }}>{label}</td>
                  <td className="px-5 py-2 font-mono" style={{ color: "var(--text-primary)" }}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
