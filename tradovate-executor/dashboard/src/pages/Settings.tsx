import { useState, useEffect } from "react";
import { api } from "../api/client";

const STRATEGIES = [
  {
    name: "RSI Extremes",
    key: "RSI",
    params: [
      { label: "RSI Period", value: "5" },
      { label: "Oversold", value: "35" },
      { label: "Overbought", value: "65" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "10 pts" },
      { label: "Take Profit", value: "100 pts" },
      { label: "Max Hold", value: "5 bars (75 min)" },
    ],
  },
  {
    name: "IB Breakout",
    key: "IB",
    params: [
      { label: "IB Window", value: "9:30 – 10:00 ET" },
      { label: "Range Filter", value: "P25 – P75 (50 day)" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "10 pts" },
      { label: "Take Profit", value: "120 pts" },
      { label: "Max Hold", value: "15 bars (225 min)" },
      { label: "Max/Day", value: "1" },
    ],
  },
  {
    name: "Momentum Bars",
    key: "MOM",
    params: [
      { label: "ATR Period", value: "14" },
      { label: "EMA Period", value: "21" },
      { label: "Vol SMA", value: "20" },
      { label: "Contracts", value: "3" },
      { label: "Stop Loss", value: "15 pts" },
      { label: "Take Profit", value: "100 pts" },
      { label: "Max Hold", value: "5 bars (75 min)" },
    ],
  },
];

const SESSION = [
  { label: "Session Start", value: "9:30 AM ET" },
  { label: "No New Entries", value: "4:30 PM ET" },
  { label: "Flatten Time", value: "4:45 PM ET" },
  { label: "Daily Loss Limit", value: "-$3,000" },
  { label: "Monthly Loss Limit", value: "-$4,500" },
  { label: "Timezone", value: "US/Eastern" },
];

const CONTRACT = [
  { label: "Symbol", value: "MNQ (Micro Nasdaq 100)" },
  { label: "Front Month", value: "MNQM6 (June 2026)" },
  { label: "Tick Size", value: "0.25 pts = $0.50/contract" },
  { label: "Point Value", value: "$2.00/contract/point" },
];

export function Settings() {
  const [environment, setEnvironment] = useState("demo");

  useEffect(() => {
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Settings</h2>

      {/* Environment */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="font-semibold">Environment</h3>
            <p className="text-sm text-gray-400">
              Current: {environment.toUpperCase()}
            </p>
          </div>
          <span
            className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider ${
              environment === "demo"
                ? "bg-yellow-900/50 text-yellow-300 border border-yellow-700"
                : "bg-red-900/50 text-red-300 border border-red-700"
            }`}
          >
            {environment}
          </span>
        </div>
      </div>

      {/* Strategies */}
      <div>
        <h3 className="font-semibold mb-3">Strategy Parameters</h3>
        <div className="grid grid-cols-3 gap-4">
          {STRATEGIES.map((strat) => (
            <div
              key={strat.key}
              className="bg-gray-900 rounded-lg border border-gray-800 p-4"
            >
              <h4 className="font-medium text-sm mb-3">{strat.name}</h4>
              <div className="space-y-1.5">
                {strat.params.map((p) => (
                  <div
                    key={p.label}
                    className="flex justify-between text-xs"
                  >
                    <span className="text-gray-400">{p.label}</span>
                    <span className="font-mono text-gray-200">{p.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Session Config */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3 text-sm">Session Rules</h3>
          <div className="space-y-1.5">
            {SESSION.map((p) => (
              <div key={p.label} className="flex justify-between text-xs">
                <span className="text-gray-400">{p.label}</span>
                <span className="font-mono text-gray-200">{p.value}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h3 className="font-semibold mb-3 text-sm">Contract Info</h3>
          <div className="space-y-1.5">
            {CONTRACT.map((p) => (
              <div key={p.label} className="flex justify-between text-xs">
                <span className="text-gray-400">{p.label}</span>
                <span className="font-mono text-gray-200">{p.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
