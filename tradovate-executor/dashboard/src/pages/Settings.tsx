import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { RuntimeConfig } from "../types";

function formatLimit(value: number | null | undefined): string {
  if (value == null) return "LucidFlex";
  return `${value >= 0 ? "+" : "-"}$${Math.abs(value).toLocaleString("en-US")}`;
}

function describeContract(symbol: string): string {
  if (symbol.startsWith("MNQ")) return "MNQ Micro Nasdaq 100";
  if (symbol.startsWith("MES")) return "MES Micro S&P 500";
  return "Configured contract";
}

export function Settings() {
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null);

  useEffect(() => {
    api.getRuntimeConfig().then(setRuntime).catch(() => {});
  }, []);

  const session = runtime?.session;
  const env = runtime?.environment ?? "demo";
  const symbol = runtime?.symbol ?? "MNQU6";
  const params = useMemo<[string, string, string, string][]>(() => {
    if (!runtime) {
      return [
        ["Contracts", "—", "—", "—"],
        ["Stop Loss", "—", "—", "—"],
        ["Take Profit", "—", "—", "—"],
        ["Max Hold", "—", "—", "—"],
        ["Period / Window", "—", "—", "—"],
        ["Thresholds", "—", "—", "—"],
        ["Extra", "—", "—", "—"],
      ];
    }

    return [
      [
        "Contracts",
        String(runtime.rsi.contracts),
        String(runtime.ib.contracts),
        String(runtime.mom.contracts),
      ],
      [
        "Stop Loss",
        `${runtime.rsi.stop_loss_pts} pts`,
        `${runtime.ib.stop_loss_pts} pts`,
        `${runtime.mom.stop_loss_pts} pts`,
      ],
      [
        "Take Profit",
        `${runtime.rsi.take_profit_pts} pts`,
        `${runtime.ib.take_profit_pts} pts`,
        `${runtime.mom.take_profit_pts} pts`,
      ],
      [
        "Max Hold",
        `${runtime.rsi.max_hold_bars} bars`,
        `${runtime.ib.max_hold_bars} bars`,
        `${runtime.mom.max_hold_bars} bars`,
      ],
      [
        "Period / Window",
        `RSI(${runtime.rsi.period ?? "—"})`,
        `${runtime.ib.ib_start ?? "—"}–${runtime.ib.ib_end ?? "—"} ET`,
        `ATR(${runtime.mom.atr_period ?? "—"})`,
      ],
      [
        "Thresholds",
        `${runtime.rsi.oversold ?? "—"} / ${runtime.rsi.overbought ?? "—"}`,
        `P${runtime.ib.ib_range_pct_low ?? "—"}–P${runtime.ib.ib_range_pct_high ?? "—"} (${runtime.ib.ib_range_lookback ?? "—"}d)`,
        `EMA(${runtime.mom.ema_period ?? "—"})`,
      ],
      [
        "Extra",
        "—",
        "Max 1/day",
        `Vol > SMA(${runtime.mom.vol_sma_period ?? "—"})`,
      ],
    ];
  }, [runtime]);

  return (
    <div className="p-5 max-w-[800px] mx-auto space-y-5">
      <Section title="Session">
        <div className="grid grid-cols-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <BigStat
            label="Trading Session"
            value={session ? `${session.session_start} – ${session.flatten_time} ET` : "9:30 – 4:45 ET"}
          />
          <BigStat
            label="Daily Limit"
            value={session?.daily_loss_limit == null ? "LucidFlex" : formatLimit(session.daily_loss_limit)}
            color={session?.daily_loss_limit == null ? "var(--accent)" : "var(--amber)"}
          />
          <BigStat
            label="Monthly Loss Limit"
            value={formatLimit(session?.monthly_loss_limit ?? -4500)}
            color="var(--red)"
          />
        </div>
        <div className="px-4 py-2 flex gap-6 text-[11px]" style={{ color: "var(--text-muted)" }}>
          <span>
            No new entries after {session?.no_new_entries_after ?? "16:30"} ET
          </span>
          <span>Flatten at {session?.flatten_time ?? "16:45"} ET</span>
          <span>{session?.timezone ?? "US/Eastern"}</span>
          <span className="ml-auto px-2 py-0.5 rounded text-[9px] font-bold uppercase" style={{ background: env === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: env === "demo" ? "var(--amber)" : "var(--red)" }}>{env}</span>
        </div>
      </Section>

      <Section title="Contract">
        <div className="px-4 py-3 flex items-center gap-6 text-xs">
          <span className="font-mono font-semibold" style={{ color: "var(--text)" }}>{symbol}</span>
          <span style={{ color: "var(--text-muted)" }}>{describeContract(symbol)}</span>
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
            {params.map(([label, rsi, ib, mom], i) => (
              <tr key={label} style={{ background: i % 2 === 1 ? "rgba(0,212,170,0.03)" : "transparent", borderTop: "1px solid var(--border)" }}>
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel rounded overflow-hidden" style={{ borderTop: "1px solid rgba(0,212,170,0.3)" }}>
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
