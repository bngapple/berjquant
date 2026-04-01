import { useEffect, useMemo, useState } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { TerminalChart } from "../components/TerminalChart";
import { api } from "../api/client";
import type { Bar, Position, Trade, Signal } from "../types";

const STRATEGY_COLORS: Record<string, string> = {
  RSI: "#00d4aa",
  IB: "#3b82f6",
  MOM: "#f97316",
};

const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
const clr = (v: number) => (v >= 0 ? "var(--accent)" : "var(--red)");

type FeedEntry =
  | { kind: "signal"; ts: string; s: Signal }
  | { kind: "fill"; ts: string; t: Trade }
  | { kind: "exit"; ts: string; t: Trade };

export function Terminal() {
  const { bars: wsBars, trades, signals, positions, pnl, status } = useLayoutData();
  const [histBars, setHistBars] = useState<Bar[]>([]);

  // Fetch historical bars on mount
  useEffect(() => {
    api.getBars().then(setHistBars).catch(() => {});
  }, []);

  // Merge historical + live bars (live takes priority for same timestamp)
  const allBars = useMemo(() => {
    const map = new Map<string, Bar>();
    for (const b of histBars) map.set(b.timestamp, b);
    for (const b of wsBars) map.set(b.timestamp, b);   // live overwrites
    return Array.from(map.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [histBars, wsBars]);

  // Last completed bar for ticker
  const lastBar = allBars.length > 0 ? allBars[allBars.length - 1] : null;
  const prevBar = allBars.length > 1 ? allBars[allBars.length - 2] : null;
  const priceChange = lastBar && prevBar ? lastBar.close - prevBar.close : 0;
  const priceChangePct = prevBar && prevBar.close > 0 ? (priceChange / prevBar.close) * 100 : 0;

  // Execution tape — merge signals + fills + exits chronologically
  const feed: FeedEntry[] = useMemo(() => {
    const entries: FeedEntry[] = [];
    for (const s of signals) entries.push({ kind: "signal", ts: s.timestamp, s });
    for (const t of trades) {
      if (t.action === "entry") entries.push({ kind: "fill", ts: t.timestamp, t });
      else entries.push({ kind: "exit", ts: t.timestamp, t });
    }
    return entries
      .sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime())
      .slice(0, 40);
  }, [signals, trades]);

  return (
    <div className="flex flex-col h-full" style={{ background: "#0d0d0d", fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
      {/* ── Ticker Bar ─────────────────────────────────────── */}
      <div
        className="flex items-center gap-6 px-4 h-10 shrink-0 text-[11px]"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "#111" }}
      >
        <span className="font-bold tracking-widest" style={{ color: "#00d4aa", letterSpacing: "0.08em" }}>
          MNQM6
        </span>
        {lastBar ? (
          <>
            <span className="font-mono text-[15px] font-semibold tabular" style={{ color: "#e8e8e8" }}>
              {lastBar.close.toFixed(2)}
            </span>
            <span
              className="font-mono tabular text-[11px]"
              style={{ color: priceChange >= 0 ? "#00d4aa" : "#ef4444" }}
            >
              {priceChange >= 0 ? "▲" : "▼"} {Math.abs(priceChange).toFixed(2)} ({priceChangePct >= 0 ? "+" : ""}{priceChangePct.toFixed(2)}%)
            </span>
            <span style={{ color: "rgba(255,255,255,0.1)" }}>│</span>
            {lastBar.rsi != null && (
              <span style={{ color: lastBar.rsi < 35 ? "#00d4aa" : lastBar.rsi > 65 ? "#ef4444" : "#666" }}>
                RSI {lastBar.rsi.toFixed(1)}
              </span>
            )}
            {lastBar.atr != null && (
              <span style={{ color: "#555" }}>ATR {lastBar.atr.toFixed(1)}</span>
            )}
            {lastBar.ema != null && (
              <span style={{ color: "#f59e0b" }}>EMA {lastBar.ema.toFixed(2)}</span>
            )}
          </>
        ) : (
          <span style={{ color: "#444" }}>Waiting for data...</span>
        )}
        <div className="flex-1" />
        {/* P&L */}
        {pnl && (
          <>
            <span style={{ color: "#444" }}>Day</span>
            <span className="font-mono tabular" style={{ color: clr(pnl.daily) }}>{fmt(pnl.daily)}</span>
            <span style={{ color: "#444" }}>Month</span>
            <span className="font-mono tabular" style={{ color: clr(pnl.monthly) }}>{fmt(pnl.monthly)}</span>
          </>
        )}
        <span style={{ color: "#444" }}>│</span>
        <span className={`flex items-center gap-1 ${status?.running ? "text-emerald-400" : "text-zinc-600"}`}>
          <span className={`w-1.5 h-1.5 rounded-full inline-block ${status?.running ? "bg-emerald-400" : "bg-zinc-600"}`} />
          {status?.running ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      {/* ── Main Body ──────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart (left, ~72%) */}
        <div className="flex-1 overflow-hidden" style={{ borderRight: "1px solid rgba(255,255,255,0.05)" }}>
          <TerminalChart bars={allBars} trades={trades} positions={positions} />
        </div>

        {/* Right panel (28%) */}
        <div className="flex flex-col shrink-0 overflow-hidden" style={{ width: 280 }}>
          {/* Positions */}
          <div style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div
              className="px-3 py-1.5 text-[9px] tracking-widest font-semibold uppercase"
              style={{ color: "#444", borderBottom: "1px solid rgba(255,255,255,0.04)" }}
            >
              Positions
            </div>
            {["RSI", "IB", "MOM"].map((s) => {
              const pos = positions[s] as Position | null | undefined;
              return (
                <div
                  key={s}
                  className="px-3 py-2"
                  style={{
                    borderBottom: "1px solid rgba(255,255,255,0.04)",
                    borderLeft: pos ? `2px solid ${STRATEGY_COLORS[s]}` : "2px solid transparent",
                  }}
                >
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold" style={{ color: STRATEGY_COLORS[s] }}>{s}</span>
                    {pos ? (
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-bold"
                        style={{
                          background: pos.side === "Buy" ? "rgba(0,212,170,0.12)" : "rgba(239,68,68,0.12)",
                          color: pos.side === "Buy" ? "#00d4aa" : "#ef4444",
                        }}
                      >
                        {pos.side === "Buy" ? "▲ LONG" : "▼ SHORT"} {pos.contracts}ct
                      </span>
                    ) : (
                      <span className="text-[10px]" style={{ color: "#333" }}>FLAT</span>
                    )}
                  </div>
                  {pos && (
                    <div className="mt-1 space-y-0.5 text-[10px]">
                      <div className="flex justify-between">
                        <span style={{ color: "#444" }}>Entry</span>
                        <span className="font-mono tabular" style={{ color: "#777" }}>{pos.entry_price.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span style={{ color: "#444" }}>P&L</span>
                        <span className="font-mono tabular font-semibold" style={{ color: clr(pos.pnl) }}>{fmt(pos.pnl)}</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span>
                          <span style={{ color: "#ef4444" }}>SL </span>
                          <span className="font-mono tabular" style={{ color: "#666" }}>{pos.sl.toFixed(2)}</span>
                        </span>
                        <span>
                          <span style={{ color: "#00d4aa" }}>TP </span>
                          <span className="font-mono tabular" style={{ color: "#666" }}>{pos.tp.toFixed(2)}</span>
                        </span>
                        <span>
                          <span style={{ color: "#444" }}>{pos.bars_held}b</span>
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Execution Tape */}
          <div className="flex flex-col flex-1 overflow-hidden">
            <div
              className="px-3 py-1.5 text-[9px] tracking-widest font-semibold uppercase shrink-0"
              style={{ color: "#444", borderBottom: "1px solid rgba(255,255,255,0.04)" }}
            >
              Execution Tape
            </div>
            <div className="flex-1 overflow-y-auto">
              {feed.length === 0 ? (
                <div className="px-3 py-6 text-[11px] text-center" style={{ color: "#333" }}>
                  No activity
                </div>
              ) : (
                feed.map((entry, i) => <TapeRow key={i} entry={entry} />)
              )}
            </div>
          </div>

          {/* Pending signals count */}
          {(status?.pending_signals ?? 0) > 0 && (
            <div
              className="px-3 py-2 text-[10px] shrink-0 flex items-center gap-2"
              style={{ borderTop: "1px solid rgba(255,255,255,0.06)", color: "#f59e0b" }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse inline-block" />
              {status!.pending_signals} signal{status!.pending_signals > 1 ? "s" : ""} pending next bar
            </div>
          )}
        </div>
      </div>

      {/* ── Risk Bar ───────────────────────────────────────── */}
      {pnl && (
        <div
          className="flex items-center gap-6 px-4 h-7 shrink-0 text-[10px]"
          style={{ borderTop: "1px solid rgba(255,255,255,0.05)", background: "#0a0a0a" }}
        >
          <RiskMeter label="Day" value={pnl.daily} limit={pnl.daily_limit} />
          <span style={{ color: "rgba(255,255,255,0.08)" }}>│</span>
          <RiskMeter label="Month" value={pnl.monthly} limit={pnl.monthly_limit} />
          <div className="flex-1" />
          <span style={{ color: "#333" }}>15m · MNQ · {allBars.length} bars</span>
        </div>
      )}
    </div>
  );
}

function TapeRow({ entry }: { entry: FeedEntry }) {
  const time = entry.ts?.split("T")[1]?.slice(0, 8) ?? "";

  if (entry.kind === "signal") {
    const { s } = entry;
    return (
      <div
        className="flex items-center gap-2 px-3 py-[5px] text-[10px]"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}
      >
        <span className="font-mono tabular shrink-0" style={{ color: "#333" }}>{time}</span>
        <span className="shrink-0 px-1 rounded text-[8px] font-bold" style={{ background: "rgba(255,255,255,0.05)", color: "#555" }}>SIG</span>
        <span className="font-semibold shrink-0" style={{ color: STRATEGY_COLORS[s.strategy] ?? "#666" }}>{s.strategy}</span>
        <span style={{ color: s.side === "Buy" ? "#00d4aa" : "#ef4444" }}>{s.side === "Buy" ? "▲" : "▼"}</span>
        <span className="truncate" style={{ color: "#3a3a3a" }}>{s.reason?.slice(0, 30)}</span>
      </div>
    );
  }

  if (entry.kind === "fill") {
    const { t } = entry;
    return (
      <div
        className="flex items-center gap-2 px-3 py-[5px] text-[10px]"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.03)", borderLeft: `2px solid ${t.side === "Buy" ? "#00d4aa" : "#ef4444"}` }}
      >
        <span className="font-mono tabular shrink-0" style={{ color: "#444" }}>{time}</span>
        <span className="shrink-0 px-1 rounded text-[8px] font-bold" style={{ background: t.side === "Buy" ? "rgba(0,212,170,0.1)" : "rgba(239,68,68,0.1)", color: t.side === "Buy" ? "#00d4aa" : "#ef4444" }}>
          {t.side === "Buy" ? "BUY" : "SELL"}
        </span>
        <span className="font-semibold shrink-0" style={{ color: STRATEGY_COLORS[t.strategy] ?? "#666" }}>{t.strategy}</span>
        <span className="font-mono tabular" style={{ color: "#666" }}>@{t.fill_price?.toFixed(2)}</span>
        {t.slippage != null && (
          <span className="font-mono tabular" style={{ color: "#333" }}>slip {t.slippage.toFixed(1)}</span>
        )}
      </div>
    );
  }

  // exit
  const { t } = entry;
  return (
    <div
      className="flex items-center gap-2 px-3 py-[5px] text-[10px]"
      style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}
    >
      <span className="font-mono tabular shrink-0" style={{ color: "#444" }}>{time}</span>
      <span className="shrink-0 px-1 rounded text-[8px] font-bold" style={{ background: "rgba(255,255,255,0.05)", color: "#555" }}>EXIT</span>
      <span className="font-semibold shrink-0" style={{ color: STRATEGY_COLORS[t.strategy] ?? "#666" }}>{t.strategy}</span>
      <span className="shrink-0 text-[9px]" style={{ color: "#555" }}>{t.exit_reason}</span>
      {t.pnl != null && (
        <span className="font-mono tabular font-bold" style={{ color: t.pnl >= 0 ? "#00d4aa" : "#ef4444" }}>
          {fmt(t.pnl)}
        </span>
      )}
      {t.bars_held != null && (
        <span style={{ color: "#333" }}>{t.bars_held}b</span>
      )}
    </div>
  );
}

function RiskMeter({ label, value, limit }: { label: string; value: number; limit: number }) {
  const pct = value <= 0 ? Math.min((Math.abs(value) / Math.abs(limit)) * 100, 100) : 0;
  const danger = pct > 80;
  const warn = pct > 50;
  const barColor = danger ? "#ef4444" : warn ? "#f59e0b" : "#00d4aa";
  return (
    <div className="flex items-center gap-2">
      <span style={{ color: "#444" }}>{label}</span>
      <div className="w-20 h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: barColor, transition: "width 0.5s" }} />
      </div>
      <span className="font-mono tabular" style={{ color: value >= 0 ? "#00d4aa" : danger ? "#ef4444" : "#666" }}>
        {fmt(value)}
      </span>
    </div>
  );
}
