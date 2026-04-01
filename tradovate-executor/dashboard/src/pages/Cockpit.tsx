import { useState, useEffect, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { api } from "../api/client";
import type { Account, AccountStatus } from "../types";

export function Cockpit() {
  const { status, trades } = useLayoutData();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [accountStatuses, setAccountStatuses] = useState<AccountStatus[]>([]);
  const [expandedAccount, setExpandedAccount] = useState<string | null>(null);

  useEffect(() => {
    api.getAccounts().then(setAccounts).catch(console.error);
    api.getAccountStatuses().then(setAccountStatuses).catch(() => {});
  }, []);

  // Refresh statuses periodically
  useEffect(() => {
    const interval = setInterval(() => {
      api.getAccountStatuses().then(setAccountStatuses).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const running = status?.running ?? false;

  const rows = useMemo(() => {
    return accounts.map(acct => {
      const connected = status?.connected_accounts.find(a => a.name === acct.name)?.connected ?? false;
      const acctStatus = accountStatuses.find(s => s.name === acct.name);
      return {
        name: acct.name,
        isLeader: acct.is_master,
        sizing: acct.sizing_mode,
        connected,
        fixedSizes: acct.fixed_sizes,
        dayPnl: acctStatus?.daily_pnl ?? 0,
        totalPnl: acctStatus?.pnl_total ?? 0,
        tradesToday: acctStatus?.trades_today ?? 0,
        status: acctStatus,
      };
    });
  }, [accounts, status, accountStatuses]);

  const fleet = useMemo(() => {
    const connected = rows.filter(r => r.connected).length;
    const fleetDayPnl = rows.reduce((s, r) => s + r.dayPnl, 0);
    const fleetTotalPnl = rows.reduce((s, r) => s + r.totalPnl, 0);
    return { connected, total: rows.length, fleetDayPnl, fleetTotalPnl };
  }, [rows]);

  const copyFeed = useMemo(() => {
    // Only show real trades from the WS feed
    return [...trades].reverse().slice(0, 10);
  }, [trades]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  return (
    <div className="p-5 space-y-3 max-w-[1440px] mx-auto">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>LucidFlex 150K Fleet</span>
        <div className="flex items-center gap-4 text-xs">
          <span style={{ color: "var(--text-muted)" }}>{running ? "Engine Running" : "Engine Stopped"}</span>
          <span className="font-mono tabular" style={{ color: clr(fleet.fleetDayPnl) }}>Fleet: {fmt(fleet.fleetDayPnl)}</span>
        </div>
      </div>

      <div className="panel rounded overflow-hidden">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-4 py-2">Status</th>
            <th className="text-left font-normal px-3 py-2">Account</th>
            <th className="text-left font-normal px-3 py-2">Role</th>
            <th className="text-left font-normal px-3 py-2">Type</th>
            <th className="text-right font-normal px-3 py-2">Day P&L</th>
            <th className="text-right font-normal px-3 py-2">Total P&L</th>
            <th className="text-right font-normal px-3 py-2">Trades</th>
            <th className="text-right font-normal px-4 py-2">DD Used</th>
          </tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={8} className="px-4 py-6 text-center text-xs" style={{ color: "var(--text-dim)" }}>No accounts configured</td></tr>}
            {rows.map(row => (
              <>
                <tr key={row.name}
                  className="cursor-pointer transition-colors hover:bg-white/[0.02]"
                  onClick={() => setExpandedAccount(expandedAccount === row.name ? null : row.name)}
                  style={{
                    background: row.isLeader ? "var(--elevated)" : "var(--panel)",
                    borderTop: "1px solid var(--border)",
                    borderLeft: row.isLeader ? "3px solid var(--accent)" : "3px solid transparent",
                  }}>
                  <td className="px-4 py-2">
                    <span className={`w-2 h-2 rounded-full inline-block ${row.connected ? "bg-emerald-400 pulse-dot" : running ? "bg-red-500" : "bg-zinc-600"}`} />
                  </td>
                  <td className="px-3 py-2 font-medium" style={{ color: "var(--text)" }}>{row.name}</td>
                  <td className="px-3 py-2">
                    <span className="px-2 py-0.5 rounded text-[9px] font-semibold"
                      style={row.isLeader ? { background: "rgba(0,212,170,0.15)", color: "#00b894" } : { background: "transparent", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
                      {row.isLeader ? "Leader" : "Follower"}
                    </span>
                  </td>
                  <td className="px-3 py-2 capitalize text-[10px]" style={{ color: "var(--text-muted)" }}>{row.status?.account_type ?? "eval"}</td>
                  <td className="px-3 py-2 text-right font-mono tabular" style={{ color: clr(row.dayPnl) }}>{fmt(row.dayPnl)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular" style={{ color: clr(row.totalPnl) }}>{fmt(row.totalPnl)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular" style={{ color: "var(--text)" }}>{row.tradesToday}</td>
                  <td className="px-4 py-2 text-right">
                    {row.status ? (
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                          <div className="h-full rounded-full" style={{
                            width: `${Math.min(row.status.drawdown_pct_used, 100)}%`,
                            background: row.status.drawdown_pct_used > 75 ? "var(--red)" : row.status.drawdown_pct_used > 50 ? "var(--amber)" : "var(--accent)",
                          }} />
                        </div>
                        <span className="text-[10px] font-mono tabular w-8 text-right" style={{
                          color: row.status.drawdown_pct_used > 75 ? "var(--red)" : row.status.drawdown_pct_used > 50 ? "var(--amber)" : "var(--text-muted)",
                        }}>{row.status.drawdown_pct_used.toFixed(0)}%</span>
                      </div>
                    ) : <span style={{ color: "var(--text-dim)" }}>&mdash;</span>}
                  </td>
                </tr>
                {/* Expanded detail panel */}
                {expandedAccount === row.name && row.status && (
                  <tr key={`${row.name}-detail`}>
                    <td colSpan={8} style={{ background: "var(--elevated)", borderTop: "1px solid var(--border)" }}>
                      <div className="px-6 py-3 flex gap-8 text-[11px]">
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Balance</div>
                          <div className="font-mono font-semibold tabular" style={{ color: "var(--text)" }}>${row.status.balance.toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>P&L</div>
                          <div className="font-mono font-semibold tabular" style={{ color: clr(row.status.pnl_total) }}>{fmt(row.status.pnl_total)}</div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Drawdown</div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono tabular" style={{ color: row.status.drawdown_current < 0 ? "var(--red)" : "var(--text)" }}>
                              ${Math.abs(row.status.drawdown_current).toFixed(2)}
                            </span>
                            <span style={{ color: "var(--text-dim)" }}>/</span>
                            <span className="font-mono tabular" style={{ color: "var(--text-muted)" }}>${Math.abs(row.status.drawdown_max_allowed).toFixed(0)}</span>
                          </div>
                          <div className="w-24 h-1.5 rounded-full overflow-hidden mt-1" style={{ background: "rgba(255,255,255,0.04)" }}>
                            <div className="h-full rounded-full" style={{
                              width: `${Math.min(row.status.drawdown_pct_used, 100)}%`,
                              background: row.status.drawdown_pct_used > 75 ? "var(--red)" : row.status.drawdown_pct_used > 50 ? "var(--amber)" : "var(--accent)",
                            }} />
                          </div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Remaining</div>
                          <div className="font-mono tabular" style={{ color: "var(--text)" }}>${Math.abs(row.status.drawdown_remaining).toFixed(2)}</div>
                        </div>
                        {row.status.profit_target > 0 && (
                          <div>
                            <div style={{ color: "var(--text-muted)" }}>Profit Target</div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono tabular" style={{ color: "var(--text)" }}>
                                ${row.status.pnl_total.toFixed(0)} / ${row.status.profit_target.toFixed(0)}
                              </span>
                            </div>
                            <div className="w-24 h-1.5 rounded-full overflow-hidden mt-1" style={{ background: "rgba(255,255,255,0.04)" }}>
                              <div className="h-full rounded-full" style={{
                                width: `${Math.min(Math.max(row.status.profit_target_progress, 0), 100)}%`,
                                background: row.status.profit_target_progress >= 100 ? "#f59e0b" : "var(--accent)",
                              }} />
                            </div>
                          </div>
                        )}
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Daily P&L</div>
                          <div className="font-mono font-semibold tabular" style={{ color: clr(row.status.daily_pnl) }}>{fmt(row.status.daily_pnl)}</div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Today</div>
                          <div style={{ color: "var(--text)" }}>{row.status.trades_today} trades</div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)" }}>Status</div>
                          <span className="px-2 py-0.5 rounded text-[9px] font-semibold" style={{
                            background: row.status.status === "active" ? "rgba(0,212,170,0.15)" : row.status.status === "eval_passed" ? "rgba(245,158,11,0.15)" : "rgba(239,68,68,0.15)",
                            color: row.status.status === "active" ? "#00b894" : row.status.status === "eval_passed" ? "var(--amber)" : "var(--red)",
                          }}>
                            {row.status.status === "active" ? "Active" : row.status.status === "eval_passed" ? "Eval Passed" : row.status.status === "breached" ? "Breached" : "Daily Limit"}
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {/* Fleet strip */}
      <div className="flex items-center gap-5 px-4 py-2 text-[11px]" style={{ color: "var(--text-muted)" }}>
        <span>Connected: <span style={{ color: "var(--text)" }}>{fleet.connected}/{fleet.total}</span></span>
        <span>&middot;</span>
        <span>Day P&L: <span className="font-mono tabular" style={{ color: clr(fleet.fleetDayPnl) }}>{fmt(fleet.fleetDayPnl)}</span></span>
        <span>&middot;</span>
        <span>Total P&L: <span className="font-mono tabular" style={{ color: clr(fleet.fleetTotalPnl) }}>{fmt(fleet.fleetTotalPnl)}</span></span>
      </div>

      {/* Copy Activity */}
      <div className="panel rounded overflow-hidden">
        <div className="px-3 py-1.5 text-[10px] font-normal tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Copy Activity</div>
        <div className="max-h-48 overflow-y-auto">
          {copyFeed.length === 0 ? (
            <div className="py-8 text-center text-xs" style={{ color: "var(--text-dim)" }}>No copy activity yet</div>
          ) : (
            copyFeed.map((entry, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{entry.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span style={{ color: "var(--text)" }}>
                  {entry.action === "entry" ? `${entry.side} ${entry.contracts} MNQ @${entry.fill_price?.toFixed(2) ?? "—"}` : `Exit ${entry.strategy} ${entry.exit_reason ?? ""}`}
                </span>
                {entry.pnl !== undefined && entry.action === "exit" && (
                  <span className="font-mono tabular" style={{ color: clr(entry.pnl) }}>{fmt(entry.pnl)}</span>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
