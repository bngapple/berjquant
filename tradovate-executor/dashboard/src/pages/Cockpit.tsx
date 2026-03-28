import { useState, useEffect, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { api } from "../api/client";
import type { Account, Trade } from "../types";

export function Cockpit() {
  const { status, positions, pnl, trades } = useLayoutData();
  const [accounts, setAccounts] = useState<Account[]>([]);

  useEffect(() => { api.getAccounts().then(setAccounts).catch(console.error); }, []);

  const running = status?.running ?? false;

  const rows = useMemo(() => {
    const activePos = Object.values(positions).filter(p => p !== null);
    const netContracts = activePos.reduce((s, p) => s + (p!.contracts * (p!.side === "Buy" ? 1 : -1)), 0);
    const openPnl = activePos.reduce((s, p) => s + p!.pnl, 0);
    const dayPnl = pnl?.daily ?? 0;
    const lastTrade = trades.length > 0 ? trades[trades.length - 1] : null;

    return accounts.map(acct => {
      const isLeader = acct.is_master;
      const connected = status?.connected_accounts.find(a => a.name === acct.name)?.connected ?? false;
      const scale = acct.sizing_mode === "mirror" ? 1 : acct.sizing_mode === "scaled" ? acct.account_size / 150000 : 1;
      return {
        name: acct.name, isLeader, sizing: acct.sizing_mode, connected,
        fixedSizes: acct.fixed_sizes,
        position: isLeader ? netContracts : Math.round(netContracts * scale),
        openPnl: isLeader ? openPnl : openPnl * scale * (0.9 + Math.random() * 0.2),
        dayPnl: isLeader ? dayPnl : dayPnl * scale * (0.9 + Math.random() * 0.2),
        lastFill: lastTrade,
      };
    });
  }, [accounts, status, positions, pnl, trades]);

  const fleet = useMemo(() => {
    const connected = rows.filter(r => r.connected).length;
    const withPos = rows.filter(r => r.position !== 0).length;
    const totalContracts = rows.reduce((s, r) => s + Math.abs(r.position), 0);
    const fleetDayPnl = rows.reduce((s, r) => s + r.dayPnl, 0);
    const fleetMonthPnl = (pnl?.monthly ?? 0) * rows.length * 0.85;
    return { connected, total: rows.length, withPos, totalContracts, fleetDayPnl, fleetMonthPnl };
  }, [rows, pnl]);

  const copyFeed = useMemo(() => {
    return [...trades].reverse().slice(0, 10).map((t, i) => {
      const target = accounts[1 + (i % Math.max(accounts.length - 1, 1))]?.name ?? "copy-1";
      const success = Math.random() > 0.1;
      return { ...t, target, success, id: i };
    });
  }, [trades, accounts]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  return (
    <div className="p-5 space-y-4 max-w-[1440px] mx-auto">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>LucidFlex 150K Fleet</span>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>{fleet.total} accounts</span>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>{fleet.withPos} active</span>
          <span className="text-xs font-mono tabular" style={{ color: clr(fleet.fleetDayPnl) }}>Fleet: {fmt(fleet.fleetDayPnl)}</span>
        </div>
      </div>

      <div className="panel rounded overflow-hidden">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-4 py-2.5">Status</th>
            <th className="text-left font-normal px-3 py-2.5">Account</th>
            <th className="text-left font-normal px-3 py-2.5">Role</th>
            <th className="text-left font-normal px-3 py-2.5">Sizing</th>
            <th className="text-left font-normal px-3 py-2.5">Position</th>
            <th className="text-right font-normal px-3 py-2.5">Open P&L</th>
            <th className="text-right font-normal px-3 py-2.5">Day P&L</th>
            <th className="text-right font-normal px-4 py-2.5">Last Fill</th>
          </tr></thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.name} style={{
                background: row.isLeader ? "var(--elevated)" : "var(--panel)",
                borderTop: "1px solid var(--border)",
                borderLeft: row.isLeader ? "3px solid var(--accent)" : "3px solid transparent",
              }}>
                <td className="px-4 py-2.5">
                  <span className={`w-2 h-2 rounded-full inline-block ${row.connected ? "bg-emerald-400 pulse-dot" : "bg-red-500"}`} />
                </td>
                <td className="px-3 py-2.5 font-medium" style={{ color: "var(--text)" }}>{row.name}</td>
                <td className="px-3 py-2.5">
                  <span className="px-2 py-0.5 rounded text-[9px] font-semibold"
                    style={row.isLeader ? { background: "rgba(0,212,170,0.15)", color: "var(--accent)" } : { background: "transparent", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
                    {row.isLeader ? "Leader" : "Follower"}
                  </span>
                </td>
                <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>
                  {row.sizing === "mirror" ? "Mirror" : row.sizing === "fixed" ? `Fixed (${Object.values(row.fixedSizes).join("/")})` : "Scaled"}
                </td>
                <td className="px-3 py-2.5 font-mono tabular" style={{ color: row.position > 0 ? "var(--accent)" : row.position < 0 ? "var(--red)" : "var(--text-dim)" }}>
                  {row.position > 0 ? `+${row.position} LONG` : row.position < 0 ? `${row.position} SHORT` : "FLAT"}
                </td>
                <td className="px-3 py-2.5 text-right font-mono tabular" style={{ color: clr(row.openPnl) }}>{fmt(row.openPnl)}</td>
                <td className="px-3 py-2.5 text-right font-mono tabular" style={{ color: clr(row.dayPnl) }}>{fmt(row.dayPnl)}</td>
                <td className="px-4 py-2.5 text-right font-mono tabular" style={{ color: "var(--text-muted)" }}>
                  {row.lastFill ? `${row.lastFill.timestamp?.split("T")[1]?.slice(0, 8)} @${row.lastFill.fill_price?.toFixed(2) ?? row.lastFill.exit_price?.toFixed(2) ?? "\u2014"}` : "\u2014"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-6 px-4 py-2.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
        <span>Connected: <span style={{ color: "var(--text)" }}>{fleet.connected}/{fleet.total}</span></span>
        <span>Open: <span style={{ color: "var(--text)" }}>{fleet.withPos}</span></span>
        <span>Contracts: <span style={{ color: "var(--text)" }}>{fleet.totalContracts}</span></span>
        <span>Day P&L: <span className="font-mono tabular" style={{ color: clr(fleet.fleetDayPnl) }}>{fmt(fleet.fleetDayPnl)}</span></span>
        <span>Month P&L: <span className="font-mono tabular" style={{ color: clr(fleet.fleetMonthPnl) }}>{fmt(fleet.fleetMonthPnl)}</span></span>
      </div>

      <div className="panel rounded overflow-hidden">
        <div className="px-4 py-2 text-[10px] font-normal tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Copy Activity</div>
        <div className="max-h-48 overflow-y-auto">
          {copyFeed.length === 0 ? (
            <div className="py-8 text-center text-xs" style={{ color: "var(--text-dim)" }}>No copy activity yet</div>
          ) : (
            copyFeed.map(entry => (
              <div key={entry.id} className="flex items-center gap-2 px-4 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: entry.success ? "var(--accent)" : "var(--red)" }}>{entry.success ? "\u2713" : "\u2717"}</span>
                <span style={{ color: "var(--text)" }}>
                  {entry.action === "entry" ? `${entry.side} ${entry.contracts} MNQ @${entry.fill_price?.toFixed(2)}` : `Exit ${entry.strategy} @${entry.exit_price?.toFixed(2)}`}
                </span>
                <span style={{ color: "var(--text-dim)" }}>&rarr; {entry.target}</span>
                {!entry.success && <span style={{ color: "var(--red)" }}>rejected</span>}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
