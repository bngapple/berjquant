import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useWebSocket } from "../hooks/useWebSocket";
import { FlattenModal } from "./FlattenModal";
import { api } from "../api/client";

const nav = [
  { to: "/", label: "Terminal", d: "M3 3h18v3H3zM3 9h12v3H3zm0 6h9v3H3zm14-3l4 4-4 4" },
  { to: "/dashboard", label: "Dashboard", d: "M3 3h7v9H3zm11-0h7v5h-7zm0 9h7v9h-7zM3 16h7v5H3z" },
  { to: "/calendar", label: "Calendar", d: "M8 7V3m8 4V3m-9 4h10a2 2 0 012 2v11a2 2 0 01-2 2H7a2 2 0 01-2-2V9a2 2 0 012-2zm-2 4h14" },
  { to: "/cockpit", label: "Cockpit", d: "M4 6h6v6H4zm10 0h6v6h-6zM7 14v4m10-4v4M7 18h10" },
  { to: "/setup", label: "Setup", d: "M12 12a4 4 0 100-8 4 4 0 000 8zm0 2c-4 0-8 2-8 4v2h16v-2c0-2-4-4-8-4z" },
  { to: "/settings", label: "Settings", d: "M4 21V14m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6" },
];

export function Layout() {
  const ws = useWebSocket();
  const [loading, setLoading] = useState("");
  const [showFlatten, setShowFlatten] = useState(false);
  const running = ws.status?.running ?? false;

  const act = async (key: string, fn: () => Promise<unknown>) => {
    setLoading(key);
    try { await fn(); } catch (e) { console.error(e); } finally { setLoading(""); }
  };

  return (
    <div className="flex flex-col h-screen" style={{ background: "var(--bg)" }}>
      {/* ── Top Header (48px) ────────────────────────────── */}
      <header className="flex items-center justify-between h-12 px-4 shrink-0" style={{ background: "var(--bg)", borderBottom: "1px solid var(--border)" }}>
        <span className="text-sm font-semibold tracking-tight" style={{ color: "var(--text)" }}>HTF Executor</span>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${running ? "bg-emerald-400 pulse-dot" : "bg-zinc-600"}`} />
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>{running ? "Running" : "Stopped"}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={() => act("start", api.startEngine)} disabled={running || loading === "start"}
            className="px-3 py-1 text-[11px] font-medium rounded disabled:opacity-30" style={{ color: "var(--accent)", border: "1px solid rgba(0,212,170,0.3)" }}>
            {loading === "start" ? "..." : "Start"}
          </button>
          <button onClick={() => act("stop", api.stopEngine)} disabled={!running || loading === "stop"}
            className="px-3 py-1 text-[11px] font-medium rounded disabled:opacity-30" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>
            Stop
          </button>
          <button onClick={() => setShowFlatten(true)} disabled={loading === "flatten"}
            className="px-3 py-1 text-[11px] font-medium rounded bg-red-600 hover:bg-red-500 text-white disabled:opacity-30">
            Flatten All
          </button>
        </div>
      </header>

      {showFlatten && <FlattenModal onConfirm={async () => { setShowFlatten(false); await act("flatten", api.flattenAll); }} onCancel={() => setShowFlatten(false)} />}

      {/* ── Body ─────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar (56px, expand to 200px on hover) */}
        <nav className="group/s flex flex-col shrink-0 w-14 hover:w-[200px] transition-all duration-150 overflow-hidden"
          style={{ background: "var(--bg)", borderRight: "1px solid var(--border)" }}>
          <div className="flex-1 py-3 flex flex-col gap-0.5">
            {nav.map(n => (
              <NavLink key={n.to} to={n.to} end={n.to === "/"}
                className={({ isActive }) => `relative flex items-center h-10 gap-3 text-[13px] whitespace-nowrap ${isActive ? "text-white" : "text-zinc-500 hover:text-zinc-300"}`}
                style={({ isActive }) => ({ paddingLeft: isActive ? 13 : 16, borderLeft: isActive ? "3px solid rgba(255,255,255,0.5)" : "3px solid transparent" })}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                  <path d={n.d} />
                </svg>
                <span className="opacity-0 group-hover/s:opacity-100 transition-opacity duration-150">{n.label}</span>
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Content */}
        <main className="flex-1 overflow-auto">
          {loading && <div className="h-0.5 overflow-hidden" style={{ background: "var(--panel)" }}><div className="h-full w-1/3 loading-bar" style={{ background: "var(--accent)" }} /></div>}
          <Outlet context={ws} />
        </main>
      </div>
    </div>
  );
}
