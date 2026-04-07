import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useWebSocket } from "../hooks/useWebSocket";
import { FlattenModal } from "./FlattenModal";
import { api } from "../api/client";

const nav = [
  { to: "/dashboard", label: "Dashboard", d: "M3 3h7v9H3zm11-0h7v5h-7zm0 9h7v9h-7zM3 16h7v5H3z" },
  { to: "/calendar", label: "Calendar", d: "M8 7V3m8 4V3m-9 4h10a2 2 0 012 2v11a2 2 0 01-2 2H7a2 2 0 01-2-2V9a2 2 0 012-2zm-2 4h14" },
  { to: "/setup", label: "Accounts", d: "M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 7a4 4 0 100 8 4 4 0 000-8zm13 10v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" },
  { to: "/settings", label: "Settings", d: "M4 21V14m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6" },
];

export function Layout() {
  const ws = useWebSocket();
  const [loading, setLoading] = useState("");
  const [showFlatten, setShowFlatten] = useState(false);
  const [notice, setNotice] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const running = ws.status?.running ?? false;

  useEffect(() => {
    if (!notice) {
      return undefined;
    }
    const timeout = window.setTimeout(() => setNotice(null), 4000);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  const act = async (key: string, fn: () => Promise<{ status?: string }>) => {
    setLoading(key);
    setNotice(null);
    try {
      const result = await fn();
      if (key === "start") {
        setNotice({ tone: "success", text: "Engine started." });
      } else if (key === "stop") {
        setNotice({ tone: "success", text: "Engine stopped." });
      } else if (key === "flatten") {
        setNotice({ tone: "success", text: result.status ?? "Flatten request sent." });
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Action failed.";
      const normalized = message.toLowerCase();
      if (key === "start" && normalized.includes("already running")) {
        setNotice({ tone: "info", text: "Engine is already running." });
      } else if (key === "stop" && normalized.includes("not running")) {
        setNotice({ tone: "info", text: "Engine is already stopped." });
      } else {
        setNotice({ tone: "error", text: message });
        console.error(e);
      }
    } finally {
      await ws.refreshStatus();
      if (key === "start" || key === "stop") {
        window.setTimeout(() => { void ws.refreshStatus(); }, 900);
      }
      setLoading("");
    }
  };

  const startActive = !running;
  const stopActive = running;

  return (
    <div className="flex flex-col h-screen" style={{ background: "var(--bg)" }}>
      <div style={{ height: 2, background: "linear-gradient(90deg, transparent 0%, #00d4aa 40%, #00b894 60%, transparent 100%)", flexShrink: 0 }} />
      {/* ── Top Header (48px) ────────────────────────────── */}
      <header className="flex items-center justify-between h-12 px-4 shrink-0" style={{ background: "var(--bg)", borderBottom: "1px solid var(--border)" }}>
        <span className="text-sm font-semibold tracking-tight glow-accent" style={{ color: "var(--accent)", letterSpacing: "0.06em", fontFamily: "'SF Mono', 'JetBrains Mono', monospace" }}>HTF EXECUTOR</span>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs" style={{ background: running ? "rgba(0,212,170,0.08)" : "rgba(255,255,255,0.04)", border: `1px solid ${running ? "rgba(0,212,170,0.2)" : "rgba(255,255,255,0.06)"}` }}>
          <span className={`w-1.5 h-1.5 rounded-full ${running ? "bg-emerald-400 pulse-dot" : "bg-zinc-600"}`} />
          <span className="text-[10px] font-medium" style={{ color: running ? "var(--accent)" : "var(--text-muted)" }}>{running ? "LIVE" : "OFFLINE"}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {notice && (
            <div
              className="hidden md:flex items-center px-2.5 py-1 rounded-full text-[10px] font-medium max-w-[220px] truncate"
              title={notice.text}
              style={{
                background:
                  notice.tone === "success"
                    ? "rgba(0,212,170,0.10)"
                    : notice.tone === "info"
                      ? "rgba(255,255,255,0.06)"
                      : "rgba(239,68,68,0.12)",
                color:
                  notice.tone === "success"
                    ? "var(--accent)"
                    : notice.tone === "info"
                      ? "var(--text)"
                      : "#fca5a5",
                border:
                  notice.tone === "success"
                    ? "1px solid rgba(0,212,170,0.22)"
                    : notice.tone === "info"
                      ? "1px solid rgba(255,255,255,0.10)"
                      : "1px solid rgba(239,68,68,0.22)",
              }}
            >
              {notice.text}
            </div>
          )}
          <button onClick={() => act("start", api.startEngine)} disabled={running || loading === "start"}
            className="px-3 py-1 text-[11px] font-semibold rounded disabled:opacity-30 transition-all"
            style={{
              background: startActive ? "rgba(0,212,170,0.14)" : "transparent",
              color: startActive ? "var(--accent)" : "var(--text-muted)",
              border: startActive ? "1px solid rgba(0,212,170,0.28)" : "1px solid var(--border)",
              boxShadow: startActive ? "inset 0 0 12px rgba(0,212,170,0.08)" : "none",
            }}>
            {loading === "start" ? "Starting..." : "Start"}
          </button>
          <button onClick={() => act("stop", api.stopEngine)} disabled={!running || loading === "stop"}
            className="px-3 py-1 text-[11px] font-medium rounded disabled:opacity-30 transition-all"
            style={{
              background: stopActive ? "rgba(239,68,68,0.12)" : "transparent",
              color: stopActive ? "#fca5a5" : "var(--text-muted)",
              border: stopActive ? "1px solid rgba(239,68,68,0.24)" : "1px solid var(--border)",
              boxShadow: stopActive ? "inset 0 0 12px rgba(239,68,68,0.08)" : "none",
            }}>
            {loading === "stop" ? "Stopping..." : "Stop"}
          </button>
          <button onClick={() => setShowFlatten(true)} disabled={loading === "flatten"}
            className="px-3 py-1 text-[11px] font-medium rounded bg-red-600 hover:bg-red-500 text-white disabled:opacity-30 hover:shadow-[0_0_12px_rgba(239,68,68,0.4)]">
            {loading === "flatten" ? "Flattening..." : "Flatten All"}
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
                className={({ isActive }) =>
                  `relative flex items-center h-10 gap-3 text-[13px] whitespace-nowrap transition-colors ${
                    isActive ? "text-white" : "text-zinc-600 hover:text-zinc-400"
                  }`
                }
                style={({ isActive }) => ({
                  paddingLeft: isActive ? 13 : 16,
                  borderLeft: isActive ? "2px solid var(--accent)" : "2px solid transparent",
                  background: isActive ? "rgba(0,212,170,0.05)" : "transparent",
                  boxShadow: isActive ? "inset 2px 0 8px rgba(0,212,170,0.08)" : "none",
                })}>
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
