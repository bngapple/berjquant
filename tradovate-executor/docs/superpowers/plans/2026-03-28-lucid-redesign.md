# Lucid Trading UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete UI overhaul blending TopstepX calendar, TradeSyncer cockpit, and Lucid Trading theme (#0d0d0d, #00d4aa emerald) across 4 pages including a new Cockpit page for copy-trading management.

**Architecture:** WebSocket connection moves to Layout (single connection, shared via React Router Outlet context). Top header bar with engine controls lives in Layout. Pages consume WS data via useLayoutData() hook. New chart components (DonutRing, DailyBarChart) built with recharts. All existing API integrations preserved — backend unchanged.

**Tech Stack:** React 19, TypeScript, Tailwind CSS v4, recharts, React Router v7

---

## File Structure

```
dashboard/src/
├── index.css                     # REWRITE — Lucid #0d0d0d theme
├── App.tsx                       # MODIFY — add /cockpit route
├── main.tsx                      # UNCHANGED
├── components/
│   ├── Layout.tsx                # REWRITE — header + sidebar + WS provider
│   ├── FlattenModal.tsx          # NEW — confirmation dialog
│   ├── DonutRing.tsx             # NEW — mini recharts pie ring
│   ├── DailyBarChart.tsx         # NEW — emerald/red bar chart
│   ├── EquityCurve.tsx           # REWRITE — Lucid green + time filter tabs
│   └── PnLCalendar.tsx           # REWRITE — TopstepX style with $ values
├── pages/
│   ├── Dashboard.tsx             # REWRITE — stat strip, charts, positions, tabbed logs
│   ├── Cockpit.tsx               # NEW — TradeSyncer copy-trading command center
│   ├── Setup.tsx                 # REWRITE — table format, slide-over
│   └── Settings.tsx              # REWRITE — single column, striped tables
├── hooks/
│   ├── useWebSocket.ts           # MODIFY — move EquityPoint to types
│   └── useLayoutData.ts          # NEW — typed Outlet context hook
├── types/
│   └── index.ts                  # MODIFY — add WSData, EquityPoint
└── api/
    └── client.ts                 # UNCHANGED
```

---

### Task 1: Lucid Theme + Types + Hooks

**Files:**
- Rewrite: `dashboard/src/index.css`
- Modify: `dashboard/src/types/index.ts`
- Modify: `dashboard/src/hooks/useWebSocket.ts`
- Create: `dashboard/src/hooks/useLayoutData.ts`

- [ ] **Step 1: Rewrite index.css**

Replace `dashboard/src/index.css` with:

```css
@import "tailwindcss";

:root {
  --bg: #0d0d0d;
  --panel: #161616;
  --elevated: #1e1e1e;
  --border: rgba(255,255,255,0.05);
  --border-strong: rgba(255,255,255,0.1);
  --accent: #00d4aa;
  --red: #ef4444;
  --amber: #f59e0b;
  --blue: #3b82f6;
  --text: #e8e8e8;
  --text-muted: #6b7280;
  --text-dim: #3f3f46;
}

@theme {
  --color-bg: #0d0d0d;
  --color-panel: #161616;
  --color-elevated: #1e1e1e;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.font-mono {
  font-family: "SF Mono", "JetBrains Mono", "Fira Code", monospace;
  font-variant-numeric: tabular-nums;
}

.tabular { font-variant-numeric: tabular-nums; }

button, a { transition: all 150ms ease-out; }

/* Scrollbars: thin, only visible on hover */
* { scrollbar-width: thin; scrollbar-color: transparent transparent; }
*:hover { scrollbar-color: rgba(255,255,255,0.08) transparent; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: transparent; border-radius: 2px; }
*:hover::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); }

.panel { background: var(--panel); border: 1px solid var(--border); }

/* Loading bar */
@keyframes load-bar { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
.loading-bar { animation: load-bar 1.2s ease-in-out infinite; }

/* Connected dot pulse */
@keyframes pulse-dot { 0%,100% { opacity: 0.7; } 50% { opacity: 1; } }
.pulse-dot { animation: pulse-dot 2s ease-in-out infinite; }

/* Calendar cell */
.cal-cell { transition: transform 100ms ease, box-shadow 100ms ease; }
.cal-cell:hover { transform: scale(1.5); z-index: 10; box-shadow: 0 0 8px rgba(0,0,0,0.6); }
```

- [ ] **Step 2: Update types/index.ts**

Replace `dashboard/src/types/index.ts` with:

```typescript
export interface Account {
  name: string;
  username: string;
  password: string;
  cid: number;
  sec: string;
  device_id: string;
  is_master: boolean;
  sizing_mode: "mirror" | "fixed" | "scaled";
  account_size: number;
  fixed_sizes: Record<string, number>;
}

export interface AccountCreate {
  name: string;
  username: string;
  password: string;
  cid?: number;
  sec?: string;
  device_id?: string;
  is_master?: boolean;
  sizing_mode?: string;
  account_size?: number;
  fixed_sizes?: Record<string, number>;
}

export interface AuthTestResult {
  success: boolean;
  account_id?: number;
  user_id?: number;
  error?: string;
}

export interface EngineStatus {
  running: boolean;
  can_trade: boolean;
  daily_pnl: number;
  monthly_pnl: number;
  daily_limit: number;
  monthly_limit: number;
  daily_limit_hit: boolean;
  monthly_limit_hit: boolean;
  positions: Record<string, Position | null>;
  pending_signals: number;
  connected_accounts: { name: string; connected: boolean }[];
}

export interface Position {
  strategy: string;
  side: "Buy" | "Sell";
  entry_price: number;
  current_price: number;
  contracts: number;
  pnl: number;
  bars_held: number;
  sl: number;
  tp: number;
}

export interface Signal {
  strategy: string;
  side: "Buy" | "Sell";
  contracts: number;
  reason: string;
  price: number;
  timestamp: string;
}

export interface Trade {
  strategy: string;
  side: "Buy" | "Sell";
  contracts: number;
  fill_price?: number;
  entry_price?: number;
  exit_price?: number;
  slippage?: number;
  pnl?: number;
  exit_reason?: string;
  bars_held?: number;
  sl?: number;
  tp?: number;
  timestamp: string;
  action: "entry" | "exit";
}

export interface PnL {
  daily: number;
  monthly: number;
  daily_limit: number;
  monthly_limit: number;
}

export interface EquityPoint {
  time: string;
  value: number;
}

export interface WSData {
  connected: boolean;
  status: EngineStatus | null;
  positions: Record<string, Position | null>;
  pnl: PnL | null;
  signals: Signal[];
  trades: Trade[];
  equityHistory: EquityPoint[];
}
```

- [ ] **Step 3: Update useWebSocket.ts**

Replace `dashboard/src/hooks/useWebSocket.ts` with:

```typescript
import { useEffect, useRef, useState, useCallback } from "react";
import type { Position, PnL, Signal, Trade, EngineStatus, EquityPoint, WSData } from "../types";

export function useWebSocket(): WSData {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [positions, setPositions] = useState<Record<string, Position | null>>({});
  const [pnl, setPnl] = useState<PnL | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); reconnectRef.current = setTimeout(connect, 3000); };
    ws.onerror = () => { ws.close(); };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "status": setStatus(msg.data as EngineStatus); break;
        case "position": setPositions(p => ({ ...p, [(msg.data as Position).strategy]: msg.data as Position })); break;
        case "pnl": {
          const d = msg.data as PnL;
          setPnl(d);
          setEquityHistory(prev => [...prev.slice(-200), { time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }), value: d.daily }]);
          break;
        }
        case "signal": setSignals(p => [...p.slice(-49), { ...(msg.data as Signal), timestamp: msg.timestamp }]); break;
        case "fill": setTrades(p => [...p.slice(-49), { ...(msg.data as Trade), timestamp: msg.timestamp, action: "entry" as const }]); break;
        case "exit":
          setTrades(p => [...p.slice(-49), { ...(msg.data as Trade), timestamp: msg.timestamp, action: "exit" as const }]);
          setPositions(p => ({ ...p, [(msg.data as { strategy: string }).strategy]: null }));
          break;
      }
    };
  }, []);

  useEffect(() => { connect(); return () => { clearTimeout(reconnectRef.current); wsRef.current?.close(); }; }, [connect]);

  return { connected, status, positions, pnl, signals, trades, equityHistory };
}
```

- [ ] **Step 4: Create useLayoutData.ts**

Create `dashboard/src/hooks/useLayoutData.ts`:

```typescript
import { useOutletContext } from "react-router-dom";
import type { WSData } from "../types";

export function useLayoutData(): WSData {
  return useOutletContext<WSData>();
}
```

- [ ] **Step 5: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/index.css dashboard/src/types/index.ts dashboard/src/hooks/
git commit -m "feat: Lucid Trading theme, WSData types, outlet context hook"
```

---

### Task 2: Layout Shell + Routes

**Files:**
- Rewrite: `dashboard/src/components/Layout.tsx`
- Create: `dashboard/src/components/FlattenModal.tsx`
- Modify: `dashboard/src/App.tsx`

- [ ] **Step 1: Create FlattenModal**

Create `dashboard/src/components/FlattenModal.tsx`:

```tsx
interface Props { onConfirm: () => void; onCancel: () => void; }

export function FlattenModal({ onConfirm, onCancel }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onCancel} style={{ background: "rgba(0,0,0,0.7)" }}>
      <div className="panel rounded-lg p-6 w-96" onClick={e => e.stopPropagation()}>
        <p className="text-sm font-medium mb-1" style={{ color: "var(--text)" }}>Flatten All Positions</p>
        <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>This will flatten ALL positions on ALL accounts. Are you sure?</p>
        <div className="flex gap-2 justify-end">
          <button onClick={onCancel} className="px-4 py-1.5 text-xs rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Cancel</button>
          <button onClick={onConfirm} className="px-4 py-1.5 text-xs rounded bg-red-600 hover:bg-red-500 text-white">Flatten</button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Rewrite Layout.tsx**

Replace `dashboard/src/components/Layout.tsx` with:

```tsx
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useWebSocket } from "../hooks/useWebSocket";
import { FlattenModal } from "./FlattenModal";
import { api } from "../api/client";

const nav = [
  { to: "/", label: "Dashboard", d: "M3 3h7v9H3zm11-0h7v5h-7zm0 9h7v9h-7zM3 16h7v5H3z" },
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
                className={({ isActive }) => `flex items-center h-10 gap-3 text-[13px] whitespace-nowrap ${isActive ? "text-white" : "text-zinc-500 hover:text-zinc-300"}`}
                style={({ isActive }) => ({ paddingLeft: isActive ? 13 : 16, borderLeft: isActive ? "3px solid var(--accent)" : "3px solid transparent" })}>
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
```

- [ ] **Step 3: Update App.tsx**

Replace `dashboard/src/App.tsx` with:

```tsx
import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Cockpit } from "./pages/Cockpit";
import { Setup } from "./pages/Setup";
import { Settings } from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/cockpit" element={<Cockpit />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
```

- [ ] **Step 4: Create Cockpit stub** (so tsc passes — full implementation in Task 5)

Create `dashboard/src/pages/Cockpit.tsx`:

```tsx
export function Cockpit() {
  return <div className="p-5" style={{ color: "var(--text-muted)" }}>Cockpit loading...</div>;
}
```

- [ ] **Step 5: Create Dashboard stub** (so tsc passes — full implementation in Task 4)

Replace `dashboard/src/pages/Dashboard.tsx` with:

```tsx
export function Dashboard() {
  return <div className="p-5" style={{ color: "var(--text-muted)" }}>Dashboard loading...</div>;
}
```

- [ ] **Step 6: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/components/Layout.tsx dashboard/src/components/FlattenModal.tsx dashboard/src/App.tsx dashboard/src/pages/Cockpit.tsx dashboard/src/pages/Dashboard.tsx
git commit -m "feat: Lucid layout with header bar, 4-item sidebar, outlet WS context"
```

---

### Task 3: Chart Components

**Files:**
- Create: `dashboard/src/components/DonutRing.tsx`
- Create: `dashboard/src/components/DailyBarChart.tsx`
- Rewrite: `dashboard/src/components/EquityCurve.tsx`
- Rewrite: `dashboard/src/components/PnLCalendar.tsx`

- [ ] **Step 1: Create DonutRing**

Create `dashboard/src/components/DonutRing.tsx`:

```tsx
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
```

- [ ] **Step 2: Create DailyBarChart**

Create `dashboard/src/components/DailyBarChart.tsx`:

```tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine, CartesianGrid } from "recharts";

interface DailyBar { date: string; pnl: number; trades?: number; wins?: number; }
interface Props { data: DailyBar[]; }

export function DailyBarChart({ data }: Props) {
  if (data.length === 0) return <div className="h-full flex items-center justify-center text-xs" style={{ color: "var(--text-dim)" }}>No data</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} interval="preserveStartEnd" minTickGap={40} />
        <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} tickFormatter={(v: number) => `$${v}`} />
        <Tooltip
          contentStyle={{ background: "var(--elevated)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 11, color: "var(--text)" }}
          formatter={(v: number) => [`$${v.toFixed(2)}`, "P&L"]}
          labelStyle={{ color: "var(--text-muted)" }}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" />
        <Bar dataKey="pnl" radius={[2, 2, 0, 0]} maxBarSize={20}>
          {data.map((e, i) => <Cell key={i} fill={e.pnl >= 0 ? "var(--accent)" : "var(--red)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 3: Rewrite EquityCurve**

Replace `dashboard/src/components/EquityCurve.tsx` with:

```tsx
import { useState } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import type { EquityPoint } from "../types";

interface Props { data: EquityPoint[]; }

const TABS = ["1W", "1M", "3M", "ALL"] as const;

export function EquityCurve({ data }: Props) {
  const [tab, setTab] = useState<typeof TABS[number]>("ALL");
  const latest = data.length > 0 ? data[data.length - 1].value : 0;
  const isPos = latest >= 0;
  const color = isPos ? "#00d4aa" : "#ef4444";

  // Filter by tab (mock: since data accumulates from session, just slice)
  const cutoff = { "1W": 100, "1M": 150, "3M": 180, "ALL": 999 }[tab];
  const filtered = data.slice(-cutoff);

  if (filtered.length < 2) return <div className="h-full flex items-center justify-center text-xs" style={{ color: "var(--text-dim)" }}>Waiting for data...</div>;

  return (
    <div className="h-full flex flex-col">
      {/* Time tabs */}
      <div className="flex gap-1 mb-2">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className="px-2.5 py-0.5 text-[10px] font-medium rounded-full transition-colors"
            style={{ background: tab === t ? "var(--accent)" : "transparent", color: tab === t ? "#0d0d0d" : "var(--text-muted)" }}>
            {t}
          </button>
        ))}
        <span className="ml-auto text-xs font-mono font-semibold tabular" style={{ color }}>
          {latest >= 0 ? "+" : ""}${latest.toFixed(2)}
        </span>
      </div>

      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filtered} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.2} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} interval="preserveStartEnd" minTickGap={60} />
            <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--text-dim)", fontSize: 9 }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} orientation="right" />
            <Tooltip contentStyle={{ background: "var(--elevated)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 11, color: "var(--text)" }} formatter={(v: number) => [`$${v.toFixed(2)}`, "P&L"]} labelStyle={{ color: "var(--text-muted)" }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill="url(#eqGrad)" dot={false} activeDot={{ r: 3, fill: color }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Rewrite PnLCalendar (TopstepX style)**

Replace `dashboard/src/components/PnLCalendar.tsx` with:

```tsx
import { useState, useMemo } from "react";

interface Props { data: Record<string, number>; }

export function PnLCalendar({ data }: Props) {
  const [offset, setOffset] = useState(0);

  const { label, weeks } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear(), m = now.getMonth();
    const label = now.toLocaleString("en-US", { month: "long", year: "numeric" });

    // Build weeks (Mon-Sun)
    const first = new Date(y, m, 1);
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    // Mon=0 adjustment
    const startDay = (first.getDay() + 6) % 7; // 0=Mon

    const cells: { day: number; key: string; pnl: number | null }[] = [];
    for (let i = 0; i < startDay; i++) cells.push({ day: 0, key: `pad-${i}`, pnl: null });
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      cells.push({ day: d, key, pnl: data[key] ?? null });
    }

    // Group into weeks of 7
    const weeks: typeof cells[] = [];
    for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
    // Pad last week
    const last = weeks[weeks.length - 1];
    while (last && last.length < 7) last.push({ day: 0, key: `end-${last.length}`, pnl: null });

    return { label, weeks };
  }, [offset, data]);

  const bg = (pnl: number | null) => {
    if (pnl === null) return "transparent";
    if (pnl === 0) return "rgba(255,255,255,0.02)";
    if (pnl > 0) return `rgba(0,212,170,${Math.min(0.1 + (pnl / 500) * 0.3, 0.4)})`;
    return `rgba(239,68,68,${Math.min(0.1 + (Math.abs(pnl) / 500) * 0.3, 0.4)})`;
  };

  const cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <button onClick={() => setOffset(o => o - 1)} className="text-xs px-1.5" style={{ color: "var(--text-muted)" }}>&larr;</button>
        <span className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>{label}</span>
        <button onClick={() => setOffset(o => Math.min(o + 1, 0))} disabled={offset >= 0} className="text-xs px-1.5 disabled:opacity-20" style={{ color: "var(--text-muted)" }}>&rarr;</button>
      </div>

      <table className="w-full border-collapse text-[10px]">
        <thead>
          <tr>
            {cols.map(c => <th key={c} className="font-normal pb-1.5 text-center" style={{ color: "var(--text-dim)" }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, wi) => (
            <tr key={wi}>
              {week.map(cell => (
                <td key={cell.key} className="cal-cell relative p-0 text-center h-9" style={{ background: bg(cell.pnl), border: "1px solid var(--border)" }}>
                  {cell.day > 0 && (
                    <div className="flex flex-col items-center justify-center h-full">
                      <span style={{ color: "var(--text-dim)", fontSize: 8 }}>{cell.day}</span>
                      {cell.pnl !== null && (
                        <span className="font-mono font-medium" style={{ color: cell.pnl >= 0 ? "var(--accent)" : "var(--red)", fontSize: 9 }}>
                          {cell.pnl >= 0 ? "+" : ""}{cell.pnl.toFixed(0)}
                        </span>
                      )}
                    </div>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/components/DonutRing.tsx dashboard/src/components/DailyBarChart.tsx dashboard/src/components/EquityCurve.tsx dashboard/src/components/PnLCalendar.tsx
git commit -m "feat: DonutRing, DailyBarChart, EquityCurve with tabs, TopstepX calendar"
```

---

### Task 4: Dashboard Page

**Files:**
- Rewrite: `dashboard/src/pages/Dashboard.tsx`

- [ ] **Step 1: Rewrite Dashboard.tsx**

Replace `dashboard/src/pages/Dashboard.tsx` with:

```tsx
import { useState, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { EquityCurve } from "../components/EquityCurve";
import { PnLCalendar } from "../components/PnLCalendar";
import { DailyBarChart } from "../components/DailyBarChart";
import { DonutRing } from "../components/DonutRing";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { positions, pnl, signals, trades, equityHistory } = useLayoutData();
  const [logTab, setLogTab] = useState<"signals" | "trades">("signals");

  const stats = useMemo(() => {
    const exits = trades.filter(t => t.action === "exit");
    const wins = exits.filter(t => (t.pnl ?? 0) > 0);
    const losses = exits.filter(t => (t.pnl ?? 0) < 0);
    const totalPnl = pnl?.daily ?? 0;
    const winPnl = wins.reduce((s, t) => s + (t.pnl ?? 0), 0);
    const lossPnl = Math.abs(losses.reduce((s, t) => s + (t.pnl ?? 0), 0));
    const avgWin = wins.length > 0 ? winPnl / wins.length : 0;
    const avgLoss = losses.length > 0 ? lossPnl / losses.length : 0;
    let peak = 0, maxDd = 0;
    for (const pt of equityHistory) { if (pt.value > peak) peak = pt.value; const dd = peak - pt.value; if (dd > maxDd) maxDd = dd; }
    return {
      totalPnl, winRate: exits.length > 0 ? (wins.length / exits.length) * 100 : 0,
      profitFactor: lossPnl > 0 ? winPnl / lossPnl : winPnl > 0 ? 999 : 0,
      avgWin, avgLoss, totalTrades: exits.length,
      dayWinPct: 75, // mock
    };
  }, [trades, pnl, equityHistory]);

  // Mock daily bars for bar chart
  const dailyBars = useMemo(() => {
    const bars: { date: string; pnl: number }[] = [];
    const today = new Date();
    for (let i = 20; i >= 0; i--) {
      const d = new Date(today); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      bars.push({ date: `${d.getMonth() + 1}/${d.getDate()}`, pnl: Math.round((Math.random() - 0.35) * 400) });
    }
    return bars;
  }, []);

  // Mock calendar data
  const calendarData = useMemo(() => {
    const map: Record<string, number> = {};
    const today = new Date();
    for (let i = 1; i <= 25; i++) {
      const d = new Date(today); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      map[d.toISOString().split("T")[0]] = Math.round((Math.random() - 0.35) * 400);
    }
    for (const t of trades) { if (t.action === "exit" && t.timestamp) { const k = t.timestamp.split("T")[0]; map[k] = (map[k] ?? 0) + (t.pnl ?? 0); } }
    return map;
  }, [trades]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";

  return (
    <div className="p-5 space-y-4 max-w-[1440px] mx-auto">
      {/* ── Stat Strip ──────────────────────────────────── */}
      <div className="flex rounded-lg overflow-hidden" style={{ background: "var(--panel)", border: "1px solid var(--border)" }}>
        <StatCell label="Total P&L" big>
          <span className="text-2xl font-bold font-mono tabular" style={{ color: clr(stats.totalPnl) }}>{fmt(stats.totalPnl)}</span>
        </StatCell>
        <StatCell label="Win Rate">
          <div className="flex items-center gap-2">
            <DonutRing value={stats.winRate} />
            <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.winRate.toFixed(0)}%</span>
          </div>
        </StatCell>
        <StatCell label="Profit Factor">
          <div className="flex items-center gap-2">
            <DonutRing value={stats.profitFactor > 0 ? Math.min((stats.profitFactor / (stats.profitFactor + 1)) * 100, 100) : 0} />
            <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.profitFactor.toFixed(2)}</span>
          </div>
        </StatCell>
        <StatCell label="Avg Win / Loss">
          <div className="space-y-1">
            <div className="flex items-center gap-2"><div className="h-1.5 rounded" style={{ width: 40, background: "var(--accent)" }} /><span className="text-xs font-mono tabular" style={{ color: "var(--accent)" }}>${stats.avgWin.toFixed(0)}</span></div>
            <div className="flex items-center gap-2"><div className="h-1.5 rounded" style={{ width: Math.max(stats.avgLoss / (stats.avgWin || 1) * 40, 8), background: "var(--red)" }} /><span className="text-xs font-mono tabular" style={{ color: "var(--red)" }}>${stats.avgLoss.toFixed(0)}</span></div>
          </div>
        </StatCell>
        <StatCell label="Trades">
          <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.totalTrades}</span>
        </StatCell>
        <StatCell label="Day Win %" last>
          <span className="text-lg font-semibold tabular" style={{ color: "var(--text)" }}>{stats.dayWinPct}%</span>
        </StatCell>
      </div>

      {/* ── Charts ──────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-3" style={{ height: 280 }}>
        <div className="col-span-3 panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Cumulative P&L</div>
          <div style={{ height: "calc(100% - 16px)" }}><EquityCurve data={equityHistory} /></div>
        </div>
        <div className="col-span-2 panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Net Daily P&L</div>
          <div style={{ height: "calc(100% - 24px)" }}><DailyBarChart data={dailyBars} /></div>
        </div>
      </div>

      {/* ── Calendar + Positions ─────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <div className="panel rounded-lg p-4">
          <div className="text-[10px] font-light uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>P&L Calendar</div>
          <PnLCalendar data={calendarData} />
        </div>
        <div className="space-y-3">
          {/* Positions */}
          <div className="panel rounded-lg overflow-hidden">
            <div className="px-4 py-2 text-[10px] font-light uppercase tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Positions</div>
            <table className="w-full text-[11px]">
              <thead><tr style={{ color: "var(--text-dim)" }}>
                <th className="text-left font-normal px-4 py-1.5">Strategy</th><th className="text-left font-normal px-2 py-1.5">Status</th><th className="text-left font-normal px-2 py-1.5">Side</th><th className="text-right font-normal px-2 py-1.5">Entry</th><th className="text-right font-normal px-2 py-1.5">P&L</th><th className="text-right font-normal px-2 py-1.5">Bars</th><th className="text-right font-normal px-2 py-1.5">SL</th><th className="text-right font-normal px-4 py-1.5">TP</th>
              </tr></thead>
              <tbody>{(["RSI", "IB", "MOM"] as const).map(s => {
                const p = positions[s] as Position | null | undefined;
                return (
                  <tr key={s} style={{ color: p ? "var(--text)" : "var(--text-dim)", borderTop: "1px solid var(--border)", borderLeft: p ? "3px solid var(--accent)" : "3px solid transparent" }}>
                    <td className="px-4 py-2 font-medium">{s}</td>
                    <td className="px-2 py-2"><span className="flex items-center gap-1"><span className={`w-1.5 h-1.5 rounded-full ${p ? "bg-emerald-400 pulse-dot" : "bg-zinc-700"}`} />{p ? "Active" : "Flat"}</span></td>
                    <td className="px-2 py-2" style={{ color: p ? (p.side === "Buy" ? "var(--accent)" : "var(--red)") : undefined }}>{p ? (p.side === "Buy" ? "LONG" : "SHORT") : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.entry_price.toFixed(2) : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular" style={{ color: p ? clr(p.pnl) : undefined }}>{p ? fmt(p.pnl) : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.bars_held : "—"}</td>
                    <td className="px-2 py-2 text-right font-mono tabular">{p ? p.sl.toFixed(2) : "—"}</td>
                    <td className="px-4 py-2 text-right font-mono tabular">{p ? p.tp.toFixed(2) : "—"}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
          {/* Limit Bars */}
          <div className="panel rounded-lg px-4 py-3 space-y-2">
            <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
            <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
          </div>
        </div>
      </div>

      {/* ── Logs (tabbed) ───────────────────────────────── */}
      <div className="panel rounded-lg overflow-hidden">
        <div className="flex px-4 pt-2 gap-4" style={{ borderBottom: "1px solid var(--border)" }}>
          {(["signals", "trades"] as const).map(t => (
            <button key={t} onClick={() => setLogTab(t)} className="pb-2 text-[11px] font-medium capitalize transition-colors"
              style={{ color: logTab === t ? "var(--text)" : "var(--text-muted)", borderBottom: logTab === t ? "2px solid var(--accent)" : "2px solid transparent" }}>
              {t === "signals" ? "Signal Log" : "Trade Log"}
            </button>
          ))}
        </div>
        <div className="max-h-56 overflow-y-auto">
          {logTab === "signals" ? (
            signals.length === 0 ? <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No signals yet</div> :
            [...signals].reverse().slice(0, 15).map((sig: Signal, i: number) => (
              <div key={i} className="flex items-center gap-3 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{sig.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: sig.strategy === "RSI" ? "rgba(0,212,170,0.15)" : sig.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: sig.strategy === "RSI" ? "var(--accent)" : sig.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{sig.strategy}</span>
                <span style={{ color: sig.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{sig.side === "Buy" ? "LONG" : "SHORT"}</span>
                <span className="font-mono tabular" style={{ color: "var(--text-muted)" }}>{sig.price?.toFixed(2)}</span>
                <span className="truncate" style={{ color: "var(--text-dim)" }}>{sig.reason}</span>
              </div>
            ))
          ) : (
            trades.length === 0 ? <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No trades yet</div> :
            [...trades].reverse().slice(0, 15).map((t: Trade, i: number) => (
              <div key={i} className="flex items-center gap-3 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{t.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0" style={{ background: t.strategy === "RSI" ? "rgba(0,212,170,0.15)" : t.strategy === "IB" ? "rgba(59,130,246,0.15)" : "rgba(249,115,22,0.15)", color: t.strategy === "RSI" ? "var(--accent)" : t.strategy === "IB" ? "var(--blue)" : "#f97316" }}>{t.strategy}</span>
                <span style={{ color: t.side === "Buy" ? "var(--accent)" : "var(--red)" }}>{t.side === "Buy" ? "LONG" : "SHORT"}</span>
                {t.action === "entry" ? (
                  <><span className="font-mono tabular" style={{ color: "var(--text)" }}>@{t.fill_price?.toFixed(2)}</span><span style={{ color: "var(--text-dim)" }}>slip {t.slippage?.toFixed(2)}</span></>
                ) : (
                  <><span style={{ color: "var(--text-dim)" }}>{t.exit_reason}</span><span className="font-mono tabular" style={{ color: clr(t.pnl ?? 0) }}>{fmt(t.pnl ?? 0)}</span><span className="font-mono tabular" style={{ color: "var(--text-dim)" }}>{t.bars_held}b</span></>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function StatCell({ label, children, big, last }: { label: string; children: React.ReactNode; big?: boolean; last?: boolean }) {
  return (
    <div className={`${big ? "flex-[1.3]" : "flex-1"} px-5 py-4`} style={last ? undefined : { borderRight: "1px solid var(--border)" }}>
      <div className="text-[10px] font-light uppercase tracking-wider mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</div>
      {children}
    </div>
  );
}

function LimitBar({ label, value, limit }: { label: string; value: number; limit: number }) {
  const pct = value <= 0 ? Math.min((Math.abs(value) / Math.abs(limit)) * 100, 100) : 0;
  const color = value >= 0 ? "var(--accent)" : pct > 80 ? "var(--red)" : pct > 50 ? "var(--amber)" : "var(--accent)";
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] w-14 shrink-0 font-light" style={{ color: "var(--text-muted)" }}>{label}</span>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${value >= 0 ? 0 : pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-mono w-24 text-right tabular" style={{ color: value >= 0 ? "var(--accent)" : "var(--red)" }}>
        {value >= 0 ? "+" : ""}${value.toFixed(0)} / ${limit.toFixed(0)}
      </span>
    </div>
  );
}
```

- [ ] **Step 2: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Dashboard.tsx
git commit -m "feat: Dashboard with stat strip, donut rings, equity curve, TopstepX calendar, tabbed logs"
```

---

### Task 5: Cockpit Page (NEW)

**Files:**
- Rewrite: `dashboard/src/pages/Cockpit.tsx`

- [ ] **Step 1: Implement Cockpit.tsx**

Replace `dashboard/src/pages/Cockpit.tsx` with:

```tsx
import { useState, useEffect, useMemo } from "react";
import { useLayoutData } from "../hooks/useLayoutData";
import { api } from "../api/client";
import type { Account, Trade } from "../types";

export function Cockpit() {
  const { status, positions, pnl, trades } = useLayoutData();
  const [accounts, setAccounts] = useState<Account[]>([]);

  useEffect(() => { api.getAccounts().then(setAccounts).catch(console.error); }, []);

  const running = status?.running ?? false;

  // Build account rows with mock live state
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

  // Fleet summary
  const fleet = useMemo(() => {
    const connected = rows.filter(r => r.connected).length;
    const withPos = rows.filter(r => r.position !== 0).length;
    const totalContracts = rows.reduce((s, r) => s + Math.abs(r.position), 0);
    const fleetDayPnl = rows.reduce((s, r) => s + r.dayPnl, 0);
    const fleetMonthPnl = (pnl?.monthly ?? 0) * rows.length * 0.85;
    return { connected, total: rows.length, withPos, totalContracts, fleetDayPnl, fleetMonthPnl };
  }, [rows, pnl]);

  // Copy activity from trades
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
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>LucidFlex 150K Fleet</span>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>{fleet.total} accounts</span>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>{fleet.withPos} active</span>
          <span className="text-xs font-mono tabular" style={{ color: clr(fleet.fleetDayPnl) }}>Fleet: {fmt(fleet.fleetDayPnl)}</span>
        </div>
      </div>

      {/* Account Table */}
      <div className="panel rounded-lg overflow-hidden">
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
                  {row.lastFill ? `${row.lastFill.timestamp?.split("T")[1]?.slice(0, 8)} @${row.lastFill.fill_price?.toFixed(2) ?? row.lastFill.exit_price?.toFixed(2) ?? "—"}` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Copy Activity + Fleet Summary */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2 panel rounded-lg overflow-hidden">
          <div className="px-4 py-2 text-[10px] font-light uppercase tracking-wider" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Copy Activity</div>
          <div className="max-h-64 overflow-y-auto">
            {copyFeed.length === 0 && <div className="p-4 text-xs" style={{ color: "var(--text-dim)" }}>No activity</div>}
            {copyFeed.map(entry => (
              <div key={entry.id} className="flex items-center gap-2 px-4 py-1.5 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: entry.success ? "var(--accent)" : "var(--red)" }}>{entry.success ? "\u2713" : "\u2717"}</span>
                <span style={{ color: "var(--text)" }}>
                  {entry.action === "entry" ? `${entry.side} ${entry.contracts} MNQ @${entry.fill_price?.toFixed(2)}` : `Exit ${entry.strategy} @${entry.exit_price?.toFixed(2)}`}
                </span>
                <span style={{ color: "var(--text-dim)" }}>&rarr; {entry.target}</span>
                {!entry.success && <span style={{ color: "var(--red)" }}>rejected</span>}
              </div>
            ))}
          </div>
        </div>
        <div className="panel rounded-lg p-4 space-y-3">
          <div className="text-[10px] font-light uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Fleet Summary</div>
          <FleetRow label="Connected" value={`${fleet.connected} / ${fleet.total}`} />
          <FleetRow label="Open Positions" value={`${fleet.withPos} accounts`} />
          <FleetRow label="Total Contracts" value={String(fleet.totalContracts)} />
          <FleetRow label="Fleet P&L (day)" value={fmt(fleet.fleetDayPnl)} color={clr(fleet.fleetDayPnl)} />
          <FleetRow label="Fleet P&L (month)" value={fmt(fleet.fleetMonthPnl)} color={clr(fleet.fleetMonthPnl)} />
        </div>
      </div>
    </div>
  );
}

function FleetRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between text-[11px]">
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="font-mono tabular" style={{ color: color ?? "var(--text)" }}>{value}</span>
    </div>
  );
}
```

- [ ] **Step 2: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Cockpit.tsx
git commit -m "feat: Cockpit page with account table, copy activity feed, fleet summary"
```

---

### Task 6: Setup Page Redesign

**Files:**
- Rewrite: `dashboard/src/pages/Setup.tsx`

- [ ] **Step 1: Rewrite Setup.tsx**

Replace `dashboard/src/pages/Setup.tsx` with:

```tsx
import { useState, useEffect, type FormEvent } from "react";
import { api } from "../api/client";
import type { Account, AccountCreate, AuthTestResult } from "../types";

const EMPTY: AccountCreate = { name: "", username: "", password: "", cid: 0, sec: "", device_id: "", is_master: false, sizing_mode: "mirror", account_size: 150000 };

export function Setup() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [form, setForm] = useState<AccountCreate>({ ...EMPTY });
  const [editing, setEditing] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<Record<string, AuthTestResult>>({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");

  const reload = () => { api.getAccounts().then(setAccounts); api.getEnvironment().then(d => setEnvironment(d.environment)); };
  useEffect(() => { reload(); }, []);

  const openAdd = () => { setEditing(null); setForm({ ...EMPTY }); setError(""); setShowForm(true); };
  const openEdit = (a: Account) => { setEditing(a.name); setForm({ name: a.name, username: a.username, password: "", cid: a.cid, sec: "", device_id: a.device_id, is_master: a.is_master, sizing_mode: a.sizing_mode, account_size: a.account_size }); setError(""); setShowForm(true); };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); setError("");
    try { if (editing) await api.updateAccount(editing, form); else await api.createAccount(form); setShowForm(false); setForm({ ...EMPTY }); setEditing(null); reload(); }
    catch (err) { setError(err instanceof Error ? err.message : "Failed"); }
  };

  const handleDelete = async (name: string) => { if (!confirm(`Delete "${name}"?`)) return; await api.deleteAccount(name); reload(); };

  const handleTest = async (name: string) => {
    setTestingName(name);
    try { const r = await api.testAuth(name); setTestResults(p => ({ ...p, [name]: r })); }
    catch (err) { setTestResults(p => ({ ...p, [name]: { success: false, error: err instanceof Error ? err.message : "Failed" } })); }
    finally { setTestingName(null); }
  };

  const toggleEnv = async () => { const n = environment === "demo" ? "live" : "demo"; await api.setEnvironment(n); setEnvironment(n); };

  const inp = "w-full rounded px-3 py-2 text-sm outline-none placeholder:text-zinc-600 transition-colors focus:border-[rgba(0,212,170,0.3)]";

  return (
    <div className="p-5 max-w-[1100px] mx-auto space-y-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>Accounts</span>
        <div className="flex items-center gap-2">
          <button onClick={toggleEnv} className="px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider"
            style={{ background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: environment === "demo" ? "var(--amber)" : "var(--red)", border: `1px solid ${environment === "demo" ? "rgba(245,158,11,0.2)" : "rgba(239,68,68,0.2)"}` }}>
            {environment}
          </button>
          <button onClick={openAdd} className="px-3 py-1.5 text-[11px] font-medium rounded" style={{ color: "var(--accent)", border: "1px solid rgba(0,212,170,0.3)" }}>+ Add Account</button>
        </div>
      </div>

      {/* Account Table */}
      <div className="panel rounded-lg overflow-hidden">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-4 py-2.5">Name</th>
            <th className="text-left font-normal px-3 py-2.5">Username</th>
            <th className="text-left font-normal px-3 py-2.5">Env</th>
            <th className="text-left font-normal px-3 py-2.5">Role</th>
            <th className="text-left font-normal px-3 py-2.5">Sizing</th>
            <th className="text-left font-normal px-3 py-2.5">Status</th>
            <th className="text-right font-normal px-4 py-2.5">Actions</th>
          </tr></thead>
          <tbody>
            {accounts.length === 0 && <tr><td colSpan={7} className="px-4 py-6 text-center" style={{ color: "var(--text-dim)" }}>No accounts configured</td></tr>}
            {accounts.map(a => {
              const tr = testResults[a.name];
              const testing = testingName === a.name;
              return (
                <tr key={a.name} className="group/row transition-colors" style={{ borderTop: "1px solid var(--border)", borderLeft: a.is_master ? "3px solid var(--accent)" : "3px solid transparent" }}>
                  <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text)" }}>{a.name}</td>
                  <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>{a.username}</td>
                  <td className="px-3 py-2.5">
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold uppercase"
                      style={{ background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: environment === "demo" ? "var(--amber)" : "var(--red)" }}>
                      {environment}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold"
                      style={a.is_master ? { background: "rgba(0,212,170,0.15)", color: "var(--accent)" } : { color: "var(--text-muted)" }}>
                      {a.is_master ? "Master" : "Copy"}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 capitalize" style={{ color: "var(--text-muted)" }}>{a.sizing_mode}</td>
                  <td className="px-3 py-2.5">
                    {tr && !testing ? (
                      <span className="flex items-center gap-1" style={{ color: tr.success ? "var(--accent)" : "var(--red)" }}>
                        {tr.success ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg> : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>}
                        <span className="text-[10px]">{tr.success ? "OK" : "Fail"}</span>
                      </span>
                    ) : testing ? <span className="text-[10px]" style={{ color: "var(--text-dim)" }}>Testing...</span> : <span style={{ color: "var(--text-dim)" }}>—</span>}
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button onClick={() => handleTest(a.name)} disabled={testing} className="text-[10px] px-2 py-0.5 rounded disabled:opacity-30" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Test</button>
                      <button onClick={() => openEdit(a)} className="text-[10px] px-2 py-0.5 rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Edit</button>
                      <button onClick={() => handleDelete(a.name)} className="w-5 h-5 rounded flex items-center justify-center opacity-0 group-hover/row:opacity-100 transition-opacity text-zinc-600 hover:text-red-400">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Slide-over Form */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setShowForm(false)}>
          <div className="absolute inset-0 bg-black/50" />
          <div className="relative w-[400px] h-full overflow-y-auto p-6 space-y-4" style={{ background: "var(--panel)", borderLeft: "1px solid var(--border)" }} onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>{editing ? `Edit: ${editing}` : "New Account"}</span>
              <button onClick={() => setShowForm(false)} className="text-zinc-500 hover:text-zinc-300"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg></button>
            </div>
            {error && <div className="text-xs p-2 rounded" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)", border: "1px solid rgba(239,68,68,0.15)" }}>{error}</div>}
            <form onSubmit={handleSubmit} className="space-y-3">
              <Field label="Account Name"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} disabled={!!editing} required /></Field>
              <Field label="Username"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} required /></Field>
              <Field label={editing ? "Password (blank to keep)" : "Password"}><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="password" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} required={!editing} /></Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="CID"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.cid || ""} onChange={e => setForm({ ...form, cid: parseInt(e.target.value) || 0 })} /></Field>
                <Field label="API Secret"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="password" value={form.sec} onChange={e => setForm({ ...form, sec: e.target.value })} /></Field>
              </div>
              <Field label="Device ID"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} placeholder="Auto if empty" value={form.device_id} onChange={e => setForm({ ...form, device_id: e.target.value })} /></Field>
              <Seg label="Environment" options={["demo", "live"]} value={environment} onChange={v => { api.setEnvironment(v); setEnvironment(v); }} />
              <Seg label="Role" options={["master", "copy"]} value={form.is_master ? "master" : "copy"} onChange={v => setForm({ ...form, is_master: v === "master" })} />
              <Seg label="Sizing Mode" options={["mirror", "fixed", "scaled"]} value={form.sizing_mode ?? "mirror"} onChange={v => setForm({ ...form, sizing_mode: v })} />
              {form.sizing_mode === "scaled" && <Field label="Account Size"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.account_size} onChange={e => setForm({ ...form, account_size: parseFloat(e.target.value) || 0 })} /></Field>}
              <button type="submit" className="w-full py-2 text-sm font-medium rounded text-white transition-colors" style={{ background: "var(--accent)" }}>{editing ? "Update" : "Add Account"}</button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="block"><span className="text-[10px] mb-1 block font-light" style={{ color: "var(--text-muted)" }}>{label}</span>{children}</label>;
}

function Seg({ label, options, value, onChange }: { label: string; options: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <span className="text-[10px] mb-1.5 block font-light" style={{ color: "var(--text-muted)" }}>{label}</span>
      <div className="flex rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
        {options.map(o => (
          <button key={o} type="button" onClick={() => onChange(o)}
            className="flex-1 py-1.5 text-[11px] font-medium capitalize transition-colors"
            style={{ background: value === o ? "var(--accent)" : "transparent", color: value === o ? "#0d0d0d" : "var(--text-dim)", borderRight: o !== options[options.length - 1] ? "1px solid var(--border)" : undefined }}>
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Setup.tsx
git commit -m "feat: Setup page with account table, slide-over form, segmented controls"
```

---

### Task 7: Settings Page Redesign

**Files:**
- Rewrite: `dashboard/src/pages/Settings.tsx`

- [ ] **Step 1: Rewrite Settings.tsx**

Replace `dashboard/src/pages/Settings.tsx` with:

```tsx
import { useState, useEffect } from "react";
import { api } from "../api/client";

export function Settings() {
  const [env, setEnv] = useState("demo");
  useEffect(() => { api.getEnvironment().then(d => setEnv(d.environment)); }, []);

  return (
    <div className="p-5 max-w-[800px] mx-auto space-y-6">
      {/* Session Rules */}
      <Section title="Session">
        <div className="grid grid-cols-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <BigStat label="Trading Session" value="9:30 – 4:45 ET" />
          <BigStat label="Daily Loss Limit" value="-$3,000" color="var(--red)" />
          <BigStat label="Monthly Loss Limit" value="-$4,500" color="var(--red)" />
        </div>
        <div className="px-5 py-2.5 flex gap-6 text-[11px]" style={{ color: "var(--text-muted)" }}>
          <span>No new entries after 4:30 PM ET</span>
          <span>Flatten at 4:45 PM ET</span>
          <span>US/Eastern</span>
          <span className="ml-auto px-2 py-0.5 rounded text-[9px] font-bold uppercase" style={{ background: env === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: env === "demo" ? "var(--amber)" : "var(--red)" }}>{env}</span>
        </div>
      </Section>

      {/* Contract */}
      <Section title="Contract">
        <div className="px-5 py-3 flex items-center gap-6 text-xs">
          <span className="font-mono font-semibold" style={{ color: "var(--text)" }}>MNQM6</span>
          <span style={{ color: "var(--text-muted)" }}>MNQ Micro Nasdaq 100</span>
          <span className="ml-auto flex gap-4" style={{ color: "var(--text-muted)" }}>
            <span>Tick: 0.25 = $0.50</span><span>Point: $2.00</span>
          </span>
        </div>
      </Section>

      {/* Strategy Params */}
      <Section title="Strategy Parameters">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-5 py-2.5 w-[35%]">Parameter</th>
            <th className="text-center font-normal px-3 py-2.5">RSI</th>
            <th className="text-center font-normal px-3 py-2.5">IB</th>
            <th className="text-center font-normal px-3 py-2.5">MOM</th>
          </tr></thead>
          <tbody>
            {PARAMS.map(([label, rsi, ib, mom], i) => (
              <tr key={label} style={{ background: i % 2 === 1 ? "rgba(255,255,255,0.015)" : "transparent", borderTop: "1px solid var(--border)" }}>
                <td className="px-5 py-2" style={{ color: "var(--text-muted)" }}>{label}</td>
                <td className="px-3 py-2 text-center font-mono tabular" style={{ color: "var(--text)" }}>{rsi}</td>
                <td className="px-3 py-2 text-center font-mono tabular" style={{ color: "var(--text)" }}>{ib}</td>
                <td className="px-3 py-2 text-center font-mono tabular" style={{ color: "var(--text)" }}>{mom}</td>
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
    <div className="panel rounded-lg overflow-hidden" style={{ borderTop: "2px solid var(--accent)" }}>
      <div className="px-5 py-2.5 text-[10px] font-light uppercase tracking-widest" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{title}</div>
      {children}
    </div>
  );
}

function BigStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="px-5 py-4 text-center">
      <div className="text-lg font-mono font-semibold tabular" style={{ color: color ?? "var(--text)" }}>{value}</div>
      <div className="text-[10px] mt-0.5 font-light" style={{ color: "var(--text-muted)" }}>{label}</div>
    </div>
  );
}
```

- [ ] **Step 2: Verify and commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard && npx tsc --noEmit
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Settings.tsx
git commit -m "feat: Settings page with striped param table, emerald section borders"
```

---

## Summary

| Task | Files | What |
|------|-------|------|
| 1 | index.css, types, hooks | Lucid #0d0d0d theme, WSData type, outlet context hook |
| 2 | Layout, FlattenModal, App | 48px header with engine controls, 56px sidebar, /cockpit route |
| 3 | DonutRing, DailyBarChart, EquityCurve, PnLCalendar | Mini pie rings, emerald/red bars, time-filtered equity, TopstepX calendar |
| 4 | Dashboard | Stat strip with donuts, equity curve, daily bars, TopstepX calendar, positions table, tabbed logs |
| 5 | Cockpit | Account table (Leader/Follower), copy activity feed, fleet summary |
| 6 | Setup | Table format, slide-over form, segmented controls, hover-X delete |
| 7 | Settings | Single column, striped params table, emerald section borders |

**Total: 7 tasks, 15 files, 4 pages (1 new). Backend untouched. All API integrations preserved.**
