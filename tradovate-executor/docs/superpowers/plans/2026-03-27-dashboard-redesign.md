# Dashboard UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign all three frontend pages from boxy card-grid to a professional trading terminal aesthetic (TradeSea/Lucid Trading style) — near-black backgrounds, slim icon sidebar, equity curve, P&L calendar, compact positions table, and refined Setup/Settings pages.

**Architecture:** Pure frontend rewrite — no backend changes. Install recharts for the equity curve. Replace Tailwind gray defaults with custom CSS variables for near-black color scheme. Slim the sidebar to icon-only (expand on hover). Split Dashboard into composable sections with proper visual hierarchy. Redesign Setup with modal form and animated feedback. Clean up Settings to table format.

**Tech Stack:** React 18, TypeScript, Tailwind CSS v4, recharts, existing API client + WebSocket hook

---

## File Structure

```
dashboard/src/
├── index.css                    # REWRITE — custom CSS vars, near-black theme
├── App.tsx                      # UNCHANGED
├── main.tsx                     # UNCHANGED
├── components/
│   ├── Layout.tsx               # REWRITE — slim icon sidebar, expand on hover
│   ├── StatusBadge.tsx          # DELETE — no longer needed
│   ├── EquityCurve.tsx          # NEW — recharts area chart
│   └── PnLCalendar.tsx          # NEW — monthly P&L heatmap grid
├── pages/
│   ├── Dashboard.tsx            # REWRITE — complete redesign
│   ├── Setup.tsx                # REWRITE — modal form, segmented control, hover-delete
│   └── Settings.tsx             # REWRITE — table format
├── hooks/
│   └── useWebSocket.ts          # MODIFY — add equityHistory accumulation
├── types/
│   └── index.ts                 # UNCHANGED
└── api/
    └── client.ts                # UNCHANGED
```

---

### Task 1: Install recharts, update CSS theme, update WebSocket hook

**Files:**
- Modify: `dashboard/package.json` (add recharts)
- Rewrite: `dashboard/src/index.css`
- Modify: `dashboard/src/hooks/useWebSocket.ts`

- [ ] **Step 1: Install recharts**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npm install recharts
```

- [ ] **Step 2: Rewrite index.css with custom theme**

Replace `dashboard/src/index.css` with:

```css
@import "tailwindcss";

/* ── Custom color tokens ───────────────────────────────────── */
:root {
  --bg-base: #0a0a0f;
  --bg-panel: #111116;
  --bg-surface: #1a1a22;
  --bg-hover: #222230;
  --border: rgba(255, 255, 255, 0.06);
  --border-strong: rgba(255, 255, 255, 0.1);
  --text-primary: #e4e4e7;
  --text-secondary: #71717a;
  --text-muted: #3f3f46;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-yellow: #f59e0b;
  --accent-blue: #3b82f6;
}

@theme {
  --color-base: #0a0a0f;
  --color-panel: #111116;
  --color-surface: #1a1a22;
  --color-hover: #222230;
  --color-subtle-border: rgba(255, 255, 255, 0.06);
  --color-pnl-green: #10b981;
  --color-pnl-red: #ef4444;
}

body {
  background: var(--bg-base);
  color: var(--text-primary);
  font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.font-mono {
  font-family: "SF Mono", "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
}

/* ── Global transitions ────────────────────────────────────── */
* {
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

button, a, [role="button"] {
  transition: all 150ms ease;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar {
  width: 4px;
  height: 4px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.15);
}

/* ── Utility classes ───────────────────────────────────────── */
.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
}

.surface {
  background: var(--bg-surface);
  border: 1px solid var(--border);
}

/* Skeleton loading */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton {
  background: linear-gradient(90deg, var(--bg-surface) 25%, var(--bg-hover) 50%, var(--bg-surface) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

/* P&L calendar cell hover */
.cal-cell {
  transition: transform 100ms ease, box-shadow 100ms ease;
}
.cal-cell:hover {
  transform: scale(1.8);
  z-index: 10;
  box-shadow: 0 0 8px rgba(0,0,0,0.5);
}
```

- [ ] **Step 3: Update useWebSocket to track equity history**

Replace `dashboard/src/hooks/useWebSocket.ts` with:

```typescript
import { useEffect, useRef, useState, useCallback } from "react";
import type { Position, PnL, Signal, Trade, EngineStatus } from "../types";

export interface EquityPoint {
  time: string;
  value: number;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [positions, setPositions] = useState<Record<string, Position | null>>(
    {}
  );
  const [pnl, setPnl] = useState<PnL | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case "status":
          setStatus(msg.data as EngineStatus);
          break;
        case "position":
          setPositions((prev) => ({
            ...prev,
            [(msg.data as Position).strategy]: msg.data as Position,
          }));
          break;
        case "pnl": {
          const pnlData = msg.data as PnL;
          setPnl(pnlData);
          setEquityHistory((prev) => [
            ...prev.slice(-200),
            {
              time: new Date().toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
              }),
              value: pnlData.daily,
            },
          ]);
          break;
        }
        case "signal":
          setSignals((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Signal), timestamp: msg.timestamp },
          ]);
          break;
        case "fill":
          setTrades((prev) => [
            ...prev.slice(-49),
            {
              ...(msg.data as Trade),
              timestamp: msg.timestamp,
              action: "entry" as const,
            },
          ]);
          break;
        case "exit":
          setTrades((prev) => [
            ...prev.slice(-49),
            {
              ...(msg.data as Trade),
              timestamp: msg.timestamp,
              action: "exit" as const,
            },
          ]);
          setPositions((prev) => ({
            ...prev,
            [(msg.data as { strategy: string }).strategy]: null,
          }));
          break;
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    connected,
    status,
    positions,
    pnl,
    signals,
    trades,
    equityHistory,
  };
}
```

- [ ] **Step 4: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/package.json dashboard/package-lock.json dashboard/src/index.css dashboard/src/hooks/useWebSocket.ts
git commit -m "feat: install recharts, custom dark theme, equity history tracking"
```

---

### Task 2: Layout Redesign — Slim Icon Sidebar

**Files:**
- Rewrite: `dashboard/src/components/Layout.tsx`
- Delete: `dashboard/src/components/StatusBadge.tsx`

- [ ] **Step 1: Rewrite Layout.tsx with slim icon sidebar**

Replace `dashboard/src/components/Layout.tsx` with:

```tsx
import { NavLink, Outlet } from "react-router-dom";

const navItems = [
  {
    to: "/",
    label: "Dashboard",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="9" rx="1" />
        <rect x="14" y="3" width="7" height="5" rx="1" />
        <rect x="14" y="12" width="7" height="9" rx="1" />
        <rect x="3" y="16" width="7" height="5" rx="1" />
      </svg>
    ),
  },
  {
    to: "/setup",
    label: "Setup",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
      </svg>
    ),
  },
  {
    to: "/settings",
    label: "Settings",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
        <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
        <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
        <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" />
      </svg>
    ),
  },
];

export function Layout() {
  return (
    <div className="flex h-screen" style={{ background: "var(--bg-base)" }}>
      {/* Slim sidebar — icon only, expand on hover */}
      <nav className="group/nav flex flex-col shrink-0 w-[52px] hover:w-[180px] transition-all duration-200 overflow-hidden"
           style={{ background: "var(--bg-panel)", borderRight: "1px solid var(--border)" }}>
        {/* Logo */}
        <div className="flex items-center h-12 px-3.5 gap-3 shrink-0"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="w-[22px] h-[22px] rounded bg-emerald-500/20 flex items-center justify-center shrink-0">
            <span className="text-emerald-400 text-[10px] font-bold">T</span>
          </div>
          <span className="text-sm font-semibold whitespace-nowrap opacity-0 group-hover/nav:opacity-100 transition-opacity duration-200"
                style={{ color: "var(--text-primary)" }}>
            Tradovate
          </span>
        </div>

        {/* Nav items */}
        <div className="flex-1 py-3 flex flex-col gap-0.5">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center h-9 px-[15px] gap-3 text-[13px] whitespace-nowrap transition-colors ${
                  isActive
                    ? "text-white"
                    : "text-zinc-500 hover:text-zinc-300"
                }`
              }
              style={({ isActive }) =>
                isActive
                  ? { background: "rgba(255,255,255,0.04)" }
                  : undefined
              }
            >
              <span className="shrink-0">{item.icon}</span>
              <span className="opacity-0 group-hover/nav:opacity-100 transition-opacity duration-200">
                {item.label}
              </span>
            </NavLink>
          ))}
        </div>

        {/* Version */}
        <div className="px-4 py-3 opacity-0 group-hover/nav:opacity-100 transition-opacity duration-200"
             style={{ borderTop: "1px solid var(--border)" }}>
          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            HTF Swing v3
          </span>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Delete StatusBadge.tsx**

Run:
```bash
rm /Users/berjourlian/berjquant/tradovate-executor/dashboard/src/components/StatusBadge.tsx
```

Remove the import from `dashboard/src/pages/Dashboard.tsx` if it references StatusBadge (it will be fully rewritten in Task 4, but ensure no build errors in between). Create a temporary stub:

Replace `dashboard/src/pages/Dashboard.tsx` with a temporary placeholder so it compiles without StatusBadge:

```tsx
export function Dashboard() {
  return <div style={{ padding: 24, color: "var(--text-secondary)" }}>Redesigning...</div>;
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/components/Layout.tsx dashboard/src/pages/Dashboard.tsx
git rm dashboard/src/components/StatusBadge.tsx
git commit -m "feat: slim icon sidebar layout, remove StatusBadge"
```

---

### Task 3: EquityCurve and PnLCalendar Components

**Files:**
- Create: `dashboard/src/components/EquityCurve.tsx`
- Create: `dashboard/src/components/PnLCalendar.tsx`

- [ ] **Step 1: Create EquityCurve component**

Create `dashboard/src/components/EquityCurve.tsx`:

```tsx
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { EquityPoint } from "../hooks/useWebSocket";

interface Props {
  data: EquityPoint[];
}

export function EquityCurve({ data }: Props) {
  const latest = data.length > 0 ? data[data.length - 1].value : 0;
  const isPositive = latest >= 0;
  const strokeColor = isPositive ? "#10b981" : "#ef4444";
  const fillColor = isPositive ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)";

  if (data.length < 2) {
    return (
      <div className="h-full flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
        <span className="text-sm">Waiting for data...</span>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={strokeColor} stopOpacity={0.15} />
            <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.03)"
          vertical={false}
        />
        <XAxis
          dataKey="time"
          axisLine={false}
          tickLine={false}
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          axisLine={false}
          tickLine={false}
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
        />
        <Tooltip
          contentStyle={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-strong)",
            borderRadius: 6,
            fontSize: 12,
            color: "var(--text-primary)",
          }}
          formatter={(value: number) => [`$${value.toFixed(2)}`, "P&L"]}
          labelStyle={{ color: "var(--text-secondary)" }}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
        <Area
          type="monotone"
          dataKey="value"
          stroke={strokeColor}
          strokeWidth={1.5}
          fill="url(#equityGrad)"
          dot={false}
          activeDot={{ r: 3, fill: strokeColor }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 2: Create PnLCalendar component**

Create `dashboard/src/components/PnLCalendar.tsx`:

```tsx
import { useState, useMemo } from "react";

interface Props {
  /** Map of "YYYY-MM-DD" → daily P&L in USD */
  data: Record<string, number>;
}

export function PnLCalendar({ data }: Props) {
  const [offset, setOffset] = useState(0); // 0 = current month, -1 = prev, etc.

  const { year, month, label, days } = useMemo(() => {
    const now = new Date();
    now.setMonth(now.getMonth() + offset);
    const y = now.getFullYear();
    const m = now.getMonth();
    const label = now.toLocaleString("en-US", { month: "long", year: "numeric" });

    const firstDay = new Date(y, m, 1).getDay(); // 0=Sun
    const daysInMonth = new Date(y, m + 1, 0).getDate();

    const days: { day: number; key: string; pnl: number | null }[] = [];
    // Padding
    for (let i = 0; i < firstDay; i++) {
      days.push({ day: 0, key: `pad-${i}`, pnl: null });
    }
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      days.push({ day: d, key, pnl: data[key] ?? null });
    }
    return { year: y, month: m, label, days };
  }, [offset, data]);

  const cellColor = (pnl: number | null): string => {
    if (pnl === null) return "transparent";
    if (pnl === 0) return "rgba(255,255,255,0.03)";
    if (pnl > 0) {
      const intensity = Math.min(pnl / 500, 1);
      return `rgba(16,185,129,${0.15 + intensity * 0.5})`;
    }
    const intensity = Math.min(Math.abs(pnl) / 500, 1);
    return `rgba(239,68,68,${0.15 + intensity * 0.5})`;
  };

  const weekdays = ["S", "M", "T", "W", "T", "F", "S"];

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setOffset((o) => o - 1)}
          className="text-zinc-500 hover:text-zinc-300 text-xs px-1.5 py-0.5"
        >
          &larr;
        </button>
        <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
          {label}
        </span>
        <button
          onClick={() => setOffset((o) => Math.min(o + 1, 0))}
          disabled={offset >= 0}
          className="text-zinc-500 hover:text-zinc-300 text-xs px-1.5 py-0.5 disabled:opacity-20"
        >
          &rarr;
        </button>
      </div>

      {/* Weekday headers */}
      <div className="grid grid-cols-7 gap-[3px] mb-[3px]">
        {weekdays.map((d, i) => (
          <div key={i} className="text-center text-[9px]" style={{ color: "var(--text-muted)" }}>
            {d}
          </div>
        ))}
      </div>

      {/* Calendar grid */}
      <div className="grid grid-cols-7 gap-[3px]">
        {days.map((d) => (
          <div
            key={d.key}
            className="cal-cell relative aspect-square rounded-[3px] flex items-center justify-center group/cell"
            style={{ background: cellColor(d.pnl) }}
            title={d.pnl !== null ? `$${d.pnl.toFixed(0)}` : undefined}
          >
            {d.day > 0 && (
              <span className="text-[8px]" style={{ color: d.pnl !== null ? "rgba(255,255,255,0.5)" : "var(--text-muted)" }}>
                {d.day}
              </span>
            )}
            {/* Tooltip on hover */}
            {d.pnl !== null && (
              <div className="absolute -top-7 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded text-[9px] font-mono whitespace-nowrap opacity-0 group-hover/cell:opacity-100 pointer-events-none z-20"
                   style={{
                     background: "var(--bg-surface)",
                     border: "1px solid var(--border-strong)",
                     color: d.pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)",
                   }}>
                {d.pnl >= 0 ? "+" : ""}${d.pnl.toFixed(0)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/components/EquityCurve.tsx dashboard/src/components/PnLCalendar.tsx
git commit -m "feat: EquityCurve (recharts) and PnLCalendar components"
```

---

### Task 4: Dashboard Page — Complete Redesign

**Files:**
- Rewrite: `dashboard/src/pages/Dashboard.tsx`

- [ ] **Step 1: Rewrite Dashboard.tsx**

Replace `dashboard/src/pages/Dashboard.tsx` with:

```tsx
import { useState, useMemo } from "react";
import { api } from "../api/client";
import { useWebSocket } from "../hooks/useWebSocket";
import { EquityCurve } from "../components/EquityCurve";
import { PnLCalendar } from "../components/PnLCalendar";
import type { Position, Signal, Trade } from "../types";

export function Dashboard() {
  const { connected, status, positions, pnl, signals, trades, equityHistory } =
    useWebSocket();
  const [loading, setLoading] = useState("");
  const [confirmFlatten, setConfirmFlatten] = useState(false);

  const running = status?.running ?? false;

  const handleAction = async (action: string, fn: () => Promise<unknown>) => {
    setLoading(action);
    try {
      await fn();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading("");
    }
  };

  const handleFlatten = async () => {
    setConfirmFlatten(false);
    await handleAction("flatten", api.flattenAll);
  };

  // Compute stats from trades
  const stats = useMemo(() => {
    const exits = trades.filter((t) => t.action === "exit");
    const wins = exits.filter((t) => (t.pnl ?? 0) > 0);
    const totalPnl = pnl?.daily ?? 0;
    const monthlyPnl = pnl?.monthly ?? 0;
    // Compute max drawdown from equity history
    let peak = 0;
    let maxDd = 0;
    for (const pt of equityHistory) {
      if (pt.value > peak) peak = pt.value;
      const dd = peak - pt.value;
      if (dd > maxDd) maxDd = dd;
    }
    return {
      totalPnl,
      monthlyPnl,
      maxDrawdown: maxDd,
      totalTrades: exits.length,
      winRate: exits.length > 0 ? (wins.length / exits.length) * 100 : 0,
    };
  }, [trades, pnl, equityHistory]);

  // Generate mock calendar data from trades
  const calendarData = useMemo(() => {
    const map: Record<string, number> = {};
    // Seed some mock history for visual richness
    const today = new Date();
    for (let i = 1; i <= 25; i++) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue; // skip weekends
      const key = d.toISOString().split("T")[0];
      map[key] = Math.round((Math.random() - 0.35) * 400);
    }
    // Add real trades
    for (const t of trades) {
      if (t.action === "exit" && t.timestamp) {
        const key = t.timestamp.split("T")[0];
        map[key] = (map[key] ?? 0) + (t.pnl ?? 0);
      }
    }
    return map;
  }, [trades]);

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const fmtPct = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
  const clr = (v: number) => (v >= 0 ? "var(--accent-green)" : "var(--accent-red)");

  return (
    <div className="p-5 space-y-4 max-w-[1400px] mx-auto">
      {/* ── Header Bar ──────────────────────────────────────── */}
      <div className="flex items-center justify-between h-10">
        <div className="flex items-center gap-3">
          <div
            className={`w-2 h-2 rounded-full ${running ? "bg-emerald-500 animate-pulse" : "bg-zinc-600"}`}
          />
          <span className="text-sm font-medium" style={{ color: running ? "var(--text-primary)" : "var(--text-secondary)" }}>
            {running ? "Engine Running" : "Engine Stopped"}
          </span>
          {/* Account status pills */}
          <div className="flex items-center gap-2 ml-4">
            {(status?.connected_accounts ?? []).map((a) => (
              <span
                key={a.name}
                className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full"
                style={{
                  background: "rgba(255,255,255,0.03)",
                  color: a.connected ? "var(--accent-green)" : "var(--text-muted)",
                }}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${a.connected ? "bg-emerald-500" : "bg-zinc-700"}`} />
                {a.name}
              </span>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleAction("start", api.startEngine)}
            disabled={running || loading === "start"}
            className="px-3 py-1.5 text-xs font-medium rounded border border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {loading === "start" ? "Starting..." : "Start"}
          </button>
          <button
            onClick={() => handleAction("stop", api.stopEngine)}
            disabled={!running || loading === "stop"}
            className="px-3 py-1.5 text-xs font-medium rounded text-zinc-400 hover:text-zinc-200 hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed"
            style={{ border: "1px solid var(--border)" }}
          >
            Stop
          </button>
          <button
            onClick={() => setConfirmFlatten(true)}
            disabled={loading === "flatten"}
            className="px-3 py-1.5 text-xs font-medium rounded border border-red-500/40 text-red-400 hover:bg-red-500/10 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Flatten All
          </button>
        </div>
      </div>

      {/* Flatten confirmation dialog */}
      {confirmFlatten && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setConfirmFlatten(false)}>
          <div className="panel rounded-lg p-5 w-80" onClick={(e) => e.stopPropagation()}>
            <p className="text-sm mb-4" style={{ color: "var(--text-primary)" }}>
              Flatten all positions across all accounts?
            </p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmFlatten(false)} className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200">
                Cancel
              </button>
              <button onClick={handleFlatten} className="px-3 py-1.5 text-xs rounded bg-red-600 hover:bg-red-500 text-white">
                Confirm Flatten
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Top Stats Bar ───────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard
          label="Daily P&L"
          value={fmt(stats.totalPnl)}
          color={clr(stats.totalPnl)}
          large
        />
        <StatCard
          label="Max Drawdown"
          value={`-$${stats.maxDrawdown.toFixed(0)}`}
          color={stats.maxDrawdown > 0 ? "var(--accent-red)" : "var(--text-secondary)"}
        />
        <StatCard
          label="Trades"
          value={String(stats.totalTrades)}
          color="var(--text-primary)"
        />
        <StatCard
          label="Win Rate"
          value={`${stats.winRate.toFixed(0)}%`}
          color={stats.winRate >= 50 ? "var(--accent-green)" : "var(--text-secondary)"}
        />
      </div>

      {/* ── Limit Bars (thin) ───────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <LimitBar label="Daily" value={pnl?.daily ?? 0} limit={-3000} />
        <LimitBar label="Monthly" value={pnl?.monthly ?? 0} limit={-4500} />
      </div>

      {/* ── Equity Curve + Calendar ─────────────────────────── */}
      <div className="grid grid-cols-5 gap-3" style={{ height: 260 }}>
        <div className="col-span-3 panel rounded-lg p-4">
          <div className="text-[11px] mb-2" style={{ color: "var(--text-secondary)" }}>
            Equity Curve
          </div>
          <div style={{ height: "calc(100% - 20px)" }}>
            <EquityCurve data={equityHistory} />
          </div>
        </div>
        <div className="col-span-2 panel rounded-lg p-4">
          <div className="text-[11px] mb-2" style={{ color: "var(--text-secondary)" }}>
            P&L Calendar
          </div>
          <PnLCalendar data={calendarData} />
        </div>
      </div>

      {/* ── Positions Table ─────────────────────────────────── */}
      <div className="panel rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>
          Positions
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th className="text-left font-normal px-4 py-2 w-20">Strategy</th>
              <th className="text-left font-normal px-3 py-2 w-16">Side</th>
              <th className="text-right font-normal px-3 py-2">Entry</th>
              <th className="text-right font-normal px-3 py-2">Current</th>
              <th className="text-right font-normal px-3 py-2">P&L</th>
              <th className="text-right font-normal px-3 py-2">Bars</th>
              <th className="text-right font-normal px-3 py-2">SL</th>
              <th className="text-right font-normal px-4 py-2">TP</th>
            </tr>
          </thead>
          <tbody>
            {(["RSI", "IB", "MOM"] as const).map((s) => {
              const pos = positions[s] as Position | null | undefined;
              const isFlat = !pos;
              return (
                <tr
                  key={s}
                  className="transition-colors"
                  style={{
                    color: isFlat ? "var(--text-muted)" : "var(--text-primary)",
                    borderTop: "1px solid var(--border)",
                  }}
                >
                  <td className="px-4 py-2.5 font-medium">{s}</td>
                  <td className="px-3 py-2.5">
                    {pos ? (
                      <span className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${pos.side === "Buy" ? "bg-emerald-500" : "bg-red-500"}`} />
                        <span style={{ color: pos.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {pos.side === "Buy" ? "LONG" : "SHORT"}
                        </span>
                      </span>
                    ) : (
                      <span>FLAT</span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono">
                    {pos ? pos.entry_price.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono">
                    {pos ? pos.current_price.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: pos ? clr(pos.pnl) : undefined }}>
                    {pos ? fmt(pos.pnl) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono">
                    {pos ? pos.bars_held : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono">
                    {pos ? pos.sl.toFixed(2) : "—"}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono">
                    {pos ? pos.tp.toFixed(2) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Signal + Trade Logs ─────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        {/* Signal Log */}
        <div className="panel rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>
            Signals
          </div>
          <div className="max-h-48 overflow-y-auto">
            {signals.length === 0 && (
              <div className="px-4 py-3 text-xs" style={{ color: "var(--text-muted)" }}>No signals yet</div>
            )}
            {[...signals].reverse().slice(0, 20).map((sig: Signal, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 px-4 py-1.5 text-[11px]"
                style={{ borderBottom: "1px solid var(--border)" }}
              >
                <span className="font-mono w-14 shrink-0" style={{ color: "var(--text-muted)" }}>
                  {sig.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span
                  className="w-10 font-medium shrink-0"
                  style={{ color: sig.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}
                >
                  {sig.strategy}
                </span>
                <span style={{ color: sig.side === "Buy" ? "var(--accent-green)" : "var(--accent-red)" }}>
                  {sig.side === "Buy" ? "LONG" : "SHORT"}
                </span>
                <span className="truncate" style={{ color: "var(--text-muted)" }}>{sig.reason}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Trade Log */}
        <div className="panel rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 text-[11px] font-medium" style={{ color: "var(--text-secondary)", borderBottom: "1px solid var(--border)" }}>
            Trades
          </div>
          <div className="max-h-48 overflow-y-auto">
            {trades.length === 0 && (
              <div className="px-4 py-3 text-xs" style={{ color: "var(--text-muted)" }}>No trades yet</div>
            )}
            {[...trades].reverse().slice(0, 20).map((t: Trade, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 px-4 py-1.5 text-[11px]"
                style={{ borderBottom: "1px solid var(--border)" }}
              >
                <span className="font-mono w-14 shrink-0" style={{ color: "var(--text-muted)" }}>
                  {t.timestamp?.split("T")[1]?.slice(0, 8) ?? ""}
                </span>
                <span className="w-10 font-medium shrink-0" style={{ color: "var(--text-secondary)" }}>
                  {t.strategy}
                </span>
                <span
                  className="w-10 shrink-0"
                  style={{ color: t.action === "entry" ? "var(--accent-blue)" : "#f97316" }}
                >
                  {t.action === "entry" ? "ENTRY" : "EXIT"}
                </span>
                {t.action === "entry" ? (
                  <>
                    <span className="font-mono" style={{ color: "var(--text-primary)" }}>
                      @{t.fill_price?.toFixed(2)}
                    </span>
                    <span style={{ color: "var(--text-muted)" }}>
                      slip {t.slippage?.toFixed(2)}
                    </span>
                  </>
                ) : (
                  <>
                    <span style={{ color: "var(--text-muted)" }}>{t.exit_reason}</span>
                    <span className="font-mono" style={{ color: clr(t.pnl ?? 0) }}>
                      {fmt(t.pnl ?? 0)}
                    </span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Helper Components ─────────────────────────────────────── */

function StatCard({
  label,
  value,
  color,
  large,
}: {
  label: string;
  value: string;
  color: string;
  large?: boolean;
}) {
  return (
    <div className="panel rounded-lg px-4 py-3">
      <div className="text-[11px] mb-1" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div
        className={`font-mono font-semibold ${large ? "text-xl" : "text-base"}`}
        style={{ color }}
      >
        {value}
      </div>
    </div>
  );
}

function LimitBar({
  label,
  value,
  limit,
}: {
  label: string;
  value: number;
  limit: number;
}) {
  const pct = value <= 0 ? Math.min((Math.abs(value) / Math.abs(limit)) * 100, 100) : 0;
  const barColor =
    value >= 0
      ? "var(--accent-green)"
      : pct > 80
        ? "var(--accent-red)"
        : pct > 50
          ? "var(--accent-yellow)"
          : "var(--accent-green)";

  return (
    <div className="panel rounded-lg px-4 py-2.5 flex items-center gap-3">
      <span className="text-[11px] w-14 shrink-0" style={{ color: "var(--text-muted)" }}>
        {label}
      </span>
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${value >= 0 ? 0 : pct}%`, background: barColor }}
        />
      </div>
      <span className="text-[11px] font-mono w-24 text-right" style={{ color: value >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
        {value >= 0 ? "+" : ""}${value.toFixed(0)} / ${limit.toFixed(0)}
      </span>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Dashboard.tsx
git commit -m "feat: redesigned Dashboard with equity curve, P&L calendar, compact layout"
```

---

### Task 5: Setup Page Redesign

**Files:**
- Rewrite: `dashboard/src/pages/Setup.tsx`

- [ ] **Step 1: Rewrite Setup.tsx**

Replace `dashboard/src/pages/Setup.tsx` with:

```tsx
import { useState, useEffect, type FormEvent } from "react";
import { api } from "../api/client";
import type { Account, AccountCreate, AuthTestResult } from "../types";

const EMPTY_FORM: AccountCreate = {
  name: "",
  username: "",
  password: "",
  cid: 0,
  sec: "",
  device_id: "",
  is_master: false,
  sizing_mode: "mirror",
  account_size: 150000,
};

export function Setup() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [form, setForm] = useState<AccountCreate>({ ...EMPTY_FORM });
  const [editing, setEditing] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<Record<string, AuthTestResult>>({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");

  const reload = () => {
    api.getAccounts().then(setAccounts).catch(console.error);
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  };

  useEffect(() => { reload(); }, []);

  const openAdd = () => {
    setEditing(null);
    setForm({ ...EMPTY_FORM });
    setError("");
    setShowForm(true);
  };

  const openEdit = (acct: Account) => {
    setEditing(acct.name);
    setForm({
      name: acct.name,
      username: acct.username,
      password: "",
      cid: acct.cid,
      sec: "",
      device_id: acct.device_id,
      is_master: acct.is_master,
      sizing_mode: acct.sizing_mode,
      account_size: acct.account_size,
    });
    setError("");
    setShowForm(true);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (editing) {
        await api.updateAccount(editing, form);
      } else {
        await api.createAccount(form);
      }
      setShowForm(false);
      setForm({ ...EMPTY_FORM });
      setEditing(null);
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete account "${name}"?`)) return;
    await api.deleteAccount(name);
    reload();
  };

  const handleTestAuth = async (name: string) => {
    setTestingName(name);
    setTestResults((prev) => ({ ...prev, [name]: undefined as unknown as AuthTestResult }));
    try {
      const result = await api.testAuth(name);
      setTestResults((prev) => ({ ...prev, [name]: result }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [name]: { success: false, error: err instanceof Error ? err.message : "Failed" },
      }));
    } finally {
      setTestingName(null);
    }
  };

  const toggleEnv = async () => {
    const next = environment === "demo" ? "live" : "demo";
    await api.setEnvironment(next);
    setEnvironment(next);
  };

  const SIZING_MODES = ["mirror", "fixed", "scaled"] as const;

  const inputCls =
    "w-full rounded px-3 py-2 text-sm outline-none placeholder:text-zinc-600 transition-colors focus:border-blue-500/50" +
    " " +
    "bg-[var(--bg-surface)] border border-[rgba(255,255,255,0.06)] text-[var(--text-primary)]";

  return (
    <div className="p-5 max-w-[1000px] mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>Accounts</h2>
        <div className="flex items-center gap-3">
          <button
            onClick={toggleEnv}
            className="px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider"
            style={{
              background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)",
              color: environment === "demo" ? "var(--accent-yellow)" : "var(--accent-red)",
              border: `1px solid ${environment === "demo" ? "rgba(245,158,11,0.2)" : "rgba(239,68,68,0.2)"}`,
            }}
          >
            {environment}
          </button>
          <button
            onClick={openAdd}
            className="px-3 py-1.5 text-xs font-medium rounded text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10"
          >
            + Add Account
          </button>
        </div>
      </div>

      {/* Account List */}
      <div className="space-y-2">
        {accounts.length === 0 && (
          <div className="panel rounded-lg p-8 text-center">
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              No accounts configured. Add one to get started.
            </p>
          </div>
        )}
        {accounts.map((acct) => {
          const testResult = testResults[acct.name];
          const isTesting = testingName === acct.name;
          const testPending = isTesting || (testResult === undefined && testingName === acct.name);

          return (
            <div
              key={acct.name}
              className="group/card panel rounded-lg p-4 relative transition-colors hover:border-[rgba(255,255,255,0.1)]"
              style={acct.is_master ? { borderColor: "rgba(245,158,11,0.2)" } : undefined}
            >
              {/* Hover delete X */}
              <button
                onClick={() => handleDelete(acct.name)}
                className="absolute top-3 right-3 w-5 h-5 rounded flex items-center justify-center text-zinc-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover/card:opacity-100 transition-opacity"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>

              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {acct.is_master && (
                    <span className="text-amber-400 text-xs">&#9733;</span>
                  )}
                  <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    {acct.name}
                  </span>
                  {acct.is_master && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.1)", color: "var(--accent-yellow)" }}>
                      MASTER
                    </span>
                  )}
                </div>
                <span className="text-[10px] uppercase font-medium" style={{ color: "var(--text-muted)" }}>
                  {acct.sizing_mode}
                </span>
              </div>

              <div className="flex items-center gap-4 text-xs mb-3" style={{ color: "var(--text-secondary)" }}>
                <span>{acct.username}</span>
                <span className="font-mono">${acct.account_size.toLocaleString()}</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleTestAuth(acct.name)}
                  disabled={isTesting}
                  className="text-[11px] px-2.5 py-1 rounded transition-colors disabled:opacity-50"
                  style={{ background: "rgba(255,255,255,0.04)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
                >
                  {isTesting ? "Testing..." : "Test Connection"}
                </button>
                <button
                  onClick={() => openEdit(acct)}
                  className="text-[11px] px-2.5 py-1 rounded transition-colors"
                  style={{ background: "rgba(255,255,255,0.04)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
                >
                  Edit
                </button>

                {/* Test result indicator */}
                {testResult && !isTesting && (
                  <span className="flex items-center gap-1 text-[11px] ml-1">
                    {testResult.success ? (
                      <span className="flex items-center gap-1" style={{ color: "var(--accent-green)" }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                        Connected (ID: {testResult.account_id})
                      </span>
                    ) : (
                      <span className="flex items-center gap-1" style={{ color: "var(--accent-red)" }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                        {testResult.error}
                      </span>
                    )}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Slide-out Form Modal ─────────────────────────────── */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setShowForm(false)}>
          <div className="absolute inset-0 bg-black/50" />
          <div
            className="relative w-[400px] h-full overflow-y-auto p-6 space-y-4"
            style={{ background: "var(--bg-panel)", borderLeft: "1px solid var(--border)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                {editing ? `Edit: ${editing}` : "New Account"}
              </h3>
              <button onClick={() => setShowForm(false)} className="text-zinc-500 hover:text-zinc-300">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {error && (
              <div className="text-xs p-2 rounded" style={{ background: "rgba(239,68,68,0.1)", color: "var(--accent-red)", border: "1px solid rgba(239,68,68,0.2)" }}>
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <label className="block">
                <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>Account Name</span>
                <input className={inputCls} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} disabled={!!editing} required />
              </label>
              <label className="block">
                <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>Username</span>
                <input className={inputCls} value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required />
              </label>
              <label className="block">
                <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>
                  {editing ? "Password (leave blank to keep)" : "Password"}
                </span>
                <input className={inputCls} type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required={!editing} />
              </label>
              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>CID</span>
                  <input className={inputCls} type="number" value={form.cid || ""} onChange={(e) => setForm({ ...form, cid: parseInt(e.target.value) || 0 })} />
                </label>
                <label className="block">
                  <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>API Secret</span>
                  <input className={inputCls} type="password" value={form.sec} onChange={(e) => setForm({ ...form, sec: e.target.value })} />
                </label>
              </div>
              <label className="block">
                <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>Device ID</span>
                <input className={inputCls} placeholder="Auto-generated if empty" value={form.device_id} onChange={(e) => setForm({ ...form, device_id: e.target.value })} />
              </label>

              {/* Sizing mode segmented control */}
              <div>
                <span className="text-[11px] mb-1.5 block" style={{ color: "var(--text-muted)" }}>Sizing Mode</span>
                <div className="flex rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                  {SIZING_MODES.map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setForm({ ...form, sizing_mode: mode })}
                      className="flex-1 py-1.5 text-xs font-medium capitalize transition-colors"
                      style={{
                        background: form.sizing_mode === mode ? "rgba(255,255,255,0.08)" : "transparent",
                        color: form.sizing_mode === mode ? "var(--text-primary)" : "var(--text-muted)",
                        borderRight: mode !== "scaled" ? "1px solid var(--border)" : undefined,
                      }}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              <label className="block">
                <span className="text-[11px] mb-1 block" style={{ color: "var(--text-muted)" }}>Account Size</span>
                <input className={inputCls} type="number" value={form.account_size} onChange={(e) => setForm({ ...form, account_size: parseFloat(e.target.value) || 0 })} />
              </label>

              <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: "var(--text-secondary)" }}>
                <input
                  type="checkbox"
                  checked={form.is_master}
                  onChange={(e) => setForm({ ...form, is_master: e.target.checked })}
                  className="rounded"
                />
                Master account
              </label>

              <button
                type="submit"
                className="w-full py-2 text-sm font-medium rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
              >
                {editing ? "Update Account" : "Add Account"}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Setup.tsx
git commit -m "feat: redesigned Setup page with modal form, segmented control, hover delete"
```

---

### Task 6: Settings Page Redesign

**Files:**
- Rewrite: `dashboard/src/pages/Settings.tsx`

- [ ] **Step 1: Rewrite Settings.tsx**

Replace `dashboard/src/pages/Settings.tsx` with:

```tsx
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/berjourlian/berjquant/tradovate-executor/dashboard
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /Users/berjourlian/berjquant/tradovate-executor
git add dashboard/src/pages/Settings.tsx
git commit -m "feat: redesigned Settings page with table format and prominent session info"
```

---

## Summary

| Task | What changes | Key features |
|------|-------------|--------------|
| 1 | Theme + deps + WS hook | recharts, CSS vars (#0a0a0f), equity history tracking |
| 2 | Layout.tsx | 52px icon sidebar, 180px on hover, removed StatusBadge |
| 3 | EquityCurve + PnLCalendar | recharts area chart, monthly heatmap grid with hover |
| 4 | Dashboard.tsx | Stats bar, limit bars, positions table, compact logs, flatten dialog |
| 5 | Setup.tsx | Slide-out modal form, segmented control, hover-X delete, gold master |
| 6 | Settings.tsx | Table format, prominent session times, compact contract info |

**Total: 6 tasks. All API integration preserved. Backend untouched.**
