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
