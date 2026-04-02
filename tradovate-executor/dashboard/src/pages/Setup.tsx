import { useState, useEffect, useMemo, type FormEvent } from "react";
import { api } from "../api/client";
import { useLayoutData } from "../hooks/useLayoutData";
import type { Account, AccountCreate, AuthTestResult, AccountStatus, FleetAlert, RuntimeConfig, EngineStatus, Trade, NTOnlySetupUpdate } from "../types";

// LucidFlex exact tier presets from official plan
const LUCID_TIERS = [
  { label: "25K",  size: 25000,  profit_target: 1250,  max_drawdown: -1000,  monthly_loss_limit: -1000,  contracts: 1 },
  { label: "50K",  size: 50000,  profit_target: 3000,  max_drawdown: -2000,  monthly_loss_limit: -2000,  contracts: 1 },
  { label: "100K", size: 100000, profit_target: 6000,  max_drawdown: -3000,  monthly_loss_limit: -3000,  contracts: 2 },
  { label: "150K", size: 150000, profit_target: 9000,  max_drawdown: -4500,  monthly_loss_limit: -4500,  contracts: 3 },
] as const;

const EMPTY: AccountCreate = {
  name: "", username: "", password: "", cid: 0, sec: "", device_id: "",
  is_master: false, sizing_mode: "scaled", account_size: 150000,
  starting_balance: 150000, profit_target: 9000, max_drawdown: -4500,
  account_type: "eval", monthly_loss_limit: -4500, min_contracts: 3,
};

function tierForRuntime(runtime: RuntimeConfig | null) {
  if (!runtime) return LUCID_TIERS[3];
  return (
    LUCID_TIERS.find(
      (tier) =>
        tier.monthly_loss_limit === runtime.session.monthly_loss_limit &&
        tier.contracts === runtime.rsi.contracts,
    ) ?? LUCID_TIERS[3]
  );
}

export function Setup() {
  const { status, trades } = useLayoutData();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [accountStatuses, setAccountStatuses] = useState<AccountStatus[]>([]);
  const [alerts, setAlerts] = useState<FleetAlert[]>([]);
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null);
  const [form, setForm] = useState<AccountCreate>({ ...EMPTY });
  const [editing, setEditing] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<Record<string, AuthTestResult>>({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [expandedAccount, setExpandedAccount] = useState<string | null>(null);

  const running = status?.running ?? false;

  const reload = () => {
    api.getAccounts().then(setAccounts);
    api.getEnvironment().then(d => setEnvironment(d.environment));
    api.getRuntimeConfig().then(setRuntime).catch(() => {});
    api.getAccountStatuses().then(setAccountStatuses).catch(() => {});
    api.getFleetAlerts().then(setAlerts).catch(() => {});
  };
  useEffect(() => { reload(); }, []);

  useEffect(() => {
    const t = setInterval(() => {
      api.getAccountStatuses().then(setAccountStatuses).catch(() => {});
    }, 30000);
    return () => clearInterval(t);
  }, []);

  const rows = useMemo(() => accounts.map(acct => {
    const connected = status?.connected_accounts?.find(a => a.name === acct.name)?.connected ?? false;
    const acctStatus = accountStatuses.find(s => s.name === acct.name);
    return { acct, connected, acctStatus };
  }), [accounts, status, accountStatuses]);

  const fleet = useMemo(() => {
    const connected = rows.filter(r => r.connected).length;
    const dayPnl = rows.reduce((s, r) => s + (r.acctStatus?.daily_pnl ?? 0), 0);
    const totalPnl = rows.reduce((s, r) => s + (r.acctStatus?.pnl_total ?? 0), 0);
    return { connected, total: rows.length, dayPnl, totalPnl };
  }, [rows]);

  const copyFeed = useMemo(() => [...trades].reverse().slice(0, 8), [trades]);

  const openAdd = () => { setEditing(null); setForm({ ...EMPTY }); setError(""); setShowForm(true); };
  const openEdit = (a: Account) => {
    setEditing(a.name);
    setForm({
      name: a.name, username: a.username, password: "", cid: a.cid, sec: "", device_id: a.device_id,
      is_master: a.is_master, sizing_mode: "scaled", account_size: a.account_size,
      starting_balance: a.starting_balance ?? 150000, profit_target: a.profit_target ?? 9000,
      max_drawdown: a.max_drawdown ?? -4500, account_type: a.account_type ?? "eval",
      monthly_loss_limit: a.monthly_loss_limit ?? -4500, min_contracts: a.min_contracts ?? 1,
    });
    setError(""); setShowForm(true);
  };

  const applyTier = (tier: typeof LUCID_TIERS[number]) => {
    setForm(f => ({
      ...f,
      account_size: tier.size,
      starting_balance: tier.size,
      profit_target: tier.profit_target,
      max_drawdown: tier.max_drawdown,
      monthly_loss_limit: tier.monthly_loss_limit,
      min_contracts: tier.contracts,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); setError(""); setSaving(true);
    try {
      if (editing) await api.updateAccount(editing, form);
      else await api.createAccount(form);
      setShowForm(false); setForm({ ...EMPTY }); setEditing(null); reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally { setSaving(false); }
  };

  const handleDelete = async (name: string) => { await api.deleteAccount(name); setDeleting(null); reload(); };
  const handleTest = async (name: string) => {
    setTestingName(name);
    try { const r = await api.testAuth(name); setTestResults(p => ({ ...p, [name]: r })); }
    catch (err) { setTestResults(p => ({ ...p, [name]: { success: false, error: err instanceof Error ? err.message : "Failed" } })); }
    finally { setTestingName(null); }
  };
  const toggleEnv = async () => { const n = environment === "demo" ? "live" : "demo"; await api.setEnvironment(n); setEnvironment(n); };

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";
  const inp = "w-full rounded px-3 py-2 text-sm outline-none placeholder:text-zinc-600 transition-colors focus:border-[rgba(0,212,170,0.3)]";

  if (runtime?.nt_only) {
    return (
      <NTOnlySetup
        runtime={runtime}
        status={status}
        trades={trades}
        alerts={alerts}
        environment={environment}
        onToggleEnv={toggleEnv}
        onReload={reload}
      />
    );
  }

  return (
    <div className="p-5 space-y-4 max-w-[1200px] mx-auto">
      {/* ── Alerts ── */}
      {alerts.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {alerts.map((a, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px]"
              style={{
                background: a.type === "danger" ? "rgba(240,74,74,0.1)" : a.type === "warning" ? "rgba(245,158,11,0.1)" : "rgba(0,212,170,0.1)",
                color: a.type === "danger" ? "var(--red)" : a.type === "warning" ? "var(--amber)" : "var(--accent)",
                border: `1px solid ${a.type === "danger" ? "rgba(240,74,74,0.15)" : a.type === "warning" ? "rgba(245,158,11,0.15)" : "rgba(0,212,170,0.15)"}`,
              }}>
              <span>{a.type === "success" ? "✓" : "⚠"}</span>
              <span>{a.account}: {a.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold tracking-wide" style={{ color: "var(--text)" }}>Accounts</span>
          {runtime?.nt_only && (
            <span
              className="px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-widest"
              style={{
                background: "rgba(0,212,170,0.1)",
                color: "var(--accent)",
                border: "1px solid rgba(0,212,170,0.18)",
              }}
            >
              NT-Only
            </span>
          )}
          <div className="flex items-center gap-3 text-[11px]" style={{ color: "var(--text-muted)" }}>
            <span>{fleet.connected}/{fleet.total} connected</span>
            <span style={{ color: "var(--text-dim)" }}>·</span>
            <span className="font-mono tabular" style={{ color: clr(fleet.dayPnl) }}>Day {fmt(fleet.dayPnl)}</span>
            <span style={{ color: "var(--text-dim)" }}>·</span>
            <span className="font-mono tabular" style={{ color: clr(fleet.totalPnl) }}>Total {fmt(fleet.totalPnl)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={toggleEnv} className="px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider"
            style={{ background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(240,74,74,0.1)", color: environment === "demo" ? "var(--amber)" : "var(--red)", border: `1px solid ${environment === "demo" ? "rgba(245,158,11,0.2)" : "rgba(240,74,74,0.2)"}` }}>
            {environment}
          </button>
          <button onClick={openAdd} className="px-3 py-1.5 text-[11px] font-medium rounded" style={{ color: "var(--accent)", border: "1px solid rgba(0,212,170,0.3)", background: "rgba(0,212,170,0.06)" }}>
            + Add Account
          </button>
        </div>
      </div>

      {/* ── Accounts Table ── */}
      <div className="panel rounded overflow-hidden">
        <table className="w-full text-[11px]">
          <thead>
            <tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
              <th className="text-left font-normal px-4 py-2 w-6"></th>
              <th className="text-left font-normal px-3 py-2">Account</th>
              <th className="text-left font-normal px-3 py-2">Role</th>
              <th className="text-left font-normal px-3 py-2">Type</th>
              <th className="text-left font-normal px-3 py-2">Size</th>
              <th className="text-right font-normal px-3 py-2">Day P&amp;L</th>
              <th className="text-right font-normal px-3 py-2">Total P&amp;L</th>
              <th className="text-right font-normal px-3 py-2">DD Used</th>
              <th className="text-right font-normal px-4 py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={9} className="px-4 py-8 text-center text-xs" style={{ color: "var(--text-dim)" }}>
                  {runtime?.nt_only
                    ? "NT-only mode active — the engine can run through NinjaTrader without Tradovate accounts."
                    : "No accounts configured — add one to get started"}
                </td>
              </tr>
            )}
            {rows.map(({ acct: a, connected, acctStatus }) => {
              const tr = testResults[a.name];
              const testing = testingName === a.name;
              const isExpanded = expandedAccount === a.name;
              return (
                <>
                  <tr key={a.name}
                    className="cursor-pointer transition-colors hover:bg-white/[0.015]"
                    onClick={() => setExpandedAccount(isExpanded ? null : a.name)}
                    style={{
                      borderTop: "1px solid var(--border)",
                      borderLeft: a.is_master ? "2px solid var(--accent)" : "2px solid transparent",
                      background: a.is_master ? "rgba(0,212,170,0.02)" : "transparent",
                    }}>
                    {/* Connection dot */}
                    <td className="px-4 py-2.5">
                      <span className={`w-1.5 h-1.5 rounded-full inline-block ${connected ? "bg-emerald-400 pulse-dot" : running ? "bg-red-500" : "bg-zinc-700"}`} />
                    </td>
                    <td className="px-3 py-2.5 font-medium" style={{ color: "var(--text)" }}>{a.name}</td>
                    <td className="px-3 py-2.5">
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold"
                        style={a.is_master ? { background: "rgba(0,212,170,0.15)", color: "#00b894" } : { color: "var(--text-muted)", border: "1px solid var(--border)" }}>
                        {a.is_master ? "Master" : "Copy"}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 capitalize text-[10px]" style={{ color: "var(--text-muted)" }}>{a.account_type ?? "eval"}</td>
                    <td className="px-3 py-2.5 text-[10px] font-mono" style={{ color: "var(--text-muted)" }}>
                      ${(a.account_size / 1000).toFixed(0)}K
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono tabular" style={{ color: clr(acctStatus?.daily_pnl ?? 0) }}>
                      {acctStatus ? fmt(acctStatus.daily_pnl) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono tabular" style={{ color: clr(acctStatus?.pnl_total ?? 0) }}>
                      {acctStatus ? fmt(acctStatus.pnl_total) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      {acctStatus ? (
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-14 h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                            <div className="h-full rounded-full transition-all duration-500" style={{
                              width: `${Math.min(acctStatus.drawdown_pct_used, 100)}%`,
                              background: acctStatus.drawdown_pct_used > 75 ? "var(--red)" : acctStatus.drawdown_pct_used > 50 ? "var(--amber)" : "var(--accent)",
                              boxShadow: `0 0 4px ${acctStatus.drawdown_pct_used > 75 ? "rgba(240,74,74,0.5)" : "rgba(0,212,170,0.4)"}`,
                            }} />
                          </div>
                          <span className="text-[10px] font-mono tabular w-7 text-right" style={{ color: acctStatus.drawdown_pct_used > 75 ? "var(--red)" : "var(--text-muted)" }}>
                            {acctStatus.drawdown_pct_used.toFixed(0)}%
                          </span>
                        </div>
                      ) : <span style={{ color: "var(--text-dim)" }}>—</span>}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="flex items-center justify-end gap-1.5" onClick={e => e.stopPropagation()}>
                        {tr && !testing ? (
                          <span style={{ color: tr.success ? "var(--accent)" : "var(--red)" }} className="text-[10px]">{tr.success ? "✓ OK" : "✗"}</span>
                        ) : testing ? <span className="text-[10px]" style={{ color: "var(--text-dim)" }}>…</span> : null}
                        <button onClick={() => handleTest(a.name)} disabled={testing} className="text-[10px] px-2 py-0.5 rounded disabled:opacity-30" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Test</button>
                        <button onClick={() => openEdit(a)} className="text-[10px] px-2 py-0.5 rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Edit</button>
                        <button onClick={() => setDeleting(a.name)} className="w-5 h-5 rounded flex items-center justify-center text-zinc-600 hover:text-red-400 transition-colors">
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                  {/* Expanded detail */}
                  {isExpanded && acctStatus && (
                    <tr key={`${a.name}-detail`}>
                      <td colSpan={9} style={{ background: "var(--elevated)", borderTop: "1px solid var(--border)" }}>
                        <div className="px-6 py-3 flex flex-wrap gap-8 text-[11px]">
                          <Stat label="Balance" value={`$${acctStatus.balance.toLocaleString("en-US", { minimumFractionDigits: 0 })}`} />
                          <Stat label="Total P&L" value={fmt(acctStatus.pnl_total)} color={clr(acctStatus.pnl_total)} />
                          <Stat label="Drawdown" value={`$${Math.abs(acctStatus.drawdown_current).toFixed(0)} / $${Math.abs(acctStatus.drawdown_max_allowed).toFixed(0)}`} color={acctStatus.drawdown_pct_used > 75 ? "var(--red)" : "var(--text)"} />
                          <Stat label="DD Remaining" value={`$${Math.abs(acctStatus.drawdown_remaining).toFixed(0)}`} />
                          {acctStatus.profit_target > 0 && <Stat label="Target Progress" value={`$${acctStatus.pnl_total.toFixed(0)} / $${acctStatus.profit_target.toFixed(0)}`} color={acctStatus.profit_target_progress >= 100 ? "var(--amber)" : "var(--accent)"} />}
                          <Stat label="Today" value={`${acctStatus.trades_today} trades`} />
                          <div>
                            <div className="mb-0.5" style={{ color: "var(--text-muted)" }}>Status</div>
                            <span className="px-2 py-0.5 rounded text-[9px] font-semibold" style={{
                              background: acctStatus.status === "active" ? "rgba(0,212,170,0.15)" : acctStatus.status === "eval_passed" ? "rgba(245,158,11,0.15)" : "rgba(240,74,74,0.15)",
                              color: acctStatus.status === "active" ? "#00b894" : acctStatus.status === "eval_passed" ? "var(--amber)" : "var(--red)",
                            }}>
                              {acctStatus.status === "active" ? "Active" : acctStatus.status === "eval_passed" ? "Eval Passed" : "Breached"}
                            </span>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Copy Activity ── */}
      {copyFeed.length > 0 && (
        <div className="panel rounded overflow-hidden">
          <div className="px-3 py-1.5 text-[9px] font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)", letterSpacing: "0.12em" }}>Copy Activity</div>
          <div className="max-h-32 overflow-y-auto">
            {copyFeed.map((entry, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{entry.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span style={{ color: "var(--text)" }}>
                  {entry.action === "entry" ? `${entry.side} ${entry.contracts} MNQ @${entry.fill_price?.toFixed(2) ?? "—"}` : `Exit ${entry.strategy} ${entry.exit_reason ?? ""}`}
                </span>
                {entry.pnl !== undefined && entry.action === "exit" && (
                  <span className="font-mono tabular ml-auto" style={{ color: clr(entry.pnl) }}>{fmt(entry.pnl)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Delete confirm ── */}
      {deleting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={() => setDeleting(null)} style={{ background: "rgba(0,0,0,0.7)" }}>
          <div className="panel rounded p-5 w-80" onClick={e => e.stopPropagation()}>
            <p className="text-sm mb-4" style={{ color: "var(--text)" }}>Delete <span className="font-medium">{deleting}</span>?</p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setDeleting(null)} className="px-3 py-1.5 text-xs rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Cancel</button>
              <button onClick={() => handleDelete(deleting)} className="px-3 py-1.5 text-xs rounded bg-red-600 hover:bg-red-500 text-white">Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Add / Edit slide-in ── */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setShowForm(false)}>
          <div className="absolute inset-0 bg-black/50" />
          <div className="relative w-[420px] h-full overflow-y-auto p-6 space-y-4" style={{ background: "var(--panel)", borderLeft: "1px solid var(--border)" }} onClick={e => e.stopPropagation()}>

            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>{editing ? `Edit: ${editing}` : "New Account"}</span>
              <button onClick={() => setShowForm(false)} className="text-zinc-500 hover:text-zinc-300">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>

            {error && <div className="text-xs p-2 rounded" style={{ background: "rgba(240,74,74,0.1)", color: "var(--red)", border: "1px solid rgba(240,74,74,0.15)" }}>{error}</div>}

            <form onSubmit={handleSubmit} className="space-y-3">
              {/* Credentials */}
              <F label="Account Name"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} disabled={!!editing} required /></F>
              <F label="Username"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} required /></F>
              <F label={editing ? "Password (blank to keep)" : "Password"}><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="password" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} required={!editing} /></F>
              <div className="grid grid-cols-2 gap-3">
                <F label="CID"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.cid || ""} onChange={e => setForm({ ...form, cid: parseInt(e.target.value) || 0 })} /></F>
                <F label="API Secret"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="password" value={form.sec} onChange={e => setForm({ ...form, sec: e.target.value })} /></F>
              </div>
              <F label="Device ID"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} placeholder="Auto if empty" value={form.device_id} onChange={e => setForm({ ...form, device_id: e.target.value })} /></F>

              {/* Account Settings */}
              <div className="pt-1 pb-0.5 text-[9px] uppercase tracking-widest font-medium" style={{ color: "var(--text-muted)", letterSpacing: "0.12em" }}>Account Settings</div>
              <Seg label="Role" options={["master", "copy"]} value={form.is_master ? "master" : "copy"} onChange={v => setForm({ ...form, is_master: v === "master" })} />
              <Seg label="Account Type" options={["eval", "funded"]} value={form.account_type ?? "eval"} onChange={v => setForm({ ...form, account_type: v })} />

              {/* LucidFlex Account Size — draggable snap slider */}
              {(() => {
                const tierIdx = LUCID_TIERS.findIndex(t => t.size === form.account_size);
                const idx = tierIdx >= 0 ? tierIdx : 3;
                const activeTier = LUCID_TIERS[idx];
                const fillPct = (idx / 3) * 100;
                return (
                  <div className="space-y-3">
                    <div className="text-[9px] uppercase tracking-widest font-medium" style={{ color: "var(--text-muted)", letterSpacing: "0.12em" }}>Account Size</div>
                    <div className="space-y-2 px-1">
                      <input
                        type="range"
                        min={0} max={3} step={1}
                        value={idx}
                        onChange={e => applyTier(LUCID_TIERS[parseInt(e.target.value)])}
                        className="lucid-slider"
                        style={{
                          background: `linear-gradient(to right, var(--accent) ${fillPct}%, rgba(255,255,255,0.08) ${fillPct}%)`,
                        }}
                      />
                      <div className="flex justify-between">
                        {LUCID_TIERS.map((tier, i) => (
                          <button key={tier.label} type="button" onClick={() => applyTier(tier)}
                            className="text-[11px] font-semibold transition-colors"
                            style={{ color: idx === i ? "var(--accent)" : "var(--text-dim)" }}>
                            {tier.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    {/* Read-only values */}
                    <div className="rounded px-3 py-2.5 grid grid-cols-3 gap-2 text-center" style={{ background: "rgba(0,212,170,0.04)", border: "1px solid rgba(0,212,170,0.1)" }}>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Profit Target</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--accent)" }}>${activeTier.profit_target.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Max Drawdown</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--red)" }}>-${Math.abs(activeTier.max_drawdown).toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Contracts</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--text)" }}>{activeTier.contracts}/strat</div>
                      </div>
                    </div>
                  </div>
                );
              })()}

              <button type="submit" disabled={saving} className="w-full py-2 text-sm font-semibold rounded text-black transition-all disabled:opacity-60" style={{ background: "var(--accent)" }}>
                {saving ? "Validating credentials..." : editing ? "Update Account" : "Add Account"}
              </button>
              {saving && (
                <div className="h-0.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                  <div className="h-full w-1/3 loading-bar" style={{ background: "var(--accent)" }} />
                </div>
              )}
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

type NTOnlySetupProps = {
  runtime: RuntimeConfig;
  status: EngineStatus | null;
  trades: Trade[];
  alerts: FleetAlert[];
  environment: string;
  onToggleEnv: () => Promise<void>;
  onReload: () => void;
};

type NTOnlyForm = NTOnlySetupUpdate & {
  account_size: number;
};

function NTOnlySetup({
  runtime,
  status,
  trades,
  alerts,
  environment,
  onToggleEnv,
  onReload,
}: NTOnlySetupProps) {
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const current = runtime.nt_accounts[0] ?? null;
  const activeTier = tierForRuntime(runtime);
  const bridgeName = current ? `NinjaTrader (${current.host}:${current.port})` : "";
  const connected = status?.connected_accounts?.find((a) => a.name === bridgeName)?.connected ?? false;
  const copyFeed = useMemo(() => [...trades].reverse().slice(0, 8), [trades]);

  const [form, setForm] = useState<NTOnlyForm>({
    account_name: current?.name ?? "",
    host: current?.host ?? "",
    port: current?.port ?? 6000,
    symbol: runtime.symbol,
    contracts: runtime.rsi.contracts,
    monthly_loss_limit: runtime.session.monthly_loss_limit,
    account_size: activeTier.size,
  });

  useEffect(() => {
    if (!showForm) {
      const tier = tierForRuntime(runtime);
      setForm({
        account_name: current?.name ?? "",
        host: current?.host ?? "",
        port: current?.port ?? 6000,
        symbol: runtime.symbol,
        contracts: runtime.rsi.contracts,
        monthly_loss_limit: runtime.session.monthly_loss_limit,
        account_size: tier.size,
      });
    }
  }, [runtime, current, showForm]);

  const applyTierToForm = (tier: typeof LUCID_TIERS[number]) => {
    setForm((prev) => ({
      ...prev,
      account_size: tier.size,
      contracts: tier.contracts,
      monthly_loss_limit: tier.monthly_loss_limit,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setSaving(true);
    try {
      await api.saveNTOnlySetup({
        account_name: form.account_name,
        host: form.host,
        port: form.port,
        symbol: form.symbol,
        contracts: form.contracts,
        monthly_loss_limit: form.monthly_loss_limit,
      });
      setShowForm(false);
      onReload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const fmt = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(2)}`;
  const clr = (v: number) => v >= 0 ? "var(--accent)" : "var(--red)";
  const inp = "w-full rounded px-3 py-2 text-sm outline-none placeholder:text-zinc-600 transition-colors focus:border-[rgba(0,212,170,0.3)]";

  return (
    <div className="p-5 space-y-4 max-w-[1200px] mx-auto">
      {alerts.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {alerts.map((a, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px]"
              style={{
                background: a.type === "danger" ? "rgba(240,74,74,0.1)" : a.type === "warning" ? "rgba(245,158,11,0.1)" : "rgba(0,212,170,0.1)",
                color: a.type === "danger" ? "var(--red)" : a.type === "warning" ? "var(--amber)" : "var(--accent)",
                border: `1px solid ${a.type === "danger" ? "rgba(240,74,74,0.15)" : a.type === "warning" ? "rgba(245,158,11,0.15)" : "rgba(0,212,170,0.15)"}`,
              }}>
              <span>{a.type === "success" ? "✓" : "⚠"}</span>
              <span>{a.account}: {a.message}</span>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold tracking-wide" style={{ color: "var(--text)" }}>Accounts</span>
          <span
            className="px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-widest"
            style={{
              background: "rgba(0,212,170,0.1)",
              color: "var(--accent)",
              border: "1px solid rgba(0,212,170,0.18)",
            }}
          >
            NT-Only
          </span>
          <div className="flex items-center gap-3 text-[11px]" style={{ color: "var(--text-muted)" }}>
            <span>{connected ? "Bridge connected" : "Bridge configured"}</span>
            <span style={{ color: "var(--text-dim)" }}>·</span>
            <span>{runtime.symbol}</span>
            <span style={{ color: "var(--text-dim)" }}>·</span>
            <span>{activeTier.label}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onToggleEnv} className="px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider"
            style={{ background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(240,74,74,0.1)", color: environment === "demo" ? "var(--amber)" : "var(--red)", border: `1px solid ${environment === "demo" ? "rgba(245,158,11,0.2)" : "rgba(240,74,74,0.2)"}` }}>
            {environment}
          </button>
          <button onClick={() => setShowForm(true)} className="px-3 py-1.5 text-[11px] font-medium rounded" style={{ color: "var(--accent)", border: "1px solid rgba(0,212,170,0.3)", background: "rgba(0,212,170,0.06)" }}>
            {current ? "Edit Account" : "+ Add Account"}
          </button>
        </div>
      </div>

      <div className="panel rounded overflow-hidden">
        <div className="grid grid-cols-[1.2fr_1fr_1fr_1fr_1fr] text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="px-4 py-2" style={{ color: "var(--text-dim)" }}>Bridge Account</div>
          <div className="px-3 py-2" style={{ color: "var(--text-dim)" }}>Contract</div>
          <div className="px-3 py-2" style={{ color: "var(--text-dim)" }}>Tier</div>
          <div className="px-3 py-2" style={{ color: "var(--text-dim)" }}>Risk</div>
          <div className="px-4 py-2 text-right" style={{ color: "var(--text-dim)" }}>Status</div>
        </div>
        {current ? (
          <div className="grid grid-cols-[1.2fr_1fr_1fr_1fr_1fr] items-center text-[11px]" style={{ borderLeft: "2px solid var(--accent)", background: "rgba(0,212,170,0.02)" }}>
            <div className="px-4 py-3">
              <div className="font-medium" style={{ color: "var(--text)" }}>{current.name}</div>
              <div className="font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>{current.host}:{current.port}</div>
            </div>
            <div className="px-3 py-3 font-mono" style={{ color: "var(--text)" }}>{runtime.symbol}</div>
            <div className="px-3 py-3">
              <div className="font-mono" style={{ color: "var(--text)" }}>{activeTier.label}</div>
              <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>{runtime.rsi.contracts} ct / strat</div>
            </div>
            <div className="px-3 py-3">
              <div className="font-mono" style={{ color: "var(--red)" }}>-${Math.abs(runtime.session.monthly_loss_limit).toFixed(0)}</div>
              <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>monthly limit</div>
            </div>
            <div className="px-4 py-3 flex items-center justify-end gap-2">
              <span className={`w-1.5 h-1.5 rounded-full inline-block ${connected ? "bg-emerald-400 pulse-dot" : "bg-zinc-700"}`} />
              <span style={{ color: connected ? "var(--accent)" : "var(--text-muted)" }}>{connected ? "Live" : "Idle"}</span>
            </div>
          </div>
        ) : (
          <div className="px-4 py-8 text-center text-xs" style={{ color: "var(--text-dim)" }}>
            No NT bridge account configured yet.
          </div>
        )}
      </div>

      <div className="panel rounded overflow-hidden">
        <div className="px-3 py-1.5 text-[9px] font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)", letterSpacing: "0.12em" }}>
          Setup Notes
        </div>
        <div className="px-4 py-3 grid grid-cols-2 gap-4 text-[11px]">
          <div>
            <div style={{ color: "var(--text-muted)" }}>NinjaTrader account name</div>
            <div style={{ color: "var(--text)" }}>Must match the Accounts tab exactly.</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>VM host / port</div>
            <div style={{ color: "var(--text)" }}>Use the Windows `ipconfig` IPv4 and the PythonBridge port.</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>Chart</div>
            <div style={{ color: "var(--text)" }}>Attach `PythonBridge` to a live 15-minute chart for the same contract.</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>Tier preset</div>
            <div style={{ color: "var(--text)" }}>Sets contracts per strategy and monthly drawdown to match the account tier.</div>
          </div>
        </div>
      </div>

      {copyFeed.length > 0 && (
        <div className="panel rounded overflow-hidden">
          <div className="px-3 py-1.5 text-[9px] font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)", letterSpacing: "0.12em" }}>Live Activity</div>
          <div className="max-h-32 overflow-y-auto">
            {copyFeed.map((entry, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-1 text-[11px]" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="font-mono w-14 shrink-0 tabular" style={{ color: "var(--text-dim)" }}>{entry.timestamp?.split("T")[1]?.slice(0, 8)}</span>
                <span style={{ color: "var(--text)" }}>
                  {entry.action === "entry" ? `${entry.side} ${entry.contracts} ${runtime.symbol.replace("U6", "").replace("M6", "")} @${entry.fill_price?.toFixed(2) ?? "—"}` : `Exit ${entry.strategy} ${entry.exit_reason ?? ""}`}
                </span>
                {entry.pnl !== undefined && entry.action === "exit" && (
                  <span className="font-mono tabular ml-auto" style={{ color: clr(entry.pnl) }}>{fmt(entry.pnl)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {showForm && (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setShowForm(false)}>
          <div className="absolute inset-0 bg-black/50" />
          <div className="relative w-[420px] h-full overflow-y-auto p-6 space-y-4" style={{ background: "var(--panel)", borderLeft: "1px solid var(--border)" }} onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>{current ? `Edit: ${current.name}` : "New NT Account"}</span>
              <button onClick={() => setShowForm(false)} className="text-zinc-500 hover:text-zinc-300">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>

            {error && <div className="text-xs p-2 rounded" style={{ background: "rgba(240,74,74,0.1)", color: "var(--red)", border: "1px solid rgba(240,74,74,0.15)" }}>{error}</div>}

            <form onSubmit={handleSubmit} className="space-y-3">
              <F label="NinjaTrader Account Name">
                <input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.account_name} onChange={e => setForm({ ...form, account_name: e.target.value })} required />
              </F>
              <div className="grid grid-cols-2 gap-3">
                <F label="VM IPv4 / Host">
                  <input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.host} onChange={e => setForm({ ...form, host: e.target.value })} required />
                </F>
                <F label="TCP Port">
                  <input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.port} onChange={e => setForm({ ...form, port: parseInt(e.target.value, 10) || 6000 })} required />
                </F>
              </div>
              <F label="Contract Symbol">
                <input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} value={form.symbol} onChange={e => setForm({ ...form, symbol: e.target.value.toUpperCase() })} required />
              </F>

              <div className="pt-1 pb-0.5 text-[9px] uppercase tracking-widest font-medium" style={{ color: "var(--text-muted)", letterSpacing: "0.12em" }}>Account Tier</div>
              {(() => {
                const tierIdx = LUCID_TIERS.findIndex(t => t.size === form.account_size);
                const idx = tierIdx >= 0 ? tierIdx : 0;
                const selectedTier = LUCID_TIERS[idx];
                const fillPct = (idx / 3) * 100;
                return (
                  <div className="space-y-3">
                    <div className="space-y-2 px-1">
                      <input
                        type="range"
                        min={0}
                        max={3}
                        step={1}
                        value={idx}
                        onChange={e => applyTierToForm(LUCID_TIERS[parseInt(e.target.value, 10)])}
                        className="lucid-slider"
                        style={{
                          background: `linear-gradient(to right, var(--accent) ${fillPct}%, rgba(255,255,255,0.08) ${fillPct}%)`,
                        }}
                      />
                      <div className="flex justify-between">
                        {LUCID_TIERS.map((tier, i) => (
                          <button key={tier.label} type="button" onClick={() => applyTierToForm(tier)}
                            className="text-[11px] font-semibold transition-colors"
                            style={{ color: idx === i ? "var(--accent)" : "var(--text-dim)" }}>
                            {tier.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="rounded px-3 py-2.5 grid grid-cols-3 gap-2 text-center" style={{ background: "rgba(0,212,170,0.04)", border: "1px solid rgba(0,212,170,0.1)" }}>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Profit Target</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--accent)" }}>${selectedTier.profit_target.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Max Drawdown</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--red)" }}>-${Math.abs(selectedTier.max_drawdown).toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Contracts</div>
                        <div className="font-mono text-[12px] font-semibold" style={{ color: "var(--text)" }}>{selectedTier.contracts}/strat</div>
                      </div>
                    </div>
                  </div>
                );
              })()}

              <button type="submit" disabled={saving} className="w-full py-2 text-sm font-semibold rounded text-black transition-all disabled:opacity-60" style={{ background: "var(--accent)" }}>
                {saving ? "Saving bridge setup..." : current ? "Update Account" : "Save Account"}
              </button>
              {saving && (
                <div className="h-0.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                  <div className="h-full w-1/3 loading-bar" style={{ background: "var(--accent)" }} />
                </div>
              )}
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] mb-0.5" style={{ color: "var(--text-muted)" }}>{label}</div>
      <div className="font-mono font-semibold tabular text-[12px]" style={{ color: color ?? "var(--text)" }}>{value}</div>
    </div>
  );
}

function F({ label, children }: { label: string; children: React.ReactNode }) {
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
            style={{ background: value === o ? "var(--accent)" : "transparent", color: value === o ? "#07100e" : "var(--text-dim)", borderRight: o !== options[options.length - 1] ? "1px solid var(--border)" : undefined }}>
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}
