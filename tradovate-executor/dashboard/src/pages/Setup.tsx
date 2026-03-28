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
