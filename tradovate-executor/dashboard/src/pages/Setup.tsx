import { useState, useEffect, type FormEvent } from "react";
import { api } from "../api/client";
import type { Account, AccountCreate, AuthTestResult } from "../types";

const EMPTY: AccountCreate = {
  name: "", username: "", password: "", cid: 0, sec: "", device_id: "",
  is_master: false, sizing_mode: "mirror", account_size: 150000,
  starting_balance: 150000, profit_target: 9000, max_drawdown: -4500, account_type: "eval",
};

export function Setup() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [form, setForm] = useState<AccountCreate>({ ...EMPTY });
  const [editing, setEditing] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<Record<string, AuthTestResult>>({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const reload = () => { api.getAccounts().then(setAccounts); api.getEnvironment().then(d => setEnvironment(d.environment)); };
  useEffect(() => { reload(); }, []);

  const openAdd = () => { setEditing(null); setForm({ ...EMPTY }); setError(""); setShowForm(true); };
  const openEdit = (a: Account) => {
    setEditing(a.name);
    setForm({
      name: a.name, username: a.username, password: "", cid: a.cid, sec: "", device_id: a.device_id,
      is_master: a.is_master, sizing_mode: a.sizing_mode, account_size: a.account_size,
      starting_balance: a.starting_balance ?? 150000, profit_target: a.profit_target ?? 9000,
      max_drawdown: a.max_drawdown ?? -4500, account_type: a.account_type ?? "eval",
    });
    setError(""); setShowForm(true);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); setError(""); setSaving(true);
    try {
      if (editing) await api.updateAccount(editing, form);
      else await api.createAccount(form);
      setShowForm(false); setForm({ ...EMPTY }); setEditing(null); reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (name: string) => { await api.deleteAccount(name); setDeleting(null); reload(); };

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

      <div className="panel rounded overflow-hidden">
        <table className="w-full text-[11px]">
          <thead><tr style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>
            <th className="text-left font-normal px-4 py-2">Name</th>
            <th className="text-left font-normal px-3 py-2">Username</th>
            <th className="text-left font-normal px-3 py-2">Env</th>
            <th className="text-left font-normal px-3 py-2">Role</th>
            <th className="text-left font-normal px-3 py-2">Type</th>
            <th className="text-left font-normal px-3 py-2">Status</th>
            <th className="text-right font-normal px-4 py-2">Actions</th>
          </tr></thead>
          <tbody>
            {accounts.length === 0 && <tr><td colSpan={7} className="px-4 py-6 text-center" style={{ color: "var(--text-dim)" }}>No accounts configured</td></tr>}
            {accounts.map(a => {
              const tr = testResults[a.name];
              const testing = testingName === a.name;
              return (
                <tr key={a.name} className="group/row transition-colors" style={{ borderTop: "1px solid var(--border)", borderLeft: a.is_master ? "3px solid var(--accent)" : "3px solid transparent" }}>
                  <td className="px-4 py-2 font-medium" style={{ color: "var(--text)" }}>{a.name}</td>
                  <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{a.username}</td>
                  <td className="px-3 py-2">
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold uppercase"
                      style={{ background: environment === "demo" ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)", color: environment === "demo" ? "var(--amber)" : "var(--red)" }}>
                      {environment}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold"
                      style={a.is_master ? { background: "rgba(0,212,170,0.15)", color: "#00b894" } : { color: "var(--text-muted)" }}>
                      {a.is_master ? "Master" : "Copy"}
                    </span>
                  </td>
                  <td className="px-3 py-2 capitalize text-[10px]" style={{ color: "var(--text-muted)" }}>{a.account_type ?? "eval"}</td>
                  <td className="px-3 py-2">
                    {tr && !testing ? (
                      <span className="flex items-center gap-1" style={{ color: tr.success ? "var(--accent)" : "var(--red)" }}>
                        {tr.success ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg> : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>}
                        <span className="text-[10px]">{tr.success ? "OK" : tr.error?.slice(0, 30) ?? "Fail"}</span>
                      </span>
                    ) : testing ? <span className="text-[10px]" style={{ color: "var(--text-dim)" }}>Testing...</span> : <span style={{ color: "var(--text-dim)" }}>&mdash;</span>}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button onClick={() => handleTest(a.name)} disabled={testing} className="text-[10px] px-2 py-0.5 rounded disabled:opacity-30" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Test</button>
                      <button onClick={() => openEdit(a)} className="text-[10px] px-2 py-0.5 rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Edit</button>
                      <button onClick={() => setDeleting(a.name)} className="w-5 h-5 rounded flex items-center justify-center opacity-0 group-hover/row:opacity-100 transition-opacity duration-150 text-zinc-600 hover:text-red-400">
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

              <div className="pt-2 pb-1 text-[10px] uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>Account Settings</div>

              <Seg label="Environment" options={["demo", "live"]} value={environment} onChange={v => { api.setEnvironment(v); setEnvironment(v); }} />
              <Seg label="Role" options={["master", "copy"]} value={form.is_master ? "master" : "copy"} onChange={v => setForm({ ...form, is_master: v === "master" })} />
              <Seg label="Account Type" options={["eval", "funded"]} value={form.account_type ?? "eval"} onChange={v => setForm({ ...form, account_type: v })} />
              <Seg label="Sizing Mode" options={["mirror", "fixed", "scaled"]} value={form.sizing_mode ?? "mirror"} onChange={v => setForm({ ...form, sizing_mode: v })} />
              {form.sizing_mode === "scaled" && <Field label="Account Size"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.account_size} onChange={e => setForm({ ...form, account_size: parseFloat(e.target.value) || 0 })} /></Field>}

              <div className="pt-2 pb-1 text-[10px] uppercase tracking-wider" style={{ color: "var(--text-dim)" }}>Risk Parameters</div>

              <div className="grid grid-cols-3 gap-3">
                <Field label="Starting Balance"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.starting_balance} onChange={e => setForm({ ...form, starting_balance: parseFloat(e.target.value) || 0 })} /></Field>
                <Field label="Profit Target"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.profit_target} onChange={e => setForm({ ...form, profit_target: parseFloat(e.target.value) || 0 })} /></Field>
                <Field label="Max Drawdown"><input className={inp} style={{ background: "var(--elevated)", border: "1px solid var(--border)", color: "var(--text)" }} type="number" value={form.max_drawdown} onChange={e => setForm({ ...form, max_drawdown: parseFloat(e.target.value) || 0 })} /></Field>
              </div>

              <button type="submit" disabled={saving} className="w-full py-2 text-sm font-medium rounded text-white transition-colors disabled:opacity-60" style={{ background: "var(--accent)" }}>
                {saving ? "Validating credentials..." : editing ? "Update" : "Add Account"}
              </button>
              {saving && (
                <div className="h-0.5 rounded overflow-hidden mt-1" style={{ background: "rgba(255,255,255,0.04)" }}>
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
