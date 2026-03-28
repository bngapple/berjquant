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
  const [error, setError] = useState("");
  const [testResults, setTestResults] = useState<
    Record<string, AuthTestResult>
  >({});
  const [testingName, setTestingName] = useState<string | null>(null);
  const [environment, setEnvironment] = useState("demo");

  const reload = () => {
    api.getAccounts().then(setAccounts).catch(console.error);
    api.getEnvironment().then((d) => setEnvironment(d.environment));
  };

  useEffect(() => {
    reload();
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (editing) {
        await api.updateAccount(editing, form);
      } else {
        await api.createAccount(form);
      }
      setForm({ ...EMPTY_FORM });
      setEditing(null);
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    }
  };

  const handleEdit = (acct: Account) => {
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
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete account "${name}"?`)) return;
    await api.deleteAccount(name);
    reload();
  };

  const handleTestAuth = async (name: string) => {
    setTestingName(name);
    try {
      const result = await api.testAuth(name);
      setTestResults((prev) => ({ ...prev, [name]: result }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [name]: {
          success: false,
          error: err instanceof Error ? err.message : "Failed",
        },
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

  const inputCls =
    "w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500";

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Account Setup</h2>
        <button
          onClick={toggleEnv}
          className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider ${
            environment === "demo"
              ? "bg-yellow-900/50 text-yellow-300 border border-yellow-700"
              : "bg-red-900/50 text-red-300 border border-red-700"
          }`}
        >
          {environment}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
          <h3 className="font-semibold mb-4">
            {editing ? `Edit: ${editing}` : "Add Account"}
          </h3>
          {error && (
            <div className="mb-3 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded p-2">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              className={inputCls}
              placeholder="Account name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              disabled={!!editing}
              required
            />
            <input
              className={inputCls}
              placeholder="Tradovate username"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required
            />
            <input
              className={inputCls}
              type="password"
              placeholder={editing ? "New password (leave blank to keep)" : "Password"}
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required={!editing}
            />
            <div className="grid grid-cols-2 gap-3">
              <input
                className={inputCls}
                type="number"
                placeholder="CID"
                value={form.cid || ""}
                onChange={(e) =>
                  setForm({ ...form, cid: parseInt(e.target.value) || 0 })
                }
              />
              <input
                className={inputCls}
                type="password"
                placeholder="API Secret"
                value={form.sec}
                onChange={(e) => setForm({ ...form, sec: e.target.value })}
              />
            </div>
            <input
              className={inputCls}
              placeholder="Device ID (auto-generated if empty)"
              value={form.device_id}
              onChange={(e) => setForm({ ...form, device_id: e.target.value })}
            />
            <div className="grid grid-cols-2 gap-3">
              <select
                className={inputCls}
                value={form.sizing_mode}
                onChange={(e) =>
                  setForm({ ...form, sizing_mode: e.target.value })
                }
              >
                <option value="mirror">Mirror</option>
                <option value="fixed">Fixed</option>
                <option value="scaled">Scaled</option>
              </select>
              <input
                className={inputCls}
                type="number"
                placeholder="Account size"
                value={form.account_size}
                onChange={(e) =>
                  setForm({
                    ...form,
                    account_size: parseFloat(e.target.value) || 0,
                  })
                }
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={form.is_master}
                onChange={(e) =>
                  setForm({ ...form, is_master: e.target.checked })
                }
                className="rounded bg-gray-800 border-gray-600"
              />
              Master account
            </label>
            <div className="flex gap-2">
              <button
                type="submit"
                className="flex-1 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium py-2 rounded transition-colors"
              >
                {editing ? "Update" : "Add Account"}
              </button>
              {editing && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(null);
                    setForm({ ...EMPTY_FORM });
                  }}
                  className="px-4 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Account List */}
        <div className="space-y-3">
          {accounts.length === 0 && (
            <p className="text-gray-500 text-sm">No accounts configured.</p>
          )}
          {accounts.map((acct) => {
            const testResult = testResults[acct.name];
            const isTesting = testingName === acct.name;
            return (
              <div
                key={acct.name}
                className="bg-gray-900 rounded-lg border border-gray-800 p-4"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="font-semibold">{acct.name}</span>
                    {acct.is_master && (
                      <span className="ml-2 text-xs bg-blue-900/50 text-blue-300 px-1.5 py-0.5 rounded">
                        MASTER
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500 uppercase">
                    {acct.sizing_mode}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mb-3">
                  <span>{acct.username}</span>
                  <span className="mx-2">·</span>
                  <span className="font-mono">
                    ${acct.account_size.toLocaleString()}
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleTestAuth(acct.name)}
                    disabled={isTesting}
                    className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors disabled:opacity-50"
                  >
                    {isTesting ? "Testing..." : "Test Connection"}
                  </button>
                  <button
                    onClick={() => handleEdit(acct)}
                    className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(acct.name)}
                    className="text-xs bg-gray-800 hover:bg-red-900/50 text-gray-400 hover:text-red-300 px-3 py-1.5 rounded transition-colors"
                  >
                    Delete
                  </button>
                </div>
                {testResult && (
                  <div
                    className={`mt-2 text-xs p-2 rounded ${
                      testResult.success
                        ? "bg-green-900/20 text-green-400 border border-green-800"
                        : "bg-red-900/20 text-red-400 border border-red-800"
                    }`}
                  >
                    {testResult.success
                      ? `Connected — Account ID: ${testResult.account_id}, User ID: ${testResult.user_id}`
                      : `Failed: ${testResult.error}`}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
