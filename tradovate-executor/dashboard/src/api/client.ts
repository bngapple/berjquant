import type {
  Account,
  AccountCreate,
  AuthTestResult,
  EngineStatus,
  HistoryStats,
  DailyPnL,
  AccountStatus,
  FleetAlert,
} from "../types";

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(error.detail || resp.statusText);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

export const api = {
  // Accounts
  getAccounts: () => request<Account[]>("/accounts"),

  createAccount: (data: AccountCreate) =>
    request<Account>("/accounts", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  updateAccount: (name: string, data: Partial<AccountCreate>) =>
    request<Account>(`/accounts/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  deleteAccount: (name: string) =>
    request<void>(`/accounts/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  testAuth: (name: string) =>
    request<AuthTestResult>("/auth/test", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  // Engine
  startEngine: () =>
    request<{ status: string }>("/engine/start", { method: "POST" }),

  stopEngine: () =>
    request<{ status: string }>("/engine/stop", { method: "POST" }),

  flattenAll: () =>
    request<{ status: string }>("/engine/flatten", { method: "POST" }),

  getStatus: () => request<EngineStatus>("/engine/status"),

  // Environment
  getEnvironment: () => request<{ environment: string }>("/environment"),

  setEnvironment: (env: string) =>
    request<{ environment: string }>("/environment", {
      method: "PUT",
      body: JSON.stringify({ environment: env }),
    }),

  // History
  getHistoryStats: () => request<HistoryStats>("/history/stats"),

  getHistoryDaily: () => request<DailyPnL>("/history/daily"),

  getHistoryEquity: () =>
    request<{ date: string; value: number }[]>("/history/equity"),

  getHistoryTrades: (limit = 50) =>
    request<Record<string, unknown>[]>(`/history/trades?limit=${limit}`),

  // Account status
  getAccountStatuses: () => request<AccountStatus[]>("/accounts/status"),

  getAccountStatus: (name: string) =>
    request<AccountStatus>(`/accounts/status/${encodeURIComponent(name)}`),

  getFleetAlerts: () => request<FleetAlert[]>("/accounts/alerts"),

  // Health
  getHealth: () => request<Record<string, unknown>>("/health"),
};
