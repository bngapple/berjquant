import type {
  Account,
  AccountCreate,
  AuthTestResult,
  EngineStatus,
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

  startEngine: () =>
    request<{ status: string }>("/engine/start", { method: "POST" }),

  stopEngine: () =>
    request<{ status: string }>("/engine/stop", { method: "POST" }),

  flattenAll: () =>
    request<{ status: string }>("/engine/flatten", { method: "POST" }),

  getStatus: () => request<EngineStatus>("/engine/status"),

  getEnvironment: () => request<{ environment: string }>("/environment"),

  setEnvironment: (env: string) =>
    request<{ environment: string }>("/environment", {
      method: "PUT",
      body: JSON.stringify({ environment: env }),
    }),
};
