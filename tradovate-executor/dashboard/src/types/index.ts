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

export interface WSMessage {
  type: "status" | "position" | "pnl" | "signal" | "fill" | "exit";
  data: unknown;
  timestamp: string;
}
