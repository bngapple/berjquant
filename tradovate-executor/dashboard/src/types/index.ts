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
  starting_balance: number;
  profit_target: number;
  max_drawdown: number;
  account_type: "eval" | "funded";
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
  starting_balance?: number;
  profit_target?: number;
  max_drawdown?: number;
  account_type?: string;
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
  error?: string;
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

export interface AccountStatus {
  name: string;
  balance: number;
  starting_balance: number;
  pnl_total: number;
  drawdown_current: number;
  drawdown_max_allowed: number;
  drawdown_remaining: number;
  drawdown_pct_used: number;
  profit_target: number;
  profit_target_progress: number;
  daily_pnl: number;
  trades_today: number;
  status: "active" | "eval_passed" | "breached" | "daily_limit_hit";
  is_master: boolean;
  account_type: "eval" | "funded";
  environment: string;
}

export interface HistoryStats {
  total_pnl: number;
  win_rate: number | null;
  profit_factor: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  total_trades: number;
  winners: number;
  losers: number;
}

export interface DailyPnL {
  [date: string]: { pnl: number; trades: number; wins: number; losses: number };
}

export interface FleetAlert {
  account: string;
  type: "danger" | "warning" | "success";
  message: string;
}
