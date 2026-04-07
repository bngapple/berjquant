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
  monthly_loss_limit?: number;
  min_contracts?: number;
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
  monthly_loss_limit?: number;
  min_contracts?: number;
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
  monthly_limit: number;
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
  bars: Bar[];
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
  status: "active" | "eval_passed" | "breached";
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

export interface Bar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi?: number | null;
  atr?: number | null;
  ema?: number | null;
}

export interface StrategyConfig {
  contracts: number;
  stop_loss_pts: number;
  take_profit_pts: number;
  max_hold_bars: number;
  period?: number;
  oversold?: number;
  overbought?: number;
  ib_start?: string;
  ib_end?: string;
  ib_range_lookback?: number;
  ib_range_pct_low?: number;
  ib_range_pct_high?: number;
  atr_period?: number;
  ema_period?: number;
  vol_sma_period?: number;
}

export interface SessionSettings {
  session_start: string;
  no_new_entries_after: string;
  flatten_time: string;
  monthly_loss_limit: number;
  daily_loss_limit?: number | null;
  timezone: string;
}

export interface NTConnection {
  name: string;
  host: string;
  port: number;
}

export interface RuntimeConfig {
  environment: string;
  symbol: string;
  nt_enabled: boolean;
  nt_only: boolean;
  nt_accounts: NTConnection[];
  session: SessionSettings;
  rsi: StrategyConfig;
  ib: StrategyConfig;
  mom: StrategyConfig;
}

export interface NTOnlySetupUpdate {
  account_name: string;
  host: string;
  port: number;
  symbol: string;
  contracts: number;
  monthly_loss_limit: number;
}
