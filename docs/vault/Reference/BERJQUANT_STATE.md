---
date: 2026-03-26
type: reference
tags: [berjquant, system-docs, architecture]
source: /Users/berjourlian/berjquant/ANTHONY/BERJQUANT_STATE.md
---

# BerjQuant — Comprehensive System Overview

**Date:** March 26, 2026
**Repository:** berjquant
**Primary Asset:** MNQ (Micro E-mini Nasdaq 100 Futures)
**Primary Strategy:** HTF Swing v3 (Aggressive, 3 contracts)

---

## Table of Contents

1. [What Is BerjQuant](#1-what-is-berjquant)
2. [Architecture & Engine](#2-architecture--engine)
3. [HTF Swing v3 — The Production Strategy](#3-htf-swing-v3--the-production-strategy)
4. [Performance Results](#4-performance-results)
5. [Monte Carlo Validation](#5-monte-carlo-validation)
6. [Trade Audit](#6-trade-audit)
7. [PineScript (TradingView) Implementation](#7-pinescript-tradingview-implementation)
8. [ANTHONY Folder CSV Guide](#8-anthony-folder-csv-guide)
9. [Other Strategies & Development History](#9-other-strategies--development-history)
10. [Engine Internals](#10-engine-internals)
11. [Signal Library](#11-signal-library)
12. [Validation Framework](#12-validation-framework)
13. [File Map](#13-file-map)

---

## 1. What Is BerjQuant

BerjQuant is a **Monte Carlo Quantitative (MCQ) Trading Engine** — a full-stack intraday strategy research and execution platform for NQ/MNQ futures, purpose-built for prop trading accounts (Topstep, Apex, LucidFlex).

The system spans 5 phases:

```
Raw Data (MotiveWave CSV / Databento / YFinance)
  → Data Pipeline (ingest, clean, resample to Parquet)
    → Vectorized Backtesting (lookahead-free, bar N+1 fills)
      → Monte Carlo Stress Testing (10K sims, bootstrap + jitter + gaps)
        → Walk-Forward Validation (IS/OOS split, overfit detection)
          → Paper/Live Trading (real-time signal engine + risk enforcement)
```

**Key design principles:**
- **Lookahead-free**: Signals fire on bar N, fills execute at bar N+1 open. No future data leaks.
- **Risk-first**: The RiskManager runs 7 mandatory pre-trade checks on every single entry. It cannot be bypassed.
- **Prop-firm native**: Enforces daily loss limits, trailing max drawdown, kill switches, consistency rules, and EOD flatten.
- **No data leakage in strategy selection**: Y1 data selects the strategy. Y2 and Blind periods are never touched during optimization.

---

## 2. Architecture & Engine

### Core Pipeline

| Component | File | Purpose |
|-----------|------|---------|
| Data Structures | `engine/utils.py` | ContractSpec, PropFirmRules, AccountState, Position, Trade, BacktestConfig, PerformanceMetrics |
| Data Pipeline | `engine/data_pipeline.py` | MotiveWave CSV ingestion, cleaning, session tagging, bar resampling |
| Data Fetcher | `engine/data_fetcher.py` | YFinance integration for NQ=F proxy data |
| Backtester | `engine/backtester.py` | VectorizedBacktester with Strategy protocol, SlippageModel, pending-entry mechanism |
| Fast Backtester | `engine/fast_backtester.py` | Current-bar fills variant for HFT/scalping strategies |
| Tiered Backtester | `engine/tiered_backtester.py` | Multi-tier priority system (champion preempts grinders) |
| Adaptive | `engine/adaptive.py` | ATR-adaptive exits, regime detection (trending/ranging), volatility regimes |
| Risk Manager | `engine/risk_manager.py` | 7-step pre-trade validation, kill switch at 80% daily limit, event blackouts |
| Metrics | `engine/metrics.py` | Performance analytics, TradeLogger (SQLite), Sharpe/PF/DD calculations |
| Parallel Runner | `engine/parallel_runner.py` | Multi-process batch execution, composite scoring, leaderboards |

### MNQ Contract Spec

| Parameter | Value |
|-----------|-------|
| Tick size | 0.25 points |
| Tick value | $0.50 |
| Point value | $2.00 |
| 1 point | 4 ticks |

### Prop Firm Profiles

| Profile | Daily Loss Limit | Max Drawdown | DD Type | Max MNQ | Consistency |
|---------|-----------------|--------------|---------|---------|-------------|
| Topstep 50K | $1,000 | $2,000 | Trailing | 15 | None |
| Topstep 150K | $3,000 | $4,500 | Trailing | 45 | None |
| Apex 50K | $1,100 | $2,500 | Trailing | 20 | None |
| LucidFlex 50K | $1,000 | $2,000 | Trailing | 15 | Max 30% single day |

Commission: $0.62/contract/side. Exchange fee: $0.27/contract/side.

### Trading Sessions (US/Eastern)

| Window | Time |
|--------|------|
| Pre-market | 08:00 – 09:30 |
| Core (RTH) | 09:30 – 16:00 |
| Post-close | 16:00 – 17:00 |
| Flat by | 17:00 ET |
| Event blackout | ±15 min around NFP, CPI, FOMC, etc. |

---

## 3. HTF Swing v3 — The Production Strategy

HTF Swing v3 is a **multi-signal intraday system** that runs 3 independent sub-strategies simultaneously on **15-minute bars**. Each sub-strategy generates its own entries/exits; all 3 trade in parallel with separate position tracking.

### Configuration (Aggressive)

| Parameter | Value |
|-----------|-------|
| Contracts per strategy | 3 MNQ |
| Max simultaneous positions | 3 (one per sub-strategy) |
| Max contracts at any time | 9 MNQ |
| Daily loss limit | -$3,000 |
| Monthly loss limit (MLL) | -$4,500 |
| Evaluation target | $9,000 |
| EOD flatten | 16:45 ET |
| Slippage | 2 ticks per side |

### Sub-Strategy 1: RSI Extremes

Oscillator-based mean reversion on rapid RSI readings.

| Parameter | Value |
|-----------|-------|
| RSI Period | 7 |
| Oversold (long trigger) | RSI < 30 |
| Overbought (short trigger) | RSI > 70 |
| Stop Loss | 15 points (60 ticks) = $30/contract |
| Take Profit | 100 points (400 ticks) = $200/contract |
| Max Hold | 5 bars (~75 minutes) |
| Typical monthly volume | 65–115 trades |

**Logic:** When RSI(7) drops below 30, enter long. When RSI(7) rises above 70, enter short. Exit on SL, TP, max hold, or EOD flatten.

### Sub-Strategy 2: IB Breakout

Session-structure breakout based on the Initial Balance range.

| Parameter | Value |
|-----------|-------|
| IB Window | 09:30 – 09:45 ET (first 2 bars on 15m chart) |
| Post-IB Trading Window | 10:00 – 15:30 ET |
| Stop Loss | 20 points (80 ticks) = $40/contract |
| Take Profit | 120 points (480 ticks) = $240/contract |
| Max Hold | 10 bars (~150 minutes) |
| Max trades/day | 1 (once either direction fires) |
| IB Range Filter | 25th–75th percentile of last 50 IB ranges |
| Typical monthly volume | 4–18 trades |

**Logic:** Capture the high and low of the first 15 minutes (IB). After the IB period ends, if price breaks above IB high → long; below IB low → short. Filter out abnormally tight or wide IB ranges using a percentile gate.

### Sub-Strategy 3: Momentum Bar

Volume-confirmed directional momentum aligned with trend.

| Parameter | Value |
|-----------|-------|
| ATR Multiplier | 1.0× (bar range must exceed ATR(14)) |
| Volume Multiplier | 1.0× (volume must exceed 20-bar SMA) |
| Trend Filter | EMA(21) — bar direction must agree |
| Stop Loss | 15 points (60 ticks) = $30/contract |
| Take Profit | 100 points (400 ticks) = $200/contract |
| Max Hold | 5 bars (~75 minutes) |
| Typical monthly volume | 38–78 trades |

**Logic:** Detect bars where range > ATR(14) AND volume > SMA(volume, 20). If the bar is bullish (close > open) and price > EMA(21), enter long. If bearish and price < EMA(21), enter short.

### Risk/Reward Profile

All three sub-strategies share asymmetric R:R — small stop losses with large take profit targets:

| Sub-Strategy | SL (points) | TP (points) | R:R Ratio | Win Rate Needed to Break Even |
|-------------|------------|------------|-----------|------------------------------|
| RSI | 15 | 100 | 1:6.67 | ~13% |
| IB | 20 | 120 | 1:6.00 | ~14% |
| MOM | 15 | 100 | 1:6.67 | ~13% |

The system wins only ~25-27% of the time, but winners are 6-7× larger than losers.

### Cost Structure Per Trade (3 contracts)

| Component | Cost |
|-----------|------|
| Commission (RT) | $0.62 × 2 sides × 3 contracts = $3.72 |
| Exchange fee (RT) | $0.27 × 2 sides × 3 contracts = $1.62 |
| Slippage (2 ticks each side) | 4 ticks × $0.50 × 3 = $6.00 |
| **Total cost per trade** | **$11.34** |

---

## 4. Performance Results

### Data Periods

| Period | Date Range | Purpose | Months |
|--------|------------|---------|--------|
| Blind (early) | Dec 2017 – Jun 2019 | Never seen during development | 19 |
| Blind (extended) | 2020 – 2024 | Never seen during development | ~48 |
| Y1 (In-Sample) | Mar 2024 – May 2025 | Strategy selection & optimization | 15 |
| Y2 (Out-of-Sample) | Jun 2025 – Mar 2026 | Forward validation | 13 |

### Aggressive (3 Contracts) — Primary Configuration

#### Y2 Out-of-Sample (Jun 2025 – Mar 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 1,810 |
| Win Rate | 26.96% |
| Net P&L | +$92,822.10 |
| Monthly Average | +$7,140.16 |
| Best Month | +$12,208.44 |
| Worst Month | +$3,008.52 |
| **Months Profitable** | **13/13 (100%)** |
| Best Day | +$3,216.60 |
| Worst Day | -$1,043.40 |
| Max Drawdown | -$1,761.30 |
| Avg Bars Held | 1.78 |
| Trades Per Day | 7.07 |
| Consistency Score | 3.47 |
| Y1 Sharpe | 6.939 |

#### Blind Test (Dec 2017 – Mar 2024, unseen data)

| Metric | Value |
|--------|-------|
| Total Trades | 4,129 |
| Win Rate | 24.70% |
| Net P&L | +$90,764.64 |
| Monthly Average | +$3,241.59 |
| Best Month | +$10,228.26 |
| Worst Month | -$1,404.36 |
| Months Profitable | 20/28 (71.4%) |
| Max Drawdown | -$4,361.70 |
| MLL Breached | No |

#### Full 8-Year Cumulative (Dec 2017 – Mar 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 14,426 |
| Cumulative Net P&L | +$313,670.16 |
| Total Trading Months | ~100 |

### Recent Monthly Breakdown (Y2 OOS, Aggressive)

| Month | Trades | RSI | IB | MOM | Win Rate | Net P&L |
|-------|--------|-----|----|----|----------|---------|
| Jun 2025 | 154 | 82 | 8 | 64 | 24.7% | +$3,578.64 |
| Jul 2025 | 145 | 67 | 8 | 70 | 29.7% | +$5,669.70 |
| Aug 2025 | 150 | 82 | 10 | 58 | 22.7% | +$2,739.00 |
| Sep 2025 | 148 | 80 | 13 | 55 | 27.0% | +$6,948.18 |
| Oct 2025 | 203 | 113 | 12 | 78 | 22.7% | +$4,602.48 |
| Nov 2025 | 157 | 99 | 4 | 54 | 26.1% | +$10,675.62 |
| Dec 2025 | 141 | 66 | 14 | 61 | 29.1% | +$7,569.06 |
| Jan 2026 | 163 | 88 | 12 | 63 | 25.8% | +$9,094.08 |
| Feb 2026 | 157 | 92 | 8 | 57 | 26.8% | +$11,505.12 |
| Mar 2026* | 85 | 41 | 6 | 38 | 25.9% | +$6,072.60 |

*Mar 2026 is partial (13 trading days).

### Conservative (2 Contracts) — Comparison

| Metric | Y2 | Blind |
|--------|----|-------|
| Net P&L | +$61,881.40 | +$60,509.76 |
| Monthly Average | +$4,760.11 | +$2,161.06 |
| Best Month | +$8,138.96 | +$6,818.84 |
| Worst Month | +$2,005.68 | -$936.24 |
| Max Drawdown | -$1,174.20 | -$2,907.80 |

---

## 5. Monte Carlo Validation

Monte Carlo simulations bootstrap-resample trades, perturb slippage (1–6 ticks random), jitter P&L by ±15%, and inject adverse gaps (2% probability, 5–25 point size). The target is to reach $9,000 cumulative profit without breaching the MLL (-$4,500).

### Aggressive (3 Contracts)

| Scenario | Pass Rate | Blowup Rate | Median Days to Target | P95 Days |
|----------|-----------|-------------|----------------------|----------|
| Baseline (1.0×) | 99.62% | 0.38% | 25 | 43 |
| Conservative (0.7×) | 100% | 0% | 35 | 56 |

### Conservative (2 Contracts)

| Scenario | Pass Rate | Blowup Rate | Median Days to Target | P95 Days |
|----------|-----------|-------------|----------------------|----------|
| Baseline (1.0×) | 100% | 0% | 37 | 58 |
| Conservative (0.7×) | 100% | 0% | 53 | 76 |

**Interpretation:** Even with 30% degraded performance (conservative 0.7× multiplier), the aggressive config passes 100% of simulations. The system has substantial margin of safety.

---

## 6. Trade Audit

A deep audit (`reports/deep_audit_v2.json`) verified trade-by-trade execution integrity:

| Check | Result |
|-------|--------|
| Total Trades Audited | 414 |
| Winners | 109 (26.33%) |
| Losers | 305 (73.67%) |
| TP Price Matches | 94/94 (100%) |
| SL Price Matches | 303/303 (100%) |
| Overlapping Trades | 0 |
| Exit Reason Breakdown (Winners) | 94 TP + 15 max hold |
| Exit Reason Breakdown (Losers) | 303 SL + 2 max hold |
| Avg Bars Held (Winners) | 2.04 |
| Avg Bars Held (Losers) | 1.34 |
| **Verdict** | **ALL CHECKS PASS** |

No price mismatches, no overlapping trades, no execution anomalies.

---

## 7. PineScript (TradingView) Implementation

Two PineScript files implement HTF Swing v3 for live demo trading on TradingView:

### `htf_swing_v3.pine` (Standard Session)
- RTH session: 09:30 – 15:45 ET
- EOD flatten: 15:45 ET
- All 3 sub-strategies (RSI, IB, MOM) with identical parameters to Python
- Visual overlays: entry markers, IB level lines, EMA(21), status table

### `htf_swing_v3_lucidflex_fixed.pine` (Extended Session)
- Extended session: 09:30 – 16:45 ET (LucidFlex rules)
- EOD flatten: 16:45 ET
- Fixed IB period to capture both 9:30 and 9:45 bars correctly on 15m chart
- All other logic identical

**TradingView limitation:** Free TradingView plans allow only 1 simultaneous position. The Python backtest runs all 3 sub-strategies in parallel (up to 9 contracts). TradingView results will be approximately 1/3 of Python results due to this constraint.

---

## 8. ANTHONY Folder CSV Guide

The `ANTHONY/` folder contains three CSV exports from `run_htf_swing_export.py` covering the full 8-year backtest (Dec 2017 – Mar 2026) at the aggressive 3-contract configuration.

### `htf_swing_v3_all_trades.csv` — Complete Trade Log

**14,426 rows** (one per trade). Key columns:

| Column Group | Fields | Description |
|-------------|--------|-------------|
| **Identity** | `trade_number`, `strategy` (RSI/IB/MOM), `direction` (LONG/SHORT), `contracts` | Which sub-strategy and direction |
| **Timing** | `signal_bar_timestamp`, `entry_bar_timestamp`, `exit_bar_timestamp` | Signal fires bar N, entry fills bar N+1 |
| **Pricing** | `entry_price`, `entry_price_raw`, `exit_price`, `exit_price_raw` | Raw = pre-slippage, main = post-slippage |
| **Exit** | `exit_reason` (SL/TP/max_hold/eod_flatten) | Why the trade closed |
| **P&L** | `gross_pnl`, `slippage_cost`, `commission_cost`, `exchange_fee`, `total_cost`, `net_pnl`, `cumulative_pnl` | Full cost decomposition |
| **Duration** | `bars_held`, `hold_time_minutes`, `points_moved` | How long and how far |
| **Risk** | `sl_distance_points`, `tp_distance_points`, `risk_dollars`, `reward_dollars` | Per-trade risk/reward |
| **Technicals** | `rsi_at_signal`, `atr_at_signal`, `volume_at_signal`, `avg_volume_20`, `bar_range_at_signal`, `ema21_at_signal`, `vwap_at_signal` | Market state at entry |
| **Context** | `day_of_week`, `hour_of_entry`, `month`, `year`, `is_first_hour`, `is_last_hour` | When did the trade occur |
| **Validation** | `data_period`, `validation_status`, `daily_trade_number`, `daily_pnl_before_this_trade`, `daily_pnl_after_this_trade` | Which test period and intraday tracking |

**How to use:** Filter by `strategy` to isolate RSI/IB/MOM performance. Filter by `data_period` to see Y1/Y2/Blind separately. Sort by `cumulative_pnl` to see equity curve progression. Check `exit_reason` distributions for edge validation.

### `htf_swing_v3_monthly_summary.csv` — Monthly Aggregates

**100 rows** (one per calendar month, Dec 2017 – Mar 2026). Key columns:

| Column | Description |
|--------|-------------|
| `total_trades`, `rsi_trades`, `ib_trades`, `mom_trades` | Volume breakdown by sub-strategy |
| `winners`, `losers`, `win_rate` | Monthly win/loss stats |
| `gross_pnl`, `total_costs`, `net_pnl`, `cumulative_pnl` | Monthly and running P&L |
| `best_trade`, `worst_trade`, `avg_trade`, `median_trade` | Trade distribution within month |
| `best_day`, `worst_day` | Best/worst single day P&L |
| `trading_days`, `avg_trades_per_day` | Activity level |
| `max_consecutive_wins`, `max_consecutive_losses` | Streak tracking |
| `avg_bars_held` | Average holding period |

**How to use:** This is the primary file for monthly performance review. Plot `net_pnl` over time to see profitability trends. Check `worst_day` to ensure daily limits are respected. Compare `rsi_trades` vs `ib_trades` vs `mom_trades` to see which sub-strategy is most active.

### `htf_swing_v3_daily_summary.csv` — Daily Aggregates

**~2,000+ rows** (one per trading day). Key columns:

| Column | Description |
|--------|-------------|
| `date`, `day_of_week` | Calendar date and weekday |
| `total_trades`, `rsi_trades`, `ib_trades`, `mom_trades` | Daily volume by sub-strategy |
| `winners`, `losers`, `win_rate` | Daily win/loss |
| `net_pnl`, `cumulative_pnl` | Daily P&L and running total |
| `max_open_positions` | Peak concurrent positions that day |
| `first_trade_time`, `last_trade_time` | Trading activity window |

**How to use:** Plot `net_pnl` as a histogram to see daily P&L distribution. Check `max_open_positions` to verify position limits are respected. Filter by `day_of_week` to look for weekday edge effects.

---

## 9. Other Strategies & Development History

The repo contains 40+ `run_*.py` scripts representing the full R&D journey. HTF Swing v3 is the final production system; everything below is development history.

### Search & Discovery Phase

| Script | Purpose |
|--------|---------|
| `run_search.py`, `run_search2.py`, `run_search3.py` | Brute-force combinatorial strategy search across signal library |
| `run_search_oct.py` | Targeted search on Oct 2025 NQZ5 contract |
| `run_search_1yr.py` | Year 1 (Mar 2024 – Mar 2025) focused search |
| `run_pipeline.py` | Full automated pipeline: data → search → MC → validate → paper |

### Evolution & Breeding Phase

| Script | Purpose |
|--------|---------|
| `run_evolve.py` | Genetic algorithm: mutate top strategies, 2000 MC sims per candidate |
| `run_breed_150k.py` | Sniper mode breeding (tight SL 10–18pt, large TP 100–300pt) |
| `run_crossbreed_v4.py` | Cross-family evolution (swap entry signals between strategy families) |
| `run_crossbreed_v4_merge.py` | Merge parallel batch breeding results + decorrelation analysis |
| `run_portfolio_breed.py` | Breed 3–5 strategy portfolio combinations |
| `run_portfolio_breed_y2.py` | Year 2 OOS validation of bred portfolios |

### Maximization Phase

| Script | Purpose |
|--------|---------|
| `run_maximize.py` | Push Stochastic+Imbalance edge to $20K+/month |
| `run_maximize_150k.py` | Aggressive sizing on 150K account |
| `run_maximize_champion.py` | Mutate ROC×KC champion to $15K+/month target |
| `run_grinder_maximize.py` | Squeeze 4-strategy grinder portfolio to $7K+/month |
| `run_safe_maximize.py` | Max profit while capping losses at $4K/month |
| `run_max_profit.py` | Dual-year validation with position sizing optimization |

### Validation Phase

| Script | Purpose |
|--------|---------|
| `run_oos_test.py` | Out-of-sample test: 39 evolved strategies on unseen Oct 2025 |
| `run_final_validation.py` | Full year + OOS + MC 5000 sims |
| `run_nuclear_validation.py` | Year 2 blind test (never seen in optimization) |
| `run_extended_validation.py` | 2022–2024 blind test on strategies built for 2024–2026 |
| `run_final_push.py` / `phase2` / `phase3` | Progressive validation tightening and multi-TF testing |
| `run_frequency_fix.py` | Ensure strategies trade every month across both years |

### Scalper / HFT Experiments

| Script | Purpose |
|--------|---------|
| `run_hft_scalper.py` | 20–50 trades/day, 52%+ WR, 5–15pt SL/TP |
| `run_hft_v2.py` | Realistic HFT execution (limit fills, 0 entry slippage) |
| `run_adaptive_scalper.py` | Rolling 2-week re-optimization with ATR-adaptive exits |
| `run_adaptive_scalper_v2.py` | Memory-efficient precomputed features |

### Always-On Systems (trade every day, no flatten)

| Script | Purpose |
|--------|---------|
| `run_always_on.py` | 3-layer system: VWAP mean reversion + session structure + volatility momentum |
| `run_always_on_v2.py` | Added multi-TF VWAP, momentum continuation |
| `run_always_on_v3.py` | Removed L1, blind tested on 2022–2024 |
| `run_always_on_v4.py` | Honest exits (no same-bar exits, conservative SL/TP) |
| `run_always_on_mc.py` | MC simulation (5K sims) on Always-On for Topstep 150K |
| `run_always_on_databento.py` | Blind test on Databento 2022–2024 data |

### HTF Swing Evolution (leading to v3)

| Script | Purpose |
|--------|---------|
| `run_htf_swing.py` | v1: 6 strategy families × 3 timeframes (15m/1h/4h) |
| `run_htf_swing_v2.py` | v2: Prop-firm constrained 2–3 strategy combo |
| `run_htf_swing_v3.py` | **v3: Production system** — no data leakage, Y1 selects, Y2/blind OOS |
| `run_htf_swing_v4.py` | v4: Adaptive contract sizing based on conditions score |
| `run_htf_swing_v5.py` | v5: Drop IB, vol filter, broader training window |
| `run_htf_swing_8yr.py` | 8-year backtest across merged datasets |
| `run_htf_swing_5m.py` | Same params on 5m bars for higher frequency |
| `run_htf_swing_lucid.py` | LucidFlex variant: 5 daily limits × 2 session lengths |
| `run_htf_swing_competition.py` | Head-to-head: Topstep vs LucidFlex |
| `run_htf_swing_diagnose.py` | Trade/signal log for Pine Script divergence diagnosis |
| `run_htf_swing_export.py` | Export all trades to CSV (generates ANTHONY/ folder files) |

### Multi-Strategy Systems

| Script | Purpose |
|--------|---------|
| `run_tiered_backtest.py` | Champion + 5 grinder tiers with priority preemption |
| `run_unified_system.py` | Alpha (rare, big) + base (frequent, small) combination |
| `run_champion_frequency.py` | Tune ROC×KC champion to trade 5+ times/month |
| `run_edge_finder.py` | Find strategies: WR>25%, winner>1.5× loser |
| `run_trailing_profit.py` | Add trailing stops to dual-year survivors |

---

## 10. Engine Internals

### Backtester Execution Flow (per bar)

```
1. Day boundary? → Reset daily P&L, kill switch, trades_today
2. EOD flatten time? → Close all positions at market
3. Pending entry from previous bar? → Fill at this bar's open + slippage
4. Overnight gap check → If open gaps beyond SL, fill at open (worst case)
5. Update trailing stop → Ratchet stop if profit activation reached
6. Check SL/TP → If both hit on same bar, use open-distance heuristic
7. Check exit signals → Close on signal
8. Check entry signals → Queue for next bar (lookahead-free)
9. Update equity curve → Record intrabar unrealized P&L
```

### Risk Manager — 7 Pre-Trade Checks (in order)

1. **Kill switch**: At 80% of daily loss limit → block ALL entries for rest of day
2. **Session bounds**: Only 08:00–17:00 ET
3. **Event blackout**: ±15 min of NFP, CPI, FOMC, etc.
4. **Daily loss limit**: Block if daily P&L ≤ limit
5. **Max drawdown**: Block if current DD ≤ limit
6. **Max contracts**: Enforce position size caps
7. **EOD proximity**: No new entries within 5 min of 17:00 ET

### Strategy Protocol

Any strategy must implement:

```python
class Strategy(Protocol):
    name: str
    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        # Must return: entry_long, entry_short, exit_long, exit_short (bool columns)
    def get_stop_loss(self, entry_price, direction) -> float | None
    def get_take_profit(self, entry_price, direction) -> float | None
    def get_position_size(self, account_state, contract_spec, prop_rules) -> int
```

### Slippage Model

- Base: 2 ticks per side (configurable)
- +2 ticks if volume < 100 (thin market)
- +1 tick if off-hours (before 9 AM or after 4 PM ET)
- Applied against trade direction (longs fill higher, shorts fill lower)

### Key Bug Fixes Embedded in Engine

| Fix | Description |
|-----|-------------|
| #1 | Trailing stop ratchets correctly based on activation/distance |
| #2 | SL/TP same-bar resolution uses open-distance heuristic |
| #3 | Lookahead-free: signal bar N → fill bar N+1 open |
| #5 | Slippage range 1–6 ticks with time-of-day penalty |
| #7 | Recurring event date generation (NFP, CPI, FOMC patterns) |
| #9 | Intrabar equity curve for realistic max drawdown |
| #14 | Stochastic zone check within 3 bars (not just exact cross) |
| #29 | Per-bar ATR array for correct per-bar stop/target lookups |
| #30 | DD-protection: halve position size when DD > 50% of allowed |
| #32 | Overnight gap handling: fill at open if gap beyond SL |

---

## 11. Signal Library

The `signals/` directory contains a modular registry of 30+ signal functions organized by category.

### Trend Signals (`signals/trend.py`)

| Signal | Parameters | Entry Columns |
|--------|-----------|---------------|
| EMA Crossover | fast [5–50], slow [10–100] | `entry_long_ema_cross`, `entry_short_ema_cross` |
| EMA Slope | period, slope_lookback | `signal_ema_slope_up/down` |
| EMA Ribbon | periods (list) | `signal_ema_ribbon_bullish/bearish` |
| Linear Regression Slope | period | `signal_linreg_up/down` |
| Heikin-Ashi | none | `signal_ha_bullish/bearish` |
| Supertrend | period, multiplier | `signal_supertrend_bullish/bearish` |

### Momentum Signals (`signals/momentum.py`)

| Signal | Parameters | Entry Columns |
|--------|-----------|---------------|
| RSI | period, overbought, oversold | `entry_long_rsi`, `entry_short_rsi` |
| MACD | fast, slow, signal_period | `entry_long_macd`, `entry_short_macd` |
| Stochastic | k_period, d_period, OB, OS | `entry_long_stoch`, `entry_short_stoch` |

### Volatility Signals (`signals/volatility.py`)

| Signal | Parameters | Output |
|--------|-----------|--------|
| ATR | period | `atr_{period}` (indicator only) |
| Bollinger Bands | period, std_dev | `entry_long_bb`, `entry_short_bb`, bands |
| Keltner Channels | ema_period, atr_period, multiplier | Entry/filter signals |

### Volume Signals (`signals/volume.py`)

| Signal | Parameters | Output |
|--------|-----------|--------|
| VWAP | none (session-daily) | `entry_long_vwap`, `entry_short_vwap`, bands |
| Volume Delta | none | `cumulative_delta` (indicator) |

### Orderflow Signals (`signals/orderflow.py`)

| Signal | Parameters | Output |
|--------|-----------|--------|
| Delta Divergence | lookback | `signal_delta_div_bullish/bearish` |
| Absorption | volume_threshold, price_threshold | Absorption bar detection |

### Price Action (`signals/price_action.py`)

| Signal | Parameters | Output |
|--------|-----------|--------|
| Session Levels | none | `signal_above/below_session_open` |
| Previous Day Levels | none | `entry_long_prev_low_bounce`, `entry_short_prev_high_reject` |

### Time Filters (`signals/time_filters.py`)

| Signal | Parameters | Output |
|--------|-----------|--------|
| Time of Day | start/end hour:minute | `signal_in_time_window` |
| Day of Week | allowed_days list | `signal_day_allowed` |

### Strategy Generator

The `GeneratedStrategy` class (`strategies/generator.py`) can combine any of the above signals into a complete strategy by specifying:
- `entry_signals`: List of signal functions (combined AND or OR)
- `entry_filters`: List of filter functions (always AND — restrict, never widen)
- `exit_rules`: SL type (fixed/ATR/percent), TP type, trailing stop, time exit
- `sizing_rules`: Fixed contracts, ATR-scaled, or risk-percent based

This combinatorial generator powered the search/evolution phases that ultimately led to HTF Swing v3.

---

## 12. Validation Framework

### Walk-Forward Validation (`validation/walk_forward.py`)

Rolling train/test windows (default 60-day train, 20-day test, 20-day step):
- **Walk-Forward Efficiency (WFE)** = OOS Sharpe / IS Sharpe
  - \>0.7 = Great (genuine edge)
  - \>0.5 = Good
  - <0.3 = Likely overfit
- Tracks: per-window Sharpe, P&L, % profitable windows, Sharpe stability

### Regime Analysis (`validation/regime.py`)

Classifies market into regimes and measures strategy robustness across them:
- Trending Up / Trending Down / Ranging / High Volatility / Low Volatility / Breakout
- **Regime sensitivity** (0–1, lower = more robust across conditions)
- Per-regime: win rate, Sharpe, profit factor, trade count

### Overfitting Detection (`validation/correlation.py`)

- **Parameter sensitivity**: Jitter params ±10% and measure metric stability
- **Time stability**: Compare Sharpe across time periods, check for decay
- **IS/OOS degradation**: Compare in-sample vs out-of-sample performance drop
- **Red flags**: Automatically flagged if high sensitivity or severe degradation

### Monte Carlo Stress Testing (`monte_carlo/simulator.py`)

- **10,000 simulations** (default) with:
  - Trade shuffle (bootstrap resampling with replacement)
  - Slippage perturbation (1–6 ticks random)
  - Parameter jitter (±15% P&L scaling)
  - Gap injection (2% probability, 5–25 point adverse gaps)
- Outputs: probability of profit, probability of ruin, prop firm pass rate, equity fan charts
- **Scoring** (0–100 composite): 20% Sharpe + 25% pass rate + 15% drawdown + 10% profit factor + 10% consistency + 10% ruin + 10% robustness
- **Grading**: A+ through F with viability flag

---

## 13. File Map

```
berjquant/
├── engine/                          # Core backtesting & risk engine
│   ├── utils.py                     # Data structures (ContractSpec, Trade, etc.)
│   ├── backtester.py                # VectorizedBacktester, Strategy protocol
│   ├── risk_manager.py              # 7-step pre-trade validation
│   ├── data_pipeline.py             # CSV → Parquet pipeline
│   ├── data_fetcher.py              # YFinance data fetcher
│   ├── metrics.py                   # Performance analytics, SQLite logger
│   ├── parallel_runner.py           # Multi-process batch execution
│   ├── fast_backtester.py           # HFT current-bar fills
│   ├── tiered_backtester.py         # Multi-tier priority strategies
│   ├── adaptive.py                  # ATR-adaptive exits, regime detection
│   └── adaptive_backtester.py       # Adaptive backtester wrapper
├── strategies/                      # Strategy definitions
│   ├── ema_crossover.py             # Reference EMA crossover strategy
│   ├── generator.py                 # GeneratedStrategy (combinatorial assembly)
│   └── serializer.py                # JSON save/load for strategies
├── signals/                         # Signal library (30+ functions)
│   ├── registry.py                  # Signal catalog & dependency tracking
│   ├── trend.py                     # EMA, Supertrend, Heikin-Ashi, LinReg
│   ├── momentum.py                  # RSI, MACD, Stochastic
│   ├── volatility.py                # ATR, Bollinger Bands, Keltner
│   ├── volume.py                    # VWAP, Volume Delta
│   ├── orderflow.py                 # Delta Divergence, Absorption
│   ├── price_action.py              # Session Levels, Previous Day
│   └── time_filters.py              # Time of Day, Day of Week
├── config/                          # Configuration files
│   ├── prop_firms.yaml              # Prop firm rules (Topstep, Apex, Lucid)
│   ├── sessions.yaml                # Trading hours, flat_by time
│   └── events_calendar.yaml         # Economic events & blackout rules
├── validation/                      # Overfitting & robustness testing
│   ├── walk_forward.py              # Walk-forward optimization
│   ├── regime.py                    # Market regime detection
│   └── correlation.py               # Portfolio diversification, overfit detection
├── monte_carlo/                     # Stress testing
│   ├── simulator.py                 # MC engine (10K sims, bootstrap + jitter)
│   ├── scoring.py                   # Strategy scoring & grading (A+ to F)
│   └── visualization.py             # Equity fan charts, distribution plots
├── live/                            # Real-time execution
│   ├── signal_engine.py             # Bar-by-bar signal generation
│   ├── paper_trader.py              # Paper trading simulator
│   ├── alerts.py                    # Risk breach notifications
│   └── dashboard.py                 # Web monitoring UI
├── data/                            # Market data (Parquet)
│   └── processed/MNQ/              # 1m raw + 1h monthly partitions (2017–2026)
├── results/                         # Output storage
│   └── trades.db                    # SQLite trade database
├── reports/                         # ~95 JSON/CSV/HTML/log report files
├── ANTHONY/                         # Trade exports for independent review
│   ├── BERJQUANT_STATE.md           # This document
│   ├── htf_swing_v3_all_trades.csv  # 14,426 trades (full 8-year history)
│   ├── htf_swing_v3_monthly_summary.csv  # 100 months of aggregated data
│   └── htf_swing_v3_daily_summary.csv    # ~2,000+ daily aggregates
├── htf_swing_v3.pine                # TradingView strategy (standard session)
├── htf_swing_v3_lucidflex_fixed.pine # TradingView strategy (extended session)
├── run_htf_swing_v3.py              # Production backtest script
├── run_htf_swing_export.py          # CSV export generator
├── run_*.py (40+ scripts)           # Development history (search, breed, validate)
├── pyproject.toml                   # Project config & dependencies
├── requirements.txt                 # pip dependencies
├── README.md                        # Quick-start guide
└── AGENTS.md                        # CLI automation guide
```

### Dependencies

```
polars >= 0.20       # Primary DataFrame engine
pandas >= 2.1        # Secondary (YFinance compat)
numpy >= 1.26        # Numerical computation
scipy >= 1.12        # Statistical functions
pyarrow >= 14        # Parquet I/O
plotly >= 5.18       # Interactive charts
matplotlib >= 3.8    # Static charts
pyyaml >= 6.0        # Config parsing
pytest >= 7.0        # Testing
```

---

*Generated March 26, 2026. For the latest results, re-run `run_htf_swing_export.py` and `run_htf_swing_v3.py`.*
