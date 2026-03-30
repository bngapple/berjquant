# BerjQuant — HTF Swing v3 Independent Review Package

This package contains everything needed to independently reproduce and validate the HTF Swing v3 trading strategy for MNQ (Micro E-mini Nasdaq 100 Futures).

---

## Quick Start

```bash
# 1. Install dependencies
pip install polars numpy

# 2. Run from inside this directory
cd anthony/

# 3. Run the production backtest (HTF Swing v3)
python run_htf_swing_v3.py

# 4. Run the full 8-year backtest
python run_htf_swing_8yr.py

# 5. Export all trades to CSV
python run_htf_swing_export.py
```

---

## Dependencies

```
polars >= 0.20    # DataFrame engine (required)
numpy >= 1.26     # Numerical computation (required)
```

Optional (for other scripts): `scipy`, `matplotlib`, `plotly`. The core scripts only need `polars` and `numpy`.

Full list in `requirements.txt`.

---

## Scripts — What Each Does

### Production Scripts (start here)

| Script | What It Does | Expected Output |
|--------|-------------|-----------------|
| `run_htf_swing_v3.py` | **Main backtest.** Tests all 2-3 strategy combos on Y1 (2024-2025), selects best Sharpe, then runs Y2 OOS (2025-2026) and Blind (2022-2024) at 2c and 3c sizing. Runs Monte Carlo (5000 sims). | JSON report to `reports/htf_swing_v3.json` |
| `run_htf_swing_8yr.py` | Full 8-year backtest (2018-2026) merging all 3 datasets. Tests RSI+IB+MOM combo across entire history. | JSON report to `reports/htf_swing_8yr.json` |
| `run_htf_swing_export.py` | Exports all trades from the 8-year backtest to CSV: all_trades, monthly_summary, daily_summary. | 3 CSV files in `reports/` |
| `run_htf_swing_competition.py` | Head-to-head: Topstep vs LucidFlex prop firm rules on same strategy. | JSON report to `reports/htf_swing_competition.json` |
| `run_htf_swing_lucid.py` | LucidFlex-specific variant: tests 5 daily limits x 2 session lengths. | JSON report to `reports/htf_swing_lucid.json` |

### Base Module (imported by all scripts)

| Script | What It Does |
|--------|-------------|
| `run_htf_swing.py` | **Base module.** Contains all signal functions (`sig_rsi_extreme`, `sig_ib_breakout`, `sig_momentum_bar`, etc.), the backtester, data loader, cost model, and constants. All other scripts import from this. Also runs a standalone 2-year backtest if executed directly. |

### Evolution Scripts (development history)

| Script | What It Does |
|--------|-------------|
| `run_htf_swing_v2.py` | v2: Prop-firm constrained 2-3 strategy portfolio selection |
| `run_htf_swing_v4.py` | v4: Adaptive contract sizing based on conditions score |
| `run_htf_swing_v5.py` | v5: Dropped IB, added vol filter, broader training window |
| `run_htf_swing_5m.py` | Same v3 params but on 5-minute bars (higher frequency test) |
| `run_htf_swing_diagnose.py` | Trade/signal log for diagnosing Pine Script vs Python divergence |

---

## Data Files

All data is MNQ 1-minute OHLCV bars from Databento, stored as Parquet files.

### Primary Data (used by production scripts)

| File | Date Range | Size | Used By |
|------|-----------|------|---------|
| `data/processed/MNQ/1m/full_2yr.parquet` | Mar 2024 – Mar 2026 | 3.4 MB | v3 (Y1+Y2), competition, lucid, export |
| `data/processed/MNQ/1m/databento_extended.parquet` | Jan 2022 – Mar 2024 | 9.2 MB | v3 (Blind), 8yr, competition, export |
| `data/processed/MNQ/1m/databento_8yr_ext.parquet` | Dec 2017 – Dec 2021 | 15 MB | 8yr, export |

### Additional Data

| File | Description | Size |
|------|-------------|------|
| `data/processed/MNQ/1m/extended_history.parquet` | Extended historical data | 8.0 MB |
| `data/processed/MNQ/1m/all.parquet` | Combined recent data | 220 KB |
| `data/processed/MNQ/1m/2026-02.parquet` | February 2026 partition | 61 KB |
| `data/processed/MNQ/1m/2026-03.parquet` | March 2026 partition | 93 KB |
| `data/processed/MNQ/databento_nq_1m_raw.parquet` | Raw NQ 1m data (pre-processing) | 16 MB |
| `data/processed/MNQ/5m/*.parquet` | 5-minute resampled (4 files) | ~166 KB |
| `data/processed/MNQ/1h/*.parquet` | 1-hour resampled (27 monthly files, 2023-10 to 2026-03) | ~93 KB total |

### Parquet Schema

All 1m files share this schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Bar timestamp (UTC) |
| `open` | float64 | Open price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Close price |
| `volume` | int64 | Volume |
| `tick_count` | int64 | Number of ticks in bar |

---

## Reports (Pre-Computed Results)

These are the results from our runs. You can regenerate them by running the scripts above.

### JSON Reports

| File | Description |
|------|-------------|
| `reports/htf_swing_v3.json` | **Primary.** Y1 selection, Y2 OOS, Blind test, Monte Carlo results for 2c and 3c |
| `reports/htf_swing_8yr.json` | 8-year backtest results (2018-2026) |
| `reports/htf_swing_competition.json` | Topstep vs LucidFlex comparison |
| `reports/htf_swing_lucid.json` | LucidFlex variant results |
| `reports/htf_swing_v1.json` | v1 results (6 families x 3 timeframes) |
| `reports/htf_swing_v2.json` | v2 results (prop-firm constrained) |
| `reports/htf_swing_v4.json` | v4 results (adaptive sizing) |
| `reports/htf_swing_v5.json` | v5 results (no IB, vol filter) |
| `reports/htf_swing_5m.json` | 5-minute variant results |

### CSV Reports (Trade-Level Data)

| File | Rows | Description |
|------|------|-------------|
| `reports/htf_swing_v3_all_trades.csv` | 14,426 | Every trade across 8 years with 65 columns (P&L, technicals, costs, context) |
| `reports/htf_swing_v3_monthly_summary.csv` | 100 | Monthly aggregates (Dec 2017 – Mar 2026) |
| `reports/htf_swing_v3_daily_summary.csv` | ~2,000 | Daily aggregates with strategy breakdown |

---

## Expected Results (for validation)

When you run `python run_htf_swing_v3.py`, you should see results close to:

### HTF Swing v3 — Aggressive (3 contracts)

| Metric | Y2 (OOS) | Blind |
|--------|----------|-------|
| Total Trades | 1,810 | 4,129 |
| Win Rate | 26.96% | 24.70% |
| Net P&L | +$92,822 | +$90,765 |
| Monthly Average | +$7,140 | +$3,242 |
| Best Month | +$12,208 | +$10,228 |
| Worst Month | +$3,009 | -$1,404 |
| Max Drawdown | -$1,761 | -$4,362 |
| Months Profitable | 13/13 (100%) | 20/28 (71%) |
| MC Pass Rate (baseline) | 99.62% | — |
| MC Pass Rate (conservative 0.7x) | 100% | — |

### HTF Swing v3 — Conservative (2 contracts)

| Metric | Y2 (OOS) | Blind |
|--------|----------|-------|
| Net P&L | +$61,881 | +$60,510 |
| Monthly Average | +$4,760 | +$2,161 |
| Max Drawdown | -$1,174 | -$2,908 |
| MC Pass Rate | 100% | — |

### Full 8-Year (run_htf_swing_8yr.py)

| Metric | Value |
|--------|-------|
| Total Trades | 14,426 |
| Cumulative Net P&L | +$313,670 |
| Date Range | Dec 2017 – Mar 2026 |

---

## Strategy Overview

HTF Swing v3 runs **3 independent sub-strategies** simultaneously on 15-minute MNQ bars:

### Sub-Strategy 1: RSI Extremes
- RSI(7) < 30 → Long | RSI(7) > 70 → Short
- SL: 15 points | TP: 100 points | Max hold: 5 bars
- ~70-80 trades/month

### Sub-Strategy 2: IB Breakout
- Captures Initial Balance (9:30-9:45 ET high/low)
- Breakout above IB → Long | Below → Short
- IB range filtered by 25th-75th percentile of last 50 IB ranges
- SL: 20 points | TP: 120 points | Max hold: 10 bars | Max 1 trade/day
- ~4-14 trades/month

### Sub-Strategy 3: Momentum Bar
- Bar range > ATR(14) AND volume > SMA(vol, 20) AND direction agrees with EMA(21)
- SL: 15 points | TP: 100 points | Max hold: 5 bars
- ~40-70 trades/month

### Key Design
- **Asymmetric R:R**: Wins only ~25% of the time, but winners are 6-7x larger than losers
- **Lookahead-free**: Signal on bar N, fill at bar N+1 open
- **Costs included**: $11.34/trade (commission + exchange + 2-tick slippage per side)
- **Risk controls**: -$3,000 daily limit, -$4,500 monthly limit, EOD flatten at 16:45 ET

---

## Pine Script (TradingView)

| File | Session |
|------|---------|
| `htf_swing_v3.pine` | Standard RTH (9:30 – 15:45 ET) |
| `htf_swing_v3_lucidflex_fixed.pine` | Extended (9:30 – 16:45 ET, LucidFlex rules) |

**Note:** Free TradingView allows only 1 position at a time. Python runs all 3 sub-strategies in parallel (up to 9 contracts). TradingView results will be ~1/3 of Python results.

To use: Open TradingView → Pine Script Editor → paste file contents → Add to Chart on MNQ 15m.

---

## Directory Structure

```
anthony/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── run_htf_swing.py                 # Base module (signals, backtester, data loader)
├── run_htf_swing_v3.py              # Production backtest (start here)
├── run_htf_swing_8yr.py             # 8-year full history backtest
├── run_htf_swing_export.py          # Export trades to CSV
├── run_htf_swing_competition.py     # Topstep vs LucidFlex comparison
├── run_htf_swing_lucid.py           # LucidFlex variant
├── run_htf_swing_v2.py              # v2 (development history)
├── run_htf_swing_v4.py              # v4 (development history)
├── run_htf_swing_v5.py              # v5 (development history)
├── run_htf_swing_5m.py              # 5-minute variant
├── run_htf_swing_diagnose.py        # Pine Script divergence diagnosis
├── htf_swing_v3.pine                # TradingView strategy (standard)
├── htf_swing_v3_lucidflex_fixed.pine # TradingView strategy (LucidFlex)
├── data/processed/MNQ/
│   ├── databento_nq_1m_raw.parquet  # Raw NQ 1m (16 MB)
│   ├── 1m/
│   │   ├── full_2yr.parquet         # 2024-2026 (3.4 MB)
│   │   ├── databento_extended.parquet # 2022-2024 (9.2 MB)
│   │   ├── databento_8yr_ext.parquet # 2018-2021 (15 MB)
│   │   ├── extended_history.parquet # Extended history (8 MB)
│   │   └── ...                      # Monthly partitions
│   ├── 5m/                          # 5-minute bars
│   └── 1h/                          # 1-hour bars (27 monthly files)
└── reports/
    ├── htf_swing_v3.json            # Production results
    ├── htf_swing_8yr.json           # 8-year results
    ├── htf_swing_v3_all_trades.csv  # 14,426 trades
    ├── htf_swing_v3_monthly_summary.csv
    ├── htf_swing_v3_daily_summary.csv
    └── ...                          # Other version reports
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'polars'"**
→ `pip install polars numpy`

**"FileNotFoundError: data/processed/MNQ/..."**
→ Make sure you're running scripts from inside the `anthony/` directory, not from a parent.

**Different P&L numbers than expected**
→ Minor floating-point differences across platforms are normal. Results should be within $50 of expected values.

**Script imports fail (e.g., "cannot import from run_htf_swing")**
→ All scripts must be in the same directory. Run from `anthony/`.
