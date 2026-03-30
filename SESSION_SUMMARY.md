# BerjQuant Session Summary — March 26-29, 2026

## What We Did This Session

### 1. Comprehensive System Documentation
- Created `ANTHONY/BERJQUANT_STATE.md` — 743-line full system overview (pushed to GitHub)
- Packaged `anthony.zip` (51 MB) with all data, scripts, reports, Pine Scripts for Anthony's independent review
- Created Obsidian vault at `~/Documents/BerjQuant-Vault/` with 21 interlinked notes covering all 14 conversations

### 2. Parameter Optimization (found Hybrid v2)
- **`run_htf_swing_optimize.py`** — Grid search: 12,000 combos across RSI/IB/MOM with train/test/blind splits
- **Discovery:** SL=10 dominates across all strategies. RSI period=5 with wider bands (35/65) generates 72% more signals
- **`run_htf_swing_v3_hybrid.py`** — Tested TradingView-validated params (RSI+IB optimized, MOM ATR=1.4)
- **`run_htf_swing_v3_hybrid_v2.py`** — Reverted MOM to ATR=1.0, which recovered $1,373/mo. This is the final config.

### 3. Final Params — Hybrid v2 (DEPLOY THIS)
```
RSI: period=5, OS=35, OB=65, SL=10pts, TP=100pts, hold=5
IB:  SL=10pts, TP=120pts, hold=15, filter=True
MOM: ATR=1.0, vol=1.0, SL=15pts, TP=100pts, hold=5
Session: LucidFlex (9:30-16:45 ET), 3 MNQ/strategy, 2-tick slippage
```

### 4. Validation & Auditing
- **`run_htf_swing_v3_audit_hybrid.py`** — 27/27 PASS paranoia audit (data, signals, execution, statistics, parameter stability, shuffle test)
- **`run_htf_swing_v3_realworld_mc.py`** — 10K sim real-world Monte Carlo with 6 degradation factors → realistic median $6,307/mo (42% haircut from backtest $10,946)
- **`run_htf_swing_v3_tp_optimize.py`** — TP=100/120 confirmed optimal. Only 4.3% of losers near-missed TP.
- **`run_htf_swing_v3_annual.py`** — Year-by-year: 7/8 full years profitable. 8yr cumulative $563K vs current's $314K.
- **`run_htf_swing_v3_sharpe.py`** — 8-year daily Sharpe 6.08, monthly Sharpe 3.63

### 5. Regime & Pre-Market Analysis
- **`live/regime_detector.py`** — System earns $12,884/mo in high-vol, loses $1,286/mo in low-vol. Current regime: HIGH VOL (381pt avg daily range)
- **`run_htf_swing_v3_premarket_filter.py`** — London ATR proxy has r=0.71 correlation with RTH range, but filter rejected (helps 2018-19, hurts 2022+)

### 6. CSV Exports for Anthony
- **`run_htf_swing_export_v2.py`** — Generated hybrid v2 trade exports in `ANTHONY/`:
  - `htf_swing_v2_all_trades.csv` (20,058 trades, 6.5 MB)
  - `htf_swing_v2_monthly_summary.csv` (100 months)
  - `htf_swing_v2_daily_summary.csv` (2,123 days)

## Key Numbers

| Metric | Value |
|--------|-------|
| Y2 OOS monthly avg | $10,946 |
| Realistic (MC P50) | $6,307/mo |
| Conservative (MC P25) | $5,711/mo |
| 8-year cumulative | $563,408 |
| 8-year daily Sharpe | 6.08 |
| MC eval pass rate | 99.2% |
| Paranoia audit | 27/27 PASS |
| Worst year (2018) | -$602 (nearly flat) |

## Files Created This Session

```
run_htf_swing_optimize.py          # Parameter grid search
run_htf_swing_v3_hybrid.py         # Hybrid v1 (ATR=1.4) backtest
run_htf_swing_v3_hybrid_v2.py      # Hybrid v2 (ATR=1.0) — THE WINNER
run_htf_swing_v3_audit_hybrid.py   # 27-point paranoia audit
run_htf_swing_v3_realworld_mc.py   # Real-world Monte Carlo (6 factors)
run_htf_swing_v3_tp_optimize.py    # TP fill optimization
run_htf_swing_v3_annual.py         # Year-by-year breakdown
run_htf_swing_v3_sharpe.py         # 8-year Sharpe calculation
run_htf_swing_v3_premarket_filter.py # Pre-market volatility filter
run_htf_swing_export_v2.py         # CSV export with hybrid v2 params
live/regime_detector.py            # Volatility regime classifier
ANTHONY/BERJQUANT_STATE.md         # Full system doc (on GitHub)
SESSION_SUMMARY.md                 # This file
```

## What's Left To Do

1. **Deploy:** Purchase LucidFlex 150K eval (target May 2026)
2. **Pine Script:** Update `htf_swing_v3.pine` with hybrid v2 params for TradingView forward testing
3. **Anthony:** Send him the hybrid v2 CSVs + BERJQUANT_STATE.md for independent review
4. **Forward test:** Run the Databento pipeline (`berjquant_forward_test.zip`) with hybrid v2 params
5. **Commit:** New scripts aren't committed to GitHub yet
