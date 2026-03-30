---
date: 2026-03-29
type: reference
tags: [berjquant, performance, results, metrics]
---

# Performance Results

All results from Hybrid v2 (aggressive 3-contract configuration).

## Y2 Out-of-Sample (Jun 2025 – Mar 2026)

| Month | Trades | WR | Net P&L |
|-------|--------|----|---------|
| Jun 2025 | 154 | 24.7% | +$3,578 |
| Jul 2025 | 145 | 29.7% | +$5,669 |
| Aug 2025 | 150 | 22.7% | +$2,739 |
| Sep 2025 | 148 | 27.0% | +$6,948 |
| Oct 2025 | 203 | 22.7% | +$4,602 |
| Nov 2025 | 157 | 26.1% | +$10,675 |
| Dec 2025 | 141 | 29.1% | +$7,569 |
| Jan 2026 | 163 | 25.8% | +$9,094 |
| Feb 2026 | 157 | 26.8% | +$11,505 |
| Mar 2026* | 85 | 25.9% | +$6,072 |

*Mar 2026 partial (13 trading days)

**Y2 Summary:** 13/13 months profitable. Monthly avg +$7,140. Max DD -$1,761.

## Full 8-Year (Dec 2017 – Mar 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 14,426 |
| Cumulative P&L | +$313,670 |
| Years Profitable | 7/8 |
| Worst Year (2018) | -$602 (nearly flat) |
| Daily Sharpe | 6.08 |
| Monthly Sharpe | 3.63 |

## Blind Test (Dec 2017 – Mar 2024)

| Metric | Value |
|--------|-------|
| Total Trades | 4,129 |
| Net P&L | +$90,764 |
| Monthly Avg | +$3,241 |
| Months Profitable | 20/28 (71.4%) |
| Max DD | -$4,361 |

## Monte Carlo (10K Sims, 6 Degradation Factors)

| Metric | Value |
|--------|-------|
| Median monthly | $6,307 |
| P25 (conservative) | $5,711 |
| Eval pass rate | 99.2% |
| Loss-limit breach | 0% |
| Haircut from backtest | 42% |

## Regime Dependency

| Regime | Monthly P&L |
|--------|-------------|
| High volatility | +$12,884 |
| Low volatility | -$1,286 |
| Current (Mar 2026) | HIGH VOL (381pt avg daily range) |

## Edge Profile

- 69% profit in first trading hour
- Strategy correlation: -0.046 (excellent diversification)
- Win rate: 26-28% with 3.3-3.8 R:R
- Only 4.3% of losers within 2pts of TP

See also: [[14 - Hybrid v2 Finalization and Deployment]], [[Strategy Parameters]]
