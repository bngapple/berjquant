---
date: 2026-03-27
type: conversation
tags: [berjquant, hybrid-v2, paranoia-audit, monte-carlo, regime-analysis, deployment]
url: https://claude.ai/chat/99c1e1ef-5e93-45bb-806f-f3e67126fdfb
chat_number: 14
---

# Hybrid v2 Finalization and Deployment

> [!tip] This is the culminating session

## Summary

Hybrid v2 params finalized and fully validated. 27/27 paranoia audit clean. Monte Carlo 10K sims confirmed. Deployment plan set.

## Key Decisions / Findings

- Hybrid v2 params finalized: RSI (period=5, OS=35, OB=65, SL=10, TP=100, hold=5), IB (SL=10, TP=120, hold=15), MOM (ATR=1.0)
- 27/27 paranoia audit clean
- Improvement explained: 72% more RSI signals + $74 vs $104 per-loss
- MC 10K sims (6 degradation factors): median $6,307/mo, 99.2% pass, 0% loss-limit breach
- TP optimal at 100/120
- 7/8 years profitable, Sharpe 3.63
- Regime: high-vol +$12,884/mo, low-vol -$1,286/mo
- London filter rejected (overfitting)
- Deploy: LucidFlex May 2026, Pine Script forward testing, Tradovate API in separate project

## Deliverables

- Hybrid v2 parameter set (final)
- 27-point paranoia audit
- Real-world Monte Carlo (10K sims, 6 degradation factors)
- TP optimization analysis
- Year-by-year breakdown
- Regime analysis
- Pre-market filter evaluation (rejected)
- Deployment plan

## Open Questions

-

## Links

- Previous: [[13 - Full Audit and Forward Test Pipeline]]
- Next: —
- Related: [[Strategy Parameters]], [[Performance Results]], [[Key Learnings and Hard Rules]]
