# BerjQuant — Map of Content

> Quantitative algorithmic trading system targeting consistent monthly income through prop firm accounts trading MNQ (Micro Nasdaq 100) futures.

## Current System

**HTF Swing v3 Hybrid v2** — 3 strategies, 9 contracts, 15-minute MNQ bars
- RSI Extremes: period=5, OS=35, OB=65, SL=10pts, TP=100pts, hold=5 bars
- IB Breakout: SL=10pts, TP=120pts, hold=15 bars
- Momentum Bars: ATR multiplier=1.0

**Key Metrics (Y2 OOS):** ~$7,140/month | Sharpe 3.63 | 99%+ MC pass | 13/13 months profitable

---

## Active TODO

→ [[Next Steps]]

---

## Conversations (Chronological)

| # | Date | Title | Key Outcome |
|---|------|-------|-------------|
| 1 | Mar 19 | [[01 - Building a Monte Carlo Trading Algorithm]] | Project spec, 5-phase roadmap |
| 2 | Mar 19 | [[02 - Monte Carlo Spec Review]] | 7 spec fixes identified |
| 3 | Mar 19 | [[03 - Overfitting Detection in Backtesting]] | 30-day→2-year validation gap |
| 4 | Mar 19 | [[04 - Topstep vs Lucid Trading Comparison]] | Rules deep-dive |
| 5 | Mar 19 | [[05 - Program Status and Codebase Audit]] | 19K line audit, Supabase |
| 6 | Mar 22 | [[06 - Maximizing Profits - Mutation and Breeding]] | Look-ahead bias found, G4 discovered |
| 7 | Mar 23 | [[07 - Trading Strategy Optimization Dilemma]] | 17+ failed runs, HFT rejected |
| 8 | Mar 23 | [[08 - Current State of the Quant]] | Portfolio Breed Y2 failure |
| 9 | Mar 24 | [[09 - Polymarket Assessment]] | Futures path reaffirmed |
| 10 | Mar 24 | [[10 - Algorithmic Trading Strategy Redesign]] | Institutional pivot proposed (not implemented) |
| 11 | Mar 25 | [[11 - Profit Allocation and Tax Planning]] | Wealth stacking, LLC roadmap |
| 12 | Mar 26 | [[12 - Open Source Backtesting and Strategy Review]] | External HTF v3 review |
| 13 | Mar 27 | [[13 - Full Audit and Forward Test Pipeline]] | 14,426 trade audit clean, LucidFlex selected |
| 14 | Mar 27 | [[14 - Hybrid v2 Finalization and Deployment]] | Hybrid v2 validated, deployment planned |

---

## Reference

- [[BERJQUANT_STATE]] — Full system documentation
- [[Key Learnings and Hard Rules]] — Non-negotiable principles
- [[System Architecture]] — Engine, signals, validation
- [[Prop Firm Comparison]] — LucidFlex vs Topstep
- [[Strategy Parameters]] — Current hybrid v2 config
- [[Performance Results]] — All validated metrics

---

## Quick Commands

| Command | What it does |
|---------|-------------|
| `vault-add conv "Title"` | New conversation note (auto-numbered, linked) |
| `vault-add daily` | Today's daily note |
| `vault-add research "Title"` | New research note |
| `vault-sync` | Push vault changes to GitHub repo |
