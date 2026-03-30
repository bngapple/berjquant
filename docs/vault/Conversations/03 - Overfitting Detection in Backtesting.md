---
date: 2026-03-19
type: conversation
tags: [berjquant, overfitting, validation]
url: https://claude.ai/chat/e1c91328-d8f3-4cfc-af3a-358b614e63d9
chat_number: 03
---

# Overfitting Detection in Backtesting

## Summary

Engine correctly rejecting narrow-window strategies. The 30-day to 2-year validation gap is where retail algos die.

## Key Decisions / Findings

- Engine's overfit detection is working as intended
- 30→2yr validation gap identified as the critical failure zone for retail algo traders
- Three paths forward: scale contracts, combine uncorrelated signals, or calibrate targets for single-signal reality

## Deliverables

- Diagnostic analysis of overfitting behavior

## Open Questions

- Which path to take: scale, combine, or calibrate?

## Links

- Previous: [[02 - Monte Carlo Spec Review]]
- Next: [[04 - Topstep vs Lucid Trading Comparison]]
- Related: [[Key Learnings and Hard Rules]]
