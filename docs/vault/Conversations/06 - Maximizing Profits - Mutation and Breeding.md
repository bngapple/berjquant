---
date: 2026-03-22
type: conversation
tags: [berjquant, genetic-algorithm, breeding, look-ahead-bias, G4-strategy]
url: https://claude.ai/chat/ee0e28e5-fcac-4120-a09c-35cde6f48562
chat_number: 06
---

# Maximizing Profits - Mutation and Breeding

## Summary

Critical bugs discovered in the genetic algorithm pipeline. G4 strategy discovered but later invalidated.

## Key Decisions / Findings

> [!warning] Critical bugs found
> (1) Genetic monoculture — all 57 winners identical. (2) Look-ahead bias — fills on current bar close. (3) UTC/ET timezone broken. (4) MC costs too optimistic.

- 38 fixes documented across BREED_150K_FIX_LIST.md + COMPLETE_FIX_LIST.md
- G4 discovered (RSI+session_levels): consistent, zero negative months, scaled to 9 contracts
- DD-protection bug at generator.py line 338

> [!danger] Hard rule established
> **No trailing stops — ever.** Trailing stops turned $10,913/month into -$3,420/month.

## Deliverables

- BREED_150K_FIX_LIST.md
- COMPLETE_FIX_LIST.md (38 fixes)
- G4 strategy identification

## Open Questions

- G4 durability in Y2 (later proved insufficient)

## Links

- Previous: [[05 - Program Status and Codebase Audit]]
- Next: [[07 - Trading Strategy Optimization Dilemma]]
- Related: [[Key Learnings and Hard Rules]]
