---
date: 2026-03-19
type: conversation
tags: [berjquant, project-setup, data-sources, spec]
url: https://claude.ai/chat/51edf409-5e80-43c0-a484-18bcb07189ec
chat_number: 01
---

# Building a Monte Carlo Trading Algorithm

## Summary

Initial requirements-gathering session. Produced full MCQ Engine project specification covering asset selection, timeframes, risk parameters, and a 5-phase development roadmap.

## Key Decisions / Findings

- Asset: MNQ/NQ futures
- Timeframes: tick/1m/5m
- Trading window: 8am-5pm ET
- Hold time: 30s-2hr
- Account: $50K prop (Topstep/Apex/Lucid)
- Success criteria: Sharpe>=2.0, PF>=1.5, DD in limits, MC>=85%
- Hard rules: no overnight, FOMC/CPI/NFP blackouts, kill switch at 80% daily loss

## Deliverables

- MCQ_Engine_Project_Spec.md
- 5-phase roadmap
- Prop firm profiles
- Tech stack selection (Python, pandas, polars, numpy, vectorbt)

## Open Questions

- Free CME intraday data doesn't exist — need paid source
- Data options: Databento ($125 credit), IB API (~$10/mo), FirstRate Data (~$15-30)
- MotiveWave = CSV exports only

## Links

- Previous: —
- Next: [[02 - Monte Carlo Spec Review]]
- Related: [[BERJQUANT_STATE]]
