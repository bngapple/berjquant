---
date: 2026-03-29
type: reference
tags: [berjquant, prop-firm, lucid-trading, topstep]
---

# Prop Firm Comparison

LucidFlex 150K won every round in the Python backtest competition ([[13 - Full Audit and Forward Test Pipeline]]).

## Head-to-Head

| Feature | LucidFlex 150K | Topstep 150K |
|---------|---------------|--------------|
| Daily loss limit | None | -$3,000 |
| Max drawdown | -$4,500 EOD trailing | -$4,500 trailing |
| Funded consistency rule | None | None |
| Cost | One-time ~$300 | $149/month |
| Payout speed | 15 minutes | 1-2 business days |
| Payout split | 90/10 | 90/10 |
| Session hours | 9:30 AM – 4:45 PM ET | 9:30 AM – 3:45 PM ET |
| Eval target | $9,000 | $9,000 |
| Eval consistency | 50% max single day | None |

## Why LucidFlex Won

1. **No daily loss limit** — system worst day -$1,252, irrelevant for LucidFlex but close to Topstep's -$3,000 threshold
2. **No funded consistency rule** — no artificial caps on profitable days
3. **One-time ~$300 vs $149/month** — LucidFlex pays for itself in 2 months
4. **15-minute payouts** — immediate access to profits
5. **Extra trading hour** — 4:45 PM vs 3:45 PM ET worth $901-$1,638/month in backtest
6. **EOD-only trailing drawdown** — intraday drawdowns don't count against the limit

## Decision

LucidFlex 150K selected. Target purchase: May 2026.

See also: [[04 - Topstep vs Lucid Trading Comparison]]
