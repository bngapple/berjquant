---
date: 2026-03-29
type: reference
tags: [berjquant, rules, principles]
---

# Key Learnings and Hard Rules

Non-negotiable principles extracted from all 14 conversations.

## Hard Rules

1. **No trailing stops — ever.** $10,913/mo → -$3,420/mo ([[06 - Maximizing Profits - Mutation and Breeding]]). Trailing stops destroy the asymmetric R:R that makes the system work.

2. **15-minute bars over 1-minute bars.** Intrabar ordering makes 1m outcomes depend on unknowable sequencing. 15m stop = ~28% of bar range vs 150% on 1m.

3. **Fixed SL/TP only.** TP optimization confirmed 100/120pts optimal. Only 4.3% of losers within 2pts of TP ([[14 - Hybrid v2 Finalization and Deployment]]).

4. **Portfolio/parameter selection on training data only.** Y2 OOS is the only reliable benchmark. Pre-2020 data structurally different.

5. **TradingView "After order is filled" must stay unchecked.** Inflates WR from 27% to 46%.

6. **TV single-position = ~1/3 of Python backtest.** Platform constraint, not bug.

7. **Regime dependency known and accepted.** High-vol profitable, low-vol modest loss ([[14 - Hybrid v2 Finalization and Deployment]]).

8. **London pre-market filter rejected.** r=0.71 but overfitting risk ([[14 - Hybrid v2 Finalization and Deployment]]).

9. **Edge in first trading hour.** 69% of total profit ([[13 - Full Audit and Forward Test Pipeline]]).

10. **Strategy correlation near-zero.** Avg pairwise -0.046 = genuine diversification ([[13 - Full Audit and Forward Test Pipeline]]).

11. **Low WR / high R:R.** 26-28% WR, 3.3-3.8 R:R. Psychologically hard, mathematically validated.

12. **Experiments in separate files.** Never contaminate validated systems ([[07 - Trading Strategy Optimization Dilemma]]).

13. **Simplicity over complexity.** MCP→CLI. Portfolio→single strategy scaled. The institutional 5-layer architecture was proposed but simpler HTF Swing v3 won ([[10 - Algorithmic Trading Strategy Redesign]]).
