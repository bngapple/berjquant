---
date: 2026-03-29
type: reference
tags: [berjquant, strategy, hybrid-v2, parameters]
---

# Strategy Parameters — Hybrid v2

Current production configuration validated in [[14 - Hybrid v2 Finalization and Deployment]].

## RSI Extremes

| Parameter | Value |
|-----------|-------|
| RSI Period | 5 |
| Oversold (long trigger) | < 35 |
| Overbought (short trigger) | > 65 |
| Stop Loss | 10 points |
| Take Profit | 100 points |
| Max Hold | 5 bars (~75 min) |
| Contracts | 3 MNQ |

## IB Breakout

| Parameter | Value |
|-----------|-------|
| IB Window | 9:30 – 10:00 ET |
| Post-IB Window | 10:00 – 15:30 ET |
| Stop Loss | 10 points |
| Take Profit | 120 points |
| Max Hold | 15 bars (~225 min) |
| IB Range Filter | P25-P75 of trailing 50 days |
| Max trades/day | 1 |
| Contracts | 3 MNQ |

## Momentum Bars

| Parameter | Value |
|-----------|-------|
| ATR Multiplier | 1.0× (bar range > ATR(14)) |
| Volume Multiplier | 1.0× (volume > SMA(vol, 20)) |
| Trend Filter | EMA(21) direction agreement |
| Stop Loss | 15 points |
| Take Profit | 100 points |
| Max Hold | 5 bars (~75 min) |
| Contracts | 3 MNQ |

## System-Level

| Parameter | Value |
|-----------|-------|
| Total contracts | 9 MNQ (3 per strategy) |
| Max simultaneous positions | 3 (one per strategy) |
| Session | 9:30 AM – 4:45 PM ET (LucidFlex) |
| EOD flatten | 4:45 PM ET |
| Slippage | 2 ticks per side |
| Daily loss limit | -$3,000 |
| Monthly loss limit | -$4,500 |

## Cost Per Trade (3 contracts)

| Component | Cost |
|-----------|------|
| Commission (RT) | $3.72 |
| Exchange fee (RT) | $1.62 |
| Slippage (2 ticks/side) | $6.00 |
| **Total** | **$11.34** |

## Changes from Default v3

Hybrid v2 differs from original v3 defaults:
- RSI period: 7 → **5** (72% more signals)
- RSI bands: 30/70 → **35/65** (wider)
- RSI SL: 15pts → **10pts** ($74 vs $104 per loss)
- IB SL: 20pts → **10pts**
- MOM ATR: was briefly 1.4 in Hybrid v1, reverted to **1.0** (recovered $1,373/mo)
