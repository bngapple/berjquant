# HTF Swing v3 Hybrid v2 — Forward Tester

## What This Is

A daily forward tester that fetches real MNQ 1-minute bars from Databento, resamples to 15m, and runs all 3 strategies (RSI, IB, MOM) with Hybrid v2 parameters. Same engine as the backtest — exact signal parity, no TradingView single-position distortion.

## Quick Start

```bash
cd /Users/berjourlian/berjquant
python3 live/forward_test.py
```

## Commands

| Command | What it does |
|---------|-------------|
| `python3 live/forward_test.py` | Fetch new bars, show new trades since last run |
| `python3 live/forward_test.py --summary` | Print current state, no fetch |
| `python3 live/forward_test.py --backfill` | Rebuild everything from scratch |

## When to Run

Run **once daily after 6 PM ET** (after market close). Databento has a ~24hr lag on CME data, so today's bars become available tomorrow morning. Running it in the evening gets you yesterday's complete data + any newly available bars.

## What It Does (Step by Step)

1. Loads your API key from `.env`
2. Fetches new MNQ 1m bars from Databento since last run
3. Merges with existing data (`full_2yr.parquet` + previously fetched bars)
4. Resamples to 15m RTH bars (9:30–16:00 ET)
5. Runs all 3 strategies with Hybrid v2 params:
   - **RSI**: period=5, OS=35, OB=65, SL=10pts, TP=100pts, hold=5
   - **IB**: SL=10pts, TP=120pts, hold=15, percentile filter
   - **MOM**: ATR=1.0, vol=1.0, SL=15pts, TP=100pts, hold=5
6. Compares against last run → prints only NEW trades
7. Saves state, updates trade log and daily P&L log

## Output Files

| File | Description |
|------|-------------|
| `live/forward_state.json` | Last fetch date, total trades, cumulative P&L |
| `live/forward_trades.csv` | Every trade (entry/exit time, strategy, P&L, reason) |
| `live/forward_daily.csv` | Daily P&L summary |
| `live/forward_test_bars.parquet` | Cached bars fetched from Databento (in `data/processed/MNQ/1m/`) |

## Cost

~$0.10/day. Your Databento account has a $125 free credit. A month of daily runs costs ~$2-3.

## Config

- Forward test start date: **Feb 5, 2026** (hardcoded in `forward_test.py` as `FT_START`)
- Flatten time: 1645 (LucidFlex session)
- Contracts: 3 per strategy (9 total)
- Slippage: 2 ticks/side

## Automate (Optional)

To run automatically every weekday at 6 PM ET:

```bash
crontab -e
```

Add this line:

```
0 18 * * 1-5 cd /Users/berjourlian/berjquant && /usr/bin/python3 live/forward_test.py >> live/forward_test.log 2>&1
```

Then check `live/forward_test.log` for results.

## Current Results (as of initial backfill)

- **Period**: Feb 5 – Mar 30, 2026 (37 trading days)
- **Trades**: 403 (24.8% WR)
- **Net P&L**: +$31,896
- **Avg/day**: +$862
- **Max DD**: -$1,414
- **Profit Factor**: 2.29
- **Sharpe**: 18.3

## vs TradingView

TV shows ~1/3 of Python results due to single-position mode. This forward tester runs the real 3-strategy parallel engine. Use this for risk assessment, use TV only for visual chart confirmation.
