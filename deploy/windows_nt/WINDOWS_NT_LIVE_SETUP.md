# Windows NinjaTrader Live Setup

This is the Windows-PC-native live path for the practical single-account prop setup.

Read this file first on the Windows machine.

## Use This Strategy File

Import or copy this exact strategy into NinjaTrader:

- `tradovate-executor/NinjaTrader/HTFSwingV3HybridV2.cs`

The `.txt` companion exists only so the repo keeps the NinjaScript source mirrored:

- `tradovate-executor/NinjaTrader/HTFSwingV3HybridV2.txt`

Do not use `PythonBridge.cs` for this live setup.

## What This Branch Is For

This branch is for:

- native NinjaTrader execution on Windows
- one trade at a time
- no same-direction adds
- no opposite-direction entry while a trade is open
- no Python bridge required for live runtime

This branch is not for:

- canonical multi-book parity
- Mac-hosted live execution
- Python bridge live execution
- bias-gated variants

## Expected Live Behavior

This setup is intentionally the practical single-account approximation.

The strategy will:

- evaluate the existing RSI / IB / MOM logic internally
- queue signals from the completed 15-minute bar
- execute on the next bar open
- allow only one live position at a time
- reject additional entries while any position is open
- flatten at end of day based on the strategy settings

Implementation note:

- `OnePositionAtATime = true` is the default in this branch
- the strategy already processes the completed bar before executing pending signals
- while a position is open, pending entries are cleared instead of adding or reversing

## Windows Install Steps

1. Clone this branch on the Windows PC.
2. Copy `tradovate-executor/NinjaTrader/HTFSwingV3HybridV2.cs` to:

```text
Documents\NinjaTrader 8\bin\Custom\Strategies\HTFSwingV3HybridV2.cs
```

3. Open NinjaTrader.
4. Open the NinjaScript Editor.
5. Compile.

If NinjaTrader reports duplicate strategy definitions, remove older copies and keep only one `HTFSwingV3HybridV2.cs` in `Custom\Strategies`.

## Chart Setup

Attach the strategy to:

- the active MNQ contract
- a `15 Minute` chart
- the live prop or funded account you intend to trade

Expected chart/session assumptions:

- RTH-oriented 15-minute workflow
- regular trading session starting at 9:30 AM ET
- no new entries after the strategy cutoff
- end-of-day flatten at the configured time

## Recommended NinjaTrader Strategy Settings

Use these settings when you add the strategy:

- `Account`: your live prop/funded account
- `Enabled`: checked
- `Calculate`: `On each tick`
- `Start behavior`: `Wait until flat`
- `Exit on session close`: off
- `Time in force`: `GTC`
- `Stop & target submission`: `Per entry execution`
- `Set order quantity`: `Strategy`

Behavior settings for this branch:

- `One Position At A Time`: `true`
- `Persist Risk State`: `true`
- `Use LucidFlex Presets`: `true` unless you intentionally want manual sizing

Preset sizing behavior:

- `25k`: 1 contract per strategy preset
- `50k`: 2 contracts per strategy preset
- `100k`: 2 contracts per strategy preset
- `150k`: 3 contracts per strategy preset

For a single-account prop setup, the important live constraint is the open-position guard, not multi-book parity.

## What To Ignore In This Repo

These files are useful for research or other runtime paths, but not required for this Windows-native live setup:

- `tradovate-executor/NinjaTrader/PythonBridge.cs`
- `tradovate-executor/app.py`
- `tradovate-executor/signal_engine.py`
- backtest and research scripts in the repo root

Those remain in the repo for research and parity work. They are not the primary live runtime for this branch.

## Before You Leave It Running

Check all of these:

- NinjaTrader connection is green
- correct live account is selected on the strategy
- the chart is on the active MNQ contract
- chart timeframe is `15 Minute`
- the strategy is enabled
- the Windows machine will not sleep
- automatic trading permissions in NinjaTrader are enabled

## Caveats

- This is the practical single-account NT-native version, not exact canonical parity.
- Because this is a netted single-account setup, it intentionally does not allow concurrent independent sub-strategy books.
- If you later want exact canonical behavior, that belongs on the Python/canonical path, not this branch.
