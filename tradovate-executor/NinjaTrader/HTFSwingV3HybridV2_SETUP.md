# HTF Swing v3 Hybrid v2 — NinjaTrader Setup

This is the fully native NinjaTrader version.

- No Mac app is required for live trading.
- No `PythonBridge` is required for live trading.
- All signal logic and order placement run inside NinjaTrader.

## 1. Install NinjaTrader

1. Install NinjaTrader Desktop on Windows.
2. Log in with your own NinjaTrader / Lucid credentials.

## 2. Create the broker connection

1. In NinjaTrader, go to `Tools -> Account Connections`.
2. Click `Add`.
3. Set up the broker/account connection using your broker credentials.
4. You can name the connection anything you want.
5. If you want it to match this setup, name it `QUANT`.
6. Finish setup.
7. Connect it from `Connections -> QUANT` or whatever name you used.

Important:

- The connection should be green when fully connected.
- If it is orange, do not trust it for unattended live trading.

## 3. Confirm the live account exists

1. Open the `Accounts` tab.
2. Make sure your real live account is visible there.
3. Make sure the account row is healthy and the connection is green.

## 4. Install the strategy

1. Copy this file:
   `HTFSwingV3HybridV2.cs`
2. Put it here:
   `Documents\NinjaTrader 8\bin\Custom\Strategies\HTFSwingV3HybridV2.cs`
3. Open NinjaTrader.
4. Open the NinjaScript Editor.
5. Compile.

Important:

- Put it in `Strategies`, not `AddOns`.
- If you get duplicate-definition errors, remove old copies and keep only one file in `Strategies`.

## 5. Open the chart

1. Open a chart for the active MNQ contract.
2. Set the timeframe to `15 Minute`.

Current example:

- `MNQ 06-26`

When contracts roll, update the chart to the active contract.

## 6. Add the strategy to the chart

1. Right-click the chart.
2. Open `Strategies`.
3. Add `HTFSwingV3HybridV2`.
4. Enable it.

## 7. Use these strategy settings

### Setup

- `Account =` your real live account
- `Calculate = On each tick`
- `Bars required to trade = 0`
- `Start behavior = Wait until flat`
- `Enabled = checked`

### Behavior

- `One Position At A Time = false`
- `Persist Risk State = true`
- `Use LucidFlex Presets = true`
- `LucidFlex Account Size = 25`

### Order handling

- `Entries per direction = 3`
- `Entry handling = Unique entries`
- `Exit on session close = off`
- `Stop & target submission = Per entry execution`

### Order properties

- `Set order quantity = Strategy`
- `Time in force = GTC`

## 8. What the presets do

- `25k = 1 contract per strategy, $1000 monthly halt`
- `50k = 2 contracts per strategy, $2000 monthly halt`
- `100k = 2 contracts per strategy, $3000 monthly halt`
- `150k = 3 contracts per strategy, $4500 monthly halt`

If `Use LucidFlex Presets = true`, the preset values override the manual contract and monthly-limit fields.

## 9. Trading session

The strategy is set to:

- start trading at `9:30 AM ET`
- stop taking new entries after `4:30 PM ET`
- flatten all positions at `4:45 PM ET`

## 10. Before leaving it live

Make sure:

- the broker connection is green
- the live account is visible in `Accounts`
- the chart is updating
- the strategy is enabled
- the Windows PC or VM will not sleep

## 11. Notes

- You do not need RSI, IB, or Momentum indicators on the chart.
- The strategy calculates everything internally.
- The connection name `QUANT` is only a label.
- The important part is that the connection is live and the strategy account is set to the real account.
