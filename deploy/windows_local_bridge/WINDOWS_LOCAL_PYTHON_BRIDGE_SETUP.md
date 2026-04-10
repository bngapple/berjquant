# Windows-Local Python Bridge Setup

This is the deployment path where the full live runtime stays on one Windows PC:

- NinjaTrader runs on the Windows PC
- the Python executor runs on the same Windows PC
- `signal_engine.py` remains the canonical live strategy logic
- NinjaTrader is only the execution endpoint and market-data bridge

Read this file first on the Windows machine.

## What Runs On The Windows PC

These all run locally on the same Windows box:

- NinjaTrader Desktop
- the cloned repo
- a Python virtual environment
- the live Python executor
- the NinjaTrader TCP bridge strategy

There is no Mac runtime dependency in this deployment path.

## Canonical Live Logic

Canonical strategy logic stays in Python:

- `tradovate-executor/signal_engine.py`

Live execution / market-data plumbing on Python side:

- `tradovate-executor/app.py`
- `tradovate-executor/market_data.py`
- `tradovate-executor/ninjatrader_bridge.py`

NinjaTrader-side bridge strategy:

- `tradovate-executor/NinjaTrader/PythonBridge.cs`

Runtime config:

- `tradovate-executor/config.json`

Useful connectivity check:

- `tradovate-executor/scripts/test_nt_connection.py`

## Live Execution Policy

This deployment keeps canonical signal generation in Python, but applies the chosen single-account live execution constraint in the live executor:

- one trade at a time
- no same-direction adds
- no opposite-direction entry while a trade is open
- no bias gate

That means:

- Python still generates RSI / IB / MOM signals canonically
- the live acceptance layer in `app.py` blocks new entries whenever the account already has an open live direction

## The Exact NinjaTrader File To Use

Use this exact NinjaTrader file for this deployment:

- `tradovate-executor/NinjaTrader/PythonBridge.cs`

Do not use these as the main live strategy in this deployment:

- `tradovate-executor/NinjaTrader/HTFSwingV3HybridV2.cs`
- `tradovate-executor/NinjaTrader/HTFSwingV3HybridV2.txt`

Those are for the native-NinjaTrader path, not this Python-canonical bridge path.

## Windows Setup

### 1. Clone the repo on Windows

Clone this branch on the Windows PC.

### 2. Install Python

Install Python 3.12 or newer on Windows, and make sure the launcher works:

```bat
py --version
```

### 3. Create a virtual environment

Open Command Prompt or PowerShell in:

```text
...\berjquant\tradovate-executor
```

Then run:

```bat
py -3 -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 4. Prepare config.json

Copy this example file:

- `deploy/windows_local_bridge/config.windows_local_nt_only.example.json`

to:

- `tradovate-executor/config.json`

Then edit:

- the NinjaTrader account name key so it exactly matches the NinjaTrader Accounts tab
- `symbol` and `ninjatrader.symbol` to the active contract

For same-PC Windows deployment, leave:

- `host = 127.0.0.1`
- `port = 6000`

Important:

- This deployment can run in NT-only mode, so `accounts` stays empty.
- `app.py` will synthesize the master session from the first `ninjatrader.accounts` key.

## What To Load In NinjaTrader

### Strategy file

Copy:

- `tradovate-executor/NinjaTrader/PythonBridge.cs`

to:

```text
Documents\NinjaTrader 8\bin\Custom\Strategies\PythonBridge.cs
```

Then compile it in NinjaTrader.

### Chart setup

Attach `PythonBridge` to:

- the active MNQ contract
- a `15 Minute` chart
- the same live prop/funded account named in `config.json`

### Strategy parameters in NinjaTrader

Set or confirm:

- `TcpPort = 6000`
- `EOD Flatten Time = 164500`
- chart timeframe = `15 Minute`
- the strategy is enabled
- automated trading is enabled in NinjaTrader

The Python side and the NinjaTrader side must use the same port.

## What To Run In Python

From:

```text
...\berjquant\tradovate-executor
```

the exact live entrypoint is:

```bat
venv\Scripts\python.exe app.py --live
```

Before that, it is smart to test the TCP bridge first:

```bat
venv\Scripts\python.exe scripts\test_nt_connection.py
```

There is also a helper launcher in:

- `deploy/windows_local_bridge/run_live_nt_bridge.bat`

It runs the TCP connectivity test first, then launches the live app.

## Recommended Startup Order

1. Start Windows.
2. Open NinjaTrader.
3. Open the `15 Minute` MNQ chart.
4. Enable the `PythonBridge` strategy on that chart.
5. Confirm the account and port are correct.
6. Open Command Prompt in `tradovate-executor`.
7. Run `venv\Scripts\python.exe scripts\test_nt_connection.py`
8. Run `venv\Scripts\python.exe app.py --live`

## What Not To Run

Do not do any of these for this deployment:

- do not use the Mac as part of the live runtime
- do not run `HTFSwingV3HybridV2.cs` as the main live strategy
- do not run both `PythonBridge` and the native merged NT strategy on the same instrument/account
- do not run both bridge mode and some second conflicting NT automation against the same account

## Important Notes

- The Python executor is the live source of truth for signal generation.
- NinjaTrader is the execution endpoint and market-data bridge.
- This path is still a practical single-account approximation, not exact independent multi-book parity.
- If the contract rolls, update both `symbol` and the NT chart instrument.

