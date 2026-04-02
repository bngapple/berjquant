# NinjaTrader Bridge Setup

Routes order execution through NinjaTrader 8.1 on a Windows VM instead of the Tradovate REST API.
Used when LucidFlex prop-firm accounts don't have Tradovate API credentials for programmatic order placement.

```
macOS (Python engine)   ←──── market data ────   Tradovate WS
macOS (Python engine)   ──── TCP :6000 ────────►  Windows VM (NinjaTrader 8.1)
                                                        │ NinjaScript strategy
                                                        │ manages SL/TP brackets
                                                        └─── LucidFlex account
```

---

## 1 — VMware Fusion Network Setup

Configure a **host-only** network adapter so the Mac and the Windows VM can communicate:

1. VMware Fusion → Virtual Machine → Network Adapter → **Host-Only**
2. In Windows VM: note the IP shown in `ipconfig` for the host-only adapter (e.g. `192.168.56.101`)
3. On macOS: verify connectivity — `ping 192.168.56.101`

Update `config.json` with the Windows VM IP:

```json
{
  "nt": {
    "host": "192.168.56.101",
    "port": 6000
  }
}
```

For local development (both engine and NT on the same machine) use `"host": "127.0.0.1"`.

---

## 2 — Install NinjaTrader 8.1

1. Download from [ninjatrader.com](https://ninjatrader.com) and install on the Windows VM
2. Log in with your NinjaTrader / LucidFlex credentials
3. Open a chart for **MNQM6** (or whichever contract is configured in `config.json`)

---

## 3 — Install PythonBridge Strategy

1. Copy `NinjaTrader/PythonBridge.cs` from this repo to:
   ```
   C:\Users\<YourName>\Documents\NinjaTrader 8\bin\Custom\Strategies\PythonBridge.cs
   ```

2. In NinjaTrader: **Tools → Edit NinjaScript → Strategies → PythonBridge**

3. Click **Compile** (F5). Resolve any errors (typically missing `using` directives or version mismatches).

4. If compilation succeeds, NinjaTrader shows "Compiled successfully" in the output pane.

> **Newtonsoft.Json**: NinjaTrader 8 bundles Newtonsoft.Json. If the compiler can't find it,
> add a reference: right-click the project → Add Reference → Browse to
> `C:\Program Files\NinjaTrader 8\bin\Newtonsoft.Json.dll`

---

## 4 — Add Strategy to Chart

1. Open a **1-minute MNQM6** chart (File → New Chart → MNQM6, 1 Min)
2. Right-click chart → **Strategies → Add Strategy → PythonBridge**
3. Configure parameters:
   - **TCP Port**: `6000` (must match `config.json → nt.port`)
   - **EOD Flatten Time**: `164500` (= 4:45 PM ET, format HHMMSS)
4. Click **OK** — strategy loads and starts listening on port 6000
5. Confirm in the NinjaTrader **Strategy** panel: status shows **Active**
6. The Output window shows: `[PB] Listening on port 6000`

---

## 5 — Configure the Python Engine

Add the `nt` section to `config.json`:

```json
{
  "environment": "live",
  "symbol": "MNQM6",
  "nt": {
    "host": "192.168.56.101",
    "port": 6000
  },
  "accounts": [
    {
      "name": "LFE150k",
      "username": "LTT07T22GBH",
      "password": "enc:...",
      "is_master": true,
      ...
    }
  ]
}
```

> **Note**: The `accounts` section is still required for market data authentication.
> The Python engine uses the Tradovate WebSocket for price ticks; only *order execution*
> routes through NinjaTrader. If market data credentials are also unavailable, see
> [Alternative Market Data](#alternative-market-data) below.

---

## 6 — Run the Engine

```bash
cd /path/to/tradovate-executor
source venv/bin/activate
python app.py
```

Expected startup sequence:
```
[INFO] NT mode: routing execution through NinjaTrader at 192.168.56.101:6000
[INFO] Market data WebSocket connected
[INFO] Subscribed to: MNQM6
[INFO] [NT] Connected to NinjaTrader at 192.168.56.101:6000
[INFO] EXECUTOR RUNNING — waiting for signals
```

---

## 7 — Verify the Connection

### From Python side

```python
# Quick connectivity test (run from project root)
import asyncio
from ninjatrader_bridge import NinjaTraderBridge
from config import AppConfig, NTConfig
from unittest.mock import MagicMock

async def test():
    cfg = AppConfig()
    cfg.nt = NTConfig(host="192.168.56.101", port=6000)
    session = MagicMock(); session.name = "test"
    bridge = NinjaTraderBridge(cfg, session, "192.168.56.101", 6000)
    await bridge.connect()
    await asyncio.sleep(1)
    print("Connected:", bridge.connected)
    await bridge.shutdown()

asyncio.run(test())
```

### From NinjaTrader side

The Output window (View → Output Window) shows:
```
[PB] Python connected from 192.168.56.101
```

---

## 8 — EOD Flatten

At `164500` (4:45 PM ET), the `PythonBridge` strategy automatically:
1. Calls `ExitLong()` and `ExitShort()` for all open positions
2. Sends `{"type": "exit", "exit_type": "EOD", ...}` to Python for each open strategy
3. Python logs the exit, updates P&L, and marks the strategy flat

This mirrors the Tradovate path's `flatten_time` in `SessionConfig`.

---

## 9 — Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ConnectionRefusedError` connecting to NT | Check NT is running and strategy is Active |
| `[PB] Listening on port 6000` not shown | Recompile strategy; check Output window for errors |
| Fill timeout — record stays WORKING | NT received command but didn't execute (check NT permissions, account connection) |
| EOD flatten not firing | Confirm NinjaTrader server time zone is ET, or adjust `EodFlattenTime` |
| `Newtonsoft.Json` compile error | Add reference to NT's bundled DLL (see Step 3) |
| Windows Firewall blocking port 6000 | Add inbound rule: `netsh advfirewall firewall add rule name="NTPythonBridge" dir=in action=allow protocol=TCP localport=6000` |

---

## Alternative Market Data

If Tradovate WebSocket is also unavailable, market data can come from NinjaTrader itself
(future feature — not yet implemented). In the interim:

1. Use a separate **demo Tradovate account** for market data only (free tier)
2. Authenticate that account in `config.json` as the master account (no trading)
3. Keep `accounts` pointing to the demo account for WS auth; NT handles actual trades

---

## Protocol Reference

All messages are newline-delimited JSON (`\n` terminated).

### Python → NinjaTrader

| Message | Fields | Description |
|---------|--------|-------------|
| `ENTRY` | `id`, `strategy`, `side`, `qty`, `sl_pts`, `tp_pts` | Place market entry with ATM bracket |
| `FLATTEN` | `id`, `strategy` | Close one strategy's position |
| `FLATTEN_ALL` | `id` | Close all open positions |
| `PING` | — | Keepalive |

### NinjaTrader → Python

| Message | Fields | Description |
|---------|--------|-------------|
| `fill` | `id`, `strategy`, `side`, `qty`, `fill_price`, `sl_price`, `tp_price` | Entry fill confirmed |
| `exit` | `strategy`, `exit_type`, `fill_price`, `qty` | SL/TP/EOD/Command exit |
| `ack` | `id` | Flatten command acknowledged |
| `pong` | — | Keepalive response |
| `error` | `id`, `message` | Command failed |
