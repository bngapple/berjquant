#!/usr/bin/env python3
"""
Standalone NinjaTrader connection test.
Reads ninjatrader config from config.json and tests TCP connectivity to each account.

No engine imports — uses only asyncio + json + os + sys from stdlib.

Usage:
    python scripts/test_nt_connection.py

Exit codes:
    0 — all accounts reachable
    1 — one or more accounts unreachable
"""

import asyncio
import json
import os
import sys

CONNECT_TIMEOUT = 5.0  # Seconds to wait per connection attempt


def load_nt_accounts() -> dict:
    """Read ninjatrader accounts from config.json at the project root."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.json",
    )
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        data = json.load(f)

    nt_raw = data.get("ninjatrader") or data.get("nt")
    if not nt_raw:
        print("ERROR: No 'ninjatrader' key found in config.json")
        print("Add a 'ninjatrader' section with your account connection details.")
        sys.exit(1)

    if "accounts" in nt_raw:
        accounts = {
            name: acct
            for name, acct in nt_raw["accounts"].items()
            if not name.startswith("_")
        }
        if not accounts:
            print("ERROR: ninjatrader.accounts is empty — add your account connection details")
            sys.exit(1)
        return accounts

    # Old single-host "nt" format (backward compat)
    return {"default": {"host": nt_raw.get("host", "127.0.0.1"), "port": nt_raw.get("port", 6000)}}


async def test_account(name: str, host: str, port: int) -> bool:
    """Try opening a TCP connection to one NinjaTrader account. Returns True on success."""
    print(f"  [{name}] {host}:{port} ... ", end="", flush=True)

    # Warn if host looks like a placeholder
    if host.startswith("REPLACE_"):
        print("SKIPPED — placeholder host not yet configured")
        return False

    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=CONNECT_TIMEOUT,
        )
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:
            pass
        print("SUCCESS")
        return True

    except asyncio.TimeoutError:
        print(f"TIMEOUT (>{CONNECT_TIMEOUT}s)")
        print(f"         Check: VM running? PythonBridge strategy active on a chart?")
        return False

    except ConnectionRefusedError:
        print("REFUSED — nothing listening on that port")
        print(f"         Check: PythonBridge loaded in NinjaTrader? Port matches config?")
        return False

    except OSError as e:
        print(f"ERROR — {e}")
        print(f"         Check: is {host!r} the VM's 'ipconfig' IPv4 address?")
        return False


async def main():
    print("NinjaTrader Connection Test")
    print("=" * 50)

    accounts = load_nt_accounts()
    print(f"Found {len(accounts)} account(s) in config:\n")

    results = {}
    for name, acct in accounts.items():
        host = acct.get("host", "127.0.0.1") if isinstance(acct, dict) else acct.host
        port = acct.get("port", 6000) if isinstance(acct, dict) else acct.port
        results[name] = await test_account(name, host, port)

    print("\n" + "=" * 50)
    all_ok = all(results.values())
    passed = sum(1 for ok in results.values() if ok)

    if all_ok:
        print(f"All {len(results)} account(s) reachable.")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"Result: {passed}/{len(results)} reachable — FAILED: {', '.join(failed)}")
        print()
        print("Troubleshooting checklist:")
        print("  1. Windows VM is running (check VMware Fusion / Parallels)")
        print("  2. NinjaTrader is open with the PythonBridge strategy active on a chart")
        print("  3. Windows Firewall has an inbound rule allowing TCP on the configured port")
        print("  4. Host IP in config.json matches the VM's ipconfig IPv4 address")
        print("     To find it: open Command Prompt in the VM and run:")
        print("       ipconfig | findstr IPv4")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
