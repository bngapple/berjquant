#!/usr/bin/env python3
"""
Dev runner — starts both the FastAPI backend and Vite frontend.
Backend: uvicorn on port 8000
Frontend: Vite dev server on port 8080 (proxies /api and /ws to backend)

Usage: python run_dashboard.py
"""

import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(ROOT, "dashboard")


def main():
    procs = []

    try:
        # Start backend
        print("Starting backend on :8000 ...")
        backend = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "server.api:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--reload-dir", "server",
            ],
            cwd=ROOT,
        )
        procs.append(backend)

        # Start frontend
        print("Starting frontend on :8080 ...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host"],
            cwd=DASHBOARD,
        )
        procs.append(frontend)

        print("\n  Dashboard:  http://localhost:8080")
        print("  API:        http://localhost:8000/docs")
        print("  Press Ctrl+C to stop both.\n")

        # Wait for either to exit
        for proc in procs:
            proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for proc in procs:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
