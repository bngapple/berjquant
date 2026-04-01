#!/usr/bin/env python3
"""
HTF Executor — Native macOS Desktop Application
Starts the FastAPI backend in a background thread and opens a native WebKit window.
"""

import os
import sys
import signal
import socket
import threading
import time
import logging

# Resolve paths relative to the app bundle or script location
if getattr(sys, "frozen", False):
    # Running inside py2app bundle
    APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))))
    RESOURCE_DIR = os.path.join(APP_DIR, "Contents", "Resources")
    os.chdir(RESOURCE_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

# Add project root to Python path
sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launcher")

HOST = "127.0.0.1"
PORT = 8080
URL = f"http://{HOST}:{PORT}"


def port_in_use(port: int) -> bool:
    """Check if a port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0


def start_server():
    """Run uvicorn in the current thread (called from a daemon thread)."""
    import uvicorn
    uvicorn.run(
        "server.api:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )


def wait_for_server(timeout: float = 30.0) -> bool:
    """Poll until the server responds or timeout."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{URL}/api/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def stop_engine_gracefully():
    """Try to stop the trading engine before exit."""
    import urllib.request
    import json
    try:
        req = urllib.request.Request(
            f"{URL}/api/engine/stop",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("Engine stopped gracefully")
    except Exception:
        pass  # Engine wasn't running or already stopped


def on_closing():
    """Called when the native window is closed."""
    logger.info("Window closing — shutting down...")
    stop_engine_gracefully()


def main():
    # Check if port is already in use
    if port_in_use(PORT):
        logger.error(f"Port {PORT} is already in use. Is another instance running?")
        try:
            import webview
            webview.create_window(
                "HTF Executor — Error",
                html=f"""
                <html>
                <body style="background:#0d0d0d;color:#e8e8e8;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
                <div style="text-align:center;">
                    <h2 style="color:#ef4444;">Port {PORT} Already In Use</h2>
                    <p style="color:#6b7280;">Another instance may be running. Close it and try again.</p>
                </div>
                </body>
                </html>
                """,
                width=500,
                height=300,
            )
            webview.start()
        except Exception:
            pass
        sys.exit(1)

    # Start FastAPI server in a background thread
    logger.info(f"Starting server on {URL}...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    if not wait_for_server():
        logger.error("Server failed to start within 30 seconds")
        sys.exit(1)

    logger.info("Server ready — opening window")

    # Open native macOS window
    import webview
    window = webview.create_window(
        "HTF Executor",
        URL,
        width=1400,
        height=900,
        resizable=True,
        min_size=(1000, 700),
        text_select=True,
    )
    window.events.closing += on_closing

    webview.start(
        debug=not getattr(sys, "frozen", False),  # Debug mode in development only
    )

    # After window closes, exit
    logger.info("Exited")


if __name__ == "__main__":
    main()
