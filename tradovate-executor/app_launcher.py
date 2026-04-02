#!/usr/bin/env python3
"""
HTF Executor — Native macOS Desktop Application
Starts the FastAPI backend in a background thread and opens a native WebKit window.
"""

import os
import sys
import socket
import threading
import time
import logging
import shutil

# ---------------------------------------------------------------------------
# Path resolution — dev vs PyInstaller bundle
# ---------------------------------------------------------------------------

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # PyInstaller bundle — store config/logs in a user-writable location
    import pathlib, platform
    _plat = platform.system()
    if _plat == "Windows":
        DATA_DIR = str(pathlib.Path.home() / "AppData" / "Roaming" / "HTFExecutor")
    elif _plat == "Darwin":
        DATA_DIR = str(pathlib.Path.home() / "Library" / "Application Support" / "HTFExecutor")
    else:
        DATA_DIR = str(pathlib.Path.home() / ".htfexecutor")
    os.makedirs(DATA_DIR, exist_ok=True)
    bundled_config = os.path.join(sys._MEIPASS, "config.json")
    data_config = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(bundled_config) and not os.path.exists(data_config):
        shutil.copy2(bundled_config, data_config)
    os.chdir(DATA_DIR)
    sys.path.insert(0, sys._MEIPASS)
elif getattr(sys, "frozen", False):
    # py2app bundle (legacy fallback)
    APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))))
    RESOURCE_DIR = os.path.join(APP_DIR, "Contents", "Resources")
    os.chdir(RESOURCE_DIR)
    sys.path.insert(0, os.getcwd())
else:
    # Development — run from project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0


def start_server():
    """Run uvicorn — use object import so PyInstaller can trace the dependency."""
    import uvicorn
    from server.api import app as fastapi_app
    uvicorn.run(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )


def wait_for_server(timeout: float = 30.0) -> bool:
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
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{URL}/api/engine/stop",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("Engine stopped gracefully")
    except Exception:
        pass


def on_closing():
    logger.info("Window closing — shutting down...")
    stop_engine_gracefully()


def main():
    if port_in_use(PORT):
        logger.error(f"Port {PORT} is already in use. Is another instance running?")
        try:
            import webview
            webview.create_window(
                "HTF Executor — Error",
                html=f"""<html><body style="background:#0d0d0d;color:#e8e8e8;font-family:system-ui;
                display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
                <div style="text-align:center;">
                <h2 style="color:#ef4444;">Port {PORT} Already In Use</h2>
                <p style="color:#6b7280;">Close the other instance and try again.</p>
                </div></body></html>""",
                width=500, height=300,
            )
            webview.start()
        except Exception:
            pass
        sys.exit(1)

    logger.info(f"Starting server on {URL}...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    if not wait_for_server():
        logger.error("Server failed to start within 30 seconds")
        sys.exit(1)

    logger.info("Server ready — opening window")

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
    webview.start(debug=not getattr(sys, "frozen", False))
    logger.info("Exited")


if __name__ == "__main__":
    main()
