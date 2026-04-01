"""
py2app setup script for HTF Executor macOS application.

Usage:
    python setup_app.py py2app
    # Output: dist/HTF Executor.app
"""

from setuptools import setup

APP = ["app_launcher.py"]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/icon.icns",
    "plist": {
        "CFBundleName": "HTF Executor",
        "CFBundleDisplayName": "HTF Executor",
        "CFBundleIdentifier": "com.htfswing.executor",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
    "includes": [
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "starlette.routing",
        "starlette.middleware",
        "starlette.middleware.cors",
        "websockets",
        "httpx",
        "pydantic",
        "cryptography",
        "webview",
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
        "httpcore",
        "h11",
        "sniffio",
        "certifi",
        "idna",
    ],
    "packages": [
        "server",
    ],
    "resources": [
        "dashboard/dist",
        "logs",
        "config.py",
        "app.py",
        "auth_manager.py",
        "websocket_client.py",
        "market_data.py",
        "signal_engine.py",
        "order_executor.py",
        "copy_engine.py",
        "risk_manager.py",
        "trade_logger.py",
        "position_sync.py",
        "indicators.py",
    ],
    "site_packages": True,
}

setup(
    app=APP,
    name="HTF Executor",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
