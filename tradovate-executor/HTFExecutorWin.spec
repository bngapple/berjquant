# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HTF Executor — Windows single-file EXE.
Built on a Windows GitHub Actions runner (windows-latest).
"""

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
webview_datas = collect_data_files("webview")

a = Analysis(
    ["app_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("dashboard/dist", "dashboard/dist"),
        ("server", "server"),
        *webview_datas,
    ],
    hiddenimports=[
        "uvicorn", "uvicorn.main", "uvicorn.config",
        "uvicorn.lifespan.on", "uvicorn.lifespan.off",
        "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "fastapi", "fastapi.middleware.cors", "fastapi.staticfiles", "fastapi.responses",
        "starlette", "starlette.middleware.cors", "starlette.staticfiles",
        "starlette.responses", "starlette.routing", "starlette.websockets",
        "pydantic", "websockets", "websockets.legacy", "websockets.legacy.server",
        "websockets.legacy.client", "websockets.connection",
        "cryptography", "cryptography.fernet",
        "cryptography.hazmat.primitives.kdf.pbkdf2",
        "httpx", "httpcore", "numpy", "multipart",
        "webview", "webview.platforms.edgechromium", "webview.platforms.winforms",
        "anyio", "anyio.abc", "anyio._backends._asyncio",
        "sniffio", "h11", "zoneinfo",
        "email.mime.text", "email.mime.multipart", "logging.handlers",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "test", "unittest"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single-file EXE — everything packed in (slower start, simpler distribution)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="HTFExecutor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # No cmd window
    icon=None,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
