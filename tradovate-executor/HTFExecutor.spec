# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HTF Executor macOS .app bundle.
Run: pyinstaller HTFExecutor.spec --clean --noconfirm
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect pywebview bundled JS files
webview_datas = collect_data_files("webview")

a = Analysis(
    ["app_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # Built React dashboard
        ("dashboard/dist", "dashboard/dist"),
        # Default runtime config for first launch
        ("config.json", "."),
        # Server Python package
        ("server", "server"),
        # pywebview JS injection files
        *webview_datas,
    ],
    hiddenimports=[
        # uvicorn internals (not auto-detected via string import)
        "uvicorn",
        "uvicorn.main",
        "uvicorn.config",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.protocols.websockets.wsproto_impl",
        # fastapi / starlette
        "fastapi",
        "fastapi.middleware.cors",
        "fastapi.staticfiles",
        "fastapi.responses",
        "starlette",
        "starlette.middleware.cors",
        "starlette.staticfiles",
        "starlette.responses",
        "starlette.routing",
        "starlette.websockets",
        # pydantic
        "pydantic",
        "pydantic.v1",
        # websockets
        "websockets",
        "websockets.legacy",
        "websockets.legacy.server",
        "websockets.legacy.client",
        "websockets.connection",
        # cryptography / fernet
        "cryptography",
        "cryptography.fernet",
        "cryptography.hazmat.primitives.kdf.pbkdf2",
        "cryptography.hazmat.primitives.ciphers",
        # httpx
        "httpx",
        "httpcore",
        # numpy
        "numpy",
        # pywebview macOS backend
        "webview",
        "webview.platforms.cocoa",
        "webview.util",
        # multipart / form data
        "multipart",
        "python_multipart",
        # zoneinfo (timezone support)
        "zoneinfo",
        "zoneinfo._tzdata",
        # h11 (HTTP/1.1)
        "h11",
        # anyio
        "anyio",
        "anyio.abc",
        "anyio._backends._asyncio",
        # sniffio
        "sniffio",
        # other stdlib
        "email.mime.text",
        "email.mime.multipart",
        "logging.handlers",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "test", "unittest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HTFExecutor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,       # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=True, # macOS: allow file-open events
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HTFExecutor",
)

app = BUNDLE(
    coll,
    name="HTFExecutor.app",
    icon=None,
    bundle_identifier="com.berjquant.htfexecutor",
    info_plist={
        "CFBundleDisplayName": "HTF Executor",
        "CFBundleName": "HTFExecutor",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "NSAppleScriptEnabled": False,
        "LSMinimumSystemVersion": "12.0",
        "LSUIElement": False,
    },
)
