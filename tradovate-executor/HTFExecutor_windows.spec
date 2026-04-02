# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HTF Executor Windows .exe
Run on Windows: pyinstaller HTFExecutor_windows.spec --clean --noconfirm
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
        # uvicorn
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
        # cryptography
        "cryptography",
        "cryptography.fernet",
        "cryptography.hazmat.primitives.kdf.pbkdf2",
        "cryptography.hazmat.primitives.ciphers",
        # httpx
        "httpx",
        "httpcore",
        # numpy
        "numpy",
        # pywebview Windows backend (Edge WebView2)
        "webview",
        "webview.platforms.edgechromium",
        "webview.platforms.winforms",
        "webview.util",
        # multipart
        "multipart",
        "python_multipart",
        # zoneinfo
        "zoneinfo",
        "zoneinfo._tzdata",
        # h11
        "h11",
        # anyio
        "anyio",
        "anyio.abc",
        "anyio._backends._asyncio",
        # sniffio
        "sniffio",
        # stdlib
        "email.mime.text",
        "email.mime.multipart",
        "logging.handlers",
        # Windows-specific
        "clr",
        "System",
        "System.Windows.Forms",
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
    console=False,        # No console window
    disable_windowed_traceback=False,
    argv_emulation=False, # Windows — not needed
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,            # Replace with "assets/icon.ico" if you add one
    version_file=None,
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
