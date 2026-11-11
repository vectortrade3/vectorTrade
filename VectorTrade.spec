# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
project_root = Path.cwd()

datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "offline" / "templates"), "offline/templates"),
]

hiddenimports = [
    "pystray",
    "pystray._win32",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "simple_websocket",
    "flask_sock",
    "pymongo",
    "dns",
    "dns.resolver",
    "pytz",
    "ctypes",
    "ctypes.wintypes",
    "xgboost",
    "sklearn",
    "joblib",
    "offline.app",
    "offline.scheduler",
    "offline.json_manager",
    "offline.daily_analyzer",
    "offline.mongo_syncer",
    "offline.ma50_service",
    "strategies.manager",
    "strategies.base_strategy",
    "strategies.global_regime_strategy",
    "strategies.ma50_strategy",
    "online.services.risk_calculator",
    "online.services.megabull_service",
    "online.services.orders_service",
    "online.db",
]

a = Analysis(
    ["tray_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VectorTrade",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "assets" / "vectortrade_icon.ico"),
)
