"""
VectorTrade - Standalone PyInstaller Executable Builder
Compiles the Offline Tray Application + Flask Backend into a native Windows executable:
dist/VectorTradeOffline/VectorTradeOffline.exe

Configured with --noconsole to ensure zero terminal distractions on Windows boot.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_DIR = BASE_DIR / "offline"
TRAY_SCRIPT = OFFLINE_DIR / "tray_app.py"


def build_exe(onefile: bool = False):
    print("==================================================================")
    print("  VectorTrade: Building Windows Executable via PyInstaller...")
    print("==================================================================")

    templates_dir = OFFLINE_DIR / "templates"
    models_dir = OFFLINE_DIR / "models"
    stock_names = BASE_DIR / "stock_names.json"
    final_features = BASE_DIR / "final_features.txt"
    global_features = BASE_DIR / "global_features.txt"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "VectorTradeOffline",
        "-y",           # Overwrite output directory without asking
        "--noconsole",  # Hides command prompt window on boot
        "--clean",
        "--add-data", f"{templates_dir};offline/templates",
        "--add-data", f"{models_dir};offline/models",
        "--add-data", f"{stock_names};.",
        "--add-data", f"{final_features};.",
        "--add-data", f"{global_features};.",
        "--hidden-import", "pystray._win32",
        "--hidden-import", "PIL",
        "--hidden-import", "flask",
        "--hidden-import", "jinja2",
        "--hidden-import", "pymongo",
        "--hidden-import", "mongomock",
        "--hidden-import", "dns",
        "--hidden-import", "dns.resolver",
        "--hidden-import", "pydantic",
        "--hidden-import", "pydantic_settings",
        "--hidden-import", "dotenv",
        "--hidden-import", "pytz",
        "--hidden-import", "requests",
        "--hidden-import", "urllib3",
        "--hidden-import", "sklearn",
        "--hidden-import", "xgboost",
        "--hidden-import", "joblib",
        "--hidden-import", "live_predictor_job",
        str(TRAY_SCRIPT)
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    print(f"Executing: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=str(BASE_DIR))

    if res.returncode == 0:
        print("\n==================================================================")
        print("  BUILD SUCCESSFUL!")
        print("==================================================================")
        out_path = BASE_DIR / "dist" / "VectorTradeOffline" / "VectorTradeOffline.exe"
        if onefile:
            out_path = BASE_DIR / "dist" / "VectorTradeOffline.exe"
        print(f"Executable generated at:\n  {out_path}")
        print("\nTo auto-start on Windows boot silently in System Tray, run:")
        print("  python offline/register_startup.py")
        print("==================================================================")
    else:
        print(f"\nBuild failed with exit code: {res.returncode}")


if __name__ == "__main__":
    onefile_mode = "--onefile" in sys.argv
    build_exe(onefile=onefile_mode)
