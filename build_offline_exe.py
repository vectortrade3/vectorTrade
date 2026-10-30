"""
VectorTrade - Standalone PyInstaller Executable Builder
Compiles the Offline Tray Application + Flask Backend into a native Windows executable:
dist/VectorTradeOffline/VectorTradeOffline.exe

Configured with --noconsole to ensure zero terminal distractions on Windows boot.
Explicitly bundles all C++ dynamic libraries for XGBoost, Scikit-Learn, and metadata
so that no 'Cannot find XGBoost Library in candidate path' errors ever occur.
"""

import sys
import shutil
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

    # Resolve XGBoost dynamic library (xgboost.dll) and version metadata
    xgb_binaries = []
    xgb_datas = []
    xgb_submodules = [
        'xgboost',
        'xgboost._data_utils',
        'xgboost._typing',
        'xgboost.callback',
        'xgboost.collective',
        'xgboost.compat',
        'xgboost.config',
        'xgboost.core',
        'xgboost.data',
        'xgboost.federated',
        'xgboost.libpath',
        'xgboost.objective',
        'xgboost.plotting',
        'xgboost.sklearn',
        'xgboost.tracker',
        'xgboost.training'
    ]

    try:
        import xgboost
        xgb_path = Path(xgboost.__file__).resolve().parent
        xgb_dll = xgb_path / "lib" / "xgboost.dll"
        xgb_ver = xgb_path / "VERSION"
        xgb_typed = xgb_path / "py.typed"

        if xgb_dll.exists():
            xgb_binaries.append((str(xgb_dll), "xgboost/lib"))
            xgb_binaries.append((str(xgb_dll), "."))
            print(f"[FOUND] XGBoost DLL: {xgb_dll}")
        if xgb_ver.exists():
            xgb_datas.append((str(xgb_ver), "xgboost"))
        if xgb_typed.exists():
            xgb_datas.append((str(xgb_typed), "xgboost"))
    except Exception as e:
        print(f"[WARNING] Could not auto-detect XGBoost directory: {e}")

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
        "--hidden-import", "joblib",
        "--hidden-import", "live_predictor_job",
    ]

    # Add XGBoost binaries and data
    for src, dst in xgb_binaries:
        cmd.extend(["--add-binary", f"{src};{dst}"])
    for src, dst in xgb_datas:
        cmd.extend(["--add-data", f"{src};{dst}"])
    for mod in xgb_submodules:
        cmd.extend(["--hidden-import", mod])

    cmd.append(str(TRAY_SCRIPT))

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    print(f"Executing: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=str(BASE_DIR))

    if res.returncode == 0:
        # Post-build verification and backup copy of XGBoost DLL
        out_path = BASE_DIR / "dist" / "VectorTradeOffline" / "VectorTradeOffline.exe"
        if not onefile:
            internal_dir = BASE_DIR / "dist" / "VectorTradeOffline" / "_internal"
            if internal_dir.exists():
                xgb_internal_lib = internal_dir / "xgboost" / "lib"
                xgb_internal_lib.mkdir(parents=True, exist_ok=True)
                if 'xgb_dll' in locals() and xgb_dll.exists():
                    shutil.copy2(xgb_dll, xgb_internal_lib / "xgboost.dll")
                    shutil.copy2(xgb_dll, internal_dir / "xgboost.dll")
                    print(f"[VERIFIED] Copied {xgb_dll.name} to {xgb_internal_lib}")
                if 'xgb_ver' in locals() and xgb_ver.exists():
                    shutil.copy2(xgb_ver, internal_dir / "xgboost" / "VERSION")

        print("\n==================================================================")
        print("  BUILD SUCCESSFUL!")
        print("==================================================================")
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
