"""
VectorTrade - Windows Startup Registration Helper
Creates a background autostart launcher in the user's Windows Startup folder:
%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup

When Windows boots, VectorTrade automatically starts and docks silently into
the System Tray without any visible command prompt or distracting windows.
"""

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAY_SCRIPT = BASE_DIR / "offline" / "tray_app.py"
EXE_PATH = BASE_DIR / "dist" / "VectorTradeOffline" / "VectorTradeOffline.exe"
ONEFILE_EXE = BASE_DIR / "dist" / "VectorTradeOffline.exe"


def register_startup() -> Path:
    startup_dir = Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"))
    if not startup_dir.exists():
        raise FileNotFoundError(f"Startup folder not found at: {startup_dir}")

    # Determine best launch command:
    # 1. Compiled EXE if built
    # 2. pythonw.exe (windowless python) running tray_app.py
    startup_file = startup_dir / "VectorTrade_AutoStart.vbs"

    if EXE_PATH.exists():
        target = f'"{EXE_PATH}"'
        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{BASE_DIR}"
WshShell.Run {target}, 0, False
'''
    elif ONEFILE_EXE.exists():
        target = f'"{ONEFILE_EXE}"'
        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{BASE_DIR}"
WshShell.Run {target}, 0, False
'''
    else:
        # Use pythonw (windowless python - 0 console window)
        python_exe = Path(sys.executable)
        pythonw_exe = python_exe.parent / "pythonw.exe"
        if not pythonw_exe.exists():
            pythonw_exe = python_exe

        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{BASE_DIR}"
WshShell.Run """{pythonw_exe}"" ""{TRAY_SCRIPT}""", 0, False
'''

    with open(startup_file, "w", encoding="utf-8") as f:
        f.write(vbs_content)

    print("==================================================================")
    print("  VectorTrade Windows Startup Registration Successful!")
    print("==================================================================")
    print(f"Created silent autostart script at:\n  {startup_file}")
    print("\nWhen you turn on your PC, VectorTrade will start automatically")
    print("and remain silently in your System Tray without opening any window.")
    print("==================================================================")
    return startup_file


def unregister_startup() -> None:
    startup_dir = Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"))
    startup_file = startup_dir / "VectorTrade_AutoStart.vbs"
    if startup_file.exists():
        startup_file.unlink()
        print("VectorTrade removed from Windows Startup.")
    else:
        print("VectorTrade was not registered in Windows Startup.")


if __name__ == "__main__":
    if "--remove" in sys.argv or "--unregister" in sys.argv:
        unregister_startup()
    else:
        register_startup()
