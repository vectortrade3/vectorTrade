"""
VectorTrade - Robust Windows Startup Registration Helper
Creates a bulletproof background autostart launcher in the user's Windows Startup folder:
%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\VectorTrade_AutoStart.vbs

When Windows boots or the script is double-clicked:
1. Launches VectorTrade silently into the System Tray with 0 console windows.
2. Serves the control dashboard on http://127.0.0.1:5000.
3. Automatically triggers EOD analysis at 16:00 IST (or on boot if after 16:00).
4. Features full VBScript error-suppression (On Error Resume Next) and Chr(34) quote-safety.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = BASE_DIR / "offline"
TRAY_SCRIPT = OFFLINE_DIR / "tray_app.py"
EXE_PATH = BASE_DIR / "dist" / "VectorTradeOffline" / "VectorTradeOffline.exe"
ONEFILE_EXE = BASE_DIR / "dist" / "VectorTradeOffline.exe"


def register_startup() -> Path:
    startup_dir = Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"))
    if not startup_dir.exists():
        raise FileNotFoundError(f"Startup folder not found at: {startup_dir}")

    startup_file = startup_dir / "VectorTrade_AutoStart.vbs"

    python_exe = Path(sys.executable)
    pythonw_exe = python_exe.parent / "pythonw.exe"
    if not pythonw_exe.exists():
        pythonw_exe = python_exe

    # Construct bulletproof VBScript with On Error Resume Next, FSO file verification, and Chr(34) quotes
    vbs_content = f'''On Error Resume Next
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = "{BASE_DIR}"
exePath = "{EXE_PATH}"
onefilePath = "{ONEFILE_EXE}"
pythonwPath = "{pythonw_exe}"
trayScript = "{TRAY_SCRIPT}"

WshShell.CurrentDirectory = baseDir

If fso.FileExists(exePath) Then
    WshShell.Run Chr(34) & exePath & Chr(34), 0, False
ElseIf fso.FileExists(onefilePath) Then
    WshShell.Run Chr(34) & onefilePath & Chr(34), 0, False
ElseIf fso.FileExists(pythonwPath) And fso.FileExists(trayScript) Then
    WshShell.Run Chr(34) & pythonwPath & Chr(34) & " " & Chr(34) & trayScript & Chr(34), 0, False
Else
    WshShell.Run "pythonw " & Chr(34) & trayScript & Chr(34), 0, False
End If
'''

    with open(startup_file, "w", encoding="utf-8") as f:
        f.write(vbs_content)

    # Also keep a local copy in offline/
    local_vbs = OFFLINE_DIR / "VectorTrade_AutoStart.vbs"
    with open(local_vbs, "w", encoding="utf-8") as f:
        f.write(vbs_content)

    # Also create desktop & project dashboard launcher
    launcher_vbs = BASE_DIR / "Launch_VectorTrade_Dashboard.vbs"
    dashboard_vbs_content = f'''On Error Resume Next
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = "{BASE_DIR}"
exePath = "{EXE_PATH}"
onefilePath = "{ONEFILE_EXE}"
pythonwPath = "{pythonw_exe}"
trayScript = "{TRAY_SCRIPT}"

WshShell.CurrentDirectory = baseDir

If fso.FileExists(exePath) Then
    WshShell.Run Chr(34) & exePath & Chr(34) & " --open-browser", 0, False
ElseIf fso.FileExists(onefilePath) Then
    WshShell.Run Chr(34) & onefilePath & Chr(34) & " --open-browser", 0, False
ElseIf fso.FileExists(pythonwPath) And fso.FileExists(trayScript) Then
    WshShell.Run Chr(34) & pythonwPath & Chr(34) & " " & Chr(34) & trayScript & Chr(34) & " --open-browser", 0, False
Else
    WshShell.Run "pythonw " & Chr(34) & trayScript & Chr(34) & " --open-browser", 0, False
End If
'''
    with open(launcher_vbs, "w", encoding="utf-8") as f:
        f.write(dashboard_vbs_content)

    print("==================================================================")
    print("  VectorTrade Windows Startup Registration Successful!")
    print("==================================================================")
    print(f"Created silent autostart script at:\n  {startup_file}")
    print(f"Interactive dashboard launcher at:\n  {launcher_vbs}")
    print(f"Local backup script at:\n  {local_vbs}")
    print("\nWhen you turn on your PC, VectorTrade will start automatically")
    print("and remain silently in your System Tray without opening any window.")
    print("Double-click Launch_VectorTrade_Dashboard.vbs anytime to open the UI.")
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
