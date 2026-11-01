@echo off
setlocal EnableDelayedExpansion
title Port Killer Utility
color 0A

echo ============================================================
echo               Port Killer Utility
echo ============================================================
echo.

set "PORT="
set /p "PORT=Enter Port Number [Default: 5000]: "
if "%PORT%"=="" set "PORT=5000"

echo.
echo Scanning for processes listening on port %PORT%...
echo.

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr /c:":%PORT% "') do (
    set "PID=%%a"
    if not "!PID!"=="0" (
        set "FOUND=1"
        echo [*] Terminating process PID !PID! on port %PORT%...
        taskkill /F /PID !PID! >nul 2>&1
        if !errorlevel! equ 0 (
            echo     [+] PID !PID! successfully killed.
        ) else (
            echo     [-] Could not kill PID !PID!. Run this script as Administrator.
        )
    )
)

if "!FOUND!"=="0" (
    echo [i] No active process was found on port %PORT%. The port is already free.
) else (
    echo.
    echo [SUCCESS] Port %PORT% has been freed!
)

echo.
echo ============================================================
pause
