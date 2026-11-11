@echo off
setlocal
echo ========================================================
echo  VectorTrade Host Registration: http://vectortrade:30307
echo ========================================================

:: Check for Administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Requesting Administrator privileges to update hosts file...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    exit /b
)

set HOSTS_FILE=%WINDIR%\System32\drivers\etc\hosts

:: Check if already registered
findstr /i "vectortrade" "%HOSTS_FILE%" >nul
if %errorlevel% equ 0 (
    echo [OK] 'vectortrade' is already registered in %HOSTS_FILE%!
) else (
    echo.>> "%HOSTS_FILE%"
    echo # VectorTrade Desktop Engine loopback mapping >> "%HOSTS_FILE%"
    echo 127.0.0.1       vectortrade >> "%HOSTS_FILE%"
    echo [SUCCESS] Successfully registered 'vectortrade' in %HOSTS_FILE%!
)

:: Flush DNS to make changes take effect immediately
ipconfig /flushdns >nul
echo [OK] Windows DNS cache flushed.
echo.
echo You can now navigate to:
echo   http://vectortrade:30307
echo.
timeout /t 5
