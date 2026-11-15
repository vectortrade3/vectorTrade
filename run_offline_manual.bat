@echo off
title VectorTrade Offline Dashboard (Manual Mode)
cd /d "D:\Master_Folder\vectorTrade"

echo ==============================================================================
echo   VectorTrade Offline Web Dashboard - Manual Launch Mode
echo ==============================================================================
echo   Starting local Flask engine on http://localhost:30307...
echo   Press Ctrl+C in this terminal window to stop the server at any time.
echo ==============================================================================

rem Open browser to the orders view after 2 seconds
start "" "http://localhost:30307/orders"

rem Start the offline web application
python offline\app.py --port 30307

pause
