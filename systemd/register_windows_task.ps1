<#
.SYNOPSIS
    Registers VectorTrade EOD Offline Daemon as a scheduled Windows Task.
.DESCRIPTION
    Runs daily at 16:05 IST (Asia/Kolkata).
    Configured with battery-awareness, wake-to-run option, and automatic termination.
    Consumes 0 MB RAM / 0% CPU outside execution.
#>

$TaskName = "VectorTrade_EOD_Daemon"
$WorkingDir = "D:\Master_Folder\vectorTrade"
$PythonExe = (Get-Command python).Source

Write-Host "Registering $TaskName..." -ForegroundColor Cyan
Write-Host "Working Directory: $WorkingDir"
Write-Host "Python Executable: $PythonExe"

# Define the action
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "-m offline.daily_eod_daemon" -WorkingDirectory $WorkingDir

# Define daily trigger at 16:05 IST
$Trigger = New-ScheduledTaskTrigger -Daily -At "4:05 PM"

# Define task settings: Battery-aware, WakeToRun, stop after 45 minutes
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45) `
    -StartWhenAvailable

# Register or update task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "VectorTrade Autonomous EOD Offline Pipeline" -Force | Out-Null

Write-Host "Task $TaskName registered successfully to run daily at 16:05 IST." -ForegroundColor Green
