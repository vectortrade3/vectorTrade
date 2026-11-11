' ==============================================================================
' VectorTrade - Silent Background Startup Script
' Launches VectorTrade.exe headlessly in the Windows System Tray on port 30307
' WindowStyle: 0 (Invisible / Headless, No Command Prompt / Terminal Window)
' WaitOnReturn: False (Asynchronous Background Execution)
' ==============================================================================
Option Explicit

Dim WshShell, fso, exePath, projectDir

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

exePath = "D:\Master_Folder\vectorTrade\dist\VectorTrade.exe"
projectDir = "D:\Master_Folder\vectorTrade"

If fso.FileExists(exePath) Then
    ' Set working directory to project root so data_store and models are preserved
    WshShell.CurrentDirectory = projectDir
    ' Run completely hidden without console window (0 = hide window)
    WshShell.Run """" & exePath & """", 0, False
Else
    ' Fallback to pythonw silent execution if exe not present
    WshShell.CurrentDirectory = projectDir
    WshShell.Run "pythonw.exe tray_app.py", 0, False
End If

Set WshShell = Nothing
Set fso = Nothing
