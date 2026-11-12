' ==============================================================================
' VectorTrade - Silent Background Startup Script
' Runs latest VectorTrade.exe headlessly in the Windows System Tray on port 30307
' WindowStyle: 0 (Invisible / Headless, No Command Prompt / Terminal Window)
' WaitOnReturn: False (Asynchronous Background Execution)
' ==============================================================================
Option Explicit

Dim WshShell, fso, projectDir, exePath, pythonwPath, trayScript

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = "D:\Master_Folder\vectorTrade"
exePath = "D:\Master_Folder\vectorTrade\dist\VectorTrade.exe"
pythonwPath = "C:\Users\bared\AppData\Local\Programs\Python\Python311\pythonw.exe"
trayScript = "D:\Master_Folder\vectorTrade\tray_app.py"

WshShell.CurrentDirectory = projectDir

If fso.FileExists(exePath) Then
    WshShell.Run """" & exePath & """", 0, False
ElseIf fso.FileExists(pythonwPath) And fso.FileExists(trayScript) Then
    WshShell.Run """" & pythonwPath & """ """ & trayScript & """", 0, False
Else
    WshShell.Run "pythonw.exe tray_app.py", 0, False
End If

Set WshShell = Nothing
Set fso = Nothing
