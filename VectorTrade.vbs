' ==============================================================================
' VectorTrade - Silent Background Time Monitor Script
' Launches time_monitor.py headlessly using pythonw.exe
' Checks time every 2 seconds; triggers daily EOD pipeline at 4:00 PM (16:00 IST).
' WindowStyle: 0 (Invisible / Headless, No Terminal / Command Prompt Window)
' WaitOnReturn: False (Asynchronous Background Daemon Execution)
' ==============================================================================
Option Explicit

Dim WshShell, fso, projectDir, pythonwPath, monitorScript

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = "D:\Master_Folder\vectorTrade"
pythonwPath = "C:\Users\bared\AppData\Local\Programs\Python\Python311\pythonw.exe"
monitorScript = "D:\Master_Folder\vectorTrade\time_monitor.py"

WshShell.CurrentDirectory = projectDir

If fso.FileExists(pythonwPath) And fso.FileExists(monitorScript) Then
    WshShell.Run """" & pythonwPath & """ """ & monitorScript & """", 0, False
ElseIf fso.FileExists(monitorScript) Then
    WshShell.Run "pythonw.exe time_monitor.py", 0, False
End If

Set WshShell = Nothing
Set fso = Nothing
