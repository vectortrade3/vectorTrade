On Error Resume Next
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = "D:\Master_Folder\vectorTrade"
appScript = "D:\Master_Folder\vectorTrade\offline\app.py"
pythonwPath = "C:\Users\bared\AppData\Local\Programs\Python\Python311\pythonw.exe"

WshShell.CurrentDirectory = baseDir

' Launch Flask Offline App silently in background (0 console, low RAM/CPU)
If fso.FileExists(pythonwPath) And fso.FileExists(appScript) Then
    WshShell.Run Chr(34) & pythonwPath & Chr(34) & " " & Chr(34) & appScript & Chr(34), 0, False
Else
    WshShell.Run "pythonw " & Chr(34) & appScript & Chr(34), 0, False
End If

' Allow server to bind port and launch default web browser
WScript.Sleep 1500
WshShell.Run "http://127.0.0.1:5000", 1, False
