On Error Resume Next
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = "D:\Master_Folder\vectorTrade"
exePath = "D:\Master_Folder\vectorTrade\dist\VectorTradeOffline\VectorTradeOffline.exe"
onefilePath = "D:\Master_Folder\vectorTrade\dist\VectorTradeOffline.exe"
pythonwPath = "C:\Users\bared\AppData\Local\Programs\Python\Python311\pythonw.exe"
trayScript = "D:\Master_Folder\vectorTrade\offline\tray_app.py"

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
