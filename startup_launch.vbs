' Startup Launcher for Productivity Monitor
' monitor-sheets.py handles window capture internally

Set WshShell = CreateObject("WScript.Shell")

projectPath = "\\wsl.localhost\Ubuntu\home\abl96\projects\project-cft"
WshShell.CurrentDirectory = projectPath

pythonExe = "C:\Users\abl96\AppData\Local\Microsoft\WindowsApps\python.exe"

' Start monitor (captures windows + classifies + uploads)
WshShell.Run """" & pythonExe & """ """ & projectPath & "\monitor-sheets.py""", 0, False

WScript.Quit