' Startup Launcher for Productivity Monitor (source-tree / dev mode).
'
' NOTE: After installing the packaged build via cft-setup.exe, the installer
' drops its own Startup-folder shortcut and this script is no longer needed.
' Keep this for running directly from the WSL source tree during development.

Set WshShell = CreateObject("WScript.Shell")

projectPath = "\\wsl.localhost\Ubuntu\home\abl96\projects\project-cft"
WshShell.CurrentDirectory = projectPath

pythonExe = "C:\Users\abl96\AppData\Local\Microsoft\WindowsApps\python.exe"

' Start monitor (captures windows + classifies + uploads)
WshShell.Run """" & pythonExe & """ """ & projectPath & "\run.py""", 0, False

WScript.Quit