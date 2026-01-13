' Startup Launcher for Productivity Monitor
' This runs all components silently in the background

Set WshShell = CreateObject("WScript.Shell")

' === EDIT THIS PATH TO MATCH YOUR SYSTEM ===
projectPath = "C:\Users\abl96\Downloads\project cft"
' ============================================

' Change to project directory
WshShell.CurrentDirectory = projectPath

' Get Python executable path
pythonExe = "C:\Users\abl96\AppData\Local\Microsoft\WindowsApps\python.exe"

' Start window logger (win-log.py)
WshShell.Run """" & pythonExe & """ """ & projectPath & "\win-log.py""", 0, False

' Start monitor (monitor-sheets.py)
WshShell.Run """" & pythonExe & """ """ & projectPath & "\monitor-sheets.py""", 0, False

' Optional: Start keystroke logger if you want it
' WshShell.Run """" & pythonExe & """ """ & projectPath & "\key-log.py""", 0, False

' Optional: Start mouse logger
' WshShell.Run """" & pythonExe & """ """ & projectPath & "\mouse-log.py""", 0, False

WScript.Quit