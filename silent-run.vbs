Set WshShell = CreateObject("WScript.Shell")
projectPath = "C:\Users\abl96\Downloads\project cft"
WshShell.CurrentDirectory = projectPath
pythonExe = "C:\Users\abl96\AppData\Local\Microsoft\WindowsApps\python.exe"
WshShell.Run """" & pythonExe & """ """ & projectPath & "\win-log.py""", 0, False
WshShell.Run """" & pythonExe & """ """ & projectPath & "\monitor-sheets.py""", 0, False