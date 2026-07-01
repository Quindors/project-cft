; installer.iss — Inno Setup script for project-cft
;
; Compile with: iscc.exe installer.iss
; (Inno Setup 6+ required.)
;
; Wraps dist\cft\ (PyInstaller onedir output) into a single self-extracting
; installer: dist\cft-setup.exe. Installs to %LOCALAPPDATA%\cft, drops a
; Start Menu shortcut, and a Startup-folder shortcut that launches the app
; minimized at boot.

#define MyAppName     "cft"
#define MyAppVersion  "0.1.0"
#define MyAppExeName  "cft.exe"

[Setup]
AppId={{4F2A4E9C-5D58-4E9B-9D9F-7E4F4C2A8C11}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=cft-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "dist\cft\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";       Filename: "{app}\{#MyAppExeName}"
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Comment: "Launch cft at login"; WorkingDir: "{app}"; \
    Flags: runminimized

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; \
    Flags: nowait postinstall skipifsilent runminimized

[UninstallDelete]
; Remove the bundle dir on uninstall, but leave the user's .env if they put one next to the exe.
Type: filesandordirs; Name: "{app}\_internal"
