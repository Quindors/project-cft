# build.ps1 - Build script for project-cft (hosted-API edition).
#
# Run from Windows PowerShell (not WSL):
#   .\build.ps1
#
# Steps:
#   1. Ensure a Windows venv exists at .\build-venv\
#   2. pip install -r requirements-build.txt
#   3. pyinstaller cft.spec  ->  .\dist\cft\
#   4. (optional) iscc installer.iss  ->  .\dist\cft-setup.exe
#
# The app talks to a hosted LLM (OpenRouter or OpenAI). On first launch it
# prompts for an API key and saves it to the .env next to the exe.

$ErrorActionPreference = 'Stop'

# --- 1. Venv ---------------------------------------------------------------
if (-not (Test-Path '.\build-venv\Scripts\python.exe')) {
    Write-Host "[venv] creating build-venv..."
    python -m venv build-venv
}
$py = '.\build-venv\Scripts\python.exe'

Write-Host "[pip ] installing build deps..."
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r requirements-build.txt

# --- 2. PyInstaller --------------------------------------------------------
Write-Host "[pyi ] running PyInstaller..."
& $py -m PyInstaller cft.spec --noconfirm

# --- 3. Inno Setup (optional) ----------------------------------------------
$iscc = Get-Command 'iscc.exe' -ErrorAction SilentlyContinue
if (-not $iscc) { $iscc = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue }
if ($iscc) {
    Write-Host "[inno] compiling installer..."
    & $iscc.Source 'installer.iss'
    Write-Host "[done] installer at .\dist\cft-setup.exe"
} else {
    Write-Host "[warn] Inno Setup (iscc.exe) not on PATH - skipping installer step."
    Write-Host "       Onedir build is at .\dist\cft\cft.exe"
}

Write-Host "[ok  ] build complete."
