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

# --- 2. Build & stage the focus-dashboard (local mode) ---------------------
# The bundled dashboard reads from the exe's local /api (SQLite), so it needs
# only npm — no Supabase env. When npm is absent the bundle is skipped and the
# tray falls back to a hosted DASHBOARD_URL.
$DASH_BRANCH = 'integrate-dashboard-hosting'   # branch with the local data source
$npm = Get-Command npm -ErrorAction SilentlyContinue
if ($npm) {
    if (-not (Test-Path '.\.dashboard-src')) {
        Write-Host "[dash] cloning focus-dashboard ($DASH_BRANCH)..."
        git clone --depth 1 --branch $DASH_BRANCH https://github.com/Quindors/focus-dashboard.git .dashboard-src
    }
    Push-Location .dashboard-src
    Write-Host "[dash] npm ci + build (local mode)..."
    & npm ci --no-audit --no-fund
    $env:VITE_DATA_SOURCE = 'local'
    & npm run build
    Pop-Location
    Remove-Item .\dashboard -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item .\.dashboard-src\dist .\dashboard -Recurse
    Write-Host "[dash] staged local-mode dashboard into .\dashboard"
} else {
    Write-Host "[dash] npm not on PATH - skipping dashboard bundle."
    Write-Host "       Tray will use DASHBOARD_URL from .env instead."
}

# --- 3. PyInstaller --------------------------------------------------------
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
