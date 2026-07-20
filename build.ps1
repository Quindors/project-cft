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

# --- 2. Stage the focus-dashboard (local mode) -----------------------------
# Preferred: a pre-staged .\dashboard (build it in WSL: `VITE_DATA_SOURCE=local
# npm run build`, then copy dist -> .\dashboard). This avoids needing Node on
# the Windows PATH. Fallback: build here if npm is available. If neither, the
# bundle is skipped and the tray uses a hosted DASHBOARD_URL.
$DASH_BRANCH = 'main'
if (Test-Path '.\dashboard\index.html') {
    Write-Host "[dash] using pre-staged .\dashboard (built elsewhere)"
} else {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
    if ($npm) {
        if (-not (Test-Path '.\.dashboard-src')) {
            Write-Host "[dash] cloning focus-dashboard ($DASH_BRANCH)..."
            git clone --depth 1 --branch $DASH_BRANCH https://github.com/Quindors/focus-dashboard.git .dashboard-src
        }
        Push-Location .dashboard-src
        Write-Host "[dash] npm ci + build (local mode)..."
        & $npm.Source ci --no-audit --no-fund
        $env:VITE_DATA_SOURCE = 'local'
        & $npm.Source run build
        Pop-Location
        if (Test-Path '.\.dashboard-src\dist\index.html') {
            Copy-Item .\.dashboard-src\dist .\dashboard -Recurse -Force
            Write-Host "[dash] staged local-mode dashboard into .\dashboard"
        } else {
            Write-Host "[dash] WARN: dashboard build produced no dist - bundle skipped"
        }
    } else {
        Write-Host "[dash] no pre-staged .\dashboard and no npm - skipping bundle."
        Write-Host "       Tray will use DASHBOARD_URL from .env instead."
    }
}

# --- 3. PyInstaller --------------------------------------------------------
# PyInstaller wipes dist\cft, which would delete the user's .env (API key,
# DASHBOARD_URL). Preserve it across the rebuild.
$envBackup = $null
if (Test-Path '.\dist\cft\.env') {
    $envBackup = Get-Content '.\dist\cft\.env' -Raw
    Write-Host "[env ] preserving existing dist\cft\.env"
}

# A running cft.exe holds locks on dist\cft and makes PyInstaller's cleanup
# fail halfway, leaving a corrupted dist. Stop it first.
$running = Get-Process cft -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "[pyi ] stopping running cft.exe (it locks dist\cft)..."
    $running | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Write-Host "[pyi ] running PyInstaller..."
& $py -m PyInstaller cft.spec --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed (exit $LASTEXITCODE) - dist\cft may be incomplete"
}

# --- 3b. Restore or seed the exe's .env ------------------------------------
if ($envBackup) {
    Set-Content '.\dist\cft\.env' -Value $envBackup -Encoding utf8 -NoNewline
    Write-Host "[env ] restored dist\cft\.env"
} elseif (Test-Path '.\.env') {
    # Fresh build: seed DASHBOARD_URL from the dev .env so the hosted (Vercel)
    # dashboard is CORS-allowed out of the box. The API key is prompted on
    # first launch and appended by the app itself.
    $dash = (Select-String -Path '.\.env' -Pattern '^DASHBOARD_URL=').Line
    if ($dash) {
        Set-Content '.\dist\cft\.env' -Value "$dash`n" -Encoding utf8
        Write-Host "[env ] seeded dist\cft\.env with $dash"
    }
}

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
