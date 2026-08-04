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

# Run a native exe without 'Stop' treating its stderr as a fatal error.
#
# Windows PowerShell 5.1 wraps every stderr line from a native command in an
# ErrorRecord, so under $ErrorActionPreference='Stop' anything that merely
# WRITES to stderr kills the build — pip's "a new release is available"
# notice and PyInstaller's own progress output both do. Exit code is the only
# honest success signal for these, so check that instead.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe,
          [Parameter(ValueFromRemainingArguments)][string[]]$Arguments,
          [string]$What = 'command')
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try   { & $Exe @Arguments }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

# --- 1. Venv ---------------------------------------------------------------
if (-not (Test-Path '.\build-venv\Scripts\python.exe')) {
    Write-Host "[venv] creating build-venv..."
    Invoke-Native 'python' '-m' 'venv' 'build-venv' -What 'venv creation'
}
$py = '.\build-venv\Scripts\python.exe'

Write-Host "[pip ] installing build deps..."
Invoke-Native $py '-m' 'pip' 'install' '--quiet' '--upgrade' 'pip' -What 'pip self-upgrade'
Invoke-Native $py '-m' 'pip' 'install' '--quiet' '-r' 'requirements-build.txt' -What 'pip install'

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
            # git reports progress on stderr; npm reports warnings there too.
            Invoke-Native 'git' 'clone' '--depth' '1' '--branch' $DASH_BRANCH `
                'https://github.com/Quindors/focus-dashboard.git' '.dashboard-src' -What 'git clone'
        }
        Push-Location .dashboard-src
        try {
            Write-Host "[dash] npm ci + build (local mode)..."
            Invoke-Native $npm.Source 'ci' '--no-audit' '--no-fund' -What 'npm ci'
            $env:VITE_DATA_SOURCE = 'local'
            Invoke-Native $npm.Source 'run' 'build' -What 'npm run build'
        } finally { Pop-Location }
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
try {
    Invoke-Native $py '-m' 'PyInstaller' 'cft.spec' '--noconfirm' -What 'PyInstaller'
} catch {
    throw "$_ - dist\cft may be incomplete"
}

# --- 3b. Restore or seed the exe's .env ------------------------------------
# Write without a BOM: PowerShell 5.1's `-Encoding utf8` prepends one, which
# corrupts the first line's variable name when the app parses the .env.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
if ($envBackup) {
    # Strip a BOM a previous buggy restore may have baked into the backup.
    $envBackup = $envBackup.TrimStart([char]0xFEFF)
    [System.IO.File]::WriteAllText((Resolve-Path '.\dist\cft').ProviderPath + '\.env', $envBackup, $utf8NoBom)
    Write-Host "[env ] restored dist\cft\.env"
} elseif (Test-Path '.\.env') {
    # Fresh build: seed DASHBOARD_URL from the dev .env so the hosted (Vercel)
    # dashboard is CORS-allowed out of the box. The API key is prompted on
    # first launch and appended by the app itself.
    $dash = (Select-String -Path '.\.env' -Pattern '^DASHBOARD_URL=').Line
    if ($dash) {
        [System.IO.File]::WriteAllText((Resolve-Path '.\dist\cft').ProviderPath + '\.env', "$dash`n", $utf8NoBom)
        Write-Host "[env ] seeded dist\cft\.env with $dash"
    }
}

# --- 3. Inno Setup (optional) ----------------------------------------------
$iscc = Get-Command 'iscc.exe' -ErrorAction SilentlyContinue
if (-not $iscc) { $iscc = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue }
if ($iscc) {
    Write-Host "[inno] compiling installer..."
    Invoke-Native $iscc.Source 'installer.iss' -What 'Inno Setup'
    Write-Host "[done] installer at .\dist\cft-setup.exe"
} else {
    Write-Host "[warn] Inno Setup (iscc.exe) not on PATH - skipping installer step."
    Write-Host "       Onedir build is at .\dist\cft\cft.exe"
}

Write-Host "[ok  ] build complete."
