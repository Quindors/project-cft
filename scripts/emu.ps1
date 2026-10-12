<#
Drive the Windows-side Android emulator (AVD cft_test) for testing the cft-mobile app.

  .\scripts\emu.ps1 boot            resume the AVD from its quick-boot snapshot (visible window, host GPU)
  .\scripts\emu.ps1 boot -Headless  same, no window
  .\scripts\emu.ps1 install <apk>   adb install -r (an x86_64 APK from the "emulator"/"preview-emu" EAS profiles)
  .\scripts\emu.ps1 dev             open the dev client against Metro running in WSL (start Metro first:
                                    preview "cft-mobile-web-start", or `npx expo start` in ~/projects/cft-mobile)
  .\scripts\emu.ps1 launch          cold-launch the installed app and wait until the summary is rendered
  .\scripts\emu.ps1 shot [path]     screenshot (default .\emu-shot.png)
  .\scripts\emu.ps1 stop            adb emu kill (saves the snapshot the next boot resumes from)

Gotchas (see memory note android-emulator): always -dns-server or the guest cannot resolve names;
never -no-snapshot-load (80 s cold boot vs 20 s resume); adb reverse does NOT reach Metro because the
Windows relay for 8081 is IPv6-only, so the dev client uses the WSL NAT IP from `wsl hostname -I`.
#>
param(
  [Parameter(Position=0)][ValidateSet('boot','install','dev','launch','shot','stop','status')][string]$Cmd = 'status',
  [Parameter(Position=1)][string]$Arg,
  [switch]$Headless
)
$ErrorActionPreference = 'Stop'
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
$adb = "$sdk\platform-tools\adb.exe"
$pkg = 'com.quindors89.cftmobile'

function Booted { ((& $adb -e shell getprop sys.boot_completed 2>$null) -join '').Trim() -eq '1' }

function Wait-Boot([int]$sec = 150) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while (-not (Booted)) { if ($sw.Elapsed.TotalSeconds -gt $sec) { throw "emulator did not boot in $sec s" }; Start-Sleep 1 }
  "booted in $([int]$sw.Elapsed.TotalSeconds) s"
}

function Wait-Rendered([int]$sec = 40) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $sec) {
    $ui = & $adb -e exec-out uiautomator dump /dev/tty 2>$null
    if ($ui -match 'monitored today|Session active|reach Supabase') { return "rendered in $($sw.ElapsedMilliseconds) ms" }
    # dev client first-run: welcome sheet (Continue) then the full dev menu (BACK closes it)
    if ($ui -match 'text="Continue"') { & $adb -e shell input tap 540 2107 | Out-Null; continue }
    if ($ui -match 'Toggle performance monitor') { & $adb -e shell input keyevent BACK | Out-Null; continue }
    Start-Sleep -Milliseconds 400
  }
  "not rendered after $sec s (focus: $((& $adb -e shell dumpsys window | Select-String mCurrentFocus) -replace '\s+',' '))"
}

switch ($Cmd) {
  'status' {
    & $adb devices
    if (Booted) { "booted; focus: $((& $adb -e shell dumpsys window | Select-String mCurrentFocus) -replace '\s+',' ')" } else { 'not running' }
  }
  'boot' {
    if (Booted) { 'already running'; break }
    $a = @('-avd','cft_test','-dns-server','8.8.8.8,1.1.1.1','-no-boot-anim','-no-audio')
    if ($Headless) { $a += @('-no-window','-gpu','swiftshader_indirect') } else { $a += @('-gpu','host','-cores','4') }
    Start-Process -FilePath "$sdk\emulator\emulator.exe" -ArgumentList $a -RedirectStandardOutput "$env:TEMP\cft-emulator.log" -RedirectStandardError "$env:TEMP\cft-emulator.err" | Out-Null
    Wait-Boot
  }
  'install' {
    if (-not $Arg) { throw 'usage: emu.ps1 install <path-to-apk>' }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $adb -e install -r $Arg
    "installed in $([int]$sw.Elapsed.TotalSeconds) s: $([int]((Get-Item $Arg).Length / 1MB)) MB"
  }
  'dev' {
    $ip = ((wsl -d Ubuntu hostname -I) -split '\s+')[0]
    try { $null = Invoke-WebRequest -Uri "http://${ip}:8081/status" -TimeoutSec 5 -UseBasicParsing } catch { throw "Metro not reachable at http://${ip}:8081 - start it first (preview cft-mobile-web-start)" }
    $url = [uri]::EscapeDataString("http://${ip}:8081")
    & $adb -e shell am force-stop $pkg
    & $adb -e shell am start -a android.intent.action.VIEW -d "cftmobile://expo-development-client/?url=$url" | Out-Null
    "dev client -> http://${ip}:8081"
    Wait-Rendered 60
  }
  'launch' {
    & $adb -e shell am force-stop $pkg
    & $adb -e shell am start -n "$pkg/.MainActivity" | Out-Null
    Wait-Rendered
  }
  'shot' {
    $out = if ($Arg) { $Arg } else { Join-Path (Get-Location) 'emu-shot.png' }
    $out = [IO.Path]::GetFullPath($out)
    # never redirect adb binary output through PowerShell (it re-encodes as text)
    & $adb -e shell screencap -p /sdcard/emu-shot.png; & $adb -e pull /sdcard/emu-shot.png $out | Out-Null
    "saved $out"
  }
  'stop' { & $adb -e emu kill; 'stopping (snapshot save takes ~20 s)' }
}
