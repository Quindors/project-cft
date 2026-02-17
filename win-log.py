#!/usr/bin/env python3
"""
Active window logger (cross-platform)
Output JSONL: {"timestamp":"YYYY-MM-DD HH:MM:SS","key":"WIN:<title>"}

Supports:
  - Windows: Win32 API via ctypes
  - Linux (X11): xdotool
"""

import os, sys, time, json, subprocess, platform
from datetime import datetime

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def daily_log_path(dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"windows_{dt.strftime('%Y-%m-%d')}.jsonl")

# ---------- tunables ----------
POLL_SEC = float(os.getenv("BL_POLL_SEC", "0.7"))
TITLE_MAXLEN = int(os.getenv("BL_TITLE_MAXLEN", "180"))

# ---------- platform detection ----------
PLATFORM = platform.system()  # "Windows", "Linux", "Darwin"

def _is_wsl() -> bool:
    """Detect if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

IS_WSL = PLATFORM == "Linux" and _is_wsl()

# ---------- active window: Windows ----------
def _get_active_window_windows() -> str:
    """Get active window title using Win32 API."""
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value or ""

# ---------- active window: Linux (X11) ----------
def _get_active_window_linux() -> str:
    """Get active window title using xdotool."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        print("[error] xdotool not found. Install it: sudo apt install xdotool", file=sys.stderr)
    except Exception:
        pass
    return ""

# ---------- active window: WSL ----------
def _get_active_window_wsl() -> str:
    """Get active window title from Windows host via PowerShell."""
    try:
        ps_cmd = (
            "Add-Type -TypeDefinition '"
            "using System; using System.Runtime.InteropServices; "
            "public class W { "
            "[DllImport(\"user32.dll\")] public static extern IntPtr GetForegroundWindow(); "
            "[DllImport(\"user32.dll\", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder t, int m); "
            "}"
            "'; "
            "$h=[W]::GetForegroundWindow(); "
            "$b=New-Object System.Text.StringBuilder 256; "
            "[void][W]::GetWindowText($h,$b,256); "
            "$b.ToString()"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        print("[error] powershell.exe not found in WSL PATH", file=sys.stderr)
    except Exception:
        pass
    return ""

# ---------- pick the right function ----------
if PLATFORM == "Windows":
    get_active_window = _get_active_window_windows
elif IS_WSL:
    get_active_window = _get_active_window_wsl
elif PLATFORM == "Linux":
    get_active_window = _get_active_window_linux
else:
    print(f"[error] Unsupported platform: {PLATFORM}", file=sys.stderr)
    sys.exit(1)

# ---------- state ----------
_current_day = datetime.now().date()
_current_path = daily_log_path(datetime.now())
last_win_key = ""

# ---------- utils ----------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _clean(s: str, maxlen: int) -> str:
    s = " ".join((s or "").split())
    return (s[:maxlen - 1] + "…") if (len(s) > maxlen) else s

def _append_record(key_text: str):
    global _current_day, _current_path
    rec = {"timestamp": _now_str(), "key": key_text}
    now = datetime.now()
    if now.date() != _current_day:
        _current_day = now.date()
        _current_path = daily_log_path(now)
    with open(_current_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- main ----------
def main():
    global last_win_key

    mode = "WSL→Windows" if IS_WSL else PLATFORM
    print(f"[info] Active window logger ({mode}) → {LOG_DIR}/windows_YYYY-MM-DD.jsonl")

    try:
        while True:
            title = _clean(get_active_window(), TITLE_MAXLEN)

            if title:
                win_key = f"WIN:{title}"
                if win_key != last_win_key:
                    _append_record(win_key)
                    last_win_key = win_key

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\n[info] Stopped cleanly.")

if __name__ == "__main__":
    main()