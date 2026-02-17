# monitor/window_capture.py
"""
Cross-platform active window title capture.

Supports:
  - Windows: Win32 API via ctypes
  - WSL: PowerShell call to Windows host
  - Linux (X11): xdotool
"""

from __future__ import annotations

import platform
import subprocess
import sys


# ---------- platform detection ----------

PLATFORM = platform.system()


def _is_wsl() -> bool:
    """Detect if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


IS_WSL = PLATFORM == "Linux" and _is_wsl()


# ---------- Windows ----------

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


# ---------- WSL ----------

_PS_CMD = (
    "Add-Type -TypeDefinition '"
    "using System; using System.Runtime.InteropServices; "
    "public class W { "
    '[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow(); '
    '[DllImport("user32.dll", CharSet=CharSet.Unicode)] '
    "public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder t, int m); "
    "}"
    "'; "
    "$h=[W]::GetForegroundWindow(); "
    "$b=New-Object System.Text.StringBuilder 256; "
    "[void][W]::GetWindowText($h,$b,256); "
    "$b.ToString()"
)


def _get_active_window_wsl() -> str:
    """Get active window title from Windows host via PowerShell."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _PS_CMD],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return ""


# ---------- Linux (X11) ----------

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
        print("[error] xdotool not found. Install: sudo apt install xdotool", file=sys.stderr)
    except Exception:
        pass
    return ""


# ---------- public API ----------

if PLATFORM == "Windows":
    _capture_fn = _get_active_window_windows
    _mode = "Windows"
elif IS_WSL:
    _capture_fn = _get_active_window_wsl
    _mode = "WSL→Windows"
elif PLATFORM == "Linux":
    _capture_fn = _get_active_window_linux
    _mode = "Linux/X11"
else:
    _capture_fn = lambda: ""
    _mode = f"Unsupported ({PLATFORM})"
    print(f"[warn] Window capture not supported on {PLATFORM}", file=sys.stderr)


def get_active_window() -> str:
    """Return the current active window title, or empty string if unavailable."""
    title = _capture_fn()
    # Normalize whitespace and trim
    return " ".join((title or "").split())


def get_mode() -> str:
    """Return a string describing the capture mode (for logging)."""
    return _mode