# monitor/idle.py
"""How long since the user last touched the machine.

The monitor's ledger rests on one assumption: that the window in front of the
user is what they are doing. An empty chair breaks it — a productive window
left open keeps logging "work" all night, and a Beeminder goal fed by those
hours becomes a machine you can cheat by going to bed. This module supplies
the missing fact: seconds since the last keyboard or mouse input, so the
runner can stop counting when nobody is there.

Platform support mirrors window_capture.py:
  - Windows: GetLastInputInfo via ctypes. Microseconds, no subprocess.
  - WSL:     PowerShell call to the Windows host — opt-in via
             CFT_IDLE_WSL_PROBE=1, because spawning powershell.exe on every
             poll is heavy and live monitoring from WSL is a dev convenience,
             not the deployment. Probes are cached for a few seconds.
  - Linux:   xprintidle, if installed.
  - Anywhere else: None.

seconds_since_input() returning None means "no idle signal on this platform",
and callers must fail open (treat the user as present). Guessing "away" on a
missing signal would pause logging forever.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

from monitor.window_capture import PLATFORM, IS_WSL


def _ticks_to_seconds(now_ticks: int, last_input_ticks: int) -> float:
    """Tick difference in seconds, honouring 32-bit wraparound.

    GetTickCount and LASTINPUTINFO.dwTime are both DWORDs that wrap every
    49.7 days; modulo arithmetic keeps the difference right across the wrap.
    """
    return ((now_ticks - last_input_ticks) & 0xFFFFFFFF) / 1000.0


# ---------- Windows ----------

def _make_windows_probe():
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    # Default restype is a SIGNED int; past 24.8 days of uptime the tick
    # count would come back negative and the wrap math would be wrong.
    kernel32.GetTickCount.restype = ctypes.c_uint32

    def probe() -> Optional[float]:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not user32.GetLastInputInfo(ctypes.byref(lii)):
            return None
        return _ticks_to_seconds(kernel32.GetTickCount(), lii.dwTime)

    return probe


# ---------- WSL (PowerShell call to the Windows host) ----------

_PS_IDLE_CMD = (
    "Add-Type -TypeDefinition '"
    "using System; using System.Runtime.InteropServices; "
    "public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; } "
    "public class I { "
    '[DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii); '
    '[DllImport("kernel32.dll")] public static extern uint GetTickCount(); '
    "}"
    "'; "
    "$l = New-Object LASTINPUTINFO; "
    "$l.cbSize = [System.Runtime.InteropServices.Marshal]::SizeOf($l); "
    "[void][I]::GetLastInputInfo([ref]$l); "
    "(([I]::GetTickCount() - $l.dwTime) -band 0xFFFFFFFF)"
)

# One powershell.exe spawn costs the better part of a second, so the WSL
# probe answers from this cache between real calls. Idle thresholds are
# minutes; a reading a few seconds old is the same reading.
_WSL_CACHE_SEC = 15.0
_wsl_cached_at: float = 0.0
_wsl_cached_val: Optional[float] = None


def _probe_wsl() -> Optional[float]:
    global _wsl_cached_at, _wsl_cached_val
    now = time.monotonic()
    if _wsl_cached_val is not None and (now - _wsl_cached_at) < _WSL_CACHE_SEC:
        # Idle keeps growing while nobody types; ride the clock between probes.
        return _wsl_cached_val + (now - _wsl_cached_at)
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _PS_IDLE_CMD],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        val = int(result.stdout.strip()) / 1000.0
    except Exception:
        return None
    _wsl_cached_at, _wsl_cached_val = now, val
    return val


# ---------- Linux (X11) ----------

_xprintidle_missing = False


def _probe_linux() -> Optional[float]:
    global _xprintidle_missing
    if _xprintidle_missing:
        return None
    try:
        result = subprocess.run(
            ["xprintidle"], capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1000.0
    except FileNotFoundError:
        # Remember, or a 2s poll loop would print this forever.
        _xprintidle_missing = True
        print("[idle] xprintidle not found — idle detection off. "
              "Install: sudo apt install xprintidle")
    except Exception:
        pass
    return None


# ---------- public API ----------

if PLATFORM == "Windows":
    _probe = _make_windows_probe()
    _mode = "Windows (GetLastInputInfo)"
elif IS_WSL:
    if os.getenv("CFT_IDLE_WSL_PROBE", "").strip().lower() in ("1", "true", "yes"):
        _probe = _probe_wsl
        _mode = "WSL→Windows (PowerShell, cached)"
    else:
        _probe = lambda: None
        _mode = "WSL (off — set CFT_IDLE_WSL_PROBE=1 to probe the host)"
elif PLATFORM == "Linux":
    _probe = _probe_linux
    _mode = "Linux/X11 (xprintidle)"
else:
    _probe = lambda: None
    _mode = f"unsupported ({PLATFORM})"


def seconds_since_input() -> Optional[float]:
    """Seconds since the last keyboard/mouse input, or None if unknowable."""
    try:
        return _probe()
    except Exception:
        return None


def get_mode() -> str:
    """A string describing the idle-probe backend (for boot logging)."""
    return _mode
