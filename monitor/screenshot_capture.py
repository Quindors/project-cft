# monitor/screenshot_capture.py
"""
Cross-platform screenshot capture.

Supports:
  - Windows: PIL.ImageGrab
  - WSL: PowerShell screenshot via .NET, piped back as base64
  - Linux (X11): scrot to temp file, read back
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import tempfile
from typing import Optional

from monitor.window_capture import PLATFORM, IS_WSL


# ---------- Windows ----------

def _screenshot_windows() -> Optional[bytes]:
    """Screenshot via PIL.ImageGrab (Windows native)."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return buf.getvalue()
    except Exception as e:
        print(f"[warn] Windows screenshot failed: {e}")
        return None


# ---------- WSL ----------

_PS_SCREENSHOT_CMD = """
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$gfx.Dispose()

$ms = New-Object System.IO.MemoryStream
$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$bmp.Dispose()

$bytes = $ms.ToArray()
$ms.Dispose()

[Convert]::ToBase64String($bytes)
"""


def _screenshot_wsl() -> Optional[bytes]:
    """Screenshot via PowerShell .NET from WSL."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _PS_SCREENSHOT_CMD],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return base64.b64decode(result.stdout.strip())
    except FileNotFoundError:
        print("[warn] powershell.exe not found for WSL screenshot", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[warn] WSL screenshot timed out", file=sys.stderr)
    except Exception as e:
        print(f"[warn] WSL screenshot failed: {e}", file=sys.stderr)
    return None


# ---------- Linux (X11) ----------

def _screenshot_linux() -> Optional[bytes]:
    """Screenshot via scrot (X11) or grim (Wayland)."""
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)

        # Try scrot first (X11)
        result = subprocess.run(
            ["scrot", "-o", tmp_path],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            # Try grim (Wayland)
            result = subprocess.run(
                ["grim", tmp_path],
                capture_output=True, timeout=5,
            )

        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, "rb") as f:
                return f.read()

    except FileNotFoundError:
        print("[warn] Neither scrot nor grim found. Install: sudo apt install scrot", file=sys.stderr)
    except Exception as e:
        print(f"[warn] Linux screenshot failed: {e}", file=sys.stderr)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return None


# ---------- public API ----------

if PLATFORM == "Windows":
    _capture_fn = _screenshot_windows
    _mode = "Windows (PIL)"
elif IS_WSL:
    _capture_fn = _screenshot_wsl
    _mode = "WSL→Windows (PowerShell)"
elif PLATFORM == "Linux":
    _capture_fn = _screenshot_linux
    _mode = "Linux (scrot/grim)"
else:
    _capture_fn = lambda: None
    _mode = f"Unsupported ({PLATFORM})"


def take_screenshot() -> Optional[bytes]:
    """
    Capture a screenshot and return as JPEG bytes.
    Returns None if capture fails.
    """
    return _capture_fn()


def take_screenshot_b64() -> Optional[str]:
    """
    Capture a screenshot and return as base64-encoded JPEG string.
    Returns None if capture fails.
    """
    data = take_screenshot()
    if data:
        return base64.b64encode(data).decode("utf-8")
    return None
    return _mode