# monitor/alerts.py
"""Cross-platform popup alerts."""

import platform
import subprocess

PLATFORM = platform.system()


def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


IS_WSL = PLATFORM == "Linux" and _is_wsl()


def _popup_windows(message: str, title: str) -> None:
    import ctypes
    flags = 0x40 | 0x00010000 | 0x00040000  # INFO | FOREGROUND | TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)


def _popup_wsl(message: str, title: str) -> None:
    """Call Windows MessageBox from WSL via PowerShell."""
    ps_cmd = (
        f"Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.MessageBox]::Show('{message}', '{title}', 'OK', 'Information')"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"[popup] {title}: {message}")


def _popup_linux(message: str, title: str) -> None:
    """Try notify-send (non-blocking), fall back to console."""
    try:
        subprocess.Popen(
            ["notify-send", "-u", "critical", "-t", "5000", title, message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(f"[popup] {title}: {message}")
        print("[hint] Install notify-send: sudo apt install libnotify-bin")


def _popup_macos(message: str, title: str) -> None:
    """macOS native notification via osascript."""
    try:
        subprocess.Popen(
            ["osascript"