# monitor/alerts.py
"""Cross-platform popup alerts — always on top."""

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
    """Win32 MessageBox with TOPMOST + FOREGROUND flags."""
    import ctypes
    flags = 0x40 | 0x00010000 | 0x00040000  # INFO | FOREGROUND | TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)


def _popup_wsl(message: str, title: str) -> None:
    """Show a TopMost WinForms popup from WSL via PowerShell."""
    message = message.replace("'", "").replace('"', '').replace('\n', ' ')
    title = title.replace("'", "").replace('"', '')

    ps_cmd = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "Add-Type -AssemblyName System.Drawing;"
        " $f = New-Object System.Windows.Forms.Form;"
        " $f.Text = '" + title + "';"
        " $f.TopMost = $true;"
        " $f.StartPosition = 'CenterScreen';"
        " $f.Size = New-Object System.Drawing.Size(420,200);"
        " $f.FormBorderStyle = 'FixedDialog';"
        " $f.MaximizeBox = $false;"
        " $f.MinimizeBox = $false;"
        " $lbl = New-Object System.Windows.Forms.Label;"
        " $lbl.Text = '" + message + "';"
        " $lbl.AutoSize = $false;"
        " $lbl.Size = New-Object System.Drawing.Size(380,100);"
        " $lbl.Location = New-Object System.Drawing.Point(15,15);"
        " $f.Controls.Add($lbl);"
        " $btn = New-Object System.Windows.Forms.Button;"
        " $btn.Text = 'OK';"
        " $btn.Size = New-Object System.Drawing.Size(80,30);"
        " $btn.Location = New-Object System.Drawing.Point(160,120);"
        " $btn.Add_Click({ $f.Close() });"
        " $f.Controls.Add($btn);"
        " $f.Add_Shown({ $f.Activate() });"
        " [void]$f.ShowDialog()"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"[popup] {title}: {message}")


def _popup_linux(message: str, title: str) -> None:
    """Try zenity (topmost), fall back to notify-send, then console."""
    try:
        subprocess.Popen(
            ["zenity", "--info", "--title", title, "--text", message,
             "--width=350", "--timeout=10", "--no-markup"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return
    except FileNotFoundError:
        pass

    try:
        subprocess.Popen(
            ["notify-send", "-u", "critical", "-t", "10000", title, message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return
    except FileNotFoundError:
        pass

    print(f"[popup] {title}: {message}")
    print("[hint] Install zenity or notify-send: sudo apt install zenity libnotify-bin")


def _popup_macos(message: str, title: str) -> None:
    """macOS dialog via osascript — always on top unlike notifications."""
    message = message.replace('"', '\\"')
    title = title.replace('"', '\\"')
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f'display dialog "{message}" with title "{title}" '
             f'buttons {{"OK"}} default button "OK" giving up after 10'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"[popup] {title}: {message}")


def show_popup(message: str, title: str = "Productivity Alert") -> None:
    """Route to the correct platform popup."""
    message = message.replace("'", "").replace('"', '')
    title = title.replace("'", "").replace('"', '')

    if PLATFORM == "Windows":
        _popup_windows(message, title)
    elif IS_WSL:
        _popup_wsl(message, title)
    elif PLATFORM == "Darwin":
        _popup_macos(message, title)
    elif PLATFORM == "Linux":
        _popup_linux(message, title)
    else:
        print(f"[popup] {title}: {message}")