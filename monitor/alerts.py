# monitor/alerts.py
"""Cross-platform popup alerts - always on top.

On Windows the alert is the full-screen sheet in monitor/focus_alert.py: a
MessageBox is far too easy to dismiss without reading. The MessageBox is kept
as the fallback for when there is no usable Tk display.
"""

import re
import subprocess

from monitor.window_capture import PLATFORM, IS_WSL

# Which pipeline stage produced a verdict is written into the reason as a
# leading tag: "[cache] [Vision] The screen shows...". That is exactly right
# in the log and noise on screen — the user is being interrupted, so every
# word in front of them should be about their work, not about which stage
# decided it.
#
# A whitelist, not a general "[...]" match: window titles and quoted text
# contain brackets too, and eating those would mangle the very sentence the
# alert exists to show.
_STAGE_TAG_RE = re.compile(
    r"\[(?:cache|vision|correction|intent|error|parse_error)\]\s*", re.IGNORECASE)


def strip_stage_tags(text: str) -> str:
    """Remove pipeline stage tags from text headed for a human."""
    return " ".join(_STAGE_TAG_RE.sub(" ", text or "").split())


def _popup_messagebox(message: str, title: str) -> None:
    """Win32 MessageBox with TOPMOST + FOREGROUND flags."""
    import ctypes
    flags = 0x40 | 0x00010000 | 0x00040000  # INFO | FOREGROUND | TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)


def _popup_windows(message: str, title: str, on_correct=None) -> None:
    try:
        from monitor import focus_alert
        if focus_alert.show(message, title, on_correct=on_correct):
            return
    except Exception as e:
        print(f"[popup] full-screen alert unavailable ({e}); using MessageBox")
    # The MessageBox fallback has no room for a correction button; the same
    # ruling is always available from the dashboard.
    _popup_messagebox(message, title)


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


def alert_is_open() -> bool:
    """Is a full-screen drift alert on screen right now?

    The runner pauses on this: while the sheet is up the only thing to see is
    the sheet, and a user who hasn't realised they can just switch windows
    shouldn't have their hesitation logged as more drift.
    """
    if PLATFORM != "Windows":
        return False
    try:
        from monitor import focus_alert
        return focus_alert.is_open()
    except Exception:
        return False


def show_popup(message: str, title: str = "Productivity Alert", on_correct=None) -> None:
    """Route to the correct platform popup.

    on_correct is offered as a button on the full-screen Windows alert ("this
    IS my session work"); the other platforms' popups ignore it.

    Stage tags are stripped here rather than at each call site, so every
    backend below shows the same human-readable text.
    """
    message = strip_stage_tags(message).replace("'", "").replace('"', '')
    title = title.replace("'", "").replace('"', '')

    if PLATFORM == "Windows":
        _popup_windows(message, title, on_correct)
    elif IS_WSL:
        _popup_wsl(message, title)
    elif PLATFORM == "Darwin":
        _popup_macos(message, title)
    elif PLATFORM == "Linux":
        _popup_linux(message, title)
    else:
        print(f"[popup] {title}: {message}")