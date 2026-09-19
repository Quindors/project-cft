# monitor/browser.py
"""Open the dashboard without piling up browser tabs.

The monitor opens the dashboard at startup and from the tray. Done naively
(webbrowser.open) that spawns a fresh tab every time — and since the autostart
task relaunches the exe at logon, tabs accumulate across days.

Two mechanisms avoid that, because neither alone is enough:

  1. focus_existing() raises a browser window that is already showing the
     dashboard. A window title only reports its ACTIVE tab, so this finds the
     dashboard only when it is the frontmost tab of some window.

  2. client_seen_within() asks the local API whether a dashboard page has
     polled it recently. The page polls every few seconds, so a recent request
     proves a dashboard is open even in a background or pinned tab — exactly
     the case a window-title search cannot see.

Startup therefore stays quiet when either says a dashboard exists; the tray,
where the user explicitly asked to see it, falls back to opening a tab when it
cannot raise an existing window.
"""
from __future__ import annotations

import sys
import time
import webbrowser
from typing import Optional

# Both the hosted and bundled builds set <title>focus-dashboard</title>, so a
# browser window showing it carries this in its title bar.
DASHBOARD_TITLE = "focus-dashboard"

_SW_RESTORE = 9


def _find_window(title_token: str) -> Optional[int]:
    """HWND of a visible window whose title contains title_token, or None."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    token = title_token.lower()
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if token in buf.value.lower():
            found.append(hwnd)
            return False  # stop enumerating
        return True

    try:
        user32.EnumWindows(callback, 0)
    except Exception as e:
        print(f"[browser] window scan failed ({e})")
        return None
    return found[0] if found else None


def focus_existing(title_token: str = DASHBOARD_TITLE) -> bool:
    """Raise an existing dashboard window. True if one was found and raised."""
    hwnd = _find_window(title_token)
    if hwnd is None:
        return False
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, _SW_RESTORE)
        # Windows only honors SetForegroundWindow from the foreground process
        # or with user input in play; a refusal is reported, not raised.
        return bool(user32.SetForegroundWindow(hwnd))
    except Exception as e:
        print(f"[browser] could not raise dashboard window ({e})")
        return False


def dashboard_is_open(within_sec: float = 45.0) -> bool:
    """Is a dashboard page already open anywhere (including a background tab)?"""
    if _find_window(DASHBOARD_TITLE) is not None:
        return True
    try:
        from monitor.dashboard_server import client_seen_within
        return client_seen_within(within_sec)
    except Exception:
        return False


def open_dashboard(url: str, *, raise_existing: bool = True) -> str:
    """Show the dashboard, reusing an open one where possible.

    Returns what happened: "raised", "already-open", or "opened".
    """
    if not url:
        return "no-url"
    if raise_existing and focus_existing():
        print("[browser] dashboard already open — raised its window")
        return "raised"
    webbrowser.open(url)
    print(f"[browser] opened dashboard: {url}")
    return "opened"


def open_dashboard_if_absent(url: str, *, grace_sec: float = 5.0) -> str:
    """Startup path: open the dashboard only if one isn't already open.

    Waits up to grace_sec for an existing page to poll the API — a pinned or
    background tab announces itself that way and nothing else does. A page
    that has lost the monitor probes for it every 2s (focus-dashboard's
    useMonitorLive), so 5s is a couple of probes plus slack; it used to be
    20s, sized for the page's 15s idle poll, and that was most of the wait
    between the timer card appearing and the dashboard doing so. Never
    raises a window: stealing focus on every logon would be worse than the
    stray tab this avoids.
    """
    if not url:
        return "no-url"
    deadline = time.time() + max(0.0, grace_sec)
    while True:
        if dashboard_is_open():
            print("[browser] dashboard already open — not opening another tab")
            return "already-open"
        if time.time() >= deadline:
            break
        time.sleep(0.5)
    webbrowser.open(url)
    print(f"[browser] opened dashboard: {url}")
    return "opened"
