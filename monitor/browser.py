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

import os
import sys
import threading
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
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, _SW_RESTORE)
        # Windows only honors SetForegroundWindow from the thread that owns
        # the foreground window (or with user input in play); from a
        # background process it just flashes the taskbar button. Attaching
        # this thread's input queue to the foreground thread's for the call
        # is the sanctioned way to borrow that right. A refusal is reported,
        # not raised.
        fg = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        me = kernel32.GetCurrentThreadId()
        attached = bool(fg_thread) and fg_thread != me and \
            bool(user32.AttachThreadInput(me, fg_thread, True))
        try:
            user32.BringWindowToTop(hwnd)
            ok = bool(user32.SetForegroundWindow(hwnd))
        finally:
            if attached:
                user32.AttachThreadInput(me, fg_thread, False)
        return ok and user32.GetForegroundWindow() == hwnd
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


# --- Pop-out on session and break transitions --------------------------------
#
# A session starting or ending, or a break running out, is the moment the
# dashboard is wanted: the session card, the countdown, the break prompt. From
# outside the browser a background tab cannot be raised (a window title names
# only its active tab), so a pop is: nothing when the dashboard is already the
# front window; raise its window when it is the active tab of some window;
# otherwise open a fresh tab. The page keeps that from piling up — the
# dashboard's singleTab closes the older copy when a new one announces itself.
# Opt out with CFT_POPOUT=0.

_pop_url: Optional[str] = None


def set_pop_url(url: Optional[str]) -> None:
    """Where a pop opens the dashboard (app.py: DASHBOARD_URL or the local copy)."""
    global _pop_url
    _pop_url = (url or "").strip() or None


def _pop_enabled() -> bool:
    return os.getenv("CFT_POPOUT", "1").strip().lower() not in ("0", "false", "no")


def _default_pop_url() -> Optional[str]:
    env = os.getenv("DASHBOARD_URL", "").strip()
    if env:
        return env
    try:
        from monitor.dashboard_server import _api_port
        return f"http://127.0.0.1:{_api_port()}/"
    except Exception:
        return None


def dashboard_in_foreground() -> bool:
    """Is the dashboard the active tab of the window the user is looking at?"""
    if sys.platform != "win32":
        return False
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return DASHBOARD_TITLE in buf.value.lower()
    except Exception as e:
        print(f"[browser] foreground check failed ({e})")
        return False


def pop_dashboard(reason: str, url: Optional[str] = None) -> str:
    """Bring the dashboard forward for a transition.

    Returns what happened: "disabled", "foreground" (already in front),
    "raised", "opened" or "no-url".
    """
    if not _pop_enabled():
        print(f"[browser] pop-out disabled ({reason})")
        return "disabled"
    if dashboard_in_foreground():
        print(f"[browser] dashboard already in front ({reason})")
        return "foreground"
    url = url or _pop_url or _default_pop_url()
    if not url:
        print(f"[browser] no dashboard URL to pop ({reason})")
        return "no-url"
    how = open_dashboard(url)
    if how == "opened":
        # The browser puts the new tab in front of its window, but Windows may
        # keep that window behind whatever the user is typing into. Once the
        # tab has loaded, its title is on the window — raise it ourselves.
        deadline = time.time() + 4.0
        while time.time() < deadline:
            time.sleep(0.25)
            if dashboard_in_foreground():
                break
            if focus_existing():
                how = "opened+raised"
                break
    print(f"[browser] popped dashboard ({reason}): {how}")
    return how


def pop_dashboard_async(reason: str) -> None:
    """pop_dashboard on a thread of its own: transitions are noticed on the
    runner loop or inside an API request, and neither should wait on a window
    scan or a browser launch."""
    threading.Thread(target=pop_dashboard, args=(reason,), daemon=True,
                     name="pop-dashboard").start()
