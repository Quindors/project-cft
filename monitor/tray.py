# monitor/tray.py
"""System-tray icon for the background monitor.

The monitor normally runs hidden (launched via VBS / Startup shortcut). The
tray gives it a visible entry point: "Open Dashboard", focus-session and break
controls, the timer-widget toggle, and "Quit". pystray is imported lazily so
environments without it (dev on WSL, headless) don't fail at import time.

The icon is a live indicator, not decoration: it takes its color from the same
state monitor/status.py hands the timer widget, so a glance at the tray tells
you whether you're on your intention, drifting off it, or on a break.
"""
from __future__ import annotations

import threading
import webbrowser
from typing import Callable, Optional

from monitor import breaks, browser, intentions, status, timer_widget

# Same palette as the timer widget, so tray and widget always agree.
_STATE_COLORS = {
    status.STATE_ON:       (16, 185, 129),
    status.STATE_SESSION:  (52, 211, 153),
    status.STATE_PARTIAL:  (245, 158, 11),
    status.STATE_OFF:      (239, 68, 68),
    status.STATE_BREAK:    (56, 189, 248),
    status.STATE_TRACKING: (100, 116, 139),
    status.STATE_IDLE:     (148, 163, 184),
}
_ICON_REFRESH_SEC = 5.0
_stop_icon_thread = threading.Event()


def _icon_image(state: str = status.STATE_IDLE):
    from PIL import Image, ImageDraw

    rgb = _STATE_COLORS.get(state, _STATE_COLORS[status.STATE_IDLE])
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=rgb + (255,))
    if state == status.STATE_BREAK:
        # A pause bar reads as "stopped" at 16px where a hollow disc doesn't.
        d.rectangle((25, 22, 30, 42), fill=(255, 255, 255, 255))
        d.rectangle((34, 22, 39, 42), fill=(255, 255, 255, 255))
    else:
        d.ellipse((24, 24, 40, 40), fill=(255, 255, 255, 255))
    return img


def _snapshot() -> dict:
    try:
        return status.full_snapshot()
    except Exception:
        return {}


def _status_text(item=None) -> str:
    """Dynamic menu text: what the monitor is doing right now."""
    snap = _snapshot()
    if not snap:
        return "Status: (unavailable)"

    left = snap.get("remaining_sec")
    mins = None if left is None else max(0, int(left) // 60)

    if snap.get("phase") == status.PHASE_BREAK:
        return f"On a break — {mins}m left"
    if snap.get("phase") == status.PHASE_SESSION:
        text = snap.get("label") or ""
        if len(text) > 34:
            text = text[:31] + "..."
        return f"{text} — {mins}m left"
    return "No focus session"


def _has_active_intention(item=None) -> bool:
    snap = _snapshot()
    return snap.get("phase") == status.PHASE_SESSION


def _on_break(item=None) -> bool:
    snap = _snapshot()
    return snap.get("phase") == status.PHASE_BREAK


def start_tray(dashboard_url: Optional[str], intention_url: Optional[str], on_quit: Callable[[], None]):
    """Create and run (detached) the tray icon. Returns the icon, or None on failure."""
    try:
        import pystray
    except Exception as e:
        print(f"[tray] pystray unavailable ({e}); tray disabled")
        return None

    def _open(icon, item):
        # Raise the dashboard if it's already open in a window; only open a new
        # tab when there is nothing to raise (the user asked to see it now).
        if dashboard_url:
            browser.open_dashboard(dashboard_url)

    def _start_session(icon, item):
        # The hosted dashboard is where sessions get set when one is
        # configured; the local /intention page is the fallback for setups
        # without a DASHBOARD_URL.
        if dashboard_url:
            browser.open_dashboard(dashboard_url)
        elif intention_url:
            webbrowser.open(intention_url)

    def _end_session(icon, item):
        try:
            ended = intentions.end_active("completed")
            if ended:
                print(f"[tray] focus session ended: \"{ended['text']}\"")
        except Exception as e:
            print(f"[tray] could not end session ({e})")

    def _start_break(icon, item):
        try:
            breaks.start(breaks.DEFAULT_BREAK_MIN)
        except Exception as e:
            print(f"[tray] could not start break ({e})")

    def _end_break(icon, item):
        try:
            breaks.end()
        except Exception as e:
            print(f"[tray] could not end break ({e})")

    def _toggle_widget(icon, item):
        timer_widget.toggle_hidden()

    def _reset_widget(icon, item):
        """Recovery for a timer that ended up somewhere unreachable."""
        timer_widget.reset_position()

    def _quit(icon, item):
        try:
            icon.stop()
        finally:
            on_quit()

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", _open, default=True, enabled=bool(dashboard_url)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_status_text, None, enabled=False),
        pystray.MenuItem("Start Focus Session...", _start_session,
                         enabled=lambda item: bool(dashboard_url or intention_url)
                         and not _on_break()),
        pystray.MenuItem("Finish Session (start break)", _end_session,
                         enabled=_has_active_intention),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(f"Take a {breaks.DEFAULT_BREAK_MIN}-Minute Break", _start_break,
                         enabled=lambda item: not _on_break()),
        pystray.MenuItem("End Break", _end_break, enabled=_on_break),
        pystray.MenuItem("Show Timer", _toggle_widget,
                         checked=lambda item: not timer_widget.is_hidden()),
        pystray.MenuItem("Reset Timer Position", _reset_widget),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("project-cft", _icon_image(), "project-cft focus monitor", menu)
    try:
        icon.run_detached()
    except Exception as e:
        print(f"[tray] could not start tray ({e})")
        return None

    threading.Thread(target=_keep_icon_current, args=(icon,),
                     name="tray-icon", daemon=True).start()
    return icon


def _keep_icon_current(icon) -> None:
    """Repaint the tray icon whenever the monitor's state changes."""
    last = None
    while True:
        try:
            snap = _snapshot()
            state = snap.get("state", status.STATE_IDLE)
            if state != last:
                last = state
                icon.icon = _icon_image(state)
                icon.title = f"project-cft — {_status_text()}"
        except Exception:
            pass
        if _stop_icon_thread.wait(_ICON_REFRESH_SEC):
            return


def stop_icon_updates() -> None:
    _stop_icon_thread.set()
