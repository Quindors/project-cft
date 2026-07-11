# monitor/tray.py
"""System-tray icon for the background monitor.

The monitor normally runs hidden (launched via VBS / Startup shortcut). The
tray gives it a visible entry point: "Open Dashboard" and "Quit". pystray is
imported lazily so environments without it (dev on WSL, headless) don't fail
at import time.
"""
from __future__ import annotations

import webbrowser
from typing import Callable, Optional


def _icon_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=(16, 185, 129, 255))   # emerald disc
    d.ellipse((24, 24, 40, 40), fill=(255, 255, 255, 255))  # focus dot
    return img


def start_tray(dashboard_url: Optional[str], on_quit: Callable[[], None]):
    """Create and run (detached) the tray icon. Returns the icon, or None on failure."""
    try:
        import pystray
    except Exception as e:
        print(f"[tray] pystray unavailable ({e}); tray disabled")
        return None

    def _open(icon, item):
        if dashboard_url:
            webbrowser.open(dashboard_url)

    def _quit(icon, item):
        try:
            icon.stop()
        finally:
            on_quit()

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", _open, default=True, enabled=bool(dashboard_url)),
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("project-cft", _icon_image(), "project-cft focus monitor", menu)
    try:
        icon.run_detached()
    except Exception as e:
        print(f"[tray] could not start tray ({e})")
        return None
    return icon
