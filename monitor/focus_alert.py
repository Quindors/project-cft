# monitor/focus_alert.py
"""The full-screen drift alert — the thing you cannot scroll past.

A Win32 MessageBox is a small grey box in the middle of a screen full of
whatever pulled you away; it loses. This puts a dimmed sheet over the entire
display with a single large card on it, and it does not go away on its own.
No timeout, no auto-dismiss: it sits there until you acknowledge it, because
the whole point is to interrupt, not to notify.

It deliberately stops short of trapping input. There is no global grab and no
disabled close path — Esc, Enter, Space and the button all dismiss it, and the
window is topmost rather than input-exclusive. An alert that could wedge and
lock you out of your own machine would be a far worse bug than one you can
click past.

Runs on the shared Tk thread (monitor/tk_host.py). Callers use show(); when
there is no display it returns False and monitor/alerts.py falls back to the
old MessageBox.
"""
from __future__ import annotations

import threading
from typing import Optional

from monitor import tk_host
from monitor.window_capture import PLATFORM

BACKDROP = "#05070d"
CARD_BG = "#111827"
CARD_BORDER = "#7f1d1d"
ACCENT = "#ef4444"
INK = "#f8fafc"
MUTED = "#cbd5e1"

# The magenta that -transparentcolor knocks out, giving the card round corners.
CHROMA = "#ff00fe"

# One alert at a time. A second decision landing while the first is still up
# should not stack sheets over each other.
_open = threading.Event()


class _AlertWindow:
    """A dimmed full-screen backdrop with an opaque card floating on it.

    Two toplevels rather than one, because Tk's alpha is per-window: painting
    the card on the same window as the backdrop makes the card translucent
    too, and whatever you were doing shows straight through the text you are
    supposed to read.
    """

    def __init__(self, message: str, title: str, on_close):
        import tkinter as tk
        from tkinter import font as tkfont

        self.on_close = on_close
        root = tk_host.root()

        # --- the dimmed sheet over everything ---
        self.backdrop = tk.Toplevel(root)
        self.backdrop.configure(bg=BACKDROP)
        self.backdrop.overrideredirect(True)
        self.backdrop.attributes("-topmost", True)
        sw = self.backdrop.winfo_screenwidth()
        sh = self.backdrop.winfo_screenheight()
        self.backdrop.geometry(f"{sw}x{sh}+0+0")
        try:
            self.backdrop.attributes("-alpha", 0.82)
        except Exception:
            pass

        # --- the card, fully opaque ---
        # The process is DPI-aware (tk_host), so this pixel layout is scaled
        # to keep its physical size while being drawn at full resolution.
        s = tk_host.ui_scale(self.backdrop)
        px = lambda n: max(1, int(round(n * s)))

        family = "Segoe UI" if PLATFORM == "Windows" else "TkDefaultFont"
        f_title = tkfont.Font(family=family, size=-px(45), weight="bold")
        f_body = tkfont.Font(family=family, size=-px(21))
        f_btn = tkfont.Font(family=family, size=-px(17), weight="bold")
        f_hint = tkfont.Font(family=family, size=-px(13))

        pad = px(8)
        card_w = max(px(520), min(px(860), int(sw * 0.66)))
        card_h = px(380)
        win_w, win_h = card_w + 2 * pad, card_h + 2 * pad

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=CHROMA)
        try:
            self.win.attributes("-transparentcolor", CHROMA)
        except Exception:
            self.win.configure(bg=CARD_BG)
        self.win.geometry(
            f"{win_w}x{win_h}+{int((sw - win_w) / 2)}+{int((sh - win_h) / 2)}")

        canvas = tk.Canvas(self.win, width=win_w, height=win_h, bg=CHROMA,
                           highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)

        x1, y1 = pad, pad
        x2, y2 = pad + card_w, pad + card_h
        cx = win_w / 2
        self._round_rect(canvas, x1, y1, x2, y2, px(24), fill=CARD_BG, outline=CARD_BORDER)
        # A bar of the alert color across the top, clipped to the rounded head.
        self._round_rect(canvas, x1, y1, x2, y1 + px(26), px(24), fill=ACCENT, outline="")
        canvas.create_rectangle(x1, y1 + px(14), x2, y1 + px(26), fill=ACCENT, outline="")

        canvas.create_text(cx, y1 + px(96), text=title, fill=INK,
                           font=f_title, anchor="center")

        # The message arrives as "<headline>\n<detail>" from the runner.
        lines = [ln for ln in (message or "").split("\n") if ln.strip()]
        y = y1 + px(158)
        for line in lines[:4]:
            canvas.create_text(cx, y, text=line, fill=MUTED, font=f_body,
                               anchor="center", width=card_w - px(100))
            y += px(38)

        btn_w, btn_h = px(240), px(52)
        bx1, by1 = cx - btn_w / 2, y2 - px(104)
        bx2, by2 = cx + btn_w / 2, by1 + btn_h
        btn = self._round_rect(canvas, bx1, by1, bx2, by2, px(14),
                               fill=ACCENT, outline="")
        label = canvas.create_text((bx1 + bx2) / 2, (by1 + by2) / 2,
                                   text="Back to it", fill="#ffffff",
                                   font=f_btn, anchor="center")
        for item in (btn, label):
            canvas.tag_bind(item, "<Button-1>", lambda _e: self.close())

        canvas.create_text(cx, y2 - px(32), text="Esc to dismiss",
                           fill="#64748b", font=f_hint, anchor="center")

        # Only the button and the keyboard dismiss it. Clicking the backdrop
        # deliberately does nothing: a stray click on the way back to the tab
        # that distracted you should not count as acknowledging the alert.
        for win in (self.win, self.backdrop):
            for key in ("<Escape>", "<Return>", "<space>"):
                win.bind(key, lambda _e: self.close())

        self.backdrop.lift()
        self.win.lift()
        try:
            self.win.focus_force()
        except Exception:
            pass

    @staticmethod
    def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r,
               x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
               x1, y1 + r, x1, y1]
        return canvas.create_polygon(pts, smooth=True, **kw)

    def close(self) -> None:
        for win in (self.win, self.backdrop):
            try:
                win.destroy()
            except Exception:
                pass
        self.on_close()


def show(message: str, title: str = "Focus Alert") -> bool:
    """Put the alert on screen. False if there is no usable display."""
    if not tk_host.start():
        return False
    if _open.is_set():
        return True  # one is already up; don't stack
    _open.set()

    def _build():
        try:
            _AlertWindow(message, title, on_close=_open.clear)
        except Exception as e:
            _open.clear()
            print(f"[alert] full-screen alert failed ({type(e).__name__}: {e})")

    if not tk_host.submit(_build):
        _open.clear()
        return False
    return True


def is_open() -> bool:
    return _open.is_set()
