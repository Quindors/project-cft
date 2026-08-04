# monitor/timer_widget.py
"""The floating focus timer: a small always-on-top countdown.

It counts *down*, never up. What it counts down is whatever you are actually
in the middle of:

    a focus session   the time left in it, under your own intention text
    a break           the time left before monitoring switches back on
    neither           no digits at all, just "Focus" or "Tracking" — there is
                      nothing to count down to, and a fake number would be
                      worse than none

The color is the drift signal: emerald while you are on your intention, amber
when the work has drifted sideways, red when it is unrelated, sky blue on a
break, slate when nothing is declared. That is the same state the tray icon
and the dashboard read, resolved once in monitor/status.py so the three can
never disagree.

Drag it anywhere; click the chevron and it collapses into a thin colored bump
flush against the nearest screen edge, which is still a live drift indicator
but takes up almost nothing. Right-click for session and break actions.

All Tk work happens on the shared UI thread (monitor/tk_host.py). Position
and collapsed/hidden state persist in app_state so the widget comes back where
you left it.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from monitor import app_state, breaks, status, tk_host
from monitor.window_capture import PLATFORM

_STATE_KEY = "widget"

# Geometry. The toplevel is bigger than the visible card: the margin is
# chroma-keyed away on Windows, which is what gives the card round corners.
MARGIN = 6
CARD_W, CARD_H = 274, 62
WIN_W, WIN_H = CARD_W + 2 * MARGIN, CARD_H + 2 * MARGIN
BUMP_W, BUMP_H = 24, 78
BUMP_WIN_W, BUMP_WIN_H = BUMP_W + MARGIN, BUMP_H + 2 * MARGIN

# The magenta that -transparentcolor knocks out. Nothing in the design uses it.
CHROMA = "#ff00fe"

TICK_MS = 1000
# Hitting sqlite once a second for three tiny reads is wasteful; the countdown
# is computed locally between refreshes.
REFRESH_EVERY_TICKS = 2

INK = "#f8fafc"
MUTED = "#94a3b8"

# (accent, card background, border) per state from monitor/status.py.
PALETTE = {
    status.STATE_ON:       ("#10b981", "#0f1c19", "#14532d"),
    status.STATE_PARTIAL:  ("#f59e0b", "#1c170f", "#78350f"),
    status.STATE_OFF:      ("#ef4444", "#1e1113", "#7f1d1d"),
    status.STATE_BREAK:    ("#38bdf8", "#0f1a22", "#0c4a6e"),
    status.STATE_SESSION:  ("#34d399", "#0f1c19", "#14532d"),
    status.STATE_TRACKING: ("#64748b", "#111827", "#1e293b"),
    status.STATE_IDLE:     ("#94a3b8", "#111827", "#1e293b"),
}
_FALLBACK = PALETTE[status.STATE_IDLE]

_widget: Optional["_TimerWidget"] = None


# ---------- persisted placement ----------

def _load_prefs() -> Dict:
    try:
        data = json.loads(app_state.get_value(_STATE_KEY) or "{}")
    except (ValueError, TypeError):
        data = {}
    return data if isinstance(data, dict) else {}


def _save_prefs(**fields) -> None:
    prefs = _load_prefs()
    prefs.update(fields)
    try:
        app_state.set_value(_STATE_KEY, json.dumps(prefs))
    except Exception as e:
        print(f"[timer] could not save widget position ({e})")


def _fmt_countdown(seconds: Optional[int]) -> str:
    """M:SS under an hour, H:MM:SS above it."""
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _fmt_compact(seconds: Optional[int]) -> str:
    """The collapsed bump has room for about three characters."""
    if seconds is None:
        return ""
    mins = max(0, int(seconds)) // 60
    if mins >= 100:
        return f"{mins // 60}h"
    return str(mins)


class _TimerWidget:
    def __init__(self, dashboard_url: Optional[str], intention_url: Optional[str]):
        import tkinter as tk
        from tkinter import font as tkfont

        self.tk = tk
        self.dashboard_url = dashboard_url
        self.intention_url = intention_url

        prefs = _load_prefs()
        self.collapsed = bool(prefs.get("collapsed"))
        self.hidden = bool(prefs.get("hidden"))

        self.win = tk.Toplevel(tk_host.root())
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=CHROMA)
        # Windows only: paint the margin away so the card reads as a rounded
        # pill instead of a rectangle. Elsewhere the margin is just dead space.
        try:
            self.win.attributes("-transparentcolor", CHROMA)
        except Exception:
            self.win.configure(bg="#111827")
        try:
            self.win.attributes("-alpha", 0.96)
        except Exception:
            pass

        self.canvas = tk.Canvas(self.win, highlightthickness=0, bd=0, bg=CHROMA)
        self.canvas.pack(fill="both", expand=True)

        family = "Segoe UI" if PLATFORM == "Windows" else "TkDefaultFont"
        self.f_time = tkfont.Font(family=family, size=22, weight="bold")
        self.f_word = tkfont.Font(family=family, size=15, weight="bold")
        self.f_label = tkfont.Font(family=family, size=8)
        self.f_bump = tkfont.Font(family=family, size=8, weight="bold")

        self._snapshot: Dict = status.full_snapshot()
        self._ticks = 0
        self._drag: Optional[Dict] = None

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_menu)
        self.canvas.bind("<Enter>", lambda _e: self._set_alpha(1.0))
        self.canvas.bind("<Leave>", lambda _e: self._set_alpha(0.96))

        self._place(prefs.get("x"), prefs.get("y"))
        if self.hidden:
            self.win.withdraw()
        self._render()
        self.win.after(TICK_MS, self._tick)

    # ---------- placement ----------

    def _screen(self):
        return self.win.winfo_screenwidth(), self.win.winfo_screenheight()

    def _size(self):
        return (BUMP_WIN_W, BUMP_WIN_H) if self.collapsed else (WIN_W, WIN_H)

    def _place(self, x=None, y=None) -> None:
        """Position the window, clamped on screen (and edge-flush when collapsed)."""
        sw, sh = self._screen()
        w, h = self._size()
        if x is None or y is None:
            x, y = sw - w - 18, 90
        x, y = int(x), int(y)
        if self.collapsed:
            # A bump belongs against an edge — snap to whichever is nearer.
            x = sw - w if (x + w // 2) >= sw // 2 else 0
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h))
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.canvas.configure(width=w, height=h)
        self._x, self._y = x, y

    def _set_alpha(self, value: float) -> None:
        try:
            self.win.attributes("-alpha", value)
        except Exception:
            pass

    # ---------- state ----------

    def _remaining(self) -> Optional[int]:
        """Seconds left, recomputed locally between DB refreshes."""
        return status.seconds_until(self._snapshot.get("ends_at"))

    def _colors(self):
        return PALETTE.get(self._snapshot.get("state"), _FALLBACK)

    def _tick(self) -> None:
        self._ticks += 1
        if self._ticks % REFRESH_EVERY_TICKS == 0:
            try:
                self._snapshot = status.full_snapshot()
            except Exception as e:
                print(f"[timer] status refresh failed ({e})")
        if not self.hidden:
            self._render()
        self.win.after(TICK_MS, self._tick)

    # ---------- drawing ----------

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r,
               x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
               x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _ellipsize(self, text: str, font, max_px: int) -> str:
        if font.measure(text) <= max_px:
            return text
        while text and font.measure(text + "…") > max_px:
            text = text[:-1]
        return text + "…"

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.collapsed:
            self._render_bump()
        else:
            self._render_card()

    def _render_bump(self) -> None:
        """The collapsed form: a thin tab, rounded on the side facing inward.

        The rounded rect is drawn wider than the window on purpose — the half
        hanging off the screen edge is clipped, which leaves the tab flush
        against the edge with only its inner corners rounded.
        """
        accent, _bg, _border = self._colors()
        sw, _ = self._screen()
        on_right = self._x + BUMP_WIN_W // 2 >= sw // 2
        if on_right:
            x1, x2 = MARGIN, MARGIN + 2 * BUMP_W
        else:
            x1, x2 = -BUMP_W, BUMP_W
        y1, y2 = MARGIN, MARGIN + BUMP_H

        # Filled with the state color rather than outlined in it: at this size
        # a thin accent line just reads as a scratch on the desktop, while a
        # solid tab is legible as a status from across the room.
        self._round_rect(x1, y1, x2, y2, 11, fill=accent, outline="")
        grip_x = (MARGIN + BUMP_W / 2) if on_right else (BUMP_W / 2)

        left = self._remaining()
        if left is None:
            # No countdown to show: a grip mark, so the tab still looks grabbable.
            self._round_rect(grip_x - 1.5, y1 + 26, grip_x + 1.5, y2 - 26, 1.5,
                             fill="#00000055", outline="")
            return
        self.canvas.create_text(grip_x, (y1 + y2) / 2, text=_fmt_compact(left),
                                fill="#0b1220", font=self.f_bump, anchor="center")

    def _render_card(self) -> None:
        accent, bg, border = self._colors()
        x1, y1 = MARGIN, MARGIN
        x2, y2 = MARGIN + CARD_W, MARGIN + CARD_H
        self._round_rect(x1, y1, x2, y2, 16, fill=bg, outline=border)

        dot_x, mid = x1 + 18, (y1 + y2) / 2
        self.canvas.create_oval(dot_x - 5, mid - 5, dot_x + 5, mid + 5,
                                fill=accent, outline="")

        text_x = x1 + 34
        chevron_x = x2 - 16
        text_right = chevron_x - 12
        text_w = text_right - text_x
        snap = self._snapshot
        left = self._remaining()

        if left is None:
            # No session, no break: the state word IS the readout. There is
            # nothing to count down to, so no digits are shown at all.
            self.canvas.create_text(text_x, mid - 8, text=snap.get("label", "Focus"),
                                    fill=INK, font=self.f_word, anchor="w")
            # The category if the monitor has classified anything yet. No
            # "stale" wording here: before the first decision of a run there
            # simply is no reading, and saying so alarms for no reason.
            sub = snap.get("category") or "no session set"
            self.canvas.create_text(text_x, mid + 13,
                                    text=self._ellipsize(sub, self.f_label, text_w),
                                    fill=MUTED, font=self.f_label, anchor="w")
        else:
            self.canvas.create_text(text_x, y1 + 21, text=_fmt_countdown(left),
                                    fill=INK, font=self.f_time, anchor="w")
            self.canvas.create_text(
                text_x, y1 + 41,
                text=self._ellipsize(snap.get("label", ""), self.f_label, text_w),
                fill=MUTED, font=self.f_label, anchor="w")
            self._render_progress(text_x, y2 - 9, text_right)

        self.canvas.create_text(chevron_x, mid, text="›", fill=MUTED,
                                font=self.f_word, anchor="center")

    def _render_progress(self, x1: float, y: float, x2: float) -> None:
        """Thin bar: how much of the session or break has been used up."""
        remaining = status.seconds_until(self._snapshot.get("ends_at"))
        started = self._snapshot.get("started_at")
        used = 0.0
        if started and remaining is not None:
            from datetime import datetime
            try:
                begun = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
                span = (datetime.now() - begun).total_seconds() + remaining
                if span > 0:
                    used = min(1.0, max(0.0, 1 - remaining / span))
            except ValueError:
                pass
        accent = self._colors()[0]
        self._round_rect(x1, y, x2, y + 3, 1.5, fill="#1e293b", outline="")
        if used > 0:
            self._round_rect(x1, y, x1 + (x2 - x1) * used, y + 3, 1.5,
                             fill=accent, outline="")

    # ---------- interaction ----------

    def _on_press(self, event) -> None:
        self._drag = {"x": event.x_root, "y": event.y_root,
                      "wx": self._x, "wy": self._y, "moved": False}

    def _on_drag(self, event) -> None:
        if self._drag is None:
            return
        dx = event.x_root - self._drag["x"]
        dy = event.y_root - self._drag["y"]
        if abs(dx) > 3 or abs(dy) > 3:
            self._drag["moved"] = True
        self._place(self._drag["wx"] + dx, self._drag["wy"] + dy)

    def _on_release(self, event) -> None:
        drag = self._drag
        self._drag = None
        if drag is None:
            return
        if drag["moved"]:
            _save_prefs(x=self._x, y=self._y)
            self._render()  # the bump's rounded end may have swapped sides
            return
        # A click, not a drag. The chevron collapses; the body opens the page
        # where you actually set a session.
        if not self.collapsed and event.x < MARGIN + CARD_W - 30:
            self._open_intention()
        else:
            self.toggle_collapsed()

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        _save_prefs(collapsed=self.collapsed)
        self._place(self._x, self._y)
        self._render()

    def set_hidden(self, hidden: bool) -> None:
        self.hidden = hidden
        _save_prefs(hidden=hidden)
        if hidden:
            self.win.withdraw()
        else:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self._render()

    def _open_dashboard(self) -> None:
        """Raise an already-open dashboard rather than stacking another tab."""
        if not self.dashboard_url:
            return
        from monitor import browser
        try:
            browser.open_dashboard(self.dashboard_url)
        except Exception as e:
            print(f"[timer] could not open dashboard ({e})")

    def _open_intention(self) -> None:
        url = self.intention_url or self.dashboard_url
        if not url:
            return
        if url == self.dashboard_url:
            self._open_dashboard()
            return
        import webbrowser
        webbrowser.open(url)

    def _on_menu(self, event) -> None:
        snap = self._snapshot
        menu = self.tk.Menu(self.win, tearoff=0)
        if snap.get("phase") == status.PHASE_BREAK:
            menu.add_command(label="End break now", command=self._end_break)
        else:
            if snap.get("intention"):
                menu.add_command(label="Finish session (start break)",
                                 command=lambda: self._end_session("completed"))
                menu.add_command(label="Abandon session",
                                 command=lambda: self._end_session("abandoned"))
            else:
                menu.add_command(label="Start a focus session…",
                                 command=self._open_intention)
            menu.add_command(label=f"Take a {breaks.DEFAULT_BREAK_MIN}-minute break",
                             command=self._start_break)
        menu.add_separator()
        menu.add_command(label="Expand" if self.collapsed else "Collapse",
                         command=self.toggle_collapsed)
        menu.add_command(label="Open dashboard", command=self._open_dashboard)
        menu.add_command(label="Hide timer", command=lambda: self.set_hidden(True))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _refresh_now(self) -> None:
        try:
            self._snapshot = status.full_snapshot()
        except Exception:
            pass
        self._render()

    def _start_break(self) -> None:
        try:
            breaks.start(breaks.DEFAULT_BREAK_MIN)
        except Exception as e:
            print(f"[timer] could not start break ({e})")
        self._refresh_now()

    def _end_break(self) -> None:
        try:
            breaks.end()
        except Exception as e:
            print(f"[timer] could not end break ({e})")
        self._refresh_now()

    def _end_session(self, status_name: str) -> None:
        from monitor import intentions
        try:
            intentions.end_active(status_name)
        except Exception as e:
            print(f"[timer] could not end session ({e})")
        self._refresh_now()

    def destroy(self) -> None:
        try:
            self.win.destroy()
        except Exception:
            pass


# ---------- public API (safe to call from any thread) ----------

def start(dashboard_url: Optional[str] = None,
          intention_url: Optional[str] = None) -> bool:
    """Show the floating timer. False if this machine has no usable UI."""
    if not tk_host.start():
        return False

    def _build():
        global _widget
        if _widget is not None:
            _widget.set_hidden(False)
            return
        try:
            _widget = _TimerWidget(dashboard_url, intention_url)
        except Exception as e:
            print(f"[timer] widget could not start ({type(e).__name__}: {e})")

    return tk_host.submit(_build)


def set_hidden(hidden: bool) -> None:
    tk_host.submit(lambda: _widget and _widget.set_hidden(hidden))


def is_hidden() -> bool:
    """Read from the persisted preference — safe off the Tk thread."""
    return bool(_load_prefs().get("hidden"))


def toggle_hidden() -> None:
    set_hidden(not is_hidden())


def stop() -> None:
    def _kill():
        global _widget
        if _widget is not None:
            _widget.destroy()
            _widget = None
    tk_host.submit(_kill)
