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

Drag it anywhere; click the chevron and it collapses into a thin colored tab
that docks to whichever screen edge you drag it near, turning sideways on the
top and bottom edges so it always lies flat along the edge.

All Tk work happens on the shared UI thread (monitor/tk_host.py). Position,
edge and collapsed/hidden state persist in app_state so the widget comes back
where you left it.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Optional

from monitor import app_state, breaks, status, tk_host
from monitor.window_capture import PLATFORM

_STATE_KEY = "widget"

# --- Geometry, in reference pixels at 100% scaling ---------------------------
# Everything here is multiplied by the display's DPI scale at build time (see
# _px), so the widget is the same physical size on a 100% and a 150% display
# but is rendered at the panel's real resolution rather than bitmap-stretched.
MARGIN = 6            # chroma-keyed gutter that lets the card have round corners
CARD_W, CARD_H = 274, 62
BUMP_THICK = 13       # the collapsed tab's thin dimension
BUMP_LONG = 78        # ... and its long one

# The magenta that -transparentcolor knocks out. Nothing in the design uses it.
CHROMA = "#ff00fe"

TICK_MS = 1000
# Hitting sqlite once a second for three tiny reads is wasteful; the countdown
# is computed locally between refreshes.
REFRESH_EVERY_TICKS = 2

# How close to an edge (reference px) a drag has to end to dock there.
SIDES = ("left", "right", "top", "bottom")

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

# Tk canvas colors are opaque — "#rrggbbaa" raises TclError. Grip marks are a
# hand-picked solid instead of black-at-33%.
GRIP_ON_ACCENT = "#0f172a"

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
    """The collapsed tab has room for about three characters."""
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
        self.side = prefs.get("side") if prefs.get("side") in SIDES else "right"

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

        # Render at the panel's real pixel density. The process is marked
        # DPI-aware (tk_host), so without scaling these constants the widget
        # would come out physically smaller on a scaled display.
        self.scale = tk_host.ui_scale(self.win)
        self.margin = self._px(MARGIN)
        self.card_w, self.card_h = self._px(CARD_W), self._px(CARD_H)
        self.thick, self.long = self._px(BUMP_THICK), self._px(BUMP_LONG)

        self.canvas = tk.Canvas(self.win, highlightthickness=0, bd=0, bg=CHROMA)
        self.canvas.pack(fill="both", expand=True)

        family = "Segoe UI" if PLATFORM == "Windows" else "TkDefaultFont"
        # Negative size means pixels, which is what the scaled layout above is
        # in; point sizes would be scaled a second time by Tk.
        self.f_time = tkfont.Font(family=family, size=-self._px(29), weight="bold")
        self.f_word = tkfont.Font(family=family, size=-self._px(20), weight="bold")
        self.f_label = tkfont.Font(family=family, size=-self._px(11))
        self.f_tab = tkfont.Font(family=family, size=-self._px(10), weight="bold")

        self._snapshot: Dict = status.full_snapshot()
        self._ticks = 0
        self._drag: Optional[Dict] = None
        self._last_error = (None, 0.0)

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

    def _px(self, n: float) -> int:
        return max(1, int(round(n * self.scale)))

    # ---------- placement ----------

    def _screen(self):
        return self.win.winfo_screenwidth(), self.win.winfo_screenheight()

    def _size(self):
        """Outer window size for the current form and docked edge."""
        if not self.collapsed:
            return self.card_w + 2 * self.margin, self.card_h + 2 * self.margin
        if self.side in ("left", "right"):
            return self.thick + self.margin, self.long + 2 * self.margin
        return self.long + 2 * self.margin, self.thick + self.margin

    def _nearest_side(self, x: int, y: int) -> str:
        """Which screen edge is this window closest to?"""
        sw, sh = self._screen()
        w, h = self._size()
        gaps = {
            "left": x,
            "right": sw - (x + w),
            "top": y,
            "bottom": sh - (y + h),
        }
        return min(gaps, key=gaps.get)

    def _place(self, x=None, y=None) -> None:
        """Position the window, clamped on screen and edge-flush when collapsed."""
        sw, sh = self._screen()
        w, h = self._size()
        if x is None or y is None:
            x, y = sw - w - self._px(18), self._px(90)
        x, y = int(x), int(y)

        if self.collapsed:
            # A tab belongs against its edge: pin the crossing axis, and let
            # the drag decide only how far along the edge it sits.
            if self.side == "left":
                x = 0
            elif self.side == "right":
                x = sw - w
            elif self.side == "top":
                y = 0
            else:
                y = sh - h

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

    def _log_once(self, msg: str) -> None:
        """Log a repeating failure at most once a minute.

        This runs once a second; an unthrottled error here filled megabytes of
        cft.log the last time something in here threw.
        """
        last, when = self._last_error
        now = time.time()
        if msg != last or (now - when) > 60:
            print(f"[timer] {msg}")
            self._last_error = (msg, now)

    def _tick(self) -> None:
        try:
            self._ticks += 1
            if self._ticks % REFRESH_EVERY_TICKS == 0:
                self._snapshot = status.full_snapshot()
            if not self.hidden:
                self._render()
        except Exception as e:
            self._log_once(f"tick failed ({type(e).__name__}: {e})")
        finally:
            # Re-arm no matter what. Without this a single bad frame means
            # `after` is never scheduled again and the widget is frozen on a
            # stale reading for the rest of the session — which is exactly
            # what an alpha-hex canvas color once did to it.
            try:
                self.win.after(TICK_MS, self._tick)
            except Exception:
                pass

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
        """Repaint. Never raises: a bad frame is not worth a dead clock."""
        try:
            self.canvas.delete("all")
            if self.collapsed:
                self._render_tab()
            else:
                self._render_card()
        except Exception as e:
            self._log_once(f"render failed ({type(e).__name__}: {e})")

    def _render_tab(self) -> None:
        """The collapsed form: a thin tab lying flat along its screen edge.

        The rounded rect is drawn overhanging the edge on purpose — the half
        off-screen is clipped, leaving the tab flush against the edge with
        only its inner corners rounded. On the top and bottom edges the whole
        thing turns sideways, including the label, so it never sticks out.
        """
        accent, _bg, _border = self._colors()
        m, t, ln = self.margin, self.thick, self.long
        vertical = self.side in ("left", "right")

        if self.side == "right":
            x1, x2, y1, y2 = m, m + 2 * t, m, m + ln
        elif self.side == "left":
            x1, x2, y1, y2 = -t, t, m, m + ln
        elif self.side == "top":
            x1, x2, y1, y2 = m, m + ln, -t, t
        else:  # bottom
            x1, x2, y1, y2 = m, m + ln, m, m + 2 * t

        # Filled with the state color rather than outlined in it: at this size
        # a thin accent line just reads as a scratch on the desktop, while a
        # solid tab is legible as a status from across the room.
        self._round_rect(x1, y1, x2, y2, self._px(6), fill=accent, outline="")

        if vertical:
            cx = (m + t / 2) if self.side == "right" else (t / 2)
            cy = m + ln / 2
        else:
            cx = m + ln / 2
            cy = (t / 2) if self.side == "top" else (m + t / 2)

        left = self._remaining()
        if left is None:
            # Nothing to count down: a grip mark, so the tab still reads as
            # something you can grab rather than a stray colored bar.
            half = self._px(12)
            if vertical:
                self._round_rect(cx - self._px(1.5), cy - half,
                                 cx + self._px(1.5), cy + half,
                                 self._px(1.5), fill=GRIP_ON_ACCENT, outline="")
            else:
                self._round_rect(cx - half, cy - self._px(1.5),
                                 cx + half, cy + self._px(1.5),
                                 self._px(1.5), fill=GRIP_ON_ACCENT, outline="")
            return

        self.canvas.create_text(cx, cy, text=_fmt_compact(left),
                                fill=GRIP_ON_ACCENT, font=self.f_tab,
                                anchor="center", angle=90 if vertical else 0)

    def _render_card(self) -> None:
        accent, bg, border = self._colors()
        m = self.margin
        x1, y1 = m, m
        x2, y2 = m + self.card_w, m + self.card_h
        self._round_rect(x1, y1, x2, y2, self._px(16), fill=bg, outline=border)

        dot_x, mid = x1 + self._px(18), (y1 + y2) / 2
        r = self._px(5)
        self.canvas.create_oval(dot_x - r, mid - r, dot_x + r, mid + r,
                                fill=accent, outline="")

        text_x = x1 + self._px(34)
        chevron_x = x2 - self._px(16)
        text_right = chevron_x - self._px(12)
        text_w = text_right - text_x
        snap = self._snapshot
        left = self._remaining()

        if left is None:
            # No session, no break: the state word IS the readout. There is
            # nothing to count down to, so no digits are shown at all.
            self.canvas.create_text(text_x, mid - self._px(8),
                                    text=snap.get("label", "Focus"),
                                    fill=INK, font=self.f_word, anchor="w")
            # The category if the monitor has classified anything yet. No
            # "stale" wording here: before the first decision of a run there
            # simply is no reading, and saying so alarms for no reason.
            sub = snap.get("category") or "no session set"
            self.canvas.create_text(text_x, mid + self._px(13),
                                    text=self._ellipsize(sub, self.f_label, text_w),
                                    fill=MUTED, font=self.f_label, anchor="w")
        else:
            self.canvas.create_text(text_x, y1 + self._px(21),
                                    text=_fmt_countdown(left),
                                    fill=INK, font=self.f_time, anchor="w")
            self.canvas.create_text(
                text_x, y1 + self._px(41),
                text=self._ellipsize(snap.get("label", ""), self.f_label, text_w),
                fill=MUTED, font=self.f_label, anchor="w")
            self._render_progress(text_x, y2 - self._px(9), text_right)

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
        h, r = self._px(3), self._px(1.5)
        self._round_rect(x1, y, x2, y + h, r, fill="#1e293b", outline="")
        if used > 0:
            self._round_rect(x1, y, x1 + (x2 - x1) * used, y + h, r,
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
        if abs(dx) > self._px(3) or abs(dy) > self._px(3):
            self._drag["moved"] = True

        nx, ny = self._drag["wx"] + dx, self._drag["wy"] + dy
        if self.collapsed:
            # Re-dock live, so the tab visibly turns sideways as you drag it
            # onto the top or bottom edge instead of snapping at the end.
            side = self._nearest_side(nx, ny)
            if side != self.side:
                self.side = side
                self._place(nx, ny)
                self._render()
                return
        self._place(nx, ny)

    def _on_release(self, event) -> None:
        drag = self._drag
        self._drag = None
        if drag is None:
            return
        if drag["moved"]:
            _save_prefs(x=self._x, y=self._y, side=self.side)
            self._render()  # the tab's rounded end may have swapped sides
            return
        # A click, not a drag. The chevron collapses; the body opens the page
        # where you actually set a session.
        if not self.collapsed and event.x < self.margin + self.card_w - self._px(30):
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
        except Exception as e:
            self._log_once(f"refresh failed ({type(e).__name__}: {e})")
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
