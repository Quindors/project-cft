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

It is a peek, not a panel. The resting form is a thin colored tab docked to a
screen edge; clicking it slides the card out, and the card puts itself away
again — when you go back to another window, or after a few seconds of being
ignored. Anything that looks like you are still using it (the pointer resting
on it, a drag, an open menu) holds it open. Drag it near any edge to re-dock,
and it turns sideways on the top and bottom edges so it always lies flat.

All Tk work happens on the shared UI thread (monitor/tk_host.py). Position,
edge and collapsed/hidden state persist in app_state so the widget comes back
where you left it.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Dict, Optional, Tuple

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
GUTTER = 32           # the chevron's hit area, drawn so it reads as a button

# The magenta that -transparentcolor knocks out. Nothing in the design uses it.
CHROMA = "#ff00fe"

TICK_MS = 1000
# Hitting sqlite once a second for three tiny reads is wasteful; the countdown
# is computed locally between refreshes.
REFRESH_EVERY_TICKS = 2
# How often to shoulder back to the front of the topmost band. Every few
# seconds is enough to recover from a taskbar click without churning z-order.
TOPMOST_EVERY_TICKS = 4

# Open/close animation. 500ms reads as deliberate rather than sluggish.
ANIM_MS = 500.0
ANIM_FRAME_MS = 16          # ~60fps
# How long an ignored card stays out before putting itself away.
AUTO_COLLAPSE_SEC = 10.0

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

# The collapse affordance points at the edge the card will fly to, so it
# doubles as a reminder of where the tab is about to appear.
CHEVRON = {"left": "◂", "right": "▸", "top": "▴", "bottom": "▾"}

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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Blend two #rrggbb colors. Tk has no alpha, so blending is explicit."""
    try:
        a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    except (ValueError, IndexError):
        return c2
    t = min(1.0, max(0.0, t))
    return "#%02x%02x%02x" % tuple(int(round(_lerp(a[i], b[i], t))) for i in range(3))


def _ease(t: float) -> float:
    """Ease-in-out cubic — slow at both ends, quick through the middle."""
    t = min(1.0, max(0.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def _foreground_window() -> int:
    """HWND of whatever the user is working in, or 0 where unsupported."""
    if sys.platform != "win32":
        return 0
    try:
        import ctypes
        return int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        return 0


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
        self.gutter = self._px(GUTTER)

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
        self._anim: Optional[Dict] = None
        self._hover_chevron = False
        self._pointer_in = False
        self._menu_open = False
        # Auto-collapse bookkeeping: when the card came out, and what the user
        # was working in at that moment.
        self._shown_at = time.time()
        self._fg_at_show = _foreground_window()

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_menu)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

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

    def _work_area(self):
        """Desktop rect EXCLUDING the taskbar, as (left, top, right, bottom).

        Docking to the full screen height parks the tab underneath the
        taskbar, where it is invisible and unclickable — the bottom edge of
        the screen is exactly where the taskbar lives. The work area is the
        rect Windows reserves for ordinary windows, and it moves correctly
        when the taskbar is on another edge or auto-hidden.
        """
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                rect = wintypes.RECT()
                SPI_GETWORKAREA = 0x0030
                if ctypes.windll.user32.SystemParametersInfoW(
                        SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                    if rect.right > rect.left and rect.bottom > rect.top:
                        return rect.left, rect.top, rect.right, rect.bottom
            except Exception:
                pass
        sw, sh = self._screen()
        return 0, 0, sw, sh

    def _keep_on_top(self) -> None:
        """Re-assert topmost without stealing focus.

        Clicking the taskbar (itself a topmost window) re-orders the topmost
        band and can leave the widget buried. SetWindowPos with NOACTIVATE
        puts us back at the front without taking focus from whatever the user
        is typing into.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes

            HWND_TOPMOST = -1
            SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
            hwnd = self.win.winfo_id()
            user32 = ctypes.windll.user32
            parent = user32.GetParent(hwnd)   # Tk may wrap the toplevel
            user32.SetWindowPos(parent or hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
        except Exception:
            try:
                self.win.attributes("-topmost", True)
            except Exception:
                pass

    def _size_for(self, collapsed: bool, side: str) -> Tuple[int, int]:
        if not collapsed:
            return self.card_w + 2 * self.margin, self.card_h + 2 * self.margin
        if side in ("left", "right"):
            return self.thick + self.margin, self.long + 2 * self.margin
        return self.long + 2 * self.margin, self.thick + self.margin

    def _size(self):
        return self._size_for(self.collapsed, self.side)

    def _geometry_for(self, collapsed: bool, side: str, x, y) -> Tuple[int, int, int, int]:
        """Where a form would sit: pinned to its edge and inside the work area."""
        wl, wt, wr, wb = self._work_area()
        w, h = self._size_for(collapsed, side)
        if x is None or y is None:
            x, y = wr - w - self._px(18), wt + self._px(90)
        x, y = int(x), int(y)
        if collapsed:
            # A tab belongs against its edge: pin the crossing axis, and let
            # the drag decide only how far along the edge it sits.
            if side == "left":
                x = wl
            elif side == "right":
                x = wr - w
            elif side == "top":
                y = wt
            else:
                y = wb - h
        x = max(wl, min(x, wr - w))
        y = max(wt, min(y, wb - h))
        return x, y, w, h

    def _apply_geometry(self, x, y, w, h) -> None:
        self.win.geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")
        self.canvas.configure(width=int(w), height=int(h))
        self._x, self._y = int(x), int(y)

    def _nearest_side(self, x: int, y: int) -> str:
        """Which working edge is this window closest to?"""
        wl, wt, wr, wb = self._work_area()
        w, h = self._size()
        gaps = {"left": x - wl, "right": wr - (x + w),
                "top": y - wt, "bottom": wb - (y + h)}
        return min(gaps, key=gaps.get)

    def _place(self, x=None, y=None) -> None:
        """Position the window, clamped to the work area (never under the taskbar)."""
        self._apply_geometry(*self._geometry_for(self.collapsed, self.side, x, y))
        self._keep_on_top()

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

    def _busy(self) -> bool:
        """Is the user visibly in the middle of using the card?"""
        return bool(self._drag) or self._menu_open or self._pointer_in

    def _maybe_auto_collapse(self) -> None:
        """Put an ignored card away.

        Two triggers, both meaning "you're done looking at this": the user has
        gone back to some other window, or the card has simply sat there. A
        pointer resting on it, a drag, or an open menu all count as still in
        use and hold it open — collapsing under the cursor would send the next
        click through to whatever is behind.
        """
        if self.collapsed or self.hidden or self._anim or self._busy():
            if self._busy():
                self._shown_at = time.time()   # still in use; restart the clock
            return
        fg = _foreground_window()
        switched = bool(fg) and bool(self._fg_at_show) and fg != self._fg_at_show
        if switched or (time.time() - self._shown_at) >= AUTO_COLLAPSE_SEC:
            self.toggle_collapsed(animate=True, remember=False)

    def _tick(self) -> None:
        try:
            self._ticks += 1
            if self._ticks % REFRESH_EVERY_TICKS == 0:
                self._snapshot = status.full_snapshot()
            if not self.hidden:
                self._maybe_auto_collapse()
                if not self._anim:
                    self._render()
                if self._ticks % TOPMOST_EVERY_TICKS == 0:
                    self._keep_on_top()
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

    # ---------- animation ----------

    def toggle_collapsed(self, animate: bool = True, remember: bool = True) -> None:
        """Slide between the card and the tab.

        `remember` is False for automatic collapses: putting itself away is a
        transient courtesy, and shouldn't overwrite the form you last chose.
        """
        target = not self.collapsed
        if remember:
            _save_prefs(collapsed=target)
        if not animate:
            self.collapsed = target
            self._place(self._x, self._y)
            self._render()
            self._note_shown()
            return

        start = self._geometry_for(self.collapsed, self.side, self._x, self._y)
        end = self._geometry_for(target, self.side, self._x, self._y)
        self._anim = {"t0": time.time(), "from": start, "to": end, "target": target}
        self._animate()

    def _animate(self) -> None:
        """One frame of the open/close slide."""
        anim = self._anim
        if anim is None:
            return
        try:
            p = _ease((time.time() - anim["t0"]) / (ANIM_MS / 1000.0))
            rect = [_lerp(a, b, p) for a, b in zip(anim["from"], anim["to"])]
            self._apply_geometry(*rect)
            self._render_morph(p if anim["target"] else 1.0 - p)

            if p >= 1.0:
                self._anim = None
                self.collapsed = anim["target"]
                self._place(self._x, self._y)
                self._render()
                self._note_shown()
                return
        except Exception as e:
            # Never strand the widget mid-slide at some interpolated size.
            self._anim = None
            self.collapsed = anim["target"]
            self._log_once(f"animation failed ({type(e).__name__}: {e})")
            try:
                self._place(self._x, self._y)
                self._render()
            except Exception:
                pass
            return
        self.win.after(ANIM_FRAME_MS, self._animate)

    def _render_morph(self, toward_tab: float) -> None:
        """The in-between frame: just the shape, morphing card -> tab.

        No text during the slide. Interpolating a layout of glyphs at 60fps
        looks like a squash; a shape that shrinks into the edge reads as the
        card tucking itself away.
        """
        accent, bg, border = self._colors()
        self.canvas.delete("all")
        w = int(self.canvas.cget("width") or 0)
        h = int(self.canvas.cget("height") or 0)
        m = self.margin
        fill = _lerp_color(bg, accent, toward_tab)
        edge = _lerp_color(border, accent, toward_tab)
        radius = _lerp(self._px(16), self._px(6), toward_tab)
        self._round_rect(m, m, max(m + 1, w - m), max(m + 1, h - m),
                         radius, fill=fill, outline=edge)

    def _note_shown(self) -> None:
        """Restart the auto-collapse clock and remember the window in front."""
        self._shown_at = time.time()
        self._fg_at_show = _foreground_window()

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

    def _chevron_zone(self) -> Tuple[float, float]:
        """(x1, x2) of the collapse button, in canvas coordinates."""
        x1, x2 = self.margin, self.margin + self.card_w
        if self.side == "left":
            return x1, x1 + self.gutter
        return x2 - self.gutter, x2

    def _render_card(self) -> None:
        accent, bg, border = self._colors()
        m = self.margin
        x1, y1 = m, m
        x2, y2 = m + self.card_w, m + self.card_h
        mid = (y1 + y2) / 2
        r = self._px(16)
        self._round_rect(x1, y1, x2, y2, r, fill=bg, outline="")

        # The collapse button, drawn as its own region so the boundary between
        # "put this away" and "open the session page" is visible rather than
        # something you discover by clicking the wrong half. Painted on the
        # card's exact rect and BEFORE its outline, so the rounded corners line
        # up with the card's instead of leaving a notch.
        gx1, gx2 = self._chevron_zone()
        on_left = self.side == "left"
        # Rest is a quiet panel that merely marks the boundary; hover is
        # unmistakably the accent, confirming the click will put the card away
        # rather than open the session page.
        zone_fill = (_lerp_color(bg, accent, 0.55) if self._hover_chevron
                     else _lerp_color(bg, border, 0.45))
        self._round_rect(gx1, y1, gx2, y2, r, fill=zone_fill, outline="")
        # Square off the inner side so it reads as a slice of the card rather
        # than a pill floating on it.
        inner = gx1 if on_left else gx2
        self.canvas.create_rectangle(inner - (r if on_left else 0), y1,
                                     inner + (0 if on_left else r), y2,
                                     fill=zone_fill, outline="")
        divider = gx2 if on_left else gx1
        self.canvas.create_line(divider, y1 + self._px(10), divider, y2 - self._px(10),
                                fill=_lerp_color(bg, accent, 0.38),
                                width=max(1, self._px(1)))
        # The card's border last, so it runs unbroken over the button.
        self._round_rect(x1, y1, x2, y2, r, fill="", outline=border)

        chevron_x = (gx1 + gx2) / 2
        if on_left:
            content_x1, text_right = x1 + self.gutter, x2 - self._px(14)
        else:
            content_x1, text_right = x1, gx1 - self._px(12)

        dot_x = content_x1 + self._px(18)
        dr = self._px(5)
        self.canvas.create_oval(dot_x - dr, mid - dr, dot_x + dr, mid + dr,
                                fill=accent, outline="")

        text_x = content_x1 + self._px(34)
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

        self.canvas.create_text(chevron_x, mid, text=CHEVRON.get(self.side, "▸"),
                                fill=INK if self._hover_chevron else MUTED,
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

    def _over_chevron(self, x: float) -> bool:
        if self.collapsed:
            return False
        gx1, gx2 = self._chevron_zone()
        return gx1 <= x <= gx2

    def _on_motion(self, event) -> None:
        hover = self._over_chevron(event.x)
        if hover != self._hover_chevron:
            self._hover_chevron = hover
            if not self._anim:
                self._render()

    def _on_enter(self, _event) -> None:
        self._pointer_in = True
        self._set_alpha(1.0)

    def _on_leave(self, _event) -> None:
        self._pointer_in = False
        self._hover_chevron = False
        self._set_alpha(0.96)
        self._note_shown()   # the countdown to putting itself away starts here
        if not self._anim:
            self._render()

    def _on_press(self, event) -> None:
        self._drag = {"x": event.x_root, "y": event.y_root,
                      "wx": self._x, "wy": self._y, "moved": False}

    def _on_drag(self, event) -> None:
        if self._drag is None or self._anim:
            return
        dx = event.x_root - self._drag["x"]
        dy = event.y_root - self._drag["y"]
        if abs(dx) > self._px(3) or abs(dy) > self._px(3):
            self._drag["moved"] = True

        nx, ny = self._drag["wx"] + dx, self._drag["wy"] + dy
        # Track the nearest edge whether collapsed or not: collapsed, the tab
        # re-docks live so you can see it turn sideways mid-drag; expanded,
        # the chevron flips to point at the edge it would collapse into.
        side = self._nearest_side(nx, ny)
        changed = side != self.side
        self.side = side
        self._place(nx, ny)
        if changed:
            self._render()

    def _on_release(self, event) -> None:
        drag = self._drag
        self._drag = None
        if drag is None:
            return
        if drag["moved"]:
            _save_prefs(x=self._x, y=self._y, side=self.side)
            self._note_shown()
            self._render()  # the tab's rounded end may have swapped sides
            return
        # A click, not a drag. The chevron collapses; the body opens the page
        # where you actually set a session.
        if self.collapsed or self._over_chevron(event.x):
            self.toggle_collapsed()
        else:
            self._open_intention()

    def reset_position(self) -> None:
        """Bring the timer back to a known-good spot.

        The escape hatch for "I can't find it any more" — un-hides, expands,
        and drops it at the default corner of the work area.
        """
        self._anim = None
        self.hidden = False
        self.collapsed = False
        self.side = "right"
        self.win.deiconify()
        self._place(None, None)
        self._render()
        self._note_shown()
        _save_prefs(hidden=False, collapsed=False, side="right",
                    x=self._x, y=self._y)

    def set_hidden(self, hidden: bool) -> None:
        self.hidden = hidden
        _save_prefs(hidden=hidden)
        if hidden:
            self.win.withdraw()
        else:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self._note_shown()
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
        """Open wherever sessions actually get set.

        The hosted dashboard when one is configured — that is the page with
        the session card on it, and going through open_dashboard reuses a tab
        that is already open. The monitor's own localhost /intention page is
        the fallback for setups with no DASHBOARD_URL.
        """
        if self.dashboard_url:
            self._open_dashboard()
            return
        if self.intention_url:
            import webbrowser
            webbrowser.open(self.intention_url)

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
        menu.add_command(label="Reset position", command=self.reset_position)
        menu.add_command(label="Open dashboard", command=self._open_dashboard)
        menu.add_command(label="Hide timer", command=lambda: self.set_hidden(True))
        # An open menu counts as using the card, so it can't fold up underneath
        # the menu the user is reading.
        self._menu_open = True
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
            self._menu_open = False
            self._note_shown()

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


def reset_position() -> None:
    """Un-hide and re-place the timer at its default corner (tray recovery)."""
    tk_host.submit(lambda: _widget and _widget.reset_position())


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
