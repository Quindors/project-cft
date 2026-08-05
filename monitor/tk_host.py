# monitor/tk_host.py
"""One Tk main loop for the whole app, on its own thread.

The monitor is a headless loop with a tray icon; it has no UI thread of its
own. Two features need one — the floating timer widget and the full-screen
focus alert — and Tk allows exactly one interpreter per process driven from a
single thread. So this module owns that thread: it creates a hidden root,
runs mainloop() forever, and hands out `submit()` for other threads to queue
work onto it.

Never touch a Tk object from outside; go through submit(). The queue is
drained by a repeating `after` callback rather than calling `root.after` from
foreign threads, which Tcl does not actually promise is safe.

Everything degrades quietly: on a box with no display or no tkinter (dev on
WSL, headless CI), available() is False and callers fall back — the monitor
itself never depends on the UI being there.
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Callable, Optional

_DRAIN_MS = 40

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_root = None
_ready = threading.Event()
_failed = False
_calls: "queue.Queue[Callable[[], None]]" = queue.Queue()


def _drain() -> None:
    """Run queued callbacks on the Tk thread, then reschedule."""
    while True:
        try:
            fn = _calls.get_nowait()
        except queue.Empty:
            break
        try:
            fn()
        except Exception as e:
            print(f"[ui] callback failed ({type(e).__name__}: {e})")
    if _root is not None:
        _root.after(_DRAIN_MS, _drain)


def _enable_dpi_awareness() -> None:
    """Tell Windows we draw in real pixels, before any window exists.

    A DPI-unaware process gets a virtualized screen: on a 150% display it
    renders at 1280x720 and Windows bitmap-stretches the result, which is why
    text in an unaware Tk window looks soft. Declaring awareness hands us the
    panel's true resolution — callers then scale their layout by ui_scale()
    to keep the same physical size, but every glyph is drawn sharp.

    Must run before the first window is created; afterwards it is a no-op.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    # These report failure by RETURN VALUE, not by raising. Declaring argtypes
    # matters: DPI_AWARENESS_CONTEXT is a pointer-sized handle, and passing the
    # negative constant without it hands the API a 32-bit int, which quietly
    # fails and leaves the process virtualized (blurry) with no error anywhere.
    try:  # Windows 10 1703+: per-monitor v2, handles dragging between screens
        fn = ctypes.windll.user32.SetProcessDpiAwarenessContext
        fn.argtypes = [wintypes.HANDLE]
        fn.restype = wintypes.BOOL
        if fn(wintypes.HANDLE(-4)):
            return
    except Exception:
        pass
    try:  # Windows 8.1+: per-monitor v1 (0 == S_OK)
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except Exception:
        pass
    try:  # Vista+: system-wide awareness
        if ctypes.windll.user32.SetProcessDPIAware():
            return
    except Exception:
        pass
    print("[ui] could not claim DPI awareness; the timer may look soft on a "
          "scaled display")


def ui_scale(widget=None) -> float:
    """Display scale factor (1.5 at 150%), for sizing a pixel layout.

    Falls back to 1.0 when the DPI can't be read, which is the correct
    no-scaling answer rather than a guess.
    """
    target = widget if widget is not None else _root
    if target is None:
        return 1.0
    try:
        dpi = float(target.winfo_fpixels("1i"))
    except Exception:
        return 1.0
    if not (48.0 <= dpi <= 480.0):   # nonsense reading; don't scale by it
        return 1.0
    return dpi / 96.0


def _run() -> None:
    global _root, _failed
    try:
        import tkinter as tk

        _enable_dpi_awareness()
        _root = tk.Tk()
        _root.withdraw()          # the root itself is never shown
        _root.title("cft")
        _root.after(_DRAIN_MS, _drain)
    except Exception as e:
        _failed = True
        print(f"[ui] no Tk display available ({type(e).__name__}: {e}); "
              f"timer widget and blocking alerts are off")
        _ready.set()
        return

    _ready.set()
    try:
        _root.mainloop()
    except Exception as e:
        print(f"[ui] Tk main loop stopped ({type(e).__name__}: {e})")
    finally:
        _failed = True
        _root = None


def start(timeout: float = 5.0) -> bool:
    """Start the Tk thread if it isn't running. True if a UI is usable."""
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            if _failed and _thread is not None:
                return False  # already tried and there is no display
            _ready.clear()
            _thread = threading.Thread(target=_run, name="tk-ui", daemon=True)
            _thread.start()
    _ready.wait(timeout)
    return available()


def available() -> bool:
    return _root is not None and not _failed


def root():
    """The hidden Tk root. Only valid on the Tk thread."""
    return _root


def submit(fn: Callable[[], None]) -> bool:
    """Queue `fn` to run on the Tk thread. False if there is no UI."""
    if not available():
        return False
    _calls.put(fn)
    return True


def stop() -> None:
    """Tear down the Tk thread (called on shutdown)."""
    if not available():
        return
    def _quit():
        try:
            _root.quit()
        except Exception:
            pass
    submit(_quit)
