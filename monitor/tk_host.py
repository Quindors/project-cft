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


def _run() -> None:
    global _root, _failed
    try:
        import tkinter as tk

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
