# monitor/sources.py
from __future__ import annotations

import os
import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.window_capture import get_active_window, get_mode

# pynput needs X11 on Linux / WSL — catch ALL errors
try:
    from pynput import keyboard
except Exception as e:
    keyboard = None
    if "No module" in str(e):
        print("[warn] pynput not installed. Live KeyLogger disabled. pip install pynput")
    else:
        print(f"[warn] pynput cannot start ({type(e).__name__}): display/X11 not available")
        print("[warn] Live KeyLogger disabled. Keystroke capture requires a display server.")


def format_keys_as_text(keys: List[str]) -> str:
    out = []
    for k in keys:
        if k == "Key.space":
            out.append(" ")
        elif k == "Key.enter":
            out.append("[ENTER] ")
        elif str(k).startswith("Key."):
            out.append(f"[{str(k).replace('Key.', '').upper()}]")
        elif isinstance(k, str) and len(k) == 1:
            out.append(k)
        else:
            out.append(f"[{k}]")
    return "".join(out)


# ==================== BASE CLASSES ====================

class EventSource:
    def now_dt(self) -> datetime: ...
    def poll(self) -> Optional[str]: ...
    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]: ...


class KeystrokeSource:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def summary(self, now_dt: datetime, window_seconds: int = 60) -> str: ...
    def typed_text_between(self, start_dt: datetime, end_dt: datetime) -> str: ...


# ==================== LIVE SOURCES (IN-MEMORY) ====================

class LiveEventSource(EventSource):
    """Captures active window titles in memory. No file I/O."""

    TITLE_MAXLEN = 180

    def __init__(self, settings: Settings):
        self.settings = settings
        self._buffer: deque[Tuple[datetime, str]] = deque(maxlen=500)
        self._last_title: str = ""

    def now_dt(self) -> datetime:
        return datetime.now()

    def poll(self) -> Optional[str]:
        """
        Check current window. Append to buffer if title changed.
        Returns the new title, or None if unchanged.
        """
        title = get_active_window()
        if not title:
            return None

        # Trim long titles
        if len(title) > self.TITLE_MAXLEN:
            title = title[:self.TITLE_MAXLEN - 1] + "…"

        win_key = f"WIN:{title}"

        if win_key != self._last_title:
            self._buffer.append((datetime.now(), win_key))
            self._last_title = win_key
            return win_key

        return None

    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]:
        """Return the most recent events as (timestamp_str, key) tuples."""
        items = list(self._buffer)[-limit:]
        return [(dt.strftime("%Y-%m-%d %H:%M:%S"), key) for dt, key in items]


class LiveKeystrokeSource(KeystrokeSource):
    """Captures keystrokes in memory via pynput. No file I/O."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._buffer: deque[Tuple[datetime, str]] = deque(maxlen=5000)
        self._lock = threading.Lock()
        self._listener = None

    def _token_from_key(self, k) -> str:
        try:
            if hasattr(k, "char") and k.char is not None:
                return k.char
            name = getattr(k, "name", None)
            if name:
                return f"Key.{name}"
        except Exception:
            pass
        return str(k)

    def _on_press(self, key):
        token = self._token_from_key(key)
        with self._lock:
            self._buffer.append((datetime.now(), token))

    def start(self) -> None:
        if not keyboard:
            print("[warn] KeyLogger disabled: pynput not available on this platform.")
            return
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        print("[info] KeyLogger started (in-memory)")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def summary(self, now_dt: datetime, window_seconds: int = 60) -> str:
        """Summarize recent keystrokes for the LLM prompt."""
        cutoff = now_dt - timedelta(seconds=window_seconds)

        with self._lock:
            recent = [(dt, k) for dt, k in self._buffer if dt >= cutoff]

        if not recent:
            return "Idle (0 keys)"

        count = len(recent)
        special_keys = set()
        for _, k in recent:
            if len(k) > 1 or k.startswith("Key."):
                special_keys.add(k)

        special_str = ", ".join(list(special_keys)[:5])
        return f"{count} keys typed. Special: [{special_str}]"

    def typed_text_between(self, start_dt: datetime, end_dt: datetime) -> str:
        """Reconstruct typed text between two timestamps."""
        with self._lock:
            keys = [k for dt, k in self._buffer if start_dt < dt <= end_dt]

        return format_keys_as_text(keys)


# ==================== TRACE SOURCES (FILE-BASED, FOR TESTING) ====================

def _load_trace(path: str) -> Tuple[List[Tuple[datetime, str]], Dict[str, str]]:
    """Load trace file. Returns (events, labels_dict).

    labels_dict maps "YYYY-MM-DD HH:MM:SS" -> expected category name.
    Only populated for events that have an "expected" field.
    """
    out: List[Tuple[datetime, str]] = []
    labels: Dict[str, str] = {}
    if not os.path.exists(path):
        return out, labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
                key = str(rec.get("key", ""))
                out.append((ts, key))
                if "expected" in rec:
                    labels[rec["timestamp"]] = rec["expected"]
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    return out, labels


@dataclass
class TraceData:
    win: List[Tuple[datetime, str]]
    keys: List[Tuple[datetime, str]]
    sim_start: datetime
    labels: Dict[str, str]


def load_trace_data(settings: Settings) -> TraceData:
    win, labels = _load_trace(settings.trace.trace_win_path)
    keys, _ = _load_trace(settings.trace.trace_key_path)
    if not win and not keys:
        raise RuntimeError("Trace mode enabled but both trace files are missing/empty.")
    sim_start = min([x[0] for x in (win[:1] + keys[:1]) if x] or [datetime.now()])
    return TraceData(win=win, keys=keys, sim_start=sim_start, labels=labels)


@dataclass
class TraceSession:
    trace: TraceData
    wall_start: float
    speed: float

    def now_dt(self) -> datetime:
        elapsed = (time.time() - self.wall_start) * self.speed
        return self.trace.sim_start + timedelta(seconds=elapsed)


class TraceEventSource(EventSource):
    def __init__(self, session: TraceSession):
        self.session = session
        self.sim_start = session.trace.sim_start

    def now_dt(self) -> datetime:
        return self.session.now_dt()

    def poll(self) -> Optional[str]:
        """No-op for trace mode — events come from the file."""
        return None

    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]:
        eligible = [x for x in self.session.trace.win if x[0] <= now_dt]
        tail = eligible[-limit:]
        return [(dt.strftime("%Y-%m-%d %H:%M:%S"), key) for dt, key in tail]

    def event_times(self):
        return sorted({dt for (dt, _key) in self.session.trace.win})
    
    def get_label(self, dt: datetime) -> Optional[str]:
        """Get the expected label for a timestamp, or None if unlabeled."""
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        return self.session.trace.labels.get(ts_str)


class TraceKeystrokeSource(KeystrokeSource):
    def __init__(self, session: TraceSession):
        self.session = session

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def summary(self, now_dt: datetime, window_seconds: int = 60) -> str:
        window_start = now_dt - timedelta(seconds=window_seconds)
        keys = [k for (dt, k) in self.session.trace.keys if window_start <= dt <= now_dt]
        if not keys:
            return "Idle (0 keys)"
        specials = {k for k in keys if (len(str(k)) > 1 or str(k).startswith("Key."))}
        return f"{len(keys)} keys typed. Special: [{', '.join(list(specials)[:5])}]"

    def typed_text_between(self, start_dt: datetime, end_dt: datetime) -> str:
        keys = [k for (dt, k) in self.session.trace.keys if start_dt < dt <= end_dt]
        return format_keys_as_text(keys)


# ==================== FACTORY ====================

def make_sources(
    use_trace: Optional[bool] = None,
    settings: Settings = DEFAULT_SETTINGS,
):
    if use_trace is None:
        use_trace = settings.trace.use_trace_file

    if use_trace:
        trace = load_trace_data(settings)
        session = TraceSession(trace=trace, wall_start=time.time(), speed=settings.trace.trace_speed)
        return TraceEventSource(session), TraceKeystrokeSource(session)

    print(f"[info] Window capture mode: {get_mode()}")
    return LiveEventSource(settings), LiveKeystrokeSource(settings)