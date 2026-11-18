# monitor/sources.py
from __future__ import annotations

import os
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.window_capture import get_active_window, get_mode

# The monitor reads window titles and, for the vision second look, the
# screen. It does not read the keyboard: there is no keystroke source here,
# and nothing anywhere in the app should start one. Idle detection
# (monitor/idle.py) asks the OS for the time since the last input and never
# sees which keys were pressed.


# ==================== BASE CLASSES ====================

class EventSource:
    def now_dt(self) -> datetime: ...
    def poll(self) -> Optional[str]: ...
    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]: ...
    def reset(self) -> None: ...


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

    def reset(self) -> None:
        """Drop captured history and forget the last title.

        Used when monitoring resumes after a break: clearing `_last_title` is
        the important half — without it, a user still sitting on the same
        window as before the break produces no change, so poll() appends
        nothing and the loop never reaches min_events again.
        """
        self._buffer.clear()
        self._last_title = ""


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
    sim_start: datetime
    labels: Dict[str, str]


def load_trace_data(settings: Settings) -> TraceData:
    win, labels = _load_trace(settings.trace.trace_win_path)
    if not win:
        raise RuntimeError("Trace mode enabled but the window trace file is missing/empty.")
    return TraceData(win=win, sim_start=win[0][0], labels=labels)


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

    def reset(self) -> None:
        """No-op: trace events are replayed from the file, not captured."""
        return

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


# ==================== FACTORY ====================

def make_sources(
    use_trace: Optional[bool] = None,
    settings: Settings = DEFAULT_SETTINGS,
) -> EventSource:
    """The window-title source for this run: replayed from a trace file in
    trace mode, read from the foreground window live."""
    if use_trace is None:
        use_trace = settings.trace.use_trace_file

    if use_trace:
        trace = load_trace_data(settings)
        session = TraceSession(trace=trace, wall_start=time.time(), speed=settings.trace.trace_speed)
        return TraceEventSource(session)

    print(f"[info] Window capture mode: {get_mode()}")
    return LiveEventSource(settings)