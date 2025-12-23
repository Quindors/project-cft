# monitor/sources.py
from __future__ import annotations

import os
import json
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from monitor.config import (
    LOG_DIR,
    KEY_LOG_DIR,
    TRACE_WIN_PATH,
    TRACE_KEY_PATH,
    TRACE_SPEED,
)

# optional dependency for live key capture
try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    print("[warn] pynput not installed. KeyLogger disabled. pip install pynput")


# ===================== helpers =====================

def _tail_lines(path: str, max_lines: int = 300) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()[-max_lines:]
    except Exception:
        return []

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


# ===================== KeyLogger (live mode) =====================

class KeyLogger:
    def __init__(self, log_dir: str):
        self.buffer = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_day = datetime.now().date()
        self.curr_path = self._daily_log_path(datetime.now())

        self.listener = None
        self.flush_thread = None

    def _daily_log_path(self, dt: datetime) -> str:
        return os.path.join(self.log_dir, f"keystrokes_{dt:%Y-%m-%d}.jsonl")

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

    def on_press(self, key):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        token = self._token_from_key(key)
        with self.lock:
            self.buffer.append({"timestamp": ts, "key": token})

    def _flush_loop(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)

            with self.lock:
                records = self.buffer[:]
                self.buffer.clear()

            now = datetime.now()
            if now.date() != self.current_day:
                self.current_day = now.date()
                self.curr_path = self._daily_log_path(now)

            if not records:
                continue

            try:
                with open(self.curr_path, "a", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[error] keylog flush: {e}")

    def start(self):
        if not keyboard:
            print("[warn] KeyLogger disabled: pynput not available.")
            return
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print(f"[info] KeyLogger started. Logs in {self.log_dir}")

    def stop(self):
        if not keyboard:
            return
        if self.listener:
            self.listener.stop()
        self.stop_event.set()
        if self.flush_thread:
            self.flush_thread.join(timeout=2.0)


# ===================== Events: Live + Trace =====================

def _todays_windows_log_path(now_dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"windows_{now_dt:%Y-%m-%d}.jsonl")

def _read_last_events_file(path: str, limit: int) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        return events
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        for line in lines:
            try:
                rec = json.loads(line)
                ts, key = rec.get("timestamp"), rec.get("key")
                if ts and key:
                    events.append((ts, key))
            except Exception:
                continue
    except Exception:
        return []
    events.sort(key=lambda x: x[0])
    return events[-limit:]


@dataclass
class TraceData:
    win: List[Tuple[datetime, str]]
    keys: List[Tuple[datetime, str]]
    sim_start: datetime


def _load_trace(path: str) -> List[Tuple[datetime, str]]:
    out: List[Tuple[datetime, str]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
                key = str(rec.get("key", ""))
                out.append((ts, key))
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    return out


def load_trace_data() -> TraceData:
    win = _load_trace(TRACE_WIN_PATH)
    keys = _load_trace(TRACE_KEY_PATH)
    if not win and not keys:
        raise RuntimeError("Trace mode enabled but both trace files are missing/empty.")
    sim_start = min([x[0] for x in (win[:1] + keys[:1]) if x] or [datetime.now()])
    return TraceData(win=win, keys=keys, sim_start=sim_start)


@dataclass
class TraceSession:
    trace: TraceData
    wall_start: float

    @property
    def sim_start(self) -> datetime:
        return self.trace.sim_start

    def now_dt(self) -> datetime:
        elapsed = (time.time() - self.wall_start) * TRACE_SPEED
        return self.trace.sim_start + timedelta(seconds=elapsed)


class EventSource:
    def now_dt(self) -> datetime: ...
    def now_str(self) -> str: ...
    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]: ...


class LiveEventSource(EventSource):
    def now_dt(self) -> datetime:
        return datetime.now()

    def now_str(self) -> str:
        return self.now_dt().strftime("%Y-%m-%d %H:%M:%S")

    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]:
        path = _todays_windows_log_path(now_dt)
        return _read_last_events_file(path, limit)


class TraceEventSource(EventSource):
    def __init__(self, session: TraceSession):
        self.session = session

    @property
    def sim_start(self) -> datetime:
        return self.session.sim_start

    def now_dt(self) -> datetime:
        return self.session.now_dt()

    def now_str(self) -> str:
        return self.now_dt().strftime("%Y-%m-%d %H:%M:%S")

    def last_events(self, now_dt: datetime, limit: int) -> List[Tuple[str, str]]:
        eligible = [x for x in self.session.trace.win if x[0] <= now_dt]
        tail = eligible[-limit:]
        return [(dt.strftime("%Y-%m-%d %H:%M:%S"), key) for dt, key in tail]


# ===================== Keystrokes: Live + Trace =====================

class KeystrokeSource:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def summary(self, now_dt: datetime, window_seconds: int = 60) -> str: ...
    def typed_text_between(self, start_dt: datetime, end_dt: datetime) -> str: ...


class LiveKeystrokeSource(KeystrokeSource):
    def __init__(self, log_dir: str = KEY_LOG_DIR, enable_logger: bool = True):
        self.log_dir = log_dir
        self.logger = KeyLogger(log_dir) if enable_logger else None

    def start(self) -> None:
        if self.logger:
            self.logger.start()

    def stop(self) -> None:
        if self.logger:
            self.logger.stop()

    def _path_for_day(self, now_dt: datetime) -> str:
        return os.path.join(self.log_dir, f"keystrokes_{now_dt:%Y-%m-%d}.jsonl")

    def summary(self, now_dt: datetime, window_seconds: int = 60) -> str:
        path = self._path_for_day(now_dt)
        if not os.path.exists(path):
            return ""

        threshold = time.time() - window_seconds
        lines = _tail_lines(path, max_lines=500)

        count = 0
        special_keys = set()

        for line in reversed(lines):
            try:
                rec = json.loads(line)
                dt = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
                if dt.timestamp() < threshold:
                    break
                count += 1
                key = rec.get("key", "")
                if key and (len(key) > 1 or str(key).startswith("Key.")):
                    special_keys.add(str(key))
            except Exception:
                continue

        if count == 0:
            return "Idle (0 keys)"
        special_str = ", ".join(list(special_keys)[:5])
        return f"{count} keys typed. Special: [{special_str}]"

    def typed_text_between(self, start_dt: datetime, end_dt: datetime) -> str:
        path = self._path_for_day(end_dt)
        if not os.path.exists(path):
            return ""

        lines = _tail_lines(path, max_lines=2000)
        keys_interval: List[str] = []
        for line in lines:
            try:
                rec = json.loads(line)
                dt = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
                if start_dt < dt <= end_dt:
                    keys_interval.append(rec.get("key", ""))
            except Exception:
                continue

        return format_keys_as_text(keys_interval)


class TraceKeystrokeSource(KeystrokeSource):
    def __init__(self, session: TraceSession):
        self.session = session

    def start(self) -> None:
        # no-op
        return

    def stop(self) -> None:
        # no-op
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


# ===================== factories =====================

def make_sources(use_trace: bool):
    """
    Returns (event_source, keystroke_source)
    """
    if use_trace:
        trace = load_trace_data()
        session = TraceSession(trace=trace, wall_start=time.time())
        return TraceEventSource(session), TraceKeystrokeSource(session)

    return LiveEventSource(), LiveKeystrokeSource(KEY_LOG_DIR, enable_logger=True)