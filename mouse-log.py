#!/usr/bin/env python3
import os, threading, time, json, signal
from datetime import datetime
from typing import List, Literal, Optional
from pynput import mouse, keyboard

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def daily_log_path(dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"mouse_{dt.strftime('%Y-%m-%d')}.jsonl")

# --- Config ---
FLUSH_INTERVAL_SEC = 1.0
MOVE_SAMPLE_MS = int(os.getenv("MOUSE_MOVE_SAMPLE_MS", "100"))

# --- State ---
BUFFER: List[dict] = []
BUFFER_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

_current_day = datetime.now().date()
_current_path = daily_log_path(datetime.now())
_last_move_ts_ms: Optional[int] = None

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append_record(x: int, y: int, state: Literal["move", "down", "up", "scroll"]) -> None:
    with BUFFER_LOCK:
        BUFFER.append({"timestamp": _now_ts(), "x": int(x), "y": int(y), "state": state})

def on_move(x, y):
    global _last_move_ts_ms
    if MOVE_SAMPLE_MS <= 0:
        _append_record(x, y, "move"); return
    now_ms = int(time.perf_counter() * 1000)
    if _last_move_ts_ms is None or (now_ms - _last_move_ts_ms) >= MOVE_SAMPLE_MS:
        _last_move_ts_ms = now_ms
        _append_record(x, y, "move")

def on_click(x, y, button, pressed):
    _append_record(x, y, "down" if pressed else "up")

def on_scroll(x, y, dx, dy):
    _append_record(x, y, "scroll")

def _flush_loop():
    global _current_day, _current_path
    while not STOP_EVENT.is_set():
        time.sleep(FLUSH_INTERVAL_SEC)
        with BUFFER_LOCK:
            records = BUFFER.copy()
            BUFFER.clear()

        now = datetime.now()
        if now.date() != _current_day:
            _current_day = now.date()
            _current_path = daily_log_path(now)

        if not records:
            continue

        os.makedirs(os.path.dirname(_current_path), exist_ok=True)
        with open(_current_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _final_flush():
    with BUFFER_LOCK:
        records = BUFFER.copy()
        BUFFER.clear()
    if records:
        with open(_current_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _install_signal_handlers():
    def _sig(_sig, _frm):
        STOP_EVENT.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)

def _start_hotkey_kill():
    def kill():
        STOP_EVENT.set()
    t = threading.Thread(
        target=lambda: keyboard.GlobalHotKeys({
            '<ctrl>+<shift>+q': kill
        }).__enter__() or keyboard.Listener().join(),
        daemon=True
    )
    t.start()

def main():
    print(f"Mouse logger (JSONL). Writing daily files in: {LOG_DIR}")
    print(f"[info] Move sampling: {MOVE_SAMPLE_MS} ms (set MOUSE_MOVE_SAMPLE_MS=0 to disable)")
    print("[stop] Press Ctrl+Shift+Q to stop (works even if Ctrl+C is captured).")

    _install_signal_handlers()
    _start_hotkey_kill()

    t_flush = threading.Thread(target=_flush_loop, daemon=True)
    t_flush.start()

    m_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    m_listener.start()
    try:
        while not STOP_EVENT.is_set():
            time.sleep(0.1)
    finally:
        m_listener.stop()
        STOP_EVENT.set()
        t_flush.join()
        _final_flush()
        print("[info] Stopped cleanly.")

if __name__ == "__main__":
    main()