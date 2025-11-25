#!/usr/bin/env python3
import os, threading, time, json, signal
from datetime import datetime
from typing import List
from pynput import keyboard

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def daily_log_path(dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"keystrokes_{dt.strftime('%Y-%m-%d')}.jsonl")

# --- State ---
BUFFER: List[dict] = []
BUFFER_LOCK = threading.Lock()
FLUSH_INTERVAL_SEC = 1.0
STOP_EVENT = threading.Event()

_current_day = datetime.now().date()
_current_path = daily_log_path(datetime.now())

def _token_from_key(k) -> str:
    try:
        if hasattr(k, "char") and k.char is not None:
            return k.char
        name = getattr(k, "name", None)
        if name:
            return f"Key.{name}"
    except Exception:
        pass
    return str(k)

def on_press(key):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token = _token_from_key(key)
    with BUFFER_LOCK:
        BUFFER.append({"timestamp": ts, "key": token})

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
    # Windows Ctrl+Break
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)

def _start_hotkey_kill():
    # Global hotkey: Ctrl+Shift+Q → STOP_EVENT.set()
    def kill():
        STOP_EVENT.set()
    t = threading.Thread(
        target=lambda: keyboard.GlobalHotKeys({
            '<ctrl>+<shift>+q': kill
        }).__enter__() or keyboard.Listener().join(),  # keep context alive
        daemon=True
    )
    t.start()

def main():
    print(f"Keystroke logger (JSONL). Writing daily files in: {LOG_DIR}")
    print("[stop] Press Ctrl+Shift+Q to stop (works even if Ctrl+C is captured).")

    _install_signal_handlers()
    _start_hotkey_kill()

    t_flush = threading.Thread(target=_flush_loop, daemon=True)
    t_flush.start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while not STOP_EVENT.is_set():
            time.sleep(0.1)
    finally:
        listener.stop()
        STOP_EVENT.set()
        t_flush.join()
        _final_flush()
        print("[info] Stopped cleanly.")

if __name__ == "__main__":
    main()