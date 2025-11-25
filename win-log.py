#!/usr/bin/env python3
# Active window logger (Windows-only)
# Output JSONL (compatible with your focus monitor):
#   {"timestamp":"YYYY-MM-DD HH:MM:SS","key":"WIN:<title>"}

import os, time, json
from datetime import datetime
import pyautogui

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def daily_log_path(dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"windows_{dt.strftime('%Y-%m-%d')}.jsonl")

# ---------- tunables ----------
POLL_SEC = float(os.getenv("BL_POLL_SEC", "0.7"))        # loop cadence
TITLE_MAXLEN = int(os.getenv("BL_TITLE_MAXLEN", "180"))  # trim very long titles

# ---------- state ----------
_current_day = datetime.now().date()
_current_path = daily_log_path(datetime.now())
last_handle = None
last_win_key = ""

# ---------- utils ----------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _clean(s: str, maxlen: int) -> str:
    s = " ".join((s or "").split())
    return (s[:maxlen - 1] + "…") if (len(s) > maxlen) else s

def _append_record(key_text: str):
    """Append one log record to today's JSONL file."""
    global _current_day, _current_path
    rec = {"timestamp": _now_str(), "key": key_text}
    now = datetime.now()
    if now.date() != _current_day:
        _current_day = now.date()
        _current_path = daily_log_path(now)
    with open(_current_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- main ----------
def main():
    global last_handle, last_win_key
    print(f"[info] Active window logger → {LOG_DIR}\\windows_YYYY-MM-DD.jsonl")

    try:
        while True:
            aw = pyautogui.getActiveWindow()
            if aw is None:
                time.sleep(POLL_SEC)
                continue

            hwnd = getattr(aw, "_hWnd", None)
            title = _clean(aw.title or "", TITLE_MAXLEN)

            # Always check for title changes, not just handle changes
            if title:
                win_key = f"WIN:{title}"
                if win_key != last_win_key:
                    _append_record(win_key)
                    last_win_key = win_key

            last_handle = hwnd
            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\n[info] Stopped cleanly.")

if __name__ == "__main__":
    main()