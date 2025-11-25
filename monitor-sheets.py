#!/usr/bin/env python3
# focus_monitor_to_sheets_daily_tabs_buffered.py
# pip install gspread google-auth openai

import os, json, time, ctypes, re, random
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict, Literal
import gspread
from google.oauth2.service_account import Credentials
import openai
from gspread.exceptions import APIError

# ===================== MONITOR CONFIG =====================
LOG_DIR = r".\logs"               # where windows_YYYY-MM-DD.jsonl lives (from your window/url logger)
MODEL = "gpt-4o-mini"
INTERVAL_SEC    = 3
BATCH_FLUSH_MAX = 20              # ~1 flush/minute at steady state
IDLE_FLUSH_SEC  = 20.0
APPEND_ONLY     = True            # if you don't need per-ts upserts
MAX_EVENTS = 4                    # total events to send (from windows/URLs)

# ---- OFF-TASK threshold (single) ----
OFF_THRESHOLD = 0.60              # OFF_SCORE >= this → OFF-TASK
# ====================================

# ===================== SHEETS CONFIG ======================
SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
WORKSHEET_PREFIX = "Focus Logs"   # daily tabs: "<prefix> - YYYY-MM-DD"
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
MAX_EVENT_COLS = 5                # flatten up to N events: e{i}_ts, e{i}_key

# --------- WRITE THROTTLING / BUFFERING (tune here) ---------
APPEND_ONLY: bool = False         # True = never update existing ts rows; fastest/lowest quota
BATCH_FLUSH_MAX: int = 15         # flush after this many buffered rows
IDLE_FLUSH_SEC: float = 15.0      # or when buffer is idle this long
MAX_RETRIES: int = 6              # retries on 429/5xx
BACKOFF_BASE: float = 0.8         # seconds, exponential with jitter

# ============================================================

last_alert_ts: Optional[str] = None

# ------------------------ OpenAI --------------------------
def require_api_key():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

def build_prompt(events: List[Tuple[str, str]]) -> str:
    tokens = [k for _, k in events]
    return (
        "You are a focus and productivity judge. Look at the *pattern* of recent active window titles.\n\n"
        "Event format:\n"
        "- WIN:<window title>\n\n"
        "Goal: Determine how OFF-TASK the user is based on intent.\n"
        "Potentially distractive apps or titles include (non-exhaustive): Discord, YouTube, Reddit, Twitter, Instagram, TikTok, Netflix, Twitch.\n\n"
        "Rules (intent-based):\n"
        "1) If the title clearly shows *work or learning intent*, treat as ON-TASK.\n"
        "2) Use surrounding context (last ~5 events) for tie-breaks.\n"
        "3) Ambiguity rule: prefer ON-TASK if context suggests work.\n"
        "4) Weigh most recent windows higher.\n\n"
        "Output (strict, single line):\n"
        "  LABEL=<ON-TASK or OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<short explanation if OFF-TASK, else brief note>\n\n"
        "Recent activity (oldest → newest):\n" + ", ".join(tokens)
    )

def ask_gpt(prompt: str, model: str) -> Tuple[str, str, Optional[float], Optional[float]]:
    """
    Returns:
      label: 'ON-TASK' or 'OFF-TASK' (model's raw label)
      reason: short explanation (may be empty)
      off_score: float in [0,1] (higher = more off-task)
      confidence: model's self-reported confidence in its judgment
    """
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()

        m_label = re.search(r"LABEL\s*=\s*(ON-TASK|OFF-TASK)", content, flags=re.I)
        if m_label:
            label = m_label.group(1).upper()
        else:
            upper = content.upper()
            if "OFF-TASK" in upper:
                label = "OFF-TASK"
            elif "ON-TASK" in upper:
                label = "ON-TASK"
            else:
                label = "[WARN]"

        m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
        off_score: Optional[float] = None
        if m_off:
            try:
                val = float(m_off.group(1))
                if 0.0 <= val <= 1.0:
                    off_score = val
            except Exception:
                off_score = None

        m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
        confidence: Optional[float] = None
        if m_conf:
            try:
                val = float(m_conf.group(1))
                if 0.0 <= val <= 1.0:
                    confidence = val
            except Exception:
                confidence = None

        reason = ""
        m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
        if m_reason:
            reason = m_reason.group(1).strip()

        if off_score is None:
            off_score = 0.8 if label == "OFF-TASK" else 0.2
        if confidence is None:
            confidence = 0.5

        return label, reason, off_score, confidence

    except Exception as e:
        return "[ERROR]", str(e), None, None

# ----------------------- Windows Log ----------------------
def todays_log_path() -> str:
    return os.path.join(LOG_DIR, f"windows_{datetime.now():%Y-%m-%d}.jsonl")

def read_last_events(path: str, limit: int) -> List[Tuple[str, str]]:
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
        pass
    events.sort(key=lambda x: x[0])
    return events[-limit:]

# ----------------------- UI: Popup ------------------------
def show_popup(message: str, title: str = "Productivity Alert"):
    MB_ICONINFORMATION = 0x40
    MB_SETFOREGROUND  = 0x00010000
    MB_TOPMOST        = 0x00040000
    flags = MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)

# --------------------- Google Sheets ----------------------
def _open_client_and_spreadsheet():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(SHEET_URL_OR_KEY) if SHEET_URL_OR_KEY.startswith("http") else gc.open_by_key(SHEET_URL_OR_KEY)
    return gc, sh

def _expected_headers() -> List[str]:
    # Column layout:
    # ts | label | confidence | ai_reason | human_label | human_reason | e1_key | e2_key | ...
    base = [
        "ts",
        "label",
        "confidence",
        "ai_reason",
        "human_label",
        "human_reason",
    ]
    for i in range(1, MAX_EVENT_COLS + 1):
        base.append(f"e{i}_key")
    return base

def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def _ensure_headers(ws) -> None:
    headers = _expected_headers()
    end_col = _col_letter(len(headers))
    ws.update(f"A1:{end_col}1", [headers], value_input_option="RAW")

def _get_or_create_daily_ws(sh, day: date):
    title = f"{WORKSHEET_PREFIX} - {day:%Y-%m-%d}"
    try:
        ws = sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=title,
            rows="200",
            cols=str(len(_expected_headers())),
            index=0,
        )
        _ensure_headers(ws)
    return ws

def _read_existing_ts(ws) -> Dict[str, int]:
    headers = _expected_headers()
    idx_ts = headers.index("ts")
    values = ws.get_all_values()
    key_to_row: Dict[str, int] = {}
    for i, r in enumerate(values[1:], start=2):
        key = (r[idx_ts] if idx_ts < len(r) else "").strip()
        if key:
            key_to_row[key] = i
    return key_to_row

def _flatten_row(record: Dict) -> List[str]:
    def get(k, default=""):
        v = record.get(k, default)
        return "" if v is None else str(v)

    # ts | label | confidence | ai_reason | human_label | human_reason
    row: List[str] = [
        get("ts"),
        get("label"),
        get("confidence"),
        get("reason"),   # goes under ai_reason
        "",              # human_label (left empty)
        "",              # human_reason (left empty)
    ]

    # Event columns: e{i}_key only
    events = record.get("events") or []
    for i in range(MAX_EVENT_COLS):
        if i < len(events) and isinstance(events[i], dict):
            row.append(str(events[i].get("key", "")) or "")
        else:
            row.append("")

    return row

def _retryable_call(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            code = getattr(e, "response", {}).get("status", None)
            msg = str(e)
            if "429" in msg or (code in (429, 500, 502, 503, 504)):
                delay = (BACKOFF_BASE * (2 ** attempt)) + random.random()
                print(f"[warn] Sheets API throttled (attempt {attempt+1}/{MAX_RETRIES}), sleeping {delay:.2f}s")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Exceeded max retries for Sheets API call")

def _flush_buffer(ws, key_to_row: Dict[str, int], buffer_rows: List[List[str]]):
    """Efficient batch flush with minimal API calls."""
    if not buffer_rows:
        return {"updated": 0, "appended": 0}

    if APPEND_ONLY:
        _retryable_call(ws.append_rows, buffer_rows, value_input_option="USER_ENTERED")
        start_row_guess = len(key_to_row) + 2
        for row in buffer_rows:
            ts = (row[0] or "").strip()
            if ts and ts not in key_to_row:
                key_to_row[ts] = start_row_guess
                start_row_guess += 1
        return {"updated": 0, "appended": len(buffer_rows)}

    headers = _expected_headers()
    idx_ts = headers.index("ts")
    end_col = _col_letter(len(headers))

    updates: List[Tuple[int, List[str]]] = []
    appends: List[List[str]] = []

    for row in buffer_rows:
        key = (row[idx_ts] or "").strip()
        if key and key in key_to_row:
            updates.append((key_to_row[key], row))
        else:
            appends.append(row)

    for row_idx, values in updates:
        _retryable_call(ws.update, f"A{row_idx}:{end_col}{row_idx}", [values], value_input_option="USER_ENTERED")

    if appends:
        _retryable_call(ws.append_rows, appends, value_input_option="USER_ENTERED")
        start_row_guess = max(key_to_row.values(), default=1) + 1
        for row in appends:
            ts = (row[idx_ts] or "").strip()
            if ts and ts not in key_to_row:
                key_to_row[ts] = start_row_guess
                start_row_guess += 1

    return {"updated": len(updates), "appended": len(appends)}

# ------------------------- Main ---------------------------
def main():
    require_api_key()

    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise SystemExit(f"[fatal] Service account JSON not found: {SERVICE_ACCOUNT_JSON}")

    gc, sh = _open_client_and_spreadsheet()

    current_day = date.today()
    ws = _get_or_create_daily_ws(sh, current_day)
    key_to_row = _read_existing_ts(ws)
    print(f"[info] connected to sheet; tab='{ws.title}', existing rows tracked: {len(key_to_row)}")

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[info] watching {LOG_DIR}, checking every {INTERVAL_SEC}s")

    global last_alert_ts

    # NEW: track last events signature (sequence of keys) so we only log on change
    last_events_signature: Optional[Tuple[str, ...]] = None

    buffer: List[List[str]] = []
    last_flush_time = time.time()

    current_state: str = "ON-TASK"

    while True:
        loop_start = time.time()
        # Daily rollover
        today = date.today()
        if today != current_day:
            if buffer:
                _ensure_headers(ws)
                _flush_buffer(ws, key_to_row, buffer)
                buffer.clear()
            current_day = today
            ws = _get_or_create_daily_ws(sh, current_day)
            key_to_row = _read_existing_ts(ws)
            last_written_newest_ts = None
            last_events_signature = None  # reset for new tab/day
            print(f"[info] rolled over → new tab '{ws.title}'")
            last_flush_time = time.time()

        path = todays_log_path()
        events = read_last_events(path, MAX_EVENTS)

        events_for_decision = [e for e in events if not last_alert_ts or e[0] > last_alert_ts]

        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_before = last_alert_ts

        if not events_for_decision:
            decision_raw, reason, off_score, confidence = "[idle]", f"no new events in {path}", None, None
            newest_ts = events[-1][0] if events else None
        else:
            prompt = build_prompt(events_for_decision)
            decision_raw, reason, off_score, confidence = ask_gpt(prompt, MODEL)
            newest_ts = events_for_decision[-1][0]

            off_val = off_score if off_score is not None else 0.5
            prev_state = current_state
            current_state = "OFF-TASK" if off_val >= OFF_THRESHOLD else "ON-TASK"

            if prev_state == "ON-TASK" and current_state == "OFF-TASK":
                show_popup("You're drifting OFF-TASK!\n" + (reason or ""))
                last_alert_ts = newest_ts
                print(f"{ts_now}  OFF-TASK (off={off_val:.2f}, conf={confidence if confidence is not None else 'n/a'}) | {reason}")
            else:
                print(f"{ts_now}  {current_state} (off={off_val:.2f}, conf={confidence if confidence is not None else 'n/a'})")

        if decision_raw == "[idle]":
            label_to_log = decision_raw
        else:
            label_to_log = current_state

        record = {
            "ts": ts_now,
            "label": label_to_log,
            "reason": reason or "",
            "confidence": confidence if confidence is not None else "",
            "events": [{"timestamp": ts, "key": key} for ts, key in events_for_decision] if events_for_decision else []
        }

        if label_to_log != "[idle]":
            # ---------- ONLY LOG WHEN EVENTS CHANGE ----------
            # Build an events signature based on the keys of up to MAX_EVENT_COLS events
            events_list = record["events"] or []
            trimmed = events_list[:MAX_EVENT_COLS]
            keys = []
            for e in trimmed:
                if isinstance(e, dict):
                    keys.append(str(e.get("key", "")) or "")
                else:
                    keys.append("")
            # pad so signature length is stable
            while len(keys) < MAX_EVENT_COLS:
                keys.append("")
            current_events_signature = tuple(keys)

            if last_events_signature is not None and current_events_signature == last_events_signature:
                # Events haven't changed → skip logging this row
                # print("[debug] skipping row, events unchanged")
                pass
            else:
                row = _flatten_row(record)
                buffer.append(row)
                last_events_signature = current_events_signature

        now = time.time()
        should_flush = len(buffer) >= BATCH_FLUSH_MAX or (buffer and (now - last_flush_time) >= IDLE_FLUSH_SEC)
        if should_flush:
            try:
                _ensure_headers(ws)
                result = _flush_buffer(ws, key_to_row, buffer)
                print(f"[flush] updated={result['updated']} appended={result['appended']} (sent {len(buffer)})")
                buffer.clear()
                last_flush_time = now
            except Exception as e:
                print(f"[error] flush failed: {e}")

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")
