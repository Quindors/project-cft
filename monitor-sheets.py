#!/usr/bin/env python3
# focus_monitor_to_sheets_daily_tabs_buffered.py
# pip install gspread google-auth openai python-dotenv

import os, json, time, ctypes, re, random, base64, tempfile, gspread, openai
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
from dotenv import load_dotenv

load_dotenv()

b64 = os.getenv("GCP_SERVICE_ACCOUNT_B64")
if not b64:
    raise RuntimeError("GCP_SERVICE_ACCOUNT_B64 not set in .env")

# Decode to a temp file
data = base64.b64decode(b64)
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
tmp.write(data)
tmp.flush()
tmp.close()

SERVICE_ACCOUNT_JSON = tmp.name  # path to use with Credentials.from_service_account_file

# ===================== MONITOR CONFIG =====================
LOG_DIR = r".\logs"               # where windows_YYYY-MM-DD.jsonl lives (from your window/url logger)
MODEL = "gpt-4o-mini"
INTERVAL_SEC    = 3
MAX_EVENTS = 4                    # total events to send (from windows/URLs)

# ---- OFF-TASK threshold (single) ----
OFF_THRESHOLD = 0.60              # OFF_SCORE >= this → OFF-TASK

# ===================== CRITIC PASS (FN-optimized) ==========
CRITIC_ENABLED: bool = True
CRITIC_MODEL: str = MODEL         # you can set a stronger model here if you want
CRITIC_MAX_TOKENS: int = 90
CRITIC_TEMPERATURE: float = 0.0

# When Pass 1 says ON-TASK, run critic if ANY of these triggers hit:
CRITIC_TRIGGER_CONF_MAX: float = 0.70    # low-confidence ON → critic
CRITIC_TRIGGER_OFF_MIN: float = 0.30     # moderately off-ish ON → critic
CRITIC_TRIGGER_RISKY_KEYWORDS: bool = True

RISKY_KEYWORDS = [
    "youtube", "youtu.be", "reddit", "discord", "twitter", "x.com",
    "instagram", "tiktok", "netflix", "twitch", "hulu", "prime video",
    "steam", "epic games", "roblox", "minecraft",
    "shopping", "amazon", "ebay", "aliexpress"
]
# ============================================================

# ===================== SHEETS CONFIG ======================
SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
WORKSHEET_PREFIX = "Focus Logs"   # daily tabs: "<prefix> - YYYY-MM-DD"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
MAX_EVENT_COLS = 5                # flatten up to N events: e{i}_key

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
        "Potentially distractive apps or titles include (non-exhaustive): Discord, YouTube, Reddit, Twitter/X, Instagram, TikTok, Netflix, Twitch.\n\n"
        "Rules (intent-based):\n"
        "1) If the title clearly shows *work or learning intent*, treat as ON-TASK.\n"
        "2) Use surrounding context (last few events) for tie-breaks.\n"
        "3) Ambiguity rule: prefer ON-TASK if context suggests work.\n"
        "4) Weigh most recent windows higher.\n\n"
        "Output (strict, single line):\n"
        "  LABEL=<ON-TASK or OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<short explanation if OFF-TASK, else brief note>\n\n"
        "Recent activity (oldest → newest):\n" + ", ".join(tokens)
    )

def build_critic_prompt(events: List[Tuple[str, str]], p1_summary: str) -> str:
    tokens = [k for _, k in events]
    return (
        "You are a strict productivity *auditor* (a critic).\n"
        "Your job is to challenge ON-TASK decisions and reduce false negatives.\n"
        "Assume the user *might* be off-task. Try to find evidence.\n"
        "If evidence is mixed or ambiguous, err slightly toward OFF-TASK.\n\n"
        "You will be shown the initial model's summary. You may agree or disagree.\n\n"
        f"Initial model summary: {p1_summary}\n\n"
        "Output (strict, single line):\n"
        "  LABEL=<ON-TASK or OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<short explanation>\n\n"
        "Recent activity (oldest → newest):\n" + ", ".join(tokens)
    )

def _parse_llm_line(content: str) -> Tuple[str, str, Optional[float], Optional[float]]:
    content = (content or "").strip()

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

    # Normalize: if label says OFF, guarantee it clears the threshold (FN-biased)
    if label == "OFF-TASK" and off_score < OFF_THRESHOLD:
        off_score = OFF_THRESHOLD

    return label, reason, off_score, confidence

def ask_gpt(prompt: str, model: str, max_tokens: int = 60, temperature: float = 0.0) -> Tuple[str, str, Optional[float], Optional[float], str]:
    """
    Returns:
      label, reason, off_score, confidence, raw_text
    """
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, confidence = _parse_llm_line(content)
        return label, reason, off_score, confidence, content
    except Exception as e:
        return "[ERROR]", str(e), None, None, ""

def _is_risky_context(events: List[Tuple[str, str]]) -> bool:
    text = " ".join([k for _, k in events]).lower()
    return any(kw in text for kw in RISKY_KEYWORDS)

def _should_run_critic(p1_label: str, p1_off: float, p1_conf: float, events: List[Tuple[str, str]]) -> bool:
    if not CRITIC_ENABLED:
        return False
    if p1_label != "ON-TASK":
        return False
    if p1_conf is not None and p1_conf < CRITIC_TRIGGER_CONF_MAX:
        return True
    if p1_off is not None and p1_off >= CRITIC_TRIGGER_OFF_MIN:
        return True
    if CRITIC_TRIGGER_RISKY_KEYWORDS and _is_risky_context(events):
        return True
    return False

def decide_with_critic(events_context: List[Tuple[str, str]]) -> Tuple[str, str, float, float, bool]:
    """
    Returns:
      final_label, final_reason, final_off, final_conf, critic_ran
    """
    p1_prompt = build_prompt(events_context)
    p1_label, p1_reason, p1_off, p1_conf, _p1_raw = ask_gpt(
        p1_prompt, MODEL, max_tokens=60, temperature=0.0
    )

    # If pass1 errored, just return it (don’t double-fail)
    if p1_label in ("[ERROR]", "[WARN]") or p1_off is None or p1_conf is None:
        return p1_label, (p1_reason or ""), (p1_off or 0.5), (p1_conf or 0.5), False

    critic_ran = _should_run_critic(p1_label, p1_off, p1_conf, events_context)
    if not critic_ran:
        return p1_label, (p1_reason or ""), p1_off, p1_conf, False

    p1_summary = f"LABEL={p1_label} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(events_context, p1_summary)
    c_label, c_reason, c_off, c_conf, _c_raw = ask_gpt(
        c_prompt, CRITIC_MODEL, max_tokens=CRITIC_MAX_TOKENS, temperature=CRITIC_TEMPERATURE
    )

    # If critic fails, fall back to pass1
    if c_label in ("[ERROR]", "[WARN]") or c_off is None or c_conf is None:
        return p1_label, (p1_reason or ""), p1_off, p1_conf, True

    # FN-biased combine: OFF if either says OFF
    if c_label == "OFF-TASK" or (c_off >= OFF_THRESHOLD):
        final_label = "OFF-TASK"
        final_off = max(p1_off, c_off)
        final_conf = c_conf
        final_reason = (c_reason or p1_reason or "").strip()
        # include a short trace without being too verbose
        final_reason = f"[Critic] {final_reason}"
        return final_label, final_reason, final_off, final_conf, True

    # Otherwise keep pass1
    return p1_label, (p1_reason or ""), p1_off, p1_conf, True

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

    row: List[str] = [
        get("ts"),
        get("label"),
        get("confidence"),
        get("reason"),   # goes under ai_reason
        "",              # human_label
        "",              # human_reason
    ]

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

    # --- TEST: mark that the script started ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    startup_log = os.path.join(base_dir, "startup_test.log")
    with open(startup_log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  started via Task Scheduler\n")
    # ------------------------------------------

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
            last_events_signature = None
            print(f"[info] rolled over → new tab '{ws.title}'")
            last_flush_time = time.time()

        path = todays_log_path()
        events_context = read_last_events(path, MAX_EVENTS)  # always keep a stable short context window

        # Decide if there are new events since last alert (used only to avoid repeated popups)
        events_for_decision = [e for e in events_context if not last_alert_ts or e[0] > last_alert_ts]

        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not events_for_decision:
            decision_raw = "[idle]"
            reason = f"no new events in {path}"
            off_score = None
            confidence = None
        else:
            # classify using the full context window (better accuracy than only the delta)
            label, reason, off_score, confidence, critic_ran = decide_with_critic(events_context)

            off_val = off_score if off_score is not None else 0.5
            prev_state = current_state
            current_state = "OFF-TASK" if (label == "OFF-TASK" or off_val >= OFF_THRESHOLD) else "ON-TASK"

            if prev_state == "ON-TASK" and current_state == "OFF-TASK":
                show_popup("You're drifting OFF-TASK!\n" + (reason or ""))
                # mark newest context ts as alerted so we don't re-alert on same window set
                last_alert_ts = events_context[-1][0] if events_context else None
                tag = " (critic)" if critic_ran else ""
                print(f"{ts_now}  OFF-TASK{tag} (off={off_val:.2f}, conf={confidence if confidence is not None else 'n/a'}) | {reason}")
            else:
                tag = " (critic checked)" if critic_ran else ""
                print(f"{ts_now}  {current_state}{tag} (off={off_val:.2f}, conf={confidence if confidence is not None else 'n/a'})")

            decision_raw = label

        label_to_log = decision_raw if decision_raw == "[idle]" else current_state

        record = {
            "ts": ts_now,
            "label": label_to_log,
            "reason": reason or "",
            "confidence": confidence if confidence is not None else "",
            "events": [{"timestamp": ts, "key": key} for ts, key in events_for_decision] if events_for_decision else []
        }

        if label_to_log != "[idle]":
            # ---------- ONLY LOG WHEN EVENTS CHANGE ----------
            events_list = record["events"] or []
            trimmed = events_list[:MAX_EVENT_COLS]
            keys = []
            for e in trimmed:
                if isinstance(e, dict):
                    keys.append(str(e.get("key", "")) or "")
                else:
                    keys.append("")
            while len(keys) < MAX_EVENT_COLS:
                keys.append("")
            current_events_signature = tuple(keys)

            if last_events_signature is not None and current_events_signature == last_events_signature:
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
