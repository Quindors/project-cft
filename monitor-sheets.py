#!/usr/bin/env python3
# focus_monitor_to_sheets_daily_tabs_buffered.py
# pip install gspread google-auth openai python-dotenv Pillow pynput

import os
import json
import time
import ctypes
import re
import random
import base64
import tempfile
import io
import threading
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict, Any
from monitor.config import *   # keeps your existing variable names unchanged
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key

load_env(override=True)
SERVICE_ACCOUNT_JSON = build_service_account_json_from_b64()

import gspread
import openai
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
from dotenv import load_dotenv

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    print("[warn] Pillow not installed. pip install Pillow")

try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    print("[warn] pynput not installed. pip install pynput")



# ===================== PROMPTS =====================

def _read_prompt_file(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[warn] failed to read {filename}: {e}")
        return ""

def build_prompt(events: List[Tuple[str, str]], keystroke_summary: str = "") -> str:
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity (last 60s): {keystroke_summary}" if keystroke_summary else ""

    template = _read_prompt_file("prompt-main.txt")
    if not template:
        return (
            "You are a focus judge. Determine if the user is ON-TASK or OFF-TASK.\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )
    return template.replace("[insert here]", activity_str + extra)

def build_critic_prompt(events: List[Tuple[str, str]], p1_summary: str, keystroke_summary: str = "") -> str:
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity: {keystroke_summary}" if keystroke_summary else ""

    template = _read_prompt_file("prompt-critic.txt")
    if not template:
        return (
            "You are a strict critic. Improve the classification.\n"
            f"Initial model summary: {p1_summary}\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )

    if template.count("[insert here]") >= 2:
        template = template.replace("[insert here]", p1_summary, 1)
        template = template.replace("[insert here]", activity_str + extra, 1)
    else:
        template += f"\n\nInitial: {p1_summary}\nActivity: {activity_str}{extra}"
    return template


# ===================== LLM PARSING / CALLS =====================

def _parse_llm_line(content: str) -> Tuple[str, str, float, float]:
    content = (content or "").strip()

    m_label = re.search(r"LABEL\s*=\s*(ON-TASK|OFF-TASK)", content, flags=re.I)
    label = m_label.group(1).upper() if m_label else ("OFF-TASK" if "OFF-TASK" in content.upper() else "ON-TASK" if "ON-TASK" in content.upper() else "[WARN]")

    m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)

    off_score = float(m_off.group(1)) if m_off else (0.8 if label == "OFF-TASK" else 0.2)
    conf = float(m_conf.group(1)) if m_conf else 0.5

    off_score = min(max(off_score, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)

    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    reason = m_reason.group(1).strip() if m_reason else ""

    # FN-biased normalization
    if label == "OFF-TASK" and off_score < OFF_THRESHOLD:
        off_score = OFF_THRESHOLD

    return label, reason, off_score, conf

def ask_gpt(prompt: str, model: str, max_tokens: int = 60, temperature: float = 0.0) -> Tuple[str, str, float, float, str]:
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf = _parse_llm_line(content)
        return label, reason, off_score, conf, content
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5, ""


# ===================== VISION TIEBREAKER =====================

def take_screenshot_b64() -> Optional[str]:
    if not ImageGrab:
        return None
    try:
        img = ImageGrab.grab()
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[warn] screenshot failed: {e}")
        return None

def ask_gpt_vision(screenshot_b64: str, events: List[Tuple[str, str]], main_reason: str, critic_reason: str) -> Tuple[str, str, float, float]:
    activity_str = ", ".join([k for _, k in events])
    prompt = (
        "You are the final judge. Resolving a disagreement.\n"
        f"Context: {activity_str}\n"
        f"Main Model said ON-TASK: {main_reason}\n"
        f"Critic Model said OFF-TASK: {critic_reason}\n\n"
        "Look at the screen. What is the user *actually* doing?\n"
        "Output strict single line format:\n"
        "LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<explanation>"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                ],
            }],
            max_tokens=120,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf = _parse_llm_line(content)
        return label, reason, off_score, conf
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5


# ===================== CRITIC LOGIC =====================

def _is_risky_context(events: List[Tuple[str, str]]) -> bool:
    text = " ".join([k for _, k in events]).lower()
    return any(kw in text for kw in RISKY_KEYWORDS)

def _should_run_critic(p1_label: str, p1_off: float, p1_conf: float, events: List[Tuple[str, str]]) -> bool:
    if not CRITIC_ENABLED:
        return False

    # Only run critic on ON-TASK calls (optimize costs)
    if p1_label != "ON-TASK":
        return False

    if p1_conf <= CRITIC_TRIGGER_CONF_MAX:
        return True

    if p1_off >= CRITIC_TRIGGER_OFF_MIN:
        return True

    if CRITIC_TRIGGER_RISKY_KEYWORDS and _is_risky_context(events):
        return True

    return False

def decide_with_critic(events_context: List[Tuple[str, str]], keystroke_summary: str = "") -> Dict[str, Any]:
    p1_prompt = build_prompt(events_context, keystroke_summary)
    p1_label, p1_reason, p1_off, p1_conf, _ = ask_gpt(p1_prompt, MODEL, max_tokens=60, temperature=0.0)

    primary_res = {"label": p1_label, "reason": p1_reason or "", "off": p1_off, "conf": p1_conf}

    def mk_ret(label, reason, off, conf, critic_ran, critic_res=None):
        return {
            "final_label": label,
            "final_reason": reason,
            "final_off": off,
            "final_conf": conf,
            "critic_ran": critic_ran,
            "primary": primary_res,
            "critic": critic_res,
        }

    if p1_label in ("[ERROR]", "[WARN]"):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], False)

    if not _should_run_critic(p1_label, p1_off, p1_conf, events_context):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], False)

    p1_summary = f"LABEL={p1_label} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(events_context, p1_summary, keystroke_summary)
    c_label, c_reason, c_off, c_conf, _ = ask_gpt(c_prompt, CRITIC_MODEL, max_tokens=CRITIC_MAX_TOKENS, temperature=CRITIC_TEMPERATURE)

    critic_res = {"label": c_label, "reason": c_reason or "", "off": c_off, "conf": c_conf}

    if c_label in ("[ERROR]", "[WARN]"):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], True, critic_res)

    critic_says_off = (c_label == "OFF-TASK") or (c_off >= OFF_THRESHOLD)

    if critic_says_off:
        print("[info] Disagreement: critic says OFF. Trying vision tiebreak...")
        b64_img = take_screenshot_b64()
        if b64_img:
            v_label, v_reason, v_off, v_conf = ask_gpt_vision(b64_img, events_context, p1_reason, c_reason)
            if v_label not in ("[ERROR]", "[WARN]"):
                return mk_ret(v_label, f"[Vision] {v_reason}", v_off, v_conf, True, critic_res)

        # fallback to critic (safer)
        final_off = max(p1_off, c_off)
        return mk_ret("OFF-TASK", f"[Critic] {critic_res['reason'] or primary_res['reason']}", final_off, c_conf, True, critic_res)

    # critic agrees ON-TASK
    return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], True, critic_res)


# ===================== LOG READERS =====================

def todays_log_path() -> str:
    return os.path.join(LOG_DIR, f"windows_{datetime.now():%Y-%m-%d}.jsonl")

def read_last_events_file(path: str, limit: int) -> List[Tuple[str, str]]:
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

def trace_now(trace: TraceData, wall_start: float) -> datetime:
    elapsed = (time.time() - wall_start) * TRACE_SPEED
    return trace.sim_start + timedelta(seconds=elapsed)

def trace_last_events(trace: TraceData, sim_now: datetime, limit: int) -> List[Tuple[str, str]]:
    # last N window events up to sim_now
    eligible = [x for x in trace.win if x[0] <= sim_now]
    tail = eligible[-limit:]
    return [(dt.strftime("%Y-%m-%d %H:%M:%S"), key) for dt, key in tail]

def trace_keys_in_range(trace: TraceData, start_dt: datetime, end_dt: datetime) -> List[str]:
    return [k for (dt, k) in trace.keys if start_dt < dt <= end_dt]


# ===================== UI POPUP =====================

def show_popup(message: str, title: str = "Productivity Alert"):
    MB_ICONINFORMATION = 0x40
    MB_SETFOREGROUND = 0x00010000
    MB_TOPMOST = 0x00040000
    flags = MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)


# ===================== SHEETS =====================

def _open_client_and_spreadsheet():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(SHEET_URL_OR_KEY) if SHEET_URL_OR_KEY.startswith("http") else gc.open_by_key(SHEET_URL_OR_KEY)
    return gc, sh

def _expected_headers() -> List[str]:
    base = [
        "ts",
        "label",
        "primary_confidence",
        "primary_reason",
        "critic_confidence",
        "critic_reason",
        "human_label",
        "human_reason",
        "typed_text",
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
        ws = sh.add_worksheet(title=title, rows="200", cols=str(len(_expected_headers())), index=0)
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

def _flatten_row(record: Dict[str, Any]) -> List[str]:
    def get(k, default=""):
        v = record.get(k, default)
        return "" if v is None else str(v)

    row: List[str] = [
        get("ts"),
        get("label"),
        get("primary_confidence"),
        get("primary_reason"),
        get("critic_confidence"),
        get("critic_reason"),
        "",  # human_label
        "",  # human_reason
        get("typed_text"),
    ]

    events = record.get("events") or []
    for i in range(MAX_EVENT_COLS):
        if i < len(events) and isinstance(events[i], dict):
            row.append(str(events[i].get("key", "")) or "")
        else:
            row.append("")
    return row

def _apierror_status(e: APIError) -> Optional[int]:
    try:
        # gspread APIError.response is requests.Response
        return getattr(e.response, "status_code", None)
    except Exception:
        return None

def _retryable_call(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            code = _apierror_status(e)
            msg = str(e)
            if ("429" in msg) or (code in (429, 500, 502, 503, 504)):
                delay = (BACKOFF_BASE * (2 ** attempt)) + random.random()
                print(f"[warn] Sheets throttled (attempt {attempt+1}/{MAX_RETRIES}) sleeping {delay:.2f}s")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Exceeded max retries for Sheets API call")

def _flush_buffer(ws, key_to_row: Dict[str, int], buffer_rows: List[List[str]]):
    if not buffer_rows:
        return {"updated": 0, "appended": 0}

    if APPEND_ONLY:
        _retryable_call(ws.append_rows, buffer_rows, value_input_option="USER_ENTERED")
        start_row_guess = max(key_to_row.values(), default=1) + 1
        for row in buffer_rows:
            ts = (row[0] or "").strip()
            if ts and ts not in key_to_row:
                key_to_row[ts] = start_row_guess
                start_row_guess += 1
        return {"updated": 0, "appended": len(buffer_rows)}

    headers = _expected_headers()
    idx_ts = headers.index("ts")
    end_col = _col_letter(len(headers))

    updates_payload = []
    appends: List[List[str]] = []

    for row in buffer_rows:
        key = (row[idx_ts] or "").strip()
        if key and key in key_to_row:
            r = key_to_row[key]
            updates_payload.append({"range": f"A{r}:{end_col}{r}", "values": [row]})
        else:
            appends.append(row)

    if updates_payload:
        _retryable_call(ws.batch_update, updates_payload, value_input_option="USER_ENTERED")

    if appends:
        _retryable_call(ws.append_rows, appends, value_input_option="USER_ENTERED")
        start_row_guess = max(key_to_row.values(), default=1) + 1
        for row in appends:
            ts = (row[idx_ts] or "").strip()
            if ts and ts not in key_to_row:
                key_to_row[ts] = start_row_guess
                start_row_guess += 1

    return {"updated": len(updates_payload), "appended": len(appends)}


# ===================== KEY LOGGER =====================

class KeyLogger:
    def __init__(self, log_dir: str):
        self.buffer: List[Dict[str, str]] = []
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


def _tail_lines(path: str, max_lines: int = 300) -> List[str]:
    # Simple tail without extra deps; ok for daily logs.
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()[-max_lines:]
    except Exception:
        return []

def get_recent_keystrokes_summary(window_seconds: int = 60) -> str:
    today = datetime.now()
    path = os.path.join(KEY_LOG_DIR, f"keystrokes_{today:%Y-%m-%d}.jsonl")
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


# ===================== MAIN =====================

def main():
    require_openai_api_key(openai)

    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise SystemExit(f"[fatal] Service account JSON not found: {SERVICE_ACCOUNT_JSON}")

    gc, sh = _open_client_and_spreadsheet()

    current_day = date.today()
    ws = _get_or_create_daily_ws(sh, current_day)
    key_to_row = _read_existing_ts(ws)
    print(f"[info] connected to sheet; tab='{ws.title}', existing rows tracked: {len(key_to_row)}")

    os.makedirs(LOG_DIR, exist_ok=True)

    # Trace setup
    trace = None
    wall_start = time.time()
    last_alert_dt: Optional[datetime] = None

    if USE_TRACE_FILE:
        trace = load_trace_data()
        print(f"[info] TRACE MODE ON. sim_start={trace.sim_start}")
    else:
        print(f"[info] watching {LOG_DIR}, checking every {INTERVAL_SEC}s")

    # Key logger only in live mode
    key_logger = None
    if not USE_TRACE_FILE:
        key_logger = KeyLogger(KEY_LOG_DIR)
        key_logger.start()

    last_events_signature: Optional[Tuple[str, ...]] = None
    buffer: List[List[str]] = []
    last_flush_time = time.time()

    # typed_text interval timing
    last_logged_dt = trace.sim_start if trace else datetime.now()

    current_state = "ON-TASK"

    try:
        while True:
            loop_start = time.time()

            # daily rollover (live mode only)
            today = date.today()
            if not USE_TRACE_FILE and today != current_day:
                if buffer:
                    _ensure_headers(ws)
                    _flush_buffer(ws, key_to_row, buffer)
                    buffer.clear()

                current_day = today
                ws = _get_or_create_daily_ws(sh, current_day)
                key_to_row = _read_existing_ts(ws)
                last_events_signature = None
                last_flush_time = time.time()
                print(f"[info] rolled over → new tab '{ws.title}'")

            # keystroke summary
            ks_summary = ""
            if USE_TRACE_FILE:
                sim_now = trace_now(trace, wall_start)
                # quick summary over last 60 seconds in sim time
                window_start = sim_now - timedelta(seconds=60)
                keys = [k for (dt, k) in trace.keys if window_start <= dt <= sim_now]
                if keys:
                    specials = {k for k in keys if (len(str(k)) > 1 or str(k).startswith("Key."))}
                    ks_summary = f"{len(keys)} keys typed. Special: [{', '.join(list(specials)[:5])}]"
                else:
                    ks_summary = "Idle (0 keys)"
            else:
                ks_summary = get_recent_keystrokes_summary(window_seconds=60)

            # events context
            if USE_TRACE_FILE:
                sim_now = trace_now(trace, wall_start)
                events_context = trace_last_events(trace, sim_now, MAX_EVENTS)
                events_for_decision = [(ts, key) for (ts, key) in events_context
                                       if (last_alert_dt is None or datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") > last_alert_dt)]
                ts_now_str = sim_now.strftime("%Y-%m-%d %H:%M:%S")
            else:
                path = todays_log_path()
                events_context = read_last_events_file(path, MAX_EVENTS)
                events_for_decision = [(ts, key) for (ts, key) in events_context if (last_alert_dt is None or datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") > last_alert_dt)]
                ts_now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # decision
            reason = ""
            record_extra = {"primary_confidence": "", "primary_reason": "", "critic_confidence": "", "critic_reason": ""}

            if not events_for_decision:
                decision_raw = "[idle]"
                reason = "no new events"
            else:
                decision_res = decide_with_critic(events_context, ks_summary)

                label = decision_res["final_label"]
                reason = decision_res["final_reason"]
                off_val = float(decision_res["final_off"])
                conf_val = float(decision_res["final_conf"])
                critic_ran = decision_res["critic_ran"]

                primary = decision_res.get("primary") or {}
                critic = decision_res.get("critic") or {}

                record_extra = {
                    "primary_confidence": primary.get("conf", ""),
                    "primary_reason": primary.get("reason", ""),
                    "critic_confidence": critic.get("conf", "") if critic else "",
                    "critic_reason": critic.get("reason", "") if critic else "",
                }

                prev_state = current_state
                current_state = "OFF-TASK" if (label == "OFF-TASK" or off_val >= OFF_THRESHOLD) else "ON-TASK"

                # alert on ON->OFF transitions
                if prev_state == "ON-TASK" and current_state == "OFF-TASK":
                    show_popup("You're drifting OFF-TASK!\n" + (reason or ""))
                    # mark newest event as alerted
                    newest_ts = events_context[-1][0]
                    last_alert_dt = datetime.strptime(newest_ts, "%Y-%m-%d %H:%M:%S")
                    tag = " (critic)" if critic_ran else ""
                    print(f"{ts_now_str}  OFF-TASK{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason}")
                else:
                    tag = " (critic checked)" if critic_ran else ""
                    print(f"{ts_now_str}  {current_state}{tag} (off={off_val:.2f}, conf={conf_val:.2f})")

                decision_raw = label

            label_to_log = decision_raw if decision_raw == "[idle]" else current_state

            # typed_text interval
            typed_text_interval = ""
            if CAPTURE_TYPED_TEXT:
                if USE_TRACE_FILE:
                    sim_now = trace_now(trace, wall_start)
                    keys_interval = trace_keys_in_range(trace, last_logged_dt, sim_now)
                    typed_text_interval = format_keys_as_text(keys_interval)
                    current_dt_for_text = sim_now
                else:
                    current_dt_for_text = datetime.now()
                    # read the keystroke log files in the interval by scanning recent lines (simple)
                    # If you want a faster indexed approach, we can add it later.
                    path = os.path.join(KEY_LOG_DIR, f"keystrokes_{current_dt_for_text:%Y-%m-%d}.jsonl")
                    if os.path.exists(path):
                        lines = _tail_lines(path, max_lines=2000)
                        keys_interval = []
                        for line in lines:
                            try:
                                rec = json.loads(line)
                                dt = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
                                if last_logged_dt < dt <= current_dt_for_text:
                                    keys_interval.append(rec.get("key", ""))
                            except Exception:
                                continue
                        typed_text_interval = format_keys_as_text(keys_interval)

            # record
            record = {
                "ts": ts_now_str,
                "label": label_to_log,
                "primary_confidence": record_extra["primary_confidence"],
                "primary_reason": record_extra["primary_reason"],
                "critic_confidence": record_extra["critic_confidence"],
                "critic_reason": record_extra["critic_reason"],
                "typed_text": typed_text_interval,
                "events": [{"timestamp": ts, "key": key} for ts, key in events_for_decision] if events_for_decision else []
            }

            # log only when events change (and not idle)
            if label_to_log != "[idle]":
                events_list = record["events"] or []
                trimmed = events_list[:MAX_EVENT_COLS]
                keys = [str(e.get("key", "")) for e in trimmed]
                keys += [""] * (MAX_EVENT_COLS - len(keys))
                current_signature = tuple(keys)

                if last_events_signature != current_signature:
                    row = _flatten_row(record)
                    buffer.append(row)
                    last_events_signature = current_signature
                    last_logged_dt = current_dt_for_text if CAPTURE_TYPED_TEXT else (trace_now(trace, wall_start) if USE_TRACE_FILE else datetime.now())

            # flush buffer
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

            # sleep
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, INTERVAL_SEC - elapsed))

    finally:
        if key_logger:
            print("[info] stopping key logger...")
            key_logger.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")