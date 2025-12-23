#!/usr/bin/env python3
# focus_monitor_to_sheets_daily_tabs_buffered.py
# pip install gspread google-auth openai python-dotenv Pillow pynput

import os
import json
import time
import ctypes
import threading
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict, Any

from monitor.config import *   # keeps your existing variable names unchanged
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.decider import decide_with_critic
from monitor.sheets_sink import SheetsSink

load_env(override=True)
SERVICE_ACCOUNT_JSON = build_service_account_json_from_b64()

import openai

try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    print("[warn] pynput not installed. pip install pynput")

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

    sink = SheetsSink(SERVICE_ACCOUNT_JSON)
    sink.ensure_day(date.today())
    print(f"[info] connected to sheet; tab='{sink.ws_title}', existing rows tracked: {sink.existing_rows_tracked}")

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

    # typed_text interval timing
    last_logged_dt = trace.sim_start if trace else datetime.now()

    current_state = "ON-TASK"

    try:
        while True:
            loop_start = time.time()

            # daily rollover (live mode only)
            if not USE_TRACE_FILE:
                sink.ensure_day(date.today())

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
            current_dt_for_text = None  # <-- add this

            if CAPTURE_TYPED_TEXT:
                if USE_TRACE_FILE:
                    sim_now = trace_now(trace, wall_start)
                    keys_interval = trace_keys_in_range(trace, last_logged_dt, sim_now)
                    typed_text_interval = format_keys_as_text(keys_interval)
                    current_dt_for_text = sim_now
                else:
                    current_dt_for_text = datetime.now()
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
                    sink.enqueue_record(record)

                    # ✅ CRITICAL: update signature so you don't enqueue forever
                    last_events_signature = current_signature

                    # ✅ CRITICAL: advance typed-text cursor only when you actually log
                    if CAPTURE_TYPED_TEXT and current_dt_for_text is not None:
                        last_logged_dt = current_dt_for_text

            res = sink.flush_if_needed()
            if res:
                print(f"[flush] updated={res.updated} appended={res.appended} (sent {res.sent})")

            # sleep
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, INTERVAL_SEC - elapsed))

    finally:
        print("[info] final sheets flush...")
        sink.close()
        if key_logger:
            print("[info] stopping key logger...")
            key_logger.stop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")