#!/usr/bin/env python3
# focus_monitor_to_sheets_daily_tabs_buffered.py
# pip install gspread google-auth openai python-dotenv Pillow pynput

import os
import time
import ctypes
from datetime import datetime, date
from typing import Optional, Tuple

import openai

from monitor.config import *  # keep existing variable names unchanged
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.decider import decide_with_critic
from monitor.sheets_sink import SheetsSink
from monitor.sources import make_sources

# ===================== ENV / CREDS =====================

load_env(override=True)
SERVICE_ACCOUNT_JSON = build_service_account_json_from_b64()

# ===================== UI POPUP =====================

def show_popup(message: str, title: str = "Productivity Alert"):
    MB_ICONINFORMATION = 0x40
    MB_SETFOREGROUND = 0x00010000
    MB_TOPMOST = 0x00040000
    flags = MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
    ctypes.windll.user32.MessageBoxW(0, message, title, flags)

# ===================== MAIN =====================

def main():
    require_openai_api_key(openai)

    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise SystemExit(f"[fatal] Service account JSON not found: {SERVICE_ACCOUNT_JSON}")

    os.makedirs(LOG_DIR, exist_ok=True)

    # Sources (live or trace)
    event_src, key_src = make_sources(USE_TRACE_FILE)

    if USE_TRACE_FILE and hasattr(event_src, "sim_start"):
        print(f"[info] TRACE MODE ON. sim_start={event_src.sim_start}")
    else:
        print(f"[info] watching {LOG_DIR}, checking every {INTERVAL_SEC}s")

    # Sheets sink (pick correct day based on source clock)
    now_dt0 = event_src.now_dt()
    sink = SheetsSink(SERVICE_ACCOUNT_JSON)
    sink.ensure_day(now_dt0.date())
    print(f"[info] connected to sheet; tab='{sink.ws_title}', existing rows tracked: {sink.existing_rows_tracked}")

    # Start key capture only in live mode (trace start() is no-op)
    key_src.start()

    last_alert_ts: Optional[str] = None  # compare as string "YYYY-MM-DD HH:MM:SS"
    current_state = "ON-TASK"

    last_events_signature: Optional[Tuple[str, ...]] = None

    # typed_text cursor (we advance every loop to avoid “giant” intervals)
    last_logged_dt = now_dt0

    try:
        while True:
            loop_start = time.time()

            now_dt = event_src.now_dt()
            ts_now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

            # Ensure sheet tab matches the same clock as the source (works for trace too)
            sink.ensure_day(now_dt.date())

            ks_summary = key_src.summary(now_dt, window_seconds=60)
            events_context = event_src.last_events(now_dt, MAX_EVENTS)

            # “new since last alert” filter (string compare is safe for this timestamp format)
            if last_alert_ts is None:
                events_new = events_context
            else:
                events_new = [(ts, key) for (ts, key) in events_context if ts > last_alert_ts]

            # Decision
            record_extra = {"primary_confidence": "", "primary_reason": "", "critic_confidence": "", "critic_reason": ""}
            decision_raw = "[idle]"
            reason = "no new events"

            if events_new:
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

                tag = " (critic)" if critic_ran else ""
                if prev_state == "ON-TASK" and current_state == "OFF-TASK":
                    show_popup("You're drifting OFF-TASK!\n" + (reason or ""))
                    # mark newest event as alerted (guard against empty context)
                    if events_context:
                        last_alert_ts = events_context[-1][0]
                    print(f"{ts_now_str}  OFF-TASK{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason}")
                else:
                    tag2 = " (critic checked)" if critic_ran else ""
                    print(f"{ts_now_str}  {current_state}{tag2} (off={off_val:.2f}, conf={conf_val:.2f})")

                decision_raw = label

            label_to_log = decision_raw if decision_raw == "[idle]" else current_state

            # Typed text (advance cursor ONLY when we actually log a row)
            typed_text_interval = ""
            current_dt_for_text = None

            if CAPTURE_TYPED_TEXT:
                typed_text_interval = key_src.typed_text_between(last_logged_dt, now_dt)
                current_dt_for_text = now_dt

            # Record (log the same context you decided on)
            record = {
                "ts": ts_now_str,
                "label": label_to_log,
                "primary_confidence": record_extra["primary_confidence"],
                "primary_reason": record_extra["primary_reason"],
                "critic_confidence": record_extra["critic_confidence"],
                "critic_reason": record_extra["critic_reason"],
                "typed_text": typed_text_interval,
                "events": [{"timestamp": ts, "key": key} for ts, key in events_context] if events_context else [],
            }

            # Log only when events change (and not idle)
            if label_to_log != "[idle]":
                trimmed = (record["events"] or [])[:MAX_EVENT_COLS]
                keys = [str(e.get("key", "")) for e in trimmed]
                keys += [""] * (MAX_EVENT_COLS - len(keys))
                current_signature = tuple(keys)

                if last_events_signature != current_signature:
                    sink.enqueue_record(record)
                    last_events_signature = current_signature

                    # advance typed-text cursor only when we actually log
                    if CAPTURE_TYPED_TEXT and current_dt_for_text is not None:
                        last_logged_dt = current_dt_for_text

            res = sink.flush_if_needed()
            if res:
                print(f"[flush] updated={res.updated} appended={res.appended} (sent {res.sent})")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, INTERVAL_SEC - elapsed))

    finally:
        print("[info] final sheets flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")