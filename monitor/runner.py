# monitor/runner.py
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Tuple, Callable, Any

from monitor.config import (
    INTERVAL_SEC,
    MAX_EVENTS,
    OFF_THRESHOLD,
    MAX_EVENT_COLS,
    CAPTURE_TYPED_TEXT,
)
from monitor.decider import decide_with_critic


def run_monitor_loop(
    *,
    event_src: Any,
    key_src: Any,
    sink: Any,
    show_popup: Callable[[str, str], None],
) -> None:
    """
    event_src: provides now_dt() and last_events(now_dt, limit)
    key_src: provides start(), stop(), summary(now_dt, window_seconds), typed_text_between(dt0, dt1)
    sink: SheetsSink-like with ensure_day(date), enqueue_record(dict), flush_if_needed(), close()
    """

    last_alert_ts: Optional[str] = None
    current_state = "ON-TASK"

    last_events_signature: Optional[Tuple[str, ...]] = None
    last_logged_dt = event_src.now_dt()  # cursor for typed_text intervals

    while True:
        loop_start = time.time()

        now_dt = event_src.now_dt()
        ts_now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        # keep sheet tabs aligned with the source clock (trace-friendly)
        sink.ensure_day(now_dt.date())

        ks_summary = key_src.summary(now_dt, window_seconds=60)
        events_context = event_src.last_events(now_dt, MAX_EVENTS)

        # only call LLM if there are events newer than the last alert
        if last_alert_ts is None:
            events_new = events_context
        else:
            events_new = [(ts, key) for (ts, key) in events_context if ts > last_alert_ts]

        record_extra = {
            "primary_confidence": "",
            "primary_reason": "",
            "critic_confidence": "",
            "critic_reason": "",
        }

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

            if prev_state == "ON-TASK" and current_state == "OFF-TASK":
                show_popup("You're drifting OFF-TASK!\n" + (reason or ""), "Productivity Alert")

                # guard against empty context
                if events_context:
                    last_alert_ts = events_context[-1][0]

                tag = " (critic)" if critic_ran else ""
                print(f"{ts_now_str}  OFF-TASK{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason}")
            else:
                tag = " (critic checked)" if critic_ran else ""
                print(f"{ts_now_str}  {current_state}{tag} (off={off_val:.2f}, conf={conf_val:.2f})")

            decision_raw = label

        label_to_log = decision_raw if decision_raw == "[idle]" else current_state

        # typed text: compute each loop, but advance cursor ONLY if we log a row (your “undo”)
        typed_text_interval = ""
        current_dt_for_text = None
        if CAPTURE_TYPED_TEXT:
            typed_text_interval = key_src.typed_text_between(last_logged_dt, now_dt)
            current_dt_for_text = now_dt

        record = {
            "ts": ts_now_str,
            "label": label_to_log,
            "primary_confidence": record_extra["primary_confidence"],
            "primary_reason": record_extra["primary_reason"],
            "critic_confidence": record_extra["critic_confidence"],
            "critic_reason": record_extra["critic_reason"],
            "typed_text": typed_text_interval,
            # log full context used for decision
            "events": [{"timestamp": ts, "key": key} for ts, key in events_context] if events_context else [],
        }

        # log only when events change (and not idle)
        if label_to_log != "[idle]":
            trimmed = (record["events"] or [])[:MAX_EVENT_COLS]
            keys = [str(e.get("key", "")) for e in trimmed]
            keys += [""] * (MAX_EVENT_COLS - len(keys))
            current_signature = tuple(keys)

            if last_events_signature != current_signature:
                sink.enqueue_record(record)
                last_events_signature = current_signature

                if CAPTURE_TYPED_TEXT and current_dt_for_text is not None:
                    last_logged_dt = current_dt_for_text

        res = sink.flush_if_needed()
        if res:
            print(f"[flush] updated={res.updated} appended={res.appended} (sent {res.sent})")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, INTERVAL_SEC - elapsed))