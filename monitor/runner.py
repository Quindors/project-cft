# monitor/runner.py
from __future__ import annotations

import os
import json
import time
from typing import Optional, Callable, Any, List
from datetime import datetime

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.decider import decide_with_critic


def _is_testing_mode(settings) -> bool:
    # Treat trace replay as testing mode (and/or add a dedicated flag later)
    return bool(getattr(settings, "trace", None) and settings.trace.use_trace_file)


def _save_results_to_file(results: List[dict], settings: Settings) -> None:
    """Save testing results to a file in the results folder."""
    results_dir = os.path.join(os.path.dirname(settings.log_dir), "tests/results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.jsonl"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Also create a summary file
    summary_file = os.path.join(results_dir, f"test_summary_{timestamp}.txt")
    off_task_count = sum(1 for r in results if r["label"] == "OFF-TASK")
    abstain_count = sum(1 for r in results if r["label"] == "ABSTAIN")
    on_task_count = sum(1 for r in results if r["label"] == "ON-TASK")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Test Results Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total decisions: {len(results)}\n")
        f.write(f"ON-TASK:  {on_task_count} ({on_task_count/len(results)*100:.1f}%)\n")
        f.write(f"OFF-TASK: {off_task_count} ({off_task_count/len(results)*100:.1f}%)\n")
        f.write(f"ABSTAIN:  {abstain_count} ({abstain_count/len(results)*100:.1f}%)\n")
        f.write(f"\nDetailed results saved to: {filename}\n")
    
    print(f"[info] Results saved to {results_dir}/")
    print(f"       - {filename} (detailed)")
    print(f"       - {os.path.basename(summary_file)} (summary)")


def _run_one_tick(
    *,
    now_dt: datetime,
    event_src: Any,
    key_src: Any,
    sink: Any,
    show_popup: Callable[[str, str], None],
    settings: Settings,
    state: dict,
) -> Optional[dict]:
    """
    Shared logic for one evaluation tick at a given now_dt.
    Uses `state` to persist cursors/signatures across ticks.
    Returns decision info for results tracking (only in testing mode).
    """

    ts_now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    sink.ensure_day(now_dt.date())

    ks_summary = key_src.summary(now_dt, window_seconds=60)
    events_context = event_src.last_events(now_dt, settings.max_events)

    # only decide when there are events newer than last alert (string compare is safe here)
    last_alert_ts: Optional[str] = state["last_alert_ts"]
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
    decision_info = None  # For results tracking

    current_state: str = state["current_state"]

    if events_new:
        decision_res = decide_with_critic(events_context, ks_summary, settings=settings)

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
        current_state = "OFF-TASK" if (label == "OFF-TASK" or off_val >= settings.off_threshold) else "ON-TASK"

        # TESTING MODE: always print reason (trace mode)
        testing_mode = bool(getattr(settings, "trace", None) and settings.trace.use_trace_file)

        reason_str = (reason or "").strip()
        if testing_mode and not reason_str:
            reason_str = "(no reason returned)"

        # Create decision info for results tracking
        if testing_mode and (current_state in ["OFF-TASK", "ABSTAIN"] or label == "ABSTAIN"):
            decision_info = {
                "timestamp": ts_now_str,
                "label": label if label == "ABSTAIN" else current_state,
                "off_score": off_val,
                "confidence": conf_val,
                "reason": reason_str,
                "critic_ran": critic_ran,
                "events": [key for _, key in events_context],
            }

        if prev_state == "ON-TASK" and current_state == "OFF-TASK":
            show_popup("You're drifting OFF-TASK!\n" + (reason_str or ""), "Productivity Alert")
            if events_context:
                state["last_alert_ts"] = events_context[-1][0]
            tag = " (critic)" if critic_ran else ""

            # OFF always prints reason
            print(f"{ts_now_str}  OFF-TASK{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason_str}")
        else:
            tag = " (critic checked)" if critic_ran else ""

            if testing_mode:
                # ✅ ON-TASK in testing mode prints reason too
                print(f"{ts_now_str}  {current_state}{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason_str}")
            else:
                # live mode: keep it clean
                print(f"{ts_now_str}  {current_state}{tag} (off={off_val:.2f}, conf={conf_val:.2f})")


        decision_raw = label

    state["current_state"] = current_state
    label_to_log = decision_raw if decision_raw == "[idle]" else current_state

    # typed text: compute every tick, advance cursor ONLY when a row is logged
    typed_text_interval = ""
    current_dt_for_text = None
    if settings.keystrokes.capture_typed_text:
        typed_text_interval = key_src.typed_text_between(state["last_logged_dt"], now_dt)
        current_dt_for_text = now_dt

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

    if label_to_log != "[idle]":
        trimmed = (record["events"] or [])[:settings.sheets.max_event_cols]
        keys = [str(e.get("key", "")) for e in trimmed]
        keys += [""] * (settings.sheets.max_event_cols - len(keys))
        current_signature = tuple(keys)

        if state["last_events_signature"] != current_signature:
            sink.enqueue_record(record)
            state["last_events_signature"] = current_signature

            if settings.keystrokes.capture_typed_text and current_dt_for_text is not None:
                state["last_logged_dt"] = current_dt_for_text

    res = sink.flush_if_needed()
    if res:
        print(f"[flush] updated={res.updated} appended={res.appended} (sent {res.sent})")

    return decision_info


def _trace_event_times(event_src: Any) -> List[datetime]:
    """
    Build the replay timeline from actual trace window event timestamps.
    """
    # We rely on TraceEventSource exposing session.trace.win
    sess = getattr(event_src, "session", None)
    if not sess or not getattr(sess, "trace", None):
        return []

    win = getattr(sess.trace, "win", [])  # List[tuple[datetime, str]]
    if not win:
        return []

    # unique, sorted timestamps
    times = sorted({dt for (dt, _key) in win})
    return times


def run_monitor_loop(
    *,
    event_src: Any,
    key_src: Any,
    sink: Any,
    show_popup: Callable[[str, str], None],
    settings: Settings = DEFAULT_SETTINGS,
) -> None:
    state = {
        "last_alert_ts": None,
        "current_state": "ON-TASK",
        "last_events_signature": None,     # for logging rows
        "last_decision_signature": None,   # for deciding/alerts
        "last_logged_dt": event_src.now_dt(),
        "last_popup_time": 0.0,
        "last_off_reason": "",
        "last_off_ts": "",
    }

    # ✅ TRACE MODE: event-driven replay (no wall-clock sim)
    if settings.trace.use_trace_file and hasattr(event_src, "event_times"):
        times = event_src.event_times()
        if not times:
            print("[warn] trace replay: no window events found.")
            return

        print(f"[info] trace replay: {times[0]} -> {times[-1]} ({len(times)} ticks)")

        # start typed-text cursor at the first event time
        state["last_logged_dt"] = times[0]

        # Track results for testing mode
        results: List[dict] = []

        for now_dt in times:
            decision_info = _run_one_tick(
                now_dt=now_dt,
                event_src=event_src,
                key_src=key_src,
                sink=sink,
                show_popup=show_popup,
                settings=settings,
                state=state,
            )
            
            # Collect results for OFF-TASK and ABSTAIN decisions
            if decision_info:
                results.append(decision_info)

        print("[info] trace replay complete.")
        
        # Save results to file
        if results:
            _save_results_to_file(results, settings)
            print(f"[info] Tracked {len(results)} OFF-TASK/ABSTAIN decisions")
        else:
            print("[info] No OFF-TASK or ABSTAIN decisions recorded")
        
        return

    # Live mode (non-trace)
    while True:
        loop_start = time.time()
        now_dt = event_src.now_dt()
        print("[debug] now_dt =", now_dt) # debug
        ts_now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        sink.ensure_day(now_dt.date())

        ks_summary = key_src.summary(now_dt, window_seconds=60)
        events_context = event_src.last_events(now_dt, settings.max_events)

        # ---- Signature of the current context ----
        # (Use keys only; include timestamps too if you want even more sensitivity.)
        sig_keys = tuple([k for (_ts, k) in events_context])
        decision_needed = (sig_keys != state["last_decision_signature"])

        reason = ""
        record_extra = {
            "primary_confidence": "",
            "primary_reason": "",
            "critic_confidence": "",
            "critic_reason": "",
        }

        if events_context and decision_needed:
            state["last_decision_signature"] = sig_keys

            decision_res = decide_with_critic(events_context, ks_summary, settings=settings)

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

            prev_state = state["current_state"]
            new_state = "OFF-TASK" if (label == "OFF-TASK" or off_val >= settings.off_threshold) else "ON-TASK"
            state["current_state"] = new_state

            # Save last OFF reason for repeat nagging
            if new_state == "OFF-TASK":
                state["last_off_reason"] = reason or ""
                state["last_off_ts"] = ts_now_str

            # ---- Transition alert (ON -> OFF) ----
            if settings.alerts.enabled and prev_state == "ON-TASK" and new_state == "OFF-TASK":
                state["last_popup_time"] = time.time()
                tag = " (critic)" if critic_ran else ""
                print(f"{ts_now_str}  [FLAG] OFF-TASK{tag} (off={off_val:.2f}, conf={conf_val:.2f}) | {reason}")
                show_popup("You're drifting OFF-TASK!\n" + (reason or ""), "Productivity Alert")

            else:
                tag = " (critic checked)" if critic_ran else ""
                print(f"{ts_now_str}  {new_state}{tag} (off={off_val:.2f}, conf={conf_val:.2f})")

        # ---- Repeat alerts while OFF-TASK (nag) ----
        if settings.alerts.enabled and settings.alerts.repeat_while_offtask:
            if state["current_state"] == "OFF-TASK":
                if (time.time() - state["last_popup_time"]) >= settings.alerts.cooldown_sec:
                    state["last_popup_time"] = time.time()
                    msg = state["last_off_reason"] or "Still OFF-TASK."
                    print(f"{ts_now_str}  [NAG] still OFF-TASK (last off @ {state['last_off_ts']})")
                    show_popup("Still OFF-TASK:\n" + msg, "Productivity Alert")

        # ---- typed text (cursor only advances when we log a row) ----
        typed_text_interval = ""
        current_dt_for_text = None
        if settings.keystrokes.capture_typed_text:
            typed_text_interval = key_src.typed_text_between(state["last_logged_dt"], now_dt)
            current_dt_for_text = now_dt

        label_to_log = state["current_state"] if events_context else "[idle]"

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

        # log only when events change (and not idle)
        if label_to_log != "[idle]":
            trimmed = (record["events"] or [])[:settings.sheets.max_event_cols]
            keys = [str(e.get("key", "")) for e in trimmed]
            keys += [""] * (settings.sheets.max_event_cols - len(keys))
            current_signature = tuple(keys)

            if state["last_events_signature"] != current_signature:
                sink.enqueue_record(record)
                state["last_events_signature"] = current_signature
                if settings.keystrokes.capture_typed_text and current_dt_for_text is not None:
                    state["last_logged_dt"] = current_dt_for_text

        res = sink.flush_if_needed()
        if res:
            print(f"[flush] updated={res.updated} appended={res.appended} (sent {res.sent})")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, settings.interval_sec - elapsed))