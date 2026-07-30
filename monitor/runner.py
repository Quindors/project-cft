"""
monitor/runner.py - Main monitoring loop for cft productivity tracker

Category-based classification system with critic verification.
Called by monitor/app.py which handles all initialization.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Callable

from monitor.config import Settings
from monitor.decider import decide
from monitor.sources import EventSource, KeystrokeSource, TraceEventSource
from monitor import intentions as intentions_store
from monitor import app_state


def run_monitor_loop(
    event_src: EventSource,
    key_src: KeystrokeSource,
    sink,
    show_popup: Callable,
    settings: Settings,
    categories: Optional[List[Dict]] = None,
    corrections: Optional[List[Dict]] = None,
    prior_classifications: Optional[List[Dict]] = None,
    stop_event=None,
) -> None:
    """Main monitoring loop. Called by monitor/app.py.

    categories/corrections/prior_classifications are preloaded in parallel during
    boot. If callers omit them (older code paths), we lazy-load here as a
    backstop.
    """

    if categories is None:
        categories = sink.get_categories()
    if corrections is None:
        corrections = sink.load_corrections()
    if prior_classifications is None:
        prior_classifications = sink.load_prior_classifications()

    if categories:
        print(f"[runner] {len(categories)} categories:")
        for cat in categories:
            prod = cat.get("is_productive")
            status = "✓" if prod is True else ("✗" if prod is False else "?")
            print(f"[runner]   {status} {cat['name']}")
    else:
        print("[runner] No categories — using fallback prompts")

    if corrections:
        print(f"[runner] {len(corrections)} user corrections:")
        for c in corrections[:5]:
            print(f"[runner]   {c['window_title'][:40]:40s} {c['ai_category']} -> {c['corrected_category']}")
        if len(corrections) > 5:
            print(f"[runner]   ... and {len(corrections) - 5} more")

    if prior_classifications:
        print(f"[runner] {len(prior_classifications)} prior classifications:")
        for p in prior_classifications[:5]:
            print(f"[runner]   {p['normalized_title'][:40]:40s} -> {p['category_name']}")
        if len(prior_classifications) > 5:
            print(f"[runner]   ... and {len(prior_classifications) - 5} more")

    # Dispatch to trace or live loop
    if isinstance(event_src, TraceEventSource):
        _run_trace_loop(event_src, key_src, sink, show_popup, settings, categories, corrections, prior_classifications)
    else:
        _run_live_loop(event_src, key_src, sink, show_popup, settings, categories, corrections, prior_classifications, stop_event)


def _run_live_loop(
    event_src: EventSource,
    key_src: KeystrokeSource,
    sink,
    show_popup: Callable,
    settings: Settings,
    categories: List[Dict],
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
    stop_event=None,
) -> None:
    """Live monitoring loop — polls every poll_interval, decides every decision_interval."""

    events_buffer: List[Tuple[str, str]] = []
    last_decision_time: Optional[float] = None
    decision_count = 0
    loop_start = time.time()
    waiting_logged = False

    # Session-intention drift tracking. While an intention is active the
    # generic off-task popup is replaced by a dwell-based alert: only
    # sustained off-intention time (settings.intention.dwell_sec) triggers it.
    current_intention_id: Optional[int] = None
    off_intention_since: Optional[float] = None
    last_drift_alert: Optional[float] = None

    # Monitor mode (dashboard toggle applies live). Tracking always runs;
    # alerts only fire in distraction-free mode.
    current_mode = app_state.get_mode()
    print(f"[runner] mode: {current_mode}")

    # Corrections made in the dashboard while we run take effect within this
    # interval — no restart needed. decide() checks corrections before its
    # session cache, so a refreshed correction overrides cached classifications.
    try:
        corrections_refresh_sec = float(os.getenv("CFT_CORRECTIONS_REFRESH_SEC", "120"))
    except ValueError:
        corrections_refresh_sec = 120.0
    last_corrections_reload = time.time()

    print(f"\n[runner] Starting LIVE monitoring loop")
    print(f"[runner] Poll interval: {settings.poll_interval}s")
    print(f"[runner] Decision interval: {settings.decision_interval}s")
    print(f"[runner] Context window: {settings.context_window_size} events")
    print(f"[runner] Min events needed: {settings.min_events}")
    print(f"[runner] Press Ctrl+C to stop\n")

    try:
        while not (stop_event is not None and stop_event.is_set()):
            now_dt = event_src.now_dt()

            # --- poll for new window title (in-memory capture) ---
            event_src.poll()

            # --- collect recent events from buffer ---
            recent = event_src.last_events(now_dt, limit=settings.context_window_size)
            if recent:
                events_buffer = recent

            # --- log event accumulation progress before first decision ---
            if decision_count == 0 and not waiting_logged:
                elapsed = time.time() - loop_start
                print(f"[runner] {len(events_buffer)}/{settings.min_events} events collected ({elapsed:.0f}s since loop start)")
                if len(events_buffer) >= settings.min_events:
                    waiting_logged = True
                    print(f"[runner] Min events reached — waiting for decision interval ({settings.decision_interval}s)...")

            # --- ensure correct daily sheet ---
            sink.ensure_day(now_dt.date())

            # --- periodic corrections refresh (dashboard edits apply live) ---
            if corrections_refresh_sec > 0 and (time.time() - last_corrections_reload) >= corrections_refresh_sec:
                last_corrections_reload = time.time()
                try:
                    fresh = sink.load_corrections()
                    if len(fresh) != len(corrections or []):
                        print(f"[runner] corrections refreshed: {len(corrections or [])} -> {len(fresh)}")
                    corrections = fresh
                except Exception as e:
                    print(f"[runner] corrections refresh failed ({e}); keeping current set")

            # --- decide if it's time ---
            should_decide = (
                last_decision_time is None
                or (time.time() - last_decision_time) >= settings.decision_interval
            )

            if should_decide and len(events_buffer) >= settings.min_events:
                if last_decision_time is not None:
                    prev_dt = datetime.fromtimestamp(last_decision_time)
                else:
                    prev_dt = now_dt
                typed_text = key_src.typed_text_between(prev_dt, now_dt)

                last_decision_time = time.time()
                decision_count += 1

                print(f"\n[runner] Decision #{decision_count} on {len(events_buffer)} events...")
                if decision_count == 1:
                    print(f"[runner] Time from loop start to first decision: {time.time() - loop_start:.1f}s")

                # --- monitor mode (dashboard toggle applies live) ---
                try:
                    mode = app_state.get_mode()
                except Exception as e:
                    print(f"[runner] mode lookup failed ({e}); keeping {current_mode}")
                    mode = current_mode
                if mode != current_mode:
                    current_mode = mode
                    print(f"[runner] mode changed -> {current_mode}")
                alerts_enabled = current_mode == app_state.MODE_DISTRACTION_FREE

                # --- active session intention (dashboard/tray edits apply live) ---
                intention = None
                try:
                    intention = intentions_store.get_active()
                except Exception as e:
                    print(f"[runner] intention lookup failed ({e}); continuing without")
                new_id = intention["id"] if intention else None
                if new_id != current_intention_id:
                    current_intention_id = new_id
                    off_intention_since = None
                    last_drift_alert = None
                    if intention:
                        print(f"[runner] session intention active: \"{intention['text']}\" until {intention['ends_at']}")
                    else:
                        print("[runner] session intention ended")

                t = time.time()
                decision = decide(
                    events_context=events_buffer,
                    settings=settings,
                    categories=categories,
                    corrections=corrections,
                    prior_classifications=prior_classifications,
                    intention=intention,
                )
                llm_time = time.time() - t
                print(f"[runner] LLM pipeline took {llm_time:.2f}s (vision_ran={decision.get('vision_ran', False)})")

                # Log to console
                _log_decision(decision)

                # Error verdicts (LLM/parse failures) are transient noise, not
                # activity — keep them out of the log entirely.
                if decision.get("final_category_name") in ("[ERROR]", "[PARSE_ERROR]"):
                    print("[runner] error verdict - not logged")
                    time.sleep(settings.poll_interval)
                    continue

                # Write to sheets
                t = time.time()
                record = _build_record(decision, events_buffer, typed_text, now_dt, intention=intention)
                sink.enqueue_record(record)
                flush_result = sink.flush_if_needed()
                sheets_time = time.time() - t
                if flush_result:
                    print(f"[runner] Flushed to sheets ({sheets_time:.2f}s, appended={flush_result.appended}, updated={flush_result.updated})")

                # Alerts. Time-tracking mode suppresses them entirely (dwell
                # state resets so a stale streak can't fire on mode switch).
                # With an active intention: dwell-based drift alert.
                # Without: the original immediate off-task popup.
                if not alerts_enabled:
                    off_intention_since = None
                elif intention is not None:
                    off_intention = _is_off_intention(decision, settings)
                    if off_intention:
                        if off_intention_since is None:
                            off_intention_since = time.time()
                        dwell = time.time() - off_intention_since
                        due = (
                            dwell >= settings.intention.dwell_sec
                            and (last_drift_alert is None
                                 or (time.time() - last_drift_alert) >= settings.intention.realert_sec)
                        )
                        if due and settings.popup.enabled:
                            cat_name = decision.get("final_category_name", "Unknown")
                            msg = (f"Off-intention for {int(dwell // 60)}m ({cat_name})\n"
                                   f"Session: {intention['text']}")
                            show_popup(msg, "Focus Alert")
                            last_drift_alert = time.time()
                    else:
                        off_intention_since = None
                elif decision.get("final_is_productive") is False and settings.popup.enabled:
                    cat_name = decision.get("final_category_name", "Unknown")
                    reason = decision.get("final_reason", "")
                    msg = f"Off-task: {cat_name}" + (f"\n{reason}" if reason else "")
                    show_popup(msg, "Focus Alert")

                if decision_count == 1:
                    total_boot_to_first = time.time() - loop_start
                    print(f"\n[runner] === FIRST DECISION COMPLETE — {total_boot_to_first:.1f}s from loop start ===\n")

            time.sleep(settings.poll_interval)

    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user")
    if stop_event is not None and stop_event.is_set():
        print("\n[runner] Shutdown requested from tray")


def _run_trace_loop(
    event_src: TraceEventSource,
    key_src: KeystrokeSource,
    sink,
    show_popup: Callable,
    settings: Settings,
    categories: List[Dict],
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
) -> None:
    """Trace mode loop — iterates through event timestamps directly, no sleeping."""
    import json, os

    event_times = event_src.event_times()
    if not event_times:
        print("[runner] No events in trace file.")
        return

    print(f"\n[runner] Starting TRACE monitoring loop")
    print(f"[runner] {len(event_times)} event timestamps in trace")
    print(f"[runner] Sim range: {event_times[0]} -> {event_times[-1]}")
    print(f"[runner] Min events for decision: {settings.min_events}\n")

    # --- result logging setup ---
    results_dir = "tests/results"
    os.makedirs(results_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"test_results_{run_ts}.jsonl")
    summary_path = os.path.join(results_dir, f"test_summary_{run_ts}.txt")
    results_file = open(results_path, "w", encoding="utf-8")

    decision_count = 0
    category_counts: Dict[str, int] = {}
    productive_count = 0
    unproductive_count = 0
    ambiguous_count = 0
    prev_evt_time = event_times[0]

    # --- accuracy tracking ---
    labeled_count = 0
    exact_match = 0
    productivity_match = 0
    accuracy_details: List[Dict] = []

    for evt_time in event_times:
        # Get all events visible at this point in time
        events_buffer = event_src.last_events(evt_time, limit=settings.context_window_size)

        if len(events_buffer) < settings.min_events:
            continue

        # Ensure correct daily sheet
        sink.ensure_day(evt_time.date())

        # Get keystroke context
        typed_text = key_src.typed_text_between(prev_evt_time, evt_time)

        print(f"\n[runner] Decision at {evt_time} ({len(events_buffer)} events)...")

        t = time.time()
        decision = decide(
            events_context=events_buffer,
            settings=settings,
            categories=categories,
            corrections=corrections,
            prior_classifications=prior_classifications,
        )
        llm_time = time.time() - t
        print(f"[runner] LLM pipeline took {llm_time:.2f}s")
        decision_count += 1
        prev_evt_time = evt_time

        # Log to console
        _log_decision(decision)

        # Track counts
        cat_name = decision.get("final_category_name", "Unknown")
        is_productive = decision.get("final_is_productive")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        if is_productive is True:
            productive_count += 1
        elif is_productive is False:
            unproductive_count += 1
        else:
            ambiguous_count += 1

        # --- accuracy comparison ---
        expected_label = event_src.get_label(evt_time)
        if expected_label:
            labeled_count += 1
            predicted = cat_name
            is_exact = predicted.lower() == expected_label.lower()
            if is_exact:
                exact_match += 1

            # Look up expected productivity from categories list
            expected_productive = None
            for cat in categories:
                if cat["name"].lower() == expected_label.lower():
                    expected_productive = cat.get("is_productive")
                    break

            prod_correct = (is_productive == expected_productive)
            if prod_correct:
                productivity_match += 1

            status = "✅" if is_exact else ("⚠️ " if prod_correct else "❌")
            print(f"[eval] {status} predicted={predicted:20s} expected={expected_label:20s}"
                  f" | prod_match={prod_correct}")

            accuracy_details.append({
                "timestamp": evt_time.strftime("%Y-%m-%d %H:%M:%S"),
                "current_window": events_buffer[-1][1] if events_buffer else "",
                "predicted": predicted,
                "expected": expected_label,
                "exact_match": is_exact,
                "predicted_productive": is_productive,
                "expected_productive": expected_productive,
                "productivity_match": prod_correct,
                "confidence": decision.get("final_conf", 0.0),
                "vision_ran": decision.get("vision_ran", False),
            })

        # Write to results JSONL — full decision pipeline
        primary = decision.get("primary", {})
        vision = decision.get("vision") or {}
        result_record = {
            "timestamp": evt_time.strftime("%Y-%m-%d %H:%M:%S"),
            "category_name": cat_name,
            "is_productive": is_productive,
            "off_score": decision.get("final_off", 0.0),
            "confidence": decision.get("final_conf", 0.0),
            "reason": decision.get("final_reason", ""),
            "primary_category": primary.get("category_name", ""),
            "primary_conf": primary.get("conf", ""),
            "primary_reason": primary.get("reason", ""),
            "vision_ran": decision.get("vision_ran", False),
            "vision_category": vision.get("category_name", ""),
            "vision_conf": vision.get("conf", ""),
            "expected": event_src.get_label(evt_time),
            "typed_text": typed_text,
            "events": [k for _, k in events_buffer],
        }
        results_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")

        # Write to sheets
        record = _build_record(decision, events_buffer, typed_text, evt_time)
        sink.enqueue_record(record)
        flush_result = sink.flush_if_needed()
        if flush_result:
            print(f"[runner] Flushed to sheets (appended={flush_result.appended}, updated={flush_result.updated})")

    results_file.close()

    # --- write summary ---
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Test Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total decisions: {decision_count}\n")
        if decision_count > 0:
            f.write(f"Productive:   {productive_count} ({productive_count/decision_count*100:.1f}%)\n")
            f.write(f"Unproductive: {unproductive_count} ({unproductive_count/decision_count*100:.1f}%)\n")
            f.write(f"Ambiguous:    {ambiguous_count} ({ambiguous_count/decision_count*100:.1f}%)\n")
            f.write("\nCategory breakdown:\n")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {cat:25s} {count:4d} ({count/decision_count*100:.1f}%)\n")
        if labeled_count > 0:
            f.write(f"\nAccuracy:\n")
            f.write(f"  Category:     {exact_match}/{labeled_count} ({exact_match/labeled_count*100:.1f}%)\n")
            f.write(f"  Productivity: {productivity_match}/{labeled_count} ({productivity_match/labeled_count*100:.1f}%)\n")

    print(f"\n[runner] Done. {decision_count} decisions made.")
    print(f"[runner] Results: {results_path}")
    print(f"[runner] Summary: {summary_path}")

    if labeled_count > 0:
        exact_pct = exact_match / labeled_count * 100
        prod_pct = productivity_match / labeled_count * 100
        print(f"\n{'='*50}")
        print(f"  ACCURACY REPORT")
        print(f"{'='*50}")
        print(f"  Category accuracy:     {exact_match}/{labeled_count} ({exact_pct:.1f}%)")
        print(f"  Productivity accuracy: {productivity_match}/{labeled_count} ({prod_pct:.1f}%)")
        errors = [d for d in accuracy_details if not d["exact_match"]]
        if errors:
            print(f"\n  Misclassifications ({len(errors)}):")
            for d in errors:
                print(f"    {d['timestamp']} | {d['predicted']:20s} -> should be {d['expected']}")
        print(f"{'='*50}")


# --------------- helpers ---------------

def _is_off_intention(decision: Dict[str, Any], settings: Settings) -> bool:
    """Is this decision drifting from the active session intention?

    ALIGN is authoritative when the LLM produced it; the shortcut paths
    (corrections, cache of pre-intention decisions, System) have no ALIGN and
    fall back to the category's productivity. System/neutral never counts as
    drift.
    """
    align = decision.get("final_align")
    if align is not None:
        return align < settings.intention.align_threshold
    return decision.get("final_is_productive") is False


def _log_decision(decision: Dict[str, Any]) -> None:
    """Print decision to console."""
    cat_name = decision.get("final_category_name", "Unknown")
    is_productive = decision.get("final_is_productive")
    conf = decision.get("final_conf", 0.0)
    reason = decision.get("final_reason", "")
    vision_ran = decision.get("vision_ran", False)
    align = decision.get("final_align")

    p_emoji = "✓" if is_productive else "✗"
    v_emoji = "🔍" if vision_ran else "  "
    align_str = f" align={align:.2f}" if align is not None else ""
    print(f"[runner] {v_emoji} {p_emoji} {cat_name} (conf={conf:.2f}{align_str})")
    print(f"[runner]   {reason}")


def _build_record(
    decision: Dict[str, Any],
    events_buffer: List[Tuple[str, str]],
    typed_text: str,
    now_dt: datetime,
    intention: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Flatten a decision into the dict that SupabaseSink.enqueue_record expects."""
    primary = decision.get("primary", {})
    return {
        "timestamp":          now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "category_name":      decision.get("final_category_name", ""),
        "is_productive":      decision.get("final_is_productive"),
        "primary_confidence": primary.get("conf", ""),
        "primary_reason":     primary.get("reason", ""),
        "main_label":         primary.get("category_name", ""),
        "typed_text":         typed_text,
        "events":             [{"key": k} for _, k in events_buffer],
        "intention_id":       intention["id"] if intention else None,
        "align_score":        decision.get("final_align"),
    }