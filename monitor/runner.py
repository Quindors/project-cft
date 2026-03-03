"""
monitor/runner.py - Main monitoring loop for cft productivity tracker

Category-based classification system with critic verification.
Called by monitor/app.py which handles all initialization.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Callable

from monitor.config import Settings
from monitor.decider import decide_with_critic
from monitor.sources import EventSource, KeystrokeSource, TraceEventSource


def run_monitor_loop(
    event_src: EventSource,
    key_src: KeystrokeSource,
    sink,                        # SheetsSink or NullSink
    show_popup: Callable,
    settings: Settings,
) -> None:
    """
    Main monitoring loop. Called by monitor/app.py after all
    initialization (env, credentials, sources, sink) is done.
    """

    # Load categories from sheets
    categories = sink.get_categories()
    if categories:
        print(f"[runner] Loaded {len(categories)} categories:")
        for cat in categories:
            prod = cat.get("is_productive")
            status = "✓" if prod is True else ("✗" if prod is False else "?")
            print(f"[runner]   {status} {cat['name']}")
    else:
        print("[runner] No categories loaded (using fallback prompts)")

    # Dispatch to trace or live loop
    if isinstance(event_src, TraceEventSource):
        _run_trace_loop(event_src, key_src, sink, show_popup, settings, categories)
    else:
        _run_live_loop(event_src, key_src, sink, show_popup, settings, categories)


def _run_live_loop(
    event_src: EventSource,
    key_src: KeystrokeSource,
    sink,
    show_popup: Callable,
    settings: Settings,
    categories: List[Dict],
) -> None:
    """Live monitoring loop — polls every poll_interval, decides every decision_interval."""

    events_buffer: List[Tuple[str, str]] = []
    last_decision_time: Optional[float] = None

    print(f"\n[runner] Starting LIVE monitoring loop")
    print(f"[runner] Poll interval: {settings.poll_interval}s")
    print(f"[runner] Decision interval: {settings.decision_interval}s")
    print(f"[runner] Context window: {settings.context_window_size} events")
    print(f"[runner] Press Ctrl+C to stop\n")

    try:
        while True:
            now_dt = event_src.now_dt()

            # --- poll for new window title (in-memory capture) ---
            event_src.poll()

            # --- collect recent events from buffer ---
            recent = event_src.last_events(now_dt, limit=settings.context_window_size)
            if recent:
                events_buffer = recent

            # --- ensure correct daily sheet ---
            sink.ensure_day(now_dt.date())

            # --- decide if it's time ---
            should_decide = (
                last_decision_time is None
                or (time.time() - last_decision_time) >= settings.decision_interval
            )

            if should_decide and len(events_buffer) >= settings.min_events:
                keystroke_summary = key_src.summary(now_dt, window_seconds=60)

                if last_decision_time is not None:
                    prev_dt = datetime.fromtimestamp(last_decision_time)
                else:
                    prev_dt = now_dt
                typed_text = key_src.typed_text_between(prev_dt, now_dt)

                last_decision_time = time.time()

                print(f"\n[runner] Making decision on {len(events_buffer)} events...")
                decision = decide_with_critic(
                    events_context=events_buffer,
                    keystroke_summary=keystroke_summary,
                    settings=settings,
                    window_dwell_info=None,
                    categories=categories,
                )

                # Log to console
                _log_decision(decision)

                # Write to sheets
                record = _build_record(decision, events_buffer, typed_text, now_dt)
                sink.enqueue_record(record)
                flush_result = sink.flush_if_needed()
                if flush_result:
                    print(f"[runner] Flushed to sheets (appended={flush_result.appended}, updated={flush_result.updated})")

                # Popup alert if off-task
                if decision.get("final_is_productive") is False and settings.popup.enabled:
                    _show_offtask_alert(show_popup, decision, settings)

            time.sleep(settings.poll_interval)

    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user")


def _run_trace_loop(
    event_src: TraceEventSource,
    key_src: KeystrokeSource,
    sink,
    show_popup: Callable,
    settings: Settings,
    categories: List[Dict],
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
        keystroke_summary = key_src.summary(evt_time, window_seconds=60)
        typed_text = key_src.typed_text_between(prev_evt_time, evt_time)

        print(f"\n[runner] Decision at {evt_time} ({len(events_buffer)} events)...")
        decision = decide_with_critic(
            events_context=events_buffer,
            keystroke_summary=keystroke_summary,
            settings=settings,
            window_dwell_info=None,
            categories=categories,
        )

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
                "critic_ran": decision.get("critic_ran", False),
            })
        
        # Write to results JSONL — full decision pipeline
        primary = decision.get("primary", {})
        critic = decision.get("critic") or {}
        vision = decision.get("vision") or {}
        result_record = {
            "timestamp": evt_time.strftime("%Y-%m-%d %H:%M:%S"),
            # Final decision
            "category_name": cat_name,
            "is_productive": is_productive,
            "off_score": decision.get("final_off", 0.0),
            "confidence": decision.get("final_conf", 0.0),
            "reason": decision.get("final_reason", ""),
            # Pipeline details
            "primary_category": primary.get("category_name", ""),
            "primary_conf": primary.get("conf", ""),
            "primary_reason": primary.get("reason", ""),
            "critic_ran": decision.get("critic_ran", False),
            "critic_category": critic.get("category_name", ""),
            "critic_conf": critic.get("conf", ""),
            "critic_reason": critic.get("reason", ""),
            "vision_category": vision.get("category_name", ""),
            "vision_conf": vision.get("conf", ""),
            # Context
            "expected": event_src.get_label(evt_time),
            "keystroke_summary": keystroke_summary,
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
                f.write(f"  {cat}: {count} ({count/decision_count*100:.1f}%)\n")

        # --- accuracy section ---
        if labeled_count > 0:
            exact_pct = exact_match / labeled_count * 100
            prod_pct = productivity_match / labeled_count * 100
            f.write("\n" + "=" * 50 + "\n")
            f.write("ACCURACY (vs ground truth labels)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Labeled events:        {labeled_count}/{decision_count}\n")
            f.write(f"Category accuracy:     {exact_match}/{labeled_count} ({exact_pct:.1f}%)\n")
            f.write(f"Productivity accuracy: {productivity_match}/{labeled_count} ({prod_pct:.1f}%)\n")

            f.write(f"\n{'Timestamp':<22s} {'Predicted':<22s} {'Expected':<22s} {'Match':>6s} {'Prod':>6s}\n")
            f.write("-" * 80 + "\n")
            for d in accuracy_details:
                match_sym = "✅" if d["exact_match"] else "❌"
                prod_sym = "✅" if d["productivity_match"] else "❌"
                f.write(f"{d['timestamp']:<22s} {d['predicted']:<22s} {d['expected']:<22s} {match_sym:>6s} {prod_sym:>6s}\n")

            errors = [d for d in accuracy_details if not d["exact_match"]]
            if errors:
                f.write(f"\nMISCLASSIFICATIONS ({len(errors)}):\n")
                f.write("-" * 80 + "\n")
                for d in errors:
                    if d["productivity_match"]:
                        prod_note = " (productivity correct, wrong category)"
                    else:
                        prod_note = f" (productivity WRONG: predicted={d['predicted_productive']}, expected={d['expected_productive']})"
                    f.write(f"  {d['timestamp']} | {d['current_window'][:50]}\n")
                    f.write(f"    predicted: {d['predicted']}  ->  expected: {d['expected']}{prod_note}\n")
        else:
            f.write("\nNo ground truth labels found in trace file.\n")
            f.write("Add \"expected\" field to trace_win.jsonl events to enable accuracy tracking.\n")

        f.write(f"\nDetailed results saved to: {results_path}\n")

    # --- console output ---
    print(f"\n[runner] Trace complete. {decision_count} decisions made.")
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

def _log_decision(decision: Dict[str, Any]) -> None:
    """Print decision to console."""
    cat_name = decision.get("final_category_name", "Unknown")
    is_productive = decision.get("final_is_productive")
    conf = decision.get("final_conf", 0.0)
    reason = decision.get("final_reason", "")
    critic_ran = decision.get("critic_ran", False)

    p_emoji = "✓" if is_productive else "✗"
    c_emoji = "🔍" if critic_ran else "  "
    print(f"[runner] {c_emoji} {p_emoji} {cat_name} (conf={conf:.2f})")
    print(f"[runner]   {reason}")


def _build_record(
    decision: Dict[str, Any],
    events_buffer: List[Tuple[str, str]],
    typed_text: str,
    now_dt: datetime,
) -> Dict[str, Any]:
    """Flatten a decision into the dict that SheetsSink.enqueue_record expects."""

    primary = decision.get("primary", {})
    critic = decision.get("critic") or {}
    vision = decision.get("vision") or {}

    return {
        "timestamp":          now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        # Category
        "category_name":      decision.get("final_category_name", ""),
        "is_productive":      decision.get("final_is_productive"),
        # Primary
        "main_label":         primary.get("category_name", ""),
        "primary_confidence": primary.get("conf", ""),
        "primary_reason":     primary.get("reason", ""),
        # Critic
        "critic_label":       critic.get("category_name", ""),
        "critic_confidence":  critic.get("conf", ""),
        "critic_reason":      critic.get("reason", ""),
        # Vision
        "vision_label":       vision.get("category_name", ""),
        "vision_confidence":  vision.get("conf", ""),
        "vision_reason":      "",
        # Typed text
        "typed_text":         typed_text,
        # Events list (SheetsSink._flatten_row handles e1_key .. eN_key)
        "events":             [{"key": k} for _, k in events_buffer],
    }


def _show_offtask_alert(
    show_popup: Callable,
    decision: Dict[str, Any],
    settings: Settings,
) -> None:
    """Show a popup when the user is in a non-productive category."""
    cat_name = decision.get("final_category_name", "Unknown")
    reason = decision.get("final_reason", "")

    title = "Focus Alert"
    msg = f"Off-task: {cat_name}"
    if reason:
        msg += f"\n{reason}"

    show_popup(msg, title)