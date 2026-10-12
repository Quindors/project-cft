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
from monitor import breaks as breaks_store
from monitor import status as status_store
from monitor import alerts as alerts_module
from monitor import idle as idle_module
from monitor import problems


class _FailureStreak:
    """Consecutive failures of one pipeline stage, reported to the status
    snapshot once they form a pattern and withdrawn on the next success.

    One failed call is weather — a dropped packet, a hiccup at the provider —
    and a box for every hiccup would teach people to ignore the box. Two in
    a row is a state, and a state with a fix belongs on the dashboard
    (monitor/status.py). `count` rides along so the banner can say how many
    checks have been lost.
    """

    def __init__(self, stage: str, *, after: int = 2):
        self.stage = stage
        self.after = max(1, after)
        self.count = 0

    def fail(self, problem: problems.Problem) -> None:
        self.count += 1
        if self.count >= self.after:
            status_store.report_problem(self.stage, problem, count=self.count)

    def ok(self) -> None:
        if self.count:
            status_store.clear_problem(self.stage)
        self.count = 0


# What a failure at each stage means for the person, appended to the
# provider's refusal so the box reads as one sentence about THEIR app.
CLASSIFIER_CONSEQUENCE = "Nothing is being classified or logged until it's sorted."
VISION_CONSEQUENCE = ("The screen check is off, so verdicts are going by window "
                      "title alone.")


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
    # Decide on the next tick regardless of the interval, WITHOUT touching
    # last_decision_time — the typed-text window is anchored to it, and
    # nulling it made the forced decision swallow everything typed since the
    # previous one.
    force_decision = False
    decision_count = 0
    loop_start = time.time()
    waiting_logged = False

    # Session-intention drift tracking. While an intention is active the
    # generic off-task popup is replaced by a dwell-based alert: only a
    # sustained run in the same drift tier (off-intent, or the more patient
    # partial-intent) earns one. drift_tier is the tier being timed; the
    # timer restarts whenever the tier changes.
    current_intention_id: Optional[int] = None
    drift_tier: Optional[str] = None
    drift_since: Optional[float] = None
    last_drift_alert: Optional[float] = None
    # Alert mode as of the last decision, only so a switch is logged once.
    current_alert_mode: Optional[str] = None
    # What earned the current streak. A drift alert may fire during a later
    # no-signal hold (a System window keeps the streak alive), and when it
    # does, the popup names this — never the System window in front.
    drift_category: Optional[str] = None
    drift_window: str = ""

    # Idle (user-away) state. Past the threshold with no keyboard or mouse
    # input, the chair is empty: nothing is judged and nothing is logged.
    # Without this, a productive window parked overnight logs hour after hour
    # of "work" — free credit on any Beeminder goal fed by those rows.
    idle_threshold = settings.idle.threshold_sec
    user_away = False

    # Break state. A break is a hard stop, not a quieter mode: while one runs
    # we do not poll windows, do not capture keystrokes, do not classify, do
    # not log and do not alert. `on_break` tracks the transition so capture is
    # torn down once on the way in and rebuilt once on the way out.
    on_break = False

    # Alert state: decisions stop while the sheet is up, and stay stopped for
    # a moment after it closes so the verdict lands on where the user went.
    alert_was_open = False
    resume_at: Optional[float] = None

    # Corrections made in the dashboard apply at the very next decision: the
    # HTTP thread flags app_state.corrections_changed and the loop reloads on
    # the spot. This timed reload is only the backstop for writers that don't
    # signal (review.py, direct DB edits). decide() checks corrections before
    # its session cache, so a fresh correction overrides cached classifications.
    try:
        corrections_refresh_sec = float(os.getenv("CFT_CORRECTIONS_REFRESH_SEC", "120"))
    except ValueError:
        corrections_refresh_sec = 120.0
    last_corrections_reload = time.time()

    # Failures that don't stop the loop but do have to become visible: an
    # account out of credits otherwise reads as a quiet afternoon.
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    classifier_health = _FailureStreak("classifier")
    vision_health = _FailureStreak("vision")
    storage_health = _FailureStreak("storage", after=1)
    corrections_health = _FailureStreak("corrections")
    status_store.clear_problems()

    print(f"\n[runner] Starting LIVE monitoring loop")
    if idle_threshold > 0:
        print(f"[runner] Idle detection: {idle_module.get_mode()} — "
              f"away after {idle_threshold:.0f}s without input")
    else:
        print("[runner] Idle detection: OFF (CFT_IDLE_SEC <= 0)")
    print(f"[runner] Poll interval: {settings.poll_interval}s")
    print(f"[runner] Decision interval: {settings.decision_interval}s")
    print(f"[runner] Context window: {settings.context_window_size} events")
    print(f"[runner] Min events needed: {settings.min_events}")
    print(f"[runner] Press Ctrl+C to stop\n")

    try:
        while not (stop_event is not None and stop_event.is_set()):
            # --- break: monitoring is off entirely ---
            try:
                active_break = breaks_store.get_active()
            except Exception as e:
                print(f"[runner] break lookup failed ({e}); assuming no break")
                active_break = None

            if active_break is not None:
                if not on_break:
                    on_break = True
                    key_src.stop()
                    status_store.set_paused(True)
                    drift_tier, drift_since, last_drift_alert = None, None, None
                    secs = active_break.get("remaining_sec") or 0
                    span = f"~{secs // 60}m" if secs >= 60 else f"{secs}s"
                    word = ("PAUSE" if active_break.get("source") == "pause"
                            else "BREAK")
                    print(f"[runner] {word} — tracking and alerts off for {span} "
                          f"(until {active_break['ends_at'][11:16]})")
                time.sleep(settings.poll_interval)
                continue

            if on_break:
                # Break over: start capture again from a clean slate. Titles
                # and keystrokes from before the break are not evidence about
                # what you are doing now.
                on_break = False
                events_buffer = []
                event_src.reset()
                key_src.start()
                status_store.set_paused(False)
                last_decision_time = time.time()
                print("[runner] break over — monitoring resumed")

            # --- idle: nobody at the desk means nothing to judge or log ---
            # The first threshold minutes of an absence still log (they are
            # indistinguishable from reading), but past it the loop stands
            # down until the next keypress or mouse move.
            idle_sec = idle_module.seconds_since_input() if idle_threshold > 0 else None
            if idle_sec is not None and idle_sec >= idle_threshold:
                if not user_away:
                    user_away = True
                    status_store.set_away(True)
                    drift_tier, drift_since = None, None
                    drift_category, drift_window = None, ""
                    # Settle the queue now: Beeminder reads the database all
                    # night, and rows buffered here would miss the day's total
                    # until morning.
                    sink.flush_if_needed()
                    print(f"[runner] no input for {int(idle_sec // 60)}m — "
                          f"away; nothing will be logged")
                time.sleep(settings.poll_interval)
                continue

            if user_away:
                # Input again. Judge what they do from here on — the titles
                # that sat in front of an empty chair are not evidence.
                user_away = False
                events_buffer = []
                event_src.reset()
                status_store.set_away(False)
                last_decision_time = time.time()
                print("[runner] input resumed — monitoring resumed")

            # --- an alert is on screen: decide nothing until it is gone ---
            # The sheet covers the display, so a screenshot shows the alert
            # rather than the user's work, and the window behind it hasn't
            # changed yet anyway. Judging through it would log the seconds
            # someone spends reading (or hesitating over) the alert as more of
            # the drift it is complaining about. Capture keeps running; only
            # the verdict waits.
            if alerts_module.alert_is_open():
                if not alert_was_open:
                    alert_was_open = True
                    print("[runner] alert on screen — decisions paused")
                time.sleep(settings.poll_interval)
                continue

            if alert_was_open:
                alert_was_open = False
                resume_at = time.time() + settings.popup.resume_grace_sec
                print(f"[runner] alert dismissed — resuming in "
                      f"{settings.popup.resume_grace_sec:.0f}s")

            if resume_at is not None:
                if time.time() < resume_at:
                    time.sleep(settings.poll_interval)
                    continue
                # Grace over. Judge where they went, not the click that closed
                # the alert, so the buffer starts from what is in front now.
                resume_at = None
                events_buffer = []
                event_src.reset()
                print("[runner] monitoring resumed")

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

            # --- corrections refresh (dashboard edits apply live) ---
            edited = app_state.corrections_changed.is_set()
            if edited or (
                corrections_refresh_sec > 0
                and (time.time() - last_corrections_reload) >= corrections_refresh_sec
            ):
                app_state.corrections_changed.clear()
                last_corrections_reload = time.time()
                try:
                    fresh = sink.load_corrections()
                    if len(fresh) != len(corrections or []):
                        print(f"[runner] corrections refreshed: {len(corrections or [])} -> {len(fresh)}")
                    corrections = fresh
                    # Priors move with corrections: a label just written also
                    # changes which rows count as uncorrected precedent.
                    prior_classifications = sink.load_prior_classifications()
                    corrections_health.ok()
                except Exception as e:
                    print(f"[runner] corrections refresh failed ({e}); keeping current set")
                    corrections_health.fail(problems.Problem(
                        "Corrections made on the dashboard aren't reaching the "
                        "classifier, so it keeps using the labels it had.",
                        f"The app reported: {' '.join(str(e).split())[:160]}. "
                        f"It retries at every check. {problems.LOOK_IN_THE_LOG}"))
                    if edited:
                        # The edit never reached `corrections`. Put the signal
                        # back so the next tick retries, and hold the forced
                        # re-judge until the reload lands — deciding now would
                        # re-log the very classification the user just fixed.
                        app_state.corrections_changed.set()
                        edited = False
                if edited:
                    # Re-judge the window in front right now — a correction
                    # that waits out the decision interval reads as ignored.
                    force_decision = True

            # --- decide if it's time ---
            should_decide = (
                force_decision
                or last_decision_time is None
                or (time.time() - last_decision_time) >= settings.decision_interval
            )

            if should_decide and len(events_buffer) >= settings.min_events:
                force_decision = False
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

                # --- active session intention (dashboard/tray edits apply live) ---
                intention = None
                try:
                    intention = intentions_store.get_active()
                except Exception as e:
                    print(f"[runner] intention lookup failed ({e}); continuing without")
                new_id = intention["id"] if intention else None
                if new_id != current_intention_id:
                    current_intention_id = new_id
                    drift_tier = None
                    drift_since = None
                    last_drift_alert = None
                    drift_category, drift_window = None, ""
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
                # vision_ran is a property of the verdict, not of this pass:
                # a cached hit replays it and reads as though vision just ran.
                from_cache = str(decision.get("final_reason", "")).startswith("[cache]")
                how = ("replayed from cache"
                       if from_cache else f"vision_ran={decision.get('vision_ran', False)}")
                print(f"[runner] LLM pipeline took {llm_time:.2f}s ({how})")

                # Log to console
                _log_decision(decision, settings)

                # Error verdicts (LLM/parse failures) are not activity — keep
                # them out of the log. They are not silent either: two in a
                # row puts the provider's refusal, and the fix, on the
                # dashboard, since nothing else in the app would show it.
                cat = decision.get("final_category_name")
                if cat in ("[ERROR]", "[PARSE_ERROR]"):
                    print("[runner] error verdict - not logged")
                    err = decision.get("final_reason", "") if cat == "[ERROR]" else "[PARSE_ERROR]"
                    classifier_health.fail(problems.llm(
                        err, CLASSIFIER_CONSEQUENCE, provider=provider, model=settings.model))
                    time.sleep(settings.poll_interval)
                    continue
                classifier_health.ok()

                # The screen check failing leaves a verdict, just a weaker
                # one. A cached replay carries the old failure, not a new one.
                if not from_cache:
                    if decision.get("vision_error"):
                        vision_health.fail(problems.llm(
                            decision["vision_error"], VISION_CONSEQUENCE,
                            provider=provider, model=settings.critic.vision_model))
                    elif decision.get("vision_ran"):
                        vision_health.ok()

                # Write to sheets
                t = time.time()
                record = _build_record(decision, events_buffer, typed_text, now_dt, intention=intention)
                try:
                    sink.enqueue_record(record)
                    flush_result = sink.flush_if_needed()
                    storage_health.ok()
                except Exception as e:
                    print(f"[runner] could not save this verdict ({e})")
                    flush_result = None
                    storage_health.fail(problems.Problem(
                        "Your activity can't be saved to the local database, "
                        "so this time is being lost.",
                        f"The app reported: {' '.join(str(e).split())[:160]}. "
                        f"Check that the disk isn't full and that focus.db in "
                        f"%LOCALAPPDATA%\\cft isn't open in another program, "
                        f"then restart the monitor from the tray."))
                sheets_time = time.time() - t
                if flush_result:
                    print(f"[runner] Flushed to sheets ({sheets_time:.2f}s, appended={flush_result.appended}, updated={flush_result.updated})")

                # Alerts. With an active intention: dwell-based drift alert.
                # Without: the original immediate off-task popup.
                tier = _intent_tier(decision, settings, intention) if intention is not None else None

                # Popups need a reader. Below the away threshold activity
                # still logs — reading produces no input either — but an
                # alert only fires if someone touched the machine recently
                # enough to plausibly be looking at the screen.
                user_present = (idle_sec is None
                                or idle_sec <= settings.idle.alert_recent_input_sec)

                # Alert mode (dashboard or tray, read every decision so a
                # switch applies on the next one). Tracker mode silences
                # everything except a session filed under a project. Logging
                # above is untouched either way. While alerts are off the
                # drift streak is not kept, so switching back to focus never
                # fires an alert for drift that happened while silenced.
                try:
                    alert_mode = app_state.get_mode()
                except Exception as e:
                    print(f"[runner] alert mode lookup failed ({e}); alerting as usual")
                    alert_mode = app_state.DEFAULT_MODE
                if alert_mode != current_alert_mode:
                    current_alert_mode = alert_mode
                    print(f"[runner] alert mode -> {alert_mode}")
                alerts_on = app_state.alerts_on(alert_mode, intention)

                if intention is not None and not alerts_on:
                    drift_tier, drift_since = None, None
                    drift_category, drift_window = None, ""
                elif intention is not None:
                    now_t = time.time()
                    drift_tier, drift_since = _update_drift(tier, drift_tier, drift_since, now_t)
                    if tier in (INTENT_OFF, INTENT_PARTIAL):
                        drift_category = decision.get("final_category_name", "Unknown")
                        drift_window = events_buffer[-1][1] if events_buffer else ""
                    # Alert whenever the *streak* is drift, not only when this
                    # decision is: a no-signal window holds the streak, and
                    # holding must not also silence an alert it already
                    # earned. The popup names what earned the streak, so a
                    # System window is never the culprit.
                    if drift_tier in (INTENT_OFF, INTENT_PARTIAL) and drift_since is not None:
                        dwell = now_t - drift_since
                        due = (
                            dwell >= _dwell_for(drift_tier, settings)
                            and (last_drift_alert is None
                                 or (now_t - last_drift_alert) >= settings.intention.realert_sec)
                        )
                        if due and settings.popup.enabled and user_present:
                            lead = ("Off-intent" if drift_tier == INTENT_OFF
                                    else "Only loosely related to your session")
                            msg = (f"{lead} for {int(dwell // 60)}m "
                                   f"({drift_category or 'Unknown'})\n"
                                   f"Session: {intention['text']}")
                            show_popup(msg, "Focus Alert",
                                       on_correct=_intent_correction_hook(intention, drift_window))
                            last_drift_alert = now_t
                elif (decision.get("final_is_productive") is False
                      and alerts_on and settings.popup.enabled and user_present):
                    cat_name = decision.get("final_category_name", "Unknown")
                    reason = decision.get("final_reason", "")
                    msg = f"Unproductive: {cat_name}" + (f"\n{reason}" if reason else "")
                    show_popup(msg, "Focus Alert")

                # Hand this verdict to the UI: the timer widget colors itself
                # by tier, and /api/status reports it to the dashboard.
                status_store.publish(
                    tier=tier,
                    category=decision.get("final_category_name"),
                    align=decision.get("final_align"),
                    is_productive=decision.get("final_is_productive"),
                    drift_since=drift_since,
                )

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

INTENT_ON = "on-intent"
INTENT_PARTIAL = "partial-intent"
INTENT_OFF = "off-intent"
# No evidence either way (a System window, a neutral category with no ALIGN).
# Distinct from on-intent: alt-tabbing past the tray for one decision is not
# evidence you returned to your session, so it must not clear a drift streak.
INTENT_UNKNOWN = "no-signal"


def _intent_tier(decision: Dict[str, Any], settings: Settings,
                 intention: Optional[Dict] = None) -> str:
    """How well does this decision serve the active session intention?

    Returns INTENT_ON / INTENT_PARTIAL / INTENT_OFF / INTENT_UNKNOWN.

    ALIGN is authoritative when the LLM produced it; the shortcut paths
    (system windows, cached pre-intention decisions) have no ALIGN and fall
    back to the category's productivity, with neutral meaning "no signal".

    Home-category prior: a partial-band score is the model hedging, and a
    hedge inside the category the intention itself maps to usually IS the
    declared work approached from a title that undersells it. Those hedges
    resolve to on-intent. The prior never touches the off band — being in
    the right category is not doing the declared work.
    """
    cfg = settings.intention
    align = decision.get("final_align")
    if align is None:
        productive = decision.get("final_is_productive")
        if productive is False:
            return INTENT_OFF
        if productive is True:
            return INTENT_ON
        return INTENT_UNKNOWN
    if align >= cfg.on_intent_min:
        return INTENT_ON
    if align >= cfg.off_intent_max:
        if cfg.home_category_prior and intention is not None:
            home = (intention.get("llm_category") or "").strip().lower()
            cat = (decision.get("final_category_name") or "").strip().lower()
            if home and cat == home:
                print(f"[runner] home-category prior: partial ALIGN in "
                      f"{intention.get('llm_category')} counts as on-intent")
                return INTENT_ON
        return INTENT_PARTIAL
    return INTENT_OFF


def _dwell_for(tier: str, settings: Settings) -> float:
    """Seconds of continuous drift in `tier` before it earns an alert."""
    return (settings.intention.partial_dwell_sec if tier == INTENT_PARTIAL
            else settings.intention.dwell_sec)


def _intent_correction_hook(intention: Dict, window_title: str):
    """Build the "this IS my session work" action offered on the alert.

    Returns None when there is nothing to record against, so the alert simply
    doesn't draw the button.
    """
    if not intention or not window_title:
        return None

    def _record() -> None:
        from monitor import intent_corrections
        intent_corrections.save(intention["text"], window_title, aligned=True)
        print(f"[intent] user ruled '{window_title[:50]}' ON-intent for "
              f"\"{intention['text'][:40]}\"")

    return _record


def _update_drift(tier: str, drift_tier, drift_since, now: float):
    """Advance the drift streak by one decision -> (drift_tier, drift_since).

    - on-intent clears the streak (you came back)
    - no-signal holds it unchanged (a System blip is not coming back)
    - a different drift tier restarts the clock
    """
    if tier == INTENT_UNKNOWN:
        return drift_tier, drift_since
    if tier == INTENT_ON:
        return None, None
    if tier != drift_tier:
        return tier, now
    return drift_tier, drift_since


def _log_decision(decision: Dict[str, Any], settings: Optional[Settings] = None) -> None:
    """Print decision to console."""
    cat_name = decision.get("final_category_name", "Unknown")
    is_productive = decision.get("final_is_productive")
    conf = decision.get("final_conf", 0.0)
    reason = decision.get("final_reason", "")
    vision_ran = decision.get("vision_ran", False)
    align = decision.get("final_align")

    p_emoji = "✓" if is_productive else "✗"
    v_emoji = "🔍" if vision_ran else "  "
    if align is None:
        align_str = ""
    elif settings is not None:
        align_str = f" {_intent_tier(decision, settings)}={align:.2f}"
    else:
        align_str = f" align={align:.2f}"
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