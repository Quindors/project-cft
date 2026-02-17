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
from monitor.sources import EventSource, KeystrokeSource


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
            print(f"[runner]   {status} {cat.get('emoji', '')} {cat['name']}")
    else:
        print("[runner] No categories loaded (using fallback prompts)")

    # Load user goals
    user_goals = settings.load_user_goals()
    print(f"[runner] Loaded user goals ({len(user_goals)} chars)")

    # State
    events_buffer: List[Tuple[str, str]] = []
    last_decision_time: Optional[float] = None

    print(f"\n[runner] Starting monitoring loop")
    print(f"[runner] Poll interval: {settings.poll_interval}s")
    print(f"[runner] Decision interval: {settings.decision_interval}s")
    print(f"[runner] Context window: {settings.context_window_size} events")
    print(f"[runner] Press Ctrl+C to stop\n")

    try:
        while True:
            now_dt = event_src.now_dt()

            # --- collect window events ---
            recent = event_src.last_events(now_dt, limit=settings.context_window_size)
            if recent:
                events_buffer = recent          # replace with freshest snapshot

            # --- ensure correct daily sheet ---
            sink.ensure_day(now_dt.date())

            # --- decide if it's time ---
            should_decide = (
                last_decision_time is None
                or (time.time() - last_decision_time) >= settings.decision_interval
            )

            if should_decide and len(events_buffer) >= settings.min_events:
                # Get keystroke summary
                keystroke_summary = key_src.summary(now_dt, window_seconds=60)

                # Get typed text for logging
                if last_decision_time is not None:
                    prev_dt = datetime.fromtimestamp(last_decision_time)
                else:
                    prev_dt = now_dt
                typed_text = key_src.typed_text_between(prev_dt, now_dt)

                # Make decision
                print(f"\n[runner] Making decision on {len(events_buffer)} events...")
                decision = decide_with_critic(
                    events_context=events_buffer,
                    keystroke_summary=keystroke_summary,
                    settings=settings,
                    window_dwell_info=None,
                    categories=categories,
                    user_goals=user_goals,
                )

                last_decision_time = time.time()

                # --- log to console ---
                cat_name = decision.get("final_category_name", "Unknown")
                is_productive = decision.get("final_is_productive")
                conf = decision.get("final_conf", 0.0)
                reason = decision.get("final_reason", "")
                critic_ran = decision.get("critic_ran", False)

                p_emoji = "✓" if is_productive else "✗"
                c_emoji = "🔍" if critic_ran else "  "
                print(f"[runner] {c_emoji} {p_emoji} {cat_name} (conf={conf:.2f})")
                print(f"[runner]   {reason}")

                # --- write to sheets ---
                record = _build_record(decision, events_buffer, typed_text, now_dt)
                sink.enqueue_record(record)
                flush_result = sink.flush_if_needed()
                if flush_result:
                    print(f"[runner] Flushed to sheets (appended={flush_result.appended}, updated={flush_result.updated})")

                # --- popup alert if off-task ---
                if is_productive is False and settings.popup.enabled:
                    _show_offtask_alert(show_popup, decision, settings)

            # sleep until next poll
            time.sleep(settings.poll_interval)

    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user")


# --------------- helpers ---------------

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
        "category_id":        decision.get("final_category_id", ""),
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