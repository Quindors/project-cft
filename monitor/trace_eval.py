# monitor/trace_eval.py
"""Replay the local trace through the live pipeline and score it against the
trace's baked-in `expected` labels.

This is the single "run the trace" entry point shared by the trace-focused
learning loop (analyze_accuracy.py, tune_prompt.py). Every decision uses
byte-identical inputs (same events, same screenshot fixtures), so any change in
the numbers is the prompt/model, not the workload.

The corrections and prior-classification fast-paths are disabled (=[]) so every
decision actually hits the model instead of short-circuiting.

Output rows use learning.py's schema so score()/confusion_pairs()/etc. work
unchanged:
    category_name  = the model's prediction   (analogous to the AI's call)
    human_label    = the trace's expected label (analogous to ground truth)
    current_window = the window title (WIN: prefix stripped)
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from typing import Callable, Dict, List, Optional

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.decider import decide
from monitor.sources import make_sources


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w")
    old = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old
        devnull.close()


def _strip_win(title: str) -> str:
    s = (title or "").strip()
    return s[4:] if s.upper().startswith("WIN:") else s


def build_trace_settings(
    base: Settings = DEFAULT_SETTINGS,
    *,
    win_path: Optional[str] = None,
    key_path: Optional[str] = None,
    use_vision: bool = True,
) -> Settings:
    """A trace-mode settings object derived from `base` (usually load_settings())."""
    trace = dataclasses.replace(
        base.trace,
        use_trace_file=True,
        upload_to_sheets=False,
        replay_realtime=False,
        trace_win_path=win_path or base.trace.trace_win_path,
        trace_key_path=key_path or base.trace.trace_key_path,
    )
    return dataclasses.replace(
        base,
        trace=trace,
        critic=dataclasses.replace(base.critic, vision_enabled=use_vision, vision_model=base.model),
    )


def decision_points(settings: Settings) -> List[Dict]:
    """The (model-independent) list of decision points from the trace."""
    event_src, key_src = make_sources(settings=settings)
    try:
        key_src.start()
    except Exception:
        pass

    points: List[Dict] = []
    for evt_time in event_src.event_times():
        events = event_src.last_events(evt_time, limit=settings.context_window_size)
        if len(events) < settings.min_events:
            continue
        points.append({
            "evt_time": evt_time,
            "events": events,
            "expected": event_src.get_label(evt_time),
        })

    try:
        key_src.stop()
    except Exception:
        pass
    return points


def run_trace(
    settings: Settings,
    categories: List[Dict],
    *,
    use_vision: bool = True,
    quiet: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    """Replay the whole trace through decide(). Returns per-decision rows.

    Requires the LLM client to be configured first:
        from monitor.creds import configure_llm
        from monitor.decider import set_llm_client
        set_llm_client(configure_llm()[0])
    """
    s = build_trace_settings(settings, use_vision=use_vision)
    points = decision_points(s)

    rows: List[Dict] = []
    total = len(points)
    for i, p in enumerate(points, 1):
        if progress:
            progress(i, total)
        with _quiet(quiet):
            d = decide(
                events_context=p["events"],
                settings=s,
                categories=categories,
                corrections=[],
                prior_classifications=[],
            )
        primary = d.get("primary") or {}
        rows.append({
            "timestamp": p["evt_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "current_window": _strip_win(p["events"][-1][1]) if p["events"] else "",
            "category_name": d.get("final_category_name", "") or "",   # prediction
            "human_label": p["expected"] or "",                        # ground truth
            "final_is_productive": d.get("final_is_productive"),
            "confidence": d.get("final_conf", 0.0),
            "vision_ran": bool(d.get("vision_ran")),
            "primary_category": primary.get("category_name", "") or "",
            "reason": d.get("final_reason", "") or "",
        })
    return rows
