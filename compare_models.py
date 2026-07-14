#!/usr/bin/env python3
"""
compare_models.py — replay one fixed trace through the full pipeline
(primary text-LLM + vision tiebreaker) for several models and print a
side-by-side comparison.

Every model sees byte-identical inputs: the same trace events and the same
screenshot fixtures (tests/traces/screenshots/), scored against the human
labels baked into the trace. Any difference in the numbers is the model, not
the workload.

The corrections fast-path and the prior-classification cache are disabled here
(corrections=[] and prior_classifications=[]) so every decision actually hits
the model instead of short-circuiting on a cache hit. The deterministic
system-window fast-path is left on — it's model-independent.

Single global swap: each model name is applied to BOTH the primary
(settings.model) and the vision tiebreaker (settings.critic.vision_model).

Requires the configure_llm() helper added to monitor/creds.py. (It does not
need the config.py/app.py edits — it builds its own settings.)

Usage:
    python compare_models.py openai/gpt-4o-mini anthropic/claude-3.7-sonnet google/gemini-2.0-flash-001
    python compare_models.py --no-vision openai/gpt-4o-mini ...        # primary only
    python compare_models.py --win tests/traces/trace_win.jsonl \
                             --key tests/traces/trace_key.jsonl  <models...>
    python compare_models.py --verbose <models...>                    # show pipeline logs
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import os
import sys
import time
from datetime import datetime

import openai

from monitor.creds import load_env, configure_llm
from monitor.decider import set_llm_client, decide
from monitor.config import DEFAULT_SETTINGS
from monitor.sources import make_sources


# Used only if categories can't be read from the sink. Productivity flags
# mirror the live config inferred from prior trace runs; if these are stale the
# CAT% column is still valid, only PROD% could drift — hence the loud warning.
_FALLBACK_CATEGORIES = [
    {"name": "Deep Work", "description": "Focused project/coding work", "is_productive": True},
    {"name": "Homework", "description": "School assignments and study", "is_productive": True},
    {"name": "Research & Learning", "description": "Reading/learning relevant material", "is_productive": True},
    {"name": "Communication", "description": "Work/school communication", "is_productive": True},
    {"name": "Social Media", "description": "Social media browsing", "is_productive": False},
    {"name": "Entertainment", "description": "Video, music, streaming for fun", "is_productive": False},
    {"name": "Gaming", "description": "Playing games", "is_productive": False},
    {"name": "Shopping", "description": "Online shopping", "is_productive": False},
    {"name": "Ambiguous", "description": "Default catch-all, treated as off-task", "is_productive": False},
    {"name": "System", "description": "System/utility windows", "is_productive": None},
]

_ERROR_LABELS = ("[ERROR]", "[PARSE_ERROR]")


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Silence the pipeline's chatty stdout prints during a decision."""
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


def build_base_settings(win_path: str, key_path: str):
    """A trace-mode settings object shared by every model (model swapped per run)."""
    trace = dataclasses.replace(
        DEFAULT_SETTINGS.trace,
        use_trace_file=True,
        upload_to_sheets=False,
        replay_realtime=False,
        trace_win_path=win_path,
        trace_key_path=key_path,
    )
    return dataclasses.replace(DEFAULT_SETTINGS, trace=trace)


def load_categories(settings):
    """Read categories from the sink; fall back to the built-in list on failure."""
    try:
        from monitor.supabase_sink import SupabaseSink
        sink = SupabaseSink(settings=settings)
        cats = sink.get_categories()
        if cats:
            return cats, "supabase"
    except Exception as e:
        print(f"[warn] couldn't load categories from sink ({e})", file=sys.stderr)
    print("[warn] using FALLBACK categories — PROD% may differ from your live config",
          file=sys.stderr)
    return _FALLBACK_CATEGORIES, "fallback"


def expected_productive(label: str, categories):
    if not label:
        return None
    for c in categories:
        if c["name"].lower() == label.lower():
            return c.get("is_productive")
    return None


def precompute_decisions(base_settings):
    """Build the (identical-for-every-model) list of decision points from the trace."""
    event_src, key_src = make_sources(settings=base_settings)
    try:
        key_src.start()
    except Exception:
        pass

    points = []
    for evt_time in event_src.event_times():
        events = event_src.last_events(evt_time, limit=base_settings.context_window_size)
        if len(events) < base_settings.min_events:
            continue
        points.append({
            "evt_time": evt_time,
            "events": events,                      # [(ts_str, window_title), ...]
            "expected": event_src.get_label(evt_time),
        })

    try:
        key_src.stop()
    except Exception:
        pass
    return points


def run_model(model: str, points, categories, base_settings, use_vision: bool, quiet: bool):
    """Replay every decision point through the pipeline for one model."""
    settings_m = dataclasses.replace(
        base_settings,
        model=model,
        critic=dataclasses.replace(
            base_settings.critic,
            vision_model=model,            # single global swap → vision too
            vision_enabled=use_vision,
        ),
    )

    detail = []
    total = len(points)
    for i, p in enumerate(points, 1):
        sys.stderr.write(f"\r  {model}: {i}/{total}")
        sys.stderr.flush()

        t0 = time.time()
        with _quiet(quiet):
            decision = decide(
                events_context=p["events"],
                settings=settings_m,
                categories=categories,
                corrections=[],                # force LLM (no corrections fast-path)
                prior_classifications=[],      # force LLM (no cache short-circuit)
            )
        latency = time.time() - t0

        final_cat = decision.get("final_category_name", "") or ""
        primary = decision.get("primary", {}) or {}
        primary_cat = (primary.get("category_name", "") or "")
        vision_ran = bool(decision.get("vision_ran"))
        vision_overrode = (
            decision.get("vision") is not None
            and final_cat.lower() != primary_cat.lower()
        )

        detail.append({
            "model": model,
            "timestamp": p["evt_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "window": p["events"][-1][1] if p["events"] else "",
            "expected": p["expected"] or "",
            "final_category": final_cat,
            "final_is_productive": decision.get("final_is_productive"),
            "final_conf": decision.get("final_conf", 0.0),
            "primary_category": primary_cat,
            "vision_ran": vision_ran,
            "vision_overrode": vision_overrode,
            "latency_s": round(latency, 3),
            "reason": decision.get("final_reason", ""),
        })

    sys.stderr.write("\r" + " " * (len(model) + 24) + "\r")
    sys.stderr.flush()
    return detail


def summarize(model: str, detail, categories):
    n = len(detail)
    errors = sum(1 for d in detail if d["final_category"] in _ERROR_LABELS)
    scored = [d for d in detail
              if d["expected"] and d["final_category"] not in _ERROR_LABELS]
    labeled = len(scored)

    cat_correct = sum(1 for d in scored
                      if d["final_category"].lower() == d["expected"].lower())
    prod_correct = sum(1 for d in scored
                       if d["final_is_productive"] == expected_productive(d["expected"], categories))

    vision_ran = sum(1 for d in detail if d["vision_ran"])
    vision_over = sum(1 for d in detail if d["vision_overrode"])

    return {
        "model": model,
        "decisions": n,
        "labeled": labeled,
        "cat_acc": (cat_correct / labeled * 100) if labeled else 0.0,
        "prod_acc": (prod_correct / labeled * 100) if labeled else 0.0,
        "vision_rate": (vision_ran / n * 100) if n else 0.0,
        "vision_override": (vision_over / n * 100) if n else 0.0,
        "avg_conf": (sum(d["final_conf"] for d in detail) / n) if n else 0.0,
        "avg_latency": (sum(d["latency_s"] for d in detail) / n) if n else 0.0,
        "errors": errors,
    }


def print_table(summaries, use_vision: bool):
    head = f"{'MODEL':<30} {'N':>4} {'LBL':>4} {'CAT%':>6} {'PROD%':>6}"
    if use_vision:
        head += f" {'VIS%':>6} {'VOVR%':>6}"
    head += f" {'CONF':>5} {'LAT(s)':>7} {'ERR':>4}"
    print(head)
    print("-" * len(head))
    for s in summaries:
        row = (f"{s['model'][:30]:<30} {s['decisions']:>4d} {s['labeled']:>4d} "
               f"{s['cat_acc']:>6.1f} {s['prod_acc']:>6.1f}")
        if use_vision:
            row += f" {s['vision_rate']:>6.1f} {s['vision_override']:>6.1f}"
        row += f" {s['avg_conf']:>5.2f} {s['avg_latency']:>7.2f} {s['errors']:>4d}"
        print(row)
    print()
    print("CAT% = exact category match | PROD% = productive/not match | "
          "VIS% = vision fired | VOVR% = vision changed the call")


def write_outputs(summaries, details):
    os.makedirs("tests/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = f"tests/results/model_comparison_{ts}.csv"
    detail_jsonl = f"tests/results/model_comparison_{ts}_detail.jsonl"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    with open(detail_jsonl, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return summary_csv, detail_jsonl


def main():
    ap = argparse.ArgumentParser(
        description="Compare models on a fixed trace through the full primary+vision pipeline.")
    ap.add_argument("models", nargs="+",
                    help="Model slugs (OpenRouter-namespaced, e.g. openai/gpt-4o-mini)")
    ap.add_argument("--win", default=DEFAULT_SETTINGS.trace.trace_win_path,
                    help="Window trace path")
    ap.add_argument("--key", default=DEFAULT_SETTINGS.trace.trace_key_path,
                    help="Keystroke trace path")
    ap.add_argument("--no-vision", action="store_true",
                    help="Primary only — skip the vision tiebreaker")
    ap.add_argument("--verbose", action="store_true",
                    help="Show the pipeline's per-decision logging")
    args = ap.parse_args()

    load_env(override=True)
    client, provider = configure_llm()
    set_llm_client(client)
    use_vision = not args.no_vision
    quiet = not args.verbose

    base_settings = build_base_settings(args.win, args.key)
    categories, cat_source = load_categories(base_settings)

    print(f"provider={provider}  vision={'on' if use_vision else 'off'}  "
          f"categories={cat_source} ({len(categories)})")
    print(f"trace: {args.win}\n")

    points = precompute_decisions(base_settings)
    if not points:
        print("No decision points (trace empty or below min_events). Check --win/--key paths.",
              file=sys.stderr)
        sys.exit(1)

    labeled = sum(1 for p in points if p["expected"])
    print(f"{len(points)} decision points ({labeled} labeled) x {len(args.models)} models\n")

    all_detail, summaries = [], []
    for model in args.models:
        detail = run_model(model, points, categories, base_settings, use_vision, quiet)
        all_detail.extend(detail)
        summaries.append(summarize(model, detail, categories))

    print()
    print_table(summaries, use_vision)
    csv_path, jsonl_path = write_outputs(summaries, all_detail)
    print(f"\nsummary: {csv_path}")
    print(f"detail:  {jsonl_path}")
    print("(detail rows are per-decision — diff two models to see exactly where they disagree.)")


if __name__ == "__main__":
    main()