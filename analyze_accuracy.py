#!/usr/bin/env python3
"""
analyze_accuracy.py — Loop 1: measure classifier accuracy on the LOCAL TRACE.

Replays tests/traces/ through the current prompt + model and scores each
decision against the trace's baked-in `expected` labels. Prints overall
accuracy, accuracy per predicted category, the confusion matrix
(model said -> trace expected), and the most-missed windows. The per-decision
detail is saved to tests/results/ so you can diff runs.

This is fully local and reproducible: same trace, same screenshot fixtures every
run. Re-run it after tune_prompt.py to see whether accuracy moved.

Note: this makes one LLM call per decision point (plus vision). The trace is
small, so it's cheap; pass --no-vision to halve the calls.

Usage:
    python analyze_accuracy.py                 # run the trace, report
    python analyze_accuracy.py --no-vision
    python analyze_accuracy.py --win path.jsonl --key path.jsonl
    python analyze_accuracy.py --source supabase   # old DB-label mode instead
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

from monitor.creds import load_env, configure_llm
from monitor.config import load_settings
from monitor.decider import set_llm_client
from monitor import learning
from monitor import trace_eval
from monitor.supabase_sink import SupabaseSink


def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "·" * (width - filled)


def _load_categories(settings):
    try:
        return [c for c in SupabaseSink(settings=settings).get_categories() if c.get("name")]
    except Exception as e:
        print(f"[warn] couldn't load categories from Supabase ({e})", file=sys.stderr)
        return []


def _prod_lookup(categories):
    return {c["name"].lower(): c.get("is_productive") for c in categories}


def _report(rows, categories, header):
    print("=" * 70)
    print(f"  {header}")
    print("=" * 70)

    labeled = [r for r in rows if learning.is_labeled(r)]
    if not labeled:
        print("\nNo labeled decisions to score.")
        return

    s = learning.score(labeled)
    print(f"\nScored: {s['reviewed']}   Correct: {s['confirmed']}   Wrong: {s['corrected']}")
    print(f"Category accuracy: {s['accuracy_pct']:.1f}%   {_bar(s['accuracy_pct'])}")

    # productivity accuracy (right/wrong side of the productive line)
    prod = _prod_lookup(categories)
    prod_scored = [r for r in labeled if r["human_label"].lower() in prod]
    if prod_scored:
        prod_ok = sum(1 for r in prod_scored
                      if r.get("final_is_productive") == prod.get(r["human_label"].lower()))
        print(f"Productivity accuracy: {100.0*prod_ok/len(prod_scored):.1f}%  "
              f"({prod_ok}/{len(prod_scored)})")

    # taxonomy drift: expected labels not in the current category set
    known = {c["name"].lower() for c in categories}
    drift = sorted({r["human_label"] for r in labeled if r["human_label"].lower() not in known})
    if drift:
        print("\n[!] Trace labels NOT in your current categories (can never be predicted):")
        for d in drift:
            n = sum(1 for r in labeled if r["human_label"] == d)
            print(f"      {d}  ({n}x)  — add the category, or relabel with review.py")

    per_cat = learning.per_category_accuracy(labeled)
    if per_cat:
        print("\nAccuracy by predicted category:")
        print(f"  {'model predicted':<22} {'acc%':>6} {'n':>4}")
        for cat, cs in sorted(per_cat.items(), key=lambda kv: kv[1]["accuracy_pct"]):
            print(f"  {cat[:22]:<22} {cs['accuracy_pct']:>6.1f} {cs['reviewed']:>4}")

    pairs = learning.confusion_pairs(labeled)
    if pairs:
        print("\nConfusion (model said -> trace expected):")
        print(f"  {'n':>3}  {'model said':<22} -> {'expected'}")
        for (ai, human), n in pairs.most_common(20):
            print(f"  {n:>3}  {ai[:22]:<22} -> {human}")

    worst = learning.worst_windows(labeled, top_n=15)
    if worst:
        print("\nMost-missed windows:")
        for win, n, ai, human in worst:
            print(f"  {n:>3}x  {win[:48]:<48}  {ai} -> {human}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure classifier accuracy on the local trace.")
    ap.add_argument("--source", choices=["trace", "supabase"], default="trace")
    ap.add_argument("--no-vision", action="store_true", help="Skip the vision tiebreaker")
    ap.add_argument("--win", default=None, help="Window trace path")
    ap.add_argument("--key", default=None, help="Keystroke trace path")
    ap.add_argument("--days", type=int, default=None, help="[supabase mode] last N days")
    args = ap.parse_args()

    load_env(override=True)
    settings = load_settings()

    if args.source == "supabase":
        rows = SupabaseSink(settings=settings).load_labeled(days_back=args.days)
        cats = _load_categories(settings)
        _report(rows, cats, "ACCURACY REPORT (Supabase labels)")
        return 0

    # --- trace mode (default) ---
    client, provider = configure_llm()
    set_llm_client(client)
    cats = _load_categories(settings)
    if args.win:
        settings = trace_eval.build_trace_settings(settings, win_path=args.win, key_path=args.key)

    def progress(i, total):
        sys.stderr.write(f"\r  running trace: {i}/{total}")
        sys.stderr.flush()

    print(f"model={settings.model}  vision={'off' if args.no_vision else 'on'}  "
          f"categories={len(cats)}")
    rows = trace_eval.run_trace(settings, cats, use_vision=not args.no_vision, progress=progress)
    sys.stderr.write("\r" + " " * 40 + "\r")

    # save detail for diffing runs
    os.makedirs("tests/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = f"tests/results/accuracy_{ts}.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _report(rows, cats, "ACCURACY REPORT (local trace)")
    print(f"detail: {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
