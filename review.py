#!/usr/bin/env python3
"""
review.py — curate the LOCAL TRACE's ground-truth labels.

The trace (tests/traces/trace_win.jsonl) is the reproducible test set the
learning loop measures against. Each event may carry an `expected` category;
that's the ground truth analyze_accuracy.py / tune_prompt.py score against.

This walks the trace events and lets you set / confirm / fix each `expected`
label against your CURRENT categories — so you can reconcile drifted labels
(e.g. "Advanced Calc" that no longer exists) and fill in unlabeled events.
Changes are written back to the trace file (a timestamped backup is made first).

Usage:
    python review.py                    # walk every trace event
    python review.py --unlabeled-only   # only events with no expected label
    python review.py --drift-only        # only labels not in current categories
    python review.py --win path.jsonl    # a different trace file

At each event:
    [Enter]   keep the current label
    <number>  set to that category
    c         type a custom label (not in the category list)
    x         clear the label (mark unlabeled)
    q         save and quit
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from monitor.creds import load_env
from monitor.config import load_settings
from monitor.supabase_sink import SupabaseSink


def _strip_win(title: str) -> str:
    s = (title or "").strip()
    return s[4:] if s.upper().startswith("WIN:") else s


def _load_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _write_lines(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate the local trace's expected labels.")
    ap.add_argument("--win", default=None, help="Trace window file (default from settings)")
    ap.add_argument("--unlabeled-only", action="store_true", help="Only events with no label")
    ap.add_argument("--drift-only", action="store_true", help="Only labels not in current categories")
    args = ap.parse_args()

    load_env(override=True)
    settings = load_settings()
    path = args.win or settings.trace.trace_win_path
    if not os.path.exists(path):
        print(f"[!] Trace file not found: {path}")
        return 1

    try:
        categories = sorted(
            [c for c in SupabaseSink(settings=settings).get_categories() if c.get("name")],
            key=lambda c: c["name"].lower(),
        )
    except Exception as e:
        print(f"[!] Could not load categories: {e}")
        return 1
    known = {c["name"].lower() for c in categories}

    rows = _load_lines(path)
    # index of rows we'll actually offer
    idxs = list(range(len(rows)))
    if args.unlabeled_only:
        idxs = [i for i in idxs if not (rows[i].get("expected") or "").strip()]
    if args.drift_only:
        idxs = [i for i in idxs if (rows[i].get("expected") or "").strip()
                and (rows[i]["expected"].strip().lower() not in known)]

    if not idxs:
        print("Nothing to review with those filters. 🎉")
        return 0

    print(f"{len(idxs)} event(s) to review in {path}\n")
    changed = 0
    quit_early = False
    for n, i in enumerate(idxs, 1):
        if quit_early:
            break
        row = rows[i]
        win = _strip_win(row.get("key", ""))
        cur = (row.get("expected") or "").strip()
        cur_tag = cur if cur else "(unlabeled)"
        drift_mark = "  ⚠ not a current category" if cur and cur.lower() not in known else ""

        print("=" * 70)
        print(f"[{n}/{len(idxs)}]  {row.get('timestamp','')}")
        print(f"  WINDOW  : {win}")
        print(f"  EXPECTED: {cur_tag}{drift_mark}")
        print("-" * 70)
        for j, c in enumerate(categories, 1):
            prod = c.get("is_productive")
            tag = "prod" if prod is True else ("off " if prod is False else " -- ")
            print(f"    {j:2d}. [{tag}] {c['name']}")
        print("\n  [Enter]=keep   <n>=set   c=custom   x=clear   q=save&quit")
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        new_label = None  # None = no change
        if raw == "":
            pass
        elif raw.lower() == "q":
            quit_early = True
        elif raw.lower() == "x":
            new_label = ""
        elif raw.lower() == "c":
            try:
                new_label = input("  custom label> ").strip()
            except (EOFError, KeyboardInterrupt):
                new_label = None
        else:
            try:
                pick = int(raw)
                if 1 <= pick <= len(categories):
                    new_label = categories[pick - 1]["name"]
                else:
                    print("  (out of range — kept)")
            except ValueError:
                print("  (unrecognized — kept)")

        if new_label is not None and new_label != cur:
            if new_label == "":
                row.pop("expected", None)
                print("  ✓ cleared")
            else:
                row["expected"] = new_label
                print(f"  ✓ set: {new_label}")
            changed += 1

    if changed:
        backup = f"{path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        shutil.copy2(path, backup)
        _write_lines(path, rows)
        print(f"\n✓ {changed} label(s) changed. Backup: {backup}")
        print("Re-run  python analyze_accuracy.py  to score against the updated trace.")
    else:
        print("\nNo changes made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
