#!/usr/bin/env python3
"""
tune_prompt.py — Loop 2: fix systematic mistakes using the LOCAL TRACE.

Replays the trace, finds every decision where the model's prediction != the
trace's expected label, clusters those misses into recurring patterns, and asks
the LLM to rewrite prompt-main.txt's rules (and optionally tighten category
descriptions) so they stop happening. It then RE-RUNS the trace and reports the
accuracy delta — so you see whether the change actually helped, on identical
inputs.

By default it AUTO-APPLIES:
  - prompt-main.txt is rewritten (timestamped backup in backups/ first; a diff
    is printed; it's also tracked in git).
  - category description edits are pushed to the Supabase `categories` table
    (current descriptions backed up to backups/ as JSON first).

Use --dry-run to preview + report nothing-written. Trace labels that aren't in
your current categories (e.g. "Advanced Calc") are excluded from tuning — the
prompt can't emit a category that doesn't exist; add it or relabel via review.py.

Usage:
    python tune_prompt.py                 # analyze -> apply -> re-measure
    python tune_prompt.py --dry-run       # preview only
    python tune_prompt.py --no-categories # prompt only
    python tune_prompt.py --no-vision     # cheaper trace runs
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from datetime import datetime

from monitor.creds import load_env, configure_llm
from monitor.config import load_settings
from monitor.decider import set_llm_client
from monitor import learning
from monitor import trace_eval
from monitor.paths import resource_path
from monitor.supabase_sink import SupabaseSink

PROMPT_PATH = str(resource_path("prompt-main.txt"))
BACKUP_DIR = "backups"
PLACEHOLDERS = ("[insert categories here]", "[insert here]")


def _progress(label):
    def p(i, total):
        sys.stderr.write(f"\r  {label}: {i}/{total}")
        sys.stderr.flush()
    return p


def _clear():
    sys.stderr.write("\r" + " " * 40 + "\r")
    sys.stderr.flush()


def _build_analysis(rows) -> str:
    clusters = learning.keyword_clusters(rows, top_pairs=8, top_tokens=8)
    lines = ["SYSTEMATIC MISSES ON THE TRACE (most frequent first):", ""]
    for c in clusters:
        toks = ", ".join(f"{t}({n})" for t, n in c["top_tokens"]) or "(no recurring keywords)"
        lines.append(f"- {c['n']}x  model said \"{c['ai_category']}\"  ->  should be \"{c['corrected_category']}\"")
        lines.append(f"      recurring window keywords: {toks}")
        for ex in c["examples"][:3]:
            lines.append(f"      e.g. {ex}")
        lines.append("")
    return "\n".join(lines)


def _meta_prompt(current_prompt, categories, analysis, allow_categories) -> str:
    cat_block = "\n".join(f'- "{c["name"]}": {c.get("description","")}' for c in categories)
    cat_instruction = (
        'Also propose tightened "category_edits" for any category whose description, if '
        "clearer, would have prevented these misses. Only include categories you actually change."
        if allow_categories else
        'Do NOT propose category edits; return an empty "category_edits" list.'
    )
    return f"""You are improving the system prompt of a window-activity classifier by
learning from its mistakes on a fixed test trace. Below is the CURRENT prompt,
the category descriptions, and the classifier's systematic misses. Rewrite the
prompt so these misses stop happening, WITHOUT overfitting to individual window
titles — encode general rules.

HARD CONSTRAINTS (violating any of these breaks the program):
  1. Keep these placeholder tokens verbatim, each unchanged and on its own line:
       {PLACEHOLDERS[0]}
       {PLACEHOLDERS[1]}
  2. Keep the final strict output-format line intact:
       CATEGORY=<...> | OFF_SCORE=<...> | CONF=<...> | REASON=<...>
  3. Keep the same overall structure and scoring bands; refine rules, don't gut them.
  4. Only reference categories that exist in the list below.

CURRENT PROMPT (prompt-main.txt):
<<<PROMPT
{current_prompt}
PROMPT>>>

CATEGORY DESCRIPTIONS:
{cat_block}

{analysis}

{cat_instruction}

Respond with ONLY a JSON object, no markdown fences, of this exact shape:
{{
  "prompt": "<the full rewritten prompt-main.txt content>",
  "category_edits": [{{"name": "<existing category name>", "new_description": "<improved description>"}}],
  "notes": "<2-4 sentences: what you changed and which misses it targets>"
}}"""


def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def _validate(new_prompt: str) -> list:
    errs = []
    for ph in PLACEHOLDERS:
        if ph not in new_prompt:
            errs.append(f"missing required placeholder: {ph}")
    if "CATEGORY=" not in new_prompt or "OFF_SCORE=" not in new_prompt:
        errs.append("missing the strict output-format line")
    if len(new_prompt) < 300:
        errs.append("rewritten prompt is suspiciously short")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-tune the classifier prompt from trace misses.")
    ap.add_argument("--dry-run", action="store_true", help="Preview only; write nothing")
    ap.add_argument("--no-categories", action="store_true", help="Tune the prompt only")
    ap.add_argument("--no-vision", action="store_true", help="Cheaper trace runs (no vision)")
    ap.add_argument("--min-misses", type=int, default=2, help="Refuse below this many misses")
    ap.add_argument("--force", action="store_true", help="Proceed even below --min-misses")
    args = ap.parse_args()

    load_env(override=True)
    client, provider = configure_llm()
    set_llm_client(client)
    settings = load_settings()
    sink = SupabaseSink(settings=settings)
    categories = [c for c in sink.get_categories() if c.get("name")]
    known = {c["name"].lower() for c in categories}

    use_vision = not args.no_vision
    print("=" * 70)
    print(f"  TRACE TUNING  (model={settings.model}, vision={'on' if use_vision else 'off'})")
    print("=" * 70)

    # --- BEFORE run ---
    before = trace_eval.run_trace(settings, categories, use_vision=use_vision,
                                  progress=_progress("before"))
    _clear()
    before_labeled = [r for r in before if learning.is_labeled(r)]
    before_score = learning.score(before_labeled)

    # misses the prompt could actually fix (expected label is a real category)
    fixable = [r for r in before_labeled if r["human_label"].lower() in known]
    misses = [r for r in fixable if learning.is_correction(r)]
    drift = sorted({r["human_label"] for r in before_labeled if r["human_label"].lower() not in known})

    print(f"\nBaseline accuracy: {before_score['accuracy_pct']:.1f}% "
          f"({before_score['confirmed']}/{before_score['reviewed']})")
    if drift:
        print(f"[!] Excluding trace labels not in your categories: {', '.join(drift)}")
    if len(misses) < args.min_misses and not args.force:
        print(f"\nOnly {len(misses)} fixable miss(es) (need {args.min_misses}). "
              f"Nothing to tune. Use --force to proceed anyway.")
        return 0

    analysis = _build_analysis(misses)
    print("\n" + analysis)

    current_prompt = open(PROMPT_PATH, "r", encoding="utf-8").read()
    allow_categories = not args.no_categories
    meta = _meta_prompt(current_prompt, categories, analysis, allow_categories)

    print("Asking the model for improvements...\n")
    try:
        resp = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": meta}],
            temperature=0.2, max_tokens=3000,
        )
        result = _parse_json((resp.choices[0].message.content or "").strip())
    except Exception as e:
        print(f"[!] Model call / JSON parse failed: {e}")
        return 1

    new_prompt = (result.get("prompt") or "").strip() + "\n"
    cat_edits = [] if args.no_categories else (result.get("category_edits") or [])
    cat_edits = [e for e in cat_edits if (e.get("name") or "").lower() in known and e.get("new_description")]
    notes = (result.get("notes") or "").strip()

    errs = _validate(new_prompt)
    if errs:
        print("[!] Rewritten prompt failed validation — NOT applying:")
        for e in errs:
            print(f"    - {e}")
        return 1

    diff = list(difflib.unified_diff(
        current_prompt.splitlines(), new_prompt.splitlines(),
        fromfile="prompt-main.txt (current)", tofile="prompt-main.txt (proposed)", lineterm="",
    ))
    print("--- prompt-main.txt diff ---")
    print("\n".join(diff) if diff else "(no changes proposed)")
    if cat_edits:
        print("\n--- proposed category edits ---")
        for e in cat_edits:
            print(f"  * {e['name']}: {e['new_description']}")
    if notes:
        print(f"\nmodel notes: {notes}")

    if args.dry_run:
        print("\n[dry-run] Nothing written. Re-run without --dry-run to apply + re-measure.")
        return 0
    if not diff and not cat_edits:
        print("\nNo changes to apply.")
        return 0

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if diff:
        with open(os.path.join(BACKUP_DIR, f"prompt-main_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(current_prompt)
        with open(PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(new_prompt)
        print(f"\n✓ prompt-main.txt updated (backup: backups/prompt-main_{ts}.txt)")
    if cat_edits:
        before_desc = {c["name"]: c.get("description", "") for c in categories}
        with open(os.path.join(BACKUP_DIR, f"categories_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(before_desc, f, indent=2, ensure_ascii=False)
        applied = 0
        for e in cat_edits:
            try:
                sink.sb.table("categories").update(
                    {"description": e["new_description"]}).eq("name", e["name"]).execute()
                applied += 1
            except Exception as ex:
                print(f"[warn] could not update category '{e['name']}': {ex}")
        sink._categories_cache = None  # force reload so the AFTER run sees edits
        categories = [c for c in sink.get_categories() if c.get("name")]
        print(f"✓ {applied}/{len(cat_edits)} category edit(s) applied (backup: backups/categories_{ts}.json)")

    # --- AFTER run (same trace, updated prompt/categories) ---
    print("\nRe-running the trace to measure the change...")
    after = trace_eval.run_trace(settings, categories, use_vision=use_vision,
                                 progress=_progress("after"))
    _clear()
    after_score = learning.score([r for r in after if learning.is_labeled(r)])

    delta = after_score["accuracy_pct"] - before_score["accuracy_pct"]
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    print("\n" + "=" * 70)
    print(f"  RESULT: {before_score['accuracy_pct']:.1f}%  ->  {after_score['accuracy_pct']:.1f}%"
          f"   {arrow} {delta:+.1f} pts")
    print("=" * 70)
    if delta < 0:
        print("Accuracy dropped. Revert with the backup in backups/ (git checkout prompt-main.txt),")
        print("then try --no-categories or label more trace events with review.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
