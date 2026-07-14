# monitor/learning.py
"""Shared learning-loop analysis: turn human-labeled focus_logs rows into
accuracy, confusion pairs, and per-mistake keyword clusters.

One definition of "correct" lives here so review.py, analyze_accuracy.py,
tune_prompt.py — and the compare_models scorer — all agree.

A "labeled row" is a dict with at least:
    category_name : str   # what the AI predicted
    human_label   : str   # what the human said (may equal category_name)
    current_window: str   # window title (optional, used for clustering)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def is_labeled(row: Dict) -> bool:
    """True if a human has put a label on this row (confirmed or corrected)."""
    return bool((row.get("human_label") or "").strip())


def is_correct(row: Dict) -> bool:
    """True if the human confirmed the AI (human_label == category_name)."""
    return is_labeled(row) and _norm(row.get("human_label")) == _norm(row.get("category_name"))


def is_correction(row: Dict) -> bool:
    """True if the human overrode the AI (human_label != category_name)."""
    return is_labeled(row) and not is_correct(row)


def score(rows: Iterable[Dict]) -> Dict[str, float]:
    """Overall accuracy over the labeled subset of `rows`."""
    labeled = [r for r in rows if is_labeled(r)]
    confirmed = sum(1 for r in labeled if is_correct(r))
    n = len(labeled)
    return {
        "reviewed": n,
        "confirmed": confirmed,
        "corrected": n - confirmed,
        "accuracy_pct": (100.0 * confirmed / n) if n else 0.0,
    }


def confusion_pairs(rows: Iterable[Dict]) -> "Counter[Tuple[str, str]]":
    """Counter of (ai_category, corrected_category) over corrections only."""
    c: "Counter[Tuple[str, str]]" = Counter()
    for r in rows:
        if is_correction(r):
            c[((r.get("category_name") or "(none)").strip(),
               (r.get("human_label") or "").strip())] += 1
    return c


def per_category_accuracy(rows: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """Accuracy broken down by the AI's predicted category."""
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if is_labeled(r):
            buckets[(r.get("category_name") or "(none)").strip()].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for cat, rs in buckets.items():
        s = score(rs)
        out[cat] = s
    return out


# --- keyword clustering: what tokens recur in a given confusion pair ---------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "-", "|", "new", "tab", "google", "chrome", "microsoft", "edge", "mozilla",
    "firefox", "window", "windows", "and", "at",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']+")


def _tokens(title: str) -> List[str]:
    toks = _TOKEN_RE.findall((title or "").lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def keyword_clusters(
    rows: Iterable[Dict],
    top_pairs: int = 6,
    top_tokens: int = 8,
) -> List[Dict]:
    """For each frequent confusion pair, the window keywords that recur in it.

    Returns a list of dicts sorted by pair frequency:
        {ai_category, corrected_category, n, examples[], top_tokens[(tok,count)]}
    """
    rows = list(rows)
    pairs = confusion_pairs(rows)

    results: List[Dict] = []
    for (ai_cat, human_cat), n in pairs.most_common(top_pairs):
        tok_counts: Counter = Counter()
        examples: List[str] = []
        for r in rows:
            if not is_correction(r):
                continue
            if (r.get("category_name") or "(none)").strip() != ai_cat:
                continue
            if (r.get("human_label") or "").strip() != human_cat:
                continue
            win = (r.get("current_window") or "").strip()
            if win and len(examples) < 5:
                examples.append(win)
            tok_counts.update(_tokens(win))
        results.append({
            "ai_category": ai_cat,
            "corrected_category": human_cat,
            "n": n,
            "examples": examples,
            "top_tokens": tok_counts.most_common(top_tokens),
        })
    return results


def worst_windows(rows: Iterable[Dict], top_n: int = 15) -> List[Tuple[str, int, str, str]]:
    """Windows most often corrected. Returns (window, count, ai_cat, human_cat)."""
    counter: "Counter[str]" = Counter()
    detail: Dict[str, Tuple[str, str]] = {}
    for r in rows:
        if not is_correction(r):
            continue
        win = (r.get("current_window") or "").strip()
        if not win:
            continue
        counter[win] += 1
        detail[win] = ((r.get("category_name") or "").strip(),
                       (r.get("human_label") or "").strip())
    return [(w, n, detail[w][0], detail[w][1]) for w, n in counter.most_common(top_n)]
