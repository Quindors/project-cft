# monitor/intent_review.py
"""Ruling on logged rows: "this WAS my session work" (or wasn't), after the fact.

The alert's button and /api/intent-correct only teach the FUTURE: they pin how
a window is judged under an intention from now on. The rows already logged
keep the model's ALIGN, so a misjudged half hour still reads as drift — in the
forest's watering, the weekly report, and Beeminder's neutral-category rule.

A ruling here does both, the way a category correction does for categories:
  - the logged rows get focus_logs.human_align (1 on-intent, 0 off-intent),
    which every reader takes over the model's align_score (the model's score
    is kept, untouched, for the learning loop and for undo);
  - the (intention text, window title) pair is pinned in intent_corrections,
    so the next decision on that window in that kind of session is right too.

Clearing a ruling (aligned=None) removes both.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from monitor.problems import UserError

MAX_IDS = 2000


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


def align_sql(conn: sqlite3.Connection, alias: str = "") -> str:
    """SQL for a row's EFFECTIVE alignment: the human ruling when there is one
    (1.0 / 0.0), else the model's align_score. Falls back to the bare score on
    a database from before rulings existed, so no reader ever breaks on it."""
    p = f"{alias}." if alias else ""
    if not _has_column(conn, "focus_logs", "human_align"):
        return f"{p}align_score"
    return (f"COALESCE(CASE {p}human_align WHEN 1 THEN 1.0 WHEN 0 THEN 0.0 END, "
            f"{p}align_score)")


def _clean_ids(ids) -> List[int]:
    if not isinstance(ids, list) or not ids:
        raise UserError(
            "The dashboard didn't say which rows to rule on.",
            "Refresh the Review tab and try again — this usually means the "
            "page and the monitor are different versions.")
    try:
        out = sorted({int(i) for i in ids})
    except (TypeError, ValueError):
        raise UserError(
            "Those rows couldn't be read.",
            "Refresh the Review tab and try again — this usually means the "
            "page and the monitor are different versions.")
    if len(out) > MAX_IDS:
        raise UserError(
            f"That's more than {MAX_IDS} rows at once.",
            "Rule on a smaller group — the Review tab's rows are already "
            "grouped by window.")
    return out


def apply(conn: sqlite3.Connection, ids, aligned: Optional[bool]) -> Dict:
    """Rule rows on- or off-intent (aligned None clears the ruling).

    Only rows logged during a session can be ruled: outside a session there
    is no intention to be aligned with. Runs on the caller's read-write
    connection and commits. Returns what changed, including the per-row
    natural keys the cloud mirror needs.
    """
    ids = _clean_ids(ids)
    if not _has_column(conn, "focus_logs", "human_align"):
        conn.execute("ALTER TABLE focus_logs ADD COLUMN human_align INTEGER")
    marks = ",".join("?" * len(ids))
    rows = [dict(r) for r in conn.execute(
        f"SELECT id, timestamp, current_window, intention_id FROM focus_logs "
        f"WHERE id IN ({marks})", ids).fetchall()]
    if not rows:
        raise UserError(
            "Those rows aren't there any more.",
            "Refresh the Review tab — they may have been cleared since "
            "the page loaded.")
    in_session = [r for r in rows if r["intention_id"] is not None]
    if not in_session:
        raise UserError(
            "Those rows weren't logged during a focus session, so there's no "
            "intention to judge them against.",
            "Only rows from a session can be marked on or off intention. To fix "
            "what the activity was, change its category instead.")

    value = None if aligned is None else int(bool(aligned))
    keep = [r["id"] for r in in_session]
    km = ",".join("?" * len(keep))
    conn.execute(
        f"UPDATE focus_logs SET human_align = ?, reviewed_at = ? WHERE id IN ({km})",
        [value, datetime.now().isoformat()] + keep)

    # Pin the future: one rule per (intention text, window) the rows cover.
    texts = {}
    iids = sorted({r["intention_id"] for r in in_session})
    try:
        im = ",".join("?" * len(iids))
        texts = {r[0]: r[1] for r in conn.execute(
            f"SELECT id, text FROM intentions WHERE id IN ({im})", iids)}
    except sqlite3.Error:
        texts = {}
    conn.commit()

    from monitor import intent_corrections
    pinned = set()
    for r in in_session:
        text = texts.get(r["intention_id"])
        window = r["current_window"] or ""
        if not text or not window or (text, window) in pinned:
            continue
        pinned.add((text, window))
        try:
            if value is None:
                intent_corrections.clear(text, window)
            else:
                intent_corrections.save(text, window, bool(value))
        except Exception as e:
            print(f"[intent] could not pin the ruling for future decisions ({e})")

    return {
        "updated": len(keep),
        "skipped": len(rows) - len(keep),
        "aligned": None if value is None else bool(value),
        "rules": [{"intention": t, "window": w} for t, w in sorted(pinned)],
        "changes": [(r["timestamp"], r["current_window"], value) for r in in_session],
    }
