# monitor/intent_corrections.py
"""User corrections to the ALIGN verdict — "this IS my session work".

Category corrections (focus_logs.human_label) fix WHAT an activity is. This
fixes whether it serves the declared intention, which the model gets wrong in
its own way: a window whose title says nothing useful, or work the model has
no way to know is part of the task. Being nagged about the thing you are
deliberately doing is the fastest way to stop trusting the alerts, so the fix
has to be one click from the alert itself.

Keyed by the intention TEXT rather than its id: "personal statement finish up"
means the same thing tomorrow, and a correction that expired with the session
would have to be re-taught every time. Same shape as the category corrections
it sits beside — a normalized window title, so notification counts and other
title noise still match.

Stored in the local focus.db, like intentions and corrections.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from monitor.paths import data_dir
from monitor.problems import UserError

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def normalize_title(title: str) -> str:
    """Match monitor/decider.py's normalization so keys line up."""
    s = (title or "").strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    s = re.sub(r"^\(\d+\)\s*", "", s)  # leading (3) notification counts
    return " ".join(s.split()).lower()


def normalize_intention(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS intent_corrections (
               id             INTEGER PRIMARY KEY AUTOINCREMENT,
               intention_norm TEXT NOT NULL,
               window_norm    TEXT NOT NULL,
               aligned        INTEGER NOT NULL,
               intention_text TEXT,
               window_title   TEXT,
               created_at     TEXT NOT NULL,
               UNIQUE(intention_norm, window_norm)
           )"""
    )
    return conn


def save(intention_text: str, window_title: str, aligned: bool) -> Dict:
    """Record (or flip) how a window should be judged under this intention."""
    i_norm = normalize_intention(intention_text)
    w_norm = normalize_title(window_title)
    if not i_norm or not w_norm:
        raise UserError(
            "That feedback needs both a session and a window to attach to.",
            "Refresh the dashboard and try again \u2014 the session may have "
            "ended between the click and the save.")

    with _connect() as conn:
        conn.execute(
            """INSERT INTO intent_corrections
                   (intention_norm, window_norm, aligned, intention_text,
                    window_title, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(intention_norm, window_norm) DO UPDATE SET
                   aligned = excluded.aligned,
                   created_at = excluded.created_at""",
            (i_norm, w_norm, int(bool(aligned)), intention_text.strip(),
             (window_title or "").strip(), datetime.now().strftime(_TS_FMT)),
        )
    return {"intention": intention_text.strip(), "window": (window_title or "").strip(),
            "aligned": bool(aligned)}


def clear(intention_text: str, window_title: str) -> bool:
    """Forget a correction. True if one was removed."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM intent_corrections WHERE intention_norm = ? AND window_norm = ?",
            (normalize_intention(intention_text), normalize_title(window_title)),
        )
        return cur.rowcount > 0


def lookup(intention_text: str, window_title: str) -> Optional[bool]:
    """True = user says on-intent, False = off-intent, None = no correction."""
    i_norm = normalize_intention(intention_text)
    w_norm = normalize_title(window_title)
    if not i_norm or not w_norm:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT aligned FROM intent_corrections "
                "WHERE intention_norm = ? AND window_norm = ?",
                (i_norm, w_norm),
            ).fetchone()
    except Exception as e:
        print(f"[intent] correction lookup failed ({e})")
        return None
    return None if row is None else bool(row["aligned"])


def for_intention(intention_text: str) -> List[Dict]:
    """Everything the user has taught about this intention, newest first."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT window_title, aligned, created_at FROM intent_corrections "
                "WHERE intention_norm = ? ORDER BY created_at DESC",
                (normalize_intention(intention_text),),
            ).fetchall()
    except Exception:
        return []
    return [{"window_title": r["window_title"], "aligned": bool(r["aligned"]),
             "created_at": r["created_at"]} for r in rows]
