# monitor/intentions.py
"""Session intentions: "for the next N minutes I intend to be doing X".

A session intention is a short-lived, user-declared focus target. While one is
active, the classifier judges activity against it (ALIGN score) and the runner
alerts on sustained drift instead of popping on every off-task decision.

Stored in the local focus.db (local SQLite is the source of truth, same as
corrections). This module owns the `intentions` table and opens short-lived
connections so it can be called safely from the runner loop, the dashboard
HTTP threads, and the tray thread at the same time.

Timestamps are naive local "YYYY-MM-DD HH:MM:SS" strings, matching
focus_logs.timestamp, so plain string comparison is correct.

Status lifecycle: active -> completed | abandoned | superseded | expired.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional

from monitor.paths import data_dir

_TS_FMT = "%Y-%m-%d %H:%M:%S"

MAX_TEXT_LEN = 300
MIN_DURATION_MIN = 5
MAX_DURATION_MIN = 8 * 60


def _now_str() -> str:
    return datetime.now().strftime(_TS_FMT)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS intentions (
               id         INTEGER PRIMARY KEY AUTOINCREMENT,
               text       TEXT NOT NULL,
               started_at TEXT NOT NULL,
               ends_at    TEXT NOT NULL,
               ended_at   TEXT,
               status     TEXT NOT NULL DEFAULT 'active'
           )"""
    )
    return conn


def get_active() -> Optional[Dict]:
    """Return the active session intention, or None.

    A past-due active intention is expired here as a side effect, so callers
    never see a stale session.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM intentions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        now = _now_str()
        if row["ends_at"] <= now:
            conn.execute(
                "UPDATE intentions SET status = 'expired', ended_at = ? WHERE id = ?",
                (row["ends_at"], row["id"]),
            )
            return None
        return dict(row)


def start(text: str, duration_min: int) -> Dict:
    """Start a session intention, superseding any currently active one."""
    text = " ".join((text or "").split())
    if not text:
        raise ValueError("intention text must not be empty")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(f"intention text must be at most {MAX_TEXT_LEN} characters")
    duration_min = int(duration_min)
    if not (MIN_DURATION_MIN <= duration_min <= MAX_DURATION_MIN):
        raise ValueError(
            f"duration must be {MIN_DURATION_MIN}-{MAX_DURATION_MIN} minutes"
        )

    now = datetime.now()
    started_at = now.strftime(_TS_FMT)
    ends_at = (now + timedelta(minutes=duration_min)).strftime(_TS_FMT)

    with _connect() as conn:
        conn.execute(
            "UPDATE intentions SET status = 'superseded', ended_at = ? WHERE status = 'active'",
            (started_at,),
        )
        cur = conn.execute(
            "INSERT INTO intentions (text, started_at, ends_at, status) VALUES (?,?,?, 'active')",
            (text, started_at, ends_at),
        )
        return {
            "id": cur.lastrowid,
            "text": text,
            "started_at": started_at,
            "ends_at": ends_at,
            "ended_at": None,
            "status": "active",
        }


def end_active(status: str = "completed") -> Optional[Dict]:
    """End the active intention (completed/abandoned). Returns it, or None."""
    if status not in ("completed", "abandoned"):
        raise ValueError("status must be 'completed' or 'abandoned'")
    active = get_active()
    if active is None:
        return None
    with _connect() as conn:
        conn.execute(
            "UPDATE intentions SET status = ?, ended_at = ? WHERE id = ?",
            (status, _now_str(), active["id"]),
        )
    active["status"] = status
    active["ended_at"] = _now_str()
    return active
