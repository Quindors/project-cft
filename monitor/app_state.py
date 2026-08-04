# monitor/app_state.py
"""Small key/value app state in the local focus.db.

First user: the monitor mode. Two modes:

    distraction_free - full behavior: track activity AND pop alerts
                       (off-task popups / intention drift alerts).
    time_tracking    - track activity only; all alerts are suppressed.

Beyond the mode, the same table is the generic scratchpad for small bits of
cross-thread state (get_value/set_value): the active break, the timer widget's
screen position, and so on.

Like intentions, this opens short-lived connections so the runner loop, the
dashboard HTTP threads, and the tray can all touch it concurrently. The
dashboard toggles the mode via POST /api/mode; the runner re-reads it every
decision, so a toggle takes effect within one decision interval.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from monitor.paths import data_dir

MODE_DISTRACTION_FREE = "distraction_free"
MODE_TIME_TRACKING = "time_tracking"
MODES = (MODE_DISTRACTION_FREE, MODE_TIME_TRACKING)

DEFAULT_MODE = MODE_DISTRACTION_FREE


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS app_state (
               key   TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )"""
    )
    return conn


def get_mode() -> str:
    """Current monitor mode; unknown/missing values fall back to the default."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = 'mode'"
        ).fetchone()
    if row and row[0] in MODES:
        return row[0]
    return DEFAULT_MODE


def set_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return set_value("mode", mode)


# ---------- generic key/value ----------
# Small, infrequently-written scalars shared between the runner loop, the HTTP
# threads, the tray and the timer widget. Anything with structure is stored as
# a JSON string by the caller.

def get_value(key: str, default: Optional[str] = None) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else default


def set_value(key: str, value: str) -> str:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    return value


def delete_value(key: str) -> bool:
    """Remove a key. True if a row was actually deleted (single-shot guard)."""
    with _connect() as conn:
        return conn.execute("DELETE FROM app_state WHERE key = ?", (key,)).rowcount > 0
