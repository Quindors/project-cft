# monitor/app_state.py
"""Small key/value app state in the local focus.db.

The table is the generic scratchpad for small bits of cross-thread state
(get_value/set_value): the active break, the timer widget's screen position,
and so on.

Like intentions, this opens short-lived connections so the runner loop, the
dashboard HTTP threads, and the tray can all touch it concurrently.
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from monitor.paths import data_dir

# In-process signal, not DB state: the dashboard HTTP thread sets this the
# moment a human label is written, and the runner loop reloads its correction
# set immediately instead of waiting out the periodic refresh. The dashboard
# server and the runner share one process, so an Event is all it takes; the
# periodic refresh stays on as the backstop for anything else that writes rows.
corrections_changed = threading.Event()

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS app_state (
               key   TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )"""
    )
    return conn


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
