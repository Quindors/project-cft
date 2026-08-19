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

Finishing a session (completed or expired, not abandoned or superseded) hands
off to monitor/breaks.py, which starts the post-session break. The handoff is
guarded so it fires exactly once no matter which thread notices the session
ended — the runner loop, an HTTP request and the tray all race to read it.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional

from monitor.paths import data_dir

_TS_FMT = "%Y-%m-%d %H:%M:%S"

MAX_TEXT_LEN = 300
# A focus session is a real block of work: half an hour is the floor. Anything
# shorter is not a session, it's an interruption with a label on it.
MIN_DURATION_MIN = 30
MAX_DURATION_MIN = 8 * 60
# A pause is one minute: enough for the door or the kettle, not for a scroll.
PAUSE_MIN = 1


def _finished(intention: Dict, status: str) -> None:
    """A session just ended for good — roll into the post-session break, and
    record it on Beeminder if that is configured.

    breaks is imported lazily because it calls back into this module to close
    an active session, so a module-level import would be circular. The two
    hooks are kept independent: a Beeminder outage must not cost you a break.
    """
    try:
        from monitor import breaks
        breaks.after_session_end(intention, status)
    except Exception as e:
        print(f"[intention] post-session break skipped ({e})")

    try:
        from monitor import beeminder
        beeminder.after_session_end(intention, status)
    except Exception as e:
        print(f"[intention] beeminder push skipped ({e})")


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
    # The category the intention gate mapped the declared work to; the runner
    # uses it as a tie-breaking prior. NULL on pre-gate rows and gate failures.
    try:
        conn.execute("ALTER TABLE intentions ADD COLUMN llm_category TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    return conn


def get_active() -> Optional[Dict]:
    """Return the active session intention, or None.

    A past-due active intention is expired here as a side effect, so callers
    never see a stale session — and the expiry starts the post-session break.
    """
    expired = None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM intentions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        if row["ends_at"] > _now_str():
            return dict(row)
        # Past due. `AND status = 'active'` makes the flip single-shot: of the
        # threads that raced to read this row, only one gets rowcount 1 and so
        # only one starts the break.
        won = conn.execute(
            "UPDATE intentions SET status = 'expired', ended_at = ? "
            "WHERE id = ? AND status = 'active'",
            (row["ends_at"], row["id"]),
        ).rowcount == 1
        if won:
            expired = dict(row)
            expired["status"] = "expired"
            expired["ended_at"] = expired["ends_at"]

    # Outside the connection block: the hook writes to the same DB file.
    if expired is not None:
        _finished(expired, "expired")
    return None


def validate(text: str, duration_min) -> tuple:
    """Normalize and bounds-check a would-be intention -> (text, duration_min).

    Split out of start() so the HTTP layer can reject bad input before
    spending an LLM call on the intention gate.
    """
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
    return text, duration_min


def start(text: str, duration_min: int, llm_category: Optional[str] = None) -> Dict:
    """Start a session intention, superseding any currently active one.

    llm_category is the home category the intention gate mapped the declared
    work to; None when the gate was skipped or could not tell.
    """
    text, duration_min = validate(text, duration_min)

    now = datetime.now()
    started_at = now.strftime(_TS_FMT)
    ends_at = (now + timedelta(minutes=duration_min)).strftime(_TS_FMT)

    # A lingering pause belongs to the session about to be superseded; without
    # this it would sit on the new session as a stale "Paused" for up to a
    # minute. Real breaks are left alone.
    try:
        from monitor import breaks
        breaks.clear_pause()
    except Exception as e:
        print(f"[intention] could not clear the pause ({e})")

    with _connect() as conn:
        conn.execute(
            "UPDATE intentions SET status = 'superseded', ended_at = ? WHERE status = 'active'",
            (started_at,),
        )
        cur = conn.execute(
            "INSERT INTO intentions (text, started_at, ends_at, status, llm_category) "
            "VALUES (?,?,?, 'active', ?)",
            (text, started_at, ends_at, llm_category),
        )
        active = {
            "id": cur.lastrowid,
            "text": text,
            "started_at": started_at,
            "ends_at": ends_at,
            "ended_at": None,
            "status": "active",
            "llm_category": llm_category,
        }

    # Declaring an intention IS choosing focus: a session started while the
    # monitor sits in time-tracking mode would score drift and never say so,
    # which is a session in name only. One-way on purpose — ending a session
    # does not switch back, because "watch me again" is explicit and "stop
    # watching me" should never happen by side effect. Failure is logged, not
    # raised: the session matters more than the mode.
    try:
        from monitor import app_state
        if app_state.get_mode() != app_state.MODE_DISTRACTION_FREE:
            app_state.set_mode(app_state.MODE_DISTRACTION_FREE)
            print("[intention] session started — switching to focus mode")
    except Exception as e:
        print(f"[intention] could not switch to focus mode ({e})")

    return active


def pause_active() -> Dict:
    """Freeze the active session for PAUSE_MIN minute(s).

    A pause is a break in all but name — implemented AS one (source="pause")
    so the runner's hard stop (no polling, no keystrokes, no classification,
    no alerts) is reused wholesale — except that the session survives it: its
    ends_at moves out by the paused minute, so no session time is lost. The
    pause expires on its own; ending it early (breaks.end) hands the unused
    part of the minute back.

    Ordering matters: the extension lands before the pause starts, which
    guarantees ends_at stays past the pause's own end — that is what makes it
    safe for status.full_snapshot to read this store while a pause runs.
    """
    from monitor import breaks

    if breaks.get_active() is not None:
        raise ValueError("already on a break")
    active = get_active()
    if active is None:
        raise ValueError("no active focus session to pause")

    old_ends = active["ends_at"]
    new_ends = (
        datetime.strptime(old_ends, _TS_FMT) + timedelta(minutes=PAUSE_MIN)
    ).strftime(_TS_FMT)
    with _connect() as conn:
        # Single-shot, like every other transition: if the session ended
        # between the read above and this write, pause nothing.
        won = conn.execute(
            "UPDATE intentions SET ends_at = ? WHERE id = ? AND status = 'active'",
            (new_ends, active["id"]),
        ).rowcount == 1
    if not won:
        raise ValueError("no active focus session to pause")

    try:
        pause = breaks.start(PAUSE_MIN, source="pause", intention_id=active["id"])
    except Exception:
        # No pause means no frozen minute to compensate for — put it back.
        with _connect() as conn:
            conn.execute(
                "UPDATE intentions SET ends_at = ? WHERE id = ? AND status = 'active'",
                (old_ends, active["id"]),
            )
        raise

    active["ends_at"] = new_ends
    return {"active": active, "pause": pause}


def shift_active_ends(intention_id: int, delta_sec: int) -> bool:
    """Move an ACTIVE intention's ends_at by delta_sec (negative = earlier).

    Used by breaks.end() to hand back the unused part of a pause. The
    status='active' guard makes it a no-op if the session ended meanwhile.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT ends_at FROM intentions WHERE id = ? AND status = 'active'",
            (intention_id,),
        ).fetchone()
        if row is None:
            return False
        new_ends = (
            datetime.strptime(row["ends_at"], _TS_FMT) + timedelta(seconds=delta_sec)
        ).strftime(_TS_FMT)
        return conn.execute(
            "UPDATE intentions SET ends_at = ? WHERE id = ? AND status = 'active'",
            (new_ends, intention_id),
        ).rowcount == 1


def end_active(status: str = "completed", *, fire_hook: bool = True) -> Optional[Dict]:
    """End the active intention (completed/abandoned). Returns it, or None.

    Completing a session starts the post-session break; abandoning one does
    not. fire_hook=False is for callers that are already handling the
    aftermath themselves (breaks.start closing a session it is replacing).
    """
    if status not in ("completed", "abandoned"):
        raise ValueError("status must be 'completed' or 'abandoned'")
    active = get_active()
    if active is None:
        return None
    with _connect() as conn:
        # Single-shot, same as expiry: two "Done" clicks must not both count.
        won = conn.execute(
            "UPDATE intentions SET status = ?, ended_at = ? "
            "WHERE id = ? AND status = 'active'",
            (status, _now_str(), active["id"]),
        ).rowcount == 1
    if not won:
        return None
    # A pause cannot outlive its session. Cleared before the hook runs, so a
    # completed session's real break is not blocked by the pause still sitting
    # in the break slot.
    try:
        from monitor import breaks
        breaks.clear_pause()
    except Exception as e:
        print(f"[intention] could not clear the pause ({e})")
    active["status"] = status
    active["ended_at"] = _now_str()
    if fire_hook:
        _finished(active, status)
    return active
