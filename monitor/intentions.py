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
from monitor.problems import UserError

_TS_FMT = "%Y-%m-%d %H:%M:%S"

MAX_TEXT_LEN = 300
# A focus session is a real block of work: half an hour is the floor. Anything
# shorter is not a session, it's an interruption with a label on it.
MIN_DURATION_MIN = 30
MAX_DURATION_MIN = 8 * 60
# A pause is one minute: enough for the door or the kettle, not for a scroll.
PAUSE_MIN = 1


def _finished(intention: Dict, status: str, *, start_break: bool = True) -> None:
    """A session just ended for good — roll into the post-session break, and
    record it on Beeminder if that is configured.

    breaks is imported lazily because it calls back into this module to close
    an active session, so a module-level import would be circular. The two
    hooks are kept independent: a Beeminder outage must not cost you a break,
    and a caller that owns the break slot itself (start_break=False) must
    still get the session recorded on Beeminder.
    """
    if start_break:
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

    _pop(f"session {status}")


def _pop(reason: str) -> None:
    """Bring the dashboard forward for a transition (monitor/browser.py). A
    browser problem is logged, never raised into the session bookkeeping."""
    try:
        from monitor import browser
        browser.pop_dashboard_async(reason)
    except Exception as e:
        print(f"[intention] pop-out skipped ({e})")


def _mirror(row: Dict) -> None:
    """Push one intention row to the cloud mirror, if configured. Never raises:
    the local write already happened and is the truth."""
    try:
        from monitor import supa_mirror
        supa_mirror.mirror_intention(row)
    except Exception as e:
        print(f"[intention] cloud mirror skipped ({e})")


def mirror_all() -> None:
    """Startup backfill: every intention ever, so the hosted dashboard can draw
    the forest from the cloud while this monitor is off."""
    try:
        from monitor import supa_mirror
        if not supa_mirror.enabled():
            return
        with _connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM intentions").fetchall()]
        supa_mirror.mirror_intentions(rows)
    except Exception as e:
        print(f"[intention] cloud backfill skipped ({e})")


def backfill_categories(limit: int = 50) -> int:
    """Classify finished sessions that predate the intention gate.

    Runs the same gate a new intention goes through, on the stored text, so
    an old "personal statement" session is a College Application session the
    way a new one would be — not whatever the monitor happened to log most
    during it. Returns the number of rows filled. One LLM call per row,
    bounded per call, and safe to run on every start: rows the gate cannot
    place stay NULL and are simply retried next time.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, text FROM intentions WHERE llm_category IS NULL "
            "AND status IN ('completed', 'expired') ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return 0
    from monitor import intent_gate
    filled = 0
    for row in rows:
        try:
            verdict = intent_gate.check(row["text"])
        except Exception as e:
            print(f"[intention] backfill check failed ({e})")
            break
        if verdict.get("reason") == "check unavailable":
            break  # no LLM right now; every further row would fail the same way
        cat = verdict.get("category")
        if not cat:
            continue
        with _connect() as conn:
            conn.execute(
                "UPDATE intentions SET llm_category = ? WHERE id = ? AND llm_category IS NULL",
                (cat, row["id"]),
            )
            full = conn.execute(
                "SELECT * FROM intentions WHERE id = ?", (row["id"],)
            ).fetchone()
        if full is not None:
            _mirror(dict(full))
        filled += 1
    if filled:
        print(f"[intention] backfilled categories on {filled} pre-gate session(s)")
    return filled


def set_category(intention_id: int, category: Optional[str]) -> Optional[Dict]:
    """Relabel a FINISHED session (completed/expired); None clears the label.

    Returns the row, or None when no finished session has that id — an
    active session's category is set at the start, not edited from the
    forest. Mirrored like every other transition.
    """
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE intentions SET llm_category = ? "
            "WHERE id = ? AND status IN ('completed', 'expired')",
            (category, intention_id),
        ).rowcount == 1
        row = conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone() if changed else None
    if row is None:
        return None
    row = dict(row)
    _mirror(row)
    return row


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
    # The project the session was declared under (monitor/projects.py), or
    # NULL for a session that stands on its own.
    try:
        conn.execute("ALTER TABLE intentions ADD COLUMN project_id INTEGER")
    except sqlite3.OperationalError:
        pass  # column already exists
    return conn


def _attach_project(row: Dict) -> Dict:
    """Hang the session's project (id, name, description, category) on the
    row as `project`, or None. Read by the classifier prompt and the status
    snapshot; the mirror whitelists its own columns, so it never ships."""
    row["project"] = None
    if row.get("project_id") is not None:
        try:
            from monitor import projects
            p = projects.get(row["project_id"])
        except Exception as e:
            print(f"[intention] project lookup failed ({e})")
            p = None
        if p is not None:
            row["project"] = {k: p.get(k) for k in ("id", "name", "description", "category")}
    return row


def set_project(intention_id: int, project_id: Optional[int]) -> Optional[Dict]:
    """File a FINISHED session under a project (None takes it out of one).

    Same shape as set_category: an active session's project is set at the
    start; this is for the forest's relabel card. Returns the row or None
    when no finished session has that id.
    """
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE intentions SET project_id = ? "
            "WHERE id = ? AND status IN ('completed', 'expired')",
            (project_id, intention_id),
        ).rowcount == 1
        row = conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone() if changed else None
    if row is None:
        return None
    row = dict(row)
    _mirror(row)
    return _attach_project(row)


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
            return _attach_project(dict(row))
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
        _mirror(expired)
        _finished(expired, "expired")
    return None


def validate(text: str, duration_min) -> tuple:
    """Normalize and bounds-check a would-be intention -> (text, duration_min).

    Split out of start() so the HTTP layer can reject bad input before
    spending an LLM call on the intention gate.
    """
    text = " ".join((text or "").split())
    if not text:
        raise UserError(
            "A focus session needs something to aim at.",
            "Write what you're about to work on \u2014 \u201cfinish the stats "
            "problem set\u201d beats \u201cwork\u201d, because the monitor "
            "checks what's on screen against it.")
    if len(text) > MAX_TEXT_LEN:
        raise UserError(
            f"That's longer than the {MAX_TEXT_LEN} characters a session "
            f"intention can hold.",
            "Shorten it to the one thing you're working on \u2014 a whole "
            "to-do list is harder to judge your screen against than a single "
            "aim.")
    duration_min = int(duration_min)
    if not (MIN_DURATION_MIN <= duration_min <= MAX_DURATION_MIN):
        raise UserError(
            f"A focus session has to run between {MIN_DURATION_MIN} and "
            f"{MAX_DURATION_MIN} minutes.",
            "Pick one of the preset lengths, or type anything in that range. "
            "You can finish early \u2014 the session counts what you actually "
            "sat through.")
    return text, duration_min


def start(text: str, duration_min: int, llm_category: Optional[str] = None,
          project_id: Optional[int] = None) -> Dict:
    """Start a session intention, superseding any currently active one.

    llm_category is the home category the intention gate mapped the declared
    work to; None when the gate was skipped or could not tell. project_id
    files the session under a project (monitor/projects.py); the caller has
    already checked the project exists and is active.
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
        superseded = [
            dict(r) for r in conn.execute(
                "SELECT * FROM intentions WHERE status = 'active'"
            ).fetchall()
        ]
        conn.execute(
            "UPDATE intentions SET status = 'superseded', ended_at = ? WHERE status = 'active'",
            (started_at,),
        )
        cur = conn.execute(
            "INSERT INTO intentions (text, started_at, ends_at, status, llm_category, project_id) "
            "VALUES (?,?,?, 'active', ?, ?)",
            (text, started_at, ends_at, llm_category, project_id),
        )
        active = {
            "id": cur.lastrowid,
            "text": text,
            "started_at": started_at,
            "ends_at": ends_at,
            "ended_at": None,
            "status": "active",
            "llm_category": llm_category,
            "project_id": project_id,
        }

    for row in superseded:
        row["status"] = "superseded"
        row["ended_at"] = started_at
        _mirror(row)
    _mirror(active)
    _pop("session started")
    return _attach_project(active)


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
        raise UserError(
            "You're on a break, so there's no session to pause.",
            "End the break first if you want to get back to work.")
    active = get_active()
    if active is None:
        raise UserError(
            "There's no focus session running to pause.",
            "Start one first \u2014 pausing only holds a session that's "
            "already going.")

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
        raise UserError(
            "The focus session ended while that was being processed.",
            "Refresh the dashboard to see where things stand.")

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
    _mirror(active)
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
        won = conn.execute(
            "UPDATE intentions SET ends_at = ? WHERE id = ? AND status = 'active'",
            (new_ends, intention_id),
        ).rowcount == 1
        full = dict(conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone()) if won else None
    if full is not None:
        _mirror(full)
    return won


def end_active(status: str = "completed", *, fire_hook: bool = True) -> Optional[Dict]:
    """End the active intention (completed/abandoned). Returns it, or None.

    Completing a session starts the post-session break; abandoning one does
    not. fire_hook=False is for callers that are already handling the break
    slot themselves (breaks.start closing a session it is replacing) — the
    Beeminder record is independent of the break and is pushed either way.
    """
    if status not in ("completed", "abandoned"):
        raise UserError(
            "A session can only be finished or abandoned.",
            "Use \u201cDone\u201d if you saw it through, or \u201cGive "
            "up\u201d if you didn't \u2014 only finished sessions count "
            "towards your goals.")
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
    _mirror(active)
    _finished(active, status, start_break=fire_hook)
    return active
