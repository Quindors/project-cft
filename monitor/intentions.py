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
# A focus session is a real block of work: twenty minutes is the floor (the
# user asked for 20-minute sessions on 2026-10-11; they earn a 10-minute
# break, see breaks.break_minutes_after). Anything shorter is not a session,
# it's an interruption with a label on it.
MIN_DURATION_MIN = 20
MAX_DURATION_MIN = 8 * 60
# A pause is one minute: enough for the door or the kettle, not for a scroll.
PAUSE_MIN = 1

# Untimed sessions run on a stopwatch instead of a countdown. They can be
# finished ("Done", which earns a break and counts as a session) only after
# UNTIMED_FINISH_MIN minutes of WORKED time — breaks don't count — and
# abandoned at any time. Breaks inside one may run up to UNTIMED_PAUSE_MAX_MIN
# minutes each, as many as needed; the stopwatch stands still while they run.
# ends_at still exists, as a hard cap: an untimed session left running expires
# like any other once it reaches MAX_DURATION_MIN.
UNTIMED_FINISH_MIN = 20
UNTIMED_PAUSE_MAX_MIN = 5


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
    # untimed: 1 for a stopwatch session. paused_sec: seconds spent in
    # in-session breaks/pauses so far, so worked time = elapsed - paused_sec.
    for col in ("untimed INTEGER NOT NULL DEFAULT 0", "paused_sec INTEGER NOT NULL DEFAULT 0"):
        try:
            conn.execute(f"ALTER TABLE intentions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def _ts(s: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.strptime(s, _TS_FMT) if s else None
    except ValueError:
        return None


def worked_sec(row: Dict, *, now: Optional[datetime] = None, pause_left_sec: int = 0) -> int:
    """Seconds of the session spent working: from its start to its end (or
    now), minus its breaks. paused_sec already includes a pause's full length
    from the moment it starts, so while one is still running, pass its
    remaining seconds back in (pause_left_sec) to get the frozen stopwatch."""
    start = _ts(row.get("started_at"))
    if start is None:
        return 0
    end = _ts(row.get("ended_at")) or now or datetime.now()
    paused = int(row.get("paused_sec") or 0) - int(pause_left_sec or 0)
    return max(0, int((end - start).total_seconds()) - max(0, paused))


def planned_min(row: Dict) -> Optional[int]:
    """A timed session's length as declared, in minutes; None for untimed."""
    if row.get("untimed"):
        return None
    start, ends = _ts(row.get("started_at")), _ts(row.get("ends_at"))
    if start is None or ends is None:
        return None
    return max(0, round(((ends - start).total_seconds() - int(row.get("paused_sec") or 0)) / 60))


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


def validate(text: str, duration_min, untimed: bool = False) -> tuple:
    """Normalize and bounds-check a would-be intention -> (text, duration_min).

    Split out of start() so the HTTP layer can reject bad input before
    spending an LLM call on the intention gate. An untimed session has no
    duration to check: duration_min comes back None.
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
    if untimed:
        return text, None
    duration_min = int(duration_min)
    if not (MIN_DURATION_MIN <= duration_min <= MAX_DURATION_MIN):
        raise UserError(
            f"A focus session has to run between {MIN_DURATION_MIN} and "
            f"{MAX_DURATION_MIN} minutes.",
            "Pick one of the preset lengths, or type anything in that range "
            "\u2014 or start an untimed session on a stopwatch instead.")
    return text, duration_min


def start(text: str, duration_min, llm_category: Optional[str] = None,
          project_id: Optional[int] = None, *, untimed: bool = False) -> Dict:
    """Start a session intention, superseding any currently active one.

    llm_category is the home category the intention gate mapped the declared
    work to; None when the gate was skipped or could not tell. project_id
    files the session under a project (monitor/projects.py); the caller has
    already checked the project exists and is active.
    """
    text, duration_min = validate(text, duration_min, untimed)

    now = datetime.now()
    started_at = now.strftime(_TS_FMT)
    # An untimed session's ends_at is only its hard cap (see UNTIMED_*).
    ends_at = (now + timedelta(minutes=MAX_DURATION_MIN if untimed else duration_min)
               ).strftime(_TS_FMT)

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
            "INSERT INTO intentions (text, started_at, ends_at, status, llm_category, "
            "project_id, untimed, paused_sec) VALUES (?,?,?, 'active', ?, ?, ?, 0)",
            (text, started_at, ends_at, llm_category, project_id, int(bool(untimed))),
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
            "untimed": int(bool(untimed)),
            "paused_sec": 0,
        }

    for row in superseded:
        row["status"] = "superseded"
        row["ended_at"] = started_at
        _mirror(row)
    _mirror(active)
    return _attach_project(active)


def pause_active(minutes=None) -> Dict:
    """Freeze the active session for PAUSE_MIN minute(s) — or, in an untimed
    session, for 1 to UNTIMED_PAUSE_MAX_MIN minutes (default the maximum).

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

    if active.get("untimed"):
        try:
            length = UNTIMED_PAUSE_MAX_MIN if minutes in (None, "") else int(minutes)
        except (TypeError, ValueError):
            length = -1
        if not (1 <= length <= UNTIMED_PAUSE_MAX_MIN):
            raise UserError(
                f"A break inside an untimed session can be 1 to "
                f"{UNTIMED_PAUSE_MAX_MIN} minutes.",
                f"Pick a shorter break \u2014 take as many as you need. The "
                f"stopwatch stands still while it runs.")
    else:
        if minutes not in (None, "", PAUSE_MIN, str(PAUSE_MIN)):
            raise UserError(
                f"A timed session can only pause for {PAUSE_MIN} minute.",
                "Use the pause for the door or the kettle. For a longer break, "
                "finish the session \u2014 or start an untimed one next time, "
                "which allows breaks of up to five minutes.")
        length = PAUSE_MIN

    old_ends = active["ends_at"]
    new_ends = (
        datetime.strptime(old_ends, _TS_FMT) + timedelta(minutes=length)
    ).strftime(_TS_FMT)
    with _connect() as conn:
        # Single-shot, like every other transition: if the session ended
        # between the read above and this write, pause nothing.
        won = conn.execute(
            "UPDATE intentions SET ends_at = ?, paused_sec = paused_sec + ? "
            "WHERE id = ? AND status = 'active'",
            (new_ends, length * 60, active["id"]),
        ).rowcount == 1
    if not won:
        raise UserError(
            "The focus session ended while that was being processed.",
            "Refresh the dashboard to see where things stand.")

    try:
        pause = breaks.start(length, source="pause", intention_id=active["id"])
    except Exception:
        # No pause means no frozen minute to compensate for — put it back.
        with _connect() as conn:
            conn.execute(
                "UPDATE intentions SET ends_at = ?, paused_sec = paused_sec - ? "
                "WHERE id = ? AND status = 'active'",
                (old_ends, length * 60, active["id"]),
            )
        raise

    active["ends_at"] = new_ends
    active["paused_sec"] = int(active.get("paused_sec") or 0) + length * 60
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
        # The handed-back seconds were never spent paused, either.
        won = conn.execute(
            "UPDATE intentions SET ends_at = ?, "
            "paused_sec = MAX(0, paused_sec + ?) WHERE id = ? AND status = 'active'",
            (new_ends, delta_sec, intention_id),
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
    if status == "completed" and active.get("untimed"):
        from monitor import breaks
        pause = breaks.get_active()
        left = pause.get("remaining_sec", 0) if pause and pause.get("source") == "pause" else 0
        worked = worked_sec(active, pause_left_sec=left)
        if worked < UNTIMED_FINISH_MIN * 60:
            more = UNTIMED_FINISH_MIN * 60 - worked
            raise UserError(
                f"An untimed session can be finished after {UNTIMED_FINISH_MIN} "
                f"minutes of work \u2014 this one has {worked // 60}.",
                f"Keep going for about {max(1, -(-more // 60))} more minute"
                f"{'' if -(-more // 60) == 1 else 's'}, or abandon it if you're "
                f"stopping for good. Breaks don't count toward the {UNTIMED_FINISH_MIN}.")
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
