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

Status lifecycle: active -> completed | abandoned | superseded | expired,
with one detour: active <-> held. A held session is shelved, not ended: it
keeps its text, category, project and the time it had left, and is neither a
finished session (no break, no Beeminder datapoint, no plant in the forest)
nor the active one (nothing is judged against it) until it is resumed.

A finished session also carries an outcome, the user's own answer to "did you
finish it?": complete | incomplete | skipped, or NULL while unanswered. It is
a note on the session, not a status: an incomplete session ran its time and
counts exactly as a complete one does.

Finishing a session (completed or expired, not abandoned or superseded) hands
off to monitor/breaks.py, which starts the post-session break. The handoff is
guarded so it fires exactly once no matter which thread notices the session
ended — the runner loop, an HTTP request and the tray all race to read it.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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

# The shelf is for the thing you were pulled away from, not a backlog.
MAX_HELD = 5
# A finished session is asked about (complete / incomplete) for this long;
# after that the moment has passed and the dashboard stops bringing it up.
OUTCOME_ASK_HOURS = 12
OUTCOMES = ("complete", "incomplete", "skipped")


def dashboard_lease_sec() -> int:
    """How long a dashboard pause lasts without the page renewing it. The
    page says "still in front" every few seconds; a page that was closed or
    crashed says nothing, and the session resumes this long after its last
    word. Short, because that silence is time the session is not running."""
    try:
        return max(2, int(os.environ.get("CFT_DASHBOARD_LEASE_SEC") or 20))
    except ValueError:
        return 20


# One dashboard-presence message at a time: two tabs renewing the same lease
# at once would each move the session's end out by the same seconds.
_presence_lock = threading.RLock()


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
    # held_at: when a held session was shelved (NULL otherwise). outcome: the
    # user's complete/incomplete answer for a finished session.
    for col in ("untimed INTEGER NOT NULL DEFAULT 0", "paused_sec INTEGER NOT NULL DEFAULT 0",
                "held_at TEXT", "outcome TEXT"):
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
    # A held session stopped working the moment it was shelved.
    end = _ts(row.get("ended_at")) or _ts(row.get("held_at")) or now or datetime.now()
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
            "\u2014 or start a marathon, which has no clock to beat.")
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
    so the runner's hard stop (no polling, no classification, no alerts) is
    reused wholesale — except that the session survives it: its
    ends_at moves out by the paused minute, so no session time is lost. The
    pause expires on its own; ending it early (breaks.end) hands the unused
    part of the minute back.

    Ordering matters: the extension lands before the pause starts, which
    guarantees ends_at stays past the pause's own end — that is what makes it
    safe for status.full_snapshot to read this store while a pause runs.
    """
    from monitor import breaks

    # Asking for a pause from the dashboard happens while the dashboard's own
    # pause is holding the session. The asked-for one replaces it: it has a
    # length, and it lasts after the dashboard is left.
    _settle_dashboard_pause()
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
                f"A break during a marathon can be 1 to "
                f"{UNTIMED_PAUSE_MAX_MIN} minutes.",
                f"Pick a shorter break \u2014 take as many as you need. The "
                f"marathon clock waits while it runs.")
    else:
        if minutes not in (None, "", PAUSE_MIN, str(PAUSE_MIN)):
            raise UserError(
                f"A timed session can only pause for {PAUSE_MIN} minute.",
                "Use the pause for the door or the kettle. For a longer break, "
                "finish the session \u2014 or start a marathon next time, "
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


def _settle_dashboard_pause() -> None:
    """End a running dashboard pause, handing its unused seconds back."""
    try:
        from monitor import breaks
        with _presence_lock:
            cur = breaks.get_active()
            if cur is not None and cur.get("source") == breaks.SOURCE_DASHBOARD:
                breaks.end()
    except Exception as e:
        print(f"[intention] could not settle the dashboard pause ({e})")


def dashboard_presence(front: bool) -> Dict:
    """The dashboard page reporting whether it is the window in front.

    While it is, the running session is paused: its countdown stands still
    and nothing is watched, logged or scored. This is the SAME pause as the
    one-minute kind (monitor/breaks.py, source="dashboard") rather than a
    second clock: it sits in the break slot, the runner stands down for it
    exactly as for any pause, the session's ends_at moves out by the paused
    time and paused_sec grows by it, and ending it early hands the unused
    part back. What differs is only the length. Nobody knows up front how
    long the dashboard stays in front, so the pause is a short lease that
    each "still in front" message renews, and a page that goes quiet
    (closed, crashed, asleep) loses it within dashboard_lease_sec().

    It never stacks: while a pause the user asked for, or a real break, holds
    the slot, presence changes nothing. Returns {"paused": bool}: whether a
    dashboard pause is holding the session after this message.
    """
    from monitor import breaks

    lease = dashboard_lease_sec()
    with _presence_lock:
        cur = breaks.get_active()
        if cur is not None and cur.get("source") != breaks.SOURCE_DASHBOARD:
            return {"paused": False}
        if not front:
            if cur is not None:
                breaks.end()
            return {"paused": False}

        now = datetime.now()
        new_ends = (now + timedelta(seconds=lease)).strftime(_TS_FMT)
        if cur is not None:
            # Renewal: the session pays for the extra seconds first, so its
            # end always stays past the pause's own (see pause_active).
            delta = int((_ts(new_ends) - _ts(cur["ends_at"])).total_seconds())
            if delta <= 0:
                return {"paused": True}
            if not shift_active_ends(cur.get("intention_id"), delta, mirror=False):
                breaks.clear_pause()      # the session it froze is gone
                return {"paused": False}
            if not breaks.renew_lease(new_ends):
                shift_active_ends(cur.get("intention_id"), -delta, mirror=False)
                return {"paused": False}
            return {"paused": True}

        active = get_active()
        if active is None:
            return {"paused": False}
        if not shift_active_ends(active["id"], lease, mirror=False):
            return {"paused": False}
        try:
            breaks.start_lease(active["id"], lease)
        except Exception:
            shift_active_ends(active["id"], -lease, mirror=False)
            raise
        return {"paused": True}


def shift_active_ends(intention_id: int, delta_sec: int, *, mirror: bool = True) -> bool:
    """Move an ACTIVE intention's ends_at by delta_sec (negative = earlier).

    Used by breaks.end() to hand back the unused part of a pause, and by the
    dashboard pause to pay for each stretch of its lease. The status='active'
    guard makes it a no-op if the session ended meanwhile. mirror=False is
    for the lease renewals, which land every few seconds: the cloud copy
    hears about the pause once, when it ends.
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
    if full is not None and mirror:
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
    # "Done" on the dashboard is clicked while the dashboard's own pause is
    # holding the session. Settle that first (the unused part of its lease
    # goes back), so the session ends with exactly the time it worked.
    _settle_dashboard_pause()
    active = get_active()
    if active is None:
        return None
    if status == "completed" and active.get("untimed"):
        from monitor import breaks
        pause = breaks.get_active()
        left = pause.get("remaining_sec", 0) if breaks.is_pause(pause) else 0
        worked = worked_sec(active, pause_left_sec=left)
        if worked < UNTIMED_FINISH_MIN * 60:
            more = UNTIMED_FINISH_MIN * 60 - worked
            raise UserError(
                f"A marathon can be finished after {UNTIMED_FINISH_MIN} "
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


# ---------- on hold ----------

def _held_view(row: Dict) -> Dict:
    """A held row as the shelf shows it: what it is, and what it has left."""
    row = _attach_project(dict(row))
    held, ends = _ts(row.get("held_at")), _ts(row.get("ends_at"))
    row["worked_sec"] = worked_sec(row)
    row["planned_min"] = planned_min(row)
    # A marathon has no countdown to preserve; its stopwatch is worked_sec.
    row["remaining_sec"] = (None if row.get("untimed") or held is None or ends is None
                            else max(0, int((ends - held).total_seconds())))
    return row


def list_held() -> List[Dict]:
    """Sessions on hold, most recently shelved first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM intentions WHERE status = 'held' ORDER BY held_at DESC, id DESC"
        ).fetchall()
    return [_held_view(dict(r)) for r in rows]


def _shelf_full() -> UserError:
    return UserError(
        f"You already have {MAX_HELD} sessions on hold.",
        "Resume or drop one of them first \u2014 the shelf is for work you "
        "were pulled away from, and it stays short on purpose.")


def hold_active() -> Dict:
    """Shelve the running session. Returns it as the shelf shows it.

    Nothing is lost and nothing is counted: the session keeps the time it had
    left (a pause in progress is settled first, so that figure is exact), and
    it is not finished, so there is no break, no Beeminder datapoint and no
    plant. Monitoring carries on as it does with no session running.
    """
    from monitor import breaks

    with _presence_lock:
        if breaks.is_pause(breaks.get_active()):
            breaks.end()
    active = get_active()
    if active is None:
        raise UserError(
            "There's no focus session running to put on hold.",
            "Start one first \u2014 or resume one from the shelf under the "
            "session box, if that's the one you meant.")
    now = _now_str()
    with _connect() as conn:
        held = conn.execute(
            "SELECT COUNT(*) FROM intentions WHERE status = 'held'").fetchone()[0]
        if held >= MAX_HELD:
            raise _shelf_full()
        won = conn.execute(
            "UPDATE intentions SET status = 'held', held_at = ? "
            "WHERE id = ? AND status = 'active'",
            (now, active["id"]),
        ).rowcount == 1
        row = dict(conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (active["id"],)).fetchone()) if won else None
    if row is None:
        raise UserError(
            "The focus session ended while that was being processed.",
            "Refresh the dashboard to see where things stand.")
    print(f"[intention] session {row['id']} put on hold")
    _mirror(row)
    return _held_view(row)


def resume_held(intention_id) -> Optional[Dict]:
    """Bring a held session back as the active one. None if it isn't on hold.

    It comes back with the time it left with: its start and end both move
    forward by the time it spent on the shelf, so it is still one block of
    the declared length, placed where it is actually being worked.

    A session that is running at that moment is not lost and not ended: it
    takes the resumed one's place on the shelf (a swap, so the shelf cannot
    overflow). A break that is running ends, because resuming is choosing to
    work; a pause ends too, with its unused time handed back.
    """
    from monitor import breaks

    try:
        iid = int(intention_id)
    except (TypeError, ValueError):
        return None
    with _connect() as conn:
        target = conn.execute(
            "SELECT * FROM intentions WHERE id = ? AND status = 'held'", (iid,)).fetchone()
    if target is None:
        return None
    target = dict(target)

    with _presence_lock:
        if breaks.get_active() is not None:
            breaks.end()
    running = get_active()

    now = datetime.now()
    now_s = now.strftime(_TS_FMT)
    away = now - (_ts(target.get("held_at")) or now)
    started = ((_ts(target["started_at"]) or now) + away).strftime(_TS_FMT)
    ends = ((_ts(target["ends_at"]) or now) + away).strftime(_TS_FMT)
    shelved = None
    with _connect() as conn:
        won = conn.execute(
            "UPDATE intentions SET status = 'active', held_at = NULL, "
            "started_at = ?, ends_at = ? WHERE id = ? AND status = 'held'",
            (started, ends, iid),
        ).rowcount == 1
        if won and running is not None:
            conn.execute(
                "UPDATE intentions SET status = 'held', held_at = ? "
                "WHERE id = ? AND status = 'active' AND id != ?",
                (now_s, running["id"], iid))
            shelved = dict(conn.execute(
                "SELECT * FROM intentions WHERE id = ?", (running["id"],)).fetchone())
        row = dict(conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (iid,)).fetchone()) if won else None
    if row is None:
        return None     # two clicks raced; the other one resumed it
    print(f"[intention] session {iid} resumed from hold"
          + (f"; session {shelved['id']} took its place" if shelved else ""))
    if shelved is not None:
        _mirror(shelved)
    _mirror(row)
    return {"active": _attach_project(row),
            "shelved": _held_view(shelved) if shelved is not None else None}


def drop_held(intention_id) -> Optional[Dict]:
    """Give up on a held session: it becomes abandoned, dated to the moment
    it was shelved (the last time it was worked). None if it isn't on hold."""
    try:
        iid = int(intention_id)
    except (TypeError, ValueError):
        return None
    with _connect() as conn:
        won = conn.execute(
            "UPDATE intentions SET status = 'abandoned', ended_at = held_at, held_at = NULL "
            "WHERE id = ? AND status = 'held'", (iid,)).rowcount == 1
        row = dict(conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (iid,)).fetchone()) if won else None
    if row is None:
        return None
    _mirror(row)
    return row


# ---------- complete / incomplete ----------

def set_outcome(intention_id, outcome: Optional[str]) -> Optional[Dict]:
    """Record whether a FINISHED session's work got done: complete,
    incomplete, skipped (asked, and the user would rather not say), or None
    to clear. Returns the row, or None when no finished session has that id.

    The outcome changes nothing about how the session counts. A session that
    ran its time is a session whether or not the work fit into it, and an
    honest "incomplete" must not cost anything, or nobody would give one.
    """
    if outcome is not None and outcome not in OUTCOMES:
        raise UserError(
            "A finished session is either complete or incomplete.",
            "Pick one of the two on the dashboard \u2014 or skip the "
            "question if you'd rather not say for this one.")
    try:
        iid = int(intention_id)
    except (TypeError, ValueError):
        return None
    with _connect() as conn:
        changed = conn.execute(
            "UPDATE intentions SET outcome = ? "
            "WHERE id = ? AND status IN ('completed', 'expired')",
            (outcome, iid),
        ).rowcount == 1
        row = conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (iid,)).fetchone() if changed else None
    if row is None:
        return None
    row = dict(row)
    _mirror(row)
    return _attach_project(row)


def outcome_ask() -> Optional[Dict]:
    """The finished session the dashboard should ask about, or None.

    Only ever the most recent finished session, only while it is unanswered,
    and only for OUTCOME_ASK_HOURS after it ended: the question is about what
    just happened, not a backlog of every session that was never marked.
    """
    cutoff = (datetime.now() - timedelta(hours=OUTCOME_ASK_HOURS)).strftime(_TS_FMT)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM intentions WHERE status IN ('completed', 'expired') "
            "ORDER BY ended_at DESC, id DESC LIMIT 1").fetchone()
    if row is None or row["outcome"] is not None or (row["ended_at"] or "") < cutoff:
        return None
    row = _attach_project(dict(row))
    row["planned_min"] = planned_min(row)
    return row


def continue_session(intention_id, *, hold: bool = True) -> Optional[Dict]:
    """Pick a finished session's work up again: a NEW session with the same
    text, category, project and declared length. None when no finished
    session has that id.

    The finished session stays finished. It was a real block of work, it has
    already been counted and its break has started, so the continuation is
    the next block rather than the old one reopened. hold=True puts it on the
    shelf (for after the break); hold=False starts it now, through
    resume_held, so a break ends and a running session swaps onto the shelf.
    """
    try:
        iid = int(intention_id)
    except (TypeError, ValueError):
        return None
    with _connect() as conn:
        src = conn.execute(
            "SELECT * FROM intentions WHERE id = ? AND status IN ('completed', 'expired')",
            (iid,)).fetchone()
    if src is None:
        return None
    src = dict(src)
    untimed = bool(src.get("untimed"))
    minutes = MAX_DURATION_MIN if untimed else min(
        MAX_DURATION_MIN, max(MIN_DURATION_MIN, planned_min(src) or MIN_DURATION_MIN))
    now = datetime.now()
    now_s = now.strftime(_TS_FMT)
    with _connect() as conn:
        if hold and conn.execute(
                "SELECT COUNT(*) FROM intentions WHERE status = 'held'").fetchone()[0] >= MAX_HELD:
            raise _shelf_full()
        # Shelved the instant it is made: started_at == held_at, so resuming
        # it moves its whole window to the moment work begins.
        cur = conn.execute(
            "INSERT INTO intentions (text, started_at, ends_at, status, llm_category, "
            "project_id, untimed, paused_sec, held_at) VALUES (?,?,?, 'held', ?, ?, ?, 0, ?)",
            (src["text"], now_s, (now + timedelta(minutes=minutes)).strftime(_TS_FMT),
             src.get("llm_category"), src.get("project_id"), int(untimed), now_s))
        row = dict(conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (cur.lastrowid,)).fetchone())
    _mirror(row)
    if hold:
        return {"held": _held_view(row)}
    resumed = resume_held(row["id"])
    return resumed if resumed is not None else {"held": _held_view(row)}
