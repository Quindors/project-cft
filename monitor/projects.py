# monitor/projects.py
"""Projects: the long-lived thing a focus session is part of.

An intention is one sitting ("finish the stats problem set, 45 minutes"); a
project is what those sittings add up to ("Stats 101"). A session may belong
to a project, or to none. The project changes nothing about what a session IS
— the declared text is still what ALIGN is scored against, the gate still
checks it, the forest still plants one tree per finished session. What a
project adds:

  - a home category, optionally. Starting a session inside a project that has
    one is the same as picking that category by hand: the gate double-checks
    it, and while the session runs the home-category prior treats a hedged
    ALIGN inside that category as on-intent (monitor/runner.py). A project
    without a category leaves the gate to decide, exactly as before.
  - context for the classifier: the project's name and description ride
    along in the prompt so screens about the project land in its category
    rather than a site's default one. That is a CATEGORY hint only; it never
    raises ALIGN.
  - a way to see the forest by project, and to file old sessions under one.

Stored in the local focus.db next to intentions, same conventions: naive
local timestamps, short-lived connections, the local row is the truth and
the cloud mirror is best-effort.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from monitor.paths import data_dir
from monitor.problems import UserError

_TS_FMT = "%Y-%m-%d %H:%M:%S"

MAX_NAME_LEN = 60
MAX_DESCRIPTION_LEN = 500

STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

_UNSET = object()


def _now_str() -> str:
    return datetime.now().strftime(_TS_FMT)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)
    return conn


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create the projects table on any connection to focus.db. Also used by
    the dashboard's own read-write connection so a cascade (a category
    rename) never has to guess whether the table exists yet."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS projects (
               id          INTEGER PRIMARY KEY AUTOINCREMENT,
               name        TEXT NOT NULL,
               description TEXT NOT NULL DEFAULT '',
               category    TEXT,
               status      TEXT NOT NULL DEFAULT 'active',
               created_at  TEXT NOT NULL,
               archived_at TEXT
           )"""
    )


def _mirror(row: Dict) -> None:
    try:
        from monitor import supa_mirror
        supa_mirror.mirror_project(row)
    except Exception as e:
        print(f"[project] cloud mirror skipped ({e})")


def mirror_all() -> None:
    """Startup backfill, like intentions.mirror_all: every project, so the
    hosted dashboard can name the trees while the monitor is off."""
    try:
        from monitor import supa_mirror
        if not supa_mirror.enabled():
            return
        with _connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM projects").fetchall()]
        supa_mirror.mirror_projects(rows)
    except Exception as e:
        print(f"[project] cloud backfill skipped ({e})")


def _clean_name(name) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise UserError(
            "A project needs a name.",
            "Something you'd recognise in a list — “Stats 101” or "
            "“the novel” — it's what you'll pick when starting "
            "a session.")
    if len(name) > MAX_NAME_LEN:
        raise UserError(
            f"A project name can be at most {MAX_NAME_LEN} characters.",
            "Keep the name short and put the rest in the description — "
            "the description is what the classifier reads.")
    return name


def _clean_description(description) -> str:
    description = " ".join(str(description or "").split())
    if len(description) > MAX_DESCRIPTION_LEN:
        raise UserError(
            f"A project description can be at most {MAX_DESCRIPTION_LEN} characters.",
            "A sentence or two about what the work is and where it happens "
            "is plenty — it's a hint for the classifier, not a plan.")
    return description


def _clean_category(category) -> Optional[str]:
    category = " ".join(str(category or "").split())
    return category or None


def _clash(conn: sqlite3.Connection, name: str, except_id: Optional[int] = None) -> bool:
    row = conn.execute(
        "SELECT id FROM projects WHERE lower(name) = lower(?)", (name,)
    ).fetchone()
    return row is not None and row["id"] != except_id


def _with_stats(conn: sqlite3.Connection, rows: List[Dict]) -> List[Dict]:
    """Finished-session count and minutes per project (the same completed +
    expired set the forest plants). Zero, not an error, before the first
    session was ever declared."""
    for r in rows:
        r["sessions"] = 0
        r["minutes"] = 0
    if not rows:
        return rows
    by_id = {r["id"]: r for r in rows}
    try:
        stats = conn.execute(
            "SELECT project_id, COUNT(*) AS n, "
            "       SUM((julianday(ended_at) - julianday(started_at)) * 1440) AS mins "
            "FROM intentions WHERE project_id IS NOT NULL "
            "  AND status IN ('completed', 'expired') GROUP BY project_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return rows   # no intentions table, or one from before projects
    for s in stats:
        r = by_id.get(s["project_id"])
        if r is not None:
            r["sessions"] = int(s["n"] or 0)
            r["minutes"] = int(round(s["mins"] or 0))
    return rows


def list_all(include_archived: bool = False) -> List[Dict]:
    """Projects, active ones first and newest first within a status, each
    with its finished-session count and minutes."""
    with _connect() as conn:
        if include_archived:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY status = 'archived', id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM projects WHERE status = 'active' ORDER BY id DESC"
            ).fetchall()
        return _with_stats(conn, [dict(r) for r in rows])


def get(project_id) -> Optional[Dict]:
    """One project by id, or None. Tolerates junk ids: the dashboard sends
    whatever was in its select."""
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    return dict(row) if row else None


def create(name, description="", category=None) -> Dict:
    """Add a project. The category, if any, must already have been validated
    against the real categories by the caller (the HTTP layer does it the
    same way it validates a session's pick)."""
    name = _clean_name(name)
    description = _clean_description(description)
    category = _clean_category(category)
    with _connect() as conn:
        if _clash(conn, name):
            raise UserError(
                f"You already have a project called “{name}”.",
                "Pick a different name, or open the existing one on the "
                "Projects tab — two projects with one name would split "
                "their sessions.")
        cur = conn.execute(
            "INSERT INTO projects (name, description, category, status, created_at) "
            "VALUES (?,?,?,?,?)",
            (name, description, category, STATUS_ACTIVE, _now_str()),
        )
        row = dict(conn.execute("SELECT * FROM projects WHERE id = ?", (cur.lastrowid,)).fetchone())
    _mirror(row)
    return row


def update(project_id, *, name=_UNSET, description=_UNSET, category=_UNSET,
           archived=_UNSET) -> Dict:
    """Edit a project. Only the fields given change; category=None clears it.

    Archiving hides a project from the pickers without touching its
    sessions — the forest keeps every tree. Unarchiving brings it back.
    """
    with _connect() as conn:
        try:
            pid = int(project_id)
        except (TypeError, ValueError):
            pid = -1
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise UserError(
                "That project isn't there any more.",
                "Refresh the Projects tab — it may have been deleted since "
                "the page loaded.")
        row = dict(row)
        if name is not _UNSET:
            new_name = _clean_name(name)
            if _clash(conn, new_name, except_id=pid):
                raise UserError(
                    f"You already have a project called “{new_name}”.",
                    "Pick a different name — two projects with one name "
                    "would split their sessions.")
            row["name"] = new_name
        if description is not _UNSET:
            row["description"] = _clean_description(description)
        if category is not _UNSET:
            row["category"] = _clean_category(category)
        if archived is not _UNSET:
            want = STATUS_ARCHIVED if archived else STATUS_ACTIVE
            if want != row["status"]:
                row["status"] = want
                row["archived_at"] = _now_str() if archived else None
        conn.execute(
            "UPDATE projects SET name = ?, description = ?, category = ?, "
            "status = ?, archived_at = ? WHERE id = ?",
            (row["name"], row["description"], row["category"], row["status"],
             row["archived_at"], pid),
        )
    _mirror(row)
    return row


def delete(project_id) -> Dict:
    """Remove a project. Its sessions stay — they just stop belonging to
    anything, the way a session with no project already does."""
    with _connect() as conn:
        try:
            pid = int(project_id)
        except (TypeError, ValueError):
            pid = -1
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise UserError(
                "That project isn't there any more.",
                "Refresh the Projects tab — it may already have been "
                "deleted.")
        row = dict(row)
        unlinked = 0
        try:
            unlinked = conn.execute(
                "UPDATE intentions SET project_id = NULL WHERE project_id = ?", (pid,)
            ).rowcount
        except sqlite3.OperationalError:
            pass   # no sessions were ever declared, so none to unlink
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
    # Every unlinked session is a changed row the cloud needs to hear about.
    if unlinked:
        try:
            from monitor import intentions
            intentions.mirror_all()
        except Exception as e:
            print(f"[project] session re-mirror skipped ({e})")
    try:
        from monitor import supa_mirror
        supa_mirror.mirror_project_delete(pid)
    except Exception as e:
        print(f"[project] cloud delete skipped ({e})")
    row["unlinked_sessions"] = unlinked
    return row


def cascade_category(conn: sqlite3.Connection, old: str, new: Optional[str]) -> List[Dict]:
    """A category was renamed (new is its new name) or deleted (new is None):
    projects that pointed at it follow. Runs on the caller's connection,
    inside the caller's transaction, and returns the changed rows so the
    caller can mirror them once it has committed."""
    ensure_table(conn)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM projects WHERE category = ? COLLATE NOCASE", (old,)
    ).fetchall()]
    if rows:
        conn.execute(
            "UPDATE projects SET category = ? WHERE category = ? COLLATE NOCASE", (new, old)
        )
        for r in rows:
            r["category"] = new
    return rows


def mirror_rows(rows: List[Dict]) -> None:
    for r in rows:
        _mirror(r)
