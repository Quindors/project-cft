# monitor/dashboard_server.py
"""Local JSON API for the focus dashboard (+ the bundled build, if present).

The dashboard reads its data from this server's /api endpoints, backed by the
local SQLite DB. Two ways to reach it:

  1. Hosted (Vercel): the page is served from https://<app>.vercel.app and
     fetches this API cross-origin at http://127.0.0.1:<CFT_API_PORT>. That
     needs a FIXED port (so the hosted page knows where to look) and CORS.
  2. Bundled: if a `dashboard/` build was staged into the exe, it's served from
     this same origin (no CORS needed).

Endpoints:
    GET  /api/status            -> the one resolved view of what's happening now:
                                   {phase, state, label, ends_at, remaining_sec,
                                    intention, break, tier, category, ...}
                                   The timer widget, the tray and the dashboard
                                   all render from this (see monitor/status.py).
    GET  /api/break             -> {active: {started_at, ends_at, remaining_sec} | null}
    POST /api/break             -> {duration_min?} starts a break: tracking AND
                                   alerts stop until it runs out
    POST /api/break/end         -> ends the break early, resuming the monitor
    GET  /api/categories        -> [{name, description, is_productive}, ...]
    GET  /api/focus?since=<ts>  -> [{timestamp, category_name, ai_category,
                                    confidence, intention_id, align_score}, ...]
                                   — category_name is the effective label
                                   (human correction wins over the classifier,
                                   as Beeminder counts it); ai_category is the
                                   classifier's original; intention_id and
                                   align_score (null outside a focus session)
                                   let the page apply Beeminder's other rule —
                                   a neutral category is productive only while
                                   declared (beeminder.NEVER_DECLARED)
    GET  /api/events?limit=<n>  -> recent rows with id/current_window/human_label,
                                   plus intention_id/align_score (0-1 intent
                                   alignment, null outside a focus session),
                                   for the dashboard's review (re-classify) tab
    GET  /api/intention         -> {active: {id, text, started_at, ends_at} | null}
    GET  /api/sessions?limit=<n>-> finished focus sessions (completed or expired
                                   — the same set Beeminder counts), newest
                                   first: [{id, text, started_at, ended_at,
                                   status, llm_category, avg_align,
                                   top_category, top_share}, ...].
                                   avg_align is the mean 0-1 ALIGN over the
                                   session's scored rows, null if none were
                                   scored. llm_category is what the intention
                                   gate DECLARED the session to be (null before
                                   the gate existed); top_category is what the
                                   monitor actually OBSERVED — the session's
                                   most-logged effective category (human
                                   correction wins) — with top_share its
                                   fraction of the session's classifiable rows.
                                   Feeds the dashboard's forest tab.
    GET  /api/beeminder         -> {configured, user, focus_goal, sessions_goal,
                                    dry_run, sync_running, last_push, goal_check,
                                    problem} — never the token. goal_check is a
                                   read-only probe of the goal (cached 60s);
                                   problem is the one actionable issue, with a
                                   reason of missing_goal | bad_token |
                                   unreachable | other, or null when healthy
    POST /api/beeminder         -> {user, token?, focus_goal, sessions_goal?,
                                    create_goal?, hours_per_day?, leeway_days?}
                                   (token may be omitted once configured — the
                                   stored one is reused, e.g. replacing a goal
                                   that was deleted on beeminder.com)
                                   verifies against Beeminder (or, with
                                   create_goal, creates the Do More goal on
                                   the user's account first), persists to the
                                   .env and starts the sync; 400 with a reason
                                   when the credentials don't check out
    POST /api/beeminder/live    -> leaves dry-run mode (BEEMINDER_DRY_RUN=0)
    POST /api/beeminder/disconnect -> blanks the Beeminder credentials
    POST /api/email-report      -> also answers {introduced: [...]}, the
                                   addresses that were sent a one-off
                                   introduction because they were new
    GET  /api/email-report      -> {configured, recipients, smtp_host, smtp_port,
                                    smtp_user, security, sender, day, hour,
                                    next_send, pending, last_sent, last_attempt,
                                    scheduler_running, problem} — never the
                                   password (monitor/email_report.py)
    GET  /api/email-report/preview -> the report for the last seven days, as
                                   the HTML that would be mailed
    POST /api/email-report      -> {to, smtp_host, smtp_port?, smtp_user?,
                                    smtp_password?, sender?, security?, day?,
                                    hour?, send_test?} — `to` is a list or a
                                   comma-separated string; the password may be
                                   omitted once stored. Logs in to the SMTP
                                   server before saving anything (400 with the
                                   reason otherwise), persists to the .env and
                                   starts the weekly scheduler; send_test mails
                                   the last seven days right away
    POST /api/email-report/send -> mails the last seven days now
    POST /api/email-report/disconnect -> blanks the recipients and SMTP settings
    POST /api/correct           -> {id, human_label} sets a human correction on a
                                   row. The label must name an existing category
                                   (matched case-insensitively, stored in the
                                   category's exact casing); null/"" clears it
    GET  /api/corrections       -> the learned correction rules, one per exact
                                   window title whose recent rows (the same
                                   window load_corrections reads) carry a
                                   human label differing from the AI's
                                   category: [{window_title, ai_category,
                                     corrected_category, rows, last_at}, ...]
    POST /api/corrections/update-> {window_title, human_label} retargets the
                                   rule's corrected rows — matched across all
                                   normalized spellings of the title, the way
                                   the decider matches — to a new category
                                   (human_label of null/"" removes the
                                   correction; confirmation rows where the
                                   label merely agrees with the AI are never
                                   touched, and a retarget never creates one)
    POST /api/category          -> {name, description, is_productive} adds a category
    POST /api/category/update   -> {name, new_name?, description?, is_productive?}
                                   edits a category; a rename cascades to
                                   focus_logs.category_name / human_label
    POST /api/category/delete   -> {name, reassign_to?} deletes a category and
                                   moves its logged rows to reassign_to
                                   (default "Ambiguous"); Ambiguous/System are
                                   protected because the classifier needs them,
                                   and a non-productive category's rows can
                                   never be reassigned into a productive one —
                                   that would retroactively count them as
                                   focused time
    GET  /api/projects          -> [{id, name, description, category, status,
                                     created_at, archived_at, sessions, minutes}]
                                   active projects (?all=1 adds archived ones)
    POST /api/project           -> {name, description?, category?} adds a
                                   project; the category, if given, must be a
                                   real one that isn't marked unproductive
    POST /api/project/update    -> {id, name?, description?, category?,
                                   archived?} edits a project (category null
                                   clears it; archived hides it from pickers)
    POST /api/project/delete    -> {id} removes a project; its sessions stay,
                                   with no project
    POST /api/sessions/project  -> {id, project_id|null} files a finished
                                   session under a project
    POST /api/intention         -> {text, duration_min, category?, confirm?,
                                   project_id?}
                                   starts a session intention (supersedes any
                                   active one). category is the user's pick:
                                   validated against the real categories (400
                                   unknown, 422 rejected if it is marked
                                   unproductive) and only CHECKED by the gate —
                                   when the gate reads the text as UNPRODUCTIVE
                                   work it answers 409 {needs_confirm, picked,
                                   suggested, reason} once; confirm=true keeps
                                   the pick anyway. A disagreement between two
                                   productive categories keeps the pick silently.
                                   Without a pick the gate decides, and 422
                                   {rejected, category, reason} refuses work
                                   that maps to an unproductive category
    POST /api/sessions/category -> {id, category|null} relabels a FINISHED
                                   session (validated like a correction; null
                                   clears); 404 if no such finished session
    POST /api/intention/end     -> {status?: completed|abandoned} ends the active one;
                                   completing one rolls into the post-session break
    POST /api/intention/pause   -> pauses the active session for 1 minute: the
                                   countdown freezes (ends_at moves out by the
                                   same minute) and monitoring stops, exactly
                                   like a short break; resumes by itself when
                                   the minute runs out. POST /api/break/end
                                   resumes early and hands the unused part of
                                   the minute back

GET /intention serves a built-in page for starting/ending a session intention
and taking a break — usable even without the bundled dashboard (the tray and
the timer widget both link to it).

Corrections apply live: correction writes flag app_state.corrections_changed
and the runner reloads its set for the very next decision. Category changes
still reach the classifier at the next monitor start.

CORS is deliberately an ALLOWLIST, never "*": this API serves your window titles
off localhost, so a wildcard would let any site you visit read them. Allowed
origins come from DASHBOARD_URL plus CFT_ALLOWED_ORIGINS (comma-separated).

`since` is a naive local timestamp string ("YYYY-MM-DD HH:MM:SS") matching how
the monitor stores focus_logs.timestamp, so a plain string comparison is correct.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Set
from urllib.parse import urlparse, parse_qs

from monitor.paths import resource_path, data_dir
from monitor.local_sink import CORRECTIONS_DAYS_BACK, _normalize_title
from monitor import supa_mirror
from monitor import intentions
from monitor import projects
from monitor import app_state
from monitor import breaks
from monitor import status as status_store
from monitor import problems


DEFAULT_API_PORT = 47113

# Set by start_local_server(): the bundled dashboard dir, or None if not bundled.
_STATIC_ROOT: Optional[str] = None

# When a dashboard page last read from this API. The page polls every few
# seconds, so a recent read means one is open somewhere — including a pinned
# or background tab, which no window-title search can find. monitor/browser.py
# uses this to avoid opening a second tab.
_last_client_seen: float = 0.0


def client_seen_within(seconds: float) -> bool:
    """True if a dashboard page read from this API in the last `seconds`."""
    return _last_client_seen > 0.0 and (time.time() - _last_client_seen) <= seconds


def _api_port() -> int:
    try:
        return int(os.getenv("CFT_API_PORT", "") or DEFAULT_API_PORT)
    except ValueError:
        return DEFAULT_API_PORT


def _origin_of(url: str) -> Optional[str]:
    try:
        p = urlparse(url.strip())
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
    except Exception:
        pass
    return None


def _allowed_origins() -> Set[str]:
    """Origins permitted to call this API cross-origin (e.g. the Vercel site)."""
    origins: Set[str] = set()
    dash = _origin_of(os.getenv("DASHBOARD_URL", ""))
    if dash:
        origins.add(dash)
    for raw in (os.getenv("CFT_ALLOWED_ORIGINS", "") or "").split(","):
        o = _origin_of(raw)
        if o:
            origins.add(o)
    return origins


def _db_ro():
    """Open the local focus DB read-only (returns None if it doesn't exist yet)."""
    path = data_dir() / "focus.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _db_rw():
    """Open the local focus DB read-write (returns None if it doesn't exist yet)."""
    path = data_dir() / "focus.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


# Categories that say nothing about what a session was FOR — the classifier's
# own shrugs and the OS. They never become a session's observed category and
# don't count toward its classifiable rows either.
_SESSION_CATEGORY_SKIP = ("ambiguous", "system", "[error]", "[parse_error]")


def _finished_sessions(conn, limit: int) -> list:
    """Finished sessions (completed or expired) newest first, each with its
    mean ALIGN and the category the monitor actually OBSERVED during it.

    llm_category is what the intention gate mapped the declared text to at
    session start — null on every session from before the gate existed.
    top_category is derived from the session's own focus_logs rows (a human
    correction wins over the classifier, as everywhere else), with top_share
    its fraction of the session's classifiable rows. So a pre-gate session
    still knows what it was, and a session declared as one thing but spent
    on another shows both.
    """
    out = None
    try:
        rows = conn.execute(
            "SELECT i.id, i.text, i.started_at, i.ended_at, "
            "       i.status, i.llm_category, i.project_id, "
            "       p.name AS project_name, "
            "       AVG(f.align_score) AS avg_align "
            "FROM intentions i "
            "LEFT JOIN projects p ON p.id = i.project_id "
            "LEFT JOIN focus_logs f ON f.intention_id = i.id "
            "WHERE i.status IN ('completed', 'expired') "
            "GROUP BY i.id ORDER BY i.started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = [dict(r) for r in rows]
    except sqlite3.OperationalError:
        # Sessions from before projects existed (no projects table, or no
        # project_id column yet): the same forest, with nothing filed.
        try:
            rows = conn.execute(
                "SELECT i.id, i.text, i.started_at, i.ended_at, "
                "       i.status, i.llm_category, "
                "       AVG(f.align_score) AS avg_align "
                "FROM intentions i "
                "LEFT JOIN focus_logs f ON f.intention_id = i.id "
                "WHERE i.status IN ('completed', 'expired') "
                "GROUP BY i.id ORDER BY i.started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except sqlite3.OperationalError:
            # No intentions table yet (no session was ever declared):
            # an empty forest, not an error.
            return []
        out = [dict(r) for r in rows]
        for s in out:
            s["project_id"] = None
            s["project_name"] = None
    for s in out:
        s["top_category"] = None
        s["top_share"] = None
    if not out:
        return out
    by_id = {s["id"]: s for s in out}
    marks = ",".join("?" * len(by_id))
    counts = conn.execute(
        "SELECT intention_id, "
        "       COALESCE(NULLIF(TRIM(human_label), ''), category_name) AS cat, "
        "       COUNT(*) AS n "
        f"FROM focus_logs WHERE intention_id IN ({marks}) "
        "GROUP BY intention_id, cat",
        list(by_id),
    ).fetchall()
    totals = {}
    best = {}
    for r in counts:
        cat = (r["cat"] or "").strip()
        if not cat or cat.lower() in _SESSION_CATEGORY_SKIP:
            continue
        iid = r["intention_id"]
        totals[iid] = totals.get(iid, 0) + r["n"]
        if iid not in best or r["n"] > best[iid][1]:
            best[iid] = (cat, r["n"])
    for iid, (cat, n) in best.items():
        by_id[iid]["top_category"] = cat
        by_id[iid]["top_share"] = round(n / totals[iid], 3)
    return out


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request stderr spam
        return

    # ---------- CORS ----------

    def _send_cors(self) -> None:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if origin and origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self) -> None:
        """Preflight for the hosted dashboard's cross-origin calls."""
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if not origin or origin not in _allowed_origins():
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        # Chrome Private Network Access: a public HTTPS page calling localhost
        # must get explicit permission here, or the request is blocked.
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    # ---------- routing ----------

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._handle_api()
            return
        if urlparse(self.path).path == "/intention":
            body = _INTENTION_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if _STATIC_ROOT is None:
            # API-only mode (dashboard is hosted): don't serve arbitrary files.
            self._send_problem(
                "This copy of the monitor doesn't include the dashboard.",
                "Open the hosted dashboard instead \u2014 it reads from this "
                "same machine while the monitor is running.", status=404)
            return
        super().do_GET()

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_problem(self, message, fix: str = "", status: int = 400) -> None:
        """Reject a request in two sentences: what went wrong, and what to do.

        `message` may be a string or a Problem/exception carrying its own fix
        (monitor/problems.py). The body is always {"error", "fix"} so the
        dashboard can set the fix in its own line; older clients that only
        read `error` still get a complete sentence.
        """
        if isinstance(message, str) and not isinstance(message, problems.Problem):
            body = problems.Problem(message, fix).as_json()
        else:
            body = problems.body(message, fix)
        self._send_json(body, status=status)

    def _send_unexpected(self, err: Exception, doing: str) -> None:
        """A 500 for something nobody predicted. The raw exception text never
        goes out on its own — it rides along inside the fix, framed as where
        to look, because a stack-trace fragment is not an instruction."""
        print(f"[api] {doing}: {err!r}")
        self._send_json(problems.unexpected(err, doing).as_json(), status=500)

    def _handle_api(self) -> None:
        global _last_client_seen
        _last_client_seen = time.time()  # a dashboard page is alive out there
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        # Intentions go through monitor/intentions.py (it creates the DB and
        # its own table on demand), not the read-only connection below.
        if u.path == "/api/intention":
            try:
                self._send_json({"active": intentions.get_active()})
            except Exception as e:
                self._send_unexpected(e, "loading your focus session")
            return

        if u.path == "/api/break":
            try:
                self._send_json({"active": breaks.get_active(),
                                 "default_min": breaks.DEFAULT_BREAK_MIN})
            except Exception as e:
                self._send_unexpected(e, "loading your break")
            return

        # Projects manage their own table too (monitor/projects.py).
        if u.path == "/api/projects":
            try:
                flag = (qs.get("all", [""])[0] or "").strip().lower()
                self._send_json(projects.list_all(
                    include_archived=flag not in ("", "0", "false", "no")))
            except Exception as e:
                self._send_unexpected(e, "loading your projects")
            return

        # Beeminder setup state, so the dashboard's setup card knows whether
        # to show itself. Never includes the token.
        if u.path == "/api/beeminder":
            try:
                from monitor import beeminder
                self._send_json(beeminder.status())
            except Exception as e:
                self._send_unexpected(e, "checking on Beeminder")
            return

        # Weekly email report: settings + health for its card, never the
        # SMTP password; and a preview of what would be mailed.
        if u.path == "/api/email-report":
            try:
                from monitor import email_report
                self._send_json(email_report.status())
            except Exception as e:
                self._send_unexpected(e, "checking on the weekly report")
            return
        if u.path == "/api/email-report/preview":
            try:
                from monitor import email_report
                from datetime import date as _date
                body = email_report.render_html(
                    email_report.build_report(_date.today())).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._send_cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_unexpected(e, "building the report preview")
            return

        # The resolved everything-at-once view the timer widget renders from.
        if u.path == "/api/status":
            try:
                self._send_json(status_store.full_snapshot())
            except Exception as e:
                self._send_unexpected(e, "loading the monitor's status")
            return

        conn = _db_ro()
        if conn is None:
            # No data yet — return empty so the dashboard renders zeros, not an error.
            self._send_json([])
            return
        try:
            if u.path == "/api/categories":
                rows = conn.execute(
                    "SELECT name, description, is_productive FROM categories"
                ).fetchall()
                data = [
                    {"name": r["name"],
                     "description": r["description"],
                     "is_productive": None if r["is_productive"] is None else bool(r["is_productive"])}
                    for r in rows
                ]
                self._send_json(data)
            elif u.path == "/api/focus":
                # category_name is the EFFECTIVE label: a human correction
                # wins over the classifier, the same rule Beeminder totals
                # by, so the pie, the split, the weekly trend and the
                # datapoint all count the same thing. ai_category keeps the
                # classifier's original call for anyone who wants it.
                # intention_id/align_score ride along for Beeminder's other
                # rule — a neutral category is productive only while it is
                # the declared work of a session — which the page applies
                # itself (focus-dashboard's useFocusData), so the pie can
                # show that time as a slice of its own.
                cols = ("SELECT timestamp, "
                        "COALESCE(NULLIF(TRIM(human_label), ''), category_name) AS category_name, "
                        "category_name AS ai_category, confidence, "
                        "intention_id, align_score FROM focus_logs ")
                since = (qs.get("since", [""])[0] or "").strip()
                if since:
                    rows = conn.execute(
                        cols + "WHERE timestamp >= ? ORDER BY timestamp DESC",
                        (since,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        cols + "ORDER BY timestamp DESC LIMIT 5000"
                    ).fetchall()
                self._send_json([dict(r) for r in rows])
            elif u.path == "/api/events":
                try:
                    limit = min(max(int(qs.get("limit", ["100"])[0]), 1), 1000)
                except ValueError:
                    limit = 100
                rows = conn.execute(
                    "SELECT id, timestamp, current_window, category_name, confidence, "
                    "human_label, reason, intention_id, align_score FROM focus_logs "
                    "WHERE current_window IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                self._send_json([dict(r) for r in rows])
            elif u.path == "/api/corrections":
                # One rule per exact window title, defined by its newest
                # corrected row, over the same recency window the decider's
                # load_corrections reads — older labeled rows are inert for
                # classification and must not be shown as live rules.
                # Confirmation rows (label agrees with the AI) are not rules.
                cutoff = (datetime.now()
                          - timedelta(days=CORRECTIONS_DAYS_BACK)).isoformat()
                rows = conn.execute(
                    "SELECT current_window, category_name, human_label, timestamp "
                    "FROM focus_logs "
                    "WHERE human_label IS NOT NULL AND human_label != '' "
                    "  AND current_window IS NOT NULL AND timestamp >= ? "
                    "ORDER BY timestamp DESC LIMIT 5000",
                    (cutoff,),
                ).fetchall()
                out = {}
                for r in rows:
                    window = (r["current_window"] or "").strip()
                    ai = (r["category_name"] or "").strip()
                    human = (r["human_label"] or "").strip()
                    if not window or not human or human.lower() == ai.lower():
                        continue
                    entry = out.get(window)
                    if entry is None:
                        out[window] = {
                            "window_title": window,
                            "ai_category": ai,
                            "corrected_category": human,
                            "rows": 1,
                            "last_at": r["timestamp"],
                        }
                    else:
                        entry["rows"] += 1
                self._send_json(list(out.values()))
            elif u.path == "/api/sessions":
                try:
                    limit = min(max(int(qs.get("limit", ["1000"])[0]), 1), 5000)
                except ValueError:
                    limit = 1000
                self._send_json(_finished_sessions(conn, limit))
            else:
                self._send_problem(
                    "The monitor doesn’t have anything at that address.",
                    "Refresh the dashboard — if it keeps happening, the "
                    "page is probably newer than the monitor.", status=404)
        except Exception as e:
            self._send_unexpected(e, "reading your activity history")
        finally:
            conn.close()

    def do_POST(self) -> None:
        u = urlparse(self.path)
        # Intention endpoints manage their own DB access (and create the DB
        # if needed), so they run before the shared read-write connection.
        no_conn_handlers = {
            "/api/intention": self._post_intention_start,
            "/api/intention/end": self._post_intention_end,
            "/api/intention/pause": self._post_intention_pause,
            "/api/break": self._post_break_start,
            "/api/break/end": self._post_break_end,
            "/api/intent-correct": self._post_intent_correct,
            "/api/project": self._post_project_add,
            "/api/project/update": self._post_project_update,
            "/api/project/delete": self._post_project_delete,
            "/api/beeminder": self._post_beeminder,
            "/api/beeminder/live": self._post_beeminder_live,
            "/api/beeminder/disconnect": self._post_beeminder_disconnect,
            "/api/email-report": self._post_email_report,
            "/api/email-report/send": self._post_email_report_send,
            "/api/email-report/disconnect": self._post_email_report_disconnect,
        }
        handlers = {
            "/api/correct": self._post_correct,
            "/api/corrections/update": self._post_corrections_update,
            "/api/category": self._post_category_add,
            "/api/category/update": self._post_category_update,
            "/api/category/delete": self._post_category_delete,
            "/api/sessions/category": self._post_session_category,
            "/api/sessions/project": self._post_session_project,
        }
        handler = no_conn_handlers.get(u.path) or handlers.get(u.path)
        if handler is None:
            self._send_problem(
                "The monitor doesn’t have anything at that address.",
                "Refresh the dashboard — if it keeps happening, the "
                "page is probably newer than the monitor.", status=404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            self._send_problem(
                "That request didn't arrive in a form the monitor understands.",
                "Refresh the dashboard and try again. If it keeps happening, "
                "the page and the monitor are probably different versions.")
            return

        if u.path in no_conn_handlers:
            try:
                handler(body)
            except Exception as e:
                self._send_unexpected(e, "saving that change")
            return

        conn = _db_rw()
        if conn is None:
            self._send_problem(
                "There's no activity history on this machine yet.",
                "Leave the monitor running for a few minutes and refresh \u2014 "
                "it starts recording as soon as you're at the screen.",
                status=503)
            return
        try:
            handler(conn, body)
        except Exception as e:
            self._send_unexpected(e, "saving that change")
        finally:
            conn.close()

    def _post_intention_start(self, body) -> None:
        from monitor import intent_gate

        try:
            text, duration_min = intentions.validate(
                body.get("text") or "", body.get("duration_min") or 0
            )
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return

        # The category is the user's call. A picked one is validated against
        # the real categories and refused only when it is marked unproductive;
        # the gate then merely CHECKS it, and says so once (409) when it
        # disagrees — unless the caller already confirmed. Nothing here needs
        # the LLM, so a dead key can't leave a session uncategorized.
        # A session may be filed under a project. The project's home
        # category, if it has one, stands in for a pick the user didn't make
        # \u2014 and goes through exactly the checks a hand pick does below.
        project = None
        raw_pid = body.get("project_id")
        if raw_pid not in (None, "", 0, "0"):
            project = projects.get(raw_pid)
            if project is None:
                self._send_problem(
                    "That project isn't there any more.",
                    "Refresh the page and pick again \u2014 it may have been "
                    "deleted since the page loaded \u2014 or start the session "
                    "without a project.", status=404)
                return
            if project["status"] != projects.STATUS_ACTIVE:
                self._send_problem(
                    f"\u201c{project['name']}\u201d is archived, so new sessions "
                    f"can't be filed under it.",
                    "Bring it back on the Projects tab first, or start the "
                    "session without a project.", status=409)
                return

        picked = (body.get("category") or "").strip() or None
        from_project = False
        if picked is None and project is not None and project.get("category"):
            picked = project["category"]
            from_project = True
        confirmed = bool(body.get("confirm"))
        chosen = None
        if picked:
            row = None
            conn = _db_ro()
            if conn is not None:
                try:
                    row = conn.execute(
                        "SELECT name, is_productive FROM categories "
                        "WHERE lower(name) = lower(?)", (picked,)
                    ).fetchone()
                finally:
                    conn.close()
            if from_project and (row is None or row["is_productive"] == 0):
                # The project's category was deleted or marked unproductive
                # since it was set. A standing pick that no longer holds is
                # not the user's fault right now: the gate decides instead.
                print(f"[gate] project {project['name']!r} names "
                      f"{picked!r}, which no longer qualifies; letting the gate decide")
                row = None
                picked = None
            elif row is None:
                self._send_json(problems.unknown_category(picked, action="start").as_json(),
                                status=400)
                return
            elif row["is_productive"] == 0:
                self._send_json({
                    "error": (f"\u201c{row['name']}\u201d isn't something you "
                              f"can declare a focus session for."),
                    "fix": (f"It's marked as time that doesn't count. Pick the "
                            f"work you actually mean to do \u2014 or, if "
                            f"{row['name']} really is work for you, change that "
                            f"on the Categories tab."),
                    "rejected": True,
                    "category": row["name"],
                    "reason": "picked category is marked unproductive",
                }, status=422)
                return
            if row is not None:
                chosen = row["name"]

        # The intention itself must be work: "scroll twitter for an hour"
        # would make every drift check score scrolling as on-intent.
        gate = intent_gate.check(text)
        if chosen is None:
            if not gate["allowed"]:
                why = gate.get("reason") or "it describes off-task activity"
                self._send_json({
                    "error": (f"\u201c{text}\u201d reads as "
                              f"{gate['category']}, which isn't work the "
                              f"session can measure you against."),
                    "fix": (f"{why} If it really is work, say what you'll "
                            f"produce \u2014 \u201cread the Chapter 4 "
                            f"problems\u201d rather than \u201clook at "
                            f"stuff\u201d."),
                    "rejected": True,
                    "category": gate.get("category"),
                    "reason": gate.get("reason"),
                }, status=422)
                return
            chosen = gate.get("category")
        elif (not confirmed and gate.get("category")
              and gate["category"].lower() != chosen.lower()):
            # Two productive categories disagreeing is a close call between
            # two labels the user is allowed to pick, so the pick stands and
            # the disagreement is only logged. The gate asks once (409) only
            # when it reads the text as a LOOPHOLE — work that maps to an
            # unproductive category — because that is the case it exists for.
            if gate["allowed"]:
                print(f"[gate] pick {chosen!r} kept over {gate['category']!r} "
                      f"for {text!r}: {gate.get('reason') or ''}")
            else:
                self._send_json({
                    "error": (f"That reads more like {gate['category']} than "
                              f"{chosen}."),
                    "fix": ("Start it anyway if you're sure \u2014 you know "
                            "what you're working on better than the gate "
                            "does."),
                    "needs_confirm": True,
                    "picked": chosen,
                    "suggested": gate["category"],
                    "reason": gate.get("reason"),
                }, status=409)
                return

        try:
            active = intentions.start(
                text, duration_min, llm_category=chosen,
                project_id=project["id"] if project is not None else None)
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, "active": active})

    # ---------- projects ----------

    def _resolve_project_category(self, label):
        """A project's home category is a standing pick, so it gets the same
        checks a session's pick does: it must exist and must not be marked
        unproductive. Returns (exact name or None, (error json, status) or None)."""
        label = (label or "").strip()
        if not label:
            return None, None
        row = None
        conn = _db_ro()
        if conn is not None:
            try:
                row = conn.execute(
                    "SELECT name, is_productive FROM categories "
                    "WHERE lower(name) = lower(?)", (label,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None, (problems.unknown_category(label, action="use").as_json(), 400)
        if row["is_productive"] == 0:
            return None, ({
                "error": (f"“{row['name']}” can't be a project's home "
                          f"category."),
                "fix": ("It's marked as time that doesn't count, and a "
                        "project's sessions are meant to. Pick the category "
                        "the project's work belongs in — or, if it really "
                        "is work for you, change that on the Categories tab."),
                "rejected": True,
                "category": row["name"],
            }, 422)
        return row["name"], None

    def _post_project_add(self, body) -> None:
        category, err = self._resolve_project_category(body.get("category"))
        if err is not None:
            self._send_json(*err)
            return
        try:
            row = projects.create(body.get("name"), body.get("description"), category)
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, "project": row})

    def _post_project_update(self, body) -> None:
        """Only the fields present in the body change, so the dashboard can
        archive a project without resending its description."""
        kw = {}
        if "name" in body:
            kw["name"] = body.get("name")
        if "description" in body:
            kw["description"] = body.get("description")
        if "category" in body:
            category, err = self._resolve_project_category(body.get("category"))
            if err is not None:
                self._send_json(*err)
                return
            kw["category"] = category
        if "archived" in body:
            kw["archived"] = bool(body.get("archived"))
        try:
            row = projects.update(body.get("id"), **kw)
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, "project": row})

    def _post_project_delete(self, body) -> None:
        try:
            row = projects.delete(body.get("id"))
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, "deleted": row["name"],
                         "unlinked_sessions": row["unlinked_sessions"]})

    def _post_session_project(self, conn, body) -> None:
        """File a finished session under a project, or take it out of one —
        the forest's relabel card, for the project half."""
        try:
            sid = int(body.get("id"))
        except (TypeError, ValueError):
            self._send_problem(
                "The dashboard didn't send everything needed for filing a session under a project.",
                "Refresh the page and try again — this usually means the "
                "page and the monitor are different versions.")
            return
        raw = body.get("project_id")
        pid = None
        if raw not in (None, ""):
            target = projects.get(raw)
            if target is None:
                self._send_problem(
                    "That project isn't there any more.",
                    "Refresh the Forest tab and pick again — it may have "
                    "been deleted since the page loaded.", status=404)
                return
            pid = target["id"]
        row = intentions.set_project(sid, pid)
        if row is None:
            self._send_problem(
                "That focus session isn't there any more.",
                "Refresh the Forest tab — it may have been removed since "
                "the page loaded, and a running session is filed when it "
                "starts, not from here.", status=404)
            return
        self._send_json({"ok": True, "session": row})

    def _post_session_category(self, conn, body) -> None:
        """Relabel a finished session — the honest fix for the sessions the
        gate never saw (or got wrong), instead of guessing from what the
        monitor happened to log."""
        try:
            sid = int(body.get("id"))
        except (TypeError, ValueError):
            self._send_problem(
                "The dashboard didn't send everything needed for changing a session's category.",
                "Refresh the page and try again \u2014 this usually means the "
                "page and the monitor are different versions.")
            return
        label = (body.get("category") or "").strip() or None
        known = None
        if label is not None:
            known = self._resolve_label(conn, label)
            if known is None:
                self._send_json(problems.unknown_category(label, action="use").as_json(),
                                status=400)
                return
        row = intentions.set_category(sid, known)
        if row is None:
            self._send_problem(
                "That focus session isn't there any more.",
                "Refresh the Forest tab \u2014 it may have been relabelled or "
                "removed since the page loaded.", status=404)
            return
        self._send_json({"ok": True, "session": row})

    def _post_intent_correct(self, body) -> None:
        """Rule a window on/off-intent for the active session's intention.

        The same ruling the alert's button records, reachable from the
        dashboard for windows you noticed after the fact.
        """
        from monitor import intent_corrections

        window = (body.get("window") or "").strip()
        text = (body.get("intention") or "").strip()
        if not text:
            active = intentions.get_active()
            if active is None:
                self._send_problem(
                    "There's no focus session running, so there's nothing to "
                    "judge that window against.",
                    "Start a session first \u2014 alignment feedback only means "
                    "something once you've declared what you're working on.")
                return
            text = active["text"]
        if not window:
            self._send_problem(
                "The dashboard didn't send everything needed for recording that feedback.",
                "Refresh the page and try again \u2014 this usually means the "
                "page and the monitor are different versions.")
            return

        if body.get("aligned") is None:
            removed = intent_corrections.clear(text, window)
            self._send_json({"ok": True, "cleared": removed, "window": window})
            return
        try:
            saved = intent_corrections.save(text, window, bool(body.get("aligned")))
        except ValueError as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, **saved})

    def _post_beeminder(self, body) -> None:
        """One-time Beeminder setup from the dashboard card.

        Two shapes, chosen by `create_goal`:
          false (default) — connect an existing goal; verified against
              Beeminder before anything is saved.
          true — create a fresh Do More goal (units: hours) on the user's own
              account at `hours_per_day`, then connect it. Creation succeeding
              is the verification.

        Either way a bad token or slug comes straight back as a 400 with the
        reason, and nothing half-configured lands in the .env. On success the
        values persist next to the exe (same file the LLM key setup writes)
        and the sync starts immediately, no restart needed.
        """
        from monitor import beeminder
        from monitor.creds import _persist_env_var

        user = str(body.get("user") or "").strip().lstrip("@")
        token = str(body.get("token") or "").strip()
        focus_goal = str(body.get("focus_goal") or "").strip()
        sessions_goal = str(body.get("sessions_goal") or "").strip()
        # Replacing a goal that vanished: the stored token still works, so
        # the card need not ask for it again. A new token, if given, wins.
        if not token and beeminder.enabled():
            token = beeminder.config().token
        if not user or not token or not focus_goal:
            self._send_problem(
                "Beeminder needs your username, your token and a goal name.",
                "Your token is on beeminder.com/api/v1/auth_token.json while "
                "you're signed in \u2014 it's a long string of letters and "
                "numbers, not your password.")
            return

        if body.get("create_goal"):
            problem = beeminder.create_goal(user, token, focus_goal,
                                            body.get("hours_per_day"),
                                            body.get("leeway_days") or 0)
            if problem:
                self._send_json(problems.body(problem), status=400)
                return
            if sessions_goal:  # the card never sends this, but keep the API honest
                problem = beeminder.verify(user, token, sessions_goal)
                if problem:
                    self._send_json(problems.body(problem), status=400)
                    return
        else:
            for goal in (focus_goal, sessions_goal):
                if not goal:
                    continue
                problem = beeminder.verify(user, token, goal)
                if problem:
                    self._send_json(problems.body(problem), status=400)
                    return

        saves = {
            "BEEMINDER_USER": user,
            "BEEMINDER_TOKEN": token,
            "BEEMINDER_FOCUS_GOAL": focus_goal,
        }
        if sessions_goal:
            saves["BEEMINDER_SESSIONS_GOAL"] = sessions_goal
        for name, value in saves.items():
            os.environ[name] = value
            _persist_env_var(name, value)

        # Past outcomes described the old goal; push once right away so the
        # card can show this one working (or not) without waiting a tick.
        beeminder.reset_health()
        beeminder.start_background_sync()
        beeminder.sync_now().join(timeout=8)
        self._send_json({"ok": True, **beeminder.status()})

    def _post_beeminder_live(self, body) -> None:
        """Leave dry-run mode: from now on datapoints are really sent."""
        from monitor import beeminder
        from monitor.creds import _persist_env_var

        os.environ["BEEMINDER_DRY_RUN"] = "0"
        _persist_env_var("BEEMINDER_DRY_RUN", "0")
        beeminder.reset_health()
        beeminder.start_background_sync()
        # Wait for the first real push (bounded) so the answer already says
        # whether it landed — the card renders straight from this.
        beeminder.sync_now().join(timeout=8)
        self._send_json({"ok": True, **beeminder.status()})

    def _post_beeminder_disconnect(self, body) -> None:
        """Forget the credentials. Blanks the values rather than deleting the
        lines, so the .env still shows which knobs exist. The sync thread
        stays up but every tick sees the integration disabled and does
        nothing; the setup card reappears."""
        from monitor import beeminder
        from monitor.creds import _persist_env_var

        for name in ("BEEMINDER_USER", "BEEMINDER_TOKEN",
                     "BEEMINDER_FOCUS_GOAL", "BEEMINDER_SESSIONS_GOAL"):
            os.environ.pop(name, None)
            _persist_env_var(name, "")
        beeminder.reset_health()
        self._send_json({"ok": True, **beeminder.status()})

    # The .env names the weekly email report writes. Blanked, not deleted, on
    # disconnect — same convention as Beeminder.
    _REPORT_ENV = ("REPORT_EMAIL_TO", "REPORT_SMTP_HOST", "REPORT_SMTP_PORT",
                   "REPORT_SMTP_USER", "REPORT_SMTP_PASSWORD", "REPORT_EMAIL_FROM",
                   "REPORT_SMTP_SECURITY", "REPORT_DAY", "REPORT_HOUR")

    def _post_email_report(self, body) -> None:
        """Set up (or edit) the weekly email report from the dashboard card.

        Recipients, the SMTP server and the schedule arrive together; the
        password may be left out once one is stored, so editing the
        recipient list never asks for it again. The server is logged into
        BEFORE anything is saved — a wrong app password comes back as a 400
        in plain words and the .env is untouched. On success the settings
        persist next to the exe and the scheduler starts at once; with
        send_test the last seven days are mailed right away, so the card can
        show whether it landed.
        """
        from monitor import email_report
        from monitor.creds import _persist_env_var

        try:
            to = email_report.parse_recipients(body.get("to"))
            day = email_report.parse_day(body.get("day", "mon"))
            hour = email_report.parse_hour(body.get("hour", 9))
            # Checked here, not after the SMTP login: the login never looks at
            # the From address, so an unchecked one would save cleanly and
            # then fail at the first send.
            given_sender = email_report.parse_sender(body.get("sender"))
        except ValueError as e:
            self._send_json(problems.body(e), status=400)
            return
        if not to:
            self._send_problem(
                "Nobody is set to receive the report.",
                "Add at least one email address \u2014 your own, or whoever "
                "should be reading it. Separate several with commas.")
            return
        host = str(body.get("smtp_host") or "").strip()
        if not host:
            self._send_problem(
                "The report needs to know which mail server to send through.",
                "Pick your provider with the buttons above and it fills "
                "itself in \u2014 Gmail is smtp.gmail.com.")
            return
        try:
            port = int(str(body.get("smtp_port") or "587").strip())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self._send_problem(
                f"\u201c{body.get('smtp_port')}\u201d isn't a port number.",
                "Use 587 for STARTTLS or 465 for SSL/TLS \u2014 between them "
                "those cover almost every provider.")
            return
        user = str(body.get("smtp_user") or "").strip()
        password = email_report.parse_app_password(body.get("smtp_password"))
        current = email_report.config()
        if not password and user and user == current.smtp_user:
            password = current.smtp_password   # editing: keep the stored one
        sender = given_sender
        if not sender and user:
            # Falling back to the login only works if the login is an address,
            # which it usually is but does not have to be.
            try:
                sender = email_report.parse_sender(user, field="Mail server username")
            except ValueError as e:
                self._send_json(problems.body(e), status=400)
                return
        if not sender:
            self._send_problem(
                "The report has no address to send from.",
                "Fill in \u201cSend from\u201d with the address the mail "
                "should come from \u2014 usually the same one you log in to "
                "the mail server with.")
            return
        security = str(body.get("security") or "").strip().lower()
        if security not in ("starttls", "ssl", "none"):
            security = "ssl" if port == 465 else "starttls"

        cfg = email_report.Config(
            to=tuple(to), smtp_host=host, smtp_port=port, smtp_user=user,
            smtp_password=password, sender=sender, security=security,
            day=day, hour=hour, timeout_sec=current.timeout_sec,
            seconds_per_row=current.seconds_per_row,
            strict_intent=current.strict_intent,
        )
        problem = email_report.verify_smtp(cfg)
        if problem:
            self._send_json(problems.body(problem), status=400)
            return

        saves = {
            "REPORT_EMAIL_TO": ", ".join(to),
            "REPORT_SMTP_HOST": host,
            "REPORT_SMTP_PORT": str(port),
            "REPORT_SMTP_USER": user,
            "REPORT_SMTP_PASSWORD": password,
            "REPORT_EMAIL_FROM": sender,
            "REPORT_SMTP_SECURITY": security,
            "REPORT_DAY": email_report.DAYS[day],
            "REPORT_HOUR": str(hour),
        }
        for name, value in saves.items():
            os.environ[name] = value
            _persist_env_var(name, value)

        email_report.reset_health()
        email_report.start_background_scheduler()

        # Anyone added here has never heard of this app. They get a short note
        # saying who set it up and that a reply stops it, before any of their
        # data - well, somebody else's data - arrives. Introducing on ADD
        # rather than at the first report means a stranger's first contact is
        # a sentence from a person, not a page of statistics.
        fresh = email_report.newcomers(cfg)
        intro_problem = email_report.send_introductions(cfg) if fresh else None

        if body.get("send_test"):
            # Bounded wait so the answer already says whether it landed.
            email_report.send_now().join(timeout=cfg.timeout_sec + 5)
        answer = {"ok": True, **email_report.status(),
                  "introduced": [] if intro_problem else fresh}
        if intro_problem:
            # The settings are saved and the weekly report will still go; only
            # the courtesy note failed, so this is a note, not a rejection.
            answer["intro_problem"] = problems.problem_for_status(intro_problem)
        self._send_json(answer)

    def _post_email_report_send(self, body) -> None:
        """Mail the last seven days to the configured recipients now."""
        from monitor import email_report

        if not email_report.enabled():
            self._send_problem(
                "The weekly report isn't set up yet, so there's nothing to send.",
                "Add recipients and a mail server first, then try again.")
            return
        email_report.send_now().join(timeout=email_report.config().timeout_sec + 5)
        self._send_json({"ok": True, **email_report.status()})

    def _post_email_report_disconnect(self, body) -> None:
        """Forget recipients and server. The scheduler thread sees the
        integration disabled at its next tick and does nothing.

        The sent-marker goes too, so switching the report back on months
        later baselines from that moment instead of catching up on a slot
        from before the disconnect and mailing a stale week.
        """
        from monitor import email_report
        from monitor.creds import _persist_env_var

        for name in self._REPORT_ENV:
            os.environ.pop(name, None)
            _persist_env_var(name, "")
        email_report.clear_marker()
        email_report.forget_introductions()
        email_report.reset_health()
        self._send_json({"ok": True, **email_report.status()})

    def _post_intention_pause(self, body) -> None:
        """Freeze the active session for a minute — no time lost, nothing watched."""
        try:
            result = intentions.pause_active()
        except ValueError as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, **result})

    def _post_intention_end(self, body) -> None:
        status = (body.get("status") or "completed").strip().lower()
        try:
            ended = intentions.end_active(status)
        except ValueError as e:
            self._send_json(problems.body(e), status=400)
            return
        # Completing a session starts the break; report it so the caller can
        # render the new countdown without a second round trip.
        self._send_json({"ok": True, "ended": ended, "break": breaks.get_active()})

    def _post_break_start(self, body) -> None:
        raw = body.get("duration_min")
        try:
            duration = breaks.DEFAULT_BREAK_MIN if raw in (None, "") else int(raw)
        except (ValueError, TypeError):
            self._send_problem(
                f"\u201c{raw}\u201d isn't a number of minutes.",
                f"Pick one of the preset lengths, or type a whole number "
                f"between {breaks.MIN_BREAK_MIN} and {breaks.MAX_BREAK_MIN}.")
            return
        try:
            active = breaks.start(duration)
        except (ValueError, TypeError) as e:
            self._send_json(problems.body(e), status=400)
            return
        self._send_json({"ok": True, "active": active})

    def _post_break_end(self, body) -> None:
        self._send_json({"ok": True, "ended": breaks.end()})

    def _post_correct(self, conn, body) -> None:
        try:
            row_id = int(body["id"])
        except (KeyError, ValueError, TypeError):
            self._send_problem(
                "The dashboard didn't send everything needed for correcting a row.",
                "Refresh the page and try again \u2014 this usually means the "
                "page and the monitor are different versions.")
            return

        label = (body.get("human_label") or "").strip() or None
        if label is not None:
            known = self._resolve_label(conn, label)
            if known is None:
                self._send_json(problems.unknown_category(label, action="use").as_json(),
                                status=400)
                return
            label = known  # exact casing, so the downstream joins match
        reviewed_at = datetime.now().isoformat() if label else None
        cur = conn.execute(
            "UPDATE focus_logs SET human_label = ?, reviewed_at = ? WHERE id = ?",
            (label, reviewed_at, row_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            self._send_problem(
                "That entry isn't in your history any more.",
                "Refresh the Review tab \u2014 it may have been corrected or "
                "cleaned up since the page loaded.", status=404)
            return

        row = conn.execute(
            "SELECT timestamp, current_window FROM focus_logs WHERE id = ?", (row_id,)
        ).fetchone()
        if row is not None:
            supa_mirror.mirror_correction(row["timestamp"], row["current_window"], label)
        app_state.corrections_changed.set()
        self._send_json({"ok": True, "id": row_id, "human_label": label})

    def _post_corrections_update(self, conn, body) -> None:
        """Manage a learned correction rule by window title.

        {window_title, human_label} retargets the rule to the new category;
        null/"" removes the correction instead. Titles are matched after
        _normalize_title — the same matching the decider applies — so every
        spelling variant ("(2) Chat | Teams") of the rule is covered, not
        just the one byte-exact spelling the list happened to display.
        Only rows whose label DIFFERS from the AI's category are touched —
        a confirmation (label agreeing with the AI) is the user's ground
        truth for the learning loop, not part of the rule being managed.

        Retargeting only rewrites rows carrying the rule's own label: rows
        of the same title corrected to some OTHER category are their own
        ground truth, not part of the rule being edited. And a retarget that
        lands on a row's AI category clears that row's label instead of
        writing it — label==category reads as a human CONFIRMATION to the
        learning loop, and no human reviewed those rows.
        """
        window = (body.get("window_title") or "").strip()
        if not window:
            self._send_problem(
                "The dashboard didn't send everything needed for updating that correction rule.",
                "Refresh the page and try again \u2014 this usually means the "
                "page and the monitor are different versions.")
            return

        label = (body.get("human_label") or "").strip() or None
        if label is not None:
            known = self._resolve_label(conn, label)
            if known is None:
                self._send_json(problems.unknown_category(label, action="use").as_json(),
                                status=400)
                return
            label = known  # exact casing from the categories table

        target_norm = _normalize_title(window)
        rows = conn.execute(
            "SELECT id, timestamp, current_window, category_name, human_label "
            "FROM focus_logs "
            "WHERE human_label IS NOT NULL AND human_label != '' "
            "  AND current_window IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 10000"
        ).fetchall()
        matched = [
            r for r in rows
            if _normalize_title(r["current_window"]) == target_norm
            and (r["human_label"] or "").strip().lower()
                != (r["category_name"] or "").strip().lower()
        ]
        if not matched:
            self._send_problem(
                "There's nothing corrected for that window any more.",
                "Refresh the Review tab \u2014 the rule may already have been "
                "changed or removed.", status=404)
            return

        # The rule the user edited: prefer the newest row with the exact
        # spelling they clicked (what the list displayed), falling back to
        # the newest normalized match.
        rule_label = next(
            ((r["human_label"] or "").strip() for r in matched
             if (r["current_window"] or "").strip() == window),
            (matched[0]["human_label"] or "").strip(),
        )
        now_iso = datetime.now().isoformat()
        changes = []  # (id, timestamp, current_window, new_label)
        for r in matched:
            if label is not None:
                if (r["human_label"] or "").strip().lower() != rule_label.lower():
                    continue
                ai = (r["category_name"] or "").strip()
                new_label = None if ai.lower() == label.lower() else label
            else:
                new_label = None
            changes.append((r["id"], r["timestamp"], r["current_window"], new_label))

        for row_id, _ts, _win, new_label in changes:
            conn.execute(
                "UPDATE focus_logs SET human_label = ?, reviewed_at = ? WHERE id = ?",
                (new_label, now_iso if new_label else None, row_id),
            )
        conn.commit()

        supa_mirror.mirror_correction_rows(
            [(ts, win, new_label) for _id, ts, win, new_label in changes])
        app_state.corrections_changed.set()
        self._send_json({"ok": True, "window_title": window,
                         "human_label": label, "rows": len(changes)})

    @staticmethod
    def _productive_flag(value):
        """JSON true/false/null -> SQLite 1/0/NULL (neutral)."""
        return None if value is None else int(bool(value))

    @staticmethod
    def _resolve_label(conn, label):
        """The exact-cased category name for `label`, or None if unknown.

        Everything downstream joins on the exact name — Beeminder's day
        query, the learning loop, the corrections list — so a free-text or
        miscased label would silently fall out of all of them (a NULL join:
        neither productive nor off-task). Labels are validated here instead.
        """
        row = conn.execute(
            "SELECT name FROM categories WHERE lower(name) = lower(?)", (label,)
        ).fetchone()
        return row["name"] if row else None

    def _post_category_add(self, conn, body) -> None:
        name = (body.get("name") or "").strip()
        if not name or len(name) > 60:
            self._send_problem(
                "A category needs a name between 1 and 60 characters.",
                "Something short and recognisable works best \u2014 it's what "
                "you'll be picking from menus and reading in the report.")
            return
        description = (body.get("description") or "").strip()
        exists = conn.execute(
            "SELECT 1 FROM categories WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if exists:
            self._send_problem(
                f"You already have a category called \u201c{name}\u201d.",
                "Pick a different name, or edit the existing one on the "
                "Categories tab.", status=409)
            return
        productive = self._productive_flag(body.get("is_productive"))
        conn.execute(
            "INSERT INTO categories (name, description, is_productive) VALUES (?,?,?)",
            (name, description, productive),
        )
        conn.commit()
        supa_mirror.mirror_category_add(name, description, None if productive is None else bool(productive))
        self._send_json({"ok": True, "name": name})

    # The classifier's mandated fallback: prompt-main.txt instructs the LLM to
    # default here, and decider tags system windows "System". Deleting either
    # would leave the monitor unable to name its own output.
    _UNDELETABLE = {"ambiguous", "system"}

    def _post_category_delete(self, conn, body) -> None:
        """Delete a category, moving any logged rows to `reassign_to`.

        Rows keep their history rather than being deleted: the default target
        is "Ambiguous" (the catch-all), and human_label follows too, exactly
        like a rename cascade. Reassignment is a one-way valve, though: rows
        may move sideways or down, never up into a productive category (see
        the guard below).
        """
        name = (body.get("name") or "").strip()
        row = conn.execute(
            "SELECT name, is_productive FROM categories WHERE name = ? COLLATE NOCASE",
            (name,),
        ).fetchone()
        if row is None:
            self._send_problem(
                f"There's no category called \u201c{name}\u201d.",
                "Refresh the Categories tab \u2014 it may have been renamed or "
                "deleted since the page loaded.", status=404)
            return
        name = row["name"]  # exact casing

        if name.lower() in self._UNDELETABLE:
            self._send_json(
                {"error": f"{name!r} is required by the classifier and cannot be deleted"},
                status=400,
            )
            return

        target = (body.get("reassign_to") or "Ambiguous").strip()
        target_row = conn.execute(
            "SELECT name, is_productive FROM categories WHERE name = ? COLLATE NOCASE",
            (target,),
        ).fetchone()
        if target_row is None:
            self._send_problem(
                f"There's no category called \u201c{target}\u201d to move that "
                f"history into.",
                "Pick an existing category to absorb the hours \u2014 Ambiguous "
                "is the usual choice when nothing else fits.")
            return
        target = target_row["name"]
        if target.lower() == name.lower():
            self._send_problem(
                "A category can't absorb its own history.",
                "Pick a different category for the old entries to move into.")
            return

        # The one-way valve: "delete Gaming, reassign to Deep Work" would
        # retroactively count every logged Gaming hour as focused work — and
        # Beeminder re-pushes past days in place, so history would quietly
        # pay out. Only the credit-minting direction is closed; tightening
        # (productive -> anything) stays unrestricted.
        if target_row["is_productive"] == 1 and row["is_productive"] != 1:
            self._send_json(
                {"error": f"cannot reassign {name!r} rows to productive {target!r} — "
                          f"its logged history would start counting as focused time. "
                          f"Pick a non-productive target (default: Ambiguous)."},
                status=400,
            )
            return

        moved = conn.execute(
            "UPDATE focus_logs SET category_name = ? WHERE category_name = ?", (target, name)
        ).rowcount
        moved += conn.execute(
            "UPDATE focus_logs SET human_label = ? WHERE human_label = ?", (target, name)
        ).rowcount
        # A project whose home category this was is left without one: its
        # next session goes through the gate like an unfiled one would.
        dropped_projects = projects.cascade_category(conn, name, None)
        conn.execute("DELETE FROM categories WHERE name = ?", (name,))
        conn.commit()

        supa_mirror.mirror_category_delete(name, target)
        projects.mirror_rows(dropped_projects)
        self._send_json({"ok": True, "deleted": name, "reassigned_to": target,
                         "moved_log_rows": moved})

    def _post_category_update(self, conn, body) -> None:
        name = (body.get("name") or "").strip()
        row = conn.execute(
            "SELECT name, description, is_productive FROM categories WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            self._send_problem(
                f"There's no category called \u201c{name}\u201d.",
                "Refresh the Categories tab \u2014 it may have been renamed or "
                "deleted since the page loaded.", status=404)
            return

        new_name = (body.get("new_name") or "").strip() or name
        if len(new_name) > 60:
            self._send_problem(
                "A category needs a name between 1 and 60 characters.",
                "Something short and recognisable works best \u2014 it's what "
                "you'll be picking from menus and reading in the report.")
            return
        if new_name.lower() != name.lower():
            clash = conn.execute(
                "SELECT 1 FROM categories WHERE name = ? COLLATE NOCASE", (new_name,)
            ).fetchone()
            if clash:
                self._send_problem(
                    f"You already have a category called \u201c{new_name}\u201d.",
                    "Pick a different name \u2014 two categories with the same "
                    "name would split your hours between them.", status=409)
                return

        description = body["description"].strip() if "description" in body else row["description"]
        productive = (
            self._productive_flag(body["is_productive"])
            if "is_productive" in body
            else row["is_productive"]
        )

        conn.execute(
            "UPDATE categories SET name = ?, description = ?, is_productive = ? WHERE name = ?",
            (new_name, description, productive, name),
        )
        renamed_logs = 0
        if new_name != name:
            # Keep history consistent: move logged events to the new name too.
            renamed_logs += conn.execute(
                "UPDATE focus_logs SET category_name = ? WHERE category_name = ?",
                (new_name, name),
            ).rowcount
            renamed_logs += conn.execute(
                "UPDATE focus_logs SET human_label = ? WHERE human_label = ?",
                (new_name, name),
            ).rowcount
        renamed_projects = (
            projects.cascade_category(conn, name, new_name) if new_name != name else []
        )
        conn.commit()
        supa_mirror.mirror_category_update(
            name, new_name, description, None if productive is None else bool(productive)
        )
        projects.mirror_rows(renamed_projects)
        self._send_json({"ok": True, "name": new_name, "renamed_log_rows": renamed_logs})


# Built-in page for session intentions and breaks. Same-origin, no build step —
# this is the tray's "Start Focus Session" target and the timer widget's click
# target, and it works even when the React dashboard isn't bundled. It renders
# from /api/status, so its countdown and colors match the floating widget's.
_INTENTION_PAGE_SRC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Focus</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0f19; --card: #111827; --line: #1e293b;
    --ink: #f1f5f9; --muted: #94a3b8; --faint: #64748b;
    --accent: #94a3b8; --accent-soft: #94a3b81f;
  }
  [data-state="on"], [data-state="session"] { --accent: #10b981; --accent-soft: #10b98122; }
  [data-state="partial"] { --accent: #f59e0b; --accent-soft: #f59e0b22; }
  [data-state="off"]     { --accent: #ef4444; --accent-soft: #ef444422; }
  [data-state="break"], [data-state="paused"] { --accent: #38bdf8; --accent-soft: #38bdf822; }
  [data-state="away"]    { --accent: #64748b; --accent-soft: #64748b22; }

  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink);
         font-family: "Segoe UI", system-ui, sans-serif;
         display: flex; align-items: center; justify-content: center; padding: 24px; }
  main { width: 100%; max-width: 460px; }

  .card { background: var(--card); border: 1px solid var(--line); border-radius: 20px;
          padding: 28px; box-shadow: 0 18px 40px #0006; }
  .chip { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
          border-radius: 999px; background: var(--accent-soft); color: var(--accent);
          font-size: .74rem; font-weight: 600; letter-spacing: .06em;
          text-transform: uppercase; }
  .chip i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

  .dial { display: flex; align-items: center; gap: 20px; margin: 22px 0 6px; }
  .dial svg { flex: none; transform: rotate(-90deg); }
  .ring-track { stroke: var(--line); }
  .ring-fill { stroke: var(--accent); transition: stroke-dashoffset 1s linear; }
  .time { font-size: 2.5rem; font-weight: 700; line-height: 1;
          font-variant-numeric: tabular-nums; }
  .word { font-size: 1.75rem; font-weight: 700; line-height: 1.1; }
  .sub { color: var(--muted); font-size: .85rem; margin-top: 7px; word-break: break-word; }
  .note { color: var(--faint); font-size: .8rem; margin-top: 16px; line-height: 1.5; }

  label { display: block; font-size: .78rem; font-weight: 600; color: var(--muted);
          text-transform: uppercase; letter-spacing: .06em; margin-bottom: 9px; }
  textarea { width: 100%; min-height: 74px; resize: vertical; font: inherit;
             padding: 11px 13px; border-radius: 12px; color: var(--ink);
             background: #0b111f; border: 1px solid var(--line); }
  textarea:focus, input:focus, select:focus { outline: none; border-color: #10b981; }
  select { width: 100%; font: inherit; padding: 10px 13px; border-radius: 12px;
           color: var(--ink); background: #0b111f; border: 1px solid var(--line);
           margin-top: 14px; }
  .chips { display: flex; gap: 7px; flex-wrap: wrap; margin: 16px 0 0; }
  .chips button { padding: 7px 14px; border-radius: 999px; font: inherit;
                  font-size: .88rem; cursor: pointer; color: var(--muted);
                  background: transparent; border: 1px solid var(--line); }
  .chips button:hover { color: var(--ink); }
  .chips button.sel { background: #10b981; border-color: #10b981; color: #04150e;
                      font-weight: 600; }
  .customrow { display: none; align-items: center; gap: 8px; margin-top: 12px; }
  .customrow.show { display: flex; }
  #custom { width: 6.5em; padding: 8px 11px; border-radius: 10px; font: inherit;
            background: #0b111f; border: 1px solid var(--line); color: var(--ink); }
  #custom.bad { border-color: #ef4444; }

  .row { display: flex; gap: 10px; margin-top: 20px; }
  button.primary { flex: 1; padding: 12px; border-radius: 12px; border: none;
                   background: #10b981; color: #04150e; font: inherit;
                   font-weight: 700; cursor: pointer; }
  button.primary:hover:not(:disabled) { background: #34d399; }
  button.primary:disabled { opacity: .45; cursor: not-allowed; }
  button.ghost { padding: 12px 16px; border-radius: 12px; font: inherit;
                 background: transparent; border: 1px solid var(--line);
                 color: var(--muted); cursor: pointer; }
  button.ghost:hover { color: var(--ink); border-color: var(--faint); }
  button.link { background: none; border: none; color: var(--faint); font: inherit;
                font-size: .82rem; cursor: pointer; padding: 0; margin-top: 14px;
                text-decoration: underline; text-underline-offset: 3px; }
  button.link:hover { color: var(--muted); }
  #err { color: #f87171; font-size: .85rem; min-height: 1.2em; margin: 12px 2px 0; }
</style>
</head>
<body>
<main>
  <div class="card" id="card" data-state="idle"><div class="sub">Loading…</div></div>
  <p id="err"></p>
</main>
<script>
// Mirrors the monitor's own validation (monitor/intentions.py, monitor/breaks.py).
const MIN_MIN = __MIN_MIN__, MAX_MIN = __MAX_MIN__, BREAK_MIN = __BREAK_MIN__;
const PRESETS = __PRESETS__;
const R = 46, CIRC = 2 * Math.PI * R;

let duration = PRESETS[0];
let snap = null;        // last /api/status payload
let form = {text: '', custom: '', project: ''};   // survives re-renders
let projectsList = [];  // active projects, for the picker

async function loadProjects() {
  try {
    const r = await fetch('/api/projects');
    const d = await r.json();
    if (Array.isArray(d)) projectsList = d;
  } catch { /* monitor unreachable — keep the last list */ }
}

function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
// Same, but safe to drop inside a quoted HTML attribute.
function escAttr(s) { return esc(s).replace(/"/g, '&quot;'); }

// The server's naive local "YYYY-MM-DD HH:MM:SS" — parsed explicitly, since
// Date's handling of that format is implementation-defined.
function parseStamp(s) {
  const m = /^(\\d{4})-(\\d{2})-(\\d{2}) (\\d{2}):(\\d{2}):(\\d{2})$/.exec(s || '');
  return m ? new Date(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]) : null;
}
function secsLeft() {
  const ends = parseStamp(snap && snap.ends_at);
  return ends ? Math.max(0, Math.round((ends - Date.now()) / 1000)) : null;
}
function fmt(secs) {
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = secs%60;
  return h ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
           : `${m}:${String(s).padStart(2,'0')}`;
}
function usedFraction() {
  const started = parseStamp(snap && snap.started_at), ends = parseStamp(snap && snap.ends_at);
  if (!started || !ends) return 0;
  const span = ends - started;
  return span > 0 ? Math.min(1, Math.max(0, (Date.now() - started) / span)) : 0;
}
function ring() {
  const off = CIRC * (1 - usedFraction());
  return `<svg width="108" height="108" viewBox="0 0 108 108">
    <circle class="ring-track" cx="54" cy="54" r="${R}" fill="none" stroke-width="7"/>
    <circle class="ring-fill" cx="54" cy="54" r="${R}" fill="none" stroke-width="7"
            stroke-linecap="round" stroke-dasharray="${CIRC.toFixed(1)}"
            stroke-dashoffset="${off.toFixed(1)}"/>
  </svg>`;
}
const CHIP = {on: 'On intention', partial: 'Drifting', off: 'Off intention',
              session: 'Session running', break: 'On a break',
              paused: 'Paused', tracking: 'Tracking only', idle: 'No session',
              away: 'Away — not logging'};

async function refresh() {
  try {
    const r = await fetch('/api/status');
    snap = await r.json();
  } catch { return; }   // monitor unreachable — keep showing the last state
  render();
}

function render(force) {
  if (!snap) return;
  const card = document.getElementById('card');
  const left = secsLeft();

  // The idle card contains a form. Rebuilding it under the user's cursor
  // every poll would eat their caret position, so while they're typing we
  // only repaint the state readout and leave the inputs alone. `force` is for
  // the cases that must rebuild it — picking a duration changes the chips.
  const typing = !force && card.dataset.phase === 'idle' && snap.phase === 'idle'
                 && card.contains(document.activeElement);
  card.dataset.state = snap.state;
  card.dataset.phase = snap.phase;
  if (typing) {
    const chip = card.querySelector('.chip');
    const word = card.querySelector('.word');
    if (chip) chip.innerHTML = `<i></i>${CHIP[snap.state] || CHIP.idle}`;
    if (word) word.textContent = snap.label;
    return;
  }

  if (snap.phase === 'break') {
    card.innerHTML = `
      <span class="chip"><i></i>${CHIP.break}</span>
      <div class="dial">${ring()}
        <div><div class="time">${fmt(left ?? 0)}</div>
             <div class="sub">until focus resumes</div></div></div>
      <p class="note">Tracking and alerts are both off. Nothing you do right now
         is classified or logged, and no intention is needed.</p>
      <div class="row"><button class="ghost" onclick="endBreak()">End break now</button></div>`;
    return;
  }

  if (snap.phase === 'session') {
    if (snap.state === 'paused') {
      // The countdown here is the pause's own: when monitoring comes back.
      card.innerHTML = `
        <span class="chip"><i></i>${CHIP.paused}</span>
        <div class="dial">${ring()}
          <div><div class="time">${fmt(left ?? 0)}</div>
               <div class="sub">until the session resumes</div></div></div>
        <p class="note">${esc(snap.label)}<br>
           Timer and monitoring are frozen — no session time is lost.</p>
        <div class="row">
          <button class="primary" onclick="resumeSession()">Resume now</button>
          <button class="ghost" onclick="endSession('abandoned')">Abandon</button>
        </div>`;
      return;
    }
    card.innerHTML = `
      <span class="chip"><i></i>${CHIP[snap.state] || CHIP.session}</span>
      <div class="dial">${ring()}
        <div><div class="time">${fmt(left ?? 0)}</div>
             <div class="sub">${esc(snap.label)}</div></div></div>
      ${snap.intention && snap.intention.project ? `<p class="note">Part of
        <strong>${esc(snap.intention.project.name)}</strong>.</p>` : ''}
      ${snap.category ? `<p class="note">Currently reading as
        <strong>${esc(snap.category)}</strong>.</p>` : ''}
      <div class="row">
        <button class="primary" onclick="endSession('completed')">Done — take a break</button>
        <button class="ghost" onclick="pauseSession()"
                title="Freeze the timer and monitoring for 1 minute">Pause 1m</button>
        <button class="ghost" onclick="endSession('abandoned')">Abandon</button>
      </div>`;
    return;
  }

  card.innerHTML = `
    <span class="chip"><i></i>${CHIP[snap.state] || CHIP.idle}</span>
    <div class="dial"><div><div class="word">${esc(snap.label)}</div>
      <div class="sub">${snap.category ? esc(snap.category) : 'Nothing declared yet'}</div></div></div>
    <label for="text">What are you working on?</label>
    <textarea id="text" maxlength="300" placeholder="e.g. finish the migration write-up"
              oninput="form.text = this.value">${esc(form.text)}</textarea>
    ${projectsList.length ? `<select id="project" aria-label="Project"
              onchange="form.project = this.value">
      <option value="">No project</option>
      ${projectsList.map(p => `<option value="${p.id}"
          ${String(p.id) === String(form.project) ? 'selected' : ''}>${esc(p.name)}${
          p.category ? ' · ' + esc(p.category) : ''}</option>`).join('')}
    </select>` : ''}
    <div class="chips">
      ${PRESETS.map(m => `<button onclick="pick(${m})"
          class="${m === duration ? 'sel' : ''}">${m}m</button>`).join('')}
      <button onclick="pick('custom')" class="${duration === 'custom' ? 'sel' : ''}">Custom</button>
    </div>
    <div class="customrow ${duration === 'custom' ? 'show' : ''}">
      <input id="custom" type="number" min="${MIN_MIN}" max="${MAX_MIN}" placeholder="120"
             value="${escAttr(form.custom)}" aria-label="Custom session length in minutes"
             oninput="form.custom = this.value; markCustom()"
             onkeydown="if (event.key === 'Enter') start()">
      <span class="sub">minutes (${MIN_MIN}–${MAX_MIN})</span>
    </div>
    <div class="row"><button class="primary" onclick="start()">Start session</button></div>
    <button class="link" onclick="startBreak()">or take a ${BREAK_MIN}-minute break now</button>`;

  const t = document.getElementById('text');
  if (t && document.activeElement !== t) t.focus();
}

function pick(m) {
  duration = m;
  render(true);
  if (m === 'custom') { const el = document.getElementById('custom'); if (el) el.focus(); }
}
// Read the live inputs, not the cached copy. `form` exists to survive a
// re-render, but the field is the truth: a value put there without an input
// event (autofill, session restore, an extension) must still submit.
function fieldValue(id) {
  const el = document.getElementById(id);
  return el ? el.value : '';
}
function customMinutes() {
  const n = Number(fieldValue('custom'));
  return Number.isInteger(n) && n >= MIN_MIN && n <= MAX_MIN ? n : null;
}
function markCustom() {
  const el = document.getElementById('custom');
  if (el) el.classList.toggle('bad', el.value !== '' && customMinutes() === null);
}

async function start() {
  let mins = duration;
  if (duration === 'custom') {
    mins = customMinutes();
    if (mins === null) {
      showError(`Enter a length between ${MIN_MIN} and ${MAX_MIN} minutes.`);
      return;
    }
  }
  const text = fieldValue('text') || form.text;
  if (!text.trim()) { showError('Say what you intend to work on first.'); return; }
  const body = {text, duration_min: mins};
  const project = fieldValue('project') || form.project;
  if (project) body.project_id = Number(project);
  if (await post('/api/intention', body)) {
    // The project stays picked: the next session is usually the same work.
    form = {text: '', custom: '', project};
  }
}
async function endSession(status) { await post('/api/intention/end', {status}); }
async function pauseSession() { await post('/api/intention/pause', {}); }
// A pause is a break under the hood, so resuming early is ending the break.
async function resumeSession() { await post('/api/break/end', {}); }
async function startBreak() { await post('/api/break', {duration_min: BREAK_MIN}); }
async function endBreak() { await post('/api/break/end', {}); }

function showError(msg) { document.getElementById('err').textContent = msg; }

async function post(url, body) {
  showError('');
  let r, d;
  try {
    r = await fetch(url, {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    d = await r.json();
  } catch {
    showError("Can't reach the monitor. Is it still running?");
    return false;
  }
  if (!r.ok || d.error) { showError(d.error || r.statusText); return false; }
  await refresh();
  return true;
}

// Tick locally every second so the countdown is smooth, and re-read the
// monitor every few seconds to pick up changes made in the tray or widget.
setInterval(() => { if (snap && snap.ends_at) render(); }, 1000);
setInterval(refresh, 5000);
setInterval(loadProjects, 60000);
loadProjects().then(refresh);
</script>
</body>
</html>
"""

_INTENTION_PAGE = (
    _INTENTION_PAGE_SRC
    .replace("__MIN_MIN__", str(intentions.MIN_DURATION_MIN))
    .replace("__MAX_MIN__", str(intentions.MAX_DURATION_MIN))
    .replace("__BREAK_MIN__", str(breaks.DEFAULT_BREAK_MIN))
    # Presets start at the real minimum, so the page can never offer a length
    # the monitor will reject.
    .replace("__PRESETS__", json.dumps(
        [m for m in (30, 45, 60, 90, 120) if intentions.MIN_DURATION_MIN <= m <= intentions.MAX_DURATION_MIN]
    ))
)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_local_server() -> Optional[str]:
    """Start the local API (and the bundled dashboard, if staged).

    Binds a FIXED port so a hosted dashboard can find it. Returns the local
    base URL, or None if the port couldn't be bound.
    """
    global _STATIC_ROOT
    root = resource_path("dashboard")
    _STATIC_ROOT = str(root) if (root / "index.html").exists() else None

    port = _api_port()
    handler = functools.partial(_Handler, directory=_STATIC_ROOT or str(data_dir()))
    try:
        httpd = _Server(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"[api] could not bind 127.0.0.1:{port} ({e}). "
              f"Is another cft already running? The hosted dashboard needs this port.")
        return None

    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http").start()
    allowed = _allowed_origins()
    print(f"[api] local API on http://127.0.0.1:{port} "
          f"(static={'bundled' if _STATIC_ROOT else 'hosted'}, allowed origins={sorted(allowed) or 'none'})")
    return f"http://127.0.0.1:{port}/"


# Back-compat alias for older callers.
serve_bundled_dashboard = start_local_server
