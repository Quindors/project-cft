# monitor/beeminder.py
"""Push focus totals to Beeminder goals.

Beeminder is a commitment contract: you pick a rate, it draws a line through
your cumulative total, and it charges you when a day ends with your total on
the wrong side of it. This module feeds that line.

Entirely opt-in, and per-user by construction — the account name is part of
the endpoint URL and the token belongs to that account — so nothing happens
until the .env names a user, a token and at least one goal.

Two things get pushed:

  focus hours   one datapoint per day, holding the productive hours logged so
                far that day. Re-sent every sync, climbing as the day goes on.
  sessions      one datapoint per focus session that reached its end with
                real logged activity behind it, on a separate count goal.

Both rely on `requestid`, which is an upsert scoped to the goal: the same id
with the same value is ignored, the same id with a different value updates the
datapoint in place. That is what makes re-sending free — there is no local
"already sent" bookkeeping to drift out of sync, an interrupted send costs
nothing, and a day whose rows are relabelled in the review tab can simply be
pushed again to correct its datapoint after the fact.

Beemind focus hours rather than distraction minutes. A do-less goal fed by a
monitor you can close is gameable in the wrong direction: quitting the exe
would report zero distraction. On a do-more goal quitting it merely stops the
credit, so the failure mode costs you instead of paying you.

Everything here is best-effort and off the critical path. Failures print and
are dropped; the local database is the source of truth and is never written to
from this module.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from monitor.config import DEFAULT_SETTINGS
from monitor.paths import data_dir

# ALIGN below this is off-intent — the same line the drift alerts draw.
OFF_INTENT_MAX = DEFAULT_SETTINGS.intention.off_intent_max

API_ROOT = "https://www.beeminder.com/api/v1"
_TS_FMT = "%Y-%m-%d %H:%M:%S"

# Sessions worth a datapoint. Abandoning a session is not finishing it, and a
# superseded one was replaced rather than done — same rule the auto-break uses.
COUNTED_SESSION_STATUSES = ("completed", "expired")

# ...and a finished session only counts if someone actually sat through it: at
# least this many seconds of activity must have been logged under its
# intention. Two holes close at once. Clicking "Done" moments after starting
# would otherwise mint a datapoint per four clicks, and a session left to
# expire in front of an empty chair logs (almost) nothing past the idle
# threshold, so it no longer counts as finished work either.
SESSION_MIN_LOGGED_SEC = 600.0


@dataclass(frozen=True)
class Config:
    user: str
    token: str
    focus_goal: str
    sessions_goal: str
    sync_sec: float
    timeout_sec: float
    seconds_per_row: float
    dry_run: bool
    strict_intent: bool


@dataclass(frozen=True)
class DayTotals:
    """A day's logged time, in hours, split by the category's productive flag.

    Ambiguous activity (a category with is_productive NULL) counts as neither,
    so the two figures deliberately do not add up to the wall-clock day.
    """
    productive_hours: float
    offtask_hours: float
    comment: str


# ------------------- configuration -------------------

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[beeminder] {name}={raw!r} is not a number; using {default}")
        return default


def config() -> Config:
    """Read the integration's settings from the environment.

    Read on every call rather than cached at boot so a .env edit takes effect
    on the next sync, and so the session hook — which runs on whichever thread
    happened to end the session — needs no handle to anything.
    """
    return Config(
        user=_env("BEEMINDER_USER"),
        token=_env("BEEMINDER_TOKEN"),
        focus_goal=_env("BEEMINDER_FOCUS_GOAL"),
        sessions_goal=_env("BEEMINDER_SESSIONS_GOAL"),
        sync_sec=_env_float("BEEMINDER_SYNC_SEC", 300.0),
        timeout_sec=_env_float("BEEMINDER_TIMEOUT_SEC", 15.0),
        # Each logged row stands for one decision interval of activity; that is
        # the only bridge from "rows in the table" to "hours at the desk".
        seconds_per_row=_env_float("BEEMINDER_SECONDS_PER_ROW",
                                   DEFAULT_SETTINGS.decision_interval),
        dry_run=_env("BEEMINDER_DRY_RUN").lower() in ("1", "true", "yes"),
        # Work done off-intent during a focus session counts as off-task, not
        # productive — a declared session means the hours goal measures the
        # work you said you'd do, and drifting to other real work doesn't pay
        # it. Set BEEMINDER_STRICT_INTENT=0 to count any productive-category
        # work regardless of the active session.
        strict_intent=_env("BEEMINDER_STRICT_INTENT", "1").lower() in ("1", "true", "yes"),
    )


def enabled(cfg: Optional[Config] = None) -> bool:
    cfg = cfg or config()
    return bool(cfg.user and cfg.token and (cfg.focus_goal or cfg.sessions_goal))


# ------------------- reading the local database -------------------

# A reviewed row's human label is the truth; the classifier's guess is the
# fallback. Corrections made in the dashboard therefore change the number that
# gets pushed, and re-pushing the day fixes the datapoint after the fact.
_LABEL = "COALESCE(NULLIF(TRIM(f.human_label), ''), f.category_name)"

# off_intent: logged during a focus session with an ALIGN score below the
# drift line — the same tier the runner's drift alerts fire on. Rows without
# an ALIGN score (no session, cache shortcuts) are never off-intent.
_DAY_QUERY = f"""
    SELECT {_LABEL} AS label, c.is_productive AS productive,
           (f.intention_id IS NOT NULL
            AND f.align_score IS NOT NULL
            AND f.align_score < ?) AS off_intent,
           COUNT(*) AS n
      FROM focus_logs f
      LEFT JOIN categories c ON c.name = {_LABEL}
     WHERE f.timestamp >= ? AND f.timestamp < ?
  GROUP BY label, productive, off_intent
  ORDER BY n DESC
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def day_totals(day: date, *, seconds_per_row: float, strict_intent: bool = True) -> DayTotals:
    """Total the rows logged on `day`. Never raises — an unreadable database
    reports an empty day, which is the honest thing to push.

    By default a productive-category row logged off-intent during a focus
    session goes to the off-task column: it was real work, but not the work
    you declared. strict_intent=False counts it as productive anyway.
    """
    start = f"{day:%Y-%m-%d} 00:00:00"
    end = f"{day + timedelta(days=1):%Y-%m-%d} 00:00:00"
    try:
        with _connect() as conn:
            rows: List[sqlite3.Row] = conn.execute(
                _DAY_QUERY, (OFF_INTENT_MAX, start, end)
            ).fetchall()
    except sqlite3.Error as e:
        # A fresh install has no focus_logs table yet.
        print(f"[beeminder] could not read focus.db ({e}); reporting an empty day")
        return DayTotals(0.0, 0.0, "")

    productive_sec = offtask_sec = drifted_sec = 0.0
    parts: List[str] = []
    for row in rows:
        seconds = row["n"] * seconds_per_row
        if row["productive"] == 1 and strict_intent and row["off_intent"]:
            offtask_sec += seconds
            drifted_sec += seconds
        elif row["productive"] == 1:
            productive_sec += seconds
            # 0.1h is the smallest slice the breakdown can render; anything
            # below it would read as a pointless "Foo 0.0h".
            if len(parts) < 3 and seconds >= 360:
                parts.append(f"{row['label']} {seconds / 3600:.1f}h")
        elif row["productive"] == 0:
            offtask_sec += seconds

    comment = f"{productive_sec / 3600:.2f}h productive"
    if parts:
        comment += " — " + ", ".join(parts)
    # Same 0.1h floor as the breakdown: below it the figure renders as 0.0h.
    if offtask_sec >= 360:
        comment += f" (off-task {offtask_sec / 3600:.1f}h"
        if drifted_sec >= 360:
            comment += f", of which off-intent {drifted_sec / 3600:.1f}h"
        comment += ")"

    return DayTotals(
        productive_hours=round(productive_sec / 3600, 2),
        offtask_hours=round(offtask_sec / 3600, 2),
        comment=comment,
    )


def session_logged_sec(intention_id, *, seconds_per_row: float) -> float:
    """Seconds of activity actually logged under this intention. Never raises.

    Rows still sitting in the runner's write buffer (a handful at most) are
    invisible here, so the figure can undershoot slightly — fine for a floor.
    An unreadable database reads as zero, the same honest fallback day_totals
    uses: when in doubt, no credit.
    """
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM focus_logs WHERE intention_id = ?",
                (intention_id,),
            ).fetchone()
    except sqlite3.Error as e:
        print(f"[beeminder] could not read focus.db ({e}); "
              f"counting the session as unlogged")
        return 0.0
    return float(row["n"] if row else 0) * seconds_per_row


# ------------------- posting -------------------

def _flatten_messages(value) -> List[str]:
    """Pull the human sentences out of an error payload of any shape.

    Beeminder nests its messages — {"errors": {"slug": ["You already have a
    goal with that goalname"]}} — and the field names add nothing the sentence
    doesn't already say. Order is preserved; blanks are dropped.
    """
    if isinstance(value, dict):
        return [m for v in value.values() for m in _flatten_messages(v)]
    if isinstance(value, (list, tuple)):
        return [m for v in value for m in _flatten_messages(v)]
    s = str(value).strip()
    return [s] if s else []


def _error_detail(err: HTTPError) -> str:
    """The API's own message as plain text, so a bad goal slug says so
    instead of '422' — and never as raw JSON."""
    try:
        body = err.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body[:300]
    if isinstance(parsed, dict):
        parsed = parsed.get("errors") or parsed.get("error") or parsed
    return "; ".join(_flatten_messages(parsed))[:300]


def post_datapoint(
    goal: str,
    value: float,
    *,
    requestid: str,
    comment: str = "",
    daystamp: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Optional[Dict]:
    """POST one datapoint. Returns the datapoint, or None if it did not land.

    The token goes in the request body, never the URL, so it stays out of
    proxy logs and crash dumps.
    """
    cfg = cfg or config()
    fields = {"auth_token": cfg.token, "value": f"{value:g}", "requestid": requestid}
    if comment:
        fields["comment"] = comment[:500]
    if daystamp:
        fields["daystamp"] = daystamp

    if cfg.dry_run:
        shown = {k: v for k, v in fields.items() if k != "auth_token"}
        print(f"[beeminder] DRY RUN {goal} <- {shown}")
        _note_push(goal, ok=None, error=None)
        return None

    url = (f"{API_ROOT}/users/{quote(cfg.user, safe='')}"
           f"/goals/{quote(goal, safe='')}/datapoints.json")
    body = urlencode(fields).encode("utf-8")

    for attempt in (1, 2):
        try:
            req = Request(url, data=body, method="POST",
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(req, timeout=cfg.timeout_sec) as resp:
                point = json.loads(resp.read().decode("utf-8"))
            print(f"[beeminder] {goal} = {value:g} ({requestid})")
            _note_push(goal, ok=True, error=None)
            return point
        except HTTPError as e:
            # Beeminder answers a re-sent requestid whose value has not
            # changed with an error status reading "Duplicate request". That
            # is the upsert working as designed - the datapoint is already
            # exactly this - not a failure, so it must not trip the card.
            if e.code < 500:
                detail = _error_detail(e)
                if "duplicate" in detail.lower():
                    _note_push(goal, ok=True, error=None)
                    return {"duplicate": True, "requestid": requestid}
            # 4xx means the request itself is wrong — a bad slug, a dead token,
            # a goal that does not exist. A second identical try cannot fix it.
            if e.code < 500 or attempt == 2:
                problem = _http_problem(e, cfg.user, goal)
                print(f"[beeminder] {goal}: {problem}")
                _note_push(goal, ok=False, error=problem, reason=_reason_for(e))
                return None
        except (URLError, OSError, ValueError) as e:
            if attempt == 2:
                problem = f"could not reach Beeminder ({e})"
                print(f"[beeminder] {goal}: {problem}")
                _note_push(goal, ok=False, error=problem, reason="unreachable")
                return None
        time.sleep(2)
    return None


# Why a request failed, in terms the card can act on:
#   missing_goal  the goal is gone (deleted, renamed) -> offer to make a new one
#   bad_token     the token no longer works           -> ask for a new token
#   unreachable   network / Beeminder down            -> nothing to do but wait
#   other         anything else, with Beeminder's own words
def _reason_for(err: HTTPError) -> str:
    if err.code in (401, 403):
        return "bad_token"
    if err.code == 404:
        return "missing_goal"
    return "other"


def _http_problem(err: HTTPError, user: str, goal: str) -> str:
    """One sentence for what Beeminder is objecting to, for humans."""
    if err.code in (401, 403):
        return "Beeminder rejected the token"
    if err.code == 404:
        return f"Beeminder has no goal '{goal}' under user '{user}' (deleted or renamed?)"
    detail = _error_detail(err)
    return detail or f"Beeminder answered HTTP {err.code}"


# ------------------- health (what the dashboard card reports) -------------------

# Outcome of the most recent push, per goal. The card reads this to tell a
# healthy integration from one that is quietly failing — a deleted goal or a
# revoked token never stops the monitor, so it has to be visible somewhere
# other than cft.log.
_health_lock = threading.Lock()
_health: Dict[str, Dict] = {}


def _note_push(goal: str, *, ok: Optional[bool], error: Optional[str],
               reason: Optional[str] = None) -> None:
    with _health_lock:
        _health[goal] = {
            "ok": ok,                       # True sent, False failed, None dry run
            "error": error,
            "reason": reason,
            "at": datetime.now().strftime(_TS_FMT),
        }


def last_push(goal: str) -> Optional[Dict]:
    with _health_lock:
        entry = _health.get(goal)
        return dict(entry) if entry else None


def reset_health() -> None:
    """Forget past outcomes — after a reconnect they describe the old goal."""
    with _health_lock:
        _health.clear()
    _forget_goal_check()


# ------------------- the two things we push -------------------

def sync_focus_day(day: Optional[date] = None, *, cfg: Optional[Config] = None) -> Optional[Dict]:
    """Push one day's productive hours. Safe to call as often as you like.

    A zero is pushed as readily as anything else: on a do-more goal it is
    equivalent to no datapoint at all, and posting it keeps the correction path
    honest — a day relabelled down to nothing can be walked back to zero.
    """
    cfg = cfg or config()
    if not cfg.focus_goal or not enabled(cfg):
        return None
    day = day or date.today()
    totals = day_totals(day, seconds_per_row=cfg.seconds_per_row,
                        strict_intent=cfg.strict_intent)
    daystamp = f"{day:%Y%m%d}"
    return post_datapoint(
        cfg.focus_goal,
        totals.productive_hours,
        requestid=f"cft-focus-{daystamp}",
        daystamp=daystamp,
        comment=totals.comment,
        cfg=cfg,
    )


def _session_daystamp(intention: Dict) -> Optional[str]:
    """Credit a session to the day it started, so a session running past
    midnight does not land on a day it barely touched."""
    started = intention.get("started_at")
    if not started:
        return None
    try:
        return f"{datetime.strptime(started, _TS_FMT):%Y%m%d}"
    except (ValueError, TypeError):
        return None


def push_session(intention: Dict, status: str, *, cfg: Optional[Config] = None) -> Optional[Dict]:
    """Record one finished focus session on the sessions goal.

    Only sessions with SESSION_MIN_LOGGED_SEC of logged activity count — how
    the session ended says less than whether anyone was there for it.
    """
    cfg = cfg or config()
    if not cfg.sessions_goal or not enabled(cfg):
        return None
    if status not in COUNTED_SESSION_STATUSES:
        return None
    if not intention.get("id"):
        return None

    logged = session_logged_sec(intention["id"], seconds_per_row=cfg.seconds_per_row)
    if logged < SESSION_MIN_LOGGED_SEC:
        print(f"[beeminder] session {intention['id']} not counted: "
              f"{logged / 60:.1f}m logged, the floor is "
              f"{SESSION_MIN_LOGGED_SEC / 60:.0f}m")
        return None

    text = " ".join(str(intention.get("text", "")).split())
    return post_datapoint(
        cfg.sessions_goal,
        1,
        requestid=f"cft-session-{intention['id']}",
        daystamp=_session_daystamp(intention),
        comment=f"{status}: {text}" if text else status,
        cfg=cfg,
    )


def after_session_end(intention: Dict, status: str) -> Optional[threading.Thread]:
    """Session-end hook, wired in from monitor/intentions.py.

    Sessions end on whichever thread first notices — usually the runner loop,
    sometimes a dashboard request — so the POST goes to a thread of its own.
    Blocking here would stall the monitor for the length of a network timeout.
    """
    if not enabled() or status not in COUNTED_SESSION_STATUSES:
        return None

    def worker() -> None:
        try:
            push_session(intention, status)
        except Exception as e:
            print(f"[beeminder] session push failed ({e})")

    thread = threading.Thread(target=worker, daemon=True, name="beeminder-session")
    thread.start()
    return thread


# ------------------- setup surface (dashboard settings card) -------------------

def status() -> Dict:
    """What the settings card needs to know — never includes the token.

    last_push is the outcome of the most recent focus-goal push: ok True/False
    (None while in dry run), an error sentence, and when. It is how a deleted
    goal or a dead token becomes visible instead of just a line in cft.log.
    """
    cfg = config()
    push = last_push(cfg.focus_goal) if cfg.focus_goal else None
    goal = check_goal(cfg)

    # One thing for the card to act on. A goal that is definitely gone wins
    # over a stale push failure; an unreachable probe says nothing either way.
    problem = None
    if goal and goal["exists"] is False:
        problem = {"reason": goal["reason"], "message": goal["error"]}
    elif push and push["ok"] is False:
        problem = {"reason": push.get("reason") or "other", "message": push["error"]}

    return {
        "configured": enabled(cfg),
        "user": cfg.user or None,
        "focus_goal": cfg.focus_goal or None,
        "sessions_goal": cfg.sessions_goal or None,
        "dry_run": cfg.dry_run,
        "sync_running": _sync_thread is not None and _sync_thread.is_alive(),
        "last_push": push,
        "goal_check": goal,
        "problem": problem,
    }


def sync_now() -> threading.Thread:
    """Push today off the calling thread, so a settings change is reflected
    in last_push within seconds rather than at the next five-minute tick."""
    thread = threading.Thread(target=lambda: sync_focus_day(), daemon=True,
                              name="beeminder-sync-now")
    thread.start()
    return thread


_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# A Do More goal is open-ended: Beeminder wants exactly two of (rate, goaldate,
# goalval), so alongside the rate we send a deadline a decade out — effectively
# endless, and the user can archive the goal whenever they like.
_GOAL_HORIZON_SEC = 10 * 365 * 24 * 3600


MAX_LEEWAY_DAYS = 30


def create_goal(user: str, token: str, slug: str, hours_per_day, leeway_days=0,
                *, timeout_sec: float = 15.0) -> Optional[str]:
    """Create a Do More goal (units: hours) on the user's own account.

    hours_per_day is the commitment — the daily rate of the bright red line.
    leeway_days is starting safety buffer: the goal begins that many days
    ahead of the line (initval = rate x days), so a rough first week can't
    derail anyone before the data source has earned their trust.

    Returns None on success, else a message fit to show the user. Creation
    doubles as credential verification — it can only succeed with a working
    token — so callers don't need a separate verify() pass. The new goal
    starts with whatever pledge Beeminder gives a fresh goal on that account;
    no payment details pass through here.
    """
    slug = (slug or "").strip()
    if not _SLUG_RE.match(slug):
        return "goal name must be letters, numbers, dashes or underscores"
    try:
        rate = float(hours_per_day)
    except (TypeError, ValueError):
        return "hours per day must be a number"
    if not (0 < rate <= 24):
        return "hours per day must be between 0 and 24"
    try:
        leeway = float(leeway_days or 0)
    except (TypeError, ValueError):
        return "leeway days must be a number"
    if not (0 <= leeway <= MAX_LEEWAY_DAYS):
        return f"leeway must be between 0 and {MAX_LEEWAY_DAYS} days"

    fields = {
        "auth_token": token,
        "slug": slug,
        "title": "Focus hours (cft)",
        "goal_type": "hustler",       # Do More
        "gunits": "hours",
        "runits": "d",
        "rate": f"{rate:g}",
        "goaldate": str(int(time.time()) + _GOAL_HORIZON_SEC),
        "initval": f"{rate * leeway:g}",
        "datasource": "api",          # data arrives from this monitor, not by hand
    }
    url = f"{API_ROOT}/users/{quote(user, safe='')}/goals.json"
    try:
        req = Request(url, data=urlencode(fields).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(req, timeout=timeout_sec) as resp:
            json.loads(resp.read().decode("utf-8"))
        print(f"[beeminder] created goal '{slug}' ({rate:g}h/day) for {user}")
        return None
    except HTTPError as e:
        if e.code in (401, 403):
            return "Beeminder rejected that token"
        detail = _error_detail(e)
        # 422 is Beeminder explaining itself (slug taken, bad rate...) — its
        # own words are more useful than anything we could paraphrase.
        return detail or f"Beeminder answered HTTP {e.code}"
    except (URLError, OSError, ValueError) as e:
        return f"could not reach Beeminder ({e})"


def probe_goal(user: str, token: str, goal: str, *, timeout_sec: float = 10.0
               ) -> Tuple[Optional[str], Optional[str]]:
    """Ask Beeminder whether (user, token, goal) is a live trio.

    Returns (reason, message): (None, None) when it checks out, else one of
    the _reason_for kinds and a sentence fit to show the user. Read-only —
    fetches the goal rather than posting a datapoint, so a typo'd slug can't
    smudge a real graph. Beeminder's API authenticates GETs by query
    parameter: over TLS to beeminder.com only, never logged here.
    """
    url = (f"{API_ROOT}/users/{quote(user, safe='')}"
           f"/goals/{quote(goal, safe='')}.json?{urlencode({'auth_token': token})}")
    try:
        with urlopen(Request(url), timeout=timeout_sec) as resp:
            json.loads(resp.read().decode("utf-8"))
        return None, None
    except HTTPError as e:
        if e.code in (401, 403):
            return "bad_token", "Beeminder rejected that token"
        if e.code == 404:
            return "missing_goal", f"Beeminder has no goal '{goal}' under user '{user}'"
        return "other", f"Beeminder answered HTTP {e.code}: {_error_detail(e)}"
    except (URLError, OSError, ValueError) as e:
        return "unreachable", f"could not reach Beeminder ({e})"


def verify(user: str, token: str, goal: str, *, timeout_sec: float = 10.0) -> Optional[str]:
    """probe_goal for callers that only want the sentence (None = fine)."""
    return probe_goal(user, token, goal, timeout_sec=timeout_sec)[1]


# The card polls status every 30s; a deleted goal must show up without
# waiting for the next five-minute push - and in dry-run mode a push would
# never notice at all. So status() probes the goal itself, at most once a
# minute, read-only.
_GOAL_CHECK_TTL_SEC = 60.0
_goal_check_lock = threading.Lock()
_goal_check: Dict = {"key": None, "at": 0.0, "result": None}


def check_goal(cfg: Optional[Config] = None, *, force: bool = False) -> Optional[Dict]:
    """Does the configured focus goal still exist? Cached for a minute.

    Returns {"exists": bool|None, "reason", "error", "checked_at"} or None
    when there is nothing configured to check. exists is None when Beeminder
    could not be reached - unknown is not the same as gone.
    """
    cfg = cfg or config()
    if not enabled(cfg) or not cfg.focus_goal:
        return None
    key = (cfg.user, cfg.focus_goal, cfg.token)
    now = time.time()
    with _goal_check_lock:
        fresh = (_goal_check["key"] == key
                 and now - _goal_check["at"] < _GOAL_CHECK_TTL_SEC)
        if fresh and not force:
            return dict(_goal_check["result"])
    reason, message = probe_goal(cfg.user, cfg.token, cfg.focus_goal)
    result = {
        "exists": None if reason == "unreachable" else reason is None,
        "reason": reason,
        "error": message,
        "checked_at": datetime.now().strftime(_TS_FMT),
    }
    with _goal_check_lock:
        _goal_check.update(key=key, at=now, result=result)
    return dict(result)


def _forget_goal_check() -> None:
    with _goal_check_lock:
        _goal_check.update(key=None, at=0.0, result=None)


# ------------------- the background sync -------------------

# One sync thread per process. The stop event registers at boot whether or not
# the integration is configured, so enabling it later from the settings card
# can start the loop with the same shutdown wiring.
_sync_lock = threading.Lock()
_sync_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None


def start_background_sync(stop_event: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """Push today's hours every sync_sec until the registered stop event fires.

    Idempotent: a live thread is returned as-is, so the boot path and the
    settings endpoint can both call this without stacking loops. Runs on its
    own thread reading its own connection, so the monitor loop never waits on
    the network, and a machine that is offline all afternoon simply catches up
    on the next tick.
    """
    global _sync_thread, _stop_event
    with _sync_lock:
        if stop_event is not None:
            _stop_event = stop_event
        stop = _stop_event
        if stop is None:
            return None
        if _sync_thread is not None and _sync_thread.is_alive():
            return _sync_thread

        cfg = config()
        if not enabled(cfg) or not cfg.focus_goal:
            return None
        _sync_thread = _spawn_sync_loop(stop, cfg)
        return _sync_thread


def _spawn_sync_loop(stop_event: threading.Event, cfg: Config) -> threading.Thread:
    def loop() -> None:
        last_day: Optional[date] = None
        while True:
            today = date.today()
            try:
                # Rolled over since the last tick: settle the day that ended
                # before moving on, or its final hour never gets reported.
                if last_day is not None and last_day != today:
                    sync_focus_day(last_day)
                sync_focus_day(today)
                last_day = today
            except Exception as e:
                print(f"[beeminder] sync failed ({e}); will retry next tick")
            if stop_event.wait(cfg.sync_sec):
                break
        try:
            sync_focus_day()  # last word on the way out
        except Exception as e:
            print(f"[beeminder] final sync failed ({e})")

    thread = threading.Thread(target=loop, daemon=True, name="beeminder-sync")
    thread.start()
    mode = " (DRY RUN)" if cfg.dry_run else ""
    print(f"[beeminder] pushing '{cfg.focus_goal}' every {cfg.sync_sec:.0f}s "
          f"as {cfg.user}{mode}")
    return thread
