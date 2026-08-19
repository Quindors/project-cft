# monitor/status.py
"""Live monitor state, published by the runner and read by the UI.

Everything else the UI needs — mode, active session, active break — is DB
backed and readable from any thread. What is *not* in the DB is what the last
decision concluded: which intent tier the current activity falls in, what it
was classified as, how long the drift has been running. The timer widget
colors itself from exactly that, and the dashboard shows it too.

So this module is a single in-memory snapshot with a lock: the runner calls
publish() after each decision, the tk widget and the HTTP threads call get().
It is deliberately not persisted — a stale tier from a previous run would be
worse than no tier at all.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

# How long a published decision stays meaningful. Past this the monitor is
# either wedged or between runs, and the UI should fall back to "no signal"
# rather than keep showing a tier from ten minutes ago.
STALE_AFTER_SEC = 180.0

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "tier": None,          # on-intent | partial-intent | off-intent | no-signal
    "category": None,      # last classified category name
    "align": None,         # 0-1 alignment with the session intention, or None
    "is_productive": None,
    "at": None,            # time.time() of the decision
    "drift_since": None,   # time.time() the current drift streak started
    "paused": False,       # monitoring suspended (a break is running)
    "away": False,         # no input past the idle threshold; nothing logs
}


def publish(**fields) -> None:
    """Record what the latest decision concluded. Unknown keys are ignored."""
    with _lock:
        for key, value in fields.items():
            if key in _state:
                _state[key] = value
        if "at" not in fields:
            _state["at"] = time.time()


def set_paused(paused: bool) -> None:
    """Mark monitoring as suspended (break) or live again."""
    with _lock:
        _state["paused"] = paused
        if paused:
            _state["tier"] = None
            _state["drift_since"] = None


def set_away(away: bool) -> None:
    """Mark the user as away (no input past the idle threshold) or back."""
    with _lock:
        _state["away"] = away
        if away:
            _state["tier"] = None
            _state["drift_since"] = None


def get() -> Dict[str, Any]:
    """A copy of the current state, with stale decisions blanked out."""
    with _lock:
        snap = dict(_state)
    at = snap.get("at")
    snap["stale"] = at is None or (time.time() - at) > STALE_AFTER_SEC
    if snap["stale"]:
        snap["tier"] = None
        snap["drift_since"] = None
    return snap


def drift_seconds() -> Optional[float]:
    """How long the current drift streak has run, or None if not drifting."""
    snap = get()
    since = snap.get("drift_since")
    return None if since is None else max(0.0, time.time() - since)


# ---------- the one view every UI renders ----------

# Phases, most specific first: a break outranks a session, a session outranks
# plain monitoring.
PHASE_BREAK = "break"
PHASE_SESSION = "session"
PHASE_IDLE = "idle"

# Color states. The timer widget, the tray icon and the dashboard all key off
# these strings, so a rename is a rename everywhere — which is the point:
# there is exactly one place that decides what state you are in.
STATE_BREAK = "break"        # on a break; nothing is being watched
STATE_ON = "on"              # doing the declared work
STATE_PARTIAL = "partial"    # loosely related to it
STATE_OFF = "off"            # not doing it
STATE_SESSION = "session"    # session running, no verdict yet
STATE_PAUSED = "paused"      # session frozen for a minute; nothing watched
STATE_TRACKING = "tracking"  # time-tracking mode, no session
STATE_IDLE = "idle"          # distraction-free, no session
STATE_AWAY = "away"          # no input at all; monitoring stood down


def full_snapshot() -> Dict[str, Any]:
    """Everything a timer/tray/dashboard needs, resolved into one view.

    Combines the DB-backed facts (mode, active session, active break) with the
    in-memory decision state above, and reduces them to a phase, a label, a
    countdown and a color state — so the widget and the dashboard can never
    disagree about what is going on.
    """
    from monitor import app_state, breaks, intentions

    snap = get()

    try:
        mode = app_state.get_mode()
    except Exception:
        mode = app_state.DEFAULT_MODE

    try:
        active_break = breaks.get_active()
    except Exception:
        active_break = None

    # Don't touch intentions during a break: get_active() expires past-due
    # sessions and would start a second break on top of this one. A pause is
    # the exception — pausing moved the session's ends_at past the pause's own
    # end, so the session cannot be past due here and the read is safe.
    is_pause = active_break is not None and active_break.get("source") == "pause"
    active_intention = None
    if active_break is None or is_pause:
        try:
            active_intention = intentions.get_active()
        except Exception:
            active_intention = None

    out: Dict[str, Any] = {
        "mode": mode,
        "break": active_break,
        "intention": active_intention,
        "tier": snap.get("tier"),
        "category": snap.get("category"),
        "align": snap.get("align"),
        "is_productive": snap.get("is_productive"),
        "stale": snap.get("stale"),
        "paused": active_break is not None,
        "away": bool(snap.get("away")),
    }

    if is_pause and active_intention is not None:
        # A paused session is still a session — the card keeps the intention,
        # and the countdown is the pause's own: when monitoring comes back.
        out.update(
            phase=PHASE_SESSION,
            state=STATE_PAUSED,
            label=active_intention.get("text") or "Focus session",
            started_at=active_break.get("started_at"),
            ends_at=active_break.get("ends_at"),
            remaining_sec=active_break.get("remaining_sec"),
        )
    elif active_break is not None:
        # Real breaks — and a pause whose session is somehow gone, which has
        # nothing left to resume and simply runs out as a break.
        out.update(
            phase=PHASE_BREAK,
            state=STATE_BREAK,
            label="Break",
            started_at=active_break.get("started_at"),
            ends_at=active_break.get("ends_at"),
            remaining_sec=active_break.get("remaining_sec"),
        )
    elif active_intention is not None:
        tier_state = {
            "on-intent": STATE_ON,
            "partial-intent": STATE_PARTIAL,
            "off-intent": STATE_OFF,
        }.get(snap.get("tier") or "", STATE_SESSION)
        out.update(
            phase=PHASE_SESSION,
            state=tier_state,
            label=active_intention.get("text") or "Focus session",
            started_at=active_intention.get("started_at"),
            ends_at=active_intention.get("ends_at"),
            remaining_sec=seconds_until(active_intention.get("ends_at")),
        )
    else:
        tracking_only = mode != app_state.MODE_DISTRACTION_FREE
        # No session means no ALIGN, so the only drift signal left is the
        # plain productive/not verdict — and only while it is fresh.
        off_task = snap.get("is_productive") is False and not snap.get("stale")
        out.update(
            phase=PHASE_IDLE,
            state=STATE_TRACKING if tracking_only
            else (STATE_OFF if off_task else STATE_IDLE),
            label="Tracking" if tracking_only else "Focus",
            started_at=None,
            ends_at=None,
            remaining_sec=None,
        )

    # Away recolors whatever is going on — a session keeps its countdown, but
    # the widget goes grey and the reason is on the card. A break outranks it:
    # monitoring is off either way, and the break says why better.
    if out["away"] and active_break is None:
        out["state"] = STATE_AWAY
        if out["phase"] == PHASE_IDLE:
            out["label"] = "Away"
    return out


def seconds_until(stamp: Optional[str]) -> Optional[int]:
    """Seconds left until a naive local "YYYY-MM-DD HH:MM:SS" stamp."""
    if not stamp:
        return None
    from datetime import datetime
    try:
        ends = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max(0, int((ends - datetime.now()).total_seconds()))
