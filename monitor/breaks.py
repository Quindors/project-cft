# monitor/breaks.py
"""Post-session breaks: a fixed stretch where the monitor stops watching.

A break is the deliberate opposite of a focus session. While one runs the
monitor is *fully* off: no window polling, no keystroke capture, no
classification, nothing written to focus_logs, and no alerts. There is no
intention to declare and nothing to drift from — the timer just counts down,
and when it hits zero the monitor returns to its normal mode on its own.

Breaks are started automatically when a focus session finishes (completed or
expired — abandoning a session does not earn one) via `after_session_end`,
which monitor/app.py wires into intentions as the session-end hook. They can
also be started on demand from the dashboard, the tray or the timer widget.

State lives in the app_state key/value table as a single JSON blob under
`break`, so the runner loop, the HTTP threads and the widget all read the same
row without a second table. Timestamps are naive local
"YYYY-MM-DD HH:MM:SS" strings, matching intentions and focus_logs.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, Optional

from monitor import app_state

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_KEY = "break"

DEFAULT_BREAK_MIN = 15
MIN_BREAK_MIN = 1
MAX_BREAK_MIN = 120


def _now_str() -> str:
    return datetime.now().strftime(_TS_FMT)


def _parse(raw: Optional[str]) -> Optional[Dict]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) and data.get("ends_at") else None


def _decorate(data: Dict) -> Dict:
    """Add the derived fields every consumer wants (seconds left)."""
    try:
        ends = datetime.strptime(data["ends_at"], _TS_FMT)
    except (KeyError, ValueError):
        return data
    data = dict(data)
    data["remaining_sec"] = max(0, int((ends - datetime.now()).total_seconds()))
    return data


def get_active() -> Optional[Dict]:
    """The running break, or None.

    A finished break is cleared here as a side effect, so callers never see a
    stale one and the monitor resumes at the first read after zero.
    """
    data = _parse(app_state.get_value(_KEY))
    if data is None:
        return None
    if data["ends_at"] <= _now_str():
        app_state.delete_value(_KEY)
        return None
    return _decorate(data)


def start(duration_min: int = DEFAULT_BREAK_MIN, *, source: str = "manual") -> Dict:
    """Start (or restart) a break. Any active focus session is completed first."""
    duration_min = int(duration_min)
    if not (MIN_BREAK_MIN <= duration_min <= MAX_BREAK_MIN):
        raise ValueError(f"break must be {MIN_BREAK_MIN}-{MAX_BREAK_MIN} minutes")

    # Starting a break mid-session means the session is over. Import locally:
    # intentions calls back into this module on session end, and a top-level
    # import either way would be circular.
    from monitor import intentions

    if source == "manual":
        try:
            intentions.end_active("completed", fire_hook=False)
        except Exception as e:
            print(f"[break] could not close the active session ({e}); starting anyway")

    now = datetime.now()
    data = {
        "started_at": now.strftime(_TS_FMT),
        "ends_at": (now + timedelta(minutes=duration_min)).strftime(_TS_FMT),
        "duration_min": duration_min,
        "source": source,
    }
    app_state.set_value(_KEY, json.dumps(data))
    print(f"[break] {duration_min}m break started ({source}) — monitoring paused until {data['ends_at'][11:16]}")
    return _decorate(data)


def end() -> Optional[Dict]:
    """End a break early. Returns the break that was cut short, or None."""
    data = get_active()
    if data is None:
        return None
    if app_state.delete_value(_KEY):
        print("[break] break ended early — monitoring resumes")
    return data


def after_session_end(intention: Dict, status: str) -> None:
    """Session-end hook: a finished session rolls straight into a break.

    Completed and expired sessions earn the break; abandoning one does not —
    you gave up on the work, not finished it.
    """
    if status not in ("completed", "expired"):
        return
    if get_active() is not None:
        return
    try:
        start(DEFAULT_BREAK_MIN, source=status)
    except Exception as e:
        print(f"[break] auto-break after session {status} failed ({e})")
