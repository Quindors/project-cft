# monitor/email_report.py
"""A weekly focus report, by email, to whoever you name.

Once a week the monitor totals the last seven days — productive hours, the
productivity rate, the split by category, each day's line, and the focus
sessions declared and finished — and mails it to the addresses in the .env.
That can be you, a parent, a study partner, a coach: anyone you want reading
the same numbers the dashboard and Beeminder see. It is opt-in, and off until
the .env names an SMTP server and at least one recipient.

The numbers are not computed here. Every hour comes out of
monitor/beeminder.py (label_rows/bucket): a human correction wins over the
classifier, off-intent work during a session counts as neither (real work,
not the declared work), a neutral category is productive only while
declared. So the figure in the inbox is the figure
that was pushed to the goal, and relabelling a day in the review tab changes
what next week's mail says about it.

Scheduling is catch-up based, because this exe is not always running at
nine on a Monday. A background thread wakes every few minutes, works out the
most recent send time that has passed (REPORT_DAY / REPORT_HOUR, local) and,
if no report has gone out for that slot yet, sends the week that ended the
day before it. A machine asleep through the slot sends when it wakes; one
off for a fortnight sends once, for the latest week. The "already sent"
marker lives in the app_state table, so a restart never re-sends — and
switching the report on stamps that marker without sending, so setting it up
on a Wednesday waits for Monday rather than mailing last week immediately.

Everything is best-effort and off the critical path: a failed send prints,
records itself for the dashboard card, and is retried an hour later. Nothing
here ever writes to focus_logs or intentions.
"""
from __future__ import annotations

import html
import json
import os
import re
import smtplib
import sqlite3
import ssl
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Dict, List, Optional, Tuple

from monitor import app_state, beeminder, problems
from monitor.problems import Problem, UserError
from monitor.config import DEFAULT_SETTINGS
from monitor.paths import data_dir

_TS_FMT = "%Y-%m-%d %H:%M:%S"

# Indexable by datetime.weekday(): Monday is 0.
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")

# Where the "last report sent" marker lives (JSON in app_state).
MARKER_KEY = "email_report_last_sent"

# How often the scheduler looks at the clock, and how long a failed send
# waits before another attempt — an SMTP server that just said no is not
# going to say yes ten seconds later.
TICK_SEC = 15 * 60.0
FIRST_TICK_SEC = 60.0
RETRY_AFTER_SEC = 60 * 60.0

MAX_RECIPIENTS = 20
# Deliberately loose: it catches the typo, not the RFC.
_ADDR_RE = re.compile(r"^[^@\s<>,;\"']+@[^@\s<>,;\"']+\.[^@\s<>,;\"']+$")

# A session counts toward Beeminder's sessions goal only past this much
# logged activity; the report uses the same floor to say how many did.
SESSION_MIN_LOGGED_SEC = beeminder.SESSION_MIN_LOGGED_SEC
# ALIGN at or above this is not off-intent — the same line beeminder.py and
# the drift alerts use.
OFF_INTENT_MAX = beeminder.OFF_INTENT_MAX

# The sender's smtplib entry points, held here so a test can swap them for a
# recorder without touching the stdlib.
_SMTP = smtplib.SMTP
_SMTP_SSL = smtplib.SMTP_SSL


# ------------------- configuration -------------------

@dataclass(frozen=True)
class Config:
    to: Tuple[str, ...]
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender: str
    security: str          # starttls | ssl | none
    day: int               # 0 = Monday ... 6 = Sunday
    hour: int              # 0-23, local time
    timeout_sec: float
    seconds_per_row: float
    strict_intent: bool


def parse_recipients(raw) -> List[str]:
    """Addresses out of a string ("a@x.org, b@y.org") or a list. Deduplicated
    in order, case-insensitively. Raises ValueError naming the first bad one."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,;\s]+", raw)
    else:
        parts = []
        for item in raw:
            parts.extend(re.split(r"[,;\s]+", str(item or "")))
    out: List[str] = []
    seen = set()
    for part in parts:
        addr = part.strip()
        if not addr:
            continue
        if not _ADDR_RE.match(addr):
            raise UserError(
                f"\u201c{addr}\u201d doesn't look like an email address.",
                "Check it for a missing @ or a missing .com. Separate several "
                "addresses with commas.")
        if addr.lower() in seen:
            continue
        seen.add(addr.lower())
        out.append(addr)
    if len(out) > MAX_RECIPIENTS:
        raise UserError(
            f"That's {len(out)} recipients, and the report sends to at most "
            f"{MAX_RECIPIENTS}.",
            "Trim the list. If a whole group needs it, point the report at one "
            "mailing-list address instead of listing everyone.")
    return out


# Providers print app passwords in groups of four so they can be read aloud:
# Google's "abcd efgh ijkl mnop" is really sixteen characters. The grouping is
# decoration, and nothing on the page that hands it to you says so - so the
# spaces come off here rather than being a rule the user has to know.
#
# Deliberately narrow. Only evenly-grouped alphanumerics are unspaced; a real
# passphrase with spaces in it ("correct horse battery staple") has uneven
# words and is left exactly as typed, because for a custom mail server those
# spaces may well be part of the secret.
_GROUPED_SECRET = re.compile(r"^[A-Za-z0-9]{4}(?: [A-Za-z0-9]{4})+$")


def parse_app_password(raw) -> str:
    """An app password as the provider printed it, as the server wants it."""
    value = str(raw if raw is not None else "").strip()
    # Tabs and non-breaking spaces ride along with a copy-paste too.
    collapsed = re.sub(r"\s+", " ", value)
    if _GROUPED_SECRET.match(collapsed):
        return collapsed.replace(" ", "")
    return value


def parse_sender(raw, *, field: str = "Send from") -> str:
    """The From address, checked the same way recipients are.

    The SMTP login never looks at this value, so an unchecked one passes
    every setup check and fails at the first real send - after the settings
    have been written. Checking it here keeps the failure where the typo is.
    """
    addr = str(raw if raw is not None else "").strip()
    if not addr:
        return ""
    if not _ADDR_RE.match(addr):
        raise UserError(
            f"\u201c{addr}\u201d isn't an email address, so the report has "
            f"nowhere to send from.",
            f"Put the address the mail should come from in \u201c{field}\u201d "
            f"\u2014 usually the same one you log in to the mail server with.")
    return addr


def parse_day(raw) -> int:
    """'mon' / 'Monday' / 0 -> 0. Raises ValueError on anything else."""
    s = str(raw if raw is not None else "").strip().lower()
    if not s:
        return 0
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 6:
            return n
        raise UserError(f"There's no day {n} in the week.",
                        "Use a name like \u201cmonday\u201d, or a number from "
                        "0 (Monday) to 6 (Sunday).")
    for i, name in enumerate(DAYS):
        if s[:3] == name:
            return i
    raise UserError(f"\u201c{raw}\u201d isn't a day of the week.",
                    "Use a name like \u201cmonday\u201d, or a number from "
                    "0 (Monday) to 6 (Sunday).")


def parse_hour(raw) -> int:
    s = str(raw if raw is not None else "").strip()
    if not s:
        return 9
    try:
        n = int(float(s))
    except ValueError:
        raise UserError(f"\u201c{raw}\u201d isn't a time of day.",
                        "Use a whole number from 0 to 23 \u2014 9 means 9am, "
                        "21 means 9pm.")
    if not 0 <= n <= 23:
        raise UserError(f"There's no {n} o'clock.",
                        "Use a whole number from 0 to 23 \u2014 9 means 9am, "
                        "21 means 9pm.")
    return n


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[report] {name}: \u201c{raw}\u201d isn't a number. "
              f"Using {default} instead.")
        return default


def _lenient(parser, raw, default, label):
    """Parse a .env value, or fall back loudly. A hand-edited .env must never
    stop the monitor booting, but it must never fail silently either."""
    try:
        return parser(raw)
    except ValueError as e:
        print(f"[report] {label}: {problems.of(e).message} Using {default!r} instead.")
        return default


def config() -> Config:
    """Read the report settings from the environment — on every call, so a
    .env edit or a save from the dashboard card takes effect at the next
    tick without a restart."""
    try:
        port = int(_env("REPORT_SMTP_PORT") or "587")
    except ValueError:
        print(f"[report] REPORT_SMTP_PORT: \u201c{_env('REPORT_SMTP_PORT')}\u201d "
              f"isn't a port number. Using 587 instead.")
        port = 587
    security = _env("REPORT_SMTP_SECURITY").lower()
    if security not in ("starttls", "ssl", "none"):
        security = "ssl" if port == 465 else "starttls"
    user = _env("REPORT_SMTP_USER")
    return Config(
        to=tuple(_lenient(parse_recipients, _env("REPORT_EMAIL_TO"), [], "REPORT_EMAIL_TO")),
        smtp_host=_env("REPORT_SMTP_HOST"),
        smtp_port=port,
        smtp_user=user,
        smtp_password=parse_app_password(_env("REPORT_SMTP_PASSWORD")),
        # The From address: explicit, else the login (what Gmail will stamp
        # on it anyway). A hand-edited .env with a bad one is reported and
        # dropped rather than carried into every send.
        sender=_lenient(parse_sender, _env("REPORT_EMAIL_FROM"), "",
                        "REPORT_EMAIL_FROM") or user,
        security=security,
        day=_lenient(parse_day, _env("REPORT_DAY"), 0, "REPORT_DAY"),
        hour=_lenient(parse_hour, _env("REPORT_HOUR"), 9, "REPORT_HOUR"),
        timeout_sec=_env_float("REPORT_TIMEOUT_SEC", 30.0),
        # The same bridge from rows to hours Beeminder uses, read from the
        # same knob, so the two can never disagree about the length of a row.
        seconds_per_row=_env_float("BEEMINDER_SECONDS_PER_ROW",
                                   DEFAULT_SETTINGS.decision_interval),
        strict_intent=_env("BEEMINDER_STRICT_INTENT", "1").lower() in ("1", "true", "yes"),
    )


def enabled(cfg: Optional[Config] = None) -> bool:
    cfg = cfg or config()
    return bool(cfg.to and cfg.smtp_host and cfg.sender)


# ------------------- the numbers -------------------

@dataclass
class DayStats:
    day: date
    logged_sec: float = 0.0
    productive_sec: float = 0.0
    offtask_sec: float = 0.0
    drifted_sec: float = 0.0

    @property
    def pct(self) -> Optional[int]:
        """Productive time over ALL logged time — System and Ambiguous rows
        included in the denominator, the way the dashboard's Today card
        counts it. None when nothing was logged."""
        return _pct(self.productive_sec, self.logged_sec)


@dataclass
class SessionStats:
    id: int
    text: str
    category: Optional[str]
    started_at: str
    ended_at: Optional[str]
    status: str                 # completed | expired | abandoned
    duration_sec: float         # wall-clock, start to end
    logged_sec: float           # activity actually logged under it
    on_intent_pct: Optional[int]  # share of scored rows not off-intent

    @property
    def finished(self) -> bool:
        return self.status in beeminder.COUNTED_SESSION_STATUSES

    @property
    def counted(self) -> bool:
        """Would Beeminder's sessions goal have taken it?"""
        return self.finished and self.logged_sec >= SESSION_MIN_LOGGED_SEC


@dataclass
class Report:
    start: date                 # first day, inclusive
    end: date                   # day after the last, exclusive
    days: List[DayStats] = field(default_factory=list)
    # (label, seconds, declared) — productive time by category, largest first.
    productive_by_category: List[Tuple[str, float, bool]] = field(default_factory=list)
    offtask_by_category: List[Tuple[str, float]] = field(default_factory=list)
    sessions: List[SessionStats] = field(default_factory=list)
    prev_logged_sec: float = 0.0
    prev_productive_sec: float = 0.0
    generated_at: str = ""

    @property
    def logged_sec(self) -> float:
        return sum(d.logged_sec for d in self.days)

    @property
    def productive_sec(self) -> float:
        return sum(d.productive_sec for d in self.days)

    @property
    def offtask_sec(self) -> float:
        return sum(d.offtask_sec for d in self.days)

    @property
    def drifted_sec(self) -> float:
        return sum(d.drifted_sec for d in self.days)

    @property
    def pct(self) -> Optional[int]:
        return _pct(self.productive_sec, self.logged_sec)

    @property
    def prev_pct(self) -> Optional[int]:
        return _pct(self.prev_productive_sec, self.prev_logged_sec)

    @property
    def active_days(self) -> int:
        return sum(1 for d in self.days if d.logged_sec > 0)

    @property
    def finished_sessions(self) -> List[SessionStats]:
        return [s for s in self.sessions if s.finished]

    @property
    def abandoned_sessions(self) -> int:
        return sum(1 for s in self.sessions if not s.finished)

    @property
    def counted_sessions(self) -> int:
        return sum(1 for s in self.sessions if s.counted)

    @property
    def session_logged_sec(self) -> float:
        return sum(s.logged_sec for s in self.finished_sessions)

    @property
    def last_day(self) -> date:
        return self.end - timedelta(days=1)


def _pct(part: float, whole: float) -> Optional[int]:
    if whole <= 0:
        return None
    return int(round(100.0 * part / whole))


def _stamp(d: date) -> str:
    return f"{d:%Y-%m-%d} 00:00:00"


def _rows(start: date, end: date):
    """label_rows over [start, end); an unreadable database is an empty week."""
    try:
        return beeminder.label_rows(_stamp(start), _stamp(end))
    except sqlite3.Error as e:
        print(f"[report] Couldn't read your activity history, so this week "
              f"reports as empty. ({e})")
        return []


def _totals(start: date, end: date, cfg: Config) -> DayStats:
    stats = DayStats(day=start)
    for row in _rows(start, end):
        seconds = row["n"] * cfg.seconds_per_row
        stats.logged_sec += seconds
        kind = beeminder.bucket(row, strict_intent=cfg.strict_intent)
        if kind == beeminder.PRODUCTIVE:
            stats.productive_sec += seconds
        elif kind == beeminder.DRIFTED:
            stats.drifted_sec += seconds
        elif kind == beeminder.OFFTASK:
            stats.offtask_sec += seconds
    return stats


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{data_dir() / 'focus.db'}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _sessions(start: date, end: date, cfg: Config) -> List[SessionStats]:
    """Sessions that STARTED in the period, with what was logged under them.
    Superseded sessions were replaced, not ended, and are left out."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, text, started_at, ended_at, status, llm_category "
                "FROM intentions WHERE started_at >= ? AND started_at < ? "
                "  AND status IN ('completed', 'expired', 'abandoned') "
                "ORDER BY started_at",
                (_stamp(start), _stamp(end)),
            ).fetchall()
            counts: Dict[int, sqlite3.Row] = {}
            if rows:
                marks = ",".join("?" * len(rows))
                from monitor.intent_review import align_sql
                align = align_sql(conn)  # a Review-tab ruling wins
                for c in conn.execute(
                    "SELECT intention_id, COUNT(*) AS n, "
                    f"       SUM({align} IS NOT NULL) AS scored, "
                    f"       SUM(CASE WHEN {align} >= ? THEN 1 ELSE 0 END) AS on_intent "
                    f"FROM focus_logs WHERE intention_id IN ({marks}) "
                    "GROUP BY intention_id",
                    [OFF_INTENT_MAX] + [r["id"] for r in rows],
                ):
                    counts[c["intention_id"]] = c
    except sqlite3.Error as e:
        # No intentions table yet: a week without sessions, not an error.
        if "no such table" not in str(e):
            print(f"[report] Couldn't read your focus sessions, so the report "
                  f"leaves them out. ({e})")
        return []

    out = []
    for r in rows:
        c = counts.get(r["id"])
        n = c["n"] if c else 0
        scored = (c["scored"] or 0) if c else 0
        on_intent = (c["on_intent"] or 0) if c else 0
        out.append(SessionStats(
            id=r["id"],
            text=" ".join(str(r["text"] or "").split()),
            category=r["llm_category"],
            started_at=r["started_at"],
            ended_at=r["ended_at"],
            status=r["status"],
            duration_sec=_span_sec(r["started_at"], r["ended_at"]),
            logged_sec=n * cfg.seconds_per_row,
            on_intent_pct=_pct(on_intent, scored) if scored else None,
        ))
    return out


def _span_sec(start: Optional[str], end: Optional[str]) -> float:
    try:
        a = datetime.strptime(start or "", _TS_FMT)
        b = datetime.strptime(end or "", _TS_FMT)
    except ValueError:
        return 0.0
    return max(0.0, (b - a).total_seconds())


def build_report(period_end: date, *, days: int = 7, cfg: Optional[Config] = None) -> Report:
    """Everything the mail says about the `days` days ending before
    `period_end` (exclusive), plus the period before it for comparison."""
    cfg = cfg or config()
    start = period_end - timedelta(days=days)
    report = Report(start=start, end=period_end,
                    generated_at=datetime.now().strftime(_TS_FMT))

    for i in range(days):
        d = start + timedelta(days=i)
        report.days.append(_totals(d, d + timedelta(days=1), cfg))

    productive: Dict[Tuple[str, bool], float] = {}
    offtask: Dict[str, float] = {}
    for row in _rows(start, period_end):
        seconds = row["n"] * cfg.seconds_per_row
        kind = beeminder.bucket(row, strict_intent=cfg.strict_intent)
        label = str(row["label"] or "Unknown")
        if kind == beeminder.PRODUCTIVE:
            key = (label, bool(row["declared"]))
            productive[key] = productive.get(key, 0.0) + seconds
        elif kind == beeminder.OFFTASK:
            offtask[label] = offtask.get(label, 0.0) + seconds
    report.productive_by_category = sorted(
        ((label, sec, declared) for (label, declared), sec in productive.items()),
        key=lambda t: -t[1])
    report.offtask_by_category = sorted(offtask.items(), key=lambda t: -t[1])

    report.sessions = _sessions(start, period_end, cfg)

    prev = _totals(start - timedelta(days=days), start, cfg)
    report.prev_logged_sec = prev.logged_sec
    report.prev_productive_sec = prev.productive_sec
    return report


# ------------------- rendering -------------------

def _hm(seconds: float) -> str:
    """'2h 05m' / '35m' / '0m'."""
    minutes = int(round(seconds / 60.0))
    h, m = divmod(minutes, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def _range_label(report: Report) -> str:
    a, b = report.start, report.last_day
    if a.month == b.month:
        return f"{a:%b} {a.day}–{b.day}"
    return f"{a:%b} {a.day} – {b:%b} {b.day}"


def _pct_label(pct: Optional[int]) -> str:
    return "—" if pct is None else f"{pct}%"


def _delta_sentence(report: Report) -> str:
    """One sentence against the week before, or an honest 'nothing to
    compare' when that week logged nothing."""
    if report.prev_logged_sec <= 0:
        return "No activity was logged the week before, so there is nothing to compare against."
    dh = (report.productive_sec - report.prev_productive_sec) / 3600.0
    parts = []
    if abs(dh) < 0.05:
        parts.append("the same productive hours as the week before")
    else:
        parts.append(f"{abs(dh):.1f}h {'more' if dh > 0 else 'less'} productive time than the week before "
                     f"({_hours(report.prev_productive_sec)})")
    if report.pct is not None and report.prev_pct is not None:
        dp = report.pct - report.prev_pct
        if dp == 0:
            parts.append("the same rate")
        else:
            parts.append(f"the rate {'up' if dp > 0 else 'down'} {abs(dp)} "
                         f"point{'' if abs(dp) == 1 else 's'} from {report.prev_pct}%")
    return "That is " + " and ".join(parts) + "."


# Where the "these people have been introduced" list lives.
INTRODUCED_KEY = "email_report_introduced"


def unsubscribe_note(cfg: Config) -> str:
    """Why this arrived and how to make it stop, in two sentences.

    It names the sender's address rather than a name because that is the only
    identity this app actually knows - and it is the address a reply reaches,
    which is the whole point.
    """
    who = cfg.sender or "the person who set this up"
    return (f"You are getting this because {who} set up a weekly focus report "
            f"and added your address. Reply to this email and they will stop "
            f"sending it.")


def _unsubscribe_header(cfg: Config) -> str:
    """RFC 2369 List-Unsubscribe, as a mailto.

    Gmail and Apple Mail render their own Unsubscribe control from this. A
    mailto rather than a link on purpose: the monitor is a desktop app that
    is off half the time, so a URL into it would be dead more often than not.
    The reply lands in the sender's inbox, which is always there.
    """
    target = cfg.sender or cfg.smtp_user
    return f"<mailto:{target}?subject=Unsubscribe%20from%20the%20focus%20report>"


def subject(report: Report) -> str:
    return (f"Focus report {_range_label(report)}: {_hours(report.productive_sec)} productive"
            + (f" ({report.pct}%)" if report.pct is not None else ""))


def render_text(report: Report, cfg: Optional[Config] = None) -> str:
    cfg = cfg or config()
    L: List[str] = []
    L.append(f"FOCUS REPORT  {_range_label(report)}, {report.start.year}")
    L.append("=" * 48)
    L.append(f"Productive time   {_hm(report.productive_sec)}")
    L.append(f"Productivity rate {_pct_label(report.pct)} of {_hm(report.logged_sec)} at the screen")
    L.append(f"Off-task time     {_hm(report.offtask_sec)}")
    if report.drifted_sec >= 60:
        L.append(f"Drifted           {_hm(report.drifted_sec)} (real work off the declared intention; counted as neither)")
    L.append(f"Active days       {report.active_days} of {len(report.days)}")
    L.append("")
    L.append(_delta_sentence(report))
    if report.logged_sec <= 0:
        L.append("")
        L.append("No activity was logged this week. Was the monitor running?")
    L.append("")
    L.append("BY DAY")
    L.append(f"{'':<14}{'productive':>11}{'off-task':>10}{'logged':>9}{'rate':>6}")
    for d in report.days:
        L.append(f"{d.day:%a %b %d}".ljust(14)
                 + f"{_hm(d.productive_sec):>11}{_hm(d.offtask_sec):>10}"
                 + f"{_hm(d.logged_sec):>9}{_pct_label(d.pct):>6}")
    if report.productive_by_category:
        L.append("")
        L.append("PRODUCTIVE TIME BY CATEGORY")
        for label, sec, declared in report.productive_by_category[:6]:
            L.append(f"  {label}{' (in session)' if declared else ''}: {_hm(sec)}")
    if report.offtask_by_category:
        L.append("")
        L.append("OFF-TASK TIME BY CATEGORY")
        for label, sec in report.offtask_by_category[:4]:
            L.append(f"  {label}: {_hm(sec)}")
    L.append("")
    L.append("FOCUS SESSIONS")
    fin = report.finished_sessions
    if not report.sessions:
        L.append("  No focus sessions were declared this week.")
    else:
        L.append(f"  {len(fin)} finished ({report.counted_sessions} with 10+ minutes logged), "
                 f"{report.abandoned_sessions} abandoned, {_hm(report.session_logged_sec)} logged in sessions")
        for s in report.sessions[:20]:
            when = s.started_at[:16]
            tag = s.status if not s.finished else ("done" if s.status == "completed" else "ran out")
            oi = "" if s.on_intent_pct is None else f", {s.on_intent_pct}% on-intent"
            cat = f" [{s.category}]" if s.category else ""
            L.append(f"  {when}  {s.text[:60]}{cat} — {_hm(s.duration_sec)}, "
                     f"{_hm(s.logged_sec)} logged{oi} ({tag})")
    L.append("")
    L.append("How this is counted: each logged row is one decision interval at the")
    L.append("screen; a row is productive when its category is, or when it was the")
    L.append("declared work of a session. Corrections made in the review tab count.")
    L.append("")
    L.append("-- ")
    L.append(unsubscribe_note(cfg))
    L.append(f"Sent by the cft monitor at {report.generated_at}.")
    return "\n".join(L) + "\n"


_CSS_CARD = ("background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;"
             "padding:18px 20px;margin:0 0 16px 0;")
_CSS_MUTED = "color:#64748b;font-size:12px;"
_CSS_TH = ("text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;"
           "color:#94a3b8;padding:6px 8px;border-bottom:1px solid #e2e8f0;")
_CSS_TD = "padding:7px 8px;border-bottom:1px solid #f1f5f9;font-size:13px;color:#334155;"


def _bar(pct: Optional[int], color: str = "#10b981") -> str:
    width = max(0, min(100, pct or 0))
    return (f'<div style="background:#f1f5f9;border-radius:4px;height:8px;width:100%;">'
            f'<div style="background:{color};border-radius:4px;height:8px;width:{width}%;"></div></div>')


def render_html(report: Report, cfg: Optional[Config] = None) -> str:
    cfg = cfg or config()
    e = html.escape
    tiles = [
        ("Productive time", _hm(report.productive_sec), "#059669"),
        ("Productivity rate", _pct_label(report.pct), "#0f172a"),
        ("Off-task time", _hm(report.offtask_sec), "#dc2626"),
        ("Focus sessions", str(len(report.finished_sessions)), "#0f172a"),
    ]
    tile_html = "".join(
        f'<td style="padding:0 6px 0 0;vertical-align:top;width:25%;">'
        f'<div style="{_CSS_CARD}margin:0;">'
        f'<div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.06em;">{e(label)}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{color};margin-top:4px;">{e(value)}</div>'
        f'</div></td>'
        for label, value, color in tiles
    )

    day_rows = "".join(
        f"<tr><td style='{_CSS_TD}white-space:nowrap;'>{d.day:%a} <span style='{_CSS_MUTED}'>{d.day:%b %d}</span></td>"
        f"<td style='{_CSS_TD}text-align:right;color:#059669;'>{_hm(d.productive_sec)}</td>"
        f"<td style='{_CSS_TD}text-align:right;color:#dc2626;'>{_hm(d.offtask_sec)}</td>"
        f"<td style='{_CSS_TD}text-align:right;'>{_hm(d.logged_sec)}</td>"
        f"<td style='{_CSS_TD}text-align:right;'>{_pct_label(d.pct)}</td>"
        f"<td style='{_CSS_TD}width:34%;'>{_bar(d.pct)}</td></tr>"
        for d in report.days
    )

    def cat_list(items, color):
        if not items:
            return f"<p style='{_CSS_MUTED}margin:0;'>Nothing logged.</p>"
        total = sum(sec for *_, sec in [(i[0], i[1]) for i in items]) or 1.0
        rows = []
        for item in items[:6]:
            label, sec = item[0], item[1]
            declared = len(item) > 2 and item[2]
            name = e(label) + (" <span style='color:#059669;font-size:11px;'>in session</span>" if declared else "")
            rows.append(
                f"<tr><td style='{_CSS_TD}'>{name}</td>"
                f"<td style='{_CSS_TD}text-align:right;white-space:nowrap;'>{_hm(sec)}</td>"
                f"<td style='{_CSS_TD}width:40%;'>{_bar(int(round(100 * sec / total)), color)}</td></tr>")
        return f"<table style='width:100%;border-collapse:collapse;'>{''.join(rows)}</table>"

    if not report.sessions:
        sessions_html = f"<p style='{_CSS_MUTED}margin:0;'>No focus sessions were declared this week.</p>"
    else:
        srows = []
        for s in report.sessions[:20]:
            if s.finished:
                tag, tag_color = ("done", "#059669") if s.status == "completed" else ("ran out", "#d97706")
            else:
                tag, tag_color = "abandoned", "#94a3b8"
            oi = "—" if s.on_intent_pct is None else f"{s.on_intent_pct}%"
            cat = f" <span style='{_CSS_MUTED}'>{e(s.category)}</span>" if s.category else ""
            srows.append(
                f"<tr><td style='{_CSS_TD}white-space:nowrap;'>{e(s.started_at[5:16])}</td>"
                f"<td style='{_CSS_TD}'>{e(s.text[:80])}{cat}</td>"
                f"<td style='{_CSS_TD}text-align:right;white-space:nowrap;'>{_hm(s.duration_sec)}</td>"
                f"<td style='{_CSS_TD}text-align:right;white-space:nowrap;'>{_hm(s.logged_sec)}</td>"
                f"<td style='{_CSS_TD}text-align:right;'>{oi}</td>"
                f"<td style='{_CSS_TD}color:{tag_color};'>{tag}</td></tr>")
        sessions_html = (
            f"<p style='margin:0 0 10px 0;font-size:13px;color:#334155;'>"
            f"<b>{len(report.finished_sessions)}</b> finished "
            f"(<b>{report.counted_sessions}</b> with 10+ minutes logged), "
            f"<b>{report.abandoned_sessions}</b> abandoned, "
            f"<b>{_hm(report.session_logged_sec)}</b> logged inside sessions.</p>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<tr><th style='{_CSS_TH}'>When</th><th style='{_CSS_TH}'>Session</th>"
            f"<th style='{_CSS_TH}text-align:right;'>Length</th><th style='{_CSS_TH}text-align:right;'>Logged</th>"
            f"<th style='{_CSS_TH}text-align:right;'>On-intent</th><th style='{_CSS_TH}'></th></tr>"
            f"{''.join(srows)}</table>")

    empty_note = ""
    if report.logged_sec <= 0:
        empty_note = (f"<div style='{_CSS_CARD}border-color:#fcd34d;background:#fffbeb;color:#92400e;font-size:13px;'>"
                      f"No activity was logged this week. Was the monitor running?</div>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{e(subject(report))}</title></head>
<body style="margin:0;padding:24px;background:#f8fafc;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a;">
<div style="max-width:680px;margin:0 auto;">
  <div style="margin:0 0 16px 0;">
    <div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.08em;">Weekly focus report</div>
    <h1 style="margin:4px 0 0 0;font-size:22px;">{e(_range_label(report))}, {report.start.year}</h1>
    <p style="margin:8px 0 0 0;font-size:14px;color:#334155;">{e(_delta_sentence(report))}
       Active on {report.active_days} of {len(report.days)} days,
       {e(_hm(report.logged_sec))} at the screen in total.</p>
  </div>
  {empty_note}
  <table style="width:100%;border-collapse:separate;border-spacing:0;margin:0 0 16px 0;"><tr>{tile_html}</tr></table>
  <div style="{_CSS_CARD}">
    <div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">By day</div>
    <table style="width:100%;border-collapse:collapse;">
      <tr><th style="{_CSS_TH}">Day</th><th style="{_CSS_TH}text-align:right;">Productive</th>
          <th style="{_CSS_TH}text-align:right;">Off-task</th><th style="{_CSS_TH}text-align:right;">Logged</th>
          <th style="{_CSS_TH}text-align:right;">Rate</th><th style="{_CSS_TH}"></th></tr>
      {day_rows}
    </table>
  </div>
  <table style="width:100%;border-collapse:separate;border-spacing:0;"><tr>
    <td style="vertical-align:top;width:50%;padding-right:8px;">
      <div style="{_CSS_CARD}"><div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Productive time</div>
      {cat_list(report.productive_by_category, '#10b981')}</div></td>
    <td style="vertical-align:top;width:50%;padding-left:8px;">
      <div style="{_CSS_CARD}"><div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Off-task time</div>
      {cat_list(report.offtask_by_category, '#f87171')}</div></td>
  </tr></table>
  <div style="{_CSS_CARD}">
    <div style="{_CSS_MUTED}text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Focus sessions</div>
    {sessions_html}
  </div>
  <p style="{_CSS_MUTED}line-height:1.5;">How this is counted: each logged row is one decision interval at the screen.
     A row is productive when its category is, or when it was the declared work of a focus session
     (a neutral category counts only while declared); real work done off-intent during a session counts as neither.
     Corrections made in the dashboard's review tab are reflected here.</p>
  <p style="{_CSS_MUTED}line-height:1.5;border-top:1px solid #e2e8f0;padding-top:12px;">
     {e(unsubscribe_note(cfg))}<br>
     Sent by the cft monitor at {e(report.generated_at)}.</p>
</div>
</body></html>
"""


# ------------------- sending -------------------

def _open_smtp(cfg: Config):
    ctx = ssl.create_default_context()
    if cfg.security == "ssl":
        server = _SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_sec, context=ctx)
    else:
        server = _SMTP(cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_sec)
        server.ehlo()
        if cfg.security == "starttls":
            server.starttls(context=ctx)
            server.ehlo()
    if cfg.smtp_user:
        server.login(cfg.smtp_user, cfg.smtp_password)
    return server


# A mail server the user picked from the provider buttons should be named
# the way they picked it. "Gmail wouldn't accept that password" lands;
# "smtp.gmail.com rejected the login" makes them look up what that is.
_PROVIDER_NAMES = {
    "smtp.gmail.com": "Gmail",
    "smtp-mail.outlook.com": "Outlook",
    "smtp.office365.com": "Outlook",
    "smtp.mail.me.com": "iCloud",
    "smtp.mail.yahoo.com": "Yahoo Mail",
}

# Where each provider hands out app passwords. Sending someone to the right
# page is the difference between a fixable error and a dead one.
_APP_PASSWORD_HELP = {
    "Gmail": ("Gmail needs a 16-character app password here, not the password "
              "you sign in with. Make one at myaccount.google.com/apppasswords "
              "\u2014 you'll need 2-Step Verification turned on first."),
    "Outlook": ("Outlook needs an app password here, not your account password. "
                "Make one under Security \u2192 Advanced security options at "
                "account.microsoft.com."),
    "iCloud": ("iCloud needs an app-specific password here, not your Apple ID "
               "password. Make one at account.apple.com under Sign-In and "
               "Security."),
}


def server_name(cfg: Config) -> str:
    """The mail server as the user thinks of it."""
    return _PROVIDER_NAMES.get((cfg.smtp_host or "").lower(),
                               cfg.smtp_host or "the mail server")


def _problem(err: Exception, cfg: Config) -> Problem:
    """What went wrong and what to do about it, fit for the dashboard card.

    Every branch ends somewhere the user can act. The SMTP exception class is
    a decent guess at the cause, so it picks the advice \u2014 but it never
    reaches the screen under its own name.
    """
    who = server_name(cfg)

    if isinstance(err, smtplib.SMTPAuthenticationError):
        return Problem(
            f"{who} wouldn't accept that password.",
            _APP_PASSWORD_HELP.get(who) or
            f"Check the username and password for {who}. Most providers want an "
            f"app-specific password here rather than the one you sign in with.")

    if isinstance(err, smtplib.SMTPRecipientsRefused):
        refused = list(err.recipients) if isinstance(err.recipients, dict) else []
        named = ", ".join(refused[:3]) if refused else "one of the recipients"
        return Problem(
            f"{who} wouldn't deliver to {named}.",
            "Check that address for a typo. Some mail servers also refuse to "
            "send to addresses outside your own domain.")

    if isinstance(err, smtplib.SMTPSenderRefused):
        return Problem(
            f"{who} wouldn't send mail as {cfg.sender}.",
            "The \u201cSend from\u201d address usually has to be the same "
            "account you logged in with, or an alias it owns.")

    if isinstance(err, smtplib.SMTPServerDisconnected):
        return Problem(
            f"{who} hung up before the report was accepted.",
            f"This is almost always the wrong encryption for the port. Try "
            f"STARTTLS on port 587, or SSL/TLS on port 465 \u2014 you have "
            f"{cfg.security.upper()} on {cfg.smtp_port}.")

    if isinstance(err, smtplib.SMTPException):
        said = " ".join(str(err).split())[:160]
        return Problem(
            f"{who} turned the report down.",
            f"It said: {said}. That message comes from {who}, not from cft \u2014 "
            f"if it mentions a sending limit, the next weekly report will go "
            f"out normally." if said else
            f"Nothing usable came back from {who}. {problems.LOOK_IN_THE_LOG}")

    if isinstance(err, TimeoutError):
        return Problem(
            f"{who} didn't answer within {cfg.timeout_sec:.0f} seconds.",
            "Check you're online. A firewall or VPN blocking outgoing mail is "
            "the usual cause when everything else on the machine works.")

    if isinstance(err, OSError):
        return Problem(
            f"Couldn't reach {who} on port {cfg.smtp_port}.",
            "Check the server name and the port. If they look right, a "
            "firewall or VPN on this machine may be blocking outgoing mail.")

    return problems.unexpected(err, "sending the report")


def cfp(cfg: Config) -> str:
    """host:port, for logs. Never for the screen \u2014 see server_name."""
    return f"{cfg.smtp_host}:{cfg.smtp_port}"


def verify_smtp(cfg: Config) -> Optional[Problem]:
    """Connect and log in, send nothing. None when it works, else a Problem
    — the same check the dashboard card runs before saving anything."""
    try:
        server = _open_smtp(cfg)
        try:
            server.quit()
        except Exception:
            pass
        return None
    except Exception as err:
        return _problem(err, cfg)


def compose(report: Report, cfg: Config) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject(report)
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(cfg.to)
    msg["Message-ID"] = make_msgid(domain="cft.local")
    msg["List-Unsubscribe"] = _unsubscribe_header(cfg)
    msg.set_content(render_text(report, cfg))
    msg.add_alternative(render_html(report, cfg), subtype="html")
    return msg


# ------------------- the introduction -------------------

def intro_subject(cfg: Config) -> str:
    return f"{cfg.sender} would like to send you a weekly focus report"


def render_intro_text(cfg: Config) -> str:
    when = f"{DAY_NAMES[cfg.day]}s at {cfg.hour:02d}:00"
    return (
        f"Hello,\n\n"
        f"{cfg.sender} uses an app called cft that keeps track of how their "
        f"time at the computer is spent, and has set it up to send you a "
        f"summary once a week.\n\n"
        f"Each one is a single page: how many hours were productive, what "
        f"share of the time that was, a line for each day, and the focus "
        f"sessions they set themselves. Nothing else from their machine is "
        f"shared, and nothing is asked of you.\n\n"
        f"They go out on {when}, starting with the next one.\n\n"
        f"If you would rather not have them, reply to this email and say so. "
        f"It goes straight to {cfg.sender} \u2014 there is no service in the "
        f"middle and no account to cancel.\n"
    )


def render_intro_html(cfg: Config) -> str:
    e = html.escape
    when = f"{DAY_NAMES[cfg.day]}s at {cfg.hour:02d}:00"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{e(intro_subject(cfg))}</title></head>
<body style="margin:0;padding:24px;background:#f8fafc;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a;">
<div style="max-width:560px;margin:0 auto;">
  <div style="{_CSS_CARD}">
    <p style="margin:0 0 12px 0;font-size:15px;">Hello,</p>
    <p style="margin:0 0 12px 0;font-size:14px;line-height:1.6;color:#334155;">
      <b>{e(cfg.sender)}</b> uses an app called cft that keeps track of how
      their time at the computer is spent, and has set it up to send you a
      summary once a week.
    </p>
    <p style="margin:0 0 12px 0;font-size:14px;line-height:1.6;color:#334155;">
      Each one is a single page: how many hours were productive, what share of
      the time that was, a line for each day, and the focus sessions they set
      themselves. Nothing else from their machine is shared, and nothing is
      asked of you.
    </p>
    <p style="margin:0 0 12px 0;font-size:14px;line-height:1.6;color:#334155;">
      They go out on <b>{e(when)}</b>, starting with the next one.
    </p>
    <p style="margin:0;font-size:14px;line-height:1.6;color:#334155;">
      If you would rather not have them, just reply to this email and say so.
      It goes straight to {e(cfg.sender)} \u2014 there is no service in the
      middle and no account to cancel.
    </p>
  </div>
  <p style="{_CSS_MUTED}line-height:1.5;">Sent by the cft monitor on
     {e(datetime.now().strftime(_TS_FMT))}.</p>
</div>
</body></html>
"""


def compose_intro(cfg: Config, to) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = intro_subject(cfg)
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(to)
    msg["Message-ID"] = make_msgid(domain="cft.local")
    msg["List-Unsubscribe"] = _unsubscribe_header(cfg)
    msg.set_content(render_intro_text(cfg))
    msg.add_alternative(render_intro_html(cfg), subtype="html")
    return msg


def introduced() -> set:
    """Addresses that have already had an introduction, lowercased."""
    try:
        raw = app_state.get_value(INTRODUCED_KEY)
    except sqlite3.Error:
        return set()
    if not raw:
        return set()
    try:
        data = json.loads(raw)
    except ValueError:
        return set()
    return {str(a).lower() for a in data} if isinstance(data, list) else set()


def _mark_introduced(addresses) -> None:
    try:
        app_state.set_value(INTRODUCED_KEY, json.dumps(
            sorted(introduced() | {str(a).lower() for a in addresses})))
    except sqlite3.Error as e:
        print(f"[report] couldn't remember who has been introduced ({e})")


def forget_introductions() -> None:
    try:
        app_state.delete_value(INTRODUCED_KEY)
    except sqlite3.Error:
        pass


def newcomers(cfg: Optional[Config] = None) -> List[str]:
    """Recipients who have never been written to."""
    cfg = cfg or config()
    seen = introduced()
    return [a for a in cfg.to if a.lower() not in seen]


def send_introductions(cfg: Optional[Config] = None) -> Optional[Problem]:
    """Introduce whoever is new, once. Returns None when there is nothing to
    do or it worked, else the problem.

    Sent when an address is added rather than at the first report, so the
    first thing a stranger reads is why they are hearing from anyone at all.
    A failure here is not fatal to the setup: the settings are already saved
    and the weekly report will still go.
    """
    cfg = cfg or config()
    if not enabled(cfg):
        return None
    fresh = newcomers(cfg)
    if not fresh:
        return None
    try:
        server = _open_smtp(cfg)
        try:
            server.send_message(compose_intro(cfg, fresh))
        finally:
            try:
                server.quit()
            except Exception:
                pass
    except Exception as err:
        problem = _problem(err, cfg)
        print(f"[report] could not introduce {', '.join(fresh)}: {problem}")
        return problem
    print(f"[report] introduced {', '.join(fresh)}")
    _mark_introduced(fresh)
    return None


# ------------------- sending the report itself -------------------

def send_report(report: Report, *, cfg: Optional[Config] = None,
                record: bool = True) -> Optional[Problem]:
    """Mail one report to every recipient. Returns None on success, else the
    problem as a sentence. With record=True the outcome is kept for the
    card, and a success stamps the sent-marker so the scheduler will not
    send the same period again."""
    cfg = cfg or config()
    if not enabled(cfg):
        return Problem(
            "The weekly report isn't set up yet, so there's nowhere to send it.",
            "Add at least one recipient and a mail server on the dashboard's "
            "Overview tab.")
    subj = subject(report)
    try:
        server = _open_smtp(cfg)
        try:
            server.send_message(compose(report, cfg))
        finally:
            try:
                server.quit()
            except Exception:
                pass
    except Exception as err:
        problem = _problem(err, cfg)
        print(f"[report] send failed: {problem}")
        if record:
            _note(ok=False, error=problem, to=cfg.to, subject=subj)
        return problem

    print(f"[report] sent \"{subj}\" to {', '.join(cfg.to)}")
    if record:
        _note(ok=True, error=None, to=cfg.to, subject=subj)
        _set_marker(report.end, cfg.to, subj)
    return None


def send_for(period_end: date, *, cfg: Optional[Config] = None) -> Optional[Problem]:
    cfg = cfg or config()
    return send_report(build_report(period_end, cfg=cfg), cfg=cfg)


# ------------------- health + marker (what the card reads) -------------------

_health_lock = threading.Lock()
_health: Dict = {}


def _note(*, ok: bool, error: Optional[str], to, subject: str) -> None:
    with _health_lock:
        _health.update(ok=ok, error=error, to=list(to), subject=subject,
                       at=datetime.now().strftime(_TS_FMT))


def last_attempt() -> Optional[Dict]:
    with _health_lock:
        return dict(_health) if _health else None


def reset_health() -> None:
    global _retry_not_before
    with _health_lock:
        _health.clear()
    _retry_not_before = None


def _set_marker(period_end: date, to, subj: str) -> None:
    app_state.set_value(MARKER_KEY, json.dumps({
        "period_end": period_end.isoformat(),
        "at": datetime.now().strftime(_TS_FMT),
        "to": list(to),
        "subject": subj,
    }))


def ensure_baseline(cfg: Optional[Config] = None, *,
                    now: Optional[datetime] = None) -> None:
    """Settle the slot that has already passed, without sending anything.

    Catch-up is what makes a sleeping machine still get its mail, but on the
    very first run there is nothing to catch up on: without this, configuring
    the report on a Wednesday would mail last week a minute later, which is
    not what "every Monday at nine" promised. So the first time the report is
    switched on, the current slot is marked settled and the first real mail
    goes at the next one. "Send now" is there for anyone who wants it sooner.

    Does nothing once any marker exists — a reconfigure must not erase a real
    send, and a disconnect clears the marker so switching back on baselines
    afresh rather than back-filling the silent weeks.
    """
    cfg = cfg or config()
    if not enabled(cfg) or last_sent() is not None:
        return
    now = now or datetime.now()
    try:
        app_state.set_value(MARKER_KEY, json.dumps({
            "period_end": last_due(now, cfg).date().isoformat(),
            "at": now.strftime(_TS_FMT),
            "baseline": True,
        }))
    except sqlite3.Error as e:
        print(f"[report] Couldn't save the first-run marker, so the first "
              f"report may cover last week rather than this one. ({e})")


def last_sent() -> Optional[Dict]:
    """The marker: which period was last reported, when, to whom."""
    try:
        raw = app_state.get_value(MARKER_KEY)
    except sqlite3.Error:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("period_end") else None


def clear_marker() -> None:
    try:
        app_state.delete_value(MARKER_KEY)
    except sqlite3.Error:
        pass


# ------------------- scheduling -------------------

def next_due(now: datetime, cfg: Config) -> datetime:
    """The first scheduled send time strictly after `now`."""
    candidate = now.replace(hour=cfg.hour, minute=0, second=0, microsecond=0)
    candidate += timedelta(days=(cfg.day - now.weekday()) % 7)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def last_due(now: datetime, cfg: Config) -> datetime:
    """The most recent scheduled send time at or before `now`."""
    return next_due(now, cfg) - timedelta(days=7)


def pending_period_end(now: datetime, cfg: Config) -> Optional[date]:
    """The period_end owed right now, or None if that slot was already sent.
    A manual send covering the same or a later period also settles it."""
    due = last_due(now, cfg).date()
    marker = last_sent()
    if marker and marker["period_end"] >= due.isoformat():
        return None
    return due


_retry_not_before: Optional[float] = None


def check_and_send(now: Optional[datetime] = None, cfg: Optional[Config] = None) -> Optional[str]:
    """One scheduler tick. Returns 'sent', None (nothing to do), or the
    problem sentence when a due report could not be sent."""
    global _retry_not_before
    import time as _time
    cfg = cfg or config()
    if not enabled(cfg):
        return None
    now = now or datetime.now()
    period_end = pending_period_end(now, cfg)
    if period_end is None:
        return None
    if _retry_not_before is not None and _time.time() < _retry_not_before:
        return None
    problem = send_for(period_end, cfg=cfg)
    if problem:
        _retry_not_before = _time.time() + RETRY_AFTER_SEC
        return problem
    _retry_not_before = None
    return "sent"


def send_now() -> threading.Thread:
    """Mail the last seven full days (ending yesterday) off the calling
    thread — the card's "Send now". It stamps the marker like a scheduled
    send, so a scheduled slot covering the same period is not sent twice."""
    def worker() -> None:
        try:
            send_for(date.today())
        except Exception as e:
            problem = problems.unexpected(e, "sending the report")
            print(f"[report] {problem}")
            _note(ok=False, error=problem, to=config().to, subject="")

    thread = threading.Thread(target=worker, daemon=True, name="email-report-now")
    thread.start()
    return thread


def status() -> Dict:
    """What the dashboard card needs — never the password."""
    cfg = config()
    now = datetime.now()
    on = enabled(cfg)
    attempt = last_attempt()
    marker = last_sent()
    if marker and marker.get("baseline"):
        marker = None   # the slot is settled, but nothing was ever mailed
    pending = pending_period_end(now, cfg) if on else None
    problem = None
    if attempt and attempt.get("ok") is False:
        problem = problems.problem_for_status(attempt.get("error"))
    return {
        "configured": on,
        "recipients": list(cfg.to),
        "smtp_host": cfg.smtp_host or None,
        "smtp_port": cfg.smtp_port,
        "smtp_user": cfg.smtp_user or None,
        "security": cfg.security,
        "sender": cfg.sender or None,
        "day": DAYS[cfg.day],
        "day_name": DAY_NAMES[cfg.day],
        "hour": cfg.hour,
        "next_send": (last_due(now, cfg) if pending else next_due(now, cfg)).strftime(_TS_FMT) if on else None,
        "pending": pending is not None,
        "last_sent": marker,
        "last_attempt": attempt,
        "scheduler_running": _thread is not None and _thread.is_alive(),
        "problem": problem,
    }


# One scheduler per process; the stop event is registered at boot so the
# settings endpoint can start the loop later with the same shutdown wiring.
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None


def start_background_scheduler(stop_event: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """Run the weekly tick until the registered stop event fires. Idempotent;
    None when nothing is configured (the card can call again after saving)."""
    global _thread, _stop_event
    with _lock:
        if stop_event is not None:
            _stop_event = stop_event
        cfg = config()
        if not enabled(cfg):
            return None
        # Before the alive check: re-enabling after a disconnect leaves the
        # loop already running, and that path needs a baseline too.
        ensure_baseline(cfg)
        stop = _stop_event
        if stop is None:
            return None
        if _thread is not None and _thread.is_alive():
            return _thread

        def loop() -> None:
            # Let the boot finish before the first look at the clock.
            if stop.wait(FIRST_TICK_SEC):
                return
            while True:
                try:
                    outcome = check_and_send()
                    if outcome and outcome != "sent":
                        print(f"[report] will retry in {RETRY_AFTER_SEC / 60:.0f}m")
                except Exception as e:
                    print(f"[report] {problems.unexpected(e, 'checking whether a report is due')}")
                if stop.wait(TICK_SEC):
                    return

        _thread = threading.Thread(target=loop, daemon=True, name="email-report")
        _thread.start()
        print(f"[report] weekly email every {DAY_NAMES[cfg.day]} {cfg.hour:02d}:00 "
              f"to {', '.join(cfg.to)} via {cfp(cfg)}")
        return _thread
