# monitor/problems.py
"""Errors written for the person reading them, not the person who wrote them.

Every failure a user can see is two sentences: **what happened**, in words
that name things they chose rather than things the code calls them, and
**what to do about it**. A message with no second half is a dead end — it
tells someone their evening is broken and leaves them to guess.

    Problem("Gmail wouldn't accept that password.",
            "Gmail needs a 16-character app password, not the password you "
            "sign in with. Make one at myaccount.google.com/apppasswords.")

`Problem` IS a string — the two halves joined — so every existing caller
keeps working: printing it, logging it, `"app password" in problem`,
`json.dumps`. The halves stay reachable so the dashboard can set the fix in
its own muted line instead of running the two together into a paragraph.

Three rules for the wording, learned from every error message that has ever
made someone close the tab:

  1. **Name what they typed, not what the field is called.** "The report
     needs to know which mail server to send through", never "smtp_host is
     required". Nobody chose the variable name.
  2. **Say who is refusing.** "Gmail wouldn't accept that password" puts the
     refusal where it belongs; "SMTPAuthenticationError" blames the void.
  3. **The fix is an action, not a diagnosis.** "Use 587 for STARTTLS or 465
     for SSL/TLS" beats "invalid port for the selected security mode".

Raw exception text is never the message. When there is genuinely nothing to
suggest, the fix says where to look — cft.log — which is still an action.
"""
from __future__ import annotations

from typing import Dict, Optional


class Problem(str):
    """One sentence of what went wrong, one of what to do about it.

    A str subclass whose value is the two joined by a space, so it drops into
    anything that wanted a plain message. `.message` and `.fix` keep them
    apart for callers that can lay them out properly.
    """

    message: str
    fix: str

    def __new__(cls, message: str, fix: str = "") -> "Problem":
        message = " ".join(str(message).split())
        fix = " ".join(str(fix or "").split())
        self = super().__new__(cls, f"{message} {fix}".strip())
        self.message = message
        self.fix = fix
        return self

    def as_json(self) -> Dict[str, str]:
        """The shape every API error body uses: {error, fix}."""
        return {"error": self.message, "fix": self.fix}


class UserError(ValueError):
    """A ValueError that carries a Problem, for validation that rejects input.

    Still a ValueError, so existing `except ValueError` handlers and tests
    that match on the text keep working unchanged.
    """

    def __init__(self, message: str, fix: str = ""):
        self.problem = Problem(message, fix)
        super().__init__(self.problem)


def of(err: object) -> Problem:
    """Whatever `err` is — a Problem, a UserError, a bare exception, a
    string — as a Problem, so a caller never has to care which it got."""
    if isinstance(err, Problem):
        return err
    problem = getattr(err, "problem", None)
    if isinstance(problem, Problem):
        return problem
    if isinstance(err, Exception):
        args = getattr(err, "args", ())
        if args and isinstance(args[0], Problem):
            return args[0]
        return Problem(str(err) or err.__class__.__name__)
    return Problem(str(err))


def body(err: object, fallback_fix: str = "") -> Dict[str, str]:
    """An API error body for anything raisable: {error, fix}."""
    problem = of(err)
    if not problem.fix and fallback_fix:
        return Problem(problem.message, fallback_fix).as_json()
    return problem.as_json()


# The fix of last resort. Used where the cause is genuinely unknown — an
# unexpected exception escaping an endpoint — because "where to look" is
# still something a person can act on, and "Internal Server Error" is not.
LOOK_IN_THE_LOG = ("If it keeps happening, cft.log next to the app has the "
                   "details — restarting the monitor from the tray clears "
                   "most one-off failures.")


def unexpected(err: Exception, doing: str) -> Problem:
    """For an exception nobody predicted: say what was being attempted, in
    plain words, and where the details are. `doing` completes "Something went
    wrong while ..." — e.g. "loading your categories"."""
    detail = " ".join(str(err).split())[:160]
    message = f"Something went wrong while {doing}."
    if detail:
        return Problem(message, f"The app reported: {detail}. {LOOK_IN_THE_LOG}")
    return Problem(message, LOOK_IN_THE_LOG)


def missing(field_label: str, fix: str) -> Problem:
    """A required value nobody filled in, named the way the UI names it."""
    return Problem(f"{field_label} is missing.", fix)


def unknown_category(name: str, *, action: str = "use") -> Problem:
    """The single most common rejection in this app: a label that is not a
    category. Always worth saying where categories come from."""
    shown = str(name).strip() or "that"
    return Problem(
        f"There's no category called “{shown}”, so it can't be {action}d.",
        "Pick one from the list, or add it first on the Categories tab.",
    )


def problem_for_status(err: Optional[object]) -> Optional[Dict[str, str]]:
    """{message, fix} for a status payload, or None when there is no error.

    Status endpoints report a *current* problem rather than a rejected
    request, so the key is `message`, matching what the cards already read.
    """
    if err is None:
        return None
    problem = of(err)
    return {"message": problem.message, "fix": problem.fix}
