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

import os
import re
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


def unknown_category(name: str, *, action: str = "use") -> Problem:
    """The single most common rejection in this app: a label that is not a
    category. Always worth saying where categories come from."""
    shown = str(name).strip() or "that"
    return Problem(
        f"There's no category called “{shown}”, so it can't be {action}d.",
        "Pick one from the list, or add it first on the Categories tab.",
    )


# ---------- the classifier's provider ----------
# The LLM behind every verdict is a hosted service the user pays for, chosen
# by LLM_PROVIDER. When it refuses, the refusal has to be translated into
# THEIR provider's words: an OpenAI user must not be sent to OpenRouter's
# credits page. Anything not listed here gets generic wording that still says
# what to do.

_PROVIDERS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "label": "OpenRouter",
        "key_var": "OPENROUTER_API_KEY",
        "credits": "Add credits at openrouter.ai/settings/credits.",
        "model_example": "openai/gpt-4o-mini",
        "status_page": "status.openrouter.ai",
    },
    "openai": {
        "label": "OpenAI",
        "key_var": "OPENAI_API_KEY",
        "credits": "Add credit at platform.openai.com/settings/organization/billing.",
        "model_example": "gpt-4o-mini",
        "status_page": "status.openai.com",
    },
}

_STATUS_CODE = re.compile(r"\b(?:Error code|status(?: code)?):?\s*(\d{3})\b", re.I)
_QUOTED_MESSAGE = re.compile(r"""['"]message['"]\s*:\s*['"]([^'"]{1,240})['"]""")
_NETWORK_WORDS = ("connection error", "connect", "timed out", "timeout",
                  "getaddrinfo", "name or service", "network is unreachable",
                  "remote end closed", "eof occurred")

# Every fix that ends with the classifier picking itself back up says so:
# the loop retries on every check, so a top-up or a network coming back is
# noticed without a restart — and people should not go restarting things
# that will fix themselves.
_RESUMES = "Classification resumes on its own at the next check; no restart needed."


def _provider_info(provider: Optional[str]) -> Dict[str, str]:
    name = (provider or os.getenv("LLM_PROVIDER", "openrouter") or "").lower()
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    label = name.capitalize() if name else "The AI provider"
    return {
        "label": label,
        "key_var": "the API key",
        "credits": f"Top up your {label} account.",
        "model_example": "",
        "status_page": "",
    }


def _status_code_of(err: object) -> Optional[int]:
    code = getattr(err, "status_code", None)
    if isinstance(code, int):
        return code
    m = _STATUS_CODE.search(str(err))
    return int(m.group(1)) if m else None


def _provider_said(err: object) -> str:
    """The provider's own sentence, dug out of the SDK's repr of its JSON."""
    m = _QUOTED_MESSAGE.search(str(err))
    if m:
        return " ".join(m.group(1).split())
    return " ".join(str(err).split())[:160]


def llm(err: object, consequence: str = "", *, provider: Optional[str] = None,
        model: str = "") -> Problem:
    """A classifier (or screen-check) failure as a Problem, in the words of
    the provider the user actually set up.

    `err` is an SDK exception or the string the decider stored in its
    [ERROR] verdict's reason. `consequence` is what the failure means for
    this app, appended to the refusal — "Nothing is being classified or
    logged until then." — because the same 402 means something different
    for the title call than for the screen check.
    """
    info = _provider_info(provider)
    label = info["label"]
    text = str(err)
    lower = text.lower()
    code = _status_code_of(err)
    said = _provider_said(err)

    def out(message: str, fix: str) -> Problem:
        return Problem(f"{message} {consequence}".strip(), fix)

    if text.strip() == "[PARSE_ERROR]" or "parse_error" in lower:
        return out(
            "The model answered in a shape the app couldn't read.",
            "That is usually a one-off. If it keeps happening, the model named "
            "by LLM_MODEL in the .env next to the app isn't following the "
            "format" + (f"; {info['model_example']} does." if info["model_example"]
                        else "."))

    if code == 402 or "insufficient credits" in lower or "insufficient_quota" in lower \
            or "exceeded your current quota" in lower:
        return out(f"{label} has run out of credits.",
                   f"{info['credits']} {_RESUMES}")

    if code in (401, 403) or "invalid api key" in lower or "incorrect api key" in lower:
        return out(
            f"{label} rejected the API key.",
            f"Check {info['key_var']} in the .env next to the app — it may have "
            f"been revoked, or pasted with a character missing — then restart "
            f"the monitor from the tray.")

    if code == 404 or "model not found" in lower or "does not exist" in lower:
        which = f"“{model}”" if model else "the model in LLM_MODEL"
        example = (f" ({info['model_example']} works)" if info["model_example"] else "")
        return out(
            f"{label} doesn't offer {which}.",
            f"Set LLM_MODEL in the .env next to the app to a model {label} "
            f"lists{example}, then restart the monitor from the tray.")

    if code == 429:
        return out(
            f"{label} is rate-limiting this app.",
            f"It said: {said}. This normally clears by itself within a minute. "
            f"{_RESUMES}")

    if code is not None and code >= 500:
        where = f" Check {info['status_page']} if it lasts." if info["status_page"] else ""
        return out(f"{label} is having an outage (it answered {code}).",
                   f"Nothing to fix on this PC.{where} {_RESUMES}")

    if code is None and any(w in lower for w in _NETWORK_WORDS):
        return out(f"Can't reach {label}.",
                   f"Check this PC's internet connection. {_RESUMES}")

    return out(f"{label} returned an error the app doesn't recognise.",
               f"It said: {said}. {LOOK_IN_THE_LOG}")


def problem_for_status(err: Optional[object]) -> Optional[Dict[str, str]]:
    """{message, fix} for a status payload, or None when there is no error.

    Status endpoints report a *current* problem rather than a rejected
    request, so the key is `message`, matching what the cards already read.
    """
    if err is None:
        return None
    problem = of(err)
    return {"message": problem.message, "fix": problem.fix}
