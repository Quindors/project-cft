# monitor/intent_gate.py
"""Sanity-check a declared intention before its session starts.

"For the next hour I intend to scroll twitter" is not a focus session, it is
a loophole: every drift check afterwards would score activity against a
target that is itself off-task, and scrolling would read as on-intent for the
whole hour. So before a session starts, the intention text is classified
against the user's own categories, and an intention whose declared work lands
in an unproductive category is rejected, with the reason.

The same call maps the intention to its home category, stored on the session
(intentions.llm_category). The runner uses that as a weak prior: a decision
that HEDGES on alignment (partial band) while the window sits in the
session's home category counts as on-intent. The prior breaks ties only — it
never rescues a confident off-intent score, because belonging to the right
category is not doing the declared work (the same conflation that once let
"planning a software project" pass as "outlining piq #2").

Fail-open by design: no LLM, no categories table, an API error or an
unrecognized category name all allow the session. The gate exists to catch
obvious loopholes, not to hold a work session hostage to an outage.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Tuple

from monitor.config import Settings, load_settings
from monitor.paths import data_dir


def _load_categories() -> List[Dict]:
    try:
        conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT name, description, is_productive FROM categories"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [{
        "name": r["name"],
        "description": r["description"] or "",
        "is_productive": None if r["is_productive"] is None else bool(r["is_productive"]),
    } for r in rows]


def _parse(raw: str) -> Tuple[Optional[str], str]:
    name, reason = None, ""
    for part in raw.replace("\n", " | ").split("|"):
        # Models decorate ("**CATEGORY=X**", "- CATEGORY: X"); strip the frame.
        part = part.strip().strip("*`").lstrip("-# ").strip()
        if part.upper().startswith("CATEGORY="):
            name = part[len("CATEGORY="):].strip().strip("*`\"'.")
        elif part.upper().startswith("REASON="):
            reason = part[len("REASON="):].strip()
    return name, reason


def _match_category(name: Optional[str], categories: List[Dict]) -> Optional[Dict]:
    """Resolve the model's CATEGORY value to one of the user's categories.

    Exact (case-insensitive) first; else a unique category name contained in
    the value ("Entertainment (watching videos)" -> Entertainment). Ambiguous
    or absent resolves to None, which fails open.
    """
    value = (name or "").strip().lower()
    if not value:
        return None
    for c in categories:
        if c["name"].strip().lower() == value:
            return c
    contained = [c for c in categories if c["name"].strip().lower() in value]
    return contained[0] if len(contained) == 1 else None


def check(text: str, categories: Optional[List[Dict]] = None,
          settings: Optional[Settings] = None) -> Dict:
    """Classify an intention before its session starts.

    Returns {"allowed", "category", "is_productive", "reason"}. allowed is
    False only when the declared work maps to a category the user marked
    unproductive; every failure or ambiguity allows the session.
    """
    from monitor import decider  # late: decider owns the configured LLM client

    if categories is None:
        categories = _load_categories()
    if not categories:
        return {"allowed": True, "category": None, "is_productive": None,
                "reason": "no categories to judge against", "raw": ""}
    if settings is None:
        settings = load_settings()

    lines = []
    for c in categories:
        prod = c.get("is_productive")
        flag = ("productive" if prod is True
                else "UNPRODUCTIVE" if prod is False else "neutral")
        lines.append(f"- {c['name']} ({flag}): {c.get('description') or ''}")

    prompt = f"""A focus-tracking app is about to start a work session. The user declared:

INTENTION: "{text}"

Their categories:
{chr(10).join(lines)}

Which category does the DECLARED WORK itself belong to? Judge the activity the
intention describes, not its phrasing: "research video essays for my film class"
is coursework; "watch some videos to unwind" is entertainment however it is
dressed up. If the intention is vague but plausibly real work, choose the
closest work category.

Answer in exactly this format:
CATEGORY=<exact category name from the list> | REASON=<one short sentence>"""

    try:
        resp = decider._client().chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[gate] intention check skipped ({e})")
        return {"allowed": True, "category": None, "is_productive": None,
                "reason": "check unavailable", "raw": ""}

    name, reason = _parse(raw)
    match = _match_category(name, categories)
    if match is None:
        return {"allowed": True, "category": None, "is_productive": None,
                "reason": reason or "no category match", "raw": raw}

    return {
        "allowed": match["is_productive"] is not False,
        "category": match["name"],
        "is_productive": match["is_productive"],
        "reason": reason,
        "raw": raw,
    }
