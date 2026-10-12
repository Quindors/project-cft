# monitor/decider.py
from __future__ import annotations

import re, time, openai
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.screenshot_capture import take_screenshot_b64
from monitor.window_capture import get_active_window

import os, base64

# Set by configure_llm() at boot. decider uses this instead of module-level openai.*
_llm_client = None

def _detect_image_mime(b64: str) -> str:
    """Sniff the image format from the leading bytes of a base64 string."""
    try:
        head = base64.b64decode(b64[:24])
    except Exception:
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"  # safe default

def set_llm_client(client) -> None:
    global _llm_client
    _llm_client = client


def _client():
    if _llm_client is None:
        # Fall back to module-level openai (works if api_key is set the old way)
        return openai
    return _llm_client

def _load_trace_screenshot(events_context, settings) -> Optional[str]:
    """Look up a screenshot fixture matching the current trace event timestamp.

    Returns base64-encoded JPEG, or None if no fixture exists for this event.
    Expected layout: tests/traces/screenshots/YYYY-MM-DD_HH-MM-SS.{jpg|jpeg|png}
    """
    if not events_context:
        return None

    ts_str = events_context[-1][0]  # "YYYY-MM-DD HH:MM:SS"
    safe_ts = ts_str.replace(" ", "_").replace(":", "-")

    screenshot_dir = getattr(settings.trace, "screenshot_dir", "tests/traces/screenshots")

    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(screenshot_dir, safe_ts + ext)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"[warn] Failed to load trace screenshot {path}: {e}")
                return None

    return None

# ==================== PROMPT FILES ====================

def _read_prompt_file(filename: str) -> str:
    from monitor.paths import resource_path
    for path in (str(resource_path(filename)), filename):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            continue
    return ""


# ==================== SESSION CACHE ====================
#
# In-process cache of (intention, normalized_title) -> (decision, timestamp).
# Lets the decision loop skip the LLM when the same window has been classified
# recently with high confidence. Keyed by the active intention id because an
# intention change flips what counts as aligned — a decision cached under the
# old intention must not leak into the new session. Reset on process restart;
# corrections and prior_classifications already handle the cross-session story.

_session_cache: Dict[str, Tuple[Dict[str, Any], float, bool]] = {}

_CACHE_TTL_SEC = 600.0   # 10 minutes
_CACHE_MIN_CONF = 0.80
_CACHE_SKIP_CATS = {"[ERROR]", "[PARSE_ERROR]", "System", "Ambiguous"}


def _cache_key(window_title: str, intention_id) -> str:
    return f"{intention_id if intention_id is not None else ''}|{_normalize_title(window_title)}"


def _cache_get(window_title: str, intention_id=None) -> Optional[Tuple[Dict[str, Any], bool]]:
    """The cached (decision, pinned) pair, or None. pinned=True means the
    entry came from a user correction (force-cached), so it is only as alive
    as the correction that made it — the caller must check."""
    key = _cache_key(window_title, intention_id)
    entry = _session_cache.get(key)
    if entry is None:
        return None
    decision, ts, pinned = entry
    if time.time() - ts > _CACHE_TTL_SEC:
        _session_cache.pop(key, None)
        return None
    return decision, pinned


def _cache_put(window_title: str, decision: Dict[str, Any], intention_id=None, force: bool = False) -> None:
    """force=True bypasses the conf/category gates — used for correction-pinned
    decisions, which are human ground truth even when the category (e.g.
    Ambiguous) would normally be excluded from caching."""
    if not force:
        if decision.get("final_conf", 0.0) < _CACHE_MIN_CONF:
            return
        if decision.get("final_category_name") in _CACHE_SKIP_CATS:
            return
    _session_cache[_cache_key(window_title, intention_id)] = (decision, time.time(), force)


# ==================== TITLE NORMALIZATION ====================

def _normalize_title(title: str) -> str:
    """Normalize a window title for cache matching.

    - Strip WIN: prefix
    - Strip leading (N) notification counts
    - Lowercase
    - Collapse whitespace
    """
    s = title.strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    # Strip leading "(2) " or "(15) " notification counts
    s = re.sub(r"^\(\d+\)\s*", "", s)
    # Collapse whitespace and lowercase
    s = " ".join(s.split()).lower()
    return s


# ==================== PROMPT BUILDING ====================

def _format_activity(events: List[Tuple[str, str]]) -> str:
    """Format events with clear separation between current window and history."""
    if not events:
        return "(no events)"

    _current_ts, current_win = events[-1]
    parts = [f">>> CURRENT WINDOW (classify this): {current_win}"]

    recent = events[max(0, len(events) - 6):-1]
    if recent:
        recent_str = "\n".join([f"  {ts}: {win}" for ts, win in reversed(recent)])
        parts.append(f"RECENT WINDOWS (newest first):\n{recent_str}")

    older = events[:max(0, len(events) - 6)]
    if older:
        unique_older = list(dict.fromkeys([win for _, win in older]))
        parts.append(f"OLDER CONTEXT ({len(older)} events): {', '.join(unique_older[-5:])}")

    return "\n".join(parts)

def _resolve_category(name: str, categories: Optional[List[Dict]]) -> Tuple[str, Optional[bool]]:
    """Match a category name case-insensitively against the category list.

    Returns (exact-cased name, is_productive) — (name, None) if not found.
    """
    if categories:
        for cat in categories:
            if cat["name"].lower() == name.lower():
                return cat["name"], cat.get("is_productive")
    return name, None


def _match_correction(current_window: str, corrections: List[Dict]) -> Optional[Dict]:
    """Check if current window matches a known correction.

    Titles are compared after _normalize_title (same normalization as the
    session cache), so "(2) Chat | Teams" still matches a correction saved
    as "(1) Chat | Teams".
    """
    if not corrections:
        return None
    current_norm = _normalize_title(current_window)
    for c in corrections:
        if _normalize_title(c["window_title"]) == current_norm:
            return c
    return None


# Words that say WHERE something happened, not WHAT it was: browser and app
# chrome, plus generic window furniture. Two windows sharing only these are
# not similar — "Claude" and "Common App activities list - Claude - Google
# Chrome" overlap on the app name alone, which is no evidence at all.
_APP_CHROME_WORDS = {
    "google", "chrome", "firefox", "edge", "safari", "brave", "opera", "arc",
    "microsoft", "windows", "app", "desktop", "window", "tab", "new", "untitled",
    "claude", "chatgpt", "gpt", "gemini", "copilot", "perplexity",
    "visual", "studio", "vscode", "profile", "home", "file", "edit", "view",
}


def _content_words(title: str) -> set:
    """Topic-bearing words of a window title, with app/browser chrome removed."""
    s = title.strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    words = re.findall(r"[a-z0-9']+", s.lower())
    return {w for w in words if len(w) > 1 and w not in _APP_CHROME_WORDS}


def _build_fewshot_prompt(
    current_window: str,
    corrections: List[Dict],
    prior_classifications: List[Dict] = None,
    max_examples: int = 8,
) -> str:
    """Build a few-shot prompt section from corrections AND prior classifications.

    Human corrections take priority. Prior classifications fill remaining slots,
    scored by CONTENT-word overlap with the current window: matching on the app
    name alone pulled in examples about unrelated topics, and the model then
    cited the example's subject as if it described the current window.
    """
    if not corrections and not prior_classifications:
        return ""

    current_words = _content_words(current_window)

    # --- Collect correction examples (highest priority) ---
    correction_lines = []
    used_titles = set()

    if corrections:
        scored = []
        for c in corrections:
            overlap = len(current_words & _content_words(c["window_title"]))
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: -x[0])

        for _, c in scored[:max_examples]:
            correction_lines.append(
                f'- "{c["window_title"]}" was {c["ai_category"]} -> corrected to {c["corrected_category"]}'
            )
            used_titles.add(c["window_title"].lower())

        # Fill remaining slots with non-overlapping corrections
        remaining = max_examples - len(correction_lines)
        if remaining > 0:
            for c in corrections:
                if c["window_title"].lower() not in used_titles:
                    correction_lines.append(
                        f'- "{c["window_title"]}" was {c["ai_category"]} -> corrected to {c["corrected_category"]}'
                    )
                    used_titles.add(c["window_title"].lower())
                    remaining -= 1
                    if remaining <= 0:
                        break

    # --- Exact-title precedent (its own block, outside the slot budget) ---
    # The session cache only pins high-confidence calls for 10 minutes under
    # one intention. Past that, what this very title was called before is the
    # strongest signal there is — dropping it made the same window flip
    # category mid-session once its cache entry aged out.
    exact_lines = []
    exact_high_conf = True
    current_norm = _normalize_title(current_window)
    corrected_norms = {_normalize_title(c["window_title"]) for c in (corrections or [])}
    if prior_classifications and current_norm not in corrected_norms:
        for p in prior_classifications:
            if p["normalized_title"] == current_norm:
                n = p.get("count", 1)
                seen = f"{n} times" if n > 1 else "once"
                exact_high_conf = p.get("high_conf", True)
                exact_lines.append(
                    f'- THIS EXACT WINDOW was classified {seen} recently as '
                    f'{p["category_name"]} (conf={p["confidence"]:.2f}).'
                )
                break

    # --- Collect prior classification examples (fill remaining slots) ---
    prior_lines = []
    remaining_slots = max_examples - len(correction_lines)

    if prior_classifications and remaining_slots > 0:
        scored_prior = []
        for p in prior_classifications:
            # Skip if already covered by a correction
            if p["window_title"].lower() in used_titles:
                continue
            # Skip exact match with current window (shown in its own block)
            if p["normalized_title"] == current_norm:
                continue
            # A title that qualified only by repetition (low conf) speaks for
            # itself, never for other windows — cross-title examples keep the
            # confidence bar. Absent flag (Supabase sink) means high-conf.
            if not p.get("high_conf", True):
                continue
            overlap = len(current_words & _content_words(p["window_title"]))
            if overlap > 0:
                scored_prior.append((overlap, p))

        scored_prior.sort(key=lambda x: -x[0])

        for _, p in scored_prior[:remaining_slots]:
            prior_lines.append(
                f'- "{p["window_title"]}" -> classified as {p["category_name"]} (conf={p["confidence"]:.2f})'
            )

    # --- Assemble output ---
    all_lines = []

    if correction_lines:
        all_lines.append("LEARNED FROM PAST CORRECTIONS (the user corrected these):")
        all_lines.extend(correction_lines)

    if exact_lines:
        all_lines.append("\nPRECEDENT FOR THIS EXACT WINDOW:")
        all_lines.extend(exact_lines)
        if exact_high_conf:
            all_lines.append(
                "Prefer this category for the current window, for consistency — repeated "
                "past judgments of this very title are strong evidence. Depart from it "
                "only when the current context clearly contradicts it, and say why. "
                "Precedent speaks to CATEGORY only, never to ALIGN."
            )
        else:
            # Repeat-qualified low-confidence precedent must never instruct:
            # telling the model to prefer its own repeated coin-flips lets a
            # few shaky calls in a row entrench the initial mistake.
            all_lines.append(
                "These were LOW-confidence calls that merely repeated: treat them as "
                "context, not an instruction — judge the current evidence on its own "
                "merits. Precedent speaks to CATEGORY only, never to ALIGN."
            )

    if prior_lines:
        all_lines.append("\nSIMILAR WINDOWS PREVIOUSLY CLASSIFIED:")
        all_lines.extend(prior_lines)

    if all_lines:
        all_lines.append(
            "\nEach correction applies to that EXACT window title. Generalize by the CONTENT "
            "of the title (its topic/subject), never by the app or site alone: one YouTube "
            "video corrected to a category says nothing about other YouTube videos - judge "
            "each video by its own title. Only reuse a correction's category when the current "
            "window is about the same specific content."
            "\n\nThese examples are PAST WINDOWS, not the current one. Never describe their "
            "subject matter as if it were on screen now, never quote their titles as evidence "
            "about the CURRENT WINDOW, and never let them raise ALIGN — alignment is judged "
            "from the current window alone."
        )

    return "\n".join(all_lines) if all_lines else ""


def _project_context(intention: Optional[Dict]) -> str:
    """Prompt lines for the project a session is filed under, or "".

    The project is context for CATEGORY only — screens about the project's
    subject belong in its home category rather than a site's default one.
    It never touches ALIGN: the declared intention is still the only thing
    alignment is scored against, so other work on the same project scores
    exactly as it would without the project.
    """
    p = (intention or {}).get("project") or {}
    name = (p.get("name") or "").strip()
    if not name:
        return ""
    lines = [f'This session is part of the project "{name}".']
    desc = (p.get("description") or "").strip()
    if desc:
        lines.append(f"About the project: {desc}")
    cat = (p.get("category") or "").strip()
    if cat:
        lines.append(
            f'Work on this project belongs in the category "{cat}" — prefer it '
            f"over a site's default category when what is on screen is plainly "
            f"about this project.")
    lines.append(
        "The project is context for CATEGORY only. It never raises ALIGN: "
        "alignment is still judged against the declared intention alone, and "
        "work on this project that is not the declared intention scores as it "
        "would without the project.")
    return "\n".join(lines) + "\n"


def _title_names_project(window_title: str, intention: Optional[Dict]) -> Optional[str]:
    """The active project's name when the window title contains it as whole
    words ("personal statement - Google Docs" for the project "Personal
    Statement"), else None. Names under 4 letters are too easy to hit by
    accident ("CS" in any title) and never count."""
    project = (intention or {}).get("project") or {}
    name = " ".join(str(project.get("name") or "").lower().split())
    if len(name.replace(" ", "")) < 4:
        return None
    title = " ".join(re.sub(r"^WIN:", "", str(window_title or "")).lower().split())
    if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", title):
        return project.get("name")
    return None


def _intention_block(intention: Dict) -> str:
    """Prompt section for an active session intention.

    Appended after the main template (like the few-shot section) so
    prompt-main.txt stays unchanged for tune_prompt/compare_models runs.
    """
    return (
        f'ACTIVE FOCUS SESSION (until {intention["ends_at"]}):\n'
        f'The user declared: "{intention["text"]}"\n'
        f'{_project_context(intention)}\n'
        "In addition to the category, judge how well the CURRENT WINDOW serves this "
        "declared session intention. Add ALIGN as an extra field placed BEFORE "
        "REASON, so REASON stays last:\n"
        "CATEGORY=... | OFF_SCORE=... | CONF=... | ALIGN=<0.0-1.0> | REASON=...\n"
        "Output that one line only — no commentary after it.\n"
        "ALIGN=<0.0-1.0> where:\n"
        "- 0.8-1.0: directly doing the declared work\n"
        "- 0.5-0.7: adjacent/supporting activity (docs, research, tooling for it)\n"
        "- 0.2-0.4: productive in general but NOT what they declared\n"
        "- 0.0-0.1: clearly off-task\n"
        "Judge alignment by content, not platform: a video or article on the declared "
        "topic is aligned; unrelated productive work is not. ALIGN is independent of "
        "CATEGORY — an activity can be a productive category with low ALIGN.\n"
        "ALIGN IS NOT A PRODUCTIVITY SCORE. Focused, skilled work on a DIFFERENT "
        "project scores LOW (0.0-0.3) — drifting into other good work is exactly what "
        "a focus session exists to catch. Declared \"finish my personal statement\" "
        "while composing music in MuseScore -> ALIGN 0.05, even though composing is "
        "real work and keeps its own productive category.\n\n"
        "SCORE ALIGN FROM THE CURRENT WINDOW ALONE. RECENT WINDOWS are history, not "
        "evidence about the window in front of the user now. That they were on the "
        "declared work a minute ago says nothing about whether THIS window serves it "
        "— having just switched away is precisely what this score exists to catch. "
        "Never quote a recent window's title as though it were the current one's. If "
        "the CURRENT WINDOW's own title says nothing tying it to the intention "
        "(a bare app name like \"Claude\" or \"Chrome\"), score it LOW and say the "
        "title carries no evidence; do not borrow a reason from history.\n\n"
        "THE DECLARED INTENTION ALSO OUTRANKS PLATFORM-BASED CATEGORY RULES.\n"
        "Category rules that name a site (Reddit -> Social Media, YouTube -> "
        "Entertainment, and so on) describe the DEFAULT use of that site. When the "
        "content on screen is plainly about the declared intention, it is work on that "
        "intention: choose the productive category that fits the work, not the site's "
        "default off-task category. Examples, with the session intention in brackets:\n"
        "- \"r/ApplyingToCollege\" [applying to college] -> the college-application "
        "category, NOT Social Media — it is a forum about the declared task.\n"
        "- \"How to write a personal statement - YouTube\" [applying to college] -> "
        "the college-application category, NOT Entertainment.\n"
        "- \"r/nba\" [applying to college] -> Social Media, as usual: unrelated content.\n"
        "Apply the site's default off-task category only when the content is unrelated "
        "to the declared intention. This paragraph is about CATEGORY only — it never "
        "raises ALIGN. An activity can take a productive category and still score "
        "near 0 for alignment."
    )


def _parse_align(content: str) -> Optional[float]:
    m = re.search(r"ALIGN\s*=\s*([01](?:\.\d+)?|\.\d+)", content or "", flags=re.I)
    if not m:
        return None
    return min(max(float(m.group(1)), 0.0), 1.0)


def _productivity_label(is_productive) -> str:
    if is_productive is True:
        return "PRODUCTIVE"
    if is_productive is False:
        return "NOT PRODUCTIVE"
    return "NEUTRAL"


# prompt-main.txt was written for the starter categories, and names their four
# off-task buckets as fixed text in rule (b) and rule 4. The user's own
# categories, and every neutral one, were invisible to those rules, so a title
# that clearly fit a neutral category (Music) was pushed into the nearest
# named bucket (Entertainment). The list is rebuilt from the real categories.
_STARTER_OFF_TASK = "(Social Media, Gaming, Entertainment, Shopping)"
_CLASSIFIER_BUCKETS = {"ambiguous", "system"}


def _fit_rules_to_categories(prompt: str, categories: List[Dict]) -> str:
    if not categories:
        return prompt
    named = [c for c in categories if (c.get("name") or "").lower() not in _CLASSIFIER_BUCKETS]
    off_task = [c["name"] for c in named if c.get("is_productive") is False]
    neutral = [c["name"] for c in named if c.get("is_productive") is None]
    specific = off_task + neutral
    if specific and _STARTER_OFF_TASK in prompt:
        prompt = prompt.replace(
            "specific off-task category " + _STARTER_OFF_TASK,
            "specific off-task or neutral category (" + ", ".join(specific) + ")")
        prompt = prompt.replace(_STARTER_OFF_TASK, "(" + ", ".join(specific) + ")")
    if neutral:
        names = ", ".join(f'"{n}"' for n in neutral)
        prompt += (
            "\n\nNEUTRAL CATEGORIES: " + names + " are neither productive nor "
            "off-task. Each is a specific answer, held to the same standard as an "
            "off-task category: use it when the CURRENT WINDOW TITLE plainly shows "
            "the activity its description names. When it does, the neutral "
            "category beats Ambiguous, and it beats any off-task category that "
            "only resembles the activity — follow the category descriptions, "
            "not the nearest platform rule. The title does not have to name a "
            "platform: a window titled with a piece or song and its composer or "
            "performers, the way music players title what they are playing "
            "(\"Symphony No. 5 • Beethoven, Berliner Philharmoniker\"), shows "
            "listening to music as plainly as \"Spotify\" does.")
    return prompt


def build_prompt(
    events: List[Tuple[str, str]],
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
    intention: Optional[Dict] = None,
) -> str:
    activity_str = _format_activity(events)

    template = _read_prompt_file("prompt-main.txt")
    if not template:
        return (
            "You are a focus judge. Categorize the user's activity.\n"
            f"Recent activity: {activity_str}\n"
            "Output: CATEGORY=<category_name> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )

    if categories:
        # Three labels, as the screen check prints them. A neutral category
        # printed as NOT PRODUCTIVE looked exactly like Entertainment, so
        # "listening to Mahler" landed in Entertainment (2026-10-07).
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']} "
            f"({_productivity_label(cat.get('is_productive'))})"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"

    filled = template
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", activity_str)
    filled = _fit_rules_to_categories(filled, categories or [])

    # Inject few-shot examples (corrections + prior classifications)
    if events:
        current_win = events[-1][1]
        fewshot_section = _build_fewshot_prompt(
            current_win, corrections or [], prior_classifications
        )
        if fewshot_section:
            filled = filled + "\n\n" + fewshot_section

    if intention:
        filled = filled + "\n\n" + _intention_block(intention)

    return filled

# ==================== LLM PARSING ====================

def _parse_llm_line(
    content: str,
    categories: List[Dict] = None,
) -> Tuple[str, float, float, str, Optional[bool]]:
    """
    Parse LLM response to extract category and other info.

    Returns:
        (category_name, off_score, conf, reason, is_productive)
    """
    content = (content or "").strip()

    # Extract CATEGORY
    m_category = re.search(r"CATEGORY\s*=\s*(.+?)(?:\||\n|$)", content, flags=re.I)
    category_name = m_category.group(1).strip().strip('"').strip("'") if m_category else None

    # Extract OFF_SCORE and CONF
    m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)

    off_score = float(m_off.group(1)) if m_off else 0.5
    conf = float(m_conf.group(1)) if m_conf else 0.5

    off_score = min(max(off_score, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)

    # Extract REASON. It is the last field, so the capture runs to the end of
    # the response — trim anything the model tacked on after it: a further
    # pipe-delimited field (e.g. "| ALIGN=0.1") or a trailing essay separated
    # by a blank line ("**Explanation of ALIGN=0.1:** ...").
    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    if m_reason:
        reason = re.split(r"\s*\|\s*[A-Z_]{3,}\s*=", m_reason.group(1))[0]
        reason = reason.split("\n\n")[0].strip()
    else:
        reason = ""

    # Look up category in list
    is_productive = None

    if categories and category_name:
        for cat in categories:
            if cat["name"].lower() == category_name.lower():
                category_name = cat["name"]  # exact casing from sheet
                is_productive = cat.get("is_productive", None)
                break
        else:
            print(f"[warn] Category '{category_name}' not found in categories list")

    if not category_name:
        category_name = "[PARSE_ERROR]"

    return category_name, off_score, conf, reason, is_productive


# ==================== LLM CALLS ====================

def ask_gpt(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    categories: List[Dict] = None,
) -> Tuple[str, float, float, str, Optional[bool], str]:
    """
    Call GPT with prompt and parse response for category.

    Returns:
        (category_name, off_score, conf, reason, is_productive, raw_content)
    """
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()

        category_name, off_score, conf, reason, is_productive = _parse_llm_line(
            content, categories
        )
        return category_name, off_score, conf, reason, is_productive, content

    except Exception as e:
        return "[ERROR]", 0.5, 0.5, str(e), None, ""


# ==================== VISION ====================
def ask_gpt_vision_verdict(
    screenshot_b64: str,
    events: List[Tuple[str, str]],
    primary_category: str,
    primary_off: float,
    primary_conf: float,
    primary_reason: str,
    *,
    model: str,
    max_tokens: int,
    categories: List[Dict] = None,
    intention: Optional[Dict] = None,
) -> Tuple[str, float, float, str, Optional[bool], str]:
    """Vision tiebreaker. Reviews the primary's window-title-only call against
    the screenshot, then either agrees or overrides.

    Returns (category_name, off_score, conf, reason, is_productive, raw_content)
    — the primary call's parse output plus the raw response (for ALIGN parsing).
    """
    if not events:
        current_window = "(unknown)"
        history_str = "(none)"
    else:
        current_window = events[-1][1]
        recent = events[max(0, len(events) - 6):-1]
        history_str = ", ".join([win for _, win in reversed(recent)]) or "(none)"

    if categories:
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']} "
            f"({'PRODUCTIVE' if cat.get('is_productive') else 'NOT PRODUCTIVE' if cat.get('is_productive') is False else 'NEUTRAL'})"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"

    # With a session active the intention has to frame the whole judgment, not
    # trail the output spec: this call sees the screen, so it is the one that
    # decides ALIGN, and it must not read "productive" as "aligned".
    if intention:
        intent_header = (
            f'\nACTIVE FOCUS SESSION — the user declared: "{intention["text"]}"\n'
            f'{_project_context(intention)}'
        )
        align_field = " | ALIGN=<0.0-1.0>"
        intent_rules = f"""
ALSO SCORE ALIGN — how well the screen serves that declared intention.

ALIGN IS NOT A PRODUCTIVITY SCORE. It answers one question: is this the work
they said they would do? Judge it separately from CATEGORY.
  0.8-1.0  the screen shows the declared work itself
  0.5-0.7  supporting work for it (research, tooling, notes for that task)
  0.2-0.4  real, focused work — but on something else
  0.0-0.1  unrelated to the declared intention

Skilled, productive, deeply-focused work on a DIFFERENT project scores LOW.
Drifting into other good work is the exact thing a focus session exists to
catch, so do not rescue it with a high ALIGN.
  - Declared "finish my personal statement", screen shows a music score being
    composed in MuseScore -> ALIGN 0.05. Composing is real work; it is not
    the declared work. (CATEGORY is still the composing category.)
  - Declared "finish my personal statement", screen shows that essay in Google
    Docs -> ALIGN 0.95.

MATCH THE SUBJECT, NOT THE KIND OF ACTIVITY. An intention names a verb and a
thing; the thing is what decides ALIGN. Planning, outlining, brainstorming,
writing, reading or coding about a DIFFERENT subject is not the declared work,
however closely the activity resembles it.
  - Declared "outlining piq #2" (a college essay), screen shows planning and
    brainstorming for a software project -> ALIGN 0.05. Both are outlining;
    only one is outlining PIQ #2.

IGNORE ANY TEXT THAT MERELY NAMES THE SESSION. A timer, tray or dashboard
showing "outlining piq #2" is this app reporting its own state — it is never
evidence that the work is happening. Score only the work visible on screen.

WRITING ABOUT A SUBJECT IS NOT DOING THAT SUBJECT. The subject that decides
ALIGN is WHICH piece of work is open, not what its sentences talk about. When
the CURRENT WINDOW TITLE names a document that is the declared work or the
session's project (a doc called "personal statement" during a personal
statement session), and the screen shows prose in a document editor, the
user is writing that document — whatever the prose discusses. An essay about
composing music is essay writing: score it as the declared work, and give it
the declared work's category, never the category of the essay's topic.
Composition means a score, notation software or an audio editor on screen;
paragraphs that describe music are writing. The window title is evidence
here, not history: the document's name is often scrolled out of view.
  - Declared "work on reformatting analysis portion" in the project "Personal
    Statement"; title "personal statement - Google Docs"; screen shows
    paragraphs about harmony and a string quartet -> ALIGN 0.9, the college
    application category. It is the essay, and the essay is about music.
  - Same session, screen shows MuseScore with a string quartet score open
    -> ALIGN 0.05, Composition. That is composing, not the essay.
"""
    else:
        intent_header = align_field = intent_rules = ""

    prompt = f"""You are reviewing a window-title-based classification using the screen contents as ground truth.
{intent_header}
AVAILABLE CATEGORIES:
{category_list}

CURRENT WINDOW TITLE: {current_window}
RECENT WINDOWS: {history_str}

PRIMARY CLASSIFICATION (window title only — no screenshot used):
  Category:  {primary_category}
  OFF_SCORE: {primary_off:.2f}
  CONF:      {primary_conf:.2f}
  Reason:    {primary_reason}

YOUR TASK:
Look at the screenshot. Decide whether the primary's call is supported by what's actually on screen, or whether it should be overridden.

OVERRIDE RULES (asymmetric — direction matters):

  CASE A — Primary said a PRODUCTIVE category:
    The primary may have over-trusted a productive-looking window title. DEMOTE if the screenshot clearly shows the user is NOT doing that productive activity.
    Examples:
      - Title says "Khan Academy" but screen shows YouTube comments scroll → Entertainment
      - Title says "myproject - VS Code" but screen shows Discord overlay in foreground → Social Media
      - Title says "Wikipedia: Calculus" but screen shows an unrelated article (celebrity gossip, sports) → Ambiguous
    NEVER demote a document for its TOPIC. If the title names a document
    ("personal statement - Google Docs", "lab report - Word") and the screen
    shows paragraphs in a document editor, the user is writing that document:
    AGREE, whatever the paragraphs are about. An essay about music, sports or
    games is still essay writing — not Composition, Sports or Gaming.

  CASE B — Primary said "Ambiguous":
    The primary defaulted because the window title had no specific productive evidence (bare "Chrome", "VS Code", "New Tab", etc.). PROMOTE to a productive category ONLY if the screenshot clearly shows a productive activity.
    Examples:
      - Title is "Google Chrome" but screen shows a calculus textbook PDF → Homework
      - Title is "Visual Studio Code" but screen shows a named project's source code → Deep Work
      - Title is "New Tab" but screen shows Wolfram Alpha computing a derivative → Research & Learning

DEFAULT BEHAVIOR — AGREE WITH PRIMARY:
  If the screenshot is unclear, low-resolution, shows a lock/idle screen, doesn't show a recognizable activity, or is consistent with the primary's call — AGREE. Do not override on uncertainty. The primary's window-title evidence stands.
{intent_rules}
CONF is how certain YOU are of your CATEGORY from what is visible. Score it
honestly — an override you are unsure of is worse than none:
  0.85-0.95  unmistakable: the activity is plainly on screen
  0.70-0.80  likely, but partly inferred from context
  below 0.70 uncertain — AGREE with the primary instead

Output format (strict, single line):
CATEGORY=<exact category name from list above> | OFF_SCORE=<0.0-1.0> | CONF=<0.0-1.0>{align_field} | REASON=<brief explanation citing what you saw on screen>"""

    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{_detect_image_mime(screenshot_b64)};base64,{screenshot_b64}"}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _parse_llm_line(content, categories) + (content,)

    except Exception as e:
        return "[ERROR]", 0.5, 0.5, str(e), None, ""

# ==================== SYSTEM WINDOW DETECTION ====================

# Known system/utility windows. Matched against the WHOLE title or the app
# name at its end — never as a substring anywhere.
#
# A substring test let any window claim to be one. A tab named "command
# prompt" (a known trick for slipping past blockers) took the shortcut below:
# no LLM call, no vision, category System — which resolves to no-signal, so it
# never counts as drift and never raises an alert. It also fired on innocent
# titles: a video called "How to use Task Manager" was silently logged as
# System. Real system windows carry bare titles; a browser appends its own
# name ("... - Google Chrome"), so anchoring the match separates them.
_SYSTEM_WINDOWS = [
    "snipping tool",
    "task switching",
    "focus alert",
    "unlockingwindow",
    "windows default lock screen",
    "narrator",
    "quick settings",
    "task manager",
    r"c:\windows\system32\cmd.exe",
    "command prompt",
]

# Distinctive enough that nothing else could plausibly contain them, so these
# stay substring matches — a lock screen must be recognised whatever decoration
# Windows hangs off the title.
_ALWAYS_SYSTEM = [
    "unlockingwindow",
    "windows default lock screen",
]


def _is_system_window(title: str) -> bool:
    """Check if window title matches a known system/utility window."""
    clean = title.strip()
    if clean.upper().startswith("WIN:"):
        clean = clean[4:]
    lower = " ".join(clean.split()).lower()
    if any(pattern in lower for pattern in _ALWAYS_SYSTEM):
        return True
    if lower in _SYSTEM_WINDOWS:
        return True
    # The last " - " segment is where the owning app names itself.
    return lower.rsplit(" - ", 1)[-1].strip() in _SYSTEM_WINDOWS


# ==================== VISION TIEBREAKER LOGIC ====================

# CONF ceiling for the default-by-absence band. Matches the band defined in
# prompt-main.txt (Rule 5 scoring): default-by-absence CONF = 0.50–0.70.
_DEFAULT_BY_ABSENCE_CONF_CEILING = 0.70


def _should_run_vision(
    is_productive: Optional[bool],
    conf: float,
    settings: Settings,
    category_name: Optional[str] = None,
) -> bool:
    """Vision looks at the screen in three cases:

      Direction A — VERIFY PRODUCTIVE:
        Primary said the user is being productive (is_productive=True).
        Vision checks whether the screen actually shows that.

      Direction B — SECOND-GUESS DEFAULT-BY-ABSENCE:
        Primary said off-task or neutral (is_productive in {False, None})
        with CONF in the default-by-absence band (≤ 0.70). The primary
        likely fell back because the title had no specific signal. Vision
        checks whether the screen reveals productive activity the title
        couldn't.

      Direction C — LOOK BEHIND AMBIGUOUS:
        Primary said Ambiguous, at any confidence. That call is the model
        saying the title told it nothing, whatever CONF it put on it, so the
        screen is the only evidence left. Whether the screen actually lifts
        it is a separate question, answered by the high bar in
        _vision_may_override — this only decides that it gets a look.

    Confident off-task calls (CONF > 0.70, is_productive=False) skip vision —
    the title already carried a specific signal (Twitter, Minecraft, etc.)
    and a screenshot won't change the answer.
    """
    if not settings.critic.vision_enabled:
        return False

    if is_productive is True:
        return True

    if conf <= _DEFAULT_BY_ABSENCE_CONF_CEILING:
        return True

    if _is_ambiguous(category_name):
        return True

    return False


def _screen_still_shows(window_key: str) -> bool:
    """True if the foreground window still matches the event being classified.

    The screenshot fires seconds after the title was polled — the primary
    LLM round-trip sits between them — so the user may have switched windows
    in the gap. A screenshot taken then is evidence about the NEW window,
    and a verdict scored from it would be cached against the old title.
    When capture cannot say (empty title, error), the screenshot is allowed
    rather than losing vision to a flaky grab.
    """
    try:
        live = get_active_window()
    except Exception:
        return True
    if not live:
        return True
    live_key = f"WIN:{live}"
    # Event keys clip long titles at TITLE_MAXLEN with a trailing ellipsis.
    if window_key.endswith("…"):
        return live_key.startswith(window_key[:-1])
    return live_key == window_key


def _screenshot_for(events_context, settings) -> Tuple[Optional[str], str]:
    """The image the vision call should look at, and where it came from.

    In trace mode the live screen has nothing to do with the historical event
    being replayed, so only a matching fixture will do. Live, the screen must
    still be showing the window under classification.
    """
    if settings.trace.use_trace_file:
        return _load_trace_screenshot(events_context, settings), "trace fixture"
    window_key = events_context[-1][1] if events_context else ""
    if window_key and not _screen_still_shows(window_key):
        print("[vision] Foreground changed since the title was polled — screenshot skipped.")
        return None, "live"
    return take_screenshot_b64(), "live"


# ==================== VISION OVERRIDE GATE ====================
#
# Vision sees the screen, so it outranks the title — but not for free. The
# classifier is center-biased on purpose: Ambiguous is where a decision lands
# without evidence, and it should take strong evidence to leave. So vision's
# CONF has to clear a bar to overturn what it reviewed, and the bar depends on
# what that was:
#
#   override_min_conf (0.70)            a specific category the title supported,
#                                       demoting it back to Ambiguous included
#   override_ambiguous_min_conf (0.85)  leaving Ambiguous, or overruling a user
#                                       correction that is open to the screen
#
# What vision decides is a verdict, never a correction: it is logged (tagged,
# so it never becomes title precedent) and cached, and nothing about it is
# written back as a human label, a correction rule or an intent ruling.

_AMBIGUOUS = "ambiguous"


def _is_ambiguous(name: Optional[str]) -> bool:
    return (name or "").strip().lower() == _AMBIGUOUS


def _correction_is_open(correction: Dict) -> bool:
    """Is this correction one the screen may look behind?

    Ambiguous on either side of it means the window's title carried no
    evidence: the model found none (Ambiguous -> X), or the user declined to
    give any (X -> Ambiguous). Such a label is only as good as whatever the
    user happened to be doing under that title when they made it — bare
    "Claude" is essays one day and robotics the next — so vision gets a say,
    at the high bar. A correction between two specific categories is a policy
    call the screen cannot see (music while working is Music, not
    Entertainment, whatever YouTube looks like) and stays final.
    """
    return (_is_ambiguous(correction.get("ai_category"))
            or _is_ambiguous(correction.get("corrected_category")))


def _known_category(name: Optional[str], categories: Optional[List[Dict]]) -> bool:
    """With a category list to check against, does `name` name one of them?"""
    if not categories:
        return True
    wanted = (name or "").strip().lower()
    return any(c["name"].lower() == wanted for c in categories)


def _vision_may_override(
    current_name: str,
    v_name: str,
    v_conf: float,
    settings: Settings,
    categories: Optional[List[Dict]],
    *,
    correction: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """Does vision's verdict clear the bar to replace `current_name`?

    Returns (override, why). A verdict naming something that is not one of
    the user's categories never overrides, however sure it sounds — an
    invented category would count as neither work nor drift.
    """
    if (v_name or "").strip().lower() == (current_name or "").strip().lower():
        return False, "agrees"
    if not _known_category(v_name, categories):
        return False, f"'{v_name}' is not one of the categories"
    if correction is not None:
        bar, what = settings.critic.override_ambiguous_min_conf, "overrule a user correction"
    elif _is_ambiguous(current_name):
        bar, what = settings.critic.override_ambiguous_min_conf, "leave Ambiguous"
    else:
        bar, what = settings.critic.override_min_conf, f"overturn {current_name}"
    if v_conf >= bar:
        return True, f"conf {v_conf:.2f} clears the {bar:.2f} bar to {what}"
    return False, f"conf {v_conf:.2f} is under the {bar:.2f} bar to {what}"


def _cache_entry_applies(cached: Dict, pinned: bool, correction: Optional[Dict]) -> bool:
    """Is a cached verdict still the answer under the correction now in force?

    - No correction: only unpinned entries count. A pinned one came from a
      correction that has since been deleted, and must not keep serving the
      retracted category.
    - A correction: the entry must agree with it — or be vision's overrule of
      this very correction, the one disagreement allowed to stand (otherwise
      the screen would be re-read every decision for as long as the overrule
      lasts). A correction retargeted since fails both tests and is judged
      afresh.
    """
    if correction is None:
        return not pinned
    want = correction["corrected_category"].strip().lower()
    if (cached.get("final_category_name") or "").strip().lower() == want:
        return True
    return bool(pinned) and (cached.get("overrode_correction") or "").strip().lower() == want


def _overridden_primary(primary: Dict, vision: Dict) -> Dict:
    """The `primary` record to report when vision overrode the title call.

    Sinks store the primary's conf and reason beside the FINAL category, so on
    an override the reason has to explain the category actually stored — what
    vision saw, tagged "[vision]" so the review tab reads it as a screen
    verdict and load_prior_classifications never mistakes it for title
    precedent. The title call's own reason stays in the text, and its conf
    stays put: that column measures how sure the title-only call was, and the
    review tab's low-confidence filter leans on it to surface exactly these
    rows.
    """
    seen = vision.get("reason") or "overrode on screen evidence"
    was = primary.get("category_name") or "?"
    said = primary.get("reason") or "no reason given"
    return dict(primary, reason=f"[vision] {seen} (overrode {was}: {said})")


def _vision_review(
    events_context,
    settings: Settings,
    categories: Optional[List[Dict]],
    intention: Optional[Dict],
    *,
    primary_category: str,
    primary_off: float,
    primary_conf: float,
    primary_reason: str,
) -> Optional[Dict[str, Any]]:
    """Vision's read of the screen behind a corrected window.

    Returns the same record the main path builds for a vision verdict —
    {category_name, is_productive, off, conf, reason, align} — or None when
    vision is off, has no screenshot, or errored, in which case callers fall
    back to the title. ALIGN is scored only under a session intention. The
    category in the record is advisory: the caller decides whether the
    correction is open to it (_correction_is_open) and whether it clears the
    bar (_vision_may_override).
    """
    if not settings.critic.vision_enabled:
        return None
    b64_img, source = _screenshot_for(events_context, settings)
    if not b64_img:
        return None

    v_name, v_off, v_conf, v_reason, v_prod, v_raw = ask_gpt_vision_verdict(
        b64_img, events_context,
        primary_category=primary_category, primary_off=primary_off,
        primary_conf=primary_conf, primary_reason=primary_reason,
        model=settings.critic.vision_model,
        max_tokens=settings.critic.vision_max_tokens,
        categories=categories, intention=intention,
    )
    if v_name in ("[ERROR]", "[PARSE_ERROR]"):
        print(f"[vision] correction review failed ({v_reason}) — falling back to the title")
        return None

    align = _parse_align(v_raw) if intention else None
    if align is not None:
        print(f"[vision] correction review ({source}): ALIGN {align:.2f} "
              f"for \"{intention['text'][:40]}\"")
    return {"category_name": v_name, "is_productive": v_prod, "off": v_off,
            "conf": v_conf, "reason": v_reason or "", "align": align}


# ==================== MAIN DECISION ====================

def decide(
    events_context: List[Tuple[str, str]],
    settings: Settings = DEFAULT_SETTINGS,
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
    intention: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Window-title primary + a vision second look that has to earn its override.

    Pipeline:
      1. User correction exact-match  -> pinned, LLM-free — unless a session
         needs ALIGN or the correction is open to the screen (step 3.5)
      2. System window shortcut       -> skip LLM
      3. Primary text-LLM (window title + history, no screenshot)
      4. Vision looks at the screen if ANY of:
           (A) primary returned a productive category
           (B) primary's CONF <= 0.70 (the default-by-absence band)
           (C) primary returned Ambiguous, at any confidence
         and overturns the primary only past a bar — override_min_conf for a
         specific category, the higher override_ambiguous_min_conf to leave
         Ambiguous or overrule a correction (_vision_may_override).

    `intention` is the active session intention (monitor/intentions.py). When
    set, prompts additionally request an ALIGN score against it and the result
    carries final_align (None on the LLM-skipping shortcut paths).

    Returns: dict with final_category_name, final_is_productive, final_off,
    final_conf, final_reason, final_align, vision_ran, vision_overrode,
    primary, vision. `primary` is what the stored row's conf/reason describe:
    the title call, or the human correction that pinned the category; on a
    vision override its reason leads with what the screen showed, tagged.
    """
    intention_id = intention["id"] if intention else None

    # --- 0. Has the user already ruled on this window for this intention? ---
    # An ALIGN correction is human ground truth about alignment, the way a
    # category correction is about category: it outranks whatever the models
    # decide below, on every return path (see mk_ret and the shortcut paths).
    forced_align: Optional[float] = None
    if intention and events_context:
        try:
            from monitor import intent_corrections
            ruled = intent_corrections.lookup(intention["text"], events_context[-1][1])
        except Exception as e:
            print(f"[intent] correction lookup skipped ({e})")
            ruled = None
        if ruled is not None:
            forced_align = 1.0 if ruled else 0.0
            print(f"[intent] user ruled this window "
                  f"{'ON' if ruled else 'OFF'}-intent for \"{intention['text'][:40]}\"")

    # --- 1. User correction exact-match ---
    # A correction pins the CATEGORY (human ground truth). Two things send it
    # past this shortcut to step 3.5 instead of returning here, LLM-free:
    #   - a session: ALIGN is intention-relative and a category label says
    #     nothing about it, so the primary call still runs to score ALIGN and
    #     the corrected category is pinned after it;
    #   - an OPEN correction (_correction_is_open): Ambiguous on either side
    #     means the title carried no evidence, so the screen gets a look and
    #     may overrule the label at the high bar.
    # A closed correction with no session is final, and free.
    correction = None
    if corrections and events_context:
        correction = _match_correction(events_context[-1][1], corrections)
    if correction and intention is None and not (
        settings.critic.vision_enabled and _correction_is_open(correction)
    ):
        cat_name, is_productive = _resolve_category(correction["corrected_category"], categories)
        print(f"[correction] Exact match: '{correction['window_title'][:40]}' -> {cat_name} (skipping LLM)")
        return {
            "final_category_name": cat_name,
            "final_is_productive": is_productive,
            "final_off": 0.0 if is_productive else 1.0,
            "final_conf": 1.0,
            "final_reason": f"User correction: was {correction['ai_category']}, corrected to {cat_name}",
            "final_align": None,
            "vision_ran": False,
            "vision_overrode": False,
            "primary": {"category_name": cat_name, "is_productive": is_productive,
                        "off": 0.0, "conf": 1.0, "reason": "user correction"},
            "vision": None,
        }

    # --- 2. System window shortcut ---
    if events_context and _is_system_window(events_context[-1][1]):
        return {
            "final_category_name": "System",
            "final_is_productive": None,
            "final_off": 0.5,
            "final_conf": 1.0,
            "final_reason": "System/utility window - not a user activity",
            "final_align": None,
            "vision_ran": False,
            "vision_overrode": False,
            "primary": {"category_name": "System", "is_productive": None,
                        "off": 0.5, "conf": 1.0, "reason": "system window shortcut"},
            "vision": None,
        }

    # --- 2.5. Session cache: same window classified recently? ---
    # A cached entry that disagrees with a (possibly newer) correction is
    # stale — bypass it so the correction wins within one decision, matching
    # the no-intention path where corrections are checked before the cache.
    # The same goes in reverse: a correction-pinned entry whose correction
    # was DELETED must not keep serving the retracted category, so with no
    # correction in force only unpinned entries count. The one disagreement
    # allowed to stand is vision's overrule of the very correction in force
    # (_cache_entry_applies).
    if events_context:
        hit = _cache_get(events_context[-1][1], intention_id)
        cached, pinned = hit if hit is not None else (None, False)
        if cached is not None and _cache_entry_applies(cached, pinned, correction):
            print(f"[cache] hit: {events_context[-1][1][:50]} -> {cached['final_category_name']}")
            result = dict(cached)
            result["final_reason"] = f"[cache] {cached.get('final_reason', '')}"
            # A ruling made after this was cached still wins.
            if forced_align is not None:
                result["final_align"] = forced_align
            return result

    current_window = events_context[-1][1] if events_context else ""

    # --- 3. Primary text-LLM call ---
    # Skipped for a correction with no session: the category is pinned and
    # there is no ALIGN to score, so the title has nothing left to say — only
    # the screen does (step 3.5).
    primary_res: Optional[Dict[str, Any]] = None
    p1_align: Optional[float] = None
    if not (correction and intention is None):
        p1_prompt = build_prompt(
            events_context,
            categories=categories,
            corrections=corrections,
            prior_classifications=prior_classifications,
            intention=intention,
        )
        p1_name, p1_off, p1_conf, p1_reason, p1_prod, p1_raw = ask_gpt(
            p1_prompt,
            settings.model,
            max_tokens=150,
            temperature=0.0,
            categories=categories,
        )
        p1_align = _parse_align(p1_raw) if intention else None
        primary_res = {
            "category_name": p1_name,
            "is_productive": p1_prod,
            "off": p1_off,
            "conf": p1_conf,
            "reason": p1_reason or "",
            "align": p1_align,
        }

    def mk_ret(cat_name, is_productive, off, conf, reason, vision_ran,
               vision_res=None, align=None, overrode: bool = False):
        """Assemble and cache a main-path verdict. `overrode`: vision replaced
        the primary's category, so the stored row has to explain itself."""
        if forced_align is not None:
            align = forced_align
        primary = primary_res
        if overrode and vision_res is not None:
            primary = _overridden_primary(primary_res, vision_res)
        result = {
            "final_category_name": cat_name,
            "final_is_productive": is_productive,
            "final_off": off,
            "final_conf": conf,
            "final_reason": reason,
            "final_align": align,
            "vision_ran": vision_ran,
            "vision_overrode": overrode,
            "primary": primary,
            "vision": vision_res,
        }
        if current_window:
            _cache_put(current_window, result, intention_id)
        return result

    # --- 3.5. Correction: pin the category, score ALIGN, let the screen look behind an open one ---
    if correction:
        cat_name, is_productive = _resolve_category(correction["corrected_category"], categories)
        pinned_off = 0.0 if is_productive else 1.0
        is_open = _correction_is_open(correction)
        want_align = intention is not None and forced_align is None

        # Vision runs for ALIGN on any correction under a session: a
        # corrected window is one whose title already misled the classifier
        # once, so scoring alignment from that same title is the weakest
        # possible evidence — "supplements prep - Google Docs" under
        # "outlining piq #2" scored 0.5, drift, while the screen showed the
        # very essay prompt named in the intention. On an open correction it
        # also gets to judge the category.
        vision_res = None
        if settings.critic.vision_enabled and (want_align or is_open):
            vision_res = _vision_review(
                events_context, settings, categories, intention,
                primary_category=cat_name, primary_off=pinned_off, primary_conf=1.0,
                primary_reason=(f"user correction — the user labelled this exact window "
                                f"title {cat_name}; the title-only call had been "
                                f"{correction['ai_category']}"),
            )
        v_align = vision_res["align"] if vision_res is not None else None
        align = v_align if v_align is not None else p1_align
        if forced_align is not None:
            align = forced_align
        align_src = ("ALIGN from screen" if v_align is not None
                     else "ALIGN from title" if align is not None else "no ALIGN")

        overrode = False
        why = ""
        if is_open and vision_res is not None:
            overrode, why = _vision_may_override(
                cat_name, vision_res["category_name"], vision_res["conf"],
                settings, categories, correction=correction,
            )
            if not overrode and why != "agrees":
                print(f"[vision] correction stands: {cat_name} — saw "
                      f"{vision_res['category_name']}, {why}")

        if not overrode:
            print(f"[correction] Exact match: '{correction['window_title'][:40]}' -> {cat_name}"
                  f" (category pinned, {align_src})")
            reason = (f"User correction: was {correction['ai_category']}, "
                      f"corrected to {cat_name} ({align_src})")
            # Report the human correction as the primary, not the LLM call we
            # made only to score ALIGN — sinks log primary conf/reason, and the
            # rejected category's reasoning would misexplain the pinned one.
            primary = {"category_name": cat_name, "is_productive": is_productive,
                       "off": 0.0, "conf": 1.0, "reason": "user correction",
                       "align": align}
            final_name, final_prod, final_off, final_conf = cat_name, is_productive, pinned_off, 1.0
        else:
            v_name = vision_res["category_name"]
            print(f"[vision] OVERRIDE correction: {cat_name} -> {v_name} ({why})")
            reason = (f"[Vision] {vision_res['reason']} — overrode the user correction "
                      f"({correction['ai_category']} -> {cat_name})")
            # The stored row now rests on the screen: vision's category, its
            # confidence (the human's 1.0 was never vision's certainty) and its
            # reason, tagged so the review tab shows a screen verdict and
            # precedent never mistakes it for a title call. None of this is
            # written back as a correction — the user's rule stays exactly as
            # it was, and the moment it changes it outranks the cache again.
            primary = {"category_name": v_name, "is_productive": vision_res["is_productive"],
                       "off": vision_res["off"], "conf": vision_res["conf"],
                       "reason": (f"[vision] {vision_res['reason']} (overrode user correction: "
                                  f"was {correction['ai_category']}, corrected to {cat_name})"),
                       "align": align}
            final_name, final_prod = v_name, vision_res["is_productive"]
            final_off, final_conf = vision_res["off"], vision_res["conf"]

        result = {
            "final_category_name": final_name,
            "final_is_productive": final_prod,
            "final_off": final_off,
            "final_conf": final_conf,
            "final_reason": reason,
            "final_align": align,
            "vision_ran": vision_res is not None,
            "vision_overrode": overrode,
            # The corrected category vision overruled, if it did: the session
            # cache serves this entry only while that exact correction stands.
            "overrode_correction": cat_name if overrode else None,
            "primary": primary,
            "vision": vision_res,
            # Keep what the title call would have said, for accuracy analysis.
            "align_primary": primary_res,
        }
        # The normal cache gate skips categories like Ambiguous; pin corrected
        # decisions regardless so neither call is repeated for 10 minutes. A
        # decision that made no call at all (open correction, no session, no
        # screenshot to be had) is not pinned, so the screen gets another try.
        if current_window and (primary_res is not None or vision_res is not None):
            _cache_put(current_window, result, intention_id, force=True)
        return result

    # Primary errored
    if p1_name in ("[ERROR]", "[PARSE_ERROR]"):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, primary_res["reason"], False)

    # --- 4. Vision second look? ---
    if not _should_run_vision(p1_prod, p1_conf, settings, p1_name):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False, align=p1_align)

    if p1_prod is True:
        direction = "verify-productive"
    elif p1_conf <= _DEFAULT_BY_ABSENCE_CONF_CEILING:
        direction = "second-guess-default"
    else:
        direction = "look-behind-ambiguous"

    # In trace mode, the live screen grab has nothing to do with the historical
    # event being evaluated. Skip the vision call but log that it would have fired
    # so the trace still reports the trigger rate and direction breakdown.
    # Get screenshot — trace fixture in trace mode, live grab otherwise
    if settings.trace.use_trace_file:
        b64_img = _load_trace_screenshot(events_context, settings)
        if b64_img is None:
            print(f"[vision] WOULD FIRE ({direction}): primary={p1_name}, conf={p1_conf:.2f} — no fixture screenshot")
            return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False, align=p1_align)
        source = "trace fixture"
    else:
        if current_window and not _screen_still_shows(current_window):
            print("[vision] Foreground changed since the title was polled — primary stands.")
            return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False, align=p1_align)
        b64_img = take_screenshot_b64()
        source = "live"

    if not b64_img:
        print("[vision] No screenshot available — primary stands.")
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False, align=p1_align)

    print(f"[vision] Triggered ({direction}, {source}): primary={p1_name}, conf={p1_conf:.2f}")

    v_name, v_off, v_conf, v_reason, v_prod, v_raw = ask_gpt_vision_verdict(
        b64_img,
        events_context,
        primary_category=p1_name,
        primary_off=p1_off,
        primary_conf=p1_conf,
        primary_reason=p1_reason,
        model=settings.critic.vision_model,
        max_tokens=settings.critic.vision_max_tokens,
        categories=categories,
        intention=intention,
    )
    v_align = _parse_align(v_raw) if intention else None
    vision_res = {
        "category_name": v_name,
        "is_productive": v_prod,
        "off": v_off,
        "conf": v_conf,
        "reason": v_reason or "",
        "align": v_align,
    }

    # Vision errored → primary stands
    if v_name in ("[ERROR]", "[PARSE_ERROR]"):
        print(f"[vision] Error ({v_reason}) — primary stands.")
        result = mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, True, vision_res, align=p1_align)
        # The verdict survives; the failure still has to reach the runner so
        # a screen check that has been dead for an hour gets reported.
        result["vision_error"] = v_reason if v_name == "[ERROR]" else "[PARSE_ERROR]"
        return result

    # Vision saw the screen; its ALIGN wins when present, else the primary's
    # stands. Alignment is judged from the screen on its own terms, so it
    # holds whether or not the category verdict below clears its bar.
    final_align = v_align if v_align is not None else p1_align

    # ...except when the window itself is the project's document. Vision
    # reads the prose, and prose has a topic: a personal statement about
    # composing reads as Composition (live incident 2026-10-07, ALIGN 0.05
    # and an off-intent alert on the essay itself). A window titled with the
    # project's name is that project's work whatever its paragraphs discuss,
    # so the screen may not score it lower than the title did, nor move it
    # out of the project's home category.
    anchor = _title_names_project(current_window, intention)
    if anchor:
        if v_align is not None and p1_align is not None and v_align < p1_align:
            print(f"[vision] title names the project \"{anchor}\" — keeping "
                  f"ALIGN {p1_align:.2f} over the screen's {v_align:.2f}")
            final_align = p1_align
        home = ((intention or {}).get("project") or {}).get("category") or ""
        if home and p1_name.lower() == home.lower() and v_name.lower() != home.lower():
            print(f"[vision] {p1_name} stands — the title names the project "
                  f"\"{anchor}\"; the screen's {v_name} is the document's topic")
            return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, True,
                          vision_res, align=final_align)

    if v_name.lower() == p1_name.lower():
        print(f"[vision] AGREE: {p1_name} (conf {p1_conf:.2f} -> {v_conf:.2f})")
        return mk_ret(v_name, v_prod, v_off, v_conf, f"[Vision] {v_reason}", True,
                      vision_res, align=final_align)

    override, why = _vision_may_override(p1_name, v_name, v_conf, settings, categories)
    if not override:
        print(f"[vision] {p1_name} stands — saw {v_name}, {why}")
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, True,
                      vision_res, align=final_align)

    print(f"[vision] OVERRIDE: {p1_name} -> {v_name} ({why})")
    return mk_ret(v_name, v_prod, v_off, v_conf, f"[Vision] {v_reason}", True,
                  vision_res, align=final_align, overrode=True)
