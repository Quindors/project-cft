# monitor/decider.py
from __future__ import annotations

import re, openai
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.screenshot_capture import take_screenshot_b64

import os, base64

# Set by configure_llm() at boot. decider uses this instead of module-level openai.*
_llm_client = None


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
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


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

def _format_activity(events: List[Tuple[str, str]], keystroke_summary: str = "") -> str:
    """Format events with clear separation between current window and history.

    keystroke_summary is accepted for backward compatibility but no longer used —
    the pipeline is window-title + screenshot only.
    """
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

def _match_correction(current_window: str, corrections: List[Dict]) -> Optional[Dict]:
    """Check if current window exactly matches a known correction."""
    if not corrections:
        return None
    clean = current_window.strip()
    # Strip WIN: prefix for matching
    if clean.upper().startswith("WIN:"):
        clean = clean[4:]
    clean_lower = clean.lower()
    for c in corrections:
        if c["window_title"].lower() == clean_lower:
            return c
    return None


def _match_prior_classification(current_window: str, prior_classifications: List[Dict]) -> Optional[Dict]:
    """Check if current window matches a prior high-confidence classification.

    Uses normalized title matching (strips WIN:, notification counts, lowercases).
    Returns the matching dict or None.
    """
    if not prior_classifications:
        return None
    normalized = _normalize_title(current_window)
    for p in prior_classifications:
        if p["normalized_title"] == normalized:
            return p
    return None


def _build_fewshot_prompt(
    current_window: str,
    corrections: List[Dict],
    prior_classifications: List[Dict] = None,
    max_examples: int = 8,
) -> str:
    """Build a few-shot prompt section from corrections AND prior classifications.

    Human corrections take priority. Prior classifications fill remaining slots,
    scored by keyword overlap with the current window.
    """
    if not corrections and not prior_classifications:
        return ""

    clean = current_window.strip()
    if clean.upper().startswith("WIN:"):
        clean = clean[4:]
    current_words = set(clean.lower().split())

    # --- Collect correction examples (highest priority) ---
    correction_lines = []
    used_titles = set()

    if corrections:
        scored = []
        for c in corrections:
            corr_words = set(c["window_title"].lower().split())
            overlap = len(current_words & corr_words)
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

    # --- Collect prior classification examples (fill remaining slots) ---
    prior_lines = []
    remaining_slots = max_examples - len(correction_lines)

    if prior_classifications and remaining_slots > 0:
        scored_prior = []
        for p in prior_classifications:
            # Skip if already covered by a correction
            if p["window_title"].lower() in used_titles:
                continue
            # Skip exact match with current window (that's handled by tier 1)
            if p["normalized_title"] == _normalize_title(current_window):
                continue
            prior_words = set(p["window_title"].lower().split())
            overlap = len(current_words & prior_words)
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

    if prior_lines:
        all_lines.append("\nSIMILAR WINDOWS PREVIOUSLY CLASSIFIED:")
        all_lines.extend(prior_lines)

    if all_lines:
        all_lines.append(
            "\nIf the current window is similar to a corrected or previously classified example, follow the pattern."
        )

    return "\n".join(all_lines) if all_lines else ""


def build_prompt(
    events: List[Tuple[str, str]],
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
) -> str:
    activity_str = _format_activity(events, keystroke_summary)

    template = _read_prompt_file("prompt-main.txt")
    if not template:
        return (
            "You are a focus judge. Categorize the user's activity.\n"
            f"Recent activity: {activity_str}\n"
            "Output: CATEGORY=<category_name> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )

    if categories:
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']} "
            f"({'PRODUCTIVE' if cat.get('is_productive') else 'NOT PRODUCTIVE'})"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"

    filled = template
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", activity_str)

    # Inject few-shot examples (corrections + prior classifications)
    if events:
        current_win = events[-1][1]
        fewshot_section = _build_fewshot_prompt(
            current_win, corrections or [], prior_classifications
        )
        if fewshot_section:
            filled = filled + "\n\n" + fewshot_section

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

    # Extract REASON
    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    reason = m_reason.group(1).strip() if m_reason else ""

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
) -> Tuple[str, float, float, str, Optional[bool]]:
    """Vision tiebreaker. Reviews the primary's window-title-only call against
    the screenshot, then either agrees or overrides.

    Returns (category_name, off_score, conf, reason, is_productive) — same
    shape as the primary call's parse output.
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

    prompt = f"""You are reviewing a window-title-based classification using the screen contents as ground truth.

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
      - Title says "Wikipedia: Calculus" but screen shows an unrelated article (celebrity gossip, sports) → Other Off-Task

  CASE B — Primary said "Other Off-Task" or "Ambiguous":
    The primary defaulted because the window title had no specific productive evidence (bare "Chrome", "VS Code", "New Tab", etc.). PROMOTE to a productive category ONLY if the screenshot clearly shows a productive activity.
    Examples:
      - Title is "Google Chrome" but screen shows a calculus textbook PDF → Homework
      - Title is "Visual Studio Code" but screen shows a named project's source code → Deep Work
      - Title is "New Tab" but screen shows Wolfram Alpha computing a derivative → Research & Learning

DEFAULT BEHAVIOR — AGREE WITH PRIMARY:
  If the screenshot is unclear, low-resolution, shows a lock/idle screen, doesn't show a recognizable activity, or is consistent with the primary's call — AGREE. Do not override on uncertainty. The primary's window-title evidence stands.

Output format (strict, single line):
CATEGORY=<exact category name from list above> | OFF_SCORE=<0.0-1.0> | CONF=<0.0-1.0> | REASON=<brief explanation citing what you saw on screen>"""

    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _parse_llm_line(content, categories)

    except Exception as e:
        return "[ERROR]", 0.5, 0.5, str(e), None

# ==================== SYSTEM WINDOW DETECTION ====================

# Known system/utility windows (lowercase substrings for matching)
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


def _is_system_window(title: str) -> bool:
    """Check if window title matches a known system/utility window."""
    clean = title.strip()
    if clean.upper().startswith("WIN:"):
        clean = clean[4:]
    lower = clean.lower()
    return any(pattern in lower for pattern in _SYSTEM_WINDOWS)


# ==================== VISION TIEBREAKER LOGIC ====================

# CONF ceiling for the default-by-absence band. Matches the band defined in
# prompt-main.txt (Rule 5 scoring): default-by-absence CONF = 0.50–0.70.
_DEFAULT_BY_ABSENCE_CONF_CEILING = 0.70


def _should_run_vision(
    is_productive: Optional[bool],
    conf: float,
    settings: Settings,
) -> bool:
    """Vision tiebreaker fires in two directions:

      Direction A — VERIFY PRODUCTIVE:
        Primary said the user is being productive (is_productive=True).
        Vision checks whether the screen actually shows that.

      Direction B — SECOND-GUESS DEFAULT-BY-ABSENCE:
        Primary said off-task or neutral (is_productive in {False, None})
        with CONF in the default-by-absence band (≤ 0.70). The primary
        likely fell back because the title had no specific signal. Vision
        checks whether the screen reveals productive activity the title
        couldn't.

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

    return False

# ==================== MAIN DECISION ====================

def decide_with_critic(
    events_context: List[Tuple[str, str]],
    keystroke_summary: str = "",          # kept for backward compat, unused
    settings: Settings = DEFAULT_SETTINGS,
    window_dwell_info: Optional[Dict[str, Any]] = None,  # kept for backward compat
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
) -> Dict[str, Any]:
    """Window-title primary + two-direction vision tiebreaker.

    Decision pipeline:
      1. User correction exact-match  → skip LLM
      2. System window shortcut       → skip LLM
      3. Primary text-LLM call (window title + history, no screenshot)
      4. Vision tiebreaker fires if EITHER:
           (A) primary returned a productive category, OR
           (B) primary returned Other Off-Task/Ambiguous with CONF ≤ 0.70
         Otherwise primary stands.

    Returns: dict with final_category_name, final_is_productive, final_off,
    final_conf, final_reason, critic_ran (= vision_ran for back-compat),
    primary, critic (= None now), vision.
    """
    # --- 1. User correction exact-match ---
    if corrections and events_context:
        match = _match_correction(events_context[-1][1], corrections)
        if match:
            cat_name = match["corrected_category"]
            is_productive = None
            if categories:
                for cat in categories:
                    if cat["name"].lower() == cat_name.lower():
                        cat_name = cat["name"]
                        is_productive = cat.get("is_productive")
                        break
            print(f"[correction] Exact match: '{match['window_title'][:40]}' -> {cat_name} (skipping LLM)")
            return {
                "final_category_name": cat_name,
                "final_is_productive": is_productive,
                "final_off": 0.0 if is_productive else 1.0,
                "final_conf": 1.0,
                "final_reason": f"User correction: was {match['ai_category']}, corrected to {cat_name}",
                "critic_ran": False,
                "primary": {"category_name": cat_name, "is_productive": is_productive,
                            "off": 0.0, "conf": 1.0, "reason": "user correction"},
                "critic": None,
                "vision": None,
            }

    # --- 2. System window shortcut ---
    if events_context and _is_system_window(events_context[-1][1]):
        return {
            "final_category_name": "System",
            "final_is_productive": None,
            "final_off": 0.5,
            "final_conf": 1.0,
            "final_reason": "System/utility window — not a user activity",
            "critic_ran": False,
            "primary": {"category_name": "System", "is_productive": None,
                        "off": 0.5, "conf": 1.0, "reason": "system window shortcut"},
            "critic": None,
            "vision": None,
        }

    # --- 3. Primary text-LLM call ---
    p1_prompt = build_prompt(
        events_context,
        categories=categories,
        corrections=corrections,
        prior_classifications=prior_classifications,
    )
    p1_name, p1_off, p1_conf, p1_reason, p1_prod, p1_raw = ask_gpt(
        p1_prompt,
        settings.model,
        max_tokens=150,
        temperature=0.0,
        categories=categories,
    )
    primary_res = {
        "category_name": p1_name,
        "is_productive": p1_prod,
        "off": p1_off,
        "conf": p1_conf,
        "reason": p1_reason or "",
    }

    def mk_ret(cat_name, is_productive, off, conf, reason, vision_ran, vision_res=None):
        return {
            "final_category_name": cat_name,
            "final_is_productive": is_productive,
            "final_off": off,
            "final_conf": conf,
            "final_reason": reason,
            "critic_ran": vision_ran,   # back-compat alias
            "primary": primary_res,
            "critic": None,
            "vision": vision_res,
        }

    # Primary errored
    if p1_name in ("[ERROR]", "[PARSE_ERROR]"):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, primary_res["reason"], False)

    # --- 4. Vision tiebreaker? ---
    if not _should_run_vision(p1_prod, p1_conf, settings):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False)

    direction = "verify-productive" if p1_prod is True else "second-guess-default"

    # In trace mode, the live screen grab has nothing to do with the historical
    # event being evaluated. Skip the vision call but log that it would have fired
    # so the trace still reports the trigger rate and direction breakdown.
    # Get screenshot — trace fixture in trace mode, live grab otherwise
    if settings.trace.use_trace_file:
        b64_img = _load_trace_screenshot(events_context, settings)
        if b64_img is None:
            print(f"[vision] WOULD FIRE ({direction}): primary={p1_name}, conf={p1_conf:.2f} — no fixture screenshot")
            return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False)
        source = "trace fixture"
    else:
        b64_img = take_screenshot_b64()
        source = "live"

    if not b64_img:
        print("[vision] No screenshot available — primary stands.")
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False)

    print(f"[vision] Triggered ({direction}, {source}): primary={p1_name}, conf={p1_conf:.2f}")

    v_name, v_off, v_conf, v_reason, v_prod = ask_gpt_vision_verdict(
        b64_img,
        events_context,
        primary_category=p1_name,
        primary_off=p1_off,
        primary_conf=p1_conf,
        primary_reason=p1_reason,
        model=settings.critic.vision_model,
        max_tokens=settings.critic.vision_max_tokens,
        categories=categories,
    )
    vision_res = {
        "category_name": v_name,
        "is_productive": v_prod,
        "off": v_off,
        "conf": v_conf,
        "reason": v_reason or "",
    }

    # Vision errored → primary stands
    if v_name in ("[ERROR]", "[PARSE_ERROR]"):
        print(f"[vision] Error ({v_reason}) — primary stands.")
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, True, vision_res)

    # Vision verdict wins
    overrode = (v_name.lower() != p1_name.lower())
    tag = "OVERRIDE" if overrode else "AGREE"
    print(f"[vision] {tag}: {p1_name} -> {v_name} (conf {p1_conf:.2f} -> {v_conf:.2f})")
    return mk_ret(v_name, v_prod, v_off, v_conf, f"[Vision] {v_reason}", True, vision_res)