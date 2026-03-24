# monitor/decider.py
from __future__ import annotations

import base64, io, re, openai
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from monitor.config import Settings, DEFAULT_SETTINGS

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None
    print("[warn] Pillow not installed. Vision tiebreaker disabled. pip install Pillow")


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
    """Format events with clear separation between current window and history."""
    if not events:
        return "(no events)"

    _current_ts, current_win = events[-1]

    parts = [f">>> CURRENT WINDOW (classify this): {current_win}"]

    # Show recent windows (up to 5 before current, newest first)
    recent = events[max(0, len(events) - 6):-1]
    if recent:
        recent_str = "\n".join([f"  {ts}: {win}" for ts, win in reversed(recent)])
        parts.append(f"RECENT WINDOWS (newest first):\n{recent_str}")

    # Summarize older context
    older = events[:max(0, len(events) - 6)]
    if older:
        unique_older = list(dict.fromkeys([win for _, win in older]))
        parts.append(f"OLDER CONTEXT ({len(older)} events): {', '.join(unique_older[-5:])}")

    if keystroke_summary:
        parts.append(f"KEYSTROKE ACTIVITY (last 60s): {keystroke_summary}")

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


def build_critic_prompt(
    events: List[Tuple[str, str]],
    p1_summary: str,
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
) -> str:
    activity_str = _format_activity(events, keystroke_summary)

    template = _read_prompt_file("prompt-critic.txt")
    if not template:
        return (
            "You are a strict critic. Review the classification.\n"
            f"Initial model summary: {p1_summary}\n"
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
    filled = filled.replace("[insert here]", p1_summary)
    filled = filled.replace("[insert activity here]", activity_str)

    # Inject few-shot examples into critic too
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
        resp = openai.chat.completions.create(
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

def take_screenshot_b64() -> Optional[str]:
    if not ImageGrab:
        return None
    try:
        img = ImageGrab.grab()
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[warn] screenshot failed: {e}")
        return None


def ask_gpt_vision_category(
    screenshot_b64: str,
    events: List[Tuple[str, str]],
    primary_category: str,
    primary_reason: str,
    critic_category: str,
    critic_reason: str,
    *,
    model: str,
    max_tokens: int,
    categories: List[Dict] = None,
) -> Tuple[str, float, float]:
    """
    Vision tiebreaker for category disagreement.

    Returns:
        (category_name, off_score, conf)
    """
    activity_str = ", ".join([k for _, k in events])

    if categories:
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']}"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"

    prompt = f"""You are the final judge. Resolving a category disagreement.

AVAILABLE CATEGORIES:
{category_list}

Context: {activity_str}

Primary Model said: {primary_category}
Reason: {primary_reason}

Critic Model said: {critic_category}
Reason: {critic_reason}

Look at the screen. What is the user *actually* doing? Which category is most accurate?

Output strict single line format:
CATEGORY=<exact category name> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<explanation>"""

    try:
        resp = openai.chat.completions.create(
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
        cat_name, off_score, conf, reason, is_productive = _parse_llm_line(
            content, categories
        )
        return cat_name, off_score, conf

    except Exception as e:
        return "[ERROR]", 0.5, 0.5


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


# ==================== CRITIC LOGIC ====================

# Known off-task signals in window titles (lowercase for matching)
_OFF_TASK_SIGNALS = [
    "twitter", "reddit", "youtube", "discord", "minecraft",
    "steam", "netflix", "tiktok", "instagram", "twitch",
    "hulu", "prime video", "roblox", "epic games",
    "facebook", "snapchat", "telegram",
]

# Categories that are productive
_PRODUCTIVE_CATS = ["deep work", "homework", "research & learning", "communication"]


def _should_run_critic(
    cat_name: str,
    conf: float,
    settings: Settings,
    events: list = None,
) -> bool:
    """Decide if critic should run based on category classification."""
    c = settings.critic
    if not c.enabled:
        return False

    # Always run critic on low confidence
    if conf <= c.trigger_conf_max:
        return True

    # Run on ambiguous categories
    ambiguous_categories = ["research & learning", "communication", "ambiguous"]
    if cat_name.lower() in ambiguous_categories:
        return True

    # Run critic when current window looks off-task but model said productive
    if events and cat_name.lower() in _PRODUCTIVE_CATS:
        current_win = events[-1][1].lower() if events else ""
        if any(sig in current_win for sig in _OFF_TASK_SIGNALS):
            print(f"[critic] Triggered: current window contains off-task signal but model said '{cat_name}'")
            return True

    return False


# ==================== MAIN DECISION ====================

def decide_with_critic(
    events_context: List[Tuple[str, str]],
    keystroke_summary: str = "",
    settings: Settings = DEFAULT_SETTINGS,
    window_dwell_info: Optional[Dict[str, Any]] = None,
    categories: List[Dict] = None,
    corrections: List[Dict] = None,
    prior_classifications: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Category-based decision with optional critic verification.

    Decision pipeline (in order):
        1. User correction exact-match → skip LLM
        2. System window shortcut → skip LLM
        3. Prior classification match → skip LLM (tier 1 cache)
        4. LLM call with few-shot hints from corrections + priors (tier 2)

    Returns dict with:
        final_category_name, final_is_productive, final_off, final_conf,
        final_reason, critic_ran, primary, critic, vision
    """
    # --- 1. Exact-match shortcut: skip LLM if user already corrected this window ---
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
                "primary": {"category_name": cat_name, "is_productive": is_productive, "off": 0.0, "conf": 1.0, "reason": "user correction"},
                "critic": None,
                "vision": None,
            }

    # --- 2. System window shortcut: skip LLM for OS-level windows ---
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

    # --- 3. Prior classification match: skip LLM if we've seen this title before ---
    if prior_classifications and events_context:
        match = _match_prior_classification(events_context[-1][1], prior_classifications)
        if match:
            cat_name = match["category_name"]
            is_productive = None
            if categories:
                for cat in categories:
                    if cat["name"].lower() == cat_name.lower():
                        cat_name = cat["name"]
                        is_productive = cat.get("is_productive")
                        break
            print(f"[prior] Cache hit: '{match['window_title'][:40]}' -> {cat_name} (skipping LLM)")
            return {
                "final_category_name": cat_name,
                "final_is_productive": is_productive,
                "final_off": 0.0 if is_productive else (1.0 if is_productive is False else 0.5),
                "final_conf": 0.95,
                "final_reason": f"[Prior] previously classified as {cat_name} (conf={match['confidence']:.2f})",
                "critic_ran": False,
                "primary": {"category_name": cat_name, "is_productive": is_productive,
                             "off": 0.0, "conf": 0.95, "reason": "prior classification cache"},
                "critic": None,
                "vision": None,
            }

    # --- 4. LLM call with few-shot hints (tier 2) ---

    # Build prompt with categories, corrections, and prior classifications
    p1_prompt = build_prompt(
        events_context,
        keystroke_summary,
        categories=categories,
        corrections=corrections,
        prior_classifications=prior_classifications,
    )

    # Primary model call
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

    def mk_ret(cat_name, is_productive, off, conf, reason, critic_ran, critic_res=None, vision_res=None):
        return {
            "final_category_name": cat_name,
            "final_is_productive": is_productive,
            "final_off": off,
            "final_conf": conf,
            "final_reason": reason,
            "critic_ran": critic_ran,
            "primary": primary_res,
            "critic": critic_res,
            "vision": vision_res,
        }

    # Handle errors
    if p1_name in ("[ERROR]", "[PARSE_ERROR]"):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, primary_res["reason"], False)

    # Check if critic should run
    if not _should_run_critic(p1_name, p1_conf, settings, events=events_context):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, False)

    # Run critic
    p1_summary = f"CATEGORY={p1_name} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(
        events_context,
        p1_summary,
        keystroke_summary,
        categories=categories,
        corrections=corrections,
        prior_classifications=prior_classifications,
    )

    c_name, c_off, c_conf, c_reason, c_prod, c_raw = ask_gpt(
        c_prompt,
        settings.critic_model(),
        max_tokens=settings.critic.max_tokens,
        temperature=settings.critic.temperature,
    )

    critic_res = {
        "category_name": c_name,
        "is_productive": c_prod,
        "off": c_off,
        "conf": c_conf,
        "reason": c_reason or "",
    }

    # Handle critic errors
    if c_name in ("[ERROR]", "[PARSE_ERROR]"):
        return mk_ret(p1_name, p1_prod, p1_off, p1_conf, primary_res["reason"], True, critic_res)

    # Check for disagreement
    critic_disagrees = (c_name.lower() != p1_name.lower())

    if critic_disagrees and settings.critic.vision_enabled:
        print(f"[info] Category disagreement: primary={p1_name}, critic={c_name}. Trying vision tiebreak...")
        b64_img = take_screenshot_b64()
        if b64_img:
            v_name, v_off, v_conf = ask_gpt_vision_category(
                b64_img,
                events_context,
                p1_name,
                p1_reason,
                c_name,
                c_reason,
                model=settings.critic.vision_model,
                max_tokens=settings.critic.vision_max_tokens,
                categories=categories,
            )
            if v_name not in ("[ERROR]", "[PARSE_ERROR]"):
                vision_res = {
                    "category_name": v_name,
                    "off": v_off,
                    "conf": v_conf,
                }
                v_prod = None
                if categories:
                    for cat in categories:
                        if cat["name"].lower() == v_name.lower():
                            v_prod = cat.get("is_productive")
                            break

                return mk_ret(v_name, v_prod, v_off, v_conf, f"[Vision] {c_reason}", True, critic_res, vision_res)

    # No vision or vision failed — use critic if it disagrees
    if critic_disagrees:
        final_off = max(p1_off, c_off)
        return mk_ret(
            c_name, c_prod, final_off, c_conf,
            f"[Critic] {critic_res['reason'] or primary_res['reason']}",
            True, critic_res,
        )

    # Critic agrees — use primary
    return mk_ret(p1_name, p1_prod, p1_off, p1_conf, p1_reason, True, critic_res)