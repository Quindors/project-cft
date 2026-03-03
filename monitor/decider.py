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

def build_prompt(
    events: List[Tuple[str, str]],
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    user_goals: str = "",
) -> str:
    activity_str = _format_activity(events, keystroke_summary)

    template = _read_prompt_file("prompt-main.txt")
    if not template:
        return (
            "You are a focus judge. Categorize the user's activity.\n"
            f"Recent activity: {activity_str}{extra}\n"
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
    filled = filled.replace("[insert goals here]", user_goals or "(no goals specified)")
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", activity_str)
    return filled


def build_critic_prompt(
    events: List[Tuple[str, str]],
    p1_summary: str,
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    user_goals: str = "",
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
    filled = filled.replace("[insert goals here]", user_goals or "(no goals specified)")
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", p1_summary)
    filled = filled.replace("[insert activity here]", activity_str)
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
    user_goals: str = "",
) -> Dict[str, Any]:
    """
    Category-based decision with optional critic verification.

    Returns dict with:
        final_category_name, final_is_productive, final_off, final_conf,
        final_reason, critic_ran, primary, critic, vision
    """

    # Build prompt with categories and goals
    p1_prompt = build_prompt(
        events_context,
        keystroke_summary,
        categories=categories,
        user_goals=user_goals,
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
        user_goals=user_goals,
    )

    c_name, c_off, c_conf, c_reason, c_prod, c_raw = ask_gpt(
        c_prompt,
        settings.critic_model(),
        max_tokens=settings.critic.max_tokens,
        temperature=settings.critic.temperature,
        categories=categories,
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