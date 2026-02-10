# monitor/decider.py
from __future__ import annotations

import base64, io, re, openai
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

from monitor.config import Settings, DEFAULT_SETTINGS

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    print("[warn] Pillow not installed. Vision tiebreaker disabled. pip install Pillow")


# ==================== NEW: FACTOR COMPUTATION ====================

def compute_factor_scores(
    events_context: List[Tuple[str, str]],
    keystroke_summary: str,
    settings: Settings,
    window_dwell_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Compute individual factor scores for composite decision.
    All scores are in range [-1, 1] where:
      - Positive = ON-TASK signal
      - Negative = OFF-TASK signal
      - 0 = Neutral
    
    Args:
        events_context: List of (timestamp, window_title) tuples
        keystroke_summary: Summary of recent keystroke activity
        settings: Configuration settings
        window_dwell_info: Optional dict with dwell time tracking
    
    Returns:
        Dict of factor_name -> score
    """
    factors = {}
    
    # Factor 1: Window Relevance
    factors["window_relevance"] = _compute_window_relevance(events_context, settings)
    
    # Factor 2: Dwell Time (requires temporal context)
    if window_dwell_info:
        factors["dwell_time"] = _compute_dwell_factor(events_context, window_dwell_info)
    else:
        factors["dwell_time"] = 0.0
    
    # Factor 3: Keystroke Activity
    factors["keystroke_activity"] = _compute_keystroke_factor(keystroke_summary)
    
    # Factor 4: Trajectory
    factors["trajectory"] = _compute_trajectory_factor(events_context, settings)
    
    # Factor 5: Risky Keywords
    factors["risky_keywords"] = _compute_risky_keyword_factor(events_context, settings)
    
    return factors

def aggregate_factor_score(factors: Dict[str, float]) -> Tuple[float, str]:
    """
    Aggregate individual factor scores into a composite decision score.
    
    Returns:
        (aggregate_score, explanation)
        aggregate_score in [-1, 1] where:
            > 0.3: suggests ON-TASK
            < -0.3: suggests OFF-TASK
            [-0.3, 0.3]: uncertain
    """
    
    # These weights can be tuned later
    weights = {
        "window_relevance": 0.30,
        "dwell_time": 0.20,
        "keystroke_activity": 0.20,
        "trajectory": 0.20,
        "risky_keywords": 0.10,
    }
    
    # Compute weighted sum
    total_score = 0.0
    explanations = []
    
    for factor_name, weight in weights.items():
        score = factors.get(factor_name, 0.0)
        contribution = score * weight
        total_score += contribution
        
        # Build explanation for significant factors
        if abs(score) > 0.3:
            direction = "positive" if score > 0 else "negative"
            strength = "strong" if abs(score) > 0.7 else "moderate"
            explanations.append(
                f"{factor_name.replace('_', ' ')}: {strength} {direction} ({score:+.2f})"
            )
    
    explanation = "; ".join(explanations) if explanations else "all factors neutral"
    
    return total_score, explanation


def decide_hybrid_simple(
    factors: Dict[str, float],
    llm_label: str,
    llm_off_score: float,
    llm_confidence: float,
    llm_reason: str,
    off_threshold: float = 0.60,
) -> Tuple[str, str, float, float]:
    """
    Simple hybrid decision: let factors override LLM only when very confident.
    
    Conservative approach:
        - If factors are decisive (|score| > 0.7), trust them
        - Otherwise, trust LLM
    
    Returns:
        (final_label, final_reason, final_off_score, final_confidence)
    """
    
    # Get factor-based score
    agg_score, factor_explanation = aggregate_factor_score(factors)
    
    # Convert aggregate score to OFF_SCORE scale [0, 1]
    # agg_score in [-1, 1] → map to [0, 1]
    # -1 (fully off-task) = 1.0, +1 (fully on-task) = 0.0
    factor_off_score = (1.0 - agg_score) / 2.0
    
    # Determine factor-based label
    if agg_score > 0.3:
        factor_label = "ON-TASK"
        factor_confidence = min(0.95, 0.6 + (agg_score - 0.3) * 0.5)
    elif agg_score < -0.3:
        factor_label = "OFF-TASK"
        factor_confidence = min(0.95, 0.6 + abs(agg_score + 0.3) * 0.5)
    else:
        factor_label = "UNCERTAIN"
        factor_confidence = 0.5
    
    # Decision logic: Only override LLM if factors are VERY strong (|score| > 0.7)
    DECISIVE_THRESHOLD = 0.7
    
    if abs(agg_score) > DECISIVE_THRESHOLD:
        # Factors are decisive - use them
        return (
            factor_label,
            f"[Factors decisive] {factor_explanation}",
            factor_off_score,
            factor_confidence,
        )
    else:
        # Factors uncertain - use LLM
        return (
            llm_label,
            f"{llm_reason} (factors: {agg_score:+.2f})",
            llm_off_score,
            llm_confidence,
        )


def _compute_window_relevance(
    events_context: List[Tuple[str, str]],
    settings: Settings
) -> float:
    """
    Score based on how task-related the window titles are.
    
    Returns:
        +0.8 to +1.0: Highly productive apps (IDE, docs, homework)
        +0.3 to +0.5: Neutral/ambiguous (browser, general tools)
        -0.5 to -1.0: Clearly distractive apps
    """
    if not events_context:
        return 0.0
    
    # Categorize windows
    productive_keywords = [
        "visual studio code", "vs code", "pycharm", "intellij",
        "onenote", "notion", "homework", "assignment",
        "terminal", "cmd", "powershell",
        "pdf", "acrobat", "chapter",
        "google sheets", "excel", "spreadsheet",
        "documentation", "docs", "reference"
    ]
    
    neutral_keywords = [
        "chrome", "firefox", "edge", "browser",
        "new tab", "search"
    ]
    
    # settings.risky_keywords already defined as distractive
    
    scores = []
    for _, key in events_context:
        key_lower = key.lower()
        
        # Check productive patterns
        if any(kw in key_lower for kw in productive_keywords):
            scores.append(0.9)
        # Check distractive patterns
        elif any(kw in key_lower for kw in settings.risky_keywords):
            scores.append(-0.9)
        # Check neutral
        elif any(kw in key_lower for kw in neutral_keywords):
            scores.append(0.2)  # Slight positive (could be research)
        # Unknown/ambiguous
        else:
            # Check for educational keywords
            if any(word in key_lower for word in ["khan", "wolfram", "desmos", "wikipedia", "learn", "tutorial"]):
                scores.append(0.6)
            else:
                scores.append(0.0)
    
    # Weight recent windows more heavily
    if len(scores) > 1:
        weights = [0.5 ** (len(scores) - 1 - i) for i in range(len(scores))]
        weight_sum = sum(weights)
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / weight_sum
        return weighted_avg
    
    return scores[0] if scores else 0.0


def _compute_dwell_factor(
    events_context: List[Tuple[str, str]],
    window_dwell_info: Dict[str, Any]
) -> float:
    """
    Score based on how long the user has been on current window.
    
    Dwell patterns:
        - 0-2 min: Neutral (0.0)
        - 2-5 min with high activity: Positive (+0.3 to +0.5)
        - 2-5 min with low activity: Slightly negative (-0.2)
        - 5-10 min with low activity: Negative (-0.5)
        - 10+ min with low activity: Very negative (-0.8) [rabbit hole]
    
    Returns:
        Score in [-1, 1]
    """
    if not events_context:
        return 0.0
    
    # Get most recent window
    _, current_window = events_context[-1]
    
    # Get dwell info
    dwell_seconds = window_dwell_info.get("current_dwell_seconds", 0)
    recent_keystroke_count = window_dwell_info.get("recent_keystroke_count", 0)
    
    # Compute activity level (keystrokes per minute)
    dwell_minutes = dwell_seconds / 60.0
    kpm = recent_keystroke_count / dwell_minutes if dwell_minutes > 0 else 0
    
    # Scoring logic
    if dwell_minutes < 2:
        return 0.0  # Too early to judge
    elif dwell_minutes < 5:
        if kpm > 30:  # Active work
            return 0.4
        elif kpm > 10:  # Some activity
            return 0.1
        else:  # Passive
            return -0.2
    elif dwell_minutes < 10:
        if kpm > 20:  # Sustained work
            return 0.5
        elif kpm > 5:  # Some work
            return 0.0
        else:  # Likely distracted
            return -0.5
    else:  # 10+ minutes
        if kpm > 15:  # Deep work
            return 0.6
        else:  # Rabbit hole / passive consumption
            return -0.8
    
    return 0.0


def _compute_keystroke_factor(keystroke_summary: str) -> float:
    """
    Score based on keystroke activity patterns.
    
    Indicators:
        - High KPM (>60): +0.7
        - Moderate KPM (20-60): +0.3
        - Low KPM (<20): -0.3
        - Idle (0 keys): -0.6
        - Special keys (Ctrl+S, etc): +0.2 bonus
    
    Returns:
        Score in [-1, 1]
    """
    if not keystroke_summary or "Idle" in keystroke_summary:
        return -0.6
    
    # Parse keystroke count from summary
    # Expected format: "45 keys typed. Special: [Key.ctrl, Key.enter]"
    try:
        count_match = re.search(r"(\d+)\s+keys", keystroke_summary)
        if not count_match:
            return 0.0
        
        count = int(count_match.group(1))
        
        # Assume 60-second window (per keystroke_summary in sources.py)
        kpm = count
        
        # Base score from KPM
        if kpm > 60:
            score = 0.7
        elif kpm > 40:
            score = 0.5
        elif kpm > 20:
            score = 0.3
        elif kpm > 10:
            score = 0.0
        elif kpm > 5:
            score = -0.2
        else:
            score = -0.4
        
        # Bonus for productive special keys
        productive_keys = ["ctrl", "alt", "tab", "enter", "backspace"]
        if any(key in keystroke_summary.lower() for key in productive_keys):
            score += 0.2
        
        # Cap at 1.0
        return min(1.0, score)
    
    except Exception:
        return 0.0


def _compute_trajectory_factor(
    events_context: List[Tuple[str, str]],
    settings: Settings
) -> float:
    """
    Analyze sequence direction to detect drift patterns.
    
    Patterns:
        - Work → Docs → Work: +0.7 (productive loop)
        - Work → Neutral → Work: +0.3 (brief check)
        - Work → Distraction (single): -0.2 (quick check)
        - Work → Distraction → Distraction: -0.7 (drift)
        - Distraction → Distraction → Distraction: -0.9 (deep off-task)
        - Work → Educational → Educational: +0.4 (research, but watch for drift)
    
    Returns:
        Score in [-1, 1]
    """
    if len(events_context) < 2:
        return 0.0
    
    # Categorize each window
    categories = []
    for _, key in events_context:
        key_lower = key.lower()
        
        # Productive work
        if any(kw in key_lower for kw in ["vs code", "visual studio", "terminal", "cmd", "homework", "assignment", "onenote"]):
            categories.append("WORK")
        # Educational (could be productive or rabbit hole)
        elif any(kw in key_lower for kw in ["khan", "wolfram", "desmos", "wikipedia", "tutorial", "documentation"]):
            categories.append("EDUCATIONAL")
        # Clear distraction
        elif any(kw in key_lower for kw in settings.risky_keywords):
            categories.append("DISTRACTION")
        # Neutral
        else:
            categories.append("NEUTRAL")
    
    # Pattern analysis (last 3-4 windows)
    recent = categories[-min(4, len(categories)):]
    
    # Count consecutive patterns
    work_count = recent.count("WORK")
    distraction_count = recent.count("DISTRACTION")
    educational_count = recent.count("EDUCATIONAL")
    
    # Detect patterns
    if len(recent) >= 3:
        # Deep distraction: 3+ consecutive distractions
        if distraction_count >= 3:
            return -0.9
        
        # Drift pattern: WORK → EDUCATIONAL → EDUCATIONAL → (no return)
        if recent[0] == "WORK" and educational_count >= 2 and recent[-1] != "WORK":
            return -0.5  # Possible rabbit hole
        
        # Drift to distraction: WORK → X → DISTRACTION → DISTRACTION
        if recent[0] == "WORK" and distraction_count >= 2:
            return -0.7
        
        # Productive loop: WORK appears at start and end
        if recent[0] == "WORK" and recent[-1] == "WORK":
            return 0.7
        
        # Research pattern: WORK → EDUCATIONAL → WORK
        if "WORK" in recent and "EDUCATIONAL" in recent and recent[-1] == "WORK":
            return 0.5
    
    # Single distraction among work
    if work_count >= 2 and distraction_count == 1:
        return -0.2  # Quick check, not too bad
    
    # Sustained work
    if work_count >= len(recent) * 0.6:
        return 0.6
    
    # Mostly distractions
    if distraction_count >= len(recent) * 0.5:
        return -0.6
    
    # Mixed/unclear
    return 0.0


def _compute_risky_keyword_factor(
    events_context: List[Tuple[str, str]],
    settings: Settings
) -> float:
    """
    Direct check for known distractive sites/apps.
    
    Returns:
        -0.9: Multiple risky keywords
        -0.7: Single risky keyword in recent window
        0.0: No risky keywords
    """
    if not events_context:
        return 0.0
    
    # Check last 3 windows
    recent = events_context[-min(3, len(events_context)):]
    
    risky_count = 0
    for _, key in recent:
        key_lower = key.lower()
        if any(kw in key_lower for kw in settings.risky_keywords):
            risky_count += 1
    
    if risky_count >= 2:
        return -0.9
    elif risky_count == 1:
        return -0.7
    else:
        return 0.0


# ==================== MODIFIED: PARSING ====================

# Updated _parse_llm_line() function for decider.py

def _parse_llm_line(
    content: str, 
    off_threshold: float,
    categories: List[Dict] = None
) -> Tuple[str, str, float, float, str, bool]:
    """
    Parse LLM response to extract category and other info.
    
    Args:
        content: Raw LLM response text
        off_threshold: Threshold for OFF-TASK (kept for backwards compatibility)
        categories: List of available categories
    
    Returns:
        (category_id, category_name, off_score, conf, reason, is_productive)
    """
    content = (content or "").strip()

    # Extract CATEGORY instead of LABEL
    m_category = re.search(r"CATEGORY\s*=\s*(.+?)(?:\||$)", content, flags=re.I)
    if m_category:
        category_name = m_category.group(1).strip().strip('"').strip("'")
    else:
        # Fallback: try to find category name in content
        category_name = None
    
    # Extract OFF_SCORE and CONF (same as before)
    m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    
    off_score = float(m_off.group(1)) if m_off else 0.5
    conf = float(m_conf.group(1)) if m_conf else 0.5
    
    # Clamp values
    off_score = min(max(off_score, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)
    
    # Extract REASON
    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    reason = m_reason.group(1).strip() if m_reason else ""
    
    # Look up category in categories list
    category_id = None
    is_productive = None
    
    if categories and category_name:
        # Find matching category (case-insensitive)
        for cat in categories:
            if cat['name'].lower() == category_name.lower():
                category_id = cat['category_id']
                category_name = cat['name']  # Use exact casing from sheet
                is_productive = cat.get('is_productive', None)
                break
    
    # If category not found, mark as unknown
    if category_id is None and category_name:
        category_id = "unknown"
        is_productive = None
        print(f"[warn] Category '{category_name}' not found in categories list")
    elif category_id is None:
        category_id = "error"
        category_name = "[PARSE_ERROR]"
        is_productive = None
    
    return category_id, category_name, off_score, conf, reason, is_productive

# ==================== MODIFIED: ASK GPT ====================

# Updated ask_gpt() function for decider.py
# Replace the existing ask_gpt() function with this

def ask_gpt(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    off_threshold: float,
    categories: List[Dict] = None,
) -> Tuple[str, str, float, float, str, bool, str]:
    """
    Call GPT with prompt and parse response for category.
    
    Args:
        prompt: Filled-in prompt string
        model: Model name (e.g., "gpt-4o-mini")
        max_tokens: Max response tokens
        temperature: Sampling temperature
        off_threshold: Threshold for OFF-TASK (kept for compatibility)
        categories: List of available categories
    
    Returns:
        (category_id, category_name, off_score, conf, reason, is_productive, raw_content)
    """
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        
        # Parse response using updated parser
        category_id, category_name, off_score, conf, reason, is_productive = _parse_llm_line(
            content, 
            off_threshold,
            categories
        )
        
        return category_id, category_name, off_score, conf, reason, is_productive, content
        
    except Exception as e:
        # Error handling - return error category
        return "error", "[ERROR]", 0.5, 0.5, str(e), None, ""

# ==================== REST OF FILE UNCHANGED ====================
# (Keep all existing functions: build_prompt, build_critic_prompt, 
#  take_screenshot_b64, ask_gpt_vision, _is_risky_context, 
#  _should_run_critic, decide_with_critic)

# ... (copy remaining functions from your current decider.py) ...


def _read_prompt_file(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


# Updated build_prompt() function for decider.py

def build_prompt(
    events: List[Tuple[str, str]], 
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    user_goals: str = ""
) -> str:
    """
    Build category-based prompt with user goals and category list.
    
    Args:
        events: List of (timestamp, window_key) tuples
        keystroke_summary: Summary of recent keystrokes
        categories: List of category dicts from sheets
        user_goals: User's current projects and goals
    
    Returns:
        Filled-in prompt string ready for LLM
    """
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity (last 60s): {keystroke_summary}" if keystroke_summary else ""
    
    template = _read_prompt_file("prompt-main.txt")
    if not template:
        # Fallback if prompt file not found
        return (
            "You are a focus judge. Categorize the user's activity.\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: CATEGORY=<category_name> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )
    
    # Build category list string
    if categories:
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']} ({'PRODUCTIVE' if cat.get('is_productive') else 'NOT PRODUCTIVE'})"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"
    
    # Fill in all placeholders
    filled = template
    filled = filled.replace("[insert goals here]", user_goals or "(no goals specified)")
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", activity_str + extra)
    
    return filled

# Updated build_critic_prompt() function for decider.py

def build_critic_prompt(
    events: List[Tuple[str, str]], 
    p1_summary: str, 
    keystroke_summary: str = "",
    categories: List[Dict] = None,
    user_goals: str = ""
) -> str:
    """
    Build critic prompt to review primary model's category choice.
    
    Args:
        events: List of (timestamp, window_key) tuples
        p1_summary: Primary model's decision summary
        keystroke_summary: Summary of recent keystrokes
        categories: List of category dicts from sheets
        user_goals: User's current projects and goals
    
    Returns:
        Filled-in critic prompt string ready for LLM
    """
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity: {keystroke_summary}" if keystroke_summary else ""

    template = _read_prompt_file("prompt-critic.txt")
    if not template:
        # Fallback if prompt file not found
        return (
            "You are a strict critic. Review the classification.\n"
            f"Initial model summary: {p1_summary}\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: CATEGORY=<category_name> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )
    
    # Build category list string
    if categories:
        category_list = "\n".join([
            f"- \"{cat['name']}\": {cat['description']} ({'PRODUCTIVE' if cat.get('is_productive') else 'NOT PRODUCTIVE'})"
            for cat in categories
        ])
    else:
        category_list = "(no categories available)"
    
    # Fill in all placeholders
    filled = template
    filled = filled.replace("[insert goals here]", user_goals or "(no goals specified)")
    filled = filled.replace("[insert categories here]", category_list)
    filled = filled.replace("[insert here]", p1_summary)
    filled = filled.replace("[insert activity here]", activity_str + extra)
    
    return filled

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

# New ask_gpt_vision_category() function for decider.py
# Add this NEW function to decider.py (after ask_gpt_vision or near other vision code)

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
    off_threshold: float,
    categories: List[Dict] = None,
) -> Tuple[str, str, float, float]:
    """
    Vision tiebreaker for category disagreement.
    
    Args:
        screenshot_b64: Base64-encoded screenshot
        events: Recent window events
        primary_category: Primary model's category choice
        primary_reason: Primary model's reasoning
        critic_category: Critic's category choice
        critic_reason: Critic's reasoning
        model: Vision model name
        max_tokens: Max tokens
        off_threshold: OFF threshold
        categories: Available categories
    
    Returns:
        (category_id, category_name, off_score, conf)
    """
    activity_str = ", ".join([k for _, k in events])
    
    # Build category list
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
        
        # Parse using same parser
        cat_id, cat_name, off_score, conf, reason, is_productive = _parse_llm_line(
            content, 
            off_threshold,
            categories
        )
        
        return cat_id, cat_name, off_score, conf
        
    except Exception as e:
        return "error", "[ERROR]", 0.5, 0.5

def _is_risky_context(events: List[Tuple[str, str]], settings: Settings) -> bool:
    text = " ".join([k for _, k in events]).lower()
    return any(kw in text for kw in settings.risky_keywords)


def _should_run_critic(p1_label: str, p1_off: float, p1_conf: float, events: List[Tuple[str, str]], settings: Settings) -> bool:
    c = settings.critic
    if not c.enabled:
        return False

    if p1_label == "ABSTAIN":
        return True

    if p1_label == "OFF-TASK":
        return False

    if p1_conf <= c.trigger_conf_max:
        return True

    if p1_off >= c.trigger_off_min:
        return True

    if c.trigger_risky_keywords and _is_risky_context(events, settings):
        return True

    return False

# Updated decide_with_critic() function for decider.py
# Replace the existing decide_with_critic() function with this

def decide_with_critic(
    events_context,
    keystroke_summary: str = "",
    settings: Settings = DEFAULT_SETTINGS,
    window_dwell_info: Optional[Dict[str, Any]] = None,
    categories: List[Dict] = None,
    user_goals: str = "",
) -> Dict[str, Any]:
    """
    Category-based decision with optional critic verification.
    
    Args:
        events_context: List of (timestamp, window_title)
        keystroke_summary: Summary of keystroke activity
        settings: Configuration
        window_dwell_info: Optional dwell time tracking (not used anymore)
        categories: List of available categories from sheets
        user_goals: User's current projects and goals
    
    Returns:
        Dict with decision info including category
    """
    
    # Build prompt with categories and goals
    p1_prompt = build_prompt(
        events_context, 
        keystroke_summary,
        categories=categories,
        user_goals=user_goals
    )
    
    # Primary model call
    p1_cat_id, p1_cat_name, p1_off, p1_conf, p1_reason, p1_is_productive, p1_raw = ask_gpt(
        p1_prompt,
        settings.model,
        max_tokens=60,
        temperature=0.0,
        off_threshold=settings.off_threshold,
        categories=categories,
    )
    
    primary_res = {
        "category_id": p1_cat_id,
        "category_name": p1_cat_name,
        "is_productive": p1_is_productive,
        "off": p1_off,
        "conf": p1_conf,
        "reason": p1_reason or "",
    }

    def mk_ret(cat_id, cat_name, is_productive, off, conf, reason, critic_ran, critic_res=None, vision_res=None):
        return {
            "final_category_id": cat_id,
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
    if p1_cat_id in ("error", "unknown"):
        return mk_ret(p1_cat_id, p1_cat_name, p1_is_productive, p1_off, p1_conf, primary_res["reason"], False)

    # Check if critic should run
    if not _should_run_critic_category(p1_cat_id, p1_cat_name, p1_conf, p1_is_productive, events_context, settings):
        return mk_ret(p1_cat_id, p1_cat_name, p1_is_productive, p1_off, p1_conf, p1_reason, False)

    # Run critic
    p1_summary = f"CATEGORY={p1_cat_name} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(
        events_context, 
        p1_summary, 
        keystroke_summary,
        categories=categories,
        user_goals=user_goals
    )

    c_cat_id, c_cat_name, c_off, c_conf, c_reason, c_is_productive, c_raw = ask_gpt(
        c_prompt,
        settings.critic_model(),
        max_tokens=settings.critic.max_tokens,
        temperature=settings.critic.temperature,
        off_threshold=settings.off_threshold,
        categories=categories,
    )
    
    critic_res = {
        "category_id": c_cat_id,
        "category_name": c_cat_name,
        "is_productive": c_is_productive,
        "off": c_off,
        "conf": c_conf,
        "reason": c_reason or "",
    }

    # Handle critic errors
    if c_cat_id in ("error", "unknown"):
        return mk_ret(p1_cat_id, p1_cat_name, p1_is_productive, p1_off, p1_conf, primary_res["reason"], True, critic_res)

    # Check for disagreement (different categories)
    critic_disagrees = (c_cat_id != p1_cat_id)

    if critic_disagrees and settings.critic.vision_enabled:
        # Vision tiebreaker
        print("[info] Category disagreement: primary says {p1_cat_name}, critic says {c_cat_name}. Trying vision tiebreak...")
        b64_img = take_screenshot_b64()
        if b64_img:
            v_cat_id, v_cat_name, v_off, v_conf = ask_gpt_vision_category(
                b64_img,
                events_context,
                p1_cat_name,
                p1_reason,
                c_cat_name,
                c_reason,
                model=settings.critic.vision_model,
                max_tokens=settings.critic.vision_max_tokens,
                off_threshold=settings.off_threshold,
                categories=categories,
            )
            if v_cat_id not in ("error", "unknown"):
                vision_res = {
                    "category_id": v_cat_id,
                    "category_name": v_cat_name,
                    "off": v_off,
                    "conf": v_conf,
                }
                # Find is_productive from category
                v_is_productive = None
                if categories:
                    for cat in categories:
                        if cat['category_id'] == v_cat_id:
                            v_is_productive = cat.get('is_productive')
                            break
                
                return mk_ret(v_cat_id, v_cat_name, v_is_productive, v_off, v_conf, f"[Vision] {c_reason}", True, critic_res, vision_res)

    # No vision or vision failed - use critic if it disagrees
    if critic_disagrees:
        final_off = max(p1_off, c_off)
        return mk_ret(
            c_cat_id,
            c_cat_name,
            c_is_productive,
            final_off,
            c_conf,
            f"[Critic] {critic_res['reason'] or primary_res['reason']}",
            True,
            critic_res,
        )

    # Critic agrees - use primary
    return mk_ret(p1_cat_id, p1_cat_name, p1_is_productive, p1_off, p1_conf, p1_reason, True, critic_res)


def _should_run_critic_category(
    cat_id: str,
    cat_name: str, 
    conf: float,
    is_productive: bool,
    events: List[Tuple[str, str]], 
    settings: Settings
) -> bool:
    """
    Decide if critic should run based on category classification.
    
    Args:
        cat_id: Category ID
        cat_name: Category name
        conf: Confidence of primary model
        is_productive: Whether category is productive
        events: Recent window events
        settings: Configuration
    
    Returns:
        True if critic should run
    """
    c = settings.critic
    if not c.enabled:
        return False

    # Always run critic on low confidence
    if conf <= c.trigger_conf_max:
        return True
    
    # Run on ambiguous categories (these can easily be wrong)
    ambiguous_categories = ["research", "communication", "research & learning"]
    if cat_name.lower() in ambiguous_categories:
        return True
    
    return False