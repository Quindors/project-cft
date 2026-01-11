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

def _parse_llm_line(content: str, off_threshold: float) -> Tuple[str, str, float, float, Dict[str, float]]:
    """
    Parse LLM response including optional factor scores.
    
    Returns:
        (label, reason, off_score, conf, factors_dict)
    """
    content = (content or "").strip()

    # Match ON-TASK, OFF-TASK, or ABSTAIN
    m_label = re.search(r"LABEL\s*=\s*(ON-TASK|OFF-TASK|ABSTAIN)", content, flags=re.I)
    if m_label:
        label = m_label.group(1).upper()
    else:
        # Fallback parsing
        upper = content.upper()
        if "ABSTAIN" in upper:
            label = "ABSTAIN"
        elif "OFF-TASK" in upper:
            label = "OFF-TASK"
        elif "ON-TASK" in upper:
            label = "ON-TASK"
        else:
            label = "[WARN]"

    m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)

    # Default OFF_SCORE
    if m_off:
        off_score = float(m_off.group(1))
    elif label == "OFF-TASK":
        off_score = 0.8
    elif label == "ABSTAIN":
        off_score = 0.5
    else:
        off_score = 0.2

    conf = float(m_conf.group(1)) if m_conf else 0.5

    off_score = min(max(off_score, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)

    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    reason = m_reason.group(1).strip() if m_reason else ""

    # FN-biased normalization
    if label == "OFF-TASK" and off_score < off_threshold:
        off_score = off_threshold

    # NEW: Extract optional factor scores from LLM
    factors = {}
    factor_names = [
        "WINDOW_RELEVANCE",
        "TYPING_ENGAGEMENT", 
        "CONTEXT_TRAJECTORY",
        "DWELL_PENALTY",
    ]
    
    for factor_name in factor_names:
        # Match formats like: WINDOW_RELEVANCE=0.8 or WINDOW_RELEVANCE = -0.3
        m = re.search(rf"{factor_name}\s*=\s*([-+]?[01](?:\.\d+)?|[-+]?\.\d+)", content, flags=re.I)
        if m:
            try:
                score = float(m.group(1))
                # Clamp to [-1, 1]
                score = min(max(score, -1.0), 1.0)
                factors[factor_name.lower()] = score
            except ValueError:
                pass

    return label, reason, off_score, conf, factors


# ==================== MODIFIED: ASK GPT ====================

def ask_gpt(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    off_threshold: float,
) -> Tuple[str, str, float, float, Dict[str, float], str]:
    """
    Modified to return factors dict.
    
    Returns:
        (label, reason, off_score, conf, factors, raw_content)
    """
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf, factors = _parse_llm_line(content, off_threshold)
        return label, reason, off_score, conf, factors, content
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5, {}, ""


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


def build_prompt(events: List[Tuple[str, str]], keystroke_summary: str = "") -> str:
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity (last 60s): {keystroke_summary}" if keystroke_summary else ""

    template = _read_prompt_file("prompt-main.txt")
    if not template:
        return (
            "You are a focus judge. Determine if the user is ON-TASK or OFF-TASK.\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )
    return template.replace("[insert here]", activity_str + extra)


def build_critic_prompt(events: List[Tuple[str, str]], p1_summary: str, keystroke_summary: str = "") -> str:
    activity_str = ", ".join([k for _, k in events])
    extra = f"\nKeystroke Activity: {keystroke_summary}" if keystroke_summary else ""

    template = _read_prompt_file("prompt-critic.txt")
    if not template:
        return (
            "You are a strict critic. Improve the classification.\n"
            f"Initial model summary: {p1_summary}\n"
            f"Recent activity: {activity_str}{extra}\n"
            "Output: LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<text>"
        )

    if template.count("[insert here]") >= 2:
        template = template.replace("[insert here]", p1_summary, 1)
        template = template.replace("[insert here]", activity_str + extra, 1)
    else:
        template += f"\n\nInitial: {p1_summary}\nActivity: {activity_str}{extra}"
    return template


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


def ask_gpt_vision(
    screenshot_b64: str,
    events: List[Tuple[str, str]],
    main_reason: str,
    critic_reason: str,
    *,
    model: str,
    max_tokens: int,
    off_threshold: float,
) -> Tuple[str, str, float, float]:
    activity_str = ", ".join([k for _, k in events])
    prompt = (
        "You are the final judge. Resolving a disagreement.\n"
        f"Context: {activity_str}\n"
        f"Main Model said ON-TASK: {main_reason}\n"
        f"Critic Model said OFF-TASK: {critic_reason}\n\n"
        "Look at the screen. What is the user *actually* doing?\n"
        "Output strict single line format:\n"
        "LABEL=<ON-TASK|OFF-TASK> | OFF_SCORE=<0..1> | CONF=<0..1> | REASON=<explanation>"
    )
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
        label, reason, off_score, conf, _ = _parse_llm_line(content, off_threshold)
        return label, reason, off_score, conf
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5


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


def decide_with_critic(
    events_context,
    keystroke_summary: str = "",
    settings: Settings = DEFAULT_SETTINGS,
    window_dwell_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Modified to include factor scoring.
    
    Args:
        events_context: List of (timestamp, window_title)
        keystroke_summary: Summary of keystroke activity
        settings: Configuration
        window_dwell_info: Optional dwell time tracking dict
    
    Returns:
        Dict with decision info + factor scores
    """
    
    # Compute local factors
    computed_factors = compute_factor_scores(
        events_context,
        keystroke_summary,
        settings,
        window_dwell_info
    )
    
    # Primary model
    p1_prompt = build_prompt(events_context, keystroke_summary)
    p1_label, p1_reason, p1_off, p1_conf, p1_factors, _ = ask_gpt(
        p1_prompt,
        settings.model,
        max_tokens=60,
        temperature=0.0,
        off_threshold=settings.off_threshold,
    )
    
    # Merge LLM factors with computed factors
    all_factors = {**computed_factors, **p1_factors}
    
    primary_res = {
        "label": p1_label,
        "reason": p1_reason or "",
        "off": p1_off,
        "conf": p1_conf,
        "factors": all_factors,
    }

    def mk_ret(label, reason, off, conf, critic_ran, critic_res=None, vision_res=None):
        return {
            "final_label": label,
            "final_reason": reason,
            "final_off": off,
            "final_conf": conf,
            "critic_ran": critic_ran,
            "primary": primary_res,
            "critic": critic_res,
            "vision": vision_res,
            "factors": all_factors,
            "aggregate_score": aggregate_factor_score(all_factors)[0],
        }

    if p1_label in ("[ERROR]", "[WARN]"):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], False)

    # NEW: Apply hybrid decision
    hybrid_label, hybrid_reason, hybrid_off, hybrid_conf = decide_hybrid_simple(
        all_factors,
        p1_label,
        p1_off,
        p1_conf,
        p1_reason,
        settings.off_threshold,
    )

    if not _should_run_critic(hybrid_label, hybrid_off, hybrid_conf, events_context, settings):
        return mk_ret(hybrid_label, hybrid_reason, hybrid_off, hybrid_conf, False)

    # Critic
    p1_summary = f"LABEL={p1_label} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(events_context, p1_summary, keystroke_summary)

    c_label, c_reason, c_off, c_conf, c_factors, _ = ask_gpt(
        c_prompt,
        settings.critic_model(),
        max_tokens=settings.critic.max_tokens,
        temperature=settings.critic.temperature,
        off_threshold=settings.off_threshold,
    )
    
    critic_res = {
        "label": c_label,
        "reason": c_reason or "",
        "off": c_off,
        "conf": c_conf,
        "factors": c_factors,
    }

    if c_label in ("[ERROR]", "[WARN]"):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], True, critic_res)

    critic_says_off = (c_label == "OFF-TASK") or (c_off >= settings.off_threshold)

    if critic_says_off:
        if settings.critic.vision_enabled:
            print("[info] Disagreement: critic says OFF. Trying vision tiebreak...")
            b64_img = take_screenshot_b64()
            if b64_img:
                v_label, v_reason, v_off, v_conf = ask_gpt_vision(
                    b64_img,
                    events_context,
                    p1_reason,
                    c_reason,
                    model=settings.critic.vision_model,
                    max_tokens=settings.critic.vision_max_tokens,
                    off_threshold=settings.off_threshold,
                )
                if v_label not in ("[ERROR]", "[WARN]"):
                    vision_res = {
                        "label": v_label,
                        "reason": v_reason,
                        "off": v_off,
                        "conf": v_conf,
                    }
                    return mk_ret(v_label, f"[Vision] {v_reason}", v_off, v_conf, True, critic_res, vision_res)

        # fallback to critic
        final_off = max(p1_off, c_off)
        return mk_ret(
            "OFF-TASK",
            f"[Critic] {critic_res['reason'] or primary_res['reason']}",
            final_off,
            c_conf,
            True,
            critic_res,
        )

    # critic agrees ON-TASK
    return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], True, critic_res)

if __name__ == "__main__":
    # Test the aggregation function
    print("Testing aggregate_factor_score()...")
    
    # Test case 1: Clear OFF-TASK
    test_factors_1 = {
        "window_relevance": -0.9,  # YouTube
        "dwell_time": -0.8,         # Long time, no activity
        "keystroke_activity": -0.6, # Idle
        "trajectory": -0.7,         # Drifting
        "risky_keywords": -0.9      # Risky site
    }
    
    score_1, explanation_1 = aggregate_factor_score(test_factors_1)
    print(f"\nTest 1 (should be OFF-TASK):")
    print(f"  Score: {score_1:+.2f}")
    print(f"  Explanation: {explanation_1}")
    print(f"  Decision: {'OFF-TASK' if score_1 < -0.3 else 'ON-TASK' if score_1 > 0.3 else 'UNCERTAIN'}")
    
    # Test case 2: Clear ON-TASK
    test_factors_2 = {
        "window_relevance": 0.9,   # VS Code
        "dwell_time": 0.5,          # Good dwell + activity
        "keystroke_activity": 0.7,  # Typing a lot
        "trajectory": 0.6,          # Stable work
        "risky_keywords": 0.0       # No risky sites
    }
    
    score_2, explanation_2 = aggregate_factor_score(test_factors_2)
    print(f"\nTest 2 (should be ON-TASK):")
    print(f"  Score: {score_2:+.2f}")
    print(f"  Explanation: {explanation_2}")
    print(f"  Decision: {'OFF-TASK' if score_2 < -0.3 else 'ON-TASK' if score_2 > 0.3 else 'UNCERTAIN'}")
    
    # Test case 3: Mixed/Uncertain
    test_factors_3 = {
        "window_relevance": 0.6,   # Educational site
        "dwell_time": -0.3,         # Been here a while, low activity
        "keystroke_activity": 0.1,  # Some typing
        "trajectory": 0.0,          # Neutral
        "risky_keywords": 0.0       # No risky keywords
    }
    
    score_3, explanation_3 = aggregate_factor_score(test_factors_3)
    print(f"\nTest 3 (should be UNCERTAIN):")
    print(f"  Score: {score_3:+.2f}")
    print(f"  Explanation: {explanation_3}")
    print(f"  Decision: {'OFF-TASK' if score_3 < -0.3 else 'ON-TASK' if score_3 > 0.3 else 'UNCERTAIN'}")
    
    print("\n✓ If these results look reasonable, move to Step 2")