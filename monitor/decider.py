# monitor/decider.py
from __future__ import annotations

import base64, io, re, openai
from typing import List, Tuple, Dict, Any, Optional

from monitor.config import Settings, DEFAULT_SETTINGS

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    print("[warn] Pillow not installed. Vision tiebreaker disabled. pip install Pillow")


# ------------------ prompts ------------------

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


# ------------------ parsing + calls ------------------

def _parse_llm_line(content: str, off_threshold: float) -> Tuple[str, str, float, float]:
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

    # Default OFF_SCORE: 0.5 for ABSTAIN (uncertain), 0.8 for OFF, 0.2 for ON
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

    # FN-biased normalization (only for definitive OFF-TASK)
    if label == "OFF-TASK" and off_score < off_threshold:
        off_score = off_threshold

    return label, reason, off_score, conf


def ask_gpt(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    off_threshold: float,
) -> Tuple[str, str, float, float, str]:
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf = _parse_llm_line(content, off_threshold)
        return label, reason, off_score, conf, content
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5, ""


# ------------------ vision tiebreak ------------------

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
        label, reason, off_score, conf = _parse_llm_line(content, off_threshold)
        return label, reason, off_score, conf
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5


# ------------------ critic logic ------------------

def _is_risky_context(events: List[Tuple[str, str]], settings: Settings) -> bool:
    text = " ".join([k for _, k in events]).lower()
    return any(kw in text for kw in settings.risky_keywords)


def _should_run_critic(p1_label: str, p1_off: float, p1_conf: float, events: List[Tuple[str, str]], settings: Settings) -> bool:
    c = settings.critic
    if not c.enabled:
        return False

    # Always run critic when main model abstains
    if p1_label == "ABSTAIN":
        return True

    # Only run critic on ON-TASK calls (cost control) - skip definitive OFF-TASK
    if p1_label == "OFF-TASK":
        return False

    # ON-TASK with low confidence or high OFF_SCORE
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
) -> Dict[str, Any]:
    # Primary
    p1_prompt = build_prompt(events_context, keystroke_summary)
    p1_label, p1_reason, p1_off, p1_conf, _ = ask_gpt(
        p1_prompt,
        settings.model,
        max_tokens=60,
        temperature=0.0,
        off_threshold=settings.off_threshold,
    )
    primary_res = {"label": p1_label, "reason": p1_reason or "", "off": p1_off, "conf": p1_conf}

    def mk_ret(label, reason, off, conf, critic_ran, critic_res=None):
        return {
            "final_label": label,
            "final_reason": reason,
            "final_off": off,
            "final_conf": conf,
            "critic_ran": critic_ran,
            "primary": primary_res,
            "critic": critic_res,
        }

    if p1_label in ("[ERROR]", "[WARN]"):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], False)

    if not _should_run_critic(p1_label, p1_off, p1_conf, events_context, settings):
        return mk_ret(p1_label, primary_res["reason"], primary_res["off"], primary_res["conf"], False)

    # Critic
    p1_summary = f"LABEL={p1_label} | OFF_SCORE={p1_off:.2f} | CONF={p1_conf:.2f} | REASON={p1_reason or ''}"
    c_prompt = build_critic_prompt(events_context, p1_summary, keystroke_summary)

    c_label, c_reason, c_off, c_conf, _ = ask_gpt(
        c_prompt,
        settings.critic_model(),
        max_tokens=settings.critic.max_tokens,
        temperature=settings.critic.temperature,
        off_threshold=settings.off_threshold,
    )
    critic_res = {"label": c_label, "reason": c_reason or "", "off": c_off, "conf": c_conf}

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
                    return mk_ret(v_label, f"[Vision] {v_reason}", v_off, v_conf, True, critic_res)

        # fallback to critic (safer)
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