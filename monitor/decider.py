# focusmon/decider.py
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import openai

from monitor.config import (
    MODEL,
    OFF_THRESHOLD,
    CRITIC_ENABLED,
    CRITIC_MODEL,
    CRITIC_MAX_TOKENS,
    CRITIC_TEMPERATURE,
    CRITIC_TRIGGER_CONF_MAX,
    CRITIC_TRIGGER_OFF_MIN,
    CRITIC_TRIGGER_RISKY_KEYWORDS,
    RISKY_KEYWORDS,
)

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    print("[warn] Pillow not installed. pip install Pillow")


# --------------------- Prompt loading ---------------------

def _read_prompt_file(filename: str) -> str:
    """
    Tries to read prompt files from:
    1) current working directory
    2) project root (parent of focusmon/)
    """
    candidates = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent.parent / filename,
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
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


def build_critic_prompt(
    events: List[Tuple[str, str]],
    p1_summary: str,
    keystroke_summary: str = "",
) -> str:
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


# --------------------- LLM parsing/calls ---------------------

def _parse_llm_line(content: str) -> Tuple[str, str, float, float]:
    content = (content or "").strip()

    m_label = re.search(r"LABEL\s*=\s*(ON-TASK|OFF-TASK)", content, flags=re.I)
    label = (
        m_label.group(1).upper()
        if m_label
        else ("OFF-TASK" if "OFF-TASK" in content.upper()
              else "ON-TASK" if "ON-TASK" in content.upper()
              else "[WARN]")
    )

    m_off = re.search(r"OFF_SCORE\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)
    m_conf = re.search(r"CONF\s*=\s*([01](?:\.\d+)?|\.\d+)", content, flags=re.I)

    off_score = float(m_off.group(1)) if m_off else (0.8 if label == "OFF-TASK" else 0.2)
    conf = float(m_conf.group(1)) if m_conf else 0.5

    off_score = min(max(off_score, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)

    m_reason = re.search(r"REASON\s*=\s*(.+)", content, flags=re.I | re.S)
    reason = m_reason.group(1).strip() if m_reason else ""

    # FN-biased normalization
    if label == "OFF-TASK" and off_score < OFF_THRESHOLD:
        off_score = OFF_THRESHOLD

    return label, reason, off_score, conf


def ask_gpt(
    prompt: str,
    model: str,
    max_tokens: int = 60,
    temperature: float = 0.0,
) -> Tuple[str, str, float, float, str]:
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf = _parse_llm_line(content)
        return label, reason, off_score, conf, content
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5, ""


# --------------------- Vision tiebreaker ---------------------

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
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                ],
            }],
            max_tokens=120,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        label, reason, off_score, conf = _parse_llm_line(content)
        return label, reason, off_score, conf
    except Exception as e:
        return "[ERROR]", str(e), 0.5, 0.5


# --------------------- Critic logic ---------------------

def _is_risky_context(events: List[Tuple[str, str]]) -> bool:
    text = " ".join([k for _, k in events]).lower()
    return any(kw in text for kw in RISKY_KEYWORDS)


def _should_run_critic(p1_label: str, p1_off: float, p1_conf: float, events: List[Tuple[str, str]]) -> bool:
    if not CRITIC_ENABLED:
        return False

    # Only run critic on ON-TASK calls (optimize costs)
    if p1_label != "ON-TASK":
        return False

    if p1_conf <= CRITIC_TRIGGER_CONF_MAX:
        return True

    if p1_off >= CRITIC_TRIGGER_OFF_MIN:
        return True

    if CRITIC_TRIGGER_RISKY_KEYWORDS and _is_risky_context(events):
        return True

    return False


def decide_with_critic(events_context: List[Tuple[str, str]], keystroke_summary: str = "") -> Dict[str, Any]:
    p1_prompt = build_prompt(events_context, keystroke_summary)
    p1_label, p1_reason, p1_off, p1_conf, _ = ask_gpt(p1_prompt, MODEL, max_tokens=60, temperature=0.0)

    primary_res = {"label": p1_label, "reason": p1_reason or "", "off": p1_off, "conf": p1_conf}