# monitor/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from monitor.paths import is_frozen as _is_frozen

# Default tracing OFF in frozen exe builds (no trace fixtures bundled),
# ON in source mode for the dev/test workflow.
tracing = not _is_frozen()

@dataclass(frozen=True)
class PopupSettings:
    """Settings for off-task popup alerts."""
    enabled: bool = True
    duration: int = 5  # seconds to show popup
    # Breathing room after an alert is dismissed before deciding again, so the
    # verdict is about where the user went, not the moment they closed it.
    resume_grace_sec: float = 2.0

@dataclass(frozen=True)
class IntentionSettings:
    """Alerting behavior while a session intention is active.

    During a declared session the generic off-task popup is replaced by a
    dwell-based drift alert. Activity falls into three intent tiers by its
    ALIGN score, and each tier gets its own patience:

      on-intent      >= on_intent_min   doing the declared work; never alerts
      partial-intent >= off_intent_max  loosely related (supporting work,
                                        adjacent research) — alerts only after
                                        a long stretch, since this is often
                                        legitimate work that drifted sideways
      off-intent     <  off_intent_max  unrelated; alerts on the short dwell
    """
    on_intent_min: float = 0.7      # ALIGN at/above this is on-intent
    off_intent_max: float = 0.5     # ALIGN below this is off-intent
    dwell_sec: float = 120.0        # continuous off-intent time before alerting
    partial_dwell_sec: float = 600.0  # ... and the longer fuse for partial-intent
    realert_sec: float = 300.0      # minimum gap between drift alerts
    # A partial-band hedge in the session's home category (the category the
    # intention gate mapped the declared work to) counts as on-intent. Ties
    # only: a confident off-intent score is never rescued by category.
    home_category_prior: bool = True


@dataclass(frozen=True)
class IdleSettings:
    """User-absence detection (monitor/idle.py).

    Past threshold_sec without keyboard/mouse input the runner treats the
    chair as empty: no decisions, nothing logged, so a productive window left
    open overnight earns nothing. threshold_sec <= 0 disables the feature.
    Override with CFT_IDLE_SEC in the .env.
    """
    threshold_sec: float = 300.0
    # Popups need a reader. Below the away threshold the user still gets the
    # benefit of the doubt for *logging*, but an alert only fires if there was
    # input this recently — never nag an empty chair.
    alert_recent_input_sec: float = 60.0


@dataclass(frozen=True)
class CriticSettings:
    # Vision tiebreaker (the "critic" text-LLM step was removed; only vision remains).
    vision_enabled: bool = True
    vision_model: str = "gpt-4o"
    vision_max_tokens: int = 120
    # How sure vision must be (its CONF) to overturn the call it reviewed.
    # Two bars, because the classifier is center-biased: Ambiguous is where a
    # decision lands without evidence, so LEAVING it — promoting the model's
    # default, or overruling a user correction that has Ambiguous on either
    # side of it — takes strong screen evidence, while overturning a specific
    # category the title supported (demoting it back to Ambiguous included)
    # takes less. Override with CFT_VISION_OVERRIDE_CONF and
    # CFT_VISION_OVERRIDE_AMBIGUOUS_CONF in the .env.
    override_min_conf: float = 0.70
    override_ambiguous_min_conf: float = 0.85


@dataclass(frozen=True)
class SupabaseSettings:
    url: str = ""
    key: str = ""
    batch_flush_max: int = 15
    idle_flush_sec: float = 15.0
    max_retries: int = 6
    backoff_base: float = 0.8

    def __post_init__(self):
        # frozen=True requires object.__setattr__ for post-init defaults
        if not self.url:
            object.__setattr__(self, 'url', os.getenv("SUPABASE_URL", ""))
        if not self.key:
            object.__setattr__(self, 'key', os.getenv("SUPABASE_KEY", ""))


@dataclass(frozen=True)
class TraceSettings:
    use_trace_file: bool = True
    trace_win_path: str = r"tests/traces/trace_win.jsonl"
    trace_key_path: str = r"tests/traces/trace_key.jsonl"
    trace_speed: float = 1.0
    upload_to_sheets: bool = False     # testing mode: don't write to database

@dataclass(frozen=True)
class Settings:
    # General
    model: str = "gpt-4o-mini"
    
    # Timing settings for runner
    poll_interval: float = 2.0  # How often to collect window events (seconds)
    decision_interval: float = 10.0  # How often to make decisions (seconds)
    
    # Context window
    context_window_size: int = 10  # Number of recent events to keep
    min_events: int = 1  # Minimum events before making first decision

    critic: CriticSettings = field(default_factory=CriticSettings)
    supabase: SupabaseSettings = field(default_factory=SupabaseSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    popup: PopupSettings = field(default_factory=PopupSettings)
    intention: IntentionSettings = field(default_factory=IntentionSettings)
    idle: IdleSettings = field(default_factory=IdleSettings)


DEFAULT_SETTINGS = Settings(
    trace=TraceSettings(
        use_trace_file=tracing,
        upload_to_sheets=not tracing,
    ),
)


def load_settings() -> Settings:
    """Read model/provider from env (call AFTER load_env so .env is in place).

    Single global swap: LLM_MODEL is applied to both the primary classifier
    and the vision tiebreaker.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    default_model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"
    model = os.getenv("LLM_MODEL", default_model)

    idle = DEFAULT_SETTINGS.idle
    raw_idle = os.getenv("CFT_IDLE_SEC", "").strip()
    if raw_idle:
        try:
            idle = replace(idle, threshold_sec=float(raw_idle))
        except ValueError:
            print(f"[config] CFT_IDLE_SEC={raw_idle!r} is not a number; "
                  f"keeping {idle.threshold_sec:g}s")

    critic = replace(DEFAULT_SETTINGS.critic, vision_model=model)
    for env, name in (("CFT_VISION_OVERRIDE_CONF", "override_min_conf"),
                      ("CFT_VISION_OVERRIDE_AMBIGUOUS_CONF", "override_ambiguous_min_conf")):
        value = _conf_from_env(env, getattr(critic, name))
        critic = replace(critic, **{name: value})
    if critic.override_ambiguous_min_conf < critic.override_min_conf:
        print("[config] CFT_VISION_OVERRIDE_AMBIGUOUS_CONF is below "
              "CFT_VISION_OVERRIDE_CONF: leaving Ambiguous is now easier than "
              "overturning a specific category, which undoes the center bias")

    return replace(
        DEFAULT_SETTINGS,
        model=model,
        critic=critic,
        idle=idle,
    )


def _conf_from_env(env: str, default: float) -> float:
    """A 0..1 confidence bar from the environment, or `default` when unset or
    unusable (a bad value is reported, never silently applied)."""
    raw = os.getenv(env, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 <= value <= 1.0:
        print(f"[config] {env}={raw!r} is not a number between 0 and 1; "
              f"keeping {default:g}")
        return default
    return value