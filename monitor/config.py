# monitor/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import List, Optional

tracing = True

@dataclass(frozen=True)
class PopupSettings:
    """Settings for off-task popup alerts."""
    enabled: bool = True
    duration: int = 5  # seconds to show popup
    sound_enabled: bool = True
    repeat_while_offtask: bool = True
    cooldown_sec: float = 5.0   # re-alert every 5s while OFF-TASK

@dataclass(frozen=True)
class CriticSettings:
    enabled: bool = True
    model: Optional[str] = None  # None => use Settings.model
    max_tokens: int = 90
    temperature: float = 0.0

    trigger_conf_max: float = 0.70

    # Vision tiebreaker
    vision_enabled: bool = True
    vision_model: str = "gpt-4o"
    vision_max_tokens: int = 120


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
    replay_realtime: bool = False      # False = run as fast as possible

@dataclass(frozen=True)
class ScreenshotSettings:
    enabled: bool = True
    capture_interval: float = 10.0   # seconds between captures
    buffer_size: int = 12            # keep last N screenshots in memory

@dataclass(frozen=True)
class Settings:
    # General
    model: str = "gpt-4o-mini"
    
    # Timing settings for runner
    poll_interval: float = 2.0  # How often to collect window events (seconds)
    decision_interval: float = 10.0  # How often to make decisions (seconds)
    
    # Context window
    context_window_size: int = 10  # Number of recent events to keep
    min_events: int = 5  # Minimum events before making first decision
    
    # Legacy settings (kept for compatibility)
    interval_sec: float = 3.0  # Old name for poll_interval
    max_events: int = 10  # Old context window size

    # OFF-TASK threshold (used by LLM parsing)
    off_threshold: float = 0.60

    critic: CriticSettings = field(default_factory=CriticSettings)
    supabase: SupabaseSettings = field(default_factory=SupabaseSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    popup: PopupSettings = field(default_factory=PopupSettings)

    def critic_model(self) -> str:
        return self.critic.model or self.model


DEFAULT_SETTINGS = Settings(
    trace=TraceSettings(
        use_trace_file=tracing,
        upload_to_sheets=not tracing,
        replay_realtime=not tracing,
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
    return replace(
        DEFAULT_SETTINGS,
        model=model,
        critic=replace(DEFAULT_SETTINGS.critic, vision_model=model),
    )