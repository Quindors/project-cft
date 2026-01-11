# monitor/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

tracing = False

@dataclass(frozen=True)
class AlertSettings:
    enabled: bool = True
    repeat_while_offtask: bool = True
    cooldown_sec: float = 5.0   # re-alert every 5s while OFF-TASK

@dataclass(frozen=True)
class CriticSettings:
    enabled: bool = True
    model: Optional[str] = None  # None => use Settings.model
    max_tokens: int = 90
    temperature: float = 0.0

    trigger_conf_max: float = 0.70
    trigger_off_min: float = 0.30
    trigger_risky_keywords: bool = True

    # Vision tiebreaker
    vision_enabled: bool = True
    vision_model: str = "gpt-4o"
    vision_max_tokens: int = 120


@dataclass(frozen=True)
class SheetsSettings:
    sheet_url_or_key: str = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
    worksheet_prefix: str = "Focus Logs"
    scopes: List[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ])

    max_event_cols: int = 10

    # Write throttling/buffering
    append_only: bool = False
    batch_flush_max: int = 15
    idle_flush_sec: float = 15.0
    max_retries: int = 6
    backoff_base: float = 0.8


@dataclass(frozen=True)
class KeystrokeSettings:
    key_log_dir: str = os.path.join(".", "logs")
    capture_typed_text: bool = True
    enable_live_keylogger: bool = True


@dataclass(frozen=True)
class TraceSettings:
    use_trace_file: bool = True
    trace_win_path: str = r"tests/traces/trace_win.jsonl"
    trace_key_path: str = r"tests/traces/trace_key.jsonl"
    trace_speed: float = 1.0
    # NEW:
    upload_to_sheets: bool = False     # testing mode: don't write to sheets
    replay_realtime: bool = False      # False = run as fast as possible; True = sleep by trace gaps/trace_speed

@dataclass(frozen=True)
class Settings:
    # General
    log_dir: str = r".\logs"
    model: str = "gpt-4o-mini"
    interval_sec: float = 3.0
    max_events: int = 10

    # OFF-TASK threshold
    off_threshold: float = 0.60

    risky_keywords: List[str] = field(default_factory=lambda: [
        "youtube", "youtu.be", "reddit", "discord", "twitter", "x.com",
        "instagram", "tiktok", "netflix", "twitch", "hulu", "prime video",
        "steam", "epic games", "roblox", "minecraft",
        "shopping", "amazon", "ebay", "aliexpress",
    ])

    critic: CriticSettings = field(default_factory=CriticSettings)
    sheets: SheetsSettings = field(default_factory=SheetsSettings)
    keystrokes: KeystrokeSettings = field(default_factory=KeystrokeSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)

    def critic_model(self) -> str:
        return self.critic.model or self.model


DEFAULT_SETTINGS = Settings(
    trace=TraceSettings(
        use_trace_file=tracing,
        upload_to_sheets=not tracing,
        replay_realtime=not tracing,
    ),
)