# focusmon/config.py
import os

# Logs / models
LOG_DIR = r".\logs"
MODEL = "gpt-4o-mini"
INTERVAL_SEC = 3
MAX_EVENTS = 4

# OFF-TASK threshold
OFF_THRESHOLD = 0.60

# Critic pass
CRITIC_ENABLED = True
CRITIC_MODEL = MODEL
CRITIC_MAX_TOKENS = 90
CRITIC_TEMPERATURE = 0.0

CRITIC_TRIGGER_CONF_MAX = 0.70
CRITIC_TRIGGER_OFF_MIN = 0.30
CRITIC_TRIGGER_RISKY_KEYWORDS = True

RISKY_KEYWORDS = [
    "youtube", "youtu.be", "reddit", "discord", "twitter", "x.com",
    "instagram", "tiktok", "netflix", "twitch", "hulu", "prime video",
    "steam", "epic games", "roblox", "minecraft",
    "shopping", "amazon", "ebay", "aliexpress"
]

# Sheets
SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
WORKSHEET_PREFIX = "Focus Logs"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
MAX_EVENT_COLS = 5

# Write throttling
APPEND_ONLY = False
BATCH_FLUSH_MAX = 15
IDLE_FLUSH_SEC = 15.0
MAX_RETRIES = 6
BACKOFF_BASE = 0.8

# Keystrokes
KEY_LOG_DIR = os.path.join(".", "logs")
CAPTURE_TYPED_TEXT = True  # NOTE: logs actual typed chars (privacy risk)

# Trace mode
USE_TRACE_FILE = True
TRACE_WIN_PATH = r"tests/traces/trace_win.jsonl"
TRACE_KEY_PATH = r"tests/traces/trace_key.jsonl"
TRACE_SPEED = 1.0