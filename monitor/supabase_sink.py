# monitor/supabase_sink.py
from __future__ import annotations

import json
import re
import threading
import time
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional

from supabase import create_client, Client


@dataclass
class FlushResult:
    updated: int
    appended: int
    sent: int


def _normalize_title(title: str) -> str:
    """Normalize a window title for cache matching."""
    s = title.strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    s = re.sub(r"^\(\d+\)\s*", "", s)  # strip leading (N) notification counts
    return " ".join(s.split()).lower()


_FALLBACK_CATEGORIES = [
    {"name": "Deep Work", "description": "VS Code, PyCharm, coding, writing, focused tasks", "is_productive": True},
    {"name": "Homework", "description": "OneNote with HW notes, PDF textbooks, Desmos, educational PDFs, calculus tools, assignment work", "is_productive": True},
    {"name": "Research & Learning", "description": "Khan Academy, Wolfram Alpha, Wikipedia for homework topics, educational YouTube tutorials, documentation", "is_productive": True},
    {"name": "Communication", "description": "Teams for schoolwork, email about classes, Slack for project collaboration, work-related messaging", "is_productive": True},
    {"name": "Social Media", "description": "Twitter/X, Instagram, TikTok, Reddit browsing, Discord chatting (non-work), Facebook", "is_productive": False},
    {"name": "Entertainment", "description": "YouTube entertainment videos, Netflix, Twitch, Hulu, Prime Video, funny cat videos, streams", "is_productive": False},
    {"name": "Gaming", "description": "Steam, Epic Games, Minecraft, Roblox, game launchers, playing games", "is_productive": False},
    {"name": "Shopping", "description": "Amazon, eBay, AliExpress, online shopping, browsing products", "is_productive": False},
    {"name": "Ambiguous", "description": "Default catch-all: activity with no specific productive evidence and no clear specific off-task signal. Treated as off-task.", "is_productive": False},
]


class NullSink:
    """No-op sink for testing/trace mode.

    Writes are no-ops. Reads delegate to `read_sink` if provided, else return
    hardcoded fallbacks so trace runs still have categories to classify into.
    """

    def __init__(self, read_sink=None):
        self.read_sink = read_sink

    def ensure_day(self, day): return
    def enqueue_record(self, record): return
    def flush_if_needed(self): return None
    def close(self): return

    def get_categories(self):
        if self.read_sink is not None:
            try:
                return self.read_sink.get_categories()
            except Exception as e:
                print(f"[warn] NullSink: read_sink.get_categories failed ({e}) - using fallback defaults")
        return _FALLBACK_CATEGORIES

    def load_corrections(self, days_back: int = 7):
        if self.read_sink is not None:
            try:
                return self.read_sink.load_corrections(days_back)
            except Exception as e:
                print(f"[warn] NullSink: read_sink.load_corrections failed ({e})")
        return []

    def load_prior_classifications(self, days_back: int = 7, min_confidence: float = 0.85):
        if self.read_sink is not None:
            try:
                return self.read_sink.load_prior_classifications(days_back, min_confidence)
            except Exception as e:
                print(f"[warn] NullSink: read_sink.load_prior_classifications failed ({e})")
        return []

    def load_labeled(self, days_back=None): return []

class SupabaseSink:
    def __init__(self, settings):
        self.settings = settings
        self.cfg = settings.supabase

        url = self.cfg.url or os.environ.get("SUPABASE_URL", "")
        key = self.cfg.key or os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        self.sb: Client = create_client(url, key)

        self.buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self._categories_cache: Optional[List[Dict]] = None

        # Sweep transient error verdicts left by older builds. Best-effort,
        # and off the boot path: this is a network round trip, and the
        # storage preload (and so the first decision) used to wait on it.
        threading.Thread(target=self._purge_error_rows, daemon=True,
                         name="supabase-purge").start()

    def _purge_error_rows(self) -> None:
        try:
            res = (self.sb.table("focus_logs").delete()
                   .in_("category_name", ["[ERROR]", "[PARSE_ERROR]"])
                   .execute())
            if res.data:
                print(f"[sink] purged {len(res.data)} error rows from Supabase")
        except Exception as e:
            print(f"[warn] Supabase error-row purge failed: {e}")

    # ------------------- public API -------------------

    def ensure_day(self, day: date) -> None:
        """No-op - Supabase uses a single table, no daily tabs."""
        return

    def enqueue_record(self, record: Dict[str, Any]) -> None:
        row = self._flatten_row(record)
        self.buffer.append(row)

    def flush_if_needed(self) -> Optional[FlushResult]:
        if not self.buffer:
            return None
        now = time.time()
        should = (
            len(self.buffer) >= self.cfg.batch_flush_max
            or (now - self.last_flush_time) >= self.cfg.idle_flush_sec
        )
        if not should:
            return None
        return self.flush(force=True)

    def flush(self, force: bool = False) -> Optional[FlushResult]:
        if not self.buffer:
            return None

        batch = self.buffer[:]
        self.buffer.clear()

        appended = 0
        for attempt in range(self.cfg.max_retries):
            try:
                self.sb.table("focus_logs").insert(batch).execute()
                appended = len(batch)
                break
            except Exception as e:
                if attempt < self.cfg.max_retries - 1:
                    delay = self.cfg.backoff_base * (2 ** attempt)
                    print(f"[warn] Supabase insert failed (attempt {attempt+1}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    print(f"[error] Supabase insert failed after {self.cfg.max_retries} attempts: {e}")
                    # Put rows back in buffer so they aren't lost
                    self.buffer = batch + self.buffer
                    return None

        self.last_flush_time = time.time()
        return FlushResult(updated=0, appended=appended, sent=appended)

    def close(self) -> None:
        self.flush(force=True)

    # ------------------- reads -------------------

    def get_categories(self) -> List[Dict]:
        if self._categories_cache is not None:
            return self._categories_cache

        try:
            result = self.sb.table("categories").select("*").execute()
            categories = []
            for r in result.data:
                cat = dict(r)
                # is_productive comes as bool or None from Postgres, no conversion needed
                categories.append(cat)
            self._categories_cache = categories
            return categories
        except Exception as e:
            print(f"[error] Failed to load categories: {e}")
            return []

    def load_corrections(self, days_back: int = 7) -> List[Dict]:
        """Load recent human corrections from the last `days_back` days."""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        try:
            result = (
                self.sb.table("focus_logs")
                .select("current_window, category_name, human_label")
                .not_.is_("human_label", "null")
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .limit(500)
                .execute()
            )
        except Exception as e:
            print(f"[error] Failed to load corrections: {e}")
            return []

        # Deduplicate by window title (latest wins — already sorted desc)
        corrections: Dict[str, Dict] = {}
        for row in result.data:
            window = (row.get("current_window") or "").strip()
            ai_cat = (row.get("category_name") or "").strip()
            human = (row.get("human_label") or "").strip()

            if not human or not window:
                continue
            if human.lower() == ai_cat.lower():
                continue  # user agreed with AI

            if window not in corrections:
                corrections[window] = {
                    "window_title": window,
                    "ai_category": ai_cat,
                    "corrected_category": human,
                }

        result_list = list(corrections.values())
        print(f"[info] Loaded {len(result_list)} corrections from last {days_back} days")
        return result_list

    def load_prior_classifications(self, days_back: int = 7, min_confidence: float = 0.85) -> List[Dict]:
        """Load recent high-confidence classifications for cache priming.

        Rows the decider tagged "[vision]" are skipped, as in the local sink:
        a screen verdict is not evidence about what the title means.
        """
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        skip_categories = {"ambiguous", "[error]", "[parse_error]"}

        try:
            result = (
                self.sb.table("focus_logs")
                .select("current_window, category_name, confidence, reason")
                .is_("human_label", "null")
                .gte("confidence", min_confidence)
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .limit(500)
                .execute()
            )
        except Exception as e:
            print(f"[error] Failed to load prior classifications: {e}")
            return []

        # Deduplicate by normalized title (latest wins)
        classifications: Dict[str, Dict] = {}
        for row in result.data:
            window = (row.get("current_window") or "").strip()
            category = (row.get("category_name") or "").strip()
            conf = row.get("confidence")

            if not window or not category:
                continue
            if category.lower() in skip_categories:
                continue
            if (row.get("reason") or "").lower().startswith("[vision]"):
                continue

            normalized = _normalize_title(window)
            if normalized not in classifications:
                classifications[normalized] = {
                    "window_title": window,
                    "normalized_title": normalized,
                    "category_name": category,
                    "confidence": conf,
                }

        result_list = list(classifications.values())
        print(f"[info] Loaded {len(result_list)} prior classifications from last {days_back} days")
        return result_list

    # ------------------- learning loop: accuracy scoring -------------------

    def load_labeled(self, days_back: Optional[int] = None) -> List[Dict]:
        """Every human-labeled row (confirmations AND corrections), for scoring.

        `days_back=None` loads all history. Returns dicts with current_window,
        category_name, human_label, confidence, timestamp.
        """
        try:
            q = (
                self.sb.table("focus_logs")
                .select("current_window, category_name, human_label, confidence, timestamp")
                .not_.is_("human_label", "null")
                .neq("human_label", "")
                .order("timestamp", desc=True)
                .limit(10000)
            )
            if days_back is not None:
                cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
                q = q.gte("timestamp", cutoff)
            result = q.execute()
        except Exception as e:
            print(f"[error] Failed to load labeled rows: {e}")
            return []
        return [dict(r) for r in result.data]

    # ------------------- internal -------------------

    @staticmethod
    def _clean_window_title(raw: str) -> str:
        """Strip WIN: prefix and trim for readability."""
        s = raw.strip()
        if s.upper().startswith("WIN:"):
            s = s[4:]
        return s

    def _flatten_row(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a decision record from runner._build_record() into a Supabase row."""
        titles = [str(e.get("key", "")) for e in (record.get("events") or []) if isinstance(e, dict)]

        current_window = self._clean_window_title(titles[-1]) if titles else ""

        # Deduplicate events, excluding the current window
        seen = set()
        events = []
        for t in titles:
            cleaned = self._clean_window_title(t)
            if cleaned and cleaned not in seen and cleaned != current_window:
                seen.add(cleaned)
                events.append(cleaned)

        try:
            confidence = float(record.get("primary_confidence")) if record.get("primary_confidence") not in ("", None) else None
        except (ValueError, TypeError):
            confidence = None

        return {
            "timestamp": record.get("timestamp", datetime.now().isoformat()),
            "current_window": current_window or None,
            "category_name": record.get("category_name") or None,
            "confidence": confidence,
            "reason": record.get("primary_reason") or None,
            "human_label": record.get("human_label") or None,
            "typed_text": record.get("typed_text") or None,
            "events": json.dumps(events) if events else None,
            # Kept in lockstep with LocalSink._flatten_row: DualSink mirrors
            # rows through here, and supa_mirror's natural-key matching
            # assumes both databases receive identical values.
            "intention_id": record.get("intention_id"),
            "align_score": record.get("align_score"),
        }