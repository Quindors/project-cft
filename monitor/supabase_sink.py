# monitor/supabase_sink.py
from __future__ import annotations

import json
import re
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


class NullSink:
    """No-op sink used in testing/trace mode."""
    ws_title = "(testing mode — no database)"
    existing_rows_tracked = 0

    def ensure_day(self, day):
        return

    def enqueue_record(self, record):
        return

    def flush_if_needed(self):
        return None

    def close(self):
        return

    def load_corrections(self):
        return []

    def load_prior_classifications(self, max_sheets=7, min_confidence=0.85):
        return []

    def get_categories(self):
        return [
            {"name": "Deep Work", "description": "VS Code, PyCharm, coding, writing, focused tasks", "is_productive": True},
            {"name": "Homework", "description": "OneNote with HW notes, PDF textbooks, Desmos, educational PDFs, calculus tools, assignment work", "is_productive": True},
            {"name": "Research & Learning", "description": "Khan Academy, Wolfram Alpha, Wikipedia for homework topics, educational YouTube tutorials, documentation", "is_productive": True},
            {"name": "Communication", "description": "Teams for schoolwork, email about classes, Slack for project collaboration, work-related messaging", "is_productive": True},
            {"name": "Social Media", "description": "Twitter/X, Instagram, TikTok, Reddit browsing, Discord chatting (non-work), Facebook", "is_productive": False},
            {"name": "Entertainment", "description": "YouTube entertainment videos, Netflix, Twitch, Hulu, Prime Video, funny cat videos, streams", "is_productive": False},
            {"name": "Gaming", "description": "Steam, Epic Games, Minecraft, Roblox, game launchers, playing games", "is_productive": False},
            {"name": "Shopping", "description": "Amazon, eBay, AliExpress, online shopping, browsing products", "is_productive": False},
            {"name": "Other Off-Task", "description": "Any other off-task activity that doesn't fit the above categories", "is_productive": False},
            {"name": "Ambiguous", "description": "Activity that cannot be clearly categorized from available context", "is_productive": None},
        ]


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

        # For boot logging compatibility
        self.ws_title = "supabase"
        self._existing_rows_count: Optional[int] = None
        self._categories_cache: Optional[List[Dict]] = None

    @property
    def existing_rows_tracked(self) -> int:
        if self._existing_rows_count is None:
            try:
                today = date.today().isoformat()
                result = (
                    self.sb.table("focus_logs")
                    .select("id", count="exact")
                    .gte("timestamp", f"{today}T00:00:00")
                    .execute()
                )
                self._existing_rows_count = result.count or 0
            except Exception as e:
                print(f"[warn] Could not count today's rows: {e}")
                self._existing_rows_count = 0
        return self._existing_rows_count

    # ------------------- public API -------------------

    def ensure_day(self, day: date) -> None:
        """No-op — Supabase uses a single table, no daily tabs."""
        self.ws_title = f"supabase ({day:%Y-%m-%d})"

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
        if self._existing_rows_count is not None:
            self._existing_rows_count += appended
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

    def load_corrections(self, max_sheets: int = 7) -> List[Dict]:
        """Load recent human corrections. max_sheets is reinterpreted as days_back."""
        days_back = max_sheets
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

    def load_prior_classifications(self, max_sheets: int = 7, min_confidence: float = 0.85) -> List[Dict]:
        """Load recent high-confidence classifications for cache priming."""
        days_back = max_sheets
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        skip_categories = {"ambiguous", "[error]", "[parse_error]"}

        try:
            result = (
                self.sb.table("focus_logs")
                .select("current_window, category_name, confidence")
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

    # ------------------- internal -------------------

    @staticmethod
    def _clean_window_title(raw: str) -> str:
        """Strip WIN: prefix and trim for readability."""
        s = raw.strip()
        if s.upper().startswith("WIN:"):
            s = s[4:]
        return s

    def _flatten_row(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a decision record into a dict for Supabase insert.

        The record comes from runner._build_record() which produces:
            events: [{"key": "window title"}, ...] — last item is current window
            primary_confidence / primary_reason — AI pipeline outputs
            category_name — final classification
        """
        # Extract event titles from list of {"key": ...} dicts
        events_raw = record.get("events") or []
        titles = []
        for e in events_raw:
            if isinstance(e, dict):
                titles.append(str(e.get("key", "")))
            elif isinstance(e, str):
                titles.append(e)

        # Current window is the last event
        current_window = ""
        if titles:
            current_window = self._clean_window_title(titles[-1])

        # Deduplicate events, excluding current_window
        seen = set()
        events = []
        for t in titles:
            cleaned = self._clean_window_title(t)
            if cleaned and cleaned not in seen and cleaned != current_window:
                seen.add(cleaned)
                events.append(cleaned)

        # Map confidence/reason — _build_record uses primary_confidence/primary_reason
        confidence = record.get("primary_confidence") or record.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None

        reason = record.get("primary_reason") or record.get("reason") or None

        return {
            "timestamp": record.get("timestamp", datetime.now().isoformat()),
            "current_window": current_window or None,
            "category_name": record.get("category_name") or None,
            "confidence": confidence,
            "reason": reason,
            "human_label": record.get("human_label") or None,
            "typed_text": record.get("typed_text") or None,
            "events": json.dumps(events) if events else None,
        }