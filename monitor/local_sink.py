# monitor/local_sink.py
"""SQLite-backed sink for standalone / distributed installs.

Mirrors the SupabaseSink interface but stores everything in a local SQLite
file under the per-user data dir. No cloud, no keys, fully private — this is
what distributed copies of the app use. Uses only the Python stdlib.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from monitor.paths import data_dir


@dataclass
class FlushResult:
    updated: int
    appended: int
    sent: int


# Seeded into a fresh database so a brand-new user has sensible categories.
_DEFAULT_CATEGORIES = [
    ("Deep Work", "VS Code, PyCharm, coding, writing, focused tasks", True),
    ("Homework", "OneNote with HW notes, PDF textbooks, Desmos, educational PDFs, assignment work", True),
    ("Research & Learning", "Khan Academy, Wolfram Alpha, Wikipedia for homework, educational YouTube, documentation", True),
    ("Communication", "Teams/email/Slack for school or work collaboration", True),
    ("Social Media", "Twitter/X, Instagram, TikTok, Reddit, Discord (non-work), Facebook", False),
    ("Entertainment", "YouTube entertainment, Netflix, Twitch, Hulu, streams", False),
    ("Gaming", "Steam, Epic Games, Minecraft, Roblox, game launchers", False),
    ("Shopping", "Amazon, eBay, AliExpress, online shopping", False),
    ("Other Off-Task", "Any other off-task activity not covered above", False),
    ("Ambiguous", "Activity that cannot be clearly categorized", None),
]


# How far back a labeled row still defines a live correction rule. The
# dashboard's rule list (GET /api/corrections) shows the same window, so it
# never displays a rule the decider no longer applies.
CORRECTIONS_DAYS_BACK = 7


def _normalize_title(title: str) -> str:
    s = title.strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    s = re.sub(r"^\(\d+\)\s*", "", s)
    return " ".join(s.split()).lower()


def _clean_window_title(raw: str) -> str:
    s = (raw or "").strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    return s


class LocalSink:
    def __init__(self, settings):
        self.settings = settings
        self.cfg = settings.supabase  # reuse batch/idle timing knobs
        self.db_path = str(data_dir() / "focus.db")

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

        self.buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self._categories_cache: Optional[List[Dict]] = None

    # ------------------- schema -------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS focus_logs (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT,
                    current_window TEXT,
                    category_name  TEXT,
                    confidence     REAL,
                    reason         TEXT,
                    human_label    TEXT,
                    typed_text     TEXT,
                    events         TEXT,
                    reviewed_at    TEXT,
                    intention_id   INTEGER,
                    align_score    REAL
                );
                CREATE INDEX IF NOT EXISTS idx_focus_ts ON focus_logs(timestamp);
                CREATE TABLE IF NOT EXISTS categories (
                    name          TEXT PRIMARY KEY,
                    description   TEXT,
                    is_productive INTEGER
                );
                """
            )
            # Migrate pre-intention databases in place.
            for col, typ in (("intention_id", "INTEGER"), ("align_score", "REAL")):
                try:
                    self._conn.execute(f"ALTER TABLE focus_logs ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            # After the column exists (older databases only gain it just
            # above). Every per-session read — the forest's /api/sessions,
            # Beeminder's session totals — groups focus_logs by intention_id,
            # and without this each one scanned the whole table.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_focus_intention "
                "ON focus_logs(intention_id) WHERE intention_id IS NOT NULL"
            )
            # Error verdicts are transient noise; sweep any that made it in
            # (e.g. rows written by older builds).
            purged = self._conn.execute(
                "DELETE FROM focus_logs WHERE category_name IN ('[ERROR]', '[PARSE_ERROR]')"
            ).rowcount
            if purged:
                print(f"[sink] purged {purged} error rows from focus.db")
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM categories")
            if cur.fetchone()["n"] == 0:
                self._conn.executemany(
                    "INSERT INTO categories(name, description, is_productive) VALUES (?,?,?)",
                    [(n, d, (None if p is None else int(p))) for (n, d, p) in _DEFAULT_CATEGORIES],
                )
            self._conn.commit()

    # ------------------- public API (writes) -------------------

    def ensure_day(self, day) -> None:
        return

    def enqueue_record(self, record: Dict[str, Any]) -> None:
        self.buffer.append(self._flatten_row(record))

    def flush_if_needed(self) -> Optional[FlushResult]:
        if not self.buffer:
            return None
        now = time.time()
        should = (
            len(self.buffer) >= self.cfg.batch_flush_max
            or (now - self.last_flush_time) >= self.cfg.idle_flush_sec
        )
        return self.flush(force=True) if should else None

    def flush(self, force: bool = False) -> Optional[FlushResult]:
        if not self.buffer:
            return None
        batch = self.buffer[:]
        self.buffer.clear()
        try:
            with self._lock:
                self._conn.executemany(
                    """INSERT INTO focus_logs
                       (timestamp, current_window, category_name, confidence,
                        reason, human_label, typed_text, events, intention_id, align_score)
                       VALUES (:timestamp, :current_window, :category_name, :confidence,
                               :reason, :human_label, :typed_text, :events,
                               :intention_id, :align_score)""",
                    batch,
                )
                self._conn.commit()
        except Exception as e:
            print(f"[error] SQLite insert failed: {e}")
            self.buffer = batch + self.buffer
            return None
        self.last_flush_time = time.time()
        return FlushResult(updated=0, appended=len(batch), sent=len(batch))

    def close(self) -> None:
        self.flush(force=True)
        with self._lock:
            self._conn.close()

    # ------------------- reads -------------------

    def get_categories(self) -> List[Dict]:
        if self._categories_cache is not None:
            return self._categories_cache
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, description, is_productive FROM categories"
            ).fetchall()
        cats = []
        for r in rows:
            ip = r["is_productive"]
            cats.append({
                "name": r["name"],
                "description": r["description"],
                "is_productive": None if ip is None else bool(ip),
            })
        self._categories_cache = cats
        return cats

    def load_corrections(self, days_back: int = CORRECTIONS_DAYS_BACK) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """SELECT current_window, category_name, human_label FROM focus_logs
                   WHERE human_label IS NOT NULL AND human_label != ''
                     AND timestamp >= ? ORDER BY timestamp DESC LIMIT 500""",
                (cutoff,),
            ).fetchall()
        corrections: Dict[str, Dict] = {}
        for row in rows:
            window = (row["current_window"] or "").strip()
            ai_cat = (row["category_name"] or "").strip()
            human = (row["human_label"] or "").strip()
            if not human or not window or human.lower() == ai_cat.lower():
                continue
            if window not in corrections:
                corrections[window] = {
                    "window_title": window,
                    "ai_category": ai_cat,
                    "corrected_category": human,
                }
        return list(corrections.values())

    def load_prior_classifications(
        self, days_back: int = 7, min_confidence: float = 0.85, min_repeats: int = 3
    ) -> List[Dict]:
        """Uncorrected precedent for the few-shot prompt.

        A title earns a slot two ways: a single classification confident
        enough to trust on its own (>= min_confidence), or the same category
        repeated min_repeats times — a window judged "Composition, conf 0.5"
        thirty times over a session is settled precedent even though no one
        row clears the confidence bar. Each entry carries how many rows agree
        ("count"), so the prompt can say how settled it is.

        Rows whose category came from the screen rather than the title (the
        decider tags their reason "[vision]") are left out: a vision override
        says the title was uninformative or wrong, which is the opposite of
        evidence about what that title means next time.
        """
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        skip = {"ambiguous", "[error]", "[parse_error]"}
        # Counting happens in SQL so the whole days_back window is surveyed:
        # a raw-row LIMIT here silently shrank "7 days" to however long 4000
        # decisions take (~11 active hours at the 10s interval). The GROUP BY
        # output is one row per distinct (title spelling, category) pair, so
        # the LIMIT below is a safety cap, not a window.
        with self._lock:
            rows = self._conn.execute(
                """SELECT current_window, category_name, COUNT(*) AS repeats,
                          MAX(confidence) AS confidence, MAX(timestamp) AS last_seen
                   FROM focus_logs
                   WHERE human_label IS NULL AND timestamp >= ?
                     AND (reason IS NULL OR reason NOT LIKE '[vision]%')
                   GROUP BY current_window, category_name
                   ORDER BY last_seen DESC LIMIT 2000""",
                (cutoff,),
            ).fetchall()
        # Groups arrive newest-first, so the first spelling seen for a pair
        # keeps the freshest title spelling, and dict order favors recency on
        # ties.
        agg: Dict[tuple, Dict] = {}
        for row in rows:
            window = (row["current_window"] or "").strip()
            category = (row["category_name"] or "").strip()
            if not window or not category or category.lower() in skip:
                continue
            norm = _normalize_title(window)
            entry = agg.get((norm, category.lower()))
            if entry is None:
                agg[(norm, category.lower())] = {
                    "window_title": window,
                    "normalized_title": norm,
                    "category_name": category,
                    "confidence": row["confidence"] or 0.0,
                    "count": row["repeats"],
                }
            else:
                entry["count"] += row["repeats"]
                entry["confidence"] = max(entry["confidence"], row["confidence"] or 0.0)
        out: Dict[str, Dict] = {}
        for entry in agg.values():
            if entry["confidence"] < min_confidence and entry["count"] < min_repeats:
                continue
            # Repeat-qualified low-confidence entries are precedent for their
            # own title only. Every leakage incident on record came from a
            # DIFFERENT window's example being cited as current evidence, so
            # cross-title few-shot keeps the high-confidence bar.
            entry["high_conf"] = entry["confidence"] >= min_confidence
            best = out.get(entry["normalized_title"])
            if best is None or entry["count"] > best["count"]:
                out[entry["normalized_title"]] = entry
        return list(out.values())

    # ------------------- learning loop: review + accuracy -------------------

    def has_review_column(self) -> bool:
        return True

    def load_labeled(self, days_back: Optional[int] = None) -> List[Dict]:
        q = ("SELECT current_window, category_name, human_label, confidence, timestamp "
             "FROM focus_logs WHERE human_label IS NOT NULL AND human_label != ''")
        params: tuple = ()
        if days_back is not None:
            q += " AND timestamp >= ?"
            params = ((datetime.now() - timedelta(days=days_back)).isoformat(),)
        q += " ORDER BY timestamp DESC LIMIT 10000"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def load_reviewable(self, limit: int = 30, days_back: Optional[int] = None) -> List[Dict]:
        q = ("SELECT id, timestamp, current_window, category_name, confidence, reason, events "
             "FROM focus_logs WHERE human_label IS NULL AND current_window IS NOT NULL")
        params: list = []
        if days_back is not None:
            q += " AND timestamp >= ?"
            params.append((datetime.now() - timedelta(days=days_back)).isoformat())
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def save_review(self, row_id, human_label: str) -> bool:
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE focus_logs SET human_label = ?, reviewed_at = ? WHERE id = ?",
                    (human_label, datetime.now().isoformat(), row_id),
                )
                self._conn.commit()
            return True
        except Exception as e:
            print(f"[error] save_review failed: {e}")
            return False

    # ------------------- internal -------------------

    def _flatten_row(self, record: Dict[str, Any]) -> Dict[str, Any]:
        titles = [str(e.get("key", "")) for e in (record.get("events") or []) if isinstance(e, dict)]
        current_window = _clean_window_title(titles[-1]) if titles else ""

        seen = set()
        events = []
        for t in titles:
            cleaned = _clean_window_title(t)
            if cleaned and cleaned not in seen and cleaned != current_window:
                seen.add(cleaned)
                events.append(cleaned)

        raw_conf = record.get("primary_confidence")
        try:
            confidence = float(raw_conf) if raw_conf not in ("", None) else None
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
            "intention_id": record.get("intention_id"),
            "align_score": record.get("align_score"),
        }
