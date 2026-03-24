# monitor/sheets_sink.py
from __future__ import annotations

import random
import time
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Any, Optional

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

from monitor.config import Settings, DEFAULT_SETTINGS


@dataclass
class FlushResult:
    updated: int
    appended: int
    sent: int


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def _normalize_title(title: str) -> str:
    """Normalize a window title for cache matching."""
    s = title.strip()
    if s.upper().startswith("WIN:"):
        s = s[4:]
    s = re.sub(r"^\(\d+\)\s*", "", s)  # strip leading (N) notification counts
    return " ".join(s.split()).lower()

class NullSink:
    """
    No-op sink used in testing/trace mode to guarantee nothing is uploaded.
    """
    ws_title = "(testing mode — no sheets)"
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


class SheetsSink:
    def __init__(self, service_account_json_path: str, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self.cfg = settings.sheets

        self.service_account_json_path = service_account_json_path
        self.gc = None
        self.sh = None
        self.ws = None
        self.ws_title = ""
        self.key_to_row: Dict[str, int] = {}

        self.buffer: List[List[str]] = []
        self.last_flush_time = time.time()

        # Categories worksheet reference
        self.categories_ws = None
        self._categories_cache: Optional[List[Dict]] = None

        self._open()

    @property
    def existing_rows_tracked(self) -> int:
        return len(self.key_to_row)

    def _open(self):
        creds = Credentials.from_service_account_file(self.service_account_json_path, scopes=self.cfg.scopes)
        self.gc = gspread.authorize(creds)
        if self.cfg.sheet_url_or_key.startswith("http"):
            self.sh = self.gc.open_by_url(self.cfg.sheet_url_or_key)
        else:
            self.sh = self.gc.open_by_key(self.cfg.sheet_url_or_key)
        
        # Ensure Categories sheet exists on startup
        self._ensure_categories_sheet()

    def _ensure_categories_sheet(self):
        """Create or get Categories worksheet and seed with defaults if new."""
        try:
            self.categories_ws = self.sh.worksheet("Categories")
            print("[info] Found existing Categories worksheet")
        except gspread.exceptions.WorksheetNotFound:
            print("[info] Creating Categories worksheet...")
            self.categories_ws = self.sh.add_worksheet(
                title="Categories",
                rows="50",
                cols="6",
                index=0  # Put at beginning of workbook
            )
            # Write headers
            headers = ["category_id", "name", "description", "color", "is_productive", "created_at"]
            self.categories_ws.update("A1:F1", [headers], value_input_option="RAW")
            
            # Seed default categories
            self._seed_default_categories()
            print("[info] Categories worksheet created and seeded with defaults")

    def _seed_default_categories(self):
        """Add default categories based on typical student workflow."""
        now_str = datetime.now().isoformat()
        
        defaults = [
            # Productive categories
            ["deep_work", "Deep Work", "VS Code, PyCharm, coding, writing, focused tasks. Examples: .py files, homework assignments, project work", "#2ecc71", "TRUE", now_str],
            ["homework", "Homework", "OneNote with HW notes, PDF textbooks, Desmos, educational PDFs, calculus tools, assignment work", "#9b59b6", "TRUE", now_str],
            ["research", "Research & Learning", "Khan Academy, Wolfram Alpha, Wikipedia for homework topics, educational YouTube tutorials, documentation", "#3498db", "TRUE", now_str],
            ["communication", "Communication", "Teams for schoolwork, email about classes, Slack for project collaboration, work-related messaging", "#f39c12", "TRUE", now_str],
            
            # Non-productive categories
            ["social_media", "Social Media", "Twitter/X, Instagram, TikTok, Reddit browsing, Discord chatting (non-work), Facebook", "#e74c3c", "FALSE", now_str],
            ["entertainment", "Entertainment", "YouTube entertainment videos, Netflix, Twitch, Hulu, Prime Video, funny cat videos, streams", "#c0392b", "FALSE", now_str],
            ["gaming", "Gaming", "Steam, Epic Games, Minecraft, Roblox, game launchers, playing games", "#8e44ad", "FALSE", now_str],
            ["shopping", "Shopping", "Amazon, eBay, AliExpress, online shopping, browsing products", "#d35400", "FALSE", now_str],
            ["other", "Other Off-Task", "Any other off-task activity that doesn't fit the above categories", "#95a5a6", "FALSE", now_str],
            
            # Ambiguous
            ["ambiguous", "Ambiguous", "Activity that cannot be clearly categorized as productive or non-productive from available context", "#bdc3c7", "", now_str],
        ]
        
        self.categories_ws.append_rows(defaults, value_input_option="USER_ENTERED")

    def get_categories(self) -> List[Dict]:
        """
        Load all categories from Categories sheet.
        Returns list of dicts with keys: category_id, name, description, color, is_productive, created_at
        """
        if self._categories_cache is not None:
            return self._categories_cache
        
        if not self.categories_ws:
            self._ensure_categories_sheet()
        
        records = self.categories_ws.get_all_records()
        
        # Convert to proper types
        categories = []
        for r in records:
            cat = dict(r)
            # Convert is_productive to boolean (empty string = None for Ambiguous)
            if isinstance(cat.get('is_productive'), str):
                val = cat['is_productive'].strip().upper()
                if val in ('TRUE', 'YES', '1'):
                    cat['is_productive'] = True
                elif val in ('FALSE', 'NO', '0'):
                    cat['is_productive'] = False
                else:
                    cat['is_productive'] = None
            categories.append(cat)
        
        self._categories_cache = categories
        return categories
    
    def load_corrections(self, max_sheets: int = 7) -> List[Dict]:
        """Scan recent daily sheets for human corrections.

        Returns deduplicated list of dicts:
            {window_title, ai_category, corrected_category}
        Keyed by window_title (latest correction wins).
        """
        headers = self._expected_headers()
        idx_window = headers.index("current_window")
        idx_ai = headers.index("category_name")
        idx_human = headers.index("human_label")

        # Find daily sheets (they start with the worksheet prefix)
        prefix = self.cfg.worksheet_prefix
        daily_sheets = []
        for ws in self.sh.worksheets():
            if ws.title.startswith(prefix):
                daily_sheets.append(ws)

        # Sort by title descending (newest first), limit
        daily_sheets.sort(key=lambda w: w.title, reverse=True)
        daily_sheets = daily_sheets[:max_sheets]

        # Collect corrections, keyed by window title (latest wins)
        corrections: Dict[str, Dict] = {}

        for ws in daily_sheets:
            try:
                rows = ws.get_all_values()
            except Exception as e:
                print(f"[warn] Could not read {ws.title}: {e}")
                continue

            for row in rows[1:]:  # skip header
                if len(row) <= max(idx_window, idx_ai, idx_human):
                    continue

                window = row[idx_window].strip()
                ai_cat = row[idx_ai].strip()
                human = row[idx_human].strip()

                if not human or not window:
                    continue
                if human.lower() == ai_cat.lower():
                    continue  # not a correction, user agreed

                # Deduplicate by window title (latest sheet = first in list)
                if window not in corrections:
                    corrections[window] = {
                        "window_title": window,
                        "ai_category": ai_cat,
                        "corrected_category": human,
                    }

        result = list(corrections.values())
        print(f"[info] Loaded {len(result)} corrections from {len(daily_sheets)} sheets")
        return result
    
    def load_prior_classifications(self, max_sheets: int = 7, min_confidence: float = 0.85) -> List[Dict]:
    """Scan recent daily sheets for high-confidence prior classifications.

    Returns deduplicated list of dicts:
        {window_title, normalized_title, category_name, confidence}
    Keyed by normalized window title (latest classification wins).
    Skips Ambiguous, errors, and rows with human corrections.
    """
    headers = self._expected_headers()
    idx_window = headers.index("current_window")
    idx_category = headers.index("category_name")
    idx_conf = headers.index("confidence")
    idx_human = headers.index("human_label")

    prefix = self.cfg.worksheet_prefix
    daily_sheets = []
    for ws in self.sh.worksheets():
        if ws.title.startswith(prefix):
            daily_sheets.append(ws)

    daily_sheets.sort(key=lambda w: w.title, reverse=True)
    daily_sheets = daily_sheets[:max_sheets]

    classifications: Dict[str, Dict] = {}
    skip_categories = {"ambiguous", "[error]", "[parse_error]"}

    for ws in daily_sheets:
        try:
            rows = ws.get_all_values()
        except Exception as e:
            print(f"[warn] Could not read {ws.title}: {e}")
            continue

        for row in rows[1:]:
            if len(row) <= max(idx_window, idx_category, idx_conf, idx_human):
                continue

            window = row[idx_window].strip()
            category = row[idx_category].strip()
            conf_str = row[idx_conf].strip()
            human = row[idx_human].strip()

            if not window or not category:
                continue
            if human:
                continue  # handled by load_corrections
            if category.lower() in skip_categories:
                continue

            try:
                conf = float(conf_str)
            except (ValueError, TypeError):
                continue
            if conf < min_confidence:
                continue

            normalized = _normalize_title(window)
            if normalized not in classifications:
                classifications[normalized] = {
                    "window_title": window,
                    "normalized_title": normalized,
                    "category_name": category,
                    "confidence": conf,
                }

    result = list(classifications.values())
    print(f"[info] Loaded {len(result)} prior classifications from {len(daily_sheets)} sheets")
    return result

    def _expected_headers(self) -> List[str]:
        """Define the headers for daily Focus Logs sheets."""
        base = [
            # --- User-facing (left side, scannable) ---
            "timestamp",
            "current_window",
            "category_name",
            "is_productive",
            "human_label",       # blank = AI is correct
            # --- AI reasoning ---
            "confidence",
            "reason",
            # --- Debug / pipeline (right side) ---
            "main_label",
            "critic_label",
            "critic_conf",
            "typed_text",
        ]
        # Event columns (recent context)
        for i in range(1, self.cfg.max_event_cols + 1):
            base.append(f"e{i}")
        return base

    def _ensure_headers(self):
        headers = self._expected_headers()
        end_col = _col_letter(len(headers))
        self.ws.update(f"A1:{end_col}1", [headers], value_input_option="RAW")

    def _apply_sheet_formatting(self):
        """Add dropdown validation on human_label and conditional formatting."""
        headers = self._expected_headers()
        ws_id = self.ws.id

        # --- Column indices (0-based for Sheets API) ---
        col_human = headers.index("human_label")
        col_productive = headers.index("is_productive")
        col_category = headers.index("category_name")

        # --- Build category names for dropdown ---
        categories = self.get_categories()
        cat_names = [c["name"] for c in categories] if categories else []

        requests = []

        # 1) Data validation dropdown on human_label (rows 2-200)
        if cat_names:
            requests.append({
                "setDataValidation": {
                    "range": {
                        "sheetId": ws_id,
                        "startRowIndex": 1,
                        "endRowIndex": 200,
                        "startColumnIndex": col_human,
                        "endColumnIndex": col_human + 1,
                    },
                    "rule": {
                        "condition": {
                            "type": "ONE_OF_LIST",
                            "values": [{"userEnteredValue": n} for n in cat_names],
                        },
                        "showCustomUi": True,
                        "strict": False,  # allow blanks
                    },
                }
            })

        # 2) Conditional formatting: green background on is_productive = TRUE
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws_id,
                        "startRowIndex": 1,
                        "endRowIndex": 200,
                        "startColumnIndex": col_productive,
                        "endColumnIndex": col_productive + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "True"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.85, "green": 0.95, "blue": 0.85},
                        },
                    },
                },
                "index": 0,
            }
        })

        # 3) Conditional formatting: red background on is_productive = False
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws_id,
                        "startRowIndex": 1,
                        "endRowIndex": 200,
                        "startColumnIndex": col_productive,
                        "endColumnIndex": col_productive + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "False"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.95, "green": 0.85, "blue": 0.85},
                        },
                    },
                },
                "index": 1,
            }
        })

        # 4) Conditional formatting: grey on is_productive = None/empty
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws_id,
                        "startRowIndex": 1,
                        "endRowIndex": 200,
                        "startColumnIndex": col_productive,
                        "endColumnIndex": col_productive + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "None"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.92, "green": 0.92, "blue": 0.92},
                        },
                    },
                },
                "index": 2,
            }
        })

        # 5) Conditional formatting: highlight human_label when filled (correction made)
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": ws_id,
                        "startRowIndex": 1,
                        "endRowIndex": 200,
                        "startColumnIndex": col_human,
                        "endColumnIndex": col_human + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "NOT_BLANK",
                        },
                        "format": {
                            "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.8},
                            "textFormat": {"bold": True},
                        },
                    },
                },
                "index": 3,
            }
        })

        # 6) Freeze header row + auto-resize current_window column
        requests.append({
            "updateSheetProperties": {
                "properties": {
                    "sheetId": ws_id,
                    "gridProperties": {"frozenRowCount": 1},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        })

        # 7) Set column widths — wide for current_window and reason, narrow for others
        col_current = headers.index("current_window")
        col_reason = headers.index("reason")
        for col_idx, width in [(col_current, 350), (col_reason, 300), (col_human, 140)]:
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": ws_id,
                        "dimension": "COLUMNS",
                        "startIndex": col_idx,
                        "endIndex": col_idx + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            })

        # Execute all formatting in one batch call
        try:
            self.sh.batch_update({"requests": requests})
        except Exception as e:
            print(f"[warn] Sheet formatting failed (non-fatal): {e}")

    def _get_or_create_daily_ws(self, day: date):
        title = f"{self.cfg.worksheet_prefix} - {day:%Y-%m-%d}"
        try:
            ws = self.sh.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = self.sh.add_worksheet(title=title, rows="200", cols=str(len(self._expected_headers())), index=0)
            self.ws = ws
            self._ensure_headers()
            self._apply_sheet_formatting()
            return ws
        return ws

    def _read_existing_ts(self) -> Dict[str, int]:
        headers = self._expected_headers()
        idx_ts = headers.index("timestamp")
        values = self.ws.get_all_values()

        key_to_row: Dict[str, int] = {}
        for i, r in enumerate(values[1:], start=2):
            key = (r[idx_ts] if idx_ts < len(r) else "").strip()
            if key:
                key_to_row[key] = i
        return key_to_row

    def ensure_day(self, day: date) -> None:
        want_title = f"{self.cfg.worksheet_prefix} - {day:%Y-%m-%d}"
        if self.ws is not None and self.ws_title == want_title:
            return

        # flush before switching
        self.flush(force=True)

        self.ws = self._get_or_create_daily_ws(day)
        self.ws_title = self.ws.title
        self.key_to_row = self._read_existing_ts()

    def enqueue_record(self, record: Dict[str, Any]) -> None:
        row = self._flatten_row(record)
        self.buffer.append(row)

    def flush_if_needed(self) -> Optional[FlushResult]:
        now = time.time()
        if not self.buffer:
            return None
        should = (len(self.buffer) >= self.cfg.batch_flush_max) or ((now - self.last_flush_time) >= self.cfg.idle_flush_sec)
        if not should:
            return None
        return self.flush(force=True)

    def close(self) -> None:
        # final flush
        self.flush(force=True)

    # ------------------- internal formatting -------------------

    @staticmethod
    def _clean_window_title(raw: str) -> str:
        """Strip WIN: prefix and trim for readability."""
        s = raw.strip()
        if s.upper().startswith("WIN:"):
            s = s[4:]
        return s

    def _flatten_row(self, record: Dict[str, Any]) -> List[str]:
        """Convert record dict to row list matching headers."""
        def get(k, default=""):
            v = record.get(k, default)
            return "" if v is None else str(v)

        # Extract current window (last event, cleaned)
        events = record.get("events") or []
        events = events[-self.cfg.max_event_cols:]
        if events:
            last = events[-1]
            raw = str(last.get("key", "")) if isinstance(last, dict) else str(last)
            current_window = self._clean_window_title(raw)
        else:
            current_window = ""

        row: List[str] = [
            # --- User-facing ---
            get("timestamp"),
            current_window,
            get("category_name"),
            get("is_productive"),
            "",  # human_label (blank = AI correct)
            # --- AI reasoning ---
            get("primary_confidence"),
            get("primary_reason"),
            # --- Debug / pipeline ---
            get("main_label"),
            get("critic_label"),
            get("critic_confidence"),
            get("typed_text"),
        ]

        # Event columns (cleaned)
        for i in range(self.cfg.max_event_cols):
            if i < len(events):
                raw = str(events[i].get("key", "")) if isinstance(events[i], dict) else str(events[i])
                row.append(self._clean_window_title(raw))
            else:
                row.append("")
        return row

    # ------------------- retry + write -------------------

    def _apierror_status(self, e: APIError) -> Optional[int]:
        try:
            return getattr(e.response, "status_code", None)
        except Exception:
            return None

    def _retryable_call(self, fn, *args, **kwargs):
        for attempt in range(self.cfg.max_retries):
            try:
                return fn(*args, **kwargs)
            except APIError as e:
                code = self._apierror_status(e)
                msg = str(e)
                if ("429" in msg) or (code in (429, 500, 502, 503, 504)):
                    delay = (self.cfg.backoff_base * (2 ** attempt)) + random.random()
                    print(f"[warn] Sheets throttled (attempt {attempt+1}/{self.cfg.max_retries}) sleeping {delay:.2f}s")
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("Exceeded max retries for Sheets API call")

    def flush(self, force: bool = False) -> Optional[FlushResult]:
        if not self.buffer:
            return None

        headers = self._expected_headers()
        idx_ts = headers.index("timestamp")
        end_col = _col_letter(len(headers))

        buffer_rows = self.buffer[:]
        self.buffer.clear()

        updated = 0
        appended = 0

        if self.cfg.append_only:
            self._retryable_call(self.ws.append_rows, buffer_rows, value_input_option="USER_ENTERED")
            # best-effort update key_to_row
            start_row_guess = max(self.key_to_row.values(), default=1) + 1
            for row in buffer_rows:
                ts = (row[idx_ts] or "").strip()
                if ts and ts not in self.key_to_row:
                    self.key_to_row[ts] = start_row_guess
                    start_row_guess += 1
            appended = len(buffer_rows)
        else:
            updates_payload = []
            appends: List[List[str]] = []

            for row in buffer_rows:
                ts = (row[idx_ts] or "").strip()
                if ts and ts in self.key_to_row:
                    r = self.key_to_row[ts]
                    updates_payload.append({"range": f"A{r}:{end_col}{r}", "values": [row]})
                else:
                    appends.append(row)

            if updates_payload:
                self._retryable_call(self.ws.batch_update, updates_payload, value_input_option="USER_ENTERED")
                updated = len(updates_payload)

            if appends:
                self._retryable_call(self.ws.append_rows, appends, value_input_option="USER_ENTERED")
                start_row_guess = max(self.key_to_row.values(), default=1) + 1
                for row in appends:
                    ts = (row[idx_ts] or "").strip()
                    if ts and ts not in self.key_to_row:
                        self.key_to_row[ts] = start_row_guess
                        start_row_guess += 1
                appended = len(appends)

        self.last_flush_time = time.time()
        return FlushResult(updated=updated, appended=appended, sent=len(buffer_rows))