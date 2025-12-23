# monitor/sheets_sink.py
from __future__ import annotations

import time
import random
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any, Optional

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

from monitor.config import (
    SHEET_URL_OR_KEY,
    WORKSHEET_PREFIX,
    SCOPES,
    MAX_EVENT_COLS,
    APPEND_ONLY,
    BATCH_FLUSH_MAX,
    IDLE_FLUSH_SEC,
    MAX_RETRIES,
    BACKOFF_BASE,
)


def expected_headers(max_event_cols: int = MAX_EVENT_COLS) -> List[str]:
    base = [
        "ts",
        "label",
        "primary_confidence",
        "primary_reason",
        "critic_confidence",
        "critic_reason",
        "human_label",
        "human_reason",
        "typed_text",
    ]
    for i in range(1, max_event_cols + 1):
        base.append(f"e{i}_key")
    return base


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def flatten_row(record: Dict[str, Any], max_event_cols: int = MAX_EVENT_COLS) -> List[str]:
    def get(k, default=""):
        v = record.get(k, default)
        return "" if v is None else str(v)

    row: List[str] = [
        get("ts"),
        get("label"),
        get("primary_confidence"),
        get("primary_reason"),
        get("critic_confidence"),
        get("critic_reason"),
        "",  # human_label
        "",  # human_reason
        get("typed_text"),
    ]

    events = record.get("events") or []
    for i in range(max_event_cols):
        if i < len(events) and isinstance(events[i], dict):
            row.append(str(events[i].get("key", "")) or "")
        else:
            row.append("")
    return row


def _apierror_status(e: APIError) -> Optional[int]:
    try:
        return getattr(e.response, "status_code", None)
    except Exception:
        return None


def _retryable_call(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            code = _apierror_status(e)
            msg = str(e)
            if ("429" in msg) or (code in (429, 500, 502, 503, 504)):
                delay = (BACKOFF_BASE * (2 ** attempt)) + random.random()
                print(f"[warn] Sheets throttled (attempt {attempt+1}/{MAX_RETRIES}) sleeping {delay:.2f}s")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Exceeded max retries for Sheets API call")


@dataclass
class FlushResult:
    updated: int = 0
    appended: int = 0
    sent: int = 0


class SheetsSink:
    """
    Owns:
      - Google Sheets connection
      - daily worksheet creation/selection
      - headers
      - buffering + flush policy (batch size / idle flush)
      - optional upsert-by-ts mode (APPEND_ONLY=False)
    """

    def __init__(self, service_account_json_path: str):
        self.service_account_json_path = service_account_json_path

        self.gc = None
        self.sh = None
        self.ws = None

        self.current_day: Optional[date] = None
        self.key_to_row: Dict[str, int] = {}  # only used when APPEND_ONLY=False

        self.buffer: List[List[str]] = []
        self.last_flush_time = time.time()

    # ---------- connection / worksheet ----------

    def connect(self):
        creds = Credentials.from_service_account_file(self.service_account_json_path, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sh = (
            self.gc.open_by_url(SHEET_URL_OR_KEY)
            if SHEET_URL_OR_KEY.startswith("http")
            else self.gc.open_by_key(SHEET_URL_OR_KEY)
        )

    def _ensure_headers(self):
        assert self.ws is not None
        headers = expected_headers()
        end_col = _col_letter(len(headers))
        _retryable_call(self.ws.update, f"A1:{end_col}1", [headers], value_input_option="RAW")

    def _get_or_create_daily_ws(self, day: date):
        assert self.sh is not None
        title = f"{WORKSHEET_PREFIX} - {day:%Y-%m-%d}"
        try:
            ws = self.sh.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = self.sh.add_worksheet(
                title=title,
                rows="200",
                cols=str(len(expected_headers())),
                index=0,
            )
            self.ws = ws
            self._ensure_headers()
        return ws

    def _read_existing_ts(self) -> Dict[str, int]:
        assert self.ws is not None
        headers = expected_headers()
        idx_ts = headers.index("ts")
        values = self.ws.get_all_values()
        key_to_row: Dict[str, int] = {}
        for i, r in enumerate(values[1:], start=2):
            key = (r[idx_ts] if idx_ts < len(r) else "").strip()
            if key:
                key_to_row[key] = i
        return key_to_row

    def ensure_day(self, day: date):
        """
        Idempotent: safe to call every loop.
        If day changes, flushes pending buffer to the previous day's sheet before switching.
        """
        if self.gc is None or self.sh is None:
            self.connect()

        if self.current_day is None:
            self.current_day = day
            self.ws = self._get_or_create_daily_ws(day)
            self._ensure_headers()
            self.key_to_row = {} if APPEND_ONLY else self._read_existing_ts()
            return

        if day != self.current_day:
            # flush remaining data to old day before switching
            self.flush(force=True)

            self.current_day = day
            self.ws = self._get_or_create_daily_ws(day)
            self._ensure_headers()
            self.key_to_row = {} if APPEND_ONLY else self._read_existing_ts()
            self.last_flush_time = time.time()
            print(f"[info] rolled over → new tab '{self.ws.title}'")

    # ---------- buffering ----------

    def enqueue_record(self, record: Dict[str, Any]):
        self.buffer.append(flatten_row(record))

    def enqueue_row(self, row: List[str]):
        self.buffer.append(row)

    @property
    def ws_title(self) -> str:
        return self.ws.title if self.ws else ""

    @property
    def existing_rows_tracked(self) -> int:
        return len(self.key_to_row)

    # ---------- flush ----------

    def _flush_buffer(self) -> FlushResult:
        assert self.ws is not None

        if not self.buffer:
            return FlushResult()

        buffer_rows = self.buffer
        self.buffer = []

        headers = expected_headers()
        idx_ts = headers.index("ts")
        end_col = _col_letter(len(headers))

        if APPEND_ONLY:
            _retryable_call(self.ws.append_rows, buffer_rows, value_input_option="USER_ENTERED")
            return FlushResult(updated=0, appended=len(buffer_rows), sent=len(buffer_rows))

        # Upsert mode by ts
        updates_payload = []
        appends: List[List[str]] = []

        for row in buffer_rows:
            key = (row[idx_ts] or "").strip()
            if key and key in self.key_to_row:
                r = self.key_to_row[key]
                updates_payload.append({"range": f"A{r}:{end_col}{r}", "values": [row]})
            else:
                appends.append(row)

        if updates_payload:
            _retryable_call(self.ws.batch_update, updates_payload, value_input_option="USER_ENTERED")

        if appends:
            _retryable_call(self.ws.append_rows, appends, value_input_option="USER_ENTERED")
            # refresh key_to_row best-effort without re-reading whole sheet:
            start_row_guess = max(self.key_to_row.values(), default=1) + 1
            for row in appends:
                key = (row[idx_ts] or "").strip()
                if key and key not in self.key_to_row:
                    self.key_to_row[key] = start_row_guess
                    start_row_guess += 1

        return FlushResult(updated=len(updates_payload), appended=len(appends), sent=len(buffer_rows))

    def flush_if_needed(self) -> Optional[FlushResult]:
        if not self.ws or not self.buffer:
            return None

        now = time.time()
        should_flush = (len(self.buffer) >= BATCH_FLUSH_MAX) or ((now - self.last_flush_time) >= IDLE_FLUSH_SEC)
        if not should_flush:
            return None

        self._ensure_headers()
        res = self._flush_buffer()
        self.last_flush_time = now
        return res

    def flush(self, force: bool = False) -> Optional[FlushResult]:
        if not self.ws or not self.buffer:
            return None
        if not force:
            return self.flush_if_needed()
        self._ensure_headers()
        res = self._flush_buffer()
        self.last_flush_time = time.time()
        return res

    def close(self):
        # best-effort final flush
        try:
            self.flush(force=True)
        except Exception as e:
            print(f"[warn] final flush failed: {e}")
