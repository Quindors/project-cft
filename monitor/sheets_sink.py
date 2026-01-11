# monitor/sheets_sink.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import date
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

    def _expected_headers(self) -> List[str]:
        base = [
            "timestamp",
            # Main LLM section
            "main_label",
            "primary_confidence",
            "primary_reason",
            # Factors section
            "factors_score",
            "factor_window_relevance",
            "factor_dwell_time",
            "factor_keystroke",
            "factor_trajectory",
            "factor_risky",
            # Critic section
            "critic_label",
            "critic_confidence",
            "critic_reason",
            # Vision section
            "vision_label",
            "vision_reason",
            # Final decision
            "label",
            # Human feedback
            "human_label",
            "human_reason",
            "typed_text",
        ]
        for i in range(1, self.cfg.max_event_cols + 1):
            base.append(f"e{i}_key")
        return base

    def _ensure_headers(self):
        headers = self._expected_headers()
        end_col = _col_letter(len(headers))
        self.ws.update(f"A1:{end_col}1", [headers], value_input_option="RAW")

    def _get_or_create_daily_ws(self, day: date):
        title = f"{self.cfg.worksheet_prefix} - {day:%Y-%m-%d}"
        try:
            ws = self.sh.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = self.sh.add_worksheet(title=title, rows="200", cols=str(len(self._expected_headers())), index=0)
            self.ws = ws
            self._ensure_headers()
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

    def _flatten_row(self, record: Dict[str, Any]) -> List[str]:
        def get(k, default=""):
            v = record.get(k, default)
            return "" if v is None else str(v)

        row: List[str] = [
            get("timestamp"),
            # Main LLM section
            get("main_label"),
            get("primary_confidence"),
            get("primary_reason"),
            # Factors section
            get("factors_score"),
            get("factor_window_relevance"),
            get("factor_dwell_time"),
            get("factor_keystroke"),
            get("factor_trajectory"),
            get("factor_risky"),
            # Critic section
            get("critic_label"),
            get("critic_confidence"),
            get("critic_reason"),
            # Vision section
            get("vision_label"),
            get("vision_reason"),
            # Final decision
            get("label"),
            # Human feedback
            "",  # human_label
            "",  # human_reason
            get("typed_text"),
        ]

        events = record.get("events") or []
        for i in range(self.cfg.max_event_cols):
            if i < len(events) and isinstance(events[i], dict):
                row.append(str(events[i].get("key", "")) or "")
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