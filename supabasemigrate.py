#!/usr/bin/env python3
"""
migrate_to_supabase.py — One-time import of historical focus logs from xlsx to Supabase.

Only imports sheets matching the current schema (v6: March 9, 2026 onwards).

Usage:
    pip install supabase openpyxl python-dotenv
    python migrate_to_supabase.py logs.xlsx

Reads SUPABASE_URL and SUPABASE_KEY from .env
"""

import os
import sys
import json
from datetime import datetime

import openpyxl
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BATCH_SIZE = 500

# The current schema headers (v6) — only sheets matching this get imported
V6_HEADERS = [
    "timestamp", "current_window", "category_name", "is_productive",
    "human_label", "confidence", "reason", "main_label", "critic_label",
    "critic_conf", "typed_text",
]


def extract_events(row, e_start, row_len, current_window):
    """Collect e1–e10 into a deduplicated list, excluding current_window."""
    seen = set()
    events = []
    for i in range(e_start, min(e_start + 10, row_len)):
        val = row[i]
        if val and val not in seen and val != current_window:
            seen.add(val)
            events.append(val)
    return events


def process_sheet(ws):
    """Convert a worksheet into a list of dicts ready for Supabase insert."""
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    headers = [h for h in rows[0] if h is not None]

    # Verify this is a v6 sheet
    if headers[:11] != V6_HEADERS:
        return None  # wrong schema, skip

    e_start = len([h for h in rows[0] if h is not None and not h.startswith("e")])
    # Find where event columns start
    all_headers = list(rows[0])
    try:
        e_start = all_headers.index("e1")
    except ValueError:
        e_start = 11  # fallback

    records = []
    for r in rows[1:]:
        if not any(r):
            continue

        row_dict = dict(zip(all_headers, r))
        current_window = row_dict.get("current_window") or ""

        # Parse timestamp
        ts = row_dict.get("timestamp")
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        elif isinstance(ts, str) and ts.strip():
            ts_str = ts.strip()
        else:
            continue  # skip rows without timestamp

        # Parse confidence
        conf = row_dict.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = None

        # Build events array
        events = extract_events(r, e_start, len(r), current_window)

        records.append({
            "timestamp": ts_str,
            "current_window": current_window or None,
            "category_name": row_dict.get("category_name") or None,
            "confidence": conf,
            "reason": row_dict.get("reason") or None,
            "human_label": row_dict.get("human_label") or None,
            "typed_text": row_dict.get("typed_text") or None,
            "events": json.dumps(events) if events else None,
        })

    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_supabase.py <path-to-xlsx>")
        sys.exit(1)

    xlsx_path = sys.argv[1]

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[error] Set SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)

    print(f"Connecting to Supabase...")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"Reading {xlsx_path}...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    total_imported = 0
    total_skipped = 0

    for name in wb.sheetnames:
        if not name.startswith("Focus Logs - 20"):
            continue

        ws = wb[name]
        records = process_sheet(ws)

        if records is None:
            print(f"  {name}: skipped (old schema)")
            total_skipped += 1
            continue

        if not records:
            print(f"  {name}: skipped (empty)")
            continue

        # Batch insert
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            sb.table("focus_logs").insert(batch).execute()

        print(f"  {name}: imported {len(records)} rows")
        total_imported += len(records)

    wb.close()

    print(f"\nDone. Imported {total_imported} rows, skipped {total_skipped} sheets.")


if __name__ == "__main__":
    main()