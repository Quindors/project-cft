#!/usr/bin/env python3
"""
dashboard-sheets.py

Creates a "Dashboard" tab with LIVE FORMULAS that auto-update.

Run this ONCE to set up the dashboard. After that, the formulas
pull from today's / this week's Focus Log sheets automatically.
Re-run only if you want to rebuild charts or extend the trend range.

Layout:
  Row 1      : Title + last-updated timestamp (formula)
  Row 2      : Stat labels
  Row 3      : Stat values (formulas — auto-update)
  Row 5+     : Today category breakdown (COUNTIF formulas)
  Below      : Week category breakdown (COUNTIF × 7 days)
  Below      : 90-day trend (one formula row per day)

Charts:
  - Today donut pie
  - Week donut pie
  - All-Time productivity trend line
"""

import os
import base64
import tempfile
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# ===================== CONFIG =====================

load_dotenv()

SHEET_URL_OR_KEY = (
    "https://docs.google.com/spreadsheets/d/"
    "1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
)
DASHBOARD_TITLE = "Dashboard"
WS_PREFIX = "Focus Logs"
INTERVAL_SEC = 10       # seconds per logged event (must match your monitor config)
TREND_DAYS = 90         # how many days back to pre-fill trend formulas

# ===================== AUTH =====================

b64 = os.getenv("GCP_SERVICE_ACCOUNT_B64")
if not b64:
    raise RuntimeError("GCP_SERVICE_ACCOUNT_B64 not set in .env")

_data = base64.b64decode(b64)
_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
_tmp.write(_data)
_tmp.flush()
_tmp.close()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
creds = Credentials.from_service_account_file(_tmp.name, scopes=SCOPES)
gc = gspread.authorize(creds)


def _open_sheet(url_or_key: str):
    if "/d/" in url_or_key:
        key = url_or_key.split("/d/")[1].split("/")[0]
    else:
        key = url_or_key
    return gc.open_by_key(key)


# ===================== HELPERS =====================

def _col_letter(n: int) -> str:
    """1-indexed column number to letter (1=A, 27=AA)."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _sheet_ref(days_ago: int = 0) -> str:
    """Return an INDIRECT formula fragment for a Focus Logs sheet N days ago.

    Example: _sheet_ref(0) → references today's sheet
             _sheet_ref(1) → references yesterday's sheet
    """
    if days_ago == 0:
        date_expr = 'TEXT(TODAY(),"YYYY-MM-DD")'
    else:
        date_expr = f'TEXT(TODAY()-{days_ago},"YYYY-MM-DD")'
    return f'"\'Focus Logs - "&{date_expr}&"\'"'


def _indirect_col(days_ago: int, col: str, start_row: int = 2) -> str:
    """Build INDIRECT("'Focus Logs - DATE'!C2:C") for a given column."""
    return f'INDIRECT({_sheet_ref(days_ago)}&"!{col}{start_row}:{col}")'


# ===================== LOAD CATEGORIES =====================

def _load_category_names(sh) -> list:
    """Get category names from the Categories sheet."""
    try:
        ws = sh.worksheet("Categories")
        records = ws.get_all_records()
    except Exception:
        return []
    names = []
    for r in records:
        name = str(r.get("name", "")).strip()
        if name:
            names.append(name)
    return names


# ===================== BUILD FORMULAS =====================

def _build_stat_formulas() -> dict:
    """Build the formulas for the 4 stat cards in row 3."""

    # Today: total events, productive events, percentage, minutes
    today_total = f'COUNTA({_indirect_col(0, "A")})'
    today_prod = f'COUNTIF({_indirect_col(0, "D")}, TRUE)'

    today_pct = f'=IFERROR(ROUND({today_prod}/{today_total}*100, 0)&"%", "0%")'
    today_sub = f'=IFERROR(ROUND({today_total}*{INTERVAL_SEC}/60, 0)&" min", "0 min")'

    # This week (last 7 days): sum totals and productive across 7 sheets
    week_total_parts = []
    week_prod_parts = []
    for d in range(7):
        week_total_parts.append(f'IFERROR(COUNTA({_indirect_col(d, "A")}), 0)')
        week_prod_parts.append(f'IFERROR(COUNTIF({_indirect_col(d, "D")}, TRUE), 0)')

    week_total = "+".join(week_total_parts)
    week_prod = "+".join(week_prod_parts)

    week_pct = f'=IFERROR(ROUND(({week_prod})/({week_total})*100, 0)&"%", "0%")'
    week_sub = f'=({week_total})&" events"'

    return {
        "today_pct": today_pct,
        "today_sub": today_sub,
        "week_pct": week_pct,
        "week_sub": week_sub,
    }


def _build_category_formulas(cat_names: list, section: str, start_row: int) -> list:
    """Build rows of [category_name, count_formula] for a category breakdown.

    section: "today" (1 day) or "week" (7 days)
    start_row: the 1-indexed row where this section starts (for reference)
    """
    rows = []
    for name in cat_names:
        escaped = name.replace('"', '""')
        if section == "today":
            formula = f'=IFERROR(COUNTIF({_indirect_col(0, "C")}, "{escaped}"), 0)'
        else:
            # Sum across 7 days
            parts = []
            for d in range(7):
                parts.append(f'IFERROR(COUNTIF({_indirect_col(d, "C")}, "{escaped}"), 0)')
            formula = f'={"+".join(parts)}'
        rows.append([name, formula])
    return rows


def _build_trend_formulas(num_days: int) -> list:
    """Build trend rows: [date, total_events, productive_events, productive_%].

    One row per day going back num_days from today.
    Each cell is a formula so it auto-updates.
    Returns rows in chronological order (oldest first).
    """
    rows = []
    for days_ago in range(num_days - 1, -1, -1):  # oldest first
        if days_ago == 0:
            date_formula = f'=TEXT(TODAY(), "YYYY-MM-DD")'
        else:
            date_formula = f'=TEXT(TODAY()-{days_ago}, "YYYY-MM-DD")'

        total_formula = f'=IFERROR(COUNTA({_indirect_col(days_ago, "A")}), "")'
        prod_formula = f'=IFERROR(COUNTIF({_indirect_col(days_ago, "D")}, TRUE), "")'

        # Productive % — only compute if there's data
        # Use the cells in the same row rather than recomputing
        # B=total, C=productive (relative to trend section)
        # We'll use a self-referencing approach after placement
        pct_formula = "PLACEHOLDER"  # will be set after we know the row

        rows.append([date_formula, total_formula, prod_formula, pct_formula])
    return rows


# ===================== WRITE DASHBOARD =====================

def _get_or_create_ws(sh, title: str, rows: int = 300, cols: int = 10):
    try:
        ws = sh.worksheet(title)
        ws.clear()
        return ws
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols), index=0)


def _delete_existing_charts(sh, ws_id: int):
    try:
        meta = sh.fetch_sheet_metadata({"includeGridData": False})
        for sheet in meta.get("sheets", []):
            if sheet["properties"]["sheetId"] == ws_id:
                for chart in sheet.get("charts", []):
                    sh.batch_update({"requests": [
                        {"deleteEmbeddedObject": {"objectId": chart["chartId"]}}
                    ]})
    except Exception:
        pass


def write_dashboard(sh, cat_names: list):
    ws = _get_or_create_ws(sh, DASHBOARD_TITLE)
    ws_id = ws.id
    _delete_existing_charts(sh, ws_id)

    stats = _build_stat_formulas()

    # ── Assemble all rows ──
    rows = []

    # Row 1 (index 0): Title
    rows.append([
        "FOCUS DASHBOARD", "", "", "",
        '="Updated "&TEXT(NOW(), "MMM DD, HH:MM")',
    ])

    # Row 2 (index 1): Stat labels
    rows.append(["Today", "", "This Week", "", "All-Time Avg", "", "Total Tracked", ""])

    # Row 3 (index 2): Stat values — placeholders, will fill with formulas below
    rows.append([
        stats["today_pct"], stats["today_sub"],
        stats["week_pct"], stats["week_sub"],
        "ALLTIME_PCT",  # placeholder — needs trend range
        "ALLTIME_SESSIONS",
        "TOTAL_HOURS",
        "DATE_RANGE",
    ])

    # Row 4 (index 3): blank spacer
    rows.append([])

    # --- Today category breakdown ---
    today_header_row = len(rows)  # 0-indexed row 4
    rows.append(["Category", "Events"])
    today_cat_rows = _build_category_formulas(cat_names, "today", today_header_row + 2)
    for r in today_cat_rows:
        rows.append(r)
    today_data_count = len(today_cat_rows)

    # blank spacer
    rows.append([])

    # --- Week category breakdown ---
    week_header_row = len(rows)
    rows.append(["Category", "Events"])
    week_cat_rows = _build_category_formulas(cat_names, "week", week_header_row + 2)
    for r in week_cat_rows:
        rows.append(r)
    week_data_count = len(week_cat_rows)

    # blank spacer
    rows.append([])

    # --- Trend data ---
    trend_header_row = len(rows)
    rows.append(["Date", "Total Events", "Productive", "Productive %"])
    trend_rows = _build_trend_formulas(TREND_DAYS)

    # Fix the % formula — now we know the actual row positions
    for i, tr in enumerate(trend_rows):
        actual_row = trend_header_row + 1 + i + 1  # +1 for header, +1 for 1-indexed
        # Col B = total, Col C = productive in that row
        tr[3] = f'=IFERROR(IF(B{actual_row}="", "", ROUND(C{actual_row}/B{actual_row}*100, 1)), "")'
        rows.append(tr)
    trend_data_count = len(trend_rows)

    # Fix all-time stats in row 3 — now we know trend range
    trend_pct_start = trend_header_row + 2  # first data row (1-indexed)
    trend_pct_end = trend_pct_start + trend_data_count - 1
    trend_total_col = f"B{trend_pct_start}:B{trend_pct_end}"
    trend_pct_col = f"D{trend_pct_start}:D{trend_pct_end}"

    # All-time avg %
    rows[2][4] = f'=IFERROR(ROUND(AVERAGEIF({trend_pct_col}, "<>"&""), 1)&"%", "—")'
    # Sessions count (days with data)
    rows[2][5] = f'=COUNTIF({trend_pct_col}, "<>"&"")&" sessions"'
    # Total tracked hours
    rows[2][6] = f'=IFERROR(ROUND(SUMIF({trend_total_col}, "<>"&"")*{INTERVAL_SEC}/3600, 1)&" hrs", "0 hrs")'
    # Date range
    first_date_cell = f"A{trend_pct_start}"
    last_date_cell = f"A{trend_pct_end}"
    rows[2][7] = f'=IFERROR({first_date_cell}&" — "&{last_date_cell}, "")'

    # ── Write all rows (USER_ENTERED so formulas evaluate) ──
    end_col = _col_letter(max(len(r) for r in rows if r))
    ws.update(
        values=rows,
        range_name=f"A1:{end_col}{len(rows)}",
        value_input_option="USER_ENTERED",
    )
    print(f"  Wrote {len(rows)} rows with live formulas")

    # ── Formatting ──
    _apply_formatting(sh, ws_id, len(rows), trend_header_row)

    return ws, {
        "today_header_row": today_header_row,
        "today_data_count": today_data_count,
        "week_header_row": week_header_row,
        "week_data_count": week_data_count,
        "trend_header_row": trend_header_row,
        "trend_data_count": trend_data_count,
    }


# ===================== FORMATTING =====================

def _apply_formatting(sh, ws_id: int, total_rows: int, trend_start: int):
    WHITE = {"red": 1.0, "green": 1.0, "blue": 1.0}
    LGRAY = {"red": 0.40, "green": 0.40, "blue": 0.40}

    reqs = []

    # Reset entire sheet
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows + 5,
                  "startColumnIndex": 0, "endColumnIndex": 10},
        "cell": {"userEnteredFormat": {
            "backgroundColor": WHITE,
            "textFormat": {"fontSize": 10, "bold": False,
                           "foregroundColor": {"red": 0, "green": 0, "blue": 0}},
        }},
        "fields": "userEnteredFormat",
    }})

    # Title row — bold, bigger
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": 10},
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": True, "fontSize": 14},
        }},
        "fields": "userEnteredFormat.textFormat",
    }})

    # Stat labels (row 2) — small gray
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2,
                  "startColumnIndex": 0, "endColumnIndex": 8},
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": False, "fontSize": 9, "foregroundColor": LGRAY},
        }},
        "fields": "userEnteredFormat.textFormat",
    }})

    # Stat values (row 3) — big bold in A, C, E, G
    for col in [0, 2, 4, 6]:
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 2, "endRowIndex": 3,
                      "startColumnIndex": col, "endColumnIndex": col + 1},
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": True, "fontSize": 22},
            }},
            "fields": "userEnteredFormat.textFormat",
        }})

    # Sub-text in B, D, F, H — small gray
    for col in [1, 3, 5, 7]:
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 2, "endRowIndex": 3,
                      "startColumnIndex": col, "endColumnIndex": col + 1},
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": False, "fontSize": 9, "foregroundColor": LGRAY},
                "verticalAlignment": "BOTTOM",
            }},
            "fields": "userEnteredFormat",
        }})

    # Hide trend source data (white text) — from trend_start onward
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": trend_start, "endRowIndex": total_rows + 5,
                  "startColumnIndex": 0, "endColumnIndex": 10},
        "cell": {"userEnteredFormat": {
            "textFormat": {"fontSize": 8, "foregroundColor": WHITE},
            "backgroundColor": WHITE,
        }},
        "fields": "userEnteredFormat",
    }})

    # Column widths
    for col, px in {0: 140, 1: 90, 2: 120, 3: 90, 4: 120, 5: 90, 6: 120, 7: 120}.items():
        reqs.append({
            "updateDimensionProperties": {
                "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                          "startIndex": col, "endIndex": col + 1},
                "properties": {"pixelSize": px},
                "fields": "pixelSize",
            }
        })

    # Freeze header row
    reqs.append({
        "updateSheetProperties": {
            "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount",
        }
    })

    sh.batch_update({"requests": reqs})
    print("  Applied formatting")


# ===================== CHARTS =====================

def add_charts(sh, ws_id: int, pos: dict):
    reqs = []

    def _pie(anchor_row, anchor_col, title, hdr_row, n_data, w=480, h=320):
        ds = hdr_row
        de = hdr_row + 1 + n_data
        return {
            "addChart": {"chart": {
                "position": {"overlayPosition": {
                    "anchorCell": {"sheetId": ws_id, "rowIndex": anchor_row, "columnIndex": anchor_col},
                    "widthPixels": w, "heightPixels": h,
                }},
                "spec": {
                    "title": title,
                    "titleTextFormat": {"fontSize": 11, "bold": True},
                    "pieChart": {
                        "legendPosition": "RIGHT_LEGEND",
                        "pieHole": 0.45,
                        "domain": {"sourceRange": {"sources": [{
                            "sheetId": ws_id,
                            "startRowIndex": ds, "endRowIndex": de,
                            "startColumnIndex": 0, "endColumnIndex": 1,
                        }]}},
                        "series": {"sourceRange": {"sources": [{
                            "sheetId": ws_id,
                            "startRowIndex": ds, "endRowIndex": de,
                            "startColumnIndex": 1, "endColumnIndex": 2,
                        }]}},
                    },
                },
            }}
        }

    # Today pie
    reqs.append(_pie(3, 0, "Today", pos["today_header_row"], pos["today_data_count"]))
    # Week pie
    reqs.append(_pie(3, 6, "This Week", pos["week_header_row"], pos["week_data_count"]))

    # All-time trend line — uses Date (col A) and Productive % (col D)
    ts = pos["trend_header_row"]
    te = ts + 1 + pos["trend_data_count"]
    reqs.append({
        "addChart": {"chart": {
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": ws_id, "rowIndex": 20, "columnIndex": 0},
                "widthPixels": 950, "heightPixels": 330,
            }},
            "spec": {
                "title": "All-Time Productivity Trend",
                "titleTextFormat": {"fontSize": 11, "bold": True},
                "basicChart": {
                    "chartType": "LINE",
                    "legendPosition": "NO_LEGEND",
                    "lineSmoothing": True,
                    "axis": [
                        {"position": "BOTTOM_AXIS", "title": "Date"},
                        {"position": "LEFT_AXIS", "title": "Productive %",
                         "viewWindowOptions": {"viewWindowMin": 0, "viewWindowMax": 100}},
                    ],
                    "domains": [{"domain": {"sourceRange": {"sources": [{
                        "sheetId": ws_id,
                        "startRowIndex": ts, "endRowIndex": te,
                        "startColumnIndex": 0, "endColumnIndex": 1,
                    }]}}}],
                    "series": [{
                        "series": {"sourceRange": {"sources": [{
                            "sheetId": ws_id,
                            "startRowIndex": ts, "endRowIndex": te,
                            "startColumnIndex": 3, "endColumnIndex": 4,
                        }]}},
                        "targetAxis": "LEFT_AXIS",
                        "type": "LINE",
                        "color": {"red": 0.20, "green": 0.83, "blue": 0.60},
                        "lineStyle": {"width": 3},
                        "pointStyle": {"size": 5},
                    }],
                    "headerCount": 1,
                },
            },
        }}
    })

    sh.batch_update({"requests": reqs})
    print("  Added 3 charts")


# ===================== MAIN =====================

def main():
    print("Opening spreadsheet...")
    sh = _open_sheet(SHEET_URL_OR_KEY)

    print("Loading categories...")
    cat_names = _load_category_names(sh)
    print(f"  Found {len(cat_names)} categories: {', '.join(cat_names)}")

    if not cat_names:
        print("No categories found in Categories sheet. Exiting.")
        return

    print("Writing Dashboard with live formulas...")
    ws, positions = write_dashboard(sh, cat_names)

    print("Adding charts...")
    add_charts(sh, ws.id, positions)

    print(f"\nDone! Dashboard is now live — no need to re-run this script.")
    print(f"Formulas auto-update as new Focus Log sheets are created.")
    print(f"Trend covers last {TREND_DAYS} days. Re-run to extend or rebuild charts.")


if __name__ == "__main__":
    main()