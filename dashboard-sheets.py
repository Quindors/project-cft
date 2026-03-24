#!/usr/bin/env python3
"""
dashboard-sheets.py

Creates / refreshes a "Dashboard" tab in your Google Sheets workbook.

Layout (matches the React prototype):
  Row 0      : Title banner
  Rows 2-3   : Stat cards — Today %, This Week %, All-Time Avg %, Total Hours
  Row 5+     : "Today" pie data   (chart overlaid to the right)
  Below      : "Week" pie data    (chart overlaid to the right)
  Below      : "All-Time" trend   (date, productive %) — wide line chart

Charts:
  - Today donut pie    (anchored col D)
  - Week donut pie     (anchored col J)
  - All-Time line      (anchored col A, full width)

Fixes the date serialisation bug by writing dates as plain strings via RAW mode.

Run:  python dashboard-sheets.py
"""

import os
import base64
import tempfile
from datetime import datetime, timedelta

import pandas as pd
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
INTERVAL_SEC = 10  # approx seconds per logged event

PRODUCTIVE_LABELS = {
    "On-Task (legacy)", "Deep Work", "Homework",
    "Research & Learning", "Communication",
}

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


# ===================== DATA LOADING =====================

def _load_categories(sh) -> dict:
    try:
        ws = sh.worksheet("Categories")
        records = ws.get_all_records()
    except Exception:
        return {}
    cats = {}
    for r in records:
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        cats[name] = {
            "color": str(r.get("color", "#999999")),
            "is_productive": r.get("is_productive"),
        }
    return cats


def _map_label(label, cats: dict) -> tuple:
    """Returns (mapped_name, is_productive)."""
    if not label or str(label).strip() == "":
        return ("Unknown", None)
    label = str(label).strip()
    if label == "ON-TASK":
        return ("On-Task (legacy)", True)
    if label == "OFF-TASK":
        return ("Off-Task (legacy)", False)
    if label == "ABSTAIN":
        return ("Ambiguous", None)
    if label == "[PARSE_ERROR]":
        return ("Unknown", None)
    if label in cats:
        prod = cats[label].get("is_productive")
        if prod in (True, 1, 1.0, "TRUE", "True"):
            return (label, True)
        elif prod in (False, 0, 0.0, "FALSE", "False"):
            return (label, False)
        else:
            return (label, None)
    return (label, None)


def fetch_all_logs(sh, cats: dict) -> pd.DataFrame:
    frames = []
    for ws in sh.worksheets():
        title = ws.title
        if not (title.startswith("Focus Logs") or title.startswith("old Focus")):
            continue
        try:
            records = ws.get_all_records()
        except Exception:
            continue
        if not records:
            continue
        df = pd.DataFrame.from_records(records)

        ts_col = next((c for c in ("timestamp", "ts") if c in df.columns), None)
        label_col = next((c for c in ("category_name", "label") if c in df.columns), None)
        if ts_col is None or label_col is None:
            continue

        sub = df[[ts_col, label_col]].copy()
        sub.columns = ["timestamp", "raw_label"]
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub = sub.dropna(subset=["timestamp"])

        mapped = sub["raw_label"].apply(lambda x: _map_label(x, cats))
        sub["label"] = mapped.apply(lambda x: x[0])
        sub["is_productive"] = mapped.apply(lambda x: x[1])
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "label", "is_productive", "date"])

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    # FIX: write dates as plain strings to avoid Sheets serial-number bug
    combined["date"] = combined["timestamp"].dt.strftime("%Y-%m-%d")
    return combined


# ===================== AGGREGATION =====================

def _is_productive(label: str) -> bool:
    return label in PRODUCTIVE_LABELS


def compute_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "total": len(g),
            "productive": g["label"].apply(_is_productive).sum(),
        })
    ).reset_index()
    daily["prod_pct"] = (daily["productive"] / daily["total"] * 100).round(1)
    return daily.sort_values("date")


def compute_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("label").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


# ===================== SHEET WRITING =====================

def _get_or_create_ws(sh, title: str, rows: int = 200, cols: int = 20):
    try:
        ws = sh.worksheet(title)
        ws.clear()
        return ws
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols), index=0)


def _delete_existing_charts(sh, ws_id: int):
    """Remove all charts so we rebuild cleanly."""
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


def write_dashboard(sh, df: pd.DataFrame, daily: pd.DataFrame):
    ws = _get_or_create_ws(sh, DASHBOARD_TITLE)
    ws_id = ws.id
    _delete_existing_charts(sh, ws_id)

    # ── Compute slices ──
    today_str = df["date"].max()
    today_df = df[df["date"] == today_str]
    today_cats = compute_category_counts(today_df)

    all_dates = sorted(df["date"].unique())
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    recent = [d for d in all_dates if d > cutoff]
    week_dates = recent[-5:] if len(recent) >= 5 else recent
    week_df = df[df["date"].isin(week_dates)]
    week_cats = compute_category_counts(week_df)

    today_total = len(today_df)
    today_prod = today_df["label"].apply(_is_productive).sum()
    today_pct = round(today_prod / today_total * 100) if today_total else 0
    today_mins = round(today_total * INTERVAL_SEC / 60)

    week_total = len(week_df)
    week_prod = week_df["label"].apply(_is_productive).sum()
    week_pct = round(week_prod / week_total * 100) if week_total else 0

    alltime_avg = round(daily["prod_pct"].mean(), 1)
    total_hours = round(len(df) * INTERVAL_SEC / 3600, 1)
    date_range_str = f"{all_dates[0][:7]} – {all_dates[-1][:7]}"

    # ── Assemble rows ──
    # Rows 0-3: visible stats.  Row 4+: chart source data (hidden via font color).
    rows = []

    # 0: Title
    rows.append(["FOCUS DASHBOARD", "", "", "",
                 f"Updated {datetime.now().strftime('%b %d, %H:%M')}"])
    # 1: Stat labels
    rows.append(["Today", "", "This Week", "", "All-Time Avg", "", "Total Tracked", ""])
    # 2: Stat values + sub-text interleaved
    rows.append([
        f"{today_pct}%", f"{today_mins} min",
        f"{week_pct}%",  f"{week_total} events",
        f"{alltime_avg}%", f"{len(daily)} sessions",
        f"{total_hours} hrs", date_range_str,
    ])
    # 3: blank spacer (charts start overlaying from row 3)
    rows.append([])

    # --- Hidden chart source data (row 4+) ---
    # Today pie data
    today_header_row = len(rows)
    rows.append(["Category", "Events"])
    for _, r in today_cats.iterrows():
        rows.append([str(r["label"]), int(r["count"])])
    today_data_count = len(today_cats)

    # Week pie data
    week_header_row = len(rows)
    rows.append(["Category", "Events"])
    for _, r in week_cats.iterrows():
        rows.append([str(r["label"]), int(r["count"])])
    week_data_count = len(week_cats)

    # Trend data
    trend_header_row = len(rows)
    rows.append(["Date", "Productive %"])
    for _, r in daily.iterrows():
        rows.append([str(r["date"]), float(r["prod_pct"])])
    trend_data_count = len(daily)

    # ── Write (RAW so dates stay as strings) ──
    ws.update(values=rows, range_name="A1", value_input_option="RAW")
    print(f"  Wrote {len(rows)} rows")

    # ── Formatting ──
    reqs = []

    WHITE  = {"red": 1.0, "green": 1.0, "blue": 1.0}
    LGRAY  = {"red": 0.40, "green": 0.40, "blue": 0.40}
    BG     = {"red": 1.0, "green": 1.0, "blue": 1.0}  # white bg

    # Reset entire sheet to white bg + default text (clears leftover dark theme)
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 200,
                  "startColumnIndex": 0, "endColumnIndex": 20},
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

    # Stat labels (row 1) — small gray uppercase
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2,
                  "startColumnIndex": 0, "endColumnIndex": 8},
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": False, "fontSize": 9, "foregroundColor": LGRAY},
        }},
        "fields": "userEnteredFormat.textFormat",
    }})

    # Stat values (row 2) — big bold numbers in A, C, E, G
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

    # Hide chart source data (row 4+) — white text on white bg
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 4, "endRowIndex": len(rows),
                  "startColumnIndex": 0, "endColumnIndex": 10},
        "cell": {"userEnteredFormat": {
            "textFormat": {"fontSize": 8, "foregroundColor": WHITE},
            "backgroundColor": BG,
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

    sh.batch_update({"requests": reqs})
    print("  Applied formatting")

    return ws, {
        "today_header_row": today_header_row,
        "today_data_count": today_data_count,
        "week_header_row": week_header_row,
        "week_data_count": week_data_count,
        "trend_header_row": trend_header_row,
        "trend_data_count": trend_data_count,
    }


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

    # Today pie — right below stats, left half
    reqs.append(_pie(3, 0, "Today", pos["today_header_row"], pos["today_data_count"]))
    # Week pie — right below stats, right half
    reqs.append(_pie(3, 6, "This Week", pos["week_header_row"], pos["week_data_count"]))

    # All-time trend line
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
                            "startColumnIndex": 1, "endColumnIndex": 2,
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
    cats = _load_categories(sh)
    print(f"  Found {len(cats)} categories")

    print("Reading all Focus Logs sheets...")
    df = fetch_all_logs(sh, cats)
    print(f"  Loaded {len(df)} total events")

    if df.empty:
        print("No data found. Exiting.")
        return

    print("Computing aggregations...")
    daily = compute_daily_trend(df)

    print("Writing Dashboard sheet...")
    ws, positions = write_dashboard(sh, df, daily)

    print("Adding charts...")
    add_charts(sh, ws.id, positions)

    print(f"\nDone! Open your spreadsheet to see the '{DASHBOARD_TITLE}' tab.")


if __name__ == "__main__":
    main()