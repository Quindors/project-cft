#!/usr/bin/env python3
"""
focus_dashboard_live.py

Single-graph dashboard:

- Stacked bars (left Y): ON-TASK% and OFF-TASK% per day
- Line (right Y): ON-TASK hours per day

Data source: Google Sheets, using a service-account JSON
stored in env var GCP_SERVICE_ACCOUNT_B64 (base64-encoded).
"""

import os
import base64
import tempfile
from typing import List

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from dotenv import load_dotenv
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ===================== CONFIG =====================

load_dotenv()

# Your sheet (URL or key)
SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"

# Sampling interval used by your monitor (seconds per event)
INTERVAL_SEC = 3

# Auto-refresh interval (ms)
REFRESH_INTERVAL_MS = 60_000  # 60 seconds

# ===================== AUTH / GSPREAD =====================

b64 = os.getenv("GCP_SERVICE_ACCOUNT_B64")
if not b64:
    raise RuntimeError("GCP_SERVICE_ACCOUNT_B64 not set in .env")

data = base64.b64decode(b64)
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
tmp.write(data)
tmp.flush()
tmp.close()
SERVICE_ACCOUNT_JSON = tmp.name

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
gc = gspread.authorize(creds)


def _open_sheet(url_or_key: str):
    """Open a Google Sheet by full URL or just key."""
    if "/d/" in url_or_key:
        key = url_or_key.split("/d/")[1].split("/")[0]
    else:
        key = url_or_key
    return gc.open_by_key(key)


# ===================== DATA LOADING & AGGREGATION =====================

def fetch_focus_logs() -> pd.DataFrame:
    """
    Read all worksheets in the spreadsheet that contain 'ts' and 'label'
    columns and concatenate them into a single DataFrame.
    """
    sh = _open_sheet(SHEET_URL_OR_KEY)

    frames: List[pd.DataFrame] = []
    for ws in sh.worksheets():
        try:
            records = ws.get_all_records()
        except Exception:
            continue

        if not records:
            continue

        df = pd.DataFrame.from_records(records)

        if "ts" not in df.columns or "label" not in df.columns:
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["ts", "label"])

    return pd.concat(frames, ignore_index=True)


def compute_daily_metrics(raw_df: pd.DataFrame, interval_sec: int) -> pd.DataFrame:
    """
    From raw focus rows (ts, label, ...), compute per-day metrics:

      - ON-TASK / OFF-TASK counts
      - hours ON / OFF (using interval_sec)
      - percentages ON / OFF
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "date_str",
                "ON-TASK",
                "OFF-TASK",
                "total",
                "on_hours",
                "off_hours",
                "on_pct",
                "off_pct",
            ]
        )

    df = raw_df.copy()
    df = df[df["label"].isin(["ON-TASK", "OFF-TASK"])]

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["date"] = df["ts"].dt.date

    daily = (
        df.groupby(["date", "label"])
          .size()
          .unstack("label", fill_value=0)
          .reset_index()
    )

    for col in ["ON-TASK", "OFF-TASK"]:
        if col not in daily.columns:
            daily[col] = 0

    daily["total"] = daily["ON-TASK"] + daily["OFF-TASK"]
    daily = daily[daily["total"] > 0]   # drop days with no events

    daily["off_count"] = daily["OFF-TASK"]
    
    # Time metrics (hours)
    daily["on_hours"] = (daily["ON-TASK"] * interval_sec) / 3600.0
    daily["off_hours"] = (daily["OFF-TASK"] * interval_sec) / 3600.0

    # Percentage metrics
    daily["on_pct"] = daily["ON-TASK"] / daily["total"] * 100.0
    daily["off_pct"] = daily["OFF-TASK"] / daily["total"] * 100.0

    # For category x-axis (only actual dates that have data)
    daily["date_str"] = daily["date"].astype(str)

    return daily.sort_values("date")

# ===================== FIGURE: COMBINED VIEW =====================

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_combined_figure(daily: pd.DataFrame):
    """
    One figure:
      - bar (primary y): ON-TASK hours per day
      - line (secondary y): OFF-TASK event count (how many times you go off-task)
    """
    if daily.empty:
        fig = go.Figure()
        fig.update_layout(title="No focus data available yet")
        return fig

    x = daily["date_str"]  # categorical x-axis (only days with data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 👉 Single bar: ON-TASK hours (primary y)
    fig.add_trace(
        go.Bar(
            x=x,
            y=daily["on_hours"],
            name="ON-TASK hours",
            hovertemplate="<b>%{x}</b><br>ON: %{y:.2f} hours<extra></extra>",
        ),
        secondary_y=False,
    )

    # 👉 Line: OFF-TASK count (secondary y)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=daily["off_count"],       # or daily["OFF-TASK"]
            name="OFF-TASK events",
            mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>OFF-TASK events: %{y:d}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="ON-TASK hours vs OFF-TASK events",
        xaxis_title="Date",
        legend_title_text="",
        hovermode="x unified",
    )

    # Primary y-axis: hours
    fig.update_yaxes(
        title_text="Hours ON-TASK",
        secondary_y=False,
    )

    # Secondary y-axis: number of times you went off-task
    fig.update_yaxes(
        title_text="OFF-TASK events",
        secondary_y=True,
        showgrid=False,
    )

    fig.update_xaxes(type="category")  # only actual dates, no gaps

    return fig

# ===================== DASH APP =====================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Focus Dashboard", style={"textAlign": "center"}),

        dcc.Graph(id="focus_graph"),

        dcc.Interval(
            id="refresh_interval",
            interval=REFRESH_INTERVAL_MS,
            n_intervals=0,
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto"},
)


@app.callback(
    Output("focus_graph", "figure"),
    Input("refresh_interval", "n_intervals"),
)
def update_focus_graph(_n):
    raw_df = fetch_focus_logs()
    daily = compute_daily_metrics(raw_df, INTERVAL_SEC)
    return make_combined_figure(daily)


# ===================== MAIN =====================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
