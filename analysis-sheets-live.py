#!/usr/bin/env python3
"""
focus_dashboard_live.py

Single-graph dashboard:

- Bar (left Y): ON-TASK hours per day
- Lines (right Y): OFF-TASK events and ABSTAIN events per day

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

      - ON-TASK / OFF-TASK / ABSTAIN counts
      - hours ON / OFF (using interval_sec)
      - percentages ON / OFF / ABSTAIN
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "date_str",
                "ON-TASK",
                "OFF-TASK",
                "ABSTAIN",
                "total",
                "on_hours",
                "off_hours",
                "on_pct",
                "off_pct",
                "abstain_pct",
            ]
        )

    df = raw_df.copy()
    # Include ABSTAIN in the analysis
    df = df[df["label"].isin(["ON-TASK", "OFF-TASK", "ABSTAIN"])]

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["date"] = df["ts"].dt.date

    daily = (
        df.groupby(["date", "label"])
          .size()
          .unstack("label", fill_value=0)
          .reset_index()
    )

    # Ensure all three columns exist
    for col in ["ON-TASK", "OFF-TASK", "ABSTAIN"]:
        if col not in daily.columns:
            daily[col] = 0

    daily["total"] = daily["ON-TASK"] + daily["OFF-TASK"] + daily["ABSTAIN"]
    daily = daily[daily["total"] > 0]   # drop days with no events

    # Event counts (for plotting)
    daily["off_count"] = daily["OFF-TASK"]
    daily["abstain_count"] = daily["ABSTAIN"]
    
    # Time metrics (hours) - ABSTAIN treated as ON-TASK for time calculation
    daily["on_hours"] = ((daily["ON-TASK"] + daily["ABSTAIN"]) * interval_sec) / 3600.0
    daily["off_hours"] = (daily["OFF-TASK"] * interval_sec) / 3600.0

    # Percentage metrics
    daily["on_pct"] = daily["ON-TASK"] / daily["total"] * 100.0
    daily["off_pct"] = daily["OFF-TASK"] / daily["total"] * 100.0
    daily["abstain_pct"] = daily["ABSTAIN"] / daily["total"] * 100.0

    # For category x-axis (only actual dates that have data)
    daily["date_str"] = daily["date"].astype(str)

    return daily.sort_values("date")

# ===================== FIGURE: COMBINED VIEW =====================

def make_combined_figure(daily: pd.DataFrame):
    """
    One figure:
      - bar (primary y): ON-TASK hours per day (includes ABSTAIN as productive time)
      - line (secondary y): OFF-TASK event count (red)
      - line (secondary y): ABSTAIN event count (yellow/orange)
    """
    if daily.empty:
        fig = go.Figure()
        fig.update_layout(title="No focus data available yet")
        return fig

    x = daily["date_str"]  # categorical x-axis (only days with data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 👉 Bar: ON-TASK hours (primary y)
    fig.add_trace(
        go.Bar(
            x=x,
            y=daily["on_hours"],
            name="ON-TASK hours",
            marker_color="#2ecc71",  # Green
            hovertemplate="<b>%{x}</b><br>ON: %{y:.2f} hours<extra></extra>",
        ),
        secondary_y=False,
    )

    # 👉 Line: OFF-TASK count (secondary y)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=daily["off_count"],
            name="OFF-TASK events",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),  # Red
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>OFF-TASK: %{y:d} events<extra></extra>",
        ),
        secondary_y=True,
    )

    # 👉 Line: ABSTAIN count (secondary y)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=daily["abstain_count"],
            name="ABSTAIN events",
            mode="lines+markers",
            line=dict(color="#f39c12", width=2, dash="dash"),  # Orange, dashed
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>ABSTAIN: %{y:d} events<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Focus Tracking: ON-TASK hours vs Events",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(250,250,250,1)",
    )

    # Primary y-axis: hours
    fig.update_yaxes(
        title_text="Hours ON-TASK",
        secondary_y=False,
        gridcolor="rgba(200,200,200,0.5)",
    )

    # Secondary y-axis: event counts
    fig.update_yaxes(
        title_text="Event Count",
        secondary_y=True,
        showgrid=False,
    )

    fig.update_xaxes(
        type="category",
        gridcolor="rgba(200,200,200,0.3)",
    )

    return fig

# ===================== DASH APP =====================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Focus Dashboard", style={"textAlign": "center", "color": "#2c3e50"}),
        
        html.Div(
            [
                html.Div([
                    html.H3("Legend", style={"fontSize": "16px", "marginBottom": "10px"}),
                    html.P("🟢 ON-TASK hours: Productive work time (includes uncertain periods)", 
                           style={"fontSize": "14px", "margin": "5px 0"}),
                    html.P("🔴 OFF-TASK events: Times you drifted off-task", 
                           style={"fontSize": "14px", "margin": "5px 0"}),
                    html.P("🟠 ABSTAIN events: Ambiguous periods needing review", 
                           style={"fontSize": "14px", "margin": "5px 0"}),
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                    "border": "1px solid #dee2e6"
                }),
            ]
        ),

        dcc.Graph(id="focus_graph"),

        dcc.Interval(
            id="refresh_interval",
            interval=REFRESH_INTERVAL_MS,
            n_intervals=0,
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
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