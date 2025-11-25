#!/usr/bin/env python3
"""
focus_line_per_day_live.py

Live dashboard: ON-TASK vs OFF-TASK counts per day as an interactive Plotly line chart,
auto-refreshing from Google Sheets using Dash.
"""

import os
from datetime import datetime

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ===================== CONFIG =====================

SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
WORKSHEET_PREFIX = "Focus Logs"  # tabs you created per day

SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# How often to refresh from Google Sheets (milliseconds)
REFRESH_INTERVAL_MS = 60_000  # 60 seconds

# ==================================================


def open_spreadsheet():
    """Authorize and open the Google Sheet."""
    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise FileNotFoundError(f"service account file not found: {SERVICE_ACCOUNT_JSON}")

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)

    if SHEET_URL_OR_KEY.startswith("http"):
        sh = gc.open_by_url(SHEET_URL_OR_KEY)
    else:
        sh = gc.open_by_key(SHEET_URL_OR_KEY)
    return sh


def load_focus_logs(sh) -> pd.DataFrame:
    """
    Load all tabs whose title starts with WORKSHEET_PREFIX into one DataFrame.

    Expected columns from your monitor script:
      ts, label, confidence, ai_reason, human_label, human_reason, e1_key, ...
    """
    frames = []
    for ws in sh.worksheets():
        if not ws.title.startswith(WORKSHEET_PREFIX):
            continue

        values = ws.get_all_values()
        if not values:
            continue

        headers = values[0]
        rows = values[1:]
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=headers)
        frames.append(df)

    if not frames:
        # return empty DataFrame instead of killing the app
        return pd.DataFrame(columns=["ts", "label"])

    df_all = pd.concat(frames, ignore_index=True)

    # Parse timestamps
    if "ts" in df_all.columns:
        df_all["ts_dt"] = pd.to_datetime(df_all["ts"], errors="coerce")
        df_all = df_all[df_all["ts_dt"].notna()]
    else:
        df_all["ts_dt"] = pd.NaT

    # Normalize labels
    if "label" in df_all.columns:
        df_all["label"] = df_all["label"].fillna("").astype(str)
    else:
        df_all["label"] = ""

    return df_all


def compute_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute counts of ON-TASK and OFF-TASK per day.
    Ignores [idle] and any other labels.
    """
    if df.empty or "ts_dt" not in df.columns:
        return pd.DataFrame(columns=["date", "label", "count"])

    df = df.copy()
    df["date"] = df["ts_dt"].dt.date

    mask = df["label"].isin(["ON-TASK", "OFF-TASK"])
    df = df[mask]

    if df.empty:
        return pd.DataFrame(columns=["date", "label", "count"])

    grouped = (
        df.groupby(["date", "label"])
          .size()
          .reset_index(name="count")
    )

    grouped["date"] = pd.to_datetime(grouped["date"])

    return grouped


def make_figure(daily_counts: pd.DataFrame):
    """Build the Plotly figure."""
    if daily_counts.empty:
        # Empty placeholder figure
        fig = px.line(
            title="ON-TASK vs OFF-TASK per day (no data yet)"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
        )
        return fig

    fig = px.line(
        daily_counts,
        x="date",
        y="count",
        color="label",
        markers=True,
        title="ON-TASK vs OFF-TASK per day",
        labels={
            "date": "Date",
            "count": "Number of events",
            "label": "Focus state",
        },
    )

    fig.update_layout(
        legend_title_text="Label",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Count",
    )
    return fig


# ===================== DASH APP =====================

app = Dash(__name__)
app.title = "Focus Monitor – Daily Line"


app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
           "padding": "20px"},
    children=[
        html.H2("Focus Monitor – ON-TASK vs OFF-TASK per day"),
        html.P(
            "This chart auto-refreshes from Google Sheets every "
            f"{REFRESH_INTERVAL_MS // 1000} seconds."
        ),
        dcc.Graph(id="daily-line-graph"),
        dcc.Interval(
            id="refresh-interval",
            interval=REFRESH_INTERVAL_MS,
            n_intervals=0,  # triggers immediately on load
        ),
    ],
)


@app.callback(
    Output("daily-line-graph", "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_graph(n):
    """Callback: re-pull data from Sheets and update figure."""
    try:
        sh = open_spreadsheet()
        df = load_focus_logs(sh)
        daily_counts = compute_daily_counts(df)
        fig = make_figure(daily_counts)
        return fig
    except Exception as e:
        # In case of error, show message in the figure title
        fig = px.line(title=f"Error loading data: {e}")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
        )
        return fig


if __name__ == "__main__":
    # debug=True gives you code hot-reload when you edit this file
    app.run(debug=True)