# monitor/app.py
from __future__ import annotations

import os, openai

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.sheets_sink import SheetsSink, NullSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop

def _noop_popup(message: str, title: str = "Productivity Alert") -> None: return

def run(settings: Settings = DEFAULT_SETTINGS) -> None:
    load_env(override=True)

    require_openai_api_key(openai)

    os.makedirs(settings.log_dir, exist_ok=True)

    event_src, key_src = make_sources(settings=settings)

    testing_mode = settings.trace.use_trace_file
    
    #popup_fn = _noop_popup if testing_mode else show_popup
    popup_fn = show_popup

    # ✅ choose sink based on trace/testing settings
    if settings.trace.use_trace_file and not settings.trace.upload_to_sheets:
        sink = NullSink()
        print("[info] TESTING MODE: Sheets upload disabled.")
    else:
        service_account_json = build_service_account_json_from_b64()
        if not os.path.exists(service_account_json):
            raise SystemExit(f"[fatal] Service account JSON not found: {service_account_json}")
        sink = SheetsSink(service_account_json, settings=settings)

        sink.ensure_day(event_src.now_dt().date())
        print(f"[info] connected to sheet; tab='{sink.ws_title}', existing rows tracked: {sink.existing_rows_tracked}")

    if settings.trace.use_trace_file and hasattr(event_src, "sim_start"):
        print(f"[info] TRACE MODE ON. sim_start={event_src.sim_start}")
    else:
        print(f"[info] watching {settings.log_dir}")

    key_src.start()
    try:
        run_monitor_loop(
            event_src=event_src,
            key_src=key_src,
            sink=sink,
            show_popup=popup_fn,
            settings=settings,
        )
    finally:
        print("[info] final sheets flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()