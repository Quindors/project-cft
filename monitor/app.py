# monitor/app.py
import os
import openai

from monitor.config import LOG_DIR, USE_TRACE_FILE
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.sheets_sink import SheetsSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop


def run() -> None:
    # env / creds
    load_env(override=True)
    service_account_json = build_service_account_json_from_b64()

    require_openai_api_key(openai)

    if not os.path.exists(service_account_json):
        raise SystemExit(f"[fatal] Service account JSON not found: {service_account_json}")

    os.makedirs(LOG_DIR, exist_ok=True)

    # sources + sink
    event_src, key_src = make_sources(USE_TRACE_FILE)
    sink = SheetsSink(service_account_json)

    sink.ensure_day(event_src.now_dt().date())
    print(f"[info] connected to sheet; tab='{sink.ws_title}', existing rows tracked: {sink.existing_rows_tracked}")

    if USE_TRACE_FILE and hasattr(event_src, "sim_start"):
        print(f"[info] TRACE MODE ON. sim_start={event_src.sim_start}")
    else:
        print(f"[info] watching {LOG_DIR}")

    key_src.start()
    try:
        run_monitor_loop(event_src=event_src, key_src=key_src, sink=sink, show_popup=show_popup)
    finally:
        print("[info] final sheets flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()