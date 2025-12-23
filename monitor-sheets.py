#!/usr/bin/env python3
import os
import openai
from datetime import date

from monitor.config import LOG_DIR, USE_TRACE_FILE
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.sheets_sink import SheetsSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop

load_env(override=True)
SERVICE_ACCOUNT_JSON = build_service_account_json_from_b64()

def main():
    require_openai_api_key(openai)

    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise SystemExit(f"[fatal] Service account JSON not found: {SERVICE_ACCOUNT_JSON}")

    os.makedirs(LOG_DIR, exist_ok=True)

    sink = SheetsSink(SERVICE_ACCOUNT_JSON)
    event_src, key_src = make_sources(USE_TRACE_FILE)

    # initial tab selection
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] stopped by user")