# monitor/app.py
from __future__ import annotations

import os, time, openai

from monitor.config import Settings, DEFAULT_SETTINGS
from monitor.creds import load_env, build_service_account_json_from_b64, require_openai_api_key
from monitor.sheets_sink import SheetsSink, NullSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop

def _noop_popup(message: str, title: str = "Productivity Alert") -> None: return

def run(settings: Settings = DEFAULT_SETTINGS) -> None:
    boot_start = time.time()
    print(f"\n{'='*50}")
    print(f"  project-cft starting up...")
    print(f"{'='*50}\n")

    # --- 1. Environment ---
    t = time.time()
    load_env(override=True)
    print(f"[boot] load .env              {time.time() - t:.2f}s")

    # --- 2. OpenAI key ---
    t = time.time()
    require_openai_api_key(openai)
    print(f"[boot] openai api key         {time.time() - t:.2f}s")

    # --- 3. Event + keystroke sources ---
    t = time.time()
    event_src, key_src = make_sources(settings=settings)
    print(f"[boot] make_sources           {time.time() - t:.2f}s")

    testing_mode = settings.trace.use_trace_file
    popup_fn = _noop_popup if testing_mode else show_popup

    # --- 4. Sheets sink ---
    if settings.trace.use_trace_file and not settings.trace.upload_to_sheets:
        sink = NullSink()
        print("[boot] TESTING MODE: Sheets upload disabled.")
    else:
        t = time.time()
        service_account_json = build_service_account_json_from_b64()
        print(f"[boot] decode service account  {time.time() - t:.2f}s")

        if not os.path.exists(service_account_json):
            raise SystemExit(f"[fatal] Service account JSON not found: {service_account_json}")

        t = time.time()
        sink = SheetsSink(service_account_json, settings=settings)
        print(f"[boot] SheetsSink connect      {time.time() - t:.2f}s")

        t = time.time()
        sink.ensure_day(event_src.now_dt().date())
        print(f"[boot] ensure_day              {time.time() - t:.2f}s")
        print(f"[boot] sheet tab='{sink.ws_title}', rows tracked={sink.existing_rows_tracked}")
        
        t = time.time()
        sink.sync_dropdowns_to_all_daily_sheets()
        print(f"[boot] sync dropdowns          {time.time() - t:.2f}s")

    if settings.trace.use_trace_file and hasattr(event_src, "sim_start"):
        print(f"[boot] TRACE MODE ON. sim_start={event_src.sim_start}")

    # --- 5. Start keystroke capture ---
    t = time.time()
    key_src.start()
    print(f"[boot] key_src.start           {time.time() - t:.2f}s")

    total = time.time() - boot_start
    print(f"\n[boot] === INIT COMPLETE in {total:.2f}s ===\n")

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