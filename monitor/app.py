# monitor/app.py
from __future__ import annotations

import os, time, openai

from monitor.config import Settings, DEFAULT_SETTINGS, load_settings
from monitor.creds import load_env, configure_llm
from monitor.supabase_sink import SupabaseSink, NullSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop

def _noop_popup(message: str, title: str = "Productivity Alert") -> None: return

def run(settings: Settings | None = None) -> None:
    boot_start = time.time()
    print(f"\n{'='*50}")
    print(f"  project-cft starting up...")
    print(f"{'='*50}\n")

    # --- 1. Environment ---
    t = time.time()
    load_env(override=True)
    print(f"[boot] load .env              {time.time() - t:.2f}s")

    # Build settings AFTER .env so LLM_MODEL takes effect
    if settings is None:
        settings = load_settings()

    # --- 2. LLM provider ---
    t = time.time()
    provider = configure_llm(openai)
    print(f"[boot] llm provider={provider} model={settings.model} ({time.time() - t:.2f}s)")

    # --- 3. Event + keystroke sources ---
    t = time.time()
    event_src, key_src = make_sources(settings=settings)
    print(f"[boot] make_sources           {time.time() - t:.2f}s")

    testing_mode = settings.trace.use_trace_file
    popup_fn = _noop_popup if testing_mode else show_popup

    # --- 4. Supabase sink ---
    if settings.trace.use_trace_file and not settings.trace.upload_to_sheets:
        t = time.time()
        try:
            read_sink = SupabaseSink(settings=settings)
            sink = NullSink(read_sink=read_sink)
            print(f"[boot] TESTING MODE: writes disabled, reads from Supabase ({time.time() - t:.2f}s).")
        except Exception as e:
            sink = NullSink()
            print(f"[boot] TESTING MODE: writes disabled, Supabase unreachable ({e}) — using fallback categories.")
    else:
        t = time.time()
        sink = SupabaseSink(settings=settings)
        print(f"[boot] SupabaseSink connect    {time.time() - t:.2f}s")

        t = time.time()
        sink.ensure_day(event_src.now_dt().date())
        print(f"[boot] ensure_day              {time.time() - t:.2f}s")
        print(f"[boot] rows tracked today={sink.existing_rows_tracked}")

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
        print("[info] final flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()