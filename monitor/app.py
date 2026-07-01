# monitor/app.py
from __future__ import annotations

import threading
import time

from monitor.config import Settings, load_settings
from monitor.creds import load_env, configure_llm
from monitor.decider import set_llm_client
from monitor.supabase_sink import SupabaseSink, NullSink
from monitor.sources import make_sources
from monitor.alerts import show_popup
from monitor.runner import run_monitor_loop


def _llm_init(box: dict) -> None:
    t = time.time()
    try:
        client, provider = configure_llm()
        set_llm_client(client)
        box["provider"] = provider
        box["elapsed"] = time.time() - t
    except Exception as e:
        box["error"] = e
        box["elapsed"] = time.time() - t


def _supabase_init(box: dict, settings: Settings, *, want_writes: bool) -> None:
    """Build the sink (writes or no-writes) and preload categories/corrections/priors."""
    t = time.time()
    try:
        if want_writes:
            sink = SupabaseSink(settings=settings)
        else:
            try:
                read_sink = SupabaseSink(settings=settings)
                sink = NullSink(read_sink=read_sink)
            except Exception:
                sink = NullSink()
        box["sink"] = sink
        box["categories"] = sink.get_categories()
        box["corrections"] = sink.load_corrections()
        box["prior_classifications"] = sink.load_prior_classifications()
        box["elapsed"] = time.time() - t
    except Exception as e:
        box["error"] = e
        box["elapsed"] = time.time() - t


def run(settings: Settings | None = None) -> None:
    boot_start = time.time()
    print(f"\n{'='*50}\n  project-cft starting up...\n{'='*50}\n")

    t = time.time()
    load_env(override=True)
    print(f"[boot] load .env                    {time.time() - t:.2f}s")

    if settings is None:
        settings = load_settings()

    testing_mode = settings.trace.use_trace_file
    want_writes = not (settings.trace.use_trace_file and not settings.trace.upload_to_sheets)

    # --- Launch slow inits in parallel ---
    llm_box: dict = {}
    supabase_box: dict = {}
    llm_thread = threading.Thread(target=_llm_init, args=(llm_box,), name="llm-init", daemon=True)
    supabase_thread = threading.Thread(target=_supabase_init, args=(supabase_box, settings),
                                       kwargs={"want_writes": want_writes}, name="supabase-init", daemon=True)
    llm_thread.start()
    supabase_thread.start()
    print("[boot] llm + supabase init dispatched in background...")

    # --- Cheap main-thread work while those run ---
    t = time.time()
    event_src, key_src = make_sources(settings=settings)
    print(f"[boot] make_sources                 {time.time() - t:.2f}s")

    popup_fn = (lambda *a, **kw: None) if testing_mode else show_popup

    if settings.trace.use_trace_file and hasattr(event_src, "sim_start"):
        print(f"[boot] TRACE MODE ON. sim_start={event_src.sim_start}")

    t = time.time()
    key_src.start()
    print(f"[boot] key_src.start                {time.time() - t:.2f}s")

    # --- Join background inits ---
    llm_thread.join()
    if "error" in llm_box:
        raise llm_box["error"]
    print(f"[boot] llm provider                 {llm_box['elapsed']:.2f}s  (provider={llm_box['provider']} model={settings.model})")

    supabase_thread.join()
    if "error" in supabase_box:
        raise supabase_box["error"]
    sink = supabase_box["sink"]
    print(f"[boot] supabase + reads             {supabase_box['elapsed']:.2f}s  "
          f"(cats={len(supabase_box['categories'])} corrections={len(supabase_box['corrections'])} priors={len(supabase_box['prior_classifications'])})")

    print(f"\n[boot] === INIT COMPLETE in {time.time() - boot_start:.2f}s ===\n")

    try:
        run_monitor_loop(
            event_src=event_src,
            key_src=key_src,
            sink=sink,
            show_popup=popup_fn,
            settings=settings,
            categories=supabase_box["categories"],
            corrections=supabase_box["corrections"],
            prior_classifications=supabase_box["prior_classifications"],
        )
    finally:
        print("[info] final flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()
