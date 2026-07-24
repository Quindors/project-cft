# monitor/app.py
from __future__ import annotations

import os
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


def _choose_storage() -> str:
    """'supabase' or 'local'. Explicit STORAGE wins; MIRROR_SUPABASE keeps local
    primary (with cloud mirroring); else Supabase iff configured."""
    storage = os.getenv("STORAGE", "").strip().lower()
    if storage in ("supabase", "local"):
        return storage
    if os.getenv("MIRROR_SUPABASE", "").strip().lower() in ("1", "true", "yes"):
        return "local"
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        return "supabase"
    return "local"


def _make_live_sink(settings: Settings):
    if _choose_storage() == "supabase":
        return SupabaseSink(settings=settings)
    from monitor.local_sink import LocalSink
    local = LocalSink(settings=settings)

    from monitor import supa_mirror
    if supa_mirror.enabled():
        try:
            from monitor.dual_sink import DualSink
            print("[boot] Supabase mirroring ON (local is primary)")
            return DualSink(local, SupabaseSink(settings=settings))
        except Exception as e:
            print(f"[boot] Supabase mirror unavailable ({e}); running local-only")
    return local


def _storage_init(box: dict, settings: Settings, *, want_writes: bool) -> None:
    """Build the sink and preload categories/corrections/priors."""
    t = time.time()
    try:
        if want_writes:
            sink = _make_live_sink(settings)
            box["backend"] = _choose_storage()
        else:
            # trace/testing: no writes; read categories from Supabase if configured
            try:
                sink = NullSink(read_sink=SupabaseSink(settings=settings))
            except Exception:
                sink = NullSink()
            box["backend"] = "null"
        box["sink"] = sink
        box["categories"] = sink.get_categories()
        box["corrections"] = sink.load_corrections()
        box["prior_classifications"] = sink.load_prior_classifications()
        box["elapsed"] = time.time() - t
    except Exception as e:
        box["error"] = e
        box["elapsed"] = time.time() - t


def _start_tray(settings: Settings, shutdown: threading.Event):
    """Start the local API, then show a tray icon pointing at the dashboard.

    The API always starts (on a fixed port) because a hosted dashboard reads
    from it. DASHBOARD_URL (e.g. the Vercel site) wins when set; otherwise we
    fall back to the bundled copy served locally.
    """
    local_url = None
    try:
        from monitor.dashboard_server import start_local_server
        local_url = start_local_server()
    except Exception as e:
        print(f"[boot] local API not started ({e})")

    dashboard_url = os.getenv("DASHBOARD_URL", "").strip() or local_url

    # Auto-open the dashboard on startup (opt out with CFT_OPEN_DASHBOARD=0).
    if dashboard_url and os.getenv("CFT_OPEN_DASHBOARD", "1").strip().lower() not in ("0", "false", "no"):
        try:
            import webbrowser
            webbrowser.open(dashboard_url)
            print(f"[boot] opened dashboard in browser")
        except Exception as e:
            print(f"[boot] could not auto-open dashboard ({e})")

    try:
        from monitor.tray import start_tray
        icon = start_tray(dashboard_url, on_quit=shutdown.set)
        print(f"[boot] tray ready (dashboard: {dashboard_url or 'not configured'})")
        return icon
    except Exception as e:
        print(f"[boot] tray not started ({e})")
        return None


def run_dashboard_only() -> None:
    """Serve the dashboard + correction API WITHOUT monitoring.

    No window capture, no keystrokes, no LLM — just the local API over the
    existing focus.db so the user can browse stats and make corrections.
    Launch with: cft.exe --dashboard
    """
    print(f"\n{'='*50}\n  project-cft dashboard (monitoring OFF)\n{'='*50}\n")
    load_env(override=True)

    shutdown = threading.Event()
    icon = _start_tray(None, shutdown)
    print("[boot] dashboard-only mode ready — quit from the tray icon")
    try:
        while not shutdown.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass


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
    supabase_thread = threading.Thread(target=_storage_init, args=(supabase_box, settings),
                                       kwargs={"want_writes": want_writes}, name="storage-init", daemon=True)
    llm_thread.start()
    supabase_thread.start()
    print("[boot] llm + storage init dispatched in background...")

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
    print(f"[boot] storage ({supabase_box.get('backend','?')}) + reads     {supabase_box['elapsed']:.2f}s  "
          f"(cats={len(supabase_box['categories'])} corrections={len(supabase_box['corrections'])} priors={len(supabase_box['prior_classifications'])})")

    # --- Tray + local dashboard (live mode only) ---
    shutdown = threading.Event()
    tray_icon = None
    if not testing_mode:
        tray_icon = _start_tray(settings, shutdown)

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
            stop_event=shutdown,
        )
    finally:
        print("[info] final flush...")
        sink.close()
        print("[info] stopping key capture...")
        key_src.stop()
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
