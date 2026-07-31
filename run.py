"""Frozen-exe entry point.

When this exe is launched without a console (e.g. via VBS launcher, scheduled
task, double-click with stdout redirected), Python's stdout/stderr can end up
unattached and print() output silently dropped. Redirect both to a file next
to the exe BEFORE any other imports, so the boot logs are always recoverable.
"""
import sys, os

if getattr(sys, "frozen", False):
    log_dir = os.path.dirname(sys.executable)
    log_path = os.path.join(log_dir, "cft.log")
    _log = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _log
    sys.stderr = _log

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--install-autostart" in args or "--uninstall-autostart" in args:
        # Register/remove the logon+unlock scheduled task, then exit.
        from monitor.autostart import install_task, uninstall_task
        ok = install_task() if "--install-autostart" in args else uninstall_task()
        sys.exit(0 if ok else 1)

    if getattr(sys, "frozen", False):
        # The task retriggers on every unlock; the mutex keeps that (and manual
        # double-launches) from stacking a second monitor onto the same DB.
        from monitor.autostart import acquire_single_instance
        if not acquire_single_instance():
            print("[cft] another instance is already running; exiting.")
            sys.exit(0)

    from monitor.app import run, run_dashboard_only

    if "--dashboard" in args:
        # Dashboard + corrections only; no monitoring, no LLM, no API key needed.
        run_dashboard_only()
    else:
        run()
