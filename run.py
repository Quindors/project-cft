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

from monitor.app import run, run_dashboard_only

if __name__ == "__main__":
    if "--dashboard" in sys.argv[1:]:
        # Dashboard + corrections only; no monitoring, no LLM, no API key needed.
        run_dashboard_only()
    else:
        run()
