# monitor/dashboard_server.py
"""Serve the bundled focus-dashboard build over localhost.

The dashboard's production build is staged into a `dashboard/` folder at build
time (see build.ps1) and bundled by PyInstaller. If present, we serve it on a
free 127.0.0.1 port in a background thread so the tray's "Open Dashboard" can
open it offline. If it isn't bundled, serve_bundled_dashboard() returns None
and the caller falls back to a hosted DASHBOARD_URL.
"""
from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import threading
from typing import Optional

from monitor.paths import resource_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request stderr spam
        return


def serve_bundled_dashboard() -> Optional[str]:
    """Start serving the bundled dashboard. Returns its URL, or None if absent."""
    root = resource_path("dashboard")
    index = root / "index.html"
    if not index.exists():
        return None

    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http").start()
    return f"http://127.0.0.1:{port}/"
