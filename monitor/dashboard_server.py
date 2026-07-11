# monitor/dashboard_server.py
"""Serve the bundled focus-dashboard build + a small JSON API over localhost.

The dashboard's production build is staged into a `dashboard/` folder at build
time (see build.ps1) and bundled by PyInstaller. When present, we serve it on a
free 127.0.0.1 port in a background thread so the tray's "Open Dashboard" can
open it offline.

For local (SQLite) installs, the bundled dashboard is built in "local" mode and
reads its data from this server's /api endpoints instead of Supabase:

    GET /api/categories        -> [{name, is_productive}, ...]
    GET /api/focus?since=<ts>   -> [{timestamp, category_name, confidence}, ...]

`since` is a naive local timestamp string ("YYYY-MM-DD HH:MM:SS") matching how
the monitor stores focus_logs.timestamp, so a plain string comparison is correct.
"""
from __future__ import annotations

import functools
import http.server
import json
import socket
import socketserver
import sqlite3
import threading
from typing import Optional
from urllib.parse import urlparse, parse_qs

from monitor.paths import resource_path, data_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _db_ro():
    """Open the local focus DB read-only (returns None if it doesn't exist yet)."""
    path = data_dir() / "focus.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request stderr spam
        return

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._handle_api()
            return
        super().do_GET()

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_api(self) -> None:
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        conn = _db_ro()
        if conn is None:
            # No data yet — return empty so the dashboard renders zeros, not an error.
            self._send_json([] if u.path != "/api/categories" else [])
            return
        try:
            if u.path == "/api/categories":
                rows = conn.execute(
                    "SELECT name, is_productive FROM categories"
                ).fetchall()
                data = [
                    {"name": r["name"],
                     "is_productive": None if r["is_productive"] is None else bool(r["is_productive"])}
                    for r in rows
                ]
                self._send_json(data)
            elif u.path == "/api/focus":
                since = (qs.get("since", [""])[0] or "").strip()
                if since:
                    rows = conn.execute(
                        "SELECT timestamp, category_name, confidence FROM focus_logs "
                        "WHERE timestamp >= ? ORDER BY timestamp DESC",
                        (since,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT timestamp, category_name, confidence FROM focus_logs "
                        "ORDER BY timestamp DESC LIMIT 5000"
                    ).fetchall()
                self._send_json([dict(r) for r in rows])
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
        finally:
            conn.close()


def serve_bundled_dashboard() -> Optional[str]:
    """Start serving the bundled dashboard + API. Returns its URL, or None if absent."""
    root = resource_path("dashboard")
    if not (root / "index.html").exists():
        return None

    port = _free_port()
    handler = functools.partial(_Handler, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http").start()
    return f"http://127.0.0.1:{port}/"
