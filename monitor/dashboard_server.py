# monitor/dashboard_server.py
"""Local JSON API for the focus dashboard (+ the bundled build, if present).

The dashboard reads its data from this server's /api endpoints, backed by the
local SQLite DB. Two ways to reach it:

  1. Hosted (Vercel): the page is served from https://<app>.vercel.app and
     fetches this API cross-origin at http://127.0.0.1:<CFT_API_PORT>. That
     needs a FIXED port (so the hosted page knows where to look) and CORS.
  2. Bundled: if a `dashboard/` build was staged into the exe, it's served from
     this same origin (no CORS needed).

Endpoints:
    GET  /api/categories        -> [{name, description, is_productive}, ...]
    GET  /api/focus?since=<ts>  -> [{timestamp, category_name, confidence}, ...]
    GET  /api/events?limit=<n>  -> recent rows with id/current_window/human_label,
                                   for the dashboard's review (re-classify) tab
    POST /api/correct           -> {id, human_label} sets a human correction on a
                                   row (human_label of null/"" clears it)
    POST /api/category          -> {name, description, is_productive} adds a category
    POST /api/category/update   -> {name, new_name?, description?, is_productive?}
                                   edits a category; a rename cascades to
                                   focus_logs.category_name / human_label

Corrections and category changes reach the classifier at the next monitor
start, when they are re-read from the DB.

CORS is deliberately an ALLOWLIST, never "*": this API serves your window titles
off localhost, so a wildcard would let any site you visit read them. Allowed
origins come from DASHBOARD_URL plus CFT_ALLOWED_ORIGINS (comma-separated).

`since` is a naive local timestamp string ("YYYY-MM-DD HH:MM:SS") matching how
the monitor stores focus_logs.timestamp, so a plain string comparison is correct.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sqlite3
import threading
from datetime import datetime
from typing import Optional, Set
from urllib.parse import urlparse, parse_qs

from monitor.paths import resource_path, data_dir


DEFAULT_API_PORT = 47113

# Set by start_local_server(): the bundled dashboard dir, or None if not bundled.
_STATIC_ROOT: Optional[str] = None


def _api_port() -> int:
    try:
        return int(os.getenv("CFT_API_PORT", "") or DEFAULT_API_PORT)
    except ValueError:
        return DEFAULT_API_PORT


def _origin_of(url: str) -> Optional[str]:
    try:
        p = urlparse(url.strip())
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
    except Exception:
        pass
    return None


def _allowed_origins() -> Set[str]:
    """Origins permitted to call this API cross-origin (e.g. the Vercel site)."""
    origins: Set[str] = set()
    dash = _origin_of(os.getenv("DASHBOARD_URL", ""))
    if dash:
        origins.add(dash)
    for raw in (os.getenv("CFT_ALLOWED_ORIGINS", "") or "").split(","):
        o = _origin_of(raw)
        if o:
            origins.add(o)
    return origins


def _db_ro():
    """Open the local focus DB read-only (returns None if it doesn't exist yet)."""
    path = data_dir() / "focus.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _db_rw():
    """Open the local focus DB read-write (returns None if it doesn't exist yet)."""
    path = data_dir() / "focus.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request stderr spam
        return

    # ---------- CORS ----------

    def _send_cors(self) -> None:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if origin and origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self) -> None:
        """Preflight for the hosted dashboard's cross-origin calls."""
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if not origin or origin not in _allowed_origins():
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        # Chrome Private Network Access: a public HTTPS page calling localhost
        # must get explicit permission here, or the request is blocked.
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    # ---------- routing ----------

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._handle_api()
            return
        if _STATIC_ROOT is None:
            # API-only mode (dashboard is hosted): don't serve arbitrary files.
            self._send_json({"error": "dashboard not bundled; use the hosted URL"}, status=404)
            return
        super().do_GET()

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _handle_api(self) -> None:
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        conn = _db_ro()
        if conn is None:
            # No data yet — return empty so the dashboard renders zeros, not an error.
            self._send_json([])
            return
        try:
            if u.path == "/api/categories":
                rows = conn.execute(
                    "SELECT name, description, is_productive FROM categories"
                ).fetchall()
                data = [
                    {"name": r["name"],
                     "description": r["description"],
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
            elif u.path == "/api/events":
                try:
                    limit = min(max(int(qs.get("limit", ["100"])[0]), 1), 1000)
                except ValueError:
                    limit = 100
                rows = conn.execute(
                    "SELECT id, timestamp, current_window, category_name, confidence, "
                    "human_label, reason FROM focus_logs "
                    "WHERE current_window IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                self._send_json([dict(r) for r in rows])
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
        finally:
            conn.close()

    def do_POST(self) -> None:
        u = urlparse(self.path)
        handlers = {
            "/api/correct": self._post_correct,
            "/api/category": self._post_category_add,
            "/api/category/update": self._post_category_update,
        }
        handler = handlers.get(u.path)
        if handler is None:
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "expected a JSON object body"}, status=400)
            return

        conn = _db_rw()
        if conn is None:
            self._send_json({"error": "no local database yet"}, status=503)
            return
        try:
            handler(conn, body)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
        finally:
            conn.close()

    def _post_correct(self, conn, body) -> None:
        try:
            row_id = int(body["id"])
        except (KeyError, ValueError, TypeError):
            self._send_json({"error": "expected JSON body {id, human_label}"}, status=400)
            return

        label = (body.get("human_label") or "").strip() or None
        reviewed_at = datetime.now().isoformat() if label else None
        cur = conn.execute(
            "UPDATE focus_logs SET human_label = ?, reviewed_at = ? WHERE id = ?",
            (label, reviewed_at, row_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            self._send_json({"error": f"no row with id {row_id}"}, status=404)
        else:
            self._send_json({"ok": True, "id": row_id, "human_label": label})

    @staticmethod
    def _productive_flag(value):
        """JSON true/false/null -> SQLite 1/0/NULL (neutral)."""
        return None if value is None else int(bool(value))

    def _post_category_add(self, conn, body) -> None:
        name = (body.get("name") or "").strip()
        if not name or len(name) > 60:
            self._send_json({"error": "category name must be 1-60 characters"}, status=400)
            return
        description = (body.get("description") or "").strip()
        exists = conn.execute(
            "SELECT 1 FROM categories WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if exists:
            self._send_json({"error": f"category {name!r} already exists"}, status=409)
            return
        conn.execute(
            "INSERT INTO categories (name, description, is_productive) VALUES (?,?,?)",
            (name, description, self._productive_flag(body.get("is_productive"))),
        )
        conn.commit()
        self._send_json({"ok": True, "name": name})

    def _post_category_update(self, conn, body) -> None:
        name = (body.get("name") or "").strip()
        row = conn.execute(
            "SELECT name, description, is_productive FROM categories WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            self._send_json({"error": f"no category named {name!r}"}, status=404)
            return

        new_name = (body.get("new_name") or "").strip() or name
        if len(new_name) > 60:
            self._send_json({"error": "category name must be 1-60 characters"}, status=400)
            return
        if new_name.lower() != name.lower():
            clash = conn.execute(
                "SELECT 1 FROM categories WHERE name = ? COLLATE NOCASE", (new_name,)
            ).fetchone()
            if clash:
                self._send_json({"error": f"category {new_name!r} already exists"}, status=409)
                return

        description = body["description"].strip() if "description" in body else row["description"]
        productive = (
            self._productive_flag(body["is_productive"])
            if "is_productive" in body
            else row["is_productive"]
        )

        conn.execute(
            "UPDATE categories SET name = ?, description = ?, is_productive = ? WHERE name = ?",
            (new_name, description, productive, name),
        )
        renamed_logs = 0
        if new_name != name:
            # Keep history consistent: move logged events to the new name too.
            renamed_logs += conn.execute(
                "UPDATE focus_logs SET category_name = ? WHERE category_name = ?",
                (new_name, name),
            ).rowcount
            renamed_logs += conn.execute(
                "UPDATE focus_logs SET human_label = ? WHERE human_label = ?",
                (new_name, name),
            ).rowcount
        conn.commit()
        self._send_json({"ok": True, "name": new_name, "renamed_log_rows": renamed_logs})


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_local_server() -> Optional[str]:
    """Start the local API (and the bundled dashboard, if staged).

    Binds a FIXED port so a hosted dashboard can find it. Returns the local
    base URL, or None if the port couldn't be bound.
    """
    global _STATIC_ROOT
    root = resource_path("dashboard")
    _STATIC_ROOT = str(root) if (root / "index.html").exists() else None

    port = _api_port()
    handler = functools.partial(_Handler, directory=_STATIC_ROOT or str(data_dir()))
    try:
        httpd = _Server(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"[api] could not bind 127.0.0.1:{port} ({e}). "
              f"Is another cft already running? The hosted dashboard needs this port.")
        return None

    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http").start()
    allowed = _allowed_origins()
    print(f"[api] local API on http://127.0.0.1:{port} "
          f"(static={'bundled' if _STATIC_ROOT else 'hosted'}, allowed origins={sorted(allowed) or 'none'})")
    return f"http://127.0.0.1:{port}/"


# Back-compat alias for older callers.
serve_bundled_dashboard = start_local_server
