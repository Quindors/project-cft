# monitor/supa_mirror.py
"""Best-effort one-way mirror of local changes up to Supabase.

Local SQLite stays the source of truth. When MIRROR_SUPABASE is enabled and
Supabase credentials exist, dashboard edits (corrections, category changes)
and session intentions are pushed to Supabase in a background thread so the
cloud copy stays current. Event rows themselves are mirrored by DualSink
(monitor/dual_sink.py). Intentions exist upstream so the hosted dashboard can
draw the forest and the session history while this exe is off.

Every call here is fire-and-forget: failures are logged and never block or
break the local write that already succeeded.

Corrections are matched in Supabase by (timestamp, current_window) — both
databases receive identical values for those columns via DualSink, so the
pair acts as a natural key even though row ids differ.
"""
from __future__ import annotations

import os
import threading
from typing import Optional


def enabled() -> bool:
    flag = os.getenv("MIRROR_SUPABASE", "").strip().lower() in ("1", "true", "yes")
    return flag and bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from supabase import create_client
            _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        return _client


def _run_async(label: str, fn) -> None:
    def worker():
        import time
        for attempt in (1, 2):
            try:
                fn(_get_client())
                print(f"[mirror] {label}: ok", flush=True)
                return
            except Exception as e:
                if attempt == 1:
                    time.sleep(2)
                else:
                    print(f"[mirror] {label} failed (local write unaffected): {e}", flush=True)

    threading.Thread(target=worker, daemon=True, name=f"mirror-{label}").start()


def mirror_correction(timestamp: str, current_window: str, human_label: Optional[str]) -> None:
    """Set (or clear) human_label on the matching Supabase row."""
    if not enabled():
        return

    def op(sb):
        (sb.table("focus_logs")
           .update({"human_label": human_label})
           .eq("timestamp", timestamp)
           .eq("current_window", current_window)
           .execute())

    _run_async("correction", op)


def mirror_correction_rows(changes) -> None:
    """Apply a rule edit's per-row outcomes to the matching Supabase rows.

    `changes` is [(timestamp, current_window, human_label-or-None), ...] —
    the exact rows the local UPDATE touched, so the cloud never re-derives
    which rows a rule covers (normalized-title matching, the confirmation
    guard, the works) and cannot drift from the local decision. Rows are
    matched by the (timestamp, current_window) natural key, see the module
    docstring.
    """
    if not enabled() or not changes:
        return

    def op(sb):
        for timestamp, current_window, human_label in changes:
            (sb.table("focus_logs")
               .update({"human_label": human_label})
               .eq("timestamp", timestamp)
               .eq("current_window", current_window)
               .execute())

    _run_async("correction-rule", op)


def mirror_intent_rows(changes) -> None:
    """Apply intent rulings from the Review tab to the matching Supabase rows.

    `changes` is [(timestamp, current_window, human_align-or-None), ...], the
    rows the local UPDATE touched, matched by the same natural key as
    corrections. Needs migrations/2026-10-11_intent_review.sql; before it is
    applied this fails, harmlessly, in the background."""
    if not enabled() or not changes:
        return

    def op(sb):
        for timestamp, current_window, human_align in changes:
            (sb.table("focus_logs")
               .update({"human_align": human_align})
               .eq("timestamp", timestamp)
               .eq("current_window", current_window)
               .execute())

    _run_async("intent-review", op)


_INTENTION_COLS = ("id", "text", "started_at", "ends_at", "ended_at", "status",
                   "llm_category", "project_id")


def _intention_payload(row) -> dict:
    return {k: row.get(k) for k in _INTENTION_COLS}


def mirror_intention(row) -> None:
    """Upsert one session intention (any status) by its local id."""
    if not enabled():
        return
    payload = _intention_payload(row)

    def op(sb):
        sb.table("intentions").upsert(payload, on_conflict="id").execute()

    _run_async(f"intention-{payload['id']}", op)


def mirror_intentions(rows) -> None:
    """Bulk upsert — the startup backfill. Idempotent, so running it on every
    start costs one request per 500 rows and never duplicates anything."""
    if not enabled() or not rows:
        return
    payloads = [_intention_payload(r) for r in rows]

    def op(sb):
        for i in range(0, len(payloads), 500):
            sb.table("intentions").upsert(payloads[i:i + 500], on_conflict="id").execute()

    _run_async(f"intentions-backfill({len(payloads)})", op)


_PROJECT_COLS = ("id", "name", "description", "category", "status", "created_at", "archived_at")


def _project_payload(row) -> dict:
    return {k: row.get(k) for k in _PROJECT_COLS}


def mirror_project(row) -> None:
    """Upsert one project by its local id."""
    if not enabled():
        return
    payload = _project_payload(row)

    def op(sb):
        sb.table("projects").upsert(payload, on_conflict="id").execute()

    _run_async(f"project-{payload['id']}", op)


def mirror_projects(rows) -> None:
    """Bulk upsert — the startup backfill, idempotent like the intentions'."""
    if not enabled() or not rows:
        return
    payloads = [_project_payload(r) for r in rows]

    def op(sb):
        for i in range(0, len(payloads), 500):
            sb.table("projects").upsert(payloads[i:i + 500], on_conflict="id").execute()

    _run_async(f"projects-backfill({len(payloads)})", op)


def mirror_project_delete(project_id: int) -> None:
    """A deleted project is gone from the cloud too: unlike categories, a
    project row is a label, not history, so there is nothing to reassign."""
    if not enabled():
        return

    def op(sb):
        sb.table("projects").delete().eq("id", project_id).execute()

    _run_async(f"project-delete-{project_id}", op)


def mirror_category_add(name: str, description: str, is_productive) -> None:
    if not enabled():
        return

    def op(sb):
        (sb.table("categories")
           .upsert({"name": name, "description": description, "is_productive": is_productive},
                   on_conflict="name")
           .execute())

    _run_async("category-add", op)


def mirror_category_delete(name: str, reassign_to: str) -> None:
    """Mirror a category delete; logged rows move to reassign_to like locally."""
    if not enabled():
        return

    def op(sb):
        # Reassign first: deleting the category before its rows move would
        # leave orphaned category_name values upstream if the update failed.
        sb.table("focus_logs").update({"category_name": reassign_to}).eq("category_name", name).execute()
        sb.table("focus_logs").update({"human_label": reassign_to}).eq("human_label", name).execute()
        sb.table("categories").delete().eq("name", name).execute()

    _run_async("category-delete", op)


def mirror_category_update(old_name: str, new_name: str, description: str, is_productive) -> None:
    """Mirror a category edit; renames cascade to focus_logs like locally."""
    if not enabled():
        return

    def op(sb):
        res = (sb.table("categories")
                 .update({"name": new_name, "description": description,
                          "is_productive": is_productive})
                 .eq("name", old_name)
                 .execute())
        if not res.data:
            # Category didn't exist upstream (e.g. created before mirroring
            # was enabled) — create it instead.
            (sb.table("categories")
               .upsert({"name": new_name, "description": description,
                        "is_productive": is_productive}, on_conflict="name")
               .execute())
        if new_name != old_name:
            sb.table("focus_logs").update({"category_name": new_name}).eq("category_name", old_name).execute()
            sb.table("focus_logs").update({"human_label": new_name}).eq("human_label", old_name).execute()

    _run_async("category-update", op)
