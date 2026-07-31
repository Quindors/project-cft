# monitor/supa_mirror.py
"""Best-effort one-way mirror of local changes up to Supabase.

Local SQLite stays the source of truth. When MIRROR_SUPABASE is enabled and
Supabase credentials exist, dashboard edits (corrections, category changes)
are pushed to Supabase in a background thread so the cloud copy stays current.
Event rows themselves are mirrored by DualSink (monitor/dual_sink.py).

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
