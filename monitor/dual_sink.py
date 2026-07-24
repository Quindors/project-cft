# monitor/dual_sink.py
"""Local-primary sink that mirrors event rows to Supabase.

Reads (categories, corrections, priors) come from the local sink only — it is
the source of truth. Writes go to both: the local write is synchronous and
authoritative; mirrored rows queue up and ship to Supabase on a background
thread so cloud latency or outages never stall the monitor loop.

All sink methods are called from the runner's single loop thread; only the
background flusher touches the Supabase sink, one flush at a time.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List


class DualSink:
    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror
        self._pending: List[Dict[str, Any]] = []
        self._flushing = threading.Lock()

    # ------------------- writes -------------------

    def ensure_day(self, day) -> None:
        self.primary.ensure_day(day)

    def enqueue_record(self, record: Dict[str, Any]) -> None:
        self.primary.enqueue_record(record)
        self._pending.append(record)

    def flush_if_needed(self):
        result = self.primary.flush_if_needed()
        if self._pending and self._flushing.acquire(blocking=False):
            batch, self._pending = self._pending, []
            threading.Thread(
                target=self._mirror_flush, args=(batch,), daemon=True, name="mirror-flush"
            ).start()
        return result

    def _mirror_flush(self, batch: List[Dict[str, Any]]) -> None:
        # _flushing was acquired by flush_if_needed; release when done.
        try:
            for record in batch:
                self.mirror.enqueue_record(record)
            self.mirror.flush(force=True)
        except Exception as e:
            print(f"[mirror] flush failed, {len(batch)} rows not mirrored (local data intact): {e}")
        finally:
            self._flushing.release()

    def close(self) -> None:
        # Final mirror flush is synchronous but bounded by the sink's own retries.
        with self._flushing:
            try:
                for record in self._pending:
                    self.mirror.enqueue_record(record)
                self._pending = []
                self.mirror.close()
            except Exception as e:
                print(f"[mirror] close failed: {e}")
        self.primary.close()

    # ------------------- reads (local is the source of truth) -------------------

    def get_categories(self) -> List[Dict]:
        return self.primary.get_categories()

    def load_corrections(self, days_back: int = 7) -> List[Dict]:
        return self.primary.load_corrections(days_back)

    def load_prior_classifications(self, days_back: int = 7, min_confidence: float = 0.85) -> List[Dict]:
        return self.primary.load_prior_classifications(days_back, min_confidence)
