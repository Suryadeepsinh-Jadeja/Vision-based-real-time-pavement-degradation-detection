"""Agent health metrics.

Frame-drop rate is published rather than hidden. If inference cannot keep up
with capture, the honest thing is to show it -- both for debugging and because
"the detector dropped 8% of frames under load" is a better answer in a viva
than a smooth-looking demo that is secretly discarding data.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class HealthStats:
    """Live counters. All reads/writes hold ``lock``."""

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    frames_in: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    detections: int = 0
    gps_in: int = 0
    gps_rejected: int = 0
    published: int = 0
    publish_failures: int = 0

    started_at: float = field(default_factory=time.monotonic)
    infer_ms_total: float = 0.0
    queue_depth: int = 0
    last_fix_ts: float | None = None

    def record_frame(self) -> None:
        with self.lock:
            self.frames_in += 1

    def record_processed(self, infer_ms: float) -> None:
        with self.lock:
            self.frames_processed += 1
            self.infer_ms_total += infer_ms

    def record_drop(self, n: int = 1) -> None:
        with self.lock:
            self.frames_dropped += n

    def record_detection(self, n: int = 1) -> None:
        with self.lock:
            self.detections += n

    def record_gps(self, accepted: bool) -> None:
        with self.lock:
            if accepted:
                self.gps_in += 1
                self.last_fix_ts = time.time()
            else:
                self.gps_rejected += 1

    def record_publish(self, ok: bool) -> None:
        with self.lock:
            if ok:
                self.published += 1
            else:
                self.publish_failures += 1

    def set_queue_depth(self, depth: int) -> None:
        with self.lock:
            self.queue_depth = depth

    # -- derived ----------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return max(1e-6, time.monotonic() - self.started_at)

    @property
    def drop_pct(self) -> float:
        with self.lock:
            total = self.frames_in
            dropped = self.frames_dropped
        return 0.0 if total == 0 else 100.0 * dropped / total

    @property
    def fps(self) -> float:
        with self.lock:
            processed = self.frames_processed
        return processed / self.elapsed_s

    @property
    def avg_infer_ms(self) -> float:
        with self.lock:
            if not self.frames_processed:
                return 0.0
            return self.infer_ms_total / self.frames_processed

    def snapshot(self) -> dict[str, float | int | None]:
        with self.lock:
            return {
                "frames_in": self.frames_in,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "detections": self.detections,
                "gps_in": self.gps_in,
                "gps_rejected": self.gps_rejected,
                "published": self.published,
                "publish_failures": self.publish_failures,
                "queue_depth": self.queue_depth,
                "elapsed_s": round(self.elapsed_s, 2),
            }
