"""Edge pipeline: capture -> infer -> geotag -> queue for transmission.

Three threads, deliberately decoupled:

* **ingest** reads frames, never blocks, drops rather than waits.
* **infer** runs the detector and stamps each detection with interpolated GPS.
* **publish** drains the outbox.

The queue between ingest and infer is bounded and drop-oldest. A slow detector
must degrade frame rate, never stall the camera -- otherwise one expensive frame
back-pressures into the capture thread and the whole survey is ruined.

This module must not import Streamlit. The UI is a read-only client of the
database; if the dashboard is not running, capture must continue.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import threading
import time
from typing import Any

import numpy as np

from ..config import Settings
from ..geo.kalman import LatLonKalman
from ..geo.quality import GPSFix, gate_fix
from ..logging_setup import get_logger
from ..transport.outbox import Outbox
from ..transport.topics import (
    TOPIC_DETECTION,
    TOPIC_GPS,
    TOPIC_STATUS,
    DetectionPayload,
    GPSPayload,
    StatusPayload,
    new_detection_id,
    topic_for,
    utc_now,
)
from ..vision.detector import DefectDetector
from .health import HealthStats
from .ingest import FrameSource, build_source

log = get_logger(__name__)

_STOP = object()


class EdgePipeline:
    """Orchestrates capture, inference and transmission."""

    def __init__(
        self,
        settings: Settings,
        *,
        source: FrameSource | None = None,
        detector: DefectDetector | None = None,
        publisher: Any = None,
        trip_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.source = source or build_source(settings.camera_url)
        self.detector = detector or DefectDetector.instance(settings.resolved_model_path())
        self.publisher = publisher
        self.outbox = Outbox(settings.outbox_db)

        self.trip_id = trip_id or f"trip-{settings.device_id}-{int(time.time())}"
        self.health = HealthStats()

        # Bounded, drop-oldest: stale frames are worthless.
        self._queue: queue.Queue = queue.Queue(maxsize=settings.queue_size)
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

        self._kalman = LatLonKalman()
        self._fixes: list[GPSFix] = []
        self._fix_lock = threading.Lock()
        self._seq = itertools.count(1)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        log.info("starting pipeline: trip=%s source=%s", self.trip_id, type(self.source).__name__)
        for name, target in (
            ("infer", self._infer_loop),
            ("publish", self._publish_loop),
            ("status", self._status_loop),
        ):
            thread = threading.Thread(target=target, name=f"roadscope-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        ingest = threading.Thread(target=self._ingest_loop, name="roadscope-ingest", daemon=True)
        ingest.start()
        self._threads.append(ingest)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop cleanly and flush the outbox before exiting."""
        log.info("stopping pipeline")
        self._stopping.set()
        # A full queue just means the infer loop is behind; the stop event
        # already tells it to exit, so the sentinel is best-effort.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_STOP)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self.source.release()
        log.info("pipeline stopped: %s", self.health.snapshot())

    # -- ingest ------------------------------------------------------------
    def _ingest_loop(self) -> None:
        """Read frames and hand them off. Never blocks on inference."""
        try:
            for frame, t_mono in self.source.frames():
                if self._stopping.is_set():
                    break
                self.health.record_frame()
                self._offer(frame, t_mono)
        except Exception as exc:
            log.exception("ingest loop failed: %s", exc)

    def _offer(self, frame: np.ndarray, t_mono: float) -> None:
        """Drop-oldest enqueue."""
        try:
            self._queue.put_nowait((frame, t_mono))
        except queue.Full:
            # Drop the oldest: a stale frame is worth less than the newest one.
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
                self.health.record_drop()
            try:
                self._queue.put_nowait((frame, t_mono))
            except queue.Full:
                self.health.record_drop()
        self.health.set_queue_depth(self._queue.qsize())

    # -- gps ---------------------------------------------------------------
    def ingest_gps(self, fix: GPSFix) -> bool:
        """Accept, filter, and enqueue a GPS fix.

        Gating happens here so the same policy applies to fixes from any source.
        """
        accepted, flag = gate_fix(
            fix,
            max_accuracy_m=self.settings.gps_max_accuracy_m,
            min_sats=self.settings.gps_min_sats,
            max_hdop=self.settings.gps_max_hdop,
            max_age_s=self.settings.gps_max_age_s,
        )
        self.health.record_gps(accepted)
        if not accepted:
            log.debug("rejected fix (%s): %.5f,%.5f", flag, fix.latitude, fix.longitude)
            return False

        filtered = self._kalman.update(fix, t_mono=fix.t_mono)
        filtered = filtered.with_quality(flag)
        with self._fix_lock:
            self._fixes.append(filtered)
            if len(self._fixes) > 600:
                del self._fixes[:300]

        self._enqueue_gps(filtered)
        return True

    def _position_at(self, t_mono: float) -> tuple[float, float, float | None]:
        """Interpolate position for a frame timestamp."""
        with self._fix_lock:
            fixes = list(self._fixes)
        if not fixes:
            return (0.0, 0.0, None)
        if t_mono is None or len(fixes) == 1:
            last = fixes[-1]
            return (last.latitude, last.longitude, last.accuracy_m)
        for a, b in itertools.pairwise(fixes):
            if (a.t_mono or 0) <= t_mono <= (b.t_mono or 0):
                span = (b.t_mono or 0) - (a.t_mono or 0)
                ratio = 0.0 if span <= 1e-6 else (t_mono - (a.t_mono or 0)) / span
                lat = a.latitude + ratio * (b.latitude - a.latitude)
                lon = a.longitude + ratio * (b.longitude - a.longitude)
                return (lat, lon, b.accuracy_m)
        last = fixes[-1]
        return (last.latitude, last.longitude, last.accuracy_m)

    def _enqueue_gps(self, fix: GPSFix) -> None:
        payload = GPSPayload(
            device_id=self.settings.device_id,
            seq=next(self._seq),
            ts_utc=fix.ts_utc,
            t_mono=fix.t_mono,
            lat=round(fix.latitude, 7),
            lon=round(fix.longitude, 7),
            speed_kmh=fix.speed_kmh,
            heading=fix.heading,
            accuracy_m=fix.accuracy_m,
            hdop=fix.hdop,
            n_sats=fix.n_sats,
            quality_flag=str(fix.quality_flag),
            trip_id=self.trip_id,
        ).to_dict()
        self.outbox.enqueue(
            topic_for(self.settings.mqtt_topic_prefix, self.settings.device_id, TOPIC_GPS), payload
        )

    # -- inference ---------------------------------------------------------
    def _infer_loop(self) -> None:
        while not self._stopping.is_set():
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                break
            frame, t_mono = item
            try:
                self._process(frame, t_mono)
            except Exception as exc:
                log.exception("frame failed: %s", exc)
            finally:
                self._queue.task_done()
                self.health.set_queue_depth(self._queue.qsize())

    def _process(self, frame: np.ndarray, t_mono: float) -> None:
        start = time.perf_counter()
        detections = self.detector.infer(
            frame,
            conf=self.settings.conf_threshold,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.imgsz,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.health.record_processed(elapsed_ms)
        self.health.record_detection(len(detections))

        lat, lon, accuracy = self._position_at(t_mono)
        if (lat, lon) == (0.0, 0.0):
            log.debug("detections without a GPS fix yet: %d", len(detections))
            return

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            payload = DetectionPayload(
                detection_id=new_detection_id(),
                device_id=self.settings.device_id,
                trip_id=self.trip_id,
                frame_id=None,
                ts_utc=utc_now(),
                lat=lat,
                lon=lon,
                class_code=detection.class_code,
                confidence=round(detection.confidence, 3),
                gps_accuracy_m=accuracy,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                model_version=self.detector.engine_name,
            ).to_dict()
            self.outbox.enqueue(
                topic_for(
                    self.settings.mqtt_topic_prefix, self.settings.device_id, TOPIC_DETECTION
                ),
                payload,
            )

    # -- transmission ------------------------------------------------------
    def _publish_loop(self) -> None:
        """Drain the outbox. Retries every loop until delivered."""
        while not self._stopping.is_set():
            rows = self.outbox.pending(limit=50)
            if not rows:
                time.sleep(0.25)
                continue
            if self.publisher is None or not getattr(self.publisher, "connected", False):
                # Broker down. Rows stay pending; this is the link-loss path.
                time.sleep(0.5)
                continue
            for row in rows:
                ok = self.publisher.publish(row.topic, row.payload, qos=self.settings.mqtt_qos)
                if ok:
                    self.outbox.mark_sent(row.id)
                    self.health.record_publish(True)
                else:
                    self.outbox.increment_attempt(row.id, "publish refused")
                    self.health.record_publish(False)

    def _status_loop(self) -> None:
        while not self._stopping.is_set():
            payload = StatusPayload(
                device_id=self.settings.device_id,
                ts_utc=utc_now(),
                fps=round(self.health.fps, 2),
                drop_pct=round(self.health.drop_pct, 2),
                queue_depth=self.health.queue_depth,
                infer_ms=round(self.health.avg_infer_ms, 1),
                outbox_depth=self.outbox.depth(),
                trip_id=self.trip_id,
                model_version=self.detector.engine_name,
            ).to_dict()
            self.outbox.enqueue(
                topic_for(self.settings.mqtt_topic_prefix, self.settings.device_id, TOPIC_STATUS),
                payload,
            )
            self._stopping.wait(5.0)

    def flush(self) -> int:
        """Publish everything pending. Returns the count delivered."""
        if self.publisher is None or not getattr(self.publisher, "connected", False):
            return 0
        delivered = 0
        for row in self.outbox.pending(limit=10_000):
            if self.publisher.publish(row.topic, row.payload, qos=self.settings.mqtt_qos):
                self.outbox.mark_sent(row.id)
                delivered += 1
        return delivered
