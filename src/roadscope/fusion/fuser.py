"""Fuses per-frame detections into deduplicated defect records.

Pipeline::

    detections (every frame)  ->  tracker  ->  confirmed tracks  ->  geofence  ->  DefectRecord

A record is emitted once per physical defect, carrying the evidence that makes
it defensible: how many frames it persisted, for how long, and the best frame
available.

The best frame is kept as a thumbnail. A geoportal pin with no photograph
attached is an assertion; a pin with a photograph is evidence.
"""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..transport.topics import DetectionPayload
from .geofence import Geofence
from .tracker import ByteTracker, Track

log = get_logger(__name__)


@dataclass
class DefectRecord:
    """One deduplicated defect, ready for transmission."""

    detection_id: str
    class_code: str
    confidence: float
    lat: float
    lon: float
    gps_accuracy_m: float | None = None
    frames_seen: int = 0
    track_duration_s: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    frame_id: str | None = None
    thumb_path: str | None = None
    model_version: str | None = None
    severity: str | None = None
    length_m: float | None = None
    width_mm: float | None = None
    area_m2: float | None = None
    image_jpeg: bytes | None = field(default=None, repr=False)

    def to_payload(self, device_id: str, trip_id: str) -> dict[str, Any]:
        return DetectionPayload(
            detection_id=self.detection_id,
            device_id=device_id,
            trip_id=trip_id,
            frame_id=self.frame_id,
            ts_utc=time.time(),
            lat=self.lat,
            lon=self.lon,
            class_code=self.class_code,
            confidence=round(self.confidence, 3),
            gps_accuracy_m=self.gps_accuracy_m,
            severity=self.severity,
            length_m=self.length_m,
            width_mm=self.width_mm,
            area_m2=self.area_m2,
            frames_seen=self.frames_seen,
            track_duration_s=round(self.track_duration_s, 2),
            bbox=list(self.bbox) if self.bbox else None,
            model_version=self.model_version,
            image_jpeg_b64=base64.b64encode(self.image_jpeg).decode("ascii")
            if self.image_jpeg
            else None,
        ).to_dict()


class DefectFuser:
    """Turns tracked detections into deduplicated, evidenced records."""

    def __init__(
        self,
        *,
        geofence_radius_m: float = 15.0,
        iou_threshold: float = 0.30,
        min_hits: int = 3,
        max_age_s: float = 2.0,
        thumb_dir: Path | None = None,
        thumb_width: int = 640,
        save_thumbnails: bool = True,
    ) -> None:
        self.tracker = ByteTracker(
            iou_threshold=iou_threshold, min_hits=min_hits, max_age_s=max_age_s
        )
        self.geofence = Geofence(radius_m=geofence_radius_m)
        self.thumb_dir = Path(thumb_dir) if thumb_dir else None
        self.thumb_width = thumb_width
        self.save_thumbnails = save_thumbnails
        self.suppressed = 0
        self.emitted = 0

    def reset(self) -> None:
        """Clear state between trips."""
        self.tracker.reset()
        self.geofence.clear()
        self.suppressed = 0
        self.emitted = 0

    # -- thumbnails -------------------------------------------------------
    def _save_thumbnail(
        self, frame: np.ndarray, track: Track, frame_id: str | None = None
    ) -> tuple[str | None, bytes | None]:
        """Persist the frame for this track, with the box drawn on.

        Returns ``(file name, JPEG bytes)``. The bytes travel to the server in
        the detection message, so the evidence image reaches the system of
        record rather than staying on the edge device.
        """
        if not self.save_thumbnails or self.thumb_dir is None or frame is None:
            return None, None
        try:
            self.thumb_dir.mkdir(parents=True, exist_ok=True)
            frame = frame.copy()  # never draw on the caller's frame
            height, width = frame.shape[:2]
            scale = 1.0
            if width > self.thumb_width:
                scale = self.thumb_width / width
                frame = cv2.resize(frame, (self.thumb_width, int(height * scale)))
            if track.best_bbox is not None:
                # The box is in full-resolution pixels; scale it with the frame,
                # or on a 1920 px phone feed it lands outside a 480 px thumbnail.
                x1, y1, x2, y2 = (round(v * scale) for v in track.best_bbox)
                color = (239, 68, 68)  # BGR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                text = f"{track.class_code} {int(track.best_confidence * 100)}%"
                (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
                cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            # Named by frame id: track ids restart at 1 every run, so naming by
            # track would overwrite the previous run's evidence.
            name = f"{frame_id or f'track-{track.track_id:06d}'}.jpg"
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None, None
            data = buf.tobytes()
            (self.thumb_dir / name).write_bytes(data)
            return name, data
        except Exception as exc:
            log.warning("thumbnail failed for track %s: %s", track.track_id, exc)
            return None, None

    # -- main -------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray | None,
        detections: list,
        ts: float,
        frame_index: int = -1,
        position: tuple[float, float, float | None] | None = None,
        model_version: str | None = None,
    ) -> list[DefectRecord]:
        """Feed one frame. Returns records newly emitted, i.e. new defects."""
        confirmed = self.tracker.update(detections, ts, frame_index=frame_index, position=position)
        records: list[DefectRecord] = []
        for track in confirmed:
            if track.emitted or track.lat is None or track.lon is None:
                continue
            # Suppress a repeat of an already-emitted physical defect.
            if self.geofence.is_suppressed(track.lat, track.lon, ts):
                self.suppressed += 1
                log.info(
                    "suppressed duplicate %s at %.5f,%.5f (within %.0fm of a known defect)",
                    track.class_code,
                    track.lat,
                    track.lon,
                    self.geofence.radius_m,
                )
                track.emitted = True
                continue
            frame_id = f"frm-{uuid.uuid4().hex[:10]}"
            thumb, jpeg = (
                self._save_thumbnail(frame, track, frame_id) if frame is not None else (None, None)
            )
            record = DefectRecord(
                detection_id=f"det-{uuid.uuid4().hex[:12]}",
                class_code=track.class_code,
                confidence=round(track.best_confidence, 3),
                lat=track.lat,
                lon=track.lon,
                gps_accuracy_m=track.gps_accuracy_m,
                frames_seen=track.hits,
                track_duration_s=track.duration_s,
                bbox=track.best_bbox,
                frame_id=frame_id if thumb else None,
                thumb_path=thumb,
                image_jpeg=jpeg,
                model_version=model_version,
            )
            track.emitted = True
            self.geofence.add(track.lat, track.lon, ts)
            self.emitted += 1
            records.append(record)
            log.info(
                "%s %.0f%% @ %.5f,%.5f  (seen in %d frames over %.2fs)",
                record.class_code,
                record.confidence * 100,
                record.lat,
                record.lon,
                record.frames_seen,
                record.track_duration_s,
            )
        return records

    def stats(self) -> dict[str, int]:
        return {
            "emitted": self.emitted,
            "suppressed": self.suppressed,
            "active_tracks": len(self.tracker.active),
            "geofence_sites": len(self.geofence),
        }
