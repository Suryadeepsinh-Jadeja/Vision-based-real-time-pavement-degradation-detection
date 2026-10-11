"""Multi-object tracking for pavement distress.

Why this exists
---------------
A detection is not a defect. On CPU the agent runs inference at ~7.9 fps
(measured: 127 ms/frame), so a single pothole passed at 20 km/h produces a
detection roughly every 125 ms. Counting detections would inflate one pothole
into many records, and every downstream number -- PCI, work-order volume,
repair cost -- inherits that error.

This is a ByteTrack-style two-stage association:

1. **High-confidence detections** (>= ``high_thresh``) are matched to existing
   tracks first. These carry enough signal to establish identity.
2. **Low-confidence detections** (>= ``low_thresh``) are then matched to tracks
   left unmatched in stage 1. They keep a track alive through occlusion and
   motion blur but cannot create one -- that is what stops noise becoming
   phantom defects.

Matching requires the same class code. A longitudinal crack must never absorb a
transverse crack's identity, or severity grading inherits the wrong geometry.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..logging_setup import get_logger

log = get_logger(__name__)

BBox = tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    """One object tracked across frames."""

    track_id: int
    class_code: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    frame_index: int = -1

    hits: int = 1
    confirmed: bool = False
    best_frame_index: int = -1
    best_confidence: float = 0.0
    best_bbox: BBox | None = None

    lat: float | None = None
    lon: float | None = None
    gps_accuracy_m: float | None = None
    emitted: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def update(
        self,
        bbox: BBox,
        confidence: float,
        ts: float,
        frame_index: int,
        lat: float | None,
        lon: float | None,
        accuracy: float | None,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = ts
        self.hits += 1
        if lat is not None and lon is not None:
            self.lat, self.lon = lat, lon
            self.gps_accuracy_m = accuracy
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_frame_index = frame_index
            self.best_bbox = bbox


class ByteTracker:
    """Two-stage association with delayed confirmation."""

    _ids = itertools.count(1)

    def __init__(
        self,
        iou_threshold: float = 0.30,
        high_thresh: float = 0.50,
        low_thresh: float = 0.25,
        max_age_s: float = 2.0,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.tracks: list[Track] = []

    def reset(self) -> None:
        self.tracks.clear()

    # -- matching ---------------------------------------------------------
    def _match(
        self,
        detections: list,
        matched_tracks: set[int],
        matched_dets: set[int],
        pending: dict[int, object],
    ) -> None:
        """Greedy highest-IoU matching within each class."""
        if not detections:
            return
        candidates: list[tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks or track.class_code != det.class_code:
                    continue
                overlap = iou(track.bbox, det.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, ti, di))
        # Highest overlap first; ties broken deterministically by index.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            pending[ti] = detections[di]

    # -- main -------------------------------------------------------------
    def update(
        self,
        detections: list,
        ts: float,
        frame_index: int = -1,
        position: tuple[float, float, float | None] | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame.

        Returns the tracks newly confirmed by this frame (possibly empty).
        """
        lat, lon, accuracy = position if position else (None, None, None)

        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        below = [d for d in detections if d.confidence < self.low_thresh]

        if below:
            log.debug("dropped %d sub-threshold detection(s)", len(below))

        # Age out before associating. If ageing ran after matching, a stale
        # track could absorb a detection from a different place entirely --
        # a vehicle 111 m and 10 s later, driving past a visually similar
        # patch of road, would extend the old track instead of starting a new
        # one, and the geofence would never get the chance to suppress it.
        self.tracks = [t for t in self.tracks if ts - t.last_seen <= self.max_age_s]

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        pending: dict[int, object] = {}

        # Stage 1: high confidence establishes identity.
        self._match(high, matched_tracks, matched_dets, pending)
        # Stage 2: low confidence keeps a track alive but cannot create one.
        self._match(low, matched_tracks, matched_dets, pending)

        for ti, det in pending.items():
            self.tracks[ti].update(det.bbox, det.confidence, ts, frame_index, lat, lon, accuracy)

        # Unmatched high-confidence detections spawn new tracks.
        for di, det in enumerate(high):
            if di in matched_dets:
                continue
            track = Track(
                track_id=next(self._ids),
                class_code=det.class_code,
                bbox=det.bbox,
                confidence=det.confidence,
                first_seen=ts,
                last_seen=ts,
                frame_index=frame_index,
                best_frame_index=frame_index,
                best_confidence=det.confidence,
                best_bbox=det.bbox,
                lat=lat,
                lon=lon,
                gps_accuracy_m=accuracy,
            )
            self.tracks.append(track)

        # Confirm tracks that have persisted long enough.
        newly_confirmed: list[Track] = []
        for track in self.tracks:
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                newly_confirmed.append(track)
        return newly_confirmed

    @property
    def active(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
