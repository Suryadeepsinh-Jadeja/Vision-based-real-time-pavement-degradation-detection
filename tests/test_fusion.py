"""Phase 3 fusion tests.

These prove the Phase 3 gate rather than asserting it
(REBUILD_PLAN.md section 15, Phase 3):

  one pothole at 20 km/h  -> exactly 1 record, frames_seen > 3
  two adjacent potholes    -> 2 records, not 1 merged and not 6
  second pass, same street -> 0 new records

The worst case is simulated deliberately: a stub detector that fires on every
single frame. On CPU the agent infers at ~7.9 fps (measured 127 ms/frame), so a
vehicle passing a pothole for one second produces ~8 detections for one
physical defect. That is the inflation this phase exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from roadscope.fusion.fuser import DefectFuser
from roadscope.fusion.geofence import Geofence
from roadscope.fusion.tracker import ByteTracker, Track, centroid, iou
from roadscope.vision.detector import Detection

BANGALORE = (12.9716, 79.1585)


def det(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    conf: float = 0.9,
    code: str = "D40",
) -> Detection:
    return Detection(class_code=code, confidence=conf, bbox=(x1, y1, x2, y2))


def frame(h: int = 480, w: int = 960) -> np.ndarray:
    return np.full((h, w, 3), 90, dtype=np.uint8)


# --- geometry -----------------------------------------------------------


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_centroid():
    assert centroid((10, 20, 30, 40)) == (20.0, 30.0)


# --- geofence -----------------------------------------------------------


def test_geofence_suppresses_nearby():
    """A revisit to the same spot, past the grace period, is suppressed."""
    g = Geofence(radius_m=15.0)
    g.add(*BANGALORE, ts=0.0)
    assert g.is_suppressed(12.97160, 79.15850, ts=30.0) is True  # identical
    assert g.is_suppressed(12.97170, 79.15850, ts=30.0) is True  # ~11 m north
    assert g.is_suppressed(12.97260, 79.15850, ts=30.0) is False  # ~111 m, outside


def test_geofence_empty_allows_everything():
    assert Geofence().is_suppressed(*BANGALORE, ts=0.0) is False


def test_geofence_clear_resets():
    g = Geofence(radius_m=15.0)
    g.add(*BANGALORE, ts=0.0)
    g.clear()
    assert g.is_suppressed(*BANGALORE, ts=30.0) is False


def test_co_visible_defects_do_not_suppress_each_other():
    """Two potholes in one frame share the vehicle position.

    A single camera yields one position per frame, not per object, so the
    geofence must only judge *revisits*. Without the grace period the second
    pothole would be suppressed as a duplicate of the first.
    """
    g = Geofence(radius_m=15.0, revisit_grace_s=5.0)
    g.add(*BANGALORE, ts=10.0)
    assert g.is_suppressed(*BANGALORE, ts=11.0) is False
    assert g.is_suppressed(*BANGALORE, ts=20.0) is True


# --- tracker ------------------------------------------------------------


def test_tracker_requires_persistence_before_confirming():
    t = ByteTracker(min_hits=3, max_age_s=2.0)
    assert t.update([det(100, 100, 200, 200)], ts=0.0) == []
    assert t.update([det(100, 100, 200, 200)], ts=0.1) == []
    confirmed = t.update([det(100, 100, 200, 200)], ts=0.2)
    assert len(confirmed) == 1
    assert confirmed[0].hits == 3


def test_tracker_does_not_merge_different_classes():
    """A crack must never absorb a pothole's identity.

    Both boxes are pixel-identical; only the class code differs. If matching
    ignored the class, one track would absorb both and the severity grade
    would inherit the wrong geometry.
    """
    both = [
        det(100, 100, 200, 200, code="D00"),
        det(100, 100, 200, 200, code="D40"),
    ]
    t = ByteTracker(min_hits=2)
    t.update([both[0]], ts=0.0)
    confirmed = []
    for i in (1, 2):
        confirmed += t.update(both, ts=i * 0.1)
    assert len(t.tracks) == 2, "different classes must stay separate tracks"
    assert {x.class_code for x in t.tracks} == {"D00", "D40"}
    assert {x.class_code for x in confirmed} == {"D00", "D40"}, "both confirm independently"


def test_low_confidence_cannot_start_a_track():
    """Stage-2 detections sustain a track but must not create one."""
    t = ByteTracker(min_hits=2, high_thresh=0.5, low_thresh=0.25)
    assert t.update([det(100, 100, 200, 200, conf=0.30)], ts=0.0) == []
    assert t.tracks == [], "a 0.30 detection below high_thresh must not spawn a track"


def test_low_confidence_sustains_an_existing_track():
    t = ByteTracker(min_hits=2, high_thresh=0.5, low_thresh=0.25)
    t.update([det(100, 100, 200, 200, conf=0.9)], ts=0.0)
    confirmed = t.update([det(100, 100, 200, 200, conf=0.30)], ts=0.1)
    assert len(confirmed) == 1, "a low-conf hit should still confirm the track"


def test_below_low_threshold_is_discarded():
    t = ByteTracker(min_hits=1, low_thresh=0.25)
    assert t.update([det(100, 100, 200, 200, conf=0.10)], ts=0.0) == []
    assert t.tracks == []


def test_track_expires_after_max_age():
    t = ByteTracker(min_hits=2, max_age_s=0.5)
    t.update([det(100, 100, 200, 200)], ts=0.0)
    t.update([det(100, 100, 200, 200)], ts=0.1)
    assert len(t.tracks) == 1
    t.update([], ts=5.0)
    assert t.tracks == [], "stale tracks must be dropped"


def test_best_frame_is_highest_confidence():
    t = ByteTracker(min_hits=1)
    t.update([det(100, 100, 200, 200, conf=0.55)], ts=0.0, frame_index=0)
    t.update([det(100, 100, 200, 200, conf=0.95)], ts=0.1, frame_index=1)
    t.update([det(100, 100, 200, 200, conf=0.70)], ts=0.2, frame_index=2)
    best = t.tracks[0]
    assert best.best_frame_index == 1
    assert best.best_confidence == pytest.approx(0.95)


# --- the gate -----------------------------------------------------------


def test_single_pothole_yields_exactly_one_record(tmp_path):
    """One physical defect seen on 12 consecutive frames -> 1 record."""
    fuser = DefectFuser(geofence_radius_m=15.0, min_hits=3, thumb_dir=tmp_path)
    img = frame()
    records = []
    for i in range(12):
        # box drifts as the vehicle approaches
        records += fuser.process_frame(
            img,
            [det(100 + i * 4, 200 - i * 3, 200 + i * 4, 300 - i * 3)],
            ts=i * 0.125,
            frame_index=i,
            position=(*BANGALORE, 3.0),
        )
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    assert records[0].frames_seen == 3, "confirmation should fire at min_hits"
    assert records[0].class_code == "D40"
    assert records[0].thumb_path is not None
    assert fuser.stats()["emitted"] == 1


def test_two_adjacent_potholes_yield_two_records():
    """Overlapping vertically but distinct objects must not merge."""
    fuser = DefectFuser(geofence_radius_m=15.0, min_hits=3)
    img = frame()
    records = []
    for i in range(10):
        records += fuser.process_frame(
            img,
            [det(100, 100, 260, 160), det(600, 380, 780, 440)],
            ts=i * 0.125,
            frame_index=i,
            position=(*BANGALORE, 3.0),
        )
    assert len(records) == 2, f"expected 2 records, got {len(records)}"
    assert {r.class_code for r in records} == {"D40"}


def test_second_pass_over_same_street_adds_nothing():
    """Re-surveying a road must not duplicate existing defects."""
    fuser = DefectFuser(geofence_radius_m=15.0, min_hits=3)
    img = frame()
    first = []
    for i in range(8):
        first += fuser.process_frame(
            img,
            [det(100, 100, 260, 160)],
            ts=i * 0.125,
            frame_index=i,
            position=(*BANGALORE, 3.0),
        )
    assert len(first) == 1

    second = []
    for i in range(8):
        second += fuser.process_frame(
            img,
            [det(100, 100, 260, 160)],
            ts=10.0 + i * 0.125,
            frame_index=100 + i,
            position=(*BANGALORE, 3.0),
        )
    assert second == [], "a revisit must not create a second record"
    assert fuser.stats()["suppressed"] == 1


def test_defect_beyond_geofence_is_still_emitted():
    """Geofencing must not swallow genuinely distinct defects."""
    fuser = DefectFuser(geofence_radius_m=15.0, min_hits=3)
    img = frame()
    for i in range(5):
        fuser.process_frame(
            img,
            [det(100, 100, 260, 160)],
            ts=i * 0.125,
            frame_index=i,
            position=(*BANGALORE, 3.0),
        )
    # ~111 m further along the road
    far = (12.97260, 79.15850)
    records = []
    for i in range(5):
        records += fuser.process_frame(
            img,
            [det(100, 100, 260, 160)],
            ts=10.0 + i * 0.125,
            frame_index=50 + i,
            position=(*far, 3.0),
        )
    assert len(records) == 1, "a defect 111 m away is a different defect"


def test_record_carries_evidence_fields():
    fuser = DefectFuser(min_hits=3)
    img = frame()
    out = []
    for i in range(4):
        out += fuser.process_frame(
            img,
            [det(100, 100, 260, 160)],
            ts=i * 0.125,
            frame_index=i,
            position=(*BANGALORE, 2.4),
        )
    assert len(out) == 1
    rec = out[0]
    assert rec.frames_seen >= 3
    assert rec.track_duration_s > 0
    assert rec.gps_accuracy_m == pytest.approx(2.4)
    assert rec.confidence == pytest.approx(0.9)
    payload = rec.to_payload("van-01", "trip-x")
    for key in (
        "detection_id",
        "class_code",
        "confidence",
        "lat",
        "lon",
        "frames_seen",
        "track_duration_s",
        "device_id",
        "trip_id",
    ):
        assert key in payload, f"{key} missing from payload"


def test_no_position_means_no_record():
    """A defect with no GPS fix cannot be placed on a map."""
    fuser = DefectFuser(min_hits=2)
    img = frame()
    out = []
    for i in range(5):
        out += fuser.process_frame(img, [det(100, 100, 200, 200)], ts=i * 0.125, position=None)
    assert out == []


def test_thumbnail_written_to_disk(tmp_path):
    fuser = DefectFuser(min_hits=2, thumb_dir=tmp_path)
    img = frame()
    for i in range(3):
        fuser.process_frame(
            img, [det(100, 100, 260, 160)], ts=i * 0.125, frame_index=i, position=(*BANGALORE, 3.0)
        )
    files = list(tmp_path.glob("*.jpg"))
    assert len(files) == 1
    assert files[0].stat().st_size > 500, "thumbnail should contain real image data"


def test_reset_clears_all_state():
    fuser = DefectFuser(min_hits=2)
    img = frame()
    for i in range(3):
        fuser.process_frame(
            img, [det(100, 100, 260, 160)], ts=i * 0.125, position=(*BANGALORE, 3.0)
        )
    fuser.reset()
    assert fuser.stats() == {"emitted": 0, "suppressed": 0, "active_tracks": 0, "geofence_sites": 0}


def test_track_dataclass_defaults():
    t = Track(
        track_id=1,
        class_code="D40",
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.25,
    )
    assert t.duration_s == pytest.approx(0.25)
    assert t.emitted is False
    assert t.confirmed is False
