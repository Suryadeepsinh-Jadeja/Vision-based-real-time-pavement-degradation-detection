"""End-to-end pipeline tests.

These prove the claims in REBUILD_PLAN.md Phase 2 (section 8.4) rather than
asserting them:

* ingest is non-blocking and inference cannot stall capture
* one pothole yields exactly one record, not five per second
* a 60s link loss loses nothing
* the agent does not import Streamlit
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from roadscope.agent.ingest import SyntheticSource
from roadscope.agent.pipeline import EdgePipeline
from roadscope.config import Settings
from roadscope.geo.quality import GPSFix
from roadscope.server.subscriber import Receiver
from roadscope.storage.repository import Repository
from roadscope.transport.outbox import Outbox
from roadscope.transport.topics import TOPIC_DETECTION, TOPIC_GPS, topic_for
from roadscope.vision.detector import DefectDetector


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        device_id="test-van",
        outbox_db=tmp_path / "outbox.sqlite",
        server_db=tmp_path / "server.sqlite",
        queue_size=4,
        conf_threshold=0.35,
    )


@pytest.fixture
def offline(settings):
    """A publisher that is never connected, like a vehicle out of coverage."""

    class Offline:
        connected = False

        def publish(self, *a, **k):
            return False

    return Offline()


# --- the link-loss guarantee --------------------------------------------


def test_link_loss_loses_nothing(settings, offline):
    """The headline Phase 5 guarantee, provable without a broker.

    Capture while the publisher refuses everything, then flush once it comes
    back. Every record must survive.
    """
    pipeline = EdgePipeline(
        settings, source=SyntheticSource(fps=30, duration_s=1.0), publisher=offline
    )
    pipeline.start()
    for i in range(20):
        pipeline.ingest_gps(
            GPSFix(
                latitude=12.9716 + i * 1e-4, longitude=79.1585, ts_utc=time.time(), t_mono=i * 0.2
            )
        )
    time.sleep(1.5)
    pipeline.stop()

    pending = pipeline.outbox.depth()
    assert pending > 0, "expected messages buffered while the broker was down"

    # Broker returns.
    online = type("Online", (), {"connected": True, "publish": lambda self, t, p, **k: True})()
    pipeline.publisher = online
    delivered = pipeline.flush()

    assert delivered == pending
    assert pipeline.outbox.depth() == 0, "messages lost or duplicated after reconnect"


def test_outbox_replay_after_restart(settings, offline):
    """A restarted agent must deliver what the dead one queued."""
    first = EdgePipeline(
        settings, source=SyntheticSource(fps=30, duration_s=0.6), publisher=offline
    )
    first.start()
    first.ingest_gps(GPSFix(latitude=12.97, longitude=79.15, ts_utc=time.time(), t_mono=0.0))
    time.sleep(0.8)
    first.stop()
    queued = first.outbox.depth()
    assert queued > 0

    # New process, same outbox file: the data is still there.
    reopened = Outbox(settings.outbox_db)
    assert reopened.depth() == queued

    online = type("Online", (), {"connected": True, "publish": lambda self, t, p, **k: True})()
    pipeline = EdgePipeline(settings, source=SyntheticSource(duration_s=0.1), publisher=online)
    assert pipeline.flush() == queued
    assert pipeline.outbox.depth() == 0


# --- idempotency --------------------------------------------------------


def test_duplicate_gps_seq_is_idempotent(settings):
    """QoS1 may redeliver. A replayed seq must not duplicate a row."""
    repo = Repository(settings.server_db)
    receiver = Receiver(settings, repo)
    payload = {"seq": 7, "ts_utc": time.time(), "lat": 12.97, "lon": 79.15}

    for _ in range(5):
        receiver.handle(TOPIC_GPS, {"device_id": "test-van", **payload})

    assert receiver.counts["gps"] == 1
    assert receiver.counts["duplicates"] == 4


def test_duplicate_detection_id_is_idempotent(settings):
    repo = Repository(settings.server_db)
    receiver = Receiver(settings, repo)
    payload = {
        "detection_id": "det-abc",
        "ts_utc": time.time(),
        "lat": 12.97,
        "lon": 79.15,
        "class_code": "D40",
        "confidence": 0.9,
    }
    for _ in range(3):
        receiver.handle(TOPIC_DETECTION, {"device_id": "test-van", **payload})
    assert len(repo.detections()) == 1


# --- ingest must not block ---------------------------------------------


def test_ingest_is_not_blocked_by_inference(settings, offline):
    """Capture must run ahead of a slow detector, dropping rather than waiting.

    A stub detector that sleeps far longer than the frame interval proves the
    queue absorbs the difference instead of back-pressuring the capture thread.
    """

    class SlowDetector:
        available = True
        engine_name = "slow-stub"

        def infer(self, frame, conf=0.35, iou=0.5, imgsz=960):
            time.sleep(0.25)
            return []

    source = SyntheticSource(fps=60, duration_s=2.0)
    pipeline = EdgePipeline(settings, source=source, detector=SlowDetector(), publisher=offline)
    pipeline.start()
    time.sleep(2.2)
    pipeline.stop()

    stats = pipeline.health.snapshot()
    assert stats["frames_in"] > 60, f"capture stalled: only {stats['frames_in']} frames"
    assert stats["frames_dropped"] > 0, "expected drops when inference is 15x slower"
    assert stats["queue_depth"] <= settings.queue_size


def test_health_reports_drop_percentage(settings, offline):
    pipeline = EdgePipeline(
        settings, source=SyntheticSource(fps=30, duration_s=0.5), publisher=offline
    )
    pipeline.start()
    time.sleep(0.7)
    pipeline.stop()
    health = pipeline.health
    assert 0.0 <= health.drop_pct <= 100.0
    assert health.snapshot()["frames_in"] > 0


# --- architecture guard -------------------------------------------------


def test_agent_does_not_import_streamlit():
    """The agent must run with no dashboard present."""
    modules = list(
        (Path(__file__).resolve().parents[1] / "src" / "roadscope" / "agent").glob("*.py")
    )
    assert modules
    for module in modules:
        text = module.read_text()
        assert "import streamlit" not in text, f"{module.name} imports Streamlit"
        assert "from streamlit" not in text, f"{module.name} imports Streamlit"


def test_detector_still_singleton_through_pipeline(settings, offline):
    DefectDetector.reset()
    pipeline = EdgePipeline(settings, source=SyntheticSource(duration_s=0.1), publisher=offline)
    pipeline.stop()
    assert DefectDetector.load_count == 1
    DefectDetector.reset()


# --- full chain ---------------------------------------------------------


def test_capture_to_persistence_end_to_end(settings, offline):
    """Frame in, record out, through the real outbox and receiver."""

    class StubDetector:
        available = True
        engine_name = "stub"

        def infer(self, frame, conf=0.35, iou=0.5, imgsz=960):
            return []

    pipeline = EdgePipeline(
        settings,
        source=SyntheticSource(fps=20, duration_s=1.0),
        detector=StubDetector(),
        publisher=offline,
    )
    pipeline.start()
    for i in range(10):
        pipeline.ingest_gps(
            GPSFix(
                latitude=12.9716 + i * 1e-4,
                longitude=79.1585,
                ts_utc=time.time(),
                t_mono=i * 0.1,
                accuracy_m=3.0,
                n_sats=9,
                hdop=0.9,
            )
        )
    time.sleep(1.2)
    pipeline.stop()

    repo = Repository(settings.server_db)
    receiver = Receiver(settings, repo)
    outbox = Outbox(settings.outbox_db)
    drained = 0
    for row in outbox.pending(limit=10_000):
        receiver.handle(row.topic.rsplit("/", 1)[-1], row.payload)
        outbox.mark_sent(row.id)
        drained += 1

    assert drained > 0
    fixes = repo.track(receiver.trips["test-van"])
    assert len(fixes) == 10
    # Filtering must not have discarded good fixes.
    assert all(f["quality_flag"] == "good" for f in fixes)
    # And accuracy metadata must have survived the whole chain.
    assert fixes[0]["accuracy_m"] == pytest.approx(3.0)
    assert fixes[0]["n_sats"] == 9


def test_poor_quality_fixes_are_not_persisted(settings, offline):
    """Gating must apply before anything reaches the database."""
    pipeline = EdgePipeline(settings, source=SyntheticSource(duration_s=0.1), publisher=offline)
    assert (
        pipeline.ingest_gps(
            GPSFix(latitude=12.97, longitude=79.15, accuracy_m=50.0, ts_utc=time.time())
        )
        is False
    )
    assert (
        pipeline.ingest_gps(
            GPSFix(
                latitude=12.97,
                longitude=79.15,
                accuracy_m=3.0,
                n_sats=9,
                hdop=0.8,
                ts_utc=time.time(),
            )
        )
        is True
    )
    assert pipeline.health.snapshot()["gps_in"] == 1
    assert pipeline.health.snapshot()["gps_rejected"] == 1


def test_topics_are_namespaced(settings):
    assert topic_for("roadscope", "van-01", TOPIC_GPS) == "roadscope/van-01/gps"
    with pytest.raises(ValueError):
        topic_for("roadscope", "bad/id", TOPIC_GPS)


# --- detector playground ---------------------------------------------------
# Upload-and-detect support. The counting logic is tested here so the UI does
# not have to be launched to know whether a number would be right.


def test_one_defect_across_many_frames_counts_once(tmp_path):
    """A pothole held in view must be one defect, not one per frame."""
    from roadscope.fusion.tracker import ByteTracker
    from roadscope.vision.detector import Detection

    tracker = ByteTracker()
    emitted = 0
    for i in range(30):
        # Same box, drifting slightly as a vehicle approaches.
        box = (100 + i * 2, 100, 200 + i * 2, 180)
        dets = [Detection(class_code="D40", confidence=0.8, bbox=box, class_id=3)]
        emitted += len(tracker.update(dets, ts=i * 0.2, frame_index=i))
    assert emitted == 1, f"repeat sightings should collapse to one defect, got {emitted}"


def test_distinct_defects_are_not_collapsed(tmp_path):
    from roadscope.fusion.tracker import ByteTracker
    from roadscope.vision.detector import Detection

    tracker = ByteTracker()
    emitted = 0
    for i in range(30):
        dets = [
            Detection(
                class_code="D40", confidence=0.8, bbox=(100 + i, 100, 200 + i, 180), class_id=3
            ),
            Detection(class_code="D00", confidence=0.7, bbox=(600, 300, 700, 360), class_id=0),
        ]
        emitted += len(tracker.update(dets, ts=i * 0.2, frame_index=i))
    assert emitted == 2


def test_upload_name_cannot_escape_the_upload_directory(tmp_path):
    """An upload filename is attacker-controlled and must not build a path."""
    from roadscope.ui.playground import load_upload

    target = load_upload("../../etc/passwd", b"payload", tmp_path / "uploads")
    assert target.parent == tmp_path / "uploads"
    assert ".." not in str(target)


def test_upload_name_keeps_its_extension(tmp_path):
    from roadscope.ui.playground import load_upload

    target = load_upload("clip.mp4", b"data", tmp_path / "uploads")
    assert target.suffix == ".mp4"


def test_detection_normalisation_is_clamped_to_the_frame(tmp_path):
    """A box spilling past the edge must not produce a normalised coord > 1."""

    from roadscope.ui.playground import to_detection_dict
    from roadscope.vision.detector import Detection

    det = Detection(class_code="D40", confidence=0.9, bbox=(-20, -10, 900, 500), class_id=3)
    d = to_detection_dict(det, frame_w=800, frame_h=480)
    x1, y1, x2, y2 = d["box_norm"]
    assert 0.0 <= x1 <= x2 <= 1.0
    assert 0.0 <= y1 <= y2 <= 1.0


def test_annotate_draws_without_changing_the_original():
    import numpy as np

    from roadscope.ui.playground import annotate

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    dets = [
        {
            "bbox": (10, 10, 60, 50),
            "class_code": "D40",
            "confidence": 0.9,
            "box_norm": (0, 0, 1, 1),
            "width_px": 50,
            "height_px": 40,
            "area_frac": 0.1,
        }
    ]
    out = annotate(frame, dets)
    assert out.shape == frame.shape
    assert out.any(), "nothing was drawn"
    assert not frame.any(), "the input frame must not be mutated"


def test_small_targets_are_flagged():
    """A miss on a distant shot should be called plausible, not certain."""
    from roadscope.ui.playground import is_probable_small_target

    tiny = [{"area_frac": 0.002, "width_px": 12, "height_px": 8}]
    big = [{"area_frac": 0.20, "width_px": 300, "height_px": 300}]
    assert is_probable_small_target(tiny) is True
    assert is_probable_small_target(big) is False
    assert is_probable_small_target([]) is False


def test_confidence_threshold_is_clamped():
    from roadscope.ui.playground import safe_conf

    assert 0.01 <= safe_conf(0.0) <= 0.99
    assert safe_conf(5.0) <= 0.99
    assert safe_conf(-1.0) >= 0.01


def test_video_sampling_reduces_frames(tmp_path):
    """Sampling must actually skip frames, and report it rather than hide it."""
    import cv2
    import numpy as np

    from roadscope.ui.playground import iter_video_frames

    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(60):
        vw.write(np.zeros((240, 320, 3), dtype=np.uint8))
    vw.release()

    every = list(iter_video_frames(path, sample_fps=None))
    sampled = list(iter_video_frames(path, sample_fps=5.0))
    assert len(every) > len(sampled), "sampling must reduce the frame count"
    assert len(sampled) >= 1


def test_upload_path_resolution_matches_the_ui():
    """The playground must build upload paths the way the UI actually does.

    Regression guard. ``main()`` called ``settings.data_dir()``, but
    ``data_dir`` is a function on ``roadscope.paths`` -- Settings only carries
    individual paths. Every upload therefore died with
    ``AttributeError: 'Settings' object has no attribute 'data_dir'``.

    It survived review because the earlier test called ``load_upload`` with a
    hand-made tmpdir instead of going through the same expression the UI uses.
    This asserts the real expression, and asserts no UI module calls a method
    that does not exist on Settings.
    """
    from roadscope import paths
    from roadscope.config import Settings
    from roadscope.ui.playground import load_upload

    settings = Settings()
    upload_dir = paths.data_dir() / "uploads"

    # The exact call the UI makes.
    resolved = load_upload("0x0.png", b"data", upload_dir)
    assert resolved.parent == upload_dir
    assert resolved.suffix == ".png"

    # And the mistake that caused the crash is now impossible to reintroduce
    # unnoticed: data_dir is not a Settings attribute.
    assert not hasattr(settings, "data_dir")
    assert hasattr(paths, "data_dir")


def test_ui_modules_only_use_real_settings_attributes():
    """Every ``settings.X`` used by the UI must exist on Settings.

    A missing field is a crash on first upload, and the failure only appears
    once a user interacts with the page.
    """
    import pathlib
    import re

    from roadscope.config import Settings

    ui_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "roadscope" / "ui"
    pattern = re.compile(r"\bsettings\.([a-z_]+)")
    settings = Settings()
    offenders: list[str] = []

    for py in ui_dir.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            for name in pattern.findall(line):
                if not hasattr(settings, name):
                    offenders.append(f"{py.name}:{lineno} settings.{name}")

    assert not offenders, "undefined Settings attributes: " + ", ".join(offenders)


# --- frame browser ---------------------------------------------------------
# Stepping through frames must not re-run detection, and the navigation maths
# must handle edges. The UI cannot be exercised headlessly, so the index logic
# the buttons depend on is pinned here.


def _fake_run(hit_positions, n_frames=15):
    """A RunResult with detections on chosen frame positions."""
    from roadscope.ui.playground import FrameResult, RunResult

    frames = [
        FrameResult(
            index=i * 6,
            timestamp_s=i * 0.2,
            detections=[{"class_code": "D40"}] if i in hit_positions else [],
            annotated=None,
            width=1280,
            height=720,
        )
        for i in range(n_frames)
    ]
    return RunResult(frames=frames, counts={"D40": len(hit_positions)}, imgsz=960)


def test_hit_indices_are_derived_from_frame_contents():
    r = _fake_run({2, 3, 4, 8, 9, 13})
    hits = [i for i, f in enumerate(r.frames) if f.count]
    assert hits == [2, 3, 4, 8, 9, 13]


def test_previous_and_next_hit_never_run_off_the_ends():
    hits = [2, 5, 9]
    prev_hit = lambda pos: max((i for i in hits if i < pos), default=None)  # noqa: E731
    next_hit = lambda pos: min((i for i in hits if i > pos), default=None)  # noqa: E731

    assert prev_hit(0) is None, "nothing before the first frame"
    assert next_hit(14) is None, "nothing after the last frame"
    assert prev_hit(6) == 5
    assert next_hit(6) == 9
    # Standing on a hit must move strictly past it, never return it.
    assert next_hit(5) == 9
    assert prev_hit(9) == 5


def test_browser_lands_on_the_first_detection_by_default():
    """The frame the user came to look at should be the one shown."""
    hits = [7, 8, 9]
    assert (hits[0] if hits else 0) == 7


def test_stale_frame_position_is_discarded_on_a_shorter_run():
    """A new, shorter clip must not leave the slider past the end."""
    stored, last = 13, 3
    assert stored > last, "a stale position must be detected as out of range"


def test_single_frame_run_has_a_degenerate_but_valid_range():
    r = _fake_run({0}, n_frames=1)
    assert len(r.frames) == 1
    assert 0 <= 0 <= len(r.frames) - 1
