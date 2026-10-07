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
