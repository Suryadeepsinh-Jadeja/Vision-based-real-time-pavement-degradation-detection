"""No-GPS behaviour of the edge pipeline.

Indoors a phone can go minutes without an acceptable fix. Detections made
then are dropped (a defect with no position cannot be mapped), so the operator
must be told, and a stationary demo may start from the last known position.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from roadscope.agent.ingest import SyntheticSource
from roadscope.agent.pipeline import EdgePipeline
from roadscope.config import Settings
from roadscope.geo.quality import GPSFix
from roadscope.vision.detector import Detection


class _Detector:
    engine_name = "stub"

    def infer(self, frame, **_):
        return [Detection("D40", 0.8, (10, 10, 60, 60))]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        device_id="test-van",
        outbox_db=tmp_path / "outbox.sqlite",
        server_db=tmp_path / "server.sqlite",
        thumb_dir=tmp_path / "thumbs",
        gps_max_accuracy_m=50.0,
    )


def _pipeline(settings) -> EdgePipeline:
    return EdgePipeline(settings, source=SyntheticSource(duration_s=0.1), detector=_Detector())


def _fix(acc: float) -> GPSFix:
    return GPSFix(
        latitude=12.8437, longitude=80.1518, accuracy_m=acc, ts_utc=time.time(), t_mono=1.0
    )


def test_detection_without_gps_is_announced(settings, caplog):
    p = _pipeline(settings)
    with caplog.at_level(logging.WARNING):
        p._process(np.zeros((120, 160, 3), np.uint8), 1.0)
    assert "NO GPS fix" in caplog.text


def test_rejected_fix_is_announced_with_reason(settings, caplog):
    p = _pipeline(settings)
    with caplog.at_level(logging.WARNING):
        assert p.ingest_gps(_fix(64.0)) is False
    assert "rejected" in caplog.text and "64" in caplog.text


def test_last_fix_survives_a_restart(settings):
    first = _pipeline(settings)
    assert first.ingest_gps(_fix(20.0))
    assert first.last_fix_path.exists()

    second = _pipeline(settings)
    assert second._position_at(5.0) == (0.0, 0.0, None)
    age = second.restore_last_fix(max_age_s=600)
    assert age is not None and age < 5
    lat, lon, _ = second._position_at(5.0)
    assert (round(lat, 4), round(lon, 4)) == (12.8437, 80.1518)


def test_stale_saved_fix_is_not_reused(settings):
    first = _pipeline(settings)
    first.ingest_gps(_fix(20.0))
    second = _pipeline(settings)
    assert second.restore_last_fix(max_age_s=-1) is None
    assert second._position_at(5.0) == (0.0, 0.0, None)
