"""Tests for the detector singleton and GPS quality gating."""

from __future__ import annotations

import math
import random

import pytest

from roadscope.geo.kalman import LatLonKalman
from roadscope.geo.quality import GPSFix, QualityFlag, gate_fix, haversine_m
from roadscope.vision.detector import DefectDetector

# --- singleton (Phase 1 gate) -------------------------------------------


@pytest.fixture(autouse=True)
def _reset_detector():
    DefectDetector.reset()
    yield
    DefectDetector.reset()


def test_detector_singleton():
    """The gate test: exactly one instance, weights loaded once.

    The legacy dashboard constructed the detector twice per rerun
    (app/dashboard.py:163 and :194), loading 22 MB of weights each time.
    """
    first = DefectDetector.instance()
    second = DefectDetector.instance()
    assert first is second
    assert DefectDetector.load_count == 1


def test_detector_does_not_raise_without_weights(tmp_path, monkeypatch):
    """Missing weights must disable the detector, never silently swap engines."""
    monkeypatch.setenv("ROADSCOPE_ROOT", str(tmp_path))
    from roadscope import paths

    paths.reset_cache()
    detector = DefectDetector.instance()
    assert detector.available is False
    assert detector.engine_name == "unavailable"


def test_single_class_model_maps_to_pothole():
    """Legacy weights have one class named '0'; that can only be a pothole."""
    detector = DefectDetector.instance()
    detector.names = {0: "0"}
    assert detector.resolve_class(0) == "D40"


def test_multiclass_model_uses_its_own_names():
    detector = DefectDetector.instance()
    detector.names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    assert detector.resolve_class(0) == "D00"
    assert detector.resolve_class(1) == "D10"
    assert detector.resolve_class(2) == "D20"
    assert detector.resolve_class(3) == "D40"


def test_named_model_maps_by_substring():
    detector = DefectDetector.instance()
    detector.names = {0: "longitudinal_crack", 1: "transverse_crack", 2: "pothole"}
    assert detector.resolve_class(0) == "D00"
    assert detector.resolve_class(1) == "D10"
    assert detector.resolve_class(2) == "D40"


# --- GPS quality gating --------------------------------------------------


def test_gate_accepts_good_fix():
    fix = GPSFix(latitude=12.9716, longitude=79.1585, accuracy_m=2.4, n_sats=11, hdop=0.9)
    accepted, reason = gate_fix(fix)
    assert accepted is True
    assert reason is QualityFlag.GOOD


def test_gate_rejects_weak_accuracy():
    fix = GPSFix(latitude=12.97, longitude=79.15, accuracy_m=45.0, n_sats=9, hdop=1.0)
    accepted, reason = gate_fix(fix)
    assert accepted is False
    assert reason is QualityFlag.WEAK_ACCURACY


def test_gate_rejects_few_satellites():
    fix = GPSFix(latitude=12.97, longitude=79.15, accuracy_m=3.0, n_sats=3, hdop=0.8)
    accepted, reason = gate_fix(fix)
    assert accepted is False
    assert reason is QualityFlag.FEW_SATELLITES


def test_gate_rejects_bad_hdop():
    fix = GPSFix(latitude=12.97, longitude=79.15, accuracy_m=3.0, n_sats=9, hdop=4.5)
    accepted, reason = gate_fix(fix)
    assert accepted is False
    assert reason is QualityFlag.BAD_HDOP


def test_gate_rejects_stale_fix():
    import time

    fix = GPSFix(latitude=12.97, longitude=79.15, ts_utc=time.time() - 30.0)
    accepted, reason = gate_fix(fix)
    assert accepted is False
    assert reason is QualityFlag.STALE


def test_gate_rejects_impossible_coordinates():
    fix = GPSFix(latitude=999.0, longitude=79.15)
    accepted, reason = gate_fix(fix)
    assert accepted is False
    assert reason is QualityFlag.INVALID


def test_gate_accepts_fix_without_metadata():
    """A phone sending only lat/lon must still work."""
    fix = GPSFix(latitude=12.97, longitude=79.15)
    accepted, _ = gate_fix(fix)
    assert accepted is True


# --- Kalman --------------------------------------------------------------


def test_kalman_first_fix_passes_through():
    kf = LatLonKalman()
    fix = GPSFix(latitude=12.9716, longitude=79.1585)
    out = kf.update(fix, t_mono=0.0)
    assert out.latitude == pytest.approx(12.9716, abs=1e-9)
    assert out.longitude == pytest.approx(79.1585, abs=1e-9)


def test_kalman_reduces_variance():
    """Filtered positions must scatter less than the raw noisy measurements."""
    random.seed(7)
    truth_lat, truth_lon = 12.9716, 79.1585
    # ~4 m of 1-sigma noise on each coordinate.
    noise_lat = 4.0 / 111_320.0
    noise_lon = 4.0 / (111_320.0 * math.cos(math.radians(truth_lat)))

    kf = LatLonKalman()
    raw_errors: list[float] = []
    filtered_errors: list[float] = []

    for i in range(60):
        noisy = GPSFix(
            latitude=truth_lat + random.gauss(0, noise_lat),
            longitude=truth_lon + random.gauss(0, noise_lon),
            accuracy_m=4.0,
        )
        filtered = kf.update(noisy, t_mono=i * 0.2)
        # Skip the settling transient.
        if i >= 20:
            raw_errors.append(haversine_m(noisy.latitude, noisy.longitude, truth_lat, truth_lon))
            filtered_errors.append(
                haversine_m(filtered.latitude, filtered.longitude, truth_lat, truth_lon)
            )

    raw_rms = math.sqrt(sum(e**2 for e in raw_errors) / len(raw_errors))
    filtered_rms = math.sqrt(sum(e**2 for e in filtered_errors) / len(filtered_errors))
    assert filtered_rms < raw_rms * 0.75, (
        f"Kalman did not reduce error: raw={raw_rms:.2f}m filtered={filtered_rms:.2f}m"
    )


def test_kalman_reset():
    kf = LatLonKalman()
    kf.update(GPSFix(latitude=1.0, longitude=1.0), t_mono=0.0)
    kf.reset()
    out = kf.update(GPSFix(latitude=2.0, longitude=2.0), t_mono=0.0)
    assert out.latitude == pytest.approx(2.0, abs=1e-9)


# --- distance ------------------------------------------------------------


def test_haversine_known_distance():
    # Bengaluru -> Chennai is ~290 km.
    d = haversine_m(12.9716, 77.5946, 13.0827, 80.2707)
    assert 285_000 < d < 295_000


def test_haversine_zero():
    assert haversine_m(12.0, 79.0, 12.0, 79.0) == pytest.approx(0.0, abs=1e-6)
