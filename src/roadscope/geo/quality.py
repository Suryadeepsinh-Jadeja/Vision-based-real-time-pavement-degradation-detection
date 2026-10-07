"""GPS fix quality gating.

The legacy system accepted every fix the phone sent and threw the reported
accuracy away. This module decides whether a fix is trustworthy enough to use,
and records why a fix was rejected so the decision is auditable.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum


class QualityFlag(StrEnum):
    """Outcome of gating a single fix."""

    GOOD = "good"
    WEAK_ACCURACY = "weak_accuracy"
    FEW_SATELLITES = "few_satellites"
    BAD_HDOP = "bad_hdop"
    STALE = "stale"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class GPSFix:
    """A single position report."""

    latitude: float
    longitude: float
    speed_kmh: float = 0.0
    heading: float = 0.0
    accuracy_m: float | None = None
    hdop: float | None = None
    n_sats: int | None = None
    seq: int = 0
    ts_utc: float = field(default_factory=time.time)
    t_mono: float | None = None
    quality_flag: QualityFlag = QualityFlag.GOOD
    road_name: str | None = None
    mapmatch_conf: float | None = None

    def with_quality(self, flag: QualityFlag) -> GPSFix:
        """Return a copy carrying the given quality flag."""
        return dataclasses.replace(self, quality_flag=flag)


def is_valid_position(lat: float, lon: float) -> bool:
    """True if the coordinates fall in a plausible range."""
    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def gate_fix(
    fix: GPSFix,
    *,
    max_accuracy_m: float = 10.0,
    min_sats: int = 6,
    max_hdop: float = 2.0,
    max_age_s: float = 2.0,
    now: float | None = None,
) -> tuple[bool, QualityFlag]:
    """Decide whether ``fix`` is fit to use.

    Returns ``(accepted, reason)``. Checks run worst-to-best so the reported
    reason is the most significant failure.

    A ``None`` accuracy/HDOP/sat-count means the client did not report it, which
    is not treated as a failure -- a phone that sends only lat/lon still works,
    it just gets no accuracy metadata downstream.
    """
    if not is_valid_position(fix.latitude, fix.longitude):
        return False, QualityFlag.INVALID

    reference = now if now is not None else time.time()
    age = reference - fix.ts_utc
    # A negative age means the client's clock runs ahead. That is clock skew,
    # handled by clock.py, not staleness, so it is not rejected here.
    if age > max_age_s:
        return False, QualityFlag.STALE

    if fix.accuracy_m is not None and fix.accuracy_m > max_accuracy_m:
        return False, QualityFlag.WEAK_ACCURACY

    if fix.n_sats is not None and fix.n_sats < min_sats:
        return False, QualityFlag.FEW_SATELLITES

    if fix.hdop is not None and fix.hdop > max_hdop:
        return False, QualityFlag.BAD_HDOP

    return True, QualityFlag.GOOD


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two points."""
    earth_radius_m = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * earth_radius_m * math.asin(math.sqrt(a))
