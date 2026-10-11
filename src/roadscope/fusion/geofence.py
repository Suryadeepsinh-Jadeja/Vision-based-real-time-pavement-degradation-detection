"""Spatial suppression for confirmed defects.

A vehicle surveying the same street twice -- or passing a large pothole for
several seconds as the box grows and drifts -- must not create a second record
for the same physical defect.

Suppression is by geodesic distance against already-emitted defects, using
haversine rather than a planar approximation so it stays correct at survey
scale.

A revisit grace period matters here. Two potholes visible in the *same* frame
share the vehicle's GPS position, because a single camera gives one position per
frame rather than one per object. Without the grace period the geofence would
correctly judge the second one to be within ``radius_m`` of the first and
suppress a genuine defect. Suppression must only apply to a *revisit*, so a
candidate is judged only against emissions older than the grace period. At20
km/h a vehicle cannot re-encounter the same defect inside 5 seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..geo.quality import haversine_m
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Geofence:
    """Remembers emitted defects and suppresses near-duplicates on revisit."""

    radius_m: float = 15.0
    revisit_grace_s: float = 5.0
    _sites: list[tuple[float, float, float]] = field(default_factory=list, repr=False)

    def add(self, lat: float, lon: float, ts: float) -> None:
        """Record an emitted defect and when it was emitted."""
        self._sites.append((lat, lon, ts))

    def is_suppressed(self, lat: float, lon: float, ts: float, extra_radius_m: float = 0.0) -> bool:
        """True if a *previously seen* defect lies within the radius.

        Emissions newer than ``revisit_grace_s`` are ignored, so co-visible
        defects in one frame do not suppress each other.
        """
        limit = self.radius_m + extra_radius_m
        return any(
            ts - site_ts >= self.revisit_grace_s
            and haversine_m(lat, lon, site_lat, site_lon) <= limit
            for site_lat, site_lon, site_ts in self._sites
        )

    def distance_to_nearest_m(self, lat: float, lon: float) -> float:
        """Distance to the nearest recorded defect. Useful for diagnostics."""
        if not self._sites:
            return math.inf
        return min(haversine_m(lat, lon, s[0], s[1]) for s in self._sites)

    def clear(self) -> None:
        self._sites.clear()

    def __len__(self) -> int:
        return len(self._sites)
