"""Constant-velocity Kalman filter for latitude/longitude.

Raw consumer GPS under urban canyon conditions jitters by several metres frame to
frame. The legacy system fed those raw fixes straight through, which is why
detections landed on the wrong side of the road.

This filter runs a local tangent-plane (ENU) Kalman filter: positions are
projected to metres east/north of an anchor, filtered, then projected back. At
urban scales (< 20 km) the projection error is negligible, and working in metres
avoids the numerical conditioning problems of filtering degrees directly.

State: ``[east, north, v_east, v_north]``
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .quality import GPSFix

_EARTH_RADIUS_M = 6_371_008.8


class LatLonKalman:
    """Constant-velocity Kalman filter for 2D positions."""

    def __init__(
        self,
        process_noise: float = 4.0,
        measurement_noise_m: float = 10.0,
        initial_variance: float = 1_000.0,
    ) -> None:
        """
        Parameters
        ----------
        process_noise:
            Accelelation variance (m^2/s^3). Higher = tracks manoeuvre faster.
        measurement_noise_m:
            Assumed 1-sigma measurement error (m). Default 10 m suits consumer GPS
            in urban conditions; lower it if the client reports better accuracy.
        initial_variance:
            Initial state covariance. Large, so the first fixes are trusted.
        """
        self.q = process_noise
        self.r = measurement_noise_m**2
        self.p0 = initial_variance

        self._anchor: tuple[float, float] | None = None
        self._x = np.zeros(4, dtype=float)
        self._p = np.eye(4, dtype=float) * self.p0
        self._initialised = False
        self._last_t: float | None = None

    # -- projection helpers -------------------------------------------------
    def _to_enu(self, lat: float, lon: float) -> tuple[float, float]:
        if self._anchor is None:
            self._anchor = (lat, lon)
        lat0, lon0 = self._anchor
        east = math.radians(lon - lon0) * _EARTH_RADIUS_M * math.cos(math.radians(lat0))
        north = math.radians(lat - lat0) * _EARTH_RADIUS_M
        return east, north

    def _from_enu(self, east: float, north: float) -> tuple[float, float]:
        assert self._anchor is not None
        lat0, lon0 = self._anchor
        lat = lat0 + math.degrees(north / _EARTH_RADIUS_M)
        lon = lon0 + math.degrees(east / (_EARTH_RADIUS_M * math.cos(math.radians(lat0))))
        return lat, lon

    # -- filter -------------------------------------------------------------
    def update(self, fix: GPSFix, t_mono: float | None = None) -> GPSFix:
        """Filter one fix and return a corrected copy.

        The first accepted fix passes through untouched. Subsequent fixes are
        blended by the filter's own covariance, so the output is always at least
        as good as a single measurement.
        """
        measurement = np.array(self._to_enu(fix.latitude, fix.longitude), dtype=float)

        if not self._initialised:
            self._x[0], self._x[1] = measurement
            self._p = np.diag([self.p0, self.p0, self.p0, self.p0]).astype(float)
            self._initialised = True
            self._last_t = t_mono
            return fix

        dt = 1.0
        if t_mono is not None and self._last_t is not None:
            dt = max(1e-3, min(60.0, t_mono - self._last_t))
        self._last_t = t_mono

        # Predict: constant velocity.
        f = np.eye(4, dtype=float)
        f[0, 2] = dt
        f[1, 3] = dt
        q = np.eye(4, dtype=float) * self.q * dt
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

        # Update.
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = self.r
        if fix.accuracy_m is not None and fix.accuracy_m > 0:
            r = float(fix.accuracy_m) ** 2
        innovation = measurement - h @ self._x
        s = h @ self._p @ h.T + np.eye(2) * r
        k = self._p @ h.T @ np.linalg.inv(s)
        self._x = self._x + k @ innovation
        identity = np.eye(4, dtype=float)
        self._p = (identity - k @ h) @ self._p

        lat, lon = self._from_enu(float(self._x[0]), float(self._x[1]))
        return _with_position(fix, lat, lon)

    def velocity_kmh(self) -> float:
        """Current filtered speed in km/h."""
        if not self._initialised:
            return 0.0
        return math.hypot(float(self._x[2]), float(self._x[3])) * 3.6

    def reset(self) -> None:
        """Forget all state."""
        self._anchor = None
        self._x = np.zeros(4, dtype=float)
        self._p = np.eye(4, dtype=float) * self.p0
        self._initialised = False
        self._last_t = None


def _with_position(fix: GPSFix, lat: float, lon: float) -> GPSFix:
    return dataclasses.replace(fix, latitude=lat, longitude=lon)
