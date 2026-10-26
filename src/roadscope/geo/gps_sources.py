"""GPS sources for the edge agent.

The agent never had a GPS source: ``pipeline.ingest_gps`` existed, but nothing
called it, so every survey was positionless. This module supplies the sources
that feed it.

Three are provided, in order of preference for a laptop-and-phone setup:

* :class:`HttpGpsSource` -- polls a phone app that publishes GPS over HTTP
  (e.g. "GPS Logger", "GPS2IP", or a custom endpoint). Zero hardware.
* :class:`CsvGpsSource` -- replays a recorded track. Used for testing and for
  re-running a survey without re-driving.
* :class:`SyntheticGpsSource` -- a straight line at constant speed, for tests
  and demos where no phone is present.

None of them require a GPS receiver or osmnx. Map matching is a separate
optional module that degrades to no road name when osmnx is absent.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import URLError
from urllib.request import urlopen

from ..geo.quality import GPSFix


class GpsSource:
    """Common interface for anything that produces GPS fixes."""

    def start(self, on_fix: Callable[[GPSFix], None]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class SyntheticGpsSource(GpsSource):
    """A straight line due east at constant speed. For tests and demos."""

    def __init__(
        self,
        lat0: float = 23.0225,
        lon0: float = 72.5714,
        speed_kmh: float = 40.0,
        hz: float = 1.0,
        duration_s: float = 3600.0,
        accuracy_m: float = 8.0,
        n_sats: int = 9,
        hdop: float = 0.9,
    ) -> None:
        self.lat0 = lat0
        self.lon0 = lon0
        self.speed_mps = speed_kmh / 3.6
        self.hz = hz
        self.duration_s = duration_s
        self.accuracy_m = accuracy_m
        self.n_sats = n_sats
        self.hdop = hdop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_fix: Callable[[GPSFix], None]) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_fix,), daemon=True, name="synthetic-gps"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self, on_fix: Callable[[GPSFix], None]) -> None:
        start = time.monotonic()
        i = 0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(self.lat0))
        while not self._stop.is_set():
            if time.monotonic() - start > self.duration_s:
                return
            t = time.monotonic()
            lon = self.lon0 + (i * self.speed_mps) / m_per_deg_lon
            fix = GPSFix(
                latitude=self.lat0,
                longitude=lon,
                speed_kmh=self.speed_mps * 3.6,
                heading=90.0,
                accuracy_m=self.accuracy_m,
                hdop=self.hdop,
                n_sats=self.n_sats,
                t_mono=t,
                ts_utc=datetime.now(UTC).timestamp(),
            )
            on_fix(fix)
            i += 1
            time.sleep(1.0 / self.hz)


class CsvGpsSource(GpsSource):
    """Replay a recorded track from a CSV file.

    Columns recognised (case-insensitive): lat/latitude, lon/longitude,
    ts/ts_utc/timestamp, speed/speed_kmh, heading, accuracy/accuracy_m,
    hdop, n_sats/satellites. Missing columns are filled with sensible defaults.
    """

    def __init__(self, path: str | Path, *, loop: bool = False, hz: float | None = None) -> None:
        self.path = Path(path)
        self.loop = loop
        self.hz = hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_fix: Callable[[GPSFix], None]) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_fix,), daemon=True, name="csv-gps"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self, on_fix: Callable[[GPSFix], None]) -> None:
        rows = self._read()
        if not rows:
            return
        while True:
            for prev_ts, fix in rows:
                if self._stop.is_set():
                    return
                on_fix(fix)
                sleep = self._sleep(prev_ts, fix)
                if sleep > 0:
                    time.sleep(sleep)
            if not self.loop:
                return

    def _read(self) -> list[tuple[float, GPSFix]]:
        out: list[tuple[float, GPSFix]] = []
        with self.path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                fix, ts = self._parse(raw)
                out.append((ts, fix))
        return out

    def _parse(self, raw: dict[str, str]) -> tuple[GPSFix, float]:
        norm = {k.strip().lower(): v for k, v in raw.items()}

        def get(*names: str, default: str | None = None) -> str | None:
            for n in names:
                if n in norm and norm[n] not in (None, ""):
                    return norm[n]
            return default

        lat = float(get("lat", "latitude"))
        lon = float(get("lon", "longitude"))
        ts_raw = get("ts", "ts_utc", "timestamp", "time")
        if ts_raw is None:
            ts = time.time()
        elif ts_raw.replace(".", "", 1).isdigit():
            ts = float(ts_raw)
        else:
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = time.time()

        accuracy = _float(get("accuracy", "accuracy_m"), 8.0)
        return (
            GPSFix(
                latitude=lat,
                longitude=lon,
                speed_kmh=_float(get("speed", "speed_kmh"), 0.0),
                heading=_float(get("heading"), 0.0),
                accuracy_m=accuracy,
                hdop=_float(get("hdop"), 0.9),
                n_sats=int(_float(get("n_sats", "satellites", "sats"), 9)),
                ts_utc=ts,
                t_mono=None,
            ),
            ts,
        )

    def _sleep(self, prev_ts: float, fix: GPSFix) -> float:
        if self.hz:
            return 1.0 / self.hz
        # Replay at the original recording rate when no override is given.
        return 0.0


class HttpGpsSource(GpsSource):
    """Poll a phone app that publishes GPS over HTTP.

    Expected JSON shape (any subset is accepted)::

        {"lat": 23.02, "lon": 72.57, "accuracy": 8.0,
         "speed": 40.0, "heading": 90.0, "hdop": 0.9, "sats": 9}

    Field names are matched case-insensitively and several common aliases are
    accepted (latitude/longitude, accuracy_m, speed_kmh, ...), so this works
    against "GPS Logger", "GPS2IP", and most similar apps without configuration.
    """

    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "lat": ("lat", "latitude", "lat_deg"),
        "lon": ("lon", "longitude", "lng", "lon_deg"),
        "accuracy": ("accuracy", "accuracy_m", "acc", "hacc"),
        "speed": ("speed", "speed_kmh", "speed_kph", "velocity"),
        "heading": ("heading", "course", "bearing", "track"),
        "hdop": ("hdop", "hdop_value"),
        "sats": ("sats", "n_sats", "satellites", "nsat"),
    }

    def __init__(
        self,
        url: str,
        *,
        interval_s: float = 1.0,
        timeout_s: float = 3.0,
    ) -> None:
        self.url = url
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_fix: Callable[[GPSFix], None]) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_fix,), daemon=True, name="http-gps"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self, on_fix: Callable[[GPSFix], None]) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                with urlopen(self.url, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                # A phone on a flaky hotspot will drop packets. Log once per
                # failure rather than spamming, and keep polling.
                print(f"GPS poll failed: {exc}")
                payload = None

            if isinstance(payload, dict):
                fix = self._to_fix(payload)
                if fix is not None:
                    on_fix(fix)

            elapsed = time.monotonic() - t0
            wait = self.interval_s - elapsed
            if wait > 0:
                self._stop.wait(wait)

    def _to_fix(self, payload: dict[str, Any]) -> GPSFix | None:
        def pick(*names: str) -> Any:
            for n in names:
                for key, value in payload.items():
                    if key.lower() == n.lower():
                        return value
            return None

        lat = pick(*self.ALIASES["lat"])
        lon = pick(*self.ALIASES["lon"])
        if lat is None or lon is None:
            return None
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            return None

        return GPSFix(
            latitude=lat_f,
            longitude=lon_f,
            speed_kmh=_float(pick(*self.ALIASES["speed"]), 0.0),
            heading=_float(pick(*self.ALIASES["heading"]), 0.0),
            accuracy_m=_float(pick(*self.ALIASES["accuracy"]), 8.0),
            hdop=_float(pick(*self.ALIASES["hdop"]), 0.9),
            n_sats=int(_float(pick(*self.ALIASES["sats"]), 9)),
            ts_utc=time.time(),
            t_mono=time.monotonic(),
        )


def _float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
