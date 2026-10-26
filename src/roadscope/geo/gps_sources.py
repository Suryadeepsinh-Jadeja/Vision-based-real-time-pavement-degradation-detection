"""GPS sources for the edge agent.

The agent never had a GPS source: ``pipeline.ingest_gps`` existed, but nothing
called it, so every survey was positionless. This module supplies the sources
that feed it.

Four are provided, in order of preference for a laptop-and-phone setup:

* :class:`PushGpsSource` -- the laptop listens and an Android app (GPSLogger)
  pushes every fix to it. The recommended live-demo path.
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
from urllib.parse import parse_qs, urlparse
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


class PushGpsSource(GpsSource):
    """Receive GPS fixes that a phone app *pushes* to the laptop.

    Most free Android GPS apps cannot serve their position for polling, but
    they can call a URL on every fix. This source runs a tiny HTTP server and
    turns each request into a :class:`GPSFix`.

    Built for **GPSLogger for Android** (BasicAirData / mendhak, free on
    F-Droid and Play). In its "Custom URL" logger set the URL to::

        http://<laptop-ip>:8765/gps?lat=%LAT&lon=%LON&acc=%ACC&spd=%SPD&dir=%DIR&sat=%SAT

    GET or POST both work; POST bodies may be form-encoded or JSON. ``spd`` is
    metres per second (GPSLogger's unit); ``speed_kmh`` is also accepted.

    ``GET /gps`` with no coordinates returns the last fix as JSON, so you can
    open ``http://<laptop-ip>:8765/gps`` in a browser to confirm the link.

    The fix is stamped with the laptop's receive time, not the phone's clock:
    the agent rejects fixes older than ``gps_max_age_s``, and a phone clock
    that is a few seconds off would otherwise make every fix "stale".
    """

    def __init__(self, port: int = 8765, host: str = "0.0.0.0") -> None:
        self.host = host
        self.port = port
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.last_fix: GPSFix | None = None
        self.received = 0

    def start(self, on_fix: Callable[[GPSFix], None]) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        source = self

        class _Handler(BaseHTTPRequestHandler):
            def _params(self) -> dict[str, Any]:
                parsed = urlparse(self.path)
                params: dict[str, Any] = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    raw = self.rfile.read(length).decode("utf-8", "replace")
                    ctype = self.headers.get("Content-Type", "")
                    if "json" in ctype or raw.lstrip().startswith("{"):
                        try:
                            body = json.loads(raw)
                            if isinstance(body, dict):
                                params.update(body)
                        except json.JSONDecodeError:
                            pass
                    else:
                        params.update({k: v[-1] for k, v in parse_qs(raw).items()})
                return params

            def _reply(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self) -> None:
                if urlparse(self.path).path.rstrip("/") not in ("/gps", ""):
                    self._reply(404, {"error": "use /gps"})
                    return
                params = self._params()
                fix = parse_push_fix(params)
                if fix is None:
                    last = source.last_fix
                    self._reply(
                        200,
                        {
                            "received": source.received,
                            "last_fix": None
                            if last is None
                            else {
                                "lat": last.latitude,
                                "lon": last.longitude,
                                "accuracy_m": last.accuracy_m,
                                "age_s": round(time.time() - last.ts_utc, 1),
                            },
                        },
                    )
                    return
                source.last_fix = fix
                source.received += 1
                on_fix(fix)
                self._reply(200, {"ok": True})

            do_GET = _handle
            do_POST = _handle

            def log_message(self, *args: Any) -> None:  # silence per-request logs
                return

        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="push-gps"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def parse_push_fix(params: dict[str, Any]) -> GPSFix | None:
    """Turn GPSLogger-style parameters into a fix. ``None`` if no position.

    GPSLogger leaves a placeholder empty (or literally ``%SAT``) when the value
    is unknown; those become ``None`` so the quality gate does not reject a fix
    for a field the phone never reported.
    """
    norm = {str(k).lower(): v for k, v in params.items()}

    def pick(*names: str) -> float | None:
        for n in names:
            v = norm.get(n)
            if v is None or v == "" or (isinstance(v, str) and v.startswith("%")):
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    lat = pick("lat", "latitude")
    lon = pick("lon", "lng", "longitude")
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0.0 and lon == 0.0:  # "no fix yet" from some apps
        return None

    speed_kmh = pick("speed_kmh", "kmh")
    if speed_kmh is None:
        mps = pick("spd", "speed")
        speed_kmh = mps * 3.6 if mps is not None else 0.0

    sats = pick("sat", "sats", "satellites", "n_sats")
    # 0 satellites means a network (cell/Wi-Fi) fix, not "zero satellites seen".
    n_sats = int(sats) if sats else None

    return GPSFix(
        latitude=lat,
        longitude=lon,
        speed_kmh=speed_kmh,
        heading=pick("dir", "heading", "bearing", "course") or 0.0,
        accuracy_m=pick("acc", "accuracy", "accuracy_m"),
        hdop=pick("hdop"),
        n_sats=n_sats,
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
