"""Frame sources.

The agent must be testable without a phone, so capture is behind one small
interface with three interchangeable implementations:

* :class:`SyntheticSource` -- draws a road scene. Deterministic, no hardware.
* :class:`VideoFileSource` -- reads a recorded clip. Same path a phone feed takes.
* :class:`MjpegSource` -- reads an MJPEG HTTP stream from a phone running
  DroidCam / IP Web Camera over the hotspot.

All sources yield ``(frame, t_mono)``. Downstream code cannot tell them apart,
which is what lets the unit tests exercise the real pipeline.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields BGR frames with a monotonic timestamp."""

    def frames(self) -> Iterator[tuple[np.ndarray, float]]: ...

    def release(self) -> None: ...


class SyntheticSource:
    """Procedurally drawn road scene.

    Deliberately simple: this exists so the pipeline can be tested with no
    hardware, not to simulate detection quality. The legacy prototype used a
    similar canvas but presented it as a demo of live capture, which it was not.
    """

    def __init__(
        self, width: int = 960, height: int = 640, fps: float = 30.0, duration_s: float = 10.0
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.duration_s = duration_s

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        start = time.monotonic()
        frame_index = 0
        while True:
            t = time.monotonic() - start
            if t > self.duration_s:
                return
            frame = np.full((self.height, self.width, 3), 80, dtype=np.uint8)
            cv2.line(
                frame,
                (int(self.width * 0.1), self.height),
                (int(self.width * 0.35), int(self.height * 0.4)),
                (200, 200, 200),
                4,
            )
            cv2.line(
                frame,
                (int(self.width * 0.9), self.height),
                (int(self.width * 0.65), int(self.height * 0.4)),
                (200, 200, 200),
                4,
            )
            for dash_y in range(int(self.height * 0.42), self.height, 45):
                cv2.line(
                    frame,
                    (int(self.width * 0.5), dash_y),
                    (int(self.width * 0.5), dash_y + 25),
                    (255, 230, 0),
                    3,
                )
            # A dark blob travelling up the frame, so consecutive frames differ.
            blob_y = self.height - int((frame_index % 60) / 60 * self.height * 0.5)
            cv2.ellipse(
                frame, (int(self.width * 0.45), blob_y), (50, 30), 15, 0, 360, (25, 25, 25), -1
            )
            yield frame, t
            frame_index += 1
            sleep = 1.0 / self.fps - (time.monotonic() - start - t)
            if sleep > 0:
                time.sleep(sleep)

    def release(self) -> None:
        return


class VideoFileSource:
    """Reads a video file as if it were a live stream."""

    def __init__(self, path: str, fps_override: float | None = None, realtime: bool = True) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.realtime = realtime

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        start = time.monotonic()
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            t = index / self.fps
            yield frame, t
            index += 1
            if self.realtime:
                sleep = 1.0 / self.fps - (time.monotonic() - start)
                if sleep > 0:
                    time.sleep(sleep)

    def release(self) -> None:
        self.cap.release()


class MjpegSource:
    """MJPEG over HTTP, e.g. a phone running DroidCam or IP Web Camera.

    This is the Phase 2 capture path. It needs no app development, which is why
    it is chosen over writing a camera app: the pipeline is proven first, the
    phone app is a later phase.
    """

    def __init__(self, url: str, reconnect_delay_s: float = 2.0) -> None:
        self.url = url
        self.reconnect_delay_s = reconnect_delay_s
        self.cap: cv2.VideoCapture | None = None
        self.reconnects = 0

    def _open(self) -> bool:
        log.info("opening MJPEG stream: %s", self.url)
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            log.error("could not open %s", self.url)
            return False
        self.cap = cap
        return True

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        if self.cap is None and not self._open():
            return
        start = time.monotonic()
        while True:
            ok, frame = self.cap.read()  # type: ignore[union-attr]
            if not ok or frame is None:
                log.warning("stream stalled, reconnecting in %.1fs", self.reconnect_delay_s)
                self.release()
                time.sleep(self.reconnect_delay_s)
                if not self._open():
                    return
                start = time.monotonic()
                continue
            yield frame, time.monotonic() - start

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.reconnects += 1


def local_ipv4_addresses() -> list[str]:
    """Best-effort list of this machine's non-loopback IPv4 addresses.

    Used only to explain a failed camera probe ("phone is on 10.0.0.5, this
    machine is on 172.30.212.1"). Never raises: a diagnostic that fails is
    worse than no diagnostic.
    """
    found: set[str] = set()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connecting a UDP socket sends nothing; it only asks the kernel
            # which local address would be used to reach the internet.
            sock.connect(("8.8.8.8", 80))
            found.add(sock.getsockname()[0])
        finally:
            sock.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in found if _is_reportable(ip))


def _is_reportable(ip: str) -> bool:
    """Drop loopback and documentation addresses from the advice message."""
    if ip.startswith("127.") or ip.startswith("169.254."):
        return False
    # RFC 5737 documentation ranges: never a real interface.
    return not any(ip.startswith(prefix) for prefix in ("192.0.2.", "198.51.100.", "203.0.113."))


def _same_private_prefix(a: str, b: str, octets: int = 2) -> bool:
    """True when two dotted quads agree on the first ``octets`` octets."""
    try:
        pa = [int(part) for part in a.split(".")[:octets]]
        pb = [int(part) for part in b.split(".")[:octets]]
    except ValueError:
        return False
    return len(pa) == octets and pa == pb


def probe_camera(url: str, *, timeout_s: float = 4.0) -> tuple[bool, str]:
    """Check an MJPEG URL is reachable before committing to a run.

    A demo that starts, captures nothing, and exits reporting zeros is worse
    than one that refuses to start and says why. This does the cheap checks:

    1. TCP connect to host:port -- catches wrong network, phone asleep, app
       not running.
    2. A single HTTP GET, reading at most 1 KiB -- catches a wrong path and
       HTML error pages that ``cv2.VideoCapture`` would silently fail on.

    Returns ``(ok, message)``. The message is written for a person standing
    next to a phone, so it names the two addresses when they differ.
    """
    parsed = urlparse(url if "://" in url else f"http://{url}")
    full_url = url if "://" in url else f"http://{url}"
    host = parsed.hostname
    if not host:
        return False, f"could not parse a host from {url!r}"
    scheme = parsed.scheme or "http"
    port = parsed.port or (443 if scheme == "https" else 80)
    target = f"{host}:{port}"

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except ConnectionRefusedError:
        return False, (
            f"{target} refused the connection. The phone is reachable but nothing "
            "is listening: open the camera app, and check the port (DroidCam 4747, "
            "IP Webcam 8080)."
        )
    except TimeoutError:
        hint = _network_hint(host)
        return False, f"{target} did not answer within {timeout_s:.0f}s.{hint}"
    except OSError as exc:
        hint = _network_hint(host)
        return False, f"cannot reach {target} ({exc.strerror or exc}).{hint}"

    try:
        with urlopen(full_url, timeout=timeout_s) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            resp.read(1024)
    except HTTPError as exc:
        return False, (
            f"{target} answered HTTP {exc.code}. The host is right but the path is "
            "wrong: DroidCam uses /video, IP Webcam uses /video (or /videofeed)."
        )
    except (URLError, TimeoutError, OSError) as exc:
        return False, f"{target} did not return a stream ({exc})."

    if "html" in content_type:
        return False, (
            f"{target} returned a web page, not a stream. That is usually the app's "
            "status page: append /video to the URL."
        )
    return True, f"{target} reachable ({content_type or 'stream'})."


def _network_hint(host: str) -> str:
    """Explain a timeout with the addresses that should have matched."""
    local = local_ipv4_addresses()
    if not local:
        return " Check that the phone and this machine are on the same Wi-Fi."
    same = [ip for ip in local if _same_private_prefix(ip, host)]
    if same:
        return (
            f" This machine is {', '.join(local)}, which shares a subnet with {host}, "
            "so the phone is probably asleep, firewalled, or the camera app is not "
            "running."
        )
    return (
        f" This machine is {', '.join(local)}; the phone is {host}. They are on "
        "different networks, so nothing will route between them. Join both to the "
        "same Wi-Fi, or enable the phone's hotspot and join it from this machine."
    )


def build_source(
    camera_url: str | None, *, fallback_video: str | None = None, duration_s: float = 10.0
) -> FrameSource:
    """Choose a capture source.

    Falls back to the synthetic source when no URL is configured, so the agent
    still runs in CI and on a developer machine with no phone attached.
    """
    if camera_url:
        return MjpegSource(camera_url)
    if fallback_video:
        log.info("no camera url; using video file %s", fallback_video)
        return VideoFileSource(fallback_video, realtime=False)
    log.info("no camera url and no video file; using synthetic source")
    return SyntheticSource(duration_s=duration_s)
