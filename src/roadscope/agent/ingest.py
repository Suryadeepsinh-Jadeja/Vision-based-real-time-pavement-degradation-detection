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

import time
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

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
