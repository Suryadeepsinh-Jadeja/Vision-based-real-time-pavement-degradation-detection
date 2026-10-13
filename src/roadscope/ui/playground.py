"""Run the detector over images or video and summarise what it found.

Kept separate from the Streamlit code so the detection behaviour can be tested
without a browser, matching the reasoning that keeps Streamlit out of the agent.

The tracker runs here too, so a video of one pothole passed for several seconds
reports one defect rather than one per frame. That is the same fusion the live
agent does, which matters: a playground that counted every frame would report
numbers the deployed system never produces.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..vision.detector import DefectDetector, Detection

#: Frames sampled from a video when the user does not ask for every one.
DEFAULT_VIDEO_FPS = 5.0

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


@dataclass
class FrameResult:
    """One processed frame."""

    index: int
    timestamp_s: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    annotated: np.ndarray | None = None
    width: int = 0
    height: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass
class RunResult:
    """Everything a run produced."""

    frames: list[FrameResult] = field(default_factory=list)
    #: Unique defects after tracking -- the headline number.
    unique: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    device: str = "cpu"
    imgsz: int = 960
    conf: float = 0.35
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def total_detections(self) -> int:
        return sum(f.count for f in self.frames)

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.total_frames,
            "raw_detections": self.total_detections,
            "unique_defects": len(self.unique),
            "counts": dict(self.counts),
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "ms_per_frame": round(self.elapsed_ms / self.total_frames, 1)
            if self.total_frames
            else 0.0,
        }


def to_detection_dict(det: Detection, frame_w: int, frame_h: int) -> dict[str, Any]:
    """Flatten a detection for display, with the box normalised to 0-1.

    Percentages rather than pixels so a marker lines up with the displayed
    image regardless of the size it was rendered at.
    """
    x1, y1, x2, y2 = det.bbox
    return {
        "class_code": det.class_code,
        "confidence": det.confidence,
        "bbox": det.bbox,
        "box_norm": (
            max(0.0, x1 / frame_w),
            max(0.0, y1 / frame_h),
            min(1.0, x2 / frame_w),
            min(1.0, y2 / frame_h),
        ),
        "width_px": x2 - x1,
        "height_px": y2 - y1,
        "area_frac": ((x2 - x1) * (y2 - y1)) / max(1, frame_w * frame_h),
    }


def annotate(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw labelled boxes on a copy of the frame."""
    out = frame.copy()
    thickness = max(1, round(out.shape[1] / 500))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        code = det["class_code"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(code, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"{code} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Keep the label inside the frame when the box sits at the top edge.
        ty = max(y1, y1 - th - 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, ty - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


#: BGR, matching the detector's colour space.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "D00": (235, 99, 37),  # blue  - longitudinal crack
    "D10": (150, 58, 235),  # purple - transverse crack
    "D20": (178, 145, 8),  # teal - alligator cracking
    "D40": (38, 38, 220),  # red   - pothole
}

CLASS_LABELS: dict[str, str] = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator cracking",
    "D40": "Pothole",
}


def iter_video_frames(
    path: str | Path, *, sample_fps: float | None = None, max_frames: int = 400
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(index, timestamp_s, frame)``, sampling if asked.

    Video at 30 fps for a minute is 1,800 frames and would take several minutes
    on CPU. Sampling is the default and the rate is reported, so a skim is
    never mistaken for full-frame coverage.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps)) if sample_fps else 1

    index = 0
    emitted = 0
    try:
        while emitted < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index % step == 0:
                yield index, index / src_fps, frame
                emitted += 1
            index += 1
    finally:
        cap.release()


def video_metadata(path: str | Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "fps": round(fps, 2),
            "frames": frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": round(frames / fps, 1) if fps else None,
        }
    finally:
        cap.release()


def run_image(
    detector: DefectDetector,
    path: str | Path,
    *,
    conf: float = 0.35,
    imgsz: int = 960,
    annotate_result: bool = True,
) -> RunResult:
    """Detect on a single image."""
    import time

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    h, w = image.shape[:2]

    t0 = time.perf_counter()
    raw = detector.infer(image, conf=conf, imgsz=imgsz)
    elapsed = (time.perf_counter() - t0) * 1000

    detections = [to_detection_dict(d, w, h) for d in raw]
    result = RunResult(
        frames=[
            FrameResult(
                index=0,
                timestamp_s=0.0,
                detections=detections,
                annotated=annotate(image, detections) if annotate_result else None,
                width=w,
                height=h,
            )
        ],
        unique=detections,
        counts=_counts(detections),
        elapsed_ms=elapsed,
        device=detector.resolve_device(),
        imgsz=imgsz,
        conf=conf,
        source=str(path),
    )
    if not detections:
        result.warnings.append(
            "Nothing detected. If this frame clearly shows damage, the detector "
            "may be wrong -- check the confidence threshold, or try a higher "
            "resolution (the model was trained at 960)."
        )
    return result


def run_video(
    detector: DefectDetector,
    path: str | Path,
    *,
    conf: float = 0.35,
    imgsz: int = 960,
    sample_fps: float | None = DEFAULT_VIDEO_FPS,
    max_frames: int = 400,
    annotate_result: bool = True,
    tracker: Any = None,
) -> RunResult:
    """Detect across a video, collapsing repeat sightings into unique defects."""
    import time

    from ..fusion.tracker import ByteTracker

    tracker = tracker or ByteTracker()

    result = RunResult(device=detector.resolve_device(), imgsz=imgsz, conf=conf, source=str(path))
    meta = video_metadata(path)
    if sample_fps and meta["fps"] and sample_fps < meta["fps"]:
        result.warnings.append(
            f"Sampling at {sample_fps:g} fps of {meta['fps']:g} -- this is a skim, not every frame."
        )

    started = time.perf_counter()
    for index, ts, frame in iter_video_frames(path, sample_fps=sample_fps, max_frames=max_frames):
        h, w = frame.shape[:2]
        raw = detector.infer(frame, conf=conf, imgsz=imgsz)
        detections = [to_detection_dict(d, w, h) for d in raw]

        # Feed the same tracker the live agent uses, so one pothole held in
        # view for a second reports once rather than once per frame.
        position = None
        # ``update`` returns every track newly *confirmed* by this frame, so a
        # track is returned exactly once in its lifetime -- the first time it
        # reaches min_hits. That is the de-duplication. DefectFuser guards with
        # ``track.emitted`` on top of this because it also has to suppress a
        # re-confirmation after a track expires; here each track_id is counted
        # once regardless, so the count is per physical defect.
        for track in tracker.update(raw, ts, frame_index=index, position=position):
            if track.best_bbox is None:
                continue
            result.unique.append(
                {
                    "class_code": track.class_code,
                    "confidence": track.best_confidence,
                    "first_frame": track.best_frame_index,
                    "frames_seen": track.hits,
                    "duration_s": round(track.last_seen - track.first_seen, 2),
                    "bbox": track.best_bbox,
                    "width_px": track.best_bbox[2] - track.best_bbox[0],
                    "height_px": track.best_bbox[3] - track.best_bbox[1],
                }
            )

        result.frames.append(
            FrameResult(
                index=index,
                timestamp_s=ts,
                detections=detections,
                annotated=annotate(frame, detections) if annotate_result else None,
                width=w,
                height=h,
            )
        )

    result.elapsed_ms = (time.perf_counter() - started) * 1000
    result.counts = _counts(result.unique)
    return result


def _counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        code = det.get("class_code", "?")
        counts[code] = counts.get(code, 0) + 1
    return counts


def load_upload(name: str, data: bytes, workdir: Path) -> Path:
    """Write an upload to disk and return the path.

    Streamlit's file uploader hands over bytes, not a path, and OpenCV needs a
    real file to decode. The name is sanitised first: an upload name is
    attacker-controlled and must never be used to build a path directly.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(name).suffix.lower()[:10]
    stem = "".join(c for c in Path(name).stem if c.isalnum() or c in "-_")[:60] or "upload"
    target = workdir / f"{stem}{suffix}"
    target.write_bytes(data)
    return target


def mean_area_fraction(detections: list[dict[str, Any]]) -> float:
    """Average share of the frame a detection covers, as a percentage."""
    if not detections:
        return 0.0
    return sum(d["area_frac"] for d in detections) / len(detections) * 100


def smallest_box_px(detections: list[dict[str, Any]]) -> tuple[int, int]:
    """Smallest detected box, or (0, 0).

    Small boxes are where the detector is weakest -- a pothole covering 0.4% of
    the frame is easy to miss -- so this is the useful thing to surface when
    someone uploads a distant shot and gets nothing.
    """
    if not detections:
        return (0, 0)
    smallest = min(detections, key=lambda d: d["width_px"] * d["height_px"])
    return (smallest["width_px"], smallest["height_px"])


def median_box_px(detections: list[dict[str, Any]]) -> tuple[int, int]:
    if not detections:
        return (0, 0)
    widths = sorted(d["width_px"] for d in detections)
    heights = sorted(d["height_px"] for d in detections)
    mid = len(widths) // 2
    return (widths[mid], heights[mid])


def is_probable_small_target(detections: list[dict[str, Any]], threshold_pct: float = 1.0) -> bool:
    """True when every detection is tiny in frame.

    Used to warn that a miss is plausible rather than certain.
    """
    if not detections:
        return False
    return all(d["area_frac"] * 100 < threshold_pct for d in detections)


def coverage_ratio(result: RunResult) -> float:
    """Share of processed frames that produced at least one detection."""
    if not result.frames:
        return 0.0
    with_det = sum(1 for f in result.frames if f.count)
    return with_det / len(result.frames) * 100


def fps_equivalent(result: RunResult) -> float:
    """Processing rate, so a user can judge whether this could run live."""
    if result.elapsed_ms <= 0 or not result.total_frames:
        return 0.0
    return result.total_frames / (result.elapsed_ms / 1000)


def describe_detection(det: dict[str, Any]) -> str:
    code = det.get("class_code", "?")
    label = CLASS_LABELS.get(code, "Distress")
    conf = float(det.get("confidence") or 0)
    size = f"{det.get('width_px', 0)}x{det.get('height_px', 0)}px"
    return f"{code} {label} {conf:.0%} ({size})"


def estimate_stretch_factor(
    video: dict[str, Any], detections: list[dict[str, Any]]
) -> float | None:
    """Rough hint that distant shots need a higher inference resolution.

    Returns a suggested imgsz when the median detection is very small. This is
    a heuristic for the UI hint only -- it is not a measurement, and it does
    not predict accuracy.
    """
    if not detections:
        return None
    mw, mh = median_box_px(detections)
    if mw <= 0 or mh <= 0:
        return None
    median_frac = ((mw * mh) / max(1, video.get("width", 1) * video.get("height", 1))) * 100
    if median_frac < 0.15:
        return 1280
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_conf(value: float) -> float:
    """Clamp a confidence threshold into a range ultralytics accepts."""
    return clamp(float(value), 0.01, 0.99)


def describe_run(result: RunResult) -> str:
    """One-line summary for the UI and for logging."""
    parts = [
        f"{result.total_frames} frame(s)",
        f"{len(result.unique)} unique defect(s)",
    ]
    if result.counts:
        parts.append(
            ", ".join(f"{CLASS_LABELS.get(k, k)} {v}" for k, v in sorted(result.counts.items()))
        )
    parts.append(f"{result.elapsed_ms:.0f} ms on {result.device}")
    return " | ".join(parts)


def nan_guard(value: float, fallback: float = 0.0) -> float:
    """Drop NaN before it reaches a st.metric, which renders 'nan'."""
    return fallback if value is None or math.isnan(value) else value
