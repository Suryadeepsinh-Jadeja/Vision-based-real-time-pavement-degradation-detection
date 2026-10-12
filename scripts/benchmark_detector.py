#!/usr/bin/env python3
"""Measure detector latency and throughput at several inference resolutions.

Why this exists
---------------
``config.Settings.imgsz`` defaults to 640 because, at the time it was written,
960 halved throughput on this machine. That reasoning predates the RDD2022
detector. Run ``r1`` was trained at imgsz=960, and ``docs/results.md`` shows the
pothole class is localisation-limited -- which is the one failure mode higher
resolution is supposed to fix. So the tradeoff changed and the default did not.

What it measures
----------------
Latency (ms/frame), throughput (fps) and detections-per-frame at each
resolution, plus the resulting frame-drop rate against a 30 fps capture.

Latency is measured on frames the detector is actually asked to work on. If no
image directory is given it falls back to the synthetic road scene, which has
almost no texture and therefore *understates* cost relative to real footage --
the numbers stay comparable across resolutions but read optimistically. Pass
real frames for numbers worth quoting.

Usage
-----
    python scripts/benchmark_detector.py --frames 40
    python scripts/benchmark_detector.py --images path/to/frames --frames 40
    python scripts/benchmark_detector.py --images frames --sizes 640 960 1280
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roadscope.vision.detector import DefectDetector

CAPTURE_FPS = 30.0


def load_frames(images_dir: Path | None, count: int) -> tuple[list, str]:
    """Return frames plus a label describing where they came from."""
    if images_dir is not None:
        paths = sorted(
            p for ext in ("jpg", "jpeg", "png", "bmp") for p in images_dir.glob(f"*.{ext}")
        )
        if not paths:
            raise SystemExit(f"no images found in {images_dir}")
        frames = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        if not frames:
            raise SystemExit(f"none of the files in {images_dir} could be read as images")
        return frames[:count], f"{images_dir} ({len(frames)} available)"

    from roadscope.agent.ingest import SyntheticSource

    # duration_s is derived from the requested count rather than fixed, so a
    # caller asking for 40 frames actually gets 40. SyntheticSource paces
    # itself in real time, so this also costs count/fps seconds of wall clock.
    src = SyntheticSource(fps=60.0, duration_s=count / 60.0 + 1.0)
    frames = [f for f, _ in src.frames()]
    src.release()
    if len(frames) < count:
        raise SystemExit(f"synthetic source yielded {len(frames)} frames, wanted {count}")
    return frames[:count], "synthetic road scene (no texture; costs read optimistically)"


def bench(detector: DefectDetector, frames: list, imgsz: int, conf: float, warmup: int) -> dict:
    for f in frames[:warmup]:
        detector.infer(f, conf=conf, imgsz=imgsz)

    timings: list[float] = []
    n_det = 0
    for f in frames:
        t0 = time.perf_counter()
        dets = detector.infer(f, conf=conf, imgsz=imgsz)
        timings.append((time.perf_counter() - t0) * 1000.0)
        n_det += len(dets)

    median = statistics.median(timings)
    return {
        "median_ms": median,
        "mean_ms": statistics.mean(timings),
        "p95_ms": sorted(timings)[int(0.95 * (len(timings) - 1))],
        "min_ms": min(timings),
        "fps": 1000.0 / median,
        "det_per_frame": n_det / len(frames),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", default="models/rdd2022-yolov8m-r1.pt")
    ap.add_argument(
        "--images", type=Path, help="directory of real frames; omit to use the synthetic scene"
    )
    ap.add_argument("--frames", type=int, default=30, help="frames to time (default: 30)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 768, 960, 1280])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument(
        "--device",
        default="auto",
        help="cpu, mps, 0 (cuda), or auto (default: MPS on Apple Silicon, else CPU)",
    )
    args = ap.parse_args()

    import torch

    print(
        f"torch {torch.__version__}, threads={torch.get_num_threads()}, "
        f"capture={CAPTURE_FPS:.0f} fps"
    )

    frames, src = load_frames(args.images, args.frames)
    h, w = frames[0].shape[:2]
    print(f"frames: {len(frames)} at {w}x{h}")
    print(f"source: {src}\n")

    detector = DefectDetector.instance(
        Path(args.weights), None if args.device == "auto" else args.device
    )
    detector._load()
    if detector.model is None:
        raise SystemExit(f"could not load weights from {args.weights}")
    print(
        f"weights: {detector.model_path.name}, {len(detector.names)} classes, "
        f"device={detector.resolve_device()}\n"
    )

    print(
        f"{'imgsz':>6} {'median':>9} {'mean':>9} {'p95':>9} {'min':>9} "
        f"{'fps':>7} {'dets/f':>7} {'drop%':>7}"
    )
    print("-" * 72)
    rows = []
    for imgsz in args.sizes:
        r = bench(detector, frames, imgsz, args.conf, args.warmup)
        drop = max(0.0, (1.0 - r["fps"] / CAPTURE_FPS)) * 100.0
        rows.append((imgsz, r, drop))
        print(
            f"{imgsz:>6} {r['median_ms']:>8.1f}m {r['mean_ms']:>8.1f}m "
            f"{r['p95_ms']:>8.1f}m {r['min_ms']:>8.1f}m {r['fps']:>6.1f} "
            f"{r['det_per_frame']:>7.2f} {drop:>6.1f}%"
        )

    fastest = max(rows, key=lambda r: r[1]["fps"])
    best_yield = max(rows, key=lambda r: r[1]["det_per_frame"])
    print()
    print(f"fastest        : imgsz={fastest[0]} at {fastest[1]['fps']:.1f} fps")
    print(
        f"most detections: imgsz={best_yield[0]} at {best_yield[1]['det_per_frame']:.2f} det/frame"
    )

    if args.images is None:
        print()
        print("NOTE: synthetic frames carry almost no texture. Re-run with --images")
        print("      pointing at real footage before quoting these numbers.")

    # At higher resolution each inference costs more, so the sample rate the
    # agent can sustain falls. The queue absorbs the gap by dropping.
    print()
    print("sustained sample rate if the agent keeps up (no drops below 30 fps):")
    for imgsz, r, _ in rows:
        print(f"  imgsz={imgsz:<5} {min(r['fps'], CAPTURE_FPS):>5.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
