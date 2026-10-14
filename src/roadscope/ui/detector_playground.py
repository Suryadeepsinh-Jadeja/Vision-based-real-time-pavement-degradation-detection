"""Detector playground: upload an image or video and see what the model finds.

    streamlit run scripts/detector_playground.py

Separate from the survey dashboard on purpose. The dashboard needs a trip in
the store before it will render anything, which is the wrong entry point for
"does this model see the pothole in this photo" -- the most common question
anyone asks about a detector, and the one that needs no hardware, no GPS and
no broker.

The detection logic lives in :mod:`roadscope.ui.playground` so it can be
tested without a browser.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import streamlit as st

from .. import paths
from ..config import get_settings
from ..ui.playground import (
    CLASS_LABELS,
    DEFAULT_VIDEO_FPS,
    RunResult,
    coverage_ratio,
    describe_run,
    fps_equivalent,
    is_image,
    is_video,
    load_upload,
    run_image,
    run_video,
    safe_conf,
    video_metadata,
)
from ..vision.detector import DefectDetector


@st.cache_resource(show_spinner=False)
def get_detector() -> DefectDetector | None:
    settings = get_settings()
    try:
        path = settings.resolved_model_path()
    except FileNotFoundError:
        return None
    if path is None:
        return None
    detector = DefectDetector.instance(path)
    detector._load()
    return detector if detector.available else None


def main() -> None:
    st.set_page_config(page_title="RoadScope Detector", page_icon="🔍", layout="wide")
    st.title("Detector playground")
    st.caption(
        "Upload a photo or a dashcam clip. The model is the RDD2022 detector "
        "used by the live agent -- the same weights, the same classes."
    )

    detector = get_detector()
    if detector is None:
        st.error(
            "No weights available. Fetch them first:\n\n"
            "```\nbash scripts/download_weights.sh --verify\n```"
        )
        return

    with st.sidebar:
        st.markdown("**Model**")
        st.write(f"`{detector.model_path.name}`")
        st.write(f"device: {detector.resolve_device()}")
        st.write(f"classes: {len(detector.names)}")
        st.caption(
            "These were trained on RDD2022 (India, Japan, US). Scores here are "
            "model output on your file, not a validated accuracy figure."
        )

        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.35, 0.05)
        imgsz = st.select_slider(
            "Inference resolution",
            options=[640, 768, 960, 1280],
            value=960,
            help="960 is what the model was trained at. Higher can help small "
            "targets but costs time and will not help on an image where the "
            "damage is genuinely too small to resolve.",
        )

    upload = st.file_uploader(
        "Image or video",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "avi", "mkv", "m4v", "webm"],
        help="Photos work best. For video, frames are sampled rather than processed exhaustively.",
    )

    if upload is None:
        st.info("Choose a file to run detection.")
        _render_legend()
        return

    # Uploads land beside the project's other runtime data. ``data_dir`` is a
    # function on ``roadscope.paths``, not a Settings field -- Settings only
    # carries the individual paths (outbox_db, server_db, thumb_dir).
    upload_dir = paths.data_dir() / "uploads"
    path = load_upload(upload.name, upload.getvalue(), upload_dir)

    if is_video(path):
        _run_video(detector, path, conf, imgsz)
    elif is_image(path):
        _run_single_image(detector, path, conf, imgsz)
    else:
        st.error(f"Unrecognised file type: {upload.name}")
        return

    _render_legend()


def _run_single_image(detector: DefectDetector, path, conf: float, imgsz: int) -> None:
    result: RunResult = run_image(detector, path, conf=safe_conf(conf), imgsz=imgsz)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Defects found", f"{len(result.unique)}")
    c2.metric("Detection time", f"{result.elapsed_ms:.0f} ms")
    c3.metric("Resolution", f"{imgsz} px")
    c4.metric("Classes", len(detector.names))

    for warning in result.warnings:
        st.warning(warning)

    if result.frames and result.frames[0].annotated is not None:
        st.image(
            result.frames[0].annotated,
            caption=f"{path.name} — {path.stat().st_size / 1024:.0f} KB",
            use_container_width=True,
        )

    if result.unique:
        st.subheader("What was found")
        st.dataframe(
            [
                {
                    "Code": d["class_code"],
                    "Type": CLASS_LABELS.get(d["class_code"], "Distress"),
                    "Confidence": f"{d['confidence']:.0%}",
                    "Box (px)": f"{d['width_px']} x {d['height_px']}",
                    "Share of frame": f"{d['area_frac'] * 100:.2f}%",
                }
                for d in result.unique
            ],
            width="stretch",
            hide_index=True,
        )
        _show_measurements_caveat(result.unique)
    else:
        _explain_miss(path)


def _run_video(detector: DefectDetector, path, conf: float, imgsz: int) -> None:
    meta = video_metadata(path)
    sample_fps = st.sidebar.slider(
        "Sample rate (fps)",
        0.5,
        15.0,
        DEFAULT_VIDEO_FPS,
        0.5,
        help="Processing every frame of a 30 fps clip is slow. This is a skim, "
        "so a fast-moving defect between samples can be missed.",
    )

    st.caption(
        f"{meta['width']}x{meta['height']}, {meta['fps']:g} fps, "
        f"{meta['frames']} frames" + (f", {meta['duration_s']:g} s" if meta["duration_s"] else "")
    )

    if st.button("Run detection", type="primary"):
        progress = st.progress(0.0, text="Detecting…")
        started = time.perf_counter()
        result = run_video(detector, path, conf=safe_conf(conf), imgsz=imgsz, sample_fps=sample_fps)
        elapsed = time.perf_counter() - started
        progress.progress(1.0, text=f"Done in {elapsed:.1f} s")
        # Held in session state because every slider drag reruns the whole
        # script. Without this the frame browser would vanish the moment it was
        # used, and detection would re-run from the video on each frame.
        st.session_state["_pg_result"] = result
        st.session_state["_pg_source"] = str(path)
        # Drop a stale frame position from a previous, possibly longer clip.
        st.session_state.pop("_pg_pos", None)

    # Show the stored result on reruns, but only for the same file -- otherwise
    # switching uploads would show the previous clip's frames.
    if st.session_state.get("_pg_source") == str(path) and "_pg_result" in st.session_state:
        _render_video_result(st.session_state["_pg_result"], meta)
    else:
        st.info(
            "Press **Run detection** to process the sampled frames. Results are "
            "kept while you step through them."
        )

    st.caption(
        f"Approximate cost at {imgsz} px on this machine: "
        f"{1000 / max(1, _ms_per_frame(imgsz)):.1f} fps."
    )


def _ms_per_frame(imgsz: int) -> float:
    """Rough per-frame cost, scaled from the recorded 960 px figure."""
    return 125.0 * (imgsz / 960) ** 2


def _render_frame_browser(result: RunResult) -> None:
    """Step through sampled frames with a slider instead of scrolling images.

    Showing every annotated frame stacked meant a long scroll for even a short
    clip, and the frames worth looking at -- the ones with a detection -- were
    buried. This presents one frame at a time and lets you jump to the next one
    that has a detection, which is what you actually want when checking whether
    a box is real.
    """
    frames = result.frames
    if not frames:
        return

    hit_indices = [i for i, f in enumerate(frames) if f.count]

    st.subheader("Frames")
    if hit_indices:
        st.caption(
            f"{len(hit_indices)} of {len(frames)} frames "
            f"({coverage_ratio(result):.0f}%) produced at least one box."
        )
    else:
        st.caption(f"{len(frames)} frames processed. No frame produced a detection.")

    # Default to the first frame that has a detection -- that is the frame the
    # user came here to look at. Reset when a new run lands fewer frames than
    # the stored position, so a shorter clip cannot leave the slider out of range.
    last = len(frames) - 1
    if st.session_state.get("_pg_pos") is None or st.session_state["_pg_pos"] > last:
        st.session_state["_pg_pos"] = hit_indices[0] if hit_indices else 0

    # No `value=` here: with a key, Streamlit takes the position from session
    # state, which is what lets the buttons below move it.
    pos = st.slider(
        "Frame",
        min_value=0,
        max_value=last,
        step=1,
        key="_pg_pos",
        help="Sampled frames in order. Use the buttons below to jump between "
        "frames that have detections.",
    )

    frame = frames[pos]
    if frame.annotated is not None:
        st.image(
            frame.annotated,
            caption=f"frame {frame.index} · {frame.timestamp_s:.1f}s · {frame.count} detection(s)",
            use_container_width=True,
        )
    else:
        st.caption("Frame could not be annotated.")

    _render_frame_nav(pos, hit_indices, last)

    if frame.detections:
        st.dataframe(
            [
                {
                    "Code": d["class_code"],
                    "Type": CLASS_LABELS.get(d["class_code"], "Distress"),
                    "Confidence": f"{d['confidence']:.0%}",
                    "Box (px)": f"{d['width_px']} x {d['height_px']}",
                    "Share of frame": f"{d['area_frac'] * 100:.2f}%",
                }
                for d in frame.detections
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No detection on this frame.")


def _set_pos(value: int) -> None:
    """Callback for the navigation buttons.

    Callbacks run before the script reruns, so assigning to the slider's key
    here is legal. Writing to it after the slider is drawn raises
    ``StreamlitAPIException: st.session_state._pg_pos cannot be modified after
    the widget with key _pg_pos is instantiated`` -- which is why these are
    callbacks rather than ``if st.button(...):`` blocks below the slider.
    """
    st.session_state["_pg_pos"] = value


def _render_frame_nav(pos: int, hit_indices: list[int], last: int) -> None:
    """Jump buttons for first/last and previous/next frame with a detection."""
    prev_hit = max((i for i in hit_indices if i < pos), default=None)
    next_hit = min((i for i in hit_indices if i > pos), default=None)

    cols = st.columns(4)
    cols[0].button(
        "⏮ First",
        disabled=pos == 0,
        use_container_width=True,
        on_click=_set_pos,
        args=(0,),
    )
    cols[1].button(
        "◀ Previous hit",
        disabled=prev_hit is None,
        use_container_width=True,
        on_click=_set_pos,
        args=(prev_hit,),
    )
    cols[2].button(
        "Next hit ▶",
        disabled=next_hit is None,
        use_container_width=True,
        on_click=_set_pos,
        args=(next_hit,),
    )
    cols[3].button(
        "⏭ Last",
        disabled=pos == last,
        use_container_width=True,
        on_click=_set_pos,
        args=(last,),
    )


def _render_video_result(result: RunResult, meta: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique defects", f"{len(result.unique)}")
    c2.metric("Frames processed", f"{result.total_frames}")
    c3.metric("Total time", f"{result.elapsed_ms / 1000:.1f} s")
    c4.metric("Rate", f"{fps_equivalent(result):.1f} fps")

    for warning in result.warnings:
        st.info(warning)

    if not result.unique:
        st.warning(
            "No defect survived tracking. Either there is nothing there, or "
            "nothing persisted across enough consecutive frames to confirm."
        )

    if result.counts:
        st.subheader("Detected classes")
        st.bar_chart({CLASS_LABELS.get(k, k): v for k, v in sorted(result.counts.items())})

    if result.frames:
        _render_frame_browser(result)

    if result.unique:
        st.subheader("Unique defects")
        st.dataframe(
            [
                {
                    "Code": d["class_code"],
                    "Type": CLASS_LABELS.get(d["class_code"], "Distress"),
                    "Confidence": f"{d['confidence']:.0%}",
                    "Best frame": d["first_frame"],
                    "Frames seen": d["frames_seen"],
                    "Tracked (s)": d["duration_s"],
                    "Box (px)": f"{d['width_px']} x {d['height_px']}",
                }
                for d in result.unique
            ],
            width="stretch",
            hide_index=True,
        )
        _show_measurements_caveat(result.unique)

        st.download_button(
            "Download detections (JSON)",
            _to_json(result),
            file_name="detections.json",
            mime="application/json",
        )
        st.caption(describe_run(result))


def _show_measurements_caveat(detections: list[dict]) -> None:
    st.info(
        "These boxes are what the model predicts. They are **not** measurements: "
        "the model was trained on bounding boxes only, so crack width and pothole "
        "depth cannot be derived from them, and the PCI engine falls back to "
        "defaulted severity for those."
    )


def _explain_miss(path) -> None:
    st.warning("Nothing was detected in this image.")
    st.markdown(
        """
Before concluding the road is fine, check:

- **Is the damage large enough in frame?** A pothole covering less than about
  0.5% of the frame is at the edge of what this model handles. The pothole
  class was its weakest in validation (AP50 0.538).
- **Try a lower confidence threshold.** Some of the output is below 0.35.
- **Try a higher resolution.** 1280 can recover small targets that 960 misses.
- **Is it road?** The model was trained on dashcam and phone footage from
  India, Japan and the US, not satellite or handheld imagery.
"""
    )


def _render_legend() -> None:
    st.divider()
    st.subheader("Classes")
    cols = st.columns(4)
    for i, (code, label) in enumerate(CLASS_LABELS.items()):
        cols[i % 4].metric(f"{code}  {label}", "")


def _to_json(result: RunResult) -> str:
    import json

    return json.dumps(
        {
            "source": result.source,
            "run_at": datetime.now(UTC).isoformat(),
            "settings": {
                "imgsz": result.imgsz,
                "conf": result.conf,
                "device": result.device,
            },
            "summary": result.summary(),
            "unique": result.unique,
            "frames": [
                {
                    "index": f.index,
                    "timestamp_s": round(f.timestamp_s, 3),
                    "detections": [
                        {
                            "class_code": d["class_code"],
                            "confidence": round(d["confidence"], 4),
                            "bbox": list(d["bbox"]),
                        }
                        for d in f.detections
                    ],
                }
                for f in result.frames
                if f.detections
            ],
        },
        indent=2,
    )


main()
