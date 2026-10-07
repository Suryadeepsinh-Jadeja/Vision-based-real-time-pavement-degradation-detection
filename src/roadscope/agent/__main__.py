"""Edge agent entry point.

Usage::

    python -m roadscope.agent --help
    python -m roadscope.agent --check      # diagnose configuration and weights

The agent runs standalone. It must never import Streamlit: the UI is a read-only
client of the database, so the capture and inference loop must stay usable when
the dashboard is not running.
"""

from __future__ import annotations

import argparse
import sys

from .. import paths
from ..config import get_settings
from ..logging_setup import get_logger, setup_logging
from ..vision.detector import DefectDetector

log = get_logger("roadscope.agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m roadscope.agent",
        description="RoadScope edge agent: capture, detect, geotag, transmit.",
    )
    parser.add_argument("--check", action="store_true", help="run diagnostics and exit")
    parser.add_argument("--device-id", help="override ROADSCOPE_DEVICE_ID")
    parser.add_argument("--camera-url", help="override ROADSCOPE_CAMERA_URL")
    parser.add_argument("--model-path", help="override ROADSCOPE_MODEL_PATH")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return parser


def _diagnose(settings) -> int:
    """Print configuration diagnostics. Returns a process exit code."""
    print(f"RoadScope {__import__('roadscope').__version__}")
    print(f"  project root      : {paths.project_root()}")
    print(f"  device id         : {settings.device_id}")
    print(f"  mqtt broker       : {settings.mqtt_url}")
    print(f"  camera url        : {settings.camera_url or '(none - synthetic source)'}")

    present = paths.available_models()
    if present:
        print(f"  weights found     : {', '.join(p.name for p in present)}")
    else:
        print(f"  weights found     : NONE in {paths.models_dir()}")

    try:
        resolved = settings.resolved_model_path()
    except FileNotFoundError as exc:
        print(f"  ERROR             : {exc}")
        return 1

    if resolved is None:
        print("  detector          : UNAVAILABLE (no weights)")
        return 1

    detector = DefectDetector.instance(resolved)
    print(f"  detector          : {detector.engine_name}")
    if detector.available:
        print(f"  model classes     : {detector.names}")
    else:
        print("  detector          : FAILED TO LOAD (is the 'yolo' extra installed?)")
        return 1

    for label, db in (("outbox db", settings.outbox_db), ("server db", settings.server_db)):
        print(f"  {label:<18}: {db}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    if args.device_id:
        settings.device_id = args.device_id
    if args.camera_url:
        settings.camera_url = args.camera_url
    if args.model_path:
        settings.model_path = args.model_path
    if args.log_level:
        settings.log_level = args.log_level

    setup_logging(settings.log_level, settings.log_file, force=True)
    log.info("RoadScope agent starting (device=%s)", settings.device_id)

    if args.check:
        return _diagnose(settings)

    log.warning(
        "Capture pipeline is not implemented yet. That is Phase 2 of the rebuild "
        "(see REBUILD_PLAN.md). Use --check to verify configuration."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
