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
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="how long to run before stopping"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="skip MQTT; replay the local outbox into the server database",
    )
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

    if args.replay:
        from ..server.subscriber import ingest_from_outbox

        count = ingest_from_outbox(settings.outbox_db, settings, repo=None)  # type: ignore[arg-type]
        log.info("replayed %d message(s) into %s", count, settings.server_db)
        return 0

    return _run(settings, args)


def _run(settings, args) -> int:
    """Run the capture -> infer -> transmit loop."""
    import time as _time

    from ..transport.mqtt_client import MQTTPublisher
    from ..transport.topics import TOPIC_STATUS, topic_for
    from .pipeline import EdgePipeline

    publisher = MQTTPublisher(settings.mqtt_url, client_id=f"roadscope-{settings.device_id}")
    publisher.set_last_will(
        topic_for(settings.mqtt_topic_prefix, settings.device_id, TOPIC_STATUS),
        {"device_id": settings.device_id, "ts_utc": _time.time(), "status": "offline"},
    )
    publisher.connect_async()
    if publisher.wait_connected(5.0):
        log.info("broker reachable at %s", settings.mqtt_url)
    else:
        log.warning(
            "no broker at %s. Capturing anyway; the outbox will buffer until one "
            "appears. Nothing is lost.",
            settings.mqtt_url,
        )

    pipeline = EdgePipeline(settings, publisher=publisher)
    pipeline.start()
    log.info("running for %.1fs", args.seconds)
    try:
        _time.sleep(args.seconds)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        pipeline.stop()
        delivered = pipeline.flush()
        log.info(
            "delivered %d queued message(s); %d still pending",
            delivered,
            pipeline.outbox.depth(),
        )
        publisher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
