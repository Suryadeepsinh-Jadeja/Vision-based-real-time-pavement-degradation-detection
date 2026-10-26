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
from ..geo.gps_sources import CsvGpsSource, HttpGpsSource, PushGpsSource, SyntheticGpsSource
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
    parser.add_argument("--gps-url", help="poll a phone app's GPS HTTP endpoint")
    parser.add_argument(
        "--gps-listen",
        type=int,
        nargs="?",
        const=8765,
        metavar="PORT",
        help="listen for GPS pushed by the phone (GPSLogger custom URL); default port 8765",
    )
    parser.add_argument(
        "--gps-max-accuracy",
        type=float,
        metavar="METRES",
        help="override ROADSCOPE_GPS_MAX_ACCURACY_M (phones indoors report 15-40 m)",
    )
    parser.add_argument(
        "--gps-reuse-last",
        type=float,
        nargs="?",
        const=120.0,
        metavar="MINUTES",
        help="start from the last accepted GPS position if newer than this (default 120 min); "
        "for a stationary demo where the phone's GPS is slow indoors",
    )
    parser.add_argument(
        "--no-geofence",
        action="store_true",
        help="record repeat defects at the same spot (stationary indoor demo only)",
    )
    parser.add_argument("--gps-csv", help="replay a recorded GPS track from CSV")
    parser.add_argument(
        "--gps-synthetic",
        action="store_true",
        help="use a synthetic straight-line GPS track (demo/testing)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="start even if the camera URL does not answer (some cameras only respond as a stream)",
    )
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

    if settings.camera_url:
        from .ingest import probe_camera

        ok, message = probe_camera(settings.camera_url)
        print(f"  camera check      : {'OK' if ok else 'FAILED'} - {message}")
        if not ok:
            return 1

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

    detector = DefectDetector.instance(
        resolved, None if settings.device == "auto" else settings.device
    )
    print(f"  detector          : {detector.engine_name}")
    if detector.available:
        print(f"  model classes     : {detector.names}")
        print(f"  inference device  : {detector.resolve_device()}")
        print(f"  inference size    : {settings.imgsz}px")
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
    if args.gps_max_accuracy is not None:
        settings.gps_max_accuracy_m = args.gps_max_accuracy
    if args.no_geofence:
        # A phone held still reports one position, so every defect after the
        # first falls inside the 15 m revisit radius and is suppressed. A
        # negative radius can never be met by a haversine distance, so nothing
        # is suppressed. Correct for a classroom demo; wrong for a real survey.
        settings.geofence_m = -1.0

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
    from .ingest import probe_camera
    from .pipeline import EdgePipeline

    # Refuse to start on a camera URL that does not answer. Without this the
    # agent runs to completion, captures nothing, and reports all zeros, which
    # looks like a broken pipeline rather than an unreachable phone.
    if settings.camera_url and not getattr(args, "skip_camera_check", False):
        ok, message = probe_camera(settings.camera_url)
        if not ok:
            log.error("camera unreachable: %s", message)
            log.error(
                "refusing to start. Fix the link, or pass --skip-camera-check to "
                "try anyway (the outbox and detector are unaffected)."
            )
            return 2
        log.info("camera check: %s", message)

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
    if args.gps_reuse_last:
        age = pipeline.restore_last_fix(args.gps_reuse_last * 60)
        if age is None:
            log.info("no saved GPS position younger than %.0f min to reuse", args.gps_reuse_last)
        else:
            log.info(
                "starting from the last known GPS position (%.0f min old) until the phone "
                "sends a new one",
                age / 60,
            )
    pipeline.start()

    # Wire a GPS source if one was requested. The pipeline already has
    # ingest_gps(); this thread just feeds fixes into it. Nothing here is
    # blocking -- the source runs on its own thread.
    gps_source = None
    if args.gps_listen:
        from .ingest import local_ipv4_addresses

        gps_source = PushGpsSource(port=args.gps_listen)
        ips = local_ipv4_addresses() or ["<laptop-ip>"]
        log.info("GPS source: listening on port %d for phone pushes", args.gps_listen)
        for ip in ips:
            log.info(
                "  GPSLogger custom URL -> http://%s:%d/gps?lat=%%LAT&lon=%%LON&acc=%%ACC"
                "&spd=%%SPD&dir=%%DIR&sat=%%SAT",
                ip,
                args.gps_listen,
            )
        log.info(
            "  GPS gate: fixes worse than %.0f m are rejected (--gps-max-accuracy to change)",
            settings.gps_max_accuracy_m,
        )
    elif args.gps_url:
        gps_source = HttpGpsSource(args.gps_url)
        log.info("GPS source: HTTP poll %s", args.gps_url)
    elif args.gps_csv:
        gps_source = CsvGpsSource(args.gps_csv)
        log.info("GPS source: CSV replay %s", args.gps_csv)
    elif args.gps_synthetic:
        gps_source = SyntheticGpsSource()
        log.info("GPS source: synthetic straight line")
    if gps_source is not None:
        gps_source.start(pipeline.ingest_gps)

    log.info("running for %.1fs", args.seconds)
    try:
        _time.sleep(args.seconds)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if gps_source is not None:
            gps_source.stop()
            if isinstance(gps_source, PushGpsSource):
                log.info("GPS pushes received from phone: %d", gps_source.received)
                if gps_source.received == 0:
                    log.warning(
                        "no GPS arrived. Is GPSLogger's custom URL enabled, pointed at "
                        "this laptop's IP, and logging every 1 s?"
                    )
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
