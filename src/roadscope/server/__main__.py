"""Receiver process: MQTT in, SQLite out.

The agent publishes detections and GPS fixes to a broker. Something has to
drain that broker into the store the dashboard reads. Until now the only
consumers were the FastAPI app and the ``--replay`` helper, which meant a live
run published into the void unless the ``server`` extra happened to be
installed.

This is that process, with no dependencies beyond the ones the agent already
needs::

    python -m roadscope.server                 # until Ctrl-C
    python -m roadscope.server --seconds 60    # bounded, for a scripted demo

It prints a running count so the operator can see a detection land while the
car is still moving. That is the difference between "the demo is broken" and
"the demo is working and we drove past no potholes".
"""

from __future__ import annotations

import argparse
import time

from ..config import get_settings
from ..logging_setup import get_logger, setup_logging
from ..storage.repository import Repository
from .subscriber import start_receiver

log = get_logger("roadscope.server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m roadscope.server",
        description="Receive agent messages over MQTT and persist them.",
    )
    parser.add_argument("--server-db", help="override ROADSCOPE_SERVER_DB")
    parser.add_argument("--mqtt-url", help="override ROADSCOPE_MQTT_URL")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="stop after this many seconds; 0 means run until interrupted",
    )
    parser.add_argument(
        "--report-every",
        type=float,
        default=5.0,
        help="seconds between throughput lines; 0 disables them",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    if args.server_db:
        settings.server_db = args.server_db
    if args.mqtt_url:
        settings.mqtt_url = args.mqtt_url
    if args.log_level:
        settings.log_level = args.log_level

    setup_logging(settings.log_level, settings.log_file, force=True)
    log.info("receiver starting (mqtt=%s, db=%s)", settings.mqtt_url, settings.server_db)

    repo = Repository(settings.server_db)
    subscriber, receiver = start_receiver(settings, repo)

    if not _wait_connected(subscriber, timeout_s=5.0):
        log.error(
            "no broker at %s; refusing to idle. Start a broker (mosquitto) or pass "
            "--mqtt-url. Without one, the agent buffers in its outbox and can be "
            "replayed later with: python -m roadscope.agent --replay",
            settings.mqtt_url,
        )
        subscriber.close()
        return 2

    started = time.monotonic()
    last_report = started
    last_counts = dict(receiver.counts)
    log.info("receiving; Ctrl-C to stop")

    try:
        while True:
            time.sleep(0.2)
            now = time.monotonic()
            if args.seconds and now - started >= args.seconds:
                break
            if args.report_every and now - last_report >= args.report_every:
                last_report = now
                if receiver.counts != last_counts:
                    last_counts = dict(receiver.counts)
                    log.info(
                        "received: %d gps, %d detections, %d status (%d duplicates, %d errors)",
                        last_counts["gps"],
                        last_counts["detection"],
                        last_counts["status"],
                        last_counts["duplicates"],
                        last_counts["errors"],
                    )
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        subscriber.close()

    final = dict(receiver.counts)
    log.info(
        "stopped: %d gps, %d detections, %d status, %d duplicates, %d errors -> %s",
        final["gps"],
        final["detection"],
        final["status"],
        final["duplicates"],
        final["errors"],
        settings.server_db,
    )
    return 0


def _wait_connected(subscriber, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if subscriber.connected:
            return True
        time.sleep(0.05)
    return subscriber.connected


if __name__ == "__main__":
    raise SystemExit(main())
