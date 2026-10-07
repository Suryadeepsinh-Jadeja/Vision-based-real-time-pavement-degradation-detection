"""MQTT to database ingest.

Runs on the receiving side. Every write is idempotent on a natural key, so a
redelivered packet cannot duplicate a row -- that is what makes the agent's
at-least-once outbox safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from ..storage.repository import DeviceStatus, Repository, new_trip_id
from ..transport.mqtt_client import MQTTSubscriber
from ..transport.topics import TOPIC_DETECTION, TOPIC_GPS, TOPIC_STATUS

log = get_logger(__name__)


class Receiver:
    """Persists agent messages into the server store."""

    def __init__(self, settings: Settings, repo: Repository | None = None) -> None:
        self.settings = settings
        self.repo = repo or Repository(settings.server_db)
        self.trips: dict[str, str] = {}
        self.counts = {"gps": 0, "detection": 0, "status": 0, "duplicates": 0, "errors": 0}

    def trip_for(self, device_id: str, payload: dict[str, Any]) -> str:
        """Resolve the trip, creating one on first contact."""
        declared = payload.get("trip_id")
        if declared:
            self.trips[device_id] = declared
            self._ensure_trip(declared, device_id)
            return declared
        if device_id not in self.trips:
            trip_id = new_trip_id(device_id)
            self.trips[device_id] = trip_id
            self._ensure_trip(trip_id, device_id)
            log.info("opened trip %s for %s", trip_id, device_id)
        return self.trips[device_id]

    def _ensure_trip(self, trip_id: str, device_id: str) -> None:
        """Register the device and trip if absent.

        ``trips.device_id`` is a foreign key, so the device row must exist
        before the trip. The agent sends only the trip_id, so the receiver
        creates both -- otherwise the first GPS packet of every run is dropped
        on a constraint violation.
        """
        self.repo.upsert_device(device_id)
        self.repo.start_trip(trip_id, device_id)

    def handle(self, channel: str, payload: dict[str, Any]) -> None:
        device_id = str(payload.get("device_id", "unknown"))
        try:
            if channel == TOPIC_GPS:
                self._gps(device_id, payload)
            elif channel == TOPIC_DETECTION:
                self._detection(device_id, payload)
            elif channel == TOPIC_STATUS:
                self._status(device_id, payload)
            else:
                log.debug("ignoring channel %s", channel)
        except Exception as exc:
            self.counts["errors"] += 1
            log.exception("failed handling %s from %s: %s", channel, device_id, exc)

    def _gps(self, device_id: str, payload: dict[str, Any]) -> None:
        trip_id = self.trip_for(device_id, payload)
        inserted = self.repo.insert_fix(trip_id, payload)
        self.counts["gps" if inserted else "duplicates"] += 1

    def _detection(self, device_id: str, payload: dict[str, Any]) -> None:
        trip_id = self.trip_for(device_id, payload)
        frame_id = payload.get("frame_id")
        if frame_id:
            self.repo.insert_frame(
                frame_id,
                trip_id,
                float(payload["ts_utc"]),
                float(payload["lat"]),
                float(payload["lon"]),
            )
        inserted = self.repo.insert_detection(trip_id, payload)
        self.counts["detection" if inserted else "duplicates"] += 1
        if inserted:
            log.info(
                "%s %.1f%% @ %.5f,%.5f (acc %s m)",
                payload.get("class_code"),
                float(payload.get("confidence", 0)) * 100,
                payload.get("lat"),
                payload.get("lon"),
                payload.get("gps_accuracy_m"),
            )

    def _status(self, device_id: str, payload: dict[str, Any]) -> None:
        self.repo.upsert_device(
            device_id,
            last_seen=float(payload.get("ts_utc", 0.0)),
            agent_version=payload.get("agent_version"),
            model_version=payload.get("model_version"),
        )
        self.repo.insert_status(
            DeviceStatus(
                device_id=device_id,
                ts_utc=float(payload.get("ts_utc", 0.0)),
                fps=payload.get("fps"),
                drop_pct=payload.get("drop_pct"),
                queue_depth=payload.get("queue_depth"),
                infer_ms=payload.get("infer_ms"),
                agent_version=payload.get("agent_version"),
                model_version=payload.get("model_version"),
            )
        )
        self.counts["status"] += 1


def start_receiver(
    settings: Settings, repo: Repository | None = None
) -> tuple[MQTTSubscriber, Receiver]:
    """Subscribe and begin persisting."""
    receiver = Receiver(settings, repo)
    subscriber = MQTTSubscriber(
        settings.mqtt_url,
        settings.mqtt_topic_prefix,
        on_message=receiver.handle,
    )
    subscriber.connect_async()
    return subscriber, receiver


def ingest_from_outbox(db_path: Path, settings: Settings, repo: Repository) -> int:
    """Test/dev helper: replay a database's outbox straight into the repository.

    Proves the full capture -> queue -> persist path without a broker.
    """
    from ..transport.outbox import Outbox

    outbox = Outbox(db_path)
    receiver = Receiver(settings, repo)
    count = 0
    for row in outbox.pending(limit=100_000):
        channel = row.topic.rsplit("/", 1)[-1]
        receiver.handle(channel, row.payload)
        outbox.mark_sent(row.id)
        count += 1
    return count
