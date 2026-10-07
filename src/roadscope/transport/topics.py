"""MQTT topic scheme and payload construction.

Wire format is documented in REBUILD_PLAN.md section 10.2. Keep this module the
single source of truth so the agent and the server cannot drift apart.

Topics::

    roadscope/{device}/gps        QoS1  position + fix quality
    roadscope/{device}/detection  QoS1  one confirmed defect
    roadscope/{device}/status     QoS1  heartbeat, retained, LWT
    roadscope/{device}/cmd        QoS1  dashboard -> agent
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

TOPIC_GPS = "gps"
TOPIC_DETECTION = "detection"
TOPIC_STATUS = "status"
TOPIC_CMD = "cmd"


def topic_for(prefix: str, device_id: str, channel: str) -> str:
    """Build a topic. Validates the device id so it cannot break the tree."""
    if not device_id or "/" in device_id:
        raise ValueError(f"invalid device_id: {device_id!r}")
    return f"{prefix.rstrip('/')}/{device_id}/{channel}"


def subscribe_filter(prefix: str, device_id: str = "+") -> str:
    """Wildcard filter for the receiver."""
    return f"{prefix.rstrip('/')}/{device_id}/+"


def new_detection_id() -> str:
    return f"det-{uuid.uuid4().hex[:12]}"


def new_frame_id() -> str:
    return f"frm-{uuid.uuid4().hex[:10]}"


@dataclass(slots=True)
class GPSPayload:
    """Position report. ``seq`` must be monotonic per device: it is the
    idempotency key that makes QoS1 redelivery safe."""

    device_id: str
    seq: int
    ts_utc: float
    t_mono: float | None
    lat: float
    lon: float
    speed_kmh: float | None = None
    heading: float | None = None
    accuracy_m: float | None = None
    hdop: float | None = None
    n_sats: int | None = None
    quality_flag: str = "good"
    trip_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(slots=True)
class DetectionPayload:
    """One confirmed defect. Every field is traceable back to evidence."""

    detection_id: str
    device_id: str
    trip_id: str
    frame_id: str | None
    ts_utc: float
    lat: float
    lon: float
    class_code: str
    confidence: float
    gps_accuracy_m: float | None = None
    severity: str | None = None
    length_m: float | None = None
    width_mm: float | None = None
    area_m2: float | None = None
    frames_seen: int | None = None
    track_duration_s: float | None = None
    bbox: list[int] | None = None
    road_name: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(slots=True)
class StatusPayload:
    """Heartbeat. Retained, so a late subscriber sees current state."""

    device_id: str
    ts_utc: float
    fps: float
    drop_pct: float
    queue_depth: int
    infer_ms: float
    outbox_depth: int
    trip_id: str | None = None
    agent_version: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def utc_now() -> float:
    return datetime.now(UTC).timestamp()
