"""Typed data access for the RoadScope store.

Every write is idempotent on a natural key, because MQTT QoS1 is at-least-once:
the broker may deliver the same packet twice, and replaying an outbox after a
reconnect must not corrupt the ledger.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import init_db, thread_connection, transaction

__all__ = [
    "DeviceStatus",
    "Repository",
    "new_trip_id",
    "utc_now",
]


def utc_now() -> float:
    return datetime.now(UTC).timestamp()


def new_trip_id(device_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"trip-{device_id}-{stamp}-{uuid.uuid4().hex[:4]}"


@dataclass(slots=True)
class DeviceStatus:
    """One agent heartbeat."""

    device_id: str
    ts_utc: float
    fps: float | None = None
    drop_pct: float | None = None
    queue_depth: int | None = None
    infer_ms: float | None = None
    agent_version: str | None = None
    model_version: str | None = None


class Repository:
    """Data access for the server store."""

    def __init__(self, db_path: Path, *, create: bool = True) -> None:
        self.db_path = Path(db_path)
        if create:
            init_db(self.db_path)

    @property
    def _conn(self) -> sqlite3.Connection:
        return thread_connection(self.db_path)

    # -- devices -----------------------------------------------------------
    def upsert_device(self, device_id: str, **fields: Any) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO devices (device_id, name, last_seen, agent_version, model_version)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(device_id) DO UPDATE SET"
                "   name=COALESCE(excluded.name, name),"
                "   last_seen=excluded.last_seen,"
                "   agent_version=COALESCE(excluded.agent_version, agent_version),"
                "   model_version=COALESCE(excluded.model_version, model_version)",
                (
                    device_id,
                    fields.get("name"),
                    fields.get("last_seen", utc_now()),
                    fields.get("agent_version"),
                    fields.get("model_version"),
                ),
            )

    def list_devices(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM devices ORDER BY device_id"))

    # -- trips -------------------------------------------------------------
    def start_trip(self, trip_id: str, device_id: str, started_at: float | None = None) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trips (trip_id, device_id, started_at) VALUES (?, ?, ?)",
                (trip_id, device_id, started_at or utc_now()),
            )

    def end_trip(
        self, trip_id: str, ended_at: float | None = None, distance_m: float | None = None
    ) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE trips SET ended_at = ?, distance_m = COALESCE(?, distance_m)"
                " WHERE trip_id = ?",
                (ended_at or utc_now(), distance_m, trip_id),
            )

    def list_trips(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self._conn.execute("SELECT * FROM trips ORDER BY started_at DESC LIMIT ?", (limit,))
        )

    # -- gps ---------------------------------------------------------------
    def insert_fix(self, trip_id: str, payload: dict[str, Any]) -> bool:
        """Insert one GPS fix. Returns False if it was a duplicate ``seq``."""
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO gps_fixes"
                " (trip_id, seq, ts_utc, lat, lon, speed_kmh, heading, accuracy_m,"
                "  hdop, n_sats, quality_flag, road_name, mapmatch_conf)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trip_id,
                    int(payload["seq"]),
                    float(payload["ts_utc"]),
                    float(payload["lat"]),
                    float(payload["lon"]),
                    payload.get("speed_kmh"),
                    payload.get("heading"),
                    payload.get("accuracy_m"),
                    payload.get("hdop"),
                    payload.get("n_sats"),
                    str(payload.get("quality_flag", "good")),
                    payload.get("road_name"),
                    payload.get("mapmatch_conf"),
                ),
            )
            return cur.rowcount > 0

    def fix_count(self, trip_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM gps_fixes WHERE trip_id = ?", (trip_id,)
        ).fetchone()
        return int(row["n"])

    def track(self, trip_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM gps_fixes WHERE trip_id = ? ORDER BY ts_utc", (trip_id,)
            )
        )

    # -- frames ------------------------------------------------------------
    def insert_frame(
        self,
        frame_id: str,
        trip_id: str,
        ts_utc: float,
        lat: float | None,
        lon: float | None,
        thumb_path: str | None = None,
    ) -> bool:
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO frames (frame_id, trip_id, ts_utc, lat, lon, thumb_path)"
                " VALUES (?,?,?,?,?,?)",
                (frame_id, trip_id, ts_utc, lat, lon, thumb_path),
            )
            return cur.rowcount > 0

    # -- detections --------------------------------------------------------
    def insert_detection(self, trip_id: str, payload: dict[str, Any]) -> bool:
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO detections"
                " (detection_id, trip_id, frame_id, ts_utc, lat, lon, gps_accuracy_m,"
                "  class_code, confidence, severity, length_m, width_mm, area_m2,"
                "  frames_seen, track_duration_s, bbox_json, road_name, model_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(payload["detection_id"]),
                    trip_id,
                    payload.get("frame_id"),
                    float(payload["ts_utc"]),
                    float(payload["lat"]),
                    float(payload["lon"]),
                    payload.get("gps_accuracy_m"),
                    str(payload["class_code"]),
                    float(payload["confidence"]),
                    payload.get("severity"),
                    payload.get("length_m"),
                    payload.get("width_mm"),
                    payload.get("area_m2"),
                    payload.get("frames_seen"),
                    payload.get("track_duration_s"),
                    json.dumps(payload["bbox"]) if payload.get("bbox") else None,
                    payload.get("road_name"),
                    payload.get("model_version"),
                ),
            )
            return cur.rowcount > 0

    def detections(self, trip_id: str | None = None, limit: int = 500) -> list[sqlite3.Row]:
        if trip_id:
            return list(
                self._conn.execute(
                    "SELECT * FROM detections WHERE trip_id = ? ORDER BY ts_utc LIMIT ?",
                    (trip_id, limit),
                )
            )
        return list(
            self._conn.execute("SELECT * FROM detections ORDER BY ts_utc DESC LIMIT ?", (limit,))
        )

    # -- status ------------------------------------------------------------
    def insert_status(self, status: DeviceStatus) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO device_status"
                " (device_id, ts_utc, fps, drop_pct, queue_depth, infer_ms,"
                "  agent_version, model_version) VALUES (?,?,?,?,?,?,?,?)",
                (
                    status.device_id,
                    status.ts_utc,
                    status.fps,
                    status.drop_pct,
                    status.queue_depth,
                    status.infer_ms,
                    status.agent_version,
                    status.model_version,
                ),
            )

    def latest_status(self, device_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM device_status WHERE device_id = ? ORDER BY ts_utc DESC LIMIT 1",
            (device_id,),
        ).fetchone()

    def latest_detection(self) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM detections ORDER BY ts_utc DESC LIMIT 1"
        ).fetchone()


def detection_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a detections row to a JSON-safe dict."""
    data = dict(row)
    if data.get("bbox_json"):
        try:
            data["bbox"] = json.loads(data.pop("bbox_json"))
        except (json.JSONDecodeError, TypeError):
            data.pop("bbox_json", None)
    return {k: v for k, v in data.items() if v is not None}


def status_to_dict(status: DeviceStatus) -> dict[str, Any]:
    return asdict(status)
