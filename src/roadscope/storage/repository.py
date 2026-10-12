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

from ..geo.quality import haversine_m
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

    # -- survey results ----------------------------------------------------
    def insert_sample_unit(self, trip_id: str, unit: dict[str, Any]) -> bool:
        """Persist one scored sample unit.

        Idempotent on ``unit_id`` so recomputing a trip's PCI (the dashboard
        recomputes on demand, and a trip can gain late detections) overwrites
        rather than duplicates.
        """
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT OR REPLACE INTO sample_units"
                " (trip_id, unit_id, chainage_start_m, chainage_end_m, area_m2,"
                "  tdv, cdv, pci, rating)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trip_id,
                    unit["unit_id"],
                    unit.get("chainage_start_m"),
                    unit.get("chainage_end_m"),
                    unit.get("area_m2"),
                    unit.get("tdv"),
                    unit.get("cdv"),
                    unit.get("pci"),
                    unit.get("rating"),
                ),
            )
            return cur.rowcount > 0

    def sample_units(self, trip_id: str) -> list[sqlite3.Row]:
        """Scored sample units for a trip, in chainage order."""
        return list(
            self._conn.execute(
                "SELECT * FROM sample_units WHERE trip_id = ? ORDER BY chainage_start_m, unit_id",
                (trip_id,),
            )
        )

    def replace_sample_units(self, trip_id: str, units: list[dict[str, Any]]) -> int:
        """Replace every unit for a trip with the supplied set.

        A recomputation that yields fewer units than before must not leave
        stale rows behind, so this deletes first rather than merging.
        """
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM sample_units WHERE trip_id = ?", (trip_id,))
        for unit in units:
            self.insert_sample_unit(trip_id, unit)
        return len(units)

    def condition_segments(self, trip_id: str) -> list[sqlite3.Row]:
        """Contiguous stretches of road sharing one PCI rating.

        Built in SQL by walking the ordered units and opening a new segment
        whenever the rating changes. The dashboard draws one polyline per
        segment so colour changes appear at real chainage boundaries instead of
        being interpolated across a whole trip.
        """
        rows = self._conn.execute(
            "SELECT u.*, f.lat, f.lon"
            "  FROM sample_units u"
            "  LEFT JOIN gps_fixes f"
            "    ON f.trip_id = u.trip_id AND f.ts_utc = ("
            "        SELECT MIN(g.ts_utc) FROM gps_fixes g"
            "         WHERE g.trip_id = u.trip_id)"
            " WHERE u.trip_id = ?"
            " ORDER BY u.chainage_start_m, u.unit_id",
            (trip_id,),
        )
        return list(rows)

    def unit_geometry(self, trip_id: str) -> dict[str, list[tuple[float, float]]]:
        """Map each unit_id to the track points it covers.

        Units are assigned by chainage, so a unit's position is the stretch of
        GPS track between its start and end chainage. Returns ``unit_id ->
        [(lat, lon), ...]`` for the map.
        """
        units = self.sample_units(trip_id)
        track = self.track(trip_id)
        if not units or not track:
            return {}

        chainage = 0.0
        points: list[tuple[float, tuple[float, float]]] = []
        for i, row in enumerate(track):
            if i:
                chainage += haversine_m(
                    track[i - 1]["lat"], track[i - 1]["lon"], row["lat"], row["lon"]
                )
            points.append((chainage, (row["lat"], row["lon"])))

        geometry: dict[str, list[tuple[float, float]]] = {}
        for unit in units:
            start = unit["chainage_start_m"]
            end = unit["chainage_end_m"]
            if start is None or end is None:
                continue
            geometry[unit["unit_id"]] = [p for ch, p in points if start <= ch <= end]
        return geometry

    def status_history(self, device_id: str, limit: int = 500) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM device_status WHERE device_id = ? ORDER BY ts_utc LIMIT ?",
                (device_id, limit),
            )
        )

    def class_summary(self, trip_id: str | None = None) -> list[sqlite3.Row]:
        """Counts per distress class, highest first."""
        if trip_id:
            return list(
                self._conn.execute(
                    "SELECT class_code, COUNT(*) AS n, AVG(confidence) AS mean_conf"
                    "  FROM detections WHERE trip_id = ? GROUP BY class_code"
                    " ORDER BY n DESC",
                    (trip_id,),
                )
            )
        return list(
            self._conn.execute(
                "SELECT class_code, COUNT(*) AS n, AVG(confidence) AS mean_conf"
                "  FROM detections GROUP BY class_code ORDER BY n DESC"
            )
        )

    def detection_counts(self, trip_id: str) -> dict[str, int]:
        return {r["class_code"]: int(r["n"]) for r in self.class_summary(trip_id)}


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
