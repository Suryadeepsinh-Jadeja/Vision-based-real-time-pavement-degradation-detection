"""FastAPI read API.

Read-only by design. The dashboard and any other client read from here or from
SQLite directly; nothing computes in the UI process. That split is what allows
capture to continue while the dashboard is closed.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from ..config import get_settings
from ..storage.repository import Repository, detection_to_dict
from .subscriber import start_receiver

settings = get_settings()
repo = Repository(settings.server_db)

app = FastAPI(
    title="RoadScope API",
    version="0.1.0",
    description="Read-only API over the RoadScope pavement survey store.",
)

_subscriber = None
_receiver = None


@app.on_event("startup")
def _startup() -> None:
    global _subscriber, _receiver
    try:
        _subscriber, _receiver = start_receiver(settings, repo)
    except Exception as exc:
        app.state.receiver_error = str(exc)


@app.on_event("shutdown")
def _shutdown() -> None:
    if _subscriber is not None:
        _subscriber.close()


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok"}
    if _receiver is not None:
        out["received"] = dict(_receiver.counts)
        out["mqtt_connected"] = _subscriber.connected if _subscriber else False
    elif getattr(app.state, "receiver_error", None):
        out["status"] = "degraded"
        out["error"] = app.state.receiver_error
    return out


@app.get("/api/trips")
def trips(limit: int = 50) -> list[dict[str, Any]]:
    return [dict(row) for row in repo.list_trips(limit=limit)]


@app.get("/api/devices")
def devices() -> list[dict[str, Any]]:
    return [dict(row) for row in repo.list_devices()]


@app.get("/api/detections")
def detections(trip_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    return [detection_to_dict(row) for row in repo.detections(trip_id=trip_id, limit=limit)]


@app.get("/api/trips/{trip_id}/track")
def track(trip_id: str) -> list[dict[str, Any]]:
    rows = repo.track(trip_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no fixes for trip {trip_id}")
    return [dict(row) for row in rows]


@app.get("/api/devices/{device_id}/status")
def device_status(device_id: str) -> dict[str, Any]:
    row = repo.latest_status(device_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no status for {device_id}")
    return dict(row)
