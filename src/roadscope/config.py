"""Application configuration.

All settings come from the environment or a local ``.env`` file. No secrets or
machine-specific values are hardcoded -- the previous prototype baked a personal
name into a function signature, which made the code unusable for anyone else.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths


class Settings(BaseSettings):
    """Runtime configuration for the edge agent and server."""

    model_config = SettingsConfigDict(
        env_prefix="ROADSCOPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identity -------------------------------------------------------
    device_id: str = "van-01"

    # --- Transport ------------------------------------------------------
    mqtt_url: str = "tcp://localhost:1883"
    mqtt_topic_prefix: str = "roadscope"
    mqtt_qos: int = 1
    mqtt_keepalive: int = 30

    # --- Capture --------------------------------------------------------
    # None => synthetic/test source (see agent.ingest).
    camera_url: str | None = None
    camera_fps: int = 30

    # --- Detection ------------------------------------------------------
    model_path: Path | None = None
    model_name: str = "roadscope-yolov8m-seg.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.50
    imgsz: int = 960
    device: str = "cpu"

    # --- Pipeline -------------------------------------------------------
    queue_size: int = 8
    infer_target_ms: int = 120
    sample_hz: float = 5.0

    # --- Fusion ---------------------------------------------------------
    geofence_m: float = 15.0
    track_min_frames: int = 3
    track_max_age_s: float = 2.0
    iou_match_threshold: float = 0.30

    # --- GPS quality gates ----------------------------------------------
    gps_max_accuracy_m: float = 10.0
    gps_min_sats: int = 6
    gps_max_hdop: float = 2.0
    gps_max_age_s: float = 2.0

    # --- Civil ----------------------------------------------------------
    sample_unit_area_m2: float = 250.0
    corridor_area_m2: float = 500.0

    # --- Storage --------------------------------------------------------
    outbox_db: Path = Field(default_factory=lambda: paths.data_dir() / "outbox.sqlite")
    server_db: Path = Field(default_factory=lambda: paths.data_dir() / "roadscope.sqlite")
    thumb_dir: Path = Field(default_factory=lambda: paths.data_dir() / "thumbs")

    # --- Logging --------------------------------------------------------
    log_level: str = "INFO"
    log_file: Path | None = None

    # --- Server ---------------------------------------------------------
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    telemetry_port: int = 5050

    def resolved_model_path(self) -> Path | None:
        """Resolve the weights file, preferring an explicit override.

        An explicit ``model_path`` that does not exist raises, rather than
        falling back to a different model. Silently loading the wrong weights is
        exactly the failure mode this rebuild exists to eliminate.
        """
        if self.model_path is not None:
            candidate = Path(self.model_path).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"ROADSCOPE_MODEL_PATH points at a missing file: {candidate}. "
                    "Refusing to silently fall back to another model."
                )
            return candidate
        return paths.resolve_model((self.model_name, *paths.MODEL_CANDIDATES))

    def topic(self, channel: str) -> str:
        """Build an MQTT topic for this device, e.g. ``roadscope/van-01/gps``."""
        return f"{self.mqtt_topic_prefix}/{self.device_id}/{channel}"


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings()
