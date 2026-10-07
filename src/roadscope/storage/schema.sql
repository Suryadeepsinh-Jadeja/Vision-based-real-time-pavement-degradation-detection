-- RoadScope server schema. System of record.
--
-- Applied idempotently by storage/db.py on first connection.
-- Design notes in REBUILD_PLAN.md section 10.5.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS devices (
    device_id     TEXT PRIMARY KEY,
    name          TEXT,
    last_seen     REAL,
    agent_version TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS trips (
    trip_id    TEXT PRIMARY KEY,
    device_id  TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    started_at REAL NOT NULL,
    ended_at   REAL,
    distance_m REAL
);

CREATE TABLE IF NOT EXISTS gps_fixes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id         TEXT NOT NULL REFERENCES trips(trip_id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    ts_utc          REAL NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    speed_kmh       REAL,
    heading         REAL,
    accuracy_m      REAL,
    hdop            REAL,
    n_sats          INTEGER,
    quality_flag    TEXT,
    road_name       TEXT,
    mapmatch_conf   REAL,
    -- Makes MQTT QoS1 redelivery idempotent: a replayed packet cannot duplicate
    -- a row. This is what allows at-least-once delivery without data corruption.
    UNIQUE (trip_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_gps_trip_ts ON gps_fixes (trip_id, ts_utc);

CREATE TABLE IF NOT EXISTS frames (
    frame_id   TEXT PRIMARY KEY,
    trip_id    TEXT NOT NULL REFERENCES trips(trip_id) ON DELETE CASCADE,
    ts_utc     REAL NOT NULL,
    lat        REAL,
    lon        REAL,
    thumb_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_frames_trip_ts ON frames (trip_id, ts_utc);

CREATE TABLE IF NOT EXISTS detections (
    detection_id    TEXT PRIMARY KEY,
    trip_id         TEXT NOT NULL REFERENCES trips(trip_id) ON DELETE CASCADE,
    frame_id        TEXT REFERENCES frames(frame_id) ON DELETE SET NULL,
    ts_utc          REAL NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    gps_accuracy_m  REAL,
    class_code      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    severity        TEXT,
    length_m        REAL,
    width_mm        REAL,
    area_m2         REAL,
    frames_seen     INTEGER,
    track_duration_s REAL,
    bbox_json       TEXT,
    road_name       TEXT,
    model_version   TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_trip_ts ON detections (trip_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_det_class ON detections (class_code);

CREATE TABLE IF NOT EXISTS sample_units (
    unit_id          TEXT PRIMARY KEY,
    trip_id          TEXT NOT NULL REFERENCES trips(trip_id) ON DELETE CASCADE,
    chainage_start_m REAL,
    chainage_end_m   REAL,
    area_m2          REAL,
    tdv              REAL,
    cdv              REAL,
    pci              REAL,
    rating           TEXT
);

-- Device-reported health, one row per status heartbeat. Retained so the
-- dashboard can show ingest rate and drop percentage over a survey.
CREATE TABLE IF NOT EXISTS device_status (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    ts_utc        REAL NOT NULL,
    fps           REAL,
    drop_pct      REAL,
    queue_depth   INTEGER,
    infer_ms      REAL,
    agent_version TEXT,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_device_ts ON device_status (device_id, ts_utc);