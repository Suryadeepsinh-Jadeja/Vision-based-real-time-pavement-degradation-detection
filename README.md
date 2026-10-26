# RoadScope — Vision-Based Road Damage Detection & GIS System

RoadScope turns a phone into a road-survey sensor. The phone streams video and
GPS to a laptop; the laptop detects potholes and cracks with a YOLOv8 model,
tags each one with its GPS position, and sends it over MQTT to a server that
stores it. A dashboard shows every detection on a map with the photo the model
flagged, and the civil-engineering engine scores road condition with the
Pavement Condition Index (PCI, ASTM D6433) and turns poor stretches into
prioritised repair work orders.

Smart Cities transdisciplinary project, VIT Chennai.

| Member | Reg. no. | Pillar |
| --- | --- | --- |
| Jadeja Suryadeepsinh Bharatsinh | 23BEL1027 | System lead, integration, transport, live demo |
| Vishnu Nair | 23BEL1028 | Computer science / AI: dataset, training, evaluation |
| Udayvardhan Singh Rathore | 23BEL1018 | Civil engineering: ASTM D6433, PCI engine |
| Ainessh Kumar S | 23BEL1039 | Geospatial & GIS: GPS quality, Kalman filter, maps |
| Shreyan Biswas | 23BEL1052 | Urban governance & UI: dashboards, work orders |

---

## Contents

1. [Status and headline numbers](#1-status-and-headline-numbers)
2. [How it works](#2-how-it-works)
3. [Installation](#3-installation)
4. [Running the live demo (phone + laptop)](#4-running-the-live-demo-phone--laptop)
5. [Other ways to run it](#5-other-ways-to-run-it)
6. [Command reference](#6-command-reference)
7. [Configuration](#7-configuration)
8. [Repository layout](#8-repository-layout)
9. [Inside each component](#9-inside-each-component)
10. [Training the model](#10-training-the-model)
11. [Results](#11-results)
12. [Testing and CI](#12-testing-and-ci)
13. [Limitations — what to claim and what not to](#13-limitations--what-to-claim-and-what-not-to)
14. [Troubleshooting](#14-troubleshooting)
15. [Project history](#15-project-history)

---

## 1. Status and headline numbers

Working end to end: phone camera → detection → GPS tag → MQTT → server
database → live dashboard, demonstrated on 26 Oct 2026 with an Android phone and
a MacBook Air (M1) over the phone's hotspot.

| Figure | Value | Where it comes from |
| --- | --- | --- |
| Overall mAP50 (4 classes) | **0.609** | RDD2022 validation set, 3,185 images |
| Pothole (D40) AP50 | **0.538** | same; target was 0.55 |
| Inference speed | **8 frames/s** (125 ms/frame) | M1 GPU (MPS) at 960 px |
| GPS accuracy in the demo | **±4–5 m** outdoors, 20–40 m indoors | phone GPS via GPSLogger |
| PCI engine | reproduces published ASTM D6433 / TM 5-623 worked examples | `tests/test_civil.py` |
| Automated tests | ~160, ruff clean | `make test` |

---

## 2. How it works

```
 ANDROID PHONE                      LAPTOP — edge agent (python -m roadscope.agent)
 ┌──────────────┐   MJPEG video     ┌────────┐   ┌────────┐   ┌─────────┐   ┌────────┐
 │ IP Webcam    │ ────────────────▶ │ Detect │ ▶ │ Geotag │ ▶ │ Confirm │ ▶ │ Outbox │
 └──────────────┘                   │ YOLOv8 │   │ gate + │   │ tracker │   │ SQLite │
 ┌──────────────┐   HTTP push /gps  └────────┘   │ Kalman │   │ + geo-  │   │ buffer │
 │ GPSLogger    │ ───────────────────────────▶   └────────┘   │ fence   │   └───┬────┘
 └──────────────┘                                             └─────────┘       │ MQTT (JSON + photo)
                                                                                ▼
                                    LAPTOP — server side
                                    ┌───────────┐   ┌──────────┐   ┌─────────────────┐   ┌───────────┐
                                    │ Mosquitto │ ▶ │ Receiver │ ▶ │ SQLite + frames/│ ▶ │ Dashboard │
                                    │ :1883     │   │make serve│   │ var/demo.sqlite │   │ Streamlit │
                                    └───────────┘   └──────────┘   └─────────────────┘   └───────────┘
```

1. **Capture.** IP Webcam on the phone serves the camera as an MJPEG stream
   (`http://<phone-ip>:8080/video`). GPSLogger on the phone calls a URL on the
   laptop with every GPS fix (`http://<laptop-ip>:8765/gps?lat=…&lon=…`).
2. **Detect.** The agent runs YOLOv8m (trained on RDD2022) on frames. Capture
   never waits for the model: the queue between them drops the oldest frame.
3. **Geotag.** GPS fixes are quality-gated, smoothed with a Kalman filter, and
   interpolated to each frame's timestamp.
4. **Confirm.** A tracker follows each box across frames; a defect is confirmed
   after 3 frames, and a 15 m geofence suppresses revisits. One pothole = one record.
5. **Send.** Each confirmed defect (class, confidence, lat/lon, accuracy, box,
   annotated 640 px JPEG) is written to a local SQLite outbox, then published
   over MQTT (QoS 1). If the network drops nothing is lost; it is resent later.
6. **Store.** The receiver subscribes to `roadscope/+/+`, writes rows to SQLite
   idempotently (a redelivered message never duplicates a row), and saves the
   photo to `var/frames/<frame-id>.jpg`.
7. **Show.** The demo dashboard polls the database every 3 s: live status,
   counts, map, latest photo with its box, and a list of all detections.
8. **Assess.** The engineering dashboard computes PCI per 250 m² sample unit
   and lists work orders, worst first.

---

## 3. Installation

Requirements: macOS or Linux, **Python 3.11**, Homebrew (macOS) for Mosquitto.

```bash
cd "/Users/suryadeepsinhjadeja/Desktop/Smart Cities"
make install                     # creates venv/ and installs all extras (~2 GB, PyTorch)
source venv/bin/activate
brew install mosquitto           # the MQTT broker
bash scripts/download_weights.sh --verify   # or copy rdd2022-yolov8m-r1.pt into models/
make check                       # must list 4 classes D00 D10 D20 D40 and a device (mps/cpu)
```

Installed extras (see `pyproject.toml`): `yolo` (ultralytics), `ui` (streamlit,
folium), `server` (FastAPI, pandas), `geo` (osmnx, optional road names), `dev`
(pytest, ruff).

**zsh tip:** zsh does not treat `#` as a comment when you paste commands. Paste
commands without the trailing comments, or run once:
`echo 'setopt interactivecomments' >> ~/.zshrc`.

---

## 4. Running the live demo (phone + laptop)

### 4.1 Phone apps (Android, both free)

| App | Get it from | Purpose |
| --- | --- | --- |
| **IP Webcam** (Pavel Khlebovich) | Play Store | Camera stream |
| **GPSLogger** (mendhak, `com.mendhak.gpslogger`) | **F-Droid** or GitHub releases — *not* on the Play Store | Pushes GPS to the laptop |

Do not use the Play Store "GPS Logger" by BasicAirData; it cannot send to a URL.

**GPSLogger, one-time setup:**

- Grant location permission ("Allow all the time").
- Performance → Logging interval **1** second, distance filter **0**, keep GPS on between fixes.
- Allow network locations as well as GPS (helps indoors).
- Custom URL → enable "Log to custom URL", method **GET**, URL:
  `http://<MAC-IP>:8765/gps?lat=%LAT&lon=%LON&acc=%ACC&spd=%SPD&dir=%DIR&sat=%SAT`
- Android settings → Apps → GPSLogger → Battery → **Unrestricted**.

### 4.2 Connect the devices

Turn on the phone's **hotspot** and join it from the Mac (best for demos: no
dependency on campus Wi-Fi; the video stays local and uses no mobile data).
Then on the Mac:

```bash
ipconfig getifaddr en0           # MAC-IP  (goes in the GPSLogger URL)
ipconfig getoption en0 router    # PHONE-IP (the hotspot is the router)
```

IP addresses can change every time the Mac joins, so check both before each run.
An IP starting `100.` or `172.30.` on the phone is its mobile-data address, which
the laptop cannot reach.

### 4.3 Four terminals, in this order

First line in **every** terminal:

```bash
cd "/Users/suryadeepsinhjadeja/Desktop/Smart Cities" && source venv/bin/activate
```

| # | Command | Wait for |
| --- | --- | --- |
| 0 | Phone: IP Webcam → **Start server**; GPSLogger → **Start logging** | — |
| 1 | `/opt/homebrew/opt/mosquitto/sbin/mosquitto` | `running`, no "Address already in use" |
| 2 | `make serve` | `subscribed to roadscope/+/+`, `receiving` |
| 3 | `ROADSCOPE_SERVER_DB=var/demo.sqlite streamlit run scripts/demo_dashboard.py` | browser opens at http://localhost:8501 |
| 4a | `make check-camera URL=http://<PHONE-IP>:8080/video` | `camera check : OK` |
| 4b | `python -m roadscope.agent --camera-url http://<PHONE-IP>:8080/video --gps-listen --gps-max-accuracy 100 --gps-reuse-last --no-geofence --seconds 1800` | `starting from the last known GPS position`, `running for 1800.0s` |

Point the phone at a pothole (a printed or full-screen photo works) **15–25 cm
away, filling about half the view, steady for 2–3 s**. You should see
`D40 85% @ 12.84…` in terminal 4, `detections` rising in terminal 2, and a new
row at the top of the dashboard with the photo and blue box.

Stop with Ctrl+C in reverse order (4, 3, 2, 1). To check GPS is arriving, open
`http://<MAC-IP>:8765/gps` in a browser: `received` should keep increasing.

### 4.4 Why the demo flags exist

| Flag | Why | For a real drive |
| --- | --- | --- |
| `--gps-max-accuracy 100` | Indoors a phone reports 20–40 m and gets worse each second. The default 10 m gate rejects every fix, and a detection without a fix is dropped. | omit (10 m default) |
| `--gps-reuse-last` | Starts from the last accepted position (saved in `var/last_gps_fix.json`) so a slow GPS start does not drop detections. | optional |
| `--no-geofence` | A phone held still reports one position, so every pothole after the first would be treated as a revisit and suppressed. | omit |

---

## 5. Other ways to run it

| What | Command | Needs |
| --- | --- | --- |
| Pre-filled data (2 synthetic trips, ~100 defects) | `make seed` | nothing |
| Engineering dashboard: PCI map, sample units, work orders, device health | `make dashboard` → http://localhost:8501 | data in `var/demo.sqlite` |
| Detector playground: upload a photo or video, see boxes, step through frames | `make playground` → http://localhost:8502 | weights only |
| Agent with a recorded GPS track | `python -m roadscope.agent --camera-url … --gps-csv tests/fixtures/sample_gps_track.csv` | camera |
| Agent with a synthetic straight-line GPS | `… --gps-synthetic` | camera |
| Agent polling a phone app that *serves* GPS JSON | `… --gps-url http://<phone>:<port>/json` | such an app |
| Replay the outbox straight into the database (no broker) | `python -m roadscope.agent --replay` | outbox rows |
| Read-only REST API over the store | `uvicorn roadscope.server.app:app --port 8000` (`/health`, `/api/trips`, `/api/detections`, …) | `server` extra |
| Broker via Docker instead of Homebrew | `docker compose up -d` | Docker |
| Benchmark inference speed | `python scripts/benchmark_detector.py --sizes 640 960 1280` | weights |
| Check a weights file is the 4-class model | `python scripts/check_model.py models/rdd2022-yolov8m-r1.pt` | weights |

---

## 6. Command reference

### Makefile

| Target | Does |
| --- | --- |
| `make install` | Create `venv/` and install all extras |
| `make check` | Diagnose weights, config, device |
| `make check-camera URL=…` | Test the phone camera link; explains network mismatches |
| `make seed` | Write synthetic trips into `var/demo.sqlite` |
| `make serve` | MQTT receiver into `var/demo.sqlite` |
| `make dashboard` | Engineering dashboard |
| `make playground` | Detector playground |
| `make test` / `make lint` | pytest / ruff check + format check |
| `make clean` | Remove `var/demo.sqlite` and uploads |

### Agent flags (`python -m roadscope.agent`)

| Flag | Meaning |
| --- | --- |
| `--check` | Print diagnostics and exit |
| `--camera-url URL` | MJPEG stream (IP Webcam: `/video`, DroidCam: `:4747/video`) |
| `--skip-camera-check` | Start even if the camera URL does not answer |
| `--gps-listen [PORT]` | Receive GPS pushed by the phone (default 8765); prints the GPSLogger URL |
| `--gps-max-accuracy M` | Reject fixes worse than M metres (default 10) |
| `--gps-reuse-last [MIN]` | Start from the last accepted fix if newer than MIN minutes (default 120) |
| `--gps-url URL` | Poll a GPS JSON endpoint instead |
| `--gps-csv FILE` | Replay a recorded track |
| `--gps-synthetic` | Straight-line synthetic track |
| `--no-geofence` | Record repeat defects at the same spot (stationary demo only) |
| `--seconds N` | Run time (default 10) |
| `--device-id`, `--model-path`, `--log-level` | Overrides |
| `--replay` | Push the outbox into the server database without a broker |

### Server (`python -m roadscope.server`, wrapped by `make serve`)

`--server-db`, `--mqtt-url`, `--seconds`, `--report-every`, `--log-level`.

---

## 7. Configuration

Every setting has a default in `src/roadscope/config.py`; override with
environment variables (prefix `ROADSCOPE_`) or a `.env` file
(`cp .env.example .env`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEVICE_ID` | `van-01` | Device name in MQTT topics |
| `MQTT_URL` / `MQTT_TOPIC_PREFIX` / `MQTT_QOS` | `tcp://localhost:1883` / `roadscope` / `1` | Transport |
| `CAMERA_URL` | none | Default camera stream |
| `MODEL_PATH` / `MODEL_NAME` | none / `rdd2022-yolov8m-r1.pt` | Weights (explicit path wins; missing file is an error) |
| `CONF_THRESHOLD` / `IOU_THRESHOLD` | 0.35 / 0.50 | Detector thresholds |
| `IMGSZ` | 960 | Inference size (trained at 960; 640 for CPU-only) |
| `DEVICE` | `auto` | `auto` picks MPS on Apple Silicon, CUDA if present, else CPU |
| `QUEUE_SIZE` | 8 | Frames buffered between capture and inference |
| `GEOFENCE_M` | 15 | Revisit suppression radius |
| `TRACK_MIN_FRAMES` / `TRACK_MAX_AGE_S` / `IOU_MATCH_THRESHOLD` | 3 / 2.0 / 0.30 | Tracker |
| `GPS_MAX_ACCURACY_M` / `GPS_MIN_SATS` / `GPS_MAX_HDOP` / `GPS_MAX_AGE_S` | 10 / 6 / 2.0 / 2.0 | GPS quality gate |
| `SAMPLE_UNIT_AREA_M2` | 250 | PCI sample unit |
| `OUTBOX_DB` / `SERVER_DB` / `THUMB_DIR` | `var/outbox.sqlite` / `var/roadscope.sqlite` / `var/thumbs` | Paths (`make serve` and the dashboards use `var/demo.sqlite`) |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / none | Logging |

---

## 8. Repository layout

```
README.md                 this file — the only documentation
Makefile                  one-word commands (section 6)
pyproject.toml            package, dependencies, pytest + ruff config
.env.example              every configurable setting
docker-compose.yml        optional Mosquitto (and PostGIS) via Docker
deploy/mosquitto.conf     broker config for the Docker route
.github/workflows/ci.yml  lint + tests on every push
models/                   rdd2022-yolov8m-r1.pt (not in git; download_weights.sh)
scripts/
  demo_dashboard.py       one-screen live demo dashboard
  dashboard.py            engineering dashboard launcher (PCI, work orders)
  detector_playground.py  upload-and-inspect launcher
  seed_demo_data.py       synthetic trips for the dashboards
  prepare_rdd2022.py      curate RDD2022 into a 4-class YOLO dataset
  download_weights.sh     fetch / verify the trained weights
  check_model.py          confirm a weights file is the 4-class model
  benchmark_detector.py   latency at several resolutions
src/roadscope/
  config.py, paths.py, logging_setup.py
  agent/      __main__ (CLI), ingest (camera sources + probe), pipeline (threads), health
  vision/     detector (YOLOv8 wrapper, device selection)
  fusion/     tracker (ByteTrack-style), geofence, fuser (records + evidence photo)
  geo/        gps_sources (push/poll/CSV/synthetic), quality (gate), kalman, mapmatch (OSM)
  transport/  topics (wire format), outbox (SQLite queue), mqtt_client
  server/     __main__ (receiver CLI), subscriber (MQTT → DB, photos), survey (PCI + work orders), app (REST API)
  storage/    schema.sql, db, repository
  civil/      distress (ASTM types), severity, dv_tables, cdv, pci
  ui/         dashboard, maps, playground, detector_playground
tests/                    unit + integration tests, fixtures/
var/                      runtime data (not in git): demo.sqlite, outbox.sqlite, frames/, thumbs/, samples/
```

---

## 9. Inside each component

### 9.1 Capture (`agent/ingest.py`)

Three interchangeable frame sources yield `(frame, t_mono)`: `MjpegSource`
(phone), `VideoFileSource` (recorded clip), `SyntheticSource` (drawn road, for
tests). `probe_camera` checks a URL before the run and explains the common
failure: phone and laptop on different networks.

### 9.2 Pipeline (`agent/pipeline.py`)

Three threads: **ingest** (never blocks), **infer**, **publish** (drains the
outbox), plus a 5 s **status** heartbeat. The ingest → infer queue is bounded and
drop-oldest: a slow model lowers frame rate but never freezes the camera. At 8 fps
processed vs 30 fps captured, ~73% of frames are skipped by design. When the model
sees a defect but there is no GPS fix, or a fix is rejected, the agent logs a
warning with the reason (rate-limited to every 5 s).

### 9.3 Detection (`vision/detector.py`)

YOLOv8m fine-tuned on RDD2022, four classes:

| Code | Damage | ASTM D6433 type | Measured as |
| --- | --- | --- | --- |
| D00 | Longitudinal crack | 10 | length (m) |
| D10 | Transverse crack | 10 | length (m) |
| D20 | Alligator (fatigue) crack | 1 | area (m²) |
| D40 | Pothole | 13 | count |

`model.names` is used verbatim. Device `auto` resolves to MPS on Apple Silicon
(2.4× faster than CPU, output identical to 1e-6).

### 9.4 Tracking and de-duplication (`fusion/`)

A detection is not a defect: one pothole appears in many frames. The tracker
(ByteTrack-style) matches high-confidence boxes first, uses low-confidence ones
only to keep tracks alive, and never matches across classes. A track is confirmed
after `TRACK_MIN_FRAMES` (3). The geofence suppresses a defect within 15 m of one
emitted more than 5 s earlier (a revisit); co-visible defects in one frame are
not suppressed. On confirmation the fuser saves the frame resized to 640 px with
the box (scaled to match) and label drawn on, named by a unique frame id.

### 9.5 GPS (`geo/`)

- **Sources:** `PushGpsSource` (laptop listens on `/gps`; GET query, form or JSON
  POST; GPSLogger speed in m/s converted to km/h; unfilled `%SAT`-style
  placeholders and 0 satellites treated as unknown; stamped with laptop receive
  time so phone clock skew cannot make fixes stale; `GET /gps` with no
  coordinates returns a status page). Also `HttpGpsSource`, `CsvGpsSource`,
  `SyntheticGpsSource`.
- **Quality gate** (`quality.py`): valid position, age ≤ 2 s, accuracy ≤ limit,
  satellites ≥ 6 and HDOP ≤ 2 when reported. Unreported fields are not failures.
- **Kalman filter** (`kalman.py`): constant-velocity model in a local east/north
  metre grid around an anchor, converted back to lat/lon; accurate at city scale.
- **Frame position:** linear interpolation between the fixes around a frame's
  timestamp; the last fix is held if none is newer.
- **Last known position:** each accepted fix is saved to `var/last_gps_fix.json`
  (`--gps-reuse-last`).
- **Road names** (`mapmatch.py`): optional OpenStreetMap lookup via osmnx;
  degrades to no name if unavailable.
- Distances use the haversine formula on WGS84 coordinates.

### 9.6 Transport (`transport/`)

| Topic | QoS | Content |
| --- | --- | --- |
| `roadscope/{device}/gps` | 1 | position, accuracy, quality flag, `seq` |
| `roadscope/{device}/detection` | 1 | one confirmed defect + `image_jpeg_b64` (~30–60 KB) |
| `roadscope/{device}/status` | 1, retained | fps, drop %, queue, inference ms, outbox depth; last-will "offline" |
| `roadscope/{device}/cmd` | 1 | reserved, dashboard → agent |

**Outbox pattern:** every message is written to SQLite first and marked sent only
after the broker accepts it, so a 60 s outage loses nothing. The broker URL
`tcp://host:port` is split into host and port before connecting.

### 9.7 Server and storage (`server/`, `storage/`)

The receiver writes every message idempotently: GPS on `(trip_id, seq)`,
detections on `detection_id`, so QoS 1 redelivery never duplicates rows. The
evidence photo is decoded, checked to be a JPEG, and written to
`frames/<frame-id>.jpg` next to the database. Tables: `devices`, `trips`,
`gps_fixes`, `frames`, `detections`, `sample_units`, `device_status`
(`storage/schema.sql`). SQLite runs in WAL mode; the schema is designed to port
to PostGIS (`docker compose --profile postgis up`).

### 9.8 Pavement Condition Index (`civil/`)

Implements ASTM D6433:

1. Split the GPS track into **sample units of 250 m²** (ASTM allows 250–500).
2. **Density** = quantity ÷ unit area × 100, with ASTM conversions: area as-is;
   cracks × 0.3048 m nominal width; each pothole = 0.0929 m² (1 ft²).
3. **Deduct value** per distress type, severity and density from digitised ASTM curves.
4. **Correction:** HDV = highest DV; m = 1 + (9/98)(100 − HDV), rounded; TDV = sum
   of the m highest DVs; q = number of DVs **> 5**; CDV read from the asphalt
   correction curve for (q, TDV).
5. **PCI = 100 − CDV** per unit; the road's PCI is the mean of its units.

Validation: the ASTM section 9 example (DVs 4, 17, 13 → TDV 34, q = 2) gives
CDV ≈ 26 (PCI ≈ 74); some secondary sources quote 24 using q = 3, but the
standard's rule gives q = 2. The TM 5-623 example (TDV 45, q = 2) gives CDV 33
exactly.

**Severity:** ASTM grades potholes by depth and cracks by width, which a bounding
box cannot measure. Severity is defaulted (potholes High, cracks Medium) and
every defaulted case is reported in the result's `limitations`. The PCI is
therefore a defect-density index following the ASTM method.

| PCI | Rating | Recommended action |
| --- | --- | --- |
| 85–100 | Good | Routine preventive maintenance and crack sealing |
| 70–84 | Satisfactory | Preventive surface seal, minor crack sealing |
| 55–69 | Fair | Cold patching of isolated potholes, slurry seal |
| 40–54 | Poor | Full-depth patch repairs, thin bituminous concrete overlay |
| 25–39 | Very Poor | Extensive mill-and-fill resurfacing |
| 10–24 | Serious | Sub-base reconstruction and asphalt base course |
| 0–9 | Failed | Complete pavement reconstruction |

### 9.9 Work orders (`server/survey.py`)

One work order per sample unit with PCI < 85, worst first, with road name (when
known), chainage, length, PCI, rating, recommended action and centroid. Reference
treatments per defect and severity, carried from the original prototype
(IRC:82-2015 practice), for the future printable PWD report:

| Defect | High | Medium | Low |
| --- | --- | --- | --- |
| D40 Pothole | Full-depth patch: saw-cut, tack coat, 75 mm hot mix (IRC:82-2015 cl. 4.3) — 24–48 h | Semi-permanent patch, cold mix — 5 days | Throw-and-roll cold patch — 14 days |
| D20 Alligator | 50 mm milling + 40 mm BC overlay — 7 days | SAMI + 25 mm overlay — 21 days | Fog / slurry seal — 45 days |
| D10 Transverse | Rout and seal, polymer-modified sealant — 14 days | Clean and rubberised sealant — 30 days | Sand-emulsion slurry — monitor 60 days |
| D00 Longitudinal | Mill 150 mm corridor + micro-surfacing — 14 days | Clean and elastomeric sealing — 30 days | Preventive seal coat — monitor 60 days |

### 9.10 Dashboards (`scripts/`, `ui/`)

- **Demo dashboard** (`scripts/demo_dashboard.py`): reads the database read-only,
  refreshes every 3 s; Live/Offline from the heartbeat, pothole and defect
  counts, last-detection age, map with coloured dots, latest photo with details
  and a Google Maps link, and a scrolling list of all detections newest first.
  "Latest session" (default) or "All sessions". A pop-up appears on each new detection.
- **Engineering dashboard** (`make dashboard`): Folium map with road stretches
  coloured by PCI rating and defect markers with photos, sample-unit table,
  work orders, device health.
- **Detector playground** (`make playground`): upload an image or video, adjust
  confidence and resolution, step through video frames.

---

## 10. Training the model

The weights are **not** in git (a 52 MB file once took the repository to 402 MB).
Get them with `bash scripts/download_weights.sh`, or retrain on Kaggle:

1. **Notebook:** Kaggle → New Notebook, Accelerator **GPU T4 ×2**, Persistence
   "Variables and Files", Internet on. `/kaggle/working` has only 20 GB.
2. **Download only India, Japan and US (2.04 GB of the 13.26 GB archive).** The
   figshare archive (`https://ndownloader.figshare.com/files/38030910`) holds
   seven STORED per-country ZIPs, so each can be fetched by HTTP byte range
   (inclusive): India `609341032-1136050223`, Japan `1136050224-2208649127`,
   United_States `12819836010-13264172022`. Check each file's size and that it
   starts with `PK\x03\x04`. Norway (10.6 GB) and China (drone/motorbike) are excluded.
3. **Unzip** each into `/kaggle/working/RDD2022` (Pascal VOC XML annotations).
4. **Curate:** clone this repository, `pip install ultralytics`, then
   ```
   python scripts/prepare_rdd2022.py --src /kaggle/working/RDD2022 --inspect
   python scripts/prepare_rdd2022.py --src /kaggle/working/RDD2022 --dst /kaggle/working/rdd2022-curated --countries India Japan United_States --val-fraction 0.2
   ```
   Read the per-class report, then delete the raw data.
5. **Train** (detect, not seg: RDD2022 has boxes, not polygons):
   ```python
   from ultralytics import YOLO
   YOLO("yolov8m.pt").train(data="/kaggle/working/rdd2022-curated/roadscope.yaml",
       epochs=40, imgsz=960, batch=16, device=[0, 1], workers=4, patience=15, cache="disk",
       hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5,
       fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.1, copy_paste=0.1, erasing=0.2,
       project="/kaggle/working/runs", name="r1")
   ```
   Halve `batch` if it runs out of memory. Run r1 took 5.0 h for 40 epochs.
6. **Evaluate per class**, never only mAP: `YOLO(best).val(split="val", imgsz=960)`
   and record AP50, precision and recall for each class.
7. **Publish** `best.pt` as a Kaggle output or GitHub Release, never in git; place
   it in `models/` and confirm with `python scripts/check_model.py`.
8. **In-domain test (still to do):** record your own road video, hand-label at
   least 150 frames (CVAT or LabelImg) and evaluate on them. That is the number
   that describes this system on Indian roads.

---

## 11. Results

Validation set: 3,185 images, 6,913 instances. Run r1 (selected): YOLOv8m,
12,743 training images, 40 epochs, 960 px, AdamW, 2× T4, 5.02 h.

| Class | Instances | Precision | Recall | AP50 | AP50-95 |
| --- | --- | --- | --- | --- | --- |
| D00 Longitudinal | 2,462 | 0.597 | 0.636 | 0.651 | 0.364 |
| D10 Transverse | 1,435 | 0.561 | 0.571 | 0.558 | 0.252 |
| D20 Alligator | 1,856 | 0.674 | 0.620 | 0.688 | 0.361 |
| D40 Pothole | 1,160 | 0.595 | 0.495 | 0.538 | 0.234 |
| **All** | 6,913 | 0.607 | 0.581 | **0.609** | 0.303 |

Gates: overall mAP50 ≥ 0.60 **passed** (0.609); pothole AP50 ≥ 0.55 **missed** (0.538).

**Run r2 (rejected):** tripling pothole images raised pothole recall 0.495 →
0.525 but lowered precision 0.595 → 0.580 and AP50 0.538 → 0.526. Pothole AP is
limited by **box placement**, not by too few examples: its AP50-95/AP50 ratio is
0.43, the worst of the four classes, because a small round object loses most of
its IoU when its box is slightly displaced. Likely fixes: 1280 px inference (needs
a GPU), box-regression-focused fine-tuning, SAM 2 box refinement. For context, the
CRDDC 2022 winner reached F1 76.9% with ensembles across six countries.

**Speed** (M1, 8 GB, median of 25 frames, `scripts/benchmark_detector.py`):

| imgsz | CPU | MPS (GPU) | MPS fps | Frames dropped at 30 fps capture |
| --- | --- | --- | --- | --- |
| 640 | 149 ms | 69 ms | 14.4 | 52% |
| **960** | 303 ms | **125 ms** | **8.0** | 73% |
| 1280 | 547 ms | 202 ms | 5.0 | 83% |

960 is the default because the model was trained at 960 and resolution is what
helps small potholes most.

---

## 12. Testing and CI

```bash
make test        # pytest
make lint        # ruff check + ruff format --check
```

Highlights: link loss loses nothing (`test_pipeline.py`); one pothole yields one
record; the agent never imports Streamlit; PCI reproduces the published examples
(`test_civil.py`); tracker and geofence behaviour (`test_fusion.py`, including
the box being drawn inside a resized thumbnail); push-GPS parsing
(`test_push_gps.py`); no-GPS warnings and last-position reuse
(`test_gps_fallback.py`); camera probe failures (`test_ingest.py`); dataset
curation (`test_prepare_rdd2022.py`). Markers: `vision` (needs weights),
`integration` (needs a broker), `field` (needs hardware). GitHub Actions runs lint
and tests on every push to `main`.

---

## 13. Limitations — what to claim and what not to

1. Accuracy is measured on RDD2022 (India, Japan, US), **not yet on our own Indian road footage**.
2. Pothole AP50 is 0.538, below the 0.55 target; the cause is localisation.
3. Severity is defaulted (no depth or crack width from a box), so PCI is a defect-density index.
4. Location is as good as phone GPS (±5 m outdoors); one position per frame, not per object.
5. Trip distance overestimates with noisy GPS (simulated 2 km: +4% at 2 m noise, +68% at 8 m), so chainage is approximate.
6. The Kalman filter and gate are unit-tested, not yet measured on a surveyed route.
7. Tested with one phone and one laptop, stationary and on a hotspot; not yet on a moving vehicle or with several devices.
8. Filming a laptop screen is harder for the model than a real road (glare, moiré): use a printout for demos.

**Do not claim:** mAP 82%, 35 FPS, HTTP REST transport, sub-metre GPS, KDE
heatmaps, UTM projection, a measured 99% bandwidth saving, or that it works on
Indian roads. None of these are true of this system.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `zsh: unknown file attribute` / `No rule to make target '#'` | Pasted a trailing `# comment` | Paste the command alone |
| Mosquitto: "Address already in use" | Old broker running | `pkill mosquitto` (or `brew services stop mosquitto`, `docker stop roadscope-mosquitto`), start again |
| Server: "no broker at tcp://localhost:1883" | Broker not running | Start terminal 1 first |
| `camera check: FAILED … different networks` | Phone on mobile data or other Wi-Fi | Join the Mac to the phone's hotspot; use the IP from `ipconfig getoption en0 router` |
| Agent: "model sees … but there is NO GPS fix yet" | No accepted fix | Start GPSLogger; add `--gps-reuse-last`; check `http://<MAC-IP>:8765/gps` |
| Agent: "GPS fix rejected (weak_accuracy …)" | Indoors | `--gps-max-accuracy 100` |
| `/gps` page shows `received: 0` | Firewall or wrong URL | Allow Python incoming connections (macOS prompt / Firewall settings); check MAC-IP in GPSLogger |
| GPS arrives rarely | GPSLogger interval/battery | Interval 1 s, distance 0, battery Unrestricted |
| Only weak "D00" detections on a pothole photo | Screen glare, pothole too small in frame | Fill half the view, cut glare, use a printout; confirm the image in `make playground` |
| Many duplicate detections | Camera pointed at the dashboard showing pothole photos | Keep the dashboard out of the camera's view |
| Dashboard unchanged after an update | Old Streamlit still running | Ctrl+C every dashboard, start one, close stale tabs |
| Dashboard "No surveys recorded yet" | Empty database | `make seed` |

---

## 15. Project history

**Prototype (deleted, recoverable at commit `a4008379` / tag `pre-rebuild-cleanup`).**
It did not work: the shipped weights were single-class COCO weights and the
"cracks" were fabricated from OpenCV contours; GPS was a hand-written CSV with no
filtering; there was no live video path (Streamlit uploads only); nothing was
persisted or transmitted; PCI used invented deduct curves over the whole road
instead of per sample unit; and the repository had grown to 402 MB from
committed weights.

**Rebuild (Oct 2026), seven gated phases:**

| Phase | Objective | Gate | Lead |
| --- | --- | --- | --- |
| 1 Foundation | Package, start training | Installs; training running | Shreyan, Vishnu |
| 2 Vertical slice | Prove the architecture | Phone video → detection → dashboard | Jadeja |
| 3 Detection | Model trained on RDD2022 | mAP50 ≥ 0.60, D40 AP50 ≥ 0.55 | Vishnu |
| 4 Position | Defensible GPS | Drift < 5% over 2 km | Ainessh |
| 5 Fusion & transport | 1 pothole = 1 record, reliable link | 60 s outage → zero loss | Jadeja |
| 6 Civil | Real ASTM PCI | Worked example reproduced | Udayvardhan, Shreyan |
| 7 Hardening | Measure, document, demo | Field tests, demo recorded | All |

**Live demo bring-up (26 Oct 2026)** fixed: MQTT client passing the full URL as
the host name (never connected); dashboard crashing on `sqlite3.Row.get` and
rendering twice; GPS push receiver added for GPSLogger; detection photo now sent
to and stored by the server; box drawn at the wrong scale on thumbnails;
thumbnails overwritten between runs; indoor GPS fixes silently rejected so every
detection was dropped (now logged, with `--gps-reuse-last`).
