# RoadScope — Rebuild Implementation Plan

> **Status:** Draft for approval · **Version:** 2.1 (phased implementation spec) · **Branch:** `main`
> **Supersedes:** the architecture described in `new.md` §2 and the prototype code in `app/`
> **Repo:** `Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection` · Python 3.11.15 · main branch

---

## 0. The plan in one page

Seven gated phases across 3 weeks, 5 people. **No phase starts until the previous phase's exit gate passes.** The single exception: **Phases 3 and 4 run concurrently** (both D3–8) — different owners, no dependency between them.

| Phase | Window | Objective | Gate (binary, checkable) | Leads |
| --- | --- | --- | --- | --- |
| **Phase 1** Foundation & Kick-Off | D1–2 | Break the monolith; start the long pole | Package installs; training **in flight** | Shreyan, Vishnu |
| **Phase 2** Vertical Slice | D2–3 | Prove the architecture end-to-end | Live phone video → detection → dashboard | Jadeja |
| **Phase 3** Detection Accuracy | D3–8 ∥ | Model that knows RDD2022 | mAP50 ≥ 0.60, D40 AP50 ≥ 0.55 | Vishnu |
| **Phase 4** Position Integrity | D3–8 ∥ | GPS you can defend with numbers | Drift < 5% over 2 km | Ainessh |
| **Phase 5** Fusion & Transport | D6–10 | 1 pothole = 1 record; reliable link | 60 s outage → **zero loss** | Jadeja |
| **Phase 6** Civil Certification | D8–11 | Replace fake numbers and fake map | ASTM worked example reproduced | Udayvardhan, Shreyan |
| **Phase 7** Hardening & Defence | D11–15 | Prove it, measure it, document it | All field tests pass; demo recorded | all |

**Three things that decide whether this works:**

1. **Launch GPU training on day 1.** It is the only task that cannot be compressed later. It waits on external quota.
2. **Phase 2 deliberately uses the bad weights.** Accuracy is irrelevant in the vertical slice — good weights would hide integration bugs behind a plausible-looking result.
3. **Phase 3 has no float.** The pothole class is the rarest in RDD2022 (6,544 instances), so plan for a fourth training run rather than discovering the need at D8.

Total float in the plan: **~4 days, all of it in Phase 7.**

---

## 1. Purpose and how to use this document

This is the working spec for rebuilding RoadScope so that the four reported failures are genuinely fixed:

| # | Reported failure | Root cause | Fixed by | Phase |
| --- | --- | --- | --- | --- |
| 1 | Model doesn't recognise potholes accurately | Never trained on RDD2022; class output is fabricated | W1, W2 | Phase 3 |
| 2 | GPS is not accurate | There is no GPS — synthetic CSV, no filtering | W3 | Phase 4 |
| 3 | Doesn't take live video feed | Streamlit-only, no camera path | W4 | Phase 2 |
| 4 | Can't transmit data anywhere | In-process HTTP receiver, nothing persisted or sent | W6 | Phase 2, Phase 5 |

**Read in this order:** §0 the one-page phase summary → §2 diagnosis (why anything is being rebuilt) → §3 target architecture → **§15 the phase plan** (gates, contingencies, day-by-day) → §12–14 the contracts, wire formats and tests everything is written against. §17 is the file-by-file migration map for whoever does the deleting.

### 1.1 Constraints and decisions

| Decision | Value | Rationale |
| --- | --- | --- |
| Capture hardware | Laptop + phone | No hardware budget |
| Model strategy | Fine-tune YOLOv8 on RDD2022, cloud GPU | Only real fix for recall |
| Transport | MQTT + FastAPI receiver | Handles flaky vehicle links properly |
| Timeline | ~3 weeks, academic review milestone | Hard constraint |
| Training hardware | Kaggle/Colab free GPU | Local box is 8 GB RAM, CPU-only |
| Model export | GitHub Release, not git | Prevents another 20 MB history entry |
| Broker | Local Mosquitto via Docker | No third-party dependency at demo time |

### 1.2 Non-goals

Explicitly out of scope for this milestone, so the team does not drift:

- PostGIS + GeoServer production geoportal (architecture permits it; SQLite ships)
- Native Android/iOS app store release (Stage B is an internal test build)
- Multi-vehicle fleet management
- Road rutting / other distress classes beyond D00/D10/D20/D40
- Offline tile rendering (tiles stay network-dependent, rendered server-side for the viewer)

---

## 2. Diagnosis — verified root causes

Every claim here was reproduced by running the current code, not inferred from reading it.

### 2.1 Detection accuracy

| Root cause | Evidence |
| --- | --- |
| Shipped weights are **single-class**, not RDD2022 | `YOLO('models/pothole_yolov8.pt').names == {0: '0'}` |
| Multi-class output is **fabricated** | `detector.py:132` — when `len(model_names)==1`, OpenCV contour "cracks" are appended and relabelled D00/D10/D20 |
| Class mapper is dead code | `detector.py:103` maps `cls_id=0` → `"D00"`, then the `len==1` branch on line 105 unconditionally overwrites it to `"D40"` |
| **No dedup between detector stages** | One drawn pothole returns `D40 bbox=[268,324,380,397]` *and* `D20 bbox=[264,320,384,410]` — same object, double-counted, double-penalised in PCI |
| Lane markings read as structural failure | Both painted curb lines detect as `D20 Alligator` at conf 0.72 |
| Severity derived from **bbox area** | `detector.py:113,185` — any box >4% of frame is "High". The 6 sample defects produce **PCI 34.6 "Very Poor"** |
| **No evaluation exists** | No validation split, no mAP, no precision/recall anywhere in the repo |

### 2.2 GPS accuracy

There is no GPS implementation. `data/sample_gps_track.csv` is 21 hand-written rows describing a near-straight line.

- `accuracy` is read from the phone (`telemetry.py:38`) then **discarded** — never stored, never used to gate a fix
- No HDOP, satellite-count or fix-age gating
- No noise filtering; raw fixes pass through unfiltered
- `interpolate_gps` (`telemetry.py:95`) linearly interpolates lat/lon — geometrically wrong on turns, cuts corners
- No `haversine` anywhere in the repo, so speed-derived distance is never cross-checked
- Phone↔laptop clock skew unhandled, yet matching depends on wall-clock timestamps
- No map matching, so urban lateral error is never corrected and there are no road names

### 2.3 Live video

- `dashboard.py` accepts **only** file uploads (line 246) or a procedurally drawn synthetic frame (lines 204–228). No camera path exists.
- Streamlit re-executes the whole script per interaction. It cannot host a 30 fps stream.
- Consequence: `RoadDamageDetector()` is constructed at **both** `dashboard.py:163` and `:194` — the 22 MB model loads twice per rerun.

### 2.4 Transmission

- `telemetry.py:17-51` — in-memory list behind a bare `HTTPServer`. **Nothing is persisted.**
- Receives only; there is no outbound code path in the repo.
- `st.session_state` is used as the database (`dashboard.py:129`) — lost on refresh, invisible to a second viewer, not queryable.

### 2.5 Additional correctness bugs

| Bug | Location | Effect |
| --- | --- | --- |
| Empty-ledger `KeyError` | `pci_engine.py:107-116` omits `deduct_values`/`severity_breakdown`; read at `dashboard.py:537` | Tab 3 **crashes** when the ledger is empty |
| CWD-relative paths | `detector.py:50`, `dashboard.py:131` | Launched from any other directory the app **silently** drops to OpenCV *and* starts with zero defects → PCI 100 "Good" on a blank map |
| Invented DV curves | `pci_engine.py:45-70` uses `a + b·log₁₀(d)` | Not ASTM D6433; produces nonsense scores |
| Global PCI | `pci_engine.compute_pci_score` | ASTM requires per-sample-unit (250–500 m²) scoring |
| Unbounded ledger growth | `dashboard.py:293,321` appends on every click, never dedups | ~5 records/sec per pothole at 5 fps |
| No OPTIONS handler | `telemetry.py` | Browser CORS preflight fails |
| Dead repo weight | `.git` = 402 MB for ~21 MB of content | ~36,000 unreachable blobs never gc'd |
| Personal name hardcoded | `work_order.py:95` default `surveyor_name` | Library default is a specific person |

---

## 3. Target architecture

The central change: **the edge agent leaves Streamlit.** Streamlit becomes a read-only client of a database.

```
┌─ VEHICLE (laptop + phone) ─────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐   MJPEG / MQTT   ┌─────────────────────────┐  │
│  │ Phone        │─────────────────▶│  EDGE AGENT (process)   │  │
│  │  • camera    │  JPEG QoS0       │                         │  │
│  │  • GPS 1–5Hz │──MQTT QoS1──────▶│  ingest ─┐              │  │
│  └──────────────┘                  │          ▼ bounded q   │  │
│                                     │  infer (YOLO seg)     │  │
│                                     │  calibrate + measure   │  │
│                                     │  track + dedup         │  │
│                                     │  outbox → publisher    │  │
│                                     └───────────┬─────────────┘  │
└─────────────────────────────────────────────────│────────────────┘
                                   MQTT QoS1, buffered, resumable
                    ┌──────────────────────────────▼───────────────┐
                    │ Mosquitto broker (local Docker, :1883)       │
                    └──────────────────────────────┬───────────────┘
                                   ┌───────────────▼───────────────┐
                                   │ SERVER                        │
                                   │  subscriber → FastAPI         │
                                   │  → SQLite (WAL) = truth       │
                                   │  → ASTM per-sample-unit PCI   │
                                   └───────────────┬───────────────┘
                                   ┌───────────────▼───────────────┐
                                   │ Streamlit dashboard (read-only)│
                                   └───────────────────────────────┘
```

### 3.1 Design principles

1. **The database is the contract.** Edge and UI never call each other.
2. **Never block ingest.** Inference sits downstream of a bounded, drop-oldest queue. A slow detector degrades frame rate; it never stalls capture. Frame-drop % is a visible metric — honest, and good viva material.
3. **Every number is traceable.** Each detection carries its source thumbnail, GPS accuracy, and model version.
4. **Offline-first.** Vehicles lose signal. The outbox means nothing is lost, ever.
5. **Measure, don't assert.** Every accuracy claim needs a held-out number behind it.

---

## 4. Repository layout

```
roadscope/
├── pyproject.toml                 # deps, ruff, pytest config
├── docker-compose.yml             # mosquitto (+ optional postgres)
├── .env.example                   # documented config template
├── README.md
├── REBUILD_PLAN.md                # this document
├── src/roadscope/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings, all env config
│   ├── logging_setup.py           # structured logging
│   ├── paths.py                   # Path(__file__)-anchored path resolution
│   ├── geo/
│   │   ├── __init__.py
│   │   ├── quality.py             # fix gating (W3.1)
│   │   ├── kalman.py              # constant-velocity filter (W3.2)
│   │   ├── mapmatch.py            # osmnx road snapping (W3.3)
│   │   ├── distance.py            # haversine + odometer (W3.4)
│   │   ├── clock.py               # skew estimation, mono time (W3.5)
│   │   └── ringbuffer.py          # timestamped GPS ring buffer
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py            # YOLO seg wrapper, singleton
│   │   ├── calibrate.py           # ground-plane homography (W2.1)
│   │   ├── measure.py             # mask → length/width/area (W2.2)
│   │   └── annotate.py            # HUD + bbox rendering
│   ├── civil/
│   │   ├── __init__.py
│   │   ├── distress.py            # RDD2022 class schema
│   │   ├── severity.py            # ASTM severity thresholds (W2.3)
│   │   ├── dv_tables.py           # published DV lookup tables (W2.4)
│   │   ├── cdv.py                 # corrected DV curve (W2.4)
│   │   └── pci.py                 # per-sample-unit PCI (W2.5)
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── tracker.py             # ByteTrack-style association (W5.1)
│   │   ├── geofence.py            # spatial suppression (W5.3)
│   │   └── fuser.py               # frame → defect record
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── mqtt_client.py         # publisher + LWT status (W6.2)
│   │   ├── outbox.py              # SQLite outbox (W6.3)
│   │   └── topics.py              # topic builders, payload schemas
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── schema.sql             # DDL (W6.5)
│   │   ├── db.py                  # connection, WAL, migrations
│   │   └── repository.py          # typed CRUD
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── ingest.py              # camera + GPS readers
│   │   ├── pipeline.py            # bounded-queue orchestration
│   │   ├── health.py              # heartbeat metrics
│   │   └── __main__.py            # python -m roadscope.agent
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI
│   │   └── subscriber.py          # MQTT → DB ingest
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py           # Streamlit (read-only)
├── scripts/
│   ├── download_weights.sh        # fetch best.pt from Release
│   ├── train_cloud.md            # Kaggle/Colab runbook
│   └── field_test.py              # field protocol harness (§14)
├── models/                        # gitignored; fetched by script
├── tests/
│   ├── test_paths.py
│   ├── test_civil.py
│   ├── test_geo.py
│   ├── test_fusion.py
│   ├── test_transport.py
│   └── test_vision.py
└── docs/
    ├── architecture.md
    ├── astm_notes.md
    └── results.md
```

---

## 5. Workstream W1 — Detection model

**Owner:** Vishnu Nair (23BEL1028) · **Depends on:** nothing · **Blocks:** W2, W4, W5

### 5.1 Dataset acquisition

Source: figshare `10.6084/m9.figshare.21431547.v1`, also `github.com/sekilab/RoadDamageDetector`.

| Property | Value |
| --- | --- |
| Total images | 47,420 |
| Annotated train | 38,385 |
| Labelled instances | 55,007 |
| Classes | `D00` Longitudinal, `D10` Transverse, `D20` Alligator, `D40` Pothole |
| Resolution | 600×600 / 720×720 |

### 5.2 Curated training mix

Per-country train counts: Japan 10,000 · India 9,000 · Norway 9,000 · US 5,000 · China 3,500 · Czech 2,000.

**Exclude China** — that subset is drone/oblique imagery, a different domain that actively degrades vehicle-mounted forward-facing inference. The dataset paper itself notes users may skip it.

Keep **India + Japan + US**: all vehicle-mounted, forward-facing, matching a phone on a dashboard.

Re-index to the canonical 4-class map. Drop `other corruption`, `block crack`, `repair`, `D01`, `D11`, `D43`, `D44`.

### 5.3 Class imbalance — the main training risk

Global instance counts:

| Class | Instances |
| --- | --- |
| D00 Longitudinal | 26,016 |
| D10 Transverse | 11,830 |
| D20 Alligator | 10,617 |
| **D40 Pothole** | **6,544** ← rarest, and the class this project is named for |

Per-country D40/D10 highlights:

| Country | D40 | D10 |
| --- | --- | --- |
| India | 3,187 | **68** |
| Japan | 2,243 | 3,979 |

India — the relevant domain for this project — has 68 transverse-crack labels against 3,187 potholes. Expect D10 recall to collapse. Mitigation: oversample D00/D10, cap D00, use `copy_paste` augmentation, and a pothole-weighted sampler. Always report **per-class AP**, never a single averaged mAP.

### 5.4 Model and training config

Use **`YOLOv8m-seg`** — segmentation, not detection alone, because the mask is what enables real crack width/length measurement in W2, which is what makes ASTM severity defensible.

```yaml
# models/rdd2022_roadscope.yaml — dataset config
path: /kaggle/input/rdd2022-curated
train: images/train
val: images/val
names:
  0: D00   # Longitudinal Crack
  1: D10   # Transverse Crack
  2: D20   # Alligator Crack
  3: D40   # Pothole
```

```python
# train config highlights
model = "yolov8m-seg.pt"
epochs = 80
imgsz = 960              # RDD2022 native is 600-720; upscale to catch thin cracks
batch = 8
device = "cuda"
patience = 15

hsv_h = 0.015            # lighting varies hugely on Indian roads
hsv_s = 0.7
hsv_v = 0.4
degrees = 10.0
translate = 0.1
scale = 0.5
fliplr = 0.5             # horizontal only — roads have no vertical flip
fliud = 0.0
mosaic = 1.0
mixup = 0.1
copy_paste = 0.1         # pothole oversampling
erasing = 0.2
```

Split: 80/20 stratified by country, held out **before** any augmentation.

### 5.5 Evaluation harness — a required deliverable

- Held-out validation split; report per-class AP50 and AP50-95, precision, recall, confusion matrix
- **In-domain test set: ≥150 frames of your own dashcam video, hand-labelled.** RDD2022 mAP will not predict performance on your roads; this set is the only number that will
- Lane-marking false-positive rate measured separately on paved-marking frames
- Write `metrics.json` per run; chart across runs to show iteration

### 5.6 Deletions

Remove from the live inference path:

- `app/detector.py:91-138` — heuristic class mapping and the `len(model_names)==1` remap
- `app/detector.py:130-136` — OpenCV crack supplement that fabricates D00/D10/D20
- `app/detector.py:113,185` — area-based severity

Retain the OpenCV extractor only as an explicitly-labelled offline demo mode, not a silent fallback.

### 5.7 Acceptance criteria

- [ ] Overall mAP50 ≥ 0.60 (challenge winner achieved F1 76.9%, so this is realistic)
- [ ] D40 pothole AP50 ≥ 0.55
- [ ] Pothole recall ≥ 0.70 on the in-domain set
- [ ] Lane markings produce no pothole false positives at the operating threshold
- [ ] `model.names` used verbatim; no inference-time class remapping
- [ ] `metrics.json` committed per run

---

## 6. Workstream W2 — Measurement and ASTM D6433

**Owner:** Udayvardhan Singh Rathore (23BEL1018) · **Depends on:** W1 (masks) · **Blocks:** PCI output

### 6.1 Ground-plane calibration

Turn pixels into metres. Currently impossible.

1. Fix the phone mount; record camera height `h` (m) and horizontal FOV → pinhole intrinsics `K`
2. Detect lane markings in-frame; estimate the vanishing point `vp`
3. Solve homography `H: image_plane → road_plane` from `K`, `h`, `vp`
4. Expose `metres_per_pixel(y)` per image row

Cross-check against a known reference: measure a 1 m object at a fixed distance.

### 6.2 Geometry from segmentation masks

- Skeletonise the mask → crack **length** in metres
- Distance transform → mean **width** in millimetres
- Pothole: convex-hull area in m², equivalent diameter in metres

### 6.3 Severity per ASTM D6433

Severity is defined by **width and length thresholds per distress type**, not by any percentage of frame. Encode the published thresholds as data tables, not magic numbers. Reference structure:

```python
# civil/severity.py
# Structure mirrors the ASTM D6433 severity criteria tables.
SEVERITY_CRITERIA = {
    "D00": {  # longitudinal crack
        "L": {"width_mm_max": 3,  "length_m_max": 3.0},
        "M": {"width_mm_max": 20, "length_m_max": 8.0},
        "H": {"width_mm_min": 20, "length_m_min": 8.0},
    },
    "D10": {...},  # transverse crack
    "D20": {...},  # alligator — area driven
    "D40": {...},  # pothole — equivalent diameter driven
}
# Threshold values to be transcribed from the standard; verify against the
# published worked example before sign-off.
```

### 6.4 Real deduct values

Delete `app/pci_engine.py:45-70` (`a + b·log₁₀(density)`).

Replace with:

1. **Published ASTM D6433 deduct-value lookup tables**, one per distress type, indexed by density
2. **The corrected DV (CDV) curve** from the standard's correction graph, digitised into a table with linear interpolation
3. Keep the `q > 5` count logic, but validate against the standard's worked example

### 6.5 Sample units

Walk the corridor by chainage; cut into **250–500 m² sample units**; compute density and PCI **per unit**; average for the corridor score.

Densities must be dimensionally correct:

| Distress | Density definition |
| --- | --- |
| D00, D10 (linear cracks) | linear metres / m² |
| D20 (alligator, area) | m² / m² |
| D40 (pothole) | count / m² |

### 6.6 Bug fix

`civil/pci.py` must return a **uniform dict shape always**, including for empty input:

```python
PCIResult = TypedDict("PCIResult", {
    "pci": float, "rating": str, "color": str, "cdv": float, "tdv": float,
    "defect_breakdown": dict, "severity_breakdown": dict,
    "deduct_values": list, "recommendation": str,
    "sample_units": list,
})
```

This is the `pci_engine.py:107-116` crash, permanently guarded by `tests/test_civil.py::test_empty_input_uniform_shape`.

### 6.7 Acceptance criteria

- [ ] Severity responds to measured mm/m — widening a crack measurably changes the grade
- [ ] Reproduces the ASTM worked example within tolerance
- [ ] Per-sample-unit PCI rendered as a coloured polyline on the map, plus a corridor average
- [ ] No `KeyError` on empty input (regression-tested)

---

## 7. Workstream W3 — GPS accuracy

**Owner:** Ainessh Kumar S (23BEL1039) · **Depends on:** none · **Blocks:** W4 (geo-tagging), W6 (payload)

### 7.1 Fix-quality gating

Accept a fix only when all hold; otherwise mark `stale` and hold the last good position.

| Condition | Threshold | Rationale |
| --- | --- | --- |
| `accuracy_m` | ≤ 10 | Reject urban multipath |
| `n_sats` | ≥ 6 | Minimum for a 2D fix |
| `hdop` | ≤ 2.0 | Reject poor geometry |
| fix age | ≤ 2 s | Reject stale positions |

Persist `hdop`, `n_sats`, and a `quality_flag` on every stored fix.

### 7.2 Kalman filter

Constant-velocity filter over lat/lon, ~40 lines of numpy. Largest single accuracy win against jitter. Replaces the straight-line `interpolate_gps`.

### 7.3 Map matching with osmnx

Snap the filtered trace to the nearest OSM way within a radius. Three returns:

1. Lateral error correction — the trace sits on the road, not 8 m beside it
2. **Real road names**, auto-filled into work orders
3. Road class, usable to weight severity

This is the single most defensible GIS addition in the plan.

### 7.4 Distance cross-check

Implement `haversine`. Compare consecutive-fix distance against `speed × dt` integrated over the trip; log drift. GPS that silently drifts is worse than GPS that reports its own error.

### 7.5 Timestamp discipline

- Publish **both** monotonic elapsed time and UTC epoch, plus a phone-assigned `seq`
- Edge agent estimates phone↔laptop clock skew once at startup, then corrects
- Interpolate on the **server**, from a GPS ring buffer, not on the phone's raw stream

### 7.6 Heading from the track

Bearing between consecutive filtered fixes, smoothed over a short window. Do not trust the phone's reported heading while stationary.

### 7.7 Acceptance criteria

- [ ] Before/after scatter plot of raw vs map-matched track from the same drive
- [ ] Every detection row carries `accuracy_m`, `hdop`, `n_sats`, `mapmatch_conf`, `road_name`
- [ ] Distance cross-check drift < 5% over a 2 km run
- [ ] Work orders auto-populate the OSM road name

---

## 8. Workstream W4 — Live video and edge agent

**Owner:** Jadeja Suryadeepsinh Bharatsinh (23BEL1027) · **Depends on:** W1 (weights) · **Blocks:** W5, W6

### 8.1 Stage A — pixels moving (days 1–2, no app development)

```
Phone runs DroidCam or IP Web Camera  →  serves MJPEG on LAN
Laptop (phone hotspot)               →  cv2.VideoCapture("http://<phone-ip>:8080/stream")
```

Proves capture, decode and inference in an afternoon. This is a complete, shippable video path on its own.

### 8.2 Stage B — one link for everything (bonus, week 3, only if weeks 1–2 are green)

Minimal Flutter app: camera frames and GPS over the same MQTT broker. Video as JPEG at ~4 fps / 640 px, QoS0 (fine on WiFi). Keep MJPEG as the bandwidth fallback.

Stronger story — video, position and telemetry over a single broker — but explicitly **not a blocker**.

### 8.3 Edge agent pipeline

A standalone process. Not Streamlit.

| Stage | Responsibility | Non-negotiable behaviour |
| --- | --- | --- |
| `ingest` | Read frames + GPS; stamp each frame with interpolated position | Never blocks |
| `infer` | YOLO seg, batched | Never called from the UI thread |
| `calibrate` | Ground-plane homography per frame | Recompute when camera pose changes |
| `fuse` | Track, dedup, emit one record per real defect | Idempotent per track |
| `publish` | Drain SQLite outbox → MQTT | Commits before publishing |
| `health` | Heartbeat: fps, inference ms, queue depth, **drop %**, GPS quality | Every 5 s + LWT |

**Queue policy:** `queue.Queue(maxsize=8)`, **drop-oldest**. Stale frames are worthless; a full queue must never stall the camera reader.

**Other agent requirements:**

- Detector loaded **once** as a module singleton — fixes the `dashboard.py:163`/`:194` double-load
- Config from `.env` via `pydantic-settings`: `DEVICE_ID`, `MQTT_URL`, `CAMERA_URL`, `MODEL_PATH`, `CONF_THRESHOLD`, `OUTBOX_DB`
- Graceful shutdown flushes the outbox before exit
- Structurally identical ingest API for MJPEG, MQTT-JPEG and file sources, so W4 testable without a phone

### 8.4 Acceptance criteria

- [ ] Sustained ≥15 fps ingest with inference not blocking it, drop rate < 10%
- [ ] Detector instantiated exactly once (asserted in a test)
- [ ] Killing the agent mid-survey loses nothing on restart (outbox replay)
- [ ] Agent runs headless, with no Streamlit import in its dependency graph

---

## 9. Workstream W5 — Tracking and de-duplication

**Owners:** Jadeja (23BEL1027) + Vishnu (23BEL1028) · **Depends on:** W4

Entirely absent today, and a correctness hole: at 5 fps the current code emits ~5 records per second per pothole.

### 9.1 Association

ByteTrack-style: IoU + centroid distance + class match across consecutive frames, with a two-stage high/low-confidence split.

### 9.2 Track confirmation

A track becomes a defect after K frames or T seconds of persistence. Discard the rest.

### 9.3 Geofence suppression

Once confirmed, do not re-emit while the vehicle is within ~15 m. Prevents a second pass over the same street creating duplicate records.

### 9.4 Best-frame selection

Keep the highest-confidence frame of the track. Store its thumbnail. Record `frames_seen` and `track_duration_s` — a defect seen in 14 frames is stronger evidence than one seen once, and the system should be able to say so.

### 9.5 Measure once

Run geometry measurement on the **best frame only**, not per frame.

### 9.6 Acceptance criteria

- [ ] One pothole at 20 km/h → exactly **1** record, `frames_seen > 3`
- [ ] Two adjacent potholes → 2 records, not 1 merged and not 6
- [ ] Second pass over the same street → 0 new records

---

## 10. Workstream W6 — Transmission and server

**Owners:** Jadeja (23BEL1027) + Shreyan Biswas (23BEL1052) · **Depends on:** W3, W4, W5

### 10.1 Broker

Local Mosquitto in Docker. Not a public broker — no third-party dependency, nothing leaves the machine, and a lab-WiFi demo cannot be disrupted by an outsider. (Public brokers are reachable from here; this is a preference, not a necessity.)

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
```

### 10.2 Topic scheme

| Topic | QoS | Retained | Payload |
| --- | --- | --- | --- |
| `roadscope/{device_id}/gps` | 1 | no | lat, lon, speed_kmh, heading, accuracy_m, hdop, n_sats, seq, ts_utc, t_mono |
| `roadscope/{device_id}/detection` | 1 | no | §10.2.1 record |
| `roadscope/{device_id}/status` | 1 | yes (LWT) | fps, drop_pct, queue_depth, infer_ms, model_version, agent_version |
| `roadscope/{device_id}/cmd` | 1 | no | threshold, start/stop, model, camera_url |

#### 10.2.1 GPS payload

```json
{
  "device_id": "van-01",
  "seq": 10432,
  "ts_utc": 1760000000.123,
  "t_mono": 812.44,
  "lat": 12.972240,
  "lon": 79.159230,
  "speed_kmh": 29.5,
  "heading": 47.2,
  "accuracy_m": 2.4,
  "hdop": 0.9,
  "n_sats": 11,
  "quality_flag": "good"
}
```

#### 10.2.2 Detection payload

```json
{
  "device_id": "van-01",
  "trip_id": "trip-2026-10-07-1530",
  "detection_id": "uuid",
  "ts_utc": 1760000008.4,
  "frame_id": "f-0004812",
  "lat": 12.972240, "lon": 79.159230,
  "gps_accuracy_m": 2.4,
  "class_code": "D40",
  "confidence": 0.91,
  "severity": "H",
  "length_m": 0.42, "width_mm": 310.0, "area_m2": 0.118,
  "frames_seen": 14, "track_duration_s": 2.8,
  "bbox": [268, 324, 380, 397],
  "road_name": "NH 44",
  "model_version": "rdd2022-yolov8m-seg-r3"
}
```

### 10.3 Outbox pattern — the reliable-link answer

The current system's transmission failure is that it never persists before sending. Fix: insert into SQLite `outbox` **before** publishing; a publisher loop drains it and marks rows sent.

Signal drops → rows accumulate → they drain on reconnect. This is what makes transmission *reliable* rather than *attempted*.

### 10.4 Server

FastAPI + MQTT subscriber → persists to SQLite in WAL mode. Zero-infra for a laptop demo, still a real queryable store. PostGIS upgrade documented in `docs/architecture.md` but not shipped.

### 10.5 Schema

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE devices (
  device_id     TEXT PRIMARY KEY,
  name          TEXT,
  last_seen     REAL,
  agent_version TEXT,
  model_version TEXT
);

CREATE TABLE trips (
  trip_id     TEXT PRIMARY KEY,
  device_id   TEXT NOT NULL REFERENCES devices(device_id),
  started_at  REAL NOT NULL,
  ended_at    REAL,
  distance_m  REAL
);

CREATE TABLE gps_fixes (
  id                 INTEGER PRIMARY KEY,
  trip_id            TEXT NOT NULL REFERENCES trips(trip_id),
  seq                INTEGER NOT NULL,
  ts_utc             REAL NOT NULL,
  lat                REAL NOT NULL,
  lon                REAL NOT NULL,
  speed_kmh          REAL,
  heading            REAL,
  accuracy_m         REAL,
  hdop               REAL,
  n_sats             INTEGER,
  quality_flag       TEXT,
  road_name          TEXT,
  mapmatch_conf      REAL,
  UNIQUE(trip_id, seq)              -- idempotent replay
);
CREATE INDEX idx_gps_trip_ts ON gps_fixes(trip_id, ts_utc);

CREATE TABLE frames (
  frame_id   TEXT PRIMARY KEY,
  trip_id    TEXT NOT NULL REFERENCES trips(trip_id),
  ts_utc     REAL NOT NULL,
  lat        REAL,
  lon        REAL,
  thumb_path TEXT
);
CREATE INDEX idx_frames_trip_ts ON frames(trip_id, ts_utc);

CREATE TABLE detections (
  detection_id    TEXT PRIMARY KEY,
  trip_id         TEXT NOT NULL REFERENCES trips(trip_id),
  frame_id        TEXT REFERENCES frames(frame_id),
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
CREATE INDEX idx_det_trip_ts ON detections(trip_id, ts_utc);
CREATE INDEX idx_det_class ON detections(class_code);

CREATE TABLE sample_units (
  unit_id         TEXT PRIMARY KEY,
  trip_id         TEXT NOT NULL REFERENCES trips(trip_id),
  chainage_start_m REAL,
  chainage_end_m   REAL,
  area_m2          REAL,
  tdv              REAL,
  cdv              REAL,
  pci              REAL,
  rating           TEXT
);

CREATE TABLE outbox (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  topic      TEXT NOT NULL,
  payload    TEXT NOT NULL,
  created_at REAL NOT NULL,
  sent_at    REAL,
  attempts   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_outbox_pending ON outbox(sent_at) WHERE sent_at IS NULL;
```

`UNIQUE(trip_id, seq)` on `gps_fixes` is what makes broker redelivery idempotent under QoS1.

### 10.6 Highest-value UI feature

`frames` enables **click a map pin → see the actual road photo**, with its measurement, severity and GPS accuracy. Today no detection carries any evidence. This is the single change that makes the geoportal convincing rather than decorative.

### 10.7 Acceptance criteria

- [ ] Kill WiFi for 60 s mid-survey, reconnect → **zero detections lost**
- [ ] A second browser session sees the same live data (proves the DB is the source of truth)
- [ ] Click any pin → thumbnail, severity, dimensions, GPS accuracy
- [ ] Broker redelivery does not create duplicate rows

---

## 11. Workstream W7 — Repo, tooling, CI

**Owner:** Shreyan Biswas (23BEL1052) · **Depends on:** all

1. **Proper package.** `src/roadscope/` + `pyproject.toml`. Delete the `sys.path` hacks at `dashboard.py:17-24` — they patched the symptom of bad layout.
2. **Paths from `Path(__file__).resolve().parent`,** never CWD-relative. `tests/test_paths.py` chdirs to `/tmp` and asserts model and dataset still resolve — a permanent guard on the §2.5 silent-downgrade bug.
3. **Weights out of git.** Upload `best.pt` to a GitHub Release; `scripts/download_weights.sh` fetches and caches into `models/`. Do not add another 20 MB history entry.
4. **`git gc --prune=now --aggressive`** to reclaim ~380 MB — run last, after a `git bundle` backup.
5. **Tests** per §13.
6. **`ruff`** for lint and format. **GitHub Actions** running ruff + pytest + a smoke test that boots the agent against a recorded sample.
7. **Docker Compose** for broker (+ optional Postgres) so the demo is one command.
8. **Structured logging** to file + console, with trip IDs, so a failed demo is diagnosable.

---

## 12. Module contracts

Signatures the implementers code against.

```python
# config.py
class Settings(BaseSettings):
    device_id: str = "van-01"
    mqtt_url: str = "tcp://localhost:1883"
    camera_url: str | None = None          # None => synthetic/test source
    model_path: Path
    conf_threshold: float = 0.35
    outbox_db: Path
    server_db: Path
    queue_size: int = 8
    infer_target_ms: int = 120             # 8 fps inference budget
    geofence_m: float = 15.0
    track_min_frames: int = 3

# geo/quality.py
def gate_fix(fix: GPSFix, max_accuracy_m: float = 10.0,
             min_sats: int = 6, max_hdop: float = 2.0,
             max_age_s: float = 2.0) -> tuple[bool, str]

# geo/kalman.py
class LatLonKalman:
    def update(self, fix: GPSFix) -> GPSFix: ...   # returns filtered lat/lon

# geo/mapmatch.py
def snap_to_network(lat: float, lon: float, radius_m: float = 30.0
                    ) -> MapMatchResult | None

# geo/distance.py
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float
def odometer_distance_m(fixes: list[GPSFix]) -> float

# geo/ringbuffer.py
class GPSRingBuffer:
    def append(self, fix: GPSFix) -> None: ...
    def position_at(self, t_mono: float) -> GPSFix | None: ...

# vision/calibrate.py
def ground_plane_homography(frame: np.ndarray, cam_height_m: float,
                            hfov_deg: float) -> np.ndarray | None
def metres_per_pixel(H: np.ndarray, y: int) -> float

# vision/measure.py
@dataclass
class DefectGeometry:
    length_m: float
    width_mm: float
    area_m2: float
    equiv_diameter_m: float

def measure(mask: np.ndarray, H: np.ndarray) -> DefectGeometry

# vision/detector.py
class DefectDetector:                          # module-level singleton
    @classmethod
    def instance(cls, model_path: Path) -> "DefectDetector": ...
    def infer(self, frame: np.ndarray, conf: float) -> list[Detection]: ...

# civil/severity.py
def classify_severity(class_code: str, geom: DefectGeometry) -> "L" | "M" | "H"

# civil/dv_tables.py
def deduct_value(class_code: str, severity: str, density: float) -> float
def interpolate_dv_table(table: np.ndarray, density: float) -> float

# civil/cdv.py
def corrected_deduct_value(deduct_values: list[float]) -> float

# civil/pci.py
def compute_sample_unit_pci(defects: list[Defect], area_m2: float) -> SampleUnit
def compute_corridor_pci(units: list[SampleUnit]) -> PCIResult   # uniform shape

# fusion/tracker.py
class ByteTracker:
    def update(self, detections: list[Detection], ts: float) -> list[Track]

# fusion/geofence.py
def is_suppressed(lat: float, lon: float, confirmed: list[Defect],
                  radius_m: float) -> bool

# transport/outbox.py
class Outbox:
    def enqueue(self, topic: str, payload: dict) -> int: ...
    def pending(self, limit: int = 100) -> list[OutboxRow]: ...
    def mark_sent(self, row_id: int) -> None: ...
    def increment_attempt(self, row_id: int) -> None: ...

# agent/pipeline.py
class EdgePipeline:
    def ingest_frame(self, frame: np.ndarray, t_mono: float) -> None
    def ingest_gps(self, fix: GPSFix) -> None
    def start(self) -> None
    def stop(self) -> None
    def health(self) -> HealthStats
```

---

## 13. Test plan

| Test | Guards against |
| --- | --- |
| `test_paths.py::test_paths_independent_of_cwd` | §2.5 silent downgrade to OpenCV + empty ledger |
| `test_civil.py::test_empty_input_uniform_shape` | Tab 3 `KeyError` crash |
| `test_civil.py::test_astm_worked_example` | Invented DV curves |
| `test_civil.py::test_density_units_are_dimensional` | Linear m/m² vs m²/m² mixups |
| `test_civil.py::test_per_sample_unit_scoring` | Global instead of per-unit PCI |
| `test_civil.py::test_severity_from_measured_geometry` | Area-based severity |
| `test_geo.py::test_haversine_known_distance` | Distance maths |
| `test_geo.py::test_kalman_reduces_variance` | GPS jitter |
| `test_geo.py::test_fix_gating_rejects_bad_fixes` | Bad fixes admitted |
| `test_geo.py::test_clock_skew_correction` | Phone↔laptop drift |
| `test_fusion.py::test_single_pothole_one_record` | ~5 records/sec per pothole |
| `test_fusion.py::test_adjacent_potholes_not_merged` | Over-merging |
| `test_fusion.py::test_geofence_suppresses_repeat_pass` | Duplicate records on revisit |
| `test_transport.py::test_outbox_replay_after_disconnect` | Data loss on signal loss |
| `test_transport.py::test_duplicate_gps_seq_is_idempotent` | QoS1 redelivery duplicates |
| `test_vision.py::test_detector_singleton` | Double model load |
| `test_vision.py::test_measure_known_square` | Calibration maths |
| `test_vision.py::test_lane_marking_not_pothole` | Painted-line false positives |

Smoke test: boot the agent against a recorded 30 s sample, assert ≥1 detection end-to-end.

---

## 14. Field test protocol

Repeatable procedure for measuring claims. Run each on a 2 km urban route with mixed road condition.

| Test | Procedure | Pass criterion |
| --- | --- | --- |
| In-domain accuracy | Drive the route; save every frame the agent annotated; hand-label ≥150 frames | Pothole recall ≥ 0.70, AP50 ≥ 0.55 |
| GPS improvement | Same drive; log raw and map-matched track | Lateral error reduced; plot both |
| Distance integrity | Compare odometer integration vs haversine total | Drift < 5% |
| Ingest rate | Read `health()` over 10 min | ≥15 fps, drop < 10% |
| Dedup | Pass 1 pothole, 2 potholes, revisit street | 1, 2, 0 new |
| Link loss | Disable WiFi 60 s mid-survey, restore | 0 detections lost |
| Measurement | Measure a known 1 m object at fixed distance | Within 10% |
| End-to-end | Full drive: capture → broker → server → map → work order | Complete, recorded |

---

## 15. Phase plan

The schedule below is organised into **gated phases**. Each phase has a single objective, a set of deliverables, and one **exit gate** — a binary, checkable condition. No phase begins until the previous phase's gate passes.

Every phase also carries a **contingency**: what to do if the gate fails. A plan with gates but no fallback is half a plan.

**Guiding rule: thin vertical slice first, then widen.** One pothole, from live phone video, geo-tagged, on the dashboard, by end of day 3. Everything after that is width.

### 15.1 Phase overview

| Phase | Window | Objective | Leads | Float |
| --- | --- | --- | --- | --- |
| **Phase 1** Foundation & Kick-Off | D1–2 | Break the monolith; start the long pole | Shreyan, Vishnu | none |
| **Phase 2** Vertical Slice | D2–3 | Prove the architecture end-to-end | Jadeja | none |
| **Phase 3** Detection Accuracy | D3–8 | Model that actually knows RDD2022 | Vishnu | none — **critical path** |
| **Phase 4** Position Integrity | D3–8 ∥ | GPS you can defend with numbers | Ainessh | ~1 day |
| **Phase 5** Fusion & Transport | D6–10 | One pothole = one record; reliable link | Jadeja | ~1 day |
| **Phase 6** Civil Certification & Geoportal | D8–11 | Replace fake numbers and fake map | Udayvardhan, Shreyan | none |
| **Phase 7** Hardening & Defence | D11–15 | Prove it, measure it, document it | all | ~2 days (Stage B) |

> **Phases 3 and 4 are numbered in sequence but run concurrently** (both D3–8). The numbering reflects workstream grouping, not ordering. They have different owners and no blocking dependency between them — Phase 4 depends on nothing from Phase 3, and Phase 5 needs both.

**Total float in the plan: about 4 days, and only Phase 7 has usable float.** Phases 1, 2 and 6 have none.

### 15.2 Dependency graph

```
                    Phase 1  Foundation & Kick-Off
                             │
                             ▼
                    Phase 2  Vertical Slice
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌────────────────────┐      ┌────────────────────┐
   │ Phase 3            │      │ Phase 4            │
   │   Detection        │      │   Position         │
   │   Accuracy         │      │   Integrity        │
   │   (Vishnu)         │      │   (Ainessh)        │
   │   D3–8             │      │   D3–8             │
   │   NO FLOAT         │      │   ~1 day float     │
   └────────────────────┘      └────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   Phase 5  Fusion & Transport
                             │
                             ▼
                   Phase 6  Civil Certification & Geoportal
                             │
                             ▼
                    Phase 7  Hardening & Defence
```

### 15.3 Phase traceability

| Phase | Workstreams | Members |
| --- | --- | --- |
| Phase 1 | W1 (start), W3 (start), W7 | Shreyan, Vishnu, Ainessh |
| Phase 2 | W4, W6 (core), W7 | Jadeja |
| Phase 3 | W1, W2 | Vishnu, Udayvardhan |
| Phase 4 | W3 | Ainessh |
| Phase 5 | W5, W4, W6 | Jadeja, Vishnu |
| Phase 6 | W2, W6, W7 | Udayvardhan, Shreyan, Ainessh |
| Phase 7 | all | all |

---

### Phase 1 — Foundation & Kick-Off · D1–2

**Objective:** break the Streamlit monolith into a package, and get GPU training in flight before anything else is ready.

**Why this phase exists:** two hard facts force it to be first. (1) Every later phase writes into the new layout, so restructuring cannot be deferred without rework. (2) Training is the only task in the plan that cannot be compressed later — it needs GPU quota, and quota is not guaranteed. Starting it on D1 while the repo is still being restructured is the single most important scheduling decision in this document.

**Deliverables**

- `pyproject.toml`, `src/roadscope/` package skeleton, `config.py`, `logging_setup.py`, `paths.py`
- `paths.py` resolving model and data from `Path(__file__)` — closing the CWD silent-downgrade bug
- CI skeleton: ruff + pytest running on every push
- RDD2022 downloaded, curated (India + Japan + US, **China excluded**), `dataset.yaml` written
- **Training run 1 launched on cloud GPU**
- GPS quality gating + Kalman filter started (§7.1–7.2)

**Exit gate**

- [ ] `pip install -e .` completes clean; `python -m roadscope.agent --help` runs
- [ ] `test_detector_singleton` passes — model loads exactly once
- [ ] `test_paths_independent_of_cwd` passes
- [ ] Training run 1 is **in flight with training loss decreasing** (not merely queued)

**Contingency if gate fails**

- *No GPU quota on D1:* switch to Colab the same day. Last resort is a CPU overnight run on a YOLOv8s subset. **Hard rule: training must be launched by end of D2.** Slipping it past D2 slips the entire plan.
- *Restructure overruns:* cut `logging_setup.py` and CI skeleton from this phase. Neither is on the critical path.

**Owners:** Shreyan (package, CI, paths) · Vishnu (dataset, training launch) · Ainessh (GPS start)

| Day | Task | Owner |
| --- | --- | --- |
| 1 | Package restructure, `pyproject.toml`, `config.py`, `paths.py`, CI skeleton. Acquire + curate RDD2022. **Launch training run 1.** | Shreyan, Vishnu |
| 2 | Training run 1 in flight. GPS quality gating + Kalman + haversine. Begin agent skeleton. | Vishnu, Ainessh, Jadeja |

---

### Phase 2 — Vertical Slice (Walking Skeleton) · D2–3

**Objective:** the thinnest possible end-to-end path — phone → detection → map — to prove the architecture before any of it is deep.

**Why this phase exists:** it converts the biggest unknown from "will these four people integrate?" to a fact by day 3 rather than day 12. **Deliberately uses the current single-class weights.** Accuracy is explicitly irrelevant here — this phase tests plumbing only. Using good weights now would hide integration bugs behind a plausible-looking result.

**Deliverables**

- MJPEG ingest from phone over hotspot (`cv2.VideoCapture`)
- Agent pipeline: bounded drop-oldest queue, batched inference, non-blocking ingest
- Detector as a module singleton
- Outbox → Mosquitto → FastAPI subscriber → SQLite
- Dashboard reading from the DB, showing one detection

**Exit gate**

- [ ] One pothole from **live phone video** appears on the dashboard within 3 s
- [ ] Ingest ≥ 15 fps with inference demonstrably not blocking it
- [ ] Dashboard contains no compute logic; it only reads the DB
- [ ] Agent runs with **no Streamlit import** in its dependency graph

**Contingency if gate fails**

- *MJPEG stream unstable:* switch the ingest source to a recorded clip. The ingest API is deliberately source-pluggable, so the slice remains a valid architectural proof.
- *Broker unavailable:* run the subscriber in-process temporarily. Do **not** carry this into Phase 5 — it defeats the offline-first design.

**Owners:** Jadeja

| Day | Task | Owner |
| --- | --- | --- |
| 2 | MJPEG ingest + agent skeleton with bounded queue | Jadeja |
| 3 | Detect → outbox → broker → FastAPI → SQLite → dashboard. **Checkpoint 1.** | Jadeja, Shreyan |

---

### Phase 3 — Detection Accuracy · D3–8

**Objective:** a model that genuinely knows RDD2022, with published per-class numbers, plus physical measurement to feed the civil engine.

**Why this phase exists:** this is the critical path. It waits on external GPU quota, and the reported failure "the model doesn't recognise potholes" is only genuinely fixed here.

**Deliverables**

- Training runs 2 and 3 (imbalance fixes; curated India/Japan/US mix)
- Evaluation harness: per-class AP50/AP50-95, precision, recall, confusion matrix, lane-marking FP rate
- In-domain test set: ≥150 self-recorded, hand-labelled dashcam frames
- `metrics.json` per run, charted across runs
- Ground-plane homography; metres-per-pixel per row
- Mask → length (m), width (mm), area (m²), equivalent diameter
- ASTM severity thresholds as data tables
- Published DV lookup tables + CDV correction curve replacing `a + b·log₁₀(d)`
- Per-sample-unit PCI (250–500 m²) + corridor average
- Uniform `PCIResult` shape on empty input

**Exit gate**

- [ ] Overall mAP50 ≥ 0.60
- [ ] **D40 pothole AP50 ≥ 0.55**; pothole recall ≥ 0.70 on the in-domain set
- [ ] Lane markings produce zero pothole false positives at the operating threshold
- [ ] Severity demonstrably changes when crack width is changed
- [ ] PCI reproduces the ASTM published worked example within tolerance
- [ ] `test_empty_input_uniform_shape` passes — Tab 3 crash closed

**Contingency if gate fails**

- *mAP50 < 0.50 after run 2:* retrain `YOLOv8s-seg` on India + Japan only — simpler and faster — ship that, and document the accuracy ceiling honestly.
- *D40 still short:* run 4 with pothole-only oversampling and stronger `copy_paste`. Potholes are the rarest class (§5.3); this is the expected failure mode.
- *D10 recall collapses:* expected — India has 68 transverse labels. Report per-class AP, state the limitation in the docs. Do not paper over it.

**Owners:** Vishnu (model, eval) · Udayvardhan (measurement, ASTM)

| Day | Task | Owner |
| --- | --- | --- |
| 3–4 | Evaluate run 1 → imbalance fixes → run 2. Label in-domain set. | Vishnu |
| 4–5 | Ground-plane calibration + mask geometry + severity tables. | Udayvardhan |
| 5–6 | DV tables + CDV curve + per-sample-unit PCI. | Udayvardhan |
| 6–8 | Run 3 on curated mix; per-class eval; measurement validated against a known 1 m reference. | Vishnu, Udayvardhan |

---

### Phase 4 — Position Integrity · D3–8 (concurrent with Phase 3)

**Objective:** GPS accuracy that is filtered, gated, map-matched, and quantified with before/after evidence.

**Why this phase exists:** runs concurrently with Phase 3 because it has no dependency on the model and a single owner. It is the second-most-questioned claim in a viva ("how do you know the GPS is right?").

**Deliverables**

- Haversine + odometer integration with drift logging
- Clock-skew estimation between phone and laptop
- Timestamped GPS ring buffer; server-side interpolation
- Heading derived from the filtered track
- osmnx map matching → lateral correction, **road names**, road class
- Before/after scatter plot of raw vs map-matched track
- `accuracy_m`, `hdop`, `n_sats`, `mapmatch_conf`, `road_name` persisted on every record

**Exit gate**

- [ ] Odometer vs haversine drift **< 5%** over a 2 km run
- [ ] Before/after plot exists showing lateral error reduction
- [ ] Road names populated on real detections
- [ ] `test_clock_skew_correction` and `test_kalman_reduces_variance` pass

**Contingency if gate fails**

- *osmnx graph download fails at demo time:* cache the OSM graph to disk ahead of time and degrade gracefully to no map-matching. Road names become optional; core function is unaffected.
- *Kalman tuning unstable:* fall back to a simple EMA filter. Less accurate, far less risk of oscillation near stops.

**Owners:** Ainessh

| Day | Task | Owner |
| --- | --- | --- |
| 3–4 | Ring buffer, clock skew, haversine, odometer cross-check. *(started D2 in Phase 1)* | Ainessh |
| 5–6 | Kalman tuning, heading from track, 2 km field run, drift measurement. | Ainessh |
| 7–8 | osmnx map matching, road names, before/after plot. | Ainessh |

---

### Phase 5 — Fusion & Transport · D6–10

**Objective:** one pothole produces exactly one record, and transmission survives signal loss.

**Why this phase exists:** these are the two failures that make the current system fundamentally untrustworthy — the ledger inflates ~5× per pothole, and nothing is persisted. Both are correctness, not polish, so they must land before the geoportal is built on top.

**Deliverables**

- ByteTrack-style association with two-stage high/low confidence split
- Track confirmation after K frames or T seconds
- Geofence suppression (~15 m)
- Best-frame selection + thumbnail persistence
- `frames_seen` / `track_duration_s` recorded as evidence strength
- Geometry measured once, on the best frame
- Outbox hardening: commit-before-publish, retry with attempt counting
- Idempotent GPS ingest via `UNIQUE(trip_id, seq)`
- FastAPI read APIs for dashboard consumption

**Exit gate**

- [ ] One pothole at 20 km/h → **exactly 1** record with `frames_seen > 3`
- [ ] Two adjacent potholes → 2 records (not merged, not duplicated)
- [ ] Second pass over the same street → 0 new records
- [ ] **60 s WiFi loss → zero detections lost** on reconnect
- [ ] Duplicate `seq` redelivery creates no duplicate rows
- [ ] All four fusion tests and both transport tests pass

**Contingency if gate fails**

- *Tracker unstable on fast road:* fall back to fixed-threshold IoU-only association and drop the high/low split. Simpler, slightly worse, shippable.
- *Outbox grows unboundedly:* add WAL checkpointing and a retention cap. Do not drop rows silently — losing data defeats the purpose of the outbox.
- *Link-loss test fails:* this blocks Phase 6. Escalate immediately; do not build the geoportal on an unreliable ledger.

**Owners:** Jadeja (fusion) · Shreyan (transport, server)

| Day | Task | Owner |
| --- | --- | --- |
| 6–7 | ByteTracker association, track confirmation, geofence. | Jadeja, Vishnu |
| 7–8 | Best-frame selection, thumbnails, measure-once. | Jadeja |
| 8–9 | Outbox hardening, idempotent ingest, FastAPI read APIs. | Shreyan |
| 9–10 | **Link-loss test and zero-loss proof.** **Checkpoint 2.** | Jadeja, Shreyan |

---

### Phase 6 — Civil Certification & Geoportal · D8–11

**Objective:** replace the invented PCI numbers and the static map with real, certified ones.

**Why this phase exists:** the geoportal and work orders are the visible deliverable. Building them on an unreliable ledger or wrong physics would mean rebuilding them, so Phase 5's gate is a hard precondition.

**Deliverables**

- PCI computed per sample unit over a real trip, not synthetic fixtures
- Coloured condition polyline along the corridor + corridor average
- Dashboard rebuilt entirely on DB reads
- **Pin → photo evidence**: thumbnail, measurement, severity, GPS accuracy
- Work orders carrying road names and measured dimensions
- Live status panel (fps, drop %, queue depth, GPS quality) from the MQTT status topic

**Exit gate**

- [ ] PCI on a real drive reproduces the ASTM worked example
- [ ] Every detection on the map has clickable photo evidence
- [ ] A second browser session sees identical live data — DB is the source of truth
- [ ] Work orders auto-populate OSM road names and measured dimensions

**Contingency if gate fails**

- *Reviewer requires PostGIS:* this is where the ~2 days is absorbed (§19 Q6). SQLite→PostGIS is a repository-layer change; the schema was designed for it.
- *Condition polyline too slow to render:* aggregate to sample-unit centroids client-side. Do not block the phase on this.

**Owners:** Udayvardhan (PCI over real data) · Shreyan (dashboard) · Ainessh (polyline, map layers)

| Day | Task | Owner |
| --- | --- | --- |
| 8–9 | Per-sample-unit PCI on real trip data; corridor average. | Udayvardhan |
| 9–10 | Dashboard rebuild on DB; condition polyline; map layers. | Shreyan, Ainessh |
| 11 | Pin → photo evidence; work orders with road names + dimensions; live status panel. | Shreyan, Ainessh |

---

### Phase 7 — Hardening & Defence · D11–15

**Objective:** prove every claim with a measurement, and have a recorded demo that runs from one command.

**Why this phase exists:** this phase converts the project from "it works on my machine" into something defensible. The field protocol (§14) and the results document are the deliverables that matter most in a viva.

**Deliverables**

- Full test suite green (§13); ruff clean
- Docker Compose one-command demo
- Complete §14 field protocol: in-domain accuracy, GPS improvement, distance integrity, ingest rate, dedup, link loss, measurement calibration, end-to-end drive
- Metrics dashboard: mAP, per-class AP, fps, drop %, GPS drift, PCI trend
- `results.md` — what works, what does not, measured limits
- README rewrite
- **Full dry run on a real drive, recorded**
- Stage-B Flutter app — **only if all gates green**, otherwise dropped without loss

**Exit gate**

- [ ] Every §14 field test passes
- [ ] CI green; demo starts with one command
- [ ] `results.md` states limitations with numbers
- [ ] Dry run recorded end to end, no manual intervention

**Contingency if gate fails**

- *Stage B not working:* drop it. Stage A MJPEG is a complete, shippable video path — losing Stage B costs nothing essential.
- *A field test fails:* report it honestly in `results.md` rather than hiding it. A documented limitation with a measured number is stronger in a viva than an unqualified success claim.

**Owners:** all five

| Day | Task | Owner |
| --- | --- | --- |
| 11–12 | Test suite, ruff, Docker Compose, README. | Shreyan |
| 12–13 | Stage-B Flutter **if green**, else polish. Field protocol runs. | Jadeja, all |
| 14 | Metrics dashboard; `results.md` written honestly. | Vishnu, Ainessh |
| 15 | **Full dry run on a real drive. Record it.** | all |

### 15.4 Critical path and float

**Critical path: Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 7**, driven end-to-end by GPU training.

- Phase 1 and Phase 2 have **zero float**. If either slips, everything slips.
- Phase 3 is where slippage accumulates, because it waits on external GPU quota and because the D40 imbalance (§5.3) will likely need a fourth training run. **Build a run into the plan** rather than discovering it at D8.
- Phase 4 has ~1 day of float — it can compress without affecting anyone else.
- Phase 7 holds the only real float (~2 days), which is exactly where Stage B is scheduled. If everything upstream goes well, spend it there; if not, it absorbs the delay.

**Checkpoint summary**

| Checkpoint | When | Proves |
| --- | --- | --- |
| Phase 1 gate | End D2 | Package sound, training in flight |
| Phase 2 gate / **Checkpoint 1** | End D3 | Live phone video → detection → dashboard |
| Phase 3 gate | End D8 | Model and physics meet their numbers |
| Phase 4 gate | End D8 | GPS improvement measured |
| Phase 5 gate / **Checkpoint 2** | End D10 | Deduped, offline-resilient |
| Phase 6 gate | End D11 | Civil-correct, evidence-attached |
| Phase 7 gate | End D15 | Everything measured and recorded |

---

## 16. Risk register

| Risk | Trigger / impact | Mitigation | Owner |
| --- | --- | --- | --- |
| **Domain gap** — RDD2022 vs your phone on Indian roads | mAP collapses at the demo; worst possible timing | Train day 1, not day 10. Label an in-domain set and tune on it. Keep the current weights as a fallback | Vishnu |
| Cloud GPU quota exhausted | Training slips, blocking all of W1 | Start day 1. Fallback: Colab, or a CPU overnight run on a YOLOv8s subset | Vishnu |
| **D10 collapse** — India has 68 transverse labels | Transverse recall near zero; examiner will ask | Oversample D00/D10, report per-class AP, state the limitation explicitly | Vishnu |
| Phone app slips (Stage B) | Video path undelivered | Stage A MJPEG is complete and shippable; Stage B is bonus only | Jadeja |
| Four workstreams in parallel, 5 people | Integration hell | Vertical slice makes seams real by day 3. One shared package, merge daily | Jadeja |
| osmnx/OSM graph download fails at demo time | Road names missing live | Cache the OSM graph to disk ahead of time; degrade gracefully to no map-matching | Ainessh |
| `git gc` surprises | Wanted objects could be pruned | `git bundle` backup first; run gc last | Shreyan |
| Over-scoping | Nothing demoable at the deadline | Every stage has its own checkpoint; if one slips the last stands | Jadeja |

---

## 17. Migration map

| Old | New | Disposition |
| --- | --- | --- |
| `app/detector.py:91-138` heuristic class mapping | `vision/detector.py` | **Delete** — guessing, and wrong |
| `app/detector.py:130-136` OpenCV crack supplement | real 4-class model | **Delete** — fabricates classes |
| `app/detector.py:113,185` area-based severity | `civil/severity.py` + `vision/measure.py` | **Delete** — bbox area has no physical meaning |
| `app/pci_engine.py:45-70` `a + b·log₁₀(d)` | `civil/dv_tables.py`, `civil/cdv.py` | **Delete** — invented |
| `app/pci_engine.py` global PCI | `civil/pci.py` per sample unit | **Rewrite** |
| `app/telemetry.py` HTTPServer + in-memory list | `transport/`, `agent/ingest.py` | **Delete** |
| `app/telemetry.py:95` linear `interpolate_gps` | `geo/kalman.py`, `geo/mapmatch.py` | **Delete** |
| `app/dashboard.py:204-228` synthetic canvas | live camera ingest | **Replace** |
| `app/dashboard.py:129` session_state as DB | `storage/` | **Replace** |
| `app/dashboard.py:17-24` sys.path hacks | `pyproject.toml` package | **Delete** |
| `app/work_order.py` | `ui/` + `server/` | **Refactor** — keep treatments, add measured dims + road names |
| `data/sample_gps_track.csv`, `sample_defects.json` | `tests/fixtures/` | **Move** — fixtures, not demo data |
| `run.sh` | `Makefile` + `docker compose` | **Rewrite** |

---

## 18. Definition of done

- [ ] Pothole in live phone video at ≥15 fps ingest, detected and geo-tagged to within ~3 m
- [ ] mAP50 ≥ 0.60 overall, D40 AP50 ≥ 0.55, published in `metrics.json` with per-class breakdown
- [ ] One pothole → exactly one record, with photo, dimensions and GPS accuracy attached
- [ ] 60 s signal loss → zero data loss
- [ ] A real drive recorded end to end: capture → broker → server → map → work order
- [ ] PCI per ASTM D6433 sample units, reproducing the standard's worked example
- [ ] Clean gc'd repo, tests green in CI, one-command demo via Docker Compose
- [ ] Results document stating what works, what does not, and the measured limits

> The last item is the most defensible thing to bring to a viva. A project that states its own limits with measured numbers is stronger than one that claims everything works.

---

## 19. Open questions

| # | Question | Blocks | Decide by |
| --- | --- | --- | --- |
| 1 | Phone model and OS? Determines Stage B feasibility and camera resolution | W4 Stage B | Day 1 |
| 2 | Camera mount height, fixed or adjustable? Determines calibration approach | W2.1 | Day 4 |
| 3 | Which 2 km route for the field test — and is it safe/legal to survey? | §14 protocol | Day 2 |
| 4 | Kaggle or Colab for GPU? Which account has quota? | W1 day 1 | **Day 1, blocking** |
| 5 | Is a public broker acceptable if local Mosquitto fails on demo day? | W6.1 fallback | Day 9 |
| 6 | Does the reviewer expect PostGIS? If so, schedule must absorb ~2 days | W6.4 | Day 3 |
---

## 20. Phase progress log

### Phase 1 — Foundation & Kick-Off ✅ (6 of 7 gates)

| Gate | Status |
| --- | --- |
| `pip install -e .` clean | ✅ |
| `python -m roadscope.agent --help` runs | ✅ |
| `test_detector_singleton` passes | ✅ |
| `test_paths_independent_of_cwd` passes | ✅ |
| ruff clean, 33 tests pass | ✅ |
| Training run 1 in flight | ❌ **blocked — needs GPU account (§19 Q4)** |

Delivered: `pyproject.toml` with optional extras, `src/roadscope/` package
(`paths`, `config`, `logging_setup`, `vision.detector`, `geo.quality`,
`geo.kalman`), CI workflow, `docker-compose.yml`, `.env.example`,
`scripts/prepare_rdd2022.py`, `scripts/train_cloud.md`,
`scripts/download_weights.sh`.

### Phase 2 — Vertical Slice ✅ (architecture proven; demo pending)

Delivered: `storage/` (schema, db, repository), `transport/` (topics, outbox,
mqtt_client), `agent/` (ingest sources, pipeline, health), `server/`
(subscriber, FastAPI app). 44 tests pass.

Verified by measurement, not assertion:

| Claim | Evidence |
| --- | --- |
| Link loss loses nothing | `test_link_loss_loses_nothing` — capture while the publisher refuses all messages, flush on reconnect, zero loss |
| A restarted agent delivers what the dead one queued | `test_outbox_replay_after_restart` |
| QoS1 redelivery is idempotent | `test_duplicate_gps_seq_is_idempotent` — 5 deliveries, 1 row, 4 duplicates counted |
| Ingest never blocks on inference | `test_ingest_is_not_blocked_by_inference` — a detector 15× slower than capture; capture continues, queue absorbs the difference |
| The agent has no UI dependency | `test_agent_does_not_import_streamlit` + runtime check: no Streamlit/Folium module loaded |
| Accuracy metadata survives the chain | end-to-end run: `acc=3.0 hdop=0.8 flag=good` persisted intact |

#### Measurement: inference cost on CPU

| `imgsz` | ms/frame | Rate |
| --- | --- | --- |
| 640 | 127 ms | 7.9 fps |
| 960 | 263 ms | 3.8 fps |

An 8-second run at 640: 206 frames captured, 64 processed, **134 dropped
(65%)**.

**Finding: inference is the binding constraint, not capture.** A 30 fps camera
cannot be fully processed on this CPU-only 8 GB laptop. The 65% drop rate is
correct behaviour — stale frames are worthless — but it has two consequences:

1. The queue depth of 8 is sized for a GPU. On CPU it should drop to 2–3 so the
   queue does not hold frames that are already stale by the time they are read.
2. The default `imgsz` is now **640**, not 960. 960 halved throughput to 3.8 fps
   and detected *zero* objects on the test frame, so it was costing accuracy for
   no benefit. Re-raise to 960 on CUDA, or once the RDD2022 model exists.

Drop percentage is published via the status topic rather than hidden, so
degradation is visible instead of silent.

#### Still open for Phase 2

- [ ] Live phone MJPEG → dashboard. Needs the phone on a hotspot with DroidCam or
      IP Web Camera serving a stream URL. Code path exists and is exercised by
      `VideoFileSource`; only the physical link is untested.
- [ ] Streamlit dashboard reading the database. Phase 4 deliverable; the
      database side is proven, the UI is not written yet.
