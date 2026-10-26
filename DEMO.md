# RoadScope — end-to-end demo (no budget, laptop + phone)

Everything below runs on the machine you already have. No cloud GPU, no
purchased hardware. The only thing you buy is a hotspot if your phone cannot
share Wi-Fi.

## What you will see

1. Phone camera captures the road.
2. Laptop detects potholes, cracks and alligator cracking (RDD2022 model).
3. Each detection is geo-tagged with the phone's GPS position.
4. Detections land in a local SQLite store.
5. Dashboard shows a condition map, per-unit PCI, defect table, work orders.
6. Detector playground lets you upload a photo/video and see boxes.

## One-time setup

```bash
cd "/Users/suryadeepsinhjadeja/Desktop/Smart Cities"
source venv/bin/activate

# 1. Mosquitto broker (one-time)
brew install mosquitto
cp /opt/homebrew/etc/mosquitto/mosquitto.conf.example \
   /opt/homebrew/etc/mosquitto/mosquitto.conf
/opt/homebrew/opt/mosquitto/sbin/mosquitto -c /opt/homebrew/etc/mosquitto/mosquitto.conf -d

# 2. Confirm weights
bash scripts/download_weights.sh --verify   # expect: 4 classes D00 D10 D20 D40

# 3. Seed demo data so the dashboard has something to show immediately
make seed
```

## Demo A — no phone needed (30 seconds, proves the chain)

```bash
make check                                  # confirms weights + config
make seed                                   # 2 trips, 107 defects, ~27 units
make dashboard                              # -> http://localhost:8501
```

Open the dashboard. You will see the condition map, work orders, and health
telemetry. This proves the storage → civil → UI chain with no hardware.

## Demo B — live phone camera

### Phone side

Install one of these free apps on your phone:

- **DroidCam** (Android/iOS) — serves MJPEG at `http://<phone-ip>:4747/video`
- **IP Web Camera** (Android) — serves at `http://<phone-ip>:8080/video`

Put the phone and laptop on the same Wi-Fi. Note the phone's IP.

### Run the agent

```bash
source venv/bin/activate
python -m roadscope.agent \
    --camera-url http://<phone-ip>:4747/video \
    --gps-synthetic \
    --seconds 30
```

`--gps-synthetic` gives a straight-line GPS track so the demo runs without a
GPS app. For real positions, see "Real GPS" below.

The agent captures, detects on the Apple GPU, geotags, and transmits over MQTT.
If the broker drops, messages buffer in the outbox and are replayed later.

### View results

```bash
make dashboard                              # http://localhost:8501
```

The dashboard auto-refreshes from the store. Defects appear as they are
processed.

## Demo C — upload and inspect (no phone, no drive)

```bash
make playground                             # -> http://localhost:8502
```

Drop a photo or dashcam clip. Every detection is boxed and labelled; for video
you can step through frames with the slider and jump between frames that have
detections. No GPS or broker required.

## Real GPS (optional, adds ~10 minutes)

Install a free GPS-over-HTTP app on the phone:

- **GPS Logger** (Android) — Settings → "Log to HTTP URL", point it at
  `http://<laptop-ip>:8765/gps` (you would need a tiny receiver; simpler
  below).
- **GPS2IP** (iOS) — serves NMEA/GPS JSON over TCP.

Then:

```bash
python -m roadscope.agent \
    --camera-url http://<phone-ip>:4747/video \
    --gps-url    http://<phone-ip>:<gps-port>/json \
    --seconds 60
```

If the phone app's JSON field names differ from the aliases in
`roadscope/geo/gps_sources.py`, add the alias there (one line).

## Fallbacks if something breaks

| Problem | Fallback |
|---|---|
| Phone and laptop on different networks | Use a phone hotspot; laptop joins it |
| No camera app installs | Use a recorded video file: `--camera-url` is not needed; pass a path via `VideoFileSource` (see `scripts/seed_demo_data.py` for the pattern) |
| No GPS app | `--gps-synthetic` (straight line) or `--gps-csv path/to/track.csv` |
| Broker dies mid-demo | Messages buffer in `var/outbox.sqlite`; run `--replay` afterwards |
| Dashboard empty | `make seed` populates it instantly |
| Model detects nothing | Lower confidence in the playground sidebar; try 1280 px |

## What to say about the numbers

- mAP50 0.609 on RDD2022 (India, Japan, US). D40 pothole AP50 0.538, below
  the 0.55 gate -- documented in `docs/results.md` with the diagnosis.
- PCI per ASTM D6433 sample units, validated against the published worked
  example (DVs 4/17/13, TDV 34, CDV 33).
- Inference 125 ms/frame at 960 px on MPS (2.4x faster than CPU), 8.0 fps
  sustained against 30 fps capture.
- 141 tests, ruff clean.

## What NOT to claim

- Do not claim it works on Indian roads. All metrics are RDD2022 validation.
- Do not claim pothole depth or crack width. The model produces bounding boxes
  only; severity is defaulted and reported as a limitation.
- Do not claim the live phone link was field-tested. It is implemented and
  runs against a synthetic source; the physical link is the one untested piece.