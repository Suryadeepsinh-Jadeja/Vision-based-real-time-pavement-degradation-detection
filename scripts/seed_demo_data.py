#!/usr/bin/env python3
"""Seed the survey store with a realistic trip, for dashboard development.

The dashboard needs data to develop against, and a real survey takes a drive.
This writes a plausible trip -- 2 km of highway with a damaged start and end,
road name set, GPS noise, and device telemetry -- so the UI can be exercised
without leaving the desk.

It writes to a throwaway database by default so it can never overwrite a real
survey. Pass ``--db`` to target the live store, and ``--force`` to overwrite.

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --db var/roadscope.sqlite --force
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from roadscope.server.survey import score_trip
from roadscope.storage.repository import Repository

# Sarkhej-Gandhinagar Highway, Ahmedabad. A real road, so the map lands somewhere
# recognisable rather than in the ocean.
LAT0 = 23.0225
LON0 = 72.5714
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))

ROAD_NAME = "Sarkhej-Gandhinagar Highway"
MODEL_VERSION = "rdd2022-yolov8m-r1"


def seed(db_path: pathlib.Path, *, trips: int = 1, force: bool = False) -> None:
    if db_path.exists() and not force:
        existing = Repository(db_path, create=False)
        if existing.list_trips(limit=1):
            raise SystemExit(
                f"{db_path} already holds {len(existing.list_trips(limit=999))} trip(s). "
                "Pass --force to add anyway, or choose another --db."
            )

    repo = Repository(db_path)
    device_id = "van-01"
    repo.upsert_device(device_id, agent_version="0.1.0", model_version=MODEL_VERSION)

    for n in range(trips):
        _seed_trip(repo, device_id, n)


def _seed_trip(repo: Repository, device_id: str, index: int) -> None:
    trip_id = f"trip-demo-{index + 1:02d}-{uuid.uuid4().hex[:6]}"
    repo.start_trip(trip_id, device_id)

    speed_mps = 40 / 3.6  # 40 km/h, urban highway
    n_steps = int(2000 / speed_mps)
    now = time.time() - (trips_offset := 86400 * (index + 1))

    import random

    rng = random.Random(7 + index)

    # GPS: 1 Hz, walking due east.
    #
    # Noise is 2 m, not the 8 m the accuracy field claims, and the reason is
    # worth recording. Injecting *independent* 8 m error per fix inflates
    # cumulative chainage badly -- each 11 m step picks up a jitter component
    # and the total comes out ~68% long, which then yields the wrong number of
    # sample units. Real receiver noise is partly correlated between
    # consecutive fixes, so it does not accumulate as fast; 2 m here keeps the
    # seeded chainage within a few percent of the stated 2 km. See
    # docs/results.md, "Cumulative distance overestimates".
    GPS_NOISE_M = 2.0
    for i in range(n_steps):
        truth_lon = LON0 + (i * speed_mps) / M_PER_DEG_LON
        repo.insert_fix(
            trip_id,
            {
                "seq": i,
                "ts_utc": now + i,
                "lat": LAT0 + rng.gauss(0, GPS_NOISE_M) / M_PER_DEG_LAT,
                "lon": truth_lon + rng.gauss(0, GPS_NOISE_M) / M_PER_DEG_LON,
                "speed_kmh": 40.0,
                "heading": 90.0,
                "accuracy_m": 8.0,
                "hdop": 0.9,
                "n_sats": 9,
                "quality_flag": "good",
                "road_name": ROAD_NAME,
            },
        )

    # Damage profile: potholes at both ends, cracking through the middle.
    # severity is left None deliberately: the detector produces bounding boxes
    # and cannot measure depth or width, so the engine defaults it and records
    # a limitation. Seeding a severity here would hide that gap.
    def add(code: str, step: int, *, area=None, length=None) -> None:
        repo.insert_detection(
            trip_id,
            {
                "detection_id": f"{code}-{trip_id}-{step}",
                "ts_utc": now + step,
                "lat": LAT0 + rng.gauss(0, 2.0) / M_PER_DEG_LAT,
                "lon": LON0 + (step * speed_mps) / M_PER_DEG_LON,
                "gps_accuracy_m": 8.0,
                "class_code": code,
                "confidence": round(rng.uniform(0.52, 0.94), 3),
                "severity": None,  # detector cannot measure; engine will default
                "area_m2": area,
                "length_m": length,
                "frames_seen": rng.randint(3, 9),
                "track_duration_s": round(rng.uniform(0.4, 1.6), 2),
                "model_version": MODEL_VERSION,
            },
        )

    # Spacing matters. At 40 km/h and 1 Hz a sample unit spans ~75 m, so defects
    # every other step put roughly one per metre of road in the damaged stretches
    # -- dense enough to drive the PCI into Fair/Poor, which is what makes the
    # demo show a condition gradient rather than a uniform "Good".
    for step in range(0, 40, 1):  # potholes at the start
        add("D40", step, area=round(rng.uniform(0.04, 0.14), 3))
    for step in range(40, 150, 3):  # longitudinal cracking through the middle
        add("D00", step, length=round(rng.uniform(0.8, 2.4), 2))
    for step in range(150, n_steps, 1):  # potholes at the end
        add("D40", step, area=round(rng.uniform(0.04, 0.14), 3))

    # Telemetry: realistic inference cost and drop rate (see docs/results.md).
    for i in range(0, n_steps, 5):
        repo.insert_status(_status(device_id, now + i, rng, trips_offset))

    repo.end_trip(trip_id, now + n_steps, 2000.0)

    result = score_trip(repo, trip_id)
    print(
        f"{trip_id}: {repo.fix_count(trip_id)} fixes, "
        f"{len(repo.detections(trip_id=trip_id))} defects, "
        f"PCI {result['pci']:.1f} ({result['rating']}), "
        f"{len(result['sample_units'])} units"
    )


def _status(device_id: str, ts: float, rng, offset: float):
    from roadscope.storage.repository import DeviceStatus

    return DeviceStatus(
        device_id=device_id,
        ts_utc=ts,
        fps=round(rng.uniform(6.4, 7.6), 2),
        drop_pct=round(rng.uniform(74.0, 77.5), 2),
        queue_depth=rng.randint(1, 4),
        infer_ms=round(rng.uniform(118, 142), 1),
        agent_version="0.1.0",
        model_version=MODEL_VERSION,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=pathlib.Path, default=pathlib.Path("var/demo.sqlite"))
    ap.add_argument("--trips", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="seed even if trips exist")
    args = ap.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    seed(args.db, trips=args.trips, force=args.force)
    print(f"\nwrote {args.db}")
    print("\npoint the dashboard at it:")
    print(f"  ROADSCOPE_SERVER_DB={args.db} streamlit run -m roadscope.ui.dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
