#!/usr/bin/env python3
"""Verify that a weights file is the trained RDD2022 detector.

Guards against the two ways weights go wrong in practice:

* the legacy single-class COCO placeholder is still present, and
* a download was truncated or is the wrong run.

Size alone is insufficient -- a truncated checkpoint still parses -- so the
class names are checked too. Those are the thing that actually distinguishes
r1 from the placeholder, and they are what the rest of the pipeline maps
D00/D10/D20/D40 to.

Usage:
    python scripts/check_model.py models/rdd2022-yolov8m-r1.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_CLASSES = ["D00", "D10", "D20", "D40"]
EXPECTED_BYTES = 52_099_026


def _class_order(names: object) -> list[str]:
    """Return class names in index order, whether keyed by int or str."""
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    path = Path(argv[1]).expanduser()
    if not path.is_file():
        print(f"FAIL  {path} does not exist")
        return 1

    size = path.stat().st_size
    failures: list[str] = []

    if size != EXPECTED_BYTES:
        failures.append(
            f"size {size:,} bytes, expected {EXPECTED_BYTES:,} "
            "(truncated download, or a different run)"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        print("WARN  ultralytics is not importable; checked size only")
        print(f"      {'FAIL' if failures else 'OK'}: {path} ({size:,} bytes)")
        for f in failures:
            print(f"      - {f}")
        return 1 if failures else 0

    model = YOLO(str(path))
    names = _class_order(model.names)

    if names != EXPECTED_CLASSES:
        failures.append(f"classes {names}, expected {EXPECTED_CLASSES}")
        hint = ""
        if names == ["person"]:
            hint = "\n      This is the legacy COCO placeholder. Delete it and retry."
        print(f"FAIL  {path}{hint}")
    else:
        print(f"OK    {path} ({size:,} bytes, {len(names)} classes: {', '.join(names)})")

    for f in failures:
        print(f"      - {f}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
