#!/usr/bin/env python3
"""Curate RDD2022 into a 4-class YOLO training set.

Why this script exists
----------------------
Two traps make naive use of RDD2022 quietly produce a bad model:

1. **Category IDs are not class indices.** RDD2022 ships 8 COCO categories
   (D00, D01, D10, D11, D20, D43, D44, D40). D40 (pothole) is category_id 7,
   not 3. Assuming ``category_id == target_class`` trains pothole detectors on
   crosswalk blur.
2. **The China subset is drone/oblique imagery.** Training on it actively
   degrades forward-facing vehicle-mounted inference, which is what a phone on a
   dashboard produces.

This script reads the category names out of the COCO file rather than assuming
IDs, drops the non-target classes, restricts to vehicle-mounted countries, and
writes a stratified train/val split in YOLO format.

Usage
-----
    python scripts/prepare_rdd2022.py \\
        --src /path/to/RDD2022 \\
        --dst data/rdd2022-curated \\
        --countries India Japan United_States

Inspect what a COCO file declares before trusting the output:

    python scripts/prepare_rdd2022.py --src /path/to/RDD2022 --inspect
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Canonical 4-class target schema. Index == YOLO class id.
TARGET_CLASSES = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator / Fatigue Crack
    "D40": 3,  # Pothole
}

# Vehicle-mounted, forward-facing. China is intentionally absent: its subset is
# drone/oblique imagery from a different domain.
DEFAULT_COUNTRIES = ("India", "Japan", "United_States")
AVAILABLE_COUNTRIES = (
    "India",
    "Japan",
    "United_States",
    "Norway",
    "Czech",
    "China",
    "China_M",
    "China_D",
)

_DCODE = re.compile(r"D\d{2}")


@dataclass
class Stats:
    """Running tally of the curation."""

    images: int = 0
    dropped_no_target: int = 0
    boxes: Counter = field(default_factory=Counter)
    per_country_images: Counter = field(default_factory=Counter)
    # Nested counter: a plain Counter returns 0 for a missing key, so indexing
    # into it to build a nested tally raises TypeError.
    per_country_boxes: defaultdict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def report(self) -> str:
        lines = ["", "=" * 62, "RDD2022 curation report", "=" * 62]
        lines.append(f"images kept        : {self.images}")
        lines.append(f"images dropped     : {self.dropped_no_target} (no target class present)")
        lines.append("")
        lines.append("box counts (class -> n):")
        for code, expected in sorted(TARGET_CLASSES.items(), key=lambda kv: kv[1]):
            n = self.boxes.get(code, 0)
            flag = "  <-- RARE" if 0 < n < 1000 else ""
            lines.append(f"  {code} (idx {expected}) : {n:>7}{flag}")
        lines.append("")
        lines.append("per country:")
        for country in sorted(self.per_country_images):
            lines.append(f"  {country:<16} images {self.per_country_images[country]:>6}")
            detail = ", ".join(
                f"{c}={n}" for c, n in sorted(self.per_country_boxes[country].items())
            )
            lines.append(f"  {'':<16} boxes  {detail}")
        lines.append("=" * 62)
        return "\n".join(lines)


def parse_category_map(coco: dict) -> dict[int, str]:
    """Build ``{category_id: D-code}`` from the COCO categories array.

    Names look like ``"D00 Longitudinal"`` or ``"D00"``, so the code is
    extracted rather than inferred from the ID.
    """
    mapping: dict[int, str] = {}
    for cat in coco.get("categories", []):
        name = str(cat.get("name", ""))
        match = _DCODE.search(name)
        if match:
            mapping[int(cat["id"])] = match.group(0)
    return mapping


def coco_bbox_to_yolo(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert COCO ``[x, y, w, h]`` to YOLO normalised ``[cx, cy, w, h]``."""
    x, y, w, h = bbox
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return cx, cy, w / width, h / height


def build_index(src: Path, countries: tuple[str, ...]) -> dict[str, Path]:
    """Map each requested country to its ``train/annotations`` directory."""
    found: dict[str, Path] = {}
    for country in countries:
        candidate = src / country / "train" / "annotations"
        if not candidate.is_dir():
            alt = src / country / "annotations"
            candidate = alt if alt.is_dir() else candidate
        found[country] = candidate
    return found


def curate_country(
    country: str,
    ann_dir: Path,
    img_root: Path,
    out_images: Path,
    out_labels: Path,
    split: str,
    stats: Stats,
    seed: int,
    val_fraction: float,
) -> None:
    """Copy one country's images/labels into the curated split."""
    coco_path = ann_dir / "coco.json"
    if not coco_path.is_file():
        print(f"  ! {country}: no coco.json at {coco_path}, skipping")
        return

    with coco_path.open() as fh:
        coco = json.load(fh)

    cat_map = parse_category_map(coco)
    if not cat_map:
        print(f"  ! {country}: no D-codes found in categories, skipping")
        return

    print(f"  {country}: {len(coco.get('images', []))} images, categories {cat_map}")

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        code = cat_map.get(int(ann["category_id"]))
        if code in TARGET_CLASSES:
            anns_by_image[int(ann["image_id"])].append(ann)

    images = [im for im in coco.get("images", []) if anns_by_image.get(int(im["id"]))]
    stats.dropped_no_target += len(coco.get("images", [])) - len(images)
    random.Random(seed).shuffle(images)

    n_val = int(len(images) * val_fraction)
    splits = (
        {"val": images[:n_val], "train": images[n_val:]} if split == "auto" else {split: images}
    )

    for split_name, subset in splits.items():
        for img in subset:
            src_img = img_root / img["file_name"]
            if not src_img.is_file():
                continue
            dst_img = out_images / split_name / Path(img["file_name"]).name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)

            w, h = int(img["width"]), int(img["height"])
            lines = []
            for ann in anns_by_image[int(img["id"])]:
                code = cat_map[int(ann["category_id"])]
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], w, h)
                lines.append(f"{TARGET_CLASSES[code]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats.boxes[code] += 1
                stats.per_country_boxes[country][code] += 1

            dst_lbl = out_labels / split_name / f"{Path(img['file_name']).stem}.txt"
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines) + "\n")

            stats.images += 1
            stats.per_country_images[country] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, required=True, help="path to the RDD2022 root")
    parser.add_argument("--dst", type=Path, default=Path("data/rdd2022-curated"))
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="print category maps and exit")
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        parser.error(f"source not found: {args.src}")

    unknown = [c for c in args.countries if c not in AVAILABLE_COUNTRIES]
    if unknown:
        parser.error(f"unknown countries {unknown}; available: {list(AVAILABLE_COUNTRIES)}")

    ann_dirs = build_index(args.src, tuple(args.countries))
    if args.inspect:
        for country, ann_dir in ann_dirs.items():
            coco_path = ann_dir / "coco.json"
            print(f"\n{country}  ({coco_path})")
            if not coco_path.is_file():
                print("  no coco.json")
                continue
            with coco_path.open() as fh:
                coco = json.load(fh)
            cat_map = parse_category_map(coco)
            for cat in coco.get("categories", []):
                code = cat_map.get(int(cat["id"]), "?")
                keep = "KEEP" if code in TARGET_CLASSES else "drop"
                print(f"  id={cat['id']:>2}  {code:<5} {cat['name']:<32} {keep}")
        return 0

    out_images = args.dst / "images"
    out_labels = args.dst / "labels"
    stats = Stats()

    print(f"Curating {args.countries} from {args.src} -> {args.dst}")
    for country, ann_dir in ann_dirs.items():
        if not (ann_dir / "coco.json").is_file():
            print(f"  ! {country}: missing coco.json, skipping")
            continue
        img_root = ann_dir.parent / "images"
        curate_country(
            country,
            ann_dir,
            img_root,
            out_images,
            out_labels,
            "auto",
            stats,
            args.seed,
            args.val_fraction,
        )

    (args.dst / "roadscope.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n\n"
        "names:\n"
        "  0: D00   # Longitudinal Crack\n"
        "  1: D10   # Transverse Crack\n"
        "  2: D20   # Alligator / Fatigue Crack\n"
        "  3: D40   # Pothole\n"
    )

    print(stats.report())
    print(f"\nDataset YAML written to {args.dst / 'roadscope.yaml'}")
    if stats.boxes.get("D10", 0) < 500:
        print(
            "\nWARNING: very few D10 (transverse crack) labels. Expect poor recall on\n"
            "         that class. Report per-class AP honestly rather than hiding it\n"
            "         behind an averaged mAP. See REBUILD_PLAN.md section 5.3."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
