#!/usr/bin/env python3
"""Curate RDD2022 into a 4-class YOLO training set.

Why this script exists
----------------------
Three traps make naive use of RDD2022 quietly produce a bad model:

1. **The annotations are Pascal VOC XML, not COCO.** The official release
   (figshare 21431547) ships ``<country>/train/annotations/xmls/*.xml``. Code
   that globs for ``coco.json`` finds nothing and reports an empty dataset.
   This script auto-detects and reads both, so it also works with the COCO
   re-uploads found on Kaggle and HuggingFace.

2. **Class names carry the schema.** VOC ``<name>`` is ``D00``/``D10``/``D20``/
   ``D40`` (sometimes with a descriptive suffix). COCO uses ``category_id``,
   which is *not* the target class index. Both are resolved to the D-code by
   parsing the name, never by assuming an id ordering.

3. **The China subset is drone/motorbike imagery.** ``China_Drone`` and
   ``China_MotorBike`` are a different domain from a phone on a dashboard and
   degrade forward-facing vehicle-mounted inference. Excluded by default.

Usage
-----
    python scripts/prepare_rdd2022.py --src /content/RDD2022 --inspect
    python scripts/prepare_rdd2022.py --src /content/RDD2022 --dst /content/rdd2022-curated
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
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

# Folder names exactly as they appear in the official release.
AVAILABLE_COUNTRIES = (
    "India",
    "Japan",
    "United_States",
    "Norway",
    "Czech",
    "China_Drone",
    "China_MotorBike",
)

# Vehicle-mounted, forward-facing. The China subsets are drone/motorbike.
DEFAULT_COUNTRIES = ("India", "Japan", "United_States")

_DCODE = re.compile(r"D\d{2}")


@dataclass
class ImageAnnotations:
    """One image and the target-class boxes on it."""

    filename: str
    width: int
    height: int
    boxes: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)


@dataclass
class CountryAnnotations:
    """Every annotation for one country, normalised across formats."""

    images: list[ImageAnnotations] = field(default_factory=list)
    source_format: str = "unknown"


@dataclass
class Stats:
    """Running tally of the curation."""

    images: int = 0
    dropped_no_target: int = 0
    dropped_missing_file: int = 0
    boxes: Counter = field(default_factory=Counter)
    per_country_images: Counter = field(default_factory=Counter)
    # Nested tally needs a defaultdict: a plain Counter returns 0 for a missing
    # key, so indexing into it to build a nested count raises TypeError.
    per_country_boxes: defaultdict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def report(self) -> str:
        lines = ["", "=" * 64, "RDD2022 curation report", "=" * 64]
        lines.append(f"images kept        : {self.images}")
        lines.append(f"dropped, no target : {self.dropped_no_target}")
        lines.append(f"dropped, no file   : {self.dropped_missing_file}")
        lines.append("")
        lines.append("box counts (class -> n):")
        for code, idx in sorted(TARGET_CLASSES.items(), key=lambda kv: kv[1]):
            n = self.boxes.get(code, 0)
            flag = "  <-- RARE, expect poor recall" if 0 < n < 1000 else ""
            lines.append(f"  {code} (idx {idx}) : {n:>7}{flag}")
        lines.append("")
        lines.append("per country:")
        for country in sorted(self.per_country_images):
            lines.append(f"  {country:<16} images {self.per_country_images[country]:>6}")
            detail = ", ".join(
                f"{c}={n}" for c, n in sorted(self.per_country_boxes[country].items())
            )
            lines.append(f"  {'':<16} boxes  {detail}")
        lines.append("=" * 64)
        return "\n".join(lines)


# --- format detection ----------------------------------------------------


def find_annotations(train_dir: Path) -> tuple[Path, str] | None:
    """Locate the annotations directory and detect its format.

    Returns ``(path, format)`` or ``None``.
    """
    candidates = [
        (train_dir / "annotations", "auto"),
        (train_dir, "auto"),
    ]
    for candidate, _ in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "coco.json").is_file():
            return candidate / "coco.json", "coco"
        xmls = candidate / "xmls"
        if xmls.is_dir() and any(xmls.glob("*.xml")):
            return xmls, "voc"
        if any(candidate.glob("*.xml")):
            return candidate, "voc"
    return None


def code_from_name(name: str) -> str | None:
    """Extract the D-code from a class name."""
    match = _DCODE.search(str(name).upper().replace(" ", ""))
    return match.group(0) if match else None


# --- COCO ---------------------------------------------------------------


def parse_category_map(coco: dict) -> dict[int, str]:
    """``{category_id: D-code}`` from the COCO categories array."""
    mapping: dict[int, str] = {}
    for cat in coco.get("categories", []):
        code = code_from_name(cat.get("name", ""))
        if code:
            mapping[int(cat["id"])] = code
    return mapping


def load_coco(coco_path: Path) -> CountryAnnotations:
    with coco_path.open() as fh:
        coco = json.load(fh)
    cat_map = parse_category_map(coco)

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        code = cat_map.get(int(ann["category_id"]))
        if code in TARGET_CLASSES:
            anns_by_image[int(ann["image_id"])].append(ann)

    out = CountryAnnotations(source_format=f"coco (categories {len(cat_map)})")
    for img in coco.get("images", []):
        boxes = []
        for ann in anns_by_image.get(int(img["id"]), []):
            x, y, w, h = ann["bbox"]
            boxes.append(
                (cat_map[int(ann["category_id"])], (float(x), float(y), float(x + w), float(y + h)))
            )
        if not boxes:
            continue
        out.images.append(
            ImageAnnotations(img["file_name"], int(img["width"]), int(img["height"]), boxes)
        )
    return out


# --- Pascal VOC ---------------------------------------------------------


def load_voc(xml_dir: Path) -> CountryAnnotations:
    """Read Pascal VOC XML.

    Handles the two quirks of the release: ``<filename>`` may lack an
    extension, and ``<name>`` may carry a descriptive suffix ("D00 Longitudinal").
    """
    out = CountryAnnotations(source_format="voc")
    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"{xml_path}: malformed XML: {exc}") from exc

        filename = (root.findtext("filename") or xml_path.stem).strip()
        size = root.find("size")
        if size is None:
            continue
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
        if width <= 0 or height <= 0:
            continue

        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        for obj in root.findall("object"):
            code = code_from_name(obj.findtext("name", ""))
            if code not in TARGET_CLASSES:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            boxes.append(
                (
                    code,
                    (
                        float(bbox.findtext("xmin", "0")),
                        float(bbox.findtext("ymin", "0")),
                        float(bbox.findtext("xmax", "0")),
                        float(bbox.findtext("ymax", "0")),
                    ),
                )
            )
        if boxes:
            out.images.append(ImageAnnotations(filename, width, height, boxes))
    return out


def load_annotations(train_dir: Path) -> CountryAnnotations | None:
    """Auto-detect format and load, whichever the release uses."""
    found = find_annotations(train_dir)
    if found is None:
        return None
    path, fmt = found
    if fmt == "coco":
        return load_coco(path)
    return load_voc(path)


# --- geometry -----------------------------------------------------------


def voc_xyxy_to_yolo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """VOC absolute xyxy -> YOLO normalised cx,cy,w,h. None if degenerate."""
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, min(xmin, xmax)), min(float(width), max(xmin, xmax))
    ymin, ymax = max(0.0, min(ymin, ymax)), min(float(height), max(ymin, ymax))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:
        return None
    return (
        (xmin + bw / 2) / width,
        (ymin + bh / 2) / height,
        bw / width,
        bh / height,
    )


def coco_bbox_to_yolo(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] -> YOLO normalised cx,cy,w,h."""
    x, y, w, h = bbox
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


# --- curation -----------------------------------------------------------


def curate_country(
    country: str,
    train_dir: Path,
    images_root: Path,
    out_images: Path,
    out_labels: Path,
    stats: Stats,
    *,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> bool:
    """Copy one country's images and labels into the curated train/val split."""
    data = load_annotations(train_dir)
    if data is None:
        print(f"  ! {country}: no annotations found under {train_dir}")
        return False
    print(f"  {country}: {len(data.images)} annotated images ({data.source_format})")

    rng = random.Random(seed)
    rng.shuffle(data.images)

    n_val = int(len(data.images) * val_fraction)
    split_of = {"val": data.images[:n_val], "train": data.images[n_val:]}

    for split, subset in split_of.items():
        for item in subset:
            src = images_root / item.filename
            if not src.is_file():
                # Some releases name the XML stem only.
                src = images_root / f"{Path(item.filename).stem}.jpg"
            if not src.is_file():
                stats.dropped_missing_file += 1
                continue

            dst_img = out_images / split / src.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_img)

            lines = []
            for code, box in item.boxes:
                xyxy = voc_xyxy_to_yolo(box, item.width, item.height)
                if xyxy is None:
                    continue
                cx, cy, nw, nh = xyxy
                lines.append(f"{TARGET_CLASSES[code]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats.boxes[code] += 1
                stats.per_country_boxes[country][code] += 1

            if not lines:
                dst_img.unlink(missing_ok=True)
                continue

            dst_lbl = out_labels / split / f"{dst_img.stem}.txt"
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines) + "\n")

            stats.images += 1
            stats.per_country_images[country] += 1
    return True


# --- CLI ----------------------------------------------------------------


def inspect(src: Path, countries: tuple[str, ...]) -> int:
    print(f"Scanning {src}\n")
    for country in countries:
        train_dir = src / country / "train"
        print(f"{country}  ->  {train_dir}")
        if not train_dir.is_dir():
            print("  country folder not found")
            continue
        found = find_annotations(train_dir)
        if found is None:
            print("  NO ANNOTATIONS (expected annotations/xmls/*.xml or coco.json)")
            print(f"  contents: {[p.name for p in sorted(train_dir.iterdir())][:8]}")
            continue
        path, fmt = found
        print(f"  format    : {fmt}")
        print(f"  path      : {path}")
        if fmt == "voc":
            data = load_voc(path)
            codes: Counter = Counter()
            for item in data.images:
                for code, _ in item.boxes:
                    codes[code] += 1
            print(f"  images    : {len(data.images)} with target annotations")
            print(
                "  classes   : "
                + ", ".join(f"{c}={codes.get(c, 0)}" for c in ("D00", "D10", "D20", "D40"))
            )
        else:
            with path.open() as fh:
                cat_map = parse_category_map(json.load(fh))
            print("  categories:")
            for cat_id, code in sorted(cat_map.items()):
                mark = "KEEP" if code in TARGET_CLASSES else "drop"
                print(f"    id={cat_id:>2}  {code:<5} {mark}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, required=True, help="path to the RDD2022 root")
    parser.add_argument("--dst", type=Path, default=Path("data/rdd2022-curated"))
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="report annotation format and class counts, then exit",
    )
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        parser.error(f"source not found: {args.src}")

    unknown = [c for c in args.countries if c not in AVAILABLE_COUNTRIES]
    if unknown:
        parser.error(f"unknown countries {unknown}; available: {list(AVAILABLE_COUNTRIES)}")

    if args.inspect:
        return inspect(args.src, tuple(args.countries))

    print(f"Curating {list(args.countries)}\n  from {args.src}\n  to   {args.dst}\n")
    # Create the output root up front. Otherwise a run that finds no annotations
    # dies with FileNotFoundError while writing the YAML, masking the real
    # problem (bad --src path or wrong layout) behind an unrelated traceback.
    args.dst.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    for country in args.countries:
        train_dir = args.src / country / "train"
        if not train_dir.is_dir():
            print(f"  ! {country}: no folder at {train_dir}, skipping")
            continue
        curate_country(
            country,
            train_dir,
            train_dir / "images",
            args.dst / "images",
            args.dst / "labels",
            stats,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )

    # Absolute path: Ultralytics resolves a relative `path` against the YAML's
    # own directory in some versions and datasets_dir in others, so relative
    # works on a Mac and fails under /content (Colab) or /kaggle/input.
    dataset_root = args.dst.resolve()
    (args.dst / "roadscope.yaml").write_text(
        f"path: {dataset_root}\ntrain: images/train\nval: images/val\n\n"
        "names:\n"
        "  0: D00   # Longitudinal Crack\n"
        "  1: D10   # Transverse Crack\n"
        "  2: D20   # Alligator / Fatigue Crack\n"
        "  3: D40   # Pothole\n"
    )

    print(stats.report())
    print(f"\nDataset YAML: {args.dst / 'roadscope.yaml'}")
    print(f"  path: {dataset_root}")
    if stats.images == 0:
        print("\nERROR: zero images curated. Run with --inspect to check the layout.")
        return 1
    if stats.boxes.get("D10", 0) < 500:
        print(
            "\nWARNING: very few D10 (transverse crack) labels. Recall on that class\n"
            "         will be poor. Report per-class AP rather than hiding it behind\n"
            "         an averaged mAP. See REBUILD_PLAN.md section 5.3."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
