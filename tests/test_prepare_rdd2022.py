"""Tests for the RDD2022 curation script.

The script sits on the critical path for Phase 3: a silent mistake here trains
a bad model for days before anyone notices.

Traps guarded here:

1. The official release ships Pascal VOC XML under ``annotations/xmls/``, not
   ``coco.json``. A COCO-only script silently curates zero images.
2. ``category_id`` (COCO) is not the target class index. D40 pothole appears at
   different ids depending on the export, so both formats resolve the D-code by
   parsing the class *name*.
3. The China subsets are drone/motorbike imagery and must be excluded.
4. The dataset YAML must carry an absolute path to work on Colab/Kaggle.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rdd2022.py"

VOC_XML = """<?xml version="1.0"?>
<annotation>
  <folder>images</folder>
  <filename>{filename}</filename>
  <size><width>600</width><height>600</height><depth>3</depth></size>
{objects}
</annotation>
"""

OBJ = """  <object>
    <name>{name}</name>
    <bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox>
  </object>
"""

COCO_CATEGORIES = [
    {"id": 0, "name": "D00 Longitudinal"},
    {"id": 1, "name": "D01 Repair"},
    {"id": 2, "name": "D10 Transverse"},
    {"id": 3, "name": "D11 Construction joint part"},
    {"id": 4, "name": "D20 Alligator"},
    {"id": 5, "name": "D43 Crosswalk blur"},
    {"id": 6, "name": "D44 White line blur"},
    {"id": 7, "name": "D40 Pothole"},
]


@pytest.fixture(scope="module")
def prepare():
    spec = importlib.util.spec_from_file_location("prepare_rdd2022", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_rdd2022"] = module
    spec.loader.exec_module(module)
    return module


def write_voc(xml_dir: Path, stem: str, filename: str, objects: list[tuple[str, ...]]) -> None:
    xml_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(OBJ.format(name=n, x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for n, *b in objects)
    (xml_dir / f"{stem}.xml").write_text(VOC_XML.format(filename=filename, objects=body))


def make_image(images_dir: Path, name: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / name).write_bytes(b"\xff\xd8\xff")


# --- format detection ---------------------------------------------------


def test_detects_voc_annotations(prepare, tmp_path):
    """The official release uses VOC XML, not coco.json."""
    train = tmp_path / "RDD2022" / "India" / "train"
    write_voc(
        train / "annotations" / "xmls",
        "India_000000",
        "India_000000.jpg",
        [("D40", 10, 10, 110, 90)],
    )
    make_image(train / "images", "India_000000.jpg")

    found = prepare.find_annotations(train)
    assert found is not None, "VOC annotations not detected"
    _, fmt = found
    assert fmt == "voc"

    data = prepare.load_annotations(train)
    assert data is not None
    assert len(data.images) == 1
    assert data.images[0].boxes[0][0] == "D40"


def test_detects_coco_annotations(prepare, tmp_path):
    """COCO re-uploads on Kaggle/HuggingFace must still work."""
    train = tmp_path / "RDD2022" / "Japan" / "train"
    (train / "annotations").mkdir(parents=True)
    make_image(train / "images", "J_0.jpg")
    (train / "annotations" / "coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "J_0.jpg", "width": 600, "height": 600}],
                "annotations": [
                    {"id": 0, "image_id": 0, "category_id": 7, "bbox": [10, 10, 100, 80]}
                ],
                "categories": COCO_CATEGORIES,
            }
        )
    )
    found = prepare.find_annotations(train)
    assert found is not None and found[1] == "coco"

    data = prepare.load_annotations(train)
    assert data is not None and data.images[0].boxes[0][0] == "D40"


def test_missing_annotations_returns_none(prepare, tmp_path):
    train = tmp_path / "RDD2022" / "Czech" / "train"
    (train / "images").mkdir(parents=True)
    assert prepare.find_annotations(train) is None
    assert prepare.load_annotations(train) is None


# --- class name resolution ---------------------------------------------


def test_code_parsing_from_various_name_forms(prepare):
    assert prepare.code_from_name("D00") == "D00"
    assert prepare.code_from_name("D00 Longitudinal") == "D00"
    assert prepare.code_from_name("d40 pothole") == "D40"
    assert prepare.code_from_name("Alligator Crack") is None


def test_coco_category_map_from_names(prepare):
    cat_map = prepare.parse_category_map({"categories": COCO_CATEGORIES})
    assert cat_map[7] == "D40"
    assert cat_map[0] == "D00"
    assert cat_map[2] == "D10"
    assert cat_map[4] == "D20"


def test_target_classes_are_canonical_four(prepare):
    assert set(prepare.TARGET_CLASSES) == {"D00", "D10", "D20", "D40"}
    assert set(prepare.TARGET_CLASSES.values()) == {0, 1, 2, 3}


def test_china_excluded_by_default(prepare):
    """Drone/motorbike imagery is a different domain from a dashboard phone."""
    ns = vars(prepare)
    defaults = ns["DEFAULT_COUNTRIES"]
    available = ns["AVAILABLE_COUNTRIES"]
    assert not any(c.startswith("China") for c in defaults)
    assert "China_Drone" in available
    assert "China_MotorBike" in available


# --- geometry -----------------------------------------------------------


def test_voc_box_to_yolo(prepare):
    # 100x80 box at (10,10) in a 600x600 image
    cx, cy, w, h = prepare.voc_xyxy_to_yolo((10, 10, 110, 90), 600, 600)
    assert cx == pytest.approx(60 / 600)
    assert cy == pytest.approx(50 / 600)
    assert w == pytest.approx(100 / 600)
    assert h == pytest.approx(80 / 600)


def test_degenerate_box_is_rejected(prepare):
    assert prepare.voc_xyxy_to_yolo((10, 10, 10, 10), 600, 600) is None
    assert prepare.voc_xyxy_to_yolo((10, 10, 11, 11), 600, 600) is None


def test_inverted_voc_box_is_normalised(prepare):
    """Some VOC files store xmin > xmax. Must not produce negative width."""
    _, _, w, h = prepare.voc_xyxy_to_yolo((110, 90, 10, 10), 600, 600)
    assert w > 0 and h > 0


def test_coco_bbox_conversion(prepare):
    cx, cy, w, h = prepare.coco_bbox_to_yolo([100, 200, 50, 25], 600, 600)
    assert cx == pytest.approx(125 / 600)
    assert cy == pytest.approx(212.5 / 600)
    assert w == pytest.approx(50 / 600)
    assert h == pytest.approx(25 / 600)


# --- end-to-end ---------------------------------------------------------


def test_voc_end_to_end(prepare, tmp_path):
    """Full curation from a VOC tree, matching the real release layout."""
    root = tmp_path / "RDD2022"
    train = root / "India" / "train"
    write_voc(
        train / "annotations" / "xmls",
        "India_000000",
        "India_000000.jpg",
        [("D40", 10, 10, 110, 90)],
    )  # pothole
    make_image(train / "images", "India_000000.jpg")
    write_voc(
        train / "annotations" / "xmls",
        "India_000001",
        "India_000001.jpg",
        [("D43", 10, 10, 110, 90)],
    )  # crosswalk blur -> dropped
    make_image(train / "images", "India_000001.jpg")
    write_voc(
        train / "annotations" / "xmls",
        "India_000002",
        "India_000002.jpg",
        [("D00", 20, 20, 120, 100), ("D20", 300, 300, 400, 380)],
    )
    make_image(train / "images", "India_000002.jpg")

    out = tmp_path / "curated"
    rc = prepare.main(["--src", str(root), "--dst", str(out), "--val-fraction", "0.0"])
    assert rc == 0

    labels = list(out.rglob("*.txt"))
    assert labels, "no labels written"
    classes = {
        int(line.split()[0])
        for label_file in labels
        for line in label_file.read_text().splitlines()
        if line.strip()
    }
    assert classes <= {0, 1, 2, 3}, f"class index outside 4-class schema: {classes}"
    # pothole present, crosswalk blur dropped
    assert 3 in classes
    assert 0 in classes and 2 in classes
    assert (out / "images" / "train" / "India_000001.jpg").exists() is False


def test_empty_dataset_fails_loudly(prepare, tmp_path):
    """Zero images must exit non-zero, not print a cheerful empty report."""
    root = tmp_path / "RDD2022"
    (root / "India" / "train" / "images").mkdir(parents=True)
    out = tmp_path / "curated"
    rc = prepare.main(["--src", str(root), "--dst", str(out)])
    assert rc == 1, "a zero-image curation must fail loudly"


def test_dataset_yaml_uses_absolute_path(prepare, tmp_path):
    """Relative paths break under /content (Colab) and /kaggle/input."""
    root = tmp_path / "RDD2022"
    train = root / "India" / "train"
    write_voc(train / "annotations" / "xmls", "I_0", "I_0.jpg", [("D40", 10, 10, 110, 90)])
    make_image(train / "images", "I_0.jpg")

    out = tmp_path / "curated"
    prepare.main(["--src", str(root), "--dst", str(out), "--val-fraction", "0.0"])
    yaml_text = (out / "roadscope.yaml").read_text()
    value = (
        next(ln for ln in yaml_text.splitlines() if ln.startswith("path:")).split(":", 1)[1].strip()
    )
    assert value.startswith("/"), f"dataset path must be absolute, got {value!r}"
    assert Path(value) == out.resolve()


def test_rejects_unknown_country(prepare, tmp_path):
    with pytest.raises(SystemExit):
        prepare.main(
            ["--src", str(tmp_path), "--dst", str(tmp_path / "x"), "--countries", "Atlantis"]
        )
