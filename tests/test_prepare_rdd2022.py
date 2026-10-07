"""Tests for the RDD2022 curation script.

The curation script sits on the critical path for Phase 3: a silent mistake here
trains a bad model for days before anyone notices.

The specific trap guarded here: RDD2022 declares 8 COCO categories and
``category_id`` is *not* the target class index. D40 (pothole) is
``category_id 7``. Code that assumes ``category_id == class_index`` writes class
7 into a 4-class dataset -- either crashing or, worse, quietly training the
pothole head on crosswalk blur.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rdd2022.py"


@pytest.fixture(scope="module")
def prepare():
    spec = importlib.util.spec_from_file_location("prepare_rdd2022", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_rdd2022"] = module
    spec.loader.exec_module(module)
    return module


# The real RDD2022 category table.
REAL_CATEGORIES = [
    {"id": 0, "name": "D00 Longitudinal"},
    {"id": 1, "name": "D01 Repair"},
    {"id": 2, "name": "D10 Transverse"},
    {"id": 3, "name": "D11 Construction joint part"},
    {"id": 4, "name": "D20 Alligator"},
    {"id": 5, "name": "D43 Crosswalk blur"},
    {"id": 6, "name": "D44 White line blur"},
    {"id": 7, "name": "D40 Pothole"},
]


def test_category_map_is_read_from_names(prepare):
    """Category ids must come from the name, not be inferred positionally."""
    cat_map = prepare.parse_category_map({"categories": REAL_CATEGORIES})
    assert cat_map[7] == "D40"
    assert cat_map[0] == "D00"
    assert cat_map[2] == "D10"
    assert cat_map[4] == "D20"
    # The four non-target classes are recognised so they can be dropped.
    assert cat_map[1] == "D01"
    assert cat_map[5] == "D43"


def test_target_classes_are_the_canonical_four(prepare):
    assert set(prepare.TARGET_CLASSES) == {"D00", "D10", "D20", "D40"}
    assert set(prepare.TARGET_CLASSES.values()) == {0, 1, 2, 3}


def test_china_excluded_by_default(prepare):
    """China is drone/oblique imagery and must not be in the default mix."""
    assert "China" not in prepare.DEFAULT_COUNTRIES
    assert "China_M" not in prepare.DEFAULT_COUNTRIES
    assert "China_D" not in prepare.DEFAULT_COUNTRIES


def test_coco_bbox_conversion(prepare):
    cx, cy, w, h = prepare.coco_bbox_to_yolo([100, 200, 50, 25], 600, 600)
    assert cx == pytest.approx(125 / 600)
    assert cy == pytest.approx(212.5 / 600)
    assert w == pytest.approx(50 / 600)
    assert h == pytest.approx(25 / 600)


@pytest.fixture
def fake_rdd2022(tmp_path):
    """A minimal RDD2022 tree using the real category layout."""
    root = tmp_path / "RDD2022"
    img_dir = root / "India" / "train" / "images"
    img_dir.mkdir(parents=True)
    ann_dir = root / "India" / "train" / "annotations"
    ann_dir.mkdir(parents=True)

    images = [
        {"id": i, "file_name": f"India_{i:05d}.jpg", "width": 600, "height": 600} for i in range(4)
    ]
    for image in images:
        (img_dir / image["file_name"]).write_bytes(b"\xff\xd8\xff")

    annotations = [
        {"id": 0, "image_id": 0, "category_id": 7, "bbox": [10, 10, 100, 80]},  # pothole
        {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 100, 80]},  # crosswalk blur
        {"id": 2, "image_id": 2, "category_id": 0, "bbox": [10, 10, 100, 80]},  # longitudinal
    ]
    (ann_dir / "coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": REAL_CATEGORIES})
    )
    return root


def test_pothole_category_id_seven_maps_to_class_three(prepare, fake_rdd2022, tmp_path):
    """The core guard. category_id 7 must become YOLO class 3."""
    out = tmp_path / "curated"
    prepare.main(["--src", str(fake_rdd2022), "--dst", str(out), "--val-fraction", "0.5"])

    labels = list(out.rglob("*.txt"))
    assert labels, "no label files written"
    classes = {int(line.split()[0]) for f in labels for line in f.read_text().split("\n") if line}
    assert classes <= {0, 1, 2, 3}, f"class index outside 4-class schema: {classes}"
    assert 3 in classes, "pothole (category_id 7) was not mapped to class 3"


def test_non_target_classes_dropped(prepare, fake_rdd2022, tmp_path):
    """Crosswalk blur (category_id 5) must not survive curation."""
    out = tmp_path / "curated"
    prepare.main(["--src", str(fake_rdd2022), "--dst", str(out), "--val-fraction", "0.5"])

    copied = {p.name for p in out.rglob("*.jpg")}
    # image 1 is crosswalk blur, image 3 has no annotations -> both dropped
    assert "India_00001.jpg" not in copied
    assert "India_00003.jpg" not in copied
    assert len(copied) == 2


def test_split_creates_both_partitions(prepare, fake_rdd2022, tmp_path):
    out = tmp_path / "curated"
    prepare.main(["--src", str(fake_rdd2022), "--dst", str(out), "--val-fraction", "0.5"])
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()
    assert (out / "roadscope.yaml").is_file()


def test_rejects_unknown_country(prepare, fake_rdd2022, tmp_path):
    with pytest.raises(SystemExit):
        prepare.main(
            ["--src", str(fake_rdd2022), "--dst", str(tmp_path / "x"), "--countries", "Atlantis"]
        )
