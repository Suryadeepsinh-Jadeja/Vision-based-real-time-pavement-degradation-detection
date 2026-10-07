# Cloud training runbook — RDD2022 fine-tune

The local machine is 8 GB RAM with no GPU. Training runs on Kaggle or Colab.

**Launch training before anything else is ready.** It is the only task in the
rebuild that cannot be compressed later, because it waits on external quota.

---

## 1. Get the dataset

RDD2022 is ~4 GB.

- figshare: <https://doi.org/10.6084/m9.figshare.21431547.v1>
- or `github.com/sekilab/RoadDamageDetector`

**Kaggle:** search "RDD2022" in Datasets, add to a notebook kernel.
**Colab:** download with `!wget` in a cell, or mount Google Drive.

Expected layout:

```
RDD2022/
  India/            train/images/*.jpg   train/annotations/coco.json
  Japan/            train/images/*.jpg   train/annotations/coco.json
  United_States/    train/images/*.jpg   train/annotations/coco.json
  Norway/ Czech/ China/ ...
```

---

## 2. Inspect before curating

```bash
python scripts/prepare_rdd2022.py --src /path/to/RDD2022 --inspect
```

This prints each country's COCO category table with a KEEP/drop marker.

> **The trap this avoids:** RDD2022 declares 8 categories, and `category_id` is
> **not** the class index. D40 (pothole) is `category_id 7`, not 3. Code that
> assumes `category_id == target_class` trains a pothole detector on crosswalk
> blur. The script parses the `Dxx` code out of each category name instead.

---

## 3. Curate

```bash
python scripts/prepare_rdd2022.py \
    --src /path/to/RDD2022 \
    --dst data/rdd2022-curated \
    --countries India Japan United_States \
    --val-fraction 0.2
```

China is excluded by default — its subset is drone/oblique imagery, a different
domain that degrades forward-facing vehicle-mounted inference. Add `Norway` if
you want more data; it is also vehicle-mounted.

Read the report. It flags rare classes. **If D10 lands below ~500 labels, expect
poor recall on transverse cracks** — India has only 68 in the raw dataset. This
is a known limitation, not a bug to hide; report per-class AP.

---

## 4. Train

```python
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # segmentation: masks enable crack width measurement
model.train(
    data="data/rdd2022-curated/roadscope.yaml",
    epochs=80,
    imgsz=960,  # upscale from native 600-720; thin cracks vanish otherwise
    batch=8,
    device="cuda",
    patience=15,
    project="runs",
    name="rdd2022-roadscope-r1",
    # lighting varies enormously on Indian roads
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,  # horizontal only — roads have no vertical flip
    fliud=0.0,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,  # pothole oversampling
    erasing=0.2,
)
```

---

## 5. Evaluate — per class, never just mAP

```python
from ultralytics import YOLO

model = YOLO("runs/rdd2022-roadscope-r1/weights/best.pt")
metrics = model.val(data="data/rdd2022-curated/roadscope.yaml", split="val")

print(f"mAP50      {metrics.box.map50:.4f}")
print(f"mAP50-95   {metrics.box.map:.4f}")
for i, name in model.names.items():
    print(
        f"  {name}  AP50 {metrics.box.ap50[i]:.4f}  AP50-95 {metrics.box.ap[i - 1] if i else metrics.box.ap[i]:.4f}"
    )
```

Reference points:

| Metric | Gate | Note |
| --- | --- | --- |
| Overall mAP50 | ≥ 0.60 | CRDDC'2022 winner scored F1 76.9%, so this is realistic |
| **D40 AP50** | ≥ 0.55 | Pothole — the rarest class, the project's namesake |
| D40 recall | ≥ 0.70 | On the in-domain set, not RDD2022 |
| Lane-marking FP | 0 | Measured separately on paved-marking frames |

**Commit `metrics.json` for every run.** The point is the trend across runs, not
the final number alone.

---

## 6. Label an in-domain test set

RDD2022 mAP will *not* predict performance on your roads. Capture at least 150
frames of your own dashcam video and hand-label them.

```bash
python -m roadscope.agent --camera-url http://<phone>:8080/stream
```

Save frames, annotate in CVAT or LabelImg, evaluate against that set. This number
is the one you quote in the viva.

---

## 7. Publish

Upload `best.pt` to a GitHub Release. **Do not commit it to git** — that is how
the repository ended up 402 MB.

```bash
gh release create v0.1.0 runs/rdd2022-roadscope-r1/weights/best.pt \
    --title "RoadScope RDD2022 detector" --notes "mAP50 0.xx, D40 AP50 0.xx"
```

Consumers fetch it with `scripts/download_weights.sh`.

---

## Expected timeline

| Run | Days | Purpose |
| --- | --- | --- |
| r1 | 3–4 | Baseline. Reveals which class collapses. |
| r2 | 2–3 | Imbalance fixes driven by r1's per-class AP. |
| r3 | 2–3 | Best curated mix, final. |

If D40 is still short at r2, run r4 with pothole-only oversampling and stronger
`copy_paste`. **Budget for four runs** — pothole is the rarest class, and this is
the phase with zero schedule float.